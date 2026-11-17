"""Server side of the worker-subprocess protocol (the worker side is workers/moss-sfx/worker.py, whose
docstring describes the messages; keep the two in sync).

Each message is a 4-byte big-endian header length, a JSON header and `header["nbytes"]` bytes of
payload. The worker reads requests on stdin and writes replies on its original stdout; library output
goes to its stderr, which a thread drains into the log (keeping the last lines for error messages), so
the pipe never fills. Writes and reads have deadlines and reads are bounded in size: a worker that stops reading, sends more than the caller
allows, closes the pipe, or stays silent past the deadline raises WorkerError.
"""
import collections
import json
import logging
import math
import os
import select
import signal
import struct
import subprocess
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

MAX_HEADER_BYTES = 1 << 20
STDERR_TAIL_LINES = 40
STDERR_LINE_BYTES = 4096


class WorkerError(Exception):
    """The worker process failed as a whole (died, hung, broke the protocol, could not be stopped)."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason  # "exited" | "timeout" | "protocol" | "start" | "unreaped"
        self.message = message


def _reject(name: str):
    raise ValueError(f"{name} is not valid JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"number {text} is out of range")
    return value


def int_field(header: dict, key: str, lo: int, hi: int) -> int:
    """header[key] as an integer in [lo, hi], else WorkerError("protocol")."""
    value = header.get(key)
    if type(value) is not int or not lo <= value <= hi:
        raise WorkerError("protocol", f"field {key!r} is {value!r}, expected an integer in [{lo}, {hi}]")
    return value


def number_field(header: dict, key: str, lo: float, hi: float) -> float:
    """header[key] as a finite number in [lo, hi], else WorkerError("protocol")."""
    value = header.get(key)
    if type(value) not in (int, float) or not lo <= value <= hi:
        raise WorkerError("protocol", f"field {key!r} is {value!r}, expected a number in [{lo}, {hi}]")
    return float(value)


class WorkerProcess:
    def __init__(self, argv: list[str], env: dict[str, str], name: str, cwd: Optional[str] = None):
        self.argv, self.env, self.name, self.cwd = argv, env, name, cwd
        self.proc: Optional[subprocess.Popen] = None
        self._tail: collections.deque[str] = collections.deque(maxlen=STDERR_TAIL_LINES)
        self._drain: Optional[threading.Thread] = None
        self._buf = bytearray()

    # --- lifecycle ---------------------------------------------------------------------------------

    def start(self) -> None:
        try:
            # Own session: stopping it also stops whatever it started (e.g. compiler subprocesses).
            self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, env=self.env, cwd=self.cwd, bufsize=0,
                                         start_new_session=True, close_fds=True)
        except OSError as e:
            raise WorkerError("start", f"could not start {self.argv[0]}: {e}") from None
        # Writes never block: send() waits for room with a deadline, so a worker that stops reading
        # cannot hold the caller (the GPU scheduler thread) forever.
        os.set_blocking(self.proc.stdin.fileno(), False)
        self._drain = threading.Thread(target=self._drain_stderr, name=f"{self.name}-stderr", daemon=True)
        self._drain.start()

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc is not None else None

    def alive(self) -> bool:
        return self.proc is not None and self._exit_status() is None

    def _exit_status(self) -> Optional[str]:
        """None while the worker runs, else how it ended. Never reaps it: while it is an unreaped zombie
        its pid (= its process group id) cannot be reused, so the group can still be killed safely."""
        proc = self.proc
        if proc.returncode is not None:
            code = proc.returncode
            return f"killed by signal {-code}" if code < 0 else f"exit code {code}"
        try:
            r = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            return "exited"
        if r is None:
            return None
        return f"exit code {r.si_status}" if r.si_code == os.CLD_EXITED else f"killed by signal {r.si_status}"

    def _wait_exit(self, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while self._exit_status() is None:
            if time.monotonic() >= end:
                return False
            time.sleep(0.05)
        return True

    def stop(self, grace: float = 10.0) -> None:
        """Ask the worker to exit, then SIGTERM and SIGKILL its process group, and reap it. Raises
        WorkerError("unreaped") if it is still there after SIGKILL (its GPU memory may still be held)."""
        proc = self.proc
        if proc is None:
            return
        if self._exit_status() is None:
            try:  # a worker that does not read its input (full pipe) goes straight on to the signals
                self.send({"type": "shutdown"}, timeout=min(grace, 1.0))
            except WorkerError:
                pass
            try:
                proc.stdin.close()
            except OSError:
                pass
            for sig, wait in ((None, grace), (signal.SIGTERM, min(grace, 5.0)), (signal.SIGKILL, 10.0)):
                if sig is not None:
                    self._kill_group(sig)
                if self._wait_exit(wait):
                    break
            else:
                raise WorkerError("unreaped", f"{self.name} (pid {proc.pid}) did not exit after SIGKILL")
        # Whatever the worker started in its group (e.g. compiler subprocesses) goes with it. Safe: the
        # worker is an unreaped zombie here, so its pid, the group id, is not reused yet.
        self._kill_group(signal.SIGKILL)
        proc.wait()
        for f in (proc.stdin, proc.stdout):
            try:
                f.close()
            except OSError:
                pass
        if self._drain is not None:
            self._drain.join(timeout=2)
        self.proc = None
        log.info("%s stopped (exit code %s)", self.name, proc.returncode)

    def _kill_group(self, sig: int) -> None:
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            self.proc.send_signal(sig)

    # --- messages ----------------------------------------------------------------------------------

    def send(self, header: dict, timeout: float) -> None:
        """Write one message within `timeout` seconds (partial writes continue as the worker reads)."""
        if self.proc is None:
            raise WorkerError("exited", f"{self.name} is not running")
        data = json.dumps(dict(header, nbytes=0)).encode()
        view = memoryview(struct.pack(">I", len(data)) + data)
        fd = self.proc.stdin.fileno()
        deadline = time.monotonic() + timeout
        while view:
            left = deadline - time.monotonic()
            if left <= 0:
                raise WorkerError("timeout", f"{self.name} did not read its input in time{self._stderr_note()}")
            try:
                _, ready, _ = select.select([], [fd], [], left)
                if ready:
                    view = view[os.write(fd, view):]
            except BlockingIOError:
                continue
            except (OSError, ValueError) as e:
                raise self._gone(f"could not send a request ({type(e).__name__})") from None

    def recv(self, timeout: float, max_payload: int = 0) -> tuple[dict, bytes]:
        """The next message: (header, payload). Payloads above `max_payload` bytes are a protocol error."""
        deadline = time.monotonic() + timeout
        (size,) = struct.unpack(">I", self._read(4, deadline))
        if size > MAX_HEADER_BYTES:
            raise WorkerError("protocol", f"{self.name} sent a {size}-byte header")
        raw = self._read(size, deadline)
        try:
            header = json.loads(raw, parse_constant=_reject, parse_float=_finite_float)
            if not isinstance(header, dict) or not isinstance(header.get("type"), str):
                raise ValueError("not an object with a string 'type'")
            nbytes = header.get("nbytes", 0)
            if type(nbytes) is not int:
                raise ValueError(f"nbytes is {nbytes!r}, not an integer")
        except (ValueError, TypeError, RecursionError) as e:
            raise WorkerError("protocol", f"{self.name} sent an invalid header: {e}") from None
        if not 0 <= nbytes <= max_payload:
            raise WorkerError("protocol", f"{self.name} sent a {nbytes}-byte payload (at most {max_payload} expected)")
        return header, self._read(nbytes, deadline) if nbytes else b""

    def _read(self, n: int, deadline: float) -> bytes:
        fd = self.proc.stdout.fileno()
        while len(self._buf) < n:
            left = deadline - time.monotonic()
            if left <= 0:
                raise WorkerError("timeout", f"{self.name} did not answer in time{self._stderr_note()}")
            ready, _, _ = select.select([fd], [], [], left)
            if not ready:
                continue
            chunk = os.read(fd, max(65536, min(n - len(self._buf), 1 << 22)))
            if not chunk:
                raise self._gone("closed its output")
            self._buf += chunk
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def _gone(self, what: str) -> WorkerError:
        status = None
        if self.proc is not None:
            self._wait_exit(2.0)
            status = self._exit_status()
        if self._drain is not None:  # let the last stderr lines (often the reason) reach the tail
            self._drain.join(timeout=1)
        note = "" if status is None else f" ({status})"
        return WorkerError("exited", f"{self.name} {what}{note}{self._stderr_note()}")

    # --- stderr --------------------------------------------------------------------------------------

    def _drain_stderr(self) -> None:
        stream = self.proc.stderr
        try:
            while True:
                line = stream.readline(STDERR_LINE_BYTES)
                if not line:
                    break
                # A progress bar redraws with \r: keep what it shows last.
                text = line.decode("utf-8", "replace").rstrip().rsplit("\r", 1)[-1].strip()
                if text:
                    self._tail.append(text)
                    log.info("[%s] %s", self.name, text)
        except (OSError, ValueError):
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def stderr_tail(self, lines: int = 8) -> list[str]:
        return list(self._tail)[-lines:]

    def _stderr_note(self) -> str:
        tail = self.stderr_tail()
        return "; last output: " + " | ".join(t[-300:] for t in tail) if tail else ""
