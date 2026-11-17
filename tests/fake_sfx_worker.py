"""A stand-in for workers/moss-sfx/worker.py without torch or a GPU, run with the service's own Python.

It reuses the real worker's channel setup and serving loop (imported by path; the real worker never
imports local_tts) with a fake model, and is noisy on purpose: it prints to stdout (Python and fd 1) and
floods stderr, which must reach neither the protocol nor a full pipe. Behaviour:
  FAKE_SFX_MODE=ok (default) | weights_missing | load_crash | never_ready | garbage | no_compile | deaf
  (deaf: reports ready, then never reads stdin again, with its stdin pipe shrunk to 4096 bytes)
  prompts containing FAIL raise (one clip fails), OOM raise an OutOfMemoryError, CRASH exit the process,
  HANG never answer. FAKE_SFX_IGNORE_TERM=1 ignores SIGTERM (so only SIGKILL stops it).
"""
import importlib.util
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "sfx_worker_main", Path(__file__).resolve().parents[1] / "workers" / "moss-sfx" / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

MODE = os.environ.get("FAKE_SFX_MODE", "ok")
SR = 48000


class OutOfMemoryError(RuntimeError):
    pass


def noise(tag: str) -> None:
    sys.stderr.write(("x" * 1000 + "\n") * 300)  # ~300 KB: more than a pipe buffer
    sys.stderr.write("progress 50%\r progress 100%\n")
    sys.stderr.flush()
    print(f"stdout noise from {tag}", flush=True)  # sys.stdout: must land on stderr
    os.write(1, f"fd 1 noise from {tag}\n".encode())  # fd 1 directly: same


def load():
    noise("load")
    if MODE == "weights_missing":
        raise worker.WorkerFailure("weights_missing", "No model files (model_index.json) in /nowhere.")
    if MODE == "load_crash":
        sys.stderr.write("fatal: simulated crash while loading\n")
        sys.stderr.flush()
        os._exit(5)
    if MODE == "never_ready":
        time.sleep(3600)
    if MODE == "garbage":
        os.write(_proto_fd, (2**31).to_bytes(4, "big"))  # claims a 2 GiB header
        time.sleep(3600)

    def generate(prompt, seconds, seed, steps, cfg_scale, sigma_shift):
        noise("generate")
        if "CRASH" in prompt:
            os._exit(3)
        if "HANG" in prompt:
            time.sleep(3600)
        if "FAIL" in prompt:
            raise RuntimeError("simulated generation failure")
        if "OOM" in prompt:
            raise OutOfMemoryError("simulated")
        n = int(SR * round(seconds, 1))
        t = np.arange(n) / SR
        return (0.2 * np.sin(2 * np.pi * (200 + seed % 100) * t))[None, :].astype(np.float32)

    compiled = MODE != "no_compile"
    return generate, lambda: None, {"sample_rate": SR, "max_seconds": 30.0, "compile": compiled,
                                    "compile_note": None if compiled else "no C compiler found (test)",
                                    "load_seconds": 0.01, "warmup_seconds": None}


if __name__ == "__main__":
    if os.environ.get("FAKE_SFX_IGNORE_TERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    inp, out = worker.setup_channels()
    _proto_fd = out.fileno()
    if MODE == "deaf":
        import fcntl
        fcntl.fcntl(0, 1031, 4096)  # F_SETPIPE_SZ: a small pipe fills with one request
        worker.write_message(out, {"type": "ready", "sample_rate": SR, "max_seconds": 30.0, "compile": True,
                                   "compile_note": None, "load_seconds": 0.0, "warmup_seconds": None})
        time.sleep(3600)
    sys.exit(worker.serve(inp, out, load))
