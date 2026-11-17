"""Adapter for OpenMOSS-Team/MOSS-SoundEffect-v2.0, run in a persistent worker subprocess.

The model needs its own environment (it pins numpy 1.26.4 and transformers 4.57.1), so load() starts
workers/moss-sfx/worker.py with that environment's Python and waits until the model is on the GPU;
unload() stops and reaps the process, which is what releases its GPU memory. This module imports
neither torch nor the model.
"""
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from local_tts.adapters.base import EffectAudio, SoundEffectAdapter
from local_tts.adapters.sfx_worker import WorkerError, WorkerProcess, int_field, number_field
from local_tts.config import PROJECT_ROOT
from local_tts.errors import TTSError

log = logging.getLogger(__name__)

# Environment variables of the service that must not leak into the worker's own Python.
SCRUB_ENV = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP", "PYTHONSAFEPATH")


def compile_cache_dir() -> Path:
    """Persistent per-user cache for torch.compile kernels (Inductor and Triton), so warm starts are fast:
    LOCAL_TTS_CACHE_DIR, else $XDG_CACHE_HOME/local-tts (~/.cache/local-tts), + /torch-compile."""
    raw = os.environ.get("LOCAL_TTS_CACHE_DIR")
    base = Path(raw).expanduser() if raw else Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "local-tts"
    return base / "torch-compile"


def c_compiler() -> Optional[str]:
    """What the worker will find (it decides for itself; this is for status before a load)."""
    cc = os.environ.get("CC")
    if cc:
        return shutil.which(cc.split()[0])
    return next((p for p in map(shutil.which, ("cc", "gcc", "clang")) if p), None)


class MossSoundEffectAdapter(SoundEffectAdapter):
    def __init__(self, entry):
        super().__init__(entry)
        o = entry.options
        worker = Path(o.get("worker", "workers/moss-sfx"))
        self.worker_dir = worker if worker.is_absolute() else PROJECT_ROOT / worker
        self.python = self.worker_dir / ".venv" / "bin" / "python"
        self.script = self.worker_dir / "worker.py"
        self.sampling = dict(o.get("sampling", {}))
        self.compile = bool(o.get("compile", False))
        self.load_timeout = float(o.get("load_timeout_seconds", 1800))
        self.generate_timeout = float(o.get("generate_timeout_seconds", 600))
        self.stop_grace = 10.0  # seconds a stopping worker gets to exit before SIGTERM / SIGKILL
        self.worker: Optional[WorkerProcess] = None
        self.ready: Optional[dict[str, Any]] = None  # the worker's ready message while loaded
        self._broken = False  # the worker failed as a whole; unload() still has to reap it
        self._next_id = 0

    # --- installation ------------------------------------------------------------------------------

    def install_command(self) -> str:
        try:
            project = self.worker_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            project = self.worker_dir
        return f"uv sync --frozen --project {project}"

    def download_command(self) -> str:
        o = self.entry.options
        cmd = f"uvx --from huggingface_hub hf download {o['repo']} --revision {o['revision']}"
        if o.get("model_dir"):  # argv() passes the configured directory instead of the HF cache
            cmd += f" --local-dir {self.argv()[-1]}"
        return cmd

    def installed(self) -> bool:
        return os.access(self.python, os.X_OK) and self.script.is_file()

    def install_error(self) -> TTSError:
        cmd = self.install_command()
        return TTSError("sfx_worker_not_installed",
                        f"The environment of sound-effect model '{self.entry.id}' is not installed "
                        f"({self.python} is missing). Install it with `{cmd}` in {PROJECT_ROOT}.", 503,
                        install_command=cmd)

    def weights_cached(self) -> Optional[bool]:
        """Whether the pinned weights are in the Hugging Face cache (None if that cannot be checked)."""
        o = self.entry.options
        try:
            from huggingface_hub import try_to_load_from_cache
            found = try_to_load_from_cache(o["repo"], "model_index.json", revision=o["revision"])
        except Exception:  # noqa: BLE001 - status only
            return None
        return isinstance(found, str)

    # --- lifecycle -----------------------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        w = self.worker
        return w is not None and self.ready is not None and not self._broken and w.alive()

    def _env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in SCRUB_ENV}
        cache = compile_cache_dir()
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / "inductor"))
        env.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
        env["PYTHONUNBUFFERED"] = "1"
        if not self.compile:
            env["TORCHDYNAMO_DISABLE"] = "1"
        return env

    def argv(self) -> list[str]:
        o = self.entry.options
        args = [str(self.python), str(self.script), "--compile", "1" if self.compile else "0"]
        if o.get("model_dir"):  # a local copy instead of the HF cache (relative to the project root)
            d = Path(o["model_dir"]).expanduser()
            return args + ["--model-dir", str(d if d.is_absolute() else PROJECT_ROOT / d)]
        return args + ["--repo", str(o["repo"]), "--revision", str(o["revision"])]

    def load(self) -> None:
        if self.worker is not None:
            raise RuntimeError("the previous worker was not stopped")  # the manager unloads first
        if not self.installed():
            raise self.install_error()
        self.warnings, self.ready, self._broken = [], None, False
        if self.compile:
            for d in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
                Path(self._env()[d]).mkdir(parents=True, exist_ok=True)
        worker = WorkerProcess(self.argv(), self._env(), name=f"sfx-worker[{self.entry.id}]",
                               cwd=str(self.worker_dir))
        self.worker = worker  # set first, so unload() reaps it whatever happens next
        try:
            worker.start()
            msg, _ = worker.recv(self.load_timeout)
        except WorkerError as e:
            raise self._worker_failed(e, "while loading") from None
        if msg.get("type") == "error":
            raise self._startup_error(msg)
        if msg.get("type") != "ready":
            raise TTSError("sfx_worker_failed", f"The sound-effect worker sent {msg.get('type')!r} instead of "
                           "ready.", 500)
        try:
            rate = int_field(msg, "sample_rate", 1, 10**7)
            longest = number_field(msg, "max_seconds", 0.0, 86400.0)
        except WorkerError as e:
            raise self._worker_failed(e, "while loading") from None
        if rate != self.sample_rate or longest < self.max_seconds:
            raise TTSError("sfx_worker_failed", f"The model reports {msg.get('sample_rate')} Hz and at most "
                           f"{msg.get('max_seconds')} s, but config/models.yaml says {self.sample_rate} Hz and "
                           f"{self.max_seconds} s.", 500)
        self.ready = msg
        if self.compile and not msg.get("compile"):
            self.warnings.append(f"Model '{self.entry.id}' runs without torch.compile (slower): "
                                 f"{msg.get('compile_note') or 'compile is unavailable'}.")
        log.info("%s ready (pid %s): compile=%s, load %.1fs, warm-up %ss", self.entry.id, worker.pid,
                 msg.get("compile"), msg.get("load_seconds") or 0, msg.get("warmup_seconds"))

    def _startup_error(self, msg: dict) -> TTSError:
        code, text = msg.get("code"), str(msg.get("message", ""))
        if code == "weights_missing":
            cmd = self.download_command()
            return TTSError("model_weights_missing", f"The weights of '{self.entry.id}' are missing: {text} "
                            f"Download them with `{cmd}`.", 503, download_command=cmd)
        if code == "no_gpu":
            return TTSError("gpu_unavailable", f"Sound-effect model '{self.entry.id}' needs an NVIDIA GPU: {text}", 503)
        return TTSError("sfx_worker_failed", f"The sound-effect worker failed to load the model: {text}", 500,
                        worker_code=code)

    def _stop_failed_worker(self) -> str:
        """Stop and reap a worker that failed as a whole, so its GPU memory is released now. Returns how
        that went, for the error message. If it cannot be stopped it stays owned (self.worker set, the
        model current in the manager): nothing else loads until a later unload() succeeds."""
        self._broken, self.ready = True, None
        worker = self.worker
        if worker is None:
            return "It is stopped"
        try:
            worker.stop(self.stop_grace)
        except WorkerError as err:
            log.error("%s: could not stop the failed worker: %s", self.entry.id, err.message)
            return (f"It could not be stopped ({err.message}); no other model loads until it is "
                    "(the next job or POST /v1/models/unload tries again)")
        self.worker = None
        return "It is stopped"

    def _worker_failed(self, e: WorkerError, when: str, clip: Optional[str] = None) -> TTSError:
        stopped = self._stop_failed_worker()
        if e.reason == "start":
            return TTSError("sfx_worker_failed", f"Could not start the sound-effect worker: {e.message}. "
                            f"Reinstall its environment with `{self.install_command()}`.", 500,
                            install_command=self.install_command())
        code = "sfx_worker_timeout" if e.reason == "timeout" else "sfx_worker_failed"
        limit = self.load_timeout if when == "while loading" else self.generate_timeout
        hint = f" (limit {limit:.0f}s)" if e.reason == "timeout" else ""
        return TTSError(code, f"The sound-effect worker failed {when}{hint}: {e.message}. {stopped}; the next "
                        "job starts a new one.", 500, **({} if clip is None else {"clip": clip}))

    def unload(self) -> None:
        worker, self.ready = self.worker, None
        if worker is None:
            return
        try:
            worker.stop(self.stop_grace)
        except WorkerError as e:
            self._broken = True
            raise RuntimeError(e.message) from None  # self.worker stays set: the next unload tries again
        self.worker, self._broken = None, False

    def release_cache(self) -> None:
        if not self.loaded:
            return
        try:
            deadline = time.monotonic() + 30.0
            rid = self._request({"type": "release_cache"}, 30.0)
            msg, _ = self.worker.recv(max(0.0, deadline - time.monotonic()))
            if msg.get("type") != "ok" or msg.get("id") != rid:
                raise WorkerError("protocol", f"unexpected reply {msg.get('type')!r} to release_cache")
        except WorkerError as e:
            log.warning("%s: worker failed on release_cache (%s); %s, the next job starts a new one",
                        self.entry.id, e.message, self._stop_failed_worker())

    def _request(self, header: dict, timeout: float) -> int:
        self._next_id += 1
        self.worker.send(dict(header, id=self._next_id), timeout)
        return self._next_id

    # --- generation ----------------------------------------------------------------------------------

    def generate(self, prompt: str, *, seconds: float, seed: int, num_inference_steps: Optional[int],
                 cfg_scale: Optional[float], clip: Optional[str] = None) -> EffectAudio:
        if not self.loaded:
            raise TTSError("sfx_worker_failed", "The sound-effect worker is not running.", 500)
        s = self.sampling
        # The model returns int(sample_rate * round(seconds, 1)) samples; allow a little more.
        max_samples = int(math.ceil((seconds + 0.1) * self.sample_rate)) + 1
        try:
            deadline = time.monotonic() + self.generate_timeout  # sending counts against the limit too
            rid = self._request({
                "type": "generate", "prompt": prompt, "seconds": seconds, "seed": int(seed),
                "num_inference_steps": int(num_inference_steps or s.get("num_inference_steps", 100)),
                "cfg_scale": float(cfg_scale if cfg_scale is not None else s.get("cfg_scale", 4.0)),
                "sigma_shift": float(s.get("sigma_shift", 5.0)),
            }, self.generate_timeout)
            t = time.perf_counter()
            msg, payload = self.worker.recv(max(0.0, deadline - time.monotonic()), max_payload=max_samples * 2 * 4)
            if msg.get("id") != rid:
                raise WorkerError("protocol", f"reply for request {msg.get('id')!r}, expected {rid}")
        except WorkerError as e:
            raise self._worker_failed(e, "while generating", clip) from None
        if msg.get("type") == "error":
            code = msg.get("code")
            if code not in ("sfx_generation_failed", "sfx_invalid_output", "gpu_out_of_memory"):
                code = "sfx_generation_failed"
            raise TTSError(code, f"Generation failed: {msg.get('message')}", 500)
        try:
            channels, samples = int_field(msg, "channels", 1, 2), int_field(msg, "samples", 1, max_samples)
            if msg.get("type") != "result" or msg.get("dtype") != "float32" \
                    or len(payload) != channels * samples * 4 \
                    or int_field(msg, "sample_rate", 1, 10**7) != self.sample_rate:
                raise WorkerError("protocol", f"inconsistent result header {msg}")
        except WorkerError as e:
            raise self._worker_failed(e, "while generating", clip) from None
        log.info("%s: %.1fs effect in %.1fs", self.entry.id, samples / self.sample_rate, time.perf_counter() - t)
        audio = np.frombuffer(payload, dtype="<f4").reshape(channels, samples).astype(np.float32)
        return EffectAudio(audio=audio, sample_rate=self.sample_rate)

    # --- status ----------------------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        ready = self.ready if self.loaded else None
        installed, worker = self.installed(), self.worker
        d: dict[str, Any] = super().status() | {
            "worker_installed": installed, "weights_cached": self.weights_cached(),
            "worker_pid": worker.pid if worker is not None else None,
            "compile": {"enabled": self.compile, "active": None if ready is None else bool(ready.get("compile")),
                        "c_compiler_found": c_compiler() is not None,
                        "note": None if ready is None else ready.get("compile_note"),
                        "override_env": "LOCAL_TTS_SFX_COMPILE"},
        }
        if not installed:
            d["install_command"] = self.install_command()
        if d["weights_cached"] is False:
            d["download_command"] = self.download_command()
        if ready is not None:
            d["load_seconds"] = ready.get("load_seconds")
            d["warmup_seconds"] = ready.get("warmup_seconds")
        return d
