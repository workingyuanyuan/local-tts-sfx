"""MOSS-SoundEffect v2.0 worker: a persistent subprocess of tts-server that holds the model on the GPU.

Runs in its own environment (workers/moss-sfx/.venv) and never imports local_tts. The server starts it
with this environment's Python, waits for "ready", sends one "generate" request at a time and stops it
to release the GPU (closing stdin or a "shutdown" request ends it; so does the server dying).

Protocol (keep in sync with local_tts/adapters/sfx_worker.py). Every message in either direction is
    4-byte big-endian header length | UTF-8 JSON header | `nbytes` bytes of payload (header["nbytes"])
Requests arrive on stdin. Replies go to the original stdout, which is moved to a private fd at start:
fd 1 and sys.stdout then point at stderr, so anything the model's libraries print can never corrupt
the protocol. stderr is free text for the server's log.
  worker -> server, once:  {"type": "ready", "sample_rate", "max_seconds", "compile": bool,
                            "compile_note": str|null, "load_seconds", ...}
                       or  {"type": "error", "code", "message"} and exit
  server -> worker:        {"type": "generate", "id", "prompt", "seconds", "seed", "num_inference_steps",
                            "cfg_scale", "sigma_shift"}
  worker -> server:        {"type": "result", "id", "sample_rate", "channels", "samples",
                            "dtype": "float32"} + payload: float32 little-endian, channel-major
                       or  {"type": "error", "id", "code", "message"} (the worker keeps serving)
  server -> worker:        {"type": "release_cache", "id"} -> {"type": "ok", "id"}
  server -> worker:        {"type": "shutdown"} -> the worker exits
Error codes: weights_missing, no_gpu, load_failed (startup); sfx_generation_failed, gpu_out_of_memory,
sfx_invalid_output, bad_request (per request).
"""
import argparse
import json
import os
import shutil
import struct
import sys
import time
import traceback

MAX_HEADER_BYTES = 1 << 20
# Requests carry no payload; a request header is small. Anything bigger is a protocol error.
MAX_REQUEST_BYTES = 1 << 20


class WorkerFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


# --- framing -----------------------------------------------------------------------------------------

def write_message(out, header: dict, payload: bytes = b"") -> None:
    header = dict(header, nbytes=len(payload))
    data = json.dumps(header, ensure_ascii=False).encode()
    out.write(struct.pack(">I", len(data)) + data)
    if payload:
        out.write(payload)
    out.flush()


def _read_exact(inp, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = inp.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return bytes(buf)


def read_message(inp) -> dict:
    (size,) = struct.unpack(">I", _read_exact(inp, 4))
    if size > MAX_HEADER_BYTES:
        raise WorkerFailure("bad_request", f"header of {size} bytes")
    header = json.loads(_read_exact(inp, size))
    nbytes = int(header.get("nbytes", 0))
    if nbytes:
        if nbytes > MAX_REQUEST_BYTES:
            raise WorkerFailure("bad_request", f"request payload of {nbytes} bytes")
        _read_exact(inp, nbytes)
    return header


def setup_channels():
    """Keep the real stdout for the protocol and send fd 1 / sys.stdout to stderr."""
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    _die_with_parent()
    return os.fdopen(0, "rb", buffering=0, closefd=False), os.fdopen(proto_fd, "wb", buffering=0)


def _die_with_parent() -> None:
    """Linux: get SIGKILL if the server dies, even mid-generation (an idle worker also sees stdin close).
    The signal follows the thread that started the worker: the server starts it from its long-lived GPU
    worker thread."""
    try:
        import ctypes
        import signal
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
        if os.getppid() == 1:  # the parent is already gone
            os._exit(1)
    except Exception:  # noqa: BLE001 - best effort; stdin EOF still ends an idle worker
        pass


def log(msg: str) -> None:
    print(f"[sfx-worker] {msg}", file=sys.stderr, flush=True)


# --- serving -------------------------------------------------------------------------------------------

def serve(inp, out, load) -> int:
    """`load()` returns (generate, release, info): generate(prompt, seconds, seed, steps, cfg_scale,
    sigma_shift) -> float32 numpy [channels, samples]; release() frees cached GPU memory; info goes into
    the ready message. It raises WorkerFailure for expected startup errors."""
    import numpy as np

    try:
        generate, release, info = load()
    except WorkerFailure as e:
        write_message(out, {"type": "error", "code": e.code, "message": e.message})
        return 2
    except Exception as e:  # noqa: BLE001 - report, then exit
        traceback.print_exc()
        write_message(out, {"type": "error", "code": "load_failed", "message": f"{type(e).__name__}: {e}"})
        return 2
    write_message(out, {"type": "ready", **info})
    max_seconds = float(info["max_seconds"])
    while True:
        try:
            req = read_message(inp)
        except EOFError:
            return 0
        kind, rid = req.get("type"), req.get("id")
        if kind == "shutdown":
            return 0
        if kind == "release_cache":
            release()
            write_message(out, {"type": "ok", "id": rid})
            continue
        if kind != "generate":
            write_message(out, {"type": "error", "id": rid, "code": "bad_request",
                                "message": f"unknown request type {kind!r}"})
            continue
        failure = None
        audio = None
        try:
            seconds = float(req["seconds"])
            if not 0 < seconds <= max_seconds:
                raise WorkerFailure("bad_request", f"seconds must be in (0, {max_seconds}], got {seconds}")
            audio = generate(str(req["prompt"]), seconds, int(req["seed"]), int(req["num_inference_steps"]),
                             float(req["cfg_scale"]), float(req["sigma_shift"]))
            audio = np.ascontiguousarray(audio, dtype="<f4")
            if audio.ndim != 2 or audio.shape[1] == 0:
                raise WorkerFailure("sfx_invalid_output", f"model returned audio of shape {audio.shape}")
            if not np.isfinite(audio).all():
                raise WorkerFailure("sfx_invalid_output", "model returned non-finite samples")
        except WorkerFailure as e:
            failure = (e.code, e.message)
        except Exception as e:  # noqa: BLE001 - one clip failed; keep serving
            traceback.print_exc()
            code = "gpu_out_of_memory" if type(e).__name__ == "OutOfMemoryError" else "sfx_generation_failed"
            failure = (code, f"{type(e).__name__}: {e}")
        if failure is not None:
            # Outside the except block: no traceback keeps the failed run's GPU tensors alive.
            audio = None
            release()
            write_message(out, {"type": "error", "id": rid, "code": failure[0], "message": failure[1]})
            continue
        write_message(out, {"type": "result", "id": rid, "sample_rate": int(info["sample_rate"]),
                            "channels": int(audio.shape[0]), "samples": int(audio.shape[1]), "dtype": "float32"},
                      audio.tobytes())


# --- the real model --------------------------------------------------------------------------------------

def c_compiler() -> str | None:
    cc = os.environ.get("CC")
    if cc:
        return shutil.which(cc.split()[0])
    return next((p for p in map(shutil.which, ("cc", "gcc", "clang")) if p), None)


def resolve_weights(args) -> str:
    """The model directory: --model-dir, else the pinned snapshot in the default HF cache. Never downloads."""
    if args.model_dir:
        if not os.path.isfile(os.path.join(args.model_dir, "model_index.json")):
            raise WorkerFailure("weights_missing", f"No model files (model_index.json) in {args.model_dir}.")
        return args.model_dir
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(args.repo, revision=args.revision, local_files_only=True)
    except Exception as e:  # noqa: BLE001 - LocalEntryNotFoundError and friends
        raise WorkerFailure("weights_missing", f"{args.repo}@{args.revision} is not in the Hugging Face cache "
                            f"({type(e).__name__}).") from None


def load_moss(args):
    model_dir = resolve_weights(args)
    note = None
    if args.compile and c_compiler() is None:
        note = "no C compiler found (cc, gcc or clang; or $CC): running without torch.compile (slower)"
        args.compile = False
    if not args.compile:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"  # must be set before torch is imported
    t = time.perf_counter()
    import torch
    if not torch.cuda.is_available():
        raise WorkerFailure("no_gpu", "torch sees no CUDA GPU (check `nvidia-smi` and the driver; CUDA 12.8 "
                            "needs driver 570 or newer).")
    from moss_soundeffect_v2 import MossSoundEffectPipeline
    pipe = MossSoundEffectPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device="cuda")
    if str(pipe.device) == "cpu":
        raise WorkerFailure("no_gpu", "the pipeline fell back to the CPU")
    load_seconds = time.perf_counter() - t

    def generate(prompt, seconds, seed, steps, cfg_scale, sigma_shift):
        audio = pipe(prompt=prompt, seconds=seconds, num_inference_steps=steps, cfg_scale=cfg_scale,
                     sigma_shift=sigma_shift, seed=seed, progress_bar_cmd=lambda it: it)
        return audio[0].float().cpu().numpy()

    def release():
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    warmup_seconds = None
    if args.compile:
        # Compile (or fetch from the kernel cache) now, so the load time includes it and the first clip
        # does not. The latent is always the full max_seconds, so a 2-step run sees the real shapes.
        t = time.perf_counter()
        generate("warm-up", 1.0, 0, 2, 4.0, 5.0)
        warmup_seconds = round(time.perf_counter() - t, 2)
        log(f"compile warm-up took {warmup_seconds}s")
    info = {"sample_rate": int(pipe.sample_rate), "max_seconds": float(pipe.max_inference_seconds),
            "compile": bool(args.compile), "compile_note": note, "load_seconds": round(load_seconds, 2),
            "warmup_seconds": warmup_seconds, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0), "model_dir": model_dir}
    return generate, release, info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", help="Hugging Face repo id (weights are taken from the local cache only)")
    p.add_argument("--revision", help="pinned revision of --repo")
    p.add_argument("--model-dir", help="a local model directory instead of --repo/--revision")
    p.add_argument("--compile", type=int, choices=(0, 1), default=1)
    args = p.parse_args(argv)
    if not args.model_dir and not (args.repo and args.revision):
        p.error("give --model-dir, or --repo and --revision")
    args.compile = bool(args.compile)
    inp, out = setup_channels()
    return serve(inp, out, lambda: load_moss(args))


if __name__ == "__main__":
    sys.exit(main())
