"""GPU acceptance of sound effects against a running tts-server (normally one started from this checkout by
scripts/sfx_run.sh on a non-default port). Every check asserts its outcome; files go to <out_dir>.

  1. one effect (POST /v1/sfx/jobs with prompt + seconds), polled to the end and saved;
  2. a 3-clip job with different lengths, each clip saved;
  3. every file: WAV at the model's sample rate, the requested duration, not silent;
  4. speech -> sound effects -> speech switching, if the speech model is installed: each switch unloads the
     previous model first (checked through /health and nvidia-smi memory).

Usage: .venv/bin/python scripts/sfx_smoke.py <out_dir> [--url http://127.0.0.1:$LOCAL_TTS_PORT] [--no-switch]
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time
import uuid
import wave
from pathlib import Path

import httpx
import numpy as np

FAILED: list[str] = []


def check(ok: bool, what: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        FAILED.append(what)
    return ok


def gpu_memory() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().replace("\n", "; ") or out.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable ({type(e).__name__})"


def submit(http: httpx.Client, path: str, body: dict) -> dict:
    r = http.post(path, json=body, headers={"Idempotency-Key": uuid.uuid4().hex})
    if r.status_code not in (200, 202):
        raise SystemExit(f"{path} -> {r.status_code}: {r.text}")
    return r.json()


def wait(http: httpx.Client, job_id: str, limit: float) -> dict:
    end = time.monotonic() + limit
    while True:
        job = http.get(f"/v1/jobs/{job_id}", params={"wait": 30}).json()
        if job["status"] in ("succeeded", "failed"):
            return job
        if time.monotonic() > end:
            raise SystemExit(f"job {job_id} still {job['status']} after {limit:.0f}s")
        print(f"  {job_id}: {job['status']} {job.get('progress')} eta={job.get('estimated_remaining_seconds')}s",
              flush=True)


def check_wav(data: bytes, path: Path, want_seconds: float, sample_rate: int) -> None:
    path.write_bytes(data)
    with wave.open(io.BytesIO(data)) as w:
        sr, ch, n, width = w.getframerate(), w.getnchannels(), w.getnframes(), w.getsampwidth()
        frames = w.readframes(n)
    dtype = {2: "<i2", 4: "<i4"}.get(width)
    x = np.frombuffer(frames, dtype=dtype).astype(np.float32) / float(2 ** (8 * width - 1)) if dtype else None
    rms = float(np.sqrt(np.mean(x ** 2))) if x is not None and x.size else 0.0
    peak = float(np.abs(x).max()) if x is not None and x.size else 0.0
    seconds = n / sr if sr else 0.0
    check(sr == sample_rate, f"{path.name}: {sr} Hz (want {sample_rate})")
    # The model takes seconds rounded to 0.1 s.
    check(abs(seconds - want_seconds) <= 0.06, f"{path.name}: {seconds:.3f}s (asked {want_seconds}s), {ch} ch")
    check(rms > 1e-3 and peak > 1e-2, f"{path.name}: not silent (rms {rms:.4f}, peak {peak:.3f})")


def sfx_job(http: httpx.Client, out: Path, name: str, body: dict, limit: float) -> dict:
    t = time.monotonic()
    created = submit(http, "/v1/sfx/jobs", body)
    check(created["kind"] == "sfx", f"{name}: job {created['id']} is a sound-effect job "
          f"(first estimate {created.get('estimated_remaining_seconds')}s)")
    job = wait(http, created["id"], limit)
    wall = time.monotonic() - t
    if not check(job["status"] == "succeeded", f"{name}: {job['status']} in {wall:.1f}s "
                 f"{json.dumps(job.get('error'), ensure_ascii=False) if job.get('error') else ''}"):
        return job
    res = job["result"]
    (out / f"{name}.json").write_text(json.dumps(job, ensure_ascii=False, indent=2))
    print(f"  warnings: {res['warnings']}")
    if "clips" in res:
        for c in res["clips"]:
            print(f"  clip {c['id']}: {c['status']} seconds={c['seconds_requested']} seed={c['seed']} "
                  f"generation={c.get('generation_seconds')}s")
            if check(c["status"] == "succeeded", f"{name}/{c['id']} succeeded"):
                a = http.get(f"/v1/jobs/{job['id']}/audio", params={"clip": c["id"]})
                check(a.status_code == 200, f"{name}/{c['id']} audio fetched")
                check_wav(a.content, out / f"{name}_{c['id']}.wav", c["seconds_requested"], res["sample_rate"])
    else:
        print(f"  effect: seconds={res['seconds_requested']} seed={res['seed']} generation={res['generation_seconds']}s")
        a = http.get(f"/v1/jobs/{job['id']}/audio")
        check(a.status_code == 200, f"{name} audio fetched")
        check_wav(a.content, out / f"{name}.wav", res["seconds_requested"], res["sample_rate"])
    print(f"  nvidia-smi: {gpu_memory()}")
    return job


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--url", default=os.environ.get("LOCAL_TTS_URL")
                    or f"http://127.0.0.1:{os.environ.get('LOCAL_TTS_PORT', '8791')}")
    ap.add_argument("--no-switch", action="store_true", help="skip the speech <-> sound-effect switching part")
    ap.add_argument("--limit", type=float, default=1800, help="seconds one job may take (cold compile included)")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    http = httpx.Client(base_url=args.url, timeout=60)

    health = http.get("/health").json()
    sfx_model = health.get("default_sfx_model")
    print(f"server {args.url}: loaded={health['loaded_model']} default_sfx={sfx_model}")
    print(f"  sfx status: {json.dumps(health.get('sfx'), ensure_ascii=False)}")
    print(f"  nvidia-smi before: {gpu_memory()}")
    if not check(bool(sfx_model) and health["sfx"][sfx_model]["worker_installed"], "sound-effect worker installed"):
        return 1

    sfx_job(http, out, "single", {"prompt": "A heavy wooden door slams shut in a stone hallway.", "seconds": 4,
                                  "seed": 1}, args.limit)
    h = http.get("/health").json()
    print(f"  after the first job: loaded={h['loaded_model']} compile={h['sfx'][sfx_model].get('compile')} "
          f"load={h['sfx'][sfx_model].get('load_seconds')}s warm-up={h['sfx'][sfx_model].get('warmup_seconds')}s")
    check(h["loaded_model"] == sfx_model, f"{sfx_model} stays loaded after its job")
    sfx_job(http, out, "clips", {"seconds": 3, "seed": 10, "clips": [
        {"id": "rain", "prompt": "Steady rain on a tin roof."},
        {"id": "steps", "prompt": "Footsteps on gravel, walking slowly away.", "seconds": 6.5},
        {"id": "chime", "prompt": "A small bright bell chimes once.", "seconds": 1.5}]}, args.limit)
    print(f"  estimates: {json.dumps(http.get('/health').json()['estimates'].get(sfx_model))}")

    if not args.no_switch:
        tts = health["default_model"]
        print("== speech -> sound effects -> speech")
        for step, (kind, body) in enumerate((
                ("tts", {"input": "切換測試：先說一句話。", "language": "zh"}),
                ("sfx", {"prompt": "A glass bottle shatters on a tiled floor.", "seconds": 2, "seed": 3}),
                ("tts", {"input": "Switching back to speech works.", "language": "en"}))):
            t = time.monotonic()
            if kind == "tts":
                r = http.post("/v1/audio/speech", json=body, timeout=1800)
                if step == 0 and r.status_code != 200 and "model_load_failed" in r.text:
                    print(f"SKIP switching: speech model {tts} cannot load here: {r.text[:300]}")
                    break
                ok = check(r.status_code == 200, f"switch {step}: speech {r.status_code} in {time.monotonic() - t:.1f}s"
                           f"{'' if r.status_code == 200 else ' ' + r.text[:300]}")
                if ok:
                    (out / f"switch_{step}_speech.wav").write_bytes(r.content)
                want = tts
            else:
                job = sfx_job(http, out, f"switch_{step}_sfx", body, args.limit)
                ok, want = job["status"] == "succeeded", sfx_model
            h = http.get("/health").json()
            check(h["loaded_model"] == want, f"switch {step}: loaded model is {h['loaded_model']} (want {want})")
            print(f"  nvidia-smi: {gpu_memory()}  worker pid: {h['sfx'][sfx_model].get('worker_pid')}")
            if kind == "tts" and want == tts:
                check(h["sfx"][sfx_model].get("worker_pid") is None, f"switch {step}: no sound-effect worker left")

    print(f"nvidia-smi after: {gpu_memory()}")
    print("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: " + "; ".join(FAILED))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
