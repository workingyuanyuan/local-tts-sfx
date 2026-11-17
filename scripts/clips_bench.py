"""Clips-job acceptance: one clips job vs one job per clip, through the running service.

Usage: .venv/bin/python scripts/clips_bench.py <out_dir> [--cases voiced_single,voiced_clips,unvoiced_single,unvoiced_clips]
                                              [--voice demo_zh_female]

Pick a voice whose reference recording is not about the deck's topic: the model can copy the
reference's words into the output (demo_zh_female says 「太陽系八大行星之一」).

Needs the local-tts service (and its GPU) and nothing else running on it. A 12-slide Chinese deck is
synthesized as twelve ordinary jobs (one after another, as an agent would call generate_speech) and as
one clips job, with and without a voice. Writes per case the clips (<case>/<id>.wav), the clips joined
with 0.5 s gaps (<case>/joined.wav, for ASR and speaker checks), and one JSON line in results.jsonl.

Then:  scripts/asr_compare.py <out_dir>/source.txt <out_dir>/*/joined.wav
       scripts/speaker_consistency.py <out_dir>/*/joined.wav
"""
import argparse
import asyncio
import io
import json
import time
import wave
from pathlib import Path

import numpy as np

from local_tts.client import Deadline, TTSClient

SLIDES = [
    "太陽系的介紹。",
    "太陽系的中心是太陽，它是一顆會自己發光發熱的恆星。",
    "八大行星依照離太陽的距離排列，最近的是水星，最遠的是海王星。",
    "水星沒有大氣層保溫，白天非常熱，夜晚又非常冷。",
    "金星被厚厚的雲層包圍，是太陽系中表面溫度最高的行星。",
    "地球是目前已知唯一有生命存在的行星，表面大部分被海洋覆蓋。",
    "火星看起來是紅色的，因為它的土壤裡含有大量的氧化鐵。",
    "木星是體積最大的行星，它的大紅斑是一個持續了幾百年的風暴。",
    "土星有一圈非常明顯的光環，主要是由冰塊和岩石碎片組成。",
    "天王星的自轉軸幾乎是橫躺的，看起來像是側著身子在轉動。",
    "海王星離太陽最遠，常常吹著太陽系中最強的風。",
    "謝謝收看，下次我們再一起認識更多的星空故事。",
]
CLIPS = [(f"s{i + 1:02d}", t) for i, t in enumerate(SLIDES)]


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch)
    return pcm, sr


def join(parts: list[bytes]) -> bytes:
    arrays, sr = [], 48000
    for data in parts:
        pcm, sr = read_wav(data)
        if arrays:
            arrays.append(np.zeros((int(0.5 * sr), pcm.shape[1]), np.int16))
        arrays.append(pcm)
    pcm = np.concatenate(arrays)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


async def audio(c: TTSClient, job_id: str, clip=None) -> bytes:
    params = {"clip": clip} if clip else None
    return (await c._request("GET", f"/v1/jobs/{job_id}/audio", params=params, timeout=60)).content


async def run_case(c: TTSClient, name: str, out: Path, voice_name: str) -> dict:
    voice = voice_name if name.startswith("voiced") else None
    d = out / name
    d.mkdir(exist_ok=True)
    parts, warnings, retries, batch = [], [], 0, 1
    t = time.perf_counter()
    if name.endswith("_single"):
        for cid, text in CLIPS:
            job = await c.submit(text, voice=voice, language="zh")
            job = await c.wait(job["id"], Deadline(None))
            if job["status"] != "succeeded":
                raise SystemExit(f"{name} {cid}: {job.get('error')}")
            r = job["result"]
            warnings += r["warnings"]
            retries += r["retries"]
            parts.append(await audio(c, job["id"]))
    else:
        job = await c.submit(None, clips=CLIPS, voice=voice, language="zh")
        job = await c.wait(job["id"], Deadline(None))
        if job["status"] != "succeeded":
            raise SystemExit(f"{name}: {job.get('error')}")
        r = job["result"]
        warnings += r["warnings"] + [w for cl in r["clips"] for w in cl.get("warnings", [])]
        retries, batch = r["retries"], r["batch_size"]
        failed = [cl["id"] for cl in r["clips"] if cl["status"] != "succeeded"]
        if failed:
            raise SystemExit(f"{name}: clips failed: {failed}")
        for cid, _ in CLIPS:
            parts.append(await audio(c, job["id"], cid))
    wall = time.perf_counter() - t
    for (cid, _), data in zip(CLIPS, parts):
        (d / f"{cid}.wav").write_bytes(data)
    (d / "joined.wav").write_bytes(join(parts))
    seconds = sum(read_wav(p)[0].shape[0] / read_wav(p)[1] for p in parts)
    return {"case": name, "voice": voice, "wall_seconds": round(wall, 2), "audio_seconds": round(seconds, 2),
            "rtf": round(wall / seconds, 3), "retries": retries, "batch_size": batch, "warnings": warnings}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--cases", default="voiced_single,voiced_clips,unvoiced_single,unvoiced_clips")
    ap.add_argument("--voice", default="demo_zh_female")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.txt").write_text("\n".join(SLIDES) + "\n", encoding="utf-8")
    c = TTSClient()
    try:
        h = await c.health()
        if h["running_job"] or h["queue_length"]:
            raise SystemExit("the service is busy; run this when it is idle")
        warm = await c.submit("暖機。", language="zh")  # load + warm-up, not timed
        await c.wait(warm["id"], Deadline(None))
        with (out / "results.jsonl").open("a", encoding="utf-8") as log:
            for name in args.cases.split(","):
                row = await run_case(c, name, out, args.voice)
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        await c.aclose()


if __name__ == "__main__":
    asyncio.run(main())
