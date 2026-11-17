"""Batched generation benchmark: throughput, peak VRAM and output sanity at several batch sizes.

Usage: .venv/bin/python scripts/batch_bench.py <out_dir> [--batches 1,2,4,8] [--reps 2] [--voice demo_zh_female]

Loads its own copy of the model (through the project's adapter): stop the local-tts service first
(`systemctl --user stop local-tts`). A long Chinese article is split with the service's own splitter
(segments of up to 30 s, as a real job), and the segments are generated in order, B at a time, with
one registered voice as the reference for every segment (the case where segments are independent).
Timing covers processor + generate + codec decode (CUDA-synchronized), not WAV writing.

Writes per configuration (rep, B): every segment's WAV, the joined narration (`joined.wav`, with the
service's 0.35 s gaps) and one JSON line in `results.jsonl`; `source.txt` is the article, for
`scripts/asr_compare.py source.txt <out_dir>/*/joined.wav`.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from local_tts.adapters.moss_local import LANGUAGES, MossLocalAdapter
from local_tts.audio import encode, silence, trim_silence
from local_tts.config import load_registry
from local_tts.engine import MAX_DURATION_RATIO, MAX_DURATION_SLACK, MIN_DURATION_RATIO, SUSPECT_DURATION_RATIO
from local_tts.text import split_text

GiB = 2**30

ARTICLE = """茶的故事，要從一片葉子說起。

相傳在很久以前，人們發現某種樹葉放進熱水裡，會讓水變得清香，喝下去之後精神也比較好。最早，茶被當作藥材使用，用來解渴、提神，甚至治療一些小病。後來，喝茶慢慢成為日常生活的一部分，從宮廷流傳到民間，也從山區的小村落，走進了熱鬧的城市。

茶葉的種類很多，大致可以依照製作方式來區分。綠茶不經過發酵，保留了葉子原本的顏色和清新的味道。紅茶經過完整的發酵，顏色深，口感比較醇厚。烏龍茶介於兩者之間，屬於部分發酵，香氣變化特別豐富。還有白茶、黃茶和黑茶，各自有不同的工序與風味。

製茶的過程需要耐心。新鮮的茶葉採下來之後，要先攤開讓水分稍微散失，這個步驟叫做萎凋。接著依照茶的種類，進行搖青、殺青、揉捻和乾燥。每一個步驟的時間和溫度，都會影響最後的味道。經驗豐富的製茶師傅，往往只要聞一聞葉子的香氣，就知道下一步該怎麼做。

泡茶也有學問。水溫太高，容易把細嫩的綠茶燙出苦味；水溫太低，烏龍茶的香氣又出不來。一般來說，綠茶適合攝氏八十度左右的水，烏龍茶和紅茶則可以用接近沸騰的水。茶葉的份量、浸泡的時間，以及沖泡的次數，也都值得慢慢調整，找到自己最喜歡的味道。

除了味道，茶也承載了許多文化。在不同的地方，人們發展出各自的喝茶習慣。有人喜歡在早上泡一壺濃茶，配著點心慢慢享用；有人習慣在工作的空檔，泡一杯清淡的綠茶，讓自己放鬆一下。朋友來訪的時候，一起坐下來喝茶聊天，更是常見的待客方式。

近年來，茶飲的形式越來越多樣。除了傳統的熱茶，冷泡茶、奶茶和各種加了水果的茶飲，也吸引了許多年輕人。有些茶農開始嘗試新的品種和製作方法，希望在保留傳統的同時，也能做出符合現代口味的產品。

不過，好的茶葉離不開好的環境。茶樹喜歡溫暖、濕潤的氣候，以及排水良好的土壤。高山上的雲霧，能讓茶葉生長得比較慢，累積更多的香氣和甘味。因此，保護山林和水源，對茶產業來說也非常重要。

下一次當你拿起一杯茶的時候，不妨停下來想一想。這杯茶從山上的一片葉子開始，經過採摘、製作、運送，最後來到你的手中。每一口茶裡，都有土地、氣候和許多人的用心。慢慢喝，好好感受，這就是茶最迷人的地方。"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--voice", default="demo_zh_female")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.txt").write_text(ARTICLE + "\n", encoding="utf-8")

    free, total = torch.cuda.mem_get_info()
    if free < 16 * GiB:
        raise SystemExit(f"only {free / GiB:.1f} GiB of GPU memory free: stop the local-tts service first")

    registry = load_registry()
    entry = registry.models[registry.default_model]
    voice = registry.resolve_voice(args.voice, entry.id)
    adapter = MossLocalAdapter(entry)
    t = time.perf_counter()
    adapter.load()
    load_s = time.perf_counter() - t
    p, model = adapter.processor, adapter.model
    ref = adapter.voice_reference(voice)
    sampling = entry.options["sampling"]
    fps = adapter.frames_per_second
    cap = int(entry.options.get("max_new_tokens", 4096))
    pad_id = int(p.model_config.audio_pad_token_id)
    sr = int(p.model_config.sampling_rate)

    segments = split_text(ARTICLE, 30)
    texts = [s.text for s in segments]
    ests = [s.est_seconds for s in segments]
    budgets = [min(cap, max(125, math.ceil((e * 2.5 + 10) * fps))) for e in ests]
    print(f"model loaded in {load_s:.1f}s; {len(texts)} segments, estimates {[round(e, 1) for e in ests]}", flush=True)

    def run_chunk(idx: list[int], seed: int):
        msgs = [[p.build_user_message(text=texts[i], reference=[ref], language=LANGUAGES["zh"])] for i in idx]
        budget = max(budgets[i] for i in idx)
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            batch = p(msgs, mode="generation")
            outputs = model.generate(input_ids=batch["input_ids"].to("cuda"),
                                     attention_mask=batch["attention_mask"].to("cuda"),
                                     max_new_tokens=budget, do_sample=True, **sampling)
            messages = p.decode(outputs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        items = []
        for i, (start_length, gen), msg in zip(idx, outputs, messages):
            # Rows after the audio-start row and the prompt are generated frames; a row whose audio
            # channels are all padding came after this item finished (the batch kept going).
            codes = gen[1 + int(start_length):, 1:]
            frames = int((~codes.eq(pad_id).all(dim=1)).sum())
            wavs = [] if msg is None else [w for w in msg.audio_codes_list if w.numel()]
            audio = torch.cat(wavs, dim=-1).float().cpu().numpy() if wavs else np.zeros((2, 0), np.float32)
            items.append((i, frames, frames >= budgets[i], trim_silence(audio, sr)))
        return elapsed, budget, items

    # Warm-up (the first generation after a load is several times slower).
    run_chunk([0], 1)

    with (out / "results.jsonl").open("a", encoding="utf-8") as log:
        for rep in range(args.reps):
            order = batches if rep % 2 == 0 else batches[::-1]  # alternate order against thermal drift
            for b in order:
                d = out / f"rep{rep}_b{b}"
                d.mkdir(exist_ok=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                wall, chunks, segs = 0.0, [], {}
                for c, start in enumerate(range(0, len(texts), b)):
                    idx = list(range(start, min(start + b, len(texts))))
                    elapsed, budget, items = run_chunk(idx, seed=10_000 * rep + 100 * c + 7)
                    wall += elapsed
                    chunks.append({"segments": idx, "seconds": round(elapsed, 2), "budget_frames": budget,
                                   "frames": [f for _, f, _, _ in items]})
                    for i, frames, truncated, audio in items:
                        segs[i] = (frames, truncated, audio)
                        (d / f"seg{i:02d}.wav").write_bytes(encode(audio, sr, "wav"))
                parts, seg_info = [], []
                for i in range(len(texts)):
                    frames, truncated, audio = segs[i]
                    dur = audio.shape[1] / sr
                    est = ests[i]
                    flag = ("empty_or_dropped" if audio.shape[1] == 0 or (est > 2 and dur < MIN_DURATION_RATIO * est - 0.5)
                            else "run_on" if dur > MAX_DURATION_RATIO * est + MAX_DURATION_SLACK
                            else "suspect_short" if est > 2 and dur < SUSPECT_DURATION_RATIO * est else None)
                    seg_info.append({"index": i, "est": round(est, 2), "seconds": round(dur, 2),
                                     "ratio": round(dur / est, 3), "frames": frames, "truncated": truncated,
                                     "flag": flag})
                    if parts:
                        parts.append(silence(2, sr, 0.35))
                    parts.append(audio)
                joined = np.concatenate(parts, axis=1)
                (d / "joined.wav").write_bytes(encode(joined, sr, "wav"))
                audio_s = joined.shape[1] / sr
                row = {"rep": rep, "batch": b, "wall_seconds": round(wall, 2), "audio_seconds": round(audio_s, 2),
                       "rtf": round(wall / audio_s, 3),
                       "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GiB, 2),
                       "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GiB, 2),
                       "flags": [s["index"] for s in seg_info if s["flag"] or s["truncated"]],
                       "chunks": chunks, "segments": seg_info}
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                print(f"rep {rep} B={b}: {wall:.1f}s for {audio_s:.1f}s audio (RTF {row['rtf']}), "
                      f"peak {row['peak_allocated_gib']} GiB alloc / {row['peak_reserved_gib']} GiB reserved, "
                      f"flags {row['flags']}", flush=True)


if __name__ == "__main__":
    main()
