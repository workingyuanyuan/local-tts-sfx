"""Drive the real tts-mcp over stdio against a running service: the sound-effect tools.
One effect, a 3-clip job followed through "pending" with exactly next.tool / next.arguments, a wrong-kind
getter (must fail with wrong_job_kind and write nothing) and an unknown argument (must be refused).
Needs the GPU and a server with the sound-effect worker installed (e.g. one started by scripts/sfx_run.sh).
Usage: sfx_mcp_stdio.py <out_dir> [port]   (port: $LOCAL_TTS_PORT, else 8791)
Writes <out_dir>/sfx_mcp_stdio.json; exits non-zero on failure."""
import asyncio
import json
import os
import sys
import time

import mcp

OUT = os.path.abspath(sys.argv[1])
PORT = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("LOCAL_TTS_PORT", "8791")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = mcp.StdioServerParameters(command=os.path.join(ROOT, ".venv/bin/tts-mcp"),
                                   env=dict(os.environ, LOCAL_TTS_URL=f"http://127.0.0.1:{PORT}"))


def error_of(r) -> dict:
    text = r.content[0].text
    return json.loads(text[text.index("{"):])["error"]


async def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    log = []
    try:
        return await run(log)
    except BaseException as e:
        log.append({"failure": f"{type(e).__name__}: {e}"})
        raise
    finally:  # keep this run's record even when it fails
        json.dump(log, open(f"{OUT}/sfx_mcp_stdio.json", "w"), ensure_ascii=False, indent=2)


async def follow(c, res: dict, log: list) -> dict:
    """Call next.tool with exactly next.arguments until the job is done."""
    while res["status"] == "pending":
        t = time.monotonic()
        r = await c.call_tool(res["next"]["tool"], res["next"]["arguments"])
        assert not r.is_error, r.content
        log.append({"call": res["next"]["tool"], "elapsed": round(time.monotonic() - t, 1), "result": r.structured_content})
        res = r.structured_content
    return res


async def run(log: list) -> int:
    async with mcp.Client(SERVER) as c:
        tools = [t.name for t in (await c.list_tools()).tools]
        log.append({"tools": tools})
        assert {"generate_sound_effect", "generate_sound_effect_clips", "get_sound_effect_job",
                "get_sound_effect_clips"} <= set(tools), tools

        t = time.monotonic()
        r = await c.call_tool("generate_sound_effect", {"prompt": "A glass bottle shatters on a tiled floor.",
                                                        "seconds": 2.5, "seed": 5,
                                                        "output_path": f"{OUT}/mcp_single.wav"})
        assert not r.is_error, r.content
        log.append({"call": "generate_sound_effect", "elapsed": round(time.monotonic() - t, 1),
                    "result": r.structured_content})
        single = await follow(c, r.structured_content, log)
        assert single["status"] == "completed" and abs(single["duration_seconds"] - 2.5) < 0.06, single

        t = time.monotonic()
        r = await c.call_tool("generate_sound_effect_clips", {
            "seconds": 2, "seed": 20, "output_dir": f"{OUT}/mcp_clips", "wait_seconds": 5, "clips": [
                {"id": "steps", "prompt": "Footsteps on a wooden floor, slow."},
                {"id": "wind", "prompt": "Strong wind through trees.", "seconds": 4},
                {"id": "click", "prompt": "A single mouse click.", "seconds": 0.5}]})
        assert not r.is_error, r.content
        res = r.structured_content
        log.append({"call": "generate_sound_effect_clips(wait 5)", "elapsed": round(time.monotonic() - t, 1),
                    "result": res})
        assert res["status"] == "pending" and res["next"]["tool"] == "get_sound_effect_clips", res
        res = await follow(c, res, log)
        assert res["status"] == "completed" and len(res["clips"]) == 3, res

        r = await c.call_tool("get_speech_job", {"job_id": single["job_id"], "output_path": f"{OUT}/wrong.wav"})
        log.append({"call": "get_speech_job(sfx job)", "error": error_of(r) if r.is_error else None})
        assert r.is_error and error_of(r)["code"] == "wrong_job_kind" and not os.path.exists(f"{OUT}/wrong.wav")
        r = await c.call_tool("generate_sound_effect", {"prompt": "x", "seconds": 1, "voice": "demo_en"})
        log.append({"call": "generate_sound_effect(voice=...)", "error": error_of(r) if r.is_error else None})
        assert r.is_error and error_of(r)["code"] == "unsupported_parameter"
    for e in log:
        print(json.dumps(e, ensure_ascii=False)[:300])
    print("ALL PASS")
    return 0


sys.exit(asyncio.run(main()))
