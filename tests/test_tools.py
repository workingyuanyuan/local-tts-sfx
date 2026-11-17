"""Unload, job list, long poll, voice reload and registration (fake adapter, no GPU)."""
import asyncio
import io
import json
import threading
import time

import httpx
import mcp
import numpy as np
import soundfile as sf

from local_tts import cli, mcp_server
from local_tts.client import Deadline, TTSClient
from tests.fakes import GATE
from tests.test_clients import error_of, tts  # noqa: F401 - fixture
from tests.test_server import wait_status


def wav_bytes(seconds: float = 4.0, sr: int = 48000, channels: int = 2, amp: float = 0.2) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    audio = np.stack([amp * np.sin(2 * np.pi * 220 * t)] * channels, axis=1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def put_voice(c, name, data=None, **params):
    return c.put(f"/v1/voices/{name}", params=params, content=wav_bytes() if data is None else data)


# --- unload ----------------------------------------------------------------------------------------

def test_unload_when_idle_and_refused_while_busy(client):
    r = client.post("/v1/models/unload")
    assert r.status_code == 200 and r.json()["unloaded"] is None
    GATE.set()
    job = client.post("/v1/jobs", json={"input": "載入一下。"}).json()
    wait_status(client, job["id"], {"succeeded"})
    assert client.get("/health").json()["loaded_model"] == "moss-tts-local-v1.5"
    assert client.post("/v1/models/unload").json()["unloaded"] == "moss-tts-local-v1.5"
    assert client.get("/health").json()["loaded_model"] is None

    GATE.clear()
    running = client.post("/v1/jobs", json={"input": "還在跑。"}).json()["id"]
    wait_status(client, running, {"running"})
    queued = client.post("/v1/jobs", json={"input": "排隊中。"}).json()["id"]
    r = client.post("/v1/models/unload")
    err = r.json()["error"]
    assert r.status_code == 409 and err["code"] == "model_busy"
    assert err["running_job"] == running and err["queue_length"] == 1 and "estimated_idle_seconds" in err
    GATE.set()
    wait_status(client, queued, {"succeeded"})
    assert client.get("/health").json()["loaded_model"] == "moss-tts-local-v1.5"  # the job still had its model


# --- job list --------------------------------------------------------------------------------------

def test_job_list_newest_first_with_text_preview(client):
    GATE.set()
    a = client.post("/v1/jobs", json={"input": "短的。"}).json()["id"]
    wait_status(client, a, {"succeeded"})
    long_text = "很長的句子。" * 30
    b = client.post("/v1/jobs", json={"input": long_text}).json()["id"]
    wait_status(client, b, {"succeeded", "failed"})  # the fake's 1 s of audio is "dropped speech" here
    jobs = client.get("/v1/jobs").json()["data"]
    assert [j["id"] for j in jobs] == [b, a]
    assert jobs[1]["text"] == "短的。" and jobs[0]["chars"] == len(long_text)
    assert len(jobs[0]["text"]) == 80 and jobs[0]["text"].endswith("…")
    assert jobs[0]["error_code"] == "segment_failed" and jobs[1]["status"] == "succeeded"
    assert jobs[1]["progress"] == {"segments_done": 1, "segments_total": 1}


# --- long poll and strict query parameters -----------------------------------------------------------

def test_long_poll_returns_on_completion_or_after_wait(client):
    job = client.post("/v1/jobs", json={"input": "等一下。"}).json()["id"]
    wait_status(client, job, {"running"})
    t = time.monotonic()
    j = client.get(f"/v1/jobs/{job}", params={"wait": 0.5}).json()
    waited = time.monotonic() - t
    assert j["status"] == "running" and 0.4 <= waited < 5
    threading.Timer(0.3, GATE.set).start()
    t = time.monotonic()
    j = client.get(f"/v1/jobs/{job}", params={"wait": 20}).json()
    assert j["status"] in ("running", "succeeded")  # completion, or progress changing first
    assert time.monotonic() - t < 10
    j = client.get(f"/v1/jobs/{job}", params={"wait": 20}).json() if j["status"] == "running" else j
    assert j["status"] == "succeeded"
    t = time.monotonic()
    client.get(f"/v1/jobs/{job}", params={"wait": 20})  # finished: answers at once
    assert time.monotonic() - t < 2


def test_query_parameters_are_checked(client):
    r = client.get("/v1/voices", params={"modle": "x"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_parameter"
    assert r.json()["error"]["parameters"] == ["modle"]
    job = client.post("/v1/jobs", json={"input": "查詢。"}).json()["id"]
    r = client.get(f"/v1/jobs/{job}", params={"wait": 31})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"
    assert client.get("/health", params={"x": 1}).json()["error"]["code"] == "unsupported_parameter"
    GATE.set()


def test_client_wait_does_not_spin_on_a_server_without_long_poll():
    hits = []

    def handler(request):
        hits.append(request.url.params.get("wait"))
        return httpx.Response(200, json={"id": "j", "status": "running", "format": "wav",
                                         "progress": {"segments_done": 0, "segments_total": 2}})
    c = TTSClient("http://x", transport=httpx.MockTransport(handler))

    async def go():
        try:
            return await c.wait("j", Deadline(2.5))
        finally:
            await c.aclose()
    assert asyncio.run(go())["status"] == "running"
    assert hits[0] is not None and 2 <= len(hits) <= 5  # about one request per second, as before


# --- voices: reload and registration ----------------------------------------------------------------

def test_user_voices_file_is_reloaded_and_errors_keep_the_last_good_set(client, user_voices):
    (user_voices / "mine.wav").write_bytes(wav_bytes())
    (user_voices / "voices.yaml").write_text(
        "voices:\n  mine:\n    description: test\n    reference_audio: mine.wav\n    models: [moss-tts-local-v1.5]\n")
    voices = {v["name"]: v for v in client.get("/v1/voices").json()["data"]}
    assert voices["mine"]["source"] == "user" and voices["demo_zh_female"]["source"] == "builtin"
    GATE.set()
    job = client.post("/v1/jobs", json={"input": "用我的聲音。", "voice": "mine"}).json()
    assert wait_status(client, job["id"], {"succeeded"})["voice"] == "mine"

    (user_voices / "voices.yaml").write_text("voices:\n  broken: {reference_audio: missing.wav}\n")
    assert "mine" in {v["name"] for v in client.get("/v1/voices").json()["data"]}
    health = client.get("/health").json()
    assert "missing.wav" in health["voice_config_error"]
    err = client.post("/v1/jobs", json={"input": "新的。", "voice": "broken"}).json()["error"]
    assert err["code"] == "unknown_voice" and "missing.wav" in err["voice_config_error"]

    (user_voices / "voices.yaml").write_text(
        "voices:\n  demo_en: {reference_audio: mine.wav, models: [moss-tts-local-v1.5]}\n")
    assert "already defined" in client.get("/health").json()["voice_config_error"]
    (user_voices / "voices.yaml").unlink()
    assert client.get("/health").json()["voice_config_error"] is None
    assert "mine" not in {v["name"] for v in client.get("/v1/voices").json()["data"]}


def test_register_replace_and_remove_a_voice(client, user_voices):
    r = put_voice(client, "narrator", description="旁白", language="Chinese")
    assert r.status_code == 201, r.json()
    body = r.json()
    assert body["replaced"] is False and body["language"] == "zh" and body["models"] == ["moss-tts-local-v1.5"]
    assert body["seconds"] == 4.0
    voices = {v["name"]: v for v in client.get("/v1/voices").json()["data"]}
    assert voices["narrator"] == {"name": "narrator", "description": "旁白", "language": "zh",
                                  "models": ["moss-tts-local-v1.5"], "source": "user"}
    first = sorted(p.name for p in user_voices.glob("narrator-*.wav"))
    assert len(first) == 1

    r = put_voice(client, "narrator")
    assert r.status_code == 409 and r.json()["error"]["code"] == "voice_exists"
    r = put_voice(client, "narrator", data=wav_bytes(5.0), replace="true")
    assert r.status_code == 200 and r.json()["replaced"] is True
    second = sorted(p.name for p in user_voices.glob("narrator-*.wav"))
    assert len(second) == 1 and second != first  # the old recording was deleted

    r = client.delete("/v1/voices/narrator")
    assert r.status_code == 200 and r.json() == {"name": "narrator", "removed": True}
    assert not list(user_voices.glob("narrator-*.wav"))
    assert client.delete("/v1/voices/narrator").json()["error"]["code"] == "unknown_voice"


def test_register_voice_validation(client):
    assert put_voice(client, "demo_en").json()["error"]["code"] == "voice_exists"
    assert client.delete("/v1/voices/demo_en").json()["error"]["code"] == "voice_not_removable"
    assert put_voice(client, "bad name").json()["error"]["code"] == "invalid_voice_name"
    assert put_voice(client, "short", data=wav_bytes(1.0)).json()["error"]["code"] == "invalid_audio"
    assert put_voice(client, "long", data=wav_bytes(31.0, sr=16000, channels=1)).json()["error"]["code"] == "invalid_audio"
    assert put_voice(client, "silent", data=wav_bytes(amp=0.0)).json()["error"]["code"] == "invalid_audio"
    assert put_voice(client, "junk", data=b"not audio" * 100).json()["error"]["code"] == "invalid_audio"
    assert put_voice(client, "x", models="nope").json()["error"]["code"] == "unknown_model"
    assert put_voice(client, "x", language="klingon").json()["error"]["code"] == "unsupported_language"
    assert put_voice(client, "x", colour="red").json()["error"]["code"] == "unsupported_parameter"
    assert client.post("/v1/jobs", json={"input": "a" * 300_000}).status_code == 413  # JSON limit unchanged


def test_replacing_a_voice_keeps_the_file_a_queued_job_needs(client, user_voices):
    assert put_voice(client, "queued_voice").status_code == 201
    old = next(user_voices.glob("queued_voice-*.wav"))
    blocker = client.post("/v1/jobs", json={"input": "先跑這個。"}).json()["id"]
    wait_status(client, blocker, {"running"})
    job = client.post("/v1/jobs", json={"input": "用舊的聲音。", "voice": "queued_voice"}).json()["id"]
    assert put_voice(client, "queued_voice", data=wav_bytes(6.0), replace="true").status_code == 200
    assert old.exists()  # still referenced by the queued job
    GATE.set()
    wait_status(client, job, {"succeeded"})


# --- clients -----------------------------------------------------------------------------------------

def call(tool, **args):
    async def go():
        async with mcp.Client(mcp_server.server) as c:
            return await c.call_tool(tool, args)
    return asyncio.run(go())


def test_mcp_tools_and_cli_for_voices_jobs_and_unload(tts, tmp_path, capsys):
    async def names():
        async with mcp.Client(mcp_server.server) as c:
            return {t.name for t in (await c.list_tools()).tools}
    assert {"register_voice", "unload_model", "list_speech_jobs"} <= asyncio.run(names())

    ref = tmp_path / "ref.wav"
    ref.write_bytes(wav_bytes())
    r = call("register_voice", name="mcp_voice", reference_audio_path=str(ref), description="d", language="zh")
    assert not r.is_error, r.content
    assert r.structured_content["name"] == "mcp_voice" and r.structured_content["replaced"] is False
    r = call("register_voice", name="rel", reference_audio_path="ref.wav")
    assert r.is_error and error_of(r)["code"] == "invalid_path"
    voices = call("list_voices").structured_content["result"]
    assert {"name": "mcp_voice", "description": "d", "language": "zh", "models": ["moss-tts-local-v1.5"],
            "source": "user"} in voices

    GATE.set()
    out = tmp_path / "v.wav"
    assert not call("generate_speech", text="測試。", output_path=str(out), voice="mcp_voice").is_error
    jobs = call("list_speech_jobs").structured_content["result"]
    assert jobs[0]["voice"] == "mcp_voice" and jobs[0]["status"] == "succeeded"
    r = call("unload_model")
    assert not r.is_error and r.structured_content["unloaded"] == "moss-tts-local-v1.5"

    assert cli.main(["--json", "add-voice", "cli_voice", str(ref), "--description", "c", "-l", "zh"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "cli_voice"
    assert cli.main(["jobs"]) == 0 and "mcp_voice" in capsys.readouterr().out
    assert cli.main(["unload"]) == 0 and "no model was loaded" in capsys.readouterr().out
    assert cli.main(["remove-voice", "cli_voice"]) == 0 and "removed" in capsys.readouterr().out
    assert cli.main(["remove-voice", "cli_voice"]) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "unknown_voice"


# --- review regressions ------------------------------------------------------------------------------

def test_long_compressed_upload_is_rejected_before_decoding(client):
    buf = io.BytesIO()
    sf.write(buf, np.zeros((48000 * 600, 1), np.float32), 48000, format="FLAC")  # 10 min, a few KB
    assert len(buf.getvalue()) < 1_000_000
    err = put_voice(client, "bomb", data=buf.getvalue()).json()["error"]
    assert err["code"] == "invalid_audio" and err["seconds"] == 600.0


def test_removing_a_voice_keeps_a_recording_another_voice_uses(client, user_voices):
    (user_voices / "shared.wav").write_bytes(wav_bytes())
    (user_voices / "voices.yaml").write_text(
        "voices:\n  a: {reference_audio: shared.wav, models: [moss-tts-local-v1.5]}\n"
        "  b: {reference_audio: shared.wav, models: [moss-tts-local-v1.5]}\n")
    assert client.delete("/v1/voices/a").status_code == 200
    assert (user_voices / "shared.wav").exists()
    assert "b" in {v["name"] for v in client.get("/v1/voices").json()["data"]}


def test_a_failed_reload_is_retried_once_the_cause_is_fixed(client, user_voices, monkeypatch):
    (user_voices / "voices.yaml").write_text(
        "voices:\n  later: {reference_audio: later.wav, models: [moss-tts-local-v1.5]}\n")
    assert "later.wav" in client.get("/health").json()["voice_config_error"]
    (user_voices / "later.wav").write_bytes(wav_bytes())  # the YAML itself is unchanged
    assert client.get("/health").json()["voice_config_error"] is None
    assert "later" in {v["name"] for v in client.get("/v1/voices").json()["data"]}

    from local_tts import config as config_module

    def denied():
        raise PermissionError("no access")
    monkeypatch.setattr(config_module, "_voice_stamp", denied)
    health = client.get("/health").json()
    assert "PermissionError" in health["voice_config_error"]
    assert "later" in {v["name"] for v in client.get("/v1/voices").json()["data"]}  # last good set


def test_generation_endpoints_reject_unknown_query_parameters(client):
    for path in ("/v1/jobs", "/v1/audio/speech"):
        r = client.post(path, params={"voiec": "x"}, json={"input": "測試。"})
        assert r.status_code == 400 and r.json()["error"]["parameters"] == ["voiec"]
    assert client.get("/v1/jobs").json()["data"] == []  # nothing was queued
