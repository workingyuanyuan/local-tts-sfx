"""Clips jobs through HTTP, the MCP tools and the CLI (fake adapter, no GPU)."""
import asyncio
import json

import mcp

from local_tts import cli, mcp_server
from tests.fakes import GATE
from tests.test_clients import error_of, tts  # noqa: F401 - fixture
from tests.test_server import wait_status

# The fake adapter returns 1 s of audio for any text: short text passes, long text "drops speech".
BAD = "這一頁的旁白特別長，" * 6 + "所以假的模型只生成一秒鐘的音訊，會被判定為漏唸而失敗。"


def call(tool, **args):
    async def go():
        async with mcp.Client(mcp_server.server) as c:
            return await c.call_tool(tool, args)
    return asyncio.run(go())


def clips_body(*texts, **extra):
    return {"clips": [{"id": f"s{i + 1}", "input": t} for i, t in enumerate(texts)], **extra}


# --- HTTP --------------------------------------------------------------------------------------------

def test_clips_job_over_http(client):
    GATE.set()
    job = client.post("/v1/jobs", json=clips_body("第一頁。", "第二頁。", language="zh")).json()
    assert job["clips_total"] == 2
    done = wait_status(client, job["id"], {"succeeded"})
    clips = done["result"]["clips"]
    assert [c["id"] for c in clips] == ["s1", "s2"] and all(c["status"] == "succeeded" for c in clips)
    assert clips[0]["duration"] == 1.0 and clips[0]["timeline"][0]["text"] == "第一頁。"
    assert done["result"]["duration"] == 2.0 and "timeline" not in done["result"]
    r = client.get(f"/v1/jobs/{job['id']}/audio", params={"clip": "s2"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.headers["X-TTS-Duration"] == "1.0"
    r = client.get(f"/v1/jobs/{job['id']}/audio")
    assert r.status_code == 400 and r.json()["error"]["code"] == "clip_required"
    assert r.json()["error"]["clips"] == ["s1", "s2"]
    r = client.get(f"/v1/jobs/{job['id']}/audio", params={"clip": "s9"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_clip"
    single = client.post("/v1/jobs", json={"input": "單一。"}).json()["id"]
    wait_status(client, single, {"succeeded"})
    assert client.get(f"/v1/jobs/{single}/audio", params={"clip": "s1"}).json()["error"]["code"] == "invalid_request"
    listed = {j["id"]: j for j in client.get("/v1/jobs").json()["data"]}
    assert listed[job["id"]]["clips"] == 2 and "clips" not in listed[single]


def test_one_failing_clip_over_http(client):
    GATE.set()
    job = client.post("/v1/jobs", json=clips_body("第一頁。", BAD)).json()["id"]
    done = wait_status(client, job, {"succeeded"})
    bad = done["result"]["clips"][1]
    assert bad["status"] == "failed" and bad["error"]["code"] == "segment_failed" and bad["error"]["clip"] == "s2"
    r = client.get(f"/v1/jobs/{job}/audio", params={"clip": "s2"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "clip_failed"
    job = client.post("/v1/jobs", json=clips_body(BAD, BAD)).json()["id"]
    failed = wait_status(client, job, {"failed"})
    assert failed["error"]["code"] == "segment_failed" and len(failed["error"]["clips"]) == 2


def test_clips_request_validation(client):
    def err(body, path="/v1/jobs"):
        r = client.post(path, json=body)
        assert r.status_code in (400, 413), r.text
        return r.json()["error"]
    assert err({"input": "a。", **clips_body("b。")})["code"] == "invalid_request"
    assert err({})["code"] == "invalid_request"
    assert err({"clips": []})["code"] == "invalid_request"
    assert err({"clips": [{"id": f"c{i}", "input": "a。"} for i in range(101)]})["code"] == "invalid_request"
    assert err({"clips": [{"id": "a", "input": "x。"}, {"id": "a", "input": "y。"}]})["code"] == "invalid_clip_id"
    assert err({"clips": [{"id": "../up", "input": "x。"}]})["code"] == "invalid_clip_id"
    assert err({"clips": [{"id": "a", "input": "  "}]})["code"] == "empty_input"
    assert err({"clips": [{"id": "a", "input": "字" * 15000}, {"id": "b", "input": "字" * 6000}]})["code"] == "input_too_long"
    assert err({"clips": [{"id": "a", "input": "x[pause 400s]"}, {"id": "b", "input": "y[pause 300s]"}]})["code"] \
        in ("invalid_pause",)
    e = err({"clips": [{"id": "a", "input": "x。", "voice": "v"}]})
    assert e["code"] == "unsupported_parameter" and e["supported"] == ["id", "input"]
    e = err(clips_body("x。"), path="/v1/audio/speech")
    assert e["code"] == "unsupported_parameter" and e["parameters"] == ["clips"]


# --- MCP ---------------------------------------------------------------------------------------------

def test_mcp_clips_completed_and_repeated(tts, tmp_path):
    GATE.set()
    out = tmp_path / "clips"
    r = call("generate_speech_clips", clips=[{"id": "intro", "text": "開場。"}, {"id": "end", "text": "結束。"}],
             output_dir=str(out), language="zh")
    assert not r.is_error, r.content
    res = r.structured_content
    assert res["status"] == "completed" and res["output_dir"] == str(out) and res["format"] == "wav"
    assert [(c["id"], c["status"], c["path"]) for c in res["clips"]] == [
        ("intro", "completed", str(out / "intro.wav")), ("end", "completed", str(out / "end.wav"))]
    assert (out / "intro.wav").exists() and (out / "end.wav").exists()
    assert "timeline" not in res["clips"][0] and "recovery" not in res
    again = call("get_speech_clips", job_id=res["job_id"], output_dir=str(out), timeline=True).structured_content
    assert again["status"] == "completed" and again["clips"][0]["timeline"][0]["text"] == "開場。"
    r = call("generate_speech_clips", clips=[{"id": "intro", "text": "再一次。"}], output_dir=str(out))
    assert r.is_error and error_of(r)["code"] == "output_exists"  # checked before submitting
    r = call("generate_speech_clips", clips=[{"id": "a b", "text": "x。"}], output_dir=str(out))
    assert error_of(r)["code"] == "invalid_clip_id"
    r = call("generate_speech_clips", clips=[{"id": "x", "text": "x。"}], output_dir="relative/dir")
    assert error_of(r)["code"] == "invalid_path"


def test_mcp_clips_pending_partial_and_recovery(tts, tmp_path):
    out = tmp_path / "deck"
    clips = [{"id": "s1", "text": "第一頁。"}, {"id": "s2", "text": BAD}, {"id": "s3", "text": "第三頁。"}]
    r = call("generate_speech_clips", clips=clips, output_dir=str(out), wait_seconds=0)
    pend = r.structured_content
    assert pend["status"] == "pending" and pend["clips_total"] == 3
    assert pend["next"] == {"tool": "get_speech_clips", "arguments": {"job_id": pend["job_id"], "output_dir": str(out)}}
    out.mkdir()
    (out / "s3.wav").write_bytes(b"someone else's file")
    GATE.set()
    res = call("get_speech_clips", **pend["next"]["arguments"]).structured_content
    by_id = {c["id"]: c for c in res["clips"]}
    assert res["status"] == "partial" and by_id["s1"]["status"] == "completed"
    assert by_id["s2"]["stage"] == "synthesis" and by_id["s2"]["error"]["code"] == "segment_failed"
    assert by_id["s3"]["stage"] == "delivery" and by_id["s3"]["error"]["code"] == "output_exists"
    assert "generate_speech_clips with only those clips" in res["recovery"]
    assert '"overwrite": true' in res["recovery"]
    first = (out / "s1.wav").stat().st_mtime_ns

    mcp_server.JOBS.clear()  # an MCP restart: the files already written are recognized, not rewritten
    res = call("get_speech_clips", job_id=pend["job_id"], output_dir=str(out), overwrite=True).structured_content
    assert {c["id"]: c["status"] for c in res["clips"]} == {"s1": "completed", "s2": "failed", "s3": "completed"}
    assert (out / "s3.wav").read_bytes()[:4] == b"RIFF"
    mcp_server.JOBS.clear()
    res = call("get_speech_clips", job_id=pend["job_id"], output_dir=str(out)).structured_content
    assert {c["id"]: c["status"] for c in res["clips"]}["s3"] == "completed"  # identical file: no output_exists
    assert (out / "s1.wav").stat().st_mtime_ns >= first


def test_mcp_wrong_fetch_tool_for_the_job_kind(tts, tmp_path):
    GATE.set()
    res = call("generate_speech_clips", clips=[{"id": "a", "text": "一。"}], output_dir=str(tmp_path / "c"))
    clips_job = res.structured_content["job_id"]
    r = call("get_speech_job", job_id=clips_job, output_path=str(tmp_path / "x.wav"))
    err = error_of(r)
    assert err["code"] == "clips_job" and "get_speech_clips" in err["recovery"]
    single = call("generate_speech", text="一。", output_path=str(tmp_path / "one.wav")).structured_content["job_id"]
    err = error_of(call("get_speech_clips", job_id=single, output_dir=str(tmp_path / "d")))
    assert err["code"] == "not_a_clips_job" and "get_speech_job" in err["recovery"]


# --- CLI ---------------------------------------------------------------------------------------------

def test_cli_clips(tts, tmp_path, capsys):
    GATE.set()
    spec = tmp_path / "clips.json"
    spec.write_text(json.dumps([{"id": "s1", "text": "第一頁。"}, {"id": "s2", "text": "第二頁。"}]), encoding="utf-8")
    out = tmp_path / "out"
    assert cli.main(["clips", str(spec), "-d", str(out), "-l", "zh"]) == 0
    text = capsys.readouterr().out
    assert "completed" in text and str(out / "s2.wav") in text and (out / "s1.wav").exists()

    spec.write_text(json.dumps([{"id": "ok", "text": "好。"}, {"id": "bad", "text": BAD}]), encoding="utf-8")
    assert cli.main(["--json", "clips", str(spec), "-d", str(out)]) == 4
    res = json.loads(capsys.readouterr().out)
    # Only a synthesis failure: fetching again cannot help, so no fetch_command (found on the GPU).
    assert res["status"] == "partial" and "fetch_command" not in res and "tts clips again" in res["recovery"]
    assert cli.main(["--json", "job", res["job_id"], "--wait", "-d", str(out)]) == 4
    assert json.loads(capsys.readouterr().out)["clips"][0]["status"] == "completed"
    (out / "ok.wav").write_bytes(b"changed")  # a clip that cannot be written: fetch, with --overwrite
    assert cli.main(["--json", "job", res["job_id"], "--wait", "-d", str(out)]) == 4
    res = json.loads(capsys.readouterr().out)
    assert res["clips"][0]["error"]["code"] == "output_exists" and res["fetch_command"].endswith("--overwrite")
    assert "tts clips again" in res["recovery"] and "fetch_command" in res["recovery"]
    assert cli.main(["job", res["job_id"], "--wait", "-d", str(out), "--overwrite"]) == 4
    text = capsys.readouterr().out
    assert "recovery:" in text and "tts job" not in text and (out / "ok.wav").read_bytes() != b"changed"

    spec.write_text("not json", encoding="utf-8")
    assert cli.main(["clips", str(spec), "-d", str(out)]) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_input"


# --- review regressions ------------------------------------------------------------------------------

def test_repeated_fetches_skip_identical_files_and_restore_missing_ones(tts, tmp_path, monkeypatch):
    GATE.set()
    out = tmp_path / "again"
    res = call("generate_speech_clips", clips=[{"id": "a", "text": "一。"}, {"id": "b", "text": "二。"}],
               output_dir=str(out)).structured_content
    downloads = []
    original = type(tts)._request

    async def spy(self, method, path, **kw):
        if path.endswith("/audio"):
            downloads.append(kw.get("params"))
        return await original(self, method, path, **kw)
    monkeypatch.setattr(type(tts), "_request", spy)
    args = {"job_id": res["job_id"], "output_dir": str(out)}
    assert call("get_speech_clips", **args).structured_content["status"] == "completed" and downloads == []
    (out / "a.wav").unlink()
    (out / "b.wav").write_bytes(b"changed")
    again = call("get_speech_clips", **args).structured_content
    by_id = {c["id"]: c for c in again["clips"]}
    assert by_id["a"]["status"] == "completed" and (out / "a.wav").read_bytes()[:4] == b"RIFF"
    assert by_id["b"]["error"]["code"] == "output_exists"
    assert call("get_speech_clips", **args, overwrite=True).structured_content["status"] == "completed"
    assert downloads == [{"clip": "a"}, {"clip": "b"}, {"clip": "b"}]


def test_clip_ids_with_a_trailing_newline_are_rejected(client, tmp_path):
    assert client.post("/v1/jobs", json={"clips": [{"id": "intro\n", "input": "x。"}]}).json()["error"]["code"] \
        == "invalid_clip_id"
    from local_tts.client import ClientError, clip_paths
    try:
        clip_paths(tmp_path, ["intro\n"], "wav")
    except ClientError as e:
        assert e.code == "invalid_clip_id"
    else:
        raise AssertionError("accepted")


def test_speech_endpoint_lists_its_own_parameters_for_clips(client):
    err = client.post("/v1/audio/speech", json={"input": "x。", "clips": []}).json()["error"]
    assert err["parameters"] == ["clips"] and "input" in err["supported"] and "voice" in err["supported"]


def test_cli_terminal_clips_failure_says_how_to_recover(tts, tmp_path, capsys):
    GATE.set()
    spec = tmp_path / "bad.json"
    spec.write_text(json.dumps([{"id": "x", "text": BAD}, {"id": "y", "text": BAD}]), encoding="utf-8")
    assert cli.main(["-q", "clips", str(spec), "-d", str(tmp_path / "o")]) == 1
    err = json.loads(capsys.readouterr().err)["error"]
    assert err["job_status"] == "failed" and "fetch_command" not in err and "tts clips again" in err["recovery"]
