"""Sound effects through the client library, the CLI and the MCP server (fake models, no GPU, no worker),
and the agent-rules installer."""
import asyncio
import io
import json
import os
import subprocess
import wave
from pathlib import Path

import mcp
import pytest

from local_tts import cli, mcp_server
from local_tts.client import ClientError, Deadline
from tests.fakes import GATE, SFX_GATE, wait_sfx_started
from tests.test_clients import call, error_of, tts  # noqa: F401 - tts is a fixture

ROOT = Path(__file__).resolve().parents[1]


def frames(path: Path) -> tuple[int, int, int]:
    with wave.open(io.BytesIO(path.read_bytes())) as w:
        return w.getframerate(), w.getnchannels(), w.getnframes()


def run(coro):
    return asyncio.run(coro)


# --- client library ---------------------------------------------------------------------------------

def test_client_single_effect_and_clips(tts, tmp_path):
    async def go():
        job = await tts.submit_sfx("door slam", seconds=1.5, seed=4, deadline=Deadline(10))
        job = await tts.wait(job["id"], Deadline(10), last=job)
        one = await tts.save(job, tmp_path / "door.wav", False)
        cj = await tts.submit_sfx(None, seconds=1, seed=10, clips=[
            {"id": "a", "prompt": "rain"}, {"id": "b", "prompt": "wind", "seconds": 2, "seed": None}])
        cj = await tts.wait(cj["id"], Deadline(10), last=cj)
        many = await tts.save_clips(cj, tmp_path / "fx", False)
        with pytest.raises(ClientError) as e:
            await tts.save(cj, tmp_path / "x.wav", False)
        return one, many, e.value
    one, many, wrong = run(go())
    assert one["status"] == "completed" and one["kind"] == "sfx" and one["seed"] == 4
    assert one["seconds_requested"] == 1.5 and "voice" not in one and "segments" not in one
    assert frames(tmp_path / "door.wav") == (100, 1, 150)
    assert many["status"] == "completed" and many["kind"] == "sfx" and "voice" not in many
    assert [(c["id"], c["seconds_requested"], c["seed"]) for c in many["clips"]] == [("a", 1.0, 10), ("b", 2.0, 11)]
    assert frames(tmp_path / "fx" / "b.wav") == (100, 1, 200)
    assert wrong.code == "clips_job" and "get_sound_effect_clips" in wrong.message


# --- CLI ----------------------------------------------------------------------------------------------

def test_cli_sfx_writes_file_and_refuses_overwrite(tts, client, tmp_path, capsys):
    out = tmp_path / "fx" / "boom.wav"
    assert cli.main(["sfx", "an explosion far away", "-s", "2", "-o", str(out), "--seed", "3", "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["path"] == str(out) and r["kind"] == "sfx" and r["seed"] == 3 and frames(out) == (100, 1, 200)
    jobs = len(client.app.state.scheduler.jobs)
    assert cli.main(["sfx", "again", "-s", "2", "-o", str(out)]) == 1  # refused before submitting
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "output_exists"
    assert len(client.app.state.scheduler.jobs) == jobs
    assert cli.main(["sfx", "again", "-s", "2", "-o", str(out), "--overwrite"]) == 0
    assert "seed=" in capsys.readouterr().out  # human output names the seed, not a voice


def test_cli_sfx_seconds_is_required(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["sfx", "a door"])
    assert e.value.code == 2 and "--seconds" in capsys.readouterr().err


def test_cli_sfx_server_errors(tts, tmp_path, capsys):
    assert cli.main(["sfx", "x", "-s", "45", "-o", str(tmp_path / "x.wav")]) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_seconds"
    assert cli.main(["sfx", "x", "-s", "1", "-m", "moss-tts-local-v1.5", "-o", str(tmp_path / "x.wav")]) == 1
    e = json.loads(capsys.readouterr().err)["error"]
    assert e["code"] == "wrong_model_kind" and e["available"] == ["moss-sfx-v2.0"]


def test_cli_sfx_no_wait_then_job_fetch(tts, tmp_path, capsys):
    SFX_GATE.clear()
    out = tmp_path / "later.flac"
    assert cli.main(["sfx", "a bell", "-s", "1", "-o", str(out), "--no-wait", "--json"]) == 3
    pend = json.loads(capsys.readouterr().out)
    assert pend["status"] == "pending" and pend["fetch_command"].startswith(f"tts job {pend['job_id']} --wait -o")
    assert cli.main(["job", pend["job_id"], "--json"]) == 3  # status only: still running
    assert json.loads(capsys.readouterr().out)["kind"] == "sfx"
    SFX_GATE.set()
    assert cli.main(["job", pend["job_id"], "--wait", "-o", str(out), "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["format"] == "flac" and r["kind"] == "sfx" and out.exists()


def test_cli_sfx_clips_partial_and_recovery(tts, tmp_path, capsys):
    spec = tmp_path / "fx.json"
    spec.write_text(json.dumps([{"id": "door", "prompt": "door slam", "seconds": 1},
                                {"id": "bad", "prompt": "FAIL"}, {"id": "rain", "prompt": "rain", "seed": 9}]))
    d = tmp_path / "out"
    assert cli.main(["sfx-clips", str(spec), "-d", str(d), "-s", "2", "--json"]) == 4
    r = json.loads(capsys.readouterr().out)
    clips = {c["id"]: c for c in r["clips"]}
    assert r["status"] == "partial" and clips["bad"]["stage"] == "synthesis"
    assert clips["rain"]["seed"] == 9 and clips["rain"]["duration_seconds"] == 2.0
    assert "tts sfx-clips again" in r["recovery"] and "fetch_command" not in r
    assert sorted(p.name for p in d.iterdir()) == ["door.wav", "rain.wav"]
    # Every file exists now: the same submission is refused up front.
    spec.write_text(json.dumps([{"id": "door", "prompt": "door slam", "seconds": 1}]))
    assert cli.main(["sfx-clips", str(spec), "-d", str(d)]) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "output_exists"
    spec.write_text(json.dumps([{"id": "door", "prompt": "x", "voice": "v"}]))
    assert cli.main(["sfx-clips", str(spec), "-d", str(d)]) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_input"


def test_cli_lists_cancel_and_models_with_sound_effects(tts, client, tmp_path, capsys):
    SFX_GATE.clear()
    spec = tmp_path / "fx.json"
    spec.write_text(json.dumps([{"id": f"c{i}", "prompt": "tick"} for i in range(3)]))
    assert cli.main(["sfx-clips", str(spec), "-d", str(tmp_path / "c"), "-s", "1", "--no-wait", "--json"]) == 3
    jid = json.loads(capsys.readouterr().out)["job_id"]
    wait_sfx_started(1)
    assert cli.main(["jobs"]) == 0 and "sfx 3.0s" in capsys.readouterr().out
    assert cli.main(["cancel", jid]) == 0 and "sound effect being generated" in capsys.readouterr().out
    SFX_GATE.set()
    assert cli.main(["models"]) == 0
    assert "kind: sound effects, 0.1-30.0 s per clip" in capsys.readouterr().out


# --- MCP ----------------------------------------------------------------------------------------------

def test_mcp_tool_schemas():
    async def go():
        async with mcp.Client(mcp_server.server) as c:
            return {t.name: t.input_schema for t in (await c.list_tools()).tools}
    tools = asyncio.run(go())
    s = tools["generate_sound_effect"]
    assert s["required"] == ["prompt", "seconds"] and s["additionalProperties"] is False
    assert "voice" not in s["properties"] and "timeline" not in s["properties"]
    assert set(tools["get_sound_effect_job"]["properties"]) == {"job_id", "wait_seconds", "output_path", "overwrite"}
    clip = tools["generate_sound_effect_clips"]["$defs"]["SfxClip"]
    assert set(clip["properties"]) == {"id", "prompt", "seconds", "seed"} and clip["additionalProperties"] is False
    assert "generate_sound_effect_clips" in mcp_server.INSTRUCTIONS


def test_mcp_single_effect_pending_then_resume(tts, tmp_path):
    out = tmp_path / "fx" / "door.wav"
    SFX_GATE.clear()
    r = call("generate_sound_effect", prompt="door slam", seconds=1.2, output_path=str(out), seed=1,
             wait_seconds=0.3).structured_content
    assert r["status"] == "pending" and r["next"]["tool"] == "get_sound_effect_job"
    assert r["next"]["arguments"] == {"job_id": r["job_id"], "output_path": str(out)}
    SFX_GATE.set()
    done = call(r["next"]["tool"], **r["next"]["arguments"]).structured_content
    assert done["status"] == "completed" and done["kind"] == "sfx" and done["duration_seconds"] == 1.2
    assert frames(out) == (100, 1, 120)
    assert call("get_sound_effect_job", **r["next"]["arguments"]).structured_content == done  # idempotent


def test_mcp_clips_pending_partial_and_recovery(tts, tmp_path):
    d = tmp_path / "fx"
    SFX_GATE.clear()
    r = call("generate_sound_effect_clips", seconds=1, output_dir=str(d), wait_seconds=0.3, clips=[
        {"id": "door", "prompt": "door"}, {"id": "bad", "prompt": "FAIL"}]).structured_content
    assert r["status"] == "pending" and r["next"]["tool"] == "get_sound_effect_clips"
    SFX_GATE.set()
    res = call(r["next"]["tool"], **r["next"]["arguments"]).structured_content
    assert res["status"] == "partial" and [c["status"] for c in res["clips"]] == ["completed", "failed"]
    assert "generate_sound_effect_clips with only those clips" in res["recovery"]
    assert (d / "door.wav").exists() and not (d / "bad.wav").exists()


def test_mcp_wrong_kind_getters_write_nothing(tts, tmp_path):
    GATE.set()
    tmp_path = tmp_path / "work"  # the fixture's config files live in tmp_path itself
    tmp_path.mkdir()
    fx, speech = tmp_path / "fx.wav", tmp_path / "speech.wav"
    sj = call("generate_sound_effect", prompt="boom", seconds=1, output_path=str(fx)).structured_content["job_id"]
    tj = call("generate_speech", text="你好。", output_path=str(speech)).structured_content["job_id"]
    mcp_server.JOBS.clear()  # as after an MCP restart: the kind must come from the service
    for tool, jid, right, args in (("get_speech_job", sj, "get_sound_effect_job", {"output_path": str(tmp_path / "a.wav")}),
                                   ("get_speech_clips", sj, "get_sound_effect_job", {"output_dir": str(tmp_path / "d")}),
                                   ("get_sound_effect_job", tj, "get_speech_job", {"output_path": str(tmp_path / "b.wav")}),
                                   ("get_sound_effect_clips", tj, "get_speech_job", {"output_dir": str(tmp_path / "e")})):
        r = call(tool, job_id=jid, **args)
        e = error_of(r)
        assert r.is_error and e["code"] == "wrong_job_kind" and e["tool"] == right and right in e["recovery"], e
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fx.wav", "speech.wav"]


def test_mcp_sfx_errors_and_unknown_arguments(tts, tmp_path):
    out = tmp_path / "exists.wav"
    out.write_bytes(b"x")
    cases = [({"prompt": "x", "seconds": 1, "output_path": str(out)}, "output_exists"),
             ({"prompt": "x", "seconds": 31}, "invalid_seconds"),
             ({"prompt": "x", "seconds": 1, "model": "moss-tts-local-v1.5"}, "wrong_model_kind"),
             ({"prompt": "x", "seconds": 1, "voice": "demo_en"}, "unsupported_parameter"),
             ({"prompt": "x", "seconds": 1, "language": "zh"}, "unsupported_parameter")]
    for args, code in cases:
        r = call("generate_sound_effect", **args)
        assert r.is_error and error_of(r)["code"] == code, r.content[0].text
    r = call("get_sound_effect_job", job_id="abc", timeline=True)
    assert r.is_error and error_of(r)["parameters"] == ["timeline"]
    r = call("generate_sound_effect_clips", seconds=1, clips=[{"id": "a", "prompt": "x", "voice": "v"}])
    assert r.is_error  # extra field in a clip
    assert out.read_bytes() == b"x"


def test_mcp_lists_cover_sound_effects(tts, client, tmp_path):
    client.app.state.scheduler.cfg.idle_unload_seconds = 3600  # keep the model loaded for unload_model
    call("generate_sound_effect", prompt="click", seconds=0.5, output_path=str(tmp_path / "c.wav"))
    jobs = call("list_speech_jobs").structured_content["result"]
    assert jobs[0]["kind"] == "sfx" and jobs[0]["status"] == "succeeded"
    models = {m["id"]: m for m in call("list_models").structured_content["result"]}
    assert models["moss-sfx-v2.0"]["kind"] == "sfx" and models["moss-sfx-v2.0"]["sfx"]["max_seconds"] == 30.0
    assert models["moss-tts-local-v1.5"]["kind"] == "tts" and "sfx" not in models["moss-tts-local-v1.5"]
    assert call("tts_status").structured_content["default_sfx_model"] == "moss-sfx-v2.0"
    assert call("unload_model").structured_content["unloaded"] == "moss-sfx-v2.0"


# --- agent rules ----------------------------------------------------------------------------------------

def install(home: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, HOME=str(home))
    return subprocess.run(["bash", str(ROOT / "scripts" / "install_agent_rules.sh")], env=env,
                          capture_output=True, text=True, timeout=60)


def test_install_agent_rules_into_a_temporary_home(tmp_path):
    home = tmp_path / "home"
    agents = home / ".codex" / "AGENTS.md"
    agents.parent.mkdir(parents=True)
    agents.write_text("# My rules\n\nKeep this line.\n")
    assert install(home).returncode == 0
    for skill in ("narration", "sound-effects"):
        text = (home / ".claude" / "skills" / skill / "SKILL.md").read_text()
        assert text.startswith(f"---\nname: {skill}\n") and "{{LOCAL_TTS_ROOT}}" not in text
    first = agents.read_text()
    assert first.startswith("# My rules\n\nKeep this line.\n")
    for name in ("local-tts", "local-tts-sfx"):
        assert first.count(f"<!-- {name}:begin -->") == 1 and first.count(f"<!-- {name}:end -->") == 1
    assert first.index("<!-- local-tts:end -->") < first.index("<!-- local-tts-sfx:begin -->")
    assert "generate_sound_effect_clips" in first.split("<!-- local-tts-sfx:begin -->")[1]
    assert (agents.parent / "AGENTS.md.bak-local-tts").read_text() == "# My rules\n\nKeep this line.\n"
    agents.write_text(first + "\nAdded later.\n")
    assert install(home).returncode == 0  # idempotent: blocks replaced in place, the rest kept
    assert agents.read_text() == first + "\nAdded later.\n"


@pytest.mark.parametrize("broken", ["<!-- local-tts-sfx:begin -->\nhalf\n",
                                    "<!-- local-tts-sfx:end -->\nx\n<!-- local-tts-sfx:begin -->\n",
                                    "<!-- local-tts:begin -->\n<!-- local-tts:begin -->\n<!-- local-tts:end -->\n"])
def test_install_agent_rules_refuses_malformed_markers(tmp_path, broken):
    home = tmp_path / "home"
    agents = home / ".codex" / "AGENTS.md"
    agents.parent.mkdir(parents=True)
    agents.write_text("mine\n" + broken)
    r = install(home)
    assert r.returncode != 0 and "nothing was changed" in r.stderr
    assert agents.read_text() == "mine\n" + broken


# --- review regressions ------------------------------------------------------------------------------

def snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else b"<dir>") for p in sorted(root.rglob("*"))}


@pytest.mark.parametrize("broken", [
    # an unmatched sound-effects begin marker hidden inside the narration block
    "<!-- local-tts:begin -->\nold\n<!-- local-tts-sfx:begin -->\n<!-- local-tts:end -->\n",
    # two well-formed pairs that overlap
    "<!-- local-tts:begin -->\na\n<!-- local-tts-sfx:begin -->\nb\n<!-- local-tts:end -->\nc\n<!-- local-tts-sfx:end -->\n",
    # one pair nested inside the other
    "<!-- local-tts:begin -->\n<!-- local-tts-sfx:begin -->\n<!-- local-tts-sfx:end -->\n<!-- local-tts:end -->\n",
    "<!-- local-tts-sfx:begin -->\nhalf\n"])
def test_install_refusal_leaves_home_untouched(tmp_path, broken):
    home = tmp_path / "home"
    agents = home / ".codex" / "AGENTS.md"
    agents.parent.mkdir(parents=True)
    agents.write_text("mine\n" + broken)
    old_skill = home / ".claude" / "skills" / "narration" / "SKILL.md"
    old_skill.parent.mkdir(parents=True)
    old_skill.write_text("an older narration skill\n")
    before = snapshot(home)
    r = install(home)
    assert r.returncode != 0 and "nothing was changed" in r.stderr, r.stdout + r.stderr
    assert snapshot(home) == before  # no skill replaced, no backup created, AGENTS.md as it was


class HeldLock:
    """A job lock another call holds for longer than this call may wait."""

    async def acquire(self):
        await asyncio.Event().wait()

    def release(self):
        pass

    def locked(self):
        return True


def test_lock_timeout_path_checks_kind_and_shape(tts, tmp_path):
    jid = call("generate_sound_effect", prompt="boom", seconds=1,
               output_path=str(tmp_path / "fx.wav")).structured_content["job_id"]
    mcp_server.JOBS[jid].lock = HeldLock()
    r = call("get_speech_job", job_id=jid, timeline=True, wait_seconds=0.2, output_path=str(tmp_path / "s.wav"))
    assert r.is_error and error_of(r)["code"] == "wrong_job_kind" and error_of(r)["tool"] == "get_sound_effect_job"
    r = call("get_sound_effect_clips", job_id=jid, wait_seconds=0.2, output_dir=str(tmp_path / "d"))
    assert r.is_error and error_of(r)["code"] == "not_a_clips_job" and "get_sound_effect_job" in error_of(r)["recovery"]
    assert not (tmp_path / "s.wav").exists() and not (tmp_path / "d").exists()


def test_clips_getter_without_dir_reports_the_kind_first(tts, tmp_path):
    GATE.set()
    tj = call("generate_speech", text="你好。", output_path=str(tmp_path / "s.wav")).structured_content["job_id"]
    sj = call("generate_sound_effect", prompt="boom", seconds=1,
              output_path=str(tmp_path / "f.wav")).structured_content["job_id"]
    mcp_server.JOBS.clear()  # after an MCP restart: no remembered output_dir
    for tool, jid, right in (("get_sound_effect_clips", tj, "get_speech_job"),
                             ("get_speech_clips", sj, "get_sound_effect_job")):
        e = error_of(call(tool, job_id=jid))
        assert e["code"] == "wrong_job_kind" and e["tool"] == right, e


def test_non_finite_numbers_are_json_errors(tts, tmp_path, capsys):
    out = str(tmp_path / "x.wav")
    for args in (["-s", "nan"], ["-s", "inf"], ["-s", "1", "--cfg-scale", "inf"], ["-s", "1", "--cfg-scale", "nan"]):
        assert cli.main(["sfx", "x", "-o", out] + args) == 1, args
        assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_request"
    spec = tmp_path / "fx.json"
    spec.write_text('[{"id": "a", "prompt": "x", "seconds": NaN}]')
    assert cli.main(["sfx-clips", str(spec), "-d", str(tmp_path / "d")]) == 1
    e = json.loads(capsys.readouterr().err)["error"]
    assert e["code"] == "invalid_request" and "clips[0].seconds" in e["message"]
    # MCP: JSON has no Infinity, so the transport sends null and argument validation refuses it.
    r = call("generate_sound_effect", prompt="x", seconds=float("inf"), output_path=out)
    assert r.is_error and "seconds" in r.content[0].text
    assert not Path(out).exists()


def test_warnings_sentence_is_about_speech():
    line = next(x for x in mcp_server.INSTRUCTIONS.splitlines() if "warnings" in x)
    assert "speech" in line


def test_rule_texts_keep_the_timeline_and_fetch_unwritten_clips():
    for rel in ("agent-rules/claude/sound-effects/SKILL.md", "agent-rules/codex/AGENTS.local-tts-sfx.md"):
        text = (ROOT / rel).read_text()
        assert "duration=first" not in text and "duration=longest" in text and "normalize=0" in text
        assert '"delivery"' in text and "recovery" in text
    assert "`LOCAL_TTS_CACHE_DIR`，預設 `~/.cache/local-tts/torch-compile`" not in (ROOT / "AGENTS.md").read_text()
