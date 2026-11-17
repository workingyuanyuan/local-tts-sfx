"""Sound-effect jobs over HTTP with a fake sound-effect model (no worker, no GPU)."""
import io
import time
import wave

import pytest

from local_tts import config as config_module
from local_tts.server import scheduler as scheduler_module
from tests.fakes import (GATE, MODEL_EVENTS, SFX_CALLS, SFX_FAIL_LOAD, SFX_FAIL_RELEASE, SFX_GATE, SFX_HOLD,
                         wait_sfx_started)
from tests.test_server import wait_status

SFX = "moss-sfx-v2.0"
TTS = "moss-tts-local-v1.5"


def sfx(c, body, key=None):
    return c.post("/v1/sfx/jobs", json=body, headers={"Idempotency-Key": key} if key else {})


def err(r, status=400):
    assert r.status_code == status, r.text
    return r.json()["error"]


def wav_info(data: bytes) -> tuple[int, int, int]:
    with wave.open(io.BytesIO(data)) as w:
        return w.getframerate(), w.getnchannels(), w.getnframes()


# --- single effect and clips ------------------------------------------------------------------------

def test_single_effect(client):
    prompt = "A door slams. Then silence.\n\nFootsteps [pause 1s] fade."
    r = sfx(client, {"prompt": prompt, "seconds": 1.5, "seed": 7, "num_inference_steps": 50, "cfg_scale": 3.5})
    assert r.status_code == 202
    job = r.json()
    assert job["kind"] == "sfx" and job["model"] == SFX and "voice" not in job and "clips_total" not in job
    done = wait_status(client, job["id"], {"succeeded"})
    res = done["result"]
    assert res["kind"] == "sfx" and res["duration"] == 1.5 and res["seed"] == 7 and res["channels"] == 1
    assert res["sample_rate"] == 100 and res["seconds_requested"] == 1.5 and res["warnings"] == []
    assert done["progress"] == {"clips_done": 1, "clips_total": 1}
    # One generation with the prompt exactly as given: no splitting, pause markers or retries.
    assert SFX_CALLS == [{"prompt": prompt, "seconds": 1.5, "seed": 7, "steps": 50, "cfg_scale": 3.5, "clip": None}]
    a = client.get(f"/v1/jobs/{job['id']}/audio")
    assert a.status_code == 200 and a.headers["content-type"] == "audio/wav"
    assert a.headers["X-TTS-Kind"] == "sfx" and a.headers["X-TTS-Seed"] == "7" and a.headers["X-TTS-Duration"] == "1.5"
    assert wav_info(a.content) == (100, 1, 150)
    assert err(client.get(f"/v1/jobs/{job['id']}/audio", params={"clip": "effect"}))["code"] == "invalid_request"
    listed = {j["id"]: j for j in client.get("/v1/jobs").json()["data"]}[job["id"]]
    assert listed["kind"] == "sfx" and listed["seconds"] == 1.5 and listed["voice"] is None and "clips" not in listed


def test_seed_is_drawn_and_reported_when_not_given(client):
    jid = sfx(client, {"prompt": "rain", "seconds": 1}).json()["id"]
    res = wait_status(client, jid, {"succeeded"})["result"]
    assert isinstance(res["seed"], int) and SFX_CALLS[0]["seed"] == res["seed"]
    assert SFX_CALLS[0]["steps"] is None and SFX_CALLS[0]["cfg_scale"] is None  # model defaults


def test_clips_with_one_failing_clip(client):
    body = {"seconds": 2, "seed": 100, "response_format": "wav", "clips": [
        {"id": "door", "prompt": "door slam", "seconds": 1},
        {"id": "bad", "prompt": "FAIL please"},
        {"id": "rain", "prompt": "rain on a roof", "seed": 5}]}
    job = sfx(client, body).json()
    assert job["clips_total"] == 3
    done = wait_status(client, job["id"], {"succeeded"})
    clips = {c["id"]: c for c in done["result"]["clips"]}
    assert clips["door"]["status"] == "succeeded" and clips["door"]["duration"] == 1.0 and clips["door"]["seed"] == 100
    assert clips["bad"]["status"] == "failed" and clips["bad"]["error"]["code"] == "sfx_generation_failed"
    assert clips["bad"]["error"]["clip"] == "bad" and clips["bad"]["seed"] == 101
    assert clips["rain"]["duration"] == 2.0 and clips["rain"]["seed"] == 5  # inherits seconds, own seed
    assert done["result"]["duration"] == 3.0 and [c["clip"] for c in SFX_CALLS] == ["door", "bad", "rain"]
    r = client.get(f"/v1/jobs/{job['id']}/audio", params={"clip": "rain"})
    assert r.status_code == 200 and wav_info(r.content) == (100, 1, 200)
    assert err(client.get(f"/v1/jobs/{job['id']}/audio", params={"clip": "bad"}), 409)["code"] == "clip_failed"
    assert err(client.get(f"/v1/jobs/{job['id']}/audio"))["code"] == "clip_required"
    assert err(client.get(f"/v1/jobs/{job['id']}/audio", params={"clip": "x"}), 404)["code"] == "unknown_clip"

    every = sfx(client, {"seconds": 1, "clips": [{"id": "a", "prompt": "FAIL"}, {"id": "b", "prompt": "FAIL"}]})
    failed = wait_status(client, every.json()["id"], {"failed"})
    assert failed["error"]["code"] == "sfx_generation_failed" and len(failed["error"]["clips"]) == 2


# --- validation ----------------------------------------------------------------------------------------

def test_validation_errors(client):
    def code(body, status=400):
        return err(sfx(client, body), status)

    assert code({"prompt": "x"})["code"] == "invalid_seconds"  # no silent default
    assert code({"clips": [{"id": "a", "prompt": "x"}]})["code"] == "invalid_seconds"
    for bad in (0, -1, 30.5, 1e9, 0.04):
        e = code({"prompt": "x", "seconds": bad})
        assert e["code"] == "invalid_seconds" and e["max_seconds"] == 30.0
    r = client.post("/v1/sfx/jobs", content=b'{"prompt": "x", "seconds": NaN}', headers={"Content-Type": "application/json"})
    assert err(r)["code"] == "invalid_seconds"
    assert code({"prompt": "x", "seconds": "5"})["code"] == "invalid_request"  # no string coercion
    assert code({"prompt": "x", "seconds": True})["code"] == "invalid_request"
    for field, value in (("voice", "v"), ("language", "zh"), ("timeline", True), ("input", "hi"), ("speed", 1)):
        e = code({"prompt": "x", "seconds": 1, field: value})
        assert e["code"] == "unsupported_parameter" and e["parameters"] == [field] and "seconds" in e["supported"]
    e = code({"seconds": 1, "clips": [{"id": "a", "prompt": "x", "voice": "v"}]})
    assert e["code"] == "unsupported_parameter" and e["supported"] == ["id", "prompt", "seconds", "seed"]
    assert code({"prompt": "x", "seconds": 1, "clips": [{"id": "a", "prompt": "y"}]})["code"] == "invalid_request"
    assert code({"seconds": 1})["code"] == "invalid_request"
    assert code({"prompt": "  ", "seconds": 1})["code"] == "empty_input"
    assert code({"prompt": "x", "seconds": 1, "response_format": "ogg"})["code"] == "unsupported_format"
    assert code({"prompt": "x", "seconds": 1, "num_inference_steps": 0})["code"] == "invalid_request"
    assert code({"prompt": "x", "seconds": 1, "num_inference_steps": 2.5})["code"] == "invalid_request"
    assert code({"prompt": "x", "seconds": 1, "cfg_scale": 50})["code"] == "invalid_request"
    assert code({"prompt": "x", "seconds": 1, "seed": -1})["code"] == "invalid_request"
    assert code({"seconds": 1, "clips": [{"id": "../x", "prompt": "y"}]})["code"] == "invalid_clip_id"
    assert code({"seconds": 1, "clips": [{"id": "a", "prompt": "y"}, {"id": "a", "prompt": "z"}]})["code"] \
        == "invalid_clip_id"
    assert sfx(client, {"prompt": "x", "seconds": 30}).status_code == 202  # the maximum itself is fine
    assert client.get("/v1/jobs").json()["data"][0]["kind"] == "sfx"
    end = time.time() + 5
    while not SFX_CALLS and time.time() < end:
        time.sleep(0.02)
    assert [c["seconds"] for c in SFX_CALLS] == [30.0]  # nothing invalid reached the model


def test_limits(client):
    lim = config_module.SfxLimits()
    e = err(sfx(client, {"prompt": "x" * (lim.max_prompt_chars + 1), "seconds": 1}))
    assert e["code"] == "input_too_long" and e["max_chars"] == lim.max_prompt_chars
    e = err(sfx(client, {"seconds": 1, "clips": [{"id": "a", "prompt": "x" * (lim.max_prompt_chars + 1)}]}))
    assert e["code"] == "input_too_long" and e["clips"] == ["a"]
    e = err(sfx(client, {"seconds": 1, "clips": [{"id": f"c{i}", "prompt": "x"} for i in range(lim.max_clips + 1)]}))
    assert e["code"] == "invalid_request" and e["max_clips"] == lim.max_clips
    e = err(sfx(client, {"seconds": 30, "clips": [{"id": f"c{i}", "prompt": "x"} for i in range(11)]}))
    assert e["code"] == "too_many_seconds" and e["max_total_seconds"] == lim.max_total_seconds
    assert err(sfx(client, {"seconds": 1, "clips": []}))["code"] == "invalid_request"


def test_limits_come_from_server_yaml(monkeypatch, tmp_path):
    (tmp_path / "server.yaml").write_text("sfx:\n  max_clips: 3\n  max_total_seconds: 10\n")
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    cfg = config_module.load_server_config()
    assert (cfg.sfx.max_clips, cfg.sfx.max_total_seconds, cfg.sfx.max_prompt_chars) == (3, 10, 2000)
    (tmp_path / "server.yaml").write_text("sfx:\n  max_clips: 0\n")
    with pytest.raises(ValueError, match="max_clips"):
        config_module.load_server_config()


def test_wrong_model_kind_both_ways(client):
    e = err(sfx(client, {"prompt": "x", "seconds": 1, "model": TTS}))
    assert e["code"] == "wrong_model_kind" and e["model_kind"] == "tts" and e["available"] == [SFX]
    for path, body in (("/v1/jobs", {"input": "你好。", "model": SFX}),
                       ("/v1/audio/speech", {"input": "你好。", "model": SFX})):
        e = err(client.post(path, json=body))
        assert e["code"] == "wrong_model_kind" and e["model_kind"] == "sfx" and e["available"] == [TTS]
    e = err(sfx(client, {"prompt": "x", "seconds": 1, "model": "nope"}), 404)
    assert e["code"] == "unknown_model" and e["available"] == [SFX]
    assert MODEL_EVENTS == []  # nothing was loaded for any of them


def test_no_sound_effect_model_configured(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from local_tts.adapters import ADAPTERS
    from local_tts.server import app as app_module
    monkeypatch.setitem(ADAPTERS, "moss_local", "tests.fakes:GatedAdapter")
    (tmp_path / "models.yaml").write_text(
        "default: m\nmodels:\n  m:\n    adapter: moss_local\n    max_new_tokens: 100\n    sampling: {}\n")
    (tmp_path / "voices.yaml").write_text("voices: {}\n")
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    with TestClient(app_module.build_app()) as c:
        e = err(sfx(c, {"prompt": "x", "seconds": 1}))
        assert e["code"] == "no_sfx_model" and e["available"] == []
        assert c.get("/health").json()["sfx"] == {}


# --- idempotency, cancellation, switching ---------------------------------------------------------------

def test_idempotency_key_is_per_kind(client):
    GATE.set()
    h = {"Idempotency-Key": "same-key"}
    speech = client.post("/v1/jobs", json={"input": "你好。"}, headers=h).json()
    effect = sfx(client, {"prompt": "x", "seconds": 1}, key="same-key").json()
    assert speech["id"] != effect["id"] and speech["kind"] == "tts" and effect["kind"] == "sfx"
    assert sfx(client, {"prompt": "x", "seconds": 1}, key="same-key").json()["id"] == effect["id"]
    assert client.post("/v1/jobs", json={"input": "你好。"}, headers=h).json()["id"] == speech["id"]
    assert len(client.app.state.scheduler.jobs) == 2
    assert err(sfx(client, {"prompt": "x", "seconds": 1}, key="k" * 200))["code"] == "invalid_request"


def test_cancel_queued_and_mid_job(client):
    SFX_GATE.clear()
    running = sfx(client, {"seconds": 1, "clips": [{"id": f"c{i}", "prompt": "x"} for i in range(3)]}).json()["id"]
    wait_sfx_started(1)  # "running" alone may still be loading, before any generation
    queued = sfx(client, {"prompt": "y", "seconds": 1}).json()["id"]
    r = client.delete(f"/v1/jobs/{queued}")
    assert r.status_code == 200 and r.json()["error"]["code"] == "cancelled"
    r = client.delete(f"/v1/jobs/{running}")
    assert r.status_code == 202 and r.json()["cancel_requested"] is True
    SFX_GATE.set()  # the clip in progress finishes and is discarded; the others never start
    failed = wait_status(client, running, {"failed"})
    assert failed["error"]["code"] == "cancelled" and len(SFX_CALLS) == 1
    assert client.get(f"/v1/jobs/{running}/audio", params={"clip": "c0"}).json()["error"]["code"] == "cancelled"


def test_switching_between_speech_and_sound_effects_unloads_first(client):
    GATE.set()
    client.app.state.scheduler.cfg.idle_unload_seconds = 3600
    assert client.post("/v1/audio/speech", json={"input": "先說話。"}).status_code == 200
    wait_status(client, sfx(client, {"prompt": "boom", "seconds": 1}).json()["id"], {"succeeded"})
    assert client.get("/health").json()["loaded_model"] == SFX
    assert client.post("/v1/audio/speech", json={"input": "再說話。"}).status_code == 200
    assert MODEL_EVENTS == [("load", TTS), ("unload", TTS), ("load", SFX), ("unload", SFX), ("load", TTS)]


def test_worker_crash_and_load_failure_leave_the_service_usable(client):
    client.app.state.scheduler.cfg.idle_unload_seconds = 3600
    job = sfx(client, {"seconds": 1, "clips": [{"id": "a", "prompt": "ok"}, {"id": "b", "prompt": "CRASH"},
                                               {"id": "c", "prompt": "never reached"}]}).json()["id"]
    e = wait_status(client, job, {"failed"})["error"]
    assert e["code"] == "sfx_worker_failed" and e["clip"] == "b" and len(SFX_CALLS) == 2
    assert client.get("/health").json()["loaded_model"] is None  # truthful: the worker is gone
    wait_status(client, sfx(client, {"prompt": "again", "seconds": 1}).json()["id"], {"succeeded"})
    assert MODEL_EVENTS == [("load", SFX), ("load", SFX)]

    client.post("/v1/models/unload")
    SFX_FAIL_LOAD["on"] = True
    e = wait_status(client, sfx(client, {"prompt": "x", "seconds": 1}).json()["id"], {"failed"})["error"]
    assert e["code"] == "sfx_worker_failed" and "Failed to load model" in e["message"]
    SFX_FAIL_LOAD["on"] = False
    wait_status(client, sfx(client, {"prompt": "x", "seconds": 1}).json()["id"], {"succeeded"})
    GATE.set()
    assert client.post("/v1/audio/speech", json={"input": "還能說話。"}).status_code == 200


# --- estimates, health, models -------------------------------------------------------------------------

def test_estimates_use_the_per_clip_figure(client):
    sched = client.app.state.scheduler
    sched.cfg.idle_unload_seconds = 3600
    clip_prior, load_prior = scheduler_module.DEFAULT_SFX_CLIP_SECONDS, scheduler_module.DEFAULT_SFX_LOAD_SECONDS
    h = client.get("/health").json()["estimates"][SFX]
    assert h == {"kind": "sfx", "clip_seconds": clip_prior, "clip_measured": False,
                 "load_seconds": load_prior, "load_measured": False}
    SFX_GATE.clear()
    # Long prompts and long clips do not matter: three clips are three clip-times.
    running = sfx(client, {"seconds": 30, "clips": [{"id": f"c{i}", "prompt": "word " * 300} for i in range(3)]}).json()
    wait_sfx_started(1)  # loaded and generating: no load time left in the estimate
    assert client.get(f"/v1/jobs/{running['id']}").json()["estimated_remaining_seconds"] == 3 * clip_prior
    queued = sfx(client, {"prompt": "x", "seconds": 1}).json()
    assert queued["estimated_remaining_seconds"] == 4 * clip_prior
    speech = client.post("/v1/jobs", json={"input": "排在音效後面。"}).json()
    est = speech["estimated_remaining_seconds"]
    assert est > 4 * clip_prior + scheduler_module.DEFAULT_LOAD_SECONDS  # plus switching back to speech
    assert sched.get(running["id"]).est_seconds == 0.0  # no speech-style estimate for effects
    SFX_GATE.set()
    GATE.set()
    for j in (running["id"], queued["id"], speech["id"]):
        wait_status(client, j, {"succeeded"})
    h = client.get("/health").json()["estimates"]
    assert h[SFX]["clip_measured"] is True and h[SFX]["clip_seconds"] < clip_prior and h[SFX]["load_measured"]
    assert h[TTS]["rtf_measured"] is False  # speech figures are separate


def test_health_and_models_describe_the_sound_effect_model(client):
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    m = models[SFX]
    assert m["kind"] == "sfx" and m["default"] is True and m["loaded"] is False
    assert m["options"] == ["cfg_scale", "num_inference_steps", "seed"] and m["languages"] == []
    assert m["sfx"]["max_seconds"] == 30.0 and m["sfx"]["channels"] == 1
    assert models[TTS]["kind"] == "tts" and models[TTS]["default"] is True and "sfx" not in models[TTS]
    h = client.get("/health").json()
    assert h["default_sfx_model"] == SFX and h["sfx"][SFX]["loaded"] is False
    wait_status(client, sfx(client, {"prompt": "x", "seconds": 1}).json()["id"], {"succeeded"})
    assert client.get("/health").json()["sfx"][SFX]["loaded"] is True


# --- config ----------------------------------------------------------------------------------------------

def test_model_kinds_and_compile_override(monkeypatch, tmp_path):
    (tmp_path / "models.yaml").write_text((config_module.CONFIG_DIR / "models.yaml").read_text())
    (tmp_path / "voices.yaml").write_text("voices: {}\n")
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("LOCAL_TTS_SFX_COMPILE", raising=False)
    monkeypatch.setenv("LOCAL_TTS_BATCH_SIZE", "4")  # applies to speech models only
    reg = config_module.load_registry()
    assert reg.models[SFX].kind == "sfx" and reg.models[TTS].kind == "tts" and reg.default_sfx == SFX
    assert reg.models[SFX].options["compile"] is True and "batch_size" not in reg.models[SFX].options
    assert reg.models[TTS].options["batch_size"] == 4
    for raw, value in (("0", False), ("false", False), ("1", True), ("yes", True)):
        monkeypatch.setenv("LOCAL_TTS_SFX_COMPILE", raw)
        assert config_module.load_registry().models[SFX].options["compile"] is value
    monkeypatch.setenv("LOCAL_TTS_SFX_COMPILE", "maybe")
    with pytest.raises(ValueError, match="compile"):
        config_module.load_registry()
    monkeypatch.delenv("LOCAL_TTS_SFX_COMPILE")
    text = (config_module.CONFIG_DIR / "models.yaml").read_text()
    for bad, match in ((text.replace("default_sfx: moss-sfx-v2.0", f"default_sfx: {TTS}"), "default_sfx"),
                       (text.replace("default: moss-tts-local-v1.5", f"default: {SFX}"), "default"),
                       (text.replace("kind: sfx", "kind: music"), "kind")):
        (tmp_path / "models.yaml").write_text(bad)
        with pytest.raises(ValueError, match=match):
            config_module.load_registry()


def test_voices_cannot_name_a_sound_effect_model(client):
    from tests.test_tools import put_voice
    e = err(put_voice(client, "v", models=SFX))
    assert e["code"] == "incompatible_voice"
    assert client.get("/v1/voices", params={"model": SFX}).json()["data"] == []


def test_first_clip_after_a_load_is_not_measured(client):
    client.app.state.scheduler.cfg.idle_unload_seconds = 3600
    wait_status(client, sfx(client, {"prompt": "x", "seconds": 1}).json()["id"], {"succeeded"})
    h = client.get("/health").json()["estimates"][SFX]
    assert h["load_measured"] is True and h["clip_measured"] is False  # cold: its only clip may still compile
    wait_status(client, sfx(client, {"prompt": "y", "seconds": 1}).json()["id"], {"succeeded"})
    assert client.get("/health").json()["estimates"][SFX]["clip_measured"] is True  # warm: measured


@pytest.mark.parametrize("prompts", [["fine", "HOLD FAIL"], ["FAIL", "HOLD FAIL"], ["HOLD FAIL"]])
def test_cancel_during_a_clip_that_then_fails(client, prompts):
    """A cancel accepted while the last clip runs wins, whether that clip succeeds or fails."""
    SFX_HOLD.clear()
    body = {"seconds": 1, "clips": [{"id": f"c{i}", "prompt": p} for i, p in enumerate(prompts)]}
    job = sfx(client, body).json()["id"]
    wait_sfx_started(len(prompts))
    assert client.delete(f"/v1/jobs/{job}").status_code == 202
    SFX_HOLD.set()
    assert wait_status(client, job, {"failed", "succeeded"})["error"]["code"] == "cancelled"


def test_single_effect_cancelled_while_failing(client):
    SFX_HOLD.clear()
    job = sfx(client, {"prompt": "HOLD FAIL", "seconds": 1}).json()["id"]
    wait_sfx_started(1)
    assert client.delete(f"/v1/jobs/{job}").status_code == 202
    SFX_HOLD.set()
    assert wait_status(client, job, {"failed", "succeeded"})["error"]["code"] == "cancelled"


def test_cache_maintenance_failure_does_not_stop_the_gpu_thread(client):
    client.app.state.scheduler.cfg.idle_unload_seconds = 3600
    SFX_FAIL_RELEASE["on"] = True  # raised by release_cache() after each job that empties the queue
    wait_status(client, sfx(client, {"prompt": "x", "seconds": 1}).json()["id"], {"succeeded"})
    wait_status(client, sfx(client, {"prompt": "y", "seconds": 1}).json()["id"], {"succeeded"})
    assert client.app.state.scheduler._thread.is_alive()
