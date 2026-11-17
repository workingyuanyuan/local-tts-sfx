"""The real worker transport and MOSS-SoundEffect adapter against a fake worker script (the service's
own Python, no torch, no GPU), and the model manager's rules around a worker that fails."""
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from local_tts.adapters import ADAPTERS
from local_tts.adapters.moss_sfx import MossSoundEffectAdapter, compile_cache_dir
from local_tts.adapters.sfx_worker import WorkerError, WorkerProcess
from local_tts.config import ModelEntry, Registry
from local_tts.engine import ModelManager
from local_tts.errors import TTSError
from local_tts.sfx import EffectClip, SoundEffectRequest, generate_effects

FAKE_WORKER = Path(__file__).with_name("fake_sfx_worker.py")
OPTIONS = {"repo": "Org/Model", "revision": "abc123", "max_seconds": 30, "sample_rate": 48000,
           "sampling": {"num_inference_steps": 100, "cfg_scale": 4.0, "sigma_shift": 5.0}, "compile": True,
           "load_timeout_seconds": 20, "generate_timeout_seconds": 20}


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.fixture
def make_adapter(monkeypatch, tmp_path):
    """A real adapter whose worker is the fake script; FAKE_SFX_* environment variables pick its behaviour."""
    monkeypatch.setenv("LOCAL_TTS_CACHE_DIR", str(tmp_path / "cache"))
    made = []

    def make(model_id="sfx", **options):
        a = MossSoundEffectAdapter(ModelEntry(model_id, "moss_sfx", "", OPTIONS | options, kind="sfx"))
        a.python, a.script, a.stop_grace = Path(sys.executable), FAKE_WORKER, 1.0
        made.append(a)
        return a
    yield make
    for a in made:  # never leave a worker behind
        if a.worker is not None and a.worker.proc is not None:
            a.worker._kill_group(9)
            a.worker.proc.wait()


def test_generate_through_a_noisy_worker(make_adapter):
    a = make_adapter()
    a.load()
    pid = a.worker.pid
    assert a.loaded and a.warnings == [] and a.status()["compile"]["active"] is True
    out = a.generate("a door slams", seconds=1.5, seed=3, num_inference_steps=None, cfg_scale=None)
    assert out.sample_rate == 48000 and out.audio.shape == (1, 72000) and out.audio.dtype == np.float32
    assert 0.19 < np.abs(out.audio).max() <= 0.2
    full = a.generate("thirty seconds", seconds=30, seed=1, num_inference_steps=10, cfg_scale=2.0)
    assert full.audio.shape == (1, 30 * 48000)  # the largest result fits the payload bound
    assert any("noise from generate" in line for line in a.worker.stderr_tail(40))  # stdout went to stderr
    a.release_cache()
    assert a.loaded
    a.unload()
    assert not a.loaded and a.worker is None and gone(pid)
    a.unload()  # idempotent


def test_clip_errors_keep_the_worker(make_adapter):
    a = make_adapter()
    a.load()
    pid = a.worker.pid
    for prompt, code in (("FAIL", "sfx_generation_failed"), ("OOM", "gpu_out_of_memory")):
        with pytest.raises(TTSError) as e:
            a.generate(prompt, seconds=1, seed=0, num_inference_steps=None, cfg_scale=None)
        assert e.value.code == code
    assert a.loaded and a.worker.pid == pid
    assert a.generate("fine", seconds=1, seed=0, num_inference_steps=None, cfg_scale=None).audio.shape == (1, 48000)
    a.unload()


def test_worker_crash_fails_the_job_and_the_manager_reloads(make_adapter):
    a = make_adapter()
    mgr = _manager_with(a)
    assert mgr.acquire("sfx") is a
    first = a.worker.pid
    req = SoundEffectRequest(model=a.entry, single=False, clips=[
        EffectClip("ok", "fine", 1.0, 1), EffectClip("boom", "CRASH", 1.0, 2), EffectClip("later", "x", 1.0, 3)])
    with pytest.raises(TTSError) as e:
        generate_effects(a, req)
    assert e.value.code == "sfx_worker_failed" and e.value.details["clip"] == "boom"
    assert "exit code 3" in e.value.message and "noise" in e.value.message  # exit status and stderr tail
    assert not a.loaded and mgr.loaded_model is None
    assert mgr.acquire("sfx") is a and a.loaded and a.worker.pid != first and gone(first)
    mgr.unload()


def test_hung_worker_times_out_and_is_killed(make_adapter, monkeypatch):
    monkeypatch.setenv("FAKE_SFX_IGNORE_TERM", "1")  # only SIGKILL stops it
    a = make_adapter(generate_timeout_seconds=1)
    a.load()
    pid = a.worker.pid
    t = time.monotonic()
    with pytest.raises(TTSError) as e:
        a.generate("HANG", seconds=1, seed=0, num_inference_steps=None, cfg_scale=None)
    assert e.value.code == "sfx_worker_timeout" and time.monotonic() - t < 10
    assert not a.loaded and gone(pid) and a.worker is None  # stopped and reaped by the failure itself
    assert "is stopped" in e.value.message
    a.unload()  # nothing left to do


@pytest.mark.parametrize("mode,code,text", [
    ("weights_missing", "model_weights_missing", "hf download Org/Model --revision abc123"),
    ("load_crash", "sfx_worker_failed", "exit code 5"),
    ("never_ready", "sfx_worker_timeout", "limit 1s"),
    ("garbage", "sfx_worker_failed", "header"),
])
def test_start_failures_are_clear_and_leave_the_manager_usable(make_adapter, monkeypatch, mode, code, text):
    monkeypatch.setenv("FAKE_SFX_MODE", mode)
    a = make_adapter(load_timeout_seconds=1)
    mgr = _manager_with(a)
    with pytest.raises(TTSError) as e:
        mgr.acquire("sfx")
    assert e.value.code == code and text in e.value.message and "Failed to load model 'sfx'" in e.value.message
    assert mgr.current is None and a.worker is None  # stopped and reaped
    monkeypatch.setenv("FAKE_SFX_MODE", "ok")
    assert mgr.acquire("sfx") is a and a.loaded
    mgr.unload()


def test_missing_worker_environment(tmp_path):
    a = MossSoundEffectAdapter(ModelEntry("sfx", "moss_sfx", "", OPTIONS | {"worker": str(tmp_path / "none")},
                                          kind="sfx"))
    assert not a.installed()
    with pytest.raises(TTSError) as e:
        a.load()
    assert e.value.code == "sfx_worker_not_installed" and e.value.http_status == 503
    assert "uv sync --frozen --project" in e.value.message and e.value.details["install_command"]
    s = a.status()
    assert s["worker_installed"] is False and s["install_command"].startswith("uv sync --frozen --project")
    assert a.worker is None


def test_missing_worker_environment_over_http(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from local_tts import config as config_module
    from local_tts.server import app as app_module
    monkeypatch.setitem(ADAPTERS, "moss_local", "tests.fakes:GatedAdapter")
    text = (config_module.CONFIG_DIR / "models.yaml").read_text()
    (tmp_path / "models.yaml").write_text(text.replace("worker: workers/moss-sfx", f"worker: {tmp_path / 'none'}"))
    (tmp_path / "voices.yaml").write_text("voices: {}\n")
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    with TestClient(app_module.build_app()) as c:
        r = c.post("/v1/sfx/jobs", json={"prompt": "x", "seconds": 1})
        e = r.json()["error"]
        assert r.status_code == 503 and e["code"] == "sfx_worker_not_installed"
        assert e["install_command"] == f"uv sync --frozen --project {tmp_path / 'none'}"
        assert c.get("/health").json()["sfx"]["moss-sfx-v2.0"]["worker_installed"] is False
        assert c.get("/v1/jobs").json()["data"] == []  # nothing was queued


def test_compile_fallback_is_a_warning(make_adapter, monkeypatch):
    monkeypatch.setenv("FAKE_SFX_MODE", "no_compile")
    a = make_adapter()
    a.load()
    c = a.status()["compile"]
    assert (c["enabled"], c["active"], c["note"]) == (True, False, "no C compiler found (test)")
    res = generate_effects(a, SoundEffectRequest(model=a.entry, single=True, clips=[EffectClip("effect", "x", 1.0, 0)]))
    assert len(res.warnings) == 1 and "without torch.compile" in res.warnings[0]
    a.unload()


def test_worker_environment_and_arguments(make_adapter, monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/somewhere/src")
    a = make_adapter(compile=False)
    env = a._env()
    assert "PYTHONPATH" not in env and env["TORCHDYNAMO_DISABLE"] == "1"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(compile_cache_dir() / "inductor")
    assert compile_cache_dir() == tmp_path / "cache" / "torch-compile"
    assert a.argv()[2:] == ["--compile", "0", "--repo", "Org/Model", "--revision", "abc123"]
    assert "TORCHDYNAMO_DISABLE" not in make_adapter(compile=True)._env()


def test_unload_failure_blocks_other_models(make_adapter, monkeypatch):
    """A worker that cannot be reaped stays current: no second model loads beside it."""
    a, b = make_adapter("sfx"), make_adapter("other")
    mgr = _manager_with(a, b)
    mgr.acquire("sfx")
    stuck = a.worker

    def refuse(grace=10.0):
        raise WorkerError("unreaped", "sfx-worker (pid 1) did not exit after SIGKILL")
    monkeypatch.setattr(stuck, "stop", refuse)
    with pytest.raises(TTSError) as e:
        mgr.acquire("other")
    assert e.value.code == "model_unload_failed" and mgr.current is a and b.worker is None
    with pytest.raises(TTSError):
        mgr.unload()
    monkeypatch.undo()  # the worker can be stopped again
    assert mgr.acquire("other") is b and a.worker is None
    mgr.unload()


def test_failed_teardown_after_a_failed_load_keeps_the_model_current(monkeypatch):
    events = []

    class Stuck:
        kind, languages, capabilities, supported_options = "tts", {}, {}, set()

        def __init__(self, entry):
            self.entry, self.loaded = entry, False

        def load(self):
            events.append(("load", self.entry.id))
            if self.entry.id == "a":
                raise RuntimeError("load failed")
            self.loaded = True

        def unload(self):
            events.append(("unload", self.entry.id))
            if self.entry.id == "a" and len(events) < 4:
                raise RuntimeError("cannot release")
            self.loaded = False

    monkeypatch.setitem(ADAPTERS, "stuck", "tests.test_sfx_worker:_STUCK")
    monkeypatch.setattr(sys.modules[__name__], "_STUCK", Stuck, raising=False)
    mgr = ModelManager(Registry("a", {m: ModelEntry(m, "stuck", "", {}) for m in "ab"}, {}))
    with pytest.raises(TTSError) as e:
        mgr.acquire("a")
    assert e.value.code == "model_load_failed" and "cannot release" in e.value.message and mgr.current is mgr.adapters["a"]
    with pytest.raises(TTSError) as e:
        mgr.acquire("b")  # a is still not released: b must not load
    assert e.value.code == "model_unload_failed" and ("load", "b") not in events
    assert mgr.acquire("b").loaded and events[-2:] == [("unload", "a"), ("load", "b")]


def test_transport_bounds(tmp_path):
    """A worker that sends more than allowed is a protocol error, not a memory spike."""
    script = tmp_path / "big.py"
    script.write_text("import os, struct, json, time\n"
                      "h = json.dumps({'type': 'result', 'nbytes': 10**9}).encode()\n"
                      "os.write(1, struct.pack('>I', len(h)) + h)\ntime.sleep(30)\n")
    w = WorkerProcess([sys.executable, str(script)], dict(os.environ), name="big")
    w.start()
    with pytest.raises(WorkerError) as e:
        w.recv(10, max_payload=1000)
    assert e.value.reason == "protocol"
    w.stop(grace=0.5)
    assert w.proc is None


def _manager_with(*adapters):
    """A ModelManager over already-built adapters (they are injected after construction)."""
    reg = Registry("none", {}, {})
    mgr = ModelManager(reg)
    mgr.adapters = {a.entry.id: a for a in adapters}
    return mgr



def test_download_command_names_a_configured_model_dir(make_adapter, tmp_path):
    assert make_adapter().download_command().endswith("--revision abc123")
    a = make_adapter(model_dir=str(tmp_path / "weights"))
    assert a.download_command().endswith(f"--revision abc123 --local-dir {tmp_path / 'weights'}")


def in_thread(fn, limit: float = 20.0):
    """Run fn in a thread; fail (instead of hanging the suite) if it does not return within `limit`."""
    out = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - handed to the test
            out["error"] = e
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(limit)
    assert not t.is_alive(), f"blocked for more than {limit}s"
    return out


def test_a_worker_that_stops_reading_cannot_block_generation(make_adapter, monkeypatch):
    monkeypatch.setenv("FAKE_SFX_MODE", "deaf")
    a = make_adapter(generate_timeout_seconds=1)
    a.load()
    pid = a.worker.pid
    out = in_thread(lambda: a.generate("雨聲" * 1000, seconds=1, seed=0, num_inference_steps=None, cfg_scale=None))
    e = out["error"]
    assert isinstance(e, TTSError) and e.code == "sfx_worker_timeout", e
    assert gone(pid) and a.worker is None


def test_stop_escalates_when_the_input_pipe_is_full(monkeypatch):
    monkeypatch.setenv("FAKE_SFX_MODE", "deaf")
    w = WorkerProcess([sys.executable, str(FAKE_WORKER)], dict(os.environ), name="deaf")
    w.start()
    assert w.recv(20)[0]["type"] == "ready"
    big = {"type": "generate", "prompt": "x" * 100_000}
    assert isinstance(in_thread(lambda: w.send(big, timeout=0.5))["error"], WorkerError)  # pipe now full
    pid = w.pid
    out = in_thread(lambda: w.stop(grace=0.1))
    assert "error" not in out and w.proc is None and gone(pid)


@pytest.mark.parametrize("header", [b'{"type": "result", "nbytes": 1e999}', b'{"type": "result", "nbytes": 1.5}',
                                    b'{"type": "result", "nbytes": true}', b'{"type": "ok", "id": NaN}',
                                    b'["not", "an", "object"]', b'{"type": 5}'])
def test_malformed_headers_are_protocol_errors(tmp_path, header):
    script = tmp_path / "bad.py"
    script.write_text("import os, struct, sys, time\nh = sys.argv[1].encode()\n"
                      "os.write(1, struct.pack('>I', len(h)) + h)\ntime.sleep(30)\n")
    w = WorkerProcess([sys.executable, str(script), header.decode()], dict(os.environ), name="bad")
    w.start()
    try:
        with pytest.raises(WorkerError) as e:
            w.recv(10, max_payload=1000)
        assert e.value.reason == "protocol"
    finally:
        w.stop(grace=0.5)


def test_bad_ready_and_result_fields_fail_cleanly(make_adapter, monkeypatch):
    """Numeric fields of the worker's messages are checked; nothing but TTSError escapes the adapter."""
    from local_tts.adapters import moss_sfx
    a = make_adapter()
    real = moss_sfx.WorkerProcess.recv

    def huge(self, timeout, max_payload=0):
        msg, payload = real(self, timeout, max_payload)
        return dict(msg, max_seconds=10**400), payload
    monkeypatch.setattr(moss_sfx.WorkerProcess, "recv", huge)
    with pytest.raises(TTSError) as e:
        a.load()
    assert e.value.code == "sfx_worker_failed"
    a.unload()
    assert a.worker is None


def test_sfx_run_refuses_a_port_another_server_answers(tmp_path):
    """scripts/sfx_run.sh must not run its checks against a service it did not start."""
    import http.server
    import subprocess

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        def log_message(self, *args):
            pass
    other = http.server.HTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=other.serve_forever, daemon=True).start()
    try:
        root = Path(__file__).resolve().parents[1]
        r = subprocess.run(["bash", str(root / "scripts" / "sfx_run.sh"), str(tmp_path / "out"),
                            str(other.server_address[1])], capture_output=True, text=True, timeout=60)
    finally:
        other.shutdown()
    assert r.returncode == 1 and "in use" in r.stdout + r.stderr, r.stdout + r.stderr
    assert not (tmp_path / "out" / "server.log").exists()  # no server was started
