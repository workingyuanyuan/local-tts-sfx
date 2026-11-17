"""Shared fakes: a gated adapter (no GPU) and a fixture running the real app on it."""
import threading
import time
import weakref

import numpy as np
import pytest
from fastapi.testclient import TestClient

from local_tts import config as config_module
from local_tts.adapters import ADAPTERS
from local_tts.adapters.base import Adapter, EffectAudio, SegmentAudio, SoundEffectAdapter
from local_tts.errors import TTSError
from local_tts.server import app as app_module

GATE = threading.Event()  # generation blocks until set
FAIL_LOAD = {"on": False}
FAIL_GEN = {"on": False}
LOAD_HELD = []  # weakrefs to objects held during a failed load
RELEASES = []  # one entry per release_cache() call
MODEL_EVENTS = []  # ("load" | "unload", model id), both kinds, in order

SFX_GATE = threading.Event()  # sound-effect generation blocks until set
SFX_FAIL_LOAD = {"on": False}
SFX_CALLS = []  # one dict per sound-effect generate() call (appended when the call returns or fails)
SFX_STARTED = []  # one prompt per generate() call, appended when it is entered (before SFX_GATE)
SFX_HOLD = threading.Event()  # prompts containing HOLD also wait for this (set by default)
SFX_FAIL_RELEASE = {"on": False}  # release_cache() raises (a broken worker reply)


def wait_sfx_started(n: int, timeout: float = 5.0) -> None:
    """Block until n sound-effect generations have started (not merely the job's status 'running')."""
    end = time.monotonic() + timeout
    while len(SFX_STARTED) < n:
        assert time.monotonic() < end, f"only {len(SFX_STARTED)} of {n} generations started"
        time.sleep(0.01)


class Big:
    """Stands in for a partially loaded GPU tensor."""


class GatedAdapter(Adapter):
    languages = {"zh": "Chinese"}
    capabilities = {"clone": True}
    supported_options = {"language", "seed"}

    def __init__(self, entry):
        super().__init__(entry)
        self._loaded = False
        self.partial = None

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        MODEL_EVENTS.append(("load", self.entry.id))
        self.partial = Big()
        if FAIL_LOAD["on"]:
            LOAD_HELD.append(weakref.ref(self.partial))
            raise RuntimeError("simulated load failure")
        self._loaded = True

    def unload(self):
        if self._loaded:
            MODEL_EVENTS.append(("unload", self.entry.id))
        self.partial = None
        self._loaded = False

    def release_cache(self):
        RELEASES.append(self.entry.id)

    def voice_reference(self, voice):
        return None

    def generate(self, text, *, language, reference, max_frames, seed):
        GATE.wait(10)
        if FAIL_GEN["on"]:
            raise RuntimeError("simulated generation failure")
        return SegmentAudio(np.full((2, 1000), 0.3, np.float32), 1000, 12, False)  # 1 s: plausible for short text


class GatedSfxAdapter(SoundEffectAdapter):
    """A sound-effect model without a worker: 100 samples per second of a constant level. Prompts
    containing FAIL fail that clip; CRASH behaves like a worker that died (the job fails, the model is
    no longer loaded)."""

    def __init__(self, entry):
        super().__init__(entry)
        self.sample_rate = 100
        self._loaded = False

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        MODEL_EVENTS.append(("load", self.entry.id))
        if SFX_FAIL_LOAD["on"]:
            raise TTSError("sfx_worker_failed", "The sound-effect worker failed while loading: simulated.", 500)
        self._loaded = True

    def unload(self):
        if self._loaded:
            MODEL_EVENTS.append(("unload", self.entry.id))
        self._loaded = False

    def release_cache(self):
        if SFX_FAIL_RELEASE["on"]:
            raise OverflowError("cannot convert float infinity to integer")

    def generate(self, prompt, *, seconds, seed, num_inference_steps, cfg_scale, clip=None):
        SFX_STARTED.append(prompt)
        SFX_GATE.wait(10)
        if "HOLD" in prompt:
            SFX_HOLD.wait(10)
        SFX_CALLS.append({"prompt": prompt, "seconds": seconds, "seed": seed, "steps": num_inference_steps,
                          "cfg_scale": cfg_scale, "clip": clip})
        if "CRASH" in prompt:
            self._loaded = False
            raise TTSError("sfx_worker_failed", "The sound-effect worker failed while generating: exit code 3.", 500)
        if "FAIL" in prompt:
            raise TTSError("sfx_generation_failed", "Generation failed: simulated.", 500)
        return EffectAudio(np.full((1, int(round(seconds * 100))), 0.1, np.float32), 100)


def make_client(monkeypatch, tmp_path):
    monkeypatch.setitem(ADAPTERS, "moss_local", "tests.fakes:GatedAdapter")
    monkeypatch.setitem(ADAPTERS, "moss_sfx", "tests.fakes:GatedSfxAdapter")
    (tmp_path / "server.yaml").write_text("queue_max: 2\nidle_unload_seconds: 1\nsync_timeout_seconds: 1\n")
    for name in ("models.yaml", "voices.yaml"):
        (tmp_path / name).write_text((config_module.CONFIG_DIR / name).read_text())
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    GATE.clear()
    FAIL_LOAD["on"] = False
    FAIL_GEN["on"] = False
    SFX_GATE.set()
    SFX_FAIL_LOAD["on"] = False
    SFX_CALLS.clear()
    SFX_STARTED.clear()
    SFX_HOLD.set()
    SFX_FAIL_RELEASE["on"] = False
    MODEL_EVENTS.clear()
    with TestClient(app_module.build_app()) as c:
        try:
            yield c
        finally:
            GATE.set()  # release a blocked job before shutdown joins the worker
            SFX_GATE.set()
            SFX_HOLD.set()
