"""Sound-effect jobs: one generation per clip, nothing else. No sentence splitting, duration-ratio
retries, silence trimming, voices or pause markers: the model is asked for `seconds` of audio and the
clip is what it returns. Runs on the single GPU worker thread."""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional

import numpy as np

from local_tts.adapters.base import SFX_CLIP_ERRORS, SoundEffectAdapter
from local_tts.config import ModelEntry
from local_tts.engine import Cancelled
from local_tts.errors import TTSError

log = logging.getLogger(__name__)


@dataclass
class EffectClip:
    id: str
    prompt: str
    seconds: float
    seed: int  # always set: drawn by the server when the request gave none, and reported back


@dataclass
class SoundEffectRequest:
    kind: ClassVar[str] = "sfx"
    model: ModelEntry
    clips: list[EffectClip]
    # True for a request with one `prompt` (one audio, fetched without ?clip=); False for `clips`.
    single: bool
    response_format: str = "wav"
    num_inference_steps: Optional[int] = None  # None: the model's configured default
    cfg_scale: Optional[float] = None
    # Attributes shared with speech requests, so job listings and voice checks treat both alike.
    voice: ClassVar[None] = None
    language: ClassVar[None] = None

    @property
    def text(self) -> str:
        return "\n\n".join(c.prompt for c in self.clips)

    @property
    def clip_ids(self) -> Optional[list[str]]:
        return None if self.single else [c.id for c in self.clips]

    @property
    def total_seconds(self) -> float:
        return sum(c.seconds for c in self.clips)


@dataclass
class EffectResult:
    id: str
    seconds: float  # requested
    seed: int
    audio: Optional[np.ndarray] = None  # None when the clip failed
    generation_seconds: Optional[float] = None
    error: Optional[dict[str, Any]] = None


@dataclass
class EffectsResult:
    clips: list[EffectResult]
    sample_rate: int
    channels: int
    warnings: list[str] = field(default_factory=list)


def generate_effects(adapter: SoundEffectAdapter, req: SoundEffectRequest,
                     progress: Optional[Callable[[int, int], None]] = None,
                     cancelled: Optional[Callable[[], bool]] = None,
                     clip_time: Optional[Callable[[float], None]] = None) -> EffectsResult:
    """Generate every clip in order. A clip error in SFX_CLIP_ERRORS fails only that clip (all clips
    failing fails the job); any other error fails the job. `cancelled` is checked before each clip (the
    clip in progress finishes and is discarded). `clip_time` receives each clip's generation seconds."""
    if not isinstance(adapter, SoundEffectAdapter):
        raise TTSError("wrong_model_kind", f"Model '{req.model.id}' is not a sound-effect model.", 400)
    total = len(req.clips)
    if progress:
        progress(0, total)
    results: list[EffectResult] = []
    for i, clip in enumerate(req.clips):
        if cancelled is not None and cancelled():
            raise Cancelled()
        name = None if req.single else clip.id
        res = EffectResult(clip.id, clip.seconds, clip.seed)
        t = time.perf_counter()
        out = None
        try:
            out = adapter.generate(clip.prompt, seconds=clip.seconds, seed=clip.seed,
                                   num_inference_steps=req.num_inference_steps, cfg_scale=req.cfg_scale, clip=name)
        except TTSError as e:
            if e.code not in SFX_CLIP_ERRORS:
                if name is not None and "clip" not in e.details:
                    e.details["clip"] = name
                raise
            label = "The effect" if name is None else f"Clip '{name}'"
            log.warning("%s failed: %s %s", label, e.code, e.message)
            res.error = {**e.to_dict(), "message": f"{label}: {e.message}", **({} if name is None else {"clip": name})}
        except Exception as e:
            if name is not None:
                e.tts_clip = name  # the scheduler reports where it crashed
            raise
        # A cancel accepted while this clip ran wins, whether the clip succeeded or failed: discard it.
        if cancelled is not None and cancelled():
            raise Cancelled()
        if res.error is None:
            res.generation_seconds = time.perf_counter() - t
            if clip_time:
                clip_time(res.generation_seconds)
            audio = np.asarray(out.audio, dtype=np.float32)
            if audio.ndim != 2 or audio.shape[1] == 0 or out.sample_rate != adapter.sample_rate:
                res.error = {"code": "sfx_invalid_output", "message": f"Clip '{clip.id}': the model returned "
                             f"audio of shape {audio.shape} at {out.sample_rate} Hz.",
                             **({} if name is None else {"clip": name})}
            else:
                res.audio = audio
        results.append(res)
        if progress:
            progress(i + 1, total)
    failed = [r for r in results if r.error is not None]
    if len(failed) == total:
        first = failed[0].error
        if req.single:
            raise TTSError(first["code"], first["message"], 500)
        raise TTSError(first["code"], f"All {total} clips failed; first: {first['message']}", 500,
                       clips=[r.error for r in failed])
    return EffectsResult(clips=results, sample_rate=adapter.sample_rate,
                         channels=max(r.audio.shape[0] for r in results if r.audio is not None),
                         warnings=list(adapter.warnings))
