from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from local_tts.config import ModelEntry, VoiceEntry
from local_tts.errors import TTSError


@dataclass
class SegmentAudio:
    audio: np.ndarray       # float32 [channels, samples]
    sample_rate: int
    frames: int             # generated codec frames
    truncated: bool         # hit the frame budget without an end token
    reference: Any = None   # adapter-specific handle reusable as a voice reference


class ModelAdapter(ABC):
    """Lifecycle shared by every model (speech or sound effects). All methods run on the GPU worker
    thread, except `loaded` and status(), which /health and /v1/models may read at any time."""

    kind: str = ""
    languages: dict[str, str] = {}       # code -> name
    capabilities: dict[str, bool] = {}
    # Per-request options (beyond the always-accepted fields) this adapter accepts.
    supported_options: set[str] = set()

    def __init__(self, entry: ModelEntry):
        self.entry = entry

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None:
        """Release the model. Raises if it could not (the manager then loads nothing else)."""

    def release_cache(self) -> None:
        """Free cached-but-unused accelerator memory between jobs (optional)."""

    def status(self) -> dict[str, Any]:
        """Extra facts for /health and /v1/models (optional)."""
        return {}


class Adapter(ModelAdapter):
    """One TTS model."""

    kind = "tts"
    frames_per_second: float = 12.5
    # Whether generate_batch() is implemented (required for batch_size > 1 in models.yaml).
    supports_batch: bool = False

    @abstractmethod
    def voice_reference(self, voice: VoiceEntry) -> Any:
        """Prepared reference for a registered voice (may be cached)."""

    @abstractmethod
    def generate(self, text: str, *, language: Optional[str], reference: Any,
                 max_frames: int, seed: Optional[int]) -> SegmentAudio: ...

    def generate_batch(self, texts: list[str], *, language: Optional[str], reference: Any,
                       max_frames: list[int], seed: Optional[int]) -> list[SegmentAudio]:
        """Generate several independent texts (same language and reference) in one call; one result
        per text, in order, each judged against its own max_frames. One seed covers the whole call, so
        output depends on the batch's composition. Adapters that implement it set supports_batch."""
        raise NotImplementedError(f"adapter for '{self.entry.id}' cannot generate in batches")

    def normalize_language(self, language: Optional[str]) -> Optional[str]:
        """Accept a language code or name (case-insensitive); return the code."""
        if language is None:
            return None
        key = language.strip().lower()
        for code, name in self.languages.items():
            if key in (code, name.lower()):
                return code
        raise TTSError("unsupported_language", f"Model '{self.entry.id}' does not support language '{language}'.",
                       400, available=sorted(self.languages))


@dataclass
class EffectAudio:
    audio: np.ndarray       # float32 [channels, samples]
    sample_rate: int


class SoundEffectAdapter(ModelAdapter):
    """One text-to-sound-effect model: a prompt and a duration in, one clip out. No segments, voices,
    languages or frame budgets."""

    kind = "sfx"
    supported_options = {"seed", "num_inference_steps", "cfg_scale"}
    min_seconds: float = 0.1
    channels: int = 1

    def __init__(self, entry: ModelEntry):
        super().__init__(entry)
        self.max_seconds = float(entry.options["max_seconds"])
        self.sample_rate = int(entry.options.get("sample_rate", 48000))
        # Notes about the loaded model the caller should see (e.g. running uncompiled); set by load().
        self.warnings: list[str] = []

    def installed(self) -> bool:
        """Whether whatever the model needs outside the service (e.g. its environment) is present."""
        return True

    def status(self) -> dict[str, Any]:
        return {"installed": self.installed(), "min_seconds": self.min_seconds, "max_seconds": self.max_seconds,
                "sample_rate": self.sample_rate, "channels": self.channels}

    def install_error(self) -> TTSError:
        """The error to report when installed() is False."""
        return TTSError("model_not_installed", f"Model '{self.entry.id}' is not installed.", 503)

    @abstractmethod
    def generate(self, prompt: str, *, seconds: float, seed: int, num_inference_steps: Optional[int],
                 cfg_scale: Optional[float], clip: Optional[str] = None) -> EffectAudio:
        """One clip (`clip` names it in errors; None for a single effect). Options left None use the
        model's configured defaults. A failure confined to this clip raises TTSError with a code in
        SFX_CLIP_ERRORS; anything else (e.g. the model process died) fails the whole job."""


# Errors that fail one sound-effect clip; any other error fails the whole job.
SFX_CLIP_ERRORS = ("sfx_generation_failed", "sfx_invalid_output", "gpu_out_of_memory")
