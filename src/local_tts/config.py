"""Load server settings and the model / voice registries from config/*.yaml.

Voices come from two files: the shipped `config/voices.yaml` (paths relative to the project root) and
the user's own registry `<user voices dir>/voices.yaml` (paths relative to that directory), which
register_voice writes. Both are re-read when they change, so adding a voice needs no restart.
"""
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from local_tts.errors import TTSError

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("LOCAL_TTS_CONFIG_DIR", PROJECT_ROOT / "config"))


def user_voices_dir() -> Path:
    """Where registered voices live (outside the repo): LOCAL_TTS_VOICES_DIR, else
    $XDG_DATA_HOME/local-tts/voices (~/.local/share/local-tts/voices)."""
    raw = os.environ.get("LOCAL_TTS_VOICES_DIR")
    if raw:
        return Path(raw).expanduser()
    data = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share")
    return Path(data).expanduser() / "local-tts" / "voices"


@dataclass
class SegmentingConfig:
    max_segment_seconds: float = 30.0
    segment_gap_seconds: float = 0.35
    paragraph_gap_seconds: float = 0.8
    unvoiced_consistency: str = "first_segment"  # or "none"
    max_retries: int = 2
    max_split_depth: int = 2


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    queue_max: int = 32
    idle_unload_seconds: float = 600
    job_ttl_seconds: float = 3600
    # Finished audio kept for polling may use at most this much RAM; beyond it submissions get 503.
    max_result_bytes: int = 4 * 2**30
    # /v1/audio/speech waits at most this long, then returns 504 sync_timeout with the job id.
    sync_timeout_seconds: float = 1800
    segmenting: SegmentingConfig = field(default_factory=SegmentingConfig)


@dataclass
class ModelEntry:
    id: str
    adapter: str
    description: str
    options: dict[str, Any]


@dataclass
class VoiceEntry:
    name: str
    description: str
    language: Optional[str]
    reference_audio: Path
    models: list[str]
    transcript: Optional[str] = None


@dataclass
class Registry:
    default_model: str
    models: dict[str, ModelEntry]
    voices: dict[str, VoiceEntry]
    # Name -> "builtin" (config/voices.yaml) or "user" (the user voices registry).
    voice_sources: dict[str, str] = field(default_factory=dict)
    # Set when a changed voices file could not be loaded (the previous voices stay in use).
    voice_error: Optional[str] = None
    _stamp: Any = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def refresh_voices(self) -> None:
        """Re-read the voice files if either changed since the last load. A broken file keeps the
        last good voices and is reported in `voice_error` (and /health) instead of failing requests."""
        with self._lock:
            try:
                stamp = _voice_stamp()
                # While an error stands, retry on every use: the cause (a missing recording, a
                # permission) may be fixed without touching the YAML files.
                if stamp == self._stamp and self.voice_error is None:
                    return
                voices, sources = load_voices(self.models)
            except (ValueError, OSError, yaml.YAMLError, KeyError, TypeError, AttributeError) as e:
                error = f"{type(e).__name__}: {e}"
                if error != self.voice_error:
                    log.warning("voice registry not reloaded (keeping %d voices): %s", len(self.voices), error)
                self.voice_error = error
                return
            if set(voices) != set(self.voices) or self.voice_error:
                log.info("voice registry reloaded: %s", ", ".join(sorted(voices)) or "(none)")
            self.voices, self.voice_sources, self.voice_error, self._stamp = voices, sources, None, stamp

    def resolve_model(self, model_id: Optional[str]) -> ModelEntry:
        if model_id is None:
            return self.models[self.default_model]
        if model_id not in self.models:
            raise TTSError("unknown_model", f"Unknown model '{model_id}'.", 404,
                           available=sorted(self.models))
        return self.models[model_id]

    def resolve_voice(self, name: Optional[str], model_id: str) -> Optional[VoiceEntry]:
        if name is None:
            return None
        self.refresh_voices()
        voices = self.voices
        compatible = sorted(v.name for v in voices.values() if model_id in v.models)
        if name not in voices:
            extra = {} if self.voice_error is None else {"voice_config_error": self.voice_error}
            raise TTSError("unknown_voice", f"Unknown voice '{name}'.", 404, available=compatible, **extra)
        voice = voices[name]
        if model_id not in voice.models:
            raise TTSError("incompatible_voice", f"Voice '{name}' does not support model '{model_id}'.",
                           400, available=compatible)
        return voice


def _read_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_server_config() -> ServerConfig:
    raw = _read_yaml("server.yaml")
    seg = SegmentingConfig(**raw.pop("segmenting", {}))
    cfg = ServerConfig(**raw, segmenting=seg)
    if "LOCAL_TTS_PORT" in os.environ:
        cfg.port = int(os.environ["LOCAL_TTS_PORT"])
    return cfg


def _batch_size(model_id: str, value: Any, source: str) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        size = 0
    if size < 1 or str(value).strip() != str(size):
        raise ValueError(f"model '{model_id}': batch_size from {source} must be a positive integer, got {value!r}")
    return size


def load_registry() -> Registry:
    raw_models = _read_yaml("models.yaml")
    models = {}
    # A machine with spare GPU memory can batch without editing the tracked models.yaml.
    override = os.environ.get("LOCAL_TTS_BATCH_SIZE")
    for model_id, m in raw_models.get("models", {}).items():
        m = dict(m)
        m["batch_size"] = (_batch_size(model_id, override, "LOCAL_TTS_BATCH_SIZE") if override is not None
                           else _batch_size(model_id, m.get("batch_size", 1), "models.yaml"))
        models[model_id] = ModelEntry(id=model_id, adapter=m.pop("adapter"),
                                      description=m.pop("description", ""), options=m)
    default = raw_models.get("default")
    if default not in models:
        raise ValueError(f"models.yaml default '{default}' is not a registered model")

    registry = Registry(default_model=default, models=models, voices={})
    # At startup a broken voices file is fatal (as before); later changes are reloaded leniently.
    registry.voices, registry.voice_sources = load_voices(models)
    registry._stamp = _voice_stamp()
    return registry


def _voice_files() -> tuple[Path, Path]:
    return CONFIG_DIR / "voices.yaml", user_voices_dir() / "voices.yaml"


def _voice_stamp() -> tuple:
    out = []
    for path in _voice_files():
        try:
            st = path.stat()
            out.append((st.st_mtime_ns, st.st_size))
        except FileNotFoundError:
            out.append(None)
    return tuple(out)


def read_voice_file(path: Path) -> dict:
    """The `voices:` mapping of one voices file ({} when the file is missing or empty)."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("voices") or {}, dict):
        raise ValueError(f"{path}: expected a mapping 'voices: {{name: {{...}}}}'")
    return raw.get("voices") or {}


def load_voices(models: dict[str, ModelEntry]) -> tuple[dict[str, VoiceEntry], dict[str, str]]:
    """Shipped voices plus the user's registered ones. A user voice may not reuse a shipped name."""
    builtin_file, user_file = _voice_files()
    voices: dict[str, VoiceEntry] = {}
    sources: dict[str, str] = {}
    for source, path, base in (("builtin", builtin_file, PROJECT_ROOT), ("user", user_file, user_file.parent)):
        for name, v in read_voice_file(path).items():
            if name in voices:
                raise ValueError(f"{path}: voice '{name}' is already defined in {builtin_file}")
            voices[name] = _voice_entry(name, v, base, models, path)
            sources[name] = source
    return voices, sources


def _voice_entry(name: str, v: Any, base: Path, models: dict[str, ModelEntry], path: Path) -> VoiceEntry:
    if not isinstance(v, dict) or "reference_audio" not in v:
        raise ValueError(f"{path}: voice '{name}' needs reference_audio")
    ref = Path(v["reference_audio"]).expanduser()
    if not ref.is_absolute():
        ref = base / ref
    if not ref.exists():
        raise ValueError(f"voice '{name}': reference_audio not found: {ref}")
    unknown = set(v.get("models", [])) - set(models)
    if unknown:
        raise ValueError(f"voice '{name}': unknown models {sorted(unknown)}")
    return VoiceEntry(name=name, description=v.get("description", ""), language=v.get("language"),
                      reference_audio=ref, models=list(v.get("models", [])), transcript=v.get("transcript"))
