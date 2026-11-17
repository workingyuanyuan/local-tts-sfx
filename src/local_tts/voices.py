"""Registering and removing user voices (server side).

Registered voices live outside the repo, in the user voices directory (config.user_voices_dir()): one
`voices.yaml` plus the reference recordings, stored as 16-bit WAV under a content-hashed name so a
replaced voice never changes the file a queued job is about to read. The registry notices the change
on its next use (config.Registry.refresh_voices).
"""
import hashlib
import io
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, ContextManager, Optional

import numpy as np
import soundfile as sf
import yaml

from local_tts.config import Registry, read_voice_file, user_voices_dir
from local_tts.errors import TTSError

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MIN_SECONDS = 3.0  # engine.MIN_REFERENCE_SECONDS: shorter references are not used for consistency either
MAX_SECONDS = 30.0  # a guard on prompt length / memory, not a measured model limit
MAX_UPLOAD_BYTES = 16 * 2**20
HEADER = ("# Voices registered with register_voice / `tts add-voice` (local-tts). reference_audio paths are\n"
          "# relative to this directory. Edits are picked up without restarting the service.\n")

_lock = threading.Lock()
# mkstemp files are 0600; stored voices get the normal permissions (0666 minus umask).
_UMASK = os.umask(0)
os.umask(_UMASK)


def _check_name(name: str) -> None:
    if not NAME_RE.fullmatch(name):
        raise TTSError("invalid_voice_name", "Voice names are 1-64 characters: letters, digits, '_', '-' or '.', "
                       "starting with a letter or digit.", 400, name=name)


def _decode(data: bytes) -> tuple[np.ndarray, int]:
    # The header is checked before decoding: a small compressed file could expand to gigabytes.
    try:
        info = sf.info(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001 - any decoder failure is the caller's file
        raise TTSError("invalid_audio", f"Cannot decode the reference audio ({type(e).__name__}: {e}); "
                       "use WAV or FLAC.", 400) from None
    sr, frames, channels = int(info.samplerate), int(info.frames), int(info.channels)
    if not 8000 <= sr <= 192000:
        raise TTSError("invalid_audio", f"Unsupported sample rate {sr} Hz.", 400)
    if channels > 2:
        raise TTSError("invalid_audio", f"The reference audio has {channels} channels; use mono or stereo.", 400)
    seconds = frames / sr
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise TTSError("invalid_audio", f"The reference audio is {seconds:.1f}s; it must be {MIN_SECONDS:.0f}-"
                       f"{MAX_SECONDS:.0f}s of one speaker.", 400, seconds=round(seconds, 2))
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True, frames=frames)
    except Exception as e:  # noqa: BLE001
        raise TTSError("invalid_audio", f"Cannot decode the reference audio ({type(e).__name__}: {e}).", 400) from None
    if float(np.abs(audio).max(initial=0.0)) < 1e-3:
        raise TTSError("invalid_audio", "The reference audio is silent.", 400)
    return audio, sr


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o666 & ~_UMASK)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_user(path: Path) -> dict[str, Any]:
    try:
        return dict(read_voice_file(path))
    except (ValueError, OSError, yaml.YAMLError) as e:
        raise TTSError("voice_config_error", f"The user voices file cannot be read ({e}); fix or remove it first.",
                       500, file=str(path)) from None


def _write_user(path: Path, voices: dict[str, Any]) -> None:
    text = HEADER + yaml.safe_dump({"voices": voices}, allow_unicode=True, sort_keys=False)
    _atomic_write(path, text.encode("utf-8"))


def _drop_file(directory: Path, filename: Optional[str], voices: dict[str, Any], guard: ContextManager,
               in_use: Callable[[Path], bool]) -> None:
    """Delete a replaced/removed recording we stored, unless another registered voice or a queued or
    running job still needs it. `guard` is held by submissions while they resolve a voice and queue
    the job, so no request can pick up the old entry between the check and the delete."""
    if not filename:
        return
    path = (directory / filename).resolve()
    if path.parent != directory.resolve():
        return
    if any(isinstance(v, dict) and v.get("reference_audio") and (directory / str(v["reference_audio"])).resolve() == path
           for v in voices.values()):
        return
    with guard:
        if path.exists() and not in_use(directory / filename) and not in_use(path):
            path.unlink()


def register(registry: Registry, name: str, data: bytes, *, description: str, language: Optional[str],
             models: list[str], replace: bool, guard: ContextManager,
             in_use: Callable[[Path], bool]) -> dict[str, Any]:
    _check_name(name)
    registry.refresh_voices()
    if registry.voice_sources.get(name) == "builtin":
        raise TTSError("voice_exists", f"'{name}' is a voice shipped with local-tts; choose another name.", 409,
                       name=name)
    audio, sr = _decode(data)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    wav = buf.getvalue()
    directory = user_voices_dir()
    filename = f"{name}-{hashlib.sha256(wav).hexdigest()[:10]}.wav"
    yaml_path = directory / "voices.yaml"
    with _lock:
        voices = _read_user(yaml_path)
        old = voices.get(name)
        if old is not None and not replace:
            raise TTSError("voice_exists", f"Voice '{name}' already exists; pass replace=true to replace it.", 409,
                           name=name)
        _atomic_write(directory / filename, wav)
        entry: dict[str, Any] = {"description": description}
        if language:
            entry["language"] = language
        entry["reference_audio"] = filename
        entry["models"] = list(models)
        voices[name] = entry
        _write_user(yaml_path, voices)
        old_file = old.get("reference_audio") if isinstance(old, dict) else None
        if old_file != filename:
            _drop_file(directory, old_file, voices, guard, in_use)
    registry.refresh_voices()
    return {"name": name, "description": description, "language": language, "models": list(models),
            "seconds": round(audio.shape[0] / sr, 2), "sample_rate": sr, "channels": int(audio.shape[1]),
            "replaced": old is not None}


def remove(registry: Registry, name: str, guard: ContextManager, in_use: Callable[[Path], bool]) -> dict[str, Any]:
    registry.refresh_voices()
    if registry.voice_sources.get(name) == "builtin":
        raise TTSError("voice_not_removable", f"'{name}' is shipped with local-tts (config/voices.yaml); only "
                       "registered voices can be removed.", 409, name=name)
    directory = user_voices_dir()
    yaml_path = directory / "voices.yaml"
    with _lock:
        voices = _read_user(yaml_path)
        if name not in voices:
            raise TTSError("unknown_voice", f"No registered voice '{name}'.", 404,
                           available=sorted(voices))
        old = voices.pop(name)
        _write_user(yaml_path, voices)
        _drop_file(directory, old.get("reference_audio") if isinstance(old, dict) else None, voices, guard, in_use)
    registry.refresh_voices()
    return {"name": name, "removed": True}
