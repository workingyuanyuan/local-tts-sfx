"""tts-server: FastAPI front end for the GPU scheduler.

OpenAI-compatible subset: POST /v1/audio/speech with `model`, `input`, `voice`, `response_format`
(wav | flac | mp3). Extensions (not in OpenAI's API): `language` (code or name, e.g. "zh" /
"Chinese") and `seed` (int, 0..2**32-1). Any other field is rejected with `unsupported_parameter`.
POST /v1/jobs and /v1/audio/speech accept an `Idempotency-Key` header: retrying a submission with the
same key returns the original job instead of generating again. Unknown query parameters are rejected
like unknown body fields.

Sound effects (models of kind "sfx"): POST /v1/sfx/jobs only (asynchronous, 202), with either `prompt`
and `seconds` (one effect) or `clips` [{id, prompt, seconds?, seed?}] (a clip without `seconds` uses the
top-level one), plus optional `model` (default: models.yaml default_sfx), `seed`, `response_format`,
`num_inference_steps` and `cfg_scale`. The job endpoints (/v1/jobs, /v1/jobs/{id}, its audio and
DELETE) serve both kinds; an Idempotency-Key is matched only against submissions of the same kind.
"""
import asyncio
import logging
import math
import re
import secrets
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import quote

from fastapi import FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from local_tts.audio import FORMATS
from local_tts import voices as voice_store
from local_tts.config import load_registry, load_server_config
from local_tts.engine import ModelManager, SynthesisRequest
from local_tts.errors import TTSError
from local_tts.sfx import EffectClip, SoundEffectRequest
from local_tts.text import pause_values
from local_tts.server.scheduler import Job, Scheduler

log = logging.getLogger("local_tts")

MAX_INPUT_CHARS = 20000
MAX_PAUSE_SECONDS = 60.0
MAX_TOTAL_PAUSE_SECONDS = 600.0
# Request bodies are refused before parsing above this size (20,000 chars of JSON-escaped text fit).
MAX_BODY_BYTES = 256 * 1024
# Per-segment seeds are derived by adding small offsets, so keep the base far from torch's 64-bit limit.
MAX_SEED = 2**32 - 1
# GET /v1/jobs/{id}?wait=N holds the request at most this long; it returns early when the job's status
# or progress changes. LONG_POLL_STEP is how often the waiting request looks.
MAX_LONG_POLL = 30.0
LONG_POLL_STEP = 0.1


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str
    model: Optional[str] = None
    voice: Optional[str] = None
    response_format: str = "wav"
    # Extensions beyond the OpenAI API.
    language: Optional[str] = None
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)


EXTENSION_FIELDS = ("language", "seed")

# A clips job (POST /v1/jobs with `clips` instead of `input`): at most this many clips, ids usable as
# file names. The input limits (characters, pauses) apply to all clips together.
MAX_CLIPS = 100
CLIP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class ClipInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    input: str


class JobRequest(SpeechRequest):
    """POST /v1/jobs: `input`, or `clips` (one audio per clip, synthesized in one pass)."""
    input: Optional[str] = None
    clips: Optional[list[ClipInput]] = None


# Sound-effect request bounds beyond the configured limits (server.yaml `sfx:`) and the model's own
# duration range.
MAX_SFX_STEPS = 500
MIN_CFG_SCALE, MAX_CFG_SCALE = 1.0, 20.0
SFX_OPTION_FIELDS = ("seed", "num_inference_steps", "cfg_scale")


class SfxClipInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    prompt: str
    seconds: Optional[float] = Field(default=None, strict=True)
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)


class SfxJobRequest(BaseModel):
    """POST /v1/sfx/jobs: `prompt` + `seconds` (one effect), or `clips` (one audio per clip)."""
    model_config = ConfigDict(extra="forbid")

    prompt: Optional[str] = None
    seconds: Optional[float] = Field(default=None, strict=True)
    clips: Optional[list[SfxClipInput]] = None
    model: Optional[str] = None
    seed: Optional[int] = Field(default=None, ge=0, le=MAX_SEED)
    response_format: str = "wav"
    num_inference_steps: Optional[int] = Field(default=None, strict=True, ge=1, le=MAX_SFX_STEPS)
    cfg_scale: Optional[float] = Field(default=None, strict=True)


def build_app() -> FastAPI:
    cfg = load_server_config()
    registry = load_registry()
    manager = ModelManager(registry)
    scheduler = Scheduler(manager, cfg)
    started = time.time()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        scheduler.start()
        log.info("tts-server ready on %s:%d (models: %s)", cfg.host, cfg.port, ", ".join(registry.models))
        yield
        scheduler.stop()

    # Held while a submission resolves its voice and queues the job, and while a replaced or removed
    # voice's recording is checked and deleted, so a job never queues a file that is being deleted.
    voice_guard = threading.Lock()

    def submit(body, key: Optional[str]) -> Job:
        with voice_guard:
            job, _ = scheduler.submit(to_request(body), key)
        return job

    app = FastAPI(title="local-tts", lifespan=lifespan)
    app.state.scheduler = scheduler
    app.add_middleware(BodyLimit, max_bytes=MAX_BODY_BYTES)

    @app.exception_handler(TTSError)
    async def tts_error(_: Request, e: TTSError):
        headers = {"Retry-After": "10"} if e.code in ("queue_full", "results_full") else None
        return JSONResponse({"error": e.to_dict()}, status_code=e.http_status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, e: RequestValidationError):
        extra = [str(err["loc"][-1]) for err in e.errors() if err["type"] == "extra_forbidden"]
        if extra:
            # ("body", "clips", <index>, <field>) is a field inside a clip; ("body", "clips") the clips field itself
            in_clip = any(err["type"] == "extra_forbidden" and len(err["loc"]) > 3 and err["loc"][1] == "clips"
                          for err in e.errors())
            if request.url.path == "/v1/sfx/jobs":
                model = SfxClipInput if in_clip else SfxJobRequest
            else:
                model = ClipInput if in_clip else JobRequest if request.url.path == "/v1/jobs" else SpeechRequest
            err = TTSError("unsupported_parameter", f"Unsupported parameter(s): {', '.join(extra)}.", 400,
                           parameters=extra, supported=sorted(model.model_fields))
        else:
            err = TTSError("invalid_request", "Invalid request body.", 400,
                           errors=[{"loc": list(x["loc"]), "msg": x["msg"]} for x in e.errors()])
        return JSONResponse({"error": err.to_dict()}, status_code=err.http_status)

    def strict_query(request: Request, allowed: tuple[str, ...] = ()) -> None:
        extra = sorted(set(request.query_params) - set(allowed))
        if extra:
            raise TTSError("unsupported_parameter", f"Unsupported query parameter(s): {', '.join(extra)}.", 400,
                           parameters=extra, supported=sorted(allowed))

    def model_info(model_id: str) -> dict:
        entry = registry.models[model_id]
        adapter = manager.adapter(model_id)
        info = {
            "id": model_id, "object": "model", "owned_by": "local", "kind": entry.kind,
            "description": entry.description,
            # the default of its kind (speech: `default`, sound effects: `default_sfx`)
            "default": model_id == (registry.default_sfx if entry.kind == "sfx" else registry.default_model),
            "loaded": manager.loaded_model == model_id,
            "languages": sorted(adapter.languages), "capabilities": adapter.capabilities,
            "options": sorted(adapter.supported_options),
        }
        if entry.kind == "sfx":
            info["sfx"] = adapter.status()
        else:
            info["batch_size"] = int(entry.options.get("batch_size", 1))
        return info

    def to_request(body: SpeechRequest) -> SynthesisRequest:
        entry = registry.resolve_model(body.model, kind="tts")
        voice = registry.resolve_voice(body.voice, entry.id)
        adapter = manager.adapter(entry.id)
        given = [f for f in EXTENSION_FIELDS if getattr(body, f) is not None]
        unsupported = [f for f in given if f not in adapter.supported_options]
        if unsupported:
            raise TTSError("unsupported_parameter",
                           f"Model '{entry.id}' does not support: {', '.join(unsupported)}.", 400,
                           parameters=unsupported, supported=sorted(adapter.supported_options))
        if body.response_format not in FORMATS:
            raise TTSError("unsupported_format", f"Unsupported response_format '{body.response_format}'.", 400,
                           available=sorted(FORMATS))
        clips = getattr(body, "clips", None)
        if (body.input is None) == (clips is None):
            raise TTSError("invalid_request", "Give either input or clips.", 400)
        if clips is not None:
            if not 1 <= len(clips) <= MAX_CLIPS:
                raise TTSError("invalid_request", f"clips must hold 1-{MAX_CLIPS} clips.", 400, max_clips=MAX_CLIPS)
            ids = [c.id for c in clips]
            bad = [i for i in ids if not CLIP_ID_RE.fullmatch(i)]
            if bad:
                raise TTSError("invalid_clip_id", "Clip ids are 1-64 characters: letters, digits, '_', '-' or '.', "
                               "starting with a letter or digit (they name the output files).", 400, ids=bad)
            dup = sorted({i for i in ids if ids.count(i) > 1})
            if dup:
                raise TTSError("invalid_clip_id", f"Duplicate clip id(s): {', '.join(dup)}.", 400, ids=dup)
            empty = [c.id for c in clips if not c.input.strip()]
            if empty:
                raise TTSError("empty_input", f"Clip(s) with empty input: {', '.join(empty)}.", 400, clips=empty)
            texts = [c.input for c in clips]
        else:
            if not body.input.strip():
                raise TTSError("empty_input", "Input is empty.", 400)
            texts = [body.input]
        if sum(map(len, texts)) > MAX_INPUT_CHARS:
            raise TTSError("input_too_long", f"Input exceeds {MAX_INPUT_CHARS} characters"
                           f"{' (all clips together)' if clips else ''}.", 400, max_chars=MAX_INPUT_CHARS)
        pauses = [p for t in texts for p in pause_values(t)]
        if any(not math.isfinite(p) or p > MAX_PAUSE_SECONDS for p in pauses) or sum(pauses) > MAX_TOTAL_PAUSE_SECONDS:
            raise TTSError("invalid_pause", f"[pause] markers must each be <= {MAX_PAUSE_SECONDS:.0f}s and "
                           f"total <= {MAX_TOTAL_PAUSE_SECONDS:.0f}s.", 400,
                           max_pause_seconds=MAX_PAUSE_SECONDS, max_total_pause_seconds=MAX_TOTAL_PAUSE_SECONDS)
        language = adapter.normalize_language(body.language)
        return SynthesisRequest(model=entry, voice=voice, text="\n\n".join(texts), response_format=body.response_format,
                                language=language, seed=body.seed,
                                clips=None if clips is None else [(c.id, c.input) for c in clips])

    def to_sfx_request(body: SfxJobRequest) -> SoundEffectRequest:
        entry = registry.resolve_model(body.model, kind="sfx")
        adapter = manager.adapter(entry.id)
        lim = cfg.sfx
        given = [f for f in SFX_OPTION_FIELDS
                 if getattr(body, f) is not None or (f == "seed" and any(c.seed is not None for c in body.clips or []))]
        unsupported = [f for f in given if f not in adapter.supported_options]
        if unsupported:
            raise TTSError("unsupported_parameter",
                           f"Model '{entry.id}' does not support: {', '.join(unsupported)}.", 400,
                           parameters=unsupported, supported=sorted(adapter.supported_options))
        if body.response_format not in FORMATS:
            raise TTSError("unsupported_format", f"Unsupported response_format '{body.response_format}'.", 400,
                           available=sorted(FORMATS))
        if body.cfg_scale is not None and not (math.isfinite(body.cfg_scale)
                                               and MIN_CFG_SCALE <= body.cfg_scale <= MAX_CFG_SCALE):
            raise TTSError("invalid_request", f"cfg_scale must be between {MIN_CFG_SCALE:g} and {MAX_CFG_SCALE:g}.",
                           400, min_cfg_scale=MIN_CFG_SCALE, max_cfg_scale=MAX_CFG_SCALE)
        if (body.prompt is None) == (body.clips is None):
            raise TTSError("invalid_request", "Give either prompt (one effect) or clips.", 400)
        if body.clips is not None:
            if not 1 <= len(body.clips) <= lim.max_clips:
                raise TTSError("invalid_request", f"clips must hold 1-{lim.max_clips} clips.", 400,
                               max_clips=lim.max_clips)
            ids = [c.id for c in body.clips]
            bad = [i for i in ids if not CLIP_ID_RE.fullmatch(i)]
            if bad:
                raise TTSError("invalid_clip_id", "Clip ids are 1-64 characters: letters, digits, '_', '-' or '.', "
                               "starting with a letter or digit (they name the output files).", 400, ids=bad)
            dup = sorted({i for i in ids if ids.count(i) > 1})
            if dup:
                raise TTSError("invalid_clip_id", f"Duplicate clip id(s): {', '.join(dup)}.", 400, ids=dup)
            # A clip's seed is its own; otherwise the job seed + the clip's index; otherwise a random one.
            specs = [(c.id, c.prompt, c.seconds if c.seconds is not None else body.seconds,
                      c.seed if c.seed is not None else None if body.seed is None else body.seed + i)
                     for i, c in enumerate(body.clips)]
        else:
            specs = [("effect", body.prompt, body.seconds, body.seed)]
        where = (lambda cid: "" if body.clips is None else f" (clip '{cid}')")
        empty = [cid for cid, prompt, _, _ in specs if not prompt.strip()]
        if empty:
            raise TTSError("empty_input", "Empty prompt" + ("." if body.clips is None else f"s: {', '.join(empty)}."),
                           400, **({} if body.clips is None else {"clips": empty}))
        long = [cid for cid, prompt, _, _ in specs if len(prompt) > lim.max_prompt_chars]
        if long:
            raise TTSError("input_too_long", f"Prompt{where(long[0])} exceeds {lim.max_prompt_chars} characters.",
                           400, max_chars=lim.max_prompt_chars, **({} if body.clips is None else {"clips": long}))
        for cid, _, seconds, _ in specs:
            if seconds is None:
                raise TTSError("invalid_seconds", f"seconds is required{where(cid)}: how long the effect should be "
                               f"({adapter.min_seconds:g}-{adapter.max_seconds:g}).", 400,
                               min_seconds=adapter.min_seconds, max_seconds=adapter.max_seconds)
            if not (math.isfinite(seconds) and adapter.min_seconds <= seconds <= adapter.max_seconds):
                raise TTSError("invalid_seconds", f"seconds{where(cid)} must be between {adapter.min_seconds:g} and "
                               f"{adapter.max_seconds:g} for model '{entry.id}', got {seconds!r}.", 400,
                               min_seconds=adapter.min_seconds, max_seconds=adapter.max_seconds)
        total = sum(seconds for _, _, seconds, _ in specs)
        if total > lim.max_total_seconds:
            raise TTSError("too_many_seconds", f"The clips ask for {total:g} s together; at most "
                           f"{lim.max_total_seconds:g} s per job.", 400, max_total_seconds=lim.max_total_seconds)
        if not adapter.installed():  # fail now rather than when the job reaches the model
            raise adapter.install_error()
        return SoundEffectRequest(
            model=entry, single=body.clips is None, response_format=body.response_format,
            num_inference_steps=body.num_inference_steps, cfg_scale=body.cfg_scale,
            clips=[EffectClip(cid, prompt, float(seconds), secrets.randbelow(MAX_SEED + 1) if seed is None else seed)
                   for cid, prompt, seconds, seed in specs])

    def job_clip_ids(job: Job) -> Optional[list[str]]:
        req = job.request
        if req.kind == "sfx":
            return req.clip_ids
        return None if req.clips is None else [c for c, _ in req.clips]

    def audio_response(job: Job, clip: Optional[str] = None) -> Response:
        ids = job_clip_ids(job)
        is_clips = ids is not None
        if is_clips and clip is None:
            raise TTSError("clip_required", f"Job '{job.id}' is a clips job: fetch each clip with ?clip=<id>.", 400,
                           clips=ids)
        if not is_clips and clip is not None:
            raise TTSError("invalid_request", f"Job '{job.id}' has one audio; omit clip.", 400)
        if is_clips and clip not in set(ids):
            raise TTSError("unknown_clip", f"Job '{job.id}' has no clip '{clip}'.", 404, clips=ids)
        status, error, http_status, audio, m = scheduler.outcome(job, clip)
        if error is not None:
            details = {k: v for k, v in error.items() if k not in ("code", "message")}
            raise TTSError(error["code"], error["message"], http_status, job_id=job.id, **details)
        if status != "succeeded":
            raise TTSError("job_not_finished", f"Job '{job.id}' is {status}.", 409, job_status=status)
        if job.request.kind == "sfx":
            info = m if clip is None else next(c for c in m["clips"] if c["id"] == clip)
            if info.get("status", "succeeded") != "succeeded":
                raise TTSError("clip_failed", f"Clip '{clip}' failed: {info['error']['message']}", 409,
                               job_id=job.id, clip=clip, clip_error=info["error"])
            headers = {"X-TTS-Job-Id": job.id, "X-TTS-Kind": "sfx", "X-TTS-Model": quote(m["model"]),
                       "X-TTS-Duration": str(info["duration"]), "X-TTS-Sample-Rate": str(m["sample_rate"]),
                       "X-TTS-Channels": str(m["channels"]), "X-TTS-Seed": str(info["seed"])}
            return Response(audio, media_type=m["content_type"], headers=headers)
        segments, duration = m["segments"], m["duration"]
        if clip is not None:
            info = next(c for c in m["clips"] if c["id"] == clip)
            if info["status"] != "succeeded":
                raise TTSError("clip_failed", f"Clip '{clip}' failed: {info['error']['message']}", 409,
                               job_id=job.id, clip=clip, clip_error=info["error"])
            segments, duration = info["segments"], info["duration"]
        headers = {  # names may be non-ASCII (registry keys), so they are percent-encoded
            "X-TTS-Job-Id": job.id, "X-TTS-Model": quote(m["model"]), "X-TTS-Voice": quote(m["voice"] or ""),
            "X-TTS-Duration": str(duration), "X-TTS-Sample-Rate": str(m["sample_rate"]),
            "X-TTS-Channels": str(m["channels"]), "X-TTS-Segments": str(segments),
            "X-TTS-Language": quote(m["language"] or ""), "X-TTS-Retries": str(m["retries"]),
            "X-TTS-Splits": str(m["splits"]),
        }
        return Response(audio, media_type=m["content_type"], headers=headers)

    @app.get("/health")
    def health(request: Request):
        strict_query(request)
        registry.refresh_voices()
        return {
            "status": "ok", "uptime_seconds": round(time.time() - started, 1),
            "loaded_model": manager.loaded_model, "default_model": registry.default_model,
            "default_sfx_model": registry.default_sfx,
            **scheduler.stats(), "queue_max": cfg.queue_max,
            "idle_unload_seconds": cfg.idle_unload_seconds,
            "estimates": scheduler.estimates(),
            "voice_config_error": registry.voice_error,
            # Sound-effect models: worker environment installed, weights cached, compile state, worker pid.
            "sfx": {m: {"loaded": manager.loaded_model == m, **manager.adapter(m).status()}
                    for m in registry.models_of("sfx")},
            "gpu": _gpu_memory(),
        }

    @app.get("/v1/models")
    def list_models(request: Request):
        strict_query(request)
        return {"object": "list", "data": [model_info(m) for m in registry.models]}

    @app.post("/v1/models/unload")
    def unload_model(request: Request):
        # 409 model_busy while a job runs or waits; otherwise unloads now (idempotent: null if none was loaded).
        strict_query(request)
        return {"unloaded": scheduler.unload(), "gpu": _gpu_memory()}

    @app.get("/v1/voices")
    def list_voices(request: Request, model: Optional[str] = None):
        strict_query(request, ("model",))
        if model is not None:
            registry.resolve_model(model)
        registry.refresh_voices()
        return {"object": "list", "data": [
            {"name": v.name, "description": v.description, "language": v.language, "models": v.models,
             "source": registry.voice_sources.get(v.name, "builtin")}
            for v in registry.voices.values() if model is None or model in v.models
        ]}

    def clone_models(models: Optional[list[str]]) -> list[str]:
        if not models:
            return [m for m in registry.models if manager.adapter(m).capabilities.get("clone")]
        for m in models:
            registry.resolve_model(m)
            if not manager.adapter(m).capabilities.get("clone"):
                raise TTSError("incompatible_voice", f"Model '{m}' cannot clone voices.", 400, model=m)
        return list(dict.fromkeys(models))

    @app.put("/v1/voices/{name}")
    async def register_voice(name: str, request: Request, description: str = "", language: Optional[str] = None,
                             models: Optional[list[str]] = Query(None), replace: bool = False):
        # Body: the reference recording itself (WAV/FLAC bytes, 3-30 s); metadata in the query string.
        strict_query(request, ("description", "language", "models", "replace"))
        targets = clone_models(models)
        if not targets:
            raise TTSError("incompatible_voice", "No registered model can clone voices.", 400)
        lang = manager.adapter(targets[0]).normalize_language(language) if language else None
        data = await request.body()
        result = await asyncio.to_thread(voice_store.register, registry, name, data, description=description,
                                         language=lang, models=targets, replace=replace, guard=voice_guard,
                                         in_use=scheduler.voice_in_use)
        return JSONResponse(result, status_code=200 if result["replaced"] else 201)

    @app.delete("/v1/voices/{name}")
    def remove_voice(name: str, request: Request):
        strict_query(request)
        return voice_store.remove(registry, name, guard=voice_guard, in_use=scheduler.voice_in_use)

    def idempotency_key(value: Optional[str]) -> Optional[str]:
        if value is not None and not (0 < len(value) <= 128 and value.isascii() and value.isprintable()):
            raise TTSError("invalid_request", "Idempotency-Key must be 1-128 printable ASCII characters.", 400)
        return value

    @app.post("/v1/audio/speech")
    async def speech(body: SpeechRequest, request: Request,
                     idempotency_key_: Optional[str] = Header(None, alias="Idempotency-Key")):
        strict_query(request)
        # Locks are taken in threads, never on the event loop (an unload holds the scheduler lock).
        job = await asyncio.to_thread(submit, body, idempotency_key(idempotency_key_))
        if not await asyncio.to_thread(job.done.wait, cfg.sync_timeout_seconds):
            # The job keeps running; the client can poll /v1/jobs/{id} instead of resubmitting.
            raise TTSError("sync_timeout", f"Not finished within {cfg.sync_timeout_seconds:.0f}s; "
                           f"poll /v1/jobs/{job.id}.", 504, job_id=job.id)
        return await asyncio.to_thread(audio_response, job)

    @app.post("/v1/jobs", status_code=202)
    def create_job(body: JobRequest, request: Request,
                   idempotency_key_: Optional[str] = Header(None, alias="Idempotency-Key")):
        # A repeated Idempotency-Key returns the job created by the first request (no new generation).
        strict_query(request)
        return scheduler.snapshot(submit(body, idempotency_key(idempotency_key_)))

    @app.post("/v1/sfx/jobs", status_code=202)
    def create_sfx_job(body: SfxJobRequest, request: Request,
                       idempotency_key_: Optional[str] = Header(None, alias="Idempotency-Key")):
        # Sound effects are asynchronous only: poll /v1/jobs/{id}, then fetch /v1/jobs/{id}/audio[?clip=].
        strict_query(request)
        key = idempotency_key(idempotency_key_)
        job, _ = scheduler.submit(to_sfx_request(body), key)
        return scheduler.snapshot(job)

    @app.get("/v1/jobs")
    def list_jobs(request: Request):
        strict_query(request)
        return {"object": "list", "data": scheduler.list_jobs()}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str, request: Request, wait: float = Query(0.0, ge=0, le=MAX_LONG_POLL)):
        # Long poll: with wait > 0, answer when the job finishes, its status or progress changes, or
        # after `wait` seconds, whichever comes first. The scheduler lock is taken in a thread (an
        # idle unload holds it for a moment), never on the event loop.
        strict_query(request, ("wait",))
        job = await asyncio.to_thread(scheduler.get, job_id)
        if wait > 0:
            end = time.monotonic() + wait
            seen = (job.status, job.segments_done, job.segments_total)
            while (not job.done.is_set() and (job.status, job.segments_done, job.segments_total) == seen
                   and time.monotonic() < end):
                await asyncio.sleep(LONG_POLL_STEP)
        return await asyncio.to_thread(scheduler.snapshot, job)

    @app.delete("/v1/jobs/{job_id}")
    def cancel_job(job_id: str, request: Request):
        strict_query(request)
        # 200: the job is over (failed with code "cancelled"). 202: it is running and fails with code
        # "cancelled" as soon as the generation in progress (one segment or one batch) returns.
        job, stopped = scheduler.cancel(job_id)
        return JSONResponse(scheduler.snapshot(job), status_code=200 if stopped else 202)

    @app.get("/v1/jobs/{job_id}/audio")
    def get_job_audio(job_id: str, request: Request, clip: Optional[str] = None):
        strict_query(request, ("clip",))
        return audio_response(scheduler.get(job_id), clip)

    return app


class BodyLimit:
    """ASGI middleware: 413 for bodies over `max_bytes`. The body is read here (at most `max_bytes`,
    checked on Content-Length and while streaming) and replayed to the app, so an oversized request
    is never read into memory or parsed."""

    def __init__(self, app, max_bytes: int, upload_bytes: int = voice_store.MAX_UPLOAD_BYTES):
        self.app, self.default_max, self.upload_bytes = app, max_bytes, upload_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD"):
            return await self.app(scope, receive, send)
        # A voice upload carries a recording; every other body is a small JSON request.
        upload = scope["method"] == "PUT" and scope["path"].startswith("/v1/voices/")
        max_bytes = self.upload_bytes if upload else self.default_max
        length = dict(scope["headers"]).get(b"content-length")
        too_large = length is not None and (not length.isdigit() or int(length) > max_bytes)
        body = bytearray()
        while not too_large:
            message = await receive()
            if message["type"] != "http.request":  # client went away
                return
            body += message.get("body", b"")
            too_large = len(body) > max_bytes
            if not message.get("more_body"):
                break
        if too_large:
            err = TTSError("body_too_large", f"Request body exceeds {max_bytes} bytes.", 413,
                           max_bytes=max_bytes)
            return await JSONResponse({"error": err.to_dict()}, status_code=413)(scope, receive, send)
        replayed = False

        async def replay():
            nonlocal replayed
            if replayed:
                return await receive()  # e.g. waiting for http.disconnect
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}
        await self.app(scope, replay, send)


def _gpu_memory() -> Optional[dict]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
                "device_used_gib": round((total - free) / 2**30, 2),
                "device_total_gib": round(total / 2**30, 2)}
    except Exception:  # noqa: BLE001 - health must not fail on GPU queries
        return None


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_server_config()
    uvicorn.run(build_app(), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
