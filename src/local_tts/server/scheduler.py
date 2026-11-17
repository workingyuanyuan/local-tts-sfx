"""Single GPU worker with a bounded FIFO queue, job tracking and idle unload.

One condition variable guards the pending queue and every job-state transition, so:
- dequeuing a job and marking it running is atomic (snapshots never show a queued job without a
  queue position);
- the idle-unload decision and the unload itself hold the lock, so a request submitted meanwhile
  waits for the unload to finish instead of racing it.
"""
import collections
import hashlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from local_tts.audio import FORMATS, encode
from local_tts.config import ServerConfig
from local_tts.engine import ModelManager, SynthesisRequest, synthesize
from local_tts.errors import TTSError
from local_tts.text import PAUSE_RE, estimate_seconds

log = logging.getLogger(__name__)

# Priors for wait estimates until this service has measured its own hardware: synthesis seconds per
# estimated second of speech, and seconds to load a model. Each finished job / load updates them.
DEFAULT_RTF = 0.75
DEFAULT_LOAD_SECONDS = 20.0
EWMA_WEIGHT = 0.3  # weight of the newest measurement


@dataclass
class Job:
    request: SynthesisRequest
    key: Optional[str] = None  # client idempotency key: resubmitting with it returns this job
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    status: str = "queued"  # queued | running | succeeded | failed (a cancelled job fails with code "cancelled")
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    segments_done: int = 0
    segments_total: Optional[int] = None
    audio: Optional[bytes] = None
    clip_audio: Optional[dict[str, bytes]] = None  # a clips job: encoded audio of each succeeded clip
    meta: dict[str, Any] = field(default_factory=dict)
    # Stored as plain data (not the exception) so no traceback keeps GPU objects alive.
    error: Optional[dict[str, Any]] = None
    error_http_status: int = 500
    cancel_requested: bool = False
    est_seconds: float = 0.0  # estimated speech (pauses excluded) in the input, for wait estimates
    done: threading.Event = field(default_factory=threading.Event)


class Scheduler:
    def __init__(self, manager: ModelManager, cfg: ServerConfig):
        self.manager = manager
        self.cfg = cfg
        self.cond = threading.Condition()
        self.pending: collections.deque[Job] = collections.deque()
        self.jobs: dict[str, Job] = {}
        self.keys: dict[str, Job] = {}
        self.running: Optional[Job] = None
        self.last_activity = time.monotonic()
        # Measured on this machine (per model): synthesis seconds per estimated speech second, load seconds.
        self.rtf: dict[str, float] = {}
        self.load_seconds: dict[str, float] = {}
        self.janitor_interval = 30.0
        self._stop = False
        self._thread = threading.Thread(target=self._worker, name="gpu-worker", daemon=True)
        # Expired jobs are purged here too, so audio is released even while a long job runs.
        self._janitor = threading.Thread(target=self._janitor_loop, name="job-janitor", daemon=True)

    def start(self) -> None:
        self._thread.start()
        self._janitor.start()

    def _janitor_loop(self) -> None:
        with self.cond:
            while not self._stop:
                self._purge()
                self.cond.wait(timeout=self.janitor_interval)

    def stop(self) -> None:
        with self.cond:
            self._stop = True
            self.cond.notify_all()
        self._thread.join(timeout=5)

    def submit(self, req: SynthesisRequest, key: Optional[str] = None) -> tuple[Job, bool]:
        """Queue a request; returns (job, created). A known `key` returns its existing job, so a
        client that lost the response to a submission can retry it without generating twice."""
        # Speech only: [pause] markers become silence without any generation work.
        job = Job(request=req, key=key, est_seconds=estimate_seconds(PAUSE_RE.sub(" ", req.text)))
        with self.cond:
            if key is not None and key in self.keys:
                return self.keys[key], False
            if len(self.pending) >= self.cfg.queue_max:
                raise TTSError("queue_full", f"Queue is full ({self.cfg.queue_max} waiting); retry later.", 503,
                               queue_length=len(self.pending))
            held = self._held_bytes()
            if held >= self.cfg.max_result_bytes:
                raise TTSError("results_full", f"Finished audio not yet expired uses {held} bytes (limit "
                               f"{self.cfg.max_result_bytes}); retry later.", 503, held_bytes=held)
            self.pending.append(job)
            self.jobs[job.id] = job
            if key is not None:
                self.keys[key] = job
            self.cond.notify_all()
            queued = len(self.pending)
        log.info("job %s queued (model=%s, %d chars, queue=%d)", job.id, req.model.id, len(req.text), queued)
        return job, True

    def get(self, job_id: str) -> Job:
        with self.cond:
            job = self.jobs.get(job_id)
        if job is None:
            raise TTSError("unknown_job", f"Unknown or expired job '{job_id}'.", 404)
        return job

    def cancel(self, job_id: str) -> tuple[Job, bool]:
        """Cancel a job; returns (job, stopped_now). A queued job fails at once; a running one stops
        before its next generation attempt (the segment being generated finishes first). Repeating
        the call is harmless; a job that already finished otherwise cannot be cancelled."""
        with self.cond:
            job = self.jobs.get(job_id)
            if job is None:
                raise TTSError("unknown_job", f"Unknown or expired job '{job_id}'.", 404)
            if job.status == "queued":
                self.pending.remove(job)
                err = TTSError("cancelled", "The job was cancelled before it started.", 409)
                job.error, job.error_http_status, job.status = err.to_dict(), err.http_status, "failed"
                job.finished = time.time()
                job.cancel_requested = True
                job.done.set()
                log.info("job %s cancelled while queued", job_id)
                return job, True
            if job.status == "running":
                if not job.cancel_requested:
                    log.info("job %s: cancel requested while running", job_id)
                job.cancel_requested = True
                return job, False
            if job.error is not None and job.error.get("code") == "cancelled":  # already cancelled
                return job, True
            raise TTSError("job_finished", f"Job '{job_id}' already {job.status}; nothing to cancel.", 409,
                           job_status=job.status)

    def unload(self) -> Optional[str]:
        """Unload the model now (returns its id, or None if none was loaded). Only when idle: the
        worker uses the adapter outside the manager's lock, so unloading under a running job would pull
        the model out from under it. Holding the scheduler lock keeps a new job from starting meanwhile."""
        with self.cond:
            if self.running is not None or self.pending:
                last = self.pending[-1] if self.pending else self.running
                raise TTSError("model_busy", "A job is running or queued; the model stays loaded until the "
                               "service is idle (wait for the jobs, or cancel them first).", 409,
                               running_job=self.running.id if self.running else None,
                               queue_length=len(self.pending), estimated_idle_seconds=self._estimate(last))
            model = self.manager.unload()
        if model:
            log.info("unloaded %s on request", model)
        return model

    def voice_in_use(self, path) -> bool:
        """Does a queued or running job still need this reference audio file?"""
        with self.cond:
            jobs = list(self.pending) + ([self.running] if self.running else [])
            return any(j.request.voice is not None and j.request.voice.reference_audio == path for j in jobs)

    def list_jobs(self) -> list[dict[str, Any]]:
        """Short summaries of every job the service still knows, newest first (to find a lost job id)."""
        with self.cond:
            out = []
            for job in sorted(self.jobs.values(), key=lambda j: j.created, reverse=True):
                text = job.request.text
                d = {"id": job.id, "status": job.status, "model": job.request.model.id,
                     "voice": job.request.voice.name if job.request.voice else None,
                     "format": job.request.response_format, "created": job.created, "finished": job.finished,
                     "chars": len(text), "text": text if len(text) <= 80 else text[:79] + "…",
                     "progress": {"segments_done": job.segments_done, "segments_total": job.segments_total}}
                if job.request.clips is not None:
                    d["clips"] = len(job.request.clips)
                if job.error is not None:
                    d["error_code"] = job.error.get("code")
                out.append(d)
            return out

    def _model_loaded(self, model_id: str) -> bool:
        return self.manager.loaded_model == model_id

    def _remaining(self, job: Job, loaded_model: Optional[str]) -> float:
        """Estimated seconds of work left in one job (caller holds the lock)."""
        model = job.request.model.id
        work = job.est_seconds * self.rtf.get(model, DEFAULT_RTF)
        if job.status == "running":
            if job.segments_total:
                work *= 1 - job.segments_done / job.segments_total
            if loaded_model != model:  # still loading
                work += self.load_seconds.get(model, DEFAULT_LOAD_SECONDS)
        return work

    def _estimate(self, job: Job) -> Optional[float]:
        """Seconds until `job` finishes: the running job, the jobs ahead of it and its own work, plus a
        model load wherever the model changes (caller holds the lock)."""
        if job.status not in ("queued", "running"):
            return None
        loaded = self.manager.loaded_model
        total = 0.0
        if job.status == "queued":
            if self.running is not None:
                total += self._remaining(self.running, loaded)
                loaded = self.running.request.model.id
            for j in self.pending:
                if j.request.model.id != loaded:
                    total += self.load_seconds.get(j.request.model.id, DEFAULT_LOAD_SECONDS)
                    loaded = j.request.model.id
                total += self._remaining(j, loaded)
                if j is job:
                    break
        else:
            total = self._remaining(job, loaded)
        return round(total, 1)

    def estimates(self) -> dict[str, Any]:
        with self.cond:
            return {m: {"rtf": round(self.rtf.get(m, DEFAULT_RTF), 3), "rtf_measured": m in self.rtf,
                        "load_seconds": round(self.load_seconds.get(m, DEFAULT_LOAD_SECONDS), 1),
                        "load_measured": m in self.load_seconds}
                    for m in self.manager.adapters}

    def snapshot(self, job: Job) -> dict[str, Any]:
        """Consistent view of a job (taken under the scheduler lock)."""
        with self.cond:
            d = {
                "id": job.id, "status": job.status, "model": job.request.model.id,
                "voice": job.request.voice.name if job.request.voice else None,
                "format": job.request.response_format,
                "created": job.created, "started": job.started, "finished": job.finished,
                "progress": {"segments_done": job.segments_done, "segments_total": job.segments_total},
                "estimated_remaining_seconds": self._estimate(job),
            }
            if job.request.clips is not None:
                d["clips_total"] = len(job.request.clips)
            if job.cancel_requested:
                d["cancel_requested"] = True
            if job.status == "queued":
                d["queue_position"] = next(i for i, j in enumerate(self.pending, 1) if j is job)
            if job.status == "succeeded":
                d["result"] = dict(job.meta)
            if job.error is not None:
                d["error"] = dict(job.error)
            return d

    def outcome(self, job: Job, clip: Optional[str] = None
                ) -> tuple[str, Optional[dict[str, Any]], int, Optional[bytes], dict[str, Any]]:
        """(status, error, error_http_status, audio, meta) read atomically; `clip` picks one clip's audio."""
        with self.cond:
            audio = job.audio if clip is None else (job.clip_audio or {}).get(clip)
            return job.status, job.error, job.error_http_status, audio, dict(job.meta)

    def stats(self) -> dict[str, Any]:
        with self.cond:
            return {"queue_length": len(self.pending),
                    "running_job": self.running.id if self.running else None}

    def _held_bytes(self) -> int:
        """RAM used by finished audio waiting to be fetched. Caller holds the lock."""
        return sum(len(j.audio or b"") + sum(map(len, (j.clip_audio or {}).values())) for j in self.jobs.values())

    def _purge(self) -> None:
        """Drop finished jobs (and their audio) older than the TTL. Caller holds the lock."""
        cutoff = time.time() - self.cfg.job_ttl_seconds
        for jid in [j.id for j in self.jobs.values() if j.finished and j.finished < cutoff]:
            job = self.jobs.pop(jid)
            if job.key is not None:
                self.keys.pop(job.key, None)

    @staticmethod
    def _observe(table: dict[str, float], model: str, value: float) -> None:
        """Fold a measurement into a per-model moving average (caller holds the lock)."""
        old = table.get(model)
        table[model] = value if old is None else (1 - EWMA_WEIGHT) * old + EWMA_WEIGHT * value

    def _worker(self) -> None:
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.pending or self._stop, timeout=1.0)
                if self._stop:
                    return
                self._purge()
                if not self.pending:
                    # Nothing running (single worker) and the queue is empty; submit() is blocked on
                    # the lock while we decide and unload.
                    if (self.manager.loaded_model
                            and time.monotonic() - self.last_activity >= self.cfg.idle_unload_seconds):
                        log.info("idle for %.0fs; unloading", time.monotonic() - self.last_activity)
                        self.manager.unload()
                    continue
                job = self.pending.popleft()
                job.status, job.started = "running", time.time()
                self.running = job
            self._run(job)
            job = None  # drop the reference so TTL purging can free the audio
            self.last_activity = time.monotonic()
            with self.cond:
                idle = not self.pending
            if idle:  # a request arriving now waits only for this (milliseconds)
                self.manager.release_cache()

    def _run(self, job: Job) -> None:
        req = job.request
        meta: dict[str, Any] = {}
        audio: Optional[bytes] = None
        clip_audio: Optional[dict[str, bytes]] = None
        size = 0
        error: Optional[TTSError] = None
        try:
            adapter = self.manager.acquire(req.model.id)

            def progress(done: int, total: int) -> None:
                with self.cond:
                    job.segments_done, job.segments_total = done, total

            # The first job after a load runs a slow warm-up (seen: 25 s for a 5 s segment), so it
            # measures the load time but not the speed.
            warm = self.manager.last_load_seconds is None
            if not warm:
                with self.cond:
                    self._observe(self.load_seconds, req.model.id, self.manager.last_load_seconds)

            t = time.perf_counter()
            res = synthesize(adapter, req, self.cfg.segmenting, progress, cancelled=lambda: job.cancel_requested)
            elapsed = time.perf_counter() - t
            if warm and job.est_seconds >= 1.0:
                with self.cond:
                    self._observe(self.rtf, req.model.id, elapsed / job.est_seconds)
            if res.clips is not None:
                clip_audio, clips, duration = {}, [], 0.0
                for c in res.clips:
                    info: dict[str, Any] = {"id": c.id, "segments": c.segments}
                    if c.error is not None:
                        clips.append(info | {"status": "failed", "error": c.error})
                        continue
                    data = encode(c.audio, res.sample_rate, req.response_format)
                    clip_audio[c.id] = data
                    seconds = c.audio.shape[1] / res.sample_rate
                    duration += seconds
                    clips.append(info | {"status": "succeeded", "duration": round(seconds, 3), "bytes": len(data),
                                         "sha256": hashlib.sha256(data).hexdigest(), "warnings": c.warnings,
                                         "timeline": c.timeline})
                size = sum(map(len, clip_audio.values()))
            else:
                audio = encode(res.audio, res.sample_rate, req.response_format)
                size, duration = len(audio), res.duration
            meta = {
                "duration": round(duration, 3), "sample_rate": res.sample_rate,
                "channels": int(res.audio.shape[0]), "format": req.response_format,
                "content_type": FORMATS[req.response_format], "bytes": size,
                "model": req.model.id, "voice": req.voice.name if req.voice else None,
                "language": req.language, "segments": res.segments, "retries": res.retries,
                "splits": res.splits, "batch_size": res.batch_size, "synthesis_seconds": round(elapsed, 2),
                "rtf": round(elapsed / duration, 3) if duration else None,
                "warnings": res.warnings,
            }
            if res.clips is not None:
                meta["clips"] = clips
            else:
                meta["timeline"] = res.timeline
            log.info("job %s done: %.1fs audio in %.1fs (%d segments, %d retries, %d splits)",
                     job.id, duration, elapsed, res.segments, res.retries, res.splits)
        except TTSError as e:
            error = TTSError(e.code, e.message, e.http_status, **e.details)
            log.warning("job %s failed: %s %s", job.id, e.code, e.message)
        except Exception as e:  # noqa: BLE001 - report any worker failure to the client
            log.exception("job %s crashed", job.id)
            seg, clip = getattr(e, "tts_segment", None), getattr(e, "tts_clip", None)
            where = ({} if seg is None else {"segment": seg}) | ({} if clip is None else {"clip": clip})
            at = "" if seg is None else (f" (segment {seg})" if clip is None else f" (clip '{clip}' segment {seg})")
            if type(e).__name__ == "OutOfMemoryError":
                error = TTSError("gpu_out_of_memory", f"GPU out of memory{at}: {e}", 500, **where)
            else:
                error = TTSError("internal_error", f"{type(e).__name__}{at}: {e}", 500, **where)
        if error is not None and error.code == "gpu_out_of_memory":
            import torch
            torch.cuda.empty_cache()
        with self.cond:
            job.finished = time.time()
            if error is None and self._held_bytes() + size > self.cfg.max_result_bytes:
                error = TTSError("results_full", f"The result ({size} bytes) does not fit in max_result_bytes "
                                 f"({self.cfg.max_result_bytes}) next to unexpired results; retry later.", 503,
                                 held_bytes=self._held_bytes(), result_bytes=size)
            if error is None:
                job.audio, job.clip_audio, job.meta, job.status = audio, clip_audio, meta, "succeeded"
            else:
                job.error, job.error_http_status, job.status = error.to_dict(), error.http_status, "failed"
            self.running = None
        job.done.set()
