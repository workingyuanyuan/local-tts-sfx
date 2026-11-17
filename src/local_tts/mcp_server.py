"""tts-mcp: MCP (stdio) front end for tts-server.

Waiting in a tool call (submit + polling) is bounded by `wait_seconds` (default 45 s, under Codex's
default 60 s MCP tool timeout, or LOCAL_TTS_MCP_WAIT when the agent allows longer calls; max 240 s). Only a finished job adds the audio download (<= 30 s).
A long narration returns status "pending" with a job_id; call get_speech_job(job_id) to keep
waiting. Jobs are never resubmitted; once a job is known, every error carries its job_id.
"""
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from local_tts.client import (ClientError, Deadline, TTSClient, as_client_error, check_target, clip_paths, pending,
                              resolve_output_dir, resolve_output_path, resumable, with_timeline)

MAX_WAIT = 240.0


def _default_wait() -> float:
    """45 s fits Codex's default 60 s tool timeout. An agent configured for longer tool calls can set
    LOCAL_TTS_MCP_WAIT (e.g. 240 with Codex tool_timeout_sec = 330) to need fewer get_speech_job rounds."""
    raw = os.environ.get("LOCAL_TTS_MCP_WAIT")
    if raw is None:
        return 45.0
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"tts-mcp: LOCAL_TTS_MCP_WAIT must be a number of seconds, got '{raw}'") from None
    if not 0 <= value <= MAX_WAIT:
        raise SystemExit(f"tts-mcp: LOCAL_TTS_MCP_WAIT must be between 0 and {MAX_WAIT:.0f}, got '{raw}'")
    return value


DEFAULT_WAIT = _default_wait()

INSTRUCTIONS = """Local text-to-speech on this machine's GPU: use it whenever a task needs spoken audio
(narration, voice-over, reading aloud). Write into the task's project with absolute paths; pass language.
- Several clips (one per slide or scene): one generate_speech_clips call. Separate calls without a voice
  sound like different speakers: pass the same voice (list_voices) to calls that must sound alike.
- "pending": call next.tool with exactly next.arguments until done; never resubmit pending text
  (list_speech_jobs finds a lost job_id). An error with a job_id: follow its "recovery".
- "warnings" on a result: a sentence may be missing; regenerate that file or clip once (overwrite=true)."""

# Writing tools create files; with overwrite=true they replace one, hence destructive. Local only.
WRITES = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False)
READS = ToolAnnotations(read_only_hint=True, open_world_hint=False)

log = logging.getLogger("local_tts.mcp")


class StrictServer(MCPServer):
    """Unknown tool arguments are an explicit `unsupported_parameter` error, never silently dropped
    (the schemas also say so with additionalProperties: false)."""

    async def list_tools(self):
        tools = await super().list_tools()
        for t in tools:
            t.input_schema["additionalProperties"] = False
        return tools

    async def call_tool(self, name, arguments, context=None):
        tool = self._tool_manager.get_tool(name)
        if tool is not None:
            known = set(tool.parameters.get("properties", {}))
            unknown = sorted(set(arguments) - known)
            if unknown:
                raise ToolError(json.dumps({"error": {
                    "code": "unsupported_parameter", "message": f"Unsupported parameter(s): {', '.join(unknown)}.",
                    "parameters": unknown, "supported": sorted(known)}}, ensure_ascii=False))
        return await super().call_tool(name, arguments, context)


server = StrictServer(name="local-tts", instructions=INSTRUCTIONS)
client = TTSClient()


@dataclass
class JobEntry:
    """Everything this process knows about one job. The lock is created once per job and guards the
    path decision and the save, so concurrent or repeated get_speech_job calls save each path once."""
    fmt: Optional[str] = None             # the job's audio format (known at submit or from the server)
    default_path: Optional[Path] = None   # output_path given to generate_speech
    default_overwrite: bool = False
    default_timeline: bool = False
    default_dir: Optional[Path] = None    # output_dir given to generate_speech_clips
    saved: dict[Path, dict] = field(default_factory=dict)  # path -> result, for idempotent repeats
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    used: float = field(default_factory=time.monotonic)


# All tools run on the server's single event loop, so plain dict access is safe.
JOBS: dict[str, JobEntry] = {}
# Entries unused this long are dropped (tts-server forgets finished jobs after an hour by default).
JOB_MEMORY_SECONDS = 2 * 3600


def _entry(job_id: str) -> JobEntry:
    now = time.monotonic()
    for jid in [j for j, e in JOBS.items() if now - e.used > JOB_MEMORY_SECONDS and not e.lock.locked()]:
        del JOBS[jid]
    entry = JOBS.setdefault(job_id, JobEntry())
    entry.used = now
    return entry


def _forget(job_id: str, e: Exception) -> None:
    """The service no longer knows this job: keeping our entry would only grow JOBS."""
    entry = JOBS.get(job_id)
    if isinstance(e, ClientError) and e.code == "unknown_job" and entry is not None and not entry.lock.locked():
        del JOBS[job_id]


def _error(e: Exception, **context: Any) -> ToolError:
    """Every failure becomes {"error": {"code", "message", ...}} (plus job context when known)."""
    if not isinstance(e, (ClientError, OSError, UnicodeError)):
        log.exception("unexpected tool failure")
    ce = as_client_error(e)
    err = ce.to_dict()
    clips = "output_dir" in context
    submit, fetch = ("generate_speech_clips", "get_speech_clips") if clips else ("generate_speech", "get_speech_job")
    if ce.code == "clips_job":
        context["recovery"] = "Call get_speech_clips with this job_id and the output_dir for its files."
    elif ce.code == "not_a_clips_job":
        context["recovery"] = "Call get_speech_job with this job_id and the output_path for its file."
    elif not resumable(ce):  # failed or forgotten: waiting again would only repeat this error
        context["recovery"] = ("The job failed or the service no longer has it (restarted or expired); it cannot "
                               "be resumed. Fix the cause above if there is one (e.g. rephrase or shorten the "
                               f"named segment) and call {submit} again.")
    elif context.get("job_id"):
        if clips:
            args = _clips_args(context["job_id"], context.get("output_dir"), context.get("overwrite", False),
                               context.get("timeline", False))
        else:
            args = _resume_args(context["job_id"], context.get("output_path"), context.get("overwrite", False),
                                context.get("timeline", False))
        context["recovery"] = (f"The job may still be running; call {fetch} with "
                               f"{json.dumps(args, ensure_ascii=False)} instead of resubmitting.")
    context.pop("overwrite", None)
    context.pop("timeline", None)
    err.update({k: v for k, v in context.items() if v is not None and k not in err})
    return ToolError(json.dumps({"error": err}, ensure_ascii=False))


def _clamp_wait(wait_seconds: float) -> float:
    return max(0.0, min(float(wait_seconds), MAX_WAIT))


def _resume_args(job_id: str, path, overwrite: bool, timeline: bool = False) -> dict[str, Any]:
    """get_speech_job arguments that finish the job as requested, even after an MCP restart."""
    args: dict[str, Any] = {"job_id": job_id}
    if path is not None:
        args["output_path"] = str(path)
    if overwrite:
        args["overwrite"] = True
    if timeline:
        args["timeline"] = True
    return args


def _clips_args(job_id: str, directory, overwrite: bool, timeline: bool) -> dict[str, Any]:
    args: dict[str, Any] = {"job_id": job_id}
    if directory is not None:
        args["output_dir"] = str(directory)
    if overwrite:
        args["overwrite"] = True
    if timeline:
        args["timeline"] = True
    return args


def _pending(job_id: str, pend: dict, path: Optional[Path], overwrite: bool, timeline: bool) -> dict[str, Any]:
    """A pending result names its target file and the exact call to keep waiting."""
    if path is None:
        return pend
    return pend | {"output_path": str(path), "next": {
        "tool": "get_speech_job", "arguments": _resume_args(job_id, path, overwrite, timeline)}}


def _target(entry: JobEntry, output_path: Optional[str], overwrite: bool) -> tuple[Optional[Path], bool]:
    """The file a call writes and whether it may replace it. Naming generate_speech's own path again
    keeps its overwrite permission (as `next` does), so following `next` finishes a regeneration."""
    if output_path is None:
        return entry.default_path, overwrite or (entry.default_path is not None and entry.default_overwrite)
    path, _ = resolve_output_path(output_path, entry.fmt)  # format_mismatch if the extension differs
    return path, overwrite or (path == entry.default_path and entry.default_overwrite)


async def _deliver(job_id: str, entry: JobEntry, output_path: Optional[str], overwrite: bool, timeline: bool,
                   deadline: Deadline, last: Optional[dict] = None) -> dict[str, Any]:
    """Under the job's lock: pick the target path, wait (within the deadline) and save it once."""
    timeline = timeline or entry.default_timeline
    try:
        async with asyncio.timeout(deadline.remaining() or 0.01):
            await entry.lock.acquire()
    except TimeoutError:  # another call for this job holds the lock until its own deadline
        pend = pending(last or await client.job(job_id, timeout=deadline.request_timeout(10.0)))
        path = Path(output_path) if output_path else entry.default_path
        return _pending(job_id, pend, path, overwrite or (path == entry.default_path and entry.default_overwrite),
                        timeline)
    try:
        if entry.fmt is None:  # recovered after a restart: ask the server for the job's format
            last = await client.job(job_id, timeout=deadline.request_timeout(10.0))
            entry.fmt = last["format"]
        path, overwrite = _target(entry, output_path, overwrite)
        if path is None:  # recovered without a path: keep using the same default file
            path, _ = resolve_output_path(None, entry.fmt)
            entry.default_path = path
        if path in entry.saved:
            return with_timeline(entry.saved[path], timeline)
        job = await client.wait(job_id, deadline, last=last)
        if job.get("clips_total") is not None:
            raise ClientError("clips_job", f"Job {job_id} is a clips job: use get_speech_clips.", job_id=job_id)
        if job["status"] in ("queued", "running"):
            return _pending(job_id, pending(job), path, overwrite, timeline)
        result = await client.save(job, path, overwrite)
        entry.saved[path] = result
        log.info("job %s saved to %s", job_id, path)
        return with_timeline(result, timeline)
    finally:
        entry.lock.release()


async def _generate(text, output_path, language, voice, model, fmt, overwrite, seed, timeline,
                    wait_seconds) -> dict[str, Any]:
    deadline = Deadline(_clamp_wait(wait_seconds))
    path, fmt = resolve_output_path(output_path, fmt)
    check_target(path, overwrite)
    job = await client.submit(text, model=model, voice=voice, language=language, fmt=fmt, seed=seed,
                              deadline=deadline)
    entry = _entry(job["id"])
    entry.fmt, entry.default_path, entry.default_overwrite, entry.default_timeline = fmt, path, overwrite, timeline
    log.info("job %s submitted -> %s", job["id"], path)
    try:
        return await _deliver(job["id"], entry, None, False, False, deadline, last=job)
    except Exception as e:  # noqa: BLE001 - attach the job so the caller can recover
        _forget(job["id"], e)
        raise _error(e, job_id=job["id"], output_path=str(path), overwrite=overwrite, timeline=timeline) from None


async def _deliver_clips(job_id: str, entry: JobEntry, output_dir: Optional[str], overwrite: bool, timeline: bool,
                         deadline: Deadline, last: Optional[dict] = None) -> dict[str, Any]:
    """The clips counterpart of _deliver: wait within the deadline, then write every clip once."""
    timeline = timeline or entry.default_timeline
    directory = resolve_output_dir(output_dir) if output_dir else entry.default_dir
    if directory is None:
        raise ClientError("invalid_path", "Pass output_dir: the directory given to generate_speech_clips.")
    overwrite = overwrite or (directory == entry.default_dir and entry.default_overwrite)
    try:
        async with asyncio.timeout(deadline.remaining() or 0.01):
            await entry.lock.acquire()
    except TimeoutError:  # another call for this job holds the lock until its own deadline
        job = last or await client.job(job_id, timeout=deadline.request_timeout(10.0))
        return _clips_pending(job_id, job, directory, overwrite, timeline)
    try:
        job = await client.wait(job_id, deadline, last=last)
        if job.get("clips_total") is None:
            raise ClientError("not_a_clips_job", f"Job {job_id} has one audio: use get_speech_job.", job_id=job_id)
        if job["status"] in ("queued", "running"):
            return _clips_pending(job_id, job, directory, overwrite, timeline)
        result = await client.save_clips(job, directory, overwrite)  # skips files already identical
        log.info("job %s: %d clip(s) in %s, status %s", job_id, len(result["clips"]), directory, result["status"])
        return _clips_recovery(with_timeline(result, timeline), overwrite, timeline)
    finally:
        entry.lock.release()


def _clips_pending(job_id: str, job: dict, directory: Path, overwrite: bool, timeline: bool) -> dict[str, Any]:
    return pending(job) | {"clips_total": job.get("clips_total"), "output_dir": str(directory),
                           "next": {"tool": "get_speech_clips",
                                    "arguments": _clips_args(job_id, directory, overwrite, timeline)}}


def _clips_recovery(result: dict[str, Any], overwrite: bool, timeline: bool) -> dict[str, Any]:
    """A partial result says what to do about each kind of failed clip."""
    if result["status"] == "completed":
        return result
    notes = []
    synth = [c["id"] for c in result["clips"] if c.get("stage") == "synthesis"]
    deliver = [c for c in result["clips"] if c.get("stage") == "delivery"]
    if synth:
        notes.append(f"Clips {', '.join(synth)} failed in synthesis (see each error): fix the cause if there is one "
                     "and call generate_speech_clips with only those clips (same voice).")
    if deliver:
        exists = any(c["error"].get("code") == "output_exists" for c in deliver)
        args = _clips_args(result["job_id"], result["output_dir"], overwrite or exists, timeline)
        notes.append(f"Clips {', '.join(c['id'] for c in deliver)} were not written (see each error): call "
                     f"get_speech_clips with {json.dumps(args, ensure_ascii=False)}"
                     + (" (overwrite replaces the existing files)." if exists and not overwrite else "."))
    return result | {"recovery": " ".join(notes)}


async def _generate_clips(clips, output_dir, language, voice, model, fmt, overwrite, seed, timeline,
                          wait_seconds) -> dict[str, Any]:
    deadline = Deadline(_clamp_wait(wait_seconds))
    directory = resolve_output_dir(output_dir)
    fmt = fmt or "wav"
    for path in clip_paths(directory, [c.id for c in clips], fmt).values():
        check_target(path, overwrite)
    job = await client.submit(None, clips=[(c.id, c.text) for c in clips], model=model, voice=voice,
                              language=language, fmt=fmt, seed=seed, deadline=deadline)
    entry = _entry(job["id"])
    entry.fmt, entry.default_dir, entry.default_overwrite, entry.default_timeline = fmt, directory, overwrite, timeline
    log.info("clips job %s submitted (%d clips) -> %s", job["id"], len(clips), directory)
    try:
        return await _deliver_clips(job["id"], entry, None, False, False, deadline, last=job)
    except Exception as e:  # noqa: BLE001 - attach the job so the caller can recover
        _forget(job["id"], e)
        raise _error(e, job_id=job["id"], output_dir=str(directory), overwrite=overwrite, timeline=timeline) from None


async def _get_clips(job_id, wait_seconds, output_dir, overwrite, timeline) -> dict[str, Any]:
    deadline = Deadline(_clamp_wait(wait_seconds))
    entry = _entry(job_id)
    try:
        return await _deliver_clips(job_id, entry, output_dir, overwrite, timeline, deadline)
    except Exception as e:  # noqa: BLE001
        _forget(job_id, e)
        directory = output_dir or (str(entry.default_dir) if entry.default_dir else None)
        raise _error(e, job_id=job_id, output_dir=directory,
                     overwrite=overwrite or (entry.default_overwrite and directory == str(entry.default_dir)),
                     timeline=timeline or entry.default_timeline) from None


async def _get_job(job_id, wait_seconds, output_path, overwrite, timeline) -> dict[str, Any]:
    deadline = Deadline(_clamp_wait(wait_seconds))
    entry = _entry(job_id)
    try:
        return await _deliver(job_id, entry, output_path, overwrite, timeline, deadline)
    except Exception as e:  # noqa: BLE001
        _forget(job_id, e)
        try:
            path, overwrite = _target(entry, output_path, overwrite)
        except ClientError:
            path = output_path
        raise _error(e, job_id=job_id, output_path=str(path) if path else None, overwrite=overwrite,
                     timeline=timeline or entry.default_timeline) from None


async def _cancel(job_id) -> dict[str, Any]:
    try:
        job = await client.cancel(job_id)
    except Exception as e:  # noqa: BLE001
        _forget(job_id, e)
        err = as_client_error(e).to_dict()
        err.setdefault("job_id", job_id)
        if err["code"] in ("unknown_job", "job_finished"):
            err["recovery"] = "Nothing to cancel: the job is over or the service no longer has it."
        else:  # interrupted: the cancel may or may not have arrived
            err["recovery"] = ("Call cancel_speech_job again with the same job_id (repeating it is harmless). "
                               "Do not call get_speech_job: it would save the unwanted audio.")
        raise ToolError(json.dumps({"error": err}, ensure_ascii=False)) from None
    if job["status"] == "failed":
        return {"status": "cancelled", "job_id": job_id}
    return {"status": "cancelling", "job_id": job_id, "job_status": job["status"],
            "message": "The job stops as soon as the segment (or batch of segments) being generated "
                       "finishes; nothing is written."}


async def _run(fn, *args) -> Any:
    try:
        return await fn(*args)
    except ToolError:
        raise
    except Exception as e:  # noqa: BLE001 - structured error for every failure
        raise _error(e) from None


@server.tool(annotations=WRITES)
async def generate_speech(
    text: str,
    output_path: Optional[str] = None,
    language: Optional[str] = None,
    voice: Optional[str] = None,
    model: Optional[str] = None,
    format: Optional[Literal["wav", "flac", "mp3"]] = None,
    overwrite: bool = False,
    seed: Optional[int] = None,
    timeline: bool = False,
    wait_seconds: float = DEFAULT_WAIT,
) -> dict[str, Any]:
    """Synthesize speech from text and write it to an audio file.

    Args:
        text: Text to speak. Supports [pause 1.5s] markers; blank lines separate paragraphs.
        output_path: Absolute path of the file to write (.wav, .flac or .mp3), normally inside the
            task's project directory. Windows paths (C:\\...) are accepted. Default:
            ~/.local/share/local-tts/output/<timestamp>.wav
        language: Language code or name, e.g. "zh", "en", "ja" (recommended when known).
        voice: Voice name from list_voices to clone. Omit for the model's default voice.
        model: Model id from list_models. Omit for the default model.
        format: wav | flac | mp3. Inferred from the output_path extension when omitted.
        overwrite: Replace output_path if it already exists (otherwise that is an error).
        seed: Optional integer (0..2**32-1) for reproducible output.
        timeline: Also return "timeline": per segment its index, paragraph, start and end seconds in
            the file, and text (segments are up to ~30 s; paragraphs always start a new segment).
        wait_seconds: How long this call waits for the result (max 240). If synthesis takes longer,
            the result is status "pending" with a job_id: call get_speech_job(job_id) to keep waiting.

    Returns: status "completed" with path, duration_seconds, sample_rate, channels, format, model,
    voice, segments (plus "warnings" when a segment stayed suspiciously short: a sentence may be
    missing, so regenerate once with overwrite=true; "timeline" if asked); or status "pending" with
    job_id, output_path, progress, estimated_remaining_seconds and
    "next" ({"tool": "get_speech_job", "arguments": {...}}: the call to keep waiting). Errors are
    JSON {"error": {"code", "message", ...}}; once the job exists they include job_id, output_path
    and recovery (how to resume, or job_status "failed" when only a new request can help).
    """
    return await _run(_generate, text, output_path, language, voice, model, format, overwrite, seed, timeline,
                      wait_seconds)


@server.tool(annotations=WRITES)
async def get_speech_job(
    job_id: str,
    wait_seconds: float = DEFAULT_WAIT,
    output_path: Optional[str] = None,
    overwrite: bool = False,
    timeline: bool = False,
) -> dict[str, Any]:
    """Wait for a pending generate_speech job and write its audio when done. Calling it again
    after completion returns the same result.

    Args:
        job_id: The job_id returned by generate_speech (status "pending" or in an error).
        wait_seconds: How long this call waits (max 240) before returning "pending" again.
        output_path: The absolute path given to generate_speech; always pass it (as in the pending
            result's "next"). Omitted, it is generate_speech's path, or after an MCP server restart a
            default file outside the project. Its extension must match the job's format.
        overwrite: Replace an existing file at the target path (e.g. to retry after output_exists).
            Passing generate_speech's own path keeps that call's overwrite setting.
        timeline: Include the per-segment timeline in the completed result (as in generate_speech).
    """
    return await _run(_get_job, job_id, wait_seconds, output_path, overwrite, timeline)


class Clip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="File name without extension: letters, digits, '_', '-', '.' (e.g. \"slide01\").")
    text: str = Field(description="Text of this clip; [pause 1.5s] markers and blank lines work as in generate_speech.")


@server.tool(annotations=WRITES)
async def generate_speech_clips(
    clips: list[Clip],
    output_dir: Optional[str] = None,
    language: Optional[str] = None,
    voice: Optional[str] = None,
    model: Optional[str] = None,
    format: Optional[Literal["wav", "flac", "mp3"]] = None,
    overwrite: bool = False,
    seed: Optional[int] = None,
    timeline: bool = False,
    wait_seconds: float = DEFAULT_WAIT,
) -> dict[str, Any]:
    """Synthesize several clips (one per slide or scene) in one job, each to <output_dir>/<id>.<format>.
    Faster than one generate_speech call per clip (clips share GPU batches), and one narrator: without
    voice, all clips use the voice of the first clip that is at least 3 s long.

    Args:
        clips: [{"id": "slide01", "text": "..."}, ...] in order, up to 100; at most 20,000 characters in all.
        output_dir: Absolute directory for the files (created if missing), normally inside the task's
            project. Default: a new folder under ~/.local/share/local-tts/output/.
        language, voice, model, overwrite, seed, timeline, wait_seconds: as in generate_speech, for all clips.
        format: wav (default) | flac | mp3.

    Returns: status "completed" (every clip written) or "partial", with output_dir, format, model, voice
    and "clips": per clip id, path, status ("completed" with duration_seconds, segments, warnings?,
    timeline?; or "failed" with stage "synthesis"/"delivery" and error), plus "recovery" saying what to do
    about failed clips. A clip with "warnings" may be missing a sentence: regenerate it once. Or status
    "pending" with job_id and "next" ({"tool": "get_speech_clips", "arguments": {...}}): call it until done.
    """
    return await _run(_generate_clips, clips, output_dir, language, voice, model, format, overwrite, seed, timeline,
                      wait_seconds)


@server.tool(annotations=WRITES)
async def get_speech_clips(
    job_id: str,
    output_dir: Optional[str] = None,
    wait_seconds: float = DEFAULT_WAIT,
    overwrite: bool = False,
    timeline: bool = False,
) -> dict[str, Any]:
    """Wait for a pending generate_speech_clips job and write its clips when done. Calling it again
    returns the same result and only writes clips that are still missing.

    Args:
        job_id: The job_id from generate_speech_clips.
        output_dir: The directory given to generate_speech_clips; always pass it (as in "next").
        wait_seconds: How long this call waits (max 240) before returning "pending" again.
        overwrite: Replace existing files (e.g. after output_exists for some clips).
        timeline: Include each clip's per-segment timeline.
    """
    return await _run(_get_clips, job_id, wait_seconds, output_dir, overwrite, timeline)


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=True,
                                         open_world_hint=False))
async def cancel_speech_job(job_id: str) -> dict[str, Any]:
    """Cancel a generate_speech job you no longer need (wrong text, superseded script), so the GPU and
    queue are free for the next one. A queued job stops at once ("cancelled"); a running one stops
    when the segment (or batch) being generated finishes ("cancelling"). No file is written. Cancelling a job that already
    completed is an error (job_finished)."""
    return await _run(_cancel, job_id)


@server.tool(annotations=READS)
async def list_speech_jobs() -> list[dict[str, Any]]:
    """Jobs the service still knows (about an hour after they finish), newest first: id, status, voice,
    created/finished times, progress and the start of the text. Use it to find a job_id you lost (e.g.
    after a context reset) instead of submitting the same text again; then call get_speech_job."""
    return await _run(client.jobs)


@server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True,
                                         open_world_hint=False))
async def unload_model() -> dict[str, Any]:
    """Unload the TTS model from the GPU now, to free its memory for other GPU work (Blender, another
    model). Call it only when all narration for the task is done: the next speech request loads the
    model again (tens of seconds). Refused with model_busy while a job runs or waits. Without it the
    model unloads by itself after the service has been idle for a while (tts_status: idle_unload_seconds)."""
    return await _run(client.unload)


@server.tool(annotations=WRITES)
async def register_voice(
    name: str,
    reference_audio_path: str,
    description: str = "",
    language: Optional[str] = None,
    models: Optional[list[str]] = None,
    replace: bool = False,
) -> dict[str, Any]:
    """Register a voice from a recording so generate_speech(voice=name) can clone it (no restart needed).
    Only register a real person's voice (the user's own recording, a colleague's) when the user has
    explicitly asked for or approved it. To keep one narrator across many files, a 3-30 s clip of
    clean speech you generated earlier (one speaker, no music) also works.

    Args:
        name: New voice name: letters, digits, '_', '-', '.' (max 64).
        reference_audio_path: Absolute path of the recording (WAV or FLAC, 3-30 s, one speaker).
        description: What the voice sounds like (shown by list_voices).
        language: The recording's language, e.g. "zh", "en".
        models: Model ids the voice is for. Default: every model that can clone.
        replace: Replace a voice you registered earlier with this name (shipped voices cannot be replaced).
    """
    async def call() -> dict[str, Any]:
        return await client.register_voice(name, reference_audio_path, description=description, language=language,
                                           models=models, replace=replace)
    return await _run(call)


@server.tool(annotations=READS)
async def list_models() -> list[dict[str, Any]]:
    """List TTS models: id, whether it is the default, whether it is loaded, supported languages,
    capabilities (clone, pause_marker, pinyin, ...) and accepted options."""
    models = await _run(client.models)
    return [{k: m[k] for k in ("id", "description", "default", "loaded", "languages", "capabilities", "options")}
            for m in models]


@server.tool(annotations=READS)
async def list_voices(model: Optional[str] = None) -> list[dict[str, Any]]:
    """List voices that can be cloned: name, description, language, compatible models and source
    ("builtin", shipped with local-tts, or "user", added with register_voice).
    Pass model to only list voices compatible with it."""
    return await _run(client.voices, model)


@server.tool(annotations=READS)
async def tts_status() -> dict[str, Any]:
    """Service health: loaded model, queue length, running job, GPU memory."""
    return await _run(client.health)


def main() -> None:
    # stdout is the MCP channel: log to stderr only.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per poll otherwise
    server.run("stdio")


if __name__ == "__main__":
    main()
