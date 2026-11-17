"""tts: command-line client for tts-server.

  tts say "你好" -o /abs/path/hello.wav --language zh [--voice NAME] [--model ID] [--overwrite]
  tts say -f script.txt -o /abs/out.mp3        (-f - reads stdin)
  tts clips clips.json -d /abs/dir -l zh [--voice NAME] (one file per clip: [{"id": "s01", "text": "..."}])
  tts sfx "door slams shut" -s 3 -o /abs/door.wav   (a sound effect; --seconds is required, at most 30)
  tts sfx-clips effects.json -d /abs/dir [-s 4]      ([{"id": "door", "prompt": "...", "seconds": 3, "seed": 1}])
  tts job <job_id> [--wait] [-o /abs/out.wav | -d /abs/dir]  (status of / fetch a job, e.g. after --no-wait)
  tts cancel <job_id>                          (stop a job that is no longer needed)
  tts jobs                                     (jobs the service still knows, newest first)
  tts unload                                   (free GPU memory now; refused while jobs run or wait)
  tts add-voice NAME /abs/ref.wav [--description D] [--language L] [--model ID] [--replace]
  tts remove-voice NAME                        (registered voices only)
  tts models | tts voices [--model ID] | tts health
  --timeline (say / clips / job): also report each segment's start/end seconds and text (speech only)

Exit codes: 0 done, 1 error (JSON on stderr), 2 usage error, 3 still pending (--no-wait / --timeout),
4 clips job finished but some clips failed or were not written (result on stdout says which).
"""
import argparse
import asyncio
import json
import shlex
import sys

from local_tts.client import (DEFAULT_URL, ClientError, Deadline, TTSClient, as_client_error, check_target, clip_paths,
                              job_kind, pending, raise_job_error, resolve_output_dir, resolve_output_path, resumable,
                              with_timeline)

EXIT_ERROR, EXIT_PENDING, EXIT_PARTIAL = 1, 3, 4


def _print(obj, as_json: bool, human: str) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2) if as_json else human)


def _progress(quiet: bool):
    last = [None]

    def show(job: dict) -> None:
        p = job.get("progress") or {}
        line = f"{job['status']}" + (f" (queue position {job['queue_position']})" if job.get("queue_position") else "")
        if p.get("segments_total"):
            line += f" {p['segments_done']}/{p['segments_total']} segments"
        elif p.get("clips_total"):
            line += f" {p['clips_done']}/{p['clips_total']} effects"
        if not quiet and line != last[0]:
            print(f"[tts] job {job['id']}: {line}", file=sys.stderr, flush=True)
            last[0] = line
    return show


def _human_result(r: dict) -> str:
    who = f"seed={r['seed']}" if r.get("kind") == "sfx" else f"voice={r['voice'] or 'default'}"
    return (f"{r['path']}  ({r['duration_seconds']:.2f}s, {r['sample_rate']} Hz, {r['channels']} ch, "
            f"model={r['model']}, {who})")


def _show_result(r: dict, a) -> None:
    r = with_timeline(r, a.timeline)
    human = _human_result(r)
    for t in r.get("timeline", []):
        human += f"\n  {t['start']:8.3f}-{t['end']:8.3f}  [p{t['paragraph']} s{t['index']}] {t['text']}"
    _print(r, a.json, human)
    if not a.json:  # in JSON they are part of the result
        for w in r.get("warnings", []):
            print(f"[tts] warning: {w}", file=sys.stderr)


def _fetch_command(a, job_id: str, path, clips: bool = False, overwrite: bool = False) -> str:
    """A copy-pasteable command that resumes waiting for a job (keeps --url / --overwrite)."""
    cmd = ["tts", "job", job_id, "--wait", "-d" if clips else "-o", str(path)]
    if a.url != DEFAULT_URL:
        cmd += ["--url", a.url]
    if a.overwrite or overwrite:
        cmd.append("--overwrite")
    if getattr(a, "timeline", False):
        cmd.append("--timeline")
    return shlex.join(cmd)


def _read_text(a) -> "str | None":
    """Text from the argument, a file or stdin; always valid UTF-8 (else invalid_input)."""
    if a.file and a.file != "-":
        source, data = a.file, open(a.file, "rb").read()
    elif a.file == "-" or a.text == "-" or (a.text is None and not sys.stdin.isatty()):
        source, data = "stdin", sys.stdin.buffer.read()
    elif a.text is not None:
        source, data = "TEXT", a.text.encode("utf-8", "surrogateescape")  # argv undecodable bytes survive here
    else:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ClientError("invalid_input", f"{source} is not UTF-8 text ({e.reason}).") from None


async def cmd_say(c: TTSClient, a) -> int:
    text = _read_text(a)
    if text is None:
        print("tts say: give TEXT, -f FILE, or pipe text on stdin", file=sys.stderr)
        return 2
    path, fmt = resolve_output_path(a.output, a.format)
    check_target(path, a.overwrite)
    deadline = Deadline(a.timeout)  # shared by the submission and the wait
    job = await c.submit(text, model=a.model, voice=a.voice, language=a.language, fmt=fmt, seed=a.seed,
                         deadline=deadline)
    return await _finish_single(c, a, job, path, deadline, "tts say")


async def cmd_sfx(c: TTSClient, a) -> int:
    path, fmt = resolve_output_path(a.output, a.format)
    check_target(path, a.overwrite)
    deadline = Deadline(a.timeout)
    job = await c.submit_sfx(a.prompt, seconds=a.seconds, model=a.model, fmt=fmt, seed=a.seed,
                             num_inference_steps=a.steps, cfg_scale=a.cfg_scale, deadline=deadline)
    return await _finish_single(c, a, job, path, deadline, "tts sfx")


async def _finish_single(c: TTSClient, a, job: dict, path, deadline: Deadline, command: str) -> int:
    """Wait for a submitted one-audio job (speech or sound effect) and save it, or say how to resume."""
    fetch = _fetch_command(a, job["id"], path)
    try:
        if not a.no_wait:
            job = await c.wait(job["id"], deadline, last=job, on_progress=_progress(a.quiet or a.json))
        if job["status"] in ("queued", "running"):
            pend = pending(job) | {"output_path": str(path), "fetch_command": fetch}
            _print(pend, a.json, f"pending: job {job['id']} ({job['status']}); fetch later with:\n  {fetch}")
            return EXIT_PENDING
        r = await c.save(job, path, a.overwrite)
    except Exception as e:  # noqa: BLE001 - the job exists: say how to recover instead of resubmitting
        err = as_client_error(e)
        err.details.setdefault("job_id", job["id"])
        err.details["output_path"] = str(path)
        if resumable(err):  # a failed or forgotten job cannot be fetched, only resubmitted
            err.details["fetch_command"] = fetch
        else:
            err.details["recovery"] = ("The job failed or the service no longer has it; fetching cannot help. Fix the "
                                       f"cause if there is one and run {command} again.")
        raise err from None
    _show_result(r, a)
    return 0


def _clips_recovery(r: dict, a, job_id: str, directory) -> dict:
    """A partial result says what to do about each kind of failed clip: fetching again only helps clips
    that were synthesized but not written; a clip that failed in synthesis needs a new tts clips run."""
    if r["status"] == "completed":
        return r
    notes = []
    synth = [c["id"] for c in r["clips"] if c.get("stage") == "synthesis"]
    deliver = [c for c in r["clips"] if c.get("stage") == "delivery"]
    if synth:
        again = ("tts sfx-clips again with only those clips (another seed or a more concrete prompt)"
                 if r.get("kind") == "sfx" else "tts clips again with only those clips (same --voice)")
        notes.append(f"Clips {', '.join(synth)} failed in synthesis (see each error); fetching cannot help: fix the "
                     f"cause if there is one and run {again}.")
    if deliver:
        exists = any(c["error"].get("code") == "output_exists" for c in deliver)
        r = r | {"fetch_command": _fetch_command(a, job_id, directory, clips=True, overwrite=exists)}
        notes.append(f"Clips {', '.join(c['id'] for c in deliver)} were not written (see each error): run "
                     "fetch_command" + (" (--overwrite replaces the existing files)." if exists and not a.overwrite
                                        else "."))
    return r | {"recovery": " ".join(notes)}


def _show_clips(r: dict, a) -> int:
    r = with_timeline(r, a.timeline)
    who = "sound effects" if r.get("kind") == "sfx" else f"voice={r['voice'] or 'default'}"
    lines = [f"{r['status']}: {r['output_dir']}  (model={r['model']}, {who})"]
    for clip in r["clips"]:
        if clip["status"] == "completed":
            lines.append(f"  {clip['id']}: {clip['path']}  ({clip['duration_seconds']:.2f}s)")
            for t in clip.get("timeline", []):
                lines.append(f"      {t['start']:8.3f}-{t['end']:8.3f}  [p{t['paragraph']} s{t['index']}] {t['text']}")
        else:
            lines.append(f"  {clip['id']}: FAILED ({clip['stage']}) {clip['error']['code']}: {clip['error']['message']}")
    if r.get("recovery"):
        lines.append(f"recovery: {r['recovery']}")
    if r.get("fetch_command"):
        lines.append(f"  {r['fetch_command']}")
    _print(r, a.json, "\n".join(lines))
    if not a.json:
        for w in r.get("warnings", []) + [w for clip in r["clips"] for w in clip.get("warnings", [])]:
            print(f"[tts] warning: {w}", file=sys.stderr)
    return 0 if r["status"] == "completed" else EXIT_PARTIAL


def _read_clips(source: str) -> list[tuple[str, str]]:
    raw = sys.stdin.buffer.read() if source == "-" else open(source, "rb").read()
    try:
        data = json.loads(raw.decode("utf-8"))
        clips = [(str(c["id"]), str(c["text"])) for c in data]
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as e:
        raise ClientError("invalid_input", f"{source}: expected a JSON list of {{\"id\": ..., \"text\": ...}} "
                          f"({type(e).__name__}: {e}).") from None
    if not clips:
        raise ClientError("invalid_input", f"{source}: no clips.")
    return clips


async def cmd_clips(c: TTSClient, a) -> int:
    clips = _read_clips(a.file)
    directory = resolve_output_dir(a.dir)
    fmt = a.format or "wav"
    for path in clip_paths(directory, [i for i, _ in clips], fmt).values():
        check_target(path, a.overwrite)
    deadline = Deadline(a.timeout)
    job = await c.submit(None, clips=clips, model=a.model, voice=a.voice, language=a.language, fmt=fmt,
                         seed=a.seed, deadline=deadline)
    return await _finish_clips(c, a, job, directory, deadline, "tts clips")


def _read_sfx_clips(source: str) -> list[dict]:
    raw = sys.stdin.buffer.read() if source == "-" else open(source, "rb").read()
    try:
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, list):
            raise TypeError("not a list")
        clips = []
        for c in data:
            unknown = sorted(set(c) - {"id", "prompt", "seconds", "seed"})
            if unknown:
                raise ClientError("invalid_input", f"{source}: unsupported clip field(s) {', '.join(unknown)}; "
                                  "a clip has id, prompt, seconds and seed.", parameters=unknown)
            clips.append({"id": str(c["id"]), "prompt": str(c["prompt"]), "seconds": c.get("seconds"),
                          "seed": c.get("seed")})
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as e:
        raise ClientError("invalid_input", f"{source}: expected a JSON list of {{\"id\": ..., \"prompt\": ..., "
                          f"\"seconds\"?: ..., \"seed\"?: ...}} ({type(e).__name__}: {e}).") from None
    if not clips:
        raise ClientError("invalid_input", f"{source}: no clips.")
    return clips


async def cmd_sfx_clips(c: TTSClient, a) -> int:
    clips = _read_sfx_clips(a.file)
    directory = resolve_output_dir(a.dir)
    fmt = a.format or "wav"
    for path in clip_paths(directory, [x["id"] for x in clips], fmt).values():
        check_target(path, a.overwrite)
    deadline = Deadline(a.timeout)
    job = await c.submit_sfx(None, clips=clips, seconds=a.seconds, model=a.model, fmt=fmt, seed=a.seed,
                             num_inference_steps=a.steps, cfg_scale=a.cfg_scale, deadline=deadline)
    return await _finish_clips(c, a, job, directory, deadline, "tts sfx-clips")


async def _finish_clips(c: TTSClient, a, job: dict, directory, deadline: Deadline, command: str) -> int:
    """Wait for a submitted clips job (speech or sound effects) and write its clips, or say how to resume."""
    fetch = _fetch_command(a, job["id"], directory, clips=True)
    try:
        if not a.no_wait:
            job = await c.wait(job["id"], deadline, last=job, on_progress=_progress(a.quiet or a.json))
        if job["status"] in ("queued", "running"):
            pend = pending(job) | {"output_dir": str(directory), "fetch_command": fetch}
            _print(pend, a.json, f"pending: job {job['id']} ({job['status']}); fetch later with:\n  {fetch}")
            return EXIT_PENDING
        r = await c.save_clips(job, directory, a.overwrite)
    except Exception as e:  # noqa: BLE001 - the job exists: say how to recover instead of resubmitting
        err = as_client_error(e)
        err.details.setdefault("job_id", job["id"])
        err.details["output_dir"] = str(directory)
        if resumable(err):
            err.details["fetch_command"] = fetch
        else:
            err.details["recovery"] = ("The job failed or the service no longer has it; fetching cannot help. Fix the "
                                       f"cause if there is one and run {command} again (only the clips still needed).")
        raise err from None
    return _show_clips(_clips_recovery(r, a, job["id"], directory), a)


async def cmd_job(c: TTSClient, a) -> int:
    # Until the job's format is known, errors name the path as given (an extensionless path gets
    # the job's extension later, so guessing .wav here could make the fetch command fail).
    if a.output and a.dir:
        raise ClientError("invalid_request", "Give -o (one audio) or -d (a clips job), not both.")
    target = a.output or a.dir
    try:
        if a.output:  # validate the target before waiting
            resolve_output_path(a.output, None)
        if a.dir:
            directory = resolve_output_dir(a.dir)
        if a.wait:
            job = await c.wait(a.job_id, Deadline(a.timeout), on_progress=_progress(a.quiet or a.json))
        else:
            job = await c.job(a.job_id)
        raise_job_error(job)
        if job["status"] == "succeeded" and a.dir:
            r = await c.save_clips(job, directory, a.overwrite)
            return _show_clips(_clips_recovery(r, a, a.job_id, directory), a)
        if job["status"] == "succeeded" and a.output:
            path, _ = resolve_output_path(a.output, job["result"]["format"])
            target = str(path)
            r = await c.save(job, path, a.overwrite)
            _show_result(r, a)
            return 0
    except Exception as e:  # noqa: BLE001 - keep the job reference so the user can retry
        err = as_client_error(e)
        err.details.setdefault("job_id", a.job_id)
        if target:
            err.details["output_dir" if a.dir else "output_path"] = target
            if resumable(err):
                err.details["fetch_command"] = _fetch_command(a, a.job_id, target, clips=bool(a.dir))
        if not resumable(err):
            err.details["recovery"] = ("The job failed or the service no longer has it; fetching cannot help. Fix the "
                                       "cause if there is one and submit it again (tts say / clips / sfx / "
                                       "sfx-clips).")
        raise err from None
    if not getattr(a, "timeline", False) and job.get("result"):  # the timeline repeats the whole input text
        job = job | {"result": with_timeline(job["result"], False)}
    _print(job, True, "")
    return 0 if job["status"] == "succeeded" else EXIT_PENDING


async def cmd_cancel(c: TTSClient, a) -> int:
    try:
        job = await c.cancel(a.job_id)
    except ClientError as e:
        e.details.setdefault("job_id", a.job_id)
        raise
    if job["status"] == "failed":
        _print({"status": "cancelled", "job_id": a.job_id}, a.json, f"cancelled: job {a.job_id}")
    else:
        what = "sound effect" if job_kind(job) == "sfx" else "segment"
        _print({"status": "cancelling", "job_id": a.job_id, "job_status": job["status"]}, a.json,
               f"cancelling: job {a.job_id} stops when the {what} being generated finishes")
    return 0


async def cmd_jobs(c: TTSClient, a) -> int:
    js = await c.jobs()
    _print(js, a.json, "\n".join(
        f"{j['id']}  {j['status']:<9} {j.get('error_code') or '':<10} "
        f"{('sfx ' + str(j.get('seconds')) + 's') if j.get('kind') == 'sfx' else (j['voice'] or '-'):<16} {j['text']}"
        for j in js) or "(no jobs)")
    return 0


async def cmd_unload(c: TTSClient, a) -> int:
    r = await c.unload()
    _print(r, a.json, f"unloaded {r['unloaded']}" if r["unloaded"] else "no model was loaded")
    return 0


async def cmd_add_voice(c: TTSClient, a) -> int:
    r = await c.register_voice(a.name, a.reference, description=a.description, language=a.language,
                               models=a.model, replace=a.replace)
    _print(r, a.json, f"{'replaced' if r['replaced'] else 'registered'} voice {r['name']} "
                      f"({r['seconds']}s, models: {', '.join(r['models'])})")
    return 0


async def cmd_remove_voice(c: TTSClient, a) -> int:
    r = await c.remove_voice(a.name)
    _print(r, a.json, f"removed voice {r['name']}")
    return 0


async def cmd_models(c: TTSClient, a) -> int:
    ms = await c.models()
    def describe(m: dict) -> str:
        head = f"{m['id']}{' (default)' if m['default'] else ''}{' [loaded]' if m['loaded'] else ''}: {m['description']}"
        if m.get("kind") == "sfx":
            x = m.get("sfx") or {}
            return (f"{head}\n  kind: sound effects, {x.get('min_seconds')}-{x.get('max_seconds')} s per clip, "
                    f"{x.get('sample_rate')} Hz; worker installed: {x.get('worker_installed')}\n"
                    f"  options: {', '.join(m['options'])}")
        return (f"{head}\n  languages: {', '.join(m['languages'])}\n"
                f"  capabilities: {', '.join(k for k, v in m['capabilities'].items() if v)}")
    _print(ms, a.json, "\n".join(describe(m) for m in ms))
    return 0


async def cmd_voices(c: TTSClient, a) -> int:
    vs = await c.voices(a.model)
    _print(vs, a.json, "\n".join(f"{v['name']} [{v['language']}] {v['description']} (models: {', '.join(v['models'])})"
                                 for v in vs) or "(no voices)")
    return 0


async def cmd_health(c: TTSClient, a) -> int:
    h = await c.health()
    _print(h, True, "")
    return 0


def main(argv=None) -> int:
    # Common options work before or after the subcommand (`tts --json health`, `tts say ... --json`):
    # the top-level parser holds the defaults; subcommands only override what is actually given.
    def add_common(p, defaults: bool) -> None:
        d = (lambda v: v) if defaults else (lambda v: argparse.SUPPRESS)
        p.add_argument("--url", default=d(DEFAULT_URL), help=f"server URL (default {DEFAULT_URL}; env LOCAL_TTS_URL)")
        p.add_argument("--json", action="store_true", default=d(False), help="machine-readable output")
        p.add_argument("-q", "--quiet", action="store_true", default=d(False), help="no progress on stderr")
    common = argparse.ArgumentParser(add_help=False)
    add_common(common, defaults=False)
    ap = argparse.ArgumentParser(prog="tts", description="Local TTS client (tts-server).")
    add_common(ap, defaults=True)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("say", parents=[common], help="synthesize text to an audio file")
    s.add_argument("text", nargs="?", help='text to speak ("-" = stdin)')
    s.add_argument("-f", "--file", help='read text from a file ("-" = stdin)')
    s.add_argument("-o", "--output", help="absolute output path (.wav/.flac/.mp3); default "
                   "~/.local/share/local-tts/output/<timestamp>.wav")
    s.add_argument("-l", "--language", help="language code or name, e.g. zh, en")
    s.add_argument("-v", "--voice", help="voice name to clone (see `tts voices`)")
    s.add_argument("-m", "--model", help="model id (see `tts models`)")
    s.add_argument("--format", choices=["wav", "flac", "mp3"], help="default: from extension, else wav")
    s.add_argument("--seed", type=int)
    s.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    s.add_argument("--no-wait", action="store_true", help="submit and print the job id (exit 3)")
    s.add_argument("--timeout", type=float, help="stop waiting after N seconds (exit 3, job keeps running)")
    s.add_argument("--timeline", action="store_true", help="report each segment's start/end seconds and text")
    s.set_defaults(func=cmd_say)

    j = sub.add_parser("job", parents=[common], help="show a job, or wait for it and save its audio")
    j.add_argument("job_id")
    j.add_argument("--wait", action="store_true")
    j.add_argument("--timeout", type=float)
    j.add_argument("-o", "--output", help="save the audio here when the job has succeeded")
    j.add_argument("-d", "--dir", help="clips job: write each clip to DIR/<id>.<format> when it has succeeded")
    j.add_argument("--overwrite", action="store_true")
    j.add_argument("--timeline", action="store_true", help="report each segment's start/end seconds and text")
    j.set_defaults(func=cmd_job)

    cl = sub.add_parser("clips", parents=[common], help="synthesize several clips, one file each, in one job")
    cl.add_argument("file", help='JSON list of {"id": ..., "text": ...} ("-" = stdin); id names the file')
    cl.add_argument("-d", "--dir", help="absolute output directory; default a new folder under "
                    "~/.local/share/local-tts/output/")
    cl.add_argument("-l", "--language")
    cl.add_argument("-v", "--voice")
    cl.add_argument("-m", "--model")
    cl.add_argument("--format", choices=["wav", "flac", "mp3"], help="default wav")
    cl.add_argument("--seed", type=int)
    cl.add_argument("--overwrite", action="store_true")
    cl.add_argument("--no-wait", action="store_true")
    cl.add_argument("--timeout", type=float)
    cl.add_argument("--timeline", action="store_true")
    cl.set_defaults(func=cmd_clips)

    def add_sfx_options(p) -> None:
        p.add_argument("-m", "--model", help="sound-effect model id (default: default_sfx, see `tts models`)")
        p.add_argument("--seed", type=int, help="seed (clips: clip i without its own seed gets SEED+i)")
        p.add_argument("--steps", type=int, help="diffusion steps (default: the model's, 100)")
        p.add_argument("--cfg-scale", type=float, help="guidance scale (default: the model's, 4.0)")
        p.add_argument("--overwrite", action="store_true", help="replace existing output files")
        p.add_argument("--no-wait", action="store_true", help="submit and print the job id (exit 3)")
        p.add_argument("--timeout", type=float, help="stop waiting after N seconds (exit 3, job keeps running)")

    fx = sub.add_parser("sfx", parents=[common], help="generate a sound effect to an audio file")
    fx.add_argument("prompt", help="the sound: source and action, concretely (English or Chinese)")
    fx.add_argument("-s", "--seconds", type=float, required=True, help="length in seconds (required; at most 30)")
    fx.add_argument("-o", "--output", help="absolute output path (.wav/.flac/.mp3); default "
                    "~/.local/share/local-tts/output/<timestamp>.wav")
    fx.add_argument("--format", choices=["wav", "flac", "mp3"], help="default: from extension, else wav")
    add_sfx_options(fx)
    fx.set_defaults(func=cmd_sfx, timeline=False)

    fc = sub.add_parser("sfx-clips", parents=[common], help="generate several sound effects, one file each, in one job")
    fc.add_argument("file", help='JSON list of {"id", "prompt", "seconds"?, "seed"?} ("-" = stdin); id names the file')
    fc.add_argument("-d", "--dir", help="absolute output directory; default a new folder under "
                    "~/.local/share/local-tts/output/")
    fc.add_argument("-s", "--seconds", type=float, help="length of every clip that gives none (each clip needs one)")
    fc.add_argument("--format", choices=["wav", "flac", "mp3"], help="default wav")
    add_sfx_options(fc)
    fc.set_defaults(func=cmd_sfx_clips, timeline=False)

    k = sub.add_parser("cancel", parents=[common], help="cancel a queued or running job")
    k.add_argument("job_id")
    k.set_defaults(func=cmd_cancel)

    sub.add_parser("jobs", parents=[common], help="list the jobs the service still knows").set_defaults(func=cmd_jobs)
    sub.add_parser("unload", parents=[common], help="unload the model now (only when idle)").set_defaults(
        func=cmd_unload)
    av = sub.add_parser("add-voice", parents=[common], help="register a voice from a 3-30 s recording of one speaker")
    av.add_argument("name", help="letters, digits, '_', '-', '.'")
    av.add_argument("reference", help="absolute path of the reference recording (WAV or FLAC)")
    av.add_argument("--description", default="")
    av.add_argument("-l", "--language", help="the recording's language, e.g. zh, en")
    av.add_argument("-m", "--model", action="append", help="model the voice is for (repeatable; default: all "
                    "models that can clone)")
    av.add_argument("--replace", action="store_true", help="replace a registered voice with this name")
    av.set_defaults(func=cmd_add_voice)
    rv = sub.add_parser("remove-voice", parents=[common], help="remove a registered voice")
    rv.add_argument("name")
    rv.set_defaults(func=cmd_remove_voice)

    sub.add_parser("models", parents=[common], help="list models").set_defaults(func=cmd_models)
    v = sub.add_parser("voices", parents=[common], help="list voices")
    v.add_argument("-m", "--model")
    v.set_defaults(func=cmd_voices)
    sub.add_parser("health", parents=[common], help="service status").set_defaults(func=cmd_health)

    a = ap.parse_args(argv)
    return asyncio.run(_run(a))


async def _run(a) -> int:
    c = TTSClient(a.url)
    try:
        return await a.func(c, a)
    except (ClientError, OSError, UnicodeError) as e:
        print(json.dumps({"error": as_client_error(e).to_dict()}, ensure_ascii=False), file=sys.stderr)
        return EXIT_ERROR
    finally:
        await c.aclose()


if __name__ == "__main__":
    sys.exit(main())
