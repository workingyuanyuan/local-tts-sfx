---
name: narration
description: Generate spoken audio with the local TTS service (MCP server local-tts). Use whenever a task's deliverable needs a voice — narration or voice-over for a video, slide deck, demo, tutorial or animation; reading teaching material, an article or a story aloud; podcast or audiobook-style audio; an audio preview of a script — including requests worded as 旁白、配音、朗讀、語音、唸出來、有聲、TTS、text to speech. Do not use it when the user only wants the script text, or asks for a different TTS provider.
---

# Narration with local TTS

A local GPU text-to-speech service (MOSS-TTS, 31 languages, 48 kHz stereo) is always available on this
machine. When a task needs spoken audio, generate it yourself: do not stop at writing a script and
asking the user to record or synthesize it elsewhere.

## Tools

MCP server `local-tts` (load them with ToolSearch if they are deferred):

- `mcp__local-tts__generate_speech` — text → audio file. Returns path, duration_seconds, sample_rate,
  channels, format, model, voice; or status `pending` with a job_id for long text.
- `mcp__local-tts__generate_speech_clips` — several clips (one per slide or scene) in one job, each to
  `<output_dir>/<id>.<format>`: faster than one call per clip, and one narrator.
- `mcp__local-tts__get_speech_job` / `get_speech_clips` — keep waiting for a pending job and write it.
- `mcp__local-tts__cancel_speech_job` — stop a job that is no longer wanted (wrong text, superseded
  script) so the GPU is free for the replacement. `list_speech_jobs` finds a lost job_id.
- `mcp__local-tts__register_voice` — make a recording usable as `voice` (see step 2).
- `mcp__local-tts__unload_model` — free the GPU memory once all narration is done.
- `mcp__local-tts__list_voices`, `mcp__local-tts__list_models`, `mcp__local-tts__tts_status`.
- Non-speech sound (effects, foley, ambience) comes from the same service: see the `sound-effects` skill.

If MCP is unavailable, use the CLI with a bounded wait:
`tts say -f script.txt -l zh -o /abs/path/narration.wav --json --timeout 45`.
Several clips: `tts clips deck.json -d /abs/dir -l zh --json --timeout 45` (deck.json:
`[{"id": "s01", "text": "..."}]`). Exit 0 = done, 1 = error (JSON on stderr), 3 = still running: the
JSON has `job_id` and a `fetch_command`; run that command with `--timeout 45` added, repeating while
it exits 3. 4 = clips finished but some failed or were not written: follow the JSON's `recovery` (a
`fetch_command` is given only when fetching again can help). Never rerun
`tts say` / `tts clips` for a job that exists.

## Workflow

1. **Write the script for the ear.** Short sentences; punctuation shapes the prosody. Write numbers,
   units, symbols and abbreviations the way they should be spoken when it is ambiguous ("2026年" is
   fine, "v1.5" → "一點五版" / "version one point five"). Use `[pause 1.0s]` for deliberate pauses and
   blank lines between paragraphs. For a mispronounced Chinese word, write it as tone-numbered pinyin
   separated by spaces, e.g. `您来自哪 zuo4 cheng2 shi4？`; prefer the common form of rare or variant
   characters (徑流, not 逕流). Save the script next to the audio (e.g.
   `narration.txt`) so it can be revised and regenerated.
2. **Choose model and voice.** Omit `model` (the default is the only one). Always pass `language`
   (`zh`, `en`, `ja`, …; the main language for mixed text).
   - Without `voice`, the model picks a voice per call. Within one call (a `generate_speech_clips`
     call included) the service tries to keep it (later segments reuse the first ≥ 3 s of output as
     reference), but separate calls sound like different speakers. Whenever separate calls must sound
     like the same narrator, pass the same `voice` to every call: call `list_voices` and pick one
     matching the language.
   - To clone a recording, register it with `register_voice` (absolute path, 3-30 s of one speaker;
     usable at once). A real person's voice, including the user's own, only with the user's explicit
     approval: ask first.
3. **Write into the task's project directory** with an absolute `output_path`, following the project's
   existing layout (e.g. `assets/audio/`, else `audio/`). Use descriptive names (`scene01.wav`).
   Format: `.wav` for editing and video pipelines, `.mp3` for sharing, `.flac` for lossless and smaller.
   Never overwrite an existing file unless you are deliberately regenerating that file
   (`overwrite: true`); on `output_exists` choose another name or ask.
4. **Split by how the audio will be used.** Video or slides: one file per scene or slide, so each
   duration sets that scene's length; generate them all with one `generate_speech_clips` call
   (`clips: [{"id": "slide01", "text": ...}, ...]`, `output_dir` in the project) rather than one call
   per clip. One continuous reading (article, story, podcast): one `generate_speech` call with the
   full text; the service splits long text itself. Pass `timeline: true` to also get each
   segment's start/end seconds, paragraph and text; paragraphs (blank lines) always start a new
   segment, so a single call can still be cut at paragraph breaks. Limits per call (all clips of a
   clips call together): 20,000 characters, each `[pause]` ≤ 60 s and all pauses ≤ 600 s in total
   (`input_too_long` / `invalid_pause` otherwise); up to 100 clips. Longer text: split at paragraph
   boundaries into batches with one fixed `voice`, generate them in order and join them (step 6).
   Jobs run one at a time on one GPU: make calls sequentially, not in parallel.
5. **Long text and interrupted calls.** Status `pending` → call `next.tool` (`get_speech_job` or
   `get_speech_clips`) with exactly `next.arguments` (job_id, the original absolute path, `overwrite`
   if you set it) until done. Always pass the path: after an MCP restart the default location is
   outside the project. Never submit text again while it is still pending; if it is no longer wanted,
   `cancel_speech_job` it first. When an error carries a `job_id`, follow its `recovery` field:
   - interrupted (connection, timeout, write error): the fetch tool with the arguments it gives;
   - `job_status: "failed"`, or `unknown_job`: the job is over; fix the cause if there is one and
     submit again.
   A clips result with status `partial` lists each failed clip with `stage` and `error`, and its
   `recovery` says what to do: clips that failed in synthesis need a new `generate_speech_clips`
   call with only those clips (same voice); clips that were not written need `get_speech_clips` again.
   Pending results carry `estimated_remaining_seconds` (measured on this machine; loading the model
   when idle adds up to ~20 s). Lost a job_id (e.g. after a context reset)? `list_speech_jobs` finds
   it; do not resubmit the text.
6. **Hand the result to the next step.** Use the returned `duration_seconds` for timing, then verify
   the final product with `ffprobe`. Typical ffmpeg steps:
   - join clips: `ffmpeg -f concat -safe 0 -i list.txt -c copy narration.wav` (same format clips)
   - narration onto video: `ffmpeg -i video.mp4 -i narration.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k out.mp4`
     (make the video at least as long as the narration; `-shortest` would cut speech off)
   - still image + narration: `ffmpeg -loop 1 -i slide.png -i slide.wav -c:v libx264 -tune stillimage -pix_fmt yuv420p -c:a aac -shortest slide.mp4`
7. **Report** the audio file paths, durations and the voice used.

## Errors

Errors are JSON `{"error": {"code", "message", ...}}`.
- `server_unreachable`: `systemctl --user start local-tts`, then `tts health`.
- A completed result (or clip) with `warnings` (a segment stayed suspiciously short: a sentence may be
  missing; the warning quotes the passage): regenerate that file or clip once (`overwrite: true`); if
  the warning persists, tell the user which file and passage.
- `queue_full`/`results_full`: wait a little and retry. `unknown_voice` / `unknown_model`: pick from `available`.
- `cancelled`: the job was cancelled (by you or someone else); submit again if it is still needed.
  `job_finished`: the job had already finished, nothing to cancel.
- `output_exists`: see step 3. `invalid_path`: the path must be absolute (Windows `C:\...` paths work).
- `segment_failed` / `segment_truncated` name the failing segment: rephrase or shorten that passage
  and submit again. `unknown_job` (the service restarted or the job expired): submit it again.
- `gpu_out_of_memory`: not enough GPU memory. Check `tts_status`, `nvidia-smi` and the logs, then tell
  the user what you found. When all narration is done and the next step needs the GPU (Blender,
  another model), `unload_model` frees the TTS model's memory at once.
- Logs: `journalctl --user -u local-tts`.
