---
name: sound-effects
description: Generate sound effects with the local sound-effect model (MCP server local-tts). Use whenever a video, animation, game prototype, UI demo or audio story needs non-speech sound — effects, foley, ambience, transitions, UI sounds, impacts — including requests worded as 音效、環境音、擬音、sound effect、SFX、foley、ambience. Not for speech (use the narration skill) and not for background music or songs (the model only makes short music-like clips).
---

# Sound effects with the local model

A local GPU text-to-sound-effect model (MOSS-SoundEffect v2.0, 48 kHz mono, at most 30 s per clip)
runs in the same service as the narration voice. When a deliverable needs non-speech sound, generate it
yourself instead of leaving silent gaps or asking the user to find stock sounds.

## Tools

MCP server `local-tts` (load them with ToolSearch if they are deferred):

- `mcp__local-tts__generate_sound_effect_clips` — several effects in one job, each to
  `<output_dir>/<id>.<format>`. The normal way: one call per deliverable.
- `mcp__local-tts__generate_sound_effect` — one effect to `output_path`.
- `mcp__local-tts__get_sound_effect_job` / `get_sound_effect_clips` — keep waiting for a pending job.
- `cancel_speech_job`, `list_speech_jobs`, `unload_model`, `list_models`, `tts_status` cover sound-effect
  jobs too.

If MCP is unavailable, use the CLI with a bounded wait:
`tts sfx "a heavy wooden door slams shut" -s 3 -o /abs/path/door.wav --json --timeout 45`, or for
several effects `tts sfx-clips effects.json -d /abs/dir --json --timeout 45` (effects.json:
`[{"id": "door", "prompt": "...", "seconds": 3}, ...]`, `-s N` for clips without `seconds`). Exit 0 =
done, 1 = error (JSON on stderr), 3 = still running: run the returned `fetch_command` with
`--timeout 45` added while it exits 3; 4 = some clips failed or were not written: follow `recovery`.
Never rerun `tts sfx` / `tts sfx-clips` for a job that exists.

## Workflow

1. **List the effects from your own timeline**: what sounds, when, how long. One clip per distinct
   sound (a door slam, rain bed, a UI click), not one long soundtrack.
2. **Describe each sound concretely**, in English or Chinese: the source, the action, the material or
   surface, the space and distance ("footsteps on wet gravel, slow, close", "large wooden door slams in
   a stone hallway, short echo"). Avoid abstract moods ("tense", "epic") and anything that needs words
   or a melody. Give `seconds` for every clip (required, ≤ 30; the model rounds to 0.1 s). Leave
   `num_inference_steps` and `cfg_scale` at their defaults.
3. **One call for all effects of a deliverable**: `generate_sound_effect_clips` with
   `clips: [{"id": "door", "prompt": ..., "seconds": 3}, ...]` and an absolute `output_dir` inside the
   project (e.g. `assets/sfx/`). Limits per call: 100 clips, 300 s in total, 2,000 characters per prompt.
   The model loads once per call; the first clip after a load is slow (compile warm-up), later clips
   take about the same time whatever their length.
4. **Do narration first or last, not interleaved.** Only one model is on the GPU: switching between
   speech and sound effects unloads one and loads the other (tens of seconds each way). Batch all
   speech, then all effects (or the reverse).
5. **Place effects yourself** with ffmpeg (they are mono 48 kHz). Delay each effect to its start time
   in milliseconds and mix, e.g. over a video with narration:
   `ffmpeg -i video.mp4 -i narration.wav -i door.wav -i rain.wav -filter_complex "[2:a]adelay=4200:all=1[d];[3:a]volume=0.4[r];[1:a][d][r]amix=inputs=3:duration=longest:normalize=0[a]" -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k out.mp4`.
   `duration=longest` keeps every effect, even one placed after the narration ends (make the video
   at least that long); `normalize=0` keeps each input's level.
   Trim or fade with `atrim` / `afade`; a longer bed can be looped with `-stream_loop`.
6. **Check what you cannot hear.** For each file: duration with
   `ffprobe -v error -show_entries format=duration -of csv=p=0 door.wav`, and loudness with
   `ffmpeg -i door.wav -af volumedetect -f null - 2>&1 | grep -E "mean_volume|max_volume"` (max_volume
   below about -40 dB means near silence). Where an audio-capable model is available, ask it to
   describe the clip without telling it the prompt, and compare. If a clip does not match, regenerate
   that clip only (`overwrite: true`) with another `seed` or a more concrete prompt.
7. **Report** the file paths, durations, prompts and seeds (the result gives the seed used).

## Jobs and errors

- Status `pending` → call `next.tool` with exactly `next.arguments` until done. Never resubmit a
  pending job; `list_speech_jobs` finds a lost job_id. Pending results carry
  `estimated_remaining_seconds`.
- An error with a `job_id` says what to do in `recovery`; follow it. `wrong_job_kind`: you used a
  speech getter on a sound-effect job (or the reverse); call the tool it names. A `partial` clips
  result: follow its `recovery`. Clips that failed in generation (`stage: "synthesis"`) need a new call
  with only those clips (another seed or prompt); clips that were not written (`stage: "delivery"`)
  already exist on the server: fetch them again with the getter it names, do not regenerate them.
- `warnings` on a result (e.g. the model runs without torch.compile): slower, not wrong; mention it.
- `invalid_seconds` / `too_many_seconds` / `input_too_long`: fix the request. `wrong_model_kind`: omit
  `model` or pick a sound-effect model from `available`.
- `sfx_worker_not_installed` / `model_weights_missing`: the service lacks the model; tell the user the
  `install_command` / `download_command` from the error. `gpu_unavailable` / `gpu_out_of_memory`:
  check `tts_status` and `nvidia-smi`, then tell the user. `sfx_worker_failed` / `sfx_worker_timeout`:
  submit once more; if it repeats, tell the user (logs: `journalctl --user -u local-tts`).
- `server_unreachable`: `systemctl --user start local-tts`. When all audio is done and the next step
  needs the GPU, `unload_model` frees it at once.
