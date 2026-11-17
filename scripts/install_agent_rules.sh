#!/bin/bash
# Install the agent rules as skills (narration, sound-effects) for Claude Code (~/.claude/skills) and
# Codex (~/.codex/skills). A skill costs an agent one description line until a task needs it, so nothing
# is written to ~/.codex/AGENTS.md, which every Codex task in every project loads.
# Earlier versions put marked blocks (<!-- local-tts:begin/end -->, <!-- local-tts-sfx:begin/end -->) in
# ~/.codex/AGENTS.md: those are removed, the rest of the file is kept, and the file is first backed up to
# AGENTS.md.bak-local-tts unless that backup exists. When the markers of a block are not exactly one
# well-ordered pair it refuses and changes nothing (no skill, no backup, no AGENTS.md).
# Idempotent. {{LOCAL_TTS_ROOT}} in the rule sources becomes this checkout's absolute path.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)

python3 - "$ROOT" "$HOME" <<'PY'
import os, shutil, sys, tempfile
root, home = sys.argv[1], sys.argv[2]
SKILLS = ("narration", "sound-effects")
TARGETS = (".claude", ".codex")
LEGACY_BLOCKS = ("local-tts", "local-tts-sfx")
agents = os.path.join(home, ".codex", "AGENTS.md")

# 1. Render and check everything, writing nothing.
skills = {}
for name in SKILLS:
    with open(os.path.join(root, "agent-rules", "skills", name, "SKILL.md"), encoding="utf-8") as f:
        skills[name] = f.read().replace("{{LOCAL_TTS_ROOT}}", root)
text = open(agents, encoding="utf-8").read() if os.path.exists(agents) else ""
new = text
for name in LEGACY_BLOCKS:
    begin, end = f"<!-- {name}:begin -->", f"<!-- {name}:end -->"
    nb, ne = new.count(begin), new.count(end)
    if (nb, ne) == (0, 0):
        continue
    if (nb, ne) != (1, 1) or new.index(begin) > new.index(end):
        sys.exit(f"{agents}: expected one ordered pair of {name} markers, found {nb} begin / {ne} end "
                 "(or blocks that overlap); fix the file by hand (nothing was changed)")
    before, after = new[:new.index(begin)].rstrip("\n"), new[new.index(end) + len(end):].strip("\n")
    new = before + ("\n\n" if before and after else "") + after + ("\n" if before or after else "")


def write_atomic(path: str, data: str, mode: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=f".{os.path.basename(path)}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, path)  # atomic: readers see the old or the new file, never a partial one
    except BaseException:
        os.unlink(tmp)
        raise


# 2. Only now write: the skills, then AGENTS.md if it held legacy blocks.
for target in TARGETS:
    for name, data in skills.items():
        path = os.path.join(home, target, "skills", name, "SKILL.md")
        write_atomic(path, data, 0o644)
        print(f"installed {path}")
if new != text:
    if not os.path.exists(agents + ".bak-local-tts"):
        shutil.copy2(agents, agents + ".bak-local-tts")
    write_atomic(agents, new, os.stat(agents).st_mode & 0o777)
    print(f"removed the old local-tts blocks from {agents}")
PY
