#!/bin/bash
# Install the agent rules: the Claude Code skills (narration, sound-effects) and one marked block per
# topic in ~/.codex/AGENTS.md (<!-- local-tts:begin/end --> for narration, <!-- local-tts-sfx:begin/end -->
# for sound effects).
# Idempotent: re-running replaces the skills and the blocks between their markers; the rest of
# AGENTS.md is kept. Everything is rendered and checked before anything is written: when the markers of
# any block are not exactly zero or one well-ordered pair, or two blocks overlap or nest, it refuses and
# changes nothing (no skill, no backup, no AGENTS.md). The first run backs up an existing AGENTS.md to
# AGENTS.md.bak-local-tts. {{LOCAL_TTS_ROOT}} in the rule sources becomes this checkout's absolute path.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)

python3 - "$ROOT" "$HOME" <<'PY'
import os, shutil, sys, tempfile
root, home = sys.argv[1], sys.argv[2]
SKILLS = ("narration", "sound-effects")
BLOCKS = (("local-tts", "agent-rules/codex/AGENTS.local-tts.md"),
          ("local-tts-sfx", "agent-rules/codex/AGENTS.local-tts-sfx.md"))
agents = os.path.join(home, ".codex", "AGENTS.md")


def render(rel: str) -> str:
    return open(os.path.join(root, rel), encoding="utf-8").read().replace("{{LOCAL_TTS_ROOT}}", root)


# 1. Render and check everything, writing nothing.
skills = {name: render(f"agent-rules/claude/{name}/SKILL.md") for name in SKILLS}
text = open(agents, encoding="utf-8").read() if os.path.exists(agents) else ""
spans, missing = [], []  # all positions refer to the original text
for name, rel in BLOCKS:
    begin, end = f"<!-- {name}:begin -->", f"<!-- {name}:end -->"
    block = f"{begin}\n{render(rel).strip()}\n{end}"
    nb, ne = text.count(begin), text.count(end)
    if (nb, ne) == (0, 0):
        missing.append(block)
    elif (nb, ne) == (1, 1) and text.index(begin) < text.index(end):
        spans.append((text.index(begin), text.index(end) + len(end), name, block))
    else:
        sys.exit(f"{agents}: expected zero or one ordered pair of {name} markers, found {nb} begin / {ne} end; "
                 "fix the file by hand (nothing was changed)")
spans.sort()
for (s1, e1, n1, _), (s2, e2, n2, _) in zip(spans, spans[1:]):
    if s2 < e1:
        sys.exit(f"{agents}: the {n1} and {n2} blocks overlap or nest; fix the file by hand (nothing was changed)")
new = text
for start, stop, _, block in reversed(spans):  # back to front: earlier positions stay valid
    new = new[:start] + block + new[stop:]
for block in missing:
    new = new.rstrip("\n") + ("\n\n" if new.strip() else "") + block + "\n"


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


# 2. Only now write: skills, the one-time backup, AGENTS.md.
for name, data in skills.items():
    path = os.path.join(home, ".claude", "skills", name, "SKILL.md")
    write_atomic(path, data, 0o644)
    print(f"installed {path}")
if os.path.exists(agents) and not os.path.exists(agents + ".bak-local-tts"):
    shutil.copy2(agents, agents + ".bak-local-tts")
write_atomic(agents, new, os.stat(agents).st_mode & 0o777 if os.path.exists(agents) else 0o644)
print(f"updated {agents} (blocks between local-tts and local-tts-sfx markers)")
PY
