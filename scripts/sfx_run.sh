#!/bin/bash
# GPU acceptance of sound effects: starts tts-server from this checkout on a non-default port, runs
# scripts/sfx_smoke.py against it, samples nvidia-smi, and stops the server by its exact PID.
# It does not stop another local-tts service: unload it first (POST /v1/models/unload) so the two do
# not compete for GPU memory.
# Usage: scripts/sfx_run.sh <out_dir> [port]   (port: $LOCAL_TTS_PORT, else 8791)
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(realpath -m "${1:?usage: scripts/sfx_run.sh <out_dir> [port]}")
PORT=${2:-${LOCAL_TTS_PORT:-8791}}
mkdir -p "$OUT"
cd "$ROOT"
export HF_HUB_OFFLINE=1 TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1

SERVER_PID="" SMI=""
cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null
  [ -n "$SMI" ] && kill "$SMI" 2>/dev/null
}
trap cleanup EXIT

# Refuse a port someone else serves: the new server would fail to bind, and the checks would then run
# against the other service (possibly another checkout) without anyone noticing.
if ! .venv/bin/python -c 'import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError as e:
    sys.exit(f"port {sys.argv[1]} is in use ({e.strerror}): pick another, e.g. scripts/sfx_run.sh <out_dir> <port>")' "$PORT"; then
  exit 1
fi
if [ ! -x workers/moss-sfx/.venv/bin/python ]; then
  echo "sound-effect worker not installed: run uv sync --frozen --project workers/moss-sfx"; exit 1
fi
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 > "$OUT/nvidia-smi.csv" &
  SMI=$!
fi
LOCAL_TTS_PORT=$PORT .venv/bin/tts-server > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
READY=0
for _ in $(seq 1 60); do
  # Our own server must still be running: a /health answer from anything else does not count.
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "server exited during startup (see $OUT/server.log)"; SERVER_PID=""; exit 1; }
  curl -sf -m 5 "localhost:$PORT/health" >/dev/null && { READY=1; break; }
  sleep 1
done
[ $READY = 1 ] && kill -0 "$SERVER_PID" 2>/dev/null || { echo "server on $PORT did not start (see $OUT/server.log)"; exit 1; }

.venv/bin/python scripts/sfx_smoke.py "$OUT" --url "http://127.0.0.1:$PORT" 2>&1 | tee "$OUT/smoke.log"
RC=${PIPESTATUS[0]}
[ -s "$OUT/nvidia-smi.csv" ] && \
  echo "nvidia-smi whole-GPU max sample (500 ms) MiB: $(cut -d, -f2 "$OUT/nvidia-smi.csv" | sort -n | tail -1)"
exit "$RC"
