#!/usr/bin/env bash
# Smoke the served K2-Horizon path: the probe backend behind the server.
#
#   tools/k2-serve-smoke.sh [install]      # default models/k2-horizon-4bit.ssdai
#
# Starts TinyTitanServer on the single-model path, then checks the surfaces a
# client actually uses: the model list, a chat completion, SSE streaming, the
# Responses API, tool calls and a stop string. Each check prints ok or FAILED;
# the exit status is the number of failures.
#
# This is a manual gate, not CI: it needs a real install and a GPU. The install
# is not modified, which is why it does not go through golden-baseline.sh (that
# drives the CLI, and the CLI has no K2 path).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER="$ROOT/.build/release/TinyTitanServer"
PORT="${PORT:-8799}"
TIMEOUT="${TIMEOUT:-900}"
INSTALL="${1:-$ROOT/models/k2-horizon-4bit.ssdai}"

failures=0
pass() { echo "  ok: $1"; }
fail() {
  echo "  FAILED: $1"
  if [ -n "${2-}" ]; then printf '%s\n' "$2" | head -5 | sed 's/^/    /'; fi
  failures=$((failures + 1))
}

if [ ! -x "$SERVER" ]; then
  echo "missing $SERVER — run: swift build -c release" >&2
  exit 2
fi
if [ ! -f "$INSTALL/manifest.json" ]; then
  echo "missing K2 install at $INSTALL" >&2
  exit 2
fi

port_busy() {
  curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/v1/models"
}
if port_busy; then
  echo "port $PORT is already serving; refusing to race it" >&2
  exit 2
fi

LOG="$(mktemp "${TMPDIR:-/tmp}/k2-smoke.XXXXXX")"
SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ "$failures" -ne 0 ]; then
    echo "server log: $LOG"
  else
    rm -f "$LOG"
  fi
}
trap cleanup EXIT INT TERM

echo "== starting $SERVER --model $INSTALL (port $PORT)"
"$SERVER" --model "$INSTALL" --port "$PORT" --reasoning high >"$LOG" 2>&1 &
SERVER_PID=$!

ready=0
waited=0
while [ "$waited" -lt "$TIMEOUT" ]; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited during load:" >&2
    tail -5 "$LOG" >&2
    exit 2
  fi
  if port_busy >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 3
  waited=$((waited + 3))
done
if [ "$ready" -ne 1 ]; then
  echo "server not ready after ${TIMEOUT}s" >&2
  tail -5 "$LOG" >&2
  exit 2
fi

BASE="http://127.0.0.1:$PORT"
MODEL="$(basename "$INSTALL")"

# post <path> <json> <output-file>: body and status, no retries.
post() {
  curl -s --max-time 600 -o "$3" -w '%{http_code}' \
    -H 'Content-Type: application/json' "$BASE$1" -d "$2"
}

# get <path> <output-file>: body and status.
get() {
  curl -s --max-time 60 -o "$2" -w '%{http_code}' "$BASE$1"
}

json_field() {  # <file> <python expression over d>
  python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(eval(sys.argv[2], {"d": d}))
' "$1" "$2"
}

echo "== models"
if [ "$(get /v1/models "$LOG.models")" = "200" ] \
  && grep -q "$MODEL" "$LOG.models"; then
  pass "model list names $MODEL"
else
  fail "model list" "$(cat "$LOG.models" 2>/dev/null)"
fi

echo "== chat completion"
status="$(post /v1/chat/completions "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Name the capital of France.\"}],\"max_tokens\":24,\"temperature\":0,\"reasoning_effort\":\"low\"}" "$LOG.chat")"
if [ "$status" = "200" ] \
  && [ "$(json_field "$LOG.chat" 'd["choices"][0]["finish_reason"]')" = "stop" ] \
  && [ -n "$(json_field "$LOG.chat" 'd["choices"][0]["message"]["content"]')" ]; then
  pass "chat completion answered: $(json_field "$LOG.chat" 'd["choices"][0]["message"]["content"][:40]')"
else
  fail "chat completion (status $status)" "$(cat "$LOG.chat" 2>/dev/null)"
fi

echo "== streaming"
stream="$(curl -s -N --max-time 600 "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Name the capital of France.\"}],\"max_tokens\":16,\"temperature\":0,\"reasoning_effort\":\"low\",\"stream\":true}" | head -20)"
if printf '%s' "$stream" | grep -q 'chat.completion.chunk' \
  && printf '%s' "$stream" | grep -q '"delta"'; then
  pass "SSE delivered chunk deltas"
else
  fail "SSE streaming" "$stream"
fi

echo "== responses api"
status="$(post /v1/responses "{\"model\":\"$MODEL\",\"input\":\"Name the capital of France.\",\"max_output_tokens\":24}" "$LOG.responses")"
if [ "$status" = "200" ] \
  && [ "$(json_field "$LOG.responses" 'd["object"]')" = "response" ]; then
  pass "responses object returned"
else
  fail "responses api (status $status)" "$(cat "$LOG.responses" 2>/dev/null)"
fi

echo "== tool call"
status="$(post /v1/chat/completions "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the get_weather tool.\"}],\"max_tokens\":96,\"temperature\":0,\"reasoning_effort\":\"low\",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Current weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]}" "$LOG.tools")"
if [ "$status" = "200" ] \
  && [ "$(json_field "$LOG.tools" 'd["choices"][0]["message"]["tool_calls"][0]["function"]["name"]')" = "get_weather" ] \
  && [ "$(json_field "$LOG.tools" 'd["choices"][0]["message"]["content"]')" = "None" ]; then
  pass "tool call parsed, content empty"
else
  fail "tool call (status $status)" "$(cat "$LOG.tools" 2>/dev/null)"
fi

echo "== stop string"
status="$(post /v1/chat/completions "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word: the capital of France.\"}],\"max_tokens\":24,\"temperature\":0,\"reasoning_effort\":\"low\",\"stop\":[\"Paris\"]}" "$LOG.stop")"
if [ "$status" = "200" ] \
  && [ "$(json_field "$LOG.stop" 'd["choices"][0]["finish_reason"]')" = "stop" ] \
  && [ "$(json_field "$LOG.stop" 'd["choices"][0]["message"]["content"]')" = "" ]; then
  pass "stop string ended the turn and was stripped"
else
  fail "stop string (status $status)" "$(cat "$LOG.stop" 2>/dev/null)"
fi

echo "== context bound"
python3 - "$MODEL" > "$LOG.long.body" <<'PY'
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": "word " * 9000}],
    "max_tokens": 8,
}))
PY
status="$(curl -s --max-time 120 -o "$LOG.long" -w '%{http_code}' \
  -H 'Content-Type: application/json' "$BASE/v1/chat/completions" \
  --data-binary @"$LOG.long.body")"
if [ "$status" = "400" ] \
  && grep -q 'context_length_exceeded' "$LOG.long"; then
  pass "overlong prompt is a 400 context_length_exceeded"
else
  fail "overlong prompt (status $status)" "$(cat "$LOG.long" 2>/dev/null)"
fi

if [ "$failures" -ne 0 ]; then
  echo "$failures check(s) failed"
  exit "$failures"
fi
echo "all checks passed"
