#!/bin/bash
# Wait for the Laguna conversion, then install and benchmark both widths.
#
# The conversion is a 235 GB stream and the installs consume their snapshots as
# they are written (the two snapshots plus an install do not fit). This runs
# unattended: it waits for both snapshot indexes, validates each snapshot
# against its shape contract before touching it, imports one width at a time
# with --consume-snapshot, and records the decode benchmark for each.
#
#   tools/laguna_pipeline.sh >> .build/laguna-pipeline.log 2>&1 &
#
# It is safe to re-run: an install that verifies is not rebuilt, and the
# benchmark is re-run against it.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SNAP4="$ROOT/.build/laguna-affine-4bit"
SNAP8="$ROOT/.build/laguna-affine-8bit"
REPACK="$ROOT/.build/out/Products/Release/TinyTitanRepack"
BENCH="$ROOT/.build/out/Products/Release/TinyTitanBench"
INSTALL4="$ROOT/models/laguna-s-2.1-4bit.ssdai"
INSTALL8="$ROOT/models/laguna-s-2.1-8bit.ssdai"
PYTHON="${TINYTITAN_PYTHON:-python3}"
PROMPT="What is the capital of France?"
DEADLINE_SECONDS=$((96 * 3600))

say() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  say "FAILED: $*"
  exit 1
}

[ -x "$REPACK" ] || die "no repacker at $REPACK"
[ -x "$BENCH" ] || die "no bench at $BENCH"

install_ok() {
  local install="$1"
  [ -d "$install" ] &&
    "$REPACK" --verify-install --input-ssdai "$install" >/dev/null 2>&1
}

need4=1
need8=1
if install_ok "$INSTALL4"; then need4=0; fi
if install_ok "$INSTALL8"; then need8=0; fi

if [ "$need4" -eq 1 ] || [ "$need8" -eq 1 ]; then
  say "waiting for the conversion (deadline ${DEADLINE_SECONDS}s)"
  started=$SECONDS
  while :; do
    if { [ "$need4" -eq 0 ] || [ -f "$SNAP4/model.safetensors.index.json" ]; } &&
      { [ "$need8" -eq 0 ] || [ -f "$SNAP8/model.safetensors.index.json" ]; }; then
      break
    fi
    if [ $((SECONDS - started)) -ge "$DEADLINE_SECONDS" ]; then
      die "conversion did not finish within ${DEADLINE_SECONDS}s"
    fi
    sleep 60
  done

  if [ "$need4" -eq 1 ]; then
    count=$(ls "$SNAP4"/layer-*.safetensors 2>/dev/null | wc -l | tr -d ' ')
    [ "$count" -eq 47 ] || die "$SNAP4 has $count layer files, expected 47"
    "$PYTHON" "$ROOT/tools/verify_laguna_snapshot.py" \
      --snapshot "$SNAP4" --bits 4 ||
      die "snapshot verification failed for $SNAP4"
  fi
  if [ "$need8" -eq 1 ]; then
    count=$(ls "$SNAP8"/layer-*.safetensors 2>/dev/null | wc -l | tr -d ' ')
    [ "$count" -eq 47 ] || die "$SNAP8 has $count layer files, expected 47"
    "$PYTHON" "$ROOT/tools/verify_laguna_snapshot.py" \
      --snapshot "$SNAP8" --bits 8 ||
      die "snapshot verification failed for $SNAP8"
  fi
  say "the snapshots still needed verify"
else
  say "both installs already verify; benchmarking them"
fi
say "both snapshots verify"

install_one() {
  local bits="$1" snapshot="$2" install="$3"
  if [ -d "$install" ]; then
    if "$REPACK" --verify-install --input-ssdai "$install" >/dev/null 2>&1; then
      say "$bits-bit install already verifies, not rebuilding"
      return 0
    fi
    die "$install exists but does not verify; inspect it rather than overwriting"
  fi
  say "repacking $bits-bit from $snapshot (consuming it)"
  "$REPACK" --input-snapshot "$snapshot" \
    --model-id "laguna-s-2.1-$bits" \
    --output "$install" --consume-snapshot ||
    die "$bits-bit repack failed"
  "$REPACK" --verify-install --input-ssdai "$install" ||
    die "$bits-bit install does not verify"
}

benchmark_one() {
  local bits="$1"
  local install="$2"
  local log="$ROOT/.build/laguna-bench-$bits-bit.log"
  say "benchmarking $bits-bit"
  df -h "$ROOT" | tail -1
  TINYTITAN_LAGUNA_CHAT=1 "$BENCH" lagunagpu "$install" "$PROMPT" 4 >"$log" 2>&1 ||
    die "$bits-bit benchmark failed (see $log)"
  cat "$log"
  say "$bits-bit: $(grep '^decode' "$log" | tail -1)"
}

install_one 4 "$SNAP4" "$INSTALL4"
benchmark_one 4 "$INSTALL4"
install_one 8 "$SNAP8" "$INSTALL8"
benchmark_one 8 "$INSTALL8"
say "PIPELINE COMPLETE"
df -h "$ROOT" | tail -1
