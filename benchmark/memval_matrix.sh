#!/usr/bin/env bash
# Runs the memory benchmarks across every 35B install: three models, 4-bit
# and 8-bit, sequentially on one port. Qwen 3.6 4-bit gets all four arms,
# because it has a frozen v2 to compare against; every other install runs
# the baseline and the shipped configuration.
#
#   benchmark/memval_matrix.sh                  # everything, ~a day
#   TINYTITAN_MEMVAL_RUNS=1 benchmark/memval_matrix.sh   # a quick pass
#
# Never edit anything under benchmark/ or tools/ while this runs: bash reads
# scripts by offset, and a shifted file mid-run is how a report step once
# executed the letter "e".
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
LOG="$ROOT/.build/benchmark-logs/memval-matrix.log"
mkdir -p "$(dirname "$LOG")"
# AUD-212: an arm's exit code used to go only into its `DONE (exit N)` log line, so
# this script's own status was the status of the final `echo` and a matrix in which
# all twelve arms failed still reported finished. `failed` collects them and the
# end of the run names them, because the operator reads the last screenful of a
# day-long run rather than diffing the log.
echo "##### MATRIX START $(date)" | tee -a "$LOG"
arms_run=0
failed=""
failed_count=0
for install in qwen36:4 qwen36:8 ornith:4 ornith:8 agentworld:4 agentworld:8; do
  model="${install%%:*}"; quant="${install##*:}"
  if [[ "$install" == "qwen36:4" ]]; then arms_book="summary auto minimal full"; arms_pong="control auto minimal full"
  else arms_book="summary auto"; arms_pong="control auto"; fi
  for bench in book pong; do
    arms="$arms_book"; [[ "$bench" == pong ]] && arms="$arms_pong"
    echo "##### $install $bench arms=[$arms] $(date)" | tee -a "$LOG"
    TINYTITAN_MEMVAL_MODEL="$model" TINYTITAN_MEMVAL_QUANT="$quant" TINYTITAN_MEMVAL_ARMS="$arms" \
      benchmark/memval_run.sh "$bench" 2>&1 | tee -a "$LOG" | tail -3
    status="${PIPESTATUS[0]}"
    arms_run=$((arms_run + 1))
    echo "##### $install $bench DONE $(date) (exit $status)" | tee -a "$LOG"
    if [ "$status" -ne 0 ]; then
      failed="$failed $install $bench(exit $status)"
      failed_count=$((failed_count + 1))
    fi
  done
done
if [ -n "$failed" ]; then
  echo "##### MATRIX FAILED $(date): $failed_count of $arms_run arms failed:$failed" | tee -a "$LOG"
  exit 1
fi
echo "##### MATRIX DONE $(date): all $arms_run arms exited 0" | tee -a "$LOG"
