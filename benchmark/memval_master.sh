#!/usr/bin/env bash
# Run the ten master prompts through the memory arms, one scenario at a time.
#
#   benchmark/memval_master.sh                    # all ten, photograph first
#   benchmark/memval_master.sh ledger filing      # named scenarios only
#
# Each scenario is a separate `memval_run.sh master` invocation, so its results
# land under memory-<scenario>-<install>/ and a failure in one does not take the
# rest with it. `TINYTITAN_MEMVAL_MODEL`/`QUANT`/`RUNS` are passed through.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ $# -gt 0 ]]; then
  scenarios=("$@")
else
  scenarios=(photograph pong ledger pigeon contract compound_k vantage kitchen cohort filing)
fi

# Each scenario's status is a verdict, not just a log line: a run in which every
# scenario refused must not exit 0 because the report happened to print.
failed=""
failed_count=0
scenarios_run=0

for scenario in "${scenarios[@]+"${scenarios[@]}"}"; do
  echo "##### master $scenario start $(date)"
  TINYTITAN_MASTER_SCENARIO="$scenario" "$ROOT/benchmark/memval_run.sh" master
  status="$?"
  echo "##### master $scenario exit=$status $(date)"
  scenarios_run=$((scenarios_run + 1))
  if [ "$status" -ne 0 ]; then
    failed="$failed $scenario(exit $status)"
    failed_count=$((failed_count + 1))
  fi
done

echo
echo "=== all master scenarios"
report_status=0
TINYTITAN_MASTER_SCENARIO=photograph python3 "$ROOT/benchmark/memory_master.py" report-all || report_status="$?"

# The report runs whether or not the scenarios did: it is the record, and the
# summary below is the verdict. Exit with the verdict, because a run whose every
# scenario refused used to end on report-all's own status, which is 0.
if [ -n "$failed" ]; then
  echo "##### MASTER FAILED $(date): $failed_count of $scenarios_run scenarios failed:$failed"
  exit 1
fi
# A scenario can exit 0 and still leave no record -- the arm ran, wrote nothing,
# and the driver above it reported success. `report-all` is the only place that
# sees the whole tree, and since AUD-240 it returns 1 when it has nothing to
# pool; discarding that status is what let this script print `all 10 scenarios
# exited 0` over a NOT MEASURED report.
if [ "$report_status" -ne 0 ]; then
  echo "##### MASTER FAILED $(date): all $scenarios_run scenarios exited 0 but report-all exited $report_status, which means it found no result record to pool -- the run measured nothing"
  exit 1
fi
echo "##### MASTER DONE $(date): all $scenarios_run scenarios exited 0"
