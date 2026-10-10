#!/usr/bin/env bash
#
# The model-process guard: the one place a shell script asks whether a model
# process is already running.
#
#   . tools/model-guard.sh
#   model_guard_matches <pattern>
#
# Answers with three statuses, and a caller must be able to tell them apart:
#
#   0  a process matches; the lines it found are on stdout
#   1  nothing matches -- the run may start
#   2  it could not ask; why is on stderr
#
# AGENTS.md makes "no process from `pgrep -fl 'TinyTitanServer|TinyTitanCLI|...'`"
# a precondition of every model run, and until now each of the three shell callers
# asked pgrep itself, in a shape that can only act on one answer
# (AUD-268):
#
#   if busy=$(pgrep -fl '...' 2>/dev/null); then          # golden-baseline.sh:130
#   if pgrep -fl "$GUARD" >/dev/null 2>&1; then           # verify_cpu_models.sh:33
#   if [[ "$(pgrep -fl '...' | wc -l | tr -d ' ')" != "0" ]]; then   # installer:184
#
# `pgrep` answers 0 matched, 1 matched nothing, 2 an error, and a missing `pgrep`
# adds the shell's 127. Measured on this host with a stub pgrep for each status
# (/tmp/aud268b/shapes.sh): all three shapes run the model anyway on status 2 --
# `GOLDEN: reached the run (busy=[])`, `INSTALLER: proceeded without warning`,
# `CPU: reached the build`, exit 0. The status 2 is reachable without a race:
# `/usr/bin/pgrep -fl '('` answers it here with `pgrep: Cannot compile regular
# expression `(' (parentheses not balanced)`. So an error read as "no model
# process running", and the run started beside a live server -- exactly what the
# guard exists to stop, and enough to make every number it prints noise.
#
# "Nothing is running" and "I could not tell you" are different answers, and only
# the second one may be treated as a refusal. A caller that cannot ask must not
# start a model.
#
# `tools/install_tinytitan.sh` carries this function as an inline copy because it
# is run through `bash -c "$(curl ...)"`, where no `tools/` directory exists
# beside it. `benchmark/test_model_process_guard.py` pins that the two copies are
# the same text.
set -uo pipefail

mg_die() { echo "error: $*" >&2; exit 1; }

model_guard_matches() {  # <pattern>
  local pattern out status=0
  if [ $# -ne 1 ]; then
    echo "model-guard: usage: model_guard_matches <pattern>" >&2
    return 2
  fi
  pattern="$1"
  # Settled once, so the reads below never have to guess whether an empty answer
  # came from pgrep or from the shell failing to find it.
  if ! command -v pgrep >/dev/null 2>&1; then
    echo "model-guard: pgrep is not on PATH, so it is not known whether a process matching '$pattern' is running" >&2
    return 2
  fi
  # pgrep's own error text goes into the same variable as its matches, so the
  # refusal can quote the reason instead of reporting an empty list.
  out="$(pgrep -fl "$pattern" 2>&1)" || status=$?
  case "$status" in
    0)
      printf '%s\n' "$out"
      return 0
      ;;
    1)
      return 1
      ;;
    *)
      echo "model-guard: pgrep exited $status and did not answer for '$pattern': $out" >&2
      return 2
      ;;
  esac
}

# A library: sourced, never run. Direct invocation says so rather than doing
# nothing and exiting 0, which is the shape this file exists to remove.
case "$0" in
  */model-guard.sh|model-guard.sh)
    mg_die "model-guard.sh is a library — source it and call model_guard_matches"
    ;;
esac
