#!/usr/bin/env bash
# Pick a Python interpreter for the tools that need the analysis stack
# (numpy, ml_dtypes, safetensors).
#
# Sourced, never executed: `source tools/lib/python.sh` then use
# `$TINYTITAN_PYTHON` (call `tinytitan_resolve_python` first).
#
# Why detection rather than a `python3` name or a pinned `python3.13`:
#
#   * A bare `python3` is not a version. On a stock macOS it is 3.9 from
#     /usr/bin, older than the 3.10 syntax these tools use and without any of
#     the dependencies -- so substituting the name breaks every conversion with
#     a confusing SyntaxError or ModuleNotFoundError.
#   * A pinned `python3.13` fails on a machine whose analysis stack is installed
#     under 3.12 or 3.14, which is the complaint this resolver exists to fix.
#
# So the candidates are tried newest-first and each one is *tested*: it must be
# at least the required version and must import the dependencies. The first that
# passes wins. `TINYTITAN_PYTHON` skips that search, for a virtualenv or a build
# that pins its own interpreter, but not the test: an untested value is not a
# resolved interpreter, and handing one back is how a typo in the variable reaches
# a converter call mid-install instead of stopping here.

# Minimum interpreter version. 3.10 is what the syntax actually needs (`X | Y`
# in annotations under `from __future__ import annotations` is fine, but `match`
# statements and the newer typing forms are not universally avoidable).
TINYTITAN_PYTHON_MIN_MAJOR=3
TINYTITAN_PYTHON_MIN_MINOR=10

# Dependency check the tools share. Kept as one string so the shell resolver and
# the Python files cannot drift.
TINYTITAN_PYTHON_DEPS="import numpy, ml_dtypes, safetensors"

tinytitan_python_note() {
  # One line, usable in any message, naming this shell's resolved interpreter.
  # Call it after `tinytitan_resolve_python`.
  echo "this checkout uses ${TINYTITAN_PYTHON:-python3}; install them for it with \"${TINYTITAN_PYTHON:-python3} -m pip install safetensors numpy ml_dtypes\", or point TINYTITAN_PYTHON at another Python 3"
}

tinytitan_python_usable() {
  # The one test every interpreter passes before it is handed to a caller: the
  # version floor and the shared dependency check. Kept in one place because the
  # search's candidates and the operator's own `TINYTITAN_PYTHON` have to be held
  # to the same bar -- a value nobody tested is not a resolved interpreter.
  # On failure, $TINYTITAN_PYTHON_WHY says what the candidate itself said.
  local candidate="$1" probe_out
  TINYTITAN_PYTHON_WHY=""
  if [[ -z "$candidate" ]]; then
    TINYTITAN_PYTHON_WHY="nothing was given"
    return 1
  fi
  if [[ ! -e "$candidate" ]]; then
    TINYTITAN_PYTHON_WHY="no such file"
    return 1
  fi
  if [[ ! -x "$candidate" ]]; then
    TINYTITAN_PYTHON_WHY="not executable"
    return 1
  fi
  probe_out="$("$candidate" -c "import sys
assert sys.version_info >= ($TINYTITAN_PYTHON_MIN_MAJOR, $TINYTITAN_PYTHON_MIN_MINOR), sys.version.split()[0]
$TINYTITAN_PYTHON_DEPS" 2>&1)" || {
    TINYTITAN_PYTHON_WHY="$(printf '%s\n' "$probe_out" | tail -n 1)"
    [[ -n "$TINYTITAN_PYTHON_WHY" ]] || TINYTITAN_PYTHON_WHY="it answered non-zero without saying why"
    return 1
  }
  return 0
}

tinytitan_resolve_python() {
  # An override is still an interpreter, so it is tested like a candidate.
  if [[ -n "${TINYTITAN_PYTHON:-}" ]]; then
    if tinytitan_python_usable "$TINYTITAN_PYTHON"; then
      printf '%s' "$TINYTITAN_PYTHON"
      return 0
    fi
    {
      echo "error: TINYTITAN_PYTHON does not name a usable interpreter: $TINYTITAN_PYTHON" >&2
      echo "  need: Python >= $TINYTITAN_PYTHON_MIN_MAJOR.$TINYTITAN_PYTHON_MIN_MINOR with $TINYTITAN_PYTHON_DEPS" >&2
      echo "  it answered: $TINYTITAN_PYTHON_WHY" >&2
      echo "  fix:  unset TINYTITAN_PYTHON to search for one, or point it at a" >&2
      echo "        Python that has $TINYTITAN_PYTHON_DEPS" >&2
    } >&2
    return 1
  fi

  local candidate
  # Newest first. `python3` is tested last on purpose: it is the name most
  # likely to be an old system interpreter.
  for candidate in \
    "$(command -v python3.14 2>/dev/null)" \
    "$(command -v python3.13 2>/dev/null)" \
    "$(command -v python3.12 2>/dev/null)" \
    "$(command -v python3.11 2>/dev/null)" \
    "$(command -v python3.10 2>/dev/null)" \
    "$(command -v python3 2>/dev/null)" \
    "$(command -v python 2>/dev/null)"
  do
    tinytitan_python_usable "$candidate" || continue
    printf '%s' "$candidate"
    return 0
  done

  {
    echo "no usable Python interpreter found for the TinyTitan tools." >&2
    echo "  need: Python >= $TINYTITAN_PYTHON_MIN_MAJOR.$TINYTITAN_PYTHON_MIN_MINOR with $TINYTITAN_PYTHON_DEPS" >&2
    echo "  tried (newest first): python3.14 python3.13 python3.12 python3.11 python3.10 python3 python" >&2
    echo "  fix:  install the packages for a Python 3, e.g." >&2
    echo "          python3 -m pip install safetensors numpy ml_dtypes" >&2
    echo "        or set TINYTITAN_PYTHON=/path/to/python3 to choose one explicitly" >&2
  } >&2
  return 1
}
