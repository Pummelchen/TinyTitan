#!/usr/bin/env bash
#
# Merge the package's test-bundle coverage into one llvm-cov JSON report.
#
#   tools/coverage-export.sh [output.json]      run the merge and its checks
#   tools/coverage-export.sh --check-file f     check one report's shape only
#   tools/coverage-export.sh --check-scope f    check shape and bundle scope
#
# Why this exists: `swift test --enable-code-coverage` writes the llvm-cov JSON
# it produces into <bin-path>/codecov/, and the toolchain's own writer names one
# test bundle — measured on Swift 6.4 / Xcode 27, the file it leaves there is
# TinyTitan.json with 27 files, one bundle's scope, while the merged
# default.profdata behind it holds 11,984 functions. Uploading that JSON is a
# coverage report that reads as package-wide and is not (audit AUD2-031).
#
# Two properties this script is built to hold:
#
#  * The object list comes from the manifest, never from a glob of the build
#    directory. The directory here carries ten .xctest bundles for seven declared
#    test targets — three are leftovers from products that no longer exist — so a
#    glob would fold dead products into the number.
#  * The scope is checked against the truth it is meant to establish. After the
#    merge, each declared bundle is exported on its own and the report must
#    strictly contain every single-bundle set. "The export collapsed to one
#    bundle" then fails, and no fixed threshold could say so honestly: one bundle
#    here maps 323 source files, so 397 merged is only whole next to the
#    per-bundle numbers, not next to a constant.
#
# Coverage stays a signal, not a gate: nothing here asserts a percentage. The
# checks are about the report describing the whole package.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.build/package-code-coverage.json"
CHECK_FILE=""
CHECK_SCOPE=""

if [ ! -e "$(command -v python3)" ]; then
  echo "coverage-export: python3 is required to read the manifest and the report" >&2
  exit 1
fi

case "${1:-}" in
  --check-file)
    if [ $# -lt 2 ]; then
      echo "usage: tools/coverage-export.sh --check-file <report.json>" >&2
      exit 2
    fi
    CHECK_FILE="$2"
    ;;
  --check-scope)
    if [ $# -lt 2 ]; then
      echo "usage: tools/coverage-export.sh --check-scope <report.json>" >&2
      exit 2
    fi
    CHECK_SCOPE="$2"
    ;;
  "") ;;
  *) OUT="$1" ;;
esac

# Structural checks on one llvm-cov JSON: every mapped file must be production
# source, which is also what proves the --ignore-filename-regex did its job.
check_report_shape() {
  python3 - "$1" <<'PY'
import json, sys

path = sys.argv[1]
try:
    with open(path) as handle:
        report = json.load(handle)
except (OSError, ValueError) as error:
    sys.exit(f"FAIL {path} is not readable coverage JSON: {error}")
data = report.get("data") or []
files = [entry for chunk in data for entry in chunk.get("files", [])]
if not files:
    sys.exit(f"FAIL {path} maps no files; a report that read nothing is not a pass")
outside = sorted(
    {f["filename"] for f in files
     if "/sources/" not in f["filename"] and not f["filename"].startswith("sources/")})
lines = sum(f["summary"]["lines"]["count"] for f in files)
covered = sum(f["summary"]["lines"]["covered"] for f in files)
if outside:
    print("  mapped files NOT under sources/: " + ", ".join(outside[:5]), file=sys.stderr)
    sys.exit(f"FAIL {path} maps {len(outside)} file(s) outside sources/; "
             "tests, dependencies and build output must be excluded")
print(f"  scope: {len(files)} source files, {covered:,}/{lines:,} lines "
      f"({100.0 * covered / lines:.2f}% of instrumented lines)")
PY
}

# Locate the merged profile and the manifest's test bundles, and refuse a state
# where llvm-cov would silently drop one. Sets PROF and OBJECTS.
resolve_bundles() {
  BIN="$(xcrun swift build --show-bin-path)"
  PROF="$BIN/codecov/default.profdata"
  if [ ! -f "$PROF" ]; then
    echo "coverage-export: no merged profile at $PROF" >&2
    echo "  run the suite with coverage first:" >&2
    echo "  swift test --no-parallel --enable-code-coverage" >&2
    exit 1
  fi

  TARGETS="$(xcrun swift package dump-package | python3 -c '
import json, sys
package = json.load(sys.stdin)
print("\n".join(sorted(t["name"] for t in package["targets"] if t.get("type") == "test")))'
  )"
  if [ -z "$TARGETS" ]; then
    echo "coverage-export: the manifest declares no test target, so a dump-package" \
         "change would export nothing and still print a number" >&2
    exit 1
  fi

  OBJECTS=()
  while IFS= read -r target; do
    [ -n "$target" ] || continue
    bundle="$BIN/$target.xctest/Contents/MacOS/$target"
    if [ ! -f "$bundle" ]; then
      echo "coverage-export: declared test target $target has no built bundle at" \
           "$bundle — the coverage run did not produce it" >&2
      exit 1
    fi
    # llvm-cov refuses an object newer than the profile with "no coverage data
    # found", which would silently drop that bundle from the merge.
    if [ "$bundle" -nt "$PROF" ]; then
      echo "coverage-export: $target was rebuilt after the profile was written, so" \
           "its coverage would be dropped. Re-run the suite with coverage and" \
           "export before building anything else." >&2
      exit 1
    fi
    OBJECTS+=("$bundle")
  done <<EOF
$TARGETS
EOF
  echo "== ${#OBJECTS[@]} manifest test bundle(s), profile $PROF =="
}

# The collapse check: export each bundle alone and require the report under test
# to strictly contain every one of them. A single bundle's JSON passing the shape
# check above is exactly the artifact this script exists to replace.
check_report_scope() {
  python3 - "$1" "$PROF" ${OBJECTS[@]+"${OBJECTS[@]}"} <<'PY'
import json, subprocess, sys

merged_path, prof = sys.argv[1:3]
objects = sys.argv[3:]


def mapped(files):
    return {f["filename"] for f in files
            if "/sources/" in f["filename"] or f["filename"].startswith("sources/")}


def export(path):
    out = subprocess.run(
        ["xcrun", "llvm-cov", "export", "--summary-only", "--instr-profile", prof,
         path, "--ignore-filename-regex", r"(^|/)(tests|\.build)/"],
        capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"FAIL single-bundle export of {path} failed: {out.stderr.strip()}")
    report = json.loads(out.stdout)
    return mapped(f for chunk in report.get("data", []) for f in chunk.get("files", []))


with open(merged_path) as handle:
    merged = mapped(f for chunk in json.load(handle).get("data", [])
                    for f in chunk.get("files", []))

if len(objects) < 2:
    sys.exit(f"FAIL expected at least two test bundles to merge, got {len(objects)}")

largest = set()
for object_path in objects:
    single = export(object_path)
    name = object_path.split("/")[-1]
    missing = sorted(single - merged)
    if missing:
        sys.exit(f"FAIL {name} maps {len(missing)} source file(s) the report does not "
                 f"(first: {missing[0]}): the merge lost a bundle's scope")
    print(f"  {name}: {len(single)} source files")
    if len(single) > len(largest):
        largest = single

if len(merged) <= len(largest):
    sys.exit(f"FAIL report maps {len(merged)} source files, no more than the single "
             f"largest bundle ({len(largest)}): the export collapsed")
print(f"  report: {len(merged)} source files, {len(merged) - len(largest)} more than "
      f"the largest single bundle")
PY
}

if [ -n "$CHECK_FILE" ]; then
  echo "== coverage shape: $CHECK_FILE =="
  check_report_shape "$CHECK_FILE"
  exit 0
fi

resolve_bundles

if [ -n "$CHECK_SCOPE" ]; then
  echo "== coverage report under test: $CHECK_SCOPE =="
  check_report_shape "$CHECK_SCOPE"
  check_report_scope "$CHECK_SCOPE"
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
# The guarded whole-array form, because `${a[@]}` on an *empty* array is
# `unbound variable` under `set -u` on the system bash 3.2.57 and the
# shell-portability gate finds that shape by eye. A slice like `${a[@]:1}` is the
# same shape, so this takes one pass and skips the first entry by index.
FIRST=""
EXTRA=()
INDEX=0
for object in ${OBJECTS[@]+"${OBJECTS[@]}"}; do
  if [ "$INDEX" -eq 0 ]; then
    FIRST="$object"
  else
    EXTRA+=(--object="$object")
  fi
  INDEX=$((INDEX + 1))
done

xcrun llvm-cov export --summary-only \
  --instr-profile="$PROF" \
  "$FIRST" ${EXTRA[@]+"${EXTRA[@]}"} \
  --ignore-filename-regex='(^|/)(tests|\.build)/' \
  > "$OUT"

check_report_shape "$OUT"
check_report_scope "$OUT"
echo "coverage-export: wrote $OUT"
