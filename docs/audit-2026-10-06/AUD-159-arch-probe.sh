#!/usr/bin/env bash
# AUD-159 probe — the arm64 assertion in the release path.
#
#   docs/audit-2026-10-06/AUD-159-arch-probe.sh [staged-release-tree]
#
# Needs a staged release tree to copy real products from. Default
# `.build/releases/tinytitan-release-5.18`; pass another if it moves. It refuses
# to run when that tree is absent: a probe that silently skips its arms is the
# defect this audit hunts, not a way to measure it. Exits non-zero if any arm
# misbehaves.
#
# What it proves, in order:
#   1  the gate passes on the real 5.18 archives (and does not trip over the 29
#      .swiftmodule bitcode files inside the library one),
#   2  it dies with a different message for each wrong shape — fat, x86_64-only,
#      arm64e, Mach-O magic over an unreadable body, and a stage with no Mach-O
#      at all (which would otherwise pass by vacuity),
#   3  the CLI mode behaves the same on a file, a directory and an archive,
#   4  the *call site* is reached: release.sh's own stage -> package -> checksum
#      lines run unmodified and stop before the sha256 when a product is the
#      wrong arch,
#   5  the sibling guard: a product that never reaches the archive is named, and
#      without that loop the same contents produce a checksummed three-binary
#      release.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SRC="${1:-$REPO/.build/releases/tinytitan-release-5.18}"
REL="$REPO/tools/release.sh"
GATE="$REPO/tools/assert-arch.sh"
STAGE_SRC="$SRC/tinytitan-5.18-macos-arm64"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/audit159-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

[ -d "$STAGE_SRC" ] || { echo "error: no staged tree at $STAGE_SRC — pass one as \$1"; exit 1; }
[ -f "$GATE" ] || { echo "error: $GATE missing"; exit 1; }

fails=0
ok() { echo "  ok   - $*"; }
bad() { echo "  FAIL - $*"; fails=$((fails + 1)); }

# ---- 1/2: the gate's own behaviour ------------------------------------------
OUT=""; RC=0
gate() { # <label> <archive>
  # $GATE is derived from $HERE at runtime, so the path is not a literal here;
  # the file exists by the check above before this line can run.
  # shellcheck source=/dev/null
  OUT="$( . "$GATE"; assert_arm64_archive "$2" "$1" 2>&1 )"
  RC=$?
}
expect_gate() { # <label> <path> <substring> <want-pass 0|1>
  local label="$1" want="$3" wantpass="$4"
  gate "$label" "$2"
  printf '%s' "$OUT" | grep -qF -- "$want" || { bad "$label: no '$want' in: $(printf '%s\n' "$OUT" | tail -1)"; return; }
  if [ "$wantpass" = 0 ] && [ "$RC" != 0 ]; then bad "$label: wanted pass, rc=$RC"
  elif [ "$wantpass" = 1 ] && [ "$RC" = 0 ]; then bad "$label: wanted a die, it passed"
  else ok "$label (rc=$RC) $(printf '%s\n' "$OUT" | tail -1 | sed 's/^ *//')"; fi
}

stage="$WORK/stage"
fresh_stage() { rm -rf "$stage"; mkdir -p "$stage"; cp -R "$STAGE_SRC/." "$stage/"; }
tars() { ( cd "$WORK" && tar czf "$1" stage ); }

echo "== gate: the real artifacts =="
expect_gate "engine-real-5.18" "$SRC/tinytitan-5.18-macos-arm64.tar.gz" "each exactly arm64" 0
expect_gate "library-real-5.18" "$SRC/tinytitan-lib-5.18-macos-arm64.tar.gz" "each exactly arm64" 0

echo "== gate: wrong shapes, one message each =="
fresh_stage; cp /usr/bin/lipo "$stage/TinyTitanCLI"; tars fat.tgz
expect_gate "planted-fat" "$WORK/fat.tgz" "reports arches 'x86_64 arm64e arm64e.x1'" 1
if lipo /usr/bin/lipo -thin x86_64 -output "$stage/TinyTitanCLI" 2>/dev/null; then
  tars x86.tgz
  expect_gate "planted-x86_64-only" "$WORK/x86.tgz" "reports arches 'x86_64'" 1
else
  echo "  skip - no x86_64 slice to thin on this host"
fi
fresh_stage
printf '\xcf\xfa\xed\xfeAAAAAAAAAAAA' > "$stage/TinyTitanRepack"; tars broken.tgz
expect_gate "magic-but-unreadable" "$WORK/broken.tgz" "lipo cannot read" 1
mkdir -p "$WORK/empty/stage" && echo text > "$WORK/empty/stage/README.txt"
( cd "$WORK/empty" && tar czf empty.tgz stage )
gate "no-macho" "$WORK/empty/empty.tgz"
if [ "$RC" != 0 ] && printf '%s' "$OUT" | grep -qF "nothing executable"; then
  ok "no-macho-vacuous (rc=$RC) an archive with no Mach-O is refused, not passed"
else
  bad "no-macho-vacuous: rc=$RC out=$(printf '%s\n' "$OUT" | tail -1)"
fi
fresh_stage
if lipo /usr/bin/lipo -thin arm64e -output "$stage/TinyTitanServer" 2>/dev/null; then
  tars arm64e.tgz
  expect_gate "planted-arm64e" "$WORK/arm64e.tgz" "TinyTitanServer reports arches 'arm64e'" 1
else
  echo "  skip - no arm64e slice to thin on this host"
fi

echo "== gate: CLI mode =="
cli() { # <path> <substring> <want-pass 0|1>
  local out rc
  out="$("$GATE" "$1" 2>&1)"; rc=$?
  printf '%s' "$out" | grep -qF -- "$2" || { bad "cli $1: no '$2' in: $(printf '%s\n' "$out" | tail -1)"; return; }
  if [ "$3" = 0 ] && [ "$rc" != 0 ]; then bad "cli $1: wanted pass, rc=$rc"
  elif [ "$3" = 1 ] && [ "$rc" = 0 ]; then bad "cli $1: wanted a die, it passed"
  else ok "cli $(basename "$1") (rc=$rc) $(printf '%s\n' "$out" | tail -1)"; fi
}
cli "$STAGE_SRC/TinyTitanCLI" "TinyTitanCLI: arm64" 0
cli /usr/bin/lipo "reports arches 'x86_64 arm64e arm64e.x1'" 1
cli "$STAGE_SRC" "each exactly arm64" 0
cli "$SRC/tinytitan-lib-5.18-macos-arm64.tar.gz" "each exactly arm64" 0
cli "$REL" "not a Mach-O, an archive or a directory" 1

# ---- 4/5: the call sites ------------------------------------------------------
# release.sh's own stage -> package -> checksum lines, verbatim, with the
# variables the earlier part of the script would have set. `die`/`step` are
# re-declared because assert-arch.sh ships only its own tt_die.
awk '/^# --- stage ---/{f=1} /^# --- the library ---/{f=0} f' "$REL" > "$WORK/tail.sh"
sed '/^# Every product the build produced/,/^done$/d' "$WORK/tail.sh" > "$WORK/tail-nomembers.sh"
grep -q 'carries no \$p' "$WORK/tail.sh" \
  && ok "extracted the shipped stage->package section ($(wc -l < "$WORK/tail.sh" | tr -d ' ') lines)" \
  || bad "the extracted tail does not contain the membership loop"
grep -q 'carries no \$p' "$WORK/tail-nomembers.sh" \
  && bad "the pre-fix variant still has the loop" \
  || ok "pre-fix variant (loop removed) built"

mk_run() { # <outfile> <tail-file>
  cat > "$1" <<PRE
#!/usr/bin/env bash
ROOT="$REPO"
VERSION="9.99-probe"; TAG="v9.99-probe"
PRODUCTS=(TinyTitanServer TinyTitanCLI TinyTitanRepack TinyTitanBench)
STAGE_ROOT="$WORK/scaffold"
STAGE="\$STAGE_ROOT/tinytitan-\$VERSION-macos-arm64"
ARCHIVE="\$STAGE_ROOT/tinytitan-\$VERSION-macos-arm64.tar.gz"
BIN="$WORK/bin"
mkdir -p "\$STAGE_ROOT"
die() { echo "error: \$*" >&2; exit 1; }
step() { printf '\\n== %s\\n' "\$*"; }
. "$GATE"
. "$WORK/$2"
PRE
}
mk_run "$WORK/run.sh" tail.sh
mk_run "$WORK/run-nomembers.sh" tail-nomembers.sh

TSHA=no
run_tail() { # <script>  ; sets OUT / RC / TSHA
  rm -rf "$WORK/scaffold"
  OUT="$(bash "$1" 2>&1)"; RC=$?
  if [ -f "$WORK/scaffold/tinytitan-9.99-probe-macos-arm64.tar.gz.sha256" ]; then TSHA=yes; else TSHA=no; fi
}
mkdir -p "$WORK/bin"
for p in TinyTitanServer TinyTitanCLI TinyTitanRepack TinyTitanBench; do
  cp "$STAGE_SRC/$p" "$WORK/bin/"
done
cp -R "$STAGE_SRC/TinyTitan_TinyTitan.bundle" "$WORK/bin/"

echo "== call site: release.sh, engine archive =="
run_tail "$WORK/run.sh"
if [ "$RC" = 0 ] && printf '%s' "$OUT" | grep -qF "each exactly arm64" && [ "$TSHA" = yes ]; then
  ok "real products: asserted, then checksummed"
else
  bad "real products: rc=$RC sha=$TSHA out=$(printf '%s\n' "$OUT" | tail -2)"
fi
cp /usr/bin/lipo "$WORK/bin/TinyTitanCLI"
run_tail "$WORK/run.sh"
if [ "$RC" != 0 ] && printf '%s' "$OUT" | grep -qF "TinyTitanCLI reports arches" && [ "$TSHA" = no ]; then
  ok "fat product: died before any sha256 existed"
else
  bad "fat product: rc=$RC sha=$TSHA out=$(printf '%s\n' "$OUT" | tail -2)"
fi
cp "$STAGE_SRC/TinyTitanCLI" "$WORK/bin/"
run_tail "$WORK/run.sh"
if [ "$RC" = 0 ]; then ok "control: real CLI restored, passes again"; else bad "control: rc=$RC"; fi

echo "== sibling: a product that never reaches the archive =="
rm -rf "$WORK/bin/TinyTitanBench"; mkdir -p "$WORK/bin/TinyTitanBench"
run_tail "$WORK/run-nomembers.sh"
if [ "$RC" = 0 ] && [ "$TSHA" = yes ]; then
  ok "pre-fix path: checksummed a 3-binary archive (the hole this closes)"
else
  bad "pre-fix path was expected to pass: rc=$RC sha=$TSHA out=$(printf '%s\n' "$OUT" | tail -1)"
fi
run_tail "$WORK/run.sh"
if [ "$RC" != 0 ] && printf '%s' "$OUT" | grep -qF "carries no TinyTitanBench" && [ "$TSHA" = no ]; then
  ok "fixed path: names TinyTitanBench and refuses the checksum"
else
  bad "fixed path: rc=$RC sha=$TSHA out=$(printf '%s\n' "$OUT" | tail -2)"
fi
rm -rf "$WORK/bin/TinyTitanBench"; cp "$STAGE_SRC/TinyTitanBench" "$WORK/bin/"

echo
if [ "$fails" = 0 ]; then
  echo "AUD-159 probe: every arm behaved."
else
  echo "AUD-159 probe: $fails arm(s) failed."
  exit 1
fi
