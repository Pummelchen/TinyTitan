#!/usr/bin/env bash
# The arm64 assertion behind RELEASE.md rule 2: a released artifact must report
# exactly `arm64` from `lipo -archs`, and a build that silently produced a fat
# or x86_64 binary is a release defect, not a build option.
#
#   tools/assert-arch.sh <file-or-dir>...      # check something on disk
#   . tools/assert-arch.sh                     # or source it and call the functions
#       assert_arm64_dir <dir> <label>         # every Mach-O under <dir> is arm64
#       assert_arm64_archive <tar.gz> <label>  # unpack it, then the same
#
# Sourced by `tools/release.sh` (both of its archives, before either checksum)
# and by `tools/build_library.sh` (the dist directory it writes). It is a separate
# file because those two are the only places that build something we hand out, and
# `build_library.sh` also runs standalone from a terminal — an assertion that lived
# only in the release script would be absent exactly when an operator packages the
# library by hand.
#
# Candidates come from the file's magic number, not from a list of expected names,
# so an artifact nobody thought to assert on is still checked. Swift bitcode is
# deliberately absent from the set: a `.swiftmodule` measures `e29ca80e`/`e29ca807`
# here and `lipo` cannot even mmap it (measured: "cannot mmap(), errno=22"), so
# keying on names would have made the module this gate's first false positive.
# Measured on the 5.18 stage tree: 162 files, 6 candidates — 4 executables, the
# dylib and `libTinyTitanLib.a` (`!<arch>`, which `lipo -archs` does read).
#
# That exemption is for a file carrying no Mach-O header, and for nothing else:
# a file this gate cannot open, and a file whose bytes are the first bytes of a
# Mach-O header and stop there, are refused. AUD-265 is the story of both being
# skipped, and of the 64-bit fat magic — the universal binary this file exists to
# catch — not being in the set at all while the line above it said "fat 32/64".
set -uo pipefail

tt_die() { echo "error: $*" >&2; exit 1; }

# thin 32/64 in either byte order, fat 32/64, and a static archive.
MACHO_MAGICS=" cffaedfe cefaedfe feedfacf feedface cafebabe bebafeca feedfacb cbfaedfe cafeabaf afabfeca 213c6172 "

# The magic read is the one step that must not be silent. Piping `od` into `tr`
# makes the assignment report `tr`'s status, so a file the gate may not open
# answers with an empty magic and takes the same branch as a README: it passed
# over an artifact with no read permission printing "each exactly arm64", and a
# 3-byte header (what a `cp` that filled the disk leaves) went the same way.
# "This is not an artifact" and "I cannot read this artifact" are different
# verdicts, and only one of them may be a skip.
macho_candidate() {  # <file> <label> -> 0 when this is a Mach-O, 1 when it is not
  local raw magic m
  raw="$(od -An -tx1 -N4 "$1" 2>/dev/null)" \
    || tt_die "$2: cannot read $1 — a file this gate cannot open is not a file it has checked"
  magic="${raw// /}"
  [ -n "$magic" ] || return 1
  case "$MACHO_MAGICS" in *" $magic "*) return 0 ;; esac
  for m in $MACHO_MAGICS; do
    case "$m" in
      "$magic"*)
        tt_die "$2: $(basename "$1") holds a $(( ${#magic} / 2 ))-byte Mach-O header of $m — a truncated artifact, not a non-artifact"
        ;;
    esac
  done
  return 1
}

assert_arm64_dir() {  # <dir> <label>
  local dir="$1" label="$2" checked=0 f arches
  [ -d "$dir" ] || tt_die "$label: no directory $dir to check"
  while IFS= read -r f; do
    macho_candidate "$f" "$label" || continue
    checked=$((checked + 1))
    arches="$(lipo -archs "$f" 2>&1)" \
      || tt_die "$label: lipo cannot read $f ($arches)"
    [ "$arches" = "arm64" ] \
      || tt_die "$label: $(basename "$f") reports arches '$arches', expected exactly 'arm64'"
  done < <(find "$dir" -type f | sort)
  # A directory with nothing in it would satisfy "every artifact is arm64" by
  # vacuity, which is the failure this gate exists to catch.
  [ "$checked" -gt 0 ] \
    || tt_die "$label: no Mach-O found under $dir — nothing executable was staged"
  echo "  $label: $checked Mach-O artifact(s), each exactly arm64"
}

assert_arm64_archive() {  # <tar.gz> <label>
  local archive="$1" label="$2" unpack
  [ -f "$archive" ] || tt_die "$label: no archive $archive"
  unpack="$(mktemp -d "${TMPDIR:-/tmp}/ttarch-XXXXXX")" || tt_die "$label: mktemp failed"
  # Unpack and check the archive rather than the staging directory it came from:
  # what a user downloads is the archive, and neither caller runs under `set -e`
  # all the way through, so a `cp` that silently failed is invisible in the
  # directory it was supposed to fill.
  tar xzf "$archive" -C "$unpack" \
    || { rm -rf "$unpack"; tt_die "$label: cannot unpack $archive to check its arch"; }
  assert_arm64_dir "$unpack" "$label"
  rm -rf "$unpack"
}

# Direct invocation checks what it is given: a directory is scanned, an archive
# is unpacked first, anything else is read as one Mach-O. Each check dies on the
# first wrong artifact, so a run says one thing: this set is arm64, or it is not.
main() {
  local target
  [ $# -gt 0 ] || tt_die "usage: tools/assert-arch.sh <file-or-dir>..."
  for target in "$@"; do
    if [ -d "$target" ]; then
      assert_arm64_dir "$target" "$(basename "$target")"
    elif [ -f "$target" ] && tar tzf "$target" >/dev/null 2>&1; then
      assert_arm64_archive "$target" "$(basename "$target")"
    else
      local arches
      # Named on the command line, so "not a Mach-O" is a refusal rather than the
      # skip the directory scan gives a README.
      macho_candidate "$target" "$(basename "$target")" \
        || tt_die "$(basename "$target"): not a Mach-O, an archive or a directory"
      arches="$(lipo -archs "$target" 2>&1)" \
        || tt_die "$(basename "$target"): lipo cannot read it ($arches)"
      [ "$arches" = "arm64" ] \
        || tt_die "$(basename "$target") reports arches '$arches', expected exactly 'arm64'"
      echo "  $(basename "$target"): arm64"
    fi
  done
}

# Only run when executed; stay quiet when sourced.
case "$0" in
  */assert-arch.sh|assert-arch.sh) main ${@+"$@"} ;;
esac
