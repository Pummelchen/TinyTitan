#!/usr/bin/env bash
#
# Build the demo as a STATIC app: everything it needs is inside the single
# archive `libTinyTitanLib.a`, so the executable carries the library with it.
#
#   ./build-static.sh [path/to/library-dist]
#
# "library-dist" is either the directory you extracted the release archive into
# (`tinytitan-lib-<version>-macos-arm64.tar.gz`) or the one a local
# `tools/build_library.sh <version>` staged in `.build/library-dist`. With no
# argument the script looks beside itself, then one level up (which is where it
# sits inside the extracted archive), then at the repository's staging.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DIST="${1:-}"
if [[ -z "$DIST" ]]; then
  # A directory counts as a distribution only when it carries the library *and*
  # the flags helper, so a stray archive beside the demo cannot be mistaken for
  # one that has the module maps too.
  if [[ -f "$HERE/libTinyTitanLib.a" && -f "$HERE/make-flags.sh" ]]; then
    DIST="$HERE"
  elif [[ -f "$HERE/../libTinyTitanLib.a" && -f "$HERE/../make-flags.sh" ]]; then
    DIST="$(cd "$HERE/.." && pwd)"
  else
    DIST="$HERE/../../.build/library-dist"
  fi
fi

if [[ ! -f "$DIST/libTinyTitanLib.a" ]]; then
  {
    echo "build-static: no libTinyTitanLib.a in $DIST"
    echo "  pass the directory holding the library (the one you extracted the"
    echo "  release archive into), or run tools/build_library.sh <version> first"
  } >&2
  exit 2
fi

# The C module maps have to be named explicitly on the command line. The archive
# ships the helper that writes the list for wherever it was extracted.
if [[ ! -f "$DIST/swiftc-flags.txt" ]]; then
  ( cd "$DIST" && ./make-flags.sh )
fi

OUT="$HERE/tinytitan-demo-static"
swiftc -O \
  -I "$DIST" \
  @"$DIST/swiftc-flags.txt" \
  -D STATIC_LINK \
  "$HERE/main.swift" \
  "$DIST/libTinyTitanLib.a" \
  -o "$OUT"

# The runtime finds its Metal shaders through Bundle.module, which resolves
# beside the running executable — so the bundles travel with the demo.
for bundle in "$DIST"/*.bundle; do
  [[ -e "$bundle" ]] || continue
  rm -rf "${HERE:?}/$(basename "$bundle")"
  cp -R "$bundle" "$HERE/"
done

echo "built $OUT"
echo "run:   \"$OUT\" --model models/qwen3.5_9B_4Bit"
