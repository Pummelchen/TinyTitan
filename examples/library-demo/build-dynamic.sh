#!/usr/bin/env bash
#
# Build the demo as a DYNAMIC app: it links `libTinyTitanLib.dylib` at run time,
# so the library stays a separate file that several apps can share and update
# independently.
#
#   ./build-dynamic.sh [path/to/library-dist]
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
  # the flags helper: this script copies the dylib into ./lib beside the app, so
  # the demo's own directory must never be mistaken for the source of it.
  if [[ -f "$HERE/libTinyTitanLib.dylib" && -f "$HERE/make-flags.sh" ]]; then
    DIST="$HERE"
  elif [[ -f "$HERE/../libTinyTitanLib.dylib" && -f "$HERE/../make-flags.sh" ]]; then
    DIST="$(cd "$HERE/.." && pwd)"
  else
    DIST="$HERE/../../.build/library-dist"
  fi
fi

if [[ ! -f "$DIST/libTinyTitanLib.dylib" ]]; then
  {
    echo "build-dynamic: no libTinyTitanLib.dylib in $DIST"
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

OUT="$HERE/tinytitan-demo-dynamic"
# The dylib lands in ./lib beside the app, and the rpath of `@executable_path/lib`
# is what makes it self-contained: it is found relative to the executable rather
# than at the absolute path it happened to be built against. Keeping it out of
# the demo's own directory also means the demo never looks like a distribution.
mkdir -p "$HERE/lib"
swiftc -O \
  -I "$DIST" \
  @"$DIST/swiftc-flags.txt" \
  -D DYNAMIC_LINK \
  -L "$DIST" -lTinyTitanLib \
  -Xlinker -rpath -Xlinker @executable_path/lib \
  "$HERE/main.swift" \
  -o "$OUT"

cp "$DIST/libTinyTitanLib.dylib" "$HERE/lib/"

# The runtime finds its Metal shaders through Bundle.module, which resolves
# beside the running executable — so the bundles travel with the demo.
for bundle in "$DIST"/*.bundle; do
  [[ -e "$bundle" ]] || continue
  rm -rf "${HERE:?}/$(basename "$bundle")"
  cp -R "$bundle" "$HERE/"
done

echo "built $OUT"
echo "run:   \"$OUT\" --model models/qwen3.5_9B_4Bit"
