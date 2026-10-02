#!/usr/bin/env bash
#
# The dependency check for docs/plan-embedded-library.md: can a package that is
# NOT this repository depend on TinyTitan and link it?
#
#   tools/embedded-dependency-check.sh                 the committed fixture, path dependency
#   tools/embedded-dependency-check.sh --tag 5.15.0    plus a throwaway consumer on a released tag
#
# Why a build and not a reading: SwiftPM diagnoses unsafe build flags in some
# configurations ("... contains unsafe build flags"), and this package uses them
# deliberately (see AGENTS.md). Whether a consumer still resolves is therefore a
# question about the toolchain, and it is answered by building. Measured
# 2026-10-02 on Swift 6.4 / Xcode 27: path, file:// tag and GitHub URL all
# resolve, debug and release.
#
# --tag takes a full semantic version (the repository tags are `v5.15`, which
# SwiftPM reads as 5.15.0) and uses file:// so the check needs no network. The
# tag must carry the library product: `TinyTitanKit` is unreleased, so `5.15.0`
# and earlier fail this arm by design — the consumer asks for a product that tag
# does not declare. It becomes meaningful from the first release that ships the
# library, which is the point of keeping it here.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE="$ROOT/examples/embedded"
TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="${2:?--tag needs a full version, e.g. 5.15.0}"
      shift 2
      ;;
    -h | --help)
      sed -n '3,17p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

step() { printf '\n== %s ==\n' "$1"; }

step "fixture package (path dependency on the working tree)"
(cd "$FIXTURE" && swift build)
(cd "$FIXTURE" && swift run EmbeddedDemo)

if [[ -n "$TAG" ]]; then
  scratch="$(mktemp -d)"
  trap 'rm -rf "$scratch"' EXIT
  mkdir -p "$scratch/Sources/consumer"
  cat > "$scratch/Package.swift" <<EOF
// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "consumer",
    platforms: [.macOS(.v26)],
    dependencies: [
        .package(url: "file://$ROOT", from: "$TAG")
    ],
    targets: [
        .executableTarget(
            name: "consumer",
            dependencies: [.product(name: "TinyTitanKit", package: "TinyTitan")]
        )
    ]
)
EOF
  cat > "$scratch/Sources/consumer/main.swift" <<'EOF'
import TinyTitanKit

print("consumer: linked \(String(describing: Engine.self))")
EOF
  step "throwaway consumer on tag $TAG (file:// dependency)"
  (cd "$scratch" && swift build && swift run consumer)
fi

step "ok: the package is consumable as a dependency"
