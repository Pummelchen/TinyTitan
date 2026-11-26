// swift-tools-version: 6.4
import PackageDescription

// A consumer package: this is what "use TinyTitan as an LLM engine" looks like
// from outside the repository. The dependency is a relative path so the fixture
// tracks the working tree it sits in; a real consumer writes the same line
// against a released tag instead:
//
//     .package(url: "https://github.com/Pummelchen/TinyTitan", from: "5.15.0")
//
// `TinyTitanLib` is the supported library product — the facade documented in
// docs/plan-embedded-library.md §4. The engine's own executables are the other
// product and are not needed here. `tools/embedded-dependency-check.sh` builds
// and runs this package so both the dependency and the facade stay proven.
//
// The standard the root package enforces (`tinytitanLanguageStandard` in the
// repository's own `Package.swift`) is restated here rather than imported: this
// is a separate package, and reaching into another manifest would tie it to a
// checkout layout. Swift 6 language mode is declared below, so strict
// concurrency is an error rather than a warning; these are the upcoming
// features the tree is already clean under, plus `-warnings-as-errors`, which a
// tools version alone does not establish. Every target carries this array, so a
// target added later cannot quietly opt out.
let demoLanguageStandard: [SwiftSetting] = [
    .enableUpcomingFeature("InferIsolatedConformances"),
    .enableUpcomingFeature("ImmutableWeakCaptures"),
    .enableUpcomingFeature("MemberImportVisibility"),
    .unsafeFlags(["-warnings-as-errors"]),
]

let package = Package(
    name: "EmbeddedDemo",
    platforms: [
        .macOS(.v26)
    ],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "EmbeddedDemo",
            dependencies: [
                .product(name: "TinyTitanLib", package: "TinyTitan")
            ],
            swiftSettings: demoLanguageStandard
        )
    ],
    swiftLanguageModes: [.v6]
)
