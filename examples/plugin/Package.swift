// swift-tools-version: 6.4
import PackageDescription

// An out-of-tree engine plugin consumer package (TT-133, TPP/1).
//
// Proves that a separate SwiftPM package outside TinyTitan can depend on
// `TinyTitanLib`, implement `EnginePlugin` in its own library target, register
// it with `PluginRegistry`, and resolve a matching install without modifying
// any TinyTitan source file. `tools/plugin-dependency-check.sh` builds and runs
// this package in CI.
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
    name: "PluginDemo",
    platforms: [
        .macOS(.v26)
    ],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .target(
            name: "EchoPlugin",
            dependencies: [
                .product(name: "TinyTitanLib", package: "TinyTitan")
            ],
            swiftSettings: demoLanguageStandard
        ),
        .executableTarget(
            name: "PluginDemo",
            dependencies: [
                "EchoPlugin",
                .product(name: "TinyTitanLib", package: "TinyTitan"),
            ],
            swiftSettings: demoLanguageStandard
        ),
    ],
    swiftLanguageModes: [.v6]
)
