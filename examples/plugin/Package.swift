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
        // `name:` is not decoration. SwiftPM derives a path dependency's
        // identity from the last component of the path, so in a checkout or
        // worktree whose directory is not called `TinyTitan` the dependency
        // would be named after that directory instead, and `package:
        // "TinyTitan"` below would fail to resolve (`unknown package
        // 'TinyTitan' ... valid packages are: '<directory name>'`). Naming it
        // here is what lets the fixture build from any directory.
        .package(name: "TinyTitan", path: "../..")
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
