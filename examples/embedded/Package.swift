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
            ]
        )
    ]
)
