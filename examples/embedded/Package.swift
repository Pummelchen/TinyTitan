// swift-tools-version: 6.4
import PackageDescription

// A consumer package: this is what "use TinyTitan as an LLM engine" looks like
// from outside the repository. The dependency is a relative path so the fixture
// tracks the working tree it sits in; a real consumer writes the same line
// against a released tag instead:
//
//     .package(url: "https://github.com/Pummelchen/TinyTitan", from: "5.15.0")
//
// Both forms are known to resolve (measured 2026-10-02, Swift 6.4 / Xcode 27,
// debug and release), which is what `tools/embedded-dependency-check.sh` reruns
// so a later manifest change cannot break it quietly. See
// docs/plan-embedded-library.md.
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
                .product(name: "TinyTitan", package: "TinyTitan")
            ]
        )
    ]
)
