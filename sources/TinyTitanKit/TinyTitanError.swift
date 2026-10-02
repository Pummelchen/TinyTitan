// The typed errors the embedded engine reports.
//
// Part of the supported facade (phase A1 of `docs/plan-embedded-library.md`,
// §4). A caller fixing a path and a caller waiting for support need to be told
// apart, so the cases are specific rather than one string.
//
// Two cases are declared but not yet produced: `unsupportedFormat` and
// `integrityFailure` need the loader's `ModelError`, which is internal to the
// engine target, so a failure there is rethrown unclassified today. See the
// phase A1 report.
import Foundation

public enum TinyTitanError: Error, Sendable {
    /// The directory does not exist.
    case modelNotFound(URL)
    /// The directory is there but is not a readable TinyTitan install.
    case notAnInstall(URL)
    /// The manifest declares a family this build does not implement.
    case unsupportedFamily(family: String)
    /// The install's format magic is not one this build reads.
    case unsupportedFormat(magic: String)
    /// A file's bytes do not match the manifest's digest.
    case integrityFailure(path: String, expected: String, actual: String)
    /// The prompt plus the requested completion do not fit the context window.
    case contextWindowExceeded(prompt: Int, window: Int)
    /// No usable Metal device, or not the one the engine must run on.
    case metalUnavailable(reason: String)
    /// The caller cancelled the generation.
    case cancelled
    /// The engine this session belonged to was unloaded.
    case engineShutDown
}
