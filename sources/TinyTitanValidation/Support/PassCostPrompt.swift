import Foundation

/// The code the pass-cost gate asks the model to restate.
///
/// A measurement must not silently substitute a different prompt: the old
/// fallback in `PassCostCommands` returned a three-line function whenever the
/// source file could not be read (a moved file, a wrong working directory), so
/// the gate could be timed on the wrong input and reported as a measurement.
/// A missing source is an error here.
package enum PassCostPrompt {
    package static let defaultSource =
        "sources/Engines/Legacy6xRuntime/Runtime/Generation/StreamingMTP.swift"

    package enum Failure: Error, CustomStringConvertible {
        case unreadable(path: String, underlying: any Error)

        package var description: String {
            switch self {
            case .unreadable(let path, let underlying):
                return "pass-cost prompt source is unreadable at \(path): \(underlying)"
            }
        }
    }

    /// The prompt, from `promptFile` when the operator supplied one, otherwise
    /// the first 140 lines of the pass-cost source. Throws rather than
    /// substituting a different prompt.
    package static func text(promptFile: String?, source: String = defaultSource) throws -> String {
        let body: String
        if let promptFile {
            body = try String(contentsOfFile: promptFile, encoding: .utf8)
        } else {
            let read: String
            do {
                read = try String(contentsOfFile: source, encoding: .utf8)
            } catch {
                throw Failure.unreadable(path: source, underlying: error)
            }
            body =
                read
                .split(separator: "\n", omittingEmptySubsequences: false)
                .prefix(140).joined(separator: "\n")
        }
        return "Below is a Swift source file.\n```swift\n\(body)\n```\n\n"
            + "Here is the same file with every doc comment shortened to one line:\n"
            + "```swift\n"
    }
}
