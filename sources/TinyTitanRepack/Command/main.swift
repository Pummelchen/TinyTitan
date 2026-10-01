import Foundation
import TinyTitanRepackCore

private let supportedModelNames = SupportedModelSource.all.map(\.name).joined(separator: "|")

private let usage = """
    Usage:
      TinyTitanRepack [--model <\(supportedModelNames)>] --output <model.ssdai> [--overwrite] [--resume]
      TinyTitanRepack --input-snapshot <affine-safetensors-dir> --model-id <id> --output <model.ssdai> [--overwrite]
      TinyTitanRepack --discard-partial --output <model.ssdai>
      TinyTitanRepack --verify-install --input-ssdai <model.ssdai>  (--input-gturbo still accepted)
      TinyTitanRepack --help

    The installer streams the selected Qwen 3.6 or text-only Ornith 1.5 checkpoint
    (default: Ornith 8-bit) from Hugging Face and repackages it without materializing
    the source checkpoint on disk. Set HF_TOKEN only if Hugging Face requests
    authentication. A cancelled or interrupted download can be continued with
    --resume or removed with --discard-partial.

    --input-snapshot imports a completed local MLX-affine safetensors snapshot.
    It is intended for reproducibly derived sidecars such as Ornith's native MTP
    draft and does not support --resume because no network payload is involved.
    """

private struct Arguments {
    var model = SupportedModelSource.default
    var modelExplicit = false
    var output: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var inputSSDAI: String?
    var inputSnapshot: String?
    var localModelID: String?
    var draftHead = false
    var shareNgramTable = false

    static func parse(_ values: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw ParseError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--model":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                guard let source = SupportedModelSource.named(values[index + 1]) else {
                    throw ParseError.invalidMode(
                        "unknown model \"\(values[index + 1])\"; supported: "
                            + SupportedModelSource.all.map(\.name).joined(separator: ", "))
                }
                parsed.model = source
                parsed.modelExplicit = true
                index += 2
            case "--draft-head":
                parsed.draftHead = true
                index += 1
            case "--share-ngram-table":
                parsed.shareNgramTable = true
                index += 1
            case "--output", "--input-ssdai", "--input-gturbo", "--input-snapshot", "--model-id":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                if flag == "--output" {
                    parsed.output = values[index + 1]
                } else if flag == "--input-ssdai" || flag == "--input-gturbo" {
                    parsed.inputSSDAI = values[index + 1]
                } else if flag == "--input-snapshot" {
                    parsed.inputSnapshot = values[index + 1]
                } else {
                    parsed.localModelID = values[index + 1]
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputSSDAI == nil,
                parsed.inputSnapshot == nil,
                parsed.localModelID == nil,
                !parsed.modelExplicit,
                !parsed.overwrite,
                !parsed.verifyInstall
            else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.inputSnapshot != nil || parsed.localModelID != nil {
            guard parsed.inputSnapshot != nil else {
                throw ParseError.missingRequired("--input-snapshot")
            }
            guard parsed.localModelID != nil else {
                throw ParseError.missingRequired("--model-id")
            }
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard !parsed.modelExplicit,
                parsed.inputSSDAI == nil,
                !parsed.resume,
                !parsed.discardPartial,
                !parsed.verifyInstall
            else {
                throw ParseError.invalidMode(
                    "local snapshot import accepts only --input-snapshot, --model-id, --output, --draft-head, --share-ngram-table, and --overwrite"
                )
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputSSDAI != nil else {
                throw ParseError.missingRequired("--input-ssdai")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume else {
                throw ParseError.invalidMode("verification accepts only --input-ssdai")
            }
        } else {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputSSDAI == nil else {
                throw ParseError.invalidMode("--input-ssdai requires --verify-install")
            }
        }
        return parsed
    }
}

private enum ParseError: Error, CustomStringConvertible {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)

    var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "missing required argument: \(flag)"
        case .invalidMode(let message): return message
        }
    }
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func run(_ values: [String]) async -> Int32 {
    let arguments: Arguments
    do {
        arguments = try Arguments.parse(values)
    } catch ParseError.help {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    if arguments.discardPartial, let output = arguments.output {
        do {
            try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            print("Discarded saved download for \(output)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.verifyInstall, let input = arguments.inputSSDAI {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputSSDAI: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    if let input = arguments.inputSnapshot,
        let modelID = arguments.localModelID,
        let output = arguments.output
    {
        do {
            let result = try await RemoteStreamingRepacker.runLocalSnapshot(
                options: LocalSnapshotRepackOptions(
                    inputSnapshotDir: input,
                    outputDir: output,
                    modelID: modelID,
                    draftHead: arguments.draftHead,
                    shareNgramTable: arguments.shareNgramTable,
                    overwrite: arguments.overwrite))
            print("Imported local snapshot")
            print("Source fingerprint: \(result.resolvedCommit)")
            print("Model: \(result.outputDir)")
            return 0
        } catch {
            printError("local import failed: \(error)")
            return 1
        }
    }

    guard let output = arguments.output else { return 2 }
    let source = arguments.model
    let options = source.installOptions(
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        resume: arguments.resume)
    do {
        let result = try await RemoteStreamingRepacker(options: options).run()
        print("Installed \(source.displayName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))
