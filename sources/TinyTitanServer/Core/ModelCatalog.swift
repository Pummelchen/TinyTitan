import Foundation
import TinyTitanCore
import TinyTitanLib
import TinyTitanPluginAPI
import TinyTitanVariants

/// Every model a models directory holds that this server can actually serve.
///
/// Identity is read, never parsed out of a directory name: an install's id,
/// family, width and name come from the plugin the registry resolves for it.
/// A directory the scan cannot read is skipped with a reason rather than
/// failing startup, because one half-copied model must not take the other nine
/// offline.
public struct ModelCatalog: Sendable {

    /// What a model is: a family label and the thinking control its template
    /// implements. Every route this host serves is a GPU plugin (TT-142
    /// slice (b)), so the engine is not a per-install choice any more.
    public enum Kind: Sendable, Equatable {
        case gpu(family: String, reasoning: ReasoningControl)

        package var reasoning: ReasoningControl {
            switch self {
            case .gpu(_, let reasoning): reasoning
            }
        }

        public var familyName: String {
            switch self {
            case .gpu(let family, _): family
            }
        }

        public var supportedReasoningLevels: [TinyTitanCore.ReasoningLevel] {
            reasoning.supportedLevels
        }

        public var levelWhenOn: TinyTitanCore.ReasoningLevel {
            reasoning.levelWhenOn
        }

        public func runtimeReasoning(
            for level: TinyTitanCore.ReasoningLevel
        ) throws -> (thinking: ModelThinkingMode, effort: ModelReasoningEffort?) {
            try reasoning.runtimeReasoning(for: level, family: familyName)
        }
    }

    public struct Entry: Sendable, Equatable {
        public let id: String
        public let name: String
        public let kind: Kind
        /// Weight width in bits: the install's routed experts.
        public let quant: Int
        public let path: URL
        /// The sampling the model will really use for anything a request
        /// omits: the tuning profile's for the install.
        public let sampling: SamplingProfile
        /// The context the checkpoint claims, where the engine honours less
        /// than the server's --max-context. Nil leaves the window to the
        /// server.
        public let contextLimit: Int?
        public let sizeBytes: Int64

        package init(
            id: String, name: String, kind: Kind, quant: Int, path: URL,
            sampling: SamplingProfile, contextLimit: Int? = nil,
            sizeBytes: Int64 = 0
        ) {
            self.id = id
            self.name = name
            self.kind = kind
            self.quant = quant
            self.path = path
            self.sampling = sampling
            self.contextLimit = contextLimit
            self.sizeBytes = sizeBytes
        }
    }

    public struct Skipped: Sendable, Equatable {
        public let path: URL
        public let reason: String
    }

    public let directory: URL
    public private(set) var entries: [Entry]
    public private(set) var skipped: [Skipped]

    public init(directory: URL, entries: [Entry], skipped: [Skipped] = []) {
        self.directory = directory
        self.entries = entries
        self.skipped = skipped
    }

    // MARK: - Scanning

    public static func scan(
        directory: URL,
        fileManager: FileManager = .default
    ) -> ModelCatalog {
        let root = directory.standardizedFileURL
        var catalog = ModelCatalog(directory: root, entries: [])
        let children: [URL]
        do {
            children = try fileManager.contentsOfDirectory(
                at: root, includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
        } catch {
            catalog.skipped.append(Skipped(path: root, reason: "cannot list: \(error)"))
            return catalog
        }
        for child in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            // A model linked in from another disk is served from where it
            // lives; the link itself does not pass as a directory.
            let target = child.resolvingSymlinksInPath()
            // Lock files and other loose files sit beside the installs; they
            // are not models and not worth a warning.
            guard (try? target.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
            else {
                continue
            }
            catalog.add(probing: target)
        }
        catalog.entries.sort { $0.id < $1.id }
        return catalog
    }

    /// Probes one directory and adds it, or records why it was skipped.
    /// Returns the entry when one was added.
    @discardableResult
    public mutating func add(probing directory: URL) -> Entry? {
        let path = directory.standardizedFileURL
        switch Self.probe(path) {
        case .success(let entry):
            if let existing = entries.first(where: { $0.id == entry.id }) {
                skipped.append(
                    Skipped(
                        path: path,
                        reason:
                            "duplicate id \(entry.id), already served from \(existing.path.path)"))
                return nil
            }
            entries.append(entry)
            return entry
        case .failure(let failure):
            skipped.append(Skipped(path: path, reason: failure.reason))
            return nil
        }
    }

    struct ProbeFailure: Error {
        let reason: String
    }

    static func probe(_ directory: URL) -> Result<Entry, ProbeFailure> {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: directory.appendingPathComponent("manifest.json").path)
        else {
            return .failure(
                ProbeFailure(reason: "no manifest.json (not a TinyTitan install)"))
        }
        return probeInstall(directory)
    }

    // MARK: - Reporting

    /// One notice for everything skipped, written once at startup.
    public func reportSkipped(to handle: FileHandle = .standardError) {
        guard !skipped.isEmpty else { return }
        var lines = [
            "catalog: skipped \(skipped.count) of \(skipped.count + entries.count) "
                + "directories in \(directory.path):"
        ]
        for item in skipped {
            lines.append("  \(item.path.lastPathComponent): \(item.reason)")
        }
        handle.write(Data((lines.joined(separator: "\n") + "\n").utf8))
    }

    /// `TinyTitanServer --catalog` output, which a launcher script parses.
    ///
    /// Assembled by hand because JSONEncoder does not keep key order, and the
    /// document should read in the order the launcher documents its fields.
    /// Strings still go through the encoder, so escaping is not hand-rolled.
    public func jsonData() throws -> Data {
        let models = try entries.map(Self.json(for:))
        return Data(("{\"models\":[" + models.joined(separator: ",") + "]}").utf8)
    }

    private static func json(for entry: Entry) throws -> String {
        let thinking = try entry.kind.supportedReasoningLevels.map { try quoted($0.rawValue) }
        let sampling =
            "{\"temperature\":\(decimal(entry.sampling.temperature, places: 4)),"
            + "\"top_p\":\(decimal(entry.sampling.topP, places: 4)),"
            + "\"top_k\":\(entry.sampling.topK)}"
        let fields: [(key: String, value: String)] = [
            ("id", try quoted(entry.id)),
            ("name", try quoted(entry.name)),
            ("family", try quoted(entry.kind.familyName)),
            ("quant", String(entry.quant)),
            ("backend", try quoted("gpu")),
            ("engines", try quoted("gpu")),
            ("path", try quoted(entry.path.path)),
            ("thinking", "[" + thinking.joined(separator: ",") + "]"),
            ("sampling", sampling),
            ("size_gb", decimal(Double(entry.sizeBytes) / 1e9, places: 1)),
        ]
        return "{" + fields.map { "\"\($0.key)\":\($0.value)" }.joined(separator: ",") + "}"
    }

    private static func quoted(_ value: String) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        return try encoder.encode(value).lossyUTF8String
    }

    /// Float 0.6 widens to 0.6000000238; a launcher showing the defaults
    /// should read the value the model card states.
    private static func decimal(_ value: some BinaryFloatingPoint, places: Int) -> String {
        let scale = pow(10.0, Double(places))
        return String((Double(value) * scale).rounded() / scale)
    }
}
