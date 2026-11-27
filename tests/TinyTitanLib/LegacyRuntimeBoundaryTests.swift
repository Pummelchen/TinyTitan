import Foundation
import Testing

/// The host reaches the 6.x runtime through the labelled adapter, never by
/// importing it directly.
///
/// TT-134 moved the runtime to `sources/Engines/Legacy6xRuntime` and put the
/// host on the 7.x factory surface (`EngineFactories`, `ResolvedPlugin`,
/// `ModelIdentity`). What is left importing the module is deliberate and each
/// file is listed below with the reason it is still there; everything else
/// must go through `TinyTitanLegacyEngine` (the adapter), whose families are
/// migrated into plugins one at a time (TT-132).
///
/// The test fails for a new importer — which is the regression it exists to
/// catch — and for a row that no longer imports, so the list cannot outlive
/// the files it describes. Model-free: it reads the source tree.
@Suite struct LegacyRuntimeBoundaryTests {
    /// Every production file allowed to import the 6.x runtime, and why.
    static let allowed: [String: String] = [
        // The adapter itself: the porting surface TT-132/134 replace.
        "sources/TinyTitanLegacyEngine/DefaultEngineConfiguration.swift": "adapter",
        "sources/TinyTitanLegacyEngine/EngineFactories.swift": "adapter",
        "sources/TinyTitanLegacyEngine/EngineLoadPolicy.swift": "adapter",
        "sources/TinyTitanLegacyEngine/LegacyBatchedMemoryBudget.swift": "adapter",
        "sources/TinyTitanLegacyEngine/LegacyInstallEstimator.swift": "adapter",
        "sources/TinyTitanLegacyEngine/ServerModelIdentity.swift": "adapter",
        "sources/TinyTitanLegacyEngine/ServerModelSession+Benchmark.swift": "adapter",
        "sources/TinyTitanLegacyEngine/ServerModelSession+ExpertWarmup.swift": "adapter",
        // The catalog's 6.x bridges: Kind resolution and reasoning targets.
        "sources/TinyTitanServer/Core/CatalogKindBridges.swift": "catalog bridges",
        // Off-disk families with no plugin yet: the CLI's own routes.
        "sources/TinyTitanCLI/RunGGUF.swift": "as-is GGUF route (TT-058)",
        // Benchmark probes of the 6.x path and the off-disk families.
        "sources/TinyTitanBench/LagunaCommands.swift": "Laguna bench (TT-054)",
        "sources/TinyTitanBench/PassCostCommands.swift":
            "pass-cost bench over the 6.x path (TT-061)",
        "sources/TinyTitanBench/VerifyProbeCommands.swift":
            "verify probe over the 6.x path (TT-061)",
    ]

    static func repositoryRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    static func importers() throws -> [String] {
        let root = repositoryRoot()
        let sources = root.appendingPathComponent("sources")
        guard
            let walker = FileManager.default.enumerator(
                at: sources, includingPropertiesForKeys: nil)
        else {
            throw BoundaryError.unescapable("cannot walk sources/")
        }
        var found: [String] = []
        for case let url as URL in walker where url.pathExtension == "swift" {
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            let imports = text.split(separator: "\n").contains { line in
                line == "import Legacy6xRuntime" || line == "@testable import Legacy6xRuntime"
            }
            if imports {
                found.append(
                    url.path.replacingOccurrences(of: root.path + "/", with: ""))
            }
        }
        return found.sorted()
    }

    enum BoundaryError: Error, CustomStringConvertible {
        case unescapable(String)
        var description: String {
            switch self {
            case .unescapable(let why): return why
            }
        }
    }

    @Test func onlyTheListedFilesImportTheSixXRuntime() throws {
        let found = try Self.importers()
        #expect(!found.isEmpty, "the scan found no importers at all; it read nothing")
        let unexpected = found.filter { Self.allowed[$0] == nil }
        #expect(
            unexpected.isEmpty,
            """
            these files import the 6.x runtime without being listed: \(unexpected). \
            Go through TinyTitanLegacyEngine (the adapter), or add the file with \
            the reason it must stay — a host file reaching back into 6.x is the \
            coupling TT-134 removed.
            """)
    }

    @Test func theListHasNoStaleRows() throws {
        let found = Set(try Self.importers())
        let stale = Self.allowed.keys.filter { !found.contains($0) }.sorted()
        #expect(
            stale.isEmpty,
            "these rows claim an import that is gone; delete them: \(stale)")
    }
}
