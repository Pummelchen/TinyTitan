import Foundation
import TinyTitanLib
import TinyTitanCore
import TinyTitanPluginAPI
import TinyTitanVariants

/// Reads a `models/` tree without loading a model or touching its weights.
///
/// The scan is deliberately the same shape as `ModelCatalog.scan` (one
/// directory per install, skipped directories carry a reason) but stays in the
/// CLI target: pulling `TinyTitanServerCore` in would link NIO into a command
/// whose whole job is to print facts. Everything the facts are derived from is
/// still the shared reader and configuration code.
enum DoctorModelScanner {
    struct ScanResult {
        let exists: Bool
        let models: [DoctorReport.Model]
        let skipped: [DoctorReport.Skipped]
    }

    static func scan(
        directory: URL,
        physicalMemory: UInt64,
        contextTokens: Int,
        environment: [String: String]
    ) -> ScanResult {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        guard
            fileManager.fileExists(
                atPath: directory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            return ScanResult(exists: false, models: [], skipped: [])
        }
        let children: [URL]
        do {
            children = try fileManager.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
        } catch {
            return ScanResult(
                exists: true, models: [],
                skipped: [
                    DoctorReport.Skipped(
                        name: directory.lastPathComponent,
                        reason: "cannot list: \(error)")
                ])
        }
        var models: [DoctorReport.Model] = []
        var skipped: [DoctorReport.Skipped] = []
        for child in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let path = child.resolvingSymlinksInPath()
            guard
                (try? path.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory
                    == true
            else { continue }
            if fileManager.fileExists(
                atPath: path.appendingPathComponent("manifest.json").path)
            {
                switch probeInstall(
                    path, physicalMemory: physicalMemory,
                    contextTokens: contextTokens, environment: environment)
                {
                case .success(let model):
                    models.append(model)
                case .failure(let failure):
                    skipped.append(
                        DoctorReport.Skipped(
                            name: child.lastPathComponent, reason: failure.reason))
                }
            }
        }
        models.sort { $0.id < $1.id }
        return ScanResult(exists: true, models: models, skipped: skipped)
    }

    private struct ProbeFailure: Error {
        let reason: String
    }

    private static func probeInstall(
        _ directory: URL,
        physicalMemory: UInt64,
        contextTokens: Int,
        environment: [String: String]
    ) -> Result<DoctorReport.Model, ProbeFailure> {
        // Identity comes from the engine plugin that serves the install, never
        // from a family table here: the adapter and the shipped plugins name
        // their own installs, and a family that cannot be served says so in its
        // identity instead of teaching doctor the family list.
        let identity: ModelIdentity
        do {
            let resolved = try EngineFactories.pluginRegistry.resolve(directory)
            guard let declared = try resolved.modelIdentity() else {
                return .failure(
                    ProbeFailure(
                        reason: "the engine plugin declares no identity for this install"))
            }
            identity = declared
        } catch let error as PluginRegistryError {
            switch error {
            case .noPlugin:
                return .failure(
                    ProbeFailure(
                        reason: "no engine plugin serves this install (\(directory.path))"))
            case .ambiguous, .incompatible:
                return .failure(ProbeFailure(reason: "unreadable manifest.json: \(error)"))
            }
        } catch {
            return .failure(
                ProbeFailure(reason: "unreadable manifest.json: \(error)"))
        }
        if let reason = identity.unsupportedReason ?? identity.cliUnsupportedReason {
            return .failure(ProbeFailure(reason: reason))
        }
        // The memory-fit arithmetic comes from the same plugin's load plan, so
        // the slot count and prefill chunk are the ones a real load takes.
        let estimate: InstallFitEstimate
        do {
            estimate = try InstallFitEstimator.estimate(
                directory: directory, physicalMemory: physicalMemory,
                contextTokens: contextTokens, environment: environment)
        } catch {
            return .failure(
                ProbeFailure(reason: "unreadable manifest.json: \(error)"))
        }
        let total = clampedSum(
            estimate.weightBytes, estimate.kvBytes, estimate.expertCacheBytes)
        let fit = DoctorReport.Model.Fit(
            weightsBytes: estimate.weightBytes,
            kvBytes: estimate.kvBytes,
            kvPrecision: KVCachePrecision.int8.label,
            contextTokens: contextTokens,
            cacheFloorBytes: estimate.cacheFloorBytes,
            expertCacheBytes: estimate.expertCacheBytes,
            totalBytes: total,
            ramBytes: physicalMemory,
            fits: total <= physicalMemory,
            utilization: physicalMemory == 0
                ? 0 : Double(total) / Double(physicalMemory))
        return .success(
            DoctorReport.Model(
                id: identity.modelID,
                directory: directory.lastPathComponent,
                family: identity.family,
                weightBits: identity.weightBits,
                sizeBytes: totalFileSize(directory),
                fit: fit,
                expertSlots: estimate.expertSlots,
                prefillChunkTokens: estimate.prefillChunkTokens,
                pageCache: DoctorReport.Model.PageCache(
                    policy: estimate.pageCachePolicy,
                    reason: estimate.pageCacheReason),
                note: nil))
    }

    private static func clampedSum(_ values: UInt64...) -> UInt64 {
        values.reduce(UInt64(0)) { total, value in
            let (sum, overflow) = total.addingReportingOverflow(value)
            return overflow ? UInt64.max : sum
        }
    }

    /// Logical file sizes, not allocated blocks.
    ///
    /// `ModelCatalog` reports allocated bytes because it describes what an
    /// install occupies; `doctor` reports logical bytes because its output is
    /// compared against a golden and APFS allocation is not stable across
    /// filesystems. Nothing here reads a byte of a weight file.
    static func totalFileSize(_ directory: URL) -> Int64 {
        let keys: [URLResourceKey] = [.fileSizeKey, .isRegularFileKey]
        guard
            let walker = FileManager.default.enumerator(
                at: directory, includingPropertiesForKeys: keys)
        else { return 0 }
        var total: Int64 = 0
        for case let file as URL in walker {
            guard
                let values = try? file.resourceValues(forKeys: Set(keys)),
                values.isRegularFile == true
            else { continue }
            total += Int64(values.fileSize ?? 0)
        }
        return total
    }
}
