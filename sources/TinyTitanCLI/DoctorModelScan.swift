import Foundation
import TinyTitanLib
import TinyTitanCore
import TinyTitanLegacyEngine
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
            } else if fileManager.fileExists(
                atPath: path.appendingPathComponent("config.json").path)
            {
                switch probeSnapshot(path) {
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
        } catch {
            return .failure(
                ProbeFailure(reason: "unreadable manifest.json: \(error)"))
        }
        if let reason = identity.unsupportedReason ?? identity.cliUnsupportedReason {
            return .failure(ProbeFailure(reason: reason))
        }
        // The memory-fit arithmetic still reads the 6.x tuning profile, so it
        // lives beside the adapter (`LegacyInstallEstimator`); the report's
        // identity above is the plugin's.
        let estimate: LegacyInstallEstimate
        do {
            estimate = try LegacyInstallEstimator.estimate(
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

    private static func probeSnapshot(
        _ directory: URL
    ) -> Result<DoctorReport.Model, ProbeFailure> {
        let data: Data
        do {
            data = try BoundedMetadataRead.read(
                fileAt: directory.appendingPathComponent("config.json"),
                maxBytes: SSDAIManifestReader.defaultMaxBytes)
        } catch {
            return .failure(
                ProbeFailure(reason: "unreadable config.json: \(error)"))
        }
        guard
            let object = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else {
            return .failure(
                ProbeFailure(reason: "config.json is not a JSON object"))
        }
        let declared =
            (object["model_id"] as? String)
            ?? (object["display_name"] as? String)
        let id =
            declared.flatMap { $0.isEmpty ? nil : $0 }
            ?? directory.lastPathComponent
        let family = (object["model_type"] as? String) ?? "cpu-snapshot"
        let bits =
            (object["quantization"] as? [String: Any])?["bits"] as? Int ?? 0
        return .success(
            DoctorReport.Model(
                id: id,
                directory: directory.lastPathComponent,
                family: family,
                weightBits: bits,
                sizeBytes: totalFileSize(directory),
                fit: nil,
                expertSlots: nil,
                prefillChunkTokens: nil,
                pageCache: nil,
                note: "CPU snapshot: fit is not estimated"))
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
