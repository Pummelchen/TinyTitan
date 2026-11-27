import Foundation
import TinyTitanCore
import TinyTitanPluginAPI
import TinyTitanVariants

/// The memory-fit facts `doctor` reports for an install the plugin registry
/// serves.
///
/// The arithmetic is the plugin's, not a family table here: the resolved
/// plugin's own load plan supplies the expert-cache slot count, the tuned
/// budget and the prefill chunk, so doctor reports what a load would really
/// allocate. This replaces the 6.x adapter's estimator when that adapter left
/// the product (TT-142).
struct InstallFitEstimate: Sendable {
    let weightBytes: UInt64
    let kvBytes: UInt64
    let cacheFloorBytes: UInt64
    let expertCacheBytes: UInt64
    let expertSlots: Int?
    let prefillChunkTokens: Int
    let pageCachePolicy: String
    let pageCacheReason: String
}

enum InstallFitEstimator {
    /// Reads the manifest and the resolved plugin's plan. No weights are
    /// touched; the only file read is the bounded manifest.
    static func estimate(
        directory: URL,
        physicalMemory: UInt64,
        contextTokens: Int,
        environment: [String: String]
    ) throws -> InstallFitEstimate {
        let resolved = try EngineFactories.pluginRegistry.resolve(directory)
        // A report must not print the load planner's diagnostic lines: the
        // plan is asked for its numbers, not its log.
        let factory = resolved.plugin.engineFactory(
            host: HostServices(
                log: { _ in },
                cacheDirectory: FileManager.default.temporaryDirectory))
        let plan = try factory.prepareLoad(
            SessionBuildOptions(
                modelDirectory: directory,
                context: nil,
                maxContext: contextTokens,
                slots: 1,
                promptCacheMaximumEntries: 0,
                promptCacheMemoryLimitBytes: 0,
                promptCacheDiskDirectory: nil,
                promptCacheDiskLimitBytes: 0,
                prefillChunkTokens: nil,
                kvCachePrecision: .int8,
                hostKVWindowTokens: nil,
                ropeScalingMode: .none,
                thinkingMode: .off,
                reasoningEffort: nil,
                expertCacheSlots: nil,
                expertCacheBudgetBytes: nil,
                mtpModelDirectory: nil,
                mtpMemoryMiB: 0,
                forceLogitsHead: true))
        let manifest = try SSDAIManifestReader.read(directoryURL: directory)
        let arch = plan.architecture
        let routedExperts = manifest.expertsPerLayer > 0 && manifest.expertStride > 0
        let expertSlots: Int?
        let cacheFloorBytes: UInt64
        let expertCacheBytes: UInt64
        if routedExperts {
            let slots = plan.expertCacheSlots
            expertSlots = slots
            cacheFloorBytes = clampedProduct(
                UInt64(RuntimeLimits.minimumExpertCacheSlots),
                manifest.expertStride,
                UInt64(arch.numLayers))
            expertCacheBytes = clampedProduct(
                UInt64(slots),
                manifest.expertStride,
                UInt64(arch.numLayers))
        } else {
            expertSlots = nil
            cacheFloorBytes = 0
            expertCacheBytes = 0
        }
        let weights = manifest.files["model_weights.bin"]?.size ?? 0
        let kvPerToken = kvBytesPerToken(arch, precision: .int8)
        let kvBytes = clampedProduct(UInt64(kvPerToken), UInt64(contextTokens))
        let installBytes = manifest.files.values.reduce(UInt64(0)) { total, file in
            let (sum, overflow) = total.addingReportingOverflow(file.size)
            return overflow ? UInt64.max : sum
        }
        let pageCache = pageCacheDecision(
            physicalMemory: physicalMemory,
            installBytes: installBytes,
            environment: environment)
        return InstallFitEstimate(
            weightBytes: weights,
            kvBytes: kvBytes,
            cacheFloorBytes: cacheFloorBytes,
            expertCacheBytes: expertCacheBytes,
            expertSlots: expertSlots,
            prefillChunkTokens: plan.prefillConfiguration.chunkTokens,
            pageCachePolicy: pageCache.policy,
            pageCacheReason: pageCache.reason)
    }

    /// K/V bytes a token occupies across every row-bearing layer, mirroring
    /// the engines' storage layout at the server's default 8-bit KV precision.
    private static func kvBytesPerToken(
        _ arch: ArchConfig,
        precision: KVCachePrecision
    ) -> Int {
        // The engines' KV quantizer packs groups of 64 values.
        let groupSize = 64
        var total = 0
        for mask in arch.fullAttentionLayerMask {
            // Linear-attention layers keep GDN state, not K/V rows.
            if mask == 2 { continue }
            let elements =
                mask == 0
                ? arch.numKVHeads * arch.headDim
                : arch.numFullKVHeads * arch.fullHeadDim
            let packed = (elements * precision.rawValue + 7) / 8
            let alignedPacked = (packed + 1) & ~1
            let groups = (elements + groupSize - 1) / groupSize
            let stride = alignedPacked + groups * 2 * MemoryLayout<Float16>.size
            // One K row and one V row per token.
            total += stride * 2
        }
        return total
    }

    /// The engines' page-cache rule, repeated here because doctor must report
    /// the policy the load will take without importing an engine module. The
    /// rule is memory and install size only, never model knowledge.
    private static func pageCacheDecision(
        physicalMemory: UInt64,
        installBytes: UInt64,
        environment: [String: String]
    ) -> (policy: String, reason: String) {
        let variable = "TINYTITAN_BOUNDED_IO"
        if let raw = environment[variable] {
            switch raw {
            case "0": return ("page-cache", "\(variable)=0 (explicit)")
            case "1": return ("bounded", "\(variable)=1 (explicit)")
            default:
                return (
                    "bounded",
                    "\(variable)=\(raw) is not 0 or 1; "
                        + "refusing to enable an undeclared cache"
                )
            }
        }
        guard physicalMemory > 0 else {
            return ("bounded", "physical memory is unavailable")
        }
        guard installBytes > 0 else {
            return ("bounded", "install size is unavailable")
        }
        let (twice, overflow) = installBytes.multipliedReportingOverflow(by: 2)
        guard !overflow else {
            return ("bounded", "install size overflows the 2x rule")
        }
        guard physicalMemory >= twice else {
            return (
                "bounded",
                "install is \(installBytes) bytes; \(physicalMemory) bytes "
                    + "of RAM is below the 2x threshold"
            )
        }
        return ("page-cache", "install is \(installBytes) bytes and RAM is at least 2x it")
    }

    private static func clampedProduct(_ values: UInt64...) -> UInt64 {
        values.reduce(UInt64(1)) { total, value in
            let (product, overflow) = total.multipliedReportingOverflow(by: value)
            return overflow ? UInt64.max : product
        }
    }
}
