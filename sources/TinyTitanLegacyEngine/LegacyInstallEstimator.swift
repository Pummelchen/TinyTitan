import Foundation
import Legacy6xRuntime
import TinyTitanCore

/// The memory-fit facts `doctor` reports for a 6.x install.
///
/// This arithmetic reads the 6.x tuning profile (`ModelProfile`) and the 6.x
/// runtime's cache rungs, so it lives beside the adapter that still owns those
/// families instead of in the CLI. The CLI supplies the plugin-resolved
/// identity for the report; this only answers "how much memory would the 6.x
/// engine need". When the last family leaves the adapter this type goes with it
/// and doctor takes the same facts from the plugin.
package struct LegacyInstallEstimate: Sendable {
    package let weightBytes: UInt64
    package let kvBytes: UInt64
    package let cacheFloorBytes: UInt64
    package let expertCacheBytes: UInt64
    package let expertSlots: Int?
    package let prefillChunkTokens: Int
    package let pageCachePolicy: String
    package let pageCacheReason: String
}

package enum LegacyInstallEstimator {
    /// Reads the manifest and computes the fit `doctor` prints. No weights are
    /// touched; the only file read is the bounded manifest.
    package static func estimate(
        directory: URL,
        physicalMemory: UInt64,
        contextTokens: Int,
        environment: [String: String]
    ) throws -> LegacyInstallEstimate {
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        let manifest = try ManifestReader.read(directoryURL: directory)
        let profile = ModelProfile.resolve(identity: identity)
        let routedExperts = manifest.expertsPerLayer > 0 && manifest.expertStride > 0
        let expertSlots: Int?
        let cacheFloorBytes: UInt64
        let expertCacheBytes: UInt64
        if routedExperts {
            let budget = RuntimeConfiguration.affordableExpertCacheBudget(
                profile.expertCacheBudgetBytes, physicalMemory: physicalMemory)
            expertSlots = RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: manifest.expertStride,
                layers: manifest.arch.numLayers,
                budgetBytes: budget)
            cacheFloorBytes = clampedProduct(
                UInt64(RuntimeConfiguration.minimumExpertCacheSlots),
                manifest.expertStride,
                UInt64(manifest.arch.numLayers))
            expertCacheBytes = clampedProduct(
                UInt64(expertSlots ?? 0),
                manifest.expertStride,
                UInt64(manifest.arch.numLayers))
        } else {
            expertSlots = nil
            cacheFloorBytes = 0
            expertCacheBytes = 0
        }
        let weights = manifest.files["model_weights.bin"]?.size ?? 0
        let kvPerToken = kvBytesPerToken(manifest.arch, precision: .int8)
        let kvBytes = clampedProduct(
            UInt64(kvPerToken), UInt64(contextTokens))
        let installBytes = PageCachePolicyResolver.installSize(
            fileSizes: manifest.files.values.map(\.size))
        let pageCache = PageCachePolicyResolver.resolve(
            physicalMemory: physicalMemory,
            installBytes: installBytes,
            environment: environment)
        let chunk =
            profile.prefillChunkTokens
            ?? defaultPrefillChunkTokens(
                family: identity.family,
                fallback: RuntimeConfiguration.production.prefillChunkTokens)
        return LegacyInstallEstimate(
            weightBytes: weights,
            kvBytes: kvBytes,
            cacheFloorBytes: cacheFloorBytes,
            expertCacheBytes: expertCacheBytes,
            expertSlots: expertSlots,
            prefillChunkTokens: chunk,
            pageCachePolicy: pageCache.policy.rawValue,
            pageCacheReason: pageCache.reason)
    }

    /// K/V bytes a token occupies across every row-bearing layer, mirroring
    /// `KVCacheManager`'s storage layout at the server's default 8-bit KV
    /// precision.
    private static func kvBytesPerToken(
        _ arch: ManifestArch,
        precision: KVCachePrecision
    ) -> Int {
        var total = 0
        for mask in arch.fullAttentionLayerMask {
            if mask == 2 { continue }
            let elements =
                mask == 0
                ? arch.numKVHeads * arch.headDim
                : arch.numFullKVHeads * arch.fullHeadDim
            let layout = KVCacheManager.rowLayout(
                elements: elements, precision: precision)
            // One K row and one V row per token.
            total += layout.stride * 2
        }
        return total
    }

    private static func clampedProduct(_ values: UInt64...) -> UInt64 {
        values.reduce(UInt64(1)) { total, value in
            let (product, overflow) = total.multipliedReportingOverflow(by: value)
            return overflow ? UInt64.max : product
        }
    }
}
