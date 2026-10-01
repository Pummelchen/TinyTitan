import Foundation
import TinyTitanFormat

// The verifier's local manifest mirror: only the fields this tool needs,
// decoded without the format layer's structural validation.
//
// Split out of `VerifiedInstallTool.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
// The five structs widened from `private` to internal because the loader that
// returns them stays in VerifiedInstallTool.swift.
struct ManifestFileEntry: Decodable {
    let size: UInt64
    let sha256: String
}

struct Manifest: Decodable {
    let files: [String: ManifestFileEntry]
    let expertsPerLayer: Int
    let numLayers: Int
    let expertStride: UInt64
    let sourceSnapshotHash: String?
    /// Decoded through `SSDAIManifestQuantV1`, whose hand-written `Codable`
    /// keeps the open set of per-tensor width keys. A synthesised decoder
    /// would drop them, which is the bug this whole check exists to catch.
    let quant: SSDAIManifestQuantV1?
}

struct PackedExpertsLayout: Decodable {
    let expertStride: UInt64
    let numLayers: Int
    let expertsPerLayer: Int
    let layers: [Layer]
}

struct Layer: Decodable {
    let layer: Int
    let file: String
    let experts: [Expert]
}

struct Expert: Decodable {
    let expert: Int?
    let offset: UInt64
    let size: UInt64
}
