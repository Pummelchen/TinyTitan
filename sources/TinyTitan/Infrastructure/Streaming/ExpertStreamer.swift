import Foundation
import TinyTitanFormat

enum StreamerError: Error, CustomStringConvertible {
    case openFailed(path: String, errno: Int32)
    case sizeMismatch(expected: UInt64, actual: UInt64)
    case offsetOutOfRange(UInt64)
    case bufferWrapFailed
    case preadFailed(errno: Int32)
    case allocFailed(errno: Int32)
    case slotOutOfRange(Int)

    public var description: String {
        switch self {
        case .openFailed(let path, let error):
            return "open(\(path)) failed: errno \(error)"
        case .sizeMismatch(let expected, let actual):
            return "file size mismatch: expected \(expected), got \(actual)"
        case .offsetOutOfRange(let offset):
            return "offset \(offset) is outside the streamed range"
        case .bufferWrapFailed:
            return "failed to wrap expert cache memory in an MTLBuffer"
        case .preadFailed(let error):
            return "pread failed: errno \(error)"
        case .allocFailed(let error):
            return "posix_memalign failed: errno \(error)"
        case .slotOutOfRange(let slot):
            return "expert cache slot \(slot) is out of range"
        }
    }
}

/// Byte layout of one routed-expert layer file.
public struct StreamLayout: Sendable {
    public let path: String
    public let streamOffset: UInt64
    public let streamSize: UInt64
    public let expertsPerLayer: Int
    public let expertStride: UInt64
    public let expertOffsets: [UInt64]?
    /// Bytes of the expert blob that decode's phase 1 (gate+up) consumes,
    /// when the manifest records where the `down` tensor starts. `nil` means
    /// whole-region reads; a split needs a boundary strictly inside the
    /// stride, which the streamer refuses anything else for.
    public let phase1Bytes: UInt64?

    /// Per-expert compressed frames, when this layer's blobs are compressed.
    ///
    /// `nil` means the layer is stored raw and every expert is `expertStride`
    /// bytes at `expertOffset(...)`. Present means those offsets address the
    /// *index* instead: each expert is one or two LZFSE frames, and a split
    /// layer has one frame per half so the reader keeps its phase-1 overlap.
    ///
    /// `package` rather than `public`: the frame type lives in
    /// `TinyTitanFormat`, whose types are package-scoped, and nothing outside
    /// this package builds a `StreamLayout`.
    package let compressedFrames: [[ExpertBlobIndex.Blob]]?
    /// The decoded size of one block in a v3 index, or 0 for a legacy (v2)
    /// index whose entries are whole-phase frames.
    package let compressedBlockBytes: UInt64

    package init(
        path: String,
        streamOffset: UInt64,
        streamSize: UInt64,
        expertsPerLayer: Int,
        expertStride: UInt64,
        expertOffsets: [UInt64]? = nil,
        phase1Bytes: UInt64? = nil,
        compressedFrames: [[ExpertBlobIndex.Blob]]? = nil,
        compressedBlockBytes: UInt64 = 0
    ) {
        self.path = path
        self.streamOffset = streamOffset
        self.streamSize = streamSize
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
        self.expertOffsets = expertOffsets
        self.phase1Bytes = phase1Bytes
        self.compressedFrames = compressedFrames
        self.compressedBlockBytes = compressedBlockBytes
    }

    /// The byte offset of one expert within this layer's file.
    ///
    /// Unchecked on purpose, and safe only because the type's contract is
    /// validated where the layout is built: `perLayer` must be representable
    /// (`expertStride * expertsPerLayer`, checked in `Model` before the
    /// `StreamLayout` is formed) and `layer`/`expert` must be within the model's
    /// and the layout's bounds. Checking here would put a reporting multiply on
    /// every expert read, which is the whole decode path.
    @inline(always)
    public func expertOffset(layer: Int, expert: Int) -> UInt64 {
        if layer == 0, let expertOffsets, expert >= 0, expert < expertOffsets.count {
            return expertOffsets[expert]
        }
        let perLayer = UInt64(expertsPerLayer) * expertStride
        return UInt64(layer) * perLayer + UInt64(expert) * expertStride
    }
}
