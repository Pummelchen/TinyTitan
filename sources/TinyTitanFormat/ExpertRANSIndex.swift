import Foundation

/// The sidecar a GPU-rANS expert layer needs.
///
/// The payload file holds one contiguous run of blocks per expert, in the order
/// the models are declared, so the streamer reads an expert with one range and
/// the GPU decodes the blocks in parallel. The sidecar carries what the decoder
/// cannot derive: the normalized frequency model per block index and where each
/// block's payload sits.
///
/// Layout, little-endian:
///
///     magic       4 bytes  "TTRX"
///     version     u16      1
///     scaleBits   u8       14
///     blockCount  u8       blocks per expert
///     expertCount u32
///     reserved    u32
///     then blockCount x 256 u16 normalized frequencies
///     then expertCount x blockCount x (payloadOffset u64, payloadLength u32,
///                                           rawOffset u32, symbolCount u32,
///                                           stride u8, role u8, reserved u16)
package enum ExpertRANSIndex {
    package static let magic: [UInt8] = Array("TTRX".utf8)
    package static let currentVersion: UInt16 = 1
    /// The scale the decoder (host reference loop and GPU kernel) implements.
    /// `RANSExpertLayerWriter` is the only producer, and it writes 14; the
    /// constant lives here so reader and writer cannot drift apart silently.
    package static let implementedScaleBits: UInt8 = 14
    package static let headerBytes = 16
    package static let blockBytes = 24
    package static let maximumBlockCount = 64

    package struct Model: Sendable, Equatable {
        package var frequencies: [UInt16]

        package init(frequencies: [UInt16]) {
            self.frequencies = frequencies
        }

        /// Cumulative frequency across all 256 symbols.
        package var total: Int { frequencies.reduce(0) { $0 + Int($1) } }
    }

    package struct Block: Sendable, Equatable {
        package let payloadOffset: UInt64
        package let payloadLength: UInt32
        /// Byte offset of this block's first symbol inside the expert.
        package let rawOffset: UInt32
        package let symbolCount: UInt32
        /// 1 for a contiguous tensor, 2 for one byte plane of a BF16 tensor.
        package let stride: UInt8
        package let role: UInt8

        package init(
            payloadOffset: UInt64, payloadLength: UInt32, rawOffset: UInt32,
            symbolCount: UInt32, stride: UInt8, role: UInt8
        ) {
            self.payloadOffset = payloadOffset
            self.payloadLength = payloadLength
            self.rawOffset = rawOffset
            self.symbolCount = symbolCount
            self.stride = stride
            self.role = role
        }
    }

    package struct Layout: Sendable, Equatable {
        package let scaleBits: UInt8
        package let models: [Model]
        package let blocks: [[Block]]

        package init(scaleBits: UInt8, models: [Model], blocks: [[Block]]) {
            self.scaleBits = scaleBits
            self.models = models
            self.blocks = blocks
        }
    }

    package enum Failure: Error, CustomStringConvertible {
        case badMagic
        case unsupportedVersion(UInt16)
        case badScaleBits(UInt8)
        case badBlockCount(UInt8)
        case truncated(expected: Int, got: Int)
        case badModel(index: Int, total: Int)
        case badExpertCount(expected: Int, got: Int)
        case badBlock(expert: Int, block: Int, reason: String)

        package var description: String {
            switch self {
            case .badMagic: return "expert rANS index is not a TTRX file"
            case .unsupportedVersion(let version):
                return "expert rANS index version \(version) is not readable here"
            case .badScaleBits(let bits):
                return "expert rANS index scale \(bits) is not the implemented scale 14"
            case .badBlockCount(let count):
                return "expert rANS index declares \(count) blocks an expert"
            case .truncated(let expected, let got):
                return "expert rANS index is \(got) bytes; \(expected) are needed"
            case .badModel(let index, let total):
                return "expert rANS model \(index) sums to \(total), not 2^scaleBits"
            case .badExpertCount(let expected, let got):
                return "expert rANS index declares \(got) experts; the layer has \(expected)"
            case .badBlock(let expert, let block, let reason):
                return "expert \(expert) block \(block): \(reason)"
            }
        }
    }

    package static func encode(_ layout: Layout) -> Data {
        precondition((1...maximumBlockCount).contains(layout.models.count))
        var data = Data(
            capacity: headerBytes + layout.models.count * 512
                + layout.blocks.count * layout.models.count * blockBytes)
        data.append(contentsOf: magic)
        append(UInt16(currentVersion), to: &data)
        data.append(layout.scaleBits)
        data.append(UInt8(layout.models.count))
        append(UInt32(layout.blocks.count), to: &data)
        append(UInt32(0), to: &data)
        for model in layout.models {
            precondition(model.frequencies.count == 256)
            for frequency in model.frequencies {
                append(frequency, to: &data)
            }
        }
        for expert in layout.blocks {
            precondition(expert.count == layout.models.count)
            for block in expert {
                append(block.payloadOffset, to: &data)
                append(block.payloadLength, to: &data)
                append(block.rawOffset, to: &data)
                append(block.symbolCount, to: &data)
                data.append(block.stride)
                data.append(block.role)
                append(UInt16(0), to: &data)
            }
        }
        return data
    }

    /// Decodes and checks what the decoder relies on: model totals, payload
    /// ranges, and that every expert's blocks tile its stride without overlap.
    package static func decode(
        _ data: Data,
        expectedExperts: Int,
        expertStride: UInt64,
        payloadSize: UInt64
    ) throws -> Layout {
        guard data.count >= headerBytes else {
            throw Failure.truncated(expected: headerBytes, got: data.count)
        }
        let bytes = [UInt8](data)
        guard Array(bytes[0..<4]) == magic else { throw Failure.badMagic }
        let version = read(UInt16.self, bytes, at: 4)
        guard version == currentVersion else {
            throw Failure.unsupportedVersion(version)
        }
        let scaleBits = bytes[6]
        // The GPU kernel is compiled for one scale: a 16,384-entry slot table,
        // `state & 16383`, `state >> 14`, renormalisation at 2^23. A sidecar
        // declaring any other scale would pass a range check and decode to
        // garbage, so the reader accepts exactly the scale the writer emits.
        guard scaleBits == Self.implementedScaleBits else {
            throw Failure.badScaleBits(scaleBits)
        }
        let blockCount = Int(bytes[7])
        guard (1...maximumBlockCount).contains(blockCount) else {
            throw Failure.badBlockCount(bytes[7])
        }
        let expertCount = Int(read(UInt32.self, bytes, at: 8))
        guard expertCount == expectedExperts else {
            throw Failure.badExpertCount(expected: expectedExperts, got: expertCount)
        }
        let modelsEnd = headerBytes + blockCount * 512
        let needed = modelsEnd + expertCount * blockCount * blockBytes
        guard data.count >= needed else {
            throw Failure.truncated(expected: needed, got: data.count)
        }
        let modelTotal = 1 << Int(scaleBits)
        var models: [Model] = []
        models.reserveCapacity(blockCount)
        var cursor = headerBytes
        for index in 0..<blockCount {
            var frequencies = [UInt16](repeating: 0, count: 256)
            for symbol in 0..<256 {
                frequencies[symbol] = read(UInt16.self, bytes, at: cursor + symbol * 2)
            }
            let total = frequencies.reduce(0) { $0 + Int($1) }
            guard total == modelTotal else {
                throw Failure.badModel(index: index, total: total)
            }
            models.append(Model(frequencies: frequencies))
            cursor += 512
        }
        var experts: [[Block]] = []
        var payloadRuns: [(expert: Int, start: UInt64, end: UInt64)] = []
        experts.reserveCapacity(expertCount)
        for expert in 0..<expertCount {
            var blocks: [Block] = []
            blocks.reserveCapacity(blockCount)
            // Payload offsets are absolute in the layer file; one expert's
            // blocks are contiguous relative to *its own* first block.
            var expectedOffset: UInt64 = 0
            for index in 0..<blockCount {
                let payloadOffset = read(UInt64.self, bytes, at: cursor)
                let payloadLength = read(UInt32.self, bytes, at: cursor + 8)
                let rawOffset = read(UInt32.self, bytes, at: cursor + 12)
                let symbolCount = read(UInt32.self, bytes, at: cursor + 16)
                let stride = bytes[cursor + 20]
                let role = bytes[cursor + 21]
                guard stride == 1 || stride == 2 else {
                    throw Failure.badBlock(expert: expert, block: index, reason: "stride \(stride)")
                }
                guard symbolCount > 0, payloadLength > 0 else {
                    throw Failure.badBlock(expert: expert, block: index, reason: "empty block")
                }
                guard
                    UInt64(rawOffset) + UInt64(symbolCount - 1) * UInt64(stride)
                        < expertStride
                else {
                    throw Failure.badBlock(
                        expert: expert, block: index, reason: "range exceeds the expert")
                }
                guard Int(role) < blockCount else {
                    throw Failure.badBlock(expert: expert, block: index, reason: "role \(role)")
                }
                let (payloadEnd, overflow) = payloadOffset.addingReportingOverflow(
                    UInt64(payloadLength))
                guard !overflow, payloadEnd <= payloadSize else {
                    throw Failure.badBlock(
                        expert: expert, block: index, reason: "payload exceeds the file")
                }
                if index > 0, payloadOffset != expectedOffset {
                    throw Failure.badBlock(
                        expert: expert, block: index,
                        reason: "payload starts at \(payloadOffset), expected \(expectedOffset)")
                }
                expectedOffset = payloadEnd
                blocks.append(
                    Block(
                        payloadOffset: payloadOffset, payloadLength: payloadLength,
                        rawOffset: rawOffset, symbolCount: symbolCount,
                        stride: stride, role: role))
                cursor += blockBytes
            }
            payloadRuns.append((expert, blocks[0].payloadOffset, expectedOffset))
            experts.append(blocks)
        }
        payloadRuns.sort { $0.start < $1.start }
        for (previous, current) in zip(payloadRuns, payloadRuns.dropFirst()) {
            guard current.start >= previous.end else {
                throw Failure.badBlock(
                    expert: current.expert, block: 0,
                    reason: "payload overlaps expert \(previous.expert)")
            }
        }
        return Layout(scaleBits: scaleBits, models: models, blocks: experts)
    }

    /// Streams per block, sized so one GPU thread decodes about 2 KiB.
    package static func streams(symbolCount: Int) -> Int {
        max(1, min(256, symbolCount / 2048))
    }

    /// The phase each block index belongs to, derived from the layer's tensors.
    ///
    /// The writer lays blocks out by sorted tensor name, so the runtime can
    /// recompute the same order from `layout.json`: `down` tensors are phase 1
    /// (the tail the GPU reads second), everything else phase 0.
    package static func phases(for layer: SSDAILayerV1) -> [UInt8] {
        guard let expert = layer.experts.first else { return [] }
        var phases: [UInt8] = []
        for name in expert.tensors.keys.sorted() {
            guard let tensor = expert.tensors[name] else { continue }
            let phase: UInt8 = name.hasPrefix("down") ? 1 : 0
            phases.append(phase)
            if tensor.dtype == "BF16" { phases.append(phase) }
        }
        return phases
    }

    /// Where a layer file's rANS sidecar lives: `layer_00.bin` -> `layer_00.ridx`.
    package static func sidecarName(forLayerFile file: String) -> String {
        (file as NSString).deletingPathExtension + ".ridx"
    }

    private static func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    private static func read<T: FixedWidthInteger>(
        _ type: T.Type, _ bytes: [UInt8], at offset: Int
    ) -> T {
        var value: T = 0
        withUnsafeMutableBytes(of: &value) { destination in
            destination.copyBytes(from: bytes[offset..<offset + MemoryLayout<T>.size])
        }
        return T(littleEndian: value)
    }
}
