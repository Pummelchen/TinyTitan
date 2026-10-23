import Foundation
import TinyTitanFormat

/// One block of an expert under the rANS codec.
public struct RANSBlockSpec: Sendable, Equatable {
    public let rawOffset: UInt32
    public let symbolCount: UInt32
    public let stride: UInt8

    public init(rawOffset: UInt32, symbolCount: UInt32, stride: UInt8) {
        self.rawOffset = rawOffset
        self.symbolCount = symbolCount
        self.stride = stride
    }
}

public struct RANSExpertLayerGeometry: Sendable, Equatable {
    public let expertStride: UInt64
    public let expertsPerLayer: Int
    public let blocks: [RANSBlockSpec]

    public init(expertStride: UInt64, expertsPerLayer: Int, blocks: [RANSBlockSpec]) {
        self.expertStride = expertStride
        self.expertsPerLayer = expertsPerLayer
        self.blocks = blocks
    }

    /// The block layout a layer's own `layout.json` describes.
    ///
    /// Weight tensors (`U32`) are contiguous blocks. Every `BF16` tensor — the
    /// scales and biases — becomes two byte-plane blocks: its even bytes
    /// (mantissa) and odd bytes (sign|exponent). Interleaved, a 64 KiB pair
    /// measures 5.58 bits/byte; the planes measure 7.89 and 1.36.
    package static func blocks(
        for layer: SSDAILayerV1
    ) -> [RANSBlockSpec] {
        guard let expert = layer.experts.first else { return [] }
        var specs: [RANSBlockSpec] = []
        for name in expert.tensors.keys.sorted() {
            guard let tensor = expert.tensors[name] else { continue }
            if tensor.dtype == "BF16" {
                let count = UInt32(tensor.size / 2)
                specs.append(
                    RANSBlockSpec(rawOffset: UInt32(tensor.offset), symbolCount: count, stride: 2))
                specs.append(
                    RANSBlockSpec(
                        rawOffset: UInt32(tensor.offset + 1), symbolCount: count, stride: 2))
            } else {
                specs.append(
                    RANSBlockSpec(
                        rawOffset: UInt32(tensor.offset), symbolCount: UInt32(tensor.size),
                        stride: 1))
            }
        }
        return specs
    }
}

public struct RANSExpertLayerResult: Sendable, Equatable {
    public let compressedBytes: UInt64
    public let indexBytes: UInt64
    public let sha256: String
    public let indexSha256: String
    public let blockCount: Int
}

/// Encodes one raw expert layer with the GPU-decoded rANS codec.
///
/// Two passes: the first counts every block's byte distribution over all
/// experts, so one model is shared by the whole layer (a per-expert model would
/// pay 512 bytes a block and, for the near-uniform mantissa planes, expand);
/// the second encodes and writes the payload, then the sidecar.
public enum RANSExpertLayerWriter {
    /// Written into the sidecar header; `ExpertRANSIndex.decode` refuses any
    /// other value, so the writer's scale and the decoder's are one constant.
    package static let scaleBits: UInt8 = ExpertRANSIndex.implementedScaleBits
    package static let modelTotal = 1 << Int(scaleBits)

    public static func write(
        sourcePath: String,
        destinationPath: String,
        indexPath: String,
        geometry: RANSExpertLayerGeometry
    ) throws -> RANSExpertLayerResult {
        guard geometry.expertsPerLayer > 0, geometry.expertStride > 0,
            !geometry.blocks.isEmpty
        else {
            throw RepackError.configurationInvalid(detail: "rANS layer has no blocks")
        }
        let rawSize = UInt64(geometry.expertsPerLayer) * geometry.expertStride
        let attributes = try FileManager.default.attributesOfItem(atPath: sourcePath)
        guard (attributes[.size] as? NSNumber)?.uint64Value == rawSize else {
            throw RepackError.configurationInvalid(
                detail: "\(sourcePath) is not \(rawSize) bytes")
        }
        let sourceFD = try Posix.openRead(sourcePath)
        defer { close(sourceFD) }
        let destinationFD = try Posix.openCreateRW(destinationPath)
        defer { close(destinationFD) }
        let buffer = UnsafeMutableRawPointer.allocate(
            byteCount: Int(geometry.expertStride), alignment: 4096)
        defer { buffer.deallocate() }

        // Pass 1: distributions per block index, over every expert.
        var counts = [[Int]](
            repeating: [Int](repeating: 0, count: 256), count: geometry.blocks.count)
        for expert in 0..<geometry.expertsPerLayer {
            try Posix.preadAll(
                fd: sourceFD, path: sourcePath, buf: buffer,
                count: Int(geometry.expertStride),
                offset: UInt64(expert) * geometry.expertStride)
            let base = UnsafeRawPointer(buffer)
            for (role, block) in geometry.blocks.enumerated() {
                var index = 0
                while index < Int(block.symbolCount) {
                    let byte = base.load(
                        fromByteOffset: Int(block.rawOffset) + index * Int(block.stride),
                        as: UInt8.self)
                    counts[role][Int(byte)] += 1
                    index += 1
                }
            }
        }
        let models = counts.map { frequencies in
            ExpertRANSIndex.Model(frequencies: normalized(frequencies, total: modelTotal))
        }

        // Pass 2: encode. Block order is the model order, and one expert's
        // blocks are written back to back.
        var hasher = Sha256Stream()
        var encodedBlocks: [[ExpertRANSIndex.Block]] = []
        encodedBlocks.reserveCapacity(geometry.expertsPerLayer)
        var written: UInt64 = 0
        for expert in 0..<geometry.expertsPerLayer {
            try Posix.preadAll(
                fd: sourceFD, path: sourcePath, buf: buffer,
                count: Int(geometry.expertStride),
                offset: UInt64(expert) * geometry.expertStride)
            let base = UnsafeRawPointer(buffer)
            var blocks: [ExpertRANSIndex.Block] = []
            blocks.reserveCapacity(geometry.blocks.count)
            for (role, spec) in geometry.blocks.enumerated() {
                let symbols = (0..<Int(spec.symbolCount)).map { index in
                    base.load(
                        fromByteOffset: Int(spec.rawOffset) + index * Int(spec.stride),
                        as: UInt8.self)
                }
                let payload = try encodeBlock(
                    symbols, model: models[role], streams: streams(count: symbols.count))
                try payload.withUnsafeBytes { raw in
                    guard let address = raw.baseAddress else { return }
                    try Posix.pwriteAll(
                        fd: destinationFD, path: destinationPath, buf: address,
                        count: raw.count, offset: written)
                }
                hasher.update(payload.withUnsafeBytes { $0 })
                blocks.append(
                    ExpertRANSIndex.Block(
                        payloadOffset: written,
                        payloadLength: UInt32(payload.count),
                        rawOffset: spec.rawOffset,
                        symbolCount: spec.symbolCount,
                        stride: spec.stride,
                        role: UInt8(role)))
                written += UInt64(payload.count)
            }
            encodedBlocks.append(blocks)
        }
        try Posix.fsync(destinationFD, path: destinationPath)
        let indexData = ExpertRANSIndex.encode(
            ExpertRANSIndex.Layout(
                scaleBits: scaleBits, models: models, blocks: encodedBlocks))
        try Posix.atomicWrite(
            indexData, to: indexPath, durableIn: (indexPath as NSString).deletingLastPathComponent)
        return RANSExpertLayerResult(
            compressedBytes: written,
            indexBytes: UInt64(indexData.count),
            sha256: hasher.finalizeHexString(),
            indexSha256: sha256(of: indexData),
            blockCount: geometry.blocks.count)
    }

    /// Streams per block, sized so one GPU thread decodes about 2 KiB.
    package static func streams(count: Int) -> Int {
        ExpertRANSIndex.streams(symbolCount: count)
    }

    package static func normalized(_ counts: [Int], total: Int) -> [UInt16] {
        var frequencies = [UInt16](repeating: 0, count: 256)
        var present: [Int] = []
        for symbol in 0..<256 where counts[symbol] > 0 {
            present.append(symbol)
            frequencies[symbol] = UInt16(
                max(1, (counts[symbol] * total + counts.reduce(0, +) / 2) / counts.reduce(0, +)))
        }
        var sum = frequencies.reduce(0) { $0 + Int($1) }
        while sum != total, let largest = present.max(by: { frequencies[$0] < frequencies[$1] }) {
            let step = sum > total ? -1 : 1
            if step < 0 && frequencies[largest] <= 1 { break }
            frequencies[largest] = UInt16(Int(frequencies[largest]) + step)
            sum += step
        }
        return frequencies
    }

    package static func encodeBlock(
        _ symbols: [UInt8], model: ExpertRANSIndex.Model, streams: Int
    ) throws -> Data {
        guard symbols.count > 0, symbols.count < Int(UInt32.max) else {
            throw RepackError.configurationInvalid(detail: "rANS block has no symbols")
        }
        var cumulative = [UInt32](repeating: 0, count: 257)
        for symbol in 0..<256 {
            cumulative[symbol + 1] = cumulative[symbol] + UInt32(model.frequencies[symbol])
        }
        let modelTotal = Int(cumulative[256])
        guard modelTotal > 0 else {
            throw RepackError.configurationInvalid(detail: "rANS model is empty")
        }
        var encodedStreams: [Data] = []
        for stream in 0..<streams {
            var positions: [Int] = []
            var position = stream
            while position < symbols.count { positions.append(position); position += streams }
            var state = UInt32(1 << 23)
            var emitted: [UInt8] = []
            for position in positions.reversed() {
                let symbol = Int(symbols[position])
                let frequency = UInt32(model.frequencies[symbol])
                while state >= UInt32((UInt32(1 << 23) >> UInt32(scaleBits)) << 8) * frequency {
                    emitted.append(UInt8(state & 0xFF))
                    state >>= 8
                }
                state =
                    ((state / frequency) << UInt32(scaleBits))
                    + (state % frequency) + cumulative[symbol]
            }
            var streamData = Data()
            streamData.append(contentsOf: [
                UInt8(state & 0xFF), UInt8((state >> 8) & 0xFF),
                UInt8((state >> 16) & 0xFF), UInt8((state >> 24) & 0xFF),
                8, 0, 0, 0,
            ])
            streamData.append(contentsOf: emitted.reversed())
            encodedStreams.append(streamData)
        }
        // Streams have variable lengths, so the block payload starts with a
        // u16 length table; the host derives each stream's offset from it.
        var payload = Data()
        for data in encodedStreams {
            guard data.count <= Int(UInt16.max) else {
                throw RepackError.configurationInvalid(
                    detail: "rANS stream of \(data.count) bytes exceeds u16")
            }
            var length = UInt16(data.count).littleEndian
            withUnsafeBytes(of: &length) { payload.append(contentsOf: $0) }
        }
        for data in encodedStreams { payload.append(data) }
        return payload
    }

    private static func sha256(of data: Data) -> String {
        var hasher = Sha256Stream()
        data.withUnsafeBytes { hasher.update($0) }
        return hasher.finalizeHexString()
    }
}
