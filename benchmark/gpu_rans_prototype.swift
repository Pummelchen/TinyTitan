import Foundation
import Metal

// GPU-decodable rANS prototype for TinyTitan expert layers.
//
// Build: swiftc -O benchmark/gpu_rans_prototype.swift -o /tmp/gpu_rans
//        /tmp/gpu_rans <raw layer_NN.bin> [experts]
//
// Measured on Ornith 4-bit layer 20 (M3):
//   ratio 0.9017-0.9020 (9.8% smaller), bit-exact on CPU and GPU,
//   decode 6.0-6.1 GB/s at 9+ experts, 2.7-4.3 GB/s for a 3-expert wave
//   (1.2-2.0 ms, against a 1.6 ms 4-bit layer window).
//
// The design is in docs/llm-compression-design.md: one static order-0 model
// per tensor role, 14-bit frequencies, 32-256 independent streams per block,
// and BF16 scales/biases split into even (mantissa) and odd (exponent) byte
// planes because the exponent plane carries 1.36 bits/byte and the mantissa
// plane 7.89 -- the interleaved pair measures 5.58 and compresses to only
// 0.70.
//
// One static order-0 byte model per tensor block, 32 interleaved streams, one
// GPU thread per stream. rANS has no context, so a stream decodes
// independently; the 32 streams of a block and all blocks run in parallel.
// Blocks are chosen by distribution:
//   - 4-bit weight tensors: 512 KiB contiguous (ratio ~0.936)
//   - BF16 scales/biases: even (mantissa) and odd (exponent) byte planes as
//     separate blocks. Interleaved, a 64 KiB pair measures 5.58 bits/byte;
//     the planes measure 7.89 and 1.36 (0.579). That is ZipMoE's bit-field
//     effect and it is what makes ~0.90 achievable.

let scaleBits = 14
let modelTotal = 1 << scaleBits
let ransL: UInt32 = 1 << 23
// streams per block are sized so each GPU thread decodes ~2 KiB
func streamsFor(_ count: Int) -> Int { max(1, min(256, count / 2048)) }
let maxSymbol = 256

struct BlockSpec {
    let name: String
    let offset: Int
    let count: Int
    let stride: Int
}

let specs: [BlockSpec] = [
    BlockSpec(name: "gate", offset: 0, count: 524_288, stride: 1),
    BlockSpec(name: "gate-mant", offset: 524_288, count: 32_768, stride: 2),
    BlockSpec(name: "gate-exp", offset: 524_289, count: 32_768, stride: 2),
    BlockSpec(name: "up", offset: 589_824, count: 524_288, stride: 1),
    BlockSpec(name: "up-mant", offset: 1_114_112, count: 32_768, stride: 2),
    BlockSpec(name: "up-exp", offset: 1_114_113, count: 32_768, stride: 2),
    BlockSpec(name: "down", offset: 1_179_648, count: 524_288, stride: 1),
    BlockSpec(name: "down-mant", offset: 1_703_936, count: 32_768, stride: 2),
    BlockSpec(name: "down-exp", offset: 1_703_937, count: 32_768, stride: 2),
]

struct EncodedBlock {
    var payload = [UInt8]()
    var freq = [UInt32](repeating: 0, count: 256)
    var cum = [UInt32](repeating: 0, count: 257)
    var streamOffsets = [Int]()
}

func normalizedFrequencies(_ counts: [Int], total: Int) -> [UInt32] {
    var freq = [UInt32](repeating: 0, count: maxSymbol)
    var present: [Int] = []
    for symbol in 0..<maxSymbol where counts[symbol] > 0 {
        present.append(symbol)
        freq[symbol] = UInt32(max(1, (counts[symbol] * modelTotal + total / 2) / total))
    }
    var sum = Int(freq.reduce(0) { $0 + UInt32($1) })
    while sum != modelTotal, let largest = present.max(by: { freq[$0] < freq[$1] }) {
        let step = sum > modelTotal ? -1 : 1
        if step < 0 && freq[largest] <= 1 { break }
        freq[largest] = UInt32(Int(freq[largest]) + step)
        sum += step
    }
    return freq
}

func encode(_ raw: [UInt8], streamCount: Int, counts: [Int]) -> EncodedBlock {
    var block = EncodedBlock()
    block.freq = normalizedFrequencies(counts, total: counts.reduce(0, +))
    for symbol in 0..<maxSymbol { block.cum[symbol + 1] = block.cum[symbol] + block.freq[symbol] }
    let freq = block.freq
    let cum = block.cum
    for stream in 0..<streamCount {
        var positions: [Int] = []
        var position = stream
        while position < raw.count { positions.append(position); position += streamCount }
        var state = UInt32(ransL)
        var emitted: [UInt8] = []
        for position in positions.reversed() {
            let symbol = Int(raw[position])
            let frequency = freq[symbol]
            while state >= UInt32((ransL >> UInt32(scaleBits)) << 8) * frequency {
                emitted.append(UInt8(state & 0xFF))
                state >>= 8
            }
            state = ((state / frequency) << UInt32(scaleBits)) + (state % frequency) + cum[symbol]
        }
        block.streamOffsets.append(block.payload.count)
        block.payload.append(contentsOf: [
            UInt8(state & 0xFF), UInt8((state >> 8) & 0xFF),
            UInt8((state >> 16) & 0xFF), UInt8((state >> 24) & 0xFF),
            8, 0, 0, 0,
        ])
        block.payload.append(contentsOf: emitted.reversed())
    }
    return block
}

func decodeCPU(_ block: EncodedBlock, symbolCount: Int, streamCount: Int) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: symbolCount)
    for stream in 0..<streamCount {
        var cursor = block.streamOffsets[stream]
        var state =
            UInt32(block.payload[cursor]) | UInt32(block.payload[cursor + 1]) << 8
            | UInt32(block.payload[cursor + 2]) << 16 | UInt32(block.payload[cursor + 3]) << 24
        var data = cursor + 8
        var position = stream
        while position < symbolCount {
            while state < ransL {
                state = (state << 8) | UInt32(block.payload[data])
                data += 1
            }
            let slot = Int(state & UInt32(modelTotal - 1))
            var low = 0
            var high = maxSymbol - 1
            while low < high {
                let mid = (low + high + 1) / 2
                if Int(block.cum[mid]) <= slot { low = mid } else { high = mid - 1 }
            }
            out[position] = UInt8(low)
            state = block.freq[low] * (state >> UInt32(scaleBits)) + UInt32(slot) - block.cum[low]
            position += streamCount
        }
    }
    return out
}

let shader = """
    #include <metal_stdlib>
    using namespace metal;

    struct StreamDesc {
        uint payloadOffset;
        uint tableOffset;
        uint outputOffset;
        uint stride;
        uint symbolCount;
        uint streamIndex;
        uint streamCount;
        uint slotOffset;
    };

    kernel void rans_decode(
        device const uchar *payload [[buffer(0)]],
        device const uint *tables [[buffer(1)]],
        constant StreamDesc *streams [[buffer(2)]],
        device uchar *output [[buffer(3)]],
        device const uint2 *slotTables [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
    {
        StreamDesc d = streams[gid];
        uint base = d.payloadOffset;
        uint state = uint(payload[base]) | (uint(payload[base + 1u]) << 8)
            | (uint(payload[base + 2u]) << 16) | (uint(payload[base + 3u]) << 24);
        uint data = base + 8u;
        device const uint *cum = tables + d.tableOffset;
        device const uint *freq = cum + 257u;
        device const uint2 *slots = slotTables + d.slotOffset;
        for (uint position = d.streamIndex; position < d.symbolCount; position += d.streamCount) {
            while (state < (1u << 23)) {
                state = (state << 8) | uint(payload[data]);
                data += 1u;
            }
            uint slot = state & 16383u;
            uint2 info = slots[slot];
            uint symbol = info.x & 0xFFu;
            output[d.outputOffset + position * d.stride] = uchar(symbol);
            state = (info.x >> 8u) * (state >> 14u) + slot - info.y;
        }
    }
    """

let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write(Data("usage: gpu_rans_proto <layer.bin> [experts]\n".utf8))
    exit(2)
}
let layer = try Data(contentsOf: URL(fileURLWithPath: args[1]), options: .mappedIfSafe)
let stride = 1_769_472
let expertCount = args.count > 2 ? Int(args[2])! : 3
let layerPointer = layer.withUnsafeBytes { $0.baseAddress! }

// Pass 1: one model per tensor role, over every expert in the layer.
var roleCounts = [[Int]](repeating: [Int](repeating: 0, count: maxSymbol), count: specs.count)
for expert in 0..<expertCount {
    for (role, spec) in specs.enumerated() {
        let base = expert * stride + spec.offset
        for offset in 0..<spec.count {
            roleCounts[role][
                Int(
                    layerPointer.load(
                        fromByteOffset: base + offset * spec.stride, as: UInt8.self))] += 1
        }
    }
}

var encoded: [EncodedBlock] = []
var rawTotal = 0
let encodeStarted = Date()
for expert in 0..<expertCount {
    for (role, spec) in specs.enumerated() {
        let base = expert * stride + spec.offset
        let raw = (0..<spec.count).map { offset in
            layerPointer.advanced(by: base + offset * spec.stride).load(as: UInt8.self)
        }
        encoded.append(encode(raw, streamCount: streamsFor(spec.count), counts: roleCounts[role]))
        rawTotal += raw.count
    }
}
let encodeSeconds = Date().timeIntervalSince(encodeStarted)
var encodedTotal = 0
for (index, block) in encoded.enumerated() {
    encodedTotal += block.payload.count + 512
    if index < specs.count {
        print(
            String(
                format: "  %@ raw=%6d coded=%6d ratio=%.3f",
                specs[index].name, specs[index].count, block.payload.count + 512,
                Double(block.payload.count + 512) / Double(specs[index].count)))
    }
}
print(
    String(
        format: "ratio=%.4f (save %.2f%%) raw=%.1f MB encode=%.2fs (%.1f MB/s)",
        Double(encodedTotal) / Double(rawTotal),
        100 * (1 - Double(encodedTotal) / Double(rawTotal)),
        Double(rawTotal) / 1e6, encodeSeconds, Double(rawTotal) / 1e6 / encodeSeconds))

var cpuMismatch = 0
let cpuStarted = Date()
for (index, block) in encoded.enumerated() {
    let spec = specs[index % specs.count]
    let base = (index / specs.count) * stride + spec.offset
    let want = (0..<spec.count).map { offset in
        layerPointer.advanced(by: base + offset * spec.stride).load(as: UInt8.self)
    }
    if decodeCPU(block, symbolCount: spec.count, streamCount: streamsFor(spec.count)) != want {
        cpuMismatch += 1
    }
}
print(
    String(
        format: "cpu: %d/%d blocks mismatched, %.1f MB/s",
        cpuMismatch, encoded.count,
        Double(rawTotal) / 1e6 / Date().timeIntervalSince(cpuStarted)))

guard let device = MTLCreateSystemDefaultDevice(), let commandQueue = device.makeCommandQueue()
else { exit(1) }
let pipeline = try device.makeComputePipelineState(
    function: device.makeLibrary(source: shader, options: nil).makeFunction(name: "rans_decode")!)

var payload: [UInt8] = []
var streamBases: [UInt32] = []
for block in encoded {
    streamBases.append(UInt32(payload.count))
    payload.append(contentsOf: block.payload)
}
var tables: [UInt32] = []
var tableOffsets: [UInt32] = []
for (role, _) in specs.enumerated() {
    tableOffsets.append(UInt32(tables.count))
    let block = encoded[role]
    tables.append(contentsOf: block.cum)
    tables.append(contentsOf: block.freq)
}
// One slot->(symbol, frequency, cumulative) table per role, eight bytes a slot.
var slotInfo: [UInt32] = []
var slotOffsets: [UInt32] = []
for (role, _) in specs.enumerated() {
    let block = encoded[role]
    slotOffsets.append(UInt32(slotInfo.count / 2))
    var symbol = 0
    for slot in 0..<modelTotal {
        while symbol < 255 && Int(block.cum[symbol + 1]) <= slot { symbol += 1 }
        slotInfo.append(UInt32(symbol) | (block.freq[symbol] << 8))
        slotInfo.append(block.cum[symbol])
    }
}
var streamDesc: [UInt32] = []
var streamTotal = 0
for (index, block) in encoded.enumerated() {
    let spec = specs[index % specs.count]
    let expert = index / specs.count
    let streamCount = streamsFor(spec.count)
    let outputBase = UInt32(expert * stride + spec.offset)
    for stream in 0..<streamCount {
        streamDesc.append(UInt32(Int(streamBases[index]) + block.streamOffsets[stream]))
        streamDesc.append(tableOffsets[index % specs.count])
        streamDesc.append(outputBase)
        streamDesc.append(UInt32(spec.stride))
        streamDesc.append(UInt32(spec.count))
        streamDesc.append(UInt32(stream))
        streamDesc.append(UInt32(streamCount))
        streamDesc.append(slotOffsets[index % specs.count])
        streamTotal += 1
    }
}
var initialOutput = [UInt8](repeating: 0, count: expertCount * stride)

let payloadBuffer = device.makeBuffer(
    bytes: payload, length: payload.count, options: .storageModeShared)!
let tableBuffer = device.makeBuffer(
    bytes: tables, length: tables.count * 4, options: .storageModeShared)!
let descBuffer = device.makeBuffer(
    bytes: streamDesc, length: streamDesc.count * 4, options: .storageModeShared)!
let slotBuffer = device.makeBuffer(
    bytes: slotInfo, length: slotInfo.count * 4, options: .storageModeShared)!
let outputBuffer = device.makeBuffer(
    bytes: initialOutput, length: initialOutput.count, options: .storageModeShared)!

func runGPU() -> Double {
    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(payloadBuffer, offset: 0, index: 0)
    encoder.setBuffer(tableBuffer, offset: 0, index: 1)
    encoder.setBuffer(descBuffer, offset: 0, index: 2)
    encoder.setBuffer(outputBuffer, offset: 0, index: 3)
    encoder.setBuffer(slotBuffer, offset: 0, index: 4)
    encoder.dispatchThreads(
        MTLSize(width: streamTotal, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
}

_ = runGPU()
let gpuSeconds = (0..<5).map { _ in runGPU() }.min() ?? 0
let outputPointer = outputBuffer.contents().bindMemory(
    to: UInt8.self, capacity: initialOutput.count)
var gpuMismatch = 0
for index in 0..<(expertCount * stride) {
    if outputPointer[index] != layerPointer.load(fromByteOffset: index, as: UInt8.self) {
        gpuMismatch += 1
    }
}
print(
    String(
        format: "gpu: mismatch=%d/%d bytes, decode=%.1f GB/s (%.0f us for %.1f MB)",
        gpuMismatch, expertCount * stride, Double(rawTotal) / gpuSeconds / 1e9,
        gpuSeconds * 1e6, Double(rawTotal) / 1e6))
