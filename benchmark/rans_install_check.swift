import Foundation
import Metal

// Decode a converted rANS install's layer on the GPU and compare it with the
// raw layer it came from, byte for byte.
//
//   rans_install_check <install dir> <raw install dir> <layer> [experts]
//
// Reads packed_experts/layer_NN.ridx (models + block map), the layer payload,
// and the raw layer; builds the same stream descriptors the runtime will, runs
// the Metal kernel, and reports mismatch, decode time and the per-wave latency.

let args = CommandLine.arguments
guard args.count >= 4 else {
    FileHandle.standardError.write(
        Data("usage: rans_install_check <install> <raw install> <layer> [experts]\n".utf8))
    exit(2)
}
let install = args[1]
let rawInstall = args[2]
let layerIndex = Int(args[3])!
let expertCount = args.count > 4 ? Int(args[4])! : 3
let layerName = String(format: "layer_%02d", layerIndex)
let payloadPath = "\(install)/packed_experts/\(layerName).bin"
let indexPath = "\(install)/packed_experts/\(layerName).ridx"
let rawPath = "\(rawInstall)/packed_experts/\(layerName).bin"
let stride = 1_769_472

func now() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }

// MARK: - sidecar

struct Model { let frequencies: [UInt16] }
struct Block {
    let payloadOffset: Int
    let payloadLength: Int
    let rawOffset: Int
    let symbolCount: Int
    let stride: Int
    let role: Int
}

let indexData = try Data(contentsOf: URL(fileURLWithPath: indexPath))
let bytes = [UInt8](indexData)
func u16(_ at: Int) -> UInt16 { UInt16(bytes[at]) | UInt16(bytes[at + 1]) << 8 }
func u32(_ at: Int) -> UInt32 {
    UInt32(bytes[at]) | UInt32(bytes[at + 1]) << 8 | UInt32(bytes[at + 2]) << 16
        | UInt32(bytes[at + 3]) << 24
}
func u64(_ at: Int) -> UInt64 {
    var value: UInt64 = 0
    for index in 0..<8 { value |= UInt64(bytes[at + index]) << (8 * index) }
    return value
}
precondition(Array(bytes[0..<4]) == Array("TTRX".utf8), "not a TTRX sidecar")
let scaleBits = Int(bytes[6])
let blockCount = Int(bytes[7])
let expertTotal = Int(u32(8))
let modelTotal = 1 << scaleBits
var models: [Model] = []
var cursor = 16
for _ in 0..<blockCount {
    var f = [UInt16](repeating: 0, count: 256)
    for symbol in 0..<256 { f[symbol] = u16(cursor + symbol * 2) }
    precondition(f.reduce(0) { $0 + Int($1) } == modelTotal, "bad model")
    models.append(Model(frequencies: f))
    cursor += 512
}
var experts: [[Block]] = []
for _ in 0..<expertTotal {
    var blocks: [Block] = []
    for _ in 0..<blockCount {
        blocks.append(
            Block(
                payloadOffset: Int(u64(cursor)),
                payloadLength: Int(u32(cursor + 8)),
                rawOffset: Int(u32(cursor + 12)),
                symbolCount: Int(u32(cursor + 16)),
                stride: Int(bytes[cursor + 20]),
                role: Int(bytes[cursor + 21])))
        cursor += 24
    }
    experts.append(blocks)
}
print("model: scaleBits=\(scaleBits) blocks=\(blockCount) experts=\(expertTotal)")

// MARK: - kernel

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
        uint state = uint(payload[d.payloadOffset]) | (uint(payload[d.payloadOffset + 1u]) << 8)
            | (uint(payload[d.payloadOffset + 2u]) << 16) | (uint(payload[d.payloadOffset + 3u]) << 24);
        uint data = d.payloadOffset + 8u;
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

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    exit(1)
}
let pipeline = try device.makeComputePipelineState(
    function: device.makeLibrary(source: shader, options: nil).makeFunction(name: "rans_decode")!)

// Tables per role: cum[257] + freq[256], and the packed slot table.
var tables: [UInt32] = []
var tableOffsets: [UInt32] = []
var slotInfo: [UInt32] = []
var slotOffsets: [UInt32] = []
for model in models {
    var cum = [UInt32](repeating: 0, count: 257)
    for symbol in 0..<256 { cum[symbol + 1] = cum[symbol] + UInt32(model.frequencies[symbol]) }
    tableOffsets.append(UInt32(tables.count))
    tables.append(contentsOf: cum)
    tables.append(contentsOf: model.frequencies.map { UInt32($0) })
    slotOffsets.append(UInt32(slotInfo.count / 2))
    var symbol = 0
    for slot in 0..<16384 {
        while symbol < 255 && Int(cum[symbol + 1]) <= slot { symbol += 1 }
        slotInfo.append(UInt32(symbol) | (UInt32(model.frequencies[symbol]) << 8))
        slotInfo.append(cum[symbol])
    }
}

let payload = try Data(contentsOf: URL(fileURLWithPath: payloadPath), options: .mappedIfSafe)
let payloadBytes = [UInt8](payload)
func payloadLength(_ at: Int) -> Int {
    Int(payloadBytes[at]) | Int(payloadBytes[at + 1]) << 8
}
let raw = try Data(contentsOf: URL(fileURLWithPath: rawPath), options: .mappedIfSafe)

// One wave: `expertCount` experts, descriptors for every stream of every block.
var descs: [UInt32] = []
var streamTotal = 0
for expert in 0..<expertCount {
    for block in experts[expert] {
        let streams = max(1, min(256, block.symbolCount / 2048))
        let outputBase = UInt32(expert * stride + block.rawOffset)
        // The block payload starts with a u16 length table; stream i's bytes
        // follow all the lengths, in order.
        var streamBase = block.payloadOffset + streams * 2
        for stream in 0..<streams {
            let length = payloadLength(block.payloadOffset + stream * 2)
            descs.append(UInt32(streamBase))
            streamBase += length
            descs.append(tableOffsets[block.role])
            descs.append(outputBase)
            descs.append(UInt32(block.stride))
            descs.append(UInt32(block.symbolCount))
            descs.append(UInt32(stream))
            descs.append(UInt32(streams))
            descs.append(slotOffsets[block.role])
            streamTotal += 1
        }
    }
}

let payloadBuffer = device.makeBuffer(
    bytes: [UInt8](payload), length: payload.count, options: .storageModeShared)!
let tableBuffer = device.makeBuffer(
    bytes: tables, length: tables.count * 4, options: .storageModeShared)!
let descBuffer = device.makeBuffer(
    bytes: descs, length: descs.count * 4, options: .storageModeShared)!
let slotBuffer = device.makeBuffer(
    bytes: slotInfo, length: slotInfo.count * 4, options: .storageModeShared)!
var output = [UInt8](repeating: 0, count: expertCount * stride)
let outputBuffer = device.makeBuffer(
    bytes: output, length: output.count, options: .storageModeShared)!

func run() -> Double {
    let cb = queue.makeCommandBuffer()!
    let encoder = cb.makeComputeCommandEncoder()!
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
    cb.commit()
    cb.waitUntilCompleted()
    return cb.gpuEndTime - cb.gpuStartTime
}

_ = run()
let seconds = (0..<5).map { _ in run() }.min() ?? 0
let out = outputBuffer.contents().bindMemory(to: UInt8.self, capacity: output.count)
var mismatch = 0
for index in 0..<output.count {
    if out[index] != raw[index] { mismatch += 1 }
}
let totalBytes = expertCount * stride
print(
    String(
        format: "gpu: mismatch=%d/%d bytes, decode=%.1f GB/s (%.0f us for %.1f MB)",
        mismatch, totalBytes, Double(totalBytes) / seconds / 1e9, seconds * 1e6,
        Double(totalBytes) / 1e6))
print(
    String(
        format: "wave: %d experts, %d streams, %d blocks",
        expertCount, streamTotal, expertCount * blockCount))
