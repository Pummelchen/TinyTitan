import Compression
import Darwin
import Foundation

// How fast does this machine decode the real expert frames, at 1..8 workers?
//
// This is the question the compressed-expert A/B could not answer: the install
// measured -42% decode, but its decode path serialised read and decode and ran
// at effectively one core. `Compression`'s LZFSE is a single-stream codec, so
// the only parallelism available is across frames -- which is exactly how the
// format stores them (two per split expert).
//
// Reads a converted layer's `.idx`, maps its `.bin`, and decodes every frame on
// N workers. Reports output GB/s (what the decoder must produce), input GB/s
// (what the SSD may skip), and the latency of one layer's worth of experts.
//
// Usage: swiftc -O benchmark/expert_decode_bench.swift -o /tmp/expert_decode_bench
//        /tmp/expert_decode_bench <layer.bin> <layer.idx> [phase1] [stride]

struct Frame {
    let offset: Int
    let length: Int
    let output: Int
}

func now() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }

func readFrames(_ path: String, phase1: Int, stride: Int) throws -> [Frame] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    guard data.count >= 12 else { throw CocoaError(.fileReadCorruptFile) }
    let bytes = [UInt8](data)
    func u32(_ at: Int) -> UInt32 {
        UInt32(bytes[at]) | UInt32(bytes[at + 1]) << 8 | UInt32(bytes[at + 2]) << 16
            | UInt32(bytes[at + 3]) << 24
    }
    func u64(_ at: Int) -> UInt64 {
        var value: UInt64 = 0
        for index in 0..<8 { value |= UInt64(bytes[at + index]) << (8 * index) }
        return value
    }
    let frames = Int(bytes[6])
    let count = Int(u32(8))
    var out: [Frame] = []
    out.reserveCapacity(count * frames)
    for expert in 0..<count {
        for frame in 0..<frames {
            let at = 12 + (expert * frames + frame) * 12
            let output: Int
            if frames == 2 {
                output = frame == 0 ? phase1 : stride - phase1
            } else {
                output = stride
            }
            out.append(
                Frame(offset: Int(u64(at)), length: Int(u32(at + 8)), output: output))
        }
    }
    return out
}

func decode(_ frame: Frame, src: UnsafeRawPointer, dst: UnsafeMutableRawPointer) -> Int {
    compression_decode_buffer(
        dst.assumingMemoryBound(to: UInt8.self), frame.output,
        src.advanced(by: frame.offset).assumingMemoryBound(to: UInt8.self), frame.length,
        nil, COMPRESSION_LZFSE)
}

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write(
        Data("usage: expert_decode_bench <layer.bin> <layer.idx> [phase1] [stride]\n".utf8))
    exit(2)
}
let phase1 = args.count > 3 ? Int(args[3]) ?? 1_179_648 : 1_179_648
let stride = args.count > 4 ? Int(args[4]) ?? 1_769_472 : 1_769_472
let frames = try readFrames(args[2], phase1: phase1, stride: stride)
let input = try Data(contentsOf: URL(fileURLWithPath: args[1]), options: .mappedIfSafe)
let rawMB = frames.reduce(0) { $0 + $1.output } / 1_000_000
print("frames=\(frames.count) packed=\(input.count / 1_000_000) MB raw=\(rawMB) MB")
let maxOutput = frames.map(\.output).max() ?? 0

func pass(workers: Int, reps: Int) -> (outGBps: Double, inGBps: Double, ms: Double, ok: Int) {
    let chunk = (frames.count + workers - 1) / workers
    var totalOut = 0
    var ok = 0
    var best = Double.infinity
    input.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        guard let base = raw.baseAddress else { return }
        for _ in 0..<reps {
            totalOut = 0
            ok = 0
            let started = now()
            DispatchQueue.concurrentPerform(iterations: workers) { worker in
                let destination = UnsafeMutableRawPointer.allocate(
                    byteCount: maxOutput, alignment: 4096)
                defer { destination.deallocate() }
                let start = worker * chunk
                let end = min(start + chunk, frames.count)
                guard start < end else { return }
                for index in start..<end {
                    let written = decode(frames[index], src: base, dst: destination)
                    if written == frames[index].output {
                        ok += 1
                        totalOut += written
                    }
                }
            }
            best = min(best, Double(now() - started) / 1e9)
        }
    }
    return (
        Double(totalOut) / best / 1e9,
        Double(input.count) / best / 1e9,
        best * 1000,
        ok
    )
}

for workers in [1, 2, 4, 8] {
    let result = pass(workers: workers, reps: 3)
    print(
        String(
            format: "workers=%d  out=%.2f GB/s  in=%.2f GB/s  pass=%.0f ms  frames_ok=%d/%d",
            workers, result.outGBps, result.inGBps, result.ms, result.ok, frames.count))
}

// One layer's worth: `experts` experts, i.e. 2x that many frames, which is the
// unit a layer actually needs before its MoE GEMM can run.
func layerLatency(workers: Int, experts: Int) -> Double {
    let picks = (0..<(experts * 2)).map { frames[($0 * 37) % frames.count] }
    let chunk = (picks.count + workers - 1) / workers
    var best = Double.infinity
    input.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        guard let base = raw.baseAddress else { return }
        for _ in 0..<10 {
            let started = now()
            DispatchQueue.concurrentPerform(iterations: workers) { worker in
                let destination = UnsafeMutableRawPointer.allocate(
                    byteCount: maxOutput, alignment: 4096)
                defer { destination.deallocate() }
                let start = worker * chunk
                let end = min(start + chunk, picks.count)
                guard start < end else { return }
                for index in start..<end {
                    _ = decode(picks[index], src: base, dst: destination)
                }
            }
            best = min(best, Double(now() - started) / 1e6)
        }
    }
    return best
}

// Can the SSD read and the decode overlap? The design's ceiling assumes they
// can: read the packed bytes while decoding the previous wave. This measures the
// three cases on the same data -- read alone (F_NOCACHE), decode alone (8
// workers), and both at once -- so a policy can be judged on measurement rather
// than on the hope that two resources leave each other alone.
func sequentialRead(path: String, chunk: Int) -> (seconds: Double, bytes: Int) {
    let fd = open(path, O_RDONLY)
    _ = fcntl(fd, F_NOCACHE, 1)
    let buffer = UnsafeMutableRawPointer.allocate(byteCount: chunk, alignment: 4096)
    defer {
        buffer.deallocate()
        close(fd)
    }
    let started = now()
    var total = 0
    while true {
        let got = pread(fd, buffer, chunk, off_t(total))
        if got <= 0 { break }
        total += got
    }
    return (Double(now() - started) / 1e9, total)
}

func decodePass(workers: Int) -> Double {
    let chunk = (frames.count + workers - 1) / workers
    let started = now()
    input.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        guard let base = raw.baseAddress else { return }
        DispatchQueue.concurrentPerform(iterations: workers) { worker in
            let destination = UnsafeMutableRawPointer.allocate(
                byteCount: maxOutput, alignment: 4096)
            defer { destination.deallocate() }
            let start = worker * chunk
            let end = min(start + chunk, frames.count)
            guard start < end else { return }
            for index in start..<end {
                _ = decode(frames[index], src: base, dst: destination)
            }
        }
    }
    return Double(now() - started) / 1e6
}

var readBest = Double.infinity
var readBytes = 0
for _ in 0..<3 {
    let result = sequentialRead(path: args[1], chunk: 4 << 20)
    readBytes = result.bytes
    readBest = min(readBest, result.seconds)
}
let decodeBest = (0..<3).map { _ in decodePass(workers: 8) }.min() ?? 0

var overlapBest = Double.infinity
var overlapRead = 0.0
for _ in 0..<3 {
    let started = now()
    let group = DispatchGroup()
    group.enter()
    var seconds = 0.0
    DispatchQueue.global(qos: .userInitiated).async {
        seconds = sequentialRead(path: args[1], chunk: 4 << 20).seconds
        group.leave()
    }
    _ = decodePass(workers: 8)
    group.wait()
    overlapBest = min(overlapBest, Double(now() - started) / 1e6)
    overlapRead = seconds
}
print(
    String(
        format:
            "read-only %.0f ms (%.2f GB/s)  decode-only %.0f ms  both-at-once %.0f ms (read part %.0f ms)  bytes=%d MB",
        readBest * 1000, Double(readBytes) / readBest / 1e9, decodeBest, overlapBest,
        overlapRead * 1000, readBytes / 1_000_000))

for experts in [3, 8, 10] {
    for workers in [4, 8] {
        print(
            String(
                format: "layer-decode experts=%d workers=%d  %.2f ms",
                experts, workers, layerLatency(workers: workers, experts: experts)))
    }
}
