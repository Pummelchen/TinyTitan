import Foundation
import Metal
import Synchronization
import Testing
import TinyTitanValidationSupport

@testable import Legacy6xRuntime
@testable import TinyTitanCore

/// How expert-read throughput scales with concurrency: three reads at a time
/// (today) against all thirty of a layer's matrices at once.
@Suite struct GGUFReadScalingTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func scalesTheReads() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let ranges: [GGUFTensorRange] = try (0..<10).flatMap { expert -> [GGUFTensorRange] in
            let weights = try GGUFExpertLayout.weights(layer: 0, expert: expert, in: model)
            return [weights.gate, weights.up, weights.down]
        }
        let bytes = ranges.reduce(0) { $0 + Int($1.byteCount) }
        let url: URL = model.shards[ranges[0].shardIndex].url

        // The destination is a parameter, not a capture: a `@Sendable` function
        // may only capture sendable values, and one 1 MiB buffer shared by every
        // reader would be a data race (all of them `pread` into it at once). Each
        // worker makes its own buffer below.
        @Sendable func read(_ range: GGUFTensorRange, into buffer: MTLBuffer) {
            let fd = open(url.path, O_RDONLY | O_CLOEXEC)
            guard fd >= 0 else { return }
            defer { close(fd) }
            var total = 0
            while total < Int(range.byteCount) {
                let count = min(Int(range.byteCount) - total, 1 << 20)
                let got = pread(
                    fd, buffer.contents(), count, off_t(range.fileOffset) + off_t(total))
                if got <= 0 { break }
                total += got
            }
        }

        func measure(concurrency: Int, rounds: Int) -> Double {
            var best = Double.infinity
            for _ in 0..<rounds {
                let start = DispatchTime.now().uptimeNanoseconds
                let claimed = Mutex(0)
                DispatchQueue.concurrentPerform(iterations: concurrency) { _ in
                    guard
                        let buffer = context.device.makeBuffer(
                            length: 1 << 20, options: .storageModeShared)
                    else { return }
                    while true {
                        let index = claimed.withLock { value -> Int in
                            let current = value
                            value += 1
                            return current
                        }
                        if index >= ranges.count { break }
                        read(ranges[index], into: buffer)
                    }
                }
                let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                best = min(best, seconds)
            }
            return best
        }

        for concurrency in [3, 10, 30] {
            _ = measure(concurrency: concurrency, rounds: 1)
            let seconds = measure(concurrency: concurrency, rounds: 4)
            let rate = Double(bytes) / seconds / 1e9
            print(
                "\(concurrency) readers: \(String(format: "%.2f", seconds * 1000)) ms, "
                    + "\(String(format: "%.2f", rate)) GB/s")
        }
        #expect(bytes > 0)
    }
}
