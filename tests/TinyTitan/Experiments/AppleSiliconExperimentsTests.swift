import Darwin
import Foundation
import Testing

@testable import TinyTitan

/// The AMX kernel is an experiment with no engine caller; these tests pin its
/// contract and its reference output, not its speed.
@Suite struct AMXKernelTests {
    @Test func probeIsStableAcrossCalls() {
        let first = AMXKernel.isAvailable
        #expect(AMXKernel.isAvailable == first)
    }

    @Test func rejectsShapesAMXCannotTile() {
        #expect(AMXKernel.multiply([1], [1], m: 8, n: 16, k: 1) == nil)
        #expect(AMXKernel.multiply([1], [1], m: 16, n: 24, k: 1) == nil)
        #expect(AMXKernel.multiply([1], [1], m: 0, n: 16, k: 1) == nil)
    }

    @Test func rejectsShortOperands() {
        // 16x16x16 needs 256 A elements and 256 B elements; one of each is not
        // enough regardless of whether AMX exists on this machine.
        #expect(AMXKernel.multiply([1], [1], m: 16, n: 16, k: 16) == nil)
    }

    @Test func matchesReferenceWhenAvailable() throws {
        guard AMXKernel.isAvailable else { return }

        let m = 32
        let n = 16
        let k = 24
        var a = [Float](repeating: 0, count: m * k)
        var b = [Float](repeating: 0, count: k * n)
        for index in 0..<a.count {
            a[index] = Float((index % 9) - 4) * 0.25
        }
        for index in 0..<b.count {
            b[index] = Float((index % 7) - 3) * 0.5
        }

        let c = try #require(AMXKernel.multiply(a, b, m: m, n: n, k: k))
        #expect(c.count == m * n)
        for row in 0..<m {
            for column in 0..<n {
                var expected = 0.0
                for inner in 0..<k {
                    expected += Double(a[row * k + inner]) * Double(b[inner * n + column])
                }
                let actual = Double(c[row * n + column])
                #expect(
                    abs(actual - expected) <= 1e-4 * max(1.0, abs(expected)),
                    "cell [\(row),\(column)] = \(actual), expected \(expected)")
            }
        }
    }

    @Test func zeroDepthYieldsZeroMatrixWhenAvailable() throws {
        guard AMXKernel.isAvailable else { return }
        let c = try #require(AMXKernel.multiply([], [], m: 16, n: 16, k: 0))
        #expect(c.count == 256)
        #expect(c.allSatisfy { $0 == 0 })
    }
}

/// `DC ZVA` zeroes 64 bytes, not the 128-byte line size; these tests pin the
/// range boundaries so a future "optimisation" to a 128-byte stride fails
/// loudly instead of leaving half a buffer dirty.
@Suite struct ZeroFillTests {
    @Test func zeroesOnlyTheRequestedAlignedRange() {
        let count = 4096
        let raw = UnsafeMutableRawPointer.allocate(byteCount: count, alignment: 64)
        defer { raw.deallocate() }
        let bytes = raw.assumingMemoryBound(to: UInt8.self)
        for index in 0..<count {
            bytes[index] = 0x5A
        }

        ZeroFill.zero(raw.advanced(by: 64), count: 1024)

        #expect(bytes[63] == 0x5A)
        #expect(bytes[64] == 0)
        #expect(bytes[64 + 1023] == 0)
        #expect(bytes[64 + 1024] == 0x5A)
    }

    @Test func unalignedBaseFallsBackWithoutTouchingThePrefix() {
        let count = 4096
        let raw = UnsafeMutableRawPointer.allocate(byteCount: count, alignment: 64)
        defer { raw.deallocate() }
        let bytes = raw.assumingMemoryBound(to: UInt8.self)
        for index in 0..<count {
            bytes[index] = 0x5A
        }

        ZeroFill.zero(raw.advanced(by: 1), count: 1024)

        #expect(bytes[0] == 0x5A)
        #expect(bytes[1] == 0)
        #expect(bytes[1 + 1023] == 0)
        #expect(bytes[1 + 1024] == 0x5A)
    }

    @Test func zeroCountDoesNothing() {
        var value: UInt8 = 0x5A
        ZeroFill.zero(&value, count: 0)
        #expect(value == 0x5A)
    }
}

/// The resident-advice knob is an A/B experiment; these tests pin the default
/// and prove each advice leaves the mapping readable.
@Suite struct MemoryAdviceTests {
    @Test func defaultsToRandomForMissingOrUnknownValues() {
        #expect(MemoryAdvice.residentAdvice(environment: [:]) == .random)
        #expect(
            MemoryAdvice.residentAdvice(
                environment: ["TINYTITAN_RESIDENT_ADVICE": "bogus"]) == .random)
        #expect(
            MemoryAdvice.residentAdvice(
                environment: ["TINYTITAN_RESIDENT_ADVICE": "WILLNEED"]) == .willneed)
        #expect(
            MemoryAdvice.residentAdvice(
                environment: ["TINYTITAN_RESIDENT_ADVICE": "none"]) == .none)
    }

    @Test func eachAdviceLeavesTheMappingReadable() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tinytitan-advice-\(UUID().uuidString)")
        try Data(repeating: 0x7, count: 8192).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let descriptor = open(url.path, O_RDONLY)
        #expect(descriptor >= 0)
        defer { close(descriptor) }

        let mapped = try #require(
            mmap(nil, 8192, PROT_READ, MAP_PRIVATE, descriptor, 0))
        defer { munmap(mapped, 8192) }
        let bytes = mapped.assumingMemoryBound(to: UInt8.self)

        for advice in ResidentAdvice.allCases {
            MemoryAdvice.apply(advice, base: mapped, length: 8192)
            #expect(bytes[0] == 0x7, "advice \(advice) corrupted the head")
            #expect(bytes[8191] == 0x7, "advice \(advice) corrupted the tail")
        }
    }
}
