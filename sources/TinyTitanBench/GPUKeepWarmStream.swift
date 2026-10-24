import Foundation
import Metal
import Synchronization

/// Benchmark-only keep-warm stream: issues a one-threadgroup dispatch on its own
/// command queue every `intervalUs` while the measured arm sleeps between
/// launches, so the device never sees a long zero-work window.
///
/// §5e measured the identical head kernel at 9.42 ms back-to-back and 12.73 ms
/// with 50 ms gaps, then closed the lever on the assertion that nothing in the
/// engine controls GPU power state. That premise was never tested. This is the
/// cheapest thing that could hold the state: a dispatch too small to matter,
/// submitted often enough that the device is never asked to decide whether it is
/// idle. If it does not recover the penalty, the clock hypothesis is closed by
/// measurement instead of by assumption.
///
/// unchecked-invariant: `dispatches` is written only by the stream queue and
/// read only after `stop()` has joined it; `probeMicroseconds` is written by
/// `calibrate()` before `start()` and never again.
/// `running` is an atomic cross-thread flag.
final class GPUKeepWarmStream: @unchecked Sendable {
    static let minimumIntervalUs = 100
    static let maximumIntervalUs = 20_000
    private static let spinRows: UInt32 = 8
    private static let threadsPerThreadgroup = 256

    private let queue: MTLCommandQueue
    private let pso: MTLComputePipelineState
    private let weights: MTLBuffer
    private let scales: MTLBuffer
    private let biases: MTLBuffer
    private let input: MTLBuffer
    private let output: MTLBuffer
    private let columns: UInt32
    private let interval: TimeInterval
    private let serial = DispatchQueue(label: "tinytitan.bench.keepwarm")
    private let finished = DispatchSemaphore(value: 0)
    private let running = Atomic<Bool>(false)
    private var dispatches = 0
    private(set) var probeMicroseconds = 0.0

    /// The spin buffers are a `Self.spinRows`-row slice of the same affine GEMV
    /// the measured arm runs, so the stream costs a single threadgroup.
    /// `rowBytes` and `groupBytes` are the measured arm's own per-row sizes.
    init(
        device: MTLDevice,
        pso: MTLComputePipelineState,
        columns: UInt32,
        rowBytes: Int,
        groupBytes: Int,
        input: MTLBuffer,
        intervalUs: Int
    ) throws {
        guard intervalUs >= Self.minimumIntervalUs, intervalUs <= Self.maximumIntervalUs else {
            throw BenchHarnessError.badConfiguration(
                "TINYTITAN_BENCH_HEAD_KEEPWARM_US must be \(Self.minimumIntervalUs)...\(Self.maximumIntervalUs), got \(intervalUs)"
            )
        }
        guard let queue = device.makeCommandQueue() else {
            throw BenchHarnessError.metalObjectUnavailable("keep-warm command queue")
        }
        self.queue = queue
        self.pso = pso
        self.columns = columns
        self.input = input
        self.interval = Double(intervalUs) / 1_000_000
        func makeBuffer(_ bytes: Int, _ value: UInt8) throws -> MTLBuffer {
            guard let buf = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw BenchHarnessError.metalObjectUnavailable("keep-warm buffer of \(bytes) bytes")
            }
            memset(buf.contents(), Int32(value), bytes)
            return buf
        }
        let rows = Int(Self.spinRows)
        self.weights = try makeBuffer(rows * rowBytes, 0x5A)
        self.scales = try makeBuffer(rows * groupBytes, 0x3C)
        self.biases = try makeBuffer(rows * groupBytes, 0x00)
        self.output = try makeBuffer(rows * 2, 0)
    }

    /// One synchronous dispatch, timed on the device, to record what the stream
    /// costs per iteration before any of it is measured against the arm.
    func calibrate() throws {
        let cb = try requireCommandBuffer(queue)
        encode(cb)
        cb.commit()
        cb.waitUntilCompleted()
        probeMicroseconds = (cb.gpuEndTime - cb.gpuStartTime) * 1_000_000
    }

    func start() {
        running.store(true, ordering: .releasing)
        serial.async { [weak self] in
            guard let self else { return }
            while self.running.load(ordering: .acquiring) {
                if let cb = self.queue.makeCommandBuffer() {
                    self.encode(cb)
                    cb.commit()
                    // Waiting bounds how many command buffers are in flight; the
                    // spin itself is microseconds, so this thread is asleep for
                    // essentially the whole interval.
                    cb.waitUntilCompleted()
                    self.dispatches += 1
                }
                Thread.sleep(forTimeInterval: self.interval)
            }
            self.finished.signal()
        }
    }

    /// Number of dispatches the stream actually issued. Blocks until the stream
    /// thread has joined, so the count is final when it returns.
    func stop() -> Int {
        running.store(false, ordering: .releasing)
        finished.wait()
        return dispatches
    }

    /// Mirrors `runHeadCadence`'s binding: the same kernel, the same argument
    /// indices, eight rows instead of the vocabulary's 248,320.
    private func encode(_ cb: MTLCommandBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var rows = Self.spinRows
        var n = columns
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: 0, index: 0)
        enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2)
        enc.setBuffer(input, offset: 0, index: 3)
        enc.setBuffer(output, offset: 0, index: 4)
        enc.setBytes(&rows, length: 4, index: 5)
        enc.setBytes(&n, length: 4, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: Self.threadsPerThreadgroup, height: 1, depth: 1))
        enc.endEncoding()
    }
}
