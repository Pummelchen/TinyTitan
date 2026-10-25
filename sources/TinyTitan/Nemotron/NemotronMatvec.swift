import TinyTitanKernelsC
import Foundation

/// The matvec surface the Nemotron graph needs. The CPU implementation calls
/// the quantized C kernels with row threading; the GPU implementation runs the
/// same two shaders the Qwen decode path uses, one dispatch per matvec (or one
/// command buffer for a batch of experts).
public protocol NemotronMatvec: AnyObject {
    /// `y = tensor * x`.
    func gemv(
        _ tensor: NemotronTensorRef,
        x: UnsafePointer<Float>,
        y: UnsafeMutablePointer<Float>)

    /// One role of a set of routed experts in a layer. `xStride` is 0 when
    /// every expert reads the same activation (the up projection) and the
    /// expert's row width when each reads its own (the down projection). `y`
    /// is `experts.count` rows, tightly packed.
    func gemvExperts(
        layer: Int,
        experts: [Int],
        role: String,
        x: UnsafePointer<Float>,
        xStride: Int,
        y: UnsafeMutablePointer<Float>)
}

/// The raw pointers a worker needs, boxed so the `concurrentPerform` closure
/// captures one `@unchecked Sendable` value instead of five non-Sendable ones.
/// Every worker writes a disjoint row range and reads the same `x`.
private struct GemvPointers: @unchecked Sendable {
    let weights: UnsafePointer<UInt8>
    let scales: UnsafePointer<UInt16>
    let biases: UnsafePointer<UInt16>
    let x: UnsafePointer<Float>
    let y: UnsafeMutablePointer<Float>
}

/// CPU matvecs: the affine GEMV kernels, rows split across the performance
/// cores. Rows are independent, which is what makes this exact.
public final class NemotronCPU: NemotronMatvec {
    private let install: NemotronInstall
    private let workers: Int

    public init(install: NemotronInstall, workers: Int? = nil) {
        self.install = install
        self.workers = workers ?? max(1, min(8, ProcessInfo.processInfo.activeProcessorCount))
    }

    public func gemv(
        _ tensor: NemotronTensorRef,
        x: UnsafePointer<Float>,
        y: UnsafeMutablePointer<Float>
    ) {
        let base = install.pointer(of: tensor)
        multiply(
            weights: base.assumingMemoryBound(to: UInt8.self),
            scales: install.pointer(of: tensor, scale: true)
                .assumingMemoryBound(to: UInt16.self),
            biases: install.pointer(of: tensor, scale: false)
                .assumingMemoryBound(to: UInt16.self),
            bits: tensor.bits ?? 4,
            rows: tensor.rows,
            columns: tensor.columns,
            x: x,
            y: y,
            workerCount: min(workers, max(1, tensor.rows / 256)))
    }

    public func gemvExperts(
        layer: Int,
        experts: [Int],
        role: String,
        x: UnsafePointer<Float>,
        xStride: Int,
        y: UnsafeMutablePointer<Float>
    ) {
        guard let entry = install.experts[layer], let tensor = entry.roles[role],
            let scalesRole = entry.roles["\(role)_scales"],
            let biasesRole = entry.roles["\(role)_biases"]
        else {
            return
        }
        for (index, expert) in experts.enumerated() {
            let base = try! install.expertPointer(layer: layer, expert: expert)
            multiply(
                weights: base.advanced(by: tensor.offset)
                    .assumingMemoryBound(to: UInt8.self),
                scales: base.advanced(by: scalesRole.offset)
                    .assumingMemoryBound(to: UInt16.self),
                biases: base.advanced(by: biasesRole.offset)
                    .assumingMemoryBound(to: UInt16.self),
                bits: tensor.bits ?? 4,
                rows: tensor.shape[0],
                columns: tensor.shape[1],
                x: xStride == 0 ? x : x + index * xStride,
                y: y + index * tensor.shape[0],
                workerCount: min(workers, max(1, tensor.shape[0] / 256)))
        }
    }

    private func multiply(
        weights: UnsafePointer<UInt8>,
        scales: UnsafePointer<UInt16>,
        biases: UnsafePointer<UInt16>,
        bits: Int,
        rows: Int,
        columns: Int,
        x: UnsafePointer<Float>,
        y: UnsafeMutablePointer<Float>,
        workerCount: Int
    ) {
        let groups = columns / 64
        if workerCount <= 1 {
            if bits == 4 {
                tinytitan_int4_affine_gemv(weights, scales, biases, x, rows, columns, y)
            } else {
                tinytitan_int8_affine_gemv(weights, scales, biases, x, rows, columns, y)
            }
            return
        }
        let raw = GemvPointers(
            weights: weights, scales: scales, biases: biases, x: x, y: y)
        DispatchQueue.concurrentPerform(iterations: workerCount) { worker in
            let first = worker * rows / workerCount
            let count = (worker + 1) * rows / workerCount - first
            if count <= 0 { return }
            let rowBytes = columns * bits / 8
            if bits == 4 {
                tinytitan_int4_affine_gemv(
                    raw.weights.advanced(by: first * rowBytes),
                    raw.scales.advanced(by: first * groups),
                    raw.biases.advanced(by: first * groups),
                    raw.x, count, columns, raw.y.advanced(by: first))
            } else {
                tinytitan_int8_affine_gemv(
                    raw.weights.advanced(by: first * columns),
                    raw.scales.advanced(by: first * groups),
                    raw.biases.advanced(by: first * groups),
                    raw.x, count, columns, raw.y.advanced(by: first))
            }
        }
    }
}
