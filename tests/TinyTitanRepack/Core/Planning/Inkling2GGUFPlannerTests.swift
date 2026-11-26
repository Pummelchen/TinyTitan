import Foundation
import Testing
import TinyTitanFormat

@testable import TinyTitanRepackCore

/// The Inkling 2 GGUF install plan (TT-121 phase 1), checked against the real
/// Q4_K_XL file's header: no bytes are copied, so it runs in a second, and it
/// pins the arithmetic that decides where every byte of the install comes from.
@Suite struct Inkling2GGUFPlannerTests {
    private static var gguf: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_GGUF"]
    }

    static func source() throws -> Inkling2GGUFPlanner.Source {
        let path = try #require(gguf)
        let handle = try #require(FileHandle(forReadingAtPath: path))
        defer { try? handle.close() }
        let header = try GGUFHeaderV3(
            bytes: [UInt8](try handle.read(upToCount: 64 << 20) ?? Data()))
        let size = try #require(
            (try FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber)?
                .uint64Value)
        return .init(
            fileName: (path as NSString).lastPathComponent, header: header, fileSize: size)
    }

    /// Every tensor's computed size ends before the next tensor starts, short of
    /// it by less than the file's alignment: the size arithmetic agrees with
    /// how the file is actually laid out, tensor by tensor.
    @Test(.enabled(if: gguf != nil)) func computedSizesTileTheFile() throws {
        let source = try Self.source()
        let tensors = source.header.tensors.sorted { $0.absoluteOffset < $1.absoluteOffset }
        for (index, tensor) in tensors.enumerated() {
            let kept = try Inkling2GGUFPlanner.kept(ggmlType: tensor.ggmlType, tensor: tensor.name)
            let size = try Inkling2GGUFPlanner.bytes(
                of: tensor.dimensions, kept: kept, tensor: tensor.name)
            let end =
                index + 1 < tensors.count ? tensors[index + 1].absoluteOffset : source.fileSize
            #expect(tensor.absoluteOffset + size <= end, "\(tensor.name) overruns")
            #expect(
                end - (tensor.absoluteOffset + size) < source.header.alignment, "\(tensor.name) gap"
            )
        }
    }

    @Test(.enabled(if: gguf != nil)) func thePlanMatchesTheModel() throws {
        let plan = try Inkling2GGUFPlanner.plan(
            source: try Self.source(), outputDir: "/tmp/inkling-plan")
        #expect(plan.arch.numLayers == 24)
        #expect(plan.arch.numExperts == 128)
        #expect(plan.arch.topKExperts == 8)
        // The checkpoint's own `local_layer_ids`: every layer but 5, 11, 17, 23.
        let local = try #require(plan.arch.variantGeometry?.lists["localLayers"])
        #expect(local == (0..<24).filter { ![5, 11, 17, 23].contains($0) })
        #expect(plan.arch.variantGeometry?.scalars["loopPasses"] == 3)
        #expect(plan.resident.entries.count == 734 - 88)
        // NVFP4 layers: 2,985,984 + 1,492,992 + two 4-byte scales, page-rounded.
        let nvfp4Stride = try Inkling2GGUFPlanner.roundUp(
            2_985_984 + 1_492_992 + 64 + 4, to: 16_384)
        #expect(plan.layers[0].expertStride == nvfp4Stride)
        #expect(plan.layers[20].expertStride > nvfp4Stride)
        #expect(plan.manifestExpertStride == plan.layers[20].expertStride)
        #expect(plan.layers[0].subTensors.first?.bitsForWeights == 4)
        #expect(plan.layers[20].subTensors.first?.bitsForWeights == 8)
        #expect(plan.passthroughFiles.first?.size == source0DataStart())
    }

    @Test(.enabled(if: gguf != nil)) func theCopyPlanAndLayoutAreValid() throws {
        let plan = try Inkling2GGUFPlanner.plan(
            source: try Self.source(), outputDir: "/tmp/inkling-plan")
        let copies = try RangeCopyPlanner.plan(repackPlan: plan, rangeChunkBytes: 64 << 20)
        // Every tensor byte is copied once, plus the header sidecar.
        let tensorBytes =
            plan.resident.entries.reduce(UInt64(0)) { $0 + $1.sizeBytes }
            + plan.layers.reduce(UInt64(0)) { total, layer in
                total + layer.subTensors.reduce(0) { $0 + $1.sizeInExpertBlob }
                    * UInt64(layer.expertsPerLayer)
            }
        let copied = copies.scalarCopies.reduce(UInt64(0)) { $0 + $1.size }
        #expect(copied == tensorBytes + (plan.passthroughFiles.first?.size ?? 0))
        let layout = try SSDAIJSON.encodeLayout(plan: plan, expertStride: plan.manifestExpertStride)
        _ = try SSDAIPackedExpertsLayoutCodec.decode(layout)
    }

    /// The header's dimensions are untrusted. A product that wraps `UInt64`
    /// must be a typed planner error, not arithmetic that traps before the
    /// planner can refuse the file.
    @Test("A dimension product that cannot fit is refused")
    func oversizedDimensionsAreRefused() throws {
        let f32 = try Inkling2GGUFPlanner.kept(ggmlType: 0, tensor: "w")
        // The rows product itself overflows: UInt64.max * 2.
        #expect(throws: RepackError.self) {
            _ = try Inkling2GGUFPlanner.bytes(
                of: [UInt64.max, 2], kept: f32, tensor: "w")
        }
        // rows * columns fits, rows * columns * elementBytes does not.
        #expect(throws: RepackError.self) {
            _ = try Inkling2GGUFPlanner.bytes(
                of: [1 << 62, 2], kept: f32, tensor: "w")
        }
        // A block encoding computes the same rows product before payloadBytes.
        let nvfp4 = try Inkling2GGUFPlanner.kept(ggmlType: 40, tensor: "w")
        #expect(throws: RepackError.self) {
            _ = try Inkling2GGUFPlanner.bytes(
                of: [64, UInt64.max, 2], kept: nvfp4, tensor: "w")
        }
    }

    @Test("Valid dimensions keep their exact byte counts")
    func validDimensionsKeepTheirByteCounts() throws {
        let f32 = try Inkling2GGUFPlanner.kept(ggmlType: 0, tensor: "w")
        #expect(try Inkling2GGUFPlanner.bytes(of: [64, 2], kept: f32, tensor: "w") == 512)
        #expect(
            try Inkling2GGUFPlanner.bytes(of: [64, 2, 3], kept: f32, tensor: "w") == 1_536)
        let q8 = try Inkling2GGUFPlanner.kept(ggmlType: 8, tensor: "w")
        // 4 rows of 32 values = 4 blocks of 34 bytes.
        #expect(try Inkling2GGUFPlanner.bytes(of: [32, 4], kept: q8, tensor: "w") == 136)
    }

    @Test("Rounding up refuses a value that cannot be aligned")
    func roundingUpRejectsOverflow() throws {
        #expect(try Inkling2GGUFPlanner.roundUp(0, to: 16_384) == 0)
        #expect(try Inkling2GGUFPlanner.roundUp(1, to: 16_384) == 16_384)
        #expect(try Inkling2GGUFPlanner.roundUp(16_384, to: 16_384) == 16_384)
        #expect(throws: RepackError.self) {
            _ = try Inkling2GGUFPlanner.roundUp(UInt64.max, to: 16_384)
        }
        #expect(throws: RepackError.self) {
            _ = try Inkling2GGUFPlanner.roundUp(0, to: 0)
        }
    }

    /// The resident index addresses names with a `UInt32`; the header/checkpoint
    /// caps keep a real table far below the limit, so the check is pinned at
    /// the boundary itself rather than by building a 4 GiB table.
    @Test("A string-table offset past UInt32 is refused")
    func stringTableOffsetIsChecked() throws {
        #expect(try Inkling2GGUFPlanner.stringTableOffset(0) == 0)
        #expect(
            try Inkling2GGUFPlanner.stringTableOffset(Int(UInt32.max)) == UInt32.max)
        #expect(throws: RepackError.self) {
            _ = try Inkling2GGUFPlanner.stringTableOffset(Int(UInt32.max) + 1)
        }
        #expect(throws: RepackError.self) {
            _ = try Inkling2GGUFPlanner.stringTableOffset(-1)
        }
    }

    private func source0DataStart() -> UInt64 {
        (try? Self.source().header.dataStart) ?? 0
    }
}
