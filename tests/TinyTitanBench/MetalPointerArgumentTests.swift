import Metal
import Testing

/// A device pointer inside a kernel argument struct must survive the encoder.
///
/// `TinyTitanBench`'s routed-MoE arms build a `RoutedBlobs`-shaped struct of
/// device pointers; the kernel receives a non-null pointer but reads zeros,
/// while the CPU sees the sentinel bytes. This pins the primitive the harness
/// depends on — one pointer struct, one sentinel buffer — without the MoE
/// kernel or its layout, so it says whether the problem is the pattern or the
/// harness (AUD-0122).
@Suite struct MetalPointerArgumentTests {
    private static let source = """
        #include <metal_stdlib>
        using namespace metal;
        struct Pointers { device const uint8_t* blob[16]; };
        kernel void read_first(
            device Pointers& pointers [[buffer(0)]],
            device uint* out [[buffer(1)]]
        ) {
            out[0] = uint(pointers.blob[0][0]);
        }
        """

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil))
    func eightLargeBlobsInTheSameStructAlsoArrive() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let library = try device.makeLibrary(source: Self.source, options: nil)
        let function = try #require(library.makeFunction(name: "read_first"))
        let pipeline = try device.makeComputePipelineState(function: function)

        var blobs: [MTLBuffer] = []
        for i in 0..<8 {
            let blob = try #require(
                device.makeBuffer(length: 2_000_000, options: .storageModeShared))
            memset(blob.contents(), Int32(0x11 + i), 2_000_000)
            blobs.append(blob)
        }
        let encoder = function.makeArgumentEncoder(bufferIndex: 0)
        let arguments = try #require(
            device.makeBuffer(length: encoder.encodedLength, options: .storageModeShared))
        encoder.setArgumentBuffer(arguments, offset: 0)
        for (index, blob) in blobs.enumerated() {
            encoder.setBuffer(blob, offset: 0, index: index)
        }

        let out = try #require(
            device.makeBuffer(length: 4, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())
        let cb = try #require(queue.makeCommandBuffer())
        let compute = try #require(cb.makeComputeCommandEncoder())
        compute.setComputePipelineState(pipeline)
        compute.setBuffer(arguments, offset: 0, index: 0)
        compute.setBuffer(out, offset: 0, index: 1)
        for blob in blobs { compute.useResource(blob, usage: .read) }
        compute.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        compute.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        #expect(cb.error == nil)
        #expect(out.contents().load(as: UInt32.self) == 0x11)
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil))
    func aPointerInAStructReachesTheKernel() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let library = try device.makeLibrary(source: Self.source, options: nil)
        let function = try #require(library.makeFunction(name: "read_first"))
        let pipeline = try device.makeComputePipelineState(function: function)

        let blob = try #require(
            device.makeBuffer(length: 16, options: .storageModeShared))
        memset(blob.contents(), 0x11, 16)

        let encoder = function.makeArgumentEncoder(bufferIndex: 0)
        let arguments = try #require(
            device.makeBuffer(length: encoder.encodedLength, options: .storageModeShared))
        encoder.setArgumentBuffer(arguments, offset: 0)
        encoder.setBuffer(blob, offset: 0, index: 0)

        let out = try #require(
            device.makeBuffer(length: 4, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())
        let cb = try #require(queue.makeCommandBuffer())
        let compute = try #require(cb.makeComputeCommandEncoder())
        compute.setComputePipelineState(pipeline)
        compute.setBuffer(arguments, offset: 0, index: 0)
        compute.setBuffer(out, offset: 0, index: 1)
        compute.useResource(blob, usage: .read)
        compute.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        compute.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        #expect(cb.error == nil)
        #expect(out.contents().load(as: UInt32.self) == 0x11)
    }
}
