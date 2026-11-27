import Foundation
import Testing

@testable import TinyTitanRepackCore

/// The accounting `--compress-experts` reports: the raw plan total and the
/// per-layer sums. The inputs are plan declarations, so an overflow is refused
/// rather than wrapping or trapping; the boundary is tested directly because a
/// full `run` would need a layer file of the same impossible size.
@Suite("Compress experts accounting")
struct CompressExpertsToolTests {
    @Test("The raw plan total is checked")
    func rawTotalIsChecked() throws {
        #expect(
            try CompressExpertsTool.totalRawBytes(
                rawLayerBytes: 1 << 40, layerCount: 4) == 1 << 42)
        #expect(
            try CompressExpertsTool.totalRawBytes(
                rawLayerBytes: UInt64.max / 2, layerCount: 2) == UInt64.max - 1)
        #expect(throws: RepackError.self) {
            _ = try CompressExpertsTool.totalRawBytes(
                rawLayerBytes: UInt64.max / 2 + 1, layerCount: 2)
        }
        #expect(throws: RepackError.self) {
            _ = try CompressExpertsTool.totalRawBytes(
                rawLayerBytes: 1, layerCount: -1)
        }
    }

    @Test("The summary sums are checked")
    func summarySumsAreChecked() throws {
        #expect(try CompressExpertsTool.totalBytes([1, 2, 3], field: "bytes") == 6)
        #expect(try CompressExpertsTool.totalBytes([], field: "bytes") == 0)
        #expect(throws: RepackError.self) {
            _ = try CompressExpertsTool.totalBytes([UInt64.max, 1], field: "bytes")
        }
    }

    /// A minimal raw packed-expert install: two layers of one 16 KiB expert.
    /// Only what `loadInput` and the layer writer read is present.
    private static func writeRawInstall(at root: URL) throws {
        let experts = root.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)
        let stride = 16_384
        func layer(_ index: Int) -> [String: Any] {
            [
                "layer": index,
                "file": String(format: "layer_%02d.bin", index),
                "experts": [
                    [
                        "expert": 0, "offset": 0, "size": stride,
                        "tensors": [
                            "gate": [
                                "offset": 0, "size": stride, "dtype": "F32",
                                "shape": [stride / 4], "bits": 4,
                            ]
                        ],
                    ]
                ],
            ]
        }
        let layout: [String: Any] = [
            "expertStride": stride, "numLayers": 2, "expertsPerLayer": 1,
            "layers": [layer(0), layer(1)],
        ]
        let layoutData = try JSONSerialization.data(withJSONObject: layout, options: [.sortedKeys])
        try layoutData.write(to: experts.appendingPathComponent("layout.json"))
        var files: [String: Any] = [
            "packed_experts/layout.json": ["size": layoutData.count, "sha256": "00"],
            "model_weights.bin": ["size": 8, "sha256": "00"],
        ]
        for index in 0..<2 {
            let name = String(format: "layer_%02d.bin", index)
            try Data(repeating: UInt8(index &+ 1), count: stride)
                .write(to: experts.appendingPathComponent(name))
            files["packed_experts/\(name)"] = ["size": stride, "sha256": "00"]
        }
        try Data(repeating: 0, count: 8).write(
            to: root.appendingPathComponent("model_weights.bin"))
        let manifest: [String: Any] = [
            "files": files, "expertsPerLayer": 1, "numLayers": 2,
            "expertStride": stride, "sourceSnapshotHash": "deadbeef",
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: root.appendingPathComponent("manifest.json"))
    }

    /// AUD-0076: `--overwrite` used to delete the destination before the
    /// conversion, and again on failure, so a failed run left neither install.
    /// It must mean "replace on success": the previous install keeps every byte,
    /// and the staging directory is gone.
    @Test("A failed overwrite leaves the existing output untouched")
    func failedOverwriteKeepsTheExistingInstall() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("compress-failure-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: root.path) }
        let input = root.appendingPathComponent("raw")
        let output = root.appendingPathComponent("compressed")
        try Self.writeRawInstall(at: input)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let sentinel = output.appendingPathComponent("previous-install.bin")
        let payload = Data((0..<1_024).map { UInt8(truncatingIfNeeded: $0) })
        try payload.write(to: sentinel)

        // The fixture's manifest has no `quant` block, so the conversion and
        // both layer writes into staging succeed and the final verification is
        // the step that fails. That is the worst case for the old code: the
        // destination had already been deleted and the new install was unusable.
        var layers: [Int] = []
        #expect(throws: RepackError.self) {
            _ = try CompressExpertsTool.run(
                options: CompressExpertsOptions(
                    inputSSDAI: input.path, output: output.path, overwrite: true),
                progress: { layers.append($0.layer) })
        }
        #expect(layers == [0, 1], "the failure did not happen after the layers converted")

        #expect(try Data(contentsOf: sentinel) == payload)
        #expect(
            try FileManager.default.contentsOfDirectory(atPath: output.path)
                == ["previous-install.bin"])
        #expect(
            !FileManager.default.fileExists(atPath: output.path + ".partial"),
            "a failed conversion left its staging directory behind")
    }

    /// The promotion half: the finished build replaces the destination in one
    /// step, and the staging path is gone afterwards.
    @Test("The staging directory is promoted onto the destination")
    func stagingIsPromoted() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("compress-promote-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: root.path) }
        let output = root.appendingPathComponent("compressed")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try Data([0x01]).write(to: output.appendingPathComponent("old.bin"))

        let staging = output.path + ".partial"
        try FileManager.default.createDirectory(atPath: staging, withIntermediateDirectories: true)
        try Data([0x02]).write(to: URL(fileURLWithPath: staging + "/new.bin"))

        try CompressExpertsTool.install(staging, at: output.path, overwrite: true)

        #expect(
            Set(try FileManager.default.contentsOfDirectory(atPath: output.path))
                == ["new.bin"])
        #expect(!FileManager.default.fileExists(atPath: staging))
    }
}
