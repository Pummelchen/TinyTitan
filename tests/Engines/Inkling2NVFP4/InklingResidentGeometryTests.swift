import Foundation
import Metal
import Testing

import TinyTitanCore
@testable import Inkling2NVFP4

/// Manifest-driven Inkling geometry refuses instead of trapping (AUD-0084).
///
/// The checkpoint's resident index is input: a shape whose element count
/// overflows, or a size that disagrees with its shape, must be a thrown error
/// rather than an arithmetic trap during load.
@Suite("Inkling resident geometry")
struct InklingResidentGeometryTests {
    private static func entry(
        shape: (UInt32, UInt32, UInt32, UInt32), size: UInt64 = 8
    ) -> ResidentIndexEntry {
        ResidentIndexEntry(
            name: "blk.0.attn_norm.weight", dtype: 1, fileOffset: 0, sizeBytes: size,
            shape: shape, scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0)
    }

    private static func withMappedFile(
        _ body: (MappedFile, MTLDevice) throws -> Void
    ) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling-geometry-\(UUID().uuidString).bin")
        try Data(repeating: 0, count: 64).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        try body(try MappedFile(url: url, device: device), device)
    }

    @Test("A converted role whose element count overflows is refused")
    func conversionRefusesAnOverflowingElementCount() throws {
        try Self.withMappedFile { file, device in
            let source = Self.entry(shape: (UInt32.max, UInt32.max, 0, 0))
            let conversion = InklingNVFP4Conversion(source: source, transform: .identity)
            #expect(throws: InklingQ4Manifest.Error.self) {
                _ = try InklingQ4Weights.converted(
                    source.name, entry: source, conversion: conversion,
                    file: file, device: device)
            }
        }
    }

    @Test("A converted role whose size is not its shape's bytes is refused")
    func conversionRefusesASizeThatDisagreesWithItsShape() throws {
        try Self.withMappedFile { file, device in
            let source = Self.entry(shape: (16, 1, 1, 1), size: 34)
            let conversion = InklingNVFP4Conversion(source: source, transform: .identity)
            #expect(throws: InklingQ4Manifest.Error.self) {
                _ = try InklingQ4Weights.converted(
                    source.name, entry: source, conversion: conversion,
                    file: file, device: device)
            }
        }
    }

    @Test("An MTP shape whose row count overflows is refused")
    func mtpRefusesAnOverflowingRowCount() {
        let entry = Self.entry(shape: (UInt32.max, UInt32.max, UInt32.max, 4))
        #expect(throws: InklingQ4Manifest.Error.self) {
            _ = try InklingMTPWeights.rowsAndColumns(entry)
        }
    }

    @Test("An ordinary MTP shape still splits into rows and columns")
    func mtpSplitsAnOrdinaryShape() throws {
        let dimensions = try InklingMTPWeights.rowsAndColumns(
            Self.entry(shape: (2, 3, 4, 0)))
        #expect(dimensions.rows == 6)
        #expect(dimensions.columns == 4)
    }
}
