import Foundation
import TinyTitanFormat
import TinyTitanCore

// The packed-expert half of `--verify-install`.
//
// Split out of `VerifiedInstallTool.swift` (2026-10-06, AUD-124) under the
// 500-line-per-file rule. `layout` and `validatePackedExpertLayout` moved as pure
// code motion — same bodies, same messages, `private` widened to internal because
// the caller stays in the other file — and the width cross-check below them is
// what the split was for: `packed_experts/layout.json` is the only place a MoE
// install records how its expert bytes are packed, so the file that reads it is
// the file that has to prove it.

extension VerifiedInstallTool {

    /// Read the packed-expert layout the manifest names.
    ///
    /// Decoded through the verifier's local mirror, which carries the fields this
    /// tool needs and none of the format layer's structural validation. That is
    /// what makes the arithmetic below the check that has to hold: nothing else
    /// here asks whether the described bytes add up.
    static func loadLayout(access: SSDAIDirectoryAccess) throws -> PackedExpertsLayout {
        do {
            let data = try loadMetadataJSON(
                access: access,
                relativePath: "packed_experts/layout.json")
            // The runtime decodes this file through the format layer, whose
            // structural validator refuses duplicate layer files, overlapping
            // experts and impossible counts. The mirror below carries only the
            // fields the width arithmetic needs, so validating it alone
            // certified installs the loader then refused -- the opposite of
            // what a verifier is for (AUD-0074). Same bytes, same predicate.
            _ = try SSDAIPackedExpertsLayoutCodec.decode(data)
            return try JSONDecoder().decode(PackedExpertsLayout.self, from: data)
        } catch {
            throw RepackError.configurationInvalid(
                detail: "packed_experts/layout.json invalid: \(error)")
        }
    }

    static func validatePackedExpertLayout(
        access: SSDAIDirectoryAccess,
        manifest: Manifest
    ) throws {
        let layoutRelativePath = "packed_experts/layout.json"
        guard manifest.files[layoutRelativePath] != nil else {
            throw RepackError.configurationInvalid(detail: "manifest missing \(layoutRelativePath)")
        }
        let layout = try loadLayout(access: access)
        let alignment = SSDAIFormatV1.alignmentBytes
        guard layout.expertStride == manifest.expertStride,
            layout.numLayers == manifest.numLayers,
            layout.expertsPerLayer == manifest.expertsPerLayer
        else {
            throw RepackError.configurationInvalid(
                detail: "packed expert layout dimensions mismatch manifest")
        }
        guard layout.expertStride % alignment == 0 else {
            throw RepackError.configurationInvalid(
                detail: "expertStride \(layout.expertStride) is not aligned to \(alignment) bytes")
        }
        guard layout.layers.count == layout.numLayers else {
            throw RepackError.configurationInvalid(
                detail: "packed expert layout layer count mismatch")
        }
        if let compression = layout.compression,
            compression != "lzfse", compression != "rans"
        {
            throw RepackError.configurationInvalid(
                detail: "packed_experts/layout.json declares unknown compression "
                    + "\"\(compression)\"")
        }
        // The routed-expert width the expert bytes below have to agree with. A
        // dense install never reaches it: its layers hold no experts, and the
        // resident index is what describes its weights.
        let declaredExpertBits = manifest.quant?.routedExpert.weightBits
        for layer in layout.layers {
            let capacity = try packedLayerCapacity(
                count: layer.expertsPerLayer ?? layout.expertsPerLayer,
                stride: layout.expertStride)
            // A raw layer ends where its last expert does. For a uniform layer
            // that is exactly `count * stride`; a compact one (Inkling's NVFP4
            // layers inside a Q8_0-sized stride) ends earlier, which the format
            // allows. A compressed layer is checked against the capacity.
            var expectedLayerSize = capacity
            if layout.compression == nil, !layer.experts.isEmpty {
                expectedLayerSize = layer.experts.reduce(UInt64(0)) {
                    max($0, $1.offset &+ $1.size)
                }
                guard expectedLayerSize <= capacity else {
                    throw RepackError.configurationInvalid(
                        detail: "packed_experts/\(layer.file) extends past its capacity")
                }
            }
            try validateLayer(
                layer, access: access, manifest: manifest, layout: layout,
                expectedLayerSize: expectedLayerSize,
                declaredExpertBits: declaredExpertBits)
        }
    }

    /// `count * stride` for one packed layer, refusing a count that cannot be a
    /// number of experts and a size that overflows.
    ///
    /// `layer.expertsPerLayer` is an `Int?` decoded from an untrusted document
    /// with no bound; the install-wide value is compared against the manifest,
    /// the per-layer override is not. `UInt64(-1)` traps -- a verifier that
    /// crashes reads as a broken tool, not a broken install (AUD-0073).
    static func packedLayerCapacity(count: Int, stride: UInt64) throws -> UInt64 {
        guard count >= 0 else {
            throw RepackError.configurationInvalid(
                detail: "packed expert layer declares \(count) experts")
        }
        let (capacity, overflow) = UInt64(count).multipliedReportingOverflow(by: stride)
        guard !overflow else {
            throw RepackError.configurationInvalid(
                detail: "packed expert layer size overflows")
        }
        return capacity
    }

    /// One layer: its file size, and every expert's own arithmetic.
    ///
    /// Split out of `validatePackedExpertLayout` so the compressed and raw
    /// cases each read as one check rather than as branches inside a loop.
    private static func validateLayer(
        _ layer: Layer,
        access: SSDAIDirectoryAccess,
        manifest: Manifest,
        layout: PackedExpertsLayout,
        expectedLayerSize: UInt64,
        declaredExpertBits: Int?
    ) throws {
        guard layer.layer >= 0 && layer.layer < layout.numLayers else {
            throw RepackError.configurationInvalid(
                detail: "packed expert layer index out of range")
        }
        let layerExperts = layer.expertsPerLayer ?? layout.expertsPerLayer
        guard layer.experts.count == layerExperts else {
            throw RepackError.configurationInvalid(
                detail: "packed_experts/\(layer.file) expert count mismatch")
        }
        try SSDAIPathValidator.validateBasename(
            layer.file, field: "packed_experts/layout.json layers[\(layer.layer)].file")
        // A layer with no routed experts has no file, and writing one empty
        // `layer_NN.bin` per layer to satisfy this loop would be worse than the
        // check: the dense Qwen 3.5 installs are exactly this shape, 24 layouts
        // and no packed experts at all. The expected size is what makes this
        // safe rather than a hole -- `expectedLayerSize` is 0 only when the
        // layer is empty, so a layer that should carry bytes still fails on a
        // missing file below, with a non-zero expected size to compare against.
        if expectedLayerSize == 0 {
            return
        }
        let relativePath = "packed_experts/\(layer.file)"
        guard let manifestEntry = manifest.files[relativePath] else {
            throw RepackError.configurationInvalid(detail: "manifest missing \(relativePath)")
        }
        let actualSize = try access.fileSize(relativePath)
        if layout.compression == nil {
            guard manifestEntry.size == expectedLayerSize else {
                throw RepackError.configurationInvalid(
                    detail:
                        "\(relativePath) manifest size \(manifestEntry.size) != \(expectedLayerSize)"
                )
            }
            guard actualSize == expectedLayerSize else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(expectedLayerSize)")
            }
        } else {
            try validateCompressedLayer(
                layer, access: access, manifest: manifest, layout: layout,
                relativePath: relativePath, manifestEntry: manifestEntry,
                actualSize: actualSize, expectedLayerSize: expectedLayerSize)
        }
        guard let declaredExpertBits else {
            throw RepackError.configurationInvalid(detail: "manifest.json has no quant block")
        }
        var seenExperts = Set<Int>()
        for (index, expert) in layer.experts.enumerated() {
            let expertID = expert.expert ?? index
            guard expertID >= 0 && expertID < layout.expertsPerLayer else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) expert id out of range")
            }
            guard seenExperts.insert(expertID).inserted else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) duplicate expert \(expertID)")
            }
            guard expert.size > 0, expert.size <= layout.expertStride else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) expert \(expertID) size mismatch")
            }
            guard expert.offset % SSDAIFormatV1.alignmentBytes == 0 else {
                throw RepackError.configurationInvalid(
                    detail:
                        "\(relativePath) expert \(expertID) offset is not aligned to \(SSDAIFormatV1.alignmentBytes) bytes"
                )
            }
            // A compressed layer's file does not contain the raw offsets the
            // layout still records -- the frames do -- so the range check is
            // the reader's (each frame's end), done by `validateCompressedLayer`.
            if layout.compression == nil {
                guard expert.offset <= actualSize,
                    expert.size <= actualSize - expert.offset
                else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) range exceeds file size")
                }
            }
            try PackedExpertBytes.validate(
                expert,
                // The expert's own extent: equal to the layout stride for a
                // uniform install, smaller in a compact layer.
                stride: expert.size,
                declaredWeightBits: declaredExpertBits,
                in: relativePath,
                rank: expertID)
        }
    }

    /// A compressed layer: its index, every frame inside the file, and the sizes
    /// the manifest claims.
    ///
    /// The frames are not decompressed here. Their bytes are already covered by
    /// the file's SHA-256 in the manifest, and a decode that disagrees with the
    /// geometry is what the reader refuses; re-decoding 17 GB to prove what the
    /// hash already proved would make verification cost as much as the install.
    private static func validateCompressedLayer(
        _ layer: Layer,
        access: SSDAIDirectoryAccess,
        manifest: Manifest,
        layout: PackedExpertsLayout,
        relativePath: String,
        manifestEntry: ManifestFileEntry,
        actualSize: UInt64,
        expectedLayerSize: UInt64
    ) throws {
        guard manifestEntry.size == actualSize else {
            throw RepackError.configurationInvalid(
                detail: "\(relativePath) size \(actualSize) != manifest \(manifestEntry.size)")
        }
        guard actualSize > 0, actualSize < expectedLayerSize else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(relativePath) compressed size \(actualSize) is not smaller than the "
                    + "raw \(expectedLayerSize) bytes; the install is not compressed")
        }
        let rans = layout.compression == "rans"
        let indexPath =
            "packed_experts/"
            + (rans
                ? ExpertRANSIndex.sidecarName(forLayerFile: layer.file)
                : ExpertBlobIndex.sidecarName(forLayerFile: layer.file))
        guard let indexEntry = manifest.files[indexPath], indexEntry.size > 0 else {
            throw RepackError.configurationInvalid(
                detail: "manifest missing the index \(indexPath) of a compressed layer")
        }
        let indexSize = try access.fileSize(indexPath)
        guard indexSize == indexEntry.size else {
            throw RepackError.configurationInvalid(
                detail: "\(indexPath) size \(indexSize) != manifest \(indexEntry.size)")
        }
        if rans {
            let cap = UInt64(
                ExpertRANSIndex.headerBytes
                    + Int(ExpertRANSIndex.maximumBlockCount) * 512
                    + layout.expertsPerLayer * Int(ExpertRANSIndex.maximumBlockCount)
                    * ExpertRANSIndex.blockBytes)
            guard indexSize <= cap else {
                throw RepackError.configurationInvalid(
                    detail: "\(indexPath) is \(indexSize) bytes; the cap is \(cap)")
            }
            let data = try access.readPrefix(indexPath, maxBytes: cap)
            do {
                _ = try ExpertRANSIndex.decode(
                    data, expectedExperts: layout.expertsPerLayer,
                    expertStride: layout.expertStride, payloadSize: actualSize)
            } catch {
                throw RepackError.configurationInvalid(detail: "\(indexPath): \(error)")
            }
            return
        }
        guard
            let cap = ExpertBlobIndex.maximumIndexBytes(
                expertsPerLayer: layout.expertsPerLayer)
        else {
            throw RepackError.configurationInvalid(
                detail: "\(indexPath) cannot be sized for \(layout.expertsPerLayer) experts")
        }
        guard indexSize <= cap else {
            throw RepackError.configurationInvalid(
                detail: "\(indexPath) is \(indexSize) bytes; an index for "
                    + "\(layout.expertsPerLayer) experts is at most \(cap)")
        }
        let indexData = try access.readPrefix(indexPath, maxBytes: cap)
        let split = ExpertBlobIndex.phase1Bytes(
            downOffset: layer.experts.first?.tensors["down"]?.offset,
            expertStride: layout.expertStride)
        let decoded: ExpertBlobIndex.Layout
        do {
            decoded = try ExpertBlobIndex.decodeLayout(
                indexData, expectedCount: layout.expertsPerLayer,
                phase0Bytes: split ?? layout.expertStride,
                phase1Bytes: split.map { layout.expertStride - $0 })
        } catch {
            throw RepackError.configurationInvalid(detail: "\(indexPath): \(error)")
        }
        let end = ExpertBlobIndex.maximumEnd(decoded.entries)
        guard end > 0, end <= actualSize else {
            throw RepackError.configurationInvalid(
                detail: "\(indexPath) addresses \(end) bytes, past the "
                    + "\(actualSize)-byte layer file")
        }
    }
}
