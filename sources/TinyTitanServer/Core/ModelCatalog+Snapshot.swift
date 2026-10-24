import Foundation
import TinyTitan
import TinyTitanLib

// The snapshot half of `ModelCatalog`'s probe, split out of `ModelCatalog.swift`
// (2026-10-24) under the 500-line-per-file rule. Pure code motion: the bodies,
// messages and access levels are unchanged except that `probeSnapshot` loses
// `private` because its caller stays in the other file.

extension ModelCatalog {

    /// A scanned directory's `config.json`, read under the manifest's bound.
    /// A probe walks every directory the scan finds, so the file in front of it
    /// is not a document this server was told to load, and it can be far larger
    /// than the bound before the bound is applied: a bound that checks the bytes
    /// *after* the read has already paid for them. K17: the size is `fstat`'d on
    /// the descriptor being read, so the cap fires before the allocation and
    /// still belongs to the file actually read.
    static func readSnapshotConfig(
        _ directory: URL,
        maxBytes: UInt64 = ManifestReader.defaultMaxBytes
    ) throws -> Data {
        do {
            return try BoundedMetadataRead.read(
                fileAt: directory.appendingPathComponent("config.json"),
                maxBytes: maxBytes)
        } catch ModelError.metadataOverBound(let document, let bytes, let cap) {
            // The catalog's own error type, because the probe below catches it
            // and turns it into the skip reason a scan reports.
            throw CPUModelBackend.CPUBackendError.unsupported(
                "\(document) is \(bytes) bytes, over the \(cap)-byte metadata bound")
        }
    }

    static func probeSnapshot(_ directory: URL) -> Result<Entry, ProbeFailure> {
        let config: [String: Any]
        do {
            let data = try readSnapshotConfig(directory)
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return .failure(ProbeFailure(reason: "config.json is not a JSON object"))
            }
            config = object
        } catch let error as CPUModelBackend.CPUBackendError {
            return .failure(ProbeFailure(reason: "\(error)"))
        } catch {
            return .failure(ProbeFailure(reason: "unreadable config.json: \(error)"))
        }
        let modelType = config["model_type"] as? String
        guard let family = CPUModelFamily.resolve(modelType: modelType) else {
            return .failure(ProbeFailure(reason: CPUModelFamily.refusal(modelType: modelType)))
        }
        guard let quantization = config["quantization"] as? [String: Any],
            let bits = quantization["bits"] as? Int
        else {
            return .failure(
                ProbeFailure(
                    reason:
                        "config.json has no quantization block; the CPU engine serves affine snapshots"
                ))
        }
        // Snapshots are written in place by a converter, so a config can
        // exist before the rest of the directory does.
        for required in ["model.safetensors.index.json", "tokenizer.json"]
        where !FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(required).path)
        {
            return .failure(ProbeFailure(reason: "incomplete snapshot: no \(required)"))
        }
        let declared = (config["model_id"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        let id = declared ?? directory.lastPathComponent
        let name = (config["display_name"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? id
        return .success(
            Entry(
                id: id,
                name: name,
                kind: .cpu(family),
                quant: bits,
                path: directory,
                sampling: family.samplingDefaults,
                contextLimit: config["max_position_embeddings"] as? Int,
                sizeBytes: sizeOnDisk(directory)))
    }
}
