import Foundation
import TinyTitanFormat

/// The manifest profile and the packed layout have to agree about how routed
/// experts are stored.
///
/// `layout.json` is the only place a per-expert width lives, and the reader
/// acts on it: a 2- or 3-bit tensor in a `standard` install is a *cold* tier
/// that the streamer expands into a wide slot, while in a `lowmem` install the
/// same bytes are read at 2 bits by the MoE kernels and must not be expanded.
/// The two interpretations of the same annotation are not interchangeable, so
/// a manifest that claims the wrong one is refused at load instead of being
/// run through the wrong arithmetic.
extension PackedExpertsLayout {

    /// The weight roles every routed expert carries.
    private static let roles = ["gate", "up", "down"]

    /// Validate the profile's storage recipe against the layout.
    ///
    /// - `lowmem`: every routed weight is annotated 2-bit, every expert fills
    ///   one uniform slot, and the layer is raw. The MoE kernels take a 2-bit
    ///   function constant, so no expansion runs.
    /// - `standard`: a cold (2/3-bit) expert is compact and is expanded into
    ///   expert 0's wide layout. That transform reads its *target* offsets
    ///   from expert 0, so expert 0 has to be wide. A uniform all-cold layer
    ///   under `standard` is exactly the lowmem recipe without its marker and
    ///   would be expanded into garbage; it is refused.
    func validateMemoryProfile(_ manifest: Manifest) throws {
        switch manifest.resolvedMemoryProfile {
        case .lowmem:
            try validateLowMemoryLayout(manifest)
        case .standard:
            try validateExpandableLayout(manifest)
        }
    }

    private func validateLowMemoryLayout(_ manifest: Manifest) throws {
        guard manifest.quant?.routedExpert.weightBits == 2 else {
            throw ModelError.indexCorrupt(
                detail: "a lowmem layout needs a 2-bit routedExpert manifest slot")
        }
        guard compression == nil else {
            throw ModelError.indexCorrupt(
                detail:
                    "a lowmem install cannot also be compressed: the compressed "
                    + "readers never run the expert width path")
        }
        guard expertsPerLayer > 0 else {
            throw ModelError.indexCorrupt(detail: "a lowmem install needs routed experts")
        }
        for layer in layers {
            guard layer.experts.count == expertsPerLayer else {
                throw ModelError.indexCorrupt(
                    detail: "layer \(layer.layer) expert count disagrees with the layout")
            }
            for (position, expert) in layer.experts.enumerated() {
                guard expert.size == expertStride else {
                    throw ModelError.indexCorrupt(
                        detail:
                            "lowmem expert \(expert.expert) is \(expert.size) bytes; "
                            + "the native 2-bit path needs one uniform "
                            + "\(expertStride)-byte slot")
                }
                guard expert.offset == UInt64(position) * expertStride else {
                    throw ModelError.indexCorrupt(
                        detail:
                            "lowmem expert \(expert.expert) is at offset "
                            + "\(expert.offset); the native path expects "
                            + "\(UInt64(position) * expertStride)")
                }
                for role in Self.roles {
                    guard let weight = expert.subTensors[role] else {
                        throw ModelError.indexCorrupt(
                            detail: "lowmem expert \(expert.expert) is missing \(role)")
                    }
                    guard weight.bits == 2 else {
                        throw ModelError.indexCorrupt(
                            detail:
                                "lowmem expert \(expert.expert) stores \(role) at "
                                + "\(weight.bits.map(String.init) ?? "inherit")-bit; "
                                + "the profile promises 2")
                    }
                }
                for role in Self.roles {
                    for component in ["_scales", "_biases"] {
                        let name = role + component
                        guard let tensor = expert.subTensors[name] else {
                            throw ModelError.indexCorrupt(
                                detail: "lowmem expert \(expert.expert) is missing \(name)")
                        }
                        guard tensor.bits == nil else {
                            throw ModelError.indexCorrupt(
                                detail:
                                    "lowmem expert \(expert.expert) annotates \(name) "
                                    + "with a weight width")
                        }
                    }
                }
            }
        }
    }

    /// A `standard` install may still be tiered (hot experts wide, cold ones
    /// compact). What it may not be is a uniform cold layer: the expansion
    /// transform takes its target offsets from expert 0, so expert 0 must be
    /// the wide reference.
    private func validateExpandableLayout(_ manifest: Manifest) throws {
        guard let firstLayer = layers.first, let wide = firstLayer.experts.first else {
            return
        }
        let wideBits = manifest.quant?.routedExpert.weightBits ?? 4
        var hasCold = false
        for layer in layers {
            for expert in layer.experts {
                guard let gate = expert.subTensors["gate"], let bits = gate.bits,
                    bits == 2 || bits == 3
                else { continue }
                hasCold = true
                for role in Self.roles {
                    guard let weight = expert.subTensors[role] else {
                        throw ModelError.indexCorrupt(
                            detail: "cold expert \(expert.expert) is missing \(role)")
                    }
                    guard weight.bits == bits else {
                        throw ModelError.indexCorrupt(
                            detail:
                                "cold expert \(expert.expert) mixes \(role) at "
                                + "\(weight.bits.map(String.init) ?? "inherit") bits "
                                + "with its \(bits)-bit gate")
                    }
                }
            }
        }
        guard hasCold else { return }
        guard
            wide.subTensors["gate"]?.bits == nil
                || wide.subTensors["gate"]?.bits == wideBits
        else {
            throw ModelError.indexCorrupt(
                detail:
                    "a tiered layer must keep expert 0 at the \(wideBits)-bit "
                    + "layout the expansion targets; its gate is annotated "
                    + "\(wide.subTensors["gate"]?.bits.map(String.init) ?? "inherit")-bit")
        }
    }
}
