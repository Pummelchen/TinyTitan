import Foundation

/// How an install spends memory on its routed experts.
///
/// `standard` is every install written before this field existed: routed
/// experts are stored at the width the manifest's `routedExpert` slot names
/// (4 or 8 bits), and the kernels read that width directly. A *tiered* install
/// (some hot experts wide, cold ones compact) is also `standard`: its
/// `packed_experts/layout.json` records the per-tensor width and the streamer
/// expands cold blobs into the wide slot, so the manifest slot still names the
/// width the kernels run.
///
/// `lowmem` means the routed experts themselves are stored at 2 bits and are
/// read at 2 bits -- no expansion step. The MoE kernels already take the width
/// as a function constant (2/4/8), so this is a real decode path rather than a
/// narrower file the runtime has to inflate first. Resident tensors keep their
/// own widths: this profile only changes the routed-expert tier.
///
/// It is **experimental**. A model-wide 2-bit cold tier built by remapping a
/// 4-bit install was measured to diverge (docs/residency-and-bottleneck-
/// verdict-2026-10-20.md, "Cold-tier verdict"), and `lowmem` is that same
/// 2-bit weight perturbation without the expansion step. The mechanics are
/// supported; the quality is not established, so the catalog refuses to serve
/// a `lowmem` install unless `TINYTITAN_ALLOW_EXPERIMENTAL_LOWMEM=1` is set.
public enum SSDAIMemoryProfile: String, Codable, Sendable, Equatable, CaseIterable {
    case standard
    case lowmem

    /// The environment switch an operator sets to serve a `lowmem` install.
    public static let experimentalOptInVariable = "TINYTITAN_ALLOW_EXPERIMENTAL_LOWMEM"

    /// Whether serving this profile needs the explicit experimental opt-in.
    public var requiresExperimentalOptIn: Bool { self == .lowmem }

    /// Whether the environment has enabled serving this profile.
    ///
    /// `standard` never needs the switch. `lowmem` is enabled only by the exact
    /// value `1`, so an empty or misspelled variable fails closed.
    public func isServable(environment: [String: String]) -> Bool {
        guard requiresExperimentalOptIn else { return true }
        return environment[Self.experimentalOptInVariable] == "1"
    }
}

extension SSDAIManifestV1 {
    /// The profile a manifest written before this field existed means.
    package var resolvedMemoryProfile: SSDAIMemoryProfile {
        memoryProfile ?? .standard
    }

    /// Enforce the profile/width contract the writer and the reader share.
    ///
    /// The lowmem profile is the only place a 2-bit routed-expert slot is
    /// legal, and it is legal there because the MoE kernels accept a 2-bit
    /// function constant. Every other slot still names a width a resident
    /// kernel implements (4 or 8 bits), so it is refused here rather than
    /// discovered later as a wrong stride.
    package func validateMemoryProfile() throws {
        let profile = resolvedMemoryProfile
        guard let quant else {
            if profile == .lowmem {
                throw TinyTitanFormatError.invalid(
                    field: "manifest.memoryProfile",
                    reason: "a lowmem install must declare its quant slots")
            }
            return
        }
        switch profile {
        case .standard:
            if quant.routedExpert.weightBits == 2 {
                throw TinyTitanFormatError.invalid(
                    field: "manifest.quant.routedExpert.weightBits",
                    reason:
                        "2-bit routed experts are the lowmem profile, and this "
                        + "manifest does not declare memoryProfile \"lowmem\". "
                        + "Without the marker the streamer would expand these "
                        + "bytes again and read them wrongly")
            }
        case .lowmem:
            guard expertsPerLayer > 0, expertStride > 0 else {
                throw TinyTitanFormatError.invalid(
                    field: "manifest.memoryProfile",
                    reason: "a lowmem install needs routed experts (expertsPerLayer > 0)")
            }
            guard quant.routedExpert.weightBits == 2 else {
                throw TinyTitanFormatError.invalid(
                    field: "manifest.quant.routedExpert.weightBits",
                    reason:
                        "a lowmem install stores routed experts at 2 bits, not "
                        + "\(quant.routedExpert.weightBits)")
            }
            for (name, slot) in [
                ("embedding", quant.embedding),
                ("attention", quant.attention),
                ("router", quant.router),
                ("sharedExpert", quant.sharedExpert),
            ] {
                guard slot.weightBits == 4 || slot.weightBits == 8 else {
                    throw TinyTitanFormatError.invalid(
                        field: "manifest.quant.\(name)",
                        reason:
                            "a lowmem profile only narrows the routed experts; "
                            + "\(name) is \(slot.weightBits)-bit, and the resident "
                            + "kernels implement 4 or 8 bits")
                }
            }
        }
    }
}
