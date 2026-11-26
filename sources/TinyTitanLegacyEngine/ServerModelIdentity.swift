import Foundation
import Legacy6xRuntime
import TinyTitanCore

package enum ServerModelIdentity {
    package static func base(manifestModelID: String, family: ModelFamily) -> String {
        var id = manifestModelID
        for suffix in ["-4bit", "-8bit", "-6bit"] where id.hasSuffix(suffix) {
            id = String(id.dropLast(suffix.count))
        }
        guard id == "unknown/snapshot" else { return id }
        switch family {
        case .qwen36: return "qwen3.6-35b-a3b"
        case .qwen36MTP: return "qwen3.6-35b-a3b-mtp"
        case .qwen38flash: return "qwen3.8-flash-next"
        case .qwen38flashMTP: return "qwen3.8-flash-next-mtp"
        case .qwen4flash: return "qwen4-omni-120b-think"
        case .qwen4flashMTP: return "qwen4-omni-120b-think-mtp"
        // Reached only for an install whose manifest carries no model id; the
        // dense converter always writes one.
        case .qwen35Dense: return "qwen3.5-dense"
        case .k2Horizon: return "k2-horizon"
        case .laguna: return "laguna-s-2-1"
        }
    }

    package static func apiModelID(
        manifestModelID: String, family: ModelFamily, weightBits: Int
    ) -> String {
        base(manifestModelID: manifestModelID, family: family) + "_\(weightBits)-Bit"
    }

    /// Human names for the installs this project ships. A manifest carries an
    /// id, not a name; an install outside this table is listed by its id.
    ///
    /// This lives with the 6.x adapter because the adapter is what still serves
    /// these families; a family that moves to its own plugin takes its name
    /// with it (`Ornith35B4BitPlugin` and `DenseQwen35Plugin` already do).
    package static let displayNames: [String: String] = [
        "qwen3.6-35b-a3b": "Qwen 3.6 35B-A3B",
        "ornith-1.5-35b-a3b": "Ornith 1.5 35B-A3B",
        "ornith-1.5-35b-a3b-bf16dense": "Ornith 1.5 35B-A3B (bf16 dense)",
        "qwen-agentworld": "Qwen AgentWorld 35B-A3B",
        "kat-coder-v2.5": "KAT-Coder-V2.5-Dev 35B-A3B",
        "qwen3.8-flash-next": "Qwen 3.8 Flash Next 125B-A6B",
        "qwen4-omni-120b-think": "Qwen 4.0 Flash (Qwen4-Omni-120B-Think)",
        // The dense CPU models. Their snapshots carried these names in
        // `config.json -> display_name`, which only the snapshot probe reads;
        // the install probe has a manifest, and a manifest carries an id, not
        // a name. Without these entries repacking them would silently rename
        // them from "Qwen 3.5 9B" to "qwen3.5-9b" in the app and the server's
        // /v1/models listing.
        "qwen3.5-2b": "Qwen 3.5 2B",
        "qwen3.5-4b": "Qwen 3.5 4B",
        "qwen3.5-9b": "Qwen 3.5 9B",
    ]

    /// Why the 6.x engine cannot serve an installed family, or nil when it can.
    ///
    /// A memory fit is not a capability claim: K2's install validates but its
    /// graph does not exist, and the catalog and doctor must say that rather
    /// than offer a model that stops at the loader's refusal.
    package static func unsupportedReason(_ family: ModelFamily) -> String? {
        switch family {
        case .qwen36MTP, .qwen38flashMTP, .qwen4flashMTP:
            return "MTP draft head, served only beside its target"
        case .k2Horizon:
            return "K2-Horizon is not served by this CLI: the server's probe "
                + "backend serves it, and `TinyTitanBench k2gpu` runs it "
                + "directly (docs/plan-k2-horizon.md)"
        case .laguna:
            return "Laguna-S-2.1's decode probe is implemented; the serving "
                + "graph is not, so the install validates but is not served "
                + "(docs/plan-laguna-s-2.1.md)"
        case .qwen36, .qwen38flash, .qwen4flash, .qwen35Dense:
            return nil
        }
    }

    /// The host-facing identity of a 6.x install, read from its manifest alone.
    package static func hostIdentity(directory: URL) throws -> ModelIdentity {
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        let base = base(manifestModelID: identity.modelID, family: identity.family)
        return ModelIdentity(
            modelID: identity.modelID,
            catalogID: apiModelID(
                manifestModelID: identity.modelID, family: identity.family,
                weightBits: identity.weightBits),
            displayName: displayNames[base] ?? base,
            family: identity.family.rawValue,
            weightBits: identity.weightBits,
            sampling: identity.family.engineSamplingProfile(identity: identity),
            reasoningControl: identity.family.engineReasoningControl,
            unsupportedReason: unsupportedReason(identity.family))
    }
}
