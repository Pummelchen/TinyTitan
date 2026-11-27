import Foundation

/// A pinned upstream checkpoint the installer knows how to repack. Each value
/// fixes the repo, revision and index fingerprint so installs are exactly
/// reproducible.
public struct SupportedModelSource: Sendable, Equatable {
    /// CLI selector value (`--model <name>`).
    public let name: String
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    /// Value recorded as `manifest.modelID` when the source fingerprint matches.
    public let modelID: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64
    /// Set for an Inkling 2 source, which the installer plans from its own
    /// tensors instead of affine index metadata.
    public var inkling2: RemoteInkling2Source?

    public func installOptions(
        outputDirectory: URL,
        overwrite: Bool,
        token: String?,
        resume: Bool = false
    )
        -> RemoteStreamingRepackOptions
    {
        RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume,
            installDraftHead: false,
            inkling2: inkling2)
    }

    /// Download estimate covers the `language_model.*` tensors plus tokenizer
    /// and metadata sidecars. Installed bytes add the resident index and
    /// per-expert 16 KB page rounding plus layout/manifest sidecars.

    /// Text-only Ornith 1.5 uses the same qwen3_5_moe tensor contract as the
    /// Qwen 3.6 target. The official MLX conversion intentionally contains
    /// only `language_model.*` tensors; vision and MTP are separate follow-up
    /// features rather than silently entering the text runtime.
    public static let ornith15 = SupportedModelSource(
        name: "ornith15",
        displayName: "Ornith 1.5 35B-A3B 4-bit (text)",
        repoID: "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
        revision: "19504d912fa8fc7622bf6b1de3db5d5d890b1f02",
        sourceIndexSHA256:
            "c118f13c0dcb729e4ca2e3d653ab193067551eb1a6410badb5192eb426104f36",
        modelID: "ornith-1.5-35b-a3b-4bit",
        approximateDownloadBytes: 19_528_995_943,
        installedBytes: 19_530_000_000,
        reserveBytes: 1_073_741_824)

    /// One-layer native Qwen3.6 MTP draft. It contains no embedding or LM
    /// head; the runtime reuses those tensors from whichever 4/6/8-bit target
    /// is loaded. The routed experts remain SSD-streamed in `.ssdai` form.

    /// Qwen3.8-Flash-Next, the Qwen4 preview architecture. Community-quantized
    /// rather than mlx-community: no official or mlx-community MLX release
    /// exists, and this artifact was verified against the official bf16
    /// weights before being pinned -- every 4-bit tensor lands within 1.02x of
    /// an ideal affine g64 round-trip, i.e. as accurate as the format allows
    /// (`benchmark/tinytitan_quant_fidelity.py`). Group size is 64, matching the
    /// format TinyTitan already uses.
    ///
    /// `approximateDownloadBytes` covers the backbone plus the 102.4 GB fp16
    /// n-gram table; the MTP draft installs separately from its own shard.

    /// The Qwen3.8-Flash-Next MTP draft head.
    ///
    /// Same repository and revision as the target, because the draft ships
    /// inside it -- but in its own `mtp-weights.safetensors` shard, so the
    /// install fetches 1.37 GB rather than re-reading the backbone. It has no
    /// embedding or head of its own; both are shared from the target it
    /// drafts for.

    /// Inkling 2 Mini from Unsloth's Q4_K_XL GGUF (TT-121): routed experts
    /// NVFP4 (Q8_0 in the last four layers), the rest Q8_0/F32/BF16, all kept
    /// as the GGUF stores them. The fingerprint is the SHA-256 of the GGUF's
    /// header (metadata, tokenizer and tensor table), which pins every tensor's
    /// encoding and offset. Converted, not yet served: the variant that serves
    /// it is TT-121 phase 2.
    public static let inkling2GGUFQ4 = SupportedModelSource(
        name: "inkling-q4-k-xl-unsloth",
        displayName: "Inkling 2 Mini UD-Q4_K_XL (Unsloth)",
        repoID: "unsloth/Inkling-2-Mini-GGUF",
        revision: "45fd08764f2b0bf16b03a5483a80e5c7aaec8772",
        sourceIndexSHA256:
            "e35ee6baa44933adb00cd5196be291a80c7e6b35aa75c2b01549b056057387e0",
        modelID: "inkling-2-mini-gguf-q4",
        approximateDownloadBytes: 17_629_759_456,
        installedBytes: 17_664_255_634,
        reserveBytes: 1_073_741_824,
        inkling2: .gguf(fileName: "Inkling-2-Mini-UD-Q4_K_XL.gguf"))

    /// Inkling 2 Mini from the original ModelOpt checkpoint (TT-121): routed
    /// experts NVFP4 (MXFP8 in the last four layers), the rest BF16/F32, the
    /// MTP head beside it, the vision tower left out. The fingerprint is the
    /// SHA-256 of `config.json` and both safetensors headers. Converted, not
    /// yet served: the variant that serves it is TT-121 phase 3.
    public static let inkling2NVFP4 = SupportedModelSource(
        name: "inkling2mini",
        displayName: "Inkling 2 Mini",
        repoID: "thinkingmachines/Inkling-2-Mini",
        revision: "ed40e7da625f594fc9c4990f5f66be24873c22d7",
        sourceIndexSHA256:
            "0b94cadc1237f8d183c2fabf35dd610d4aa7364862be2afda0371eaacef45750",
        modelID: "inkling-2-mini-nvfp4",
        approximateDownloadBytes: 19_555_000_000,
        installedBytes: 19_592_286_535,
        reserveBytes: 1_073_741_824,
        inkling2: .checkpoint)

    /// Default source when no `--model` selector is given: the shipped 4-bit
    /// Ornith, which is the matrix's default install.
    public static let `default` = ornith15

    /// The shipped matrix: Ornith 1.5 35B-A3B 4-bit and the two Inkling 2
    /// Mini installs. The sources for the retired families (Qwen 3.6/3.8,
    /// Ornith 8-bit and the MTP draft heads) left with TT-142.
    public static let all: [SupportedModelSource] = [
        ornith15, inkling2GGUFQ4, inkling2NVFP4,
    ]

    public static func named(_ name: String) -> SupportedModelSource? {
        if let direct = all.first(where: { $0.name == name }) { return direct }
        switch name.lowercased() {
        case "inkling2-gguf-q4", "inkling_q4_k_xl_unsloth", "inkling2-ud-q4-k-xl-unsloth":
            return inkling2GGUFQ4
        case "inkling2-nvfp4":
            return inkling2NVFP4
        default:
            return nil
        }
    }
}
