import Foundation
import TinyTitanCore
import TinyTitanFormat

// Split out of `PackedExpertLayoutVerification.swift` under the 500-line-per-file
// rule (AUD-0007): layout validation and the expert-byte arithmetic are two
// concerns, and the file crossed the limit when the capacity guard and the
// format validator were added. Pure code motion — same bodies, same messages.

/// The arithmetic that ties one packed expert's bytes to the width it claims.
enum PackedExpertBytes {

    /// Prove `declaredWeightBits` against one expert's own slices.
    ///
    /// A packed-expert install keeps its routed-expert widths here rather than in
    /// the resident index, which is why `validateQuantAgainstResident` steps
    /// aside for it -- and until now nothing read what this file says. So the
    /// slot that becomes the `_<bits>-Bit` name in `/v1/models`, and that the
    /// dequantizer and the Metal pipeline constant are chosen from, was compared
    /// against nothing for exactly the shapes the product ships. The writer's own
    /// comment names the failure it invites: the word count changes, the strides
    /// still divide evenly, every shape check passes, and the model answers
    /// fluently and wrongly.
    ///
    /// Three arithmetic facts, none of which asks to be trusted:
    ///
    ///   - a U32 tensor of `n` elements in `bits`-bit packing is
    ///     `n * bits / 8` bytes, so its byte extent *implies* its width;
    ///   - that implied width has to equal the tensor's own `bits` annotation
    ///     where it carries one, and the declared slot for every one of them;
    ///   - the slices of an expert fill its stride once, with page padding
    ///     allowed at the end and nowhere else — so a missing, duplicated or
    ///     wrongly-sized slice shows up as a sum that does not round to the
    ///     stride rather than as a file that still divides evenly.
    ///
    /// A BF16 tensor is two bytes an element and has no width to agree with.
    ///
    /// The bound on a *derived* width is the format layer's own (1...32 bits),
    /// not the 4-or-8 the resident check insists on. That gap is deliberate:
    /// 6-bit was withdrawn as a format, and an install whose bytes match its
    /// description is a truthful install even when the runtime will not load it.
    /// Refusing it here would make the verifier stricter than the load path, with
    /// a message telling the user to re-download weights that are fine.
    static func validate(
        _ expert: Expert,
        stride: UInt64,
        declaredWeightBits: Int,
        in file: String,
        rank: Int
    ) throws {
        var blobBytes: UInt64 = 0
        // Sorted so the first complaint is the same on every run: dictionary
        // order is not, and a verifier whose message depends on it cannot be
        // compared between two machines.
        for (name, tensor) in expert.tensors.sorted(by: { $0.key < $1.key }) {
            let at = "\(file) expert \(rank) tensor \(name)"
            let elements = try elementCount(of: tensor, at: at)
            switch tensor.dtype {
            case "U32":
                let implied = try impliedBits(of: tensor, elements: elements, at: at)
                if let annotated = tensor.bits, annotated != implied {
                    throw RepackError.configurationInvalid(
                        detail:
                            "\(at): annotated \(annotated)-bit but its \(tensor.size) bytes "
                            + "at \(describe(tensor.shape)) are \(implied)-bit. Repack it "
                            + "from its source snapshot; the bytes are fine, the "
                            + "description of them is not")
                }
                guard implied == declaredWeightBits else {
                    throw RepackError.configurationInvalid(
                        detail:
                            "\(at): its \(tensor.size) bytes at \(describe(tensor.shape)) are "
                            + "\(implied)-bit, while the install declares a routed-expert "
                            + "width of \(declaredWeightBits). The dequantizer's word "
                            + "count and the Metal pipeline constant are both taken from "
                            + "that declaration, so this install would read these bytes "
                            + "wrongly")
                }
            case "BF16":
                let (expected, byteOverflow) = elements.multipliedReportingOverflow(by: 2)
                guard !byteOverflow else {
                    throw RepackError.configurationInvalid(
                        detail:
                            "\(at): \(describe(tensor.shape)) is too many bf16 values to count "
                            + "in bytes")
                }
                guard expected == tensor.size else {
                    throw RepackError.configurationInvalid(
                        detail:
                            "\(at): \(tensor.size) bytes at \(describe(tensor.shape)) is not "
                            + "the \(expected) bytes a bf16 tensor of that shape is")
                }
            case let dtype
            where SSDAIBlockEncodingV1(rawValue: dtype) != nil
                || SSDAIAuxiliaryDTypeV1(rawValue: dtype) != nil:
                // A block encoding or a scale type: its bytes are fixed by its
                // shape. Its width is the encoding's own, so the routed-expert
                // slot is not compared -- Inkling keeps its last four layers
                // 8-bit beside 4-bit ones, each tensor naming its encoding.
                do {
                    try SSDAIPackedExpertsLayoutV1.validateSubTensorSize(tensor, field: at)
                } catch {
                    throw RepackError.configurationInvalid(detail: "\(error)")
                }
                if let encoding = SSDAIBlockEncodingV1(rawValue: dtype),
                    let annotated = tensor.bits, annotated != encoding.weightBits
                {
                    throw RepackError.configurationInvalid(
                        detail:
                            "\(at): annotated \(annotated)-bit but \(dtype) is \(encoding.weightBits)-bit"
                    )
                }
            default:
                throw RepackError.configurationInvalid(
                    detail: "\(at): unknown dtype \(tensor.dtype)")
            }
            let (sum, blobOverflow) = blobBytes.addingReportingOverflow(tensor.size)
            guard !blobOverflow else {
                throw RepackError.configurationInvalid(detail: "\(at): expert byte total overflows")
            }
            blobBytes = sum
        }
        let padded = try roundedToStride(blobBytes)
        guard padded == stride else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(file) expert \(rank): its tensors total \(blobBytes) bytes, which "
                    + "pads to \(padded) rather than the \(stride)-byte expert stride. "
                    + "Either a slice is missing, duplicated, or a different width than "
                    + "the one declared")
        }
    }

    /// The element count a tensor's own shape describes.
    private static func elementCount(of tensor: SSDAISubTensorV1, at: String) throws -> UInt64 {
        guard !tensor.shape.isEmpty, tensor.shape.allSatisfy({ $0 > 0 }) else {
            throw RepackError.configurationInvalid(
                detail: "\(at): shape \(describe(tensor.shape)) has no elements")
        }
        var elements: UInt64 = 1
        for dimension in tensor.shape {
            let (product, overflow) = elements.multipliedReportingOverflow(by: UInt64(dimension))
            guard !overflow else {
                throw RepackError.configurationInvalid(
                    detail: "\(at): shape \(describe(tensor.shape)) overflows")
            }
            elements = product
        }
        return elements
    }

    /// The width a U32 tensor's byte extent implies, in bits per element.
    ///
    /// `size * 8 / elements`, and only when it divides: a byte extent that is not
    /// a whole number of packed values per element is a broken layout, not a width
    /// to guess at.
    private static func impliedBits(
        of tensor: SSDAISubTensorV1, elements: UInt64, at: String
    ) throws -> Int {
        guard tensor.size > 0 else {
            throw RepackError.configurationInvalid(detail: "\(at): zero-length tensor")
        }
        let (bits, bitsOverflow) = tensor.size.multipliedReportingOverflow(by: 8)
        guard !bitsOverflow else {
            throw RepackError.configurationInvalid(
                detail: "\(at): \(tensor.size) bytes overflow a bit count")
        }
        guard bits % elements == 0 else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(at): \(tensor.size) bytes at \(describe(tensor.shape)) does not divide "
                    + "into a whole number of values per element")
        }
        let implied = bits / elements
        guard implied > 0, implied <= 32 else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(at): implied width \(implied) is outside the 1...32 bits a packed "
                    + "tensor can hold")
        }
        return Int(implied)
    }

    /// The stride a blob of `bytes` is cut to: the next multiple of the page.
    ///
    /// The same rounding the planner applies when it lays an expert out, which is
    /// why a 35B expert whose bytes fill their stride exactly and a 125B expert
    /// padded by 4 KiB both pass, and a blob a page short or long does not.
    private static func roundedToStride(_ bytes: UInt64) throws -> UInt64 {
        let page = SSDAIFormatV1.alignmentBytes
        let (padded, overflow) = bytes.addingReportingOverflow(page - 1)
        guard !overflow else {
            throw RepackError.configurationInvalid(
                detail: "expert byte total \(bytes) cannot be padded to a page")
        }
        return (padded / page) * page
    }

    private static func describe(_ shape: [UInt32]) -> String {
        shape.map(String.init).joined(separator: "x")
    }
}
