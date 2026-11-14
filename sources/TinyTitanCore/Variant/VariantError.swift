import Foundation

/// Errors a variant target may throw that are generic across variants. A
/// variant that has not yet had its graph ported refuses to serve rather than
/// silently falling back to another engine path (7.0 spec §3).
public enum VariantError: Error, CustomStringConvertible {
    /// The variant's graph has not been ported yet; it compiles and is gated,
    /// but it does not serve.
    case notPorted(String)

    public var description: String {
        switch self {
        case .notPorted(let id):
            return "variant \(id) has no ported graph yet; it does not serve"
        }
    }
}
