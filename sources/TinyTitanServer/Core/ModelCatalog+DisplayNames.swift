import Foundation
import TinyTitanLegacyEngine

/// The display-name table, split out of `ModelCatalog.swift` (2026-10-25) as
/// pure code motion under the 500-line-per-file rule. TT-134 moved the table
/// itself to `ServerModelIdentity`, beside the 6.x adapter that still serves
/// those families; this alias keeps the catalog's own callers and test reading
/// unchanged until those families have plugins.
extension ModelCatalog {
    static let displayNames = ServerModelIdentity.displayNames
}
