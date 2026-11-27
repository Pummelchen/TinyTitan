import Foundation

// The catalog's lookups: a base id or the path of a directory the catalog
// holds. Split out of `ModelCatalog.swift` (2026-11-20) under the 500-line
// rule; pure code motion. The `<id>@cpu` / `@gpu` engine aliases left with the
// CPU engine (TT-142 slice (b), 2026-11-28): every served install has one
// engine now, so there is no second spelling of an id.
extension ModelCatalog {
    // MARK: - Lookup

    public func entry(id: String) -> Entry? {
        entries.first { $0.id == id }
    }

    /// The entry `--model` names: a catalog id or the path of a directory the
    /// catalog holds.
    public func entry(idOrPath value: String) -> Entry? {
        if let byID = entry(id: value) { return byID }
        let path = URL(fileURLWithPath: value).standardizedFileURL.resolvingSymlinksInPath().path
        return entries.first { $0.path.resolvingSymlinksInPath().path == path }
    }
}
