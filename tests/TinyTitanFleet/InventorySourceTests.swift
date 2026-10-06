import Foundation
import Testing

import TinyTitanFleetCore

/// `--from` reads an inventory the operator names, from a file or from a pipe, and
/// both branches are bounded (AUD-142 group 3).
///
/// The bound is checked as the bytes arrive rather than from an `fstat` size,
/// because a pipe has no size to consult — which is also why the file branch and
/// the stdin branch share this one code path: capping the file and reading stdin
/// to end-of-file would have protected neither, since the pipe is the same
/// allocation under a different name.
@Suite struct InventorySourceTests {

    private static func scratch(_ name: String, _ bytes: Data) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fleet-inventory-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        try bytes.write(to: url)
        return url
    }

    @Test func aDocumentWithinTheBoundReadsWhole() throws {
        let payload = Data(#"{"ok":true,"group":"tinytitan-lan"}"#.utf8)
        let url = try Self.scratch("inventory.json", payload)
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        #expect(try FleetInventorySource.read(fileAt: url) == payload)
    }

    @Test func theStreamRefusalNamesWhatItReadAndTheCeiling() throws {
        let url = try Self.scratch("inventory.json", Data(repeating: 0x78, count: 4096))
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        #expect {
            _ = try FleetInventorySource.readLimited(handle, path: "inventory.json", maxBytes: 100)
        } throws: { error in
            guard case FleetInventoryError.oversized(let path, let bytes, let cap) = error else {
                return false
            }
            return path == "inventory.json" && cap == 100 && bytes > 100
        }
    }

    /// The file branch uses the declared ceiling, and not a bound of its own. A
    /// sparse tail stands in for a planted file: 32 MiB and one byte of *size*, so
    /// the read has to stop on the way rather than after.
    @Test func theFileBranchUsesTheDeclaredCeiling() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fleet-inventory-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("inventory.json")
        try Data("{}".utf8).write(to: url)
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: UInt64(FleetInventorySource.maxBytes + 1))
        try handle.close()
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect {
            _ = try FleetInventorySource.read(fileAt: url)
        } throws: { error in
            guard case FleetInventoryError.oversized(_, let bytes, let cap) = error else {
                return false
            }
            return cap == FleetInventorySource.maxBytes && bytes > cap
        }
    }

    @Test func anAbsentOrUnreadableSourceSaysSoRatherThanReturningNothing() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fleet-inventory-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(
            throws: FleetInventoryError.unreadable(
                path: dir.appendingPathComponent(
                    "missing.json"
                ).path)
        ) {
            _ = try FleetInventorySource.read(fileAt: dir.appendingPathComponent("missing.json"))
        }
    }
}
