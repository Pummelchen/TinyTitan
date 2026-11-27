import ContinuityCore
import Foundation

/// The result of looking for project files: the readable ones and the ones
/// that are present but could not be loaded.
public struct MemoryProjectDiscovery: Sendable {
    /// A file that is present but could not be read. `reason` is a safe
    /// description — a path and an errno, or the rejected line numbers — and
    /// never record contents.
    public struct Broken: Sendable, Equatable {
        public let url: URL
        public let workspace: String
        public let reason: String

        public init(url: URL, workspace: String, reason: String) {
            self.url = url
            self.workspace = workspace
            self.reason = reason
        }
    }

    /// The directory itself could not be listed. Distinct from `broken`: that
    /// is a file the scan found and could not read; this is a scan that did
    /// not happen, so `projects` and `broken` say nothing about the contents.
    /// `reason` is a safe description — "does not exist", "is not a
    /// directory", or an errno description — and never a file's contents.
    public struct ListingFailure: Sendable, Equatable {
        public let url: URL
        public let reason: String

        public init(url: URL, reason: String) {
            self.url = url
            self.reason = reason
        }
    }

    public let projects: [MemoryProjectFile]
    public let broken: [Broken]
    /// Nil when the directory was listed, whatever the files turned out to be.
    public let listingFailure: ListingFailure?
    /// A directory *below* the root that could not be walked, with the same
    /// safe reason as `listingFailure`. The store layout is one `.ndjson` per
    /// workspace at the top, but the scan has always descended, and a
    /// directory it could not enter used to vanish without a word: that is the
    /// same defect as an unlistable root, one level down.
    public let unlistableDirectories: [ListingFailure]

    public init(
        projects: [MemoryProjectFile], broken: [Broken],
        listingFailure: ListingFailure? = nil,
        unlistableDirectories: [ListingFailure] = []
    ) {
        self.projects = projects
        self.broken = broken
        self.listingFailure = listingFailure
        self.unlistableDirectories = unlistableDirectories
    }

    /// Whether the scan found nothing *and* could see the whole tree. A
    /// listing failure is not an empty directory, and a caller that shows the
    /// result to a person has to handle it first.
    public var isEmpty: Bool {
        projects.isEmpty && broken.isEmpty && listingFailure == nil
            && unlistableDirectories.isEmpty
    }
}

/// A project file read without opening it for writing.
///
/// This is what a person uses the day memory says something wrong: list the
/// facts, read one, see how it changed. It takes no lock, so it works on the
/// file of a server that is running, and it folds the journal the same way
/// the engine does on start -- a checkpoint resets, a memory record upserts,
/// a version extends the history.
///
/// A file can be damaged: a complete line that does not decode is skipped by
/// the journal format and the rest is still readable. That is not the same as
/// a clean file, so `damagedRecordLines` names the lines that were rejected.
/// The facts here are what *did* decode; a person reading them is owed the
/// knowledge that the view is incomplete.
public struct MemoryProjectFile: Sendable {
    public struct Fact: Sendable, Equatable {
        public let key: String
        public let value: String
        public let version: Int
        public let status: String
        public let updatedAt: Date
        /// Earlier values, oldest first.
        public let history: [(version: Int, value: String)]

        public static func == (lhs: Fact, rhs: Fact) -> Bool {
            lhs.key == rhs.key && lhs.version == rhs.version && lhs.value == rhs.value
        }
    }

    public let url: URL
    /// The workspace id, which is the file's stem.
    public let workspace: String
    public let title: String?
    public let facts: [Fact]
    public let sessionCount: Int
    public let eventCount: Int
    public let bytesOnDisk: Int
    public let modifiedAt: Date
    /// 1-based lines that were complete but did not decode. Empty for a clean
    /// file. The facts above come only from the lines that did decode.
    public let damagedRecordLines: [Int]

    /// Every project file under a memory directory, newest first.
    ///
    /// The loadable ones. A file that is present but cannot be read is not
    /// part of this list — that is what `discoverAll` is for, and a caller
    /// that shows projects to a person should be using it. Keeping this
    /// signature for callers that only want the readable set is the point of
    /// the split; silently omitting a broken file from the *tool* is the
    /// defect this replaced.
    public static func discover(in directory: URL) -> [MemoryProjectFile] {
        discoverAll(in: directory).projects
    }

    /// Every project file under a memory directory, loadable or not.
    ///
    /// `projects` are the files that parsed, newest first. `broken` are the
    /// files that are present but could not be opened or decoded, with a safe
    /// reason (a `JournalError` description: a path and an errno, or the line
    /// numbers of records that did not decode — never record contents).
    ///
    /// A project that is present but unreadable is not a missing project. The
    /// tool lists it in its warnings and refuses a command for it with the
    /// reason, so an operator can repair or remove it instead of wondering why
    /// the workspace vanished.
    public static func discoverAll(in directory: URL) -> MemoryProjectDiscovery {
        // A directory that cannot be listed is not an empty one. The
        // existence/type/permission checks come first so the common failures
        // carry a stable reason ("does not exist", "Permission denied") rather
        // than whatever a listing call reports; the walk repeats the check for
        // every level below the root.
        if let reason = listingReason(for: directory) {
            return MemoryProjectDiscovery(
                projects: [], broken: [],
                listingFailure: MemoryProjectDiscovery.ListingFailure(
                    url: directory, reason: reason))
        }
        var files: [URL] = []
        var unlistable: [MemoryProjectDiscovery.ListingFailure] = []
        walk(directory, files: &files, unlistable: &unlistable)
        var projects: [MemoryProjectFile] = []
        var broken: [MemoryProjectDiscovery.Broken] = []
        for url in files {
            do {
                projects.append(try load(url))
            } catch {
                // `load` throws only `JournalError`; this catch is its one
                // exhaustive handler, and it records the failure instead of
                // dropping it.
                broken.append(
                    MemoryProjectDiscovery.Broken(
                        url: url,
                        workspace: url.deletingPathExtension().lastPathComponent,
                        reason: error.description))
            }
        }
        return MemoryProjectDiscovery(
            projects: projects.sorted { $0.modifiedAt > $1.modifiedAt },
            broken: broken.sorted {
                ($0.workspace, $0.url.path) < ($1.workspace, $1.url.path)
            },
            unlistableDirectories: unlistable.sorted { $0.url.path < $1.url.path })
    }

    /// Every `.ndjson` file below `directory`, and every directory that could
    /// not be entered.
    ///
    /// This was `FileManager.enumerator`: that walker yields nothing for a
    /// directory it cannot descend into and offers no error, so an unreadable
    /// *nested* directory was indistinguishable from an empty one and the
    /// answer read as complete. `contentsOfDirectory` throws, so the walk can
    /// name the directory it skipped and why. The worklist is explicit rather
    /// than recursive: a directory tree is input like any other, and its depth
    /// is not this process's to bound.
    private static func walk(
        _ directory: URL, files: inout [URL],
        unlistable: inout [MemoryProjectDiscovery.ListingFailure]
    ) {
        var pending = [directory]
        while let next = pending.popLast() {
            let entries: [URL]
            do {
                entries = try FileManager.default.contentsOfDirectory(
                    at: next,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: [.skipsHiddenFiles])
            } catch {
                // The pre-check on the root produces the stable reasons
                // ("Permission denied"); this is the backstop for a directory
                // that becomes unreadable between the check and the listing,
                // and for the levels below the root.
                let code = (error as NSError).code
                let reason =
                    (error as NSError).domain == NSPOSIXErrorDomain
                    ? String(cString: strerror(Int32(code)))
                    : "an I/O error"
                unlistable.append(
                    MemoryProjectDiscovery.ListingFailure(url: next, reason: reason))
                continue
            }
            for url in entries {
                // A name ending in `.ndjson` is a project path whatever kind
                // of entry it is: a directory or FIFO there is the 0061 case,
                // reported by `load` as broken rather than walked into.
                if url.pathExtension == "ndjson" {
                    files.append(url)
                    continue
                }
                let isDirectory =
                    (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
                guard isDirectory else { continue }
                if let reason = listingReason(for: url) {
                    unlistable.append(
                        MemoryProjectDiscovery.ListingFailure(url: url, reason: reason))
                } else {
                    pending.append(url)
                }
            }
        }
    }

    /// Why `directory` cannot be listed, or nil when it can.
    ///
    /// A safe fragment — "does not exist", "is not a directory", or an errno
    /// description — and never a file's contents.
    private static func listingReason(for directory: URL) -> String? {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory)
        else { return "does not exist" }
        guard isDirectory.boolValue else { return "is not a directory" }
        let accessible = access(directory.path, R_OK | X_OK)
        guard accessible == 0 else {
            let code = errno
            return String(cString: strerror(code))
        }
        return nil
    }

    /// Picks a project by exact workspace id or by a unique prefix, so
    /// `photograph` finds `photograph-851a1c1a`.
    ///
    /// A name that matches a file which could not be read answers
    /// `.unreadable` with that file's reason, rather than "no project matches"
    /// — the project does exist, and saying otherwise sends an operator
    /// looking for a file that is right there.
    public static func resolve(_ name: String, among files: [MemoryProjectFile])
        -> Result<MemoryProjectFile, ResolveError>
    {
        resolve(name, among: files, broken: [])
    }

    public static func resolve(
        _ name: String, among files: [MemoryProjectFile],
        broken: [MemoryProjectDiscovery.Broken]
    ) -> Result<MemoryProjectFile, ResolveError> {
        if let exact = files.first(where: { $0.workspace == name }) { return .success(exact) }
        if let exact = broken.first(where: { $0.workspace == name }) {
            return .failure(.unreadable(exact.workspace, reason: exact.reason))
        }
        let matches = files.filter { $0.workspace.hasPrefix(name) }
        let brokenMatches = broken.filter { $0.workspace.hasPrefix(name) }
        switch matches.count + brokenMatches.count {
        case 0:
            return .failure(.notFound(name))
        case 1:
            if let file = matches.first { return .success(file) }
            let file = brokenMatches[0]
            return .failure(.unreadable(file.workspace, reason: file.reason))
        default:
            let names = matches.map(\.workspace) + brokenMatches.map(\.workspace)
            return .failure(.ambiguous(name, names.sorted()))
        }
    }

    public enum ResolveError: Error, CustomStringConvertible {
        case notFound(String)
        case ambiguous(String, [String])
        /// The file exists but could not be loaded. Carries only the safe
        /// reason `discoverAll` collected.
        case unreadable(String, reason: String)
        public var description: String {
            switch self {
            case .notFound(let name): return "no project matches '\(name)'"
            case .ambiguous(let name, let candidates):
                return "'\(name)' matches several projects: " + candidates.joined(separator: ", ")
            case .unreadable(let name, let reason):
                return
                    "project '\(name)' is present but could not be read: \(reason)\n"
                    + "Repair the file or remove it, then try again."
            }
        }
    }

    public static func load(_ url: URL) throws(JournalError) -> MemoryProjectFile {
        let replay = try FileJournal.readDetailed(contentsOf: url)
        let records = replay.records
        var items: [String: MemoryItem] = [:]
        var versions: [String: [(Int, String)]] = [:]
        var tasks: [UUID: ContinuityTask] = [:]
        var sessions = Set<UUID>()
        var events = 0
        for record in records {
            switch record {
            case .checkpoint(let log, let memory):
                items = [:]
                versions = [:]
                tasks = [:]
                sessions = []
                for task in log.tasks { tasks[task.id] = task }
                for session in log.sessions { sessions.insert(session.id) }
                events = log.events.count
                for item in memory.items { items[item.address] = item }
                for version in memory.versions {
                    versions["\(version.namespace).\(version.key)", default: []]
                        .append((version.version, version.value))
                }
            case .task(let task): tasks[task.id] = task
            case .session(let session): sessions.insert(session.id)
            case .event: events += 1
            case .memory(let item): items[item.address] = item
            case .memoryVersion(let version):
                versions["\(version.namespace).\(version.key)", default: []]
                    .append((version.version, version.value))
            }
        }
        let facts = items.values.map { item in
            Fact(
                key: ContinuityStore.keyText(for: item),
                value: item.value,
                version: item.version,
                status: item.status.rawValue,
                updatedAt: item.updatedAt,
                history: (versions[item.address] ?? []).sorted { $0.0 < $1.0 }
                    .map { (version: $0.0, value: $0.1) })
        }.sorted { $0.key < $1.key }
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        return MemoryProjectFile(
            url: url,
            workspace: url.deletingPathExtension().lastPathComponent,
            title: tasks.values.first?.title,
            facts: facts,
            sessionCount: sessions.count,
            eventCount: events,
            bytesOnDisk: (attributes?[.size] as? Int) ?? 0,
            modifiedAt: (attributes?[.modificationDate] as? Date) ?? .distantPast,
            damagedRecordLines: replay.damage?.lineNumbers ?? [])
    }

    /// Where the project files are, by the same rule the server uses:
    /// `TINYTITAN_MEMORY_DIR`, else `memory/` under the current directory when it
    /// exists, else the binary's own fallback.
    public static func defaultDirectory(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        currentDirectory: String = FileManager.default.currentDirectoryPath
    )
        -> URL
    {
        if let explicit = environment["TINYTITAN_MEMORY_DIR"], !explicit.isEmpty {
            return URL(fileURLWithPath: explicit)
        }
        let local = URL(fileURLWithPath: currentDirectory).appendingPathComponent("memory")
        if FileManager.default.fileExists(atPath: local.path) { return local }
        return ContinuityStorageConfiguration.defaultDirectory
    }
}
