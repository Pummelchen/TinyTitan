import Foundation

/// The path form an install receipt is bound to, and the one comparison every
/// loader makes against it.
///
/// The receipt records which directory was attested, so moving or swapping it
/// is detected. It must not record the *spelling* the operator typed: keeping
/// models on a second disk by symlinking the directory (`models -> /Volumes/ext/
/// models`) is a supported layout, and the installer places the payload at the
/// physical path. Recording the typed path made an untouched, correctly
/// installed model refuse to load with "model directory mismatch … Moving or
/// renaming an installed model invalidates its receipt", which was false, and
/// `--verify-install` — the documented cure — re-issued the same spelling.
///
/// `canonical` resolves the parent and keeps the last component, for two
/// reasons: the writer runs before the final directory exists (a non-existent
/// leaf is not resolved by Foundation), and the last component is the name the
/// installer itself created.
public enum InstallPathBinding {
    /// The physical form of an install directory, whether or not the directory
    /// exists yet.
    public static func canonical(_ url: URL) -> String {
        let parent = url.deletingLastPathComponent().resolvingSymlinksInPath()
            .standardizedFileURL.path
        return (parent as NSString).appendingPathComponent(url.lastPathComponent)
    }

    /// Whether `recorded` names the same directory as `directory`.
    ///
    /// True for the canonical form, for the fully resolved form (an install
    /// directory that is itself a symlink), and for the operator's own
    /// spelling (a receipt issued before the writer recorded the physical
    /// path). A moved or swapped directory matches none of them.
    public static func matches(_ recorded: String, directory: URL) -> Bool {
        if recorded == canonical(directory) { return true }
        let spelled = directory.standardizedFileURL.path
        if recorded == spelled { return true }
        return recorded == directory.resolvingSymlinksInPath().standardizedFileURL.path
    }
}
