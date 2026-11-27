import Testing

@testable import TinyTitanRepackCore

/// `--verify-install` has to name what it did not attest (AUD-0075).
///
/// The CLI prints these lines verbatim. `unexpectedEntries` is how an ANE
/// prefill sidecar -- written into the install by a separate step and absent
/// from the manifest -- gets reported; printing "Verified" on its own made an
/// unattested directory look attested.
@Suite("Verify install report")
struct VerifyInstallReportTests {
    @Test("An entry outside the manifest is named on its own line")
    func unattestedEntriesAreNamed() {
        let result = VerifyInstallResult(
            receiptPath: "/models/ornith/verified-install.json",
            fileCount: 3,
            bytesVerified: 4_096,
            unexpectedEntries: ["ane_prefill", "ane_prefill/sidecar.bin"])
        #expect(
            VerifiedInstallTool.reportLines(result) == [
                "Verified 3 files (4096 bytes)",
                "Unattested entry (not in manifest): ane_prefill",
                "Unattested entry (not in manifest): ane_prefill/sidecar.bin",
                "Receipt: /models/ornith/verified-install.json",
            ])
    }

    @Test("A clean install prints no unattested line")
    func cleanInstallHasNoUnattestedLine() {
        let result = VerifyInstallResult(
            receiptPath: "/models/ornith/verified-install.json",
            fileCount: 42,
            bytesVerified: 1_000,
            unexpectedEntries: [])
        #expect(
            VerifiedInstallTool.reportLines(result) == [
                "Verified 42 files (1000 bytes)",
                "Receipt: /models/ornith/verified-install.json",
            ])
    }
}
