import Foundation
import Testing

@testable import Ornith35B4Bit

/// The PLE head ranges are validated in the signed domain (AUD-0095).
///
/// The values reach `PLEHash` as bit patterns, so a negative offset or a zero
/// vocab size becomes a huge unsigned number, passes a naive coverage check and
/// makes the hash compute row ids past the table. The signed add also used to
/// trap before any check ran.
@Suite struct GGUFPLESetupRangeTests {
    @Test func aNegativeOffsetIsRefusedNotTrapped() {
        #expect(throws: GGUFPLESetupError.self) {
            _ = try GGUFPLESetup.coveredRows(
                offsets: [-2], vocabSizes: [4], tableRows: 1_000)
        }
    }

    @Test func aZeroVocabSizeIsRefused() {
        #expect(throws: GGUFPLESetupError.self) {
            _ = try GGUFPLESetup.coveredRows(
                offsets: [0], vocabSizes: [0], tableRows: 1_000)
        }
    }

    @Test func aRangePastTheTableIsRefused() {
        #expect(throws: GGUFPLESetupError.self) {
            _ = try GGUFPLESetup.coveredRows(
                offsets: [990], vocabSizes: [20], tableRows: 1_000)
        }
    }

    @Test func theHighestRangeEndIsWhatItReturns() throws {
        #expect(
            try GGUFPLESetup.coveredRows(
                offsets: [0, 10, 30], vocabSizes: [5, 5, 5], tableRows: 1_000) == 35)
    }
}
