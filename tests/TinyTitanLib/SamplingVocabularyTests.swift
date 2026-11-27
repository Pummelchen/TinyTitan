import Foundation
import Testing

@testable import TinyTitanLib

/// The sampler may only pick an id the tokenizer can decode (AUD-0101).
///
/// The lm_head is padded to the architecture's vocabulary (248,320 for the
/// Qwen builds) while the tokenizer's is smaller (248,077). An id in that gap
/// has no token, the detokenizer refuses it, and the request fails with a 500
/// — so the sampler's width is the narrower of the two.
@Suite struct SamplingVocabularyTests {
    @Test func theNarrowerVocabularyWins() {
        #expect(
            TinyTitanServerSessionShim.samplingVocabulary(
                architecture: 248_320, tokenizer: 248_077) == 248_077)
        #expect(
            TinyTitanServerSessionShim.samplingVocabulary(
                architecture: 32, tokenizer: 248_077) == 32)
    }

    @Test func aTokenizerWithNoVocabularyLeavesTheHeadWidth() {
        #expect(
            TinyTitanServerSessionShim.samplingVocabulary(
                architecture: 248_320, tokenizer: 0) == 248_320)
    }
}

/// `ServerModelSession` is package-visible; this alias keeps the test's call
/// sites readable.
private typealias TinyTitanServerSessionShim = ServerModelSession
