import Foundation
import NIOCore
import Testing

@testable import TinyTitanCore
@testable import TinyTitanLib
@testable import TinyTitanServerCore

/// A backend whose response size is a test parameter, so a stored entry can be
/// pushed over a budget without a model or a large request body.
private actor SizedTextBackend: ServerInferenceBackend {
    let contentLength: Int
    private(set) var generations = 0

    init(contentLength: Int) {
        self.contentLength = contentLength
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        generations += 1
        let text = String(repeating: "x", count: contentLength)
        onEvent(.content(text))
        return ServerCompletion(
            content: text, toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }
}

private func storePost(
    _ port: Int, _ path: String, _ json: String
) async throws -> (Data, HTTPURLResponse) {
    var request = URLRequest(url: try localURL(port: port, path))
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "content-type")
    request.httpBody = Data(json.utf8)
    let (data, response) = try await URLSession.shared.data(for: request)
    return (data, try #require(response as? HTTPURLResponse))
}

private func storeGet(_ port: Int, _ path: String) async throws -> (Data, HTTPURLResponse) {
    let (data, response) = try await URLSession.shared.data(
        from: try localURL(port: port, path))
    return (data, try #require(response as? HTTPURLResponse))
}

private func storeObject(_ data: Data) throws -> [String: Any] {
    try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
}

/// The Responses store's byte and chain bounds at the HTTP surface: what the
/// client is told about storage, what it can retrieve, and what is refused
/// before the model runs.
@Suite("Stored response budgets", .serialized)
struct StoredResponseBudgetTests {
    @Test func anOversizeGeneratedResponseIsNotClaimedStored() async throws {
        // 17 MiB of generated text makes one entry over the default 16 MiB
        // per-entry budget (the response JSON and the output item both carry
        // it). The old count-only store accepted it and answered `store: true`
        // for an id a later GET could not be guaranteed to find.
        let backend = SizedTextBackend(contentLength: 17 * 1024 * 1024)
        let server = TinyTitanHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: backend)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (data, response) = try await storePost(
            port, "/v1/responses", #"{"model":"test-model","input":"hi"}"#)
        #expect(response.statusCode == 200)
        let object = try storeObject(data)
        let id = try #require(object["id"] as? String)
        #expect(
            object["store"] as? Bool == false,
            "an over-budget response claimed to be stored")
        let (_, fetched) = try await storeGet(port, "/v1/responses/\(id)")
        #expect(fetched.statusCode == 404, "an unstored response was retrievable")

        try await server.shutdown()
    }

    @Test func anOversizeStreamedResponseIsReportedStoreFalse() async throws {
        let store = ResponseStore(
            capacity: 4, maximumEntryBytes: 2_000, maximumTotalBytes: 20_000)
        let backend = SizedTextBackend(contentLength: 1_000)
        let server = TinyTitanHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: backend,
            responseStore: store)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (data, response) = try await storePost(
            port, "/v1/responses",
            #"{"model":"test-model","input":"hi","stream":true}"#)
        #expect(response.statusCode == 200)
        let terminal = try #require(
            data.lossyUTF8String
                .components(separatedBy: "\n\n")
                .last(where: { $0.contains(#""type":"response.completed""#) }))
        let jsonStart = try #require(terminal.range(of: "data: "))
        let terminalJSON = try storeObject(
            Data(terminal[jsonStart.upperBound...].utf8))
        let final = try #require(terminalJSON["response"] as? [String: Any])
        let id = try #require(final["id"] as? String)
        #expect(
            final["store"] as? Bool == false,
            "the streamed terminal object claimed a refused entry was stored")
        #expect(store.totalRetainedBytes == 0)
        let (_, fetched) = try await storeGet(port, "/v1/responses/\(id)")
        #expect(fetched.statusCode == 404)

        try await server.shutdown()
    }

    /// A chain is refused at admission when its flattened conversation already
    /// cannot fit one entry. The model is not run and the prior response is
    /// still retrievable.
    @Test func aChainOverTheEntryBudgetIsRefusedBeforeGeneration() async throws {
        let store = ResponseStore(
            capacity: 4, maximumEntryBytes: 4_096, maximumTotalBytes: 64 * 1_024)
        let backend = SizedTextBackend(contentLength: 20)
        let server = TinyTitanHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: backend,
            responseStore: store)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (first, firstResponse) = try await storePost(
            port, "/v1/responses", #"{"model":"test-model","input":"first"}"#)
        #expect(firstResponse.statusCode == 200)
        let firstObject = try storeObject(first)
        let firstID = try #require(firstObject["id"] as? String)
        #expect(firstObject["store"] as? Bool == true)
        #expect(await backend.generations == 1)

        let longInput = String(repeating: "z", count: 5_000)
        let (refused, refusedResponse) = try await storePost(
            port, "/v1/responses",
            #"{"model":"test-model","input":"\#(longInput)","previous_response_id":"\#(firstID)"}"#)
        #expect(refusedResponse.statusCode == 400)
        let refusedObject = try storeObject(refused)
        let error = try #require(refusedObject["error"] as? [String: Any])
        #expect(error["code"] as? String == "stored_response_too_large")
        #expect(error["param"] as? String == "previous_response_id")
        #expect(
            await backend.generations == 1,
            "the over-budget chain reached the model")

        let (_, stillThere) = try await storeGet(port, "/v1/responses/\(firstID)")
        #expect(stillThere.statusCode == 200, "the refused continuation removed the prior")

        try await server.shutdown()
    }

    /// `store: false` is unchanged: it does not consult or consume the budget.
    @Test func storeFalseBypassesTheBudget() async throws {
        let store = ResponseStore(
            capacity: 4, maximumEntryBytes: 4_096, maximumTotalBytes: 64 * 1_024)
        let backend = SizedTextBackend(contentLength: 20)
        let server = TinyTitanHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: backend,
            responseStore: store)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let longInput = String(repeating: "z", count: 5_000)
        let (data, response) = try await storePost(
            port, "/v1/responses",
            #"{"model":"test-model","input":"\#(longInput)","store":false}"#)
        #expect(response.statusCode == 200)
        let object = try storeObject(data)
        let id = try #require(object["id"] as? String)
        #expect(object["store"] as? Bool == false)
        #expect(store.count == 0)
        let (_, fetched) = try await storeGet(port, "/v1/responses/\(id)")
        #expect(fetched.statusCode == 404)
        #expect(await backend.generations == 1)

        try await server.shutdown()
    }

    /// The byte budget evicts the oldest stored response, and a continuation
    /// still names its `previous_response_id` even after that prior is gone.
    @Test func theByteBudgetEvictsTheOldestStoredResponse() async throws {
        let store = ResponseStore(
            capacity: 8, maximumEntryBytes: 10_000, maximumTotalBytes: 8_000)
        let backend = SizedTextBackend(contentLength: 2_000)
        let server = TinyTitanHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: backend,
            responseStore: store)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)

        let (first, firstResponse) = try await storePost(
            port, "/v1/responses", #"{"model":"test-model","input":"first"}"#)
        #expect(firstResponse.statusCode == 200)
        let firstObject = try storeObject(first)
        let firstID = try #require(firstObject["id"] as? String)
        #expect(firstObject["store"] as? Bool == true)

        let (second, secondResponse) = try await storePost(
            port, "/v1/responses",
            #"{"model":"test-model","input":"second","previous_response_id":"\#(firstID)"}"#)
        #expect(secondResponse.statusCode == 200)
        let secondObject = try storeObject(second)
        let secondID = try #require(secondObject["id"] as? String)
        #expect(
            secondObject["store"] as? Bool == true,
            "the entry that fits the per-entry budget was refused")
        #expect(store.totalRetainedBytes <= store.maximumTotalBytes)

        let (_, evicted) = try await storeGet(port, "/v1/responses/\(firstID)")
        #expect(
            evicted.statusCode == 404,
            "two responses over the total budget kept the oldest")
        // The newest is retrievable and the accounting is inside the budget.
        let (_, newest) = try await storeGet(port, "/v1/responses/\(secondID)")
        #expect(newest.statusCode == 200)
        #expect(store.totalRetainedBytes <= 8_000)

        // Continuing from the newest still works (its chain was flattened
        // before the first was potentially evicted).
        let (third, thirdResponse) = try await storePost(
            port, "/v1/responses",
            #"{"model":"test-model","input":"third","previous_response_id":"\#(secondID)"}"#)
        #expect(thirdResponse.statusCode == 200)
        #expect(try storeObject(third)["previous_response_id"] as? String == secondID)

        try await server.shutdown()
    }
}
