import Darwin
import Foundation
import NIOCore
import Synchronization
import Testing

@testable import TinyTitanCore
@testable import TinyTitanLib
@testable import TinyTitanServerCore

// The intended trust boundary of this server is one sentence long: it binds
// 127.0.0.1, has no authentication, and the machine is the trust boundary. That
// makes a *remote* attacker out of scope and a *browser* attacker in scope:
// DNS rebinding lets a page the operator visits resolve its own hostname to
// 127.0.0.1, and then the browser's same-origin rule stops protecting this
// server. The Host/Origin/Referer policy these tests pin is what refuses that,
// without touching the documented CLI clients (which send a loopback Host and
// no Origin at all).
//
// These are raw-socket tests because URLSession will not let a test send a
// duplicate Host or an HTTP/1.0 request, and the exact wire shape is what the
// policy reads.

private enum RawHTTPError: Error {
    case systemCall(String, Int32)
    case timeout
}

private func rawConnectedSocket(port: Int) throws -> Int32 {
    let descriptor = Darwin.socket(AF_INET, SOCK_STREAM, 0)
    guard descriptor >= 0 else { throw RawHTTPError.systemCall("socket", errno) }
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = in_port_t(port).bigEndian
    address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
    let result = withUnsafePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            Darwin.connect(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    guard result == 0 else {
        let code = errno
        Darwin.close(descriptor)
        throw RawHTTPError.systemCall("connect", code)
    }
    return descriptor
}

private func rawWriteAll(socket: Int32, text: String) throws {
    let bytes = Array(text.utf8)
    var written = 0
    while written < bytes.count {
        let count = bytes.withUnsafeBytes { buffer -> Int in
            guard let base = buffer.baseAddress else { return -1 }
            return Darwin.send(socket, base.advanced(by: written), bytes.count - written, 0)
        }
        guard count > 0 else { throw RawHTTPError.systemCall("send", errno) }
        written += count
    }
}

private func rawReadUntil(
    socket: Int32, timeoutMilliseconds: Int32, condition: (String) -> Bool
) throws -> String {
    let deadline = Date().addingTimeInterval(Double(timeoutMilliseconds) / 1_000)
    var result = ""
    while Date() < deadline {
        var descriptor = pollfd(fd: socket, events: Int16(POLLIN), revents: 0)
        guard Darwin.poll(&descriptor, 1, 50) > 0 else { continue }
        var buffer = [UInt8](repeating: 0, count: 4_096)
        let count = Darwin.recv(socket, &buffer, buffer.count, 0)
        guard count >= 0 else { throw RawHTTPError.systemCall("recv", errno) }
        if count == 0 { break }
        result += buffer.prefix(count).lossyUTF8String
        if condition(result) { return result }
    }
    if condition(result) { return result }
    throw RawHTTPError.timeout
}

/// One raw request on a fresh connection; the reply, or a thrown timeout.
private func rawExchange(
    port: Int, request: String, until: @escaping (String) -> Bool = { _ in true }
) throws -> String {
    let socket = try rawConnectedSocket(port: port)
    defer { Darwin.close(socket) }
    try rawWriteAll(socket: socket, text: request)
    return try rawReadUntil(socket: socket, timeoutMilliseconds: 2_000, condition: until)
}

/// The backend a refused request must never reach. `unload()` is counted too:
/// `/v1/models/unload` is a control surface, and "the request was refused" has
/// to mean the model was not released either.
///
/// unchecked-invariant: every counter is only touched under `lock`.
private final class RefusalCountingBackend: ServerInferenceBackend, ResidencyManaging,
    @unchecked Sendable
{
    private let lock = NSLock()
    private var _generations = 0
    private var _unloads = 0

    var generations: Int { lock.withLock { _generations } }
    var unloads: Int { lock.withLock { _unloads } }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        lock.withLock { _generations += 1 }
        onEvent(.content("reached"))
        return ServerCompletion(
            content: "reached", toolCalls: [], finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1, completionTokens: 1, totalTokens: 2))
    }

    func unload() async -> Bool {
        lock.withLock { _unloads += 1 }
        return true
    }
}

@Suite("Loopback trust boundary", .serialized)
struct LoopbackTrustTests {
    private func start(
        _ backend: any ServerInferenceBackend = RefusalCountingBackend()
    ) async throws -> (TinyTitanHTTPServer, Int) {
        let server = TinyTitanHTTPServer(
            modelID: "test-model", queueLimit: 1, backend: backend)
        let channel = try await server.start(port: 0)
        return (server, try #require(channel.localAddress?.port))
    }

    private func chatRequest(port: Int, host: String, extraHeaders: String = "") -> String {
        let body = #"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#
        return
            "POST /v1/chat/completions HTTP/1.1\r\n"
            + "Host: \(host)\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.utf8.count)\r\n"
            + extraHeaders
            + "Connection: close\r\n"
            + "\r\n"
            + body
    }

    @Test func canonicalLoopbackHostFormsAreAccepted() async throws {
        let (server, port) = try await start()
        for host in [
            "127.0.0.1:\(port)", "localhost:\(port)", "LOCALHOST:\(port)", "[::1]:\(port)",
            "127.0.0.1",
        ] {
            let reply = try rawExchange(
                port: port,
                request: "GET /health HTTP/1.1\r\nHost: \(host)\r\nConnection: close\r\n\r\n",
                until: { $0.contains("HTTP/1.1 ") })
            #expect(reply.hasPrefix("HTTP/1.1 200"), "Host \(host) was refused: \(reply)")
        }
        try await server.shutdown()
    }

    @Test func aForeignHostIsRefusedBeforeRouting() async throws {
        let backend = RefusalCountingBackend()
        let (server, port) = try await start(backend)
        let reply = try rawExchange(
            port: port, request: chatRequest(port: port, host: "evil.example"),
            until: { $0.contains("HTTP/1.1 ") })
        #expect(reply.hasPrefix("HTTP/1.1 403"), "foreign Host was routed: \(reply)")
        #expect(reply.contains("untrusted_host"))
        #expect(backend.generations == 0, "a foreign-Host request reached the model")
        try await server.shutdown()
    }

    /// The refusal runs before routing, so it must also honour the HEAD rule:
    /// a refused HEAD gets a status line and no body.
    @Test func aRefusedHEADCarriesNoBody() async throws {
        let (server, port) = try await start()
        let reply = try rawExchange(
            port: port,
            request:
                "HEAD /health HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n",
            until: { $0.contains("\r\n\r\n") })
        #expect(reply.hasPrefix("HTTP/1.1 403"), "refused HEAD: \(reply)")
        let separator = try #require(reply.range(of: "\r\n\r\n"))
        #expect(reply[separator.upperBound...].isEmpty, "a refused HEAD carried a body")
        try await server.shutdown()
    }

    @Test func duplicateHostIsRefused() async throws {
        let backend = RefusalCountingBackend()
        let (server, port) = try await start(backend)
        let body = #"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#
        let request =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            + "Host: 127.0.0.1:\(port)\r\n"
            + "Host: evil.example\r\n"
            + "Content-Type: application/json\r\n"
            + "Content-Length: \(body.utf8.count)\r\n"
            + "Connection: close\r\n\r\n"
            + body
        let reply = try rawExchange(
            port: port, request: request, until: { $0.contains("HTTP/1.1 ") })
        #expect(reply.hasPrefix("HTTP/1.1 400"), "duplicate Host was resolved: \(reply)")
        #expect(reply.contains("duplicate_host"))
        #expect(backend.generations == 0)
        try await server.shutdown()
    }

    @Test func aForeignOriginIsRefused() async throws {
        let backend = RefusalCountingBackend()
        let (server, port) = try await start(backend)
        let reply = try rawExchange(
            port: port,
            request: chatRequest(
                port: port, host: "127.0.0.1:\(port)",
                extraHeaders: "Origin: https://evil.example\r\n"),
            until: { $0.contains("HTTP/1.1 ") })
        #expect(reply.hasPrefix("HTTP/1.1 403"), "foreign Origin was routed: \(reply)")
        #expect(reply.contains("untrusted_origin"))
        #expect(backend.generations == 0)
        try await server.shutdown()
    }

    @Test func aLoopbackOriginIsAccepted() async throws {
        let backend = RefusalCountingBackend()
        let (server, port) = try await start(backend)
        for origin in [
            "http://127.0.0.1:3000", "https://localhost", "http://[::1]:8080",
        ] {
            let reply = try rawExchange(
                port: port,
                request: chatRequest(
                    port: port, host: "127.0.0.1:\(port)",
                    extraHeaders: "Origin: \(origin)\r\n"),
                until: { $0.contains("HTTP/1.1 ") })
            #expect(reply.hasPrefix("HTTP/1.1 200"), "Origin \(origin) refused: \(reply)")
        }
        #expect(backend.generations == 3)
        try await server.shutdown()
    }

    @Test func aForeignRefererIsRefused() async throws {
        let (server, port) = try await start()
        let reply = try rawExchange(
            port: port,
            request:
                "GET /health HTTP/1.1\r\nHost: 127.0.0.1:\(port)\r\n"
                + "Referer: https://evil.example/page\r\nConnection: close\r\n\r\n",
            until: { $0.contains("HTTP/1.1 ") })
        #expect(reply.hasPrefix("HTTP/1.1 403"), "foreign Referer was served: \(reply)")
        #expect(reply.contains("untrusted_referer"))
        try await server.shutdown()
    }

    @Test func aLoopbackRefererIsAccepted() async throws {
        let (server, port) = try await start()
        let reply = try rawExchange(
            port: port,
            request:
                "GET /health HTTP/1.1\r\nHost: localhost:\(port)\r\n"
                + "Referer: http://localhost:3000/index.html\r\nConnection: close\r\n\r\n",
            until: { $0.contains("HTTP/1.1 ") })
        #expect(reply.hasPrefix("HTTP/1.1 200"), "loopback Referer was refused: \(reply)")
        try await server.shutdown()
    }

    /// No CORS: a preflight is not answered with an allow header, and the
    /// request is not routed. That is what stops a cross-origin *fetch* that
    /// needs `application/json` from reaching a JSON route at all.
    @Test func preflightIsNotAnsweredWithCORS() async throws {
        let backend = RefusalCountingBackend()
        let (server, port) = try await start(backend)
        let reply = try rawExchange(
            port: port,
            request:
                "OPTIONS /v1/chat/completions HTTP/1.1\r\n"
                + "Host: 127.0.0.1:\(port)\r\n"
                + "Origin: https://evil.example\r\n"
                + "Access-Control-Request-Method: POST\r\n"
                + "Access-Control-Request-Headers: content-type\r\n"
                + "Connection: close\r\n\r\n",
            until: { $0.contains("HTTP/1.1 ") })
        #expect(!reply.lowercased().contains("access-control-allow-origin: *"))
        #expect(!reply.lowercased().contains("access-control-allow-origin: https://evil.example"))
        #expect(!reply.hasPrefix("HTTP/1.1 200"))
        #expect(backend.generations == 0)
        try await server.shutdown()
    }

    @Test func http10WithoutHostIsStillServed() async throws {
        let (server, port) = try await start()
        let reply = try rawExchange(
            port: port, request: "GET /health HTTP/1.0\r\n\r\n",
            until: { $0.contains("HTTP/1.1 ") || $0.contains("HTTP/1.0 ") })
        #expect(reply.hasPrefix("HTTP/1.1 200") || reply.hasPrefix("HTTP/1.0 200"))
        try await server.shutdown()
    }

    @Test func http10WithAForeignHostIsRefused() async throws {
        let (server, port) = try await start()
        let reply = try rawExchange(
            port: port,
            request: "GET /health HTTP/1.0\r\nHost: evil.example\r\n\r\n",
            until: { $0.contains("HTTP/1.1 ") || $0.contains("HTTP/1.0 ") })
        #expect(reply.contains(" 403"), "HTTP/1.0 bypassed the Host policy: \(reply)")
        #expect(reply.contains("untrusted_host"))
        try await server.shutdown()
    }

    @Test func missingHostOnHTTP11IsStillRefused() async throws {
        let (server, port) = try await start()
        let reply = try rawExchange(
            port: port, request: "GET /health HTTP/1.1\r\nConnection: close\r\n\r\n",
            until: { $0.contains("HTTP/1.1 ") })
        #expect(reply.hasPrefix("HTTP/1.1 400"))
        #expect(reply.contains("missing_host"))
        try await server.shutdown()
    }

    /// The unload control route is the one a browser page would most like to
    /// reach (it frees the operator's model). A cross-site form POST is a simple
    /// request -- no preflight -- so the Origin/Host policy, not CORS, is what
    /// refuses it; and a same-machine form is refused by the content-type gate
    /// 0020 pinned. Neither may release the model.
    @Test func theUnloadControlRouteIsNotReachableFromAForeignOrigin() async throws {
        let backend = RefusalCountingBackend()
        let (server, port) = try await start(backend)

        let crossSite = try rawExchange(
            port: port,
            request:
                "POST /v1/models/unload HTTP/1.1\r\n"
                + "Host: 127.0.0.1:\(port)\r\n"
                + "Origin: https://evil.example\r\n"
                + "Content-Type: application/x-www-form-urlencoded\r\n"
                + "Content-Length: 3\r\nConnection: close\r\n\r\nx=1",
            until: { $0.contains("HTTP/1.1 ") })
        #expect(crossSite.hasPrefix("HTTP/1.1 403"), "cross-site unload: \(crossSite)")
        #expect(backend.unloads == 0)

        let trustedForm = try rawExchange(
            port: port,
            request:
                "POST /v1/models/unload HTTP/1.1\r\n"
                + "Host: localhost:\(port)\r\n"
                + "Content-Type: application/x-www-form-urlencoded\r\n"
                + "Content-Length: 3\r\nConnection: close\r\n\r\nx=1",
            until: { $0.contains("HTTP/1.1 ") })
        #expect(trustedForm.hasPrefix("HTTP/1.1 415"), "form unload: \(trustedForm)")
        #expect(backend.unloads == 0)

        try await server.shutdown()
    }
}
