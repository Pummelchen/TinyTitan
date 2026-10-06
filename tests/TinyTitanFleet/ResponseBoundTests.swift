import Darwin
import Foundation
import Testing
@testable import TinyTitanFleetCore

/// AUD-168's sibling: the fleet CLI dials any member the LAN admits, and `data(for:)`
/// buffered the whole answer before anything counted it. The bound lives inside the
/// response reader, so a stubbed `FleetTransport` — which is how every other test in
/// this target drives the manager — cannot reach it. Real loopback sockets are the
/// only way to prove the client stops reading.
struct URLSessionResponseBoundTests {
    /// `/small` is an ordinary inventory and must survive untouched; `/big` announces
    /// more than the cap and must be refused with the cap named.
    @Test func anAnswerOverTheCapIsDroppedRatherThanBuffered() async throws {
        let server = try LoopbackAnswerServer()
        defer { server.stop() }

        let transport = URLSessionTransport(timeout: 5)
        func request(_ path: String) -> FleetRequest {
            FleetRequest(
                method: "GET",
                target: FleetTarget(host: "127.0.0.1", port: server.port),
                path: path, token: "tinytitan-lan")
        }

        let small = try await transport.send(request("/small"))
        #expect(small.status == 200)
        #expect(
            try #require(String(bytes: small.body, encoding: .utf8)) == "{\"ok\":true}",
            "a normal inventory still arrives whole")

        // The error's own numbers are pinned, not just its type: the cap named must
        // be the cap compiled in, or a changed constant would leave this test
        // asserting an old number and pass anyway.
        do {
            _ = try await transport.send(request("/big"))
            Issue.record("an over-cap answer was accepted")
        } catch let error as FleetError {
            if case .responseTooLarge(let target, let bytes) = error {
                #expect(bytes == URLSessionTransport.maxResponseBytes)
                #expect(target.contains("127.0.0.1"), "and it names who was too large")
            } else {
                Issue.record("expected responseTooLarge, got \(error.description)")
            }
        } catch {
            Issue.record("expected responseTooLarge, got \(error)")
        }
    }
}

/// A blocking TCP listener on its own queue, shut down by `stop()`.
///
/// `@unchecked Sendable` because it is captured by the queue's `@Sendable` closure.
/// unchecked-invariant: `port` and `listenFD` are immutable after `init`; `clientFDs`
/// is only touched under `lock`; the accept loop runs on one serial queue and is the
/// only writer of a client descriptor.
private final class LoopbackAnswerServer: @unchecked Sendable {
    let port: Int
    private let listenFD: Int32
    private var clientFDs: [Int32] = []
    private let lock = NSLock()

    init() throws {
        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw LoopbackFailure.open(errno) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr = in_addr(s_addr: UInt32(0x7F00_0001).bigEndian)
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0, listen(fd, 8) == 0 else {
            Darwin.close(fd)
            throw LoopbackFailure.bind(errno)
        }
        var name = sockaddr_in()
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let read = withUnsafeMutablePointer(to: &name) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(fd, $0, &length)
            }
        }
        guard read == 0 else {
            Darwin.close(fd)
            throw LoopbackFailure.bind(errno)
        }
        listenFD = fd
        port = Int(UInt16(bigEndian: name.sin_port))
        let queue = DispatchQueue(label: "tinytitan.loopback-answer-server")
        queue.async { [weak self] in
            self?.acceptLoop()
        }
    }

    private func acceptLoop() {
        while true {
            let client = accept(listenFD, nil, nil)
            guard client >= 0 else { return }
            lock.lock()
            clientFDs.append(client)
            lock.unlock()
            serve(client)
        }
    }

    private func serve(_ client: Int32) {
        // The reader stops mid-body by design, so a later `send` here is a write to a
        // closed socket. Without SO_NOSIGPIPE that is a signal that kills the test
        // process rather than an error code the loop below already handles.
        var noSigPipe: Int32 = 1
        _ = setsockopt(
            client, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe, socklen_t(MemoryLayout<Int32>.size))
        var head = [UInt8](repeating: 0, count: 4096)
        // No MSG_WAITALL: a GET is far shorter than the buffer, and waiting for the
        // full 4096 would park this thread instead of answering.
        let read = recv(client, &head, head.count, 0)
        guard read > 0, let request = String(bytes: head[0..<Int(read)], encoding: .utf8) else {
            Darwin.close(client)
            return
        }
        if request.contains("/small") {
            let body = Array(#"{"ok":true}"#.utf8)
            writeAll(
                client,
                Array(
                    ("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                        + "content-length: \(body.count)\r\nconnection: close\r\n\r\n").utf8))
            writeAll(client, body)
            Darwin.close(client)
            return
        }
        // Announce more than the cap so the reader has to reach the bound rather
        // than finish first, then keep sending until it stops asking. The
        // connection is left open and closed by `stop()`: a `SIGPIPE`-driven
        // failure would answer the wrong question.
        let announced = URLSessionTransport.maxResponseBytes + 1024 * 1024
        writeAll(
            client,
            Array(
                ("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                    + "content-length: \(announced)\r\nconnection: close\r\n\r\n").utf8))
        var sent = 0
        let chunk = [UInt8](repeating: 0x61, count: 64 * 1024)
        while sent < URLSessionTransport.maxResponseBytes + 128 * 1024 {
            let written = chunk.withUnsafeBytes {
                send(client, $0.baseAddress, $0.count, 0)
            }
            guard written > 0 else { return }
            sent += written
        }
    }

    private func writeAll(_ client: Int32, _ bytes: [UInt8]) {
        // A dead peer is not the property under test; `stop()` closes the rest.
        _ = bytes.withUnsafeBytes { Darwin.write(client, $0.baseAddress, $0.count) }
    }

    func stop() {
        Darwin.close(listenFD)
        lock.lock()
        let fds = clientFDs
        clientFDs = []
        lock.unlock()
        for fd in fds {
            Darwin.close(fd)
        }
        // The accept loop leaves once `accept` fails on the closed listener.
    }
}

/// Carries `errno`, because "bind failed" without it sends the next reader to
/// the man pages.
private enum LoopbackFailure: Error {
    case open(Int32)
    case bind(Int32)
}
