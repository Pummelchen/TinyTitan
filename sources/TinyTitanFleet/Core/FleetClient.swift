import Foundation

/// A request the manager makes to one member of the group.
public struct FleetRequest: Sendable {
    public let method: String
    public let target: FleetTarget
    public let path: String
    public let token: String
    public let body: Data?

    public init(method: String, target: FleetTarget, path: String, token: String, body: Data? = nil)
    {
        self.method = method
        self.target = target
        self.path = path
        self.token = token
        self.body = body
    }
}

/// A member's raw answer.
public struct FleetResponse: Sendable {
    public let status: Int
    public let body: Data

    public init(status: Int, body: Data) {
        self.status = status
        self.body = body
    }
}

/// How requests are sent. A protocol so tests drive the whole manager — owner
/// resolution, fan-out, rendering — with no network and no live fleet.
public protocol FleetTransport: Sendable {
    func send(_ request: FleetRequest) async throws -> FleetResponse
}

/// Anything that went wrong talking to a member.
public enum FleetError: Error, Equatable, CustomStringConvertible {
    case badURL(String)
    case unreachable(target: String, reason: String)
    case http(target: String, status: Int, message: String)
    case decoding(target: String, reason: String)
    case notFound(String)
    case responseTooLarge(target: String, bytes: Int)

    public var description: String {
        switch self {
        case .badURL(let text):
            return "not a usable address: \(text)"
        case .unreachable(let target, let reason):
            return "\(target) is unreachable: \(reason)"
        case .http(let target, let status, let message):
            return "\(target) answered \(status): \(message)"
        case .decoding(let target, let reason):
            return "\(target) answered something unreadable: \(reason)"
        case .notFound(let what):
            return "no member of the group holds \(what)"
        case .responseTooLarge(let target, let bytes):
            return "\(target) sent more than \(bytes) bytes; the answer was dropped"
        }
    }
}

/// The real transport: one HTTP request, `x-dsh-token` carrying the group key.
public struct URLSessionTransport: FleetTransport {
    public let timeout: TimeInterval
    private let session: URLSession

    /// Bytes one answer may occupy, counted while it arrives.
    ///
    /// The response comes from any host the LAN fence admits, so its size is
    /// external input. 4 MiB is the ceiling this project already uses for an
    /// outside answer -- `maxBuffer` in the manager's `dns-sd` reader and
    /// `MAX_PEER_RESPONSE_BYTES` in its peer probe -- so one number describes every
    /// response read here instead of three that can drift apart.
    static let maxResponseBytes = 4 * 1024 * 1024

    public init(timeout: TimeInterval = 10) {
        self.timeout = timeout
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout
        self.session = URLSession(configuration: configuration)
    }

    public func send(_ request: FleetRequest) async throws -> FleetResponse {
        // An IPv6 literal must be bracketed inside a URL, and a v6 host is the
        // one case where a colon in `host` is not a port.
        let host =
            request.target.host.contains(":") ? "[\(request.target.host)]" : request.target.host
        guard let url = URL(string: "http://\(host):\(request.target.port)\(request.path)") else {
            throw FleetError.badURL("\(request.target)\(request.path)")
        }
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = request.method
        urlRequest.timeoutInterval = timeout
        urlRequest.setValue(request.token, forHTTPHeaderField: "x-dsh-token")
        if let body = request.body {
            urlRequest.httpBody = body
            urlRequest.setValue("application/json", forHTTPHeaderField: "content-type")
        }
        do {
            // The answer is counted as it arrives, not after: `data(for:)` buffers
            // the whole body first, which is a bound applied once the cost is paid.
            // Byte-at-a-time is what `URLSession.AsyncBytes` offers on this SDK —
            // `chunks(ofCount:)` measured absent at compile on Swift 6.4 — and a
            // legitimate answer is a few kilobytes, so the loop only runs long when a
            // host is streaming past the cap, where it stops at 4 MiB.
            var data = Data()
            let (stream, response) = try await session.bytes(for: urlRequest)
            for try await byte in stream {
                data.append(byte)
                guard data.count <= Self.maxResponseBytes else {
                    throw FleetError.responseTooLarge(
                        target: "\(request.target)", bytes: Self.maxResponseBytes)
                }
            }
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            return FleetResponse(status: status, body: data)
        } catch let error as FleetError {
            throw error
        } catch {
            throw FleetError.unreachable(
                target: "\(request.target)", reason: error.localizedDescription)
        }
    }
}

/// What a mutating call answers with. The body is kept verbatim so the CLI can
/// print it under `--json` without a model for every endpoint's receipt.
public struct FleetAck: Sendable, Equatable {
    public let ok: Bool
    public let message: String?
    public let raw: String

    public init(ok: Bool, message: String?, raw: String) {
        self.ok = ok
        self.message = message
        self.raw = raw
    }
}

/// The manager's client. It talks to whichever member owns the thing being
/// acted on — never through a third instance — which is the rule that keeps a
/// prompt from being relayed by a peer.
public struct FleetClient: Sendable {
    public let token: String
    public let basePath: String
    private let transport: any FleetTransport

    public init(
        token: String, basePath: String = "/dsh-lan",
        transport: any FleetTransport = URLSessionTransport()
    ) {
        self.token = token
        self.basePath = basePath
        self.transport = transport
    }

    /// Percent-encode one path component, keeping `/` encoded: a session or
    /// workspace id is a single component on the wire, and the plugin's routes
    /// reject a slash inside one.
    public static func encodePathComponent(_ value: String) -> String {
        var allowed = CharacterSet.urlPathAllowed
        allowed.remove(charactersIn: "/?#[]@")
        return value.addingPercentEncoding(withAllowedCharacters: allowed) ?? value
    }

    private func decode<T: Decodable>(
        _ type: T.Type, from response: FleetResponse, target: FleetTarget
    ) throws -> T {
        guard (200...299).contains(response.status) else {
            throw FleetError.http(
                target: "\(target)", status: response.status,
                message: Self.message(in: response.body))
        }
        do {
            return try JSONDecoder().decode(T.self, from: response.body)
        } catch {
            throw FleetError.decoding(target: "\(target)", reason: "\(error)")
        }
    }

    /// Pull `error`/`message` out of a failure body without a model per endpoint.
    public static func message(in body: Data) -> String {
        guard
            let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any]
        else {
            return String(data: body, encoding: .utf8)?.trimmingCharacters(
                in: .whitespacesAndNewlines) ?? "no body"
        }
        let error = object["error"] as? String
        let message = object["message"] as? String
        switch (error, message) {
        case (let error?, let message?): return "\(error): \(message)"
        case (let error?, nil): return error
        case (nil, let message?): return message
        default: return "no message"
        }
    }

    private func ack(_ response: FleetResponse, target: FleetTarget) throws -> FleetAck {
        let text = String(data: response.body, encoding: .utf8) ?? ""
        if !(200...299).contains(response.status) {
            throw FleetError.http(
                target: "\(target)", status: response.status,
                message: Self.message(in: response.body))
        }
        let object = try? JSONSerialization.jsonObject(with: response.body) as? [String: Any]
        guard let object, let ok = object["ok"] as? Bool else {
            throw FleetError.decoding(
                target: "\(target)", reason: "a 2xx answer carries no boolean `ok`")
        }
        return FleetAck(ok: ok, message: object["message"] as? String, raw: text)
    }

    private func post(_ path: String, to target: FleetTarget, body: [String: Any]? = nil)
        async throws -> FleetResponse
    {
        let data: Data?
        if let body {
            data = try? JSONSerialization.data(withJSONObject: body)
        } else {
            data = nil
        }
        return try await transport.send(
            FleetRequest(
                method: "POST", target: target, path: "\(basePath)\(path)", token: token, body: data
            )
        )
    }

    /// Read one member's aggregate view of the group, verbatim.
    public func inventoryData(from target: FleetTarget) async throws -> Data {
        let response = try await transport.send(
            FleetRequest(method: "GET", target: target, path: "\(basePath)/inventory", token: token)
        )
        guard (200...299).contains(response.status) else {
            throw FleetError.http(
                target: "\(target)", status: response.status,
                message: Self.message(in: response.body))
        }
        return response.body
    }

    /// Read one member's aggregate view of the group.
    public func inventory(from target: FleetTarget) async throws -> FleetInventory {
        let body = try await inventoryData(from: target)
        do {
            return try JSONDecoder().decode(FleetInventory.self, from: body)
        } catch {
            throw FleetError.decoding(target: "\(target)", reason: "\(error)")
        }
    }

    /// Send one prompt to one session, on the member that owns it.
    public func prompt(to target: FleetTarget, sessionId: String, text: String) async throws
        -> FleetAck
    {
        let response = try await post(
            "/prompt", to: target, body: ["sessionId": sessionId, "prompt": text])
        return try ack(response, target: target)
    }

    /// Register an existing folder as a workspace on one member.
    public func createWorkspace(on target: FleetTarget, path: String, title: String?) async throws
        -> FleetAck
    {
        var body: [String: Any] = ["path": path]
        if let title { body["title"] = title }
        let response = try await post("/workspaces", to: target, body: body)
        return try ack(response, target: target)
    }

    /// Archive one session on the member that owns it.
    public func archiveSession(on target: FleetTarget, sessionId: String) async throws -> FleetAck {
        let path = "/sessions/\(Self.encodePathComponent(sessionId))/archive"
        let response = try await post(path, to: target)
        return try ack(response, target: target)
    }

    /// Delete a workspace from one member's registry, archiving its sessions by
    /// default — the plugin's own rule, passed through rather than second-guessed.
    public func deleteWorkspace(on target: FleetTarget, workspaceId: String, archiveSessions: Bool)
        async throws -> FleetAck
    {
        let path = "/workspaces/\(Self.encodePathComponent(workspaceId))/delete"
        let response = try await post(path, to: target, body: ["archiveSessions": archiveSessions])
        return try ack(response, target: target)
    }
}
