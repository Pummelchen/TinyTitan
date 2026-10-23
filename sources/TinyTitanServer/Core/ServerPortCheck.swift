import Darwin

/// Whether the listener's address can be bound right now.
///
/// `main` loads the model before it binds (`ServerBootstrap.bind` runs after
/// the router's preload), so an occupied `--port` used to be reported only
/// after a full load of a multi-gigabyte install. This is the same question
/// asked before the load; the authoritative bind still happens below, and a
/// bind that fails there is still reported the same way.
package enum ServerPortCheck {
    package static func isAvailable(port: Int) -> Bool {
        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        // A failed probe says nothing about the port: let the real bind be the
        // one that reports an unexpected failure.
        guard fd >= 0 else { return true }
        defer { Darwin.close(fd) }
        // The listener sets SO_REUSEADDR, so the probe must too: otherwise a
        // just-restarted server, whose retired connections are in TIME_WAIT,
        // would be refused here and accepted by the bind.
        var reuse: Int32 = 1
        _ = Darwin.setsockopt(
            fd, SOL_SOCKET, SO_REUSEADDR, &reuse,
            socklen_t(MemoryLayout<Int32>.size))

        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port).bigEndian)
        guard inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1 else {
            return true
        }
        let bound: Int32 = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound != 0 else { return true }
        return errno != EADDRINUSE
    }
}
