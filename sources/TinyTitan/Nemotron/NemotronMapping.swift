import Foundation

final class NemotronMappedFile {
    let url: URL
    let size: Int
    let base: UnsafeRawPointer

    init(url: URL) throws {
        self.url = url
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw NemotronInstallError.resident("cannot open \(url.lastPathComponent)")
        }
        defer { close(fd) }
        var status = stat()
        guard fstat(fd, &status) == 0 else {
            throw NemotronInstallError.resident("cannot stat \(url.lastPathComponent)")
        }
        let length = Int(status.st_size)
        guard length > 0 else {
            throw NemotronInstallError.resident("\(url.lastPathComponent) is empty")
        }
        guard let mapped = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0),
            mapped != MAP_FAILED
        else {
            throw NemotronInstallError.resident("cannot map \(url.lastPathComponent)")
        }
        self.size = length
        self.base = UnsafeRawPointer(mapped)
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: base), size)
    }

    func pointer(offset: Int) -> UnsafeRawPointer {
        base.advanced(by: offset)
    }
}

public enum NemotronMath {
    @inline(__always)
    public static func bfloat16(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }
}
