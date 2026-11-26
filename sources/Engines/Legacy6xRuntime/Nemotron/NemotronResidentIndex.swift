import Foundation

/// The resident index parser, split out of the reader so the file
/// stays under the repository's 500-line production limit.
extension NemotronInstall {
    static func readResidentIndex(
        _ file: NemotronMappedFile
    ) throws -> [String: NemotronTensorRef] {
        guard file.size >= 24 else {
            throw NemotronInstallError.resident("model_weights.bin is shorter than its header")
        }
        let header = file.base.assumingMemoryBound(to: UInt64.self)
        let indexSize = Int(UInt64(littleEndian: header[0]))
        let residentSize = Int(UInt64(littleEndian: header[1]))
        let count = Int(UInt64(littleEndian: header[2]))
        guard indexSize >= 24, indexSize + residentSize <= file.size,
            24 + count * 72 <= indexSize
        else {
            throw NemotronInstallError.resident("index header is inconsistent")
        }
        var tensors: [String: NemotronTensorRef] = [:]
        tensors.reserveCapacity(count)
        let base = file.base
        for index in 0..<count {
            let entry = base.advanced(by: 24 + index * 72)
            let nameOffset = Int(entry.load(fromByteOffset: 0, as: UInt32.self))
            let nameLength = Int(entry.load(fromByteOffset: 4, as: UInt16.self))
            let dtype = entry.load(fromByteOffset: 6, as: UInt8.self)
            let fileOffset = Int(entry.load(fromByteOffset: 8, as: UInt64.self))
            let sizeBytes = Int(entry.load(fromByteOffset: 16, as: UInt64.self))
            let shape = (0..<4).map { slot in
                Int(entry.load(fromByteOffset: 24 + slot * 4, as: UInt32.self))
            }
            let scaleOffset = Int(entry.load(fromByteOffset: 40, as: UInt64.self))
            let scaleSize = Int(entry.load(fromByteOffset: 48, as: UInt64.self))
            let biasOffset = Int(entry.load(fromByteOffset: 56, as: UInt64.self))
            let biasSize = Int(entry.load(fromByteOffset: 64, as: UInt64.self))
            guard nameOffset >= 0, nameLength > 0, nameOffset + nameLength <= indexSize
            else {
                throw NemotronInstallError.resident("entry \(index) has an out-of-range name")
            }
            guard
                let name = String(
                    bytes: UnsafeRawBufferPointer(
                        start: base.advanced(by: nameOffset), count: nameLength),
                    encoding: .utf8)
            else {
                throw NemotronInstallError.resident("entry \(index) has a non-UTF8 name")
            }
            let logicalShape = shape.prefix { $0 != 0 }
            var bits: Int?
            if dtype == 0 {
                let cells = logicalShape.reduce(1, *)
                guard cells > 0, sizeBytes * 8 % (cells * 4) == 0 else {
                    throw NemotronInstallError.resident("\(name) has an impossible packed size")
                }
                let width = sizeBytes * 8 / cells
                guard width == 4 || width == 8 else {
                    throw NemotronInstallError.resident("\(name) is \(width)-bit")
                }
                bits = width
            }
            tensors[name] = NemotronTensorRef(
                name: name, dtype: dtype, shape: Array(logicalShape),
                fileOffset: fileOffset, sizeBytes: sizeBytes,
                scaleOffset: scaleOffset, scaleSize: scaleSize,
                biasOffset: biasOffset, biasSize: biasSize, bits: bits)
        }
        return tensors
    }
}
