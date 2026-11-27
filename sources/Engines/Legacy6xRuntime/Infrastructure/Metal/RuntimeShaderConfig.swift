import Foundation
import Metal
import TinyTitanCore

/// The runtime serves every shipped family through the generic Core shader
/// modules. A 7.0 variant that needs only those uses `MetalContext`'s Core
/// default directly; this entry point remains so the 6.x engine and its probes
/// build a context through one call.
extension MetalContext {
    public static func runtimeDefault(device: MTLDevice? = nil) throws -> MetalContext {
        let dev: MTLDevice
        if let device {
            dev = device
        } else {
            guard let created = MTLCreateSystemDefaultDevice() else { throw MetalError.noDevice }
            dev = created
        }
        return try MetalContext(device: dev)
    }
}
