import Foundation

extension DoctorReport {
    /// Human-readable output. Every number here is the JSON's number, so the
    /// two surfaces cannot drift.
    func renderedHuman() -> String {
        var lines: [String] = ["TinyTitan doctor"]
        let gpuName = machine.gpuName ?? "no Metal device"
        lines.append(
            "machine: \(machine.chip) (\(gpuName))  macOS \(machine.osVersion)  "
                + "RAM \(formatBytes(machine.ramBytes))")
        if modelsDirExists {
            let installedBytes = models.reduce(Int64(0)) { $0 + max(0, $1.sizeBytes) }
            lines.append(
                "models: \(modelsDir)  (\(models.count) installed, "
                    + "\(formatBytes(UInt64(installedBytes))) on disk)")
        } else {
            lines.append("models: \(modelsDir) — not found")
        }
        let free = disk.freeBytes.map(formatBytes) ?? "unavailable"
        let capacity = disk.capacityBytes.map(formatBytes) ?? "unavailable"
        lines.append(
            "disk: \(free) free of \(capacity) at \(disk.path)")
        lines.append(
            "gpu: \(machine.gpuFamily ?? "unknown")  "
                + "apple10=\(yesNo(machine.apple10TensorOps))  "
                + "metal4=\(yesNo(machine.metal4))  "
                + "ane=\(yesNo(machine.ane))")
        lines.append("tensor-ops: \(tensorOps.path) (\(tensorOps.reason))")
        lines.append("calibration: \(calibrationLine)")
        lines.append("other TinyTitan model processes: \(processLine)")
        if !messages.isEmpty {
            for message in messages {
                lines.append("notice: \(message)")
            }
        }
        lines.append("")
        lines.append("models:")
        if models.isEmpty {
            lines.append("  (none)")
        }
        for model in models {
            lines.append("  \(model.id)  (\(model.family), \(model.weightBits)-bit)")
            lines.append("    dir: \(model.directory)")
            lines.append("    size on disk: \(formatBytes(UInt64(max(0, model.sizeBytes))))")
            if let fit = model.fit {
                let cache: String
                if fit.expertCacheBytes == 0, fit.cacheFloorBytes == 0 {
                    cache = "cache: n/a  "
                } else {
                    cache =
                        "cache: \(formatBytes(fit.expertCacheBytes)) "
                        + "(floor \(formatBytes(fit.cacheFloorBytes)))  "
                }
                lines.append(
                    "    weights: \(formatBytes(fit.weightsBytes))  "
                        + "kv@\(fit.contextTokens) (\(fit.kvPrecision)): "
                        + "\(formatBytes(fit.kvBytes))  "
                        + cache
                        + "total: \(formatBytes(fit.totalBytes))")
                lines.append(
                    "    fit: \(yesNo(fit.fits)) "
                        + "(\(percent(fit.utilization)) of "
                        + "\(formatBytes(fit.ramBytes)))")
            } else {
                lines.append("    fit: not estimated (\(model.note ?? "no manifest"))")
            }
            if let slots = model.expertSlots {
                lines.append(
                    "    expert slots: \(slots)  "
                        + "prefill chunk: \(model.prefillChunkTokens.map(String.init) ?? "family default")"
                )
            } else {
                lines.append(
                    "    expert slots: n/a  prefill chunk: "
                        + "\(model.prefillChunkTokens.map(String.init) ?? "family default")")
            }
            if let pageCache = model.pageCache {
                lines.append(
                    "    page cache: \(pageCache.policy) — \(pageCache.reason)")
            }
        }
        if !skipped.isEmpty {
            lines.append("")
            lines.append("skipped directories:")
            for item in skipped {
                lines.append("  \(item.name): \(item.reason)")
            }
        }
        return lines.joined(separator: "\n")
    }

    private var calibrationLine: String {
        guard calibration.configured else { return "none" }
        guard calibration.status == "valid" else {
            return "invalid (\(calibration.detail ?? "unknown error"))"
        }
        var parts = [
            "measured: \(yesNo(calibration.measured ?? false))",
            "source: \(calibration.source ?? "unknown")",
            "target: \(calibration.modelID ?? "unknown") @ "
                + "\(calibration.contextTokens.map(String.init) ?? "unknown")",
            "machine matches: \(yesNo(calibration.machineMatches ?? false))",
        ]
        if let path = calibration.path {
            parts.append("path: \(path)")
        }
        return "valid (\(parts.joined(separator: ", ")))"
    }

    private var processLine: String {
        guard !otherModelProcesses.isEmpty else { return "none" }
        return
            otherModelProcesses
            .map { "pid \($0.pid) \($0.name)" }
            .joined(separator: ", ")
    }

    private func yesNo(_ value: Bool) -> String { value ? "yes" : "no" }

    private func percent(_ value: Double) -> String {
        String(format: "%.1f%%", value * 100)
    }

    private func formatBytes(_ bytes: UInt64) -> String {
        let gib = Double(bytes) / 1_073_741_824
        if gib >= 1 {
            return String(format: "%.2f GiB", gib)
        }
        let mib = Double(bytes) / 1_048_576
        if mib >= 1 {
            return String(format: "%.2f MiB", mib)
        }
        let kib = Double(bytes) / 1_024
        if kib >= 1 {
            return String(format: "%.2f KiB", kib)
        }
        return "\(bytes) B"
    }
}
