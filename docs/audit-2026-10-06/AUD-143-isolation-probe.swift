// Shape under test: a continuation stored in actor state from inside
// withCheckedContinuation's closure, resumed later from another actor method,
// wrapped in withTaskCancellationHandler. Mirrors ModelRouter.waitForTurn().
// Run: swiftc -O -strict-concurrency=complete AUD-143-isolation-probe.swift -o probe
//      ./probe          # waiter array unguarded, as the router has it
//      ./probe --boxed  # the same array behind a Mutex
// Not a gate and not part of any target: it is AUD-143's evidence, kept so the
// repro is runnable without this repository.
//
// Liveness is the measurement, not a hang: every waiter re-checks under a bound,
// so a lost wakeup shows up as a "spun" result rather than a deadlock.

import Foundation
import Synchronization

let boxedMode = CommandLine.arguments.contains("--boxed")
actor Router {
    struct Waiter: Sendable {
        let id: UUID
        let continuation: CheckedContinuation<Void, Never>
    }

    private let lock = Mutex<[Waiter]>([])
    private var waitersRaw: [Waiter] = []
    private var switching = false
    private var busy = false
    private var resident = "alpha"
    private var spins = 0

    init(boxed: Bool) { self.boxed = boxed }
    private let boxed: Bool

    private func withWaiters<T>(_ body: (inout [Waiter]) -> T) -> T {
        guard boxed else { return body(&waitersRaw) }
        return lock.withLock { body(&$0) }
    }

    private func waitForTurn() async {
        let id = UUID()
        await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                if Task.isCancelled {
                    continuation.resume()
                } else {
                    withWaiters { $0.append(Waiter(id: id, continuation: continuation)) }
                }
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
    }

    private func cancelWaiter(_ id: UUID) {
        let waiter = withWaiters { list -> Waiter? in
            guard let index = list.firstIndex(where: { $0.id == id }) else { return nil }
            return list.remove(at: index)
        }
        waiter?.continuation.resume()
    }

    private func wakeWaiters() {
        let woken = withWaiters { list in
            let taken = list
            list.removeAll()
            return taken
        }
        for waiter in woken { waiter.continuation.resume() }
    }

    /// Returns true if the lock was taken, false if it gave up after `limit`
    /// wakeups — a give-up with the target resident and free is a lost wakeup.
    func acquire(_ target: String, limit: Int) async -> Bool {
        var attempts = 0
        while attempts < limit {
            attempts += 1
            if !switching && !busy && resident == target {
                busy = true
                return true
            }
            // The bound is only observable on a wake, so never sleep on the
            // final attempt — that would hang instead of reporting a spin.
            if attempts >= limit { break }
            await waitForTurn()
        }
        spins += 1
        return false
    }

    func release() {
        busy = false
        wakeWaiters()
    }

    func switchTo(_ target: String) async {
        switching = true
        resident = target
        switching = false
        wakeWaiters()
    }

    func settle(rounds: Int) async {
        for round in 1...rounds {
            await switchTo(round.isMultiple(of: 2) ? "alpha" : "beta")
        }
    }

    func report() -> (spins: Int, queued: Int) {
        (spins, withWaiters { $0.count })
    }
}

let contenders = 12
for boxed in [boxedMode] {
    var spun = 0
    var crashed = 0
    for iteration in 1...40 {
        let router = Router(boxed: boxed)
        let outcomes = await withTaskGroup(of: Bool.self) { group -> [Bool] in
            for _ in 0..<contenders {
                group.addTask {
                    let got = await router.acquire("beta", limit: 60)
                    if got { await router.release() }
                    return got
                }
            }
            group.addTask { await router.switchTo("beta"); return true }
            group.addTask { await router.settle(rounds: 30); return true }
            var results: [Bool] = []
            while let result = await group.next() { results.append(result) }
            return results
        }
        let stats = await router.report()
        if outcomes.contains(where: { !$0 }) || stats.queued != 0 {
            spun += 1
            print("  boxed=\(boxed) iteration \(iteration): lost wakeup — "
                + "\(outcomes.filter { !$0 }.count) gave up, \(stats.queued) still queued")
        }
    }
    print("boxed=\(boxed) (lock guarding the waiter list: "
        + "\(boxed ? "yes" : "no")): \(spun)/40 iterations with a lost wakeup, "
        + "\(crashed)/40 crashed")
}
print("probe finished")
