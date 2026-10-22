# Page-cache policy (T25/T26)

Status: implemented and tested; the large-memory benchmark (T27) needs a
64/128 GB Mac this checkout does not have.

The routed-expert reader has two modes:

- **bounded** — `F_NOCACHE` reads through `ParallelExpertReader`. The expert
  slot budget is the only cache, so the process's declared footprint is true.
- **page-cache** — ordinary reads. The OS may keep expert bytes resident, which
  makes a repeat read a memory read but creates a second, undeclared cache.

The measured trade (24 GB M3, `docs/perf-campaign-2026-10-18.md` and
`docs/project-maya-scan-2026-10-20.md`): **+4.47% decode at 8-bit** and
**−2.1% at 4-bit**. The page cache is a win where the install fits beside the
rest of the process and a loss where it competes with it.

## Rule

`PageCachePolicyResolver` in
`sources/TinyTitan/Runtime/Configuration/PageCachePolicy.swift`:

| condition | policy | reason |
| --- | --- | --- |
| `TINYTITAN_BOUNDED_IO=0` | page-cache | explicit override |
| `TINYTITAN_BOUNDED_IO=1` | bounded | explicit override |
| any other non-empty `TINYTITAN_BOUNDED_IO` | bounded | invalid value; fail closed |
| physical memory unavailable (`0`) | bounded | cannot size the machine |
| install size unknown or `0` | bounded | cannot compare |
| `installBytes * 2` overflows | bounded | size arithmetic refused |
| `physicalMemory >= 2 * installBytes` | page-cache | RAM comfortably exceeds the install |
| otherwise | bounded | install is too large for the RAM |

The 2x factor is the task's "comfortably exceeds" rule. The install is not the
only resident claim: the resident weights, KV cache, expert slots and the
process itself all live beside the page cache, so equality is not enough.

The install size is the sum of every file size in `manifest.json`
(`PageCachePolicyResolver.installSize`). An overflow is reported as
`UInt64.max`, which the 2x rule then refuses rather than wrapping to a small
install. The decision is made once in `Model.init` and passed to every layer
streamer, so all layers agree.

## Precedence

The environment override is read before the automatic rule and wins in both
directions. An unrecognised value is **bounded**, not "enable": a typo must not
create the undeclared cache. There is deliberately no profile key for this yet;
the profile schema's knobs are the ones T16 measured, and the page-cache arm
has not been measured on a large-memory machine (T27).

## This host

The 24 GB M3 has three installs under `models/`: 19 GB (Ornith 4-bit), 34 GB
(Ornith 8-bit) and 96 GB (Qwen3.8 4-bit). All are above half of RAM (12 GB), so
the automatic rule selects **bounded** for every one of them. The policy is a
no-op on the development host, which is the task's acceptance condition, and
the manual knob (`TINYTITAN_BOUNDED_IO=0`) still exists for a deliberate A/B.

## T27

Caching on/off on a 64/128 GB Mac is not measured here. The task stays
**incomplete (hardware-gated)**: the rule is pinned by unit tests, and the arm
it selects has prior 24 GB measurements, but no large-memory evidence exists in
this checkout.
