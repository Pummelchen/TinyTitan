# Tool-coverage and language-standard proofs

§1 of the audit standard: a human check delegated to a tool counts as covered only if
the tool is *proved* to catch it, by introducing a deliberate violation and recording the
tool's output. Every run below is against the repository's own build configuration and
committed lint config, not an ad-hoc invocation, and every violating file was deleted
after the run (`git status` clean before this file was committed).

Host: the primary Mac, Xcode 27.0 / Swift 6.4. See `environment.md`.

## Language-standard proof

### Swift — non-Sendable value across a task boundary

File added to a real target (`sources/TinyTitanFormat/AuditProbe.swift`, so it compiles
under `swiftLanguageModes: [.v6]` plus `-warnings-as-errors`), built with `swift build`:

```swift
final class AuditProbeCounter {
    var total = 0
}

func auditProbeCaptureAcrossTaskBoundary() async {
    let counter = AuditProbeCounter()
    let task = Task {
        counter.total += 1
    }
    counter.total += 2          // used after being handed to the task
    await task.value
}
```

Compiler output — an **error**, and the build fails:

```
AuditProbe.swift:7:16: error: sending value of non-Sendable type '() async -> ()'
  risks causing data races [#RegionIsolation::SendingRisksDataRace]
error: Build failed
```

So Swift 6 language mode with complete concurrency checking is in force, not declared.
`SWIFT_STRICT_CONCURRENCY` is not set as a build setting because `swiftLanguageModes:
[.v6]` already implies complete checking; the proof above is what shows that.

**The first probe attempt did not fail, and that is worth recording.** This version
compiled clean:

```swift
let counter = AuditProbeCounter()
await Task { counter.total += 1 }.value        // no use of counter afterwards
```

That is *correct*, not a hole: under region-based isolation a non-Sendable local handed
to a task and never touched again is a `sending` transfer with no race. A reviewer who
wrote the probe expecting an error and read a green build as "strict concurrency is not
enforced" would have filed a false finding, and one who "fixed" the legal case would have
fought the compiler. The provable form is the use-after-hand-off above.

### Swift — force unwrap

Same file, replacing the body with `let unwrapped = value!` for an optional parameter,
run through the gate's exact invocation (`swiftlint lint --strict --no-cache --quiet`,
`.swiftlint.yml` with `force_unwrapping` in `opt_in_rules`):

```
sources/TinyTitanFormat/AuditProbe.swift:2:26: error: Force Unwrapping Violation:
  Force unwrapping should be avoided (force_unwrapping)
```

Covered. Note this is a *different* guard from `tools/lint.sh force-cast`, which rejects
`as!` and `try!` but not a postfix `!`; both are needed and both fire.

### C — implicit function declaration

File added to `sources/TinyTitanKernelsC/`, built with `swift build`:

```c
int audit_probe_implicit_declaration(void) {
  return audit_probe_undeclared_call(1);
}
```

```
AuditProbeImplicit.c:2:10: error: call to undeclared function
  'audit_probe_undeclared_call'; ISO C99 and later do not support implicit function
  declarations [-Wimplicit-function-declaration]
AuditProbeImplicit.c:1:5: error: no previous prototype for function
  'audit_probe_implicit_declaration' [-Werror,-Wmissing-prototypes]
error: Build failed
```

### C — GNU extension

```c
int audit_probe_typeof(int value) {
  typeof(value) copy = value;
  int inner(void) { return copy; }
  return inner();
}
```

```
AuditProbeGnu.c:2:3:  error: call to undeclared function 'typeof';
  ISO C99 and later do not support implicit function declarations
AuditProbeGnu.c:3:19: error: function definition is not allowed here
AuditProbeGnu.c:1:5:  error: no previous prototype for function
  'audit_probe_typeof' [-Werror,-Wmissing-prototypes]
```

The echoed compile command confirms the flags actually in force on the real target:
`-std=c99 -pedantic-errors -Wall -Wextra -Wshadow -Wconversion -Wsign-conversion
-Wcast-qual -Wwrite-strings -Wformat=2 -Wstrict-prototypes -Wmissing-prototypes -Werror
-O2`, `-target-cpu apple-m1`, `-triple arm64-apple-macosx26.0.0`. `typeof` is rejected as
an undeclared call rather than as a named GNU extension; the outcome the standard asks
for — a GNU-only construct fails the build — holds either way.

### Python — the pitfalls §1 names, one by one

Probe file in `benchmark/`, run with the pinned ruff (0.16.7) and the committed
`pyproject.toml` rule set (`select = ["E4","E7","E9","F","W","B","E722","S101","PT"]`).

| Pitfall §1 names | Probe | Result |
| --- | --- | --- |
| bare `except:` | `except:` | **caught** — `E722 Do not use bare \`except\`` |
| mutable default argument | `def f(value, items=[])` | **caught** — `B006` |
| `assert` used for validation | `assert value is not None` | **caught** — `S101` |
| `is` compared against a literal | `if value is 0:` | **caught** — `F632` |
| `subprocess` without `check=True` | `subprocess.run([...], check=False)` | **silent** |
| `time.sleep()` used to synchronize | `time.sleep(2)` | **silent** |
| `open()` without `encoding=` | `open("/tmp/audit_probe.txt")` | **silent** |
| `datetime.now()` without a timezone | `datetime.datetime.now()` | **silent** |
| a test that asserts nothing | empty test body | **silent** (no rule; `PT009`/`PT027` are excluded in `pyproject.toml`) |

Four of the nine enforced by the tool; five silent. `pyproject.toml` stated that "the rest
are the families the audit standard names", which is a claim broader than the
configuration — the same defect class this repository's own lessons list (`Say what a guard
actually reads, not what it intends`). Filed as AUD-103 rather than fixed by moving the
goalposts: the fix is to add the rule families that do cover them, or to keep the human
check and say so.

### Python — the same probe after AUD-103

`select` gained `DTZ`, `ASYNC`, `PLW1510` and `PLW1514` (with `preview = true`, which
`PLW1514` needs and which adds no other violation on the pinned ruff), and the tree was
made clean under it: 42 `subprocess.run` calls now state `check=False` where they read the
exit status on purpose, 91 text-mode `open`/`read_text`/`write_text` gained
`encoding="utf-8"`, and eight `datetime.now()` stamps became UTC — the convention the same
files already used for their `recorded_at` fields. Re-measured with the committed config
against a probe holding all nine:

| Pitfall §1 names | Now | Rule |
| --- | --- | --- |
| bare `except:` | caught | `E722` |
| mutable default argument | caught | `B006` |
| `assert` used for validation | caught | `S101` |
| `is` compared against a literal | caught | `F632` |
| `subprocess` with no `check` at all | caught | `PLW1510` |
| `datetime.now()` without a timezone | caught | `DTZ005` |
| `open()` without `encoding=` | caught | `PLW1514` |
| `time.sleep()` used to synchronize in an `async def` | caught | `ASYNC110` |
| `time.sleep()` used to synchronize in synchronous code | **silent** | no rule exists |
| a test that asserts nothing | **silent** | no rule exists |
| `subprocess.run(..., check=False)` with the status never read | **silent** | `PLW1510` reads the omission, not the value |

Seven of the nine named pitfalls are now enforced by the tool, and the three that are not
are named in `pyproject.toml` itself. The `subprocess` row moved from *without
`check=True`* to *without an explicit `check`*: `PLW1510` allows `check=False` by design,
because these scripts call `pgrep` (exit 1 means "no match"), `sysctl` and `curl` whose
failure each one handles with its own message. What the rule buys is that the choice can no
longer be an accident of the default. `time.strftime()` and `time.localtime()` sit outside
`DTZ`, which reads only `datetime` — `benchmark/tinytitan_benchmark.py:506` still labels a
run directory with a naive local stamp, and no rule sees it.


## Repository gate proofs

Each of the thirteen checks in `tools/lint.sh` was given a deliberate violation and run as
`tools/lint.sh <mode>`; the exit code is the gate's own, captured without a pipe so a
`FAIL` line cannot be reported beside a zero status. The baseline run of the first eleven
on a clean tree is green (`/tmp/tt-audit/lint-baseline.log`: 2,091 functions scanned, 24
scripts, eslint 10.11.0/prettier 3.9.9 in both plugin packages, ruff 0.16.7 clean).
`test-skip` is the twelfth, added when AUD-127 closed, and `unbounded-read` the thirteenth,
added when AUD-142 closed; each has its own probe row below.

| Gate | Violation introduced | Gate result |
| --- | --- | --- |
| `force-cast` | `value as! String` and `try! JSONSerialization…` with no opt-out | exit 1, both lines named |
| `func-length` | a 125-line function | exit 1, `NEW: …auditProbeLongFunction` |
| `unchecked-sendable` | `final class AuditProbeBox: @unchecked Sendable` with no `unchecked-invariant:` note | exit 1, `NEW: …AuditProbeBox` |
| `converter` | `stack[expert] = piece` changed to `stack[len(target["experts"])] = piece`, i.e. file by arrival order | exit 1, `experts landed by arrival order: [3, 0, 7, 1, 5, 2, 6, 4]` — the gate catches the real defect, not a proxy |
| `arch-path` | `BIN=".build/arm64-apple-macosx26.0/release/TinyTitanCLI"` in a `tools/` script | exit 1, file and line named |
| `test-skip` | (a) a temporary `tests/` suite whose `@Test` body opens `guard let path, FileManager…fileExists(atPath: path) else { return }` on an environment variable; (b) the tree as it stood with three sites already gated | (a) exit 1 naming `tests/TinyTitanServer/GateProbeTmp.swift:9`, and exit 0 once the probe file is deleted. (b) exit 1 naming one site the grep sweep had classed as a manual helper and not a defect — `ClientCLITests:170`, the stub server, which is the same early return and is now gated too. The gate was written after the three swept sites were fixed, so it is proved against the probe and that one live find, not re-run over the whole unfixed tree |
| `unbounded-read` | a temporary `sources/` file holding `try Data(contentsOf: url)` (a) with no comment above it and (b) with `// lint:allow-unbounded-read` and no reason | (a) exit 1, `sources/TinyTitan/Infrastructure/ModelIO/UnboundedProbe.swift:2` named; (b) exit 1 again — the marker without a reason fails like no marker at all; (c) the same probe with `lint:allow-unbounded-read <reason>` on the line directly above → exit 0, and exit 0 once the probe file is deleted. On the tree as it stands the gate is clean with 7 exemptions, and it found those 7 the first time it ran: the two `RemoteSnapshotLoader` temp reads (a range this process requested), the two `MetalContext` shader reads (a resource the package ships), the one `AffineSnapshot` `.alwaysMapped` weights map, and the two `CPUQwenCommands` benchmark inputs. Doc comments that merely *name* `Data(contentsOf:)` are not flagged — 3 of the 10 matches on the current tree are prose, and before the row was fixed the ratio was worse, because every reader this audit converted left its explanation behind. |
| `shell` (portability) | `mapfile -t lines …` and a bare `"${args[@]}"` under `set -u` | exit 1, both classes named separately |
| `shellcheck` | unquoted `cd $1` and an unquoted array expansion | exit 1, `SC2068` (error) and `SC2164` (warning) |
| `swiftlint` | a force unwrap, then `as!`/`try!` | exit 1, `force_unwrapping`, `force_cast`, `force_try` all as errors under `--strict` |
| `swift-format` | `func auditProbeBadlyFormatted( ){` / `let   a=1` | exit 1, five `Spacing`/`TrailingWhitespace` errors |
| `javascript` | `let unused = 1` in a plugin `src/` file | exit 1, `no-unused-vars` + `prefer-const` |
| `python` | see the Python tables above | exit 1 for four of the nine at discovery; after AUD-103 the probe in the second table is caught by eight rules, and deleting a `check=False` or an `encoding="utf-8"` from a real file fails `tools/lint.sh python` with exit 1 |

Two observations from the proofs themselves, filed as ledger tasks:

- A first probe of a *109*-line function passed the `func-length` gate. That is correct
  behaviour (the limit is 120), but it is worth recording that this gate is a ratchet on
  length only, and that the probe must exceed the limit to prove anything.
- The `converter` gate runs through `python3.13 || python3` and prints `SKIP: …
  (converter deps unavailable)` with exit 0 when `numpy` is absent, so on a host without
  the converter dependencies the most dangerous defect in this repository — experts filed
  by arrival order, which no downstream check can see — goes unexamined while the gate
  reports nothing failing. This host has `numpy`, so the proof above ran for real.
