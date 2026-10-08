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
`DTZ`, which reads only `datetime` — `benchmark/tinytitan_benchmark.py:604` still labels a
run directory with a naive local stamp, and no rule sees it.


## Repository gate proofs

Each of the eighteen checks in `tools/lint.sh` was given a deliberate violation and run as
`tools/lint.sh <mode>`; the exit code is the gate's own, captured without a pipe so a
`FAIL` line cannot be reported beside a zero status. The baseline run of the first eleven
on a clean tree is green (`/tmp/tt-audit/lint-baseline.log`: 2,091 functions scanned, 24
scripts, eslint 10.11.0/prettier 3.9.9 in both plugin packages, ruff 0.16.7 clean).
`test-skip` is the twelfth, added when AUD-127 closed, and `unbounded-read` the thirteenth,
added when AUD-142 closed; `file-length` is the fourteenth, added when AUD-151 closed, and
`test-hollow` the fifteenth, added when AUD-128 closed. Those last two are the only ones
whose probe must run in a copied tree — see the note under the table. `library-facade` is
the sixteenth, added when AUD-158 closed; its probe edits tracked files in place and puts
them back, and the last two arms of that run are the proof that it did — `git status` byte
for byte as it was before, and the gate green again. `docs` is the seventeenth, added when
AUD-160 closed; it is the only gate whose subject is the documentation of the others,
and its probe is committed (`AUD-160-docs-probe.py`) rather than narrated, because
a gate that edits tracked documents to prove itself has to be re-runnable by the person
who reads this table. `stdout-clean` is the eighteenth, added when AUD-180 closed; its
first run was the finding — over the unmodified tree it named exactly the six `print`
calls the audit had just read, which is the probe and the RED at once. Each has its own
row below.

| Gate | Violation introduced | Gate result |
| --- | --- | --- |
| `force-cast` | `value as! String` and `try! JSONSerialization…` with no opt-out | exit 1, both lines named |
| `func-length` | a 125-line function | exit 1, `NEW: …auditProbeLongFunction` |
| `file-length` | five arms, all in a copied tree: (a) `HEAD`'s pre-split `Engine.swift` (529 physical lines) as the only source; (b) the same file after the split (286); (c) a fabricated file at exactly 500, then at 501; (d) a tree whose `sources/` holds no Swift at all; (e) the same tree with `PATH=/nonexistent`, so the counter cannot start | (a) exit 1, `OVER 529 sources/Engine.swift` — the gate bites on the real defect it was written for, not a proxy; (b) exit 0; (c) exit 0 at 500 and exit 1 at 501, naming `OVER 501 sources/Exact.swift`, so the boundary is `> limit` as the rule is written; (d) exit 1, `the length counter found no Swift sources under sources/`; (e) exit 1, `the length counter exited 127; it measured nothing`. On the tree as it stands the gate is clean and prints its receipt: `ok (largest: 496 sources/TinyTitan/Runtime/Inference/RealForwardRunner.swift, 370 files scanned)` |
| `unchecked-sendable` | `final class AuditProbeBox: @unchecked Sendable` with no `unchecked-invariant:` note | exit 1, `NEW: …AuditProbeBox` |
| `converter` | `stack[expert] = piece` changed to `stack[len(target["experts"])] = piece`, i.e. file by arrival order | exit 1, `experts landed by arrival order: [3, 0, 7, 1, 5, 2, 6, 4]` — the gate catches the real defect, not a proxy |
| `arch-path` | `BIN=".build/arm64-apple-macosx26.0/release/TinyTitanCLI"` in a `tools/` script | exit 1, file and line named |
| `stdout-clean` | ten arms. On the live tree, in order: (1) the tree *before* the fix, untouched; (2) `print("audit-180 mutation probe")` appended to `sources/TinyTitanFormat/SSDAIManifestV1.swift`; (3) the same file with only `// lint:allow-stdout <reason>` above the same call; (4) the file reverted. In a copied tree (`/tmp/sc-gate`, `tools/lint.sh` plus a synthetic `Package.swift` and empty target dirs): (5) `TinyTitanLib` renamed out of the manifest; (6) a declared `path:` pointing at a directory that does not exist; (7) closure dirs present but holding no `.swift`; (8) a `print` in the root target; (9) a `print` only in the target the root depends on; (10) `FileHandle.standardOutput.write` and `debugPrint`; then the same two sites exempted, and an `executableTarget` that depends *on* the library given a `print` | (1) exit 1 naming exactly the six predicted sites — `RealForwardRunner+DecodeMoE.swift:485` and `+PrefillLayer.swift:222,223,226,229,233` — which is the finding and the proof the scope is right, taken together; (2) exit 1, `sources/TinyTitanFormat/SSDAIManifestV1.swift:486` named; (3) exit 0, so the marker works and needs a reason; (4) exit 0, `ok (4 targets in TinyTitanLib's closure, 211 Swift files, no stdout write)`, with `git status` back to the three files this change touches; (5) exit 1 `TinyTitanLib is not declared in Package.swift; the gate read nothing`; (6) exit 1 `declares TinyTitan at sources/TinyTitan, which does not exist`; (7) exit 1 `walked no Swift file; a scan that read nothing is not a pass` — the three arms that must never bless a tree the gate could not read; (8)(9)(10) all exit 1 naming the file, line and call, (9) being the arm the old rule-1 scanner could not reach because it only walked `sources/TinyTitanLib/`; the exempted pair exit 0; and the dependent target's `print` exits 0 with `2 targets in TinyTitanLib's closure`, which proves the walk follows `dependencies:` and not the reverse — a front end keeps its stdout |
| `test-skip` | (a) a temporary `tests/` suite whose `@Test` body opens `guard let path, FileManager…fileExists(atPath: path) else { return }` on an environment variable; (b) the tree as it stood with three sites already gated | (a) exit 1 naming `tests/TinyTitanServer/GateProbeTmp.swift:9`, and exit 0 once the probe file is deleted. (b) exit 1 naming one site the grep sweep had classed as a manual helper and not a defect — `ClientCLITests:170`, the stub server, which is the same early return and is now gated too. The gate was written after the three swept sites were fixed, so it is proved against the probe and that one live find, not re-run over the whole unfixed tree |
| `test-hollow` | seven arms in a copied tree (`/tmp/th-gate`, `tools/lint.sh` + `tests/` only), all in one probe file: (1) a body whose only statement is `_ = (SomeFunction, "a string")`; (2) a body whose only `#expect` is inside a `//` comment; (3) a body holding `{`/`}` and the words `@Test func`/`#expect` inside a `"…"` and a `#"{…}"#` raw string; (4) a body that asserts only through a same-file helper; (5) a body whose only assertion is `try`; (6) arm 1 with `lint:allow-hollow-test <reason>` above the `@Test`; (7) a normal asserting test placed *after* arm 3 | (1)(2)(3) exit 1, each named with file, line and test name — so the comment and the strings are not read as assertions; (4)(5)(6) are not flagged, which is the rule as written (a throwing body fails when it throws, and a helper's `#expect` is a real assertion reached by the call); (7) still scanned and passing proves arm 3's braces did not derail the match. Then the tree as it stood *before* the fix: with `HEAD`'s `RMSNormReferenceTests.swift` copied in, exit 1 naming `mismatchedLengthsTrap` at line 55 — the gate bites on the audited defect itself, not a proxy. Guards: with `tests/` moved away, exit 1 `found no @Test bodies`; with a `python3` that exits 3 first on `PATH`, exit 1 `the hollow-body counter exited 3; it measured nothing`. Clean run prints `ok (1630 test bodies scanned, none hollow)` in 2.4 s |
| `unbounded-read` | a temporary `sources/` file holding `try Data(contentsOf: url)` (a) with no comment above it and (b) with `// lint:allow-unbounded-read` and no reason | (a) exit 1, `sources/TinyTitan/Infrastructure/ModelIO/UnboundedProbe.swift:2` named; (b) exit 1 again — the marker without a reason fails like no marker at all; (c) the same probe with `lint:allow-unbounded-read <reason>` on the line directly above → exit 0, and exit 0 once the probe file is deleted. On the tree as it stands the gate is clean with 7 exemptions, and it found those 7 the first time it ran: the two `RemoteSnapshotLoader` temp reads (a range this process requested), the two `MetalContext` shader reads (a resource the package ships), the one `AffineSnapshot` `.alwaysMapped` weights map, and the two `CPUQwenCommands` benchmark inputs. Doc comments that merely *name* `Data(contentsOf:)` are not flagged — 3 of the 10 matches on the current tree are prose, and before the row was fixed the ratio was worse, because every reader this audit converted left its explanation behind. |
| `library-facade` | sixteen arms, run against the live tree and reverted after each (the last two arms are the revert itself): surface — (1) a new `public struct` in `Session.swift`; (2) `Engine.swift`'s `prefillChunk` renamed while its allowlist row stays; (3) one `:func:respond` row deleted while both overloads stay public; (4) a *unique* baseline row deleted; (5) the allowlist file moved away; (6) `@discardableResult public func` — the form a line-start matcher would have missed; (7) a bare `public` with nothing nameable after it; (8) `open class`; (9) `FACADE_UPDATE=1` on a tree with a row missing. Rule 1 — (10) `import NIOCore` at the top of `Engine.swift`; (11) a `public func` whose body is `print("mutant")`; (12) the word `stdout` in a doc comment. Scanner honesty — (13) `tools/library-facade-rules.py` with `ROOT` at a directory holding an empty `sources/TinyTitanLib`; (14) the same with a `ROOT` that does not exist; and (15)(16) the tree re-checked after every revert | (1)(3)(4)(8) exit 1 with the row named — (3) is the multiset property: deleting one of two identical rows reads as *new* surface, so an overload cannot quietly disappear behind its twin's exemption; (4) is the same rule on a row that appears once; (2) exit 1, `STALE: …Engine.swift:func:prefillChunk`, so an exemption cannot be reused; (5) exit 1, `no allowlist at tools/library-facade-baseline.txt`, never a silent bless; (6) exit 1 naming `audit158Attr`, which proves the attribute-stripping; (7) exit 1, `UNRESOLVED:` with the file and line, so an unnameable `public` fails instead of being skipped; (9) exit 0 and `allowlist rewritten`, the deliberate path; (10) exit 1, `NIO sources/…:2: import NIOCore`; (11) exit 1, `STDOUT sources/…`; (12) exit 0 — prose that *states* the rule does not trip the gate, which matters because `ServerLog.swift`'s own doc comment does exactly that; (13) exit 1 `cannot scan …: no Swift files`; (14) exit 1 `cannot scan …: not a directory`; (15) `git status` identical to the pre-probe state and (16) the gate green again. Clean tree: `ok (88 public declarations, all allowlisted)` and `ok (no NIO import, no stdout write, in 33 files)` |
| `docs` | twenty-four arms in `AUD-160-docs-probe.py`, each editing one tracked document or `tools/lint.sh` itself and reverting it before the next: (1) `CONTRIBUTING.md`'s count read as sixteen against a derived seventeen; (2) a mode name that answers to nothing, planted in a paragraph that lists three that do; (3) the handover's restated ledger counts off by one row; (4) `` `v5.18` -> `d30de44…` ``, the *tag object* sha; (5) `` `v5.18` -> `1111111…` ``, a sha that is neither; (6) a usage-header label with no `case` arm behind it; (7) an `all)` chain entry no mode invokes; (8) a real mode missing from the unknown-check message, which is what `--help` prints; the five AUD-162 additions read the counts the repository can still compute: (19) `CONTRIBUTING.md`'s `(14 rows` against `wc -l` of the ratchet file named beside it, (20) the same claim attributed to a `tools/*.txt` that does not exist, (21) the handover's `16 files under `benchmark/golden/`` against `git ls-files`, (22) its `16 targets in `tools/golden-baseline.sh`` against the target list that script prints when a target is typo'd, and (23) `48 rows per token`, a number with no ratchet file within 300 characters, and (24) a row of `tool-coverage.md` itself given a fourth column, which is how the table check proves it still reads the documents the number checks are told to skip; and four arms that must **not** fail, because a gate that fires on these is a gate people switch off — (9) a wrong count under `## What has landed`, (10) "Verification: tools/lint.sh ran sixteen gates, all clean", (11) a number with no `tools/lint.sh` or CI anchor nearby, (12) `RELEASE.md`'s quoted retired value, `"eleven checks as of …"`, (17) a table row with one cell too many — located from the file rather than hard-coded, so the arm outlives the next edit to that table — and (18) a row whose pipes are `\|` escapes inside a code span, which is how `docs/release-process.md`'s `pgrep` row is correct at nine pipes; then (13) `ROOT=/tmp` and (14) `ROOT` at a directory that does not exist, and (15)(16) the revert proof | (1)(2)(3)(4)(5) exit 1 with the file, the line and the disagreeing pair named — (3) is the bug this audit shipped once, in a commit message; (4) exit 1 *and* says "that is the *tag object*, and the tools key on the commit", (5) exit 1 without that hint, so the two diagnoses stay distinguishable; (6)(7)(8) exit 1 on the script's own internal disagreement, which is checked before any document is judged against it; (9)(10)(11)(12)(18) exit 0, and (17) exit 1 naming the line and both column counts — the shape this gate's own author typed while editing the handover, caught only by reading the line back, which is why it is in the gate; (19)(21)(22) exit 1 naming the document, the claim as written and the computed count; (20) exit 1 `cannot count the rows a claim of “14 rows” rests on (tools/no-such-baseline.txt)` -- a derivation that cannot run is reported, not waved through; (23) exit 0, because prose about a tensor is not a claim about a file here; (24) exit 1, naming the line and both column counts, in a file `check_counts` never looks at; (13) exit 1 `no tracked documents to compare`; (14) exit 1 `cannot read …/tools/lint.sh` — both of these were crashes with a traceback before this probe ran, and a traceback beside exit 1 is not a gate verdict; (15) `git status` identical to the pre-probe state; (16) the gate green again. On the tree as it stands: `ok (82 documents against 17 gates derived from tools/lint.sh, table shape in all 108; 1 owner-file note(s) reported and not enforced)`, the one note being `AGENTS.md`'s count, printed every run and deliberately not a failure |
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

Two more, from the `file-length` proof added on 2026-10-06. Neither is a ledger task; both
are traps for whoever probes a gate here next:

- **The first four arms were vacuous.** They set `ROOT=/tmp/fabricated` in the environment
  and each reported exactly what the *real* tree reports — `ok (largest: 496 …, 370 files)`
  — while claiming to measure a 501-line file. `tools/lint.sh` recomputes and re-exports
  `ROOT` from its own path near the top of the file, so no caller-supplied `ROOT` reaches a
  check. A gate whose input root is derived internally can only be probed by running that
  same script from a copied tree, which is how the five arms above were taken. It is worth
  checking the *message* names the probe file, not only the exit code: an `ok` line that
  reports the real tree's largest file is the tell.
- **A gate's exit status must mean one thing.** The first draft had python `sys.exit(1)` on
  finding offenders, and the shell treated any non-zero as a verdict; when `MAX_FILE_LINES`
  turned out not to be exported, the `KeyError` traceback also exited 1, and the failure
  printed `found no Swift sources under sources/` — a wrong diagnosis of a gate that had
  not run, which is the shape AUD-104 (`a gate that reports nothing`) and AUD-127 (`a gate
  that passes without asserting`) were filed for. The counter now
  always exits 0 when it completed and the shell decides the verdict from an `OVER` line,
  so a non-zero status can only mean "this did not run"; arm (e) above pins it.

One more, from the `test-hollow` proof added on 2026-10-07, and it is the rule rather
than a trap:

- **The sweep's count and the gate's count are different measurements, and the difference
  is the design.** A first pass over `tests/` flagged 83 of 1,630 `@Test` bodies for holding
  no assertion token. Read one by one, 51 of them assert through a helper in the same file
  (`check("fresh")`, `verify(...)`) whose own body carries the `#expect`, and 30 assert only
  that a call does not throw — which in Swift Testing *is* an assertion, because a body that
  throws fails the test. Both classes are named for what they check ("a uniform role
  override is accepted", "the fused read kernel builds"), so the gate accepts them, and the
  rule had to be written helper-aware or it would have failed the tree over fifty bodies that
  work. That leaves 2, and only after the arithmetic was done did the audit find a third
  pattern the token scan cannot see: a body that calls a function and **discards** the result
  (`_ = try MoE(context: context, topKExperts: 10)`), which throws-nothing passes even though
  the test's name claims the runner carries `k`. Two of those were in `RouterTopKTests`, and
  they are what the `#expect(moe.maxStreamedExperts == 10)` replacements are for. A rule
  cannot catch a discarded value; that one needs a reader.
