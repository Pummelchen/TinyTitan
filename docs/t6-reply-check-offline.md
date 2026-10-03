# T6 (reply check) on real replies — offline measurement

**Status: closed 2026-10-03.** Measured, the pre-registered gate failed
(66.7% precision, 38.5% recall), and T6 is not wired — see the verdict at the
end. The document is kept whole as the record of the gate as it was fixed
before the run.

T6 is the one side-engine task the 4B cannot do: given one stored fact and one
assistant reply, does the reply contradict the fact? It was measured on eight
authored cases (`docs/side-engine-tasks.md`): 4B 62% (0/3 contradicting replies
caught), 9B and the served 35B 100% (3/3, 5/5). Eight cases, three of them
positives, is not a basis for wiring anything — and the plan's own gate for the
shadow asks for precision on **free-text** replies, where the claim has to be
read out of prose rather than handed over.

This is that measurement, on replies nobody wrote for it: the 30 memory-on
replies from the photograph master-benchmark runs (3 runs x 10 sessions), with
the ground truth that drives those runs (`benchmark/master_scenarios.py`).

Harness: `benchmark/t6_prose_cases.py` (cases + labels), scored by
`benchmark/side_engine_tasks.py`'s own scorer through
`benchmark/side_engine_judges.py`.

## The gate, fixed before the run

The verdict is read against these numbers, registered here first. They restate
the plan's shadow gate (`docs/plan-memory-guard-and-shadow.md`) in the form this
case set can measure:

1. **Precision >= 95%** on fired notes, over every case: of the replies the
   judge marks as contradicting, at least 19 in 20 must really contradict.
2. **Recall >= 50%** on the contradiction cases.
3. **Silence false-alarm rate <= 5%**: a reply that never touches the fact must
   almost never be marked a contradiction. This is the failure that would fire
   on every reply.
4. **Instrument check**: the 4B must stay one-sided on the same cases (it must
   not secretly become sufficient), or the case set is not measuring what
   `docs/side-engine-tasks.md` measured.

Any of 1-3 failing keeps T6 unwired. If all pass, T6 becomes a candidate for an
opt-in, post-hoc, memory-on-only check — still with the quote-only rule, since a
fired note must be checkable against the reply by substring before it is shown.

## Method

**Cases.** For every (run, session, fact) pair:

- a **claim** case is a reply that asserts a value for the fact's key. The truth
  is YES when the asserted value differs from the known value and NO when it
  matches.
- a **silent** case is a reply that never touches the fact. The truth is NO —
  silence is not a contradiction.

`KNOWN` is the fact as the store would hold it (`characters/marcus/eyes = grey`,
`state/inn = the inn has burned down`), and `REPLY` is the recorded reply with
its leading quiz JSON removed. The key is given to the judge, so this measures
the **decision**, not the claim extraction that deployment would still need;
that is the easier half and it is named as the main limitation below.

**Labels.** The detectors that find claims are mechanical and every claim case
is written to a review file and read by hand before the run, because a
mislabeled pair is a wrong answer charged to the judge. Silent cases are
sampled and read the same way; the sample's miss rate is reported.

**Judges.** The served 35B (`qwen3.6_35B_A3B_4Bit`, the candidate) and the 4B
(`qwen3.5_4B_4Bit`, the incumbent, CPU) over the identical case file, greedy,
one word.

## Results

Run 2026-10-03, 417 cases (78 contradictions, 339 not), greedy, one word.
Served judge `qwen3.6-35b-a3b_4-Bit` on the server; 4B judge
`qwen3.5_4B_4Bit` on the CPU, over a 20-case subset (10 positives, 10
negatives sampled with a fixed seed).

| judge | scored | accuracy | precision | recall | NO half | silence alarms | seconds/judgement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| served 35B | 417 | 84.9% | **66.7%** (30/45) | **38.5%** (30/78) | 324/339 | 6/157 = 3.8% | 12.6 |
| 4B CPU | 20 | 55.0% | 100% (1/1) | **10.0%** (1/10) | 10/10 | 0/4 = 0.0% | 130.5 |

**The gate fails on the primary judge: precision 66.7% against a 95% bar, and
recall 38.5% against 50%.** Only the control passes — the 35B fired on 6 of
157 replies that never touched the fact (3.8%, inside the 5% bar). The 4B
reproduces its known one-sidedness on real prose: it fired once in twenty
cases, was right, and missed 9 of 10 contradictions. The instrument check
passes; the case set measures what `docs/side-engine-tasks.md` measured.

What that means in the deployment the plan describes: of every three notes the
shadow would raise, one would be wrong, and it would stay silent on roughly
three of every five real contradictions.

### Where the 35B fails

Per fact, right / fired / positives, over 30 cases each:

| fact | right | fired | positives |
| --- | ---: | ---: | ---: |
| `characters/marcus/eyes` | 100% | 4 | 4 |
| `setting/town` | 100% | 0 | 0 |
| `rules/weather` | 97% | 0 | 1 |
| `state/anyone_left` | 93% | 1 | 1 |
| `rules/ferry` | 90% | 2 | 1 |
| `characters/ines/eyes` | 90% | 0 | 3 |
| `state/tomas` | 87% | 4 | 8 |
| `characters/marcus/knows_photo` | 86% | 6 | 6 |
| `state/ferry` | 82% | 9 | 8 |
| `characters/aldo/eyes` | 80% | 3 | 9 |
| `characters/halvorsen/eyes` | 80% | 2 | 8 |
| `characters/rosa/eyes` | 77% | 2 | 9 |
| `characters/halvorsen/confessed` | 63% | 6 | 15 |
| `state/inn` | 63% | 6 | 5 |

**It is character-inconsistent, which is the signature of attribution rather
than of the attribute.** Marcus's eye colour is right in all 30 cases and
fires on every contradiction; the same fact for Ines, Rosa, Halvorsen and Aldo
is missed or fired on wrongly. These replies put several characters in one
paragraph, and a stated colour has to be tied back to the right person —
the same trap the labeller had to solve (see the test file).

**It reads agreement as contradiction on the state facts.** Six of the fifteen
wrong fires are a reply that says the inn burned while the known fact is that
the inn burned (`state/inn` fires 6 times on 5 positives), and one is "the
ferry was not running" against "the ferry stopped running for good". The judge
appears to match the words rather than the states.

**It misses inversions inside long prose.** Most of the 48 missed
contradictions are a reply stating a colour other than the known one
(`Dr. Halvorsen's grey eyes` against brown), phrased in a paragraph of
narrative.

### Two labels found wrong after the run, and why they are not silently fixed

The hand review was thorough but not perfect. Two of the fifteen wrong fires
are not judge errors:

- `r1s10 characters/marcus/knows_photo` — known "Marcus knows what the
  photograph shows"; the reply says the photograph stayed "hidden away from
  Marcus's knowledge". That is a contradiction, and the judge was right.
- `r3s7` — same fact, reply "a secret that was not his to know". Also a
  contradiction.

Both were labelled NO because the detector's negative branch only recognised
`does not know`/`unaware`, not `hidden from ... knowledge`/`not his to know`.
A third, `r1s6 rules/ferry` ("the ferry had not run on Sunday"), is the
ambiguity already excluded at its two sibling sessions and should have been
excluded here too.

Correcting only these three raises the 35B to **71.1% precision** (32/45) and
leaves recall unchanged. Post-hoc label fixes chosen after seeing the model's
answers are not evidence, so the table above keeps the pre-registered labels
and this paragraph exists to be honest about the direction and size of the
error. The gate fails either way, by a wide margin, on both precision and
recall.

### Cost, and a difference from the authored cases

A judgement costs **12.6 s on the served 35B** here against the 2.5 s measured
on the eight authored cases, because each call carries a whole session reply
(~700 prompt tokens) rather than two short lines; the 4B's 130.5 s on the CPU
was measured while the 35B was still running, so it is an upper bound. Even at
the authored-prompt cost, one judgement per fact per reply is a generation the
person's engine cannot serve from.

## Verdict: do not wire T6 on this evidence

The eight authored cases said 100% (5/5, 3/3) and the served model looked like
the answer. On real full-reply prose the same judge is at 66.7% precision and
38.5% recall — a note that is wrong one time in three, silent on three of five
contradictions, and paid for with the main engine's time. The pre-registered
gate was precision >= 95% and recall >= 50%: **NO-GO**, and the honest reading
is that the authored cases measured the prompt's shape, not the deployment.

This is the outcome the plan anticipated ("deployment requires extracting the
claim from prose first, and that is the step where every small model tested
did badly"): the decision step is not the only hard part, and on real prose it
is not the good part either.

What would change the verdict, in the order worth trying:

1. **Fix the attribution, not the model.** The strongest signal here is that
   one character's fact scores 100% while the identical fact for four others
   does not — so a caller that passes a resolved claim (`the reply states
   colour X for character Y`, quoted) rather than free text should be measured
   before any model change. That is a one-day measurement with this harness.
2. **Then re-measure with the same 417 cases** and the same gate. If
   attribution lifts precision over 95% and recall over 50%, T6 becomes an
   opt-in, post-hoc, memory-on-only note with the quote-only rule.
3. Only if that fails, try a larger judge or the two-judge agreement the
   2026-09-19 comparison suggested — and expect the engine-time price above.

Until then T6 stays unwired, and this document plus
`benchmark/t6_prose_cases.py` are the record. The harness, the labels and the
run artifacts are reproducible: the case file and both judge outputs are in
`.build/benchmark-logs/t6-prose/` (not committed, per the benchmark
convention), and the scorer prints the gate table above.

## Limitations

- The claim extraction from prose is bypassed: the judge is told which fact to
  check. The plan names extraction as the step every small model failed
  (71-83%), so this measures the upper bound of the deployed pipeline.
- `KNOWN` is the world's ground truth, not the store the auto arm actually
  wrote. A store-based repeat is the follow-up; the guard-protected store is
  what the plan requires before T6 is worth wiring at all.
- The photograph world is one domain (fiction, fixed attributes, a handful of
  state changes). It is the world with the most prose and the most recorded
  replies; the coder and ops worlds have far fewer.
- Labels are mechanical detection plus hand review, so a contradiction phrased
  outside every detector is a missed positive. The sampled silent check bounds
  that error rather than removing it. Two such misses were found after the run
  and are reported above rather than patched into the labels.
- One judge call per fact per reply is the deployment shape, but a real shadow
  would also have to decide which facts to check; here every fact is checked
  against every reply, which is the most favourable version of that step.
