#!/usr/bin/env python3.13
"""The ten master prompts as runnable scenarios.

`docs/benchmark-master-prompts.md` is the design; this is the executable form.
Each scenario is a session-1 brief that fixes a set of facts, a per-session
instruction, the sessions that change a fact, and a JSON quiz scored against
what is true *by then*. Nothing here is memory-specific: the driver decides
which arm (summary, memory auto) carries the facts.

    python3.13 benchmark/master_scenarios.py --check
    python3.13 benchmark/master_scenarios.py --list

**Foundation and carryable are derived, not authored.** A key whose truth never
changes is foundation — the no-regression check. A key that changes at least
once is carryable: it is what a client's own summary loses and what the
transition score counts. Authoring them by hand invited the two lists to drift
from `truth`; deriving them cannot.

**Pong is self-chosen.** Its rules are the model's own session-1 choice, so its
expected values come from session 1's answer rather than from the brief. That
is the one scenario with `self_chosen`, and the driver reads session 1's JSON
block to fill its truth. Its four prior-derivable rules are still annotated so
"the control scored this because Pong defaults are 800x600, first to 11" is a
number rather than an excuse.
"""

from __future__ import annotations

import argparse
import json
import re


def timeline(base: dict, changes: dict[int, dict]):
    """`truth(session)` for a base set plus changes from named sessions on."""

    def truth(session: int) -> dict:
        out = dict(base)
        for when in sorted(changes):
            if when <= session:
                out.update(changes[when])
        return out

    return truth


def scenario(
    name, domain, sessions, brief, work, events, base, changes, desc, prior=(), self_chosen=False
):
    changed = sorted({key for when in changes.values() for key in when})
    # Pong's rules are the model's own choice, so "what changes" cannot describe
    # its carryable set: the derivable-by-prior keys are the foundation and the
    # rest are the signal, whatever values the model picked.
    foundation = list(prior) if self_chosen else [key for key in base if key not in changed]
    carryable = [key for key in base if key not in prior] if self_chosen else changed
    return {
        "name": name,
        "domain": domain,
        "sessions": sessions,
        "brief": brief,
        "work": work,
        "events": events,
        "keys": list(base),
        "truth": timeline(base, changes),
        "desc": desc,
        "foundation": foundation,
        "carryable": carryable,
        "prior": list(prior),
        "self_chosen": self_chosen,
    }


PHOTOGRAPH_KEYS = {
    "marcus_eyes": "Marcus's eye colour",
    "ines_eyes": "Ines's eye colour",
    "halvorsen_eyes": "Dr Halvorsen's eye colour",
    "rosa_eyes": "Rosa's eye colour",
    "aldo_eyes": "Aldo's eye colour",
    "town": "the town's name",
    "weather_rule": "the weather rule",
    "ferry_day": "the day the ferry runs",
    "marcus_knows_photo": "true/false: does Marcus know what the photograph shows?",
    "tomas_status": '"missing" or "found"',
    "inn_status": '"standing" or "burned"',
    "halvorsen_confessed": "true/false: has Halvorsen confessed?",
    "ferry_running": "true/false: does the ferry still run?",
    "anyone_left_ashgrove": "true/false: has anyone left Ashgrove?",
}

SCENARIOS = {
    "photograph": scenario(
        "photograph",
        "fiction",
        10,
        brief="THE PHOTOGRAPH, a novel set in the town of Ashgrove.\n"
        "Characters: Marcus (grey eyes, the lighthouse keeper's son), Ines "
        "(green eyes, the town archivist), Dr Halvorsen (brown eyes, the "
        "physician), Rosa (hazel eyes, keeps the inn), Aldo (blue eyes, the "
        "mayor).\nRules: close third person, past tense; Marcus must not learn "
        "what the photograph shows before chapter 60; no character may leave "
        "Ashgrove before chapter 80.\nWeather rule: it never rains. The ferry "
        "runs only on Sundays.\nWrite chapters 1 to 10. Each chapter is two "
        "sentences, headed 'Chapter N'. Stay consistent with everything "
        "established so far.",
        work="Write chapters {first} to {last} of THE PHOTOGRAPH. Each chapter is "
        "two sentences, headed 'Chapter N'. Stay consistent with everything "
        "established so far.",
        events={
            2: "In chapter 12, Ines finds the photograph in the archive.",
            4: "In chapter 34, Rosa's inn burns to the ground.",
            6: "In chapter 58, Tomas is found alive, hiding in the lighthouse. "
            "Marcus learns what the photograph shows in chapter 60.",
            8: "In chapter 71, Dr Halvorsen confesses the forged certificate to Aldo.",
            9: "In chapter 90, the ferry stops running for good.",
        },
        base={
            "marcus_eyes": "grey",
            "ines_eyes": "green",
            "halvorsen_eyes": "brown",
            "rosa_eyes": "hazel",
            "aldo_eyes": "blue",
            "town": "ashgrove",
            "weather_rule": "never rains",
            "ferry_day": "sunday",
            "marcus_knows_photo": False,
            "tomas_status": "missing",
            "inn_status": "standing",
            "halvorsen_confessed": False,
            "ferry_running": True,
            "anyone_left_ashgrove": False,
        },
        changes={
            4: {"inn_status": "burned"},
            6: {"tomas_status": "found", "marcus_knows_photo": True},
            8: {"halvorsen_confessed": True},
            9: {"ferry_running": False},
        },
        desc=PHOTOGRAPH_KEYS,
    ),
    "pong": scenario(
        "pong",
        "code port",
        3,
        brief="Write a complete Pong game in Swift with two computer players that "
        "play each other automatically. No human input at all. You decide the "
        "field size, the winning score, how each paddle's AI tracks the ball, "
        "and how ball speed changes over a rally. First state those decisions "
        "as a JSON object, in a ```json block, with exactly these keys and "
        "numeric values: field_width, field_height, win_score, ball_start_speed, "
        "ball_speed_increment, ball_max_speed, paddle_speed. Then give the full "
        "code in one Swift file.",
        work="Port that Pong game to {language}. Keep exactly the same game rules and "
        "behaviour. First state the rules you are implementing as the same JSON "
        "object, then give the full code in one file.",
        events={},
        base={
            "field_width": None,
            "field_height": None,
            "win_score": None,
            "ball_start_speed": None,
            "ball_speed_increment": None,
            "ball_max_speed": None,
            "paddle_speed": None,
        },
        changes={},
        desc={
            "field_width": "the field width you chose",
            "field_height": "the field height you chose",
            "win_score": "the winning score you chose",
            "ball_start_speed": "the ball's starting speed",
            "ball_speed_increment": "the speed added per paddle hit",
            "ball_max_speed": "the ball's maximum speed",
            "paddle_speed": "the paddle speed",
        },
        prior=("field_width", "field_height", "win_score", "ball_speed_increment"),
        self_chosen=True,
    ),
    "ledger": scenario(
        "ledger",
        "code migration",
        8,
        brief="You are the engineer for Ledger API, a payments service.\nAgreed "
        "migration target: wire protocol gRPC; OLTP datastore Postgres; "
        "analytics read model ClickHouse; entity ID scheme ULID; service auth "
        "OIDC, with mTLS between internal services; rollout strangler.\nOver "
        "eight sessions, migrate one stage per session and keep the system "
        "working. Session 1: restate the target as `MIGRATION_PLAN`, then write "
        "the stage-1 steps.",
        work="Continue the Ledger API migration. Session {session}: write the "
        "stage-{session} steps, keeping the system working.",
        events={
            3: "The team has decided to defer ClickHouse: Postgres serves analytics until phase 5.",
            5: "Service auth moves from OIDC to mTLS for all callers.",
            6: "Entity IDs move from ULID to a Snowflake scheme.",
        },
        base={
            "protocol": "grpc",
            "oltp": "postgres",
            "analytics": "clickhouse",
            "id_scheme": "ulid",
            "auth": "oidc",
            "rollout": "strangler",
        },
        changes={3: {"analytics": "postgres"}, 5: {"auth": "mtls"}, 6: {"id_scheme": "snowflake"}},
        desc={
            "protocol": "the wire protocol",
            "oltp": "the OLTP datastore",
            "analytics": "the analytics datastore",
            "id_scheme": "the entity ID scheme",
            "auth": "the service auth mechanism",
            "rollout": "the rollout strategy",
        },
    ),
    "pigeon": scenario(
        "pigeon",
        "operations",
        6,
        brief="Write the runbook for `pigeon`, a three-service deployment in one "
        "region.\nService map: gateway — port 8443, datastore Postgres, owner "
        "Platform; worker — port 9100, datastore Redis, owner Data; scheduler — "
        "port 9200, datastore Postgres, owner Platform. Rollback window: 24 "
        "hours.\nOver six sessions, document deploy, rollback, backup/restore "
        "and on-call as each changes. Session 1: restate the service map, then "
        "write the deploy section.",
        work="Continue the pigeon runbook. Session {session}: write the next section.",
        events={
            2: "The gateway moved to port 9443 behind the new proxy.",
            3: "The worker's owner changed to Data-Platform.",
            4: "The rollback window is now 72 hours.",
            5: "The scheduler is decommissioned and will not be restored.",
        },
        base={
            "gateway_port": 8443,
            "worker_port": 9100,
            "worker_owner": "data",
            "rollback_hours": 24,
            "scheduler_state": "running",
        },
        changes={
            2: {"gateway_port": 9443},
            3: {"worker_owner": "data-platform"},
            4: {"rollback_hours": 72},
            5: {"scheduler_state": "decommissioned"},
        },
        desc={
            "gateway_port": "the gateway's port",
            "worker_port": "the worker's port",
            "worker_owner": "the worker's owning team",
            "rollback_hours": "the rollback window in hours",
            "scheduler_state": '"running" or "decommissioned"',
        },
    ),
    "contract": scenario(
        "contract",
        "legal",
        7,
        brief="Draft a master services agreement between Northwind Trading (client) "
        "and Calder Systems (supplier).\nAgreed commercial terms: termination "
        "notice 30 days; liability capped at 12 months' fees; governing law "
        "Singapore; sub-processors not permitted.\nOver seven sessions, "
        "negotiate and redraft one area per session. Session 1: restate the "
        "terms, then draft clause 1 (term).",
        work="Continue the Calder master services agreement. Session {session}: redraft "
        "the next area, keeping every other term as agreed.",
        events={
            2: "The amendment extends termination notice to 60 days.",
            3: "The liability cap now carves out gross negligence.",
            4: "Governing law changes to England and Wales.",
            5: "A sub-processor clause is added, permitting named sub-processors.",
        },
        base={
            "client": "northwind",
            "supplier": "calder",
            "notice_days": 30,
            "liability_cap": "flat",
            "governing_law": "singapore",
            "subprocessors": False,
        },
        changes={
            2: {"notice_days": 60},
            3: {"liability_cap": "carveout"},
            4: {"governing_law": "england"},
            5: {"subprocessors": True},
        },
        desc={
            "client": "the client's name",
            "supplier": "the supplier's name",
            "notice_days": "the termination notice in days",
            "liability_cap": '"flat" or "carveout"',
            "governing_law": '"singapore" or "england"',
            "subprocessors": "true/false: are sub-processors permitted?",
        },
    ),
    "compound_k": scenario(
        "compound_k",
        "laboratory",
        6,
        brief="Design a bench protocol to assay compound K in serum.\nStarting "
        "parameters: reagent Tris; concentration 0.5 M; incubation 30 minutes "
        "at 37 C; detection absorbance; safety limit 60 C.\nOver six sessions, "
        "validate the assay and revise the protocol. Session 1: restate the "
        "parameters, then write the materials section.",
        work="Continue the compound K protocol. Session {session}: write the next "
        "section and apply any revision you were given.",
        events={
            2: "Correction: the concentration is 0.25 M Tris, not 0.5 M.",
            3: "Incubation is extended to 45 minutes.",
            4: "Detection moves from absorbance to fluorescence.",
            5: "The safety limit is tightened to 50 C.",
        },
        base={
            "reagent": "tris",
            "concentration_m": 0.5,
            "incubation_min": 30,
            "detection": "absorbance",
            "safety_limit_c": 60,
        },
        changes={
            2: {"concentration_m": 0.25},
            3: {"incubation_min": 45},
            4: {"detection": "fluorescence"},
            5: {"safety_limit_c": 50},
        },
        desc={
            "reagent": "the reagent",
            "concentration_m": "the concentration in M",
            "incubation_min": "the incubation in minutes",
            "detection": '"absorbance" or "fluorescence"',
            "safety_limit_c": "the safety limit in C",
        },
    ),
    "vantage": scenario(
        "vantage",
        "game design",
        8,
        brief="You are the GM of a tabletop campaign set in the city of Vantage.\n"
        "Bible: the five factions are the Ember Court, the Tide Guild, the "
        "Lantern Watch, the Thorn Circle and the Hollow Choir. Named NPCs: Mira "
        "(harbourmaster), Vale (court sorcerer), Thorn (guildmaster), Sable "
        "(watch captain). The two hard rules: magic cannot create life; a death "
        "cannot be undone.\nOver eight sessions, write the campaign bible and "
        "play out the story. Session 1: restate the bible, then write the first "
        "district.",
        work="Continue the Vantage campaign. Session {session}: write the next "
        "district or scene, keeping the bible consistent.",
        events={
            3: "In session 3, the court sorcerer Vale dies in the harbour fire.",
            4: "The Ember Court and the Thorn Circle form an alliance.",
            5: "The Lantern district is destroyed.",
            6: "The death rule is clarified: a death cannot be undone except by the Ember rite.",
        },
        base={
            "binding_rule": "absolute",
            "npc_vale": "alive",
            "alliance": "none",
            "district_lantern": "standing",
        },
        changes={
            3: {"npc_vale": "dead"},
            4: {"alliance": "ember+thorn"},
            5: {"district_lantern": "destroyed"},
            6: {"binding_rule": "ember_rite_exception"},
        },
        desc={
            "binding_rule": 'how absolute the death rule is: "absolute" or "ember_rite_exception"',
            "npc_vale": '"alive" or "dead"',
            "alliance": "the current alliance",
            "district_lantern": '"standing" or "destroyed"',
        },
    ),
    "kitchen": scenario(
        "kitchen",
        "physical project",
        7,
        brief="Plan the renovation of a 1970s townhouse kitchen.\nStarting spec: room "
        "4200 x 3600 mm; galley layout; quartz countertop; gas range; budget "
        "42,000; permit pending.\nOver seven sessions, revise the spec as "
        "decisions are made. Session 1: restate the spec, then write the "
        "demolition plan.",
        work="Continue the kitchen renovation. Session {session}: update the spec and "
        "write the next section.",
        events={
            2: "Survey correction: the room is 4180 mm wide, not 4200.",
            3: "The countertop changes from quartz to soapstone.",
            4: "The range changes from gas to induction.",
            5: "The budget is revised to 46,000.",
            6: "The permit is approved.",
        },
        base={
            "width_mm": 4200,
            "layout": "galley",
            "countertop": "quartz",
            "range": "gas",
            "budget": 42000,
            "permit": "pending",
        },
        changes={
            2: {"width_mm": 4180},
            3: {"countertop": "soapstone"},
            4: {"range": "induction"},
            5: {"budget": 46000},
            6: {"permit": "approved"},
        },
        desc={
            "width_mm": "the room width in mm",
            "layout": "the cabinet layout",
            "countertop": "the countertop material",
            "range": "the range type",
            "budget": "the budget",
            "permit": "the permit status",
        },
    ),
    "cohort": scenario(
        "cohort",
        "research",
        8,
        brief="Design a cohort study of sleep and memory.\nStarting design: "
        "hypothesis — sleep duration predicts memory consolidation; cohort size "
        "240; primary endpoint word recall; analysis mixed model; stopping rule "
        "futility at the interim; inclusion adults 18-65.\nOver eight sessions, "
        "write the protocol and analysis plan. Session 1: restate the design, "
        "then write the recruitment section.",
        work="Continue the sleep-and-memory protocol. Session {session}: write the "
        "next section and apply any revision you were given.",
        events={
            2: "The power analysis raises the cohort size to 360.",
            3: "The primary endpoint changes to recognition.",
            4: "An exclusion is added: diagnosed apnoea.",
            5: "The analysis method is corrected to GEE.",
            6: "Recruitment is paused.",
        },
        base={
            "hypothesis": "sleep duration",
            "cohort_size": 240,
            "endpoint": "word recall",
            "analysis": "mixed model",
            "exclusion": "none",
            "recruitment": "running",
        },
        changes={
            2: {"cohort_size": 360},
            3: {"endpoint": "recognition"},
            4: {"exclusion": "apnoea"},
            5: {"analysis": "gee"},
            6: {"recruitment": "paused"},
        },
        desc={
            "hypothesis": "what the hypothesis predicts",
            "cohort_size": "the cohort size",
            "endpoint": "the primary endpoint",
            "analysis": "the analysis method",
            "exclusion": "the main exclusion",
            "recruitment": '"running" or "paused"',
        },
    ),
    "filing": scenario(
        "filing",
        "compliance",
        7,
        brief="Prepare the annual regulatory filing for Calder Systems under the new "
        "regime.\nStarting position: accounting standard IFRS; revenue "
        "recognised over time; deferred tax treatment; materiality threshold "
        "50,000; filing deadline 2027-03-31.\nOver seven sessions, assemble the "
        "filing position and draft one note per session. Session 1: restate the "
        "position, then draft the revenue note.",
        work="Continue the Calder annual filing. Session {session}: draft the next "
        "note and apply any revision you were given.",
        events={
            2: "A tax election is made: deferred tax is measured at fair value.",
            3: "The revenue method is corrected to point in time.",
            4: "The auditor revises the materiality threshold to 75,000.",
            5: "The filing deadline is extended to 2027-04-15.",
        },
        base={
            "standard": "ifrs",
            "revenue_method": "over time",
            "tax_treatment": "deferred",
            "materiality": 50000,
            "deadline": "2027-03-31",
        },
        changes={
            2: {"tax_treatment": "fair value"},
            3: {"revenue_method": "point in time"},
            4: {"materiality": 75000},
            5: {"deadline": "2027-04-15"},
        },
        desc={
            "standard": "the accounting standard",
            "revenue_method": "the revenue recognition method",
            "tax_treatment": "the deferred-tax treatment",
            "materiality": "the materiality threshold",
            "deadline": "the filing deadline",
        },
    ),
}

# Pong's stages are languages, not session numbers.
_PONG_LANGUAGES = {2: "Python", 3: "C99"}


def session_prompt(spec: dict, session: int, carried: str | None = None) -> str:
    parts = []
    if carried:
        parts.append("Notes from the previous session:\n" + carried + "\n")
    if session == 1:
        parts.append(spec["brief"])
    elif spec["name"] == "pong":
        parts.append(spec["work"].format(language=_PONG_LANGUAGES.get(session, "Python")))
    else:
        first, last = (session - 1) * 10 + 1, session * 10
        parts.append(spec["work"].format(session=session, first=first, last=last))
    if session in spec["events"]:
        parts.append(spec["events"][session])
    parts.append(quiz_prompt(spec))
    return "\n\n".join(parts)


def quiz_prompt(spec: dict) -> str:
    keys = ", ".join(spec["keys"])
    asks = "; ".join(f"{key} ({spec['desc'][key]})" for key in spec["keys"])
    # The quiz comes first so the instrument cannot be lost to a long or
    # meandering reply: a session that gets truncated, or that never writes the
    # block at all, used to score as a full set of memory misses. Answering from
    # retention before re-deriving anything is also the cleaner measurement.
    return (
        "Begin your reply with this continuity quiz as a JSON object in a "
        f"```json block with exactly these keys: {keys}. Meaning: {asks}. "
        "The JSON block must be the first thing in your reply, answered from "
        "what you actually remember or have been told. Then do the work above "
        "in at most 1,200 words."
    )


def extract_quiz(text: str, keys) -> dict:
    """The reply's quiz block, preferring one that carries the whole key set.

    The quiz is asked for first, so blocks are scanned in order: the first one
    holding every key wins, and a later block that merely shares a key is only
    a fallback. Reading backwards used to be right when the quiz was the last
    thing in the reply; it would now be the *work* that got mistaken for it.
    """
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates += re.findall(r"(\{[^{}]*\})", text, re.S)
    partial = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and set(parsed) & set(keys):
            if set(keys) <= set(parsed):
                return parsed
            partial = partial or parsed
    return partial or {}


def normalise(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower().strip(".,;:\"'")
    if text in ("true", "yes"):
        return True
    if text in ("false", "no"):
        return False
    return text


def hit(expected, answer) -> bool:
    """A right answer is the expected value, not a phrase containing it."""
    got = normalise(answer)
    if got is None:
        return False
    if isinstance(expected, bool):
        return got is expected
    if isinstance(expected, (int, float)):
        if isinstance(got, bool):
            return False
        numbers = re.findall(r"-?\d+(?:\.\d+)?", str(got))
        return any(abs(float(n) - float(expected)) < 1e-9 for n in numbers)
    want = str(expected).lower()
    if not want:
        return False
    return want in str(got)


def check() -> int:
    """Structural consistency, before a model run can hide an authoring bug."""
    problems = []
    for name, spec in SCENARIOS.items():
        if len(spec["keys"]) != len(spec["desc"]):
            problems.append(f"{name}: {len(spec['keys'])} keys, {len(spec['desc'])} descriptions")
        if spec["sessions"] < 2:
            problems.append(f"{name}: fewer than two sessions")
        if not spec["self_chosen"]:
            for session in range(1, spec["sessions"] + 1):
                truth = spec["truth"](session)
                if set(truth) != set(spec["keys"]):
                    problems.append(f"{name}: truth({session}) keys differ")
                for key, value in truth.items():
                    if not hit(value, value):
                        problems.append(
                            f"{name}: truth({session})[{key}]={value!r} does not match itself"
                        )
                prompt = session_prompt(spec, session)
                for key in spec["keys"]:
                    if key not in prompt:
                        problems.append(f"{name}: session {session} quiz omits {key}")
        # A scenario with no carryable key measures the model, not memory.
        if not spec["self_chosen"] and not spec["carryable"]:
            problems.append(f"{name}: no carryable key")
        if spec["self_chosen"] and not spec["prior"]:
            problems.append(f"{name}: self-chosen with no prior annotations")
    for problem in problems:
        print("FAIL", problem)
    print(
        f"{len(SCENARIOS)} scenarios, "
        f"{sum(s['sessions'] for s in SCENARIOS.values())} sessions, "
        f"{sum(len(s['carryable']) for s in SCENARIOS.values())} carryable keys"
    )
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.check:
        return check()
    if args.list:
        for name, spec in SCENARIOS.items():
            print(
                f"{name:12s} {spec['domain']:16s} sessions={spec['sessions']} "
                f"foundation={len(spec['foundation'])} carryable={len(spec['carryable'])}"
            )
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
