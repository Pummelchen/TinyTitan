/**
 * The gate that keeps dsh-lan-manager off harnesses it was not verified
 * against, and the pin it shares with the launcher and dsh-tinytitan.
 *
 * The gate is tested for the three things that make it safe to ship: it refuses,
 * it refuses *without throwing*, and it refuses without touching the context —
 * no config resolution, no route registration, no service access. The last one
 * is why the fake context's `get` is a landmine rather than a stub.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { apply } from "../src/index.js";
import { SUPPORTED_DSH_VERSION, supportDecision } from "../src/versions.js";

const LAUNCHER = new URL("../../../tools/dsh_local.sh", import.meta.url);

/** A version we positively read as *not* the supported one. */
const OTHER_VERSIONS = [
  "0.1.5-rc.2",
  "0.1.6-alpha.1",
  "0.1.6-alpha.3",
  "0.1.6-rc.1",
  "0.1.6",
  "0.2.0-rc.2",
  "0.0.0-development",
];

/** Run `apply` under `version` with a context that reports logging and defends services. */
function applyAs(version, { ctx = {}, config = {} } = {}) {
  const lines = [];
  const stderr = [];
  const realError = console.error;
  console.error = (message) => stderr.push(String(message));
  const context = {
    logger: {
      info: (message) => lines.push(String(message)),
      error: (message) => lines.push(String(message)),
    },
    get: () => {
      throw new Error("services must not be touched when refused");
    },
    ...ctx,
  };
  const previous = process.env.DSH_VERSION;
  if (version === undefined) delete process.env.DSH_VERSION;
  else process.env.DSH_VERSION = version;
  try {
    return { lines, result: apply(context, config), stderr };
  } finally {
    console.error = realError;
    if (previous === undefined) delete process.env.DSH_VERSION;
    else process.env.DSH_VERSION = previous;
  }
}

test("the launcher pins the same release the plugin supports", () => {
  const launcher = readFileSync(LAUNCHER, "utf8");
  const pinned = /^DSH_VERSION="\$\{TINYTITAN_DSH_VERSION:-([^}]+)\}"/m.exec(launcher);
  assert.ok(pinned, "tools/dsh_local.sh must pin DSH_VERSION to a default");
  assert.equal(pinned[1], SUPPORTED_DSH_VERSION);
});

test("the supported release runs", () => {
  assert.deepEqual(supportDecision(SUPPORTED_DSH_VERSION), { run: true, refusal: null });
});

test("another release is refused, by name, in one line", () => {
  for (const version of OTHER_VERSIONS) {
    const { run, refusal } = supportDecision(version);
    assert.equal(run, false, `${version} must be refused`);
    assert.match(refusal, new RegExp(version.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    assert.match(refusal, new RegExp(SUPPORTED_DSH_VERSION.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    assert.doesNotMatch(refusal, /\n/, "a refusal is one line");
  }
});

test("an unreadable version is refused too, and says so", () => {
  for (const unknown of [null, undefined, ""]) {
    const { run, refusal } = supportDecision(unknown);
    assert.equal(run, false, `${String(unknown)} must be refused, not assumed`);
    assert.match(refusal, /could not be read/);
  }
});

test("every refusal promises the rest of DSH keeps working", () => {
  for (const version of [...OTHER_VERSIONS, null]) {
    assert.match(supportDecision(version).refusal, /DSH itself is unaffected and keeps working/);
    assert.match(supportDecision(version).refusal, /remove this plugin/);
  }
});

test("an unsupported harness is refused without throwing or touching the context", () => {
  for (const version of [...OTHER_VERSIONS, undefined]) {
    const { lines, result } = applyAs(version);
    assert.equal(lines.length, 1, `${String(version)} must produce exactly one line`);
    assert.equal(result?.refused, true);
    assert.equal(result?.mounted, false);
  }
});

test("a refusal reaches stderr and the host error channel, never info", () => {
  // Verified against a real harness in a throwaway home, and against
  // `dsh-app-boot`: its startup exporter is registered with
  // `levels: { default: 2 }` and its `startupLogs` are printed only when the boot
  // itself fails. On a healthy boot nothing a plugin logs is printed, so stderr is
  // the only channel that carries the refusal.
  const infos = [];
  const errors = [];
  const stderr = [];
  const realError = console.error;
  console.error = (m) => stderr.push(String(m));
  const ctx = {
    logger: { info: (m) => infos.push(String(m)), error: (m) => errors.push(String(m)) },
    get: () => {
      throw new Error("services must not be touched when refused");
    },
  };
  const previous = process.env.DSH_VERSION;
  process.env.DSH_VERSION = "0.1.6";
  try {
    apply(ctx, {});
  } finally {
    console.error = realError;
    if (previous === undefined) delete process.env.DSH_VERSION;
    else process.env.DSH_VERSION = previous;
  }
  assert.equal(stderr.length, 1, "stderr is the channel a healthy boot does not swallow");
  assert.equal(errors.length, 1, "the host logger still gets the record");
  assert.equal(infos.length, 0, "a refusal must not go where the harness filters it out");
  assert.match(stderr[0], /not supported/);
});

test("the refusal is reported even with host logging turned off", () => {
  const { lines } = applyAs("0.1.6", { config: { logToHost: false } });
  assert.equal(lines.length, 1);
  assert.match(lines[0], /not supported/);
});

test("the supported harness reaches the normal path and is not refused", () => {
  // `get` answers "no webServer", which is the plugin's own quiet no-op path —
  // proof the gate let the call through rather than disabling the plugin.
  const { lines, result } = applyAs(SUPPORTED_DSH_VERSION, {
    ctx: { get: () => undefined },
    config: { logToHost: true },
  });
  assert.equal(result?.refused, undefined);
  assert.equal(result?.mounted, false);
  assert.match(lines.join("\n"), /no webServer in this profile/);
});
