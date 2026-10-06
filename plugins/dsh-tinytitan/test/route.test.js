import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, mkdirSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { findRepoRoot, REPO_ROOT, resolveConfig } from "../src/config.js";
import { registerRoute } from "../src/route.js";

function repo(withScript) {
  const root = mkdtempSync(join(tmpdir(), "dsh-tinytitan-route-"));
  mkdirSync(join(root, "tools"), { recursive: true });
  if (withScript) writeFileSync(join(root, "tools", "dsh_route.sh"), "#!/usr/bin/env bash\n");
  return root;
}

test("the route is refreshed by running the checkout's tool", () => {
  const calls = [];
  const messages = [];
  const result = registerRoute({
    repoRoot: repo(true),
    port: 8096,
    provider: "tinytitan",
    dshHome: "/tmp/dsh-home",
    run: (command, args, options) => {
      calls.push({ command, args, options });
      return "replaced\n";
    },
    log: (message) => messages.push(message),
  });
  assert.equal(result.status, "written");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, "bash");
  assert.deepEqual(calls[0].args.slice(1), [
    "--write",
    "--port",
    "8096",
    "--provider",
    "tinytitan",
    "--settings",
    "/tmp/dsh-home/settings.yaml",
  ]);
  assert.ok(calls[0].args[0].endsWith("tools/dsh_route.sh"));
  assert.ok(messages.some((message) => message.includes("route refreshed")));
});

test("the route refresh carries the reasoning level it was given", () => {
  // Regression: the level was resolved but never passed to the script, so the
  // script's own `medium` default won and a boot-time refresh turned thinking
  // back on for a server started with it off. On a dense Qwen that is the model
  // reasoning until its budget runs out and never answering.
  const calls = [];
  registerRoute({
    repoRoot: repo(true),
    port: 8096,
    provider: "tinytitan",
    reasoning: "off",
    dshHome: "/tmp/dsh-home",
    run: (command, args) => {
      calls.push(args);
      return "replaced\n";
    },
    log: () => {},
  });
  assert.deepEqual(calls[0].slice(1), [
    "--write",
    "--port",
    "8096",
    "--provider",
    "tinytitan",
    "--settings",
    "/tmp/dsh-home/settings.yaml",
    "--reasoning",
    "off",
  ]);
});

test("no reasoning level is passed when the caller names none", () => {
  // The script keeps its own default then, which is the historical behaviour.
  const calls = [];
  registerRoute({
    repoRoot: repo(true),
    port: 8080,
    provider: "tinytitan",
    dshHome: "/tmp/dsh-home",
    run: (command, args) => {
      calls.push(args);
      return "written\n";
    },
    log: () => {},
  });
  assert.equal(calls[0].includes("--reasoning"), false);
});

test("the route refresh carries the window and cap it was given", () => {
  // Same forwarding as the reasoning level above: `dsh_route.sh` has its own
  // defaults, so a refresh that names neither puts the launcher's 262144/32768
  // back over a route narrowed to fit the compaction arithmetic.
  const calls = [];
  registerRoute({
    repoRoot: repo(true),
    port: 8096,
    provider: "tinytitan",
    context: 131072,
    maxTokens: 8192,
    dshHome: "/tmp/dsh-home",
    run: (command, args) => {
      calls.push(args);
      return "replaced\n";
    },
    log: () => {},
  });
  assert.deepEqual(calls[0].slice(1), [
    "--write",
    "--port",
    "8096",
    "--provider",
    "tinytitan",
    "--settings",
    "/tmp/dsh-home/settings.yaml",
    "--context",
    "131072",
    "--max-tokens",
    "8192",
  ]);
});

test("no window or cap is passed when the caller names neither", () => {
  // The script keeps its own pin then, which is the historical block byte-for-byte.
  const calls = [];
  registerRoute({
    repoRoot: repo(true),
    port: 8080,
    provider: "tinytitan",
    dshHome: "/tmp/dsh-home",
    run: (command, args) => {
      calls.push(args);
      return "written\n";
    },
    log: () => {},
  });
  assert.equal(calls[0].includes("--context"), false);
  assert.equal(calls[0].includes("--max-tokens"), false);
});

test("what resolveConfig emits is what the tool is handed", () => {
  // The reported defect was exactly this seam: `context` and `maxTokens` were
  // read by both writers, but `resolveConfig` emitted neither, so the boot-time
  // and watcher refreshes — which spread the resolved config — always wrote the
  // launcher's pin. Asserting each side separately would not have caught it.
  const calls = [];
  const root = repo(true);
  registerRoute({
    ...resolveConfig({
      repoRoot: root,
      dshHome: "/tmp/dsh-home",
      context: 131072,
      maxTokens: 8192,
    }),
    run: (command, args) => {
      calls.push(args);
      return "replaced\n";
    },
    log: () => {},
  });
  const args = calls[0];
  assert.equal(args[args.indexOf("--context") + 1], "131072");
  assert.equal(args[args.indexOf("--max-tokens") + 1], "8192");
});

test("a checkout without the tool is reported, not fatal", () => {
  const messages = [];
  const result = registerRoute({
    repoRoot: repo(false),
    port: 8080,
    provider: "tinytitan",
    dshHome: "/tmp/dsh-home",
    log: (m) => messages.push(m),
  });
  assert.equal(result.status, "missing");
  assert.ok(messages.some((message) => message.includes("leaving the llm-pi-ai route as it is")));
});

test("a failed refresh is reported with the tool's own words", () => {
  const messages = [];
  const failure = new Error("exit 2");
  failure.stderr = "dsh_route: no DSH settings file at /tmp/dsh-home/settings.yaml\n";
  const result = registerRoute({
    repoRoot: repo(true),
    port: 8080,
    provider: "tinytitan",
    dshHome: "/tmp/dsh-home",
    run: () => {
      throw failure;
    },
    log: (m) => messages.push(m),
  });
  assert.equal(result.status, "failed");
  assert.equal(result.detail, "dsh_route: no DSH settings file at /tmp/dsh-home/settings.yaml");
  assert.ok(messages.some((message) => message.includes("route refresh failed")));
});

test("config defaults suit a local server and can be overridden", () => {
  const defaults = resolveConfig();
  assert.equal(defaults.port, 8080);
  assert.equal(defaults.provider, "tinytitan");
  assert.equal(defaults.presetId, "tinytitan");
  assert.equal(defaults.registerRoute, true);
  assert.equal(defaults.writeCompactionPreset, true);
  assert.equal(defaults.compactionHeadroomTokens, null, "the harness's headroom stands by default");
  assert.equal(defaults.context, null, "the route writer's own window pin stands by default");
  assert.equal(defaults.maxTokens, null, "the route writer's own cap pin stands by default");
  assert.ok(defaults.dshHome.endsWith(".dsh"));

  const configured = resolveConfig({
    port: 8096,
    provider: "local",
    presetId: "tinytitan-thin",
    registerRoute: false,
    writeCompactionPreset: false,
    setDefaultWhenUnset: false,
    compactionHeadroomTokens: 0,
    context: 131072,
    maxTokens: 8192,
    repoRoot: "/repo",
    dshHome: "/home",
  });
  assert.equal(configured.port, 8096);
  assert.equal(configured.provider, "local");
  assert.equal(configured.presetId, "tinytitan-thin");
  assert.equal(configured.registerRoute, false);
  assert.equal(configured.writeCompactionPreset, false);
  assert.equal(configured.setDefaultWhenUnset, false);
  assert.equal(configured.compactionHeadroomTokens, 0);
  assert.equal(configured.context, 131072);
  assert.equal(configured.maxTokens, 8192);
  // An explicit root is a hint: it is used when it holds the tool, and a stale
  // one falls through to the checkout this test suite lives in.
  assert.equal(configured.repoRoot, REPO_ROOT);
  assert.equal(configured.repoFound, true);
  assert.equal(configured.dshHome, "/home");
});

test("the environment picks the port when the config does not", () => {
  const saved = process.env.TINYTITAN_PORT;
  process.env.TINYTITAN_PORT = "9123";
  try {
    assert.equal(resolveConfig().port, 9123);
    // A configured port is an explicit choice and still wins.
    assert.equal(resolveConfig({ port: 8096 }).port, 8096);
  } finally {
    if (saved === undefined) delete process.env.TINYTITAN_PORT;
    else process.env.TINYTITAN_PORT = saved;
  }
});

test("the declared window and cap come from the config, then the environment", () => {
  // A route is rewritten on every refresh, so a window narrowed once has to be
  // something the plugin can be told from outside the profile — otherwise the
  // next boot writes the launcher's 262144 back over it.
  const savedContext = process.env.TINYTITAN_CONTEXT;
  const savedMax = process.env.TINYTITAN_MAX_TOKENS;
  delete process.env.TINYTITAN_CONTEXT;
  delete process.env.TINYTITAN_MAX_TOKENS;
  try {
    assert.equal(resolveConfig().context, null);
    assert.equal(resolveConfig().maxTokens, null);
    process.env.TINYTITAN_CONTEXT = "65536";
    process.env.TINYTITAN_MAX_TOKENS = "16384";
    assert.equal(resolveConfig().context, 65536);
    assert.equal(resolveConfig().maxTokens, 16384);
    // Configured values are explicit choices and still win.
    assert.equal(resolveConfig({ context: 32768, maxTokens: 4096 }).context, 32768);
    assert.equal(resolveConfig({ context: 32768, maxTokens: 4096 }).maxTokens, 4096);
  } finally {
    if (savedContext === undefined) delete process.env.TINYTITAN_CONTEXT;
    else process.env.TINYTITAN_CONTEXT = savedContext;
    if (savedMax === undefined) delete process.env.TINYTITAN_MAX_TOKENS;
    else process.env.TINYTITAN_MAX_TOKENS = savedMax;
  }
});

test("config refuses what it cannot use", () => {
  assert.throws(() => resolveConfig({ port: 0 }), /port must be a port number/);
  assert.throws(() => resolveConfig({ port: "http" }), /port must be a port number/);
  assert.throws(() => resolveConfig({ provider: "  " }), /provider must not be empty/);
  assert.throws(() => resolveConfig({ compactionHeadroomTokens: -1 }), /whole number of tokens/);
  assert.throws(() => resolveConfig({ compactionHeadroomTokens: 1.5 }), /whole number of tokens/);
  assert.throws(() => resolveConfig({ presetId: "" }), /presetId must not be empty/);
  assert.throws(() => resolveConfig({ reasoning: "" }), /reasoning must not be empty/);
  assert.throws(() => resolveConfig({ context: 0 }), /context must be a positive token count/);
  assert.throws(() => resolveConfig({ context: "wide" }), /context must be a positive token count/);
  assert.throws(() => resolveConfig({ maxTokens: -1 }), /maxTokens must be a positive token count/);
  assert.throws(
    () => resolveConfig({ maxTokens: 1.5 }),
    /maxTokens must be a positive token count/,
  );
});

test("the reasoning level comes from the config, then the environment, then the default", () => {
  // The route's declared level is what the harness asks for on every call that
  // names none, so it has to be settable from outside without editing a profile.
  const saved = process.env.TINYTITAN_REASONING;
  delete process.env.TINYTITAN_REASONING;
  try {
    assert.equal(resolveConfig().reasoning, "medium");
    process.env.TINYTITAN_REASONING = "off";
    assert.equal(resolveConfig().reasoning, "off");
    // A configured level is an explicit choice and still wins.
    assert.equal(resolveConfig({ reasoning: "xhigh" }).reasoning, "xhigh");
  } finally {
    if (saved === undefined) delete process.env.TINYTITAN_REASONING;
    else process.env.TINYTITAN_REASONING = saved;
  }
});

test("the checkout is found from a hint, the environment, the module, or the profile", () => {
  const base = mkdtempSync(join(tmpdir(), "dsh-tinytitan-find-"));
  const checkout = join(base, "checkout");
  mkdirSync(join(checkout, "tools"), { recursive: true });
  writeFileSync(join(checkout, "tools", "dsh_route.sh"), "#!/usr/bin/env bash\n");

  // An explicit root that holds the tool.
  assert.deepEqual(
    findRepoRoot({ explicit: checkout, env: {}, moduleUrl: "file:///nowhere/x.js", cwd: base }),
    { root: checkout, found: true },
  );
  // The environment variable.
  assert.deepEqual(
    findRepoRoot({
      env: { TINYTITAN_REPO: checkout },
      moduleUrl: "file:///nowhere/x.js",
      cwd: base,
    }),
    { root: checkout, found: true },
  );
  // Walking up from the module (running out of the checkout).
  const moduleUrl = pathToFileURL(
    join(checkout, "plugins", "dsh-tinytitan", "src", "config.js"),
  ).href;
  assert.deepEqual(findRepoRoot({ env: {}, moduleUrl, cwd: base }), {
    root: checkout,
    found: true,
  });

  // A profile that recorded where it installed the plugin from.
  const profile = join(base, "profile");
  const installed = join(profile, "node_modules", "dsh-tinytitan", "src");
  mkdirSync(installed, { recursive: true });
  writeFileSync(
    join(profile, "package.json"),
    JSON.stringify({
      dependencies: { "dsh-tinytitan": `file:${join(checkout, "plugins", "dsh-tinytitan")}` },
    }),
  );
  assert.deepEqual(
    findRepoRoot({
      env: {},
      moduleUrl: pathToFileURL(join(installed, "config.js")).href,
      cwd: base,
    }),
    { root: checkout, found: true },
  );

  // Nothing anywhere: the first candidate is returned, and the caller says so.
  const empty = mkdtempSync(join(tmpdir(), "dsh-tinytitan-empty-"));
  const missing = findRepoRoot({
    env: {},
    moduleUrl: pathToFileURL(join(empty, "src", "config.js")).href,
    cwd: empty,
  });
  assert.equal(missing.found, false);
});
