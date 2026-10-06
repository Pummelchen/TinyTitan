/**
 * Keep the harness's route to TinyTitan current.
 *
 * The plugin deliberately owns no adapter: the harness's own `llm-pi-ai` route
 * serves these models, and `tools/dsh_route.sh` in this checkout is the one
 * place that turns the installed models into that route's block (ids, effort
 * ladders and the three switches that are easy to get wrong by hand). Running it
 * at boot is what makes the route follow `models/` instead of a copy someone
 * typed once.
 *
 * A plugin installed from a catalogue is a plain package beside no checkout, so
 * there is nothing to run. Only then — or when `selfContained: true` asks for it
 * explicitly — this module falls back to `generate.js`, a built-in generator
 * that mirrors the shell tool. The checkout stays the source of truth wherever
 * it exists, so the two cannot drift for checkout users.
 *
 * @module dsh-tinytitan/route
 */
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";

import { generateRoute } from "./generate.js";

/** The tool this delegates to. */
export function routeScript(repoRoot) {
  return join(repoRoot, "tools", "dsh_route.sh");
}

/**
 * Write the route block into the DSH settings file.
 * @param options - resolved config fields, plus injectable `run`/`log` for tests.
 * @returns `{status}` — `written`, `written-self-contained`, `missing`, `failed`
 *   or `skipped`.
 */
export function registerRoute({
  repoRoot,
  repoFound = true,
  port,
  provider,
  dshHome,
  selfContained = false,
  serverBinary,
  modelsDir,
  context,
  maxTokens,
  reasoning,
  env = process.env,
  run = execFileSync,
  log = () => {},
}) {
  const script = routeScript(repoRoot);
  if (!selfContained && existsSync(script)) {
    const args = [
      script,
      "--write",
      "--port",
      String(port),
      "--provider",
      provider,
      "--settings",
      join(dshHome, "settings.yaml"),
    ];
    // Not decorative: without this the script falls back to its own `medium`,
    // and a boot-time refresh silently turns thinking back on for a server that
    // was started with it off — after which the model reasons until its output
    // budget is gone and the page never shows an answer.
    if (reasoning) args.push("--reasoning", String(reasoning));
    // The same forwarding, and the same reason: `dsh_route.sh` falls back to the
    // launcher's pin, so a refresh that names neither writes 262144/32768 back
    // over a route narrowed with `--context`. `null` means unsaid, and the
    // script's default then stands, exactly as with `--reasoning` above.
    if (context !== null && context !== undefined) args.push("--context", String(context));
    if (maxTokens !== null && maxTokens !== undefined) {
      args.push("--max-tokens", String(maxTokens));
    }
    try {
      const stdout = run("bash", args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
      const first = String(stdout).trim().split("\n")[0] || "written";
      log(`dsh-tinytitan: route refreshed from ${script} (${first})`);
      return { status: "written", script, detail: first };
    } catch (error) {
      const detail = String(error?.stderr ?? error?.message ?? error)
        .trim()
        .split("\n")[0];
      log(`dsh-tinytitan: route refresh failed: ${detail}`);
      return { status: "failed", script, detail };
    }
  }

  // The fallback. `repoFound === false` is the catalogue case the generator
  // exists for; a repo that was found but carries no tool is reported the same
  // way, because neither can run the checkout tool.
  if (!selfContained) {
    const why =
      repoFound === false
        ? "no TinyTitan checkout found (set TINYTITAN_REPO or the repoRoot config)"
        : `${script} is absent`;
    log(`dsh-tinytitan: ${why}; using the built-in route generator`);
  } else {
    log("dsh-tinytitan: selfContained is set; using the built-in route generator");
  }
  try {
    return generateRoute({
      port,
      provider,
      context,
      maxTokens,
      reasoning,
      repoRoot,
      serverBinary,
      modelsDir,
      dshHome,
      env,
      run,
      log,
    });
  } catch (error) {
    const detail = String(error?.message ?? error)
      .trim()
      .split("\n")[0];
    log(`dsh-tinytitan: route refresh failed: ${detail}`);
    return { status: "failed", detail };
  }
}
