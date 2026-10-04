/**
 * The handoff driver: an unfinished objective moves to a fresh context before
 * the session's window fills.
 *
 * Three live findings shape this file. First, the regression it exists for — the
 * fork bomb: the first version keyed the hop count on the child session and
 * wrote it after `start()` resolved, so a child's opening turn raced the write,
 * every child read "hop 1", and a zero budget forked until the process aborted.
 * The count is now keyed on the goal and claimed before the await. Second, a
 * forked child inherits the parent's `goal/change` events but *not* its
 * activation (that is process-local), so a hop must disarm the parent and arm the
 * child, or the hop buys exactly one turn. Third, that child is a **one-shot**
 * run: its `result` settles at the end of its prompt turn, so releasing it there
 * kills the continuation that the armed goal is about to start. A child is
 * released at idle, and only once its goal stops actively continuing.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { resolveConfig } from "../src/config.js";
import {
  DEFAULT_HANDOFF_AT_TOKENS,
  DEFAULT_HANDOFF_HOPS,
  DEFAULT_HANDOFF_MAX_CHILDREN,
  HANDOFF_PROVIDER,
  handoffPrompt,
  installHandoff,
  startHandoff,
  usageTokens,
} from "../src/handoff.js";

/** A small event bus, so service mutations reach the driver's listeners. */
function makeBus() {
  const listeners = {};
  return {
    listeners,
    on: (name, handler) => {
      (listeners[name] ??= []).push(handler);
    },
    emit: (name, ...args) => {
      for (const handler of listeners[name] ?? []) handler(...args);
    },
  };
}

/** A context over a bus, with immediate dependency injection. */
function context(bus, services = {}, { inject = true } = {}) {
  const ctx = {
    get: (name) => services[name],
    on: bus.on,
    listeners: bus.listeners,
    emit: bus.emit,
  };
  if (inject) ctx.inject = (_names, run) => run(ctx);
  return ctx;
}

/** Let the driver's promise chain settle. */
const tick = () => new Promise((resolve) => setImmediate(resolve));

/** A goal view shaped like the harness's, activation included. */
function goalView(overrides = {}) {
  return {
    id: "goal-1",
    revision: 1,
    objective: "ship the parser",
    phase: "active",
    activation: "armed",
    roundsStarted: 0,
    maxGoalRounds: 3,
    ...overrides,
  };
}

/**
 * A harness whose agents and goals behave like the measured ones: a forked
 * child inherits the parent's goal **disarmed**, its run settles on demand, and
 * service mutations announce themselves on the bus.
 */
function harness({ goal = goalView(), child = "local", gate = null, failFirst = false } = {}) {
  const bus = makeBus();
  const starts = [];
  const runs = [];
  const calls = { disarm: [], resume: [] };
  const sessions = new Set(["session-1"]);
  const agents = new Map([["session-1", { id: "session-1", status: "running" }]]);
  const goalStates = new Map([["session-1", { ...goal }]]);
  let created = 0;

  const announce = (id) => {
    bus.emit("goal/activation-changed", { sessionId: id });
  };
  const addSession = (id, state = goalView({ id: `goal-${id}`, objective: `objective ${id}` })) => {
    sessions.add(id);
    agents.set(id, { id, status: "running" });
    goalStates.set(id, { ...state });
    return agents.get(id);
  };
  const setStatus = (id, status) => {
    const agent = agents.get(id);
    if (agent === undefined) return undefined;
    agent.status = status;
    bus.emit("agent/status", { agent, status });
    return agent;
  };

  const services = {
    agents: {
      get: (id) => agents.get(id),
      withInitiator: (_agent, operation) => operation(),
    },
    goals: {
      get: (agent) => {
        const state = goalStates.get(agent?.id);
        return state === undefined ? undefined : { ...state };
      },
      disarm: (agent) => {
        calls.disarm.push(agent.id);
        const state = goalStates.get(agent.id);
        if (state === undefined) return undefined;
        state.activation = "disarmed";
        announce(agent.id);
        return { ...state };
      },
      resume: (agent, ref) => {
        calls.resume.push({ session: agent.id, ref });
        const state = goalStates.get(agent.id);
        if (state === undefined || state.id !== ref.id) throw new Error("no such goal");
        if (state.activation === "armed") throw new Error("already active and armed");
        state.activation = "armed";
        announce(agent.id);
        return { ...state };
      },
    },
    subagents: {
      start: (name, request) => {
        starts.push({ name, request });
        if (failFirst && starts.length === 1) return Promise.reject(new Error("loop inactive"));
        const id = `child-${++created}`;
        const run = { id, localAgent: undefined, result: undefined, disposed: false };
        if (child !== "remote") {
          const childAgent = { id, session: { id }, status: "running" };
          agents.set(id, childAgent);
          sessions.add(id);
          if (child !== "no-goal") {
            goalStates.set(id, { ...goal, activation: "disarmed", roundsStarted: 0 });
          }
          run.localAgent = childAgent;
        }
        run.result = new Promise((resolve) => {
          run.settle = resolve;
        });
        run.dispose = () => {
          run.disposed = true;
        };
        runs.push(run);
        return gate ? gate.then(() => run) : Promise.resolve(run);
      },
    },
  };
  return {
    bus,
    services,
    starts,
    runs,
    calls,
    sessions,
    agents,
    goalStates,
    addSession,
    setStatus,
  };
}

/** Fire listeners and let the driver's promise chain settle. */
async function fire(ctx, name, ...args) {
  ctx.emit(name, ...args);
  await tick();
}

/** Report a step's cost, then open the next turn — where the budget is judged. */
async function step(ctx, session, usage) {
  await fire(ctx, "session/event", { id: session }, { type: "assistant/message", data: { usage } });
  await fire(ctx, "session/event", { id: session }, { type: "turn/start" });
}

/** A harness with the driver installed over one armed goal. */
function installed(options = {}, resolved = {}) {
  const h = harness(options);
  const ctx = context(h.bus, h.services);
  const lines = [];
  installHandoff({
    ctx,
    resolved: { handoff: true, handoffHops: 2, handoffAtTokens: 1000, ...resolved },
    log: (line) => lines.push(line),
  });
  return { ...h, ctx, lines };
}

test("handoffPrompt leads with the objective and names the authority rule", () => {
  const prompt = handoffPrompt("fix the parser");
  assert.equal(
    prompt.split("\n")[0],
    "fix the parser",
    "the child's goal reads the objective first",
  );
  assert.match(prompt, /workspace is the authority/);
  assert.match(prompt, /solved, tested and verified/);
});

test("the budget counts cached input, not just the uncached remainder", () => {
  // Measured live: inputTokens 137 with cacheReadTokens 5454 in the same step.
  assert.equal(usageTokens({ inputTokens: 137, cacheReadTokens: 5454 }), 5591);
  assert.equal(usageTokens({ inputTokens: 1, cacheReadTokens: 2, cacheWriteTokens: 3 }), 6);
  assert.equal(usageTokens({ totalTokens: 5605 }), 5605, "a total-only adapter still counts");
  assert.equal(usageTokens({}), 0);
  assert.equal(usageTokens(undefined), 0);
});

test("the driver is off unless the profile asks for it", () => {
  const ctx = context(makeBus(), {});
  const status = installHandoff({ ctx, resolved: { handoff: false }, log: () => {} });
  assert.deepEqual(status, { installed: false, reason: "disabled" });
  assert.equal(ctx.listeners["session/event"], undefined);
});

test("a harness without the services logs why and does nothing", () => {
  const lines = [];
  const ctx = context(makeBus(), {}, { inject: false });
  const status = installHandoff({
    ctx,
    resolved: { handoff: true, handoffHops: 3 },
    log: (line) => lines.push(line),
  });
  assert.equal(status.installed, false);
  assert.equal(status.reason, "no-services");
  assert.ok(lines.some((line) => line.includes("no agents/goals/subagents")));
});

test("a turn is only handed on once the previous step spent the budget", async () => {
  const { ctx, starts, lines } = installed({}, { handoffAtTokens: 1000 });
  await step(ctx, "session-1", { inputTokens: 999 });
  assert.equal(starts.length, 0, "below the budget nothing moves");
  await step(ctx, "session-1", { inputTokens: 1000 });
  assert.equal(starts.length, 1, "at the budget the work moves");
  assert.equal(starts[0].name, HANDOFF_PROVIDER);
  assert.equal(starts[0].request.parent.id, "session-1");
  assert.match(starts[0].request.prompt[0].text, /ship the parser/);
  assert.ok(starts[0].request.signal, "start requires a cancellation signal");
  assert.ok(lines.some((line) => line.includes("hop 1/2 at 1000 tokens")));
});

test("only a spent step, and only an armed unfinished goal, can trigger", async () => {
  for (const [label, goal, usage] of [
    ["a completed goal", goalView({ phase: "complete" }), { inputTokens: 9999 }],
    ["no goal", null, { inputTokens: 9999 }],
    ["a disarmed goal", goalView({ activation: "disarmed" }), { inputTokens: 9999 }],
    ["a paused goal", goalView({ phase: "paused" }), { inputTokens: 9999 }],
    ["nothing spent yet", goalView(), { inputTokens: 0 }],
  ]) {
    const { ctx, starts } = installed({ goal }, { handoffAtTokens: 1000 });
    await step(ctx, "session-1", usage);
    assert.equal(starts.length, 0, label);
  }
});

test("one hop moves the goal: the parent stops, the child's seed is armed", async () => {
  const { ctx, starts, runs, calls, lines, goalStates } = installed({}, { handoffAtTokens: 1000 });
  await step(ctx, "session-1", { inputTokens: 1000 });
  assert.equal(starts.length, 1);
  assert.deepEqual(calls.disarm, ["session-1"], "the handing-off session is disarmed");
  assert.equal(goalStates.get("session-1").activation, "disarmed");
  assert.equal(calls.resume.length, 1, "the child's inherited goal is armed");
  assert.equal(calls.resume[0].session, "child-1");
  assert.deepEqual(calls.resume[0].ref, { id: "goal-1", revision: 1 });
  assert.equal(goalStates.get("child-1").activation, "armed");
  assert.ok(lines.some((line) => line.includes('armed "ship the parser" in the child session')));
  assert.equal(runs[0].disposed, false, "a working child is not disposed");
});

test("a later turn in the parent cannot spend a second hop", async () => {
  const { ctx, starts } = installed({}, { handoffAtTokens: 1000, handoffHops: 5 });
  await step(ctx, "session-1", { inputTokens: 5000 });
  assert.equal(starts.length, 1);
  await step(ctx, "session-1", { inputTokens: 9000 });
  assert.equal(starts.length, 1, "the parent is disarmed, so it is no longer a handoff source");
});

test("a settled run is not released while its goal is still being continued", async () => {
  const { ctx, runs, setStatus, lines } = installed({}, { handoffAtTokens: 1000 });
  await step(ctx, "session-1", { inputTokens: 1000 });
  runs[0].settle({ stopReason: "completed" });
  await tick();
  await tick();
  assert.equal(runs[0].disposed, false, "the one-shot run settling is not the end of the work");
  // The round driver queues the next round: the child is running again.
  setStatus("child-1", "idle");
  await tick();
  assert.equal(runs[0].disposed, false, "an armed goal means a round is coming");
  setStatus("child-1", "running");
  await tick();
  // The objective completes: activation goes disarmed, the child goes idle.
  ctx.get("goals").disarm(ctx.get("agents").get("child-1"));
  setStatus("child-1", "idle");
  await tick();
  assert.equal(runs[0].disposed, true, "idle with nothing to continue is the release point");
  assert.ok(lines.some((line) => line.includes("released child child-1")));
});

test("a child that handed the objective on finishes its turn first", async () => {
  const { ctx, runs, setStatus } = installed({}, { handoffAtTokens: 1000 });
  await step(ctx, "session-1", { inputTokens: 1000 });
  // The handing-off child was disarmed mid-turn by the next hop.
  ctx.get("goals").disarm(ctx.get("agents").get("child-1"));
  await tick();
  assert.equal(runs[0].disposed, false, "a running child is never torn down mid-turn");
  setStatus("child-1", "idle");
  await tick();
  assert.equal(runs[0].disposed, true);
});

test("an ancestor is not released while its own handoff child is still working", async () => {
  // Measured live: the grandchild's turn ended `{kind: "aborted", reason:
  // {kind: "disposed"}}` 3 ms after its parent was released. A disposed parent
  // disposes its subagent children, so the chain releases leaf-first.
  const { ctx, runs, calls, agents, setStatus } = installed(
    {},
    { handoffAtTokens: 1000, handoffHops: 2 },
  );
  await step(ctx, "session-1", { inputTokens: 1000 });
  await step(ctx, "child-1", { inputTokens: 2000 });
  assert.equal(runs.length, 2, "child-1 spent the second hop");
  assert.equal(calls.resume.filter((each) => each.session === "child-2").length, 1);
  setStatus("child-1", "idle");
  await tick();
  assert.equal(runs[0].disposed, false, "its own child is still working");
  ctx.get("goals").disarm(agents.get("child-2"));
  setStatus("child-2", "idle");
  await tick();
  assert.equal(runs[1].disposed, true, "the leaf goes first");
  assert.equal(runs[0].disposed, true, "then the hop above it");
});

test("the chain is capped by the goal, not by any one session", async () => {
  // The fork-bomb regression: three sessions, one goal. With the count keyed per
  // session every one of them would read "hop 1" and fork forever.
  const { sessions, ctx, starts, lines } = installed({}, { handoffHops: 2 });
  await step(ctx, "session-1", { inputTokens: 5000 });
  assert.equal(starts.length, 1);
  sessions.add("child-1");
  await step(ctx, "child-1", { inputTokens: 5000 });
  assert.equal(starts.length, 2, "the child's own turn spends the second hop");
  sessions.add("child-2");
  await step(ctx, "child-2", { inputTokens: 5000 });
  assert.equal(starts.length, 2, "the third session finds the budget spent");
  assert.ok(lines.some((line) => line.includes("reached 2 hops")));
});

test("one goal can never have two handoffs in flight", async () => {
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const { sessions, ctx, starts } = installed({ gate });
  sessions.add("child-1");
  const first = step(ctx, "session-1", { inputTokens: 5000 });
  const second = step(ctx, "child-1", { inputTokens: 5000 });
  await Promise.all([first, second]);
  assert.equal(starts.length, 1, "the second trigger waits for the first to settle");
  release();
  await tick();
  await tick();
});

test("a refused start gives the hop back", async () => {
  const { ctx, starts, lines } = installed({ failFirst: true }, { handoffAtTokens: 1000 });
  await step(ctx, "session-1", { inputTokens: 5000 });
  assert.equal(starts.length, 1);
  assert.ok(lines.some((line) => line.includes("handoff failed: loop inactive")));
  await step(ctx, "session-1", { inputTokens: 5000 });
  assert.equal(starts.length, 2, "the failed attempt did not consume the objective's budget");
  assert.ok(lines.some((line) => line.includes("hop 1/2")));
});

test("a full child cap skips the next hop loudly", async () => {
  const { ctx, starts, addSession, lines } = installed(
    {},
    { handoffAtTokens: 1000, handoffHops: 9, handoffMaxChildren: 1 },
  );
  addSession("session-2");
  await step(ctx, "session-1", { inputTokens: 5000 });
  assert.equal(starts.length, 1);
  await step(ctx, "session-2", { inputTokens: 5000 });
  assert.equal(starts.length, 1, "the process cap refuses the second child");
  assert.ok(lines.some((line) => line.includes("handoff children are already live (cap 1)")));
});

test("a remote run is released when it settles and is never guessed at", async () => {
  const { ctx, runs, calls, lines } = installed({ child: "remote" }, { handoffAtTokens: 1000 });
  await step(ctx, "session-1", { inputTokens: 5000 });
  assert.equal(runs.length, 1);
  assert.ok(lines.some((line) => line.includes("(remote run: no in-process child to arm)")));
  assert.deepEqual(calls.resume, [], "nothing to arm without a published child");
  runs[0].settle({ stopReason: "completed" });
  await tick();
  await tick();
  assert.equal(runs[0].disposed, true, "a remote run has no local agent to watch");
});

test("a child that inherited no goal says so", async () => {
  const { ctx, calls, lines } = installed({ child: "no-goal" }, { handoffAtTokens: 1000 });
  await step(ctx, "session-1", { inputTokens: 5000 });
  assert.deepEqual(calls.resume, []);
  assert.ok(lines.some((line) => line.includes("inherited no goal")));
});

test("startHandoff reports a missing service instead of throwing", async () => {
  const result = await startHandoff({
    ctx: context(makeBus(), {}),
    objective: "x",
    parent: { id: "a" },
  });
  assert.deepEqual(result, { started: false, reason: "no-agents-service" });
});

test("disposal releases a child that is still working", async () => {
  const { ctx, runs } = installed({}, { handoffAtTokens: 1000 });
  await step(ctx, "session-1", { inputTokens: 1000 });
  assert.equal(runs[0].disposed, false);
  for (const handler of ctx.listeners.dispose ?? []) handler();
  await tick();
  assert.equal(runs[0].disposed, true);
});

test("resolveConfig carries the handoff switches and refuses a zero budget", () => {
  const base = resolveConfig({ log: () => {} });
  assert.equal(base.handoff, false);
  assert.equal(base.handoffHops, DEFAULT_HANDOFF_HOPS);
  assert.equal(base.handoffAtTokens, DEFAULT_HANDOFF_AT_TOKENS);
  assert.equal(base.handoffMaxChildren, DEFAULT_HANDOFF_MAX_CHILDREN);
  const on = resolveConfig({
    handoff: true,
    handoffHops: 9,
    handoffAtTokens: 50000,
    handoffMaxChildren: 2,
    log: () => {},
  });
  assert.equal(on.handoff, true);
  assert.equal(on.handoffHops, 9);
  assert.equal(on.handoffAtTokens, 50000);
  assert.equal(on.handoffMaxChildren, 2);
  assert.equal(resolveConfig({ handoff: "true", log: () => {} }).handoff, false);
  assert.throws(() => resolveConfig({ handoffHops: 0, log: () => {} }), /positive integer/);
  assert.throws(() => resolveConfig({ handoffAtTokens: 0, log: () => {} }), /positive integer/);
  assert.throws(() => resolveConfig({ handoffAtTokens: -1, log: () => {} }), /positive integer/);
  assert.throws(() => resolveConfig({ handoffMaxChildren: 0, log: () => {} }), /positive integer/);
});
