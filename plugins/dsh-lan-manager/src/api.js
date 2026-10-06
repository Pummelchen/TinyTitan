/**
 * Harness operations behind the LAN API.
 *
 * Every capability is expressed against a host service rather than by reaching
 * into dsh's internals:
 *
 * | Capability | Service | Method |
 * |---|---|---|
 * | active workspaces | `workspaceRegistry` | `list()` |
 * | visible sessions | `workspaceRegistry` | `Workspace.sessionIds` minus the archive set |
 * | prompt one session | `agents` | `get(id)` → `followup(createUserMessage(...))` |
 * | prompt all sessions | `agents` | the same, fanned out |
 * | archive a session | `workspaceRegistry` | `archiveSession(id)` |
 * | delete a workspace | `workspaceRegistry` | `delete(id)` |
 *
 * Services are read lazily through `ctx.get(...)` on each call, never captured at
 * load time: the plugin may be composed before the registry has started, and a
 * captured `undefined` would look like a permanent capability loss.
 *
 * **"Active" means what the web page shows.** A workspace is active when the
 * registry lists it *and* it has at least one non-archived session; a session is
 * visible when the workspace accounts for it and it is not in the registry-global
 * `archivedSessionIds` set. Archiving keeps the `sessionIds` slot — it hides the
 * row without deleting history — which is exactly the semantics the UI renders.
 *
 * @module dsh-lan-manager/api
 */
import { readFileSync, readdirSync } from "node:fs";

/** Reasons an operation can fail, mapped to HTTP status by the router. */
export const Failure = Object.freeze({
  NO_REGISTRY: "workspace-registry-unavailable",
  NO_AGENTS: "agent-service-unavailable",
  NO_SESSIONS: "session-controller-unavailable",
  NO_MESSAGE_FACTORY: "user-message-factory-unavailable",
  NOT_FOUND: "not-found",
  BAD_REQUEST: "bad-request",
  BUSY: "session-busy",
});

/**
 * Test/caller seams, keyed by ctx.
 *
 * Deliberately a WeakMap rather than a property on the context: Cordis guards
 * property access on its context object, so reading `ctx.somethingUndeclared`
 * throws "cannot get property ... without inject". A seam that can trip the
 * guard is worse than no seam.
 */
const ctxOverrides = new WeakMap();

/**
 * Attach overrides to a context without mutating it.
 * @param ctx - harness context (or a plain object in tests).
 * @param values - `{ sessionCacheDir, archivedSessionIds }`.
 * @returns the same ctx, for chaining.
 */
export function setContextOverrides(ctx, values) {
  const existing = ctxOverrides.get(ctx) ?? {};
  ctxOverrides.set(ctx, { ...existing, ...values });
  return ctx;
}

/** An operation error carrying an HTTP status and a stable code. */
export class ApiError extends Error {
  /**
   * @param code - one of {@link Failure}.
   * @param message - human-readable detail.
   * @param status - HTTP status to answer with.
   */
  constructor(code, message, status = 400) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

/**
 * Build the user-message factory, tolerating an older harness that does not
 * export `createUserMessage` from `dsh-llm`.
 *
 * The primary path is upstream's own factory. It mints the id the agent loop
 * expects, but it deliberately does **not** infer a `source`: `createUserMessage`
 * takes a *complete* user message, and the loop reads `message.source.kind`, so
 * the caller supplies `{ kind: 'user' }` — omitting it delivers a message whose
 * turn dies with `Cannot read properties of undefined (reading 'kind')`, with no
 * model call and no visible error at the route. The fallback mirrors the same
 * complete shape so both paths behave identically.
 *
 * @param load - injectable dynamic importer, for tests.
 * @returns `{ create, strategy }`.
 */
export async function resolveMessageFactory(load = (specifier) => import(specifier)) {
  try {
    const mod = await load("@deepseek-ai/dsh-llm");
    if (typeof mod?.createUserMessage === "function") {
      return { create: mod.createUserMessage, strategy: "dsh-llm:createUserMessage" };
    }
  } catch {
    // Not resolvable from this profile; fall through to the literal shape.
  }
  let counter = 0;
  const create = ({ content, source }) => ({
    id: `lan-manager-${Date.now()}-${++counter}`,
    role: "user",
    content,
    source: source ?? { kind: "user" },
  });
  return { create, strategy: "inline-user-message" };
}

/**
 * Normalize a prompt into model content blocks.
 * @param prompt - a string, or an array of already-shaped content blocks.
 * @returns content blocks.
 */
export function toContent(prompt) {
  if (typeof prompt === "string") {
    const text = prompt.trim();
    if (!text) throw new ApiError(Failure.BAD_REQUEST, "prompt must be a non-empty string", 400);
    return [{ type: "text", text }];
  }
  if (Array.isArray(prompt) && prompt.length > 0) return prompt;
  throw new ApiError(
    Failure.BAD_REQUEST,
    "prompt must be a string or a non-empty block array",
    400,
  );
}

/**
 * The workspace registry, or a typed failure.
 * @param ctx - harness context.
 * @returns the registry service.
 */
export function registry(ctx) {
  const service = ctx?.get?.("workspaceRegistry");
  if (!service || typeof service.list !== "function") {
    throw new ApiError(
      Failure.NO_REGISTRY,
      "workspaceRegistry service is not composed in this profile",
      503,
    );
  }
  return service;
}

/**
 * The agent registry, or a typed failure.
 * @param ctx - harness context.
 * @returns the agent service.
 */
export function agents(ctx) {
  const service = ctx?.get?.("agents");
  if (!service || typeof service.get !== "function") {
    throw new ApiError(Failure.NO_AGENTS, "agents service is not composed in this profile", 503);
  }
  return service;
}

/**
 * Read the registry-global archive set as plain strings.
 *
 * The durable state lives beside the `workspaces` table; the public
 * `Workspace.sessionIds` does not subtract archives, so the UI's notion of a
 * visible row is reconstructed here.
 *
 * @param ctx - harness context.
 * @returns a `Set<string>` of archived session ids (possibly empty).
 */
export function archivedSet(ctx) {
  // `ctx.get(...)` only: reading an undeclared service as a property throws
  // "cannot get property ... without inject" and fails the whole request.
  const seam = ctxOverrides.get(ctx);
  if (seam?.archivedSessionIds instanceof Set) return new Set(seam.archivedSessionIds);
  if (Array.isArray(seam?.archivedSessionIds)) return new Set(seam.archivedSessionIds.map(String));
  const state = ctx?.get?.("workspaceDomainState");
  const ids = state?.archivedSessionIds;
  if (Array.isArray(ids)) return new Set(ids.map(String));

  // Fall back to the durable registry file when the state row is not exposed.
  try {
    const home = process.env.DSH_HOME || `${process.env.HOME}/.dsh`;
    const raw = readFileSync(`${home}/storages/workspace.json`, "utf8");
    const parsed = JSON.parse(raw)?.global?.archivedSessionIds;
    if (Array.isArray(parsed)) return new Set(parsed.map(String));
  } catch {
    // No readable state: treat nothing as archived rather than hiding rows.
  }
  return new Set();
}

/**
 * Project one workspace to its API shape.
 * @param workspace - a registry `Workspace`.
 * @param archived - the archive set.
 * @returns the workspace with its visible sessions.
 */
export function projectWorkspace(workspace, archived) {
  const sessionIds = Array.isArray(workspace?.sessionIds) ? workspace.sessionIds.map(String) : [];
  const visible = sessionIds.filter((id) => !archived.has(id));
  return {
    id: String(workspace?.id ?? ""),
    path: String(workspace?.path ?? ""),
    title: String(workspace?.title ?? ""),
    createdAt: workspace?.createdAt ?? null,
    updatedAt: workspace?.updatedAt ?? null,
    sessionCount: visible.length,
    hiddenSessionCount: sessionIds.length - visible.length,
    sessionIds: visible,
  };
}

/**
 * List active workspaces — those the web page shows — newest display order kept.
 * @param ctx - harness context.
 * @param options - `{ includeEmpty?: boolean }`; empty workspaces are hidden by
 *   default because the page shows a workspace only once it owns a session.
 * @returns `{ workspaces, archivedCount, strategy }`.
 */
/**
 * The registry's workspaces, or an empty list when the service is absent.
 *
 * The page's grouping is derived from sessions, not from the registry, so a
 * profile without `workspaceRegistry` can still answer "what is active" — it
 * just loses the pinned title, explicit order and stable id. Failing the whole
 * route over missing *metadata* would be wrong.
 *
 * @param ctx - harness context.
 * @returns the registry workspaces, possibly empty.
 */
function tryRegistryList(ctx) {
  try {
    return registry(ctx).list() ?? [];
  } catch {
    return [];
  }
}

/**
 * The set of visible sessions, grouped the way the web page groups them.
 *
 * **This is the correction that matters.** The UI's workspace list is not the
 * registry's: `workspaceRegistry` holds only the workspaces a user explicitly
 * added (on this machine: two), while the page's groups are derived from the
 * sessions themselves — each session's `cwd` is its workspace. So the source of
 * truth for "what the page shows" is the session projection, and the registry is
 * metadata layered on top (a pinned title, an explicit order, an id).
 *
 * @param ctx - harness context.
 * @returns `{ groups, archived }` where `groups` is a `Map<path, {sessions}>`.
 */
function groupVisibleSessions(ctx) {
  const archived = archivedSet(ctx);
  const registryEntries = tryRegistryList(ctx);
  const registryIds = new Set(registryEntries.map((w) => String(w.path)));
  // `sessionCacheDir` is an override for tests and for a caller that keeps the
  // projection somewhere other than the default home.
  const home = process.env.DSH_HOME || `${process.env.HOME}/.dsh`;
  const dir =
    ctxOverrides.get(ctx)?.sessionCacheDir ?? `${home}/storages/session_projcache/sessions`;
  const groups = new Map();

  let entries;
  try {
    entries = readdirSync(dir).filter((name) => name.endsWith(".json"));
  } catch {
    entries = [];
  }

  for (const name of entries) {
    const sessionId = name.slice(0, -5);
    if (archived.has(sessionId)) continue;
    let doc;
    try {
      doc = JSON.parse(readFileSync(`${dir}/${name}`, "utf8"));
    } catch {
      continue; // A half-written projection row is skipped, never fatal.
    }
    const identity = doc?.record?.identity ?? {};
    const path = String(identity.cwd ?? "") || "(unknown)";
    const rows = doc?.record?.rows ?? {};
    const title = rows?.title?.val ?? null;
    const stats = rows?.sessionStats?.val ?? {};
    if (!groups.has(path)) groups.set(path, []);
    groups.get(path).push({
      sessionId,
      workspacePath: path,
      title: typeof title === "string" ? title : null,
      createdAt: Number(identity.createdAt) || 0,
      turns: Number(stats.turns) || 0,
      steps: Number(stats.steps) || 0,
    });
  }

  for (const sessions of groups.values()) sessions.sort((a, b) => b.createdAt - a.createdAt);
  return { groups, archived, registryPaths: registryIds };
}

/**
 * List active workspaces — the ones the web page shows.
 * @param ctx - harness context.
 * @param options - `{ includeEmpty?: boolean }`.
 * @returns `{ workspaces, archivedCount, strategy }`.
 */
export function listActiveWorkspaces(ctx, options = {}) {
  const { groups, archived, registryPaths } = groupVisibleSessions(ctx);
  const byPath = new Map(tryRegistryList(ctx).map((w) => [String(w.path), w]));

  const workspaces = [];
  for (const [path, sessions] of groups) {
    const registered = byPath.get(path);
    workspaces.push({
      id: registered ? String(registered.id) : null,
      path,
      title: registered?.title ?? path.split("/").filter(Boolean).pop() ?? path,
      registered: Boolean(registered),
      createdAt: registered?.createdAt ?? null,
      updatedAt: registered?.updatedAt ?? null,
      sessionCount: sessions.length,
      hiddenSessionCount: 0,
      newestSessionAt: sessions[0]?.createdAt ?? 0,
      sessionIds: sessions.map((s) => s.sessionId),
    });
  }

  // A registered workspace with no visible session is still a real workspace; the
  // page shows it as an empty group. It is included only when asked for, so the
  // default answer stays "what has activity".
  if (options.includeEmpty) {
    for (const [path, w] of byPath) {
      if (groups.has(path)) continue;
      workspaces.push({
        id: String(w.id),
        path,
        title: w.title ?? path,
        registered: true,
        createdAt: w.createdAt ?? null,
        updatedAt: w.updatedAt ?? null,
        sessionCount: 0,
        hiddenSessionCount: 0,
        newestSessionAt: 0,
        sessionIds: [],
      });
    }
  }

  workspaces.sort((a, b) => b.newestSessionAt - a.newestSessionAt);
  return {
    workspaces,
    archivedCount: archived.size,
    totalWorkspaces: workspaces.length,
    registryWorkspaces: registryPaths.size,
    strategy: "session-projection",
  };
}

/**
 * Every visible session across every active workspace, newest first per workspace.
 * @param ctx - harness context.
 * @returns `{ sessions, count }`.
 */
export function listAllActiveSessions(ctx) {
  const { groups } = groupVisibleSessions(ctx);
  const byPath = new Map(tryRegistryList(ctx).map((w) => [String(w.path), w]));
  const sessions = [];
  for (const [path, list] of groups) {
    const registered = byPath.get(path);
    for (const s of list) {
      sessions.push({
        ...s,
        workspaceId: registered ? String(registered.id) : null,
        workspaceTitle: registered?.title ?? path.split("/").filter(Boolean).pop() ?? path,
      });
    }
  }
  return { sessions, count: sessions.length };
}

export function findWorkspace(ctx, selector = {}) {
  const reg = registry(ctx);
  if (selector.workspaceId) {
    const found = reg.get(selector.workspaceId);
    if (!found) throw new ApiError(Failure.NOT_FOUND, `no workspace ${selector.workspaceId}`, 404);
    return found;
  }
  if (selector.path) {
    const wanted = String(selector.path);
    const found = reg.list().find((w) => String(w.path) === wanted);
    if (!found) throw new ApiError(Failure.NOT_FOUND, `no workspace at ${wanted}`, 404);
    return found;
  }
  throw new ApiError(Failure.BAD_REQUEST, "workspaceId or path is required", 400);
}

/**
 * List the visible sessions of one workspace.
 * @param ctx - harness context.
 * @param selector - `{ workspaceId }` or `{ path }`.
 * @returns the projected workspace.
 */
export function listWorkspaceSessions(ctx, selector) {
  // Resolve in order of specificity: a registry id, then an exact session-derived
  // path, then a path suffix. The page's groups can have no registry id at all
  // (a folder that was never explicitly added), so a suffix has to work.
  const { groups } = groupVisibleSessions(ctx);
  const raw = String(selector.workspaceId ?? selector.path ?? "").replace(/\/+$/, "");
  if (!raw) throw new ApiError(Failure.BAD_REQUEST, "workspaceId or path is required", 400);

  let path;
  let registered;
  try {
    registered = findWorkspace(ctx, { workspaceId: raw });
    path = String(registered.path);
  } catch {
    registered = undefined;
  }
  if (!path) {
    if (groups.has(raw)) path = raw;
    else path = [...groups.keys()].find((p) => p === raw || p.endsWith(`/${raw}`));
  }
  if (!path) {
    const known = [...groups.keys()];
    throw new ApiError(
      Failure.NOT_FOUND,
      `no active workspace matching "${raw}"${known.length ? ` (known: ${known.join(", ")})` : ""}`,
      404,
    );
  }

  const sessions = groups.get(path) ?? [];
  const meta = registered ?? tryRegistryList(ctx).find((w) => String(w.path) === path);
  return {
    id: meta ? String(meta.id) : null,
    path,
    title: meta?.title ?? path.split("/").filter(Boolean).pop() ?? path,
    registered: Boolean(meta),
    createdAt: meta?.createdAt ?? null,
    updatedAt: meta?.updatedAt ?? null,
    sessionCount: sessions.length,
    hiddenSessionCount: 0,
    sessionIds: sessions.map((s) => s.sessionId),
  };
}

/**
 * How much history a read returns by default, and the ceiling.
 *
 * A fleet audit wants the answer, so the newest messages are the ones kept. The
 * ceiling exists because a manager fanning out over a fleet is one HTTP response
 * per member, and a long session is megabytes.
 */
export const DEFAULT_MESSAGE_LIMIT = 40;
export const MAX_MESSAGE_LIMIT = 200;

/** Per-message character cap, so one dumped tool result cannot dominate a reply. */
export const MAX_MESSAGE_CHARS = 4000;

/**
 * Render one derived message's content into something a manager can read.
 *
 * The block vocabulary belongs to the harness and grows between releases, so a
 * block this function does not recognise is *counted* rather than dropped: a
 * reader can see that something was there. Nothing here throws on an unexpected
 * shape, because a session that recorded an unfamiliar block is still a session.
 *
 * @param content - a message's `content`, as a string or a block array.
 * @returns `{ text, reasoning, otherBlocks }`.
 */
function renderMessageContent(content) {
  const text = [];
  const reasoning = [];
  let otherBlocks = 0;
  const blocks = Array.isArray(content)
    ? content
    : typeof content === "string"
      ? [{ type: "text", text: content }]
      : [];
  for (const block of blocks) {
    if (typeof block === "string") {
      text.push(block);
      continue;
    }
    if (!block || typeof block !== "object") continue;
    const type = String(block.type ?? "");
    const body =
      typeof block.text === "string"
        ? block.text
        : typeof block.content === "string"
          ? block.content
          : "";
    if (type === "text" && body !== "") text.push(body);
    else if ((type === "thinking" || type === "reasoning") && body !== "") reasoning.push(body);
    else otherBlocks += 1;
  }
  return {
    text: text.join("\n").trim(),
    reasoningChars: reasoning.join("\n").trim().length,
    otherBlocks,
  };
}

/** A positive whole limit, clamped. Anything unparseable takes the default. */
function messageLimit(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_MESSAGE_LIMIT;
  return Math.min(Math.floor(parsed), MAX_MESSAGE_LIMIT);
}

/**
 * Derive one session's messages from the harness's own persistence, without an agent.
 *
 * `agent.session.deriveMessages()` is the harness's derivation, but it needs a live
 * agent, so an idle or archived session could not be read at all (TT-030). The cold
 * path asks the harness's cold-read service — `sessionQuery.readSession()`, which
 * loads and replay-validates the stored log — and hands that validated pair back to
 * the `sessions` service's `prepare()` with `eventState: "detached"`, the same call
 * `sessionQuery` makes internally to build its own observation. The result is a
 * detached, unentered Session whose `deriveMessages()` is the same derivation the
 * live path uses, so surface markers and compaction rules stay the harness's and no
 * frame parser is hand-rolled here.
 *
 * @param ctx - harness context.
 * @param sessionId - target session.
 * @returns the derived message array.
 */
async function derivedMessagesFromStore(ctx, sessionId) {
  const query = ctx?.get?.("sessionQuery");
  const store = ctx?.get?.("sessions");
  if (typeof query?.readSession !== "function" || typeof store?.prepare !== "function") {
    throw new ApiError(
      Failure.NO_AGENTS,
      "this profile composes no cold session reader, so a session with no live agent cannot be read",
      503,
    );
  }
  let loaded;
  try {
    loaded = await query.readSession(sessionId);
  } catch (error) {
    const code = typeof error?.code === "string" ? error.code : "";
    if (code === "SESSION_QUERY_SESSION_NOT_FOUND") {
      throw new ApiError(Failure.NOT_FOUND, `no session ${sessionId}`, 404);
    }
    throw new ApiError(
      Failure.NO_AGENTS,
      `session ${sessionId} could not be read from storage: ${error instanceof Error ? error.message : String(error)}`,
      503,
    );
  }
  const { session: header, inheritedEventCount, events } = loaded ?? {};
  let detached;
  try {
    detached = store.prepare(header.id, {
      seed: events,
      meta: header,
      inheritedEventCount,
      eventState: "detached",
    });
  } catch (error) {
    throw new ApiError(
      Failure.NO_AGENTS,
      `session ${sessionId} could not be prepared for reading: ${error instanceof Error ? error.message : String(error)}`,
      503,
    );
  }
  return detached.deriveMessages() ?? [];
}

/**
 * Read the message history of a session, as far as the harness will derive it.
 *
 * The history is the harness's own derivation, so this plugin holds no opinion about
 * session format. A **live** agent is preferred and cheap — its session is already in
 * memory — and with none the stored log is read instead, so an idle or archived
 * session still answers (TT-030). A session that exists in neither is a `404` rather
 * than an empty conversation, which would read as "this session said nothing".
 *
 * @param ctx - harness context.
 * @param sessionId - target session.
 * @param options - `{ limit?: number }`.
 * @returns `{ sessionId, total, returned, truncated, messages }`.
 */
export async function readSessionMessages(ctx, sessionId, options = {}) {
  const agent = agents(ctx).get(sessionId);
  let derived;
  if (agent) {
    const session = agent.session;
    if (!session || typeof session.deriveMessages !== "function") {
      throw new ApiError(
        Failure.NO_AGENTS,
        "the agent for this session does not expose its history",
        503,
      );
    }
    derived = session.deriveMessages() ?? [];
  } else {
    derived = await derivedMessagesFromStore(ctx, sessionId);
  }
  const limit = messageLimit(options.limit);
  const kept = derived.slice(-limit);
  const messages = kept.map((message) => {
    const rendered = renderMessageContent(message?.content);
    const cut = rendered.text.length > MAX_MESSAGE_CHARS;
    return {
      id: message?.id ?? null,
      role: String(message?.role ?? "unknown"),
      text: cut ? rendered.text.slice(0, MAX_MESSAGE_CHARS) : rendered.text,
      textTruncated: cut,
      // Reasoning is not the answer an audit came for, so only its size is reported.
      reasoningChars: rendered.reasoningChars,
      otherBlocks: rendered.otherBlocks,
    };
  });

  return {
    sessionId,
    total: derived.length,
    returned: messages.length,
    truncated: derived.length > messages.length,
    messages,
  };
}

/**
 * Enqueue a prompt on one session.
 *
 * `followup` is the same entry point the SDK server uses for a queued user
 * message, and it always wakes the agent — so the receipt reports `wakeup: true`
 * as a fact about what happened, not as an echo of an option nothing reads.
 *
 * @param ctx - harness context.
 * @param sessionId - target session.
 * @param prompt - string or content blocks.
 * @param factory - a resolved message factory.
 * @returns a delivery receipt.
 */
export function promptSession(ctx, sessionId, prompt, factory) {
  const service = agents(ctx);
  const agent = service.get(sessionId);
  if (!agent) {
    throw new ApiError(
      Failure.NOT_FOUND,
      `session ${sessionId} has no live agent (start it in the UI, then retry)`,
      404,
    );
  }
  if (typeof agent.followup !== "function") {
    throw new ApiError(Failure.NO_AGENTS, "agent does not accept follow-up input", 503);
  }
  const message = factory.create({ content: toContent(prompt), source: { kind: "user" } });
  agent.followup(message);
  return {
    sessionId,
    delivered: true,
    messageId: message?.id ?? null,
    wakeup: true,
  };
}

/**
 * Fan a prompt out to every active session, or to a chosen subset.
 *
 * Delivery is per-session and never all-or-nothing: one session without a live
 * agent must not stop the rest of the fleet, so failures are reported beside
 * successes instead of aborting the request.
 *
 * @param ctx - harness context.
 * @param prompt - string or content blocks.
 * @param factory - a resolved message factory.
 * @param options - `{ sessionIds?: string[], limit?: number }`.
 * @returns `{ delivered, failed, total }`.
 */
export function promptAllActive(ctx, prompt, factory, options = {}) {
  const { sessions } = listAllActiveSessions(ctx);
  const wanted =
    Array.isArray(options.sessionIds) && options.sessionIds.length > 0
      ? sessions.filter((s) => options.sessionIds.includes(s.sessionId))
      : sessions;
  const capped =
    Number.isInteger(options.limit) && options.limit > 0 ? wanted.slice(0, options.limit) : wanted;

  const delivered = [];
  const failed = [];
  for (const session of capped) {
    try {
      delivered.push(promptSession(ctx, session.sessionId, prompt, factory));
    } catch (error) {
      failed.push({
        sessionId: session.sessionId,
        code: error?.code ?? "error",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }
  return { delivered, failed, total: capped.length, considered: sessions.length };
}

/**
 * Archive one session (hide it from the UI, keep its history and its slot).
 * @param ctx - harness context.
 * @param sessionId - target session.
 * @returns `{ sessionId, archived: true }`.
 */
export async function archiveSession(ctx, sessionId) {
  const reg = registry(ctx);
  if (typeof reg.archiveSession !== "function") {
    throw new ApiError(Failure.NO_REGISTRY, "this harness does not expose archiveSession", 503);
  }
  const exists = reg
    .list()
    .some((w) => (w.sessionIds ?? []).map(String).includes(String(sessionId)));
  if (!exists)
    throw new ApiError(Failure.NOT_FOUND, `no session ${sessionId} in any workspace`, 404);
  await reg.archiveSession(sessionId);
  return { sessionId, archived: true };
}

/**
 * Delete a workspace. With `archiveSessions: true` (the default) the sessions are
 * archived first so a stray delete does not silently drop history; the registry
 * itself never touches the folder or the session logs.
 * @param ctx - harness context.
 * @param workspaceId - target workspace.
 * @param options - `{ archiveSessions?: boolean }`.
 * @returns a deletion receipt.
 */
export async function deleteWorkspace(ctx, workspaceId, options = {}) {
  const reg = registry(ctx);
  const workspace = findWorkspace(ctx, { workspaceId });
  const sessionIds = (workspace.sessionIds ?? []).map(String);
  const archiveFirst = options.archiveSessions !== false;
  const archived = [];
  const archiveFailures = [];
  if (archiveFirst && typeof reg.archiveSession === "function") {
    for (const id of sessionIds) {
      try {
        await reg.archiveSession(id);
        archived.push(id);
      } catch (error) {
        archiveFailures.push({
          sessionId: id,
          message: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }
  const removed = await reg.delete(workspaceId);
  if (!removed) throw new ApiError(Failure.NOT_FOUND, `no workspace ${workspaceId}`, 404);
  return {
    workspaceId: String(workspaceId),
    path: workspace.path,
    deleted: true,
    archivedSessionIds: archived,
    archiveFailures,
  };
}

/**
 * Start a live agent on a new session, through the harness's own session controller.
 *
 * This is deliberately a **delegation**, not a reimplementation. Starting a session
 * is not one call: the controller composes the agent's world from the preset,
 * resolves the default model, creates the working directory, mints the session id,
 * ensures the session/agent pair, and attaches it to the workspace. Rebuilding that
 * here would be a second implementation of the harness's own logic, drifting from
 * it at every release — so the plugin asks the service that already does it.
 *
 * The service is `sessionController` (`@deepseek-ai/dsh-api-session-controller`,
 * `super(ctx, "sessionController", …)`) and **not** `sessions`: that name is the
 * raw `@deepseek-ai/dsh-session` store, whose `create(id, options)` mints a bare
 * session and takes an id first — calling it with a request object throws
 * `session header id "[object Object]" does not match session id "[object Object]"`
 * and produces no agent. Found by driving the route against a real harness, which
 * is the one thing the unit test's fake could not see.
 *
 * The 501 is kept for the profile that composes no session controller (a headless
 * or SDK-only runtime): answering "created" for something that was not created is
 * the one thing worse than saying no.
 *
 * @param ctx - harness context.
 * @param selector - `{ workspaceId }` or `{ path | cwd }`, plus optional `agentPreset`.
 * @returns `{ sessionId, agentPreset }`.
 */
export async function startSession(ctx, selector = {}) {
  const service = ctx?.get?.("sessionController");
  if (!service || typeof service.create !== "function") {
    throw new ApiError(
      Failure.NO_SESSIONS,
      "this profile does not compose the harness session controller, so a session cannot be started from here",
      501,
    );
  }
  const request = {};
  if (selector.workspaceId !== undefined) request.workspaceId = String(selector.workspaceId);
  else if (selector.path !== undefined) request.cwd = String(selector.path);
  else if (selector.cwd !== undefined) request.cwd = String(selector.cwd);
  else throw new ApiError(Failure.BAD_REQUEST, "workspaceId or path is required", 400);
  if (selector.agentPreset !== undefined) request.agentPreset = String(selector.agentPreset);

  try {
    const created = await service.create(request);
    return {
      sessionId: String(created?.sessionId ?? ""),
      agentPreset: created?.agentPreset ?? null,
    };
  } catch (error) {
    // Only an error the controller *authored* keeps its code and its message: that
    // is how a caller tells "no such workspace" from "the preset is wrong". A throw
    // with no code is not a documented failure -- it is `node:fs`, a child process,
    // or a bug -- and its message can name a path on this Mac, so it goes back out
    // unchanged and the router's catch-all logs it and answers a fixed 500. It is
    // also not the caller's bad request, which is what the old fallback said.
    if (typeof error?.code !== "string") throw error;
    const status = error.code.includes("not-found") ? 404 : 400;
    throw new ApiError(error.code, error.message, status);
  }
}

/**
 * Register an existing folder as a workspace.
 *
 * The harness's own `create(path, title)` refuses anything that is not already a
 * directory, so this adds a *record*: it does not make a folder. A manager that
 * asks for a workspace at a path that does not exist gets a clear error instead
 * of a half-made directory tree.
 *
 * @param ctx - harness context.
 * @param selector - `{ path, title }`.
 * @returns a receipt for the created (or already-present) workspace.
 */
export async function createWorkspace(ctx, selector = {}) {
  const reg = registry(ctx);
  const path = typeof selector.path === "string" ? selector.path.trim() : "";
  if (!path) throw new ApiError(Failure.BAD_REQUEST, "path is required", 400);
  if (typeof reg.create !== "function") {
    throw new ApiError(Failure.NO_REGISTRY, "this harness does not expose workspace creation", 503);
  }
  let workspace;
  try {
    workspace = await reg.create(path, selector.title);
  } catch (error) {
    // `create` throws for a missing directory and for a path that is a file;
    // both are the caller's to fix, so they are a 400 and not a 500.
    throw new ApiError(
      Failure.BAD_REQUEST,
      error instanceof Error ? error.message : String(error),
      400,
    );
  }
  return {
    workspaceId: String(workspace?.id ?? ""),
    path: String(workspace?.path ?? path),
    title: String(workspace?.title ?? selector.title ?? ""),
    created: true,
  };
}
