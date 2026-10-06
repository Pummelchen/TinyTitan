/**
 * The LAN management router.
 *
 * One prefix route carries every endpoint. The order of the guards is the
 * security design, and it is deliberate:
 *
 * 1. **Source address** — checked before the body is read and before any handler
 *    runs. A caller outside the allowlist gets `403` and the plugin does no work.
 * 2. **Shared token** — when configured, compared in constant time. Optional so a
 *    single-user LAN needs no secret, but it is the only thing that distinguishes
 *    two machines on the same private range. Because the shipped default is a
 *    published literal, it distinguishes nothing until an operator changes it, so
 *    a **mutating** request from outside loopback while the key is still the
 *    default is refused: the default is a group tag, and the trade it represents
 *    is one a user makes with their own machine and no one else's. And a request
 *    from a **link-local** source while the key is still the default gets the
 *    group routes and nothing else: `169.254/16` and `fe80::/10` are where a host
 *    lands when nothing assigned it an address, so membership there is claimed,
 *    not granted, and reading another machine's prompts on that basis is not the
 *    zero-setup trade anyone agreed to.
 * 3. **Origin, on mutating verbs only** — a browser on an allowed host must not be
 *    usable as a confused deputy by a page from elsewhere. Absent Origin (curl,
 *    another dsh instance) is fine; a foreign Origin is refused. Same-origin is
 *    judged against hosts this server can name, not against the request's own
 *    `Host` header, because an equality the caller's DNS chose is the rebinding
 *    shape.
 *
 * Responses are JSON, and a response never repeats the group key: the group is
 * named by its digest, so the string that authorises a write does not end up in a
 * log, a capture, or a screenshot of the `/health` banner (AUD-154).
 *
 * Responses are JSON. Bodies are capped so a stray client cannot stream the host
 * out of memory, and a body is only read as JSON when its `content-type` says so:
 * the three media types a browser form may post without a preflight are not JSON,
 * and refusing them is what keeps Guard 3 the *second* layer rather than the only
 * one — `enforceOrigin: false` is a documented off switch, and the layer under it
 * must not depend on it.
 *
 * @module dsh-lan-manager/router
 */

import { timingSafeEqual } from "node:crypto";

import { DEFAULT_GROUP_KEY } from "./config.js";
import {
  ApiError,
  archiveSession,
  createWorkspace,
  deleteWorkspace,
  listActiveWorkspaces,
  listAllActiveSessions,
  listWorkspaceSessions,
  promptAllActive,
  promptSession,
  readSessionMessages,
  startSession,
} from "./api.js";
import { checkAddress, isLinkLocal, isLoopback, peerAddress } from "./net.js";

/** Default request body cap: prompts are text, not uploads. */
export const DEFAULT_MAX_BODY_BYTES = 256 * 1024;

/**
 * Constant-time string comparison that tolerates length mismatch.
 * @param a - first string.
 * @param b - second string.
 * @returns true when equal.
 */
function safeEqual(a, b) {
  const left = Buffer.from(String(a ?? ""), "utf8");
  const right = Buffer.from(String(b ?? ""), "utf8");
  if (left.length !== right.length || left.length === 0) return false;
  return timingSafeEqual(left, right);
}

/**
 * The routes that answer with membership rather than content.
 *
 * Guard 2c lets a link-local source carrying the shipped key have these and
 * nothing else. Without them the mesh cannot form at all — `/health` is what a
 * probe reads and `/peers` is what a member gossips from — and what they carry is
 * a name, an address, and counts. Everything past them is content: workspace
 * titles, session lists, and under `/sessions/:id/messages` the prompts
 * themselves.
 *
 * @param route - the sub-path after the configured prefix.
 * @returns true when the route names the group instead of reading it.
 */
function isGroupRoute(route) {
  return route === "/" || route === "/health" || route === "/peers";
}

/**
 * The refusal both default-key guards answer with, in their own words.
 * @param code - which guard fired.
 * @returns the 403 body.
 */
function defaultKeyRefusal(code) {
  return {
    error: code,
    hint: "the shipped group key is public, so it groups rather than protects: set groupKey (or DSH_LAN_KEY) to a value you chose",
  };
}

/**
 * Read and parse a JSON body under a byte cap.
 *
 * A body that does not announce itself as JSON is refused with `415` before it is
 * parsed. `JSON.parse` on whatever arrives would accept a `<form>` submission —
 * `text/plain` is one of the three media types a browser may post without a
 * preflight, and it carries a JSON object through unchanged — which would leave
 * Guard 3 as the only thing between that form and a mutating route, and
 * `enforceOrigin: false` is a documented way to remove it. Requiring the header
 * costs no caller: `FleetClient.swift:89` sets `application/json` for every body
 * it writes, and a request with *no* body — the archive and delete posts — sends
 * no content-type at all and still passes, because there is nothing to misrepresent.
 *
 * @param req - the request.
 * @param maxBytes - cap.
 * @returns the parsed body, or `{}` when empty.
 */
export async function readJsonBody(req, maxBytes = DEFAULT_MAX_BODY_BYTES) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > maxBytes) {
      // Stop reading and drop the connection: leaving an unread body on a
      // keep-alive socket is how one request's bytes get read as the next one's.
      req.destroy?.();
      throw new ApiError("body-too-large", `request body exceeds ${maxBytes} bytes`, 413);
    }
    chunks.push(chunk);
  }
  if (size === 0) return {};
  const text = Buffer.concat(chunks).toString("utf8").trim();
  if (!text) return {};
  const contentType = String(req.headers?.["content-type"] ?? "").toLowerCase();
  if (!contentType.startsWith("application/json")) {
    throw new ApiError("unsupported-media-type", "content-type must be application/json", 415);
  }
  try {
    const parsed = JSON.parse(text);
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new ApiError("bad-json", "body must be a JSON object", 400);
    }
    return parsed;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError("bad-json", "body is not valid JSON", 400);
  }
}

/**
 * Write a JSON response.
 * @param res - the response.
 * @param status - HTTP status.
 * @param payload - JSON-serializable payload.
 */
export function sendJson(res, status, payload) {
  const body = `${JSON.stringify(payload, null, 2)}\n`;
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
  });
  res.end(body);
}

/**
 * Normalize a request path for matching: strip a trailing slash, keep the prefix.
 * @param url - the request url.
 * @param basePath - configured prefix.
 * @returns the sub-path after the prefix, or `undefined` when it does not match.
 */
export function subPath(url, basePath) {
  const pathname =
    String(url ?? "/")
      .split("?")[0]
      .replace(/\/+$/, "") || "/";
  if (pathname === basePath) return "/";
  if (pathname.startsWith(`${basePath}/`)) return pathname.slice(basePath.length);
  return undefined;
}

/**
 * Build the request handler.
 * @param options - `{ ctx, config, messageFactory, log }`.
 * @returns an async `(req, res) => void` handler.
 */
export function createHandler(options) {
  // Deliberately NOT destructured. `messageFactory` is supplied as a getter so
  // the router sees the factory once its background dynamic import settles;
  // destructuring would invoke that getter once, at construction, and freeze the
  // pre-settlement `undefined`. Every read goes through this accessor instead.
  const readFactory = () => {
    const value = options.messageFactory;
    // The plugin passes a getter, so a background import that settles later is
    // visible to every request; a test may pass the factory itself. A factory is
    // identifiable by its `create`, so a function without one is a getter.
    if (typeof value === "function" && typeof value.create !== "function") return value();
    return value;
  };
  const { ctx, config, log = () => {}, peers, self } = options;
  const basePath = config.basePath;
  const maxBodyBytes = config.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES;
  const netOptions = {
    ipv4Networks: config.ipv4Networks,
    ipv6Networks: config.ipv6Networks,
    allow: config.allowAddresses,
  };

  return async function handle(req, res) {
    const address = peerAddress(req);

    // Guard 1: source address, before any parsing or work.
    const verdict = checkAddress(address, netOptions);
    if (!verdict.allowed) {
      log(`denied ${address} (${verdict.reason}) ${req.method} ${req.url}`);
      sendJson(res, 403, {
        error: "source-not-allowed",
        reason: verdict.reason,
        source: verdict.address,
        hint: "add the host to allowAddresses or extend ipv4Networks in the plugin config",
      });
      return;
    }

    const route = subPath(req.url, basePath);
    if (route === undefined) {
      sendJson(res, 404, { error: "not-found", path: req.url });
      return;
    }

    // Guard 2: shared token, when configured.
    if (config.token) {
      const presented = req.headers["x-dsh-token"] ?? "";
      if (!safeEqual(presented, config.token)) {
        log(`bad token from ${address} ${req.method} ${req.url}`);
        sendJson(res, 401, { error: "unauthorized", hint: "send the shared token in x-dsh-token" });
        return;
      }
    }

    const method = String(req.method ?? "GET").toUpperCase();
    const mutating = method !== "GET" && method !== "HEAD";

    // Guard 2b: the shipped default group key only counts inside loopback.
    //
    // `tinytitan-lan` is in this repository's source, so as a door key it admits
    // anyone who has read it — which is the documented trade for a single user on
    // a network of their own, and the reason the harness binds loopback only. The
    // fence above widens past that on purpose (LAN, tailnet, and link-local and
    // CGNAT included, so a fleet needs no setup), and a caller from one of those
    // ranges is not the user. So past loopback the key has to be one an operator
    // chose. Reads stay allowed from a range the operator chose — a peer on a LAN
    // or tailnet still has to be inventoried before it can be driven — and
    // Guard 2c takes that back for the one class of source that did not get its
    // address from the operator at all.
    if (mutating && config.token === DEFAULT_GROUP_KEY && !isLoopback(verdict.address)) {
      log(`default group key from ${verdict.address} on ${method} ${req.url}`);
      sendJson(res, 403, defaultKeyRefusal("default-group-key-outside-loopback"));
      return;
    }

    // Guard 2c: the shipped key plus a self-assigned source buys membership only.
    //
    // `169.254.0.0/16` is what a host answers with when nothing handed it an
    // address, and every IPv6 host holds an `fe80::/10` one whether or not a
    // network was ever configured — so on those ranges Guard 1's "is this my LAN"
    // is really "did this host name itself". That is fine for being *listed*, and
    // it is not fine for reading another machine's prompts while the key is still
    // the published default. An operator who means the pair — a Mac linked
    // directly to a Mac — has set a key, and is untouched by this.
    if (
      config.token === DEFAULT_GROUP_KEY &&
      isLinkLocal(verdict.address) &&
      !isGroupRoute(route)
    ) {
      log(`default group key from link-local ${verdict.address} on ${method} ${req.url}`);
      sendJson(res, 403, defaultKeyRefusal("default-group-key-link-local-source"));
      return;
    }

    // Guard 3: Origin on mutating verbs only, and only when one is present.
    //
    // An absent Origin is accepted on purpose. A browser sends Origin on **every**
    // POST — same-origin, cross-origin, and a plain `<form>` submission included —
    // so there is no browser request that reaches a mutating route without it, and
    // the requests that do arrive header-less are programs: `ttlanmanager` posts
    // through URLSession, which sends no Origin (FleetClient.swift:198). Demanding
    // the header would break the sanctioned client and close no browser path.
    // What authorises a non-browser caller instead is Guard 1's source address and
    // Guard 2's token, and `test/router.test.js` pins that both still bite when
    // there is no Origin to check.
    if (mutating && config.enforceOrigin !== false) {
      const origin = req.headers.origin;
      if (origin && !isAllowedOrigin(origin, req.headers.host, config)) {
        log(`foreign origin ${origin} from ${address}`);
        sendJson(res, 403, { error: "origin-not-allowed", origin });
        return;
      }
    }

    try {
      const result = await dispatch({
        route,
        method,
        req,
        res,
        ctx,
        config,
        peers,
        self,
        messageFactory: readFactory(),
        maxBodyBytes,
        source: verdict,
      });
      if (result !== undefined) sendJson(res, result.status ?? 200, result.body ?? result);
    } catch (error) {
      if (error instanceof ApiError) {
        sendJson(res, error.status, { error: error.code, message: error.message });
        return;
      }
      log(`handler threw: ${error instanceof Error ? (error.stack ?? error.message) : error}`);
      // The detail stays in the log and out of the answer. An unexpected throw is
      // most often a filesystem or child-process one, and Node puts the absolute
      // path in its message, so echoing it tells any peer that can reach this port
      // where this Mac keeps its profiles. `ApiError` messages are authored here and
      // still go out, because they are the documented answer for a known failure.
      sendJson(res, 500, {
        error: "internal-error",
        message: "the manager failed to answer; the reason is in its log",
      });
    }
  };
}

/**
 * Is an authority one this server can name as its own?
 *
 * The question exists because a same-origin claim is only as good as the host it
 * is compared against, and the `Host` header is chosen by the caller's DNS: an
 * attacker whose domain resolves to this machine sends `Origin:` and `Host:` both
 * as `attacker.example:P`, and a bare equality calls that same-origin. That is
 * DNS rebinding, and the answer to it is a set the server knows — loopback, a
 * host inside its own address fence, or a name the operator listed in
 * `trustedHosts` / `trustedOrigins`.
 *
 * Deliberately **not** applied to every request: the sanctioned manager reads
 * peers by hostname (`ttlanmanager` connects to `mordor.local:3080`, no Origin),
 * so a blanket Host rule would break the client that is supposed to reach the
 * fleet. It gates the same-origin *inference* only.
 *
 * @param authority - a `Host` or `Origin` host, port included, brackets allowed.
 * @param config - resolved config.
 * @returns true when this server can recognise the name.
 */
export function isKnownHost(authority, config) {
  const text = String(authority ?? "").trim();
  if (!text) return false;
  const bracketed = /^\[(.+?)](?::\d+)?$/.exec(text);
  const hostname = (bracketed ? bracketed[1] : text.replace(/:\d+$/, "")).toLowerCase();
  if (!hostname) return false;
  if (hostname === "localhost") return true;
  for (const extra of config.trustedHosts ?? []) {
    const wanted = String(extra).toLowerCase();
    if (hostname === wanted || hostname === wanted.replace(/:\d+$/, "")) return true;
  }
  for (const extra of config.trustedOrigins ?? []) {
    if (hostname === String(extra).toLowerCase().replace(/:\d+$/, "")) return true;
  }
  if (config.allowPrivateOrigins === false) return false;
  return checkAddress(hostname, {
    ipv4Networks: config.originNetworks ?? config.ipv4Networks,
    ipv6Networks: config.ipv6Networks,
  }).allowed;
}

/**
 * Is an Origin header acceptable for a mutating request?
 *
 * Accepts: an Origin whose authority equals the request Host **and** names a host
 * this server recognises (see {@link isKnownHost}), a host inside the allowlist
 * (when `allowPrivateOrigins` is not disabled — the normal single-fleet case,
 * where the page is served from the same LAN), or one explicitly trusted. Rejects
 * everything else, which is what stops an unrelated site in an allowed browser
 * from driving the API.
 *
 * @param origin - the Origin header.
 * @param host - the Host header.
 * @param config - resolved config.
 * @returns true when acceptable.
 */
export function isAllowedOrigin(origin, host, config) {
  let parsed;
  try {
    parsed = new URL(origin);
  } catch {
    return false;
  }
  if (host && parsed.host === host && isKnownHost(host, config)) return true;
  for (const extra of config.trustedOrigins ?? []) {
    if (parsed.host === extra || parsed.hostname === extra) return true;
  }
  if (parsed.hostname === "localhost") return true;
  if (config.allowPrivateOrigins === false) return false;
  return checkAddress(parsed.hostname, {
    ipv4Networks: config.originNetworks ?? config.ipv4Networks,
    ipv6Networks: config.ipv6Networks,
  }).allowed;
}

/**
 * Route one request to its operation.
 * @param args - handler state.
 * @returns `{status, body}` or a value the caller serializes.
 */
async function dispatch({
  route,
  method,
  req,
  res,
  ctx,
  config,
  peers,
  self,
  messageFactory,
  maxBodyBytes,
  source,
}) {
  if (method === "OPTIONS") return { status: 204, body: { ok: true } };

  if (route === "/" || route === "/health") {
    return {
      body: {
        ok: true,
        plugin: "dsh-lan-manager",
        version: config.version ?? null,
        dshHome: process.env.DSH_HOME ?? null,
        group: config.groupDigest ?? null,
        ...(config.groupLabel ? { groupLabel: config.groupLabel } : {}),
        self: self ?? null,
        peerCount: peers?.list().length ?? 0,
        discoveryIntervalSeconds: config.discoveryIntervalSeconds ?? null,
        lastDiscovery: peers?.lastRefresh ?? null,
        discoveryErrors: peers?.lastDiscoveryErrors ?? [],
        messageStrategy: messageFactory?.strategy ?? "unavailable",
        source: { address: source.address, family: source.family, reason: source.reason },
        endpoints: [
          "GET  /health",
          "GET  /workspaces",
          "GET  /sessions",
          "GET  /sessions/:id/messages",
          "GET  /workspaces/:id/sessions",
          "GET  /peers",
          "GET  /peers/:id",
          "GET  /inventory",
          "POST /prompt",
          "POST /prompt-all",
          "POST /sessions",
          "POST /workspaces",
          "POST /sessions/:id/archive",
          "POST /workspaces/:id/delete",
        ],
      },
    };
  }

  // --- the group -----------------------------------------------------------
  // `/peers` is the light list — who is in the group, how big each one is — and
  // is also what the mesh gossips. `/inventory` is the aggregate a manager
  // reads: this instance's own workspaces and sessions, plus every member's,
  // in one call, so a manager never has to walk the fleet to draw the picture.
  if (route === "/peers") {
    const light = (peers?.list() ?? []).map((peer) => ({
      id: peer.id,
      address: peer.address,
      port: peer.port,
      name: peer.name,
      source: peer.source,
      version: peer.version,
      lastSeen: peer.lastSeen,
      rttMs: peer.rttMs,
      workspaceCount: peer.workspaceCount,
      sessionCount: peer.sessionCount,
    }));
    return {
      body: {
        ok: true,
        group: config.groupDigest ?? null,
        ...(config.groupLabel ? { groupLabel: config.groupLabel } : {}),
        self: self ?? null,
        lastDiscovery: peers?.lastRefresh ?? null,
        discoveryIntervalSeconds: config.discoveryIntervalSeconds ?? null,
        // A short list with a dead probe must not read as a small fleet: the
        // reason rides with the peers it failed to find (AUD-133).
        discoveryErrors: peers?.lastDiscoveryErrors ?? [],
        peers: light,
      },
    };
  }

  if (route === "/inventory") {
    const own = listActiveWorkspaces(ctx, { includeEmpty: config.includeEmptyWorkspaces === true });
    const sessions = listAllActiveSessions(ctx);
    return {
      body: {
        ok: true,
        group: config.groupDigest ?? null,
        ...(config.groupLabel ? { groupLabel: config.groupLabel } : {}),
        self: self ?? null,
        lastDiscovery: peers?.lastRefresh ?? null,
        workspaces: own.workspaces ?? [],
        sessions: sessions.sessions ?? [],
        peers: peers?.list() ?? [],
      },
    };
  }

  const onePeer = /^\/peers\/(.+)$/.exec(route);
  if (method === "GET" && onePeer) {
    const found = peers?.get(decodeURIComponent(onePeer[1]));
    if (!found) throw new ApiError("not-found", `no peer ${decodeURIComponent(onePeer[1])}`, 404);
    return { body: { ok: true, peer: found } };
  }

  // Register an existing folder as a workspace, optionally starting a session on
  // it. Starting is delegated to the harness's own session controller, so the
  // plugin does not reimplement agent composition (TT-028).
  if (method === "POST" && route === "/workspaces") {
    const body = await readJsonBody(req, maxBodyBytes);
    const receipt = await createWorkspace(ctx, { path: body.path, title: body.title });
    // Starting needs the workspace to exist, so it is a second step. If it fails,
    // the typed error propagates and the workspace half is still there — which the
    // message says rather than implying nothing happened.
    const session =
      body.startSession === true
        ? await startSession(ctx, {
            workspaceId: receipt.workspaceId,
            agentPreset: body.agentPreset,
          })
        : null;
    return { body: { ok: true, ...receipt, ...(session === null ? {} : { session }) } };
  }

  if (method === "POST" && route === "/sessions") {
    const body = await readJsonBody(req, maxBodyBytes);
    const session = await startSession(ctx, {
      workspaceId: body.workspaceId,
      path: body.path,
      cwd: body.cwd,
      agentPreset: body.agentPreset,
    });
    return { body: { ok: true, ...session } };
  }

  if (method === "GET" && route === "/workspaces") {
    const result = listActiveWorkspaces(ctx, {
      includeEmpty: config.includeEmptyWorkspaces === true,
    });
    return { body: { ok: true, ...result } };
  }

  if (method === "GET" && route === "/sessions") {
    const result = listAllActiveSessions(ctx);
    return { body: { ok: true, ...result } };
  }

  // Reading the history back is what makes a fleet audit an audit: `prompt-all`
  // delivers the question, and this collects the answers. `?limit=` caps how many
  // of the newest messages come back.
  const sessionMessages = /^\/sessions\/([^/]+)\/messages$/.exec(route);
  if (method === "GET" && sessionMessages) {
    const sessionId = decodeURIComponent(sessionMessages[1]);
    const limit = new URL(String(req.url ?? "/"), "http://placeholder").searchParams.get("limit");
    return { body: { ok: true, ...(await readSessionMessages(ctx, sessionId, { limit })) } };
  }

  // The tail may be a registry id or a path (a page-visible workspace that was
  // never explicitly registered has no id), so allow slashes inside it.
  const wsSessions = /^\/workspaces\/(.+)\/sessions$/.exec(route);
  if (method === "GET" && wsSessions) {
    const workspaceId = decodeURIComponent(wsSessions[1]);
    return { body: { ok: true, workspace: listWorkspaceSessions(ctx, { workspaceId }) } };
  }

  if (method === "POST" && route === "/prompt") {
    const body = await readJsonBody(req, maxBodyBytes);
    const sessionId = body.sessionId ?? body.session;
    if (!sessionId) throw new ApiError("bad-request", "sessionId is required", 400);
    if (body.prompt === undefined) throw new ApiError("bad-request", "prompt is required", 400);
    const receipt = promptSession(ctx, String(sessionId), body.prompt, messageFactory);
    return { body: { ok: true, ...receipt } };
  }

  if (method === "POST" && route === "/prompt-all") {
    const body = await readJsonBody(req, maxBodyBytes);
    if (body.prompt === undefined) throw new ApiError("bad-request", "prompt is required", 400);
    const result = promptAllActive(ctx, body.prompt, messageFactory, body);
    return {
      status: result.failed.length > 0 && result.delivered.length === 0 ? 502 : 200,
      body: { ok: result.delivered.length > 0, ...result },
    };
  }

  const archive = /^\/sessions\/([^/]+)\/archive$/.exec(route);
  if (method === "POST" && archive) {
    const sessionId = decodeURIComponent(archive[1]);
    const receipt = await archiveSession(ctx, sessionId);
    return { body: { ok: true, ...receipt } };
  }

  const removeWs = /^\/workspaces\/([^/]+)\/delete$/.exec(route);
  if (method === "POST" && removeWs) {
    const workspaceId = decodeURIComponent(removeWs[1]);
    const body = await readJsonBody(req, maxBodyBytes);
    const receipt = await deleteWorkspace(ctx, workspaceId, {
      archiveSessions: body.archiveSessions !== false,
    });
    return { body: { ok: true, ...receipt } };
  }

  sendJson(res, 404, { error: "not-found", route, method });
  return undefined;
}
