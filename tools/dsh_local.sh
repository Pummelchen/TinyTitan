#!/usr/bin/env bash
# TinyTitan's own, isolated DeepSeek Harness.
#
# `dsh web` is the optional browser window the installer can set up: a local
# page with a prompt box, pointed at the TinyTitan server. This script owns that
# runtime and nothing else.
#
# **Isolation is the point.** A user may already run DeepSeek Harness — their
# own `dsh` on PATH, their own `~/.dsh`, their own UI on 3080. Nothing here may
# touch any of it, so:
#
#   * our `dsh` is installed into a private npm prefix and is never put on PATH;
#   * DSH_HOME points at our own home, so profiles, sessions, settings and the
#     plugin copy live there and the user's ~/.dsh is never read or written;
#   * pnpm's store and pnpm's own home (`~/Library/pnpm` on macOS) are redirected
#     into the same private root;
#   * npm is told to keep its cache, its logs and its user config there too.
#     `--prefix` moves where packages are unpacked, not where npm writes: without
#     this, every install left `~/.npm/_cacache` and `~/.npm/_logs` in the user's
#     home and read their `~/.npmrc` (measured 2026-09-24, fixed here);
#   * the harness runs with a private `XDG_CACHE_HOME` and `XDG_STATE_HOME`, so a
#     tool the agent runs cannot drop a cache into the user's home either.
#     `HOME`, `XDG_CONFIG_HOME` and `XDG_DATA_HOME` are deliberately left alone:
#     the agent works inside the user's repositories, so it must be able to
#     *read* their git identity and their `gh`/registry credentials. Isolation is
#     about what this bundle writes, not about blinding the tools it drives;
#   * the browser UI binds 7788 (TINYTITAN_DSH_PORT), not DSH's default 3080,
#     and steps up to the next free port if 7788 is already taken. A port held
#     by an earlier run of *this* install is stopped and reused; a port held by
#     anything else is left alone and the search moves on.
#
# Node is reused when the machine already has one, and only fetched into
# ~/.tinytitan/dsh/node when it does not, so this never runs `brew install` and
# never writes a global npm prefix. The fetch is checked against the digest
# nodejs.org publishes for that exact version before anything is unpacked: what
# lands in the private root is executed by the harness seconds later.
#
# The DeepSeek Harness version is **pinned**. It is the version this project
# supports and has tested the plugin against; DSH is in developer preview and
# says outright that it will break compatibility between releases, so a floating
# version here would turn a working install into a broken one overnight. Moving
# the pin is a deliberate change, made with the plugin in hand.
#
#   tools/dsh_local.sh ensure [--port N] [--model ID]
#                                          install or refresh the private runtime,
#                                          and point the harness default at ID
#                                          (default: the first model the route serves)
#   tools/dsh_local.sh web [--dsh-port N]  run our `dsh web` (server already up)
#   tools/dsh_local.sh smoke [-- args]     drive the real page in a headless
#                                          browser and prove a prompt is answered;
#                                          needs a server already running, and
#                                          takes `--prompt`, `--expect`, `--timeout`
#   tools/dsh_local.sh port [--dsh-port N] print the free port `web` would use
#   tools/dsh_local.sh status              what is installed, and where
#   tools/dsh_local.sh paths               the private paths, for scripting
#   tools/dsh_local.sh --help
#
# Environment:
#   TINYTITAN_DSH_ROOT            private root (default ~/.tinytitan/dsh)
#   TINYTITAN_DSH_VERSION         pinned DeepSeek Harness version
#   TINYTITAN_DSH_NODE_VERSION    pinned Node, used only when the Mac has none
#   TINYTITAN_DSH_PNPM_VERSION    pinned pnpm, installed into the private prefix
#   TINYTITAN_DSH_PLAYWRIGHT_VERSION  pinned Playwright, used by `smoke` only
#   TINYTITAN_DSH_PORT            browser UI port (default 7788)
#   TINYTITAN_DSH_DRY_RUN=1       print what would happen, change nothing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGIN_DIR="$REPO_ROOT/plugins/dsh-tinytitan"

# Where the installed models are. A checkout keeps them in `<repo>/models`; an
# install from the release tarball keeps them one level above the tools
# (`<root>/models`, beside `bin/` and `src/`). `dsh_route.sh` resolves the same
# way when the variable is unset, but this script always exports it, so the
# wrong value here wins over that fallback: on an installed copy the exported
# `<repo>/models` is `<root>/src/models`, which holds no models, and the
# plugin's boot-time route refresh then dies with "no catalog (it lists no
# installed models)". Measured in a simulated install on 2026-09-25.
if [[ -n "${TINYTITAN_MODELS_DIR:-}" ]]; then
  MODELS_DIR="$TINYTITAN_MODELS_DIR"
elif [[ -d "$REPO_ROOT/models" ]]; then
  MODELS_DIR="$REPO_ROOT/models"
elif [[ -d "$REPO_ROOT/../models" ]]; then
  MODELS_DIR="$(cd "$REPO_ROOT/.." && pwd)/models"
else
  MODELS_DIR="$REPO_ROOT/models"
fi

# Pinned on purpose; see the header. 0.2.0-rc.2 is the version the plugin was
# tested against — verified live, not assumed: `tools/dsh_local.sh ensure`
# completes against it, the plugin mounts on its web server, and `/dsh-lan/*`
# answers (with `dsh-llm:createUserMessage`, so prompts use upstream's own
# factory rather than the fallback).
#
# It is a **release candidate**, and npm knows that: `latest` and `next` both
# still point at an older tag, and this one sits under its own pre-release tag.
# The pin is exact, so that costs nothing here — but nothing gets it by accident
# either, which is the point of pinning.
DSH_VERSION="${TINYTITAN_DSH_VERSION:-0.2.0-rc.2}"
# The notice version the pinned harness gates its first-run modal on. Held here
# next to the pin it belongs to: they move together.
WELCOME_NOTICE_VERSION="2026-08-13.1"
NODE_VERSION="${TINYTITAN_DSH_NODE_VERSION:-26.8.2}"
PNPM_VERSION="${TINYTITAN_DSH_PNPM_VERSION:-12.4.2}"
# The same rule as the two pins above, and it applies to the smoke path too: this
# is the Playwright whose browser bundle drove the real page when the pin was set,
# and the headless Chromium that bundle downloads is chosen by it. A floating
# `playwright` here would make `smoke` test whatever upstream calls latest on the
# day someone runs it — which is AUD-156.
PLAYWRIGHT_VERSION="${TINYTITAN_DSH_PLAYWRIGHT_VERSION:-1.63.0}"
DSH_PORT="${TINYTITAN_DSH_PORT:-7788}"
# The served id the harness should open on. Empty means "the first one the route
# serves", which is what a bare `ensure` from the installer uses.
DEFAULT_MODEL_ID=""
# The route's default reasoning level, and it must match the level the server was
# started with. `medium` here against a server running `--reasoning off` is not a
# harmless mismatch: the harness reads the route and turns thinking ON, and a
# dense Qwen 3.5 that is asked to think fills whatever budget it is given with
# reasoning and never emits an answer. Measured on the 4B: thinking off gives
# `finish: stop` with content "42" in 3 tokens; thinking on gives `finish: length`
# with 64/64 reasoning tokens and empty content. At the route's maxTokens of
# 32768 that is the browser page sitting on "Deep diving..." for a quarter of an
# hour. `off` matches the launcher's own default.
REASONING="off"

DSH_ROOT="${TINYTITAN_DSH_ROOT:-$HOME/.tinytitan/dsh}"
DSH_HOME_DIR="$DSH_ROOT/home"
DSH_PREFIX="$DSH_ROOT/npm-prefix"
DSH_NODE_DIR="$DSH_ROOT/node"
DSH_STORE="$DSH_ROOT/store"
DSH_BIN_DIR="$DSH_ROOT/bin"
# Test-only dependencies and their browser download. Kept in the private root
# with everything else: the smoke test is a check, not something a user installs.
SMOKE_DIR="$DSH_ROOT/smoke"
SMOKE_BROWSERS="$DSH_ROOT/browsers"
VERSION_MARKER="$DSH_ROOT/.dsh-version"
# npm's cache, log directory and user config. `--prefix` moves where packages
# are *unpacked*, not where npm keeps its cache: measured on 2026-09-24, an
# `npm install --prefix <private>` still wrote `~/.npm/_cacache` and
# `~/.npm/_logs`, and read the user's `~/.npmrc` — exactly the writes an
# isolated bundle promises not to make. Everything the bundle installs now uses
# these; the user's npm setup is neither read nor written.
DSH_NPM_CACHE="$DSH_ROOT/npm-cache"
DSH_NPMRC="$DSH_ROOT/npmrc"
# Caches and state only. HOME, XDG_CONFIG_HOME and XDG_DATA_HOME are left as the
# user's on purpose: the agent works inside their repositories, so it needs
# their git identity and their `gh`/registry credentials to read. Isolation is
# about what the bundle *writes*, not about blinding the tools it drives.
DSH_XDG_CACHE="$DSH_ROOT/xdg-cache"
DSH_XDG_STATE="$DSH_ROOT/xdg-state"
# pnpm's own home. `--store-dir` moves the store, but pnpm still ensures
# `~/Library/pnpm` exists on macOS (measured 2026-09-24 in a factory-new HOME:
# an empty directory, but created in the user's home all the same). PNPM_HOME
# moves that too.
DSH_PNPM_HOME="$DSH_ROOT/pnpm-home"

DRY_RUN=0
[[ "${TINYTITAN_DSH_DRY_RUN:-0}" == "1" ]] && DRY_RUN=1

# Whether `web` lets DSH open the default browser. `--no-open` or
# TINYTITAN_DSH_NO_OPEN=1 turns it off, for an SSH session or a test that must
# not hijack whatever browser happens to be running.
DSH_OPEN=1
[[ "${TINYTITAN_DSH_NO_OPEN:-0}" == "1" ]] && DSH_OPEN=0

# The server port the route should point at: the launcher's default, overridden
# by TINYTITAN_PORT, and set by `ensure --port`.
SERVER_PORT="${TINYTITAN_PORT:-8080}"

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# run <description> <command…>: one place that honours --dry-run and prints the
# command, so a dry run is a readable transcript rather than a guess.
run() {
  local description="$1"; shift
  if (( DRY_RUN )); then
    printf '  would %s\n    $ ' "$description"
    printf '%q ' "$@"; echo
    return 0
  fi
  "$@"
}

# --- where things are -------------------------------------------------------

dsh_bin() { printf '%s' "$DSH_PREFIX/node_modules/.bin/dsh"; }
pnpm_bin() { printf '%s' "$DSH_BIN_DIR/pnpm"; }

# The Node we will use. An existing node+npm wins; otherwise ours, if it is
# already unpacked; otherwise nothing yet (install_node supplies it).
system_node() { command -v node 2>/dev/null; }
system_npm()  { command -v npm  2>/dev/null; }
private_node(){ [[ -x "$DSH_NODE_DIR/bin/node" ]] && printf '%s' "$DSH_NODE_DIR/bin/node"; }
private_npm() { [[ -x "$DSH_NODE_DIR/bin/npm" ]] && printf '%s' "$DSH_NODE_DIR/bin/npm"; }

node_bin() {
  local system; system="$(system_node || true)"
  if [[ -n "$system" ]]; then printf '%s' "$system"; return 0; fi
  private_node
}

npm_bin() {
  local system; system="$(system_npm || true)"
  if [[ -n "$system" ]]; then printf '%s' "$system"; return 0; fi
  private_npm
}

# `node`, `npm`, `pnpm` and `dsh` are all `#!/usr/bin/env node` scripts, so every
# one of them needs *the Node we chose* on PATH — not merely a path to it. This
# mattered the moment it was tested on a Mac with no Node: the private Node
# unpacked correctly and the very next command died with
# `env: node: No such file or directory`, because nothing had put its bin
# directory on PATH. Use `tool_path` for every invocation of a JS tool.
node_bin_dir() {
  local node; node="$(node_bin)"
  [[ -n "$node" ]] && dirname "$node"
}

tool_path() {
  local node_dir; node_dir="$(node_bin_dir || true)"
  printf '%s' "${node_dir:+$node_dir:}$DSH_BIN_DIR:$DSH_PREFIX/node_modules/.bin:$PATH"
}

# Create the private npm cache, config and XDG cache/state before anything
# invokes npm, pnpm or the harness. Idempotent, and cheap enough to call at each
# entry point that might write.
private_env() {
  # A dry run must not create anything, not even these: the plan is printed and
  # the filesystem is left exactly as it was.
  if (( DRY_RUN )); then
    echo "  would create the private npm, pnpm and XDG caches under $DSH_ROOT"
    return 0
  fi
  mkdir -p "$DSH_NPM_CACHE" "$DSH_XDG_CACHE" "$DSH_XDG_STATE" "$DSH_PNPM_HOME"
  if [[ ! -f "$DSH_NPMRC" ]]; then
    {
      echo "# TinyTitan's private npm config, written by tools/dsh_local.sh."
      echo "# The user's own ~/.npmrc is deliberately not read: a pinned install"
      echo "# must not depend on their registry, proxy or prefix settings."
    } > "$DSH_NPMRC"
  fi
}

# --- node -------------------------------------------------------------------

# Check the fetched Node tarball against the digest nodejs.org publishes for this
# exact version, and stop before anything is unpacked when the bytes cannot be
# proven. There is deliberately no fall-through, and the shape is the one
# `verify_release_artifact` in tools/install_tinytitan.sh gives the engine and
# tools archives (AUD-109): the file verified here is the runtime the harness
# then executes, so "could not verify" is a reason to stop, not a line to scroll
# past. Each branch names a different cause on purpose — a missing `shasum` is
# this machine, a missing or unmatched digest is the download — because the fix
# the user reaches for depends on which one it was.
verify_node_tarball() {
  local dir="$1" tarball="$2" shasums="$3"
  # `shasum` ships with macOS, so this is a guard rather than an expectation:
  # without it a missing tool would land in the mismatch branch and blame the
  # download for a problem it does not have.
  if ! command -v shasum >/dev/null 2>&1; then
    rm -rf "$dir"
    die "shasum is missing, so the Node download cannot be verified. It ships with
  macOS. If it is genuinely unavailable, install Node yourself and re-run: an
  existing node+npm is used in preference to fetching one."
  fi
  if [[ ! -s "$dir/$shasums" ]]; then
    rm -rf "$dir"
    die "nodejs.org published no checksums for Node $NODE_VERSION, so these bytes
  cannot be verified and nothing was installed. Check the connection, or name a
  version that carries them with TINYTITAN_DSH_NODE_VERSION."
  fi
  # Pin the check to the one line that names the file we are about to run.
  # SHASUMS256.txt lists every artifact of the release — headers, the source
  # tarball, the other platforms — and handing the whole file to `shasum -c`
  # would verify all of them and pass on whichever matched.
  local line
  line="$(awk -v want="$tarball" '$2 == want { print $1 "  " $2; exit }' "$dir/$shasums")"
  if [[ -z "$line" ]]; then
    rm -rf "$dir"
    die "The published checksums for Node $NODE_VERSION do not name $tarball, so
  nothing was installed. This Mac wants an arm64 darwin build of that version;
  nodejs.org did not publish one under that name."
  fi
  printf '%s\n' "$line" > "$dir/$tarball.sha256"
  if ! ( cd "$dir" && shasum -a 256 -c "$tarball.sha256" >/dev/null 2>&1 ); then
    rm -rf "$dir"
    die "The Node download does not match the checksum nodejs.org publishes for
  v${NODE_VERSION}, so nothing was installed. Try again; if it keeps failing the
  mirror is serving damaged bytes, and installing Node by another route is
  better than unpacking these."
  fi
  ok "Node $NODE_VERSION verified against nodejs.org's published digest"
}

# Fetch Node into our own root. Only reached when the Mac has no node at all,
# because the alternative — `brew install node` — writes a system-wide package
# for a feature the user may never turn on.
install_node() {
  if [[ -n "$(system_node || true)" && -n "$(system_npm || true)" ]]; then
    ok "Using the Node already on this Mac ($(node --version))"
    return 0
  fi
  if [[ -x "$DSH_NODE_DIR/bin/node" ]]; then
    ok "Using the private Node at $DSH_NODE_DIR ($("$DSH_NODE_DIR/bin/node" --version))"
    return 0
  fi

  [[ "$(uname -m)" == "arm64" ]] \
    || die "TinyTitan is Apple Silicon only; this Mac reports $(uname -m)."
  command -v curl >/dev/null 2>&1 || die "curl is required to fetch Node."

  local tarball="node-v${NODE_VERSION}-darwin-arm64.tar.gz"
  local url="https://nodejs.org/dist/v${NODE_VERSION}/${tarball}"
  local shasums="SHASUMS256.txt"
  say "Installing a private Node $NODE_VERSION (no Homebrew, nothing system-wide)"
  echo "  This is about 50 MB and lands only in $DSH_NODE_DIR."
  local tmp; tmp="$(mktemp -d)"

  run "download Node from nodejs.org" curl -fsSL "$url" -o "$tmp/$tarball"
  # `|| true` is deliberate: `set -e` would otherwise end the run on a 404 with
  # curl's own exit code, and the reason this install stopped is not a failed
  # transfer — it is that there are no published bytes to check against.
  # verify_node_tarball is the only place that decides whether to continue, and
  # it says so in those terms.
  run "fetch nodejs.org's published checksums for v${NODE_VERSION}" \
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/${shasums}" -o "$tmp/$shasums" || true
  if (( DRY_RUN )); then rm -rf "$tmp"; return 0; fi
  verify_node_tarball "$tmp" "$tarball" "$shasums"
  mkdir -p "$DSH_NODE_DIR"
  tar -xzf "$tmp/$tarball" -C "$DSH_NODE_DIR" --strip-components=1
  rm -rf "$tmp"
  [[ -x "$DSH_NODE_DIR/bin/node" ]] || die "the Node download did not unpack as expected."
  ok "Private Node ready at $DSH_NODE_DIR"
}

# --- the private install ----------------------------------------------------

install_dsh() {
  local installed=""
  [[ -f "$VERSION_MARKER" ]] && installed="$(cat "$VERSION_MARKER")"
  if [[ "$installed" == "$DSH_VERSION" && -x "$(dsh_bin)" ]]; then
    ok "DeepSeek Harness $DSH_VERSION already installed"
    return 0
  fi

  local npm; npm="$(npm_bin)"
  [[ -n "$npm" ]] || die "no npm available (neither the Mac's nor ours)."
  private_env

  say "Installing the pinned DeepSeek Harness $DSH_VERSION"
  echo "  Into $DSH_PREFIX. Your own dsh and ~/.dsh are not touched."
  run "install @deepseek-ai/dsh@$DSH_VERSION" \
    env PATH="$(tool_path)" \
    npm_config_cache="$DSH_NPM_CACHE" \
    npm_config_userconfig="$DSH_NPMRC" \
    "$npm" install --prefix "$DSH_PREFIX" --no-fund --no-audit \
    "@deepseek-ai/dsh@$DSH_VERSION"
  (( DRY_RUN )) && return 0
  [[ -x "$(dsh_bin)" ]] || die "the DeepSeek Harness install produced no dsh binary."
  mkdir -p "$DSH_ROOT"
  printf '%s\n' "$DSH_VERSION" > "$VERSION_MARKER"
  ok "DeepSeek Harness $DSH_VERSION"
}

# `dsh plugin` forwards to pnpm, and pnpm on macOS is a trap worth naming.
#
# The `pnpm` npm package installs a **shebang-less** wrapper whose own comments
# explain why: a bin shim generated from a shebang records the interpreter, and
# pnpm generates the shim before it puts the native binary in place. A shell, or
# glibc's execvp, retries such a file through `sh`; **Apple's libc does not**, and
# neither does Node's spawnSync without a shell — so `dsh plugin`, which spawns
# `pnpm` itself, dies with `spawnSync pnpm ENOEXEC` on macOS.
#
# The fix is to give the private install a shim we control, with a real shebang,
# that execs the native binary the package did unpack (`@pnpm/exe.darwin-arm64`).
# There is a fallback to the package's own `.mjs` entry through Node, so this
# still works on a machine where the native binary was not fetched.
#
# pnpm is installed into our own prefix rather than enabled through corepack,
# which would write shims into whichever Node install happens to be on PATH.
install_pnpm() {
  local shim; shim="$(pnpm_bin)"
  if [[ -x "$shim" ]] && "$shim" --version >/dev/null 2>&1; then
    ok "pnpm already present (private shim)"
    return 0
  fi

  local native="$DSH_PREFIX/node_modules/@pnpm/exe.darwin-arm64/pnpm"
  local mjs="$DSH_PREFIX/node_modules/pnpm/bin/pnpm.mjs"
  if [[ ! -f "$mjs" ]]; then
    local npm; npm="$(npm_bin)"
    private_env
    run "install pnpm@$PNPM_VERSION (private)" \
      env PATH="$(tool_path)" \
      npm_config_cache="$DSH_NPM_CACHE" \
      npm_config_userconfig="$DSH_NPMRC" \
      "$npm" install --prefix "$DSH_PREFIX" --no-fund --no-audit "pnpm@$PNPM_VERSION"
    (( DRY_RUN )) && return 0
  fi

  # The guard above sits inside the branch that has to fetch the package, so a
  # dry run only returns there when pnpm is missing. Reaching this line means the
  # package is unpacked and the shim is not working — the repair case — and the
  # mkdir and the heredoc below would write an executable into the private root
  # during a run that promised to change nothing.
  if (( DRY_RUN )); then
    echo "  would write the private pnpm shim at $shim"
    return 0
  fi

  mkdir -p "$DSH_BIN_DIR"
  if [[ -x "$native" ]]; then
    cat > "$shim" <<SHIM
#!/bin/sh
# TinyTitan's private pnpm, written by tools/dsh_local.sh.
# Execs the native binary directly: the npm package's own shim has no shebang,
# which macOS refuses to exec (ENOEXEC) when a program spawns it without a shell.
exec "$native" "\$@"
SHIM
  else
    local node; node="$(node_bin)"
    [[ -n "$node" ]] || die "no node to run pnpm with."
    cat > "$shim" <<SHIM
#!/bin/sh
# TinyTitan's private pnpm, written by tools/dsh_local.sh.
# The native binary was not installed, so hand over to the package's own entry.
exec "$node" "$mjs" "\$@"
SHIM
  fi
  chmod +x "$shim"
  ok "pnpm $("$shim" --version 2>/dev/null || echo "$PNPM_VERSION") via $shim"
}

# A fresh DSH_HOME has no settings.yaml: DSH creates profiles/ and storages/ but
# leaves the settings file to the user. `tools/dsh_route.sh --write` refuses to
# write into a file that is not there, so we create it — which is also the only
# sane default, because an empty settings file is exactly "no overrides".
ensure_home() {
  if [[ -d "$DSH_HOME_DIR/profiles/web" ]]; then
    ok "Private DSH home already initialised"
  else
    say "Initialising the private DSH home"
    run "initialise the web profile under $DSH_HOME_DIR" \
      env DSH_HOME="$DSH_HOME_DIR" PATH="$(tool_path)" \
      "$(dsh_bin)" --profile web --dump-config
    (( DRY_RUN )) || ok "Private DSH home at $DSH_HOME_DIR"
  fi

  if [[ -f "$DSH_HOME_DIR/settings.yaml" ]]; then
    ok "settings.yaml present"
  else
    (( DRY_RUN )) && { echo "  would create $DSH_HOME_DIR/settings.yaml"; return 0; }
    mkdir -p "$DSH_HOME_DIR"
    cat > "$DSH_HOME_DIR/settings.yaml" <<'YAML'
# TinyTitan's private DeepSeek Harness settings.
#
# This file belongs to the TinyTitan install under ~/.tinytitan, not to your own
# ~/.dsh. The llm-pi-ai route below is generated by tools/dsh_route.sh from the
# models installed in the checkout.
YAML
    ok "Created $DSH_HOME_DIR/settings.yaml"
  fi
}

# A fresh DSH home has **no workspace**, and the harness will not accept a prompt
# until one is chosen: the composer is disabled and reads "Choose a workspace to
# start". That is a dead first run for someone who was promised a window they can
# type in, so one is seeded — the checkout by default, since it exists and is the
# folder TinyTitan is installed in. The person can change or add workspaces in the
# UI; this only removes the empty state. Found by driving the real page.
seed_workspace() {
  local ws="${TINYTITAN_WORKSPACE:-$REPO_ROOT}"
  if (( DRY_RUN )); then
    echo "  would seed the workspace $ws"
    return 0
  fi
  DSH_WORKSPACE="$ws" python3 - "$DSH_HOME_DIR/storages/workspace.json" <<'PY'
import datetime, json, os, sys, uuid

path = sys.argv[1]
workspace = os.environ["DSH_WORKSPACE"]
os.makedirs(os.path.dirname(path), exist_ok=True)
try:
    data = json.load(open(path, encoding="utf-8"))
except Exception:
    data = {"unit": {"name": "workspace", "version": 2},
            "global": {"initialized": True, "workspaceIds": [], "archivedSessionIds": []},
            "tables": {"workspaces": {}}}
data.setdefault("global", {}).setdefault("workspaceIds", [])
data.setdefault("tables", {}).setdefault("workspaces", {})
known = data["tables"]["workspaces"]
if any(entry.get("path") == workspace for entry in known.values()):
    sys.exit(0)
identifier = str(uuid.uuid4())
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
known[identifier] = {"path": workspace, "title": os.path.basename(workspace) or workspace,
                     "sessionIds": [], "createdAt": now, "updatedAt": now}
data["global"]["workspaceIds"] = [identifier] + [i for i in data["global"]["workspaceIds"]]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
PY
  ok "Workspace seeded: $ws"
}

# The plugin is installed from the tools tree on purpose. It is deliberately
# **not** published to npm (decided 2026-09-25): the installer already downloads
# this project's source archive for the release tag, so the bundle arrives from
# the web with the tools and nobody needs a registry account — ours or the
# user's. There is no registry package to depend on and no second copy to keep in
# step. A `file:` install is a copy, so re-running `add` is how a plugin edit
# reaches the private home; for an installed copy, re-run `ensure`.
install_plugin() {
  [[ -d "$PLUGIN_DIR" ]] || die "plugin not found at $PLUGIN_DIR"
  private_env
  say "Installing the TinyTitan plugin into the private profile"
  # `--store-dir` is passed straight through to pnpm, and it is the only lever
  # that works: pnpm ignores `npm_config_store_dir` and `PNPM_STORE_DIR`, and a
  # `.npmrc` beside the profile did not move the store either. Without it the
  # ~36 MB native pnpm binary would land in the user's own ~/Library/pnpm store.
  run "add dsh-tinytitan from the checkout" \
    env DSH_HOME="$DSH_HOME_DIR" \
        PATH="$(tool_path)" \
        npm_config_cache="$DSH_NPM_CACHE" \
        npm_config_userconfig="$DSH_NPMRC" \
        XDG_CACHE_HOME="$DSH_XDG_CACHE" \
        XDG_STATE_HOME="$DSH_XDG_STATE" \
        PNPM_HOME="$DSH_PNPM_HOME" \
        "$(dsh_bin)" plugin --profile web add --store-dir "$DSH_STORE" "file:$PLUGIN_DIR"
  (( DRY_RUN )) && return 0
  [[ -e "$DSH_HOME_DIR/profiles/web/node_modules/dsh-tinytitan" ]] \
    || die "the plugin did not install into the private profile."
  ok "Plugin installed (pinned to this checkout)"
}

# The route is what points DSH at the TinyTitan server. It is generated from the
# installs under models/, so the model the user chose is the model DSH offers.
write_route() {
  say "Writing the TinyTitan route (port $SERVER_PORT)"
  if ! run "generate the llm-pi-ai route into the private settings.yaml" \
      "$REPO_ROOT/tools/dsh_route.sh" --write \
      --settings "$DSH_HOME_DIR/settings.yaml" --port "$SERVER_PORT" \
      --reasoning "$REASONING"; then
    warn "Could not write the route — is a model installed under models/?"
    warn "Install one, then re-run: tools/dsh_local.sh ensure"
    return 1
  fi
  (( DRY_RUN )) || ok "Route written to $DSH_HOME_DIR/settings.yaml"
}

# The settings file is ours (we create it), so a block is replaced with line
# surgery rather than a YAML round-trip: that keeps our comments and the generated
# route byte-for-byte, which a parse-and-dump would rewrite.
set_settings_block() {
  local key="$1"; shift
  BLOCK_KEY="$key" python3 - "$DSH_HOME_DIR/settings.yaml" "$@" <<'PY'
import os, sys

path, key = sys.argv[1], os.environ["BLOCK_KEY"]
body = list(sys.argv[2:])
try:
    lines = open(path, encoding="utf-8").read().splitlines()
except FileNotFoundError:
    lines = []

out, i = [], 0
while i < len(lines):
    if lines[i].startswith(key + ":"):
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if nxt.strip() == "" or nxt[:1] in (" ", "\t") or nxt.startswith("#"):
                i += 1
                continue
            break
        continue
    out.append(lines[i])
    i += 1
while out and out[-1].strip() == "":
    out.pop()
out += ["", key + ":"] + body
with open(path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(out) + "\n")
PY
}

# DeepSeek Harness ships `agent-default-model` pointing at **its own hosted
# route** — `provider: deepseek-official, model: deepseek-flash`. A fresh private
# install therefore opens on a provider we have no key for and fails with
# `MISSING_CREDENTIAL: llm-deepseek`, which is the exact opposite of a window
# that is ready to go. Writing a route is not enough; the default model has to be
# pointed at it. This was found by dry-testing the install, not by reading it.
write_default_model() {
  local provider="$1" model="$2"
  [[ -n "$model" ]] || return 0
  (( DRY_RUN )) && { echo "  would set agent-default-model to $provider/$model"; return 0; }
  set_settings_block agent-default-model \
    "  provider: $provider" \
    "  model: $model"
  ok "Harness default model: $provider/$model"
}

# The harness opens on a blocking "Internal Testing Notice" — DeepSeek's own
# notice that 0.1 is a developer preview. It is gated by a single version string,
# and until Continue is clicked the modal masks the page and nothing else is
# clickable: a person promised a prompt box meets a DeepSeek-branded banner
# first. Marking it seen is the whole fix, and it is one line in a file we own.
#
# The value is the one the **pinned** harness carries. A pin that moves to a
# release with a new notice version brings the notice back, which is visible
# rather than silent — the person sees it once and can click through.
suppress_welcome_notice() {
  (( DRY_RUN )) && { echo "  would mark the harness testing notice as seen"; return 0; }
  set_settings_block ui-onboarding "  welcomeNoticeVersion: $WELCOME_NOTICE_VERSION"
  ok "Testing notice marked as seen (no first-run modal)"
}

# The first served id in the route, skipping the `<id>-fast` chat alias: the
# default a bare `ensure` gets when the caller did not name one.
first_served_model() {
  python3 - "$DSH_HOME_DIR/settings.yaml" <<'PY'
import re, sys

try:
    text = open(sys.argv[1], encoding="utf-8").read()
except FileNotFoundError:
    sys.exit(0)
for match in re.finditer(r"^\s*-\s*id:\s*(\S+)\s*$", text, re.M):
    if not match.group(1).endswith("-fast"):
        print(match.group(1))
        break
PY
}

# --- the browser port -------------------------------------------------------

# 7788 is the default, but it is a port like any other: something else may hold
# it. Rather than fail, walk upward to the first free one. The one case worth
# special handling is a previous run of *this* install, which would otherwise
# leave a new UI on 7789, then 7790, one per launch; that one is stopped and its
# port reused. A port held by anything we cannot identify as ours is never
# touched -- the launcher's rule for the server port, applied here.

port_listening() { lsof -i :"$1" -sTCP:LISTEN >/dev/null 2>&1; }

# Is every listener on this port one of our own dsh processes? Our dsh runs from
# the private prefix, and node puts that path in the command line, so the prefix
# is a reliable fingerprint that does not depend on the port.
port_holders_are_ours() {
  local pid found=0
  for pid in $(lsof -ti :"$1" -sTCP:LISTEN 2>/dev/null); do
    found=1
    ps -p "$pid" -o command= 2>/dev/null | grep -qF -- "$DSH_PREFIX" || return 1
  done
  (( found ))
}

stop_ours_on_port() {
  local pid
  for pid in $(lsof -ti :"$1" -sTCP:LISTEN 2>/dev/null); do
    if ps -p "$pid" -o command= 2>/dev/null | grep -qF -- "$DSH_PREFIX"; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  local _; for _ in $(seq 1 50); do
    port_listening "$1" || break
    sleep 0.1
  done
}

# resolve_port <preferred> -> the port to use, on stdout.
resolve_port() {
  local base="$1" candidate
  for (( candidate = base; candidate <= base + 100 && candidate <= 65535; candidate++ )); do
    if ! port_listening "$candidate"; then
      (( candidate == base )) || printf '  \033[33m!\033[0m Port %s is taken; using %s instead.\n' "$base" "$candidate" >&2
      printf '%s' "$candidate"
      return 0
    fi
    if port_holders_are_ours "$candidate"; then
      printf '  \033[33m!\033[0m Stopping the DeepSeek Harness this install left on port %s.\n' "$candidate" >&2
      stop_ours_on_port "$candidate"
      if ! port_listening "$candidate"; then printf '%s' "$candidate"; return 0; fi
    fi
  done
  die "ports ${base}-$((base + 100)) are all in use; free one or pass --dsh-port."
}

# --- commands ---------------------------------------------------------------

cmd_paths() {
  printf '%-14s %s\n' root "$DSH_ROOT"
  printf '%-14s %s\n' home "$DSH_HOME_DIR"
  printf '%-14s %s\n' prefix "$DSH_PREFIX"
  printf '%-14s %s\n' node "$DSH_NODE_DIR"
  printf '%-14s %s\n' store "$DSH_STORE"
  printf '%-14s %s\n' dsh "$(dsh_bin)"
  printf '%-14s %s\n' pnpm "$(pnpm_bin)"
  printf '%-14s %s\n' port "$DSH_PORT"
  printf '%-14s %s\n' version "$DSH_VERSION"
  printf '%-14s %s\n' plugin "$PLUGIN_DIR"
  printf '%-14s %s\n' models "$MODELS_DIR"
}

cmd_status() {
  say "TinyTitan's private DeepSeek Harness"
  echo
  local node; node="$(node_bin || true)"
  if [[ -n "$node" ]]; then
    ok "node: $node ($("$node" --version))"
  else
    warn "node: none found (ensure will fetch one privately)"
  fi
  if [[ -x "$(dsh_bin)" ]]; then
    # The marker, not the binary's presence, is what says which release is
    # installed: a harness one release behind still has a working `dsh`, and
    # reporting it as pinned is how a stale private install went unnoticed.
    local installed=""
    [[ -f "$VERSION_MARKER" ]] && installed="$(cat "$VERSION_MARKER")"
    if [[ -z "$installed" ]]; then
      warn "dsh:  $(dsh_bin) has no version marker — run: tools/dsh_local.sh ensure"
    elif [[ "$installed" != "$DSH_VERSION" ]]; then
      warn "dsh:  $(dsh_bin) is $installed, not the pinned $DSH_VERSION — run: tools/dsh_local.sh ensure"
    else
      ok "dsh:  $(dsh_bin) ($DSH_VERSION)"
    fi
  else
    warn "dsh:  not installed — run: tools/dsh_local.sh ensure"
  fi
  if [[ -x "$(pnpm_bin)" ]]; then
    ok "pnpm: $(pnpm_bin)"
  else
    warn "pnpm: not installed (needed for the plugin)"
  fi
  if [[ -d "$DSH_HOME_DIR/profiles/web" ]]; then
    ok "home: $DSH_HOME_DIR"
  else
    warn "home: not initialised"
  fi
  if [[ -e "$DSH_HOME_DIR/profiles/web/node_modules/dsh-tinytitan" ]]; then
    ok "plugin: installed"
  else
    warn "plugin: not installed"
  fi
  # 0.2.0 imports a legacy settings.yaml into the active profile patch at boot
  # and renames it `.imported`, so the route is in one of two places now: the
  # file on a harness that has not booted since, the patch after one that has.
  if [[ -f "$DSH_HOME_DIR/settings.yaml" ]] \
     && grep -q '^llm-pi-ai:' "$DSH_HOME_DIR/settings.yaml"; then
    ok "route: written"
  elif [[ -f "$DSH_HOME_DIR/profiles/web/cordis.patch.yml" ]] \
     && grep -q 'llm-pi-ai' "$DSH_HOME_DIR/profiles/web/cordis.patch.yml" \
     && grep -q 'tinytitan:' "$DSH_HOME_DIR/profiles/web/cordis.patch.yml"; then
    ok "route: written (in the profile patch)"
  else
    warn "route: not written"
  fi
  echo
  echo "Your own ~/.dsh and any dsh on PATH are left alone:"
  if [[ -e "$HOME/.dsh" ]]; then
    echo "  $HOME/.dsh exists and is not used by this install."
  else
    echo "  no $HOME/.dsh on this Mac."
  fi
}

cmd_ensure() {
  if [[ ! -f "$REPO_ROOT/Package.swift" ]]; then
    warn "This does not look like a TinyTitan checkout; continuing anyway."
  fi
  say "Setting up TinyTitan's private DeepSeek Harness"
  echo "  Private root: $DSH_ROOT"
  echo "  Nothing outside it is touched; a dsh you already run is left alone."
  echo
  install_node
  install_dsh
  install_pnpm
  ensure_home
  seed_workspace
  install_plugin
  write_route
  local model="$DEFAULT_MODEL_ID"
  [[ -n "$model" ]] || model="$(first_served_model)"
  write_default_model "tinytitan" "$model"
  suppress_welcome_notice
  echo
  say "Done."
  echo "  Start it with:  $REPO_ROOT/tools/server_launcher.sh --web"
  echo "  Or from the launcher the installer wrote:  ~/.local/bin/tinytitan-web"
}

cmd_web() {
  [[ -x "$(dsh_bin)" ]] || die "DeepSeek Harness is not installed. Run: tools/dsh_local.sh ensure"
  [[ -d "$DSH_HOME_DIR/profiles/web" ]] || die "the private DSH home is not initialised. Run: tools/dsh_local.sh ensure"
  local port; port="$(resolve_port "$DSH_PORT")"
  local open=()
  (( DSH_OPEN )) || open=(--no-open)
  # TINYTITAN_PORT / TINYTITAN_REASONING / TINYTITAN_REPO / TINYTITAN_MODELS_DIR
  # are not decoration: the dsh-tinytitan plugin refreshes the route at **every**
  # boot. With TINYTITAN_PORT unset it regenerates the baseURL for the default
  # 8080; with TINYTITAN_REASONING unset it puts back `medium`, i.e. thinking on,
  # against a server started with thinking off. The first makes the page say
  # "Retrying model request"; the second makes it sit on "Deep diving..." while
  # the model spends 32768 tokens reasoning and never answers. Both were found by
  # driving the real page, not by reading the code: `ensure` writes the route
  # once, and this is what stops the boot-time refresh from undoing it.
  # DSH opens the default browser itself and prints the tokenised URL.
  # Caches go to the private root; config does not, because the agent works in
  # the user's repositories and needs their git identity and `gh`/registry
  # credentials *readable*. Writing is what isolation is about.
  private_env
  exec env DSH_HOME="$DSH_HOME_DIR" \
           TINYTITAN_PORT="$SERVER_PORT" \
           TINYTITAN_REASONING="$REASONING" \
           TINYTITAN_REPO="$REPO_ROOT" \
           TINYTITAN_MODELS_DIR="$MODELS_DIR" \
           XDG_CACHE_HOME="$DSH_XDG_CACHE" \
           XDG_STATE_HOME="$DSH_XDG_STATE" \
           PNPM_HOME="$DSH_PNPM_HOME" \
           npm_config_cache="$DSH_NPM_CACHE" \
           PATH="$(tool_path)" \
           "$(dsh_bin)" web --port "$port" "${open[@]+"${open[@]}"}" "$@"
}

# What `web` would bind, without starting anything. Prints only the number on
# stdout, so a caller can capture it.
cmd_port() { resolve_port "$DSH_PORT"; }

# --- smoke test -------------------------------------------------------------

# Playwright and its headless Chromium, into the private root. One-time, ~150 MB,
# and never exported to the repo or the system.
# What `smoke` has actually got, read from the package's own manifest. A leftover
# from before this was pinned would otherwise satisfy an existence check and get
# used unversioned forever — which is how the floating install stayed working.
playwright_installed() {
  if [[ -f "$SMOKE_DIR/node_modules/playwright/package.json" ]]; then
    sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' \
      "$SMOKE_DIR/node_modules/playwright/package.json" | head -1
  fi
}

ensure_smoke_deps() {
  local installed; installed="$(playwright_installed)"
  if [[ "$installed" != "$PLAYWRIGHT_VERSION" ]]; then
    say "Installing the pinned Playwright $PLAYWRIGHT_VERSION${installed:+ (found $installed)}"
    private_env
    run "install playwright@$PLAYWRIGHT_VERSION" env PATH="$(tool_path)" \
      npm_config_cache="$DSH_NPM_CACHE" \
      npm_config_userconfig="$DSH_NPMRC" \
      "$(npm_bin)" install --prefix "$SMOKE_DIR" --no-fund --no-audit \
      "playwright@$PLAYWRIGHT_VERSION"
    (( DRY_RUN )) || ok "Playwright $(playwright_installed)"
  fi
  local browsers; browsers="$(compgen -G "$SMOKE_BROWSERS/chromium*" | head -1 || true)"
  if [[ -z "$browsers" ]]; then
    say "Downloading the headless browser (one time, about 150 MB)"
    run "download headless chromium" env PLAYWRIGHT_BROWSERS_PATH="$SMOKE_BROWSERS" \
      "$SMOKE_DIR/node_modules/.bin/playwright" install chromium --only-shell
    (( DRY_RUN )) || ok "Headless browser for Playwright $PLAYWRIGHT_VERSION"
  elif (( ! DRY_RUN )); then
    ok "Headless browser already at $(basename "$browsers"), which Playwright $PLAYWRIGHT_VERSION chose"
  fi
}

# Drive the real page: start our harness against an already-running server, load
# the tokenised URL in headless Chromium, type a prompt, and wait for the answer.
# Everything the page needs that a curl cannot see -- the composer appearing, the
# send button enabling, the route's model and reasoning level -- is exercised.
#
# The harness's pid lives in a script-level variable on purpose: a trap that reads
# a `local` fires after the function has returned, and under `set -u` it then dies
# with "web_pid: unbound variable" instead of cleaning up.
SMOKE_WEB_PID=""
SMOKE_WEB_LOG=""

smoke_cleanup() {
  if [[ -n "$SMOKE_WEB_PID" ]]; then
    kill "$SMOKE_WEB_PID" 2>/dev/null || true
    SMOKE_WEB_PID=""
  fi
  if [[ -n "$SMOKE_WEB_LOG" ]]; then
    rm -f "$SMOKE_WEB_LOG"
    SMOKE_WEB_LOG=""
  fi
}

cmd_smoke() {
  [[ -x "$(dsh_bin)" ]] || die "DeepSeek Harness is not installed. Run: tools/dsh_local.sh ensure"
  if ! curl -s --max-time 3 "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; then
    die "no TinyTitan server on port $SERVER_PORT.
     Start one first, in another window:
       $REPO_ROOT/tools/server_launcher.sh --client server --port $SERVER_PORT"
  fi
  ensure_smoke_deps

  local port; port="$(resolve_port "$DSH_PORT")"
  SMOKE_WEB_LOG="$(mktemp)"
  say "Starting the harness on port $port for the test"
  # Same environment `web` uses, so the plugin's boot-time route refresh keeps
  # this server's address and reasoning level (see the note in cmd_web).
  env DSH_HOME="$DSH_HOME_DIR" \
      TINYTITAN_PORT="$SERVER_PORT" \
      TINYTITAN_REASONING="$REASONING" \
      TINYTITAN_REPO="$REPO_ROOT" \
      TINYTITAN_MODELS_DIR="$MODELS_DIR" \
      PATH="$(tool_path)" \
      "$(dsh_bin)" web --no-open --port "$port" >"$SMOKE_WEB_LOG" 2>&1 &
  SMOKE_WEB_PID=$!
  trap smoke_cleanup EXIT INT TERM

  local url="" _ status=0
  for _ in $(seq 1 60); do
    # `|| true` is load-bearing: a grep that matches nothing exits 1, and an
    # assignment takes its command substitution's status, so under `set -e` the
    # whole run died silently on the first second before the URL was printed.
    url="$(grep -oE 'http://127\.0\.0\.1:[0-9]+/\?token=[A-Za-z0-9_-]+' "$SMOKE_WEB_LOG" 2>/dev/null | tail -1 || true)"
    [[ -n "$url" ]] && break
    kill -0 "$SMOKE_WEB_PID" 2>/dev/null || break
    sleep 1
  done
  if [[ -z "$url" ]]; then
    warn "the harness did not start:"
    sed 's/^/  /' "$SMOKE_WEB_LOG" >&2
    smoke_cleanup
    trap - EXIT INT TERM
    return 1
  fi

  # ESM resolves `playwright` by walking up from the importing file, and
  # NODE_PATH does not apply to it -- so the script runs from a copy inside the
  # directory that has the dependency. A copy rather than a symlink because Node
  # resolves a symlink to its real path, which puts it back in the repo where
  # `playwright` is not. The copy is refreshed every run, so it cannot go stale.
  cp -f "$REPO_ROOT/tools/dsh_smoke.mjs" "$SMOKE_DIR/dsh_smoke.mjs"
  env PLAYWRIGHT_BROWSERS_PATH="$SMOKE_BROWSERS" PATH="$(tool_path)" \
    node "$SMOKE_DIR/dsh_smoke.mjs" --url "$url" "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}" || status=$?
  smoke_cleanup
  trap - EXIT INT TERM
  return $status
}

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; }

COMMAND="${1:-}"
[[ $# -gt 0 ]] && shift
# Everything after `--` goes to `smoke`'s test harness (`--expect`, `--prompt`,
# `--timeout`), which this script does not interpret.
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) SERVER_PORT="${2:?--port needs a number}"; shift 2 ;;
    --dsh-port) DSH_PORT="${2:?--dsh-port needs a number}"; shift 2 ;;
    --model) DEFAULT_MODEL_ID="${2:?--model needs a served id}"; shift 2 ;;
    --reasoning) REASONING="${2:?--reasoning needs a level}"; shift 2 ;;
    --no-open) DSH_OPEN=0; shift ;;
    --) shift; PASSTHROUGH=("$@"); break ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

case "$COMMAND" in
  ensure) cmd_ensure ;;
  web)    cmd_web "$@" ;;
  smoke)  cmd_smoke ;;
  port)   cmd_port ;;
  status) cmd_status ;;
  paths)  cmd_paths ;;
  ""|--help|-h) usage ;;
  *) die "unknown command: $COMMAND (try --help)" ;;
esac
