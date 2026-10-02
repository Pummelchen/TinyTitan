#!/usr/bin/env bash
# Install TinyTitan for someone who has never used a terminal.
#
# Two ways to run it:
#
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/Pummelchen/TinyTitan/main/tools/install_tinytitan.sh)"
#   bash tools/install_tinytitan.sh                 # from a clone or an unzipped download
#
# Use those forms rather than `curl ... | bash`. A pipe makes this script's stdin
# the pipe, so it cannot ask anything and silently takes the default at every
# step — including "no model" and "no browser window". `bash -c "$(curl ...)"`
# downloads the script first and runs it with the terminal still on stdin, so the
# questions work. `bash tools/install_tinytitan.sh` never needs `chmod +x`
# either, which matters because a zip that lost the executable bit still runs.
#
# Nothing here needs Homebrew, Python, Node, git or Xcode. The engine is
# downloaded built, from the newest published release; the tools that drive it
# arrive as text; Node is fetched privately only if this Mac has none. The model
# is the one thing a person chooses, and the installer asks for it with a list.
#
# It checks the Mac, installs TinyTitan under ~/.tinytitan, asks which model to
# download, and installs a `tinytitan` command that starts the server — then
# offers to start it, so you finish with a base URL a client can be pointed at.
# Everything it creates is under ~/.tinytitan and ~/.local/bin, so removing those
# two directories removes the install.
#
# It can also set up a **chat window**: TinyTitan's own DeepSeek Harness, a local
# page with a prompt box that opens in the default browser, already pointed at
# the server. That is offered once a model is installed, because a window with
# nothing to load is not a working thing. It lives under ~/.tinytitan too, and is
# kept entirely separate from any DeepSeek Harness you run yourself; see
# tools/dsh_local.sh.
#
# Nothing here is destructive. It never deletes a model and never touches
# anything outside its own folders:
#   ~/.tinytitan                the engine, the tools, the models, the chat window
#   ~/.local/bin/tinytitan      the command that starts the server
#   ~/.local/bin/tinytitan-web  the command that starts it with the browser window
#   ~/TinyTitan                 only with --from-source: the clone it builds in
# Re-running it is safe: it replaces the engine and tools in place and keeps the
# models.
#
# Flags:
#   --yes, -y        answer yes to every question (unattended install)
#   --model NAME     install this model without asking. Omit it and, with a
#                    terminal, the installer shows the model list to choose from
#                    (`tools/install_models.sh --choose`); through a pipe it
#                    takes ornith15-8bit rather than hanging on a question.
#   --no-model       install no model
#   --web            also set up the browser chat window (asked interactively;
#                    --yes alone does not install it, so an unattended run stays
#                    a server and nothing else)
#   --no-web         do not offer the browser chat window
#   --version TAG    install that release instead of the newest (e.g. --version v5.7)
#   --from-source    clone and `swift build` instead of downloading a release.
#                    For contributors; needs Xcode and takes much longer.
#   --dir PATH       where --from-source clones to (default ~/TinyTitan)
#   --help, -h       this text
set -euo pipefail

REPO_URL="https://github.com/Pummelchen/TinyTitan.git"
REPO_URL_RAW="https://raw.githubusercontent.com/Pummelchen/TinyTitan/main"
DEFAULT_MODEL="ornith15-8bit"
DEFAULT_DIR="$HOME/TinyTitan"

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Ask a yes/no question. With --yes, or when nothing can answer, take the
# fallback rather than hanging a piped install.
ASSUME_YES=0
ask() {
  local prompt="$1" fallback="${2:-no}" reply
  if (( ASSUME_YES )); then
    [[ "$fallback" == "yes" ]]
    return
  fi
  if [[ ! -t 0 ]]; then
    [[ "$fallback" == "yes" ]]
    return
  fi
  printf '%s [%s] ' "$prompt" "$([[ "$fallback" == yes ]] && echo 'Y/n' || echo 'y/N')"
  read -r reply || reply=""
  reply="${reply:-$fallback}"
  [[ "$reply" =~ ^[Yy] ]]
}

# --- flags -----------------------------------------------------------------
MODEL="$DEFAULT_MODEL"
# Set by --model. It decides whether the model step shows the menu (nothing was
# named, so ask) or installs the one that was named (do not second-guess it).
MODEL_WAS_SET=0
INSTALL_MODEL=1
TARGET_DIR="$DEFAULT_DIR"
# ask | yes | no. `--yes` deliberately does not imply `yes` here: pulling ~320 MB
# of Node/DeepSeek Harness into someone's home on an unattended run is a side
# effect that has to be asked for by name, with --web.
WEB_MODE="ask"
# Empty means "the newest release", which is what a person wants and what the
# release runbook publishes. `--version v5.7` pins one; a contributor who wants
# `main` builds from source instead.
RELEASE_TAG=""
FROM_SOURCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y)    ASSUME_YES=1 ;;
    --model)     MODEL="${2:?--model needs a name}"; MODEL_WAS_SET=1; shift ;;
    --no-model)  INSTALL_MODEL=0 ;;
    --web)       WEB_MODE="yes" ;;
    --no-web)    WEB_MODE="no" ;;
    --version)   RELEASE_TAG="${2:?--version needs a tag like v5.7}"; shift ;;
    --from-source) FROM_SOURCE=1 ;;
    --dir)       TARGET_DIR="${2:?--dir needs a path}"; shift ;;
    --help|-h)   sed -n '2,/^set -euo pipefail/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; exit 0 ;;
    *)           die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

# --- where everything goes --------------------------------------------------
#
# One private root, and nothing outside it. This is what makes the install
# "uninstall by deleting a directory", and it is why the engine can arrive as a
# release tarball instead of a build: `.build/release` only exists in a checkout,
# so an installed copy keeps its binaries in `$INSTALL_ROOT/bin` and tells the
# tools where with TINYTITAN_BIN_DIR.
INSTALL_ROOT="${TINYTITAN_ROOT:-$HOME/.tinytitan}"
BIN_PATH="$INSTALL_ROOT/bin"
SRC_PATH="$INSTALL_ROOT/src"
MODELS_PATH="$INSTALL_ROOT/models"

# Steps depend on the path: a release install has no toolchain step and no build.
STEP=0
TOTAL=4
(( FROM_SOURCE )) && TOTAL=5
step() { STEP=$((STEP + 1)); say "$STEP/$TOTAL  $*"; }

say "TinyTitan installer"
echo "  This takes a while and mostly waits. You can stop it with Ctrl-C at any"
echo "  point; run it again later and it continues where it can."
echo

# A pipe takes the default at every question, and a person who expected to choose
# would otherwise never learn why nothing asked them. Say what is about to happen
# and how to get the questions, before the 37 GB download rather than after it.
if [[ ! -t 0 ]] && (( ASSUME_YES == 0 )); then
  warn "This is running from a pipe, so it cannot ask you anything."
  warn "It will take the default at every step — including downloading $MODEL"
  warn "and setting up the browser window."
  echo "     To choose instead, run:"
  echo "       bash -c \"\$(curl -fsSL $REPO_URL_RAW/tools/install_tinytitan.sh)\""
  echo "     Flags still work through a pipe:"
  echo "       curl -fsSL $REPO_URL_RAW/tools/install_tinytitan.sh | bash -s -- --no-model"
  echo
fi

# --- 1) the machine ---------------------------------------------------------
step "Checking this Mac"

os_version="$(sw_vers -productVersion 2>/dev/null || echo 0)"
os_major="${os_version%%.*}"
arch="$(uname -m)"
if [[ "$arch" != "arm64" ]]; then
  die "TinyTitan needs Apple Silicon (M1 or newer). This Mac reports $arch."
fi
ok "Apple Silicon ($arch), macOS $os_version"
if [[ "${os_major:-0}" -lt 26 ]]; then
  warn "TinyTitan targets macOS 26 or later; this is $os_version."
  warn "The engine may refuse to start. If it does, updating macOS fixes it."
fi

free_kb="$(df -Pk "$HOME" | awk 'NR==2 {print $4}')"
free_gb=$(( free_kb / 1048576 ))
if [[ "$free_gb" -lt 45 ]]; then
  warn "Only about ${free_gb} GB free on this volume."
  warn "The build needs a few GB, and a 35B 4-bit model about 20 GB."
  ask "Continue anyway?" no || die "Stopped at your request. Free up space and re-run."
else
  ok "About ${free_gb} GB free"
fi

if [[ "$(pgrep -fl 'TinyTitanServer|TinyTitanCLI' 2>/dev/null | wc -l | tr -d ' ')" != "0" ]]; then
  warn "A TinyTitan process is already running. Stop it before starting a server,"
  warn "since one model runs at a time on this Mac."
fi

# --- 2) TinyTitan itself ----------------------------------------------------
step "Installing TinyTitan"

# The release that carries the engine binaries. Empty means the newest one, which
# is what a person wants and what the release runbook publishes.
latest_tag() {
  curl -fsSL "https://api.github.com/repos/Pummelchen/TinyTitan/releases/latest" 2>/dev/null \
    | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1
}

# Download the published arm64 binaries and the matching source, and put them
# under $INSTALL_ROOT. No git, no Xcode, no Swift, no brew, no Python: the engine
# arrives built, and the tools that drive it arrive as text.
install_from_release() {
  local tag="$1" tmp asset src_url
  [[ "$tag" == v* ]] || tag="v$tag"
  asset="tinytitan-${tag#v}-macos-arm64.tar.gz"
  tmp="$(mktemp -d)"

  say "Downloading TinyTitan ${tag#v} (about 25 MB)"
  if ! curl -fL --progress-bar -o "$tmp/$asset" \
      "https://github.com/Pummelchen/TinyTitan/releases/download/$tag/$asset"; then
    rm -rf "$tmp"
    die "Could not download $tag. Check your connection, or that the release exists."
  fi
  if curl -fsSL -o "$tmp/$asset.sha256" \
      "https://github.com/Pummelchen/TinyTitan/releases/download/$tag/$asset.sha256"; then
    # `shasum` ships with macOS, so this is a guard rather than an expectation —
    # without it a missing tool would fall through to the mismatch branch and
    # blame the download for a problem it does not have.
    if ! command -v shasum >/dev/null 2>&1; then
      warn "shasum is missing, so the download could not be verified."
    elif ( cd "$tmp" && shasum -a 256 -c "$asset.sha256" >/dev/null 2>&1 ); then
      ok "Checksum verified"
    else
      rm -rf "$tmp"
      die "The download does not match its published checksum. Try again."
    fi
  else
    warn "No checksum published for $tag; continuing without verification."
  fi

  rm -rf "$BIN_PATH"
  mkdir -p "$BIN_PATH"
  tar -xzf "$tmp/$asset" -C "$BIN_PATH" --strip-components=1 || {
    rm -rf "$tmp"; die "Could not unpack the download."; }
  [[ -x "$BIN_PATH/TinyTitanServer" ]] || {
    rm -rf "$tmp"; die "The download did not contain TinyTitanServer."; }
  # Downloaded through curl rather than a browser, so there is normally no
  # quarantine attribute; clearing it anyway costs nothing and covers the case
  # where these files were moved here from a browser download.
  xattr -dr com.apple.quarantine "$BIN_PATH" 2>/dev/null || true
  ok "Engine in $BIN_PATH"

  # The tools, the DSH plugin and the docs, from the same tag. Small, and it is
  # what makes the layout identical to a checkout apart from the build.
  say "Downloading the matching tools"
  src_url="https://github.com/Pummelchen/TinyTitan/archive/refs/tags/$tag.tar.gz"
  if ! curl -fL --progress-bar -o "$tmp/src.tar.gz" "$src_url"; then
    rm -rf "$tmp"; die "Could not download the tools for $tag."
  fi
  rm -rf "$SRC_PATH"
  mkdir -p "$SRC_PATH"
  tar -xzf "$tmp/src.tar.gz" -C "$SRC_PATH" --strip-components=1 || {
    rm -rf "$tmp"; die "Could not unpack the tools."; }
  rm -rf "$tmp"
  [[ -f "$SRC_PATH/tools/server_launcher.sh" ]] || die "The tools are incomplete."
  ok "Tools in $SRC_PATH"
}

# The contributor path: a checkout, a toolchain and a build. Unchanged apart from
# being opt-in, because it is the slow one and needs Xcode.
install_from_source() {
  find_checkout() {
    local start="$1" probe
    probe="$start"
    while [[ -n "$probe" && "$probe" != "/" ]]; do
      if [[ -f "$probe/Package.swift" ]]; then printf '%s' "$probe"; return 0; fi
      probe="$(dirname "$probe")"
    done
    return 1
  }

  REPO_ROOT=""
  if ! REPO_ROOT="$(find_checkout "$(cd "$(dirname "$0")" && pwd)")"; then
    REPO_ROOT="$(find_checkout "$PWD" || true)"
  fi
  if [[ -n "$REPO_ROOT" ]]; then
    ok "Using the checkout at $REPO_ROOT"
  else
    command -v git >/dev/null 2>&1 \
      || die "git is missing. Install Xcode (App Store), open it once, then re-run."
    if [[ -d "$TARGET_DIR/.git" ]]; then
      ok "Updating the existing checkout at $TARGET_DIR"
      git -C "$TARGET_DIR" pull --ff-only || warn "Could not update; using what is there."
      REPO_ROOT="$TARGET_DIR"
    else
      [[ -e "$TARGET_DIR" ]] && die "$TARGET_DIR already exists and is not a checkout. Move it or use --dir."
      echo "  Downloading TinyTitan into $TARGET_DIR ..."
      git clone --depth 1 "$REPO_URL" "$TARGET_DIR" || die "Could not download TinyTitan. Check your connection."
      REPO_ROOT="$TARGET_DIR"
    fi
    ok "Source ready"
  fi
  cd "$REPO_ROOT"

  say "Checking the Swift toolchain"
  if ! command -v swift >/dev/null 2>&1; then
    warn "Swift is not installed yet."
    echo "  Pressing Enter opens the installer for Apple's command-line tools."
    if ask "Install them now?" yes; then
      xcode-select --install 2>/dev/null || true
      echo
      echo "  A dialog should appear. Accept it, wait for it to finish (it can take"
      echo "  several minutes), then run this installer again."
    fi
    exit 1
  fi
  swift_line="$(swift --version 2>&1 | head -1)"
  swift_ver="$(printf '%s' "$swift_line" | grep -oE 'Swift version [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
  if [[ -z "$swift_ver" ]]; then
    warn "Could not read a Swift version from: $swift_line"
    ask "Try the build anyway?" yes || exit 1
  elif (( $(printf '%s' "$swift_ver" | cut -d. -f1) < 6 )) \
    || { [[ "$(printf '%s' "$swift_ver" | cut -d. -f1)" == "6" ]] \
         && (( $(printf '%s' "$swift_ver" | cut -d. -f2) < 4 )); }; then
    die "TinyTitan needs Swift 6.4 (Xcode 27), the only supported toolchain;
     this Mac has $swift_ver.
     Update Xcode from the App Store (or set it with xcode-select), then re-run."
  elif [[ "$swift_ver" != "6.4" ]]; then
    # Supported is exactly Xcode 27 / Swift 6.4. A newer Swift may well build
    # this tree, but nothing here is measured on it, so it is not supported and
    # the user is told that before spending half an hour on a build.
    warn "The supported toolchain is Xcode 27 / Swift 6.4; this Mac has Swift $swift_ver."
    ask "Build anyway, knowing it is unsupported?" yes || exit 1
  else
    ok "Swift $swift_ver"
  fi

  say "Building (this is the slow part)"
  if ! swift build -c release; then
    die "The build failed. The last few lines explain why.
     Copy the whole message to https://github.com/Pummelchen/TinyTitan/issues and someone will help."
  fi
  ok "Build complete"

  TOOLS_PATH="$REPO_ROOT/tools"
  BIN_PATH_FINAL="$REPO_ROOT/.build/release"
  MODELS_PATH_FINAL="$REPO_ROOT/models"
}

if (( FROM_SOURCE )); then
  install_from_source
else
  if [[ -z "$RELEASE_TAG" ]]; then
    RELEASE_TAG="$(latest_tag)"
    [[ -n "$RELEASE_TAG" ]] \
      || die "Could not reach GitHub to find the newest release. Check your connection,
     or name one yourself with --version v5.7, or build from a clone with --from-source."
  fi
  install_from_release "$RELEASE_TAG"
  TOOLS_PATH="$SRC_PATH/tools"
  BIN_PATH_FINAL="$BIN_PATH"
  MODELS_PATH_FINAL="$MODELS_PATH"
fi

# Exported so every tool this installer runs — the model installer, the launcher,
# the route writer, the DSH setup — finds the binaries and the models without
# anyone having to pass them down. A checkout's defaults still apply when these
# are unset, so this changes nothing for `swift build` users.
export TINYTITAN_BIN_DIR="$BIN_PATH_FINAL"
export TINYTITAN_MODELS_DIR="$MODELS_PATH_FINAL"

# --- 3) a model -------------------------------------------------------------
step "Model"

installed_any() {
  local d
  for d in "$MODELS_PATH_FINAL"/*/manifest.json; do
    [[ -f "$d" ]] && return 0
  done
  return 1
}

if (( ! INSTALL_MODEL )); then
  ok "Skipped (--no-model)"
elif installed_any; then
  ok "A model is already installed under $MODELS_PATH_FINAL"
  echo "     Install another any time:  $TOOLS_PATH/install_models.sh --choose"
else
  # The model is the only real choice in this install, so it gets a list rather
  # than a yes/no on one default: what each model is, how much disk it takes, and
  # the verified default on the first line for someone who just presses Enter.
  # It lives in `install_models.sh --choose` so the list, the labels and the
  # sizes come from the one catalogue.
  if (( MODEL_WAS_SET )) || [[ ! -t 0 ]]; then
    echo "  No model is installed yet. TinyTitan needs one to run."
    if [[ "$MODEL" == "$DEFAULT_MODEL" ]]; then
      echo "  The recommended starting model is Ornith 1.5 35B-A3B at 8-bit,"
      echo "  about 37 GB installed. The 4-bit version is about 20 GB and faster"
      echo "  to download if that is a lot:  $TOOLS_PATH/install_models.sh ornith15"
    else
      echo "  This run was asked for '$MODEL'."
    fi
    if ask "Download $MODEL now?" yes; then
      if ! "$TOOLS_PATH/install_models.sh" "$MODEL"; then
        warn "The model download did not finish."
        echo "     Re-run this installer to continue, or start it directly:"
        echo "       $TOOLS_PATH/install_models.sh $MODEL"
      else
        ok "Model installed"
      fi
    else
      echo "  Fine — TinyTitan is installed but will have nothing to load until you run:"
      echo "       $TOOLS_PATH/install_models.sh --choose"
    fi
  elif ! "$TOOLS_PATH/install_models.sh" --choose; then
    warn "No model was installed."
    echo "     Pick one whenever you like:  $TOOLS_PATH/install_models.sh --choose"
  else
    ok "Model installed"
  fi
fi

# --- 4) a server you can start ----------------------------------------------
step "Your server"

# A command for the terminal-minded: it starts the server from these tools, and
# only ever stops a server that launcher started. The three exports are what let
# one launcher work in both layouts — a checkout's `.build/release` or an
# installed copy's `~/.tinytitan/bin`.
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/tinytitan" <<RUNNER
#!/bin/sh
# Installed by tools/install_tinytitan.sh. Starts the TinyTitan server.
export TINYTITAN_BIN_DIR="$BIN_PATH_FINAL"
export TINYTITAN_MODELS_DIR="$MODELS_PATH_FINAL"
exec "$TOOLS_PATH/server_launcher.sh" "\$@"
RUNNER
chmod +x "$HOME/.local/bin/tinytitan"
ok "Start it with: ~/.local/bin/tinytitan"

case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) echo "     Add it to your PATH to use the 'tinytitan' command anywhere:"
     echo "       echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc" ;;
esac

if ! installed_any; then
  echo
  warn "No model is installed yet, so a server would have nothing to load."
  echo "     Install one, then start the server:"
  echo "       $TOOLS_PATH/install_models.sh $MODEL"
  echo "       ~/.local/bin/tinytitan"
else
  # Staging is cleaned as widths complete; what is left is a converted snapshot
  # a second width would reuse, which is worth its disk only if that width is
  # still wanted. One command reclaims the rest, and `status` shows the size.
  echo
  echo "     Conversion staging is removed automatically once both widths of a"
  echo "     model are installed. Reclaim the rest whenever you like:"
  echo "       $TOOLS_PATH/install_models.sh clean"
fi

# The chat window. Offered only once a model is on disk: the whole point is a
# page that opens onto a model that is already there, and setting up 320 MB of
# runtime for an empty picker helps nobody.
WANT_WEB=0
if ! installed_any; then
  :
elif [[ "$WEB_MODE" == "no" ]]; then
  ok "Browser chat window: skipped (--no-web)"
elif [[ "$WEB_MODE" == "yes" ]] \
  || { (( ASSUME_YES == 0 )) && [[ -t 0 ]] \
       && ask "Also set up a chat window in your browser?" yes; }; then
  echo
  if "$TOOLS_PATH/dsh_local.sh" ensure; then
    WANT_WEB=1
    cat > "$HOME/.local/bin/tinytitan-web" <<WEBRUNNER
#!/bin/sh
# Installed by tools/install_tinytitan.sh. Starts the server and opens
# TinyTitan's own DeepSeek Harness in the browser.
export TINYTITAN_BIN_DIR="$BIN_PATH_FINAL"
export TINYTITAN_MODELS_DIR="$MODELS_PATH_FINAL"
exec "$TOOLS_PATH/server_launcher.sh" --web "\$@"
WEBRUNNER
    chmod +x "$HOME/.local/bin/tinytitan-web"
    ok "Chat window ready: ~/.local/bin/tinytitan-web"
  else
    warn "The chat window could not be set up; it does not affect the server."
    warn "Retry any time with: $TOOLS_PATH/dsh_local.sh ensure"
  fi
fi

# --- done -------------------------------------------------------------------
echo
# Hand over to the launcher whenever a person is there to watch: the launcher
# prints the base URL and the client settings, and the health check it does
# first is the proof that what was just built actually serves. An unattended
# install prints the command instead, so a pipe never blocks on a server.
if installed_any && (( ASSUME_YES == 0 )) && [[ -t 0 && -t 1 ]]; then
  if ask "Start the TinyTitan server now?" yes; then
    if (( WANT_WEB )); then
      say "Starting the model and opening the chat window in your browser."
    else
      say "Starting the server. Leave this window open; Ctrl-C stops it."
    fi
    echo
    if (( WANT_WEB )); then
      exec "$TOOLS_PATH/server_launcher.sh" --web
    fi
    exec "$TOOLS_PATH/server_launcher.sh" --client server
  fi
fi

say "Done."
echo
if (( WANT_WEB )); then
  echo "  Start TinyTitan with the browser chat window:"
  echo "    ~/.local/bin/tinytitan-web"
  echo "    (or: $TOOLS_PATH/server_launcher.sh --web)"
  echo
  echo "  A page opens in your browser with a prompt box, already pointed at"
  echo "  the model. Keep the window open while you use it; Ctrl-C stops both."
  echo "  The server alone is still:  ~/.local/bin/tinytitan"
else
  echo "  Start the TinyTitan server:"
  echo "    ~/.local/bin/tinytitan"
  echo "    (or: $TOOLS_PATH/server_launcher.sh)"
  echo
  echo "  It prints the base URL to point a client at - by default"
  echo "  http://127.0.0.1:8080/v1 with any API key; --port changes the port."
  echo "  Keep the window open while you use it; one model runs at a time."
  echo
  echo "  For a chat window in the browser instead, re-run this installer with"
  echo "  --web, or run: $TOOLS_PATH/dsh_local.sh ensure"
fi
echo
echo "  New to this? Start here:"
echo "    https://github.com/Pummelchen/TinyTitan/wiki/Getting-Started"
echo "  Questions and bug reports: https://github.com/Pummelchen/TinyTitan/issues"
