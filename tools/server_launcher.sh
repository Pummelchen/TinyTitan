#!/usr/bin/env bash
# The one TinyTitan launcher: start the server, or start it and open a coding
# client against it. Interactive by default; every question has a default, so
# pressing Enter through them launches the recommended setup.
#
#   tools/server_launcher.sh
#
# Positional form (backwards compatible with the old launcher):
#
#   tools/server_launcher.sh [<client>] [fast|full] [<model> [4|8]] \
#                            [default|concise] [<thinking>] [<ram>]
#
#   <client>   server (default), or one of the coding clients:
#              codex, claude, qwen, opencode, zed
#   <model>    an install key (ornith|qwen36|agentworld|katcoder|qwen38, or
#              qwen35-2b|qwen35-4b|qwen35-9b) optionally followed by 4|8, or a
#              catalog id such as ornith-1.5-35b-a3b_8-Bit, which names its own
#              width. A model that is not on disk is offered for download and
#              conversion at that point (see --install). The three qwen35-* keys
#              are the dense Qwen 3.5 models (2B, 4B, 9B): both engines implement
#              their family (`qwen3_5_dense`), so they are the one install shape
#              whose engine is a real choice -- GPU by default, CPU on request.
#   <thinking> off, on, or any level the chosen model lists
#              (minimal, low, medium, high, xhigh, max). The dense Qwen 3.5
#              models define the binary thinking switch, so their levels are
#              exactly off|on -- off for a direct answer, on to reason first.
#   <ram>      resident-memory target for the server in GB: any whole number
#              from 4 up (the interactive question offers 4, 6, 8, 10, 12, 14
#              or 16, a Custom value, or its default of half this Mac's physical
#              memory).
#              Anything over 50% of this Mac's physical memory is warned about
#              in red and used anyway. Omit it and an interactive run asks the
#              question, whose default passes half of physical memory as an
#              explicit target; an unattended run passes nothing and the
#              install's own measured profile applies, which the runtime holds to
#              a third of physical memory (a third of the cache plus the resident
#              weights is what the process then holds). The CPU engine has no
#              expert cache, so it does not ask and the flag does not apply there.
#
# Flags (override the positional form, and work in any order):
#
#   --client <c>    server|@CLIENTS@
#              (the list is filled from TINYTITAN_CLIENTS in tools/tinytitan_models.sh,
#              which the coder benchmark reads too)
#   --model <m>     model key or catalog id
#   --bits <4|8>    quantization for a model key
#   --engine <cpu|gpu>  which engine serves the model. Almost every install
#              declares exactly one -- a MoE family is GPU-only, a snapshot is
#              CPU-only -- and asking for the other is refused with the reason.
#              The dense Qwen 3.5 models (2B/4B/9B) are the exception: both
#              engines implement their family, so the engine is a real choice,
#              asked interactively and named by an `@cpu`/`@gpu` suffix on the
#              model id a request uses.
#   --mode <fast|full>       fast strips CLI boilerplate; full keeps tools
#   --answers <default|concise>
#   --thinking <level>
#   --ram <n>   resident-memory target for the server in GB (GPU models only),
#              any whole number from 4 up -- the menu offers 4, 6, 8, 10, 12, 14
#              and 16, a Custom value, or its default of half this Mac's memory,
#              and the benchmark harnesses pass others (9, for instance). The
#              expert cache gets what is left after the resident weights and the
#              runtime. 4 GB is the floor (the weights plus a minimum cache are
#              ~4.7 GB on the 125B install); over 50% of this Mac's physical
#              memory is warned about, not refused
#   --context <n|native|max> native 262144, or 524288/1048576 with --yarn
#   --kv <4|8|16>   KV-cache precision (default 8)
#   --yarn          enable YaRN context scaling
#   --port <n>      default 8080 (TINYTITAN_PORT overrides)
#   --concurrency <n>  generations served at once through one model (default 1:
#              one at a time). A power of two up to 256 -- 1, 2, 4, 8, 16, ... --
#              because an agentic workload may want many, and the runtime clamps
#              the width to what this Mac's memory can hold, logging what it
#              built. Above 1 each running sequence keeps its own KV cache, so
#              memory use rises, and one GPU shared between them makes every
#              answer slower -- it buys fairness (nobody queues), not throughput.
#              The prompt cache is off above 1, so follow-up turns re-read their
#              whole prompt. The launcher asks, and warns in red, because this is
#              the setting a person turns on once and then wonders why the Mac is
#              swapping. GPU models only: the CPU engine runs one generation at a
#              time whatever it is told.
#   --memory        enable persistent agent memory for this project
#   --web           after the server is up, open TinyTitan's own DeepSeek
#              Harness in the default browser: a local page with a prompt box,
#              installed under ~/.tinytitan and isolated from any dsh you run
#              yourself (see tools/dsh_local.sh). 7788 is its port, or the next
#              free one.
#   --install       download and convert a model that is not on disk yet without
#              asking. The interactive menu lists the supported models this
#              checkout does not have as extra rows, marked "not installed";
#              picking one fetches it and then starts normally. By default the
#              launcher asks before such a download when a person is there, and
#              a piped or --dry-run invocation never downloads at all: it prints
#              the `tools/install_models.sh <key>` command instead. Both the
#              download and the conversion report a percentage while they run.
#   --dry-run       print the server command and client setup; start nothing
#   --prompt-cache <multi-prefix|off>  prompt-state reuse (default multi-prefix,
#              a 256 MiB cache). `off` is for the cache A/B harnesses, which
#              have to be able to ask for the arm they measure.
#   --mtp-model <dir>   attach a native speculative draft head. GPU only (the
#              draft shares the target's embedding and head), and not a catalog
#              entry, because a sidecar has no weights for the catalog to
#              describe. `--mtp-memory-mib <n>` sets its budget (default 384).
#   --help, -h
#
# What the server runs: every installed model is reachable by name through
# the API (--models-dir), one resident at a time; GPU models are pinned to
# native 262,144-token context, a 256 MiB multi-prefix prompt cache, 8-bit KV
# and MTP off. Everything tuned per model and quantization -- the expert-cache
# budget, prefetch, the prefill chunk, sampling -- comes from the install's own
# ModelProfile row, so the launcher never overrides a measured optimum.
#
# --ram is a **process target**, not a cache size: the runtime subtracts the
# resident weights and a measured runtime reserve and gives the routed-expert
# cache what is left, stepping down the slot ladder so the server stays under
# the number. On a Qwen3.8 4-bit install the floor is about 3.7 GB, so --ram 4
# lands at the 8-slot minimum (~4.7 GB) and anything below 4 is refused. Past 50% of physical
# memory the server starts competing with the rest of the machine -- the cache
# is wired and cannot be paged out -- so this launcher warns above that line and
# passes the size on as asked; the runtime separately holds the *profile's* own
# cache value to a third of physical memory on the default (no-flag) path.
# CPU models take none of the pinned flags: that backend has no prompt cache, no
# quantized KV and no expert cache, so --kv, --context, --yarn and --ram are
# reported as not applying rather than passed or dropped in silence.
#
# Overrides: TINYTITAN_PORT, TINYTITAN_THINKING_MODE, TINYTITAN_CATALOG_JSON,
# TINYTITAN_PHYSICAL_RAM_BYTES (the RAM-ceiling test seam), TINYTITAN_MODELS_DIR (the
# installs directory, models/ by default), TINYTITAN_LAUNCHER_DRY_RUN=1, and the
# per-client ones below.
# TINYTITAN_LAUNCHER_ASSUME_TTY=1 answers the interactive questions from a pipe
# while still starting nothing (it is the test seam for this script's
# questions, not something a person needs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Where the built binaries are. A checkout builds into `.build/release`; an
# install from the release tarball keeps them in `~/.tinytitan/bin` and says so
# with TINYTITAN_BIN_DIR, because there is no `.build` in that layout — the whole
# point of installing a release is that nothing had to be built.
BIN_DIR="${TINYTITAN_BIN_DIR:-$BASE_DIR/.build/release}"
BINARY="$BIN_DIR/TinyTitanServer"
MODELS_DIR="${TINYTITAN_MODELS_DIR:-$BASE_DIR/models}"
# One catalogue for the model list, the install paths and the port.
# shellcheck source=tools/tinytitan_models.sh
source "$SCRIPT_DIR/tinytitan_models.sh"

say()  { printf '%s\n' "$*"; }
rule() { printf '%s\n' "------------------------------------------------------------"; }
# Bold red, for the warning a person has to notice before the model starts.
# Colour only when a terminal is watching stderr and NO_COLOR is unset: a pipe,
# a log file or TERM=dumb gets the same words without the escape.
warn_red() {
  if [[ -t 2 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
    printf '\033[1;31m%s\033[0m\n' "$*" >&2
  else
    printf '%s\n' "$*" >&2
  fi
}

# ============================================================
# Arguments
# ============================================================

usage() {
  # The client list is filled from the shared catalogue rather than copied
  # here, so this help cannot name a client the launcher does not accept.
  sed -n '2,/^set -euo pipefail/p' "$0" \
    | sed 's/^# \{0,1\}//' \
    | sed "s/@CLIENTS@/$(tinytitan_client_ids_csv '|')/" \
    | sed '$d'
}

DRY_RUN=0
if [[ "${TINYTITAN_LAUNCHER_DRY_RUN:-0}" == 1 ]]; then DRY_RUN=1; fi

# A dry run and a piped run must never block on a question: they take every
# default instead. Interactive only when a person is actually there -- except
# under the test seam, which answers the questions from a pipe while the dry
# run still decides that nothing is started.
INTERACTIVE=1
if [[ "$DRY_RUN" == "1" || ! -t 0 ]]; then INTERACTIVE=0; fi
if [[ "${TINYTITAN_LAUNCHER_ASSUME_TTY:-0}" == "1" ]]; then INTERACTIVE=1; fi

CLIENT=""; MODE=""; MODEL_ARG=""; BITS=""; ANSWERS=""; THINKING_ARG=""
RAM_ARG=""; CONTEXT_ARG=""; KV_ARG=""; YARN=0; PORT_ARG=""; MEMORY=0; ENGINE_ARG=""
CACHE_ARG=""; MTP_MODEL_ARG=""; MTP_MEMORY_ARG=""; CONCURRENCY_ARG=""
# --install: fetch a model that is not on disk yet instead of stopping. The
# default is to ask, because the download is 20-220 GB; a pipe or a dry run
# never downloads on its own, and `--install` is how a caller says yes ahead of
# time (a first-run script, a CI image).
INSTALL_MISSING=0
# --web: after the server is up, hand the terminal over to TinyTitan's own
# DeepSeek Harness, which opens a prompt box in the default browser. It is a
# mode rather than a `--client` entry on purpose: the client list in
# tools/tinytitan_models.sh is shared with the coder benchmark, which rejects
# any kind that is not `coder` or `editor` and asserts the id set exactly.
WEB=0
# The tiers are the sizes a person can reason about, not the runtime's own
# slot rungs. Defined here, before the positional parse, because the
# unlabelled RAM value is read there: with the definition further down the
# file, bash resolved that call to nothing and a positional RAM tier was
# silently ignored.
# The tiers are the sizes a person can reason about, not the runtime's own
# slot rungs. The runtime derives slots from the budget and the model's own
# expert stride, so the same tier means fewer slots on a wider model.
ram_tier() {
  # A whole number of GB, with or without the "G" suffix, from 4 up. The
  # runtime's --ram-budget enforces the same floor and the benchmark profile
  # passes its own value through TINYTITAN_BENCH_RAM_BUDGET. The interactive
  # question offers 4/6/8/10/12/14/16, a Custom value parsed here, or the
  # install's own profile.
  local value="${1%[Gg]}"
  case "$value" in
    *[!0-9]*|"") return 1 ;;
  esac
  # 4 GB is the floor: a streaming install holds its weight file (2.5-4.5 GB)
  # plus an 8-slot minimum expert cache, which is about 4.7 GB on Qwen3.8 4-bit.
  # A smaller target cannot be honoured, so it is refused here rather than
  # silently overshot -- the runtime refuses it too.
  (( 10#$value >= 4 )) || return 1
  echo "$(( 10#$value ))"
}

positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --client)   CLIENT="${2:?--client needs a value}"; shift 2 ;;
    --model)    MODEL_ARG="${2:?--model needs a value}"; shift 2 ;;
    --bits)     BITS="${2:?--bits needs 4 or 8}"; shift 2 ;;
    --engine)   ENGINE_ARG="${2:?--engine needs cpu or gpu}"; shift 2 ;;
    --mode)     MODE="${2:?--mode needs fast or full}"; shift 2 ;;
    --answers)  ANSWERS="${2:?--answers needs default or concise}"; shift 2 ;;
    --thinking) THINKING_ARG="${2:?--thinking needs a level}"; shift 2 ;;
    --ram)      RAM_ARG="${2:?--ram needs a GB size}"; shift 2 ;;
    --context)  CONTEXT_ARG="${2:?--context needs a value}"; shift 2 ;;
    --kv)       KV_ARG="${2:?--kv needs 4, 8 or 16}"; shift 2 ;;
    --yarn)     YARN=1; shift ;;
    # The benchmark harnesses measure the prompt cache as a variable, so the
    # launcher has to be able to turn it off; `multi-prefix` (the default) is
    # the 256 MiB cache every published number was taken with.
    --prompt-cache) CACHE_ARG="${2:?--prompt-cache needs multi-prefix or off}"; shift 2 ;;
    # The native speculative draft head. It is not a catalog entry (a sidecar
    # has no weights of its own for the catalog to describe), and it is the
    # only way to reach the speculative path, so a harness that measures MTP
    # could not otherwise go through this script at all.
    --mtp-model) MTP_MODEL_ARG="${2:?--mtp-model needs a directory}"; shift 2 ;;
    --mtp-memory-mib) MTP_MEMORY_ARG="${2:?--mtp-memory-mib needs a number}"; shift 2 ;;
    --port)     PORT_ARG="${2:?--port needs a number}"; shift 2 ;;
    # How many generations one server serves at once. The server's own default
    # is one, so this only ever raises it, and raising it is warned about.
    --concurrency) CONCURRENCY_ARG="${2:?--concurrency needs a power of two}"; shift 2 ;;
    --memory)   MEMORY=1; shift ;;
    --web)      WEB=1; shift ;;
    # Answer the "download and convert it?" question ahead of time. Without it
    # the launcher asks when a person is there, and never downloads on its own
    # from a pipe or a dry run.
    --install)  INSTALL_MISSING=1; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --help|-h)  usage; exit 0 ;;
    --)         shift; while [[ $# -gt 0 ]]; do positional+=("$1"); shift; done ;;
    -*)         echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    *)          positional+=("$1"); shift ;;
  esac
done

# Old five-argument form without a model: `... <api> <mode> <bits> ...`
# inserts the default model so the positions below line up. Kept because the
# benchmark harness and older notes call it that way.
if (( ${#positional[@]} >= 3 )); then
  case "${positional[1]}:${positional[2]}" in
    full:4|full:8|full:4bit|full:8bit|fast:4|fast:8|fast:4bit|fast:8bit)
      positional=( "${positional[0]:-}" "${positional[1]:-}" ornith "${positional[2]}" \
                   "${positional[3]:-}" "${positional[4]:-}" "${positional[5]:-}" )
      ;;
  esac
fi

# Positional defaults for anything the flags did not set. Values are consumed
# in order and disambiguated by shape, so a flag may supply the model and
# still leave the width or the later fields positional.
pos=0
# The old launcher's first position named the API (openai|anthropic) rather
# than a client. The server now speaks both at once, so such a value selects
# no client and is simply consumed.
if [[ -n "$CLIENT" && ${#positional[@]} -gt 0 ]]; then
  case "${positional[0]}" in
    openai|anthropic) pos=1 ;;
  esac
fi
# Only a value that names a client is taken as one. `ornith 4` and
# `--model ornith 4` must leave the model where the model pass looks for it.
if [[ -z "$CLIENT" && ${#positional[@]} -gt $pos ]]; then
  case "${positional[$pos]}" in
    server|codex|claude|qwen|opencode|zed|openai|anthropic|none|api)
      CLIENT="${positional[$pos]}"; pos=$((pos + 1)) ;;
  esac
fi
if [[ -z "$MODE" && ${#positional[@]} -gt $pos ]]; then
  case "${positional[$pos]}" in
    fast|full|1|0) MODE="${positional[$pos]}"; pos=$((pos + 1)) ;;
  esac
fi
if [[ -z "$MODEL_ARG" && ${#positional[@]} -gt $pos ]]; then
  MODEL_ARG="${positional[$pos]}"; pos=$((pos + 1))
fi
if [[ -z "$BITS" && ${#positional[@]} -gt $pos ]]; then
  case "${positional[$pos]}" in
    4|8|4bit|8bit) BITS="${positional[$pos]}"; pos=$((pos + 1)) ;;
  esac
fi
# Answers, thinking and RAM are optional and unlabelled, so take them in
# whatever order they appear. A value that can be none of them is ignored
# rather than silently treated as a mode.
for value in "${positional[@]:$pos}"; do
  [[ -z "$value" ]] && continue
  if [[ -z "$ANSWERS" ]]; then
    case "$value" in
      default|standard|concise) ANSWERS="$value"; continue ;;
    esac
  fi
  if [[ -z "$THINKING_ARG" ]]; then
    if normalize_level "$value" >/dev/null 2>&1; then THINKING_ARG="$value"; continue; fi
  fi
  if [[ -z "$RAM_ARG" ]]; then
    if ram_tier "$value" >/dev/null 2>&1; then RAM_ARG="$value"; continue; fi
    # A number here can only have been meant as the RAM tier, so refuse it
    # instead of ignoring it the way an unrecognised word is ignored: silently
    # dropping a positional `2` is how a person ends up running the model
    # default while believing they set a limit.
    if [[ "$value" =~ ^[0-9]+[Gg]?$ ]]; then
      echo "unknown RAM target: $value (minimum 4 GB; the weights plus a minimum expert cache are ~4.7 GB on a 125B install, so a smaller target cannot be honoured)" >&2
      exit 2
    fi
  fi
done

# ============================================================
# 1) Client
# ============================================================

client_label() {
  case "$1" in
    server) echo "Server only (no client)" ;;
    *)      tinytitan_client_label "$1" || echo "$1" ;;
  esac
}

normalize_client() {
  case "$1" in
    ""|server|none|api) echo server; return 0 ;;
    # The coding-CLI spellings earlier versions accepted.
    claude|claude-code) echo claude; return 0 ;;
    qwen|qwen-code)     echo qwen; return 0 ;;
    # The old API names meant "start the server for that API".
    openai|anthropic)   echo server; return 0 ;;
  esac
  # Everything else has to be a client the shared catalogue carries.
  if tinytitan_client_label "$1" >/dev/null 2>&1; then echo "$1"; return 0; fi
  return 1
}

# --web is its own handover, so it does not ask the client question: the browser
# page is what comes next, and a coder CLI on top of it would fight for the
# terminal.
if [[ -z "$CLIENT" ]] && (( INTERACTIVE )) && (( ! WEB )); then
  menu_ids=()
  while IFS= read -r id; do menu_ids+=("$id"); done < <(tinytitan_client_ids)
  echo "What do you want to launch?"
  echo "  1) Server only — start the API, no client (default)"
  for (( menu_index = 0; menu_index < ${#menu_ids[@]}; menu_index++ )); do
    printf '  %d) %s\n' "$((menu_index + 2))" "$(client_label "${menu_ids[$menu_index]}")"
  done
  printf "Choice [1-%d] (default 1): " "$(( ${#menu_ids[@]} + 1 ))"
  read -r client_choice || exit 1
  client_choice="${client_choice:-1}"
  if [[ "$client_choice" == "1" ]]; then
    CLIENT=server
  elif [[ "$client_choice" =~ ^[0-9]+$ ]] \
    && (( client_choice >= 2 && client_choice <= ${#menu_ids[@]} + 1 )); then
    CLIENT="${menu_ids[$((client_choice - 2))]}"
  else
    echo "invalid choice: $client_choice" >&2; exit 2
  fi
fi
[[ -z "$CLIENT" ]] && CLIENT=server
requested_client="$CLIENT"
if ! CLIENT="$(normalize_client "$requested_client")"; then
  # Name what was asked for: the assignment above has already emptied CLIENT.
  echo "unknown client: $requested_client (server|$(tinytitan_client_ids_csv '|'))" >&2
  exit 2
fi

# Which API surface the client needs. Only used to print the right setup and
# to pick the right config shape; the server speaks all of them at once.
case "$CLIENT" in
  claude) API=anthropic ;;
  *)      API=openai ;;
esac

# ============================================================
# 2) Mode: full agent loop, or fast chat
# ============================================================

if [[ -n "$MODE" ]]; then
  case "$MODE" in
    fast|1) model_word=fast ;;
    full|0) model_word=full ;;
    *) echo "unknown mode: $MODE (fast|full)" >&2; exit 2 ;;
  esac
else
  # A plain API server has no agent loop to preserve, so it does not ask.
  if [[ "$CLIENT" == "server" || ! -t 0 ]]; then
    model_word=full
  else
    echo ""
    echo "Full agent loop or fast chat?"
    echo "  1) Full (keep agent tools; multi-thousand-token prefill, slower)"
    echo "  2) Fast (strip CLI boilerplate, seconds-per-answer chat)"
    printf "Choice [1-2] (default 1): "
    read -r model_choice || exit 1
    case "${model_choice:-1}" in
      1) model_word=full ;;
      2) model_word=fast ;;
      *) echo "invalid choice: $model_choice" >&2; exit 2 ;;
    esac
  fi
fi

# ============================================================
# 3) Model and quantization
# ============================================================

# Ask once, then install one catalogue key.
#
# Answers, in order: `--install` says yes ahead of time; a person at a terminal
# is asked (Enter takes the yes, because they asked for the model); a pipe, a
# log or `--dry-run` never downloads and is told the exact command instead. The
# installer streams a percentage for the download and for the conversion, so
# nothing here has to draw progress -- it only has to say what is about to
# happen.
#
# <key> is an install_models.sh target (katcoder-8bit), <label> what to call it
# in the question, <bits> 4 or 8. Returns 0 once it is on disk.
install_model_key() {
  local key="$1" label="$2" bits="$3" gb="" reply=""
  gb="$(tinytitan_install_size_gb "$key" || true)"
  say ""
  if [[ -n "$gb" ]]; then
    say "$label ${bits}-bit is not installed. It is a ${gb} GB install, and"
    say "the download before it is larger."
  else
    say "$label ${bits}-bit is not installed."
  fi
  if (( ! INSTALL_MISSING )); then
    if (( ! INTERACTIVE )); then
      say "Install it with:  tools/install_models.sh $key"
      return 1
    fi
    printf 'Download and convert it now? [Y/n] '
    if ! read -r reply; then
      say ""
      say "No answer given; nothing was installed." >&2
      return 1
    fi
    case "${reply:-y}" in
      y|Y|yes|YES) ;;
      *)
        say "Nothing was installed. When you want it:"
        say "    tools/install_models.sh $key"
        return 1 ;;
    esac
  fi
  # A dry run starts nothing, and a 70 GB download is the largest thing this
  # script could ever start. It says what it would do and stops.
  if (( DRY_RUN )); then
    say "Would install:  tools/install_models.sh $key"
    return 1
  fi
  say ""
  say "Installing $key. The download and the conversion each report a percentage;"
  say "this takes a while (hours for a 125B model, minutes on a fast line)."
  if ! "$SCRIPT_DIR/install_models.sh" "$key"; then
    echo "ERROR: installing $key failed; the model is still not on disk." >&2
    return 1
  fi
  return 0
}

# The same, for a caller that has the runtime stem (a model named on the command
# line) rather than the catalogue key the installer is addressed by.
install_missing_model() {
  local key=""
  if ! key="$(tinytitan_install_key "$1" "$2")"; then
    return 1
  fi
  install_model_key "$key" "$3" "$2"
}

# The installed models: the server's own catalog, or the built-in list. Both
# are held to installs that are really on disk, so the menu never offers a model
# or a width that cannot be loaded.
dynamic=0
if tinytitan_load_catalog "$BINARY" "$MODELS_DIR"; then
  if tinytitan_catalog_keep_installed; then
    dynamic=1
  else
    catalog_error="none of the models it lists are under $MODELS_DIR"
  fi
else
  catalog_error="$TINYTITAN_CATALOG_ERROR"
fi
if (( ! dynamic )); then
  echo "" >&2
  echo "NOTE: the model catalog is unavailable ($catalog_error)." >&2
  echo "      Offering the installs this checkout has instead; the server will" >&2
  echo "      serve only the model chosen here, and switching needs a restart." >&2
  if ! tinytitan_static_catalog "$MODELS_DIR"; then
    echo "" >&2
    echo "ERROR: $TINYTITAN_CATALOG_ERROR." >&2
    echo "       Add one first: docs/adding-a-model.md, or tools/install_models.sh." >&2
    exit 2
  fi
  if (( ${#TINYTITAN_CATALOG_MISSING[@]} > 0 )); then
    echo "      Supported but not installed here: ${TINYTITAN_CATALOG_MISSING[*]+"${TINYTITAN_CATALOG_MISSING[*]}"}" >&2
  fi
fi

if [[ -z "$MODEL_ARG" ]]; then
  count=${#TINYTITAN_CAT_ID[@]}
  default_idx="$(tinytitan_catalog_find_dir ornith-1.5_35B_A3B_8Bit)" || default_idx=0
  echo ""
  if (( dynamic )); then
    echo "Which model? (loaded first; every other one stays available by name)"
  else
    echo "Which model?"
  fi
  # An install both engines can serve says so: `GPU+CPU`. One is the default
  # and the other is a choice the next question offers.
  engine_column() {
    local list="$1"
    if [[ "$list" == *,* ]]; then
      # `tr` rather than `${list^^}`: the uppercase expansion is bash 4+, and
      # /bin/bash is 3.2 on a factory Mac ("bad substitution", at runtime, in the
      # middle of the model menu).
      printf '%s' "$list" | tr '[:lower:]' '[:upper:]' | tr ',' '+'
    else
      [[ "$list" == cpu ]] && echo CPU || echo GPU
    fi
  }
  # Engine and thinking levels are the two things a person is choosing between
  # here, so both are in the list rather than discovered after the fact.
  printf "  %-3s %-28s %-6s %-4s %8s  %-22s %-24s %s\n" \
    "#" "model" "bits" "engine" "size" "api id" "thinking" ""
  for (( i = 0; i < count; i++ )); do
    size=""
    if [[ "${TINYTITAN_CAT_SIZE[$i]}" != "-" ]]; then size="${TINYTITAN_CAT_SIZE[$i]} GB"; fi
    note=""
    if (( i == default_idx )); then note="  (default)"; fi
    printf "  %2d) %-28s %s-bit  %-4s %8s  %-22s %-24s%s\n" "$((i + 1))" \
      "${TINYTITAN_CAT_NAME[$i]}" "${TINYTITAN_CAT_QUANT[$i]}" \
      "$( engine_column "${TINYTITAN_CAT_ENGINES[$i]:-${TINYTITAN_CAT_BACKEND[$i]}}" )" \
      "$size" "${TINYTITAN_CAT_ID[$i]}" "${TINYTITAN_CAT_THINKING[$i]//,/, }" "$note"
  done
  # What this checkout supports but does not have, as rows rather than a note:
  # the menu is where a person sees what TinyTitan can run, and a model they
  # cannot pick is a model they will not discover. Picking one fetches it and
  # then takes the same path as an installed one, so nothing downstream has to
  # know where the bytes came from.
  offers=()
  for (( i = 0; i < count; i++ )); do offers+=("installed|$i"); done
  missing_count=0
  while IFS='|' read -r m_name m_bits m_key m_label m_gb _; do
    [[ -n "$m_name" ]] || continue
    offers+=("missing|$m_name|$m_bits|$m_key|$m_label|$m_gb")
    missing_count=$((missing_count + 1))
  done < <(tinytitan_missing_offers "$MODELS_DIR")
  first_missing=$((count + 1))
  for (( i = 0; i < missing_count; i++ )); do
    IFS='|' read -r _ m_name m_bits m_key m_label m_gb <<< "${offers[$((count + i))]}"
    # The same columns as an installed row: engine and levels are what the
    # choice is between, and a row without them is a row nobody can compare.
    tinytitan_resolve_model "$m_name" 2>/dev/null || continue
    size=""
    [[ -n "$m_gb" ]] && size="$m_gb GB"
    printf "  %2d) %-28s %s-bit  %-4s %8s  %-22s %-24s%s\n" "$((first_missing + i))" \
      "$m_label" "$m_bits" \
      "$( engine_column "${TINYTITAN_MODEL_ENGINES:-${TINYTITAN_MODEL_FAMILY:-}}" )" \
      "$size" "not installed" "${TINYTITAN_MODEL_THINKING//,/, }" "  fetches it first"
  done
  if (( missing_count > 0 )); then
    echo ""
    echo "Rows $first_missing-$((first_missing + missing_count - 1)) are not installed yet:"
    echo "choosing one downloads and converts it first (a percentage is shown)."
  fi
  printf "Choice [1-%d] (default %d): " "$((count + missing_count))" "$((default_idx + 1))"
  read -r pick || exit 1
  pick="${pick:-$((default_idx + 1))}"
  if [[ "$pick" =~ ^[0-9]+$ ]] && (( pick >= 1 && pick <= count + missing_count )); then
    choice="${offers[$((pick - 1))]}"
  else
    echo "invalid choice: $pick" >&2; exit 2
  fi
  if [[ "${choice%%|*}" == "missing" ]]; then
    IFS='|' read -r _ m_name m_bits m_key m_label m_gb <<< "$choice"
    if ! install_model_key "$m_key" "$m_label" "$m_bits"; then
      exit 1
    fi
    # The install landed: the catalog has to name it before the rest of this
    # script can use it, and its directory is what the arrays carry.
    if ! tinytitan_load_catalog "$BINARY" "$MODELS_DIR" \
      || ! tinytitan_resolve_model "$m_name" 2>/dev/null \
      || ! tinytitan_resolve_quant "$m_bits" 2>/dev/null \
      || ! idx="$(tinytitan_catalog_find_dir "${TINYTITAN_MODEL_STEM}_${TINYTITAN_QUANT_DIR}")"; then
      echo "ERROR: installed $m_key, but the catalog does not list it under $MODELS_DIR." >&2
      exit 1
    fi
  else
    idx="${choice#installed|}"
  fi
else
  if idx="$(tinytitan_catalog_find_id "$MODEL_ARG")"; then
    if [[ -n "$BITS" ]]; then
      case "$BITS" in
        4|8|4bit|8bit)
          if [[ "${BITS%bit}" != "${TINYTITAN_CAT_QUANT[$idx]}" ]]; then
            echo "$MODEL_ARG is ${TINYTITAN_CAT_QUANT[$idx]}-bit, not ${BITS%bit}-bit" >&2
            exit 2
          fi ;;
        *) echo "unknown bits: $BITS (4|8)" >&2; exit 2 ;;
      esac
    fi
  else
    if ! tinytitan_resolve_model "$MODEL_ARG" 2>/dev/null; then
      echo "unknown model: $MODEL_ARG (a model id, or ornith|qwen36|agentworld|katcoder|qwen38|qwen35-2b|qwen35-4b|qwen35-9b)" >&2
      if (( dynamic )); then echo "installed: ${TINYTITAN_CAT_ID[*]+"${TINYTITAN_CAT_ID[*]}"}" >&2; fi
      exit 2
    fi
    # No width given means 8-bit, the historical default.
    tinytitan_resolve_quant "${BITS:-8}" || exit 2
    if ! idx="$(tinytitan_catalog_find_dir "${TINYTITAN_MODEL_STEM}_${TINYTITAN_QUANT_DIR}")"; then
      echo "ERROR: $TINYTITAN_MODEL_LABEL ${TINYTITAN_QUANT%bit}-bit is not installed (no ${TINYTITAN_MODEL_STEM}_${TINYTITAN_QUANT_DIR} under $MODELS_DIR)" >&2
      # Name the widths that are here: "not installed" alone reads like the
      # whole model is missing when only the one asked for is.
      installed_widths=""
      for width_idx in "${!TINYTITAN_CAT_PATH[@]}"; do
        if [[ "$(basename "${TINYTITAN_CAT_PATH[$width_idx]}")" == "${TINYTITAN_MODEL_STEM}_"* ]]; then
          installed_widths="$installed_widths ${TINYTITAN_CAT_QUANT[$width_idx]}-bit"
        fi
      done
      if [[ -n "$installed_widths" ]]; then
        echo "       installed here:$installed_widths" >&2
      fi
      # Fetch it and carry on, or stop with the exact command. The catalog is
      # re-read afterwards: the install that just landed is what the index has
      # to name before the rest of this script can use it.
      if install_missing_model "$TINYTITAN_MODEL_STEM" "${TINYTITAN_QUANT%bit}" "$TINYTITAN_MODEL_LABEL"; then
        if ! tinytitan_load_catalog "$BINARY" "$MODELS_DIR" \
          || ! idx="$(tinytitan_catalog_find_dir "${TINYTITAN_MODEL_STEM}_${TINYTITAN_QUANT_DIR}")"; then
          echo "ERROR: $TINYTITAN_MODEL_LABEL ${TINYTITAN_QUANT%bit}-bit installed, but the" >&2
          echo "       catalog still does not list it under $MODELS_DIR." >&2
          exit 1
        fi
      else
        exit 1
      fi
    fi
  fi
fi

MODEL_ID="${TINYTITAN_CAT_ID[$idx]}"
MODEL_NAME="${TINYTITAN_CAT_NAME[$idx]}"
MODEL_QUANT="${TINYTITAN_CAT_QUANT[$idx]}"
MODEL_BACKEND="${TINYTITAN_CAT_BACKEND[$idx]}"
MODEL_DIR="${TINYTITAN_CAT_PATH[$idx]}"
IFS=',' read -r -a levels <<< "${TINYTITAN_CAT_THINKING[$idx]}"

# ============================================================
# 3b) Engine: which engine serves this install, and what was asked for
# ============================================================

# The catalog says which engines can serve an install. Almost every install has
# exactly one -- a MoE family is GPU-only because the CPU engine does not
# implement those shapes, and a converted snapshot is CPU-only -- but the dense
# Qwen 3.5 models (2B/4B/9B) are implemented by *both*, from the same `.ssdai`
# payload. That is the one case where the engine is a real choice, so it is the
# one case that asks.
IFS=',' read -r -a engines <<< "${TINYTITAN_CAT_ENGINES[$idx]:-$MODEL_BACKEND}"
default_engine="${TINYTITAN_CAT_BACKEND[$idx]}"
engine_family="${TINYTITAN_CAT_FAMILY[$idx]:--}"
engine_name() { if [[ "$1" == cpu ]]; then echo CPU; else echo GPU; fi; }
engine_available() {
  local candidate
  for candidate in "${engines[@]+"${engines[@]}"}"; do [[ "$candidate" == "$1" ]] && return 0; done
  return 1
}
engine_reason() {
  if [[ "$1" == cpu ]]; then
    echo "$MODEL_NAME ${MODEL_QUANT}-bit declares family $engine_family, which the GPU engine does not implement; it runs on the CPU engine"
  else
    echo "$MODEL_NAME ${MODEL_QUANT}-bit declares family $engine_family, which the CPU engine does not implement; it runs on the GPU engine"
  fi
}

if [[ -n "$ENGINE_ARG" ]]; then
  case "$ENGINE_ARG" in
    cpu|gpu) ;;
    *) echo "unknown --engine: $ENGINE_ARG (cpu|gpu)" >&2; exit 2 ;;
  esac
  if ! engine_available "$ENGINE_ARG"; then
    echo "$(engine_reason "$default_engine")." >&2
    echo "--engine $ENGINE_ARG is not available for it." >&2
    exit 2
  fi
  ENGINE="$ENGINE_ARG"
elif (( ${#engines[@]} > 1 )) && (( INTERACTIVE )); then
  echo ""
  echo "Which engine? $MODEL_NAME ${MODEL_QUANT}-bit runs on both."
  default_choice=1
  for (( i = 0; i < ${#engines[@]}; i++ )); do
    note=""
    if [[ "${engines[$i]}" == "$default_engine" ]]; then
      default_choice=$((i + 1))
      note="  (default)"
    fi
    printf "  %d) %s%s\n" "$((i + 1))" "$(engine_name "${engines[$i]}")" "$note"
  done
  printf "Choice [1-%d] (default %d): " "${#engines[@]}" "$default_choice"
  read -r engine_choice || exit 1
  engine_choice="${engine_choice:-$default_choice}"
  if [[ "$engine_choice" =~ ^[0-9]+$ ]] \
     && (( engine_choice >= 1 && engine_choice <= ${#engines[@]} )); then
    ENGINE="${engines[$((engine_choice - 1))]}"
  else
    echo "invalid choice: $engine_choice" >&2; exit 2
  fi
else
  ENGINE="$default_engine"
fi

# The engine is stated, and whether it was a choice. The list's engine column
# shows the default; this says which one this run will use and why there is (or
# is not) an alternative.
if (( ${#engines[@]} > 1 )); then
  engine_line="$(engine_name "$ENGINE") -- your choice of $(printf '%s' "${engines[*]+"${engines[*]}"}" | tr ' ' '/')"
else
  engine_line="$(engine_name "$ENGINE") only -- $(engine_reason "$ENGINE")"
fi
if (( INTERACTIVE )); then echo "Engine: $engine_line"; fi
# The request names the engine with an `@cpu`/`@gpu` suffix when it is not the
# install's default; the server registers those aliases from the same catalog
# field, so the two cannot disagree about what is available.
MODEL_ID_LAUNCH="$MODEL_ID"
if [[ "$ENGINE" != "$default_engine" ]]; then
  MODEL_ID_LAUNCH="${MODEL_ID}@${ENGINE}"
fi
MODEL_BACKEND="$ENGINE"

# ============================================================
# 4) Answers: standard or concise
# ============================================================

if [[ -n "$ANSWERS" ]]; then
  case "$ANSWERS" in
    default|standard|1) mode_word=default ;;
    concise|2)          mode_word=concise ;;
    *) echo "unknown answers mode: $ANSWERS (default|concise)" >&2; exit 2 ;;
  esac
elif (( ! INTERACTIVE )); then
  mode_word=default
else
  echo ""
  echo "Which answer style?"
  echo "  1) Standard (default)"
  echo "  2) Concise (terse answers)"
  printf "Choice [1-2] (default 1): "
  read -r answers_choice || exit 1
  case "${answers_choice:-1}" in
    1) mode_word=default ;;
    2) mode_word=concise ;;
    *) echo "invalid choice: $answers_choice" >&2; exit 2 ;;
  esac
fi

# ============================================================
# 5) Thinking: only the levels this model lists, off first and default
# ============================================================

thinking_label() { case "$1" in xhigh) echo "extra high" ;; *) echo "$1" ;; esac; }
has_level() {
  local level
  for level in "${levels[@]+"${levels[@]}"}"; do [[ "$level" == "$1" ]] && return 0; done
  return 1
}
normalize_level() {
  local word
  word="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$word" in
    0|off|nothink|false|no) echo off ;;
    1|on|think|true|yes) echo on ;;
    minimal|low|medium|high|max) echo "$word" ;;
    xhigh|extra-high|"extra high") echo xhigh ;;
    *) return 1 ;;
  esac
}
# "on" has no exact match on a model with effort levels, so it takes medium,
# the middle of every effort scale here, or failing that the first level above
# off. A level the model lacks is refused rather than mapped to a neighbour.
level_for_model() {
  if has_level "$1"; then echo "$1"; return 0; fi
  if [[ "$1" == on ]] && (( ${#levels[@]} > 1 )); then
    if has_level medium; then echo medium; else echo "${levels[1]}"; fi
    return 0
  fi
  return 1
}
level_names() {
  local level out=""
  for level in "${levels[@]+"${levels[@]}"}"; do out="${out:+$out, }$(thinking_label "$level")"; done
  echo "$out"
}

thinking_default="${TINYTITAN_THINKING_MODE:-off}"
if ! default_word="$(normalize_level "$thinking_default")"; then
  echo "invalid TINYTITAN_THINKING_MODE: $thinking_default (off|on|minimal|low|medium|high|xhigh|max)" >&2
  exit 2
fi
if ! default_level="$(level_for_model "$default_word")"; then
  # A default the model cannot render is a person's mistake worth naming, not
  # a level to substitute in silence.
  default_level="${levels[0]}"
  echo "NOTE: TINYTITAN_THINKING_MODE=$thinking_default is not a level $MODEL_NAME ${MODEL_QUANT}-bit renders" >&2
  echo "      ($(level_names)); using $default_level." >&2
fi

if [[ -n "$THINKING_ARG" ]]; then
  if ! word="$(normalize_level "$THINKING_ARG")"; then
    echo "unknown thinking level: $THINKING_ARG (off|on|minimal|low|medium|high|xhigh|max)" >&2
    exit 2
  fi
  if ! thinking_level="$(level_for_model "$word")"; then
    echo "$MODEL_NAME ${MODEL_QUANT}-bit has no thinking level $THINKING_ARG; it has: $(level_names)" >&2
    exit 2
  fi
  if [[ "$word" != "$thinking_level" ]]; then
    echo "Thinking on -> $(thinking_label "$thinking_level"): $MODEL_NAME takes an effort level, not on/off." >&2
  fi
elif (( ${#levels[@]} == 1 )); then
  thinking_level="${levels[0]}"
elif (( ! INTERACTIVE )); then
  thinking_level="$default_level"
else
  echo ""
  if [[ "${levels[*]+"${levels[*]}"}" == "off on" ]]; then
    echo "Reasoning (thinking)? $MODEL_NAME ${MODEL_QUANT}-bit defines the binary switch"
    echo "off|on; a client can switch it per request with reasoning_effort."
  else
    echo "Reasoning effort? $MODEL_NAME ${MODEL_QUANT}-bit renders $(level_names)."
  fi
  default_choice=1
  for (( i = 0; i < ${#levels[@]}; i++ )); do
    level="${levels[$i]}"
    note=""
    case "$level" in
      off) note="direct answers" ;;
      on)  note="model reasons before answering" ;;
    esac
    if [[ "$level" == "$default_level" ]]; then
      default_choice=$((i + 1))
      note="${note:+$note, }default"
    fi
    printf "  %d) %s%s\n" "$((i + 1))" "$(thinking_label "$level")" "${note:+ ($note)}"
  done
  printf "Choice [1-%d] (default %d): " "${#levels[@]}" "$default_choice"
  read -r think_choice || exit 1
  think_choice="${think_choice:-$default_choice}"
  if [[ "$think_choice" =~ ^[0-9]+$ ]] && (( think_choice >= 1 && think_choice <= ${#levels[@]} )); then
    thinking_level="${levels[$((think_choice - 1))]}"
  else
    echo "invalid choice: $think_choice" >&2; exit 2
  fi
fi
think_word="$(thinking_label "$thinking_level")"

# ============================================================
# 6) RAM target: what the whole server may hold (the cache gets the rest)
# ============================================================

# Half of physical memory: the target this launcher recommends, and the point
# past which it warns.
#
# The number has moved twice, and the history is the reason the warning exists
# rather than a hard rule:
#
#   * 5.10 measured the half and rejected it. On a 24 GiB Mac a 12 GiB budget /
#     64 slots paged (swap 855 -> 1,610 MB) at 5.58 tok/s, against flat swap and
#     7.29 tok/s for the 40 slots a third selects.
#   * The launcher then recommended 30%: at 40% real usage ran past half of
#     physical memory, leaving 11% free and glitching CoreAudio while the model
#     was merely loaded.
#   * It is **50%** now, by the owner's decision (2026-10-05): the half is the
#     target they want the machine to give, and the menu says so plainly.
#
# What that changes in practice is narrower than it looks. An *unset* budget is
# still clamped by the runtime to a third of physical memory, so the install's
# own profile cannot reach the half the 5.10 run measured; only an explicit
# target does — this question's default, or `--ram-budget`. The cache is wired
# and cannot be paged out, so everything else resident (dense weights, KV, the
# prompt cache, macOS, whatever else is running) shares what is left, and the
# warning above the rule is what says so.
#
# `TINYTITAN_PHYSICAL_RAM_BYTES` is the test seam: this mapping has to be checkable
# on a machine of any size.
physical_ram_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
if [[ -n "${TINYTITAN_PHYSICAL_RAM_BYTES:-}" ]]; then
  physical_ram_bytes="$TINYTITAN_PHYSICAL_RAM_BYTES"
fi
case "$physical_ram_bytes" in
  *[!0-9]*|"") physical_ram_bytes=0 ;;
esac
physical_ram_gb=$(( (physical_ram_bytes + 1073741823) / 1073741824 ))
# Half, floored to whole GB -- 4 on an 8 GB Mac, 8 on 16, 12 on 24, 16 on 32,
# 32 on 64 -- because `--ram` names whole gigabytes. 4 is also the floor the
# runtime accepts, so the smallest machines land exactly on it.
ram_rule_gb=$(( physical_ram_bytes * 5 / 10 / 1073741824 ))
# An unreadable size means no warning rather than a warning at 1 GB: the same
# thing the runtime's own guard does, and the launcher must not invent a limit
# it cannot justify.
(( physical_ram_bytes > 0 )) || ram_rule_gb=0

ram_over_rule=0
# 1 when the RAM target is the question's default rather than a named choice.
ram_default=0
warn_ram_over_rule() {
  # A budget past this launcher's rule is allowed and warned about, never
  # reduced: the runtime takes an explicit --ram-budget verbatim, and this is
  # the operator's trade to make. What the warning has to be is unmissable and
  # specific -- swapping costs far more throughput than the extra slots buy.
  (( ram_rule_gb > 0 )) || return 0
  (( ram_gb > ram_rule_gb )) || return 0
  ram_over_rule=1
  warn_red "WARNING: the server would hold about ${ram_gb} GB, more than the ${ram_rule_gb} GB"
  warn_red "         this launcher recommends (50% of this Mac's ${physical_ram_gb} GB). The"
  warn_red "         expert cache is wired, so it cannot be paged out and everything"
  warn_red "         else has to fit beside it. Expect:"
  warn_red "           * system instability while the model is loaded"
  warn_red "           * heavy swapping, which stalls other apps (audio, calls)"
  warn_red "           * much slower generation: a paging cache loses more than the"
  warn_red "             extra slots gain"
  warn_red "         Starting anyway with ${ram_gb} GB, because you asked for it."
}

if [[ -n "$RAM_ARG" ]]; then
  if ! ram_gb="$(ram_tier "$RAM_ARG")"; then
    echo "unknown RAM target: $RAM_ARG (minimum 4 GB; the weights plus a minimum expert cache are ~4.7 GB on a 125B install, so a smaller target cannot be honoured)" >&2
    exit 2
  fi
  if [[ "$ENGINE" != "cpu" ]]; then warn_ram_over_rule; fi
elif [[ "$ENGINE" == "cpu" ]]; then
  # Nothing to ask: the CPU engine holds the whole model resident and has no
  # routed-expert cache, so a budget would be a number that changes nothing.
  ram_gb=""
elif (( ! INTERACTIVE )); then
  # Unattended or dry run: keep the install's own measured profile, whose cache
  # budget the runtime holds to a third of physical memory. The 50% target is a
  # person's choice, taken in the question below; a script gets the shipped
  # default, which is also what the benchmark protocol measures
  # (`TINYTITAN_BENCH_RAM_BUDGET` pins one when a harness wants a specific size).
  ram_gb=""
else
  echo ""
  echo "RAM target for the server?"
  echo "  This is what the whole server may hold, weights and runtime included;"
  echo "  the expert cache gets the remainder. More cache means fewer SSD reads"
  echo "  and faster answers, and less left for everything else on the Mac."
  if (( ram_rule_gb > 0 )); then
    echo "  It is wired, so it cannot be paged out. This launcher recommends"
    echo "  half of this Mac's ${physical_ram_gb} GB (${ram_rule_gb} GB) and warns above"
    echo "  that, because the rest of the server and everything else you run"
    echo "  has to fit in what is left."
    rule_hint="50% of this Mac is ${ram_rule_gb} GB"
  else
    rule_hint="recommended"
  fi
  echo "  1) 4 GB     2) 6 GB     3) 8 GB    4) 10 GB"
  echo "  5) 12 GB    6) 14 GB    7) 16 GB   8) Custom"
  echo "  (4 GB is the floor: the weights plus the minimum expert cache are"
  echo "   about 4.7 GB on a 125B install, so a smaller target cannot be met)"
  echo "  9) Model default (${rule_hint})"
  printf "Choice [1-9] (default 9): "
  read -r ram_choice || exit 1
  case "${ram_choice:-9}" in
    1) ram_gb=4 ;;  2) ram_gb=6 ;;  3) ram_gb=8 ;;   4) ram_gb=10 ;;
    5) ram_gb=12 ;; 6) ram_gb=14 ;; 7) ram_gb=16 ;;
    8)
      # Any whole number of GB, through the same parser the flag uses, so the
      # floor and the "with or without a G" spelling cannot differ between the
      # two ways in. The recommended half is the default, because that is the
      # number the paragraph above just named.
      custom_default="${ram_rule_gb:-16}"
      printf "RAM target in whole GB (4 or more) [default %s]: " "$custom_default"
      read -r custom_ram || exit 1
      custom_ram="${custom_ram:-$custom_default}"
      if ! custom_gb="$(ram_tier "$custom_ram")"; then
        echo "invalid choice: $custom_ram (a whole number of GB from 4 up)" >&2
        exit 2
      fi
      ram_gb="$custom_gb"
      ;;
    9)
      # The default is the recommendation now, not the install's profile: it is
      # passed as an explicit target, which is the only way to reach the half --
      # an unset budget is clamped by the runtime to a third of physical memory.
      # With no readable memory size there is nothing to recommend, so the
      # profile stands as it did before.
      ram_default=1
      ram_gb=""
      if (( ram_rule_gb > 0 )); then ram_gb="$ram_rule_gb"; fi
      ;;
    *) echo "invalid choice: $ram_choice" >&2; exit 2 ;;
  esac
  if [[ -n "$ram_gb" ]]; then warn_ram_over_rule; fi
fi
if (( ram_rule_gb > 0 )); then
  ram_note="model default (measured; 50% of this Mac is ${ram_rule_gb} GB)"
else
  ram_note="model default (measured)"
fi
if [[ -n "$ram_gb" ]]; then
  ram_note="${ram_gb} GB (your choice)"
  # The question's default is the recommended half, and it is passed as an
  # explicit target, so the banner has to say which of the two it is: a person
  # cannot otherwise tell their own choice from the recommendation.
  if (( ram_default )); then
    ram_note="${ram_gb} GB (default: 50% of this Mac)"
  fi
  # The banner is the last thing printed before the model starts, so the
  # summary line carries the risk once more for anyone who scrolled past the
  # warning itself.
  if (( ram_over_rule )); then
    ram_note="${ram_gb} GB (your choice; over 50% of this Mac's RAM)"
    if (( ram_default )); then
      ram_note="${ram_gb} GB (default; over 50% of this Mac's RAM)"
    fi
  fi
fi

# A CPU model ignores the routed-expert cache entirely, so say what the number
# does rather than leaving a budget that never takes effect.
if [[ "$ENGINE" == "cpu" ]]; then
  ram_note="not applicable (no routed-expert cache on the CPU engine)"
  if [[ -n "$RAM_ARG" ]]; then
    echo "NOTE: $MODEL_NAME runs on the CPU and has no routed-expert cache;" >&2
    echo "      --ram $RAM_ARG does not apply to it." >&2
  fi
fi

# ============================================================
# 7) Advanced server features
# ============================================================

# Context and KV are only offered interactively when the caller asked for the
# detailed choices; the defaults are what almost everyone wants.
kv_bits="${KV_ARG:-8}"
case "$kv_bits" in 4|8|16) : ;; *) echo "unknown --kv: $kv_bits (4|8|16)" >&2; exit 2 ;; esac

max_context=""
rope_scaling="none"
if (( YARN )); then
  rope_scaling="yarn"
  max_context="${CONTEXT_ARG:-1048576}"
elif [[ -n "$CONTEXT_ARG" ]]; then
  case "$CONTEXT_ARG" in
    native|max) max_context=262144 ;;
    *[!0-9]*) echo "unknown --context: $CONTEXT_ARG (a token count, native, or max)" >&2; exit 2 ;;
    *) max_context="$CONTEXT_ARG" ;;
  esac
fi

# Concurrency: how many generations one server serves at once. It is opt-in --
# the server's own default is one -- because it is the one setting that changes
# what the machine has to hold *while it works*: every running sequence keeps its
# own KV cache and scratch, and the single GPU is shared between them. The
# runtime clamps the width when the worst case does not fit in memory, and says
# so; that clamp is the backstop, not the reason to be quiet about the bill.
# The engine's slot ceiling (`KVCacheManager.maximumSlots`). Kept as a number
# here rather than derived, because this is a shell script and one constant to
# keep beside the Swift one is cheaper than a build dependency.
MAX_CONCURRENCY=256

# A power of two in range. Asking for more than this Mac can run is allowed on
# purpose -- the runtime clamps the width it builds and logs it -- but a number
# that is not a power of two is a typo, and is refused here.
valid_concurrency() {
  local value="$1"
  case "$value" in ''|*[!0-9]*) return 1 ;; esac
  (( 10#$value >= 1 && 10#$value <= MAX_CONCURRENCY )) || return 1
  (( (10#$value & (10#$value - 1)) == 0 ))
}

warn_concurrency() {
  # Unmissable and specific, the same way the expert-cache warning is: this is
  # the trade the operator is making, and the summary repeats it for anyone who
  # scrolled past.
  warn_red "WARNING: the server will serve ${concurrency} generations at once."
  warn_red "         Each running sequence keeps its own KV cache and scratch, so"
  warn_red "         memory use rises with the count: the worst case is"
  warn_red "         ${concurrency} x one sequence, on top of the weights and the"
  warn_red "         expert cache. The runtime clamps the width if that does not"
  warn_red "         fit, and says so when it does."
  warn_red "         One GPU is shared: ${concurrency} clients make progress together"
  warn_red "         instead of queueing, but each answer takes roughly"
  warn_red "         ${concurrency} times longer. This buys fairness, not throughput."
  warn_red "         The prompt cache is off above 1, so every follow-up turn"
  warn_red "         re-reads its whole prompt."
  warn_red "         Starting anyway with ${concurrency}, because you asked for it."
  if (( concurrency > 16 )); then
    warn_red "         At ${concurrency} the clamp is likely to bind: read the server's"
    warn_red "         own 'batch width' line to see how many slots it really built."
  fi
}

# A width of one is the default and needs no warning; only a raised one does.
warn_concurrency_if_many() {
  (( concurrency > 1 )) || return 0
  warn_concurrency
}

concurrency=1
# The flag is validated whether or not it can take effect, so a typo is refused
# rather than silently swallowed on the CPU engine.
if [[ -n "$CONCURRENCY_ARG" ]]; then
  if ! valid_concurrency "$CONCURRENCY_ARG"; then
    echo "unknown --concurrency: $CONCURRENCY_ARG (a power of two from 1 to $MAX_CONCURRENCY)" >&2
    exit 2
  fi
  concurrency="$CONCURRENCY_ARG"
fi

if [[ "$ENGINE" == "cpu" ]]; then
  # Nothing to ask and nothing to warn about: the CPU backend's actor runs a
  # generation to completion before the next starts, so a width above one only
  # widens the admission window. The flag is reported as not applying in the CPU
  # block below, and the value that reaches the server is pinned to one, so the
  # summary cannot advertise a width the server will not run.
  concurrency=1
elif (( INTERACTIVE )) && [[ -z "$CONCURRENCY_ARG" ]]; then
  echo ""
  echo "How many generations should the server serve at once?"
  echo "  1 is the default and the fastest per answer: the whole GPU works on"
  echo "  your request. More lets that many people (or agents) make progress"
  echo "  together instead of queueing, and costs: each extra sequence holds"
  echo "  its own KV cache, so memory use rises, and one GPU shared n ways"
  echo "  makes every answer about n times slower. The prompt cache is off"
  echo "  above 1, so follow-up turns re-read their whole prompt."
  echo "  This Mac may hold fewer than you ask for: the server clamps the width"
  echo "  it builds to the memory available, and its log says what it built."
  echo "  1) One (default)   2) Two        3) Four"
  echo "  4) Eight           5) Sixteen    6) A custom power of two"
  printf "Choice [1-6] (default 1): "
  read -r concurrency_choice || exit 1
  case "${concurrency_choice:-1}" in
    1) concurrency=1 ;; 2) concurrency=2 ;; 3) concurrency=4 ;;
    4) concurrency=8 ;; 5) concurrency=16 ;;
    6)
      printf "Power of two [1-%s] (default 32): " "$MAX_CONCURRENCY"
      read -r custom_concurrency || exit 1
      custom_concurrency="${custom_concurrency:-32}"
      if ! valid_concurrency "$custom_concurrency"; then
        echo "invalid choice: $custom_concurrency (a power of two from 1 to $MAX_CONCURRENCY)" >&2
        exit 2
      fi
      concurrency="$custom_concurrency"
      ;;
    *) echo "invalid choice: $concurrency_choice" >&2; exit 2 ;;
  esac
fi
# Outside the branches on purpose: a raised width has to warn whether it came
# from the flag or the question.
warn_concurrency_if_many

# The port is the last question about the *server* rather than the model, so it is
# asked last and it is skippable: `--port` or `TINYTITAN_PORT` answers it, and an
# unattended run takes the default. The default itself lives in one place —
# `TINYTITAN_DEFAULT_PORT` in tinytitan_models.sh — so a changed default cannot
# leave one script pointing where the others do not.
if [[ -n "$PORT_ARG" ]]; then
  PORT="$PORT_ARG"
elif [[ -n "${TINYTITAN_PORT:-}" ]]; then
  PORT="$TINYTITAN_PORT"
elif (( INTERACTIVE )); then
  echo ""
  echo "Port for the server?"
  echo "  Clients reach the API at http://127.0.0.1:<port>. ${TINYTITAN_DEFAULT_PORT} is the default;"
  echo "  change it when something else already holds that port, or to run a"
  echo "  second model beside this one."
  printf "Port [1-65535] (default %s): " "$TINYTITAN_DEFAULT_PORT"
  read -r port_choice || exit 1
  PORT="${port_choice:-$TINYTITAN_DEFAULT_PORT}"
else
  PORT="$TINYTITAN_DEFAULT_PORT"
fi
case "$PORT" in
  ''|*[!0-9]*) echo "unknown port: $PORT (a number 1-65535)" >&2; exit 2 ;;
esac
if (( PORT < 1 || PORT > 65535 )); then
  echo "unknown port: $PORT (a number 1-65535)" >&2; exit 2
fi
# Below 1024 macOS wants root. Say so now, rather than let the server exit with a
# bind error after the model has been chosen and the client wiring decided.
if (( PORT < 1024 )); then
  echo "NOTE: port $PORT is privileged; macOS requires root to bind it." >&2
fi
non_default_port=0
if (( PORT != TINYTITAN_DEFAULT_PORT )); then non_default_port=1; fi

# The context, KV and YaRN flags reach the GPU runtime only. Asking for one of
# them against a CPU model is a misunderstanding worth naming: the CPU engine
# takes its context from the model's own config (clamped by the backend) and
# has no quantized KV cache to pick a width for.
if [[ "$ENGINE" == "cpu" ]]; then
  inert_flags=()
  [[ -n "$KV_ARG" ]] && inert_flags+=(--kv)
  [[ -n "$CONTEXT_ARG" ]] && inert_flags+=(--context)
  [[ -n "$CONCURRENCY_ARG" ]] && inert_flags+=(--concurrency)
  (( YARN )) && inert_flags+=(--yarn)
  if (( ${#inert_flags[@]} > 0 )); then
    echo "NOTE: ${inert_flags[*]+"${inert_flags[*]}"} do not apply to the CPU engine; ignoring" >&2
    echo "      ${inert_flags[*]+"${inert_flags[*]}"} for $MODEL_NAME." >&2
  fi
  kv_bits=""; max_context=""; rope_scaling="none"
fi

if [[ "$MEMORY" == "1" ]]; then
  export TINYTITAN_MEMORY=1
fi

# ============================================================
# 8) The server command
# ============================================================

# Pinned for GPU models; per-install tuning comes from the profile.
prompt_cache_mode="multi-prefix"
prompt_cache_mib=256
case "$CACHE_ARG" in
  ""|multi-prefix) ;;
  off) prompt_cache_mode="off"; prompt_cache_mib=0 ;;
  *) echo "unknown --prompt-cache: $CACHE_ARG (multi-prefix or off)" >&2; exit 2 ;;
esac
gpu_runtime=(--prompt-cache-mode "$prompt_cache_mode" --prompt-cache-memory-mib "$prompt_cache_mib" --kv-bits "$kv_bits")
if [[ -n "$max_context" ]]; then
  gpu_runtime+=(--max-context "$max_context" --rope-scaling "$rope_scaling")
elif [[ "$rope_scaling" == "none" ]]; then
  gpu_runtime+=(--max-context 262144 --rope-scaling none)
fi
if [[ -n "$ram_gb" && "$MODEL_BACKEND" != "cpu" ]]; then
  gpu_runtime+=(--ram-budget "${ram_gb}G")
fi
# The draft head shares the target's embedding and head on the GPU path, so it
# attaches there only; the CPU engine has no speculative path to attach to.
if [[ -n "$MTP_MODEL_ARG" ]]; then
  if [[ "$MODEL_BACKEND" == "cpu" ]]; then
    echo "ERROR: --mtp-model does not apply to the CPU engine (no speculative path)" >&2
    exit 2
  fi
  if [[ ! -f "$MTP_MODEL_ARG/manifest.json" ]]; then
    echo "ERROR: MTP draft head not found at $MTP_MODEL_ARG" >&2
    exit 2
  fi
  gpu_runtime+=(--mtp-model "$MTP_MODEL_ARG" --mtp-memory-mib "${MTP_MEMORY_ARG:-384}")
fi

if (( dynamic )); then
  server_cmd=("$BINARY" --models-dir "$MODELS_DIR" --model "$MODEL_ID_LAUNCH" --reasoning "$thinking_level" --port "$PORT")
else
  server_cmd=("$BINARY" --model "$MODEL_DIR" --port "$PORT" --thinking "$( [[ "$thinking_level" == off ]] && echo off || echo on )")
fi

if [[ "$MODEL_BACKEND" == "cpu" ]]; then
  if (( dynamic )); then
    # The catalog knows which engine each model uses; --cpu is refused here.
    runtime_note="CPU backend | no prompt cache | context clamped by the backend | sampling from the model"
  else
    server_cmd+=(--cpu)
    runtime_note="CPU backend | no prompt cache | context clamped by the backend | sampling from the model"
  fi
  # Pin the single-generation width the CPU engine runs at anyway, so the
  # server's admission window matches what this launcher just told the operator.
  server_cmd+=(--max-concurrent-sequences 1)
  runtime_note="$runtime_note | one generation at a time"
else
  server_cmd+=("${gpu_runtime[@]+"${gpu_runtime[@]}"}" --max-concurrent-sequences "$concurrency")
  mtp_note="off"
  [[ -n "$MTP_MODEL_ARG" ]] && mtp_note="on (draft head)"
  # The cache mode in force, not the one asked for: above one slot the server
  # switches the session-wide cache off, and a summary that said "multi-prefix"
  # while the server ran `prompt_cache=off` would be a misreport.
  cache_note="$prompt_cache_mode"
  if (( concurrency > 1 )); then cache_note="off (above 1 at once)"; fi
  runtime_note="context ${max_context:-262144} | KV ${kv_bits}-bit | cache ${cache_note} | MTP ${mtp_note}"
  if (( concurrency > 1 )); then
    runtime_note="$runtime_note | ${concurrency} at once (yours)"
  else
    runtime_note="$runtime_note | one generation at a time"
  fi
  if [[ -n "$ram_gb" ]]; then
    runtime_note="$runtime_note | RAM target ${ram_gb} GB (yours)"
  else
    runtime_note="$runtime_note | expert cache, prefetch, sampling from the model profile"
  fi
fi

# The base API id is what the catalog calls the install (it ends in the
# routed-expert width, e.g. ornith-1.5-35b-a3b_8-Bit). The "<id>-fast" alias
# serves the same weights with the CLI-strip heuristic instead of the agentic
# tool loop.
# The API id is whatever the server advertises (it ends in the routed-expert
# width, e.g. ornith-1.5-35b-a3b_8-Bit; the bare name is not accepted). It is
# read back from /v1/models once the server is up, so the client config is
# written with the id that really exists rather than a guess.
resolve_launch_model() {
  local base="$1"
  if [[ "$model_word" == "fast" ]]; then
    launch_model="${base}-fast"
    api_model_note="(fast alias, seconds-per-answer chat)"
  else
    launch_model="$base"
    api_model_note="(full agent loop)"
  fi
}
launch_model="$MODEL_ID"
api_model_note=""

# ============================================================
# 9) Printing
# ============================================================

print_setup() {
  echo ""
  echo "============================================================"
  echo " TinyTitanServer ready — $MODEL_NAME ${MODEL_QUANT}-bit ($( [[ "$MODEL_BACKEND" == cpu ]] && echo CPU || echo GPU ))"
  echo "                                            answers ${mode_word}, thinking ${think_word}"
  echo "============================================================"
  echo ""
  if [[ "$API" == "anthropic" ]]; then
    echo "Anthropic API setup — point any Anthropic Messages client at this:"
    echo "  ANTHROPIC_BASE_URL=http://127.0.0.1:${PORT}"
    echo "  ANTHROPIC_API_KEY=tinytitan   (any value; the server does not authenticate)"
    echo "  Model:      $launch_model $api_model_note"
    echo "  Endpoint:   POST /v1/messages"
  else
    echo "OpenAI API setup — point any OpenAI-compatible client at this:"
    echo "  Base URL:   http://127.0.0.1:${PORT}/v1"
    echo "  API key:    any value (the server does not authenticate)"
    echo "  Model:      $launch_model $api_model_note"
    echo "  Endpoints:  POST /v1/chat/completions, POST /v1/responses"
  fi
  if (( WEB )); then
    echo ""
    echo "DeepSeek Harness is opening in your default browser: a prompt box"
    echo "pointed at this server. It is this install's own copy under"
    # shellcheck disable=SC2088  # prose in a message, not a path being used
    echo "~/.tinytitan (port ${TINYTITAN_DSH_PORT:-7788}, or the next free one), so a"
    echo "DeepSeek Harness you run yourself is not touched."
    echo "Engine is in fast mode: the coding agent's system prompt and tool"
    echo "definitions are stripped before prefill, so answers start in seconds."
  fi
  # The launcher's own route is written by `dsh_local.sh ensure --port` on the
  # --web path, so the "regenerate it yourself" advice below would be noise there.
  if (( non_default_port && ! WEB )); then
    echo ""
    echo "Port $PORT is not the default, so anything that assumes ${TINYTITAN_DEFAULT_PORT} needs"
    echo "telling. The DeepSeek Harness route is one of those; regenerate it"
    echo "against this server with:"
    echo "  TINYTITAN_PORT=$PORT tools/dsh_route.sh --write"
  fi
  echo ""
  if (( dynamic )); then
    echo "Every installed model is available by name through the API: send any"
    echo "id from GET /v1/models and the server switches to it on demand, keeping"
    echo "one model resident at a time (a switch reloads weights)."
  else
    echo "Single-model mode (no catalog): this server serves only the model above;"
    echo "switching models needs a restart."
  fi
  echo "  Runtime:    $runtime_note"
  echo "  Engine:     $engine_line"
  if [[ "$MEMORY" == "1" ]]; then
    echo "  Memory:     on (persistent, repo-scoped; guard on)"
  fi
  echo ""
  echo "Model: $MODEL_DIR"
  echo "Thinking: $think_word | RAM: $ram_note | At once: $concurrency | Port: $PORT | Ctrl-C to stop"
  if (( ram_over_rule )); then
    warn_red "WARNING: ${ram_gb} GB of RAM is over the ${ram_rule_gb} GB this launcher"
    warn_red "         recommends for this Mac (50%). Expect swapping, a less stable"
    warn_red "         system and slower tokens."
  fi
  if (( concurrency > 1 )); then
    warn_red "WARNING: ${concurrency} generations are served at once. Each holds its own KV"
    warn_red "         cache, so memory use is up to ${concurrency}x one sequence, and one GPU"
    warn_red "         shared ${concurrency} ways makes each answer about ${concurrency}x slower."
    warn_red "         The prompt cache is off above 1."
    if (( concurrency > 16 )); then
      warn_red "         This Mac may build fewer slots: see the server's 'batch width' line."
    fi
  fi
  echo "============================================================"
  echo ""
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo ""
  rule
  echo "DRY RUN — nothing started, nothing stopped."
  rule
  echo "Client:     $(client_label "$CLIENT")"
  echo "Server cmd:"
  printf '  '; printf '%q ' "${server_cmd[@]+"${server_cmd[@]}"}"; echo ""
  echo ""
  echo "Client setup the launcher would write/use:"
  case "$CLIENT" in
    server)   echo "  Server only; nothing to configure." ;;
    codex)    echo "  ~/.codex-tinytitan/config.toml -> $launch_model via $API at port $PORT" ;;
    claude)   echo "  ANTHROPIC_BASE_URL=http://127.0.0.1:$PORT, model $launch_model" ;;
    qwen)     echo "  ~/.qwen-tinytitan/settings.json -> $launch_model at port $PORT" ;;
    opencode) echo "  ~/.config/opencode/opencode.jsonc provider 'tinytitan' (model $launch_model)" ;;
    zed)      echo "  ~/.config/zed/settings.json openai_compatible 'tinytitan' (model $launch_model)" ;;
  esac
  echo ""
  print_setup
  exit 0
fi

# Persistent memory, when TINYTITAN_MEMORY=1. The workspace is the directory this
# was launched from, so each repository keeps its own memory.
tinytitan_export_memory_environment "$PWD"

if [[ "$MEMORY" == "1" ]]; then
  echo "Memory: on (in-process, ${TINYTITAN_MEMORY_DIR}${TINYTITAN_MEMORY_CACHE_MIB:+, cap ${TINYTITAN_MEMORY_CACHE_MIB} MiB}, workspace $(basename "$PWD"))"
fi

# ============================================================
# 10) Start the server
# ============================================================

# Stop only a stale TinyTitanServer on this port -- never an unrelated process --
# then wait until the port actually frees, so a slow teardown cannot race the
# new server's bind.
if lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Stopping the existing TinyTitanServer on port $PORT..."
  for pid in $(lsof -ti :"$PORT" -sTCP:LISTEN); do
    if ps -p "$pid" -o command= | grep -q TinyTitanServer; then
      kill "$pid"
    else
      echo "  skipping non-TinyTitanServer pid $pid on port $PORT" >&2
    fi
  done
  for _ in $(seq 1 100); do
    lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break
    sleep 0.1
  done
  # If it is still held, the holder is not ours — the loop above skips those on
  # purpose. Name it and stop, instead of letting the server fail to bind after
  # the model has been chosen and the client wiring decided.
  if lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "ERROR: port $PORT is held by a process that is not a TinyTitanServer:" >&2
    lsof -i :"$PORT" -sTCP:LISTEN | tail -n +2 | sed 's/^/  /' >&2
    echo "       Choose another port with --port, or stop that process yourself." >&2
    exit 2
  fi
fi

if [[ ! -e "$MODEL_DIR" ]]; then
  # Reached when the catalog named a directory that is not there (removed
  # between the catalog read and here). The install path above is the normal
  # route, so this only has to say which command fixes it.
  echo "ERROR: $MODEL_NAME ${MODEL_QUANT}-bit not found at $MODEL_DIR" >&2
  missing_key="$(tinytitan_install_key "${TINYTITAN_MODEL_STEM:-}" "${MODEL_QUANT%bit}" 2>/dev/null || true)"
  if [[ -n "$missing_key" ]]; then
    echo "       Fetch it with:  tools/install_models.sh $missing_key" >&2
  else
    echo "       Install it first: tools/install_models.sh (see --help for the target names)" >&2
  fi
  exit 1
fi

echo "Starting TinyTitanServer ($MODEL_NAME ${MODEL_QUANT}-bit, $model_word, $mode_word, thinking $think_word) on port $PORT..."

# --web runs the engine in **fast mode**: TINYTITAN_STRIP_CLI_PROMPT enables the
# same CLI-strip the "<model>-fast" alias selects, for every request. The harness
# is a coding agent, so each turn carries a multi-thousand-token system prompt
# and 27 tool definitions that a plain prompt box never needs; stripping them
# leaves the real user/assistant conversation — a few hundred tokens instead of
# several thousand — which is the difference between an answer in seconds and the
# page sitting on "Deep diving...". The engine logs a strip report per request,
# so what was removed is visible rather than assumed.
if (( WEB )); then
  TINYTITAN_STRIP_CLI_PROMPT=1 "${server_cmd[@]+"${server_cmd[@]}"}" &
else
  "${server_cmd[@]+"${server_cmd[@]}"}" &
fi
server_pid=$!

# The trap belongs here, with the server, not only on the client path below.
# A harness starts this script and later signals it; without a trap on this
# path the backgrounded server is orphaned, which leaves a model process
# running -- and this project's own guard then refuses the next golden or gate.
cleanup() { kill "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Wait for the API to answer, not for a fixed number of seconds.
for _ in $(seq 1 240); do
  curl -s --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
  kill -0 "$server_pid" 2>/dev/null || break
  sleep 2
done
if ! models_json="$(curl -s --max-time 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)" || [[ -z "$models_json" ]]; then
  echo "ERROR: TinyTitanServer did not come up on port $PORT" >&2
  kill "$server_pid" 2>/dev/null || true
  exit 1
fi
# The id the server actually advertises for this install. The catalog knows
# it when it is available; otherwise ask the running server.
if (( dynamic )); then
  MODEL="$MODEL_ID_LAUNCH"
else
  MODEL="$(printf '%s' "$models_json" \
    | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]+"' \
    | sed -E 's/.*"([^"]+)"$//' | grep -v -- '-fast$' | head -1)"
fi
if [[ -z "$MODEL" ]]; then
  echo "WARNING: could not read the model id from /v1/models" >&2
  MODEL="$MODEL_ID"
fi
resolve_launch_model "$MODEL"

# ============================================================
# 11) Client configuration
# ============================================================

BASE_URL="http://127.0.0.1:${PORT}/v1"

# Timestamped backup, once per file per run, before anything is rewritten.
backup_once() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  local backup="${file}.tinytitan-backup"
  [[ -f "$backup" ]] || cp "$file" "$backup"
}

configure_codex() {
  local dir="${CODEX_HOME_TinyTitan:-$HOME/.codex-tinytitan}"
  mkdir -p "$dir"
  backup_once "$dir/config.toml"
  cat > "$dir/config.toml" <<EOF
# Written by TinyTitan tools/server_launcher.sh
model = "$launch_model"
model_provider = "tinytitan"

[model_providers.tinytitan]
name = "TinyTitan"
base_url = "$BASE_URL"
wire_api = "responses"
EOF
  export CODEX_HOME="$dir"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
  echo "  Codex configured: $dir/config.toml"
}

configure_claude() {
  # Claude Code reads the Anthropic surface from the environment, so there is
  # no file to write. The two extra model variables matter: without them it
  # asks for a claude-* id for its background tasks and gets a 404.
  export ANTHROPIC_BASE_URL="http://127.0.0.1:${PORT}"
  export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-tinytitan}"
  export ANTHROPIC_MODEL="$launch_model"
  export ANTHROPIC_DEFAULT_HAIKU_MODEL="$launch_model"
  export ANTHROPIC_SMALL_FAST_MODEL="$launch_model"
  echo "  Claude Code configured: ANTHROPIC_BASE_URL=http://127.0.0.1:${PORT}"
}

configure_qwen() {
  # Qwen Code reads JSON settings from $QWEN_HOME (default ~/.qwen/), not a
  # Codex-style TOML. A dedicated home leaves the user's real qwen-code
  # config (providers, keys, memories) untouched.
  local dir="${QWEN_HOME_TinyTitan:-$HOME/.qwen-tinytitan}"
  mkdir -p "$dir"
  backup_once "$dir/settings.json"
  cat > "$dir/settings.json" <<EOF
{
  "modelProviders": {
    "openai": [
      {
        "id": "$launch_model",
        "name": "[TinyTitan] $launch_model",
        "baseUrl": "$BASE_URL",
        "description": "TinyTitan local server",
        "envKey": "OPENAI_API_KEY"
      }
    ]
  },
  "security": {
    "auth": {
      "selectedType": "openai"
    }
  },
  "model": {
    "name": "$launch_model"
  },
  "memory": {
    "enableManagedAutoMemory": false,
    "enableManagedAutoDream": false,
    "enableAutoSkill": false
  }
}
EOF
  export QWEN_HOME="$dir"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
  # TinyTitan's cold prefill of qwen-code's large system prompt can exceed
  # qwen-code's default 240s stream-idle timeout; disable it so the request is
  # not aborted mid-generation. Also disable the 900s stream-lifetime cap,
  # which would otherwise abort long reasoning generations mid-answer.
  export QWEN_STREAM_IDLE_TIMEOUT_MS=0
  export QWEN_STREAM_MAX_LIFETIME_MS=0
  echo "  Qwen Code configured: $dir/settings.json"
}

# Merge the TinyTitan provider into a JSONC settings file, keeping everything
# else. Comments and trailing commas are tolerated on read; the file is
# rewritten as plain JSON (Zed and OpenCode both accept that) with a backup
# kept beside it. Each client has its own schema, so the shape is passed in:
#   zed       language_models.openai_compatible.<id> = {api_url, available_models}
#   opencode  provider.<id> = {npm, name, options.baseURL, models}
merge_client_config() {
  local file="$1" schema="$2" provider="$3" model="$4" context="$5"
  mkdir -p "$(dirname "$file")"
  backup_once "$file"
  TINYTITAN_MERGE_FILE="$file" TINYTITAN_MERGE_SCHEMA="$schema" TINYTITAN_MERGE_PROVIDER="$provider" \
  TINYTITAN_MERGE_MODEL="$model" TINYTITAN_MERGE_URL="$BASE_URL" TINYTITAN_MERGE_CONTEXT="$context" \
  python3 - <<'PY'
import json, os, re, sys

path = os.environ["TINYTITAN_MERGE_FILE"]
schema = os.environ["TINYTITAN_MERGE_SCHEMA"]
provider = os.environ["TINYTITAN_MERGE_PROVIDER"]
model = os.environ["TINYTITAN_MERGE_MODEL"]
base_url = os.environ["TINYTITAN_MERGE_URL"]
context = int(os.environ["TINYTITAN_MERGE_CONTEXT"])


def strip_comments(text):
    """JSONC -> JSON: drop // and /* */ comments outside strings."""
    out, i, n, in_str, esc = [], 0, len(text), False, False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i)
            i = n if j == -1 else j + 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


try:
    raw = open(path, encoding="utf-8").read()
except FileNotFoundError:
    raw = "{}"
text = re.sub(r",(\s*[}\]])", r"\1", strip_comments(raw))
try:
    data = json.loads(text) if text.strip() else {}
except json.JSONDecodeError as error:
    print(f"  WARNING: {path} is not parseable ({error}); left untouched.", file=sys.stderr)
    sys.exit(0)
if not isinstance(data, dict):
    print(f"  WARNING: {path} is not a JSON object; left untouched.", file=sys.stderr)
    sys.exit(0)


def table(parent, key):
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


if schema == "opencode":
    block = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "TinyTitan (local)",
        "options": {"baseURL": base_url, "apiKey": "tinytitan"},
        "models": {
            model: {
                "name": f"TinyTitan — {model}",
                "limit": {"context": context, "output": 65536},
            }
        },
    }
    table(table(data, "provider"), provider).update(block)
else:  # zed
    block = {
        "api_url": base_url,
        "available_models": [
            {
                "name": model,
                "display_name": f"TinyTitan — {model}",
                "max_tokens": context,
                "capabilities": {
                    "tools": True,
                    "images": False,
                    "parallel_tool_calls": False,
                    "prompt_cache_key": False,
                    "chat_completions": True,
                    "interleaved_reasoning": False,
                    "max_tokens_parameter": False,
                },
            }
        ],
    }
    table(table(data, "language_models"), "openai_compatible")[provider] = block

with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
PY
}

configure_opencode() {
  local file="$HOME/.config/opencode/opencode.jsonc"
  # OpenCode has no environment override for a provider, so the provider block
  # is written into its config. The provider is named "tinytitan" so nothing of
  # the user's own "openai" setup is touched.
  merge_client_config "$file" opencode tinytitan "$launch_model" "${max_context:-262144}"
  echo "  Pick \"$launch_model\" under the tinytitan provider in OpenCode's model list."
}

configure_zed() {
  local file="$HOME/.config/zed/settings.json"
  merge_client_config "$file" zed tinytitan "$launch_model" "${max_context:-262144}"
  echo "  In Zed: Settings -> AI -> LLM Providers; choose the tinytitan provider's model."
}

case "$CLIENT" in
  server)   : ;;
  codex)    configure_codex ;;
  claude)   configure_claude ;;
  qwen)     configure_qwen ;;
  opencode) configure_opencode ;;
  zed)      configure_zed ;;
esac

# ============================================================
# 12) Hand over to the client
# ============================================================

# The model is loaded and the served id is real. State what was built and where
# to point a client before anything takes over the terminal: for a server-only
# run this banner IS the handover, and it is the one place the base URL, the
# served model id and the port are printed together. It was reachable only
# through --dry-run, so the default `server` path — which is what the installer
# hands off to — started a server and never told the user its address.
print_setup

# --web: bring up TinyTitan's own DeepSeek Harness and let it take the terminal.
# It opens the default browser itself and prints the tokenised URL, so there is
# nothing else to tell the user. `ensure` writes the route against *this*
# server's port, which is why it runs here rather than at install time: the port
# is not known until now. The child is not `exec`ed — the trap above has to
# survive so Ctrl-C stops the model with the page.
if (( WEB )); then
  # `--model` sets the harness's own default model: it ships pointing at
  # DeepSeek's hosted route, so without this the window opens on a provider we
  # have no key for. The launcher knows the served id; the harness does not.
  # `--reasoning` has to travel with it: the route's declared level is what the
  # harness asks for per request, and a route that says "think" against a server
  # started thinking-off is the page hanging on "Deep diving..." forever.
  if ! "$SCRIPT_DIR/dsh_local.sh" ensure --port "$PORT" --model "$MODEL" \
       --reasoning "$thinking_level"; then
    echo "ERROR: could not set up DeepSeek Harness (see above)." >&2
    echo "       The server is still running on http://127.0.0.1:${PORT}/v1 —" >&2
    echo "       point any OpenAI-compatible client at it." >&2
    wait "$server_pid"
    exit $?
  fi
  # Warm the expert cache before the window opens.
  #
  # The first request after a load pays for every expert sweep it misses.
  # Measured on the 35B-A3B 4-bit: the same ~4.2k-token prompt took **161s cold**
  # and **72s** once the cache held the working set. A throwaway prefill moves
  # that cost to startup, where this script can say what it is doing, instead of
  # leaving it on the person's first question with the page stuck on "Deep
  # diving...". Set TINYTITAN_WARM=0 to skip it, or TINYTITAN_WARM_TOKENS to
  # change the size.
  if [[ "${TINYTITAN_WARM:-1}" != "0" ]]; then
    warm_tokens="${TINYTITAN_WARM_TOKENS:-4000}"
    echo
    echo "Warming the expert cache (one throwaway prompt, ~$warm_tokens tokens;"
    echo "this is the slow part of the first answer, moved here)."
    # Built in shell rather than by python3 for two reasons. It keeps a python3
    # requirement out of a path that has to work on a Mac that has none — and a
    # single-quoted heredoc containing an apostrophe inside `$( )` is a **bash 3.2
    # parse error**, and /bin/bash is 3.2 on a factory Mac, so the script failed
    # to parse at all before it ran a line. Reproduced in isolation; the same
    # heredoc parses under 5.x, which is why it went unnoticed here.
    warm_sentence="The quick brown fox jumps over the lazy dog. "
    warm_repeat=$(( warm_tokens / 11 ))
    (( warm_repeat < 1 )) && warm_repeat=1
    warm_text=""
    warm_i=0
    while (( warm_i < warm_repeat )); do
      warm_text="${warm_text}${warm_sentence}"
      warm_i=$(( warm_i + 1 ))
    done
    warm_body="$(printf '{"model":"%s","messages":[{"role":"user","content":"%s\\nReply with the single word: ready"}],"max_tokens":1,"stream":false}' "$MODEL" "$warm_text")"
    warm_started="$(date +%s)"
    if curl -s --max-time "${TINYTITAN_WARM_TIMEOUT:-900}" \
         "http://127.0.0.1:${PORT}/v1/chat/completions" \
         -H 'Content-Type: application/json' -d "$warm_body" >/dev/null 2>&1; then
      echo "Expert cache warm ($(( $(date +%s) - warm_started ))s)."
    else
      warn_red "The warm-up did not finish; the first answer will be slower."
    fi
  fi
  echo
  echo "Opening DeepSeek Harness in your default browser..."
  # `--port` matters here as much as it does to `ensure`: `web` exports it as
  # TINYTITAN_PORT so the plugin's boot-time route refresh keeps this server's
  # address instead of rewriting it to the default 8080.
  "$SCRIPT_DIR/dsh_local.sh" web --port "$PORT"
  exit $?
fi

# A server-only run stays in the foreground so Ctrl-C stops the model.
if [[ "$CLIENT" == "server" ]]; then
  wait "$server_pid"
  exit $?
fi

client_bin() {
  local id="$1" var="" candidate override
  case "$id" in
    codex)    var=CODEX ;;
    claude)   var=CLAUDE ;;
    qwen)     var=QWEN ;;
    opencode) var=OPENCODE ;;
    zed)      var=ZED ;;
  esac
  if [[ -n "$var" && -n "${!var-}" ]]; then echo "${!var}"; return 0; fi
  # The binary names the shared catalogue carries, first one on PATH.
  for candidate in $(tinytitan_client_binaries "$id"); do
    if override="$(command -v "$candidate" 2>/dev/null)"; then
      echo "$override"
      return 0
    fi
  done
  # Where each client installs itself when PATH does not carry it.
  case "$id" in
    codex)    echo "$HOME/.local/bin/codex" ;;
    qwen)     echo "$HOME/.qwen-code/bin/qwen-code" ;;
    opencode) echo "/opt/homebrew/bin/opencode" ;;
    zed)      echo "/usr/local/bin/zed" ;;
    claude)   echo "$HOME/.local/bin/claude" ;;
  esac
}

BIN="$(client_bin "$CLIENT")"
if [[ ! -x "$BIN" ]]; then
  echo "" >&2
  echo "WARNING: $(client_label "$CLIENT") was not found at $BIN." >&2
  echo "         The server is running on http://127.0.0.1:${PORT} — point your" >&2
  echo "         client at that address, or install it and re-run." >&2
  wait "$server_pid"
  exit $?
fi

echo "Launching $(client_label "$CLIENT")..."
echo ""

# The model keeps running after the client exits; the next launcher run stops
# it on this port and starts fresh. The trap was installed with the server
# above, so the server-only path is covered too; this keeps the two lifetimes
# matched when the person Ctrl-Cs the client instead.

case "$CLIENT" in
  codex)    "$BIN" ;;
  claude)   "$BIN" ;;
  qwen)     "$BIN" -i ;;
  opencode) "$BIN" ;;
  zed)      "$BIN" "${ZED_PROJECT:-$PWD}" ;;
esac
