#!/usr/bin/env bash
# Write (or print) the DeepSeek Harness route for the TinyTitan server.
#
#   tools/dsh_route.sh                     # print the block (copy it yourself)
#   tools/dsh_route.sh --write             # install it into ~/.dsh/settings.yaml
#   tools/dsh_route.sh --models qwen38 4   # only installs whose family/key matches
#   tools/dsh_route.sh --reasoning off     # route default level (default: medium)
#
# The block is generated from the installs this checkout actually has: the
# server's own catalog (TinyTitanServer --catalog --models-dir models, or
# TINYTITAN_CATALOG_JSON when the server is not built) supplies each served id, its
# name and the thinking levels its chat template renders, so the harness's model
# picker follows `models/` instead of a hand-written list that goes stale.
#
# Three switches in the block are not obvious and are easy to get wrong by hand:
#
#   compat.thinkingFormat: chat-template   the only place TinyTitan reads
#                                          enable_thinking; pi-ai's `qwen`
#                                          format sends the switch top-level,
#                                          where TinyTitan ignores it
#   headers.authorization                  pi-ai refuses a keyless route
#                                          ("No API key for provider"); TinyTitan
#                                          accepts and ignores the header
#   streamIdleTimeoutMs                    TinyTitan says nothing until the first
#                                          token, and a cold local prefill
#                                          outlives pi-ai's five-minute default
#
# The route's `reasoning` default is what the *auxiliary* calls use too
# (compaction and session titles name no level of their own), so `medium` means
# a thinking summariser. `--reasoning off` keeps them unthinking at the cost of
# chat starting unthinking too; the plugins/dsh-tinytitan plugin is the way to have
# both (it forces thinking off for those calls).
#
# Flags:
#   --port <n>        the port the server serves on (default 8080, TINYTITAN_PORT)
#   --context <n>     contextWindow to declare (default 262144, the launcher's pin)
#   --max-tokens <n>  maxTokens to declare (default 32768)
#   --reasoning <l>   route default level: off|low|medium|xhigh (default medium)
#   --provider <name> provider route name (default tinytitan)
#   --models <a> [b]  only these ids, install keys or families
#   --thinking <l,…>  levels to declare when --from-server has to guess them
#   --from-server     read the served ids from GET /v1/models instead of the
#                     catalog (levels then come from --thinking, default off,on)
#   --print           print the block (default)
#   --write           replace the llm-pi-ai section in the DSH settings file,
#                     after backing it up
#   --settings <path> the settings file (default $DSH_HOME/settings.yaml)
#   --help, -h
#
# Exits 2 on a bad argument or when no install can be described, so a script can
# tell "nothing to write" from "wrote nothing".
set -euo pipefail

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# A checkout builds into `.build/release` and keeps models in `models/`; an
# install from the release tarball keeps both one level up (`~/.tinytitan/bin`,
# `~/.tinytitan/models`) and names them through TINYTITAN_BIN_DIR and
# TINYTITAN_MODELS_DIR. The launcher the installer writes sets those, but the
# installer also tells a person to "re-run tools/dsh_local.sh ensure" from a
# shell, where they are unset — so the installed layout is checked before giving
# up, and the failure a user sees is about a missing model rather than a missing
# path the installer put somewhere else.
if [[ -n "${TINYTITAN_BIN_DIR:-}" ]]; then
  BINARY="$TINYTITAN_BIN_DIR/TinyTitanServer"
elif [[ -x "$BASE_DIR/.build/release/TinyTitanServer" ]]; then
  BINARY="$BASE_DIR/.build/release/TinyTitanServer"
elif [[ -x "$BASE_DIR/../bin/TinyTitanServer" ]]; then
  BINARY="$(cd "$BASE_DIR/.." && pwd)/bin/TinyTitanServer"
else
  BINARY="$BASE_DIR/.build/release/TinyTitanServer"
fi
if [[ -n "${TINYTITAN_MODELS_DIR:-}" ]]; then
  MODELS_DIR="$TINYTITAN_MODELS_DIR"
elif [[ -d "$BASE_DIR/models" ]]; then
  MODELS_DIR="$BASE_DIR/models"
elif [[ -d "$BASE_DIR/../models" ]]; then
  MODELS_DIR="$(cd "$BASE_DIR/.." && pwd)/models"
else
  MODELS_DIR="$BASE_DIR/models"
fi
# shellcheck source=tools/tinytitan_models.sh
source "$SCRIPT_DIR/tinytitan_models.sh"

PORT="${TINYTITAN_PORT:-$TINYTITAN_DEFAULT_PORT}"
CONTEXT=262144
MAX_TOKENS=32768
REASONING=medium
PROVIDER=tinytitan
MODE=print
SETTINGS="${DSH_HOME:-$HOME/.dsh}/settings.yaml"
FROM_SERVER=0
THINKING="off,on"
FILTER=()

die() { echo "dsh_route: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)      PORT="${2:?--port needs a value}"; shift 2 ;;
    --context)   CONTEXT="${2:?--context needs a value}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:?--max-tokens needs a value}"; shift 2 ;;
    --reasoning) REASONING="${2:?--reasoning needs a level}"; shift 2 ;;
    --provider)  PROVIDER="${2:?--provider needs a name}"; shift 2 ;;
    --models)    shift; while [[ $# -gt 0 && "$1" != --* ]]; do FILTER+=("$1"); shift; done ;;
    --thinking)  THINKING="${2:?--thinking needs a list}"; shift 2 ;;
    --from-server) FROM_SERVER=1; shift ;;
    --settings)  SETTINGS="${2:?--settings needs a path}"; shift 2 ;;
    --print)     MODE="print"; shift ;;
    --write)     MODE="write"; shift ;;
    --help|-h)   usage; exit 0 ;;
    *)           die "unknown option: $1 (try --help)" ;;
  esac
done

case "$PORT" in *[!0-9]*|"") die "port must be a number: $PORT" ;; esac
case "$CONTEXT" in *[!0-9]*|"") die "context must be a number: $CONTEXT" ;; esac
case "$MAX_TOKENS" in *[!0-9]*|"") die "max-tokens must be a number: $MAX_TOKENS" ;; esac
case "$REASONING" in
  off|minimal|low|medium|high|xhigh|max) : ;;
  *) die "reasoning must be a pi-ai level (off|low|medium|xhigh|...), not '$REASONING'" ;;
esac

# --- The models to declare --------------------------------------------------

ids=(); names=(); levels=(); families=(); paths=()
add_model() { ids+=("$1"); names+=("$2"); levels+=("$3"); families+=("$4"); paths+=("${5:-}"); }

if (( FROM_SERVER )); then
  # A server that is already running is the cheapest source of the *served*
  # ids; it cannot say which levels a template renders, so --thinking does.
  listing="$(curl -sS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)" \
    || die "no server answered on http://127.0.0.1:${PORT}/v1/models"
  # A `while read` loop rather than `mapfile`: the latter is bash 4+, and
  # /bin/bash is 3.2 on a factory Mac. This is the `--from-server` path, which is
  # exactly what a launcher without a checkout uses.
  served=()
  # A served id is printed verbatim into the settings block, so it has to be one
  # plain token. The catalogue parser refuses an empty or multi-line field, and the
  # plugin writer does the same through its own validator; this path asked neither
  # question, so an id carrying a newline became two rows here and one carrying
  # YAML punctuation changed what the block declares. The validator prints nothing
  # unless every id passes, and its reason becomes this script's reason, so an
  # operator is not told "listed no models" when a model was listed and refused.
  if ! validated="$(printf '%s' "$listing" | python3 -c 'import json,re,sys
ids = [str(m["id"]) for m in json.load(sys.stdin)["data"]]
bad = sorted(i for i in ids if not re.fullmatch(r"[A-Za-z0-9._-]+", i))
if bad:
    raise SystemExit("ids that are not plain tokens: " + repr(bad))
for i in ids:
    print(i)' 2>&1)"; then
    die "the server on port $PORT answered with ${validated}"
  fi
  while IFS= read -r served_id; do
    [[ -n "$served_id" ]] && served+=("$served_id")
  done <<< "${validated}"
  (( ${#served[@]} > 0 )) || die "the server on port $PORT listed no models"
  for id in "${served[@]+"${served[@]}"}"; do
    # A served id carries the width and nothing else human-readable is known
    # here, so the name is the id without its `_N-Bit` suffix and the label adds
    # the width back once.
    stem="$id"
    case "$id" in *_[0-9]*-Bit) stem="${id%_*}" ;; esac
    add_model "$id" "$stem" "$THINKING" "-" ""
  done
else
  if ! tinytitan_load_catalog "$BINARY" "$MODELS_DIR"; then
    die "no catalog ($TINYTITAN_CATALOG_ERROR); build the server, or pass --from-server"
  fi
  for (( i = 0; i < ${#TINYTITAN_CAT_ID[@]}; i++ )); do
    # The static fallback has no ids ("-"): a route cannot name a model the
    # server will not answer to.
    [[ "${TINYTITAN_CAT_ID[$i]}" == "-" ]] && continue
    add_model "${TINYTITAN_CAT_ID[$i]}" "${TINYTITAN_CAT_NAME[$i]}" \
      "${TINYTITAN_CAT_THINKING[$i]}" "${TINYTITAN_CAT_FAMILY[$i]}" "${TINYTITAN_CAT_PATH[$i]}"
  done
fi
(( ${#ids[@]} > 0 )) || die "the catalog describes no servable install under $MODELS_DIR"

if (( ${#FILTER[@]} > 0 )); then
  # A filter may be a served id, an install key (qwen38), a family, or part of
  # the install directory's name. An install key resolves through the shared
  # catalogue to its stem, which is what the directory is called.
  patterns=()
  for needle in "${FILTER[@]+"${FILTER[@]}"}"; do
    patterns+=("$needle")
    if stem="$(tinytitan_resolve_model "$needle" 2>/dev/null && echo "$TINYTITAN_MODEL_STEM")"; then
      patterns+=("$stem")
    fi
  done
  keep_ids=(); keep_names=(); keep_levels=(); keep_families=(); keep_paths=()
  for (( i = 0; i < ${#ids[@]}; i++ )); do
    wanted=0
    haystack="${ids[$i]} ${names[$i]} ${families[$i]} $(basename "${paths[$i]:-}")"
    for pattern in "${patterns[@]+"${patterns[@]}"}"; do
      case "$haystack" in *"$pattern"*) wanted=1 ;; esac
    done
    (( wanted )) || continue
    keep_ids+=("${ids[$i]}"); keep_names+=("${names[$i]}")
    keep_levels+=("${levels[$i]}"); keep_families+=("${families[$i]}")
    keep_paths+=("${paths[$i]:-}")
  done
  (( ${#keep_ids[@]} > 0 )) || die "no install matches: ${FILTER[*]+"${FILTER[*]}"}"
  ids=("${keep_ids[@]+"${keep_ids[@]}"}"); names=("${keep_names[@]+"${keep_names[@]}"}")
  levels=("${keep_levels[@]+"${keep_levels[@]}"}"); families=("${keep_families[@]+"${keep_families[@]}"}")
  paths=("${keep_paths[@]+"${keep_paths[@]}"}")
fi

# --- The block --------------------------------------------------------------

# pi-ai's level vocabulary has no `on` (its levels are off|minimal|low|medium|
# high|xhigh|max), while a binary-thinking template renders exactly off|on.
# Such a family's thinking mode is therefore offered as `medium` with the wire
# value `on`: the picker shows one thinking choice, and TinyTitan reads `on`.
efforts_block() {
  local list="$1" level
  echo "          reasoningEfforts:"
  IFS=',' read -r -a split <<< "$list"
  for level in "${split[@]+"${split[@]}"}"; do
    level="${level// /}"
    [[ -z "$level" ]] && continue
    case "$level" in
      off) echo "            off:" ;;
      on)  echo "            medium: on" ;;
      *)   echo "            ${level}: ${level}" ;;
    esac
  done
}

# The label the DSH picker renders. It shows `name` and nothing else, while the
# catalogue's display name carries no width -- "Qwen 3.5 2B" names both the
# 4-bit and the 8-bit install -- so two rows arrived looking identical and the
# width could not be picked. The routed width is always on the id (`..._4-Bit`),
# which is where the plugin reads it too, so the two stay byte-identical.
route_label() {
  local name="$1" id="$2" bits=""
  case "$id" in
    *_[0-9]*-Bit) bits="${id##*_}"; bits="${bits%-Bit}" ;;
  esac
  if [[ -n "$bits" ]]; then printf '%s (%s-bit)' "$name" "$bits"
  else printf '%s' "$name"; fi
}

block=""
block+="# DeepSeek Harness route to the TinyTitan server on port ${PORT}."$'\n'
block+="# Generated by tools/dsh_route.sh from the installs under models/."$'\n'
block+="# ${#ids[@]} served model(s); settings.yaml is hot-reloaded."$'\n'
block+="llm-pi-ai:"$'\n'
block+="  providers:"$'\n'
block+="    ${PROVIDER}:"$'\n'
block+="      displayName: TinyTitan"$'\n'
block+="      api: openai-completions"$'\n'
block+="      baseURL: http://127.0.0.1:${PORT}/v1"$'\n'
block+="      # pi-ai refuses a keyless route (\"No API key for provider\");"$'\n'
block+="      # TinyTitan has no authentication and ignores the header."$'\n'
block+="      headers:"$'\n'
block+="        authorization: Bearer tinytitan-local"$'\n'
block+="      # Level for calls that name none: compaction and session titles."$'\n'
block+="      # Use \`off\` to keep those unthinking, or install plugins/dsh-tinytitan,"$'\n'
block+="      # which forces it for them without turning chat off."$'\n'
block+="      reasoning: ${REASONING}"$'\n'
block+="      # TinyTitan emits nothing until the first token; pi-ai's own default"$'\n'
block+="      # abandons an idle stream after five minutes."$'\n'
block+="      streamIdleTimeoutMs: 3600000"$'\n'
block+="      defaultContextWindow: ${CONTEXT}"$'\n'
block+="      defaultMaxTokens: ${MAX_TOKENS}"$'\n'
block+="      models:"$'\n'
for (( i = 0; i < ${#ids[@]}; i++ )); do
  block+="        - id: ${ids[$i]}"$'\n'
  block+="          name: $(route_label "${names[$i]}" "${ids[$i]}")"$'\n'
  block+="          contextWindow: ${CONTEXT}"$'\n'
  block+="          maxTokens: ${MAX_TOKENS}"$'\n'
  block+="$(efforts_block "${levels[$i]}")"$'\n'
  block+="          compat:"$'\n'
  block+="            # The only place TinyTitan reads the thinking switch; pi-ai's"$'\n'
  block+="            # \`qwen\` format sends it top-level, where TinyTitan ignores it."$'\n'
  block+="            thinkingFormat: chat-template"$'\n'
  block+="            chatTemplateKwargs:"$'\n'
  block+="              enable_thinking: { \$var: thinking.enabled }"$'\n'
  block+="              reasoning_effort: { \$var: thinking.effort }"$'\n'
  block+="            maxTokensField: max_tokens"$'\n'
  block+="            supportsUsageInStreaming: true"$'\n'
done

if [[ "$MODE" == "print" ]]; then
  printf '%s' "$block"
  exit 0
fi

# --- --write: replace the section, after a backup ---------------------------

[[ -e "$SETTINGS" ]] || die "no DSH settings file at $SETTINGS (pass --settings)"
backup="${SETTINGS}.bak-$(date -u +%Y%m%dT%H%M%S)"
cp "$SETTINGS" "$backup"

# Line-based surgery, not a YAML round-trip: the file is the person's, with
# their comments, and a parse-and-dump would rewrite all of it. A section ends
# at the next line that starts in column 0 and is not a comment or blank.
# The replacement starts at `llm-pi-ai:`, which is *below* this block's own
# three-line header — so the previous refresh's header is removed explicitly
# rather than left orphaned. Without that, every run (the plugin refreshes at
# every harness boot) would add three stale comment lines above the new block.
# The block travels in the environment, not on stdin: `python3 -` reads its
# *program* from stdin, so a heredoc and a pipe cannot both be used.
BLOCK="$block" python3 - "$SETTINGS" <<'PY'
import os, pathlib, re, sys

path = pathlib.Path(sys.argv[1])
block = os.environ["BLOCK"]
if not block.endswith("\n"):
    block += "\n"


def generated_header(line):
    """Whether a comment line is part of this tool's own generated header."""
    stripped = line.strip()
    if stripped.startswith("# DeepSeek Harness route to the "):
        return True
    if stripped.startswith("# Generated by tools/dsh_route.sh "):
        return True
    return re.fullmatch(
        r"# \d+ served model\(s\); settings\.yaml is hot-reloaded\.", stripped
    ) is not None


lines = path.read_text().splitlines(keepends=True)
out, index, replaced = [], 0, False
while index < len(lines):
    line = lines[index]
    if line.startswith("llm-pi-ai:"):
        index += 1
        while index < len(lines):
            following = lines[index]
            if following.strip() == "" or following[0] in (" ", "\t"):
                index += 1
                continue
            break
        while out and out[-1].strip() == "":
            out.pop()
        while out and generated_header(out[-1]):
            out.pop()
            while out and out[-1].strip() == "":
                out.pop()
        if out:
            out.append("\n")
        out.append(block)
        replaced = True
        continue
    out.append(line)
    index += 1
if not replaced:
    while out and out[-1].strip() == "":
        out.pop()
    if out:
        out.append("\n")
    out.append(block)
path.write_text("".join(out))
print("replaced" if replaced else "appended")
PY

echo "dsh_route: wrote ${#ids[@]} model(s) for the '${PROVIDER}' route into $SETTINGS" >&2
echo "           backup: $backup" >&2
echo "           DSH re-reads settings.yaml per request; no restart needed." >&2
