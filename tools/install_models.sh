#!/usr/bin/env bash
# Install a supported model install.
#
#   tools/install_models.sh          # what is installed, and what staging is kept
#   tools/install_models.sh ornith15 # install one
#   tools/install_models.sh --all-4bit   # every 4-bit model
#
# Every row is built from a pinned Hugging Face release and converted to .ssdai
# while it downloads: the MoE row through its converter, the two Inkling installs
# through TinyTitanRepack's own streaming path. No raw checkpoint is served.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Same override as the launcher: a checkout builds into `.build/release`, an
# install from the release tarball keeps its binaries in `~/.tinytitan/bin`.
BIN="${TINYTITAN_BIN_DIR:-$ROOT/.build/release}/TinyTitanRepack"
MODELS="${TINYTITAN_MODELS_DIR:-$ROOT/models}"
# Where a download and its conversion are staged, and where the converted
# snapshot is kept until the install exists. It must be absolute and independent
# of the caller's working directory: every staging path below used to be a bare
# `.build/...`, so a factory-new install launched from the user's home staged
# tens to hundreds of GB in `~/.build` — outside the install root, and outside
# the two directories the installer says removing uninstalls everything. Found
# and fixed 2026-09-24; `TINYTITAN_WORK_DIR` moves it to another volume.
WORK="${TINYTITAN_WORK_DIR:-$ROOT/.build}"

# The install menu's list of models, labels and sizes comes from the shared
# catalogue so this file and the installer cannot disagree about what exists.
# shellcheck source=tools/tinytitan_models.sh
source "$ROOT/tools/tinytitan_models.sh"

# The converters need a Python with numpy/ml_dtypes/safetensors at 3.10 or
# newer. Resolved by capability rather than by name: `python3` is 3.9 on a
# stock macOS and lacks the packages, while a pinned `python3.13` fails on a
# machine whose stack lives under another version. See tools/lib/python.sh.
# shellcheck source=tools/lib/python.sh
source "$ROOT/tools/lib/python.sh"

# --- disk space and staging hygiene -----------------------------------------
#
# Two rules make an install predictable on a machine that is somebody's only
# disk. A download is refused before it starts if it cannot finish, and staging
# is removed as soon as it can no longer save work — a converted snapshot exists
# so the *second* width of a model costs no fetch, so it is kept exactly until
# that width is installed.

# Free GB on the volume holding <path>, or empty when df cannot say.
free_gb() {
  local path="$1" available
  mkdir -p "$path" 2>/dev/null || true
  available="$(df -g "$path" 2>/dev/null | awk 'NR==2 {print $4}')"
  [[ "$available" =~ ^[0-9]+$ ]] && printf '%s' "$available"
}

# The catalogue's installed size for a row, in GB. May be fractional.
choice_gb() {
  local want="$1" row key _label _bits gb
  for row in "${TINYTITAN_MODEL_CHOICES[@]+"${TINYTITAN_MODEL_CHOICES[@]}"}"; do
    IFS='|' read -r key _label _bits gb _ <<<"$row"
    [[ "$key" == "$want" ]] && { printf '%s' "$gb"; return 0; }
  done
}

# Refuse when `path` cannot hold `need` GB. Empty `need` means "do not check",
# and an unreadable df means the same: this never invents a limit it cannot
# justify.
require_gb() {
  local label="$1" path="$2" need="$3" available
  [[ -n "$need" ]] || return 0
  available="$(free_gb "$path")"
  [[ -n "$available" ]] || return 0
  if awk -v a="$available" -v n="$need" 'BEGIN { exit !(a < n) }'; then
    printf '  %s holds %s GB but about %s GB is needed there.\n' "$label" "$available" "$need" >&2
    return 1
  fi
}

# The paths under $WORK that this tool owns. Deliberately a narrow pattern rather
# than "everything in $WORK": a checkout keeps its release build, benchmark
# results and scratch checkouts there too, and measuring those as "staging" made
# `clean` claim 12 GB it had nothing to do with. `ornith-mtp-src` is staging a
# removed draft-head row left behind; it is still counted so `clean` can reclaim
# it (see legacy_staging_paths).
staging_paths() {
  local dir
  for dir in "$WORK"/*-affine* "$WORK"/*-shards "$WORK"/ornith-mtp-src; do
    [[ -e "$dir" ]] && printf '%s\n' "$dir"
  done
}

# A KB count as a person reads it: MB below a gigabyte, GB above. Storage here
# spans three orders of magnitude, and "0 GB" or "1 GB" for a few megabytes is a
# number that teaches the reader to distrust the rest.
human_kb() {
  local kb="$1"
  if (( kb < 1048576 )); then
    printf '%s MB' "$(( (kb + 1023) / 1024 ))"
  else
    printf '%s GB' "$(( (kb + 1048575) / 1048576 ))"
  fi
}

# Total KB of staging, printed as a number only (empty when there is none), so
# callers can both compare and display it.
staging_kb() {
  local dir kb total=0
  while IFS= read -r dir; do
    kb="$(du -sk "$dir" 2>/dev/null | awk 'NR==1 {print $1}')"
    [[ "$kb" =~ ^[0-9]+$ ]] && total=$(( total + kb ))
  done < <(staging_paths)
  (( total > 0 )) || return 0
  printf '%s' "$total"
}

# The staging size as text ("12 GB", "850 MB"), or empty when there is none.
staging_size() {
  local kb
  kb="$(staging_kb)"
  [[ -n "$kb" ]] && human_kb "$kb"
}

# The staging paths, comma-separated, for a hint that says what is kept.
staging_names() {
  local dir names=()
  while IFS= read -r dir; do names+=("${dir#"$WORK"/}"); done < <(staging_paths)
  (( ${#names[@]} > 0 )) && printf '%s' "$(IFS=', '; printf '%s' "${names[*]+"${names[*]}"}")"
}

# The staging one catalogue row owns, one absolute path per line. `repack` rows
# stream straight into the install and stage nothing; the Qwen3.5-MoE converter
# writes a snapshot and its source shards under `$WORK/<preset>-*`. Every row is
# one width, so a snapshot has exactly one consumer: its own install (TT-142,
# 2026-11-27).
row_staging_paths() {
  local name="$1" row _n _dir _w source preset
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r _n _dir _w source preset _ <<<"$row"
    [[ "$_n" == "$name" ]] || continue
    case "$source" in
      convert_qwen35moe) printf '%s\n' "$WORK/${preset}-affine-4bit" "$WORK/${preset}-shards" ;;
    esac
    return 0
  done
}

# Every staging path any current row owns.
all_row_staging_paths() {
  local row name
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r name _ <<<"$row"
    row_staging_paths "$name"
  done
}

# Staging under `$WORK` that no catalogue row owns: a snapshot or shard directory
# a family left behind when it left the matrix. The 122B, the dense Qwen 3.5
# sizes, Qwen 3.6/3.8, AgentWorld and KAT-Coder rows each wrote their own
# `*-affine*`/`*-shards` here before they were removed, and `clean` is the one
# place that can still reclaim them; `staging_paths` counts them as staged, so
# reporting them as kept forever would be the lie. `ornith-mtp-src` is the one
# removal whose staging does not match either pattern, so it is named in
# `staging_paths` and lands here by the same rule.
legacy_staging_paths() {
  local dir owned candidate
  while IFS= read -r dir; do
    owned=0
    while IFS= read -r candidate; do
      [[ "$candidate" == "$dir" ]] && { owned=1; break; }
    done < <(all_row_staging_paths)
    (( owned )) || printf '%s\n' "$dir"
  done < <(staging_paths)
}

# The install directory one catalogue row would fill.
row_install_dir() {
  local name="$1" row _n dir
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r _n dir _ <<<"$row"
    [[ "$_n" == "$name" ]] && { printf '%s' "$dir"; return 0; }
  done
}

# Remove staging that cannot save a future install, and say what happened.
#
# One rule for both callers: a snapshot or shard directory goes once the install
# it serves exists, and is kept while that install is missing. Nothing is
# removed on a guess — an install that is not on disk may still be resumed from
# what is staged here.
cleanup_staging() {
  local name="$1" dir freed=0 removed=()
  local own_dir
  own_dir="$(row_install_dir "$name")"
  [[ -n "$own_dir" && -d "$MODELS/$own_dir" ]] || return 0
  while IFS= read -r dir; do
    [[ -n "$dir" && -e "$dir" ]] || continue
    local before
    before="$(du -sk "$dir" 2>/dev/null | awk 'NR==1 {print $1}')"
    rm -rf "$dir"
    freed=$(( freed + ${before:-0} ))
    removed+=("${dir#"$WORK"/}")
  done < <(row_staging_paths "$name")
  if (( ${#removed[@]} > 0 )); then
    printf '  staging cleaned (%s freed): %s\n' \
      "$(human_kb "$freed")" "$(IFS=', '; printf '%s' "${removed[*]+"${removed[*]}"}")"
  fi
  return 0
}

# The explicit `clean`: the rules above for every model, plus staging no row owns
# any more, plus empty shard directories, then a summary of what is still staged
# and why. Safe at any time: it only touches this script's own staging.
cmd_clean() {
  echo "Cleaning install staging under $WORK"
  local row name dir
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r name _ <<<"$row"
    cleanup_staging "$name"
  done
  # Staging from removed families: nothing left in the catalogue can consume it.
  local legacy freed=0 removed=()
  while IFS= read -r legacy; do
    [[ -n "$legacy" && -e "$legacy" ]] || continue
    local before
    before="$(du -sk "$legacy" 2>/dev/null | awk 'NR==1 {print $1}')"
    rm -rf "$legacy"
    freed=$(( freed + ${before:-0} ))
    removed+=("${legacy#"$WORK"/}")
  done < <(legacy_staging_paths)
  if (( ${#removed[@]} > 0 )); then
    printf '  staging cleaned (%s freed): %s\n' \
      "$(human_kb "$freed")" "$(IFS=', '; printf '%s' "${removed[*]+"${removed[*]}"}")"
  fi
  for dir in "$WORK"/*-shards; do
    [[ -d "$dir" && -z "$(ls -A "$dir" 2>/dev/null)" ]] && rmdir "$dir" 2>/dev/null
  done
  local staged
  staged="$(staging_size)"
  if [[ -n "$staged" ]]; then
    echo "  still staged: $staged — $(staging_names)"
    echo "  kept for an install you have not made yet; it goes automatically once"
    echo "  that install exists"
  else
    echo "  nothing left staged"
  fi
}

# name|install directory|width|source[|preset]
#
# The preset field names the converter preset; it appears on every row a
# converter branch reads (`convert_qwen35moe`). The rows are the shipped matrix
# and nothing else (TT-142, 2026-11-27): a family that leaves the matrix leaves
# this list, the menu's TINYTITAN_MODEL_CHOICES and the registry together.
#
# The keys are the installer's historical spellings; the launcher's canonical
# keys (ornith, ...) and every resolver alias are accepted too, through
# `tinytitan_install_target` in tools/tinytitan_models.sh. Both scripts print
# the same display names: the menu reads TINYTITAN_MODEL_CHOICES, whose labels
# `test_launcher_install` holds equal to `tinytitan_resolve_model`'s.
CATALOGUE=(
  # Ornith 1.5 35B-A3B, the one 35B-A3B install the matrix has. Qwen's own bf16
  # release is quantized one shard at a time by tools/prepare_agentworld.py and
  # then repacked; the snapshot is kept only until the install exists.
  "ornith15|ornith-1.5_35B_A3B_4Bit|4|convert_qwen35moe|ornith15"
  # Inkling 2 Mini: streamed directly from Hugging Face into .ssdai by
  # TinyTitanRepack without a Python converter or intermediate staging.
  "inkling-q4-k-xl-unsloth|inkling-2-mini_GGUF-Q4|4|repack"
  "inkling2mini|inkling-2-mini_NVFP4|4|repack"
)

usage() {
  cat <<'USAGE'
Commands

  tools/install_models.sh                 what is installed, and what staging is kept
  tools/install_models.sh --choose        the install menu (the one real choice)
  tools/install_models.sh <name>          download, convert and install it
  tools/install_models.sh --all-4bit      every 4-bit model
  tools/install_models.sh clean           drop staging that cannot be reused

Model keys

  Every row answers to the launcher's canonical key (ornith,
  inkling-q4-k-xl-unsloth, inkling2mini) and to the historical installer
  spelling (ornith15). Both scripts accept the same keys and print the same
  display names; the list is TINYTITAN_MODEL_CHOICES in
  tools/tinytitan_models.sh.

Where things go

  Models install to $TINYTITAN_MODELS_DIR (the installer sets
  ~/.tinytitan/models; a checkout uses <repo>/models), one directory per row
  with its verified-install.json receipt. A download and its conversion are
  staged under $TINYTITAN_WORK_DIR (default <tools>/../.build) — for an
  installed copy that is ~/.tinytitan/src/.build.

  Staging is kept only while it can save an install: the converted snapshot is
  what a resumed or repeated install of the same row reads, so it lives until
  that install exists and is removed automatically at that point.
  `clean` applies those rules to everything at once, reclaims staging a removed
  family left behind, and says what is left.

  A download is refused before it starts when the staging volume or the models
  volume cannot hold it, naming both numbers and both ways out (free space, or
  TINYTITAN_WORK_DIR on another volume). TINYTITAN_SKIP_DISK_CHECK=1 starts
  anyway. A resumed conversion does not need the room again.

What a run costs

  The sizes in the menu are the space an install occupies. Installing also
  downloads the model's own bf16 release and stages it under
  $TINYTITAN_WORK_DIR: about 70 GB for the 35B-A3B. The disk guard above names
  both volumes and both numbers before anything starts.

Coverage

  Ornith 1.5 35B-A3B        4-bit
  Inkling 2 Mini / UD-Q4_K_XL (Unsloth) 4-bit (streams directly via TinyTitanRepack)
  Inkling 2 Mini (NVFP4)    4-bit (streams directly via TinyTitanRepack)

Sources

  Every install is built from the model's own release, quantized here
  (group-64 affine) by the tools in tools/ where a converter is involved, then
  imported by TinyTitanRepack. Third-party quantizations are deliberately not
  used for the converted model: their group sizes, widths and norm conventions
  are theirs, and here the router, the shared-expert gate, the DeltaNet gating
  projections and every norm stay at bf16.

  convert_qwen35moe   tools/prepare_agentworld.py --model ornith15
                      the Qwen3.5-MoE bf16 release quantized one shard at a
                      time (about a 70 GB fetch), then repacked. The source
                      shards are deleted as they convert, so the full
                      checkpoint is never staged.
  repack              TinyTitanRepack streams the Inkling 2 Mini checkpoints
                      straight into .ssdai, with no converter and no staging.

USAGE
}

# What a catalogue `source` means, in words a person reads. The branch names are
# this script's own, so each one is translated rather than printed. Every row
# downloads first.
source_label() {
  case "$1" in
    convert_qwen35moe)
      printf 'download + convert' ;;
    repack)
      printf 'download + install' ;;
    *)
      printf '%s' "$1" ;;
  esac
}

status() {
  printf '%-20s %-8s %-13s %s\n' MODEL WIDTH STATE 'INSTALLS BY'
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r name dir width source _ _ <<<"$row"
    if [[ -d "$MODELS/$dir" ]]; then
      state="installed"
    else
      state="not installed"
    fi
    printf '%-20s %-8s %-13s %s\n' "$name" "${width}-bit" "$state" "$(source_label "$source")"
  done
  echo
  echo "Nothing needs fetching by hand: installing a row downloads that model's"
  echo "pinned release and installs it here."
  echo
  local staged
  staged="$(staging_size)"
  if [[ -n "$staged" ]]; then
    echo "staging: $staged under $WORK — $(staging_names)"
    echo "  a converted snapshot is kept until its own install exists, so a resumed"
    echo "  or repeated install needs no download; it is removed automatically then"
  fi
  echo "tools/install_models.sh --choose        the install menu: pick a model"
  echo "tools/install_models.sh <name>          download, convert and install it"
  echo "tools/install_models.sh clean           drop staging that cannot be reused"
  echo "tools/install_models.sh --help          sources, disk sizes and env vars"
}

# Does `TinyTitanRepack --model <key>` stream this one itself?
#
# These are the checkpoints the repacker fetches from Hugging Face and packages
# in one pass, with no converter involved — which means no Python. The keys are
# the same strings as this script's own, and the list is exactly what
# `TinyTitanRepack --help` accepts.
tinytitan_repack_streams() {
  case "$1" in
    inkling-q4-k-xl-unsloth|inkling2mini) return 0 ;;
    *) return 1 ;;
  esac
}

# A Mac that has only Xcode's command-line tools ships Python 3.9, below the
# converters' 3.10 floor, and there is no Homebrew to install a newer one. The
# two Inkling installs do not need a converter at all, so say which command works
# instead of leaving the person at "no usable Python interpreter found".
tinytitan_no_python_fallback() {
  local want="$1" row name dir
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r name dir _ <<<"$row"
    [[ "$name" == "$want" ]] || continue
    echo >&2
    if tinytitan_repack_streams "$want"; then
      echo "  This one does not need Python: the repacker streams it itself." >&2
      echo "      swift run -c release TinyTitanRepack --model $want --output models/$dir" >&2
      echo "  or, with the release build already made," >&2
      echo "      .build/release/TinyTitanRepack --model $want --output models/$dir" >&2
    else
      echo "  This one is quantized here from its bf16 release by a Python" >&2
      echo "  converter, so it does need Python 3.10+. The two Inkling 2 Mini" >&2
      echo "  installs stream through the repacker with no Python at all." >&2
    fi
    return 0
  done
  return 0
}

install_one() {
  local want="$1" found=0 preset_field=""
  # The launcher's canonical keys and resolver aliases (ornith, inkling2-q4)
  # mean the same rows as the historical catalogue keys; one shared function
  # decides, so the two scripts cannot disagree about what a key is.
  if ! want="$(tinytitan_install_target "$want")"; then
    echo "unknown model: $1" >&2
    status >&2
    return 2
  fi
  # Every conversion path runs a Python converter, so resolve the interpreter
  # once when the selected row needs one. Direct `repack` rows stream in Swift
  # without Python.
  local python=""
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    # Up to five fields on the MoE row, four elsewhere. The preset is read here
    # because the MoE branch addresses staging by it (`$WORK/<preset>-affine-*`);
    # it used to be left unset, which worked only while every key equalled its
    # preset by coincidence.
    IFS='|' read -r name dir width source preset_field _ <<<"$row"
    [[ "$name" == "$want" ]] || continue
    found=1
    if [[ "$source" != repack ]]; then
      python="$(tinytitan_resolve_python)" || { tinytitan_no_python_fallback "$want"; return 1; }
    fi
    if [[ -d "$MODELS/$dir" ]]; then
      echo "$name is already installed at models/$dir"
      cleanup_staging "$name"
      return 0
    fi

    # --- can this download finish? -----------------------------------------
    # The converted snapshot and the install exist at the same time, so each
    # destination has to hold a copy: the catalogue's size for the install, a
    # quarter more on the staging volume for the quantized snapshot's overhead,
    # and the shards being streamed. A Qwen3.5-MoE conversion keeps twenty
    # fetchers busy (~90 GB in flight for a 35B-A3B) and a few finished shards
    # waiting for the converter, so 104 GB is the upper bound on everything the
    # source shards can occupy. A direct repack streams straight into the
    # install and stages nothing. A snapshot that is already on disk is not
    # fetched again, so it is not counted twice. An unreadable df is not a
    # limit: require_gb says nothing then, which is the same rule the runtime's
    # own guard follows.
    if [[ "${TINYTITAN_SKIP_DISK_CHECK:-0}" != "1" ]]; then
      local size_gb="" snap_exists=0 install_exists=0 work_ok=1 models_ok=1
      local stream_gb=104 stage_need=""
      size_gb="$(choice_gb "$name")"
      if [[ -n "$size_gb" ]]; then
        # A completed snapshot is the only staging that saves the fetch. An
        # interrupted conversion leaves its directory behind with no index, and
        # `install_one` still runs the converter in that case -- so the guard
        # must ask the same question the converter branch does, or a restart
        # fetches ~90 GB of shards into a volume the guard just approved for 2
        # GB (AUD-0106).
        if [[ -n "$preset_field" && "$source" != repack &&
              -f "$WORK/${preset_field}-affine-${width}bit/model.safetensors.index.json" ]]; then
          snap_exists=1
        fi
        [[ -d "$MODELS/$dir" ]] && install_exists=1
        if [[ "$source" == repack ]]; then
          stage_need=""
        else
          stage_need="$(awk -v s="$size_gb" -v have="$snap_exists" -v stream="$stream_gb" \
               'BEGIN { printf "%d", have ? 2 : s * 1.25 + stream }')"
        fi
        require_gb "the staging volume ($WORK)" "$WORK" "$stage_need" \
          || work_ok=0
        local model_need
        model_need="$(awk -v s="$size_gb" -v have="$install_exists" \
             'BEGIN { printf "%d", have ? 2 : s + 3 }')"
        require_gb "the models volume ($MODELS)" "$MODELS" "$model_need" \
          || models_ok=0
        if (( ! work_ok || ! models_ok )); then
          echo "  Not starting $name: there is not enough room to finish it." >&2
          echo "  Free space, or stage on another volume:" >&2
          echo "      TINYTITAN_WORK_DIR=/Volumes/scratch/tt $0 $name" >&2
          echo "  (TINYTITAN_SKIP_DISK_CHECK=1 starts anyway, at your own risk.)" >&2
          return 1
        fi
      fi
    fi

    case "$source" in
      repack)
        [[ -x "$BIN" ]] || { echo "build TinyTitanRepack first: swift build -c release" >&2; return 1; }
        echo "installing $name -> models/$dir"
        # --resume is refused when there is nothing to resume; pass it only
        # when a previous attempt left its state behind.
        if [[ -f "$MODELS/$dir.resume.json" ]]; then
          "$BIN" --model "$name" --output "$MODELS/$dir" --resume
        else
          "$BIN" --model "$name" --output "$MODELS/$dir"
        fi
        ;;
      convert_qwen35moe)
        # The model's own bf16 release, quantized one shard at a time by
        # tools/prepare_agentworld.py (about 70 GB fetched; a source shard is
        # deleted once its output reaches a finished shard, so the sources since
        # the last 4 GiB flush -- ~14 GB of bf16 at this width -- stay beside the
        # fetch-ahead shards), then repacked. The snapshot is kept until the
        # install exists, so a failed or repeated run needs no new fetch.
        [[ -x "$BIN" ]] || { echo "build TinyTitanRepack first: swift build -c release" >&2; return 1; }
        local preset="${preset_field:?catalogue row has no converter preset}" model_id
        case "$preset" in
          ornith15) model_id="ornith-1.5-35b-a3b" ;;
          *) echo "unknown MoE preset: $preset" >&2; return 2 ;;
        esac
        local snap="$WORK/${preset}-affine-${width}bit"
        if [[ ! -f "$snap/model.safetensors.index.json" ]]; then
          echo "downloading and converting $preset (~70 GB fetch) -> $snap"
          "$python" "$ROOT/tools/prepare_agentworld.py" --model "$preset" --bits "$width" \
              --output "$snap" \
              --work "$WORK/${preset}-shards" || return 1
        fi
        echo "installing $name -> models/$dir"
        "$BIN" --input-snapshot "$snap" \
            --model-id "$model_id" --output "$MODELS/$dir"
        ;;
      unsupported)
        # No catalogue row uses this branch. It is kept as a generic refusal
        # rather than deleted, so wiring a row the runtime cannot execute fails
        # loudly instead of falling through this switch and reporting a
        # successful install it never performed.
        cat <<EOF
$name cannot be run by this installer.

A row reaches this branch only when it is known to build an install the runtime
cannot execute. Check --help for what that model would require; if nothing
explains it, this message is stale and the row should be fixed rather than
shipped.
EOF
        return 1
        ;;
    esac
    # Installed. Drop this row's staging: nothing else consumes it, and the
    # install it serves now exists.
    cleanup_staging "$name"
    return 0
  done
  [[ "$found" == 1 ]] || { echo "unknown model: $want" >&2; status >&2; return 2; }
}

# Install every model at one width, and report on all of them.
#
# The batch used to be the dispatcher's own loop, whose body ended in
# `[[ "$w" == 4 ]] && install_one "$n"`. A loop's status is its last body
# statement's, so the verdict was the *final catalogue row's width test*: with an
# 8-bit row last, the run exited 1 after installing everything. One refused model
# is enough to make the run incomplete, so every row's status is collected and
# the count of what was missed is what leaves the script (AUD-253). Every shipped
# row is 4-bit today, so the width is a parameter the argument still names.
install_width() {
  local want="$1" row name _ width
  local ran=() missed=()
  for row in "${CATALOGUE[@]+"${CATALOGUE[@]}"}"; do
    IFS='|' read -r name _ width _ <<<"$row"
    [[ "$width" == "$want" ]] || continue
    if install_one "$name"; then
      ran+=("$name")
    else
      missed+=("$name")
    fi
  done
  echo
  echo "installed ${#ran[@]} of $(( ${#ran[@]} + ${#missed[@]} )) ${want}-bit model(s)"
  if [[ "${#missed[@]}" -gt 0 ]]; then
    printf 'NOT INSTALLED: %s\n' "${missed[@]+"${missed[@]}"}" >&2
    return 1
  fi
  return 0
}

# The one question an install has to ask.
#
# The model is the only real choice a person makes — everything else about
# setting TinyTitan up is automatic — so it is shown as a list with the disk each
# one costs and what it is for, with the verified default first. Enter takes the
# default. The entries and their sizes come from `TINYTITAN_MODEL_CHOICES` in
# tools/tinytitan_models.sh, so the installer cannot offer a model that does not
# exist or quote a size no one re-measured.
choose_model() {
  local count=${#TINYTITAN_MODEL_CHOICES[@]}
  local index key label bits gb note reply default_label
  if (( count == 0 )); then
    echo "no models are listed in TINYTITAN_MODEL_CHOICES" >&2
    return 2
  fi
  IFS='|' read -r _ default_label _ _ _ <<<"${TINYTITAN_MODEL_CHOICES[0]}"

  echo
  echo "Which model? This is the only choice; the rest is automatic."
  echo
  printf '  %-4s %-32s %-6s %-10s %s\n' "#" "model" "bits" "installed" "what it is for"
  for (( index = 0; index < count; index++ )); do
    IFS='|' read -r key label bits gb note <<<"${TINYTITAN_MODEL_CHOICES[$index]}"
    printf '  %2d)  %-32s %-6s %7s GB  %s\n' \
      "$((index + 1))" "$label" "${bits}-bit" "$gb" "$note"
  done
  echo
  echo "  Sizes are installed sizes: installing also downloads the model's bf16"
  echo "  release first and stages it separately — --help has the numbers."
  echo
  printf 'Choice [1-%d] (Enter for %s): ' "$count" "$default_label"
  # An EOF is not an answer. Treating it as "Enter" started the recommended
  # download for a caller that never chose anything (found 2026-09-24 by
  # closing stdin on `--choose`); only a real empty line takes the default.
  if ! read -r reply; then
    echo
    echo "No answer given; nothing was installed." >&2
    return 2
  fi
  reply="${reply:-1}"
  # Both reads are forced to decimal. `(( ))` and `$(( ))` take a leading zero as
  # octal, and this guard asks the range question in the negative, so an
  # unreadable reply answers "no verdict" and means ACCEPTED: `010` passed as 8
  # and installed row 8 for a person who pointed at row 10, while `08` passed the
  # test and then died at the index below. The row a reply names is the row it
  # installs. Forcing the base is what the launcher does with a number it consumes
  # on the spot (`ram_tier` and `valid_concurrency` in server_launcher.sh); a port
  # is refused instead because it flows outward to readers that script cannot fix.
  if [[ ! "$reply" =~ ^[0-9]+$ ]] || (( 10#$reply < 1 || 10#$reply > count )); then
    echo "not a choice: $reply" >&2
    return 2
  fi
  key="${TINYTITAN_MODEL_CHOICES[$((10#$reply - 1))]%%|*}"
  echo
  install_one "$key"
}

case "${1:-}" in
  "")            status ;;
  --help|-h)     usage ;;
  clean)         cmd_clean ;;
  # The install menu: one numbered list, default first, then install the pick.
  --choose|--menu)
                 if [[ ! -t 0 ]]; then
                   echo "--choose needs a terminal to ask in; pass a model name instead" >&2
                   exit 2
                 fi
                 choose_model ;;
  --all-4bit)    install_width 4 ;;
  *)             install_one "$1" ;;
esac
