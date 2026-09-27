#!/usr/bin/env bash
# Production gates that the compiler cannot express. Run locally before a PR;
# CI runs the same script, so a green run here is a green run there.
#
#   tools/lint.sh              # all checks
#   tools/lint.sh force-cast   # one check
#
# Checks:
#   force-cast          no `as!` / `try!` in sources/ without an audited opt-out
#   func-length         no NEW function longer than MAX_FUNC_LINES (ratcheted)
#   unchecked-sendable  new `@unchecked Sendable` must document its invariant
#   converter           routed experts must land at their own index
#   arch-path           no hardcoded SwiftPM triple in a build path (see below)
#   shell-portability   scripts run on the system bash (3.2), not just the dev one
#   shell-lint          shellcheck warnings-as-errors over every script, pinned version
#   swiftlint           SwiftLint violations-as-errors under the committed config
#   swift-format        formatting enforced under the committed .swift-format
#   javascript          eslint + prettier --check over the plugin packages, pinned
#   python              ruff check + ruff format --check under pyproject.toml,
#                       with the pinned ruff version
#
# Opting out of force-cast: put `lint:allow-force <reason>` in a comment on
# the line immediately above. The reason is mandatory and is what a reviewer
# reads — an opt-out without one fails the same as no opt-out at all.
#
# Opting out of arch-path: `lint:allow-arch-path <reason>` on the line above,
# for a deliberate compatibility fallback rather than a build path.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export ROOT
BASELINE="$SCRIPT_DIR/func-length-baseline.txt"
MAX_FUNC_LINES="${MAX_FUNC_LINES:-120}"

status=0
want="${1:-all}"

# --- force-cast / force-try -------------------------------------------------
check_force_cast() {
  echo "== force-cast: as! / try! outside tests =="
  local found=0
  while IFS= read -r hit; do
    local file line
    file="${hit%%:*}"
    line="$(echo "$hit" | cut -d: -f2)"
    # Walk up the contiguous comment block directly above the hit, looking for
    # an opt-out marker followed by a reason. Scanning the whole block (not
    # just the previous line) lets the reason wrap naturally.
    local n=$((line - 1)) text ok=0
    while [ "$n" -ge 1 ]; do
      text="$(sed -n "${n}p" "$file")"
      echo "$text" | grep -qE '^[[:space:]]*//' || break
      if echo "$text" | grep -qE 'lint:allow-force[[:space:]]+[^[:space:]]'; then
        ok=1
        break
      fi
      n=$((n - 1))
    done
    [ "$ok" -eq 1 ] && continue
    echo "  ${file#$ROOT/}:$line: $(echo "$hit" | cut -d: -f3- | sed 's/^[[:space:]]*//')"
    found=1
  done < <(grep -rnE '(\bas!\s|\btry!\s)' --include='*.swift' "$ROOT/sources" 2>/dev/null)

  if [ "$found" -ne 0 ]; then
    echo "  FAIL: force cast/try without an audited 'lint:allow-force <reason>' comment above it"
    status=1
  else
    echo "  ok"
  fi
}

# --- function length --------------------------------------------------------
# Indentation-anchored: a function runs from its `func` line to the first line
# that closes a brace at the same indent. Brace-depth counting drifts on braces
# inside strings and comments; this codebase is consistently formatted, so
# indent is the more reliable anchor.
#
# Every `func` must land in exactly one of three buckets: no body (a protocol
# requirement), a body that opens and closes on one line, or a body with a
# closer at its own indent. Anything else is printed as UNRESOLVED and fails
# the check. A gate that silently skips what it cannot parse reports "ok" for
# code it never looked at, which is worse than no gate — so unparsed input is
# an error, not a shrug.
measure_functions() {
  ruby -e '
    Encoding.default_external = Encoding::UTF_8
    Encoding.default_internal = Encoding::UTF_8
    limit = Integer(ENV.fetch("MAX_FUNC_LINES", "120"))
    root = ENV.fetch("ROOT")
    scanned = 0
    Dir.glob(File.join(root, "sources", "**", "*.swift")).sort.each do |path|
      lines = File.readlines(path, chomp: true)
      rel = path.delete_prefix(root + "/")
      lines.each_with_index do |line, i|
        next unless (m = line.match(/^(\s*)(?:[\w@\(\)]+\s+)*func\s+([A-Za-z_]\w*)/))
        scanned += 1
        indent, name = m[1], m[2]

        # An inline opt-out in the contiguous comment block above, mirroring
        # lint:allow-force. Preferred over a baseline row for a function that is
        # long on purpose: the reason sits next to the code instead of in a
        # separate file, so it is reviewed whenever the function is.
        k = i - 1
        exempt = false
        while k >= 0 && lines[k] =~ /^\s*(\/\/|\/\/\/)/
          if lines[k] =~ /lint:allow-long\s+\S/
            exempt = true
            break
          end
          k -= 1
        end
        next if exempt

        # Walk the (possibly multi-line) signature looking for the body brace.
        # Stop at the next declaration or at a closer no deeper than us, which
        # is what a bodyless protocol requirement runs into.
        open_at = nil
        j = i
        while j < lines.length
          text = lines[j].sub(%r{//.*$}, "")
          if j > i && text =~ /^\s{0,#{indent.length}}(\}|func\s|var\s|let\s|case\s)/
            break
          end
          if text.include?("{")
            open_at = j
            break
          end
          j += 1
        end

        if open_at.nil?
          next # no body: protocol requirement or bodyless declaration
        end

        opener = lines[open_at].sub(%r{//.*$}, "")
        if opener.count("{") == opener.count("}") && opener.rstrip.end_with?("}")
          next # body opens and closes on one line
        end

        closer = /^#{indent}\}/
        stop = ((open_at + 1)...lines.length).find { |k| lines[k] =~ closer }
        if stop.nil?
          puts "UNRESOLVED:#{rel}:#{name}:#{i + 1}"
          next
        end
        length = stop - i
        next unless length > limit
        puts "#{rel}:#{name}:#{length}"
      end
    end
    # Coverage receipt. Without it an empty result is indistinguishable from
    # "scanner never ran", and the gate would report ok for an unexamined tree.
    puts "SCANNED:#{scanned}"
  '
}

check_func_length() {
  echo "== func-length: no NEW function over $MAX_FUNC_LINES lines =="
  local measured current new unresolved raw rc scanned

  # Capture without a pipe so the scanner's exit status survives, then check it.
  # A gate whose measurement step died must fail, not report "ok (0 new)" —
  # that is how an unexported ROOT once let this check pass while looking at
  # nothing at all.
  raw="$(measure_functions)"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL: the length scanner exited $rc; it measured nothing."
    status=1
    return
  fi
  scanned="$(echo "$raw" | sed -n 's/^SCANNED://p' | tail -1)"
  if [ -z "$scanned" ] || [ "$scanned" -eq 0 ] 2>/dev/null; then
    echo "  FAIL: the length scanner reported no functions scanned."
    echo "        Expected ~1000 under sources/; check ROOT and the glob."
    status=1
    return
  fi
  measured="$(echo "$raw" | grep -v '^SCANNED:' | sort)"

  # Coverage first: if the scanner could not resolve a function, the ratchet
  # below is reporting on an unknown subset of the tree. Fail loudly rather
  # than let an "ok" stand for code that was never measured.
  unresolved="$(echo "$measured" | grep '^UNRESOLVED:' || true)"
  if [ -n "$unresolved" ]; then
    echo "$unresolved" | sed 's/^UNRESOLVED:/  UNRESOLVED: /'
    echo "  FAIL: the length scanner could not find these functions' bounds."
    echo "        Fix tools/lint.sh — do not silence this by ignoring them."
    status=1
    return
  fi
  current="$(echo "$measured" | grep -v '^UNRESOLVED:' || true)"

  if [ ! -f "$BASELINE" ]; then
    echo "  no baseline at ${BASELINE#$ROOT/}; writing one"
    echo "$current" > "$BASELINE"
    echo "  ok (baseline created, $(echo "$current" | grep -c . ) entries)"
    return
  fi
  # Compare on file:function only, so shrinking a baselined function toward the
  # limit does not churn the file.
  local baseline_keys current_keys stale
  baseline_keys="$(cut -d: -f1,2 "$BASELINE" | sort -u)"
  current_keys="$(echo "$current" | grep -v '^$' | cut -d: -f1,2 | sort -u)"

  new="$(comm -13 <(echo "$baseline_keys") <(echo "$current_keys"))"
  if [ -n "$new" ]; then
    echo "$new" | sed 's/^/  NEW: /'
    echo "  FAIL: shorten it, or update ${BASELINE#$ROOT/} with a reason in the PR"
    status=1
    return
  fi

  # An exemption has to stay earned. Once a function is decomposed below the
  # limit it drops out of `current`, and leaving its baseline row behind would
  # let it silently grow back over the limit later under the old exemption.
  stale="$(comm -23 <(echo "$baseline_keys") <(echo "$current_keys"))"
  if [ -n "$stale" ]; then
    echo "$stale" | sed 's/^/  STALE: /'
    echo "  FAIL: these are no longer over $MAX_FUNC_LINES lines — drop them from"
    echo "        ${BASELINE#$ROOT/} so the exemption cannot be reused."
    status=1
    return
  fi

  echo "  ok ($(echo "$current" | grep -c .) baselined, 0 new, $scanned scanned)"
}

# --- unchecked Sendable -----------------------------------------------------
# `@unchecked Sendable` is a promise to the compiler that a type is safe to
# share across threads. Unlike the checked kind, nothing verifies it — so the
# reasoning has to be written down where the next reader will find it, or the
# promise is unreviewable. Existing sites are baselined; new ones must explain
# themselves.
SENDABLE_BASELINE="$SCRIPT_DIR/unchecked-sendable-baseline.txt"

check_unchecked_sendable() {
  echo "== unchecked-sendable: new conformances must document their invariant =="
  local current new stale
  current="$(ruby -e '
    Encoding.default_external = Encoding::UTF_8
    Encoding.default_internal = Encoding::UTF_8
    root = ENV.fetch("ROOT")
    Dir.glob(File.join(root, "sources", "**", "*.swift")).sort.each do |path|
      lines = File.readlines(path, chomp: true)
      rel = path.delete_prefix(root + "/")
      lines.each_with_index do |line, i|
        next unless line.include?("@unchecked Sendable")
        next if line =~ /^\s*(\/\/|\/\/\/)/      # a comment mentioning it
        # The invariant comment sits above the declaration, and the declaration
        # may wrap over several lines (swift-format breaks a long inheritance
        # clause), so walk the whole contiguous non-blank block above and collect
        # its comments. The marker is still required; only its distance from the
        # `@unchecked Sendable` token changed.
        j = i - 1
        block = []
        while j >= 0 && !lines[j].strip.empty?
          block << lines[j] if lines[j] =~ /^\s*(\/\/|\/\/\/)/
          j -= 1
        end
        text = block.join(" ").downcase
        next if text =~ /unchecked-invariant:/
        # Name the type so the row survives line-number churn.
        name = line[/(?:class|struct|enum|actor)\s+([A-Za-z_]\w*)/, 1] || "line#{i + 1}"
        puts "#{rel}:#{name}"
      end
    end
  ' | sort -u)"

  if [ ! -f "$SENDABLE_BASELINE" ]; then
    echo "$current" > "$SENDABLE_BASELINE"
    echo "  ok (baseline created, $(echo "$current" | grep -c .) entries)"
    return
  fi
  new="$(comm -13 <(sort -u "$SENDABLE_BASELINE") <(echo "$current"))"
  if [ -n "$new" ]; then
    echo "$new" | sed 's/^/  NEW: /'
    echo "  FAIL: document the invariant above it in a comment containing"
    echo "        'unchecked-invariant: <what makes this safe>'"
    status=1
    return
  fi
  stale="$(comm -23 <(sort -u "$SENDABLE_BASELINE") <(echo "$current"))"
  if [ -n "$stale" ]; then
    echo "$stale" | sed 's/^/  DOCUMENTED: /'
    echo "  These now carry an invariant — drop them from"
    echo "  ${SENDABLE_BASELINE#$ROOT/} so the exemption cannot be reused."
    status=1
    return
  fi
  echo "  ok ($(echo "$current" | grep -c .) undocumented, 0 new)"
}

# --- converter: expert placement ---------------------------------------------
# A routed-expert checkpoint may ship experts one tensor at a time, and the
# converter stacks them into the fused axis the repacker packs. The experts
# arrive in whatever order the shards and the (lexicographic) index put them --
# KAT's arrive 0, 1, 10, 100, ... -- so filing them by arrival order puts expert
# k's weights in expert j's slot. Everything downstream still passes: the bytes
# match the checkpoint, the shapes are right, `validateRoleUniformity` passes,
# the receipt verifies, and the model answers fluently from the wrong experts.
# It has to be caught here, because no Swift test can see a converter bug.
check_converter_expert_order() {
  echo "== converter-expert-order: experts file at their own index =="
  if ! command -v python3 >/dev/null 2>&1 && ! command -v python3.13 >/dev/null 2>&1; then
    echo "  SKIP: no python3 available to run the converter check"
    return
  fi
  local py out rc
  py="$(command -v python3.13 || command -v python3)"
  out="$("$py" - <<'PY' 2>&1
import sys
sys.path.insert(0, "tools")
try:
    import numpy as np
    import prepare_agentworld as P
except ImportError as exc:
    print("SKIP: {} (converter deps unavailable)".format(exc))
    sys.exit(0)

class W:
    def __init__(self): self.added = {}
    def add(self, n, v): self.added[n] = np.asarray(v)
    def flush(self): pass

NE = 8
order = [3, 0, 7, 1, 5, 2, 6, 4]          # deliberately not ascending
w = W(); f = P.FusedExperts()
for e in order:
    f.add("model.language_model.layers.0.mlp.experts.%d.gate_proj.weight" % e,
          np.full((512, 2048), float(e), dtype=np.float32), 4, w)
f.release(NE, {4: w})
key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases"
bi = w.added.get(key)
if bi is None:
    print("FAIL: the fused tensor was never emitted")
    sys.exit(1)
got = [float(bi[e].astype(np.float32).mean()) for e in range(NE)]
want = [float(e) for e in range(NE)]
# Each expert carries its index as its value, so the axis must read 0..NE-1.
if [round(x, 3) for x in got] != [round(x, 3) for x in want]:
    print("FAIL: experts landed by arrival order: %s (want %s)" % (got, want))
    sys.exit(1)
print("ok")
PY
)"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$out" | sed 's/^/  /'
    status=1
  else
    echo "  $(echo "$out" | tail -1)"
  fi
}

# --- arch-triple build paths ------------------------------------------------
# SwiftPM's product directory is not a fixed name. The layout moved from
# `.build/<triple>/release` to `.build/release -> out/Products/Release`, so a
# script that pins the triple points at nothing on a machine that built with a
# newer toolchain — or, worse, at a stale binary left behind by an older one,
# where an `-x` guard passes and the wrong executable runs quietly. Both
# happened: tools/install_models.sh refused to install on a fresh clone
# (issue #8) while this checkout would have run a two-day-old TinyTitanRepack.
# `.build/release` is the stable spelling on both layouts.
check_arch_path() {
  echo "== arch-path: no hardcoded SwiftPM triple in a build path =="
  local out rc
  out="$(cd "$ROOT" && python3 - <<'PY'
import pathlib
import re
import sys

PATTERN = re.compile(r"arm64-apple-macosx")   # lint:allow-arch-path the gate names the literal it forbids
ALLOW = re.compile(r"lint:allow-arch-path\s+\S+")
SKIP = {".build", ".qwen", ".git", "releases", "__pycache__"}
bad = []
for ext in ("*.sh", "*.py", "*.swift"):
    for path in pathlib.Path(".").rglob(ext):
        if SKIP & set(path.parts):
            continue
        lines = path.read_text(errors="replace").splitlines()
        for index, line in enumerate(lines):
            if not PATTERN.search(line) or ALLOW.search(line):
                continue
            if index and ALLOW.search(lines[index - 1]):
                continue          # the reason sits on the line above
            bad.append(f"{path}:{index + 1}: {line.strip()[:90]}")
if bad:
    print("FAIL: hardcoded SwiftPM triple in a build path; use .build/release")
    for entry in bad:
        print("  " + entry)
    sys.exit(1)
print("ok (none)")
PY
)"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$out" | sed 's/^/  /'
    status=1
  else
    echo "  $(echo "$out" | tail -1)"
  fi
}

# --- shell-portability ------------------------------------------------------
# Every user-facing script carries `#!/usr/bin/env bash`, which on a factory Mac
# resolves to `/bin/bash` **3.2.57** — not the Homebrew 5.x a developer has, and
# which `bash` finds first on a development machine. Two classes of bug reached
# main before this gate existed:
#
#   * a single-quoted heredoc containing an apostrophe inside `$( )`, which 3.2
#     refuses to *parse* — the whole script dies before its first line;
#   * bash-4 expansions and builtins (`${v^^}`, `mapfile`), which 3.2 parses and
#     then fails on at run time, halfway through a menu; and
#   * a whole-array expansion `"${a[@]}"` on an **empty** array, which 3.2 makes
#     `a[@]: unbound variable` under the `set -u` every script here sets. 5.x
#     accepts it, so it is invisible on a development machine. Write it
#     `${a[@]+"${a[@]}"}` (and likewise `[*]`), which means the same thing for a
#     non-empty array on both shells.
#
# The parse half needs the old shell, so it runs `/bin/bash` whatever that is; the
# two pattern halves are version-independent and catch the other classes anywhere.
# Opting out: `lint:allow-shell <reason>` on the line immediately above, for a
# deliberate use rather than an oversight.
check_shell_portability() {
  echo "== shell-portability: runs on the system bash, not only the developer's =="
  local old_bash="/bin/bash" version scripts=() f hit failed=0
  version="$("$old_bash" --version 2>/dev/null | sed -n 's/.*version \([0-9][0-9.]*\).*/\1/p' | head -1)"

  # Every shell script in the tree, not only the ones the installer runs: the
  # rule is a property of the shell, and `docs/paper/build.sh` is a script too.
  while IFS= read -r f; do scripts+=("$f"); done < <(
    find "$ROOT/tools" "$ROOT/benchmark" "$ROOT/docs" -name '*.sh' -not -path '*/.build/*' 2>/dev/null | sort)

  for f in "${scripts[@]+"${scripts[@]}"}"; do
    if ! "$old_bash" -n "$f" >/dev/null 2>&1; then
      echo "  FAIL: $f does not parse under $old_bash"
      "$old_bash" -n "$f" 2>&1 | head -2 | sed 's/^/    /'
      failed=1
    fi
  done

  # lint:allow-shell the rule's own pattern, not a use of the rule
  local pattern='\$\{[A-Za-z_][A-Za-z0-9_]*(\[[^]]*\])?(\^\^|,,|\^\}|,)\}|\b(mapfile|readarray)\b|declare -A|local -A|declare -n|local -n|&>>|\|&|wait -n|;;&|\[\[ -v |shopt -s globstar'
  if [ "${#scripts[@]}" -gt 0 ]; then
    while IFS= read -r hit; do
      [ -n "$hit" ] || continue
      local file line code trimmed
      file="${hit%%:*}"
      line="$(printf '%s' "$hit" | cut -d: -f2)"
      code="$(printf '%s' "$hit" | cut -d: -f3-)"
      # A comment that names the rule is not a use of it, and this repository
      # explains these constructs in prose right where it avoids them.
      trimmed="$(printf '%s' "$code" | sed 's/^[[:space:]]*//')"
      case "$trimmed" in \#*) continue ;; esac
      if [ "$line" -gt 1 ] && sed -n "$((line - 1))p" "$file" | grep -q 'lint:allow-shell'; then
        continue
      fi
      echo "  FAIL: $file:$line uses a bash 4+ feature; the system bash is 3.2"
      echo "    $trimmed"
      failed=1
    done < <(grep -rnE "$pattern" "${scripts[@]+"${scripts[@]}"}" 2>/dev/null)
  fi

  # A whole-array expansion with no `+` guard. A bare `"${a[@]}"` is correct on
  # 5.x for any array, so the only way to catch it is by shape: mask the guarded
  # `${a[@]+"${a[@]}"}` down to its test, and anything still expanding a whole
  # array was written bare.
  local array_pattern='\$\{[A-Za-z_][A-Za-z0-9_]*\[[@*]\]\}'
  if [ "${#scripts[@]}" -gt 0 ]; then
    while IFS= read -r hit; do
      [ -n "$hit" ] || continue
      local file line code trimmed masked
      file="${hit%%:*}"
      line="$(printf '%s' "$hit" | cut -d: -f2)"
      code="$(printf '%s' "$hit" | cut -d: -f3-)"
      trimmed="$(printf '%s' "$code" | sed 's/^[[:space:]]*//')"
      case "$trimmed" in \#*) continue ;; esac
      if [ "$line" -gt 1 ] && sed -n "$((line - 1))p" "$file" | grep -q 'lint:allow-shell'; then
        continue
      fi
      masked="$(printf '%s' "$code" \
        | sed -E 's/[+]"\$\{[A-Za-z_][A-Za-z0-9_]*\[[@*]\]\}"/+GUARDED/g')"
      printf '%s' "$masked" | grep -qE "$array_pattern" || continue
      echo "  FAIL: $file:$line expands a whole array without the empty-array guard"
      echo "    $trimmed"
      echo "    on bash 3.2 an empty array there is 'unbound variable' under set -u;"
      echo "    the guard is the form documented above this check"
      failed=1
    done < <(grep -rnE "$array_pattern" "${scripts[@]+"${scripts[@]}"}" 2>/dev/null)
  fi

  if [ "$failed" -eq 0 ]; then
    echo "  ok (${#scripts[@]} scripts, system bash ${version:-unknown})"
  else
    echo "  fix: use a tr/while-read equivalent, or put"
    echo "       'lint:allow-shell <reason>' on the line above a deliberate use"
    # The gates share one exit status; a returned non-zero from the `all` list
    # would be swallowed and the gate would report a failure while exiting 0.
    status=1
  fi
  return $failed
}

# --- python -----------------------------------------------------------------
# Ruff is the Python standard: the rules and the format are pinned in the
# repository's pyproject.toml. A missing or different ruff FAILS rather than
# skipping — a gate that quietly does nothing is the failure mode this check
# exists to prevent.
RUFF_PIN="0.16.7"
# The oldest Python the scripts must run on; pyproject.toml's target-version is
# the same number, and CI installs this interpreter for the gates.
PYTHON_FLOOR="3.13"

check_python() {
  echo "== python: ruff check + ruff format --check (pinned $RUFF_PIN) =="
  if ! command -v ruff >/dev/null 2>&1; then
    echo "  FAIL: ruff is not installed; this gate needs the pinned version:"
    echo "        pipx install ruff==$RUFF_PIN   (or: python3 -m pip install --user ruff==$RUFF_PIN)"
    status=1
    return 1
  fi
  local version
  version="$(ruff --version | awk '{print $2}')"
  if [ "$version" != "$RUFF_PIN" ]; then
    echo "  FAIL: ruff $version is installed, this gate pins $RUFF_PIN"
    echo "        pipx install --force ruff==$RUFF_PIN"
    status=1
    return 1
  fi
  local output
  if ! output="$(cd "$ROOT" && ruff check . 2>&1)"; then
    printf '%s\n' "$output" | tail -25
    echo "  FAIL: ruff check (fix: ruff check --fix .)"
    status=1
    return 1
  fi
  if ! output="$(cd "$ROOT" && ruff format --check . 2>&1)"; then
    printf '%s\n' "$output" | tail -10
    echo "  FAIL: ruff format --check (fix: ruff format .)"
    status=1
    return 1
  fi
  # The floor is enforced, not trusted: every script must PARSE under the pinned
  # Python, not merely under whichever interpreter is on this Mac. Ruff's
  # formatter rewrites to the target version's syntax (targeting 3.14 turned
  # `except (A, B):` into PEP 758's `except A, B:`, a syntax error on 3.13), so
  # the check is on the real files with the real floor.
  if ! output="$(cd "$ROOT" && python3 - "$PYTHON_FLOOR" <<'PY' 2>&1
import ast, pathlib, sys

floor = tuple(int(part) for part in sys.argv[1].split("."))
roots = [pathlib.Path("benchmark"), pathlib.Path("tools"), pathlib.Path("docs")]
bad = []
for root in roots:
    for path in sorted(root.rglob("*.py")):
        if ".build" in path.parts:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path),
                      feature_version=floor)
        except SyntaxError as error:
            bad.append(f"{path}:{error.lineno}: {error.msg}")
if bad:
    print("\n".join(bad))
    sys.exit(1)
PY
)"; then
    printf '%s\n' "$output" | tail -10
    echo "  FAIL: a script does not parse under Python $PYTHON_FLOOR"
    status=1
    return 1
  fi
  echo "  ok (ruff $version, check and format clean, parses under $PYTHON_FLOOR)"
  return 0
}


# --- shellcheck -------------------------------------------------------------
# The scripts are the installer, the launcher and the release tooling: a real
# bug here is a user's disk or a published artifact, not a style point. The
# pinned version matters because shellcheck's checks change between releases.
SHELLCHECK_PIN="0.11.0"

check_shellcheck() {
  echo "== shellcheck: warnings are errors over every script (pinned $SHELLCHECK_PIN) =="
  if ! command -v shellcheck >/dev/null 2>&1; then
    echo "  FAIL: shellcheck is not installed; this gate needs the pinned version:"
    echo "        brew install shellcheck  (CI downloads $SHELLCHECK_PIN from the release page)"
    status=1
    return 1
  fi
  local version
  version="$(shellcheck --version | awk '/^version:/ {print $2}')"
  if [ "$version" != "$SHELLCHECK_PIN" ]; then
    echo "  FAIL: shellcheck $version is installed, this gate pins $SHELLCHECK_PIN"
    status=1
    return 1
  fi
  local scripts=() f output
  while IFS= read -r f; do scripts+=("$f"); done < <(
    find "$ROOT/tools" "$ROOT/benchmark" "$ROOT/docs" -name '*.sh' -not -path '*/.build/*' 2>/dev/null | sort)
  if [ "${#scripts[@]}" -eq 0 ]; then
    echo "  FAIL: no shell scripts found to check"
    status=1
    return 1
  fi
  if ! output="$(shellcheck -S warning -f gcc "${scripts[@]+"${scripts[@]}"}" 2>&1)"; then
    printf '%s\n' "$output" | sed "s|$ROOT/||" | head -30
    echo "  FAIL: shellcheck found warnings (see above)"
    status=1
    return 1
  fi
  echo "  ok (shellcheck $version, ${#scripts[@]} scripts, no warnings)"
  return 0
}

# --- swiftlint --------------------------------------------------------------
# The committed `.swiftlint.yml` is the Swift standard here: the safety rules
# (force_unwrapping, implicitly_unwrapped_optional) are on and had to reach zero
# before this gate could be wired, layout is delegated to swift-format and
# size/complexity to the ratchet above, and every remaining decision is
# justified in the config itself. `--strict` promotes every warning to a
# failure. The pinned version matters because the rule set changes between
# releases.
SWIFTLINT_PIN="0.65.1"

check_swiftlint() {
  echo "== swiftlint: violations are errors under the committed config (pinned $SWIFTLINT_PIN) =="
  if ! command -v swiftlint >/dev/null 2>&1; then
    echo "  FAIL: swiftlint is not installed; this gate needs the pinned version:"
    echo "        brew install swiftlint  (or the $SWIFTLINT_PIN release binary)"
    status=1
    return 1
  fi
  local version
  version="$(swiftlint version)"
  if [ "$version" != "$SWIFTLINT_PIN" ]; then
    echo "  FAIL: swiftlint $version is installed, this gate pins $SWIFTLINT_PIN"
    status=1
    return 1
  fi
  local output
  if ! output="$(cd "$ROOT" && swiftlint lint --strict --no-cache --quiet 2>&1)"; then
    printf '%s\n' "$output" | sed "s|$ROOT/||" | head -30
    echo "  FAIL: swiftlint found violations (fix them; an exclusion needs a written reason)"
    status=1
    return 1
  fi
  echo "  ok (swiftlint $version, --strict clean)"
  return 0
}

# --- swift-format -----------------------------------------------------------
# The committed `.swift-format` is the formatting standard: 4-space indentation
# (the tree's actual style; swift-format defaults to 2) and
# `AlwaysUseLowerCamelCase` off, because the numerical vocabulary (`D`, `N`,
# `Dv`, `FmoE`, `qwen36_8bit`, ...) is the same deliberate naming the
# `.swiftlint.yml` identifier_name decision already records (AUD-018).
# Everything else is the formatter's default. The binary is the one bundled with
# the pinned Xcode 27 / Swift 6.4 toolchain, so the toolchain pin IS the version
# pin -- that build's `swift-format --version` reports the branch ("main"), which
# is why there is no separate SWIFT_FORMAT_PIN.
check_swift_format() {
  echo "== swift-format: formatting is enforced under the committed .swift-format =="
  if ! command -v xcrun >/dev/null 2>&1 || ! xcrun --find swift-format >/dev/null 2>&1; then
    echo "  FAIL: swift-format is not available from the toolchain (needs Xcode 27 / Swift 6.4)"
    status=1
    return 1
  fi
  local output
  if ! output="$(cd "$ROOT" && xcrun swift-format lint --strict --recursive \
      sources tests benchmark Package.swift 2>&1)"; then
    printf '%s\n' "$output" | sed "s|$ROOT/||" | head -30
    echo "  FAIL: swift-format found formatting drift"
    echo "        fix: xcrun swift-format format --in-place --recursive sources tests benchmark Package.swift"
    status=1
    return 1
  fi
  echo "  ok (xcrun swift-format, --strict clean)"
  return 0
}

# --- javascript -------------------------------------------------------------
# The two DSH plugin packages are npm packages with no runtime dependencies.
# ESLint and Prettier are pinned exactly in each package.json and locked in its
# package-lock.json, so `npm ci` reproduces the toolchain byte-for-byte; this
# gate FAILS when the installed versions differ from those pins instead of
# skipping, and it fails when a package has no toolchain installed at all.
ESLINT_PIN="10.11.0"
PRETTIER_PIN="3.9.9"
NODE_FLOOR="22"

check_javascript() {
  echo "== javascript: eslint + prettier --check over the plugin packages (pinned $ESLINT_PIN/$PRETTIER_PIN) =="
  if ! command -v node >/dev/null 2>&1; then
    echo "  FAIL: node is not installed; the plugin packages need Node $NODE_FLOOR or newer"
    status=1
    return 1
  fi
  local node_version node_major
  node_version="$(node --version)"
  node_major="${node_version#v}"
  node_major="${node_major%%.*}"
  if [ "$node_major" -lt "$NODE_FLOOR" ]; then
    echo "  FAIL: node $node_version is installed, the packages declare engines.node >=$NODE_FLOOR"
    status=1
    return 1
  fi
  local package name eslint prettier version output checked=0
  for package in "$ROOT"/plugins/dsh-*; do
    [ -f "$package/package.json" ] || continue
    name="$(basename "$package")"
    eslint="$package/node_modules/.bin/eslint"
    prettier="$package/node_modules/.bin/prettier"
    if [ ! -x "$eslint" ] || [ ! -x "$prettier" ]; then
      echo "  FAIL: $name has no installed toolchain; run: (cd ${package#"$ROOT"/} && npm ci)"
      status=1
      return 1
    fi
    version="$("$eslint" --version | tr -d 'v')"
    if [ "$version" != "$ESLINT_PIN" ]; then
      echo "  FAIL: $name has eslint $version, this gate pins $ESLINT_PIN"
      status=1
      return 1
    fi
    version="$("$prettier" --version)"
    if [ "$version" != "$PRETTIER_PIN" ]; then
      echo "  FAIL: $name has prettier $version, this gate pins $PRETTIER_PIN"
      status=1
      return 1
    fi
    if ! output="$(cd "$package" && "$eslint" . 2>&1)"; then
      printf '%s\n' "$output" | head -20
      echo "  FAIL: $name: eslint findings (fix: npm run lint, then npm run format)"
      status=1
      return 1
    fi
    if ! output="$(cd "$package" && "$prettier" --check . 2>&1)"; then
      printf '%s\n' "$output" | head -20
      echo "  FAIL: $name: formatting drift (fix: npm run format)"
      status=1
      return 1
    fi
    checked=$((checked + 1))
    echo "  ok: $name (node $node_version, eslint $ESLINT_PIN, prettier $PRETTIER_PIN)"
  done
  if [ "$checked" -eq 0 ]; then
    echo "  FAIL: no plugin package found under plugins/"
    status=1
    return 1
  fi
  return 0
}

case "$want" in
  all)         check_force_cast; check_func_length; check_unchecked_sendable; check_converter_expert_order; check_arch_path; check_shell_portability; check_shellcheck; check_swiftlint; check_swift_format; check_javascript; check_python ;;
  force-cast)  check_force_cast ;;
  func-length) check_func_length ;;
  sendable)    check_unchecked_sendable ;;
  converter)   check_converter_expert_order ;;
  arch-path)   check_arch_path ;;
  shell)       check_shell_portability ;;
  shellcheck)  check_shellcheck ;;
  swiftlint)   check_swiftlint ;;
  swift-format) check_swift_format ;;
  format)      check_swift_format ;;
  javascript)  check_javascript ;;
  js)          check_javascript ;;
  python)      check_python ;;
  *) echo "unknown check: $want (all|force-cast|func-length|sendable|converter|arch-path|shell|shellcheck|swiftlint|swift-format|javascript|python)" >&2; exit 2 ;;
esac

exit $status
