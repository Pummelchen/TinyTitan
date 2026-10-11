#!/usr/bin/env bash
# Regenerates the figures, inlines them into the template and prints to PDF.
#
# TINYTITAN_PAPER_CHROME names the browser to print with. It exists so the launch
# can be driven by a stub in benchmark/test_paper_build_pdf.py; unset it and the
# default is still Google Chrome at its usual absolute path.
set -euo pipefail
cd "$(dirname "$0")"
python3 figures.py >/dev/null
python3 - <<'PY'
import re, pathlib
t = pathlib.Path("paper.template.html").read_text()
t = re.sub(r"\{\{FIG:(\w+)\}\}", lambda m: pathlib.Path(f"fig/{m.group(1)}.svg").read_text(), t)
pathlib.Path("paper.html").write_text(t)
PY

html="$PWD/paper.html"
pdf="$PWD/continuitycore-paper.pdf"
chrome="${TINYTITAN_PAPER_CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"

# Before and after, not mtime versus the HTML: the launch has to have changed the
# artifact it was told to write. A Chrome that exits 0 without writing one -- an
# instance that cannot take the profile lock -- used to leave the previous paper
# on disk, and the page count below printed over it as this build's result.
before=""
if [[ -f "$pdf" ]]; then
  before="$(stat -f '%m-%z' "$pdf")"
fi

# No 2>/dev/null: the redirect also swallowed bash's own message when the binary
# was missing, so a failed build reported nothing at all.
if ! "$chrome" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$pdf" "file://$html"; then
  printf 'ERROR: %s exited non-zero while printing %s; its output is above\n' "$chrome" "$pdf" >&2
  exit 1
fi

after=""
if [[ -f "$pdf" ]]; then
  after="$(stat -f '%m-%z' "$pdf")"
fi
if [[ -z "$after" ]]; then
  printf 'ERROR: the print exited 0 but wrote no PDF at %s\n' "$pdf" >&2
  exit 1
fi
if [[ "$after" == "$before" ]]; then
  printf 'ERROR: the print exited 0 and left %s exactly as it found it, so its page count\n' "$pdf" >&2
  printf '       below is the previous build'"'"'s artifact, not a result of this one.\n' >&2
  exit 1
fi

pages="$(python3 -c 'import re,sys,pathlib; print(len(re.findall(rb"/Type\s*/Page[^s]", pathlib.Path(sys.argv[1]).read_bytes())))' "$pdf")"
if [[ "$pages" == "0" ]]; then
  printf 'ERROR: %s holds no page objects: refusing to report it as a rendered paper\n' "$pdf" >&2
  exit 1
fi
printf 'pages: %s bytes: %s\n' "$pages" "$(stat -f '%z' "$pdf")"
