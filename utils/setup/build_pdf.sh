#!/usr/bin/env bash
# Build a PDF from a Markdown source using pandoc + tectonic.
#
# Usage:
#   build_pdf.sh <input.md> [output.pdf]
#
# If <output.pdf> is omitted, the output filename is derived from the input
# (e.g. "HW1_report.md" -> "HW1_report.pdf"), and is written next to the
# input file. Relative paths are resolved against the current working
# directory.
#
# Designed to be invoked from the repo root, e.g.:
#   utils/setup/build_pdf.sh docs/report.md

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $(basename "$0") <input.md> [output.pdf]" >&2
    exit 1
fi

input="$1"
if [[ ! -f "$input" ]]; then
    echo "Error: input file '$input' not found" >&2
    exit 1
fi

if [[ $# -eq 2 ]]; then
    output="$2"
else
    output="${input%.md}.pdf"
fi

# Resolve relative resources (e.g. figures/foo.png) against the Markdown
# file's own directory, so the build works regardless of the current
# working directory.
input_dir="$(cd "$(dirname "$input")" && pwd)"

pandoc "$input" \
    --from markdown \
    --pdf-engine=tectonic \
    --resource-path="${input_dir}:." \
    -o "$output"

echo "Wrote $output"
