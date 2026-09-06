#!/usr/bin/env bash
# LP4FM workshop clone build (NeurIPS 2026): compile the assembled main.tex.
# This pack ships a single assembled file. Requires tectonic.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Compiling with tectonic"
tectonic main.tex --keep-logs --synctex=0
echo
echo "Built: paper/lp4fm/main.pdf"
