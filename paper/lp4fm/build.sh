#!/usr/bin/env bash
# Build the LP4FM workshop paper (NeurIPS 2026): compile the single assembled main.tex.
# Requires tectonic.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Compiling with tectonic"
tectonic main.tex --keep-logs --synctex=0
echo
echo "Built: paper/lp4fm/main.pdf"
