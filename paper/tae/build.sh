#!/usr/bin/env bash
# TAE workshop clone build (NeurIPS 2026): compile the assembled main.tex.
# main.tex is assembled by the integrator from skeleton.tex + blocks/; this
# script only compiles. Requires tectonic.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Compiling with tectonic"
tectonic main.tex --keep-logs --synctex=0
echo
echo "Built: paper/tae/main.pdf"
