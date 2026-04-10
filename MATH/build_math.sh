#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEX_ONLY=0
for arg in "$@"; do
  if [[ "$arg" == "--tex-only" ]]; then
    TEX_ONLY=1
    break
  fi
done

# Refresh the markdown-derived manuscript artifacts first.
python3 "$SCRIPT_DIR/build_math_from_md.py" "$@"

# Build the standalone 17A symbol guide directly from its TeX source unless
# the caller requested a TeX-only refresh of the generated manuscript twins.
if [[ "$TEX_ONLY" -eq 0 ]]; then
  pushd "$SCRIPT_DIR" >/dev/null
  xelatex -interaction=nonstopmode -halt-on-error 17A_symbol_guide.tex >/dev/null
  xelatex -interaction=nonstopmode -halt-on-error 17A_symbol_guide.tex >/dev/null
  popd >/dev/null
  echo "Wrote $SCRIPT_DIR/17A_symbol_guide.pdf"
fi
