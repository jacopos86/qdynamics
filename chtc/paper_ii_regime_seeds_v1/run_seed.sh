#!/usr/bin/env bash
set -euo pipefail
REGIME="${1:?regime_id required}"
BASE=chtc/paper_ii_regime_seeds_v1
read -r _ U G NPH SEARCH <<< "$(awk -F'\t' -v id="$REGIME" '$1==id {print $1, $2, $3, $4, $5}' "$BASE/input/regimes.tsv")"
[[ -n "${U:-}" ]] || { echo "unknown regime $REGIME" >&2; exit 3; }
OUT="raw_outputs/paper_ii_regime_seeds_v1/${REGIME}"
mkdir -p "$OUT" logs raw_outputs
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
# Fixed-structure VQE ansatz at the Paper-I regime conventions: t=1, omega0=1,
# dv=0, binary boson encoding, blocked ordering, open boundary. The cutoff is
# carried per regime (nph3 weak-phonon, nph7 strong-phonon) so the register is
# filled exactly.
python3 pipelines/time_dynamics/runners/build_fixed_vqe_conditioning_seed.py \
  --output-dir "$OUT" \
  --construction-mode conventional_fixed_layered_v1 \
  --num-sites 2 --t 1.0 --omega0 1.0 --dv 0.0 \
  --u "$U" --g-ep "$G" --n-ph-max "$NPH" \
  --layer-counts 3,4,5 --search-seed "$SEARCH" \
  --population-size 24 --generations 40 --vqe-restarts 6 \
  --delta-e-max 1.0e-4 --write-artifacts \
  --max-architecture-workers 4 --max-snapshot-workers 2
