#!/usr/bin/env bash
set -euo pipefail
REGIME="${1:?regime_id required}"
BASE=chtc/paper_ii_regime_seeds_v1
read -r _ U G NPH OSEED <<< "$(awk -F'\t' -v id="$REGIME" '$1==id {print $1, $2, $3, $4, $5}' "$BASE/input/regimes.tsv")"
[[ -n "${U:-}" ]] || { echo "unknown regime $REGIME" >&2; exit 3; }
OUT="raw_outputs/paper_ii_regime_seeds_v1/${REGIME}"
mkdir -p "$OUT" logs raw_outputs
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Frozen three-complete-layer fixed-structure VQE (the prospective HH seed
# policy three_complete_ordered_hh_hamiltonian_layers_v2): four seeded L-BFGS-B
# restarts, exact compiled adjoint gradient, hard |E - E0| <= 1e-6 gate, no
# conditioning search, no driven-snapshot fits, no exact data in the optimizer.
# Paper-I regime conventions: t=1, omega0=1, dv=0, binary boson encoding,
# blocked ordering, open boundary, (n_up, n_dn) = (1, 1). The cutoff is carried
# per regime so the phonon register is filled exactly (1, 3, or 7).
python3 pipelines/time_dynamics/runners/build_hh_fixed_layer_seed.py \
  --regime-id "$REGIME" \
  --u "$U" --g-ep "$G" --n-ph-max "$NPH" \
  --optimizer-seed "$OSEED" \
  --num-sites 2 --t 1.0 --omega0 1.0 --dv 0.0 --n-up 1 --n-dn 1 \
  --boson-encoding binary --ordering blocked --boundary open \
  --output-json "$OUT/seed.json"
