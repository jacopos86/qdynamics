# Workspace Cleanup Manifest

Date: 2026-02-28

## Scope
- Cleanup profile: safe + reversible.
- `.venv` excluded from cleanup.
- Generated artifacts archived-then-pruned (kept recoverable).
- Test layout normalized under `test/`.

## Baseline Snapshot (Before)
- `.venv`: 354M
- `Fermi-Hamil-JW-VQE-TROTTER-PIPELINE/artifacts`: 34M
- `.pytest_cache`: 28K
- `__pycache__` (repo root): 352K
- `src/__pycache__`: 8.0K
- `test/__pycache__`: 128K

## Actions Completed
1. Artifact archive/prune pass under:
   - `Fermi-Hamil-JW-VQE-TROTTER-PIPELINE/artifacts/archive_20260228_121304`
2. Kept in place:
   - `commands.txt`
   - `drive_reg`
   - `hva_uccsd_heavy`
   - `json`
   - `logs`
   - `pdf`
3. Moved into archive (`pruned/`):
   - `hva_drive_L3.json`
   - `hva_drive_L3.pdf`
   - `hva_drive_L3_heavy_20260227_133346.json`
   - `hva_drive_L3_heavy_20260227_133346.pdf`
   - `hva_drive_L4_medium.json`
   - `hva_drive_L4_medium.pdf`
   - `hva_vqe_heavy`
   - `hva_vqe_heavy_termwise`
   - `pauli_poly_fix_validation_20260225_183259`
   - `plots`
   - `run_L_drive_accurate_L2_20260223_020012`
   - `run_L_drive_accurate_L4_20260223_165840`
   - `scaling_preset_L2_L6_20260222_101331`
   - `testing3d`
4. Removed cache directories:
   - all repo-local `__pycache__/`
   - all repo-local `.pytest_cache/`
5. Updated ignore policy:
   - `.gitignore` now includes `**/.pytest_cache/` and `*.swp`.
6. Obsolete test cleanup:
   - removed `test/test_tex2text.py` because `tex2text.py` was intentionally deleted.

## Validation
- `python -m py_compile` on remaining `test/*.py` files: pass.
- `python -m pytest -q test/test_pauli_polynomial_ops.py`: pass (`3 passed`).

## Current Snapshot (After)
- `.venv`: 354M (unchanged by policy)
- `Fermi-Hamil-JW-VQE-TROTTER-PIPELINE/artifacts`: 34M (contents reorganized; archived in-place)
- `.pytest_cache`: removed in final sweep
- repo-local `__pycache__/`: removed in final sweep

## Known Blockers
- Git remote sync could not be completed in this environment:
  - `git fetch origin` failed with: `error: cannot open '.git/FETCH_HEAD': Operation not permitted`

## Notes
- Archive metadata is also recorded in:
  - `Fermi-Hamil-JW-VQE-TROTTER-PIPELINE/artifacts/archive_20260228_121304/ARCHIVE_INDEX.md`
