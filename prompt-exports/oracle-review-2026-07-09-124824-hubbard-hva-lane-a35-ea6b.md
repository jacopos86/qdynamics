# Oracle Review

## Summary

The focused Hubbard HVA lane changes look aligned with the objective: `uccsd` and `uccsd_qeb` remain distinct, the new `uccsd_qeb_hva_blocks` pool appends grouped plain-Hubbard HVA block polynomials, emitted labels match the v3 Hubbard physical-lane classifier, and `adapt_pipeline.py` now gates the protected Hubbard pools on physical route, zero `other`, positive QEB, and positive HVA for the new pool.

## Findings

### P1 — Should Fix / Verify Before Launch

- **Reference:** launch/source-locking path; `pipelines/static_adapt/adapt_pipeline.py` prelaunch audit gate  
  **Issue:** The validation shown proves the new pool for a direct L=2 smoke setting, but the launch requirement is stricter: audit the **actual source-locked Hubbard weak baseline settings** before running depth 10. A different lattice/weak artifact could alter final surviving labels or effective shortlist settings.  
  **Suggestion:** Before launch, resolve the prior depth-15 `uccsd_qeb` Hubbard weak artifact and record a settings diff showing only:
  - `adapt_pool: uccsd_qeb → uccsd_qeb_hva_blocks`
  - `adapt_max_depth: 10`
  - expected v3 route/classifier metadata from the new lane bundle  
  Then run the cheap audit on those exact settings and require:
  - `other_count == 0`
  - `exact_other_labels == []`
  - `lane_counts["qeb_excitation"] > 0`
  - `lane_counts["hva_hamiltonian_blocks"] > 0`

No code-level launch blocker is apparent in the shown HVA changes.