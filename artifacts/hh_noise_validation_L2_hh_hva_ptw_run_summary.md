# HH L=2 Noise Hardware Validation Sweep Summary

Date: 2026-03-10
Mode: execution-only (no code changes)

## Sweep request
Run order: noiseless baseline -> runtime/no-mitigation -> runtime/readout mitigation -> runtime/DD(XY4)

## Outputs directory check
- `artifacts/json` and `artifacts/pdf` were created/used.
- `artifacts/json`: generated for Leg A only.
- `artifacts/pdf`: no files generated because Leg A crashed during PDF rendering and noisy legs failed before PDF phase.

## Leg A (completed up to JSON)
- File: `artifacts/json/hh_noise_validation_L2_hh_hva_ptw_noiseless_ideal.json`
- Command exited with failure after JSON write due PDF serialization bug in this path.
- `vqe` fields:
  - `energy_noisy`: `0.24024115479998964`
  - `energy_ideal_reference`: `0.24024115479999023`
  - `delta_noisy_minus_ideal`: `-5.828670879282072e-16`
  - `success`: `true`

## Leg B (runtime no mitigation)
- Requested target: `artifacts/json/hh_noise_validation_L2_hh_hva_ptw_runtime_nomit.json`
- Status: failed before VQE
- Error: IBM token/backend resolution failure (`InvalidAccountError`, `QISKIT_IBM_TOKEN` missing/invalid)
- Fallback attempt `--noise-mode shots` used:
  - first attempt with `symmetry_mitigation_mode postselect_diag_v1` failed (`observable_not_diagonal`) unless noisy fallback enabled
  - with noisy fallback enabled, process hit environment OpenMP crash (`OMP: Error #178: Function Can't open SHM2`)
  - fake snapshot/local backend attempts also blocked by the same environment issue or coupling-map width mismatch for 5-qubit fake backend

## Leg C (runtime readout)
- Requested target: `artifacts/json/hh_noise_validation_L2_hh_hva_ptw_runtime_readout.json`
- Status: failed before VQE
- Error: same IBM credential/backend resolution failure as Leg B (`QISKIT_IBM_TOKEN` invalid)

## Leg D (runtime DD XY4)
- Requested target: `artifacts/json/hh_noise_validation_L2_hh_hva_ptw_runtime_dd_xy4.json`
- Status: failed before VQE
- Error: same IBM credential/backend resolution failure as Leg B/C (`QISKIT_IBM_TOKEN` invalid)

## Notes on requested validation check
- Strict order was executed in the attempted command sequence.
- Full comparison of `vqe` noisy-vs-ideal triplets is only available for Leg A due failures in B/C/D.
- All requested outputs under `artifacts/json` / `artifacts/pdf` are not fully produced due runtime/auth/environment blockers.
