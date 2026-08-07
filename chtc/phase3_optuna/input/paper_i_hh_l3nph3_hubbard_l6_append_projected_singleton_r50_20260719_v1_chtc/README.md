# Higher-L Append projected-singleton screen

This is a five-job, prepared-but-not-submitted Paper-I diagnostic bundle. It is
separate from the six-job HH `L=4`, `n_ph=1` bundle and preserves the immutable
Append-v4 method contract while changing only the declared model/size points,
their exact-reference locks, and required resources.

- Physics: open-boundary HH `L=3`, binary `n_ph_work=n_ph_ref=3` at
  weak/intermediate/strong `U` with strong Holstein coupling; and open-boundary
  Hubbard `L=6` at weak/strong `U`. Hubbard has no bosonic cutoff; its
  `n_ph_work=1` field is inert family-normalization metadata and `n_ph_ref` is
  null.
- Method: append-only ADAPT over hard-guarded projected-singleton children of
  the family-native full-meta parent pool. HH uses the unfiltered full-meta/HVA
  parent pool. Parent and child candidates are never mixed.
- Optimization: fresh seed 7, fixed 50 rounds, Powell `maxiter=200`, with the
  Append-v4 finite non-increasing Powell-cap continuation policy.
- Runtime locks: 12-qubit resource and exact-fidelity caps; 9000 expanded-pool
  cap; state-sector, exact-reference, pool-order, estimator-ledger, fidelity,
  and Qiskit-cost validation are required.
- Resources per job: 1 CPU, 32768 MB RAM, 61440 MB disk.
- Source archive SHA-256:
  `36bddec0e2f533d674e61318087de3ced1d9108aeaf127ed263a15024457e2a0`.
- The path-safe worker gate accepts the uppercase `L` used by all five
  canonical job identifiers.

Build and validate locally:

```bash
python3 build_bundle.py --build
python3 -m pytest -q test_bundle.py
python3 finalize_preflight.py
```

Submission is intentionally a separate user-approved CHTC action. The submit
description is `submit.sub`; do not submit this bundle merely by rebuilding it.
