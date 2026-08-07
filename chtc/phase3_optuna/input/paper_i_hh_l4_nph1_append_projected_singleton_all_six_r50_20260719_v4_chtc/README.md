# HH L4 nph1 Append projected-singleton screen

This is a six-job, prepared-but-not-submitted Paper-I diagnostic bundle. It
changes only the requested Hubbard--Holstein size/cutoff physics point and
output/resources relative to the immutable Append-v4 projected-singleton
contract.

This v4 bundle is the operational successor to cluster `8890181`, which was
removed with all six jobs still idle and `NumJobStarts=0`. V4 keeps the repaired
12-qubit execution gate and also manifest-locks the required exact-fidelity cap
to 12. The immutable scientific source archive, Hamiltonians, projected-child
pool, optimizer, seed, and Condor resources are unchanged.

- Physics: open-boundary HH, `L=4`, binary `n_ph_work=n_ph_ref=1`, all six
  `(U, lambda)` regimes from the scaling profile.
- Method: append-only ADAPT over the hard-guarded projected-singleton children
  of the unfiltered full-meta/HVA parent pool; no parent/child mixing.
- Optimization: fresh seed 7, fixed 50 rounds, Powell `maxiter=200`, accepted
  finite non-increasing Powell cap policy inherited from Append-v4.
- Resources per job: 1 CPU, 32768 MB RAM, 61440 MB disk.
- Source archive SHA-256:
  `8922435b176d635544f6fa2629da05ea7151f457e584c39e47a2ee161de94ecd`.

Build and validate locally:

```bash
python3 build_bundle.py --build
python3 -m pytest -q test_bundle.py test_archive_l4_projected_pool.py
python3 finalize_preflight.py
```

Submission is intentionally a separate user-approved CHTC action. The submit
description is `submit.sub`; do not submit this bundle merely by rebuilding it.
