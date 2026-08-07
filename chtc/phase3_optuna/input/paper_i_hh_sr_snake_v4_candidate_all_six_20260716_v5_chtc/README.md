# SR-SNAKE v4 Phase-I/II no-lambda-F six-regime bundle revision v5

This directory is the immutable-successor scaffold for the stale v2 bundle.
It preserves the six v2 Hubbard--Holstein interaction points, resources,
optimizer budgets, and fresh round-0 to round-30 horizon.  It applies the two
user-approved scientific changes: the Phase-I/II energy model below and the
padding-free same-cutoff binary truncations `n_ph=3` for weak Holstein and
`n_ph=7` for strong Holstein.

- `phase1_energy_model=first_order_fs_trust_v1`;
- `phase2_curvature_policy=measured_required_fail_closed_v1`;
- `phase2_cheap_curvature_proxy_policy=off`;
- no Phase-I/II lambda-F proxy or missing-curvature substitution;
- legacy HH quadrature preseed disabled so depth and controller round both
  start at zero; HVA generators remain available in `full_meta`;
- malformed, absent, unprovenanced, or nonfinite Phase-II curvature aborts the
  run before novelty fallback can execute.

Revision v5 is the source-only repair successor to the failed v3 submission.
The scientific argv and route digest are unchanged. The validator now accepts
the explicit `--adapt-disable-hh-seed` flag because it matches the v4 contract,
while historical profiles still reject that setting as drift. The tested source
is frozen at commit `8a746d244a15e2cb16099a732e78e1110a8e59f2` and tree
`6cb596ab953386a9c9a3b0698e7b1489e3b0f02e`.

The main agent completes the bundle only after the corrected source is committed
and pushed:

1. Replace `EXPECTED_HEAD` and `EXPECTED_TREE` in `build_bundle.py` with the
   confirmed 40-character commit and tree hashes.
2. Keep `SUBMISSION_ENABLED = False`; run `build_bundle.py` and
   `test_bundle.py` to prove local/archive-only gates and preserve the generated
   blocked preflight.
3. Record the authenticated remote image/Qiskit/FakeMarrakesh check in
   `remote_execution_gate.json`.
4. Only after every gate passes, set `SUBMISSION_ENABLED = True`, rebuild from
   the same frozen commit, rerun `test_bundle.py`, and verify the generated
   `submit.sub` uses `requirements = TARGET.HasSIF`.

The bundle validators require exact Phase-I/II policy serialization, forbid
Phase-I/II lambda-F flags in argv, parse every exact scientific argv through
the archived production CLI, require every Phase-II full-candidate occurrence to be
matched by a validated finite curvature receipt, and require zero proxy and
missing-curvature-fallback occurrences.  Existing v4 Phase III, pruning,
adaptive trust, full-response/full-refit, symmetry, padding, and Qiskit gates
remain in force.
