# SR-SNAKE v4 six-regime parent bundle

This generated bundle freezes commit
`dfe8d8cad94167ebb1be6f919eeab3a64bb904d2` and v4 contract digest
`447b8fe3f4fef340fbb1cd5d221a0234826ba80c7e4e405937004e4ab25bec93`.
It contains six fresh round-0 to round-30 parents. Weak--weak and
intermediate--weak end at round 30; the four deeper regimes may receive a
separate authenticated round-30 to round-50 continuation only after their
parents are fetched and validated.

The repaired depth-8 smoke records a non-null supported rank on every round,
one adaptive-trust update per accepted full-coordinate refit, closed estimator
accounting, ordered checkpoint/Qiskit replay, and explicit unresolved
shadow-damping receipts that are zero-query diagnostic no-ops. No mapped-seed
energy or displacement evidence is invented. The production-composition prune
regression covers one model nominee, one measured delete/refit, measured-energy
accept/reject authority, conservative rho/mu updates, zero added quantum query,
and live-only execution.

The bundle performs archive-only import/parse and focused-regression checks,
strict per-round response/checkpoint/leakage/ledger/prune validation, explicit
Condor output remapping, and writes a non-executable authenticated round-30 to
round-50 continuation template.

This bundle has **not** been submitted. See
`SUBMISSION_READY_NOT_YET_SUBMITTED.md`. The user authorized submission, the
remote image SHA-256 was rechecked, and that image imported Qiskit 2.3.1 and
instantiated the 156-qubit `fake_marrakesh` backend. `submit.sub` is therefore
enabled with `requirements = TARGET.HasSIF`; the authenticated main agent owns
the actual transfer and `condor_submit` action.

Rebuild and validate:

```bash
python3 chtc/phase3_optuna/input/paper_i_hh_sr_snake_v4_candidate_all_six_20260716_v1_chtc/build_bundle.py
python3 chtc/phase3_optuna/input/paper_i_hh_sr_snake_v4_candidate_all_six_20260716_v1_chtc/test_bundle.py
```

Preserve all hashes during transfer and do not modify the bundle between the
recorded preflight and submission.
