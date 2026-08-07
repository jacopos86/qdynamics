# Shared Scientific and Artifact Invariants

Read this file only for scientific code, mathematical defaults, route identity,
run/report behavior, or paper-support work. Paper ownership, manuscripts,
evidence transfer, and workflow-specific gates remain governed by
`MATH/AGENTS.md` and the selected paper lane.

## Algorithm synchronization

Each method lane must faithfully implement its active scientific source and
typed canonical settings. When the user requests an algorithmic change:

1. classify it as a semantic change, canonical-default change, or named
   experiment/ablation;
2. state the intended behavior and resolve only genuinely unsettled scientific
   intent;
3. update implementation, tests, and canonical settings together;
4. do not silently rewrite a manuscript;
5. stop if code, tests, settings, and manuscript mathematics disagree.

A named experiment or one-off ablation must not create a new public
architecture seam or silently alter canonical defaults.

## Quantum representation

- Use `e/x/y/z` internally and convert to `I/X/Y/Z` only at output boundaries.
- Pauli words are ordered left-to-right as `q_(n-1) ... q_0`; qubit 0 is the
  rightmost character.
- Build Jordan--Wigner ladder operators through
  `pauli_polynomial_class.py`.
- Use `n_p = (I - Z_p)/2` in the repository Pauli-string convention.
- Use `src.quantum.qubitization_module.PauliTerm` as the canonical
  `PauliTerm`.
- Do not introduce Qiskit into production/core VQE paths. Qiskit belongs in
  validation, reference, compilation-cost, and hardware-integration paths.
- The `A=0` drive must agree with no-drive within the defined safe-test
  threshold.
- Do not change Pauli ordering, Jordan--Wigner mapping, drive defaults, route
  identity, or artifact semantics without matching tests or an explicit user
  decision.

## Exact-reference isolation

AP-McLachlan support-patch decisions, integrator choices, and parameter updates
may use only measurement-compatible quantities for the prepared current or
candidate ansatz state. Exact or classical reference trajectories are post-run
inputs for plots and error metrics only; controllers, scoring, integration, and
online tuning must not query them.

The same principle applies generally: exact-energy stop targets must be
predeclared and route-bound; exact-reference information must not leak into
candidate selection or online control.

## Run and report artifacts

- Every run emits a normalized machine-readable manifest.
- Every evidence PDF includes a parameter manifest derived from the run
  manifest, normally in a final appendix.
- Reader-facing report pages lead with results, comparisons, and resource costs
  when those are in scope. Put the manifest on page one only when requested.
- User-requested evidence, status, audit, and table PDFs are LaTeX-built from
  `.tex` sources with LaTeX tables. Disposable diagnostic mockups require an
  explicit request.
- Do not downscale a production run below the scale fixed by its source lock.

## Paper-II calibration minimum

Before Paper-II scheduler submission, aggregation, evidence-PDF generation, or
table update, verify:

- every method in a comparison uses the same static-ansatz seed hash;
- no paper-facing row uses staged, fallback, pending, or recovery seeds;
- `latest_phase3_source_artifact_missing_locally` is false;
- controller settings are calibrated by coarse Hamiltonian class rather than
  individual family;
- supported comparator rows pass Qiskit parity and parity deltas are not
  reported as physical accuracy;
- exact reference data remains reporting-only;
- required spectra, fidelity, hardware-cost, and estimator-burden fields are
  present or explicitly recorded as missing.

Record failed checks and block automated evidence transfer. Report objective
facts; the user alone decides whether to repair, rerun, defer, or adopt
evidence.
