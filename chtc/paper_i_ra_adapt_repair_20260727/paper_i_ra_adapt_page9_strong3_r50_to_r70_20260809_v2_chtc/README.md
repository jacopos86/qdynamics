# Page-9 strong-sector accepted-state continuations to round 70

Revision v2 preserves the inert v1 package and changes only staging-path
portability: an explicitly supplied `/staging/jsstrobel/...` archive is compared
lexically after absolute `.`/`..` normalization, without resolving a symlinked
mount ancestor. The archive itself must still be a regular non-symlink file with
the exact declared size and SHA-256, and its selected tar members remain subject
to the same safety and content checks.

This inert package preserves the exact Page-9 v3 route and changes only
`request.execution.stop.maximum_controller_rounds` from 50 to 70. It contains
three strong-Holstein rows: weak--strong, intermediate--strong, and
strong--strong. Weak--strong and intermediate--strong are bound to their
authenticated remote full-attempt archives. Strong--strong is deliberately
blocked until its terminal round-50 full archive, worker receipt, summary, and
updated visible adapter exist.

The worker uses `AcceptedStateResume`, verifies the exact checkpoint/estimator-
ledger/verified-resume-sidecar triplet and exact accepted prefix, and composes
the Page-9 source archive with the source-locked accepted-energy-only 128-ULP
controller repair. All non-energy replay fields remain exact. The package
vendors the ijson 3.5.1 pure-Python streaming backend subset it needs, so the
remote image requires no ambient `ijson` installation.

## Validate the inert package

```bash
python -B chtc/paper_i_ra_adapt_repair_20260727/paper_i_ra_adapt_page9_strong3_r50_to_r70_20260809_v2_chtc/validate_package.py --worker-preflight
```

## Exact remote materialization commands

Run these from the active repository checkout on CHTC. They stream only the
three content-addressed resume members out of each multi-gigabyte full archive;
they do not copy or expand the full archive.

```bash
page9_package=chtc/paper_i_ra_adapt_repair_20260727/paper_i_ra_adapt_page9_strong3_r50_to_r70_20260809_v2_chtc
resume_root=/staging/jsstrobel/paper_i_ra_adapt_page9_strong3_r70_20260809_v2/resume_inputs

python -B "$page9_package/materialize_resume_input.py" \
  --job "$page9_package/jobs/phase3_qiskit_denominator_no_lanes__weak_strong__nph7__ra_global_singleton_plateau_commutation__resume_r50_to_r70.json" \
  --source-archive /staging/jsstrobel/paper_i_ra_adapt_completed_20260808/raw/phase3_qiskit_denominator_no_lanes__weak_strong__nph7__ra_global_singleton_plateau_commutation__9588784__3.tar.gz \
  --output-dir "$resume_root/phase3_qiskit_denominator_no_lanes__weak_strong__nph7__ra_global_singleton_plateau_commutation__resume_r50_to_r70"

python -B "$page9_package/materialize_resume_input.py" \
  --job "$page9_package/jobs/phase3_qiskit_denominator_no_lanes__intermediate_strong__nph7__ra_global_singleton_plateau_commutation__resume_r50_to_r70.json" \
  --source-archive /staging/jsstrobel/paper_i_ra_adapt_completed_20260808/raw/phase3_qiskit_denominator_no_lanes__intermediate_strong__nph7__ra_global_singleton_plateau_commutation__9588784__4.tar.gz \
  --output-dir "$resume_root/phase3_qiskit_denominator_no_lanes__intermediate_strong__nph7__ra_global_singleton_plateau_commutation__resume_r50_to_r70"
```

After strong--strong round 50 completes, use the same command with its exact
preserved full archive plus:

```text
--completed-adapter <updated-page9-adapter.json>
--source-worker-receipt <strong-strong-worker-receipt.json>
--source-summary <strong-strong-summary.json>
```

`materialize_resume_input.py` rejects those inputs unless all hashes, identities,
pointer targets, and the round-50 accepted prefix agree.

Activation is a separate, non-colliding overlay created by
`activate_package.py`. It refuses to render `submit.sub` until all three
materializations and a self-digested three-cell execution/submission request are
present. Activation also hashes the pinned image and runs the vendored parser
inside that image with `python -I -S -B` before rendering the descriptor. The
sealed package itself contains no `submit.sub`, authorization, or submission
state.
