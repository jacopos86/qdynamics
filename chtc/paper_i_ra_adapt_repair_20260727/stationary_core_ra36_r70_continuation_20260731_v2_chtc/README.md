# Stationary-core RA r50→r70 v2 scaffold

This is an inert, paper-facing continuation scaffold for the 36 stationary
RA-ADAPT cells.  Every eventual row is an authenticated round-50 to round-70
resume.  It does not authorize execution, create a live submit descriptor, or
contact CHTC.

The sibling binds the existing
`stationary_core_ra36_r70_continuation_20260731_v1_chtc` operational overlay
by its exact sealed manifest and job bytes.  It deliberately does not invoke
the parent overlay's dynamic semantic re-derivation.  As observed on
2026-07-31, that re-derivation currently fails on the weak/weak macro-always
`resume_source` field after later provenance state changed; this scaffold
records that as parent-validator drift, not as permission to rewrite the
sealed artifact.

Current state:

- 27 pointer-closed resume archives are referenced read-only from the sealed
  parent;
- nine exact cluster `9397758`, proc `0..8` predecessor bindings are explicit
  fail-closed placeholders.  Their compact resumes and binding receipts live
  under the separate `stationary_core_ra36_r70_continuation_20260731_input_evidence_v1`
  root, so completing them cannot mutate this v2 directory;
- all 36 scientific-setting objects and their digests are projected exactly
  from the sealed parent jobs;
- each row's transfer plan names one resume archive and one small source
  archive.  The aggregate `resume_inputs/` directory is forbidden as a Condor
  transfer input.

Validate the checked-in inert scaffold with:

```text
python3 validate_scaffold.py
```

`materialize_scaffold.py` is the exclusive initial builder and deliberately
refuses to rewrite an already materialized sibling.

`build_activation.py` never adds files to v2.  It consumes v2 plus external
evidence and, only after full validation, atomically publishes the new
`stationary_core_ra36_r70_continuation_20260731_v3_chtc` sibling.  Validation
includes a full scan of every inherited and new resume archive, revalidation
of each controlled-cycle attempt and retrieval receipt, the actual runtime
tar/member inventory, a remote `ap2001` SIF byte-verification receipt, four
successful round-50 resource-history observations, and 36 external
authorizations against deterministic final-job bytes.

Authorization is intentionally two-step and non-circular.  First provide an
input receipt with status `evidence_complete_authorizations_pending` and run:

```text
python3 build_activation.py --activation-inputs <intent-input.json> --prepare-intent
```

That prints the exact control-plane and job digests without writing v3.  After
external authorization receipts bind those digests, provide the full input
receipt with status `evidence_and_authorizations_complete` and run the command
without `--prepare-intent`.  The resulting descriptor is an ordinary queue,
starts with every row held, has no automatic release, and is not submitted.

The resource gate permits creation/submission of that all-held descriptor from
authenticated round-50 history.  It does not authorize broad release.  One
declared `single_pauli_word_v1:nph7` round-70 pilot must be separately
authorized and completed before any later broad-release receipt can be issued.
Submission, pilot release, broad release, and paper-evidence adoption remain
separate operations.
