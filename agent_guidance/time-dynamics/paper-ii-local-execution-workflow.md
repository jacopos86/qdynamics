# Paper II local execution workflow

Author-directed order (9 September 2026):

1. Automate the working results PDF: append one page per newly completed,
   validated `t=20` trajectory. Each page contains that new APM run, frozen v1
   APM, AVQDS and the matching exact solution in the three observable plots and
   both comparison tables. Keep page order stable and deduplicate by raw hash.
   Administrative notes belong in companion Markdown.
2. Both tables use the authoritative reuse-adjusted estimator cost, for APM and
   AVQDS alike. The author selected this on 9 September 2026. The old APM v1 row
   is 5.26 in both tables (units of 10^8). There is no alternative non-reuse
   reporting route. Preserve raw execution ledgers for evidence and accounting
   reconciliation; these are not the published estimator-cost column.
3. Inspect the already-submitted CHTC jobs, collect their immutable receipts,
   validate arrivals, and refresh the PDF. Follow the standing SSH/Duo procedure
   in `paper-ii-run-protocol.md`; actually initiate authentication when due.
   The six owned science procs are `10024096.0` through `.5`. Keep them running;
   do not resubmit, hold or mutate their source packages.
4. Review the complete Fable audit independently. Reconcile every finding ID,
   inspect its source and evidence, and record dispositions and a dependency-
   ordered implementation plan before changing production behavior. Follow
   `output/paper_ii_selector_performance_audit_20260909/fable_5_1/SECOND_PASS_CONTRACT.md`.

Read the current stage and receipts from
`output/paper_ii_t20_successful_settings_local_20260909_2c00f5a4/continuation_state.json`.
A scientific or algorithmic ambiguity permits discussion with the author; elapsed
waiting time is not a scientific decision. Routine automation runs directly in
the root task. The author permits subagents for performance improvement,
including bounded investigation, implementation and independent review.

## Collection and automatic page refresh

Collect the six declared actions from remote `archives_by_receipt/<sha>/` into
`output/paper_ii_t20_successful_settings_local_20260909_2c00f5a4/chtc_t20_arrivals/<sha>/`.
Copy the archive and its `.archive.json` together. The report input manifest
binds the exact package manifest and source archive. Archive names alone are not
identity.

After collection, or to inspect already-collected arrivals, run from this worktree:

```sh
/Users/lordskyler/Holstein_test/.venv/bin/python -B pipelines/shell/ram_guard.py --limit-mb 12000 -- /Users/lordskyler/Holstein_test/.venv/bin/python -B -m pipelines.reporting.update_paper_ii_working_results --inputs agent_guidance/time-dynamics/reports/paper_ii_new_route_results_inputs.json --output output/pdf/paper_ii_new_route_results --verify-python /Users/lordskyler/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

Check the existing memory contract before launching. Add `--rebuild` only for an
explicit reporting change. Normal unchanged ticks do no geometry, compilation
or PDF work. A file lock prevents overlapping refreshes.

The updater validates archive, action, source, completion and sidecar hashes;
checks the full horizon, shared seed/physics/numerics, compiled resources,
estimator ledger and search receipts; scores the immutable exact reference;
builds each new results page; and checks page placement and estimator equality.
Only a fully verified build replaces the stable PDF. Prior builds and all raw
trajectories remain preserved. Invalid arrivals receive a separate review record.

This adapter covers the submitted weak-fast, state-motion, singleton-insertion,
15%-deletion tranche. Other physical regimes need their own matched baseline and
reference manifest. Batched insertion reuse accounting requires scientific
review before extending this singleton accounting rule. An unmatched or partial
result must not be portrayed as a matched completed `t=20` trajectory.

Outputs live under `output/pdf/paper_ii_new_route_results/`: the stable PDF,
`results.json`, `pdf_build_receipt.json`, `verification_receipt.json`,
`refresh_receipt.json`, `provenance_notes.md`, persistent `data/`, and preserved
`builds/`. The existing 15-minute heartbeat invokes this workflow after due
collection; it stays quiet when nothing meaningful changes.

## Campaign continuation after the recovery tranche

The six recovery jobs do not complete the 72-cell selector campaign. Follow
`paper-ii-selector-campaign.md` → Current campaign scope for the resolved axes
and the submitted 10,000/50,000 matrix. Retain the six A1/D15% results separately from
strict A1/D1 and batched A4/D15% matrix cells. Keep the existing heartbeat active
until the full authorized campaign is accounted for, or the author changes that
scope. Advance ready matrix preparation and submission using the standing CHTC
authorization; no blanket hold is imposed by ongoing performance investigation.
