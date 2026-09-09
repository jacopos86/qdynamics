# New-route working results

Use this report when completed results from the new selectors arrive. It is a
separate working comparison; the Paper II manuscript and frozen v1 remain intact.

## Current scope and outputs

- Cell: Hubbard--Holstein weak--fast drive `(A, omega) = (0.6, 3)`, state-motion
  controller, `t=0..20`, 501 reporting times.
- Completed new arm: greedy/joint, ascending schedule, `|A| <= 1`,
  `|D| <= floor(0.15 n)`, W50000; source `2c00f5a4`.
- Comparators: the APM and AVQDS trajectories behind the frozen Paper II v1
  Table I row. The canonical exact-reference artifact supplies every error and
  reference curve.
- Main PDF: `output/pdf/paper_ii_new_route_results/paper_ii_new_route_results.pdf`.
  Page 1 contains all three observable overlays and both Paper II comparison
  tables. Page 2 contains provenance and the W10000 CHTC tranche awaiting results.
- Inputs: `agent_guidance/time-dynamics/reports/paper_ii_new_route_results_inputs.json`.
- Data and validation: `results.json`, `calibration_receipt.json`,
  `comparability.json`, `verification_receipt.json`, and per-arm `data/` beneath
  the PDF output directory. The AVQDS source bytes are copied out of temporary
  storage to `sources/avqds/` with unchanged hashes.

## Refresh

From the scoring-curation worktree, after checking the existing memory contract:

```sh
/Users/lordskyler/Holstein_test/.venv/bin/python -B pipelines/shell/ram_guard.py --limit-mb 12000 -- /Users/lordskyler/Holstein_test/.venv/bin/python -B -m pipelines.reporting.prepare_paper_ii_working_results --inputs agent_guidance/time-dynamics/reports/paper_ii_new_route_results_inputs.json --output output/pdf/paper_ii_new_route_results
/Users/lordskyler/Holstein_test/.venv/bin/python -B pipelines/shell/ram_guard.py --limit-mb 4000 -- /Users/lordskyler/Holstein_test/.venv/bin/python -B -m pipelines.reporting.build_paper_ii_new_route_results --results output/pdf/paper_ii_new_route_results/results.json
```

The prepare step hashes raw sources and completion records, streams the two
large unused ranking arrays out of a reporting-only projection, verifies the
shared run-lock fingerprint, scores all 501 states against the existing exact
reference, checks all-call work accounting and new-route search receipts, and
reconstructs the new terminal circuit. Cached terminal costs remain bound to
the projection hash and seed hash. It never runs a new trajectory.

Add a newly completed arm to the input manifest with its own immutable ID,
raw/completion paths and hashes, case settings, label, style, budget, and stored
terminal cost where available. Use IDs beginning `new_` for the new selectors so
their all-call search receipts are audited. The current receipt adapter expects
the local benchmark's completion-observation/case schema; cluster arrivals must
be adapted from their verified worker completion and effective-action records,
without relabeling a worker receipt as a local benchmark. Remove its pending row
only after validated data are available. Keep labels explicit about search,
joint/factorized evaluation and budget when more than one variant is shown.

This first report template covers the declared A1 / 15%-deletion experiment.
For other drives, horizons or batching arms, derive the page manifest from those
run records and keep their comparisons on separate pages. Do not silently mix
their grids or reuse the weak--fast reference. Preserve all completed arms,
including unmatched results; assess equal-accuracy claims separately.

## Metric and validation conventions

- Energy, mean doublon and charge-imbalance errors average all reporting times.
  Charge imbalance is computed directly; the dimer identity with twice the
  maximum site error is checked numerically.
- Table I uses accepted internal substeps and the paper's reuse-adjusted scalar
  estimator count. Table II keeps executed estimator work separate.
- Generator histories follow the original Paper II reported-support convention.
  Exchange commits use all selector calls. Incomplete finalist history retains
  its lower-bound mark. Exact-reference circuit/search entries are inapplicable.
- The saved baseline grids differ from the immutable reference grid only at
  floating-point roundoff. The report verifies agreement to `1e-12` and reuses
  the original reference; it does not replace raw time grids.
- Validate PDF/source/result hashes, numerical means, original Table I values,
  first-page co-location of both tables with the figure, text coordinates and
  TeX overflow. The initial build passed 48 checks in `verify_report.py` beside
  the PDF. Extend its arm-specific assertions when adding cases.
- Respect the repository's PDF-viewing boundary. The initial report was checked
  using text, coordinates and build logs; it was not opened or raster-rendered.

The Fable performance audit remains a separate task; report generation does not
authorize implementing its recommendations or changing running job sources.
