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
  One results page contains all three observable overlays and both Paper II
  comparison tables. Keep provenance, pending jobs and agent orientation in
  `provenance_notes.md` beside the PDF; the builder regenerates that companion
  file. Do not restore an administrative appendix to the PDF.
- Inputs: `agent_guidance/time-dynamics/reports/paper_ii_new_route_results_inputs.json`.
- Data and validation: `results.json`, `calibration_receipt.json`,
  `comparability.json`, `verification_receipt.json`, and per-arm `data/` beneath
  the PDF output directory. The AVQDS source bytes are copied out of temporary
  storage to `sources/avqds/` with unchanged hashes.

## Automated refresh

Follow [the ordered local workflow](paper-ii-local-execution-workflow.md).
`update_paper_ii_working_results` handles archive validation, deduplicated page
admission, preparation, compilation, verification and publication. Each new run
gets its own page with the same matched baselines. Never append administrative
pages. The updater preserves previous builds and leaves the stable PDF intact
if validation fails.

The receipt adapter consumes the original CHTC completion and effective action
records directly. It verifies terminal-cost provenance against the worker's
raw-bound reporting projection. It never fabricates a local benchmark receipt.
New settings or physical regimes require matching baselines and references.

## Metric and validation conventions

- Energy, mean doublon and charge-imbalance errors average all reporting times.
  Charge imbalance is computed directly; the dimer identity with twice the
  maximum site error is checked numerically.
- Table I uses accepted internal substeps. Both estimator columns use the same
  authoritative reuse-adjusted total for APM and AVQDS. Raw acquisition
  ledgers remain source evidence; they are not a second reported cost basis.
- Generator histories follow the original Paper II reported-support convention.
  Exchange commits use all selector calls. Incomplete finalist history retains
  its lower-bound mark. Exact-reference circuit/search entries are inapplicable.
- The saved baseline grids differ from the immutable reference grid only at
  floating-point roundoff. The report verifies agreement to `1e-12` and reuses
  the original reference; it does not replace raw time grids.
- Validate PDF/source/result hashes, numerical means, original Table I values,
  co-location of both tables with the figure on every results page, equality
  of both estimator columns, text coordinates and TeX overflow. The reusable
  verifier is `pipelines.reporting.verify_paper_ii_working_results`.
- Respect the repository's PDF-viewing boundary. The initial report was checked
  using text, coordinates and build logs; it was not opened or raster-rendered.

The Fable performance audit remains a separate task; report generation does not
authorize implementing its recommendations or changing running job sources.
