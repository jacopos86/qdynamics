# pipelines/reporting AGENTS.md

This subtree owns report generation, PDF assembly, bundle summaries, and
artifact interpretation. Humans mainly consume this repo through PDFs, so
reports must be both readable and recoverable by future agents.

## PDF Manifest Contract

- Every generated PDF starts with a clear parameter manifest on the first page or first text-summary page.
- Render manifests from normalized run/artifact metadata when available; do not rely on a raw command dump as the manifest.
- Include model family/name, ansatz type(s), drive enabled status, run-defining physics parameters, route/method identifiers, time grid/propagation knobs, optimizer/noise/mitigation fields when relevant, seeds when present, and artifact/git provenance when available.
- Drive-enabled reports include waveform parameters and reference-method metadata.
- HH reports include `omega0`, `g_ep`, `n_ph_max`, sector/filter information when available, and exact/reference target definitions.
- For Paper-I static HH ADAPT reports, compare the variational energy only with
  exact diagonalization at the identical `n_ph_max`. Label `exact_gs_energy`
  and `abs_delta_e` as same-cutoff quantities. Do not require or display a
  higher-cutoff reference diagnostic unless the user explicitly requests a
  separate cutoff-sensitivity report.

## PDF Builder Boundary

- Paper-facing evidence, current-status, audit, and table PDFs must be LaTeX-built from `.tex` sources with LaTeX tables unless the user explicitly asks for a disposable diagnostic mockup.
- Keep PDF/reporting modules import-light. Module import must not scan artifact trees, load large JSON/source-map bundles, compile LaTeX, create output files, or import optional heavy scientific stacks for routes that do not need them. Put that work behind `main()` or an explicit build function.
- Top-level ReportLab PDF builders are legacy/diagnostic only. New paper-facing PDF paths should emit `.tex` and use `latexmk` or the approved `tectonic` fallback.
- `test/test_reporting_pdf_imports.py` guards PDF entrypoint import latency and keeps top-level ReportLab usage on an explicit legacy allowlist.

## Report Semantics

- For HH noisy/mitigation reports, primary `|dE|` means `|E_exact - E_noisy(with mitigation)|`.
- If a report includes noisy-vs-ideal imported-circuit bias, label it explicitly as `dE_to_ideal` or equivalent.
- Safe-test scalar metrics must be visible in drive/amplitude comparison scoreboard pages.
- Full safe-test time-series pages are conditional: failure, near-threshold, or explicit verbose report mode.

## Agent Retell

- After a completed HH run, default to a short objective-aware in-chat retell before writing persistent report files.
- Default compact format is three lines with no blank line: `Objective<...>`, `Why/Intent<...>`, `Suggested Next step/how this fits into broader picture<...>`.
- Keep interpretation logic/math/physics-first rather than repo-prose-first.
- Only create or update persistent markdown/PDF report files when report output is in scope or explicitly requested.

## Artifact Scope

- Use `docs/reports/` only when the user asks for docs material or PDF/report output is in scope.
- Report artifacts should stay in the established JSON/PDF artifact directories for their pipeline surface.
- Preserve existing artifact naming conventions for comparison/bundle outputs unless changing them is an explicit task.
