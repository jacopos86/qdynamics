---
name: paper-i-noise-model-primer
description: Reproduce the Paper-I SNAKE noise-model appendix workflow, including the noise source-of-truth note, math-primer PDFs, scalar or gate-noise diagnostic plots, GPT-Pro modeling handoffs, and appendix provenance. Use only for explicit Paper-I noise-model equations, robustness artifacts, N_eff interpretation, or noisy-objective modeling; do not use for ordinary Paper-I algorithm implementation, route design, tests, or non-noise manuscript work.
---

# Paper-I Noise Model Primer Skill

Use this skill for Paper-I SNAKE noise-model appendix work: scalar value-noise math, synthetic gate-noise math, dense math-only PDF primers, noise diagnostic plots, and GPT/modeling handoffs.

This skill composes existing skills; it does not replace them.

## Required companion skills / docs

When this skill triggers, load only what the task needs:

1. `$journal-math-manuscript-refiner` for any appendix prose, caption, `.tex`, or PDF-facing edit.
2. `MATH/paper_facing/shared/journal_math_skill_supplement.md` for Paper-I manuscript constraints.
3. Global `pedagogical-math-primer` skill for standalone math-primer PDFs or equation explanations.
4. Global `pdf` skill when compiling or visually checking a PDF.
5. Global `gpt-pro-handoff` skill when creating a GPT-Pro modeling export.

Do not load every companion automatically. Load only the companion whose
workflow the current request actually requires.

## Canonical support artifacts

Current source-of-truth note:

- `MATH/paper_facing/paper_I_static_scaffold/noise_model_source_of_truth_20260609.md`

Current math-only primer:

- TeX: `MATH/paper_facing/paper_I_static_scaffold/noise_model_math_primer_20260609.tex`
- PDF: `MATH/paper_facing/paper_I_static_scaffold/noise_model_math_primer_20260609.pdf`
- shareable copy: `output/pdf/noise_model_math_primer_20260609.pdf`

Use these as templates for future updates. Do not replace them silently; create dated successors when the model materially changes.

## Implementation hooks to verify before writing math

Scalar value-noise code hooks:

- `pipelines/static_adapt/cli_config.py::Phase3OracleGradientConfig`
- `pipelines/static_adapt/cli_config.py::_resolve_value_noise_std_contract(...)`
- `pipelines/static_adapt/cli_config.py::_validate_value_noise_config(...)`
- `pipelines/static_adapt/adapt_pipeline.py::_apply_phase3_inner_value_noise(...)`

Synthetic gate-noise hooks:

- `phase3_oracle_gradient_mode = aer_density_matrix_synthetic_depolarizing`
- `phase3_oracle_execution_surface = expectation_v1`
- `phase3_oracle_inner_objective_mode = noisy_v1`
- `phase3_oracle_synthetic_depolarizing_1q_error = p1q`
- `phase3_oracle_synthetic_depolarizing_2q_error = p2q`
- defaults in `pipelines/exact_bench/noise_oracle_defaults.py`

Run-record forwarding / manifest hooks:

- `chtc/phase3_optuna/run_task.py` fields: `value_noise_contract`, `synthetic_depolarizing_contract`, `noise_contract`.

## Mathematical contract

Use this core model unless the code has changed:

\[
E_k(\theta)=\bra{\psi_k(\theta)}H\ket{\psi_k(\theta)},
\qquad
\ket{\psi_k(\theta)}=U_k(\theta)\ket{\phi_0}.
\]

Synthetic gate noise changes the expectation surface:

\[
E_k^{\rm gate}(\theta;p_{1q},p_{2q})
=\operatorname{Tr}\!\left[H\rho_k^{\rm gate}(\theta;p_{1q},p_{2q})\right].
\]

Scalar value noise adds a post-expectation Gaussian perturbation:

\[
\widetilde E_k(\theta;p_{1q},p_{2q})
=E_k^{\rm gate}(\theta;p_{1q},p_{2q})+\xi,
\qquad
\xi\sim\mathcal N(0,\sigma_E^2),
\qquad
\sigma_E=\frac{\sigma_0}{\sqrt{N_{\rm eff}}}.
\]

Claim boundary: `N_eff` is an effective scalar value-noise scale, not a physical shot count, unless a separate estimator-variance calibration is added.

## Workflow: source-of-truth MD

When the noise model, run ladder, or evidence anchor changes:

1. Update a dated source-of-truth MD in `MATH/paper_facing/paper_I_static_scaffold/`.
2. Include machine-readable YAML with:
   - noise surfaces;
   - code hooks;
   - run artifact paths;
   - SHA256 hashes for cited JSONs;
   - current parameter values;
   - claim boundaries.
3. Keep manuscript prose candidates separate from implementation facts.
4. Do not assert hardware-shot meaning for `N_eff` without calibration.

## Workflow: math-only primer PDF

For a reproducible math primer:

1. Start from the source-of-truth MD and current code hooks.
2. Use dense two-column LaTeX unless the user asks otherwise.
3. Introduce symbols by role before headline equations.
4. Derive in this order:
   - noiseless selected-energy objective;
   - synthetic gate-noisy expectation surface;
   - scalar value-noise layer;
   - error curve / visible noise band;
   - optional grouped-shot calibration relation.
5. Compile with `tectonic --keep-logs --reruns 2 <file>.tex` or `latexmk` if available.
6. Render pages to PNG with `pdftoppm` and visually inspect them.
7. Fix hard LaTeX errors and harmful overfull boxes. Existing visual clarity matters more than zero stylistic warnings, but prefer a clean compile when easy.
8. Copy final PDF to `output/pdf/` for sharing.

## Workflow: noise diagnostic plots

For appendix-ready plots:

1. Use completed JSON artifacts only.
2. Plot energy error versus ADAPT depth/iteration.
3. Use log y-axis and include the target tolerance line.
4. Keep one figure visually simple:
   - one curve per gate-noise level or one curve plus a scalar-noise band;
   - no crowded settings textbox;
   - no unnecessary noiseless overlay unless explicitly requested.
5. If showing scalar value-noise bands, use

\[
\varepsilon_{k,z}^{\rm lower}=\max\{0, |d_k|-z\sigma_E\},
\qquad
\varepsilon_{k,z}^{\rm upper}=|d_k|+z\sigma_E.
\]

6. Use figure captions to distinguish scalar value noise from gate noise.

## Workflow: GPT-Pro modeling handoff

When asking GPT-Pro about noise-model design:

1. Use the global `gpt-pro-handoff` skill.
2. Save the final export under `prompt-exports/`.
3. Make it standalone: no repo access assumed.
4. Include:
   - core equations;
   - current scalar/depolarizing model;
   - current evidence table;
   - exact modeling question;
   - requested output structure.
5. Reveal the final export in Finder per `prompt-exports/AGENTS.md`.

## Appendix edit discipline

- For manuscript `.tex` edits, use `$journal-math-manuscript-refiner` and rebuild the manuscript PDF afterward.
- Preserve machine-readable provenance comments.
- Keep reader-facing language compact and claim-bounded.
- Avoid saying the result is or is not paper-promotable; report objective evidence and let the user decide.
