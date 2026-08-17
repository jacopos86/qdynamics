# Paper-Facing Manuscript Support Workspace

Created: 2026-05-09  
Purpose: single ordered home for AI-facing support material for the journal paper program.

## Scope

This folder contains support documents only. Do not edit manuscript `.tex` files or generated PDFs from this workspace unless the user explicitly asks for manuscript editing.

There is no cross-paper `Math.md` source of truth. For Paper I, use the active
local-checkout manuscript and reader-facing build:

```text
MATH/paper_details/Paper_I.tex
MATH/paper_details/Paper_I.pdf
```

For the other papers, use the target manuscript/source pair named in
`MATH/AGENTS.md`. `MATH/Math.md` is legacy background only and must not be
used to establish active-paper notation, claims, results, or route identity.

Active paper drafts live under:

```text
MATH/paper_details/
```

## Layout

```text
MATH/paper_facing/
  shared/                  # Cross-paper writing rules and skill supplement
  journal_targets/          # Venue dossiers and target matrix
  paper_I_static_scaffold/  # Paper I support package
  paper_II_dynamics/        # Paper II support package
  paper_III_spectra/        # Paper III spectra/excited-state support package
  paper_IV_molecular_vibronic_h2o/
                            # Paper IV molecular-vibronic water support package
  paper_V_high_u_gkba/      # Paper V pointer/support stub; active workspace is paper_5/
  ml_outer_loop_paper_I/    # Future ML outer-loop paper attached to Paper I
  ml_outer_loop_paper_II/   # Future ML outer-loop paper attached to Paper II
  ml_outer_loop_paper_III/  # Future ML outer-loop paper attached to Paper III
```

## Future ML outer-loop paper placeholders

Three intentionally minimal support folders reserve future, separate papers on
machine-learning prediction or surrogate control for reducing expensive outer
loops:

- `ml_outer_loop_paper_I/` — static ADAPT / Paper-I outer loops;
- `ml_outer_loop_paper_II/` — AP-McLachlan / Paper-II outer loops;
- `ml_outer_loop_paper_III/` — QSE and excited-dynamics / Paper-III outer loops.

These folders are visual planning reminders. They do not yet define active
manuscripts, scientific claims, run routes, evidence contracts, or paper
numbers.

## Global strategy memo

Before restructuring Paper I or Paper II, read:

```text
MATH/paper_facing/two_paper_strategy.md
```

It defines the non-overlap rule, submission sequencing, venue strategy, and the
dominant novelty claims for Paper I and Paper II. For Paper III, use
`paper_III_spectra/literature_addendum.md`,
`paper_III_spectra/submission_package.md`, and
`paper_III_spectra/journal_recommendations.md`. For Paper IV, use
`paper_IV_molecular_vibronic_h2o/` plus the active manuscript source. For Paper
V, use `paper_V_high_u_gkba/` and then the standalone `paper_5/` workspace.

## Active Paper-I completion tracker

For the current Paper-I computational finish queue, including the final
SR-SNAKE route lock, the six-regime evidence matrix, matched Geo-ADAPT and
append-only ADAPT iteration-50 reruns, and appendix-only pruning, batching, and
beam ablations, read:

```text
MATH/paper_facing/paper_I_static_scaffold/
  paper_i_completion_tracker_20260717.md
```

This is the **only active Paper-I tracker**. Dated evidence, audits, source
maps, and validation reports are provenance records, not competing queues.

## Recommended load order for future writing agents

1. `MATH/AGENTS.md`.
2. `MATH/paper_facing/shared/journal_math_skill_supplement.md`.
3. `MATH/paper_facing/shared/claim_source_types.md`.
4. `MATH/paper_facing/shared/repo_to_journal_translation.md`.
5. `MATH/paper_facing/shared/ai_manuscript_style_guardrails.md`.
6. Target paper folder:
   - `paper_I_static_scaffold/`; for Paper I, start with
     `paper_i_completion_tracker_20260717.md`
   - `paper_II_dynamics/`
   - `paper_III_spectra/`
   - `paper_IV_molecular_vibronic_h2o/`
   - `paper_V_high_u_gkba/`, then `paper_5/`
7. The target paper's active source/PDF pair when exact manuscript notation,
   claims, or reader-facing status is needed. For Paper I, this is
   `MATH/paper_details/Paper_I.tex` and `MATH/paper_details/Paper_I.pdf`.
8. Target journal dossier in `journal_targets/`, if a venue is named.

## Copy policy

Do not create many full manuscript copies per journal by default. Use lightweight `journal_variants/*.md` files to record title, abstract posture, intro emphasis, evidence thresholds, and cuts/additions. Create a real copied `.tex` submission fork only after a target journal is selected.

## Paper split

- Paper I: static adaptive ansatz / operator-support acquisition under geometry and cost.
- Paper II: real-time checkpoint-adaptive McLachlan dynamics with reversible append--prune maintenance.
- Paper III: spectra/excited-state response, geometry-selected QSE excitation manifolds, transition observables, root tracking, and frozen-vs-live excited-state propagation.
- Paper IV: molecular-vibronic water application, finite active-space H2O model construction, and static SNAKE benchmarking hooks.
- Paper V: high-`U` regularization / GKBA exploratory study and quantum-computable encoding path under `paper_5/`.

## Non-overlap rule

- Paper I owns static scaffold acquisition, ground-state preparation, static Pareto/resource claims, and encoding/layout robustness.
- Paper II owns checkpoint maintenance, trajectory observables, real-time error, and append/prune/stay/repair ablations.
- Paper III owns excitation spectra, transition strengths, spectral-window error, QSE overlap conditioning, root tracking, frozen-QSE escape, and excited-state response.
- Paper IV owns the water molecular-vibronic Hamiltonian construction, derivative/alignment/cutoff diagnostics, and application-level static benchmark framing.
- Paper V owns the high-`U` instability, regularized GKBA formulation, stability diagnostics, and quantum-encoding feasibility questions.
- Do not reuse the same benchmark tables with different captions.
