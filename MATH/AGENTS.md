# MATH AGENTS.md

This subtree is mandatory for ADAPT, HH, time-dynamics, reporting, or any task
that changes mathematical defaults or route identity. For generic coding tasks,
leave it out of scope unless explicitly requested.

## Math File Contract

- `MATH/Math.md` and its generated `MATH/paper_details/Math.tex`/PDF twin are legacy background notes. They are not an authority for active-paper notation, claims, route identity, results, or manuscript synchronization; do not route agents to them by default.
- For Paper I, the active authority in the local checkout is the co-located source/PDF pair `MATH/paper_details/Paper_I.tex` and `MATH/paper_details/Paper_I.pdf`. Use the active paper manuscript together with its support folder and locked evidence, not `Math.md`.
- On 2026-07-18 the user promoted the no-ordinary-novelty SR-SNAKE manuscript copy into the canonical Paper-I source lineage. The dated `Paper_I_no_ordinary_novelty_sr_snake_20260717.*` files and the pre-promotion backup are historical provenance only; future agents must use the active co-located `Paper_I.tex` / `Paper_I.pdf` pair. The exact historical promotion receipt is `MATH/paper_facing/paper_I_static_scaffold/paper_i_canonical_manuscript_promotion_20260718.json`.
- Active user-facing paper sources live under `MATH/paper_details/`; their current source/PDF pairs are listed below. Treat older paper drafts, generated Math twins, and markdown notebooks as recovery/reference material unless the user explicitly requests them.
- `MATH/archive/archaic_repo_math.md` is archival only. Never use it, `MATH/Math.md`, or generated Math artifacts as a default source of truth.

## Paper-Program Bridge

Use this file as the bridge from root `AGENTS.md` into the active journal-paper
program. For paper, math, run, evidence, table, or manuscript tasks, route
through the support workspace before mining manuscript `.tex` files. Read or edit
manuscripts only when the task actually requires manuscript work, exact table
blocks, paper comments, or PDF/LaTeX synchronization.

### Active papers and partitions

The active repo paper program is partitioned into Papers I-V. Papers I-III are
standalone method papers with dedicated run/results gates. Paper IV is a
molecular-vibronic water application manuscript that reuses those method gates
when it invokes SNAKE, AP-McLachlan, or QSE. Paper V is the high-`U`
regularization / GKBA exploratory workspace under `paper_5/`.

| Paper | Active source/PDF pair | Scope | Primary support folder |
|---|---|---|---|
| Paper I | `MATH/paper_details/Paper_I.tex` / `MATH/paper_details/Paper_I.pdf` | Geometry- and cost-aware ADAPT ansatz construction for mixed fermion--boson systems. | `MATH/paper_facing/paper_I_static_scaffold/` |
| Paper II | `MATH/paper_details/time_dynamics_paper_II.tex` / `.pdf` | AP-McLachlan dynamics: append-prune support-patch control over time points/time iterations. | `MATH/paper_facing/paper_II_dynamics/` |
| Paper III | `MATH/paper_details/excited_spectra_dynamics_paper_III.tex` / `.pdf` | Geometry-aware QSE and excited-state dynamics. | `MATH/paper_facing/paper_III_spectra/` |
| Paper IV | `MATH/paper_details/molecular_vibronic_h2o_paper_IV.tex` / `.pdf` | Molecular-vibronic water application using finite active-space, all-three-mode, linearly coupled H2O Hamiltonians. | `MATH/paper_facing/paper_IV_molecular_vibronic_h2o/` |
| Paper V | `MATH/paper_details/paper_V_high_u_gkba.tex` / `.pdf` | High-`U` regularization / GKBA exploratory line and quantum-computable encoding study. | `paper_5/` and `MATH/paper_facing/paper_V_high_u_gkba/` |

Internal paper shorthand for repo-agent routing:

- **Paper I** / **paper one** means the RA-ADAPT/static-construction repo lane and support
  workspace.
- **Paper II** / **paper two** means the time-dynamics/AP-McLachlan repo lane
  and support workspace.
- **Paper III** / **paper three** means the QSE/excited-dynamics repo lane and
  support workspace.
- **Paper IV** / **paper four** means the molecular-vibronic water application
  lane and support workspace.
- **Paper V** / **paper five** means the high-`U` regularization / GKBA
  exploratory workspace under `paper_5/`.

This shorthand is not reader-facing manuscript prose. For manuscript text, use
the method object directly: static ADAPT/SNAKE, AP-McLachlan dynamics,
geometry-aware QSE/excited-state dynamics, molecular-vibronic water, or the
explicit high-`U`/GKBA model under discussion.

For Paper I, the reader-facing source-of-truth PDF is
`MATH/paper_details/Paper_I.pdf`, built from the co-located active source
`MATH/paper_details/Paper_I.tex`. Do not create a repository-root mirror.
Treat `MATH/paper_details/static_adapt_paper_I.tex`,
`MATH/paper_details/static_adapt_paper_I_condensed.tex`, old
`static_adapt_paper_I.pdf` copies, Overleaf upload zips, and extracted upload
copies as recovery/reference artifacts unless the user explicitly asks to
restore or package from them.

### Paper-support load order

When paper scope is triggered, load the smallest relevant slice in this order:

1. `MATH/paper_facing/README.md` for the support-workspace contract and paper
   split.
2. Shared bridge docs:
   - `MATH/paper_facing/shared/journal_math_skill_supplement.md`;
   - `MATH/paper_facing/shared/repo_to_journal_translation.md`;
   - `MATH/paper_facing/shared/ai_manuscript_style_guardrails.md`.
3. The target paper support folder from the active-papers table above.
   For Paper V, read `agent_guidance/paper-v/AGENTS.md`; no `paper_5/AGENTS.md`
   or `paper_5/README.md` exists in this checkout.
4. A target journal dossier under `MATH/paper_facing/journal_targets/` only when
   a venue is named.
5. The target manuscript `.tex`, manuscript comments, generated PDF, or
   machine-readable manifests only when the task specifically needs those files.

### Required skill gates

- Skill gates follow workflow scope, not paper membership. Algorithm
  implementation, route mathematics, unit tests, and agent-facing handoffs do
  not trigger run, results, CHTC, manuscript, pedagogical, plotting, noise, or
  GPT skills unless the current request actually performs that workflow.
- Confirm a repo-local `SKILL.md` exists before treating it as a gate. Do not
  invent or silently substitute a skill named by a stale router entry.
- Run, benchmark, CHTC, Optuna/settings, manifest, artifact aggregation,
  evidence-PDF, report, canonical-settings, or promotion work must use the
  existing relevant paper-specific run skill before commands, launches,
  aggregation, reporting, or promotion: `paper-i-run` or `paper-ii-run`.
  No Paper-III run skill exists in this checkout; fail closed only when an
  actual Paper-III paper-facing run workflow requires the missing contract.
- Table-cell/source-map/current-status/table-provenance updates from completed
  evidence must use a matching existing results skill after the run skill.
  Currently only `paper-ii-results` exists. Paper-I/III transfers must follow
  the explicit target support/evidence contract and fail closed when that
  contract is ambiguous; do not cite missing results skills. Results workflows
  consume locked evidence and must not launch runs.
- Paper-facing runs are visible-target gated. Execute or repair runs for visible
  manuscript/PDF table rows, figure panels, or explicit reader-facing claims;
  treat non-rendering `.tex` comments, `.txt` notes, and handoff notes as
  provenance/context only unless the user explicitly asks for diagnostic or
  exploratory work.
- Paper-facing reruns, repairs, noise variants, comparator extensions, and CHTC
  submissions must start from the current best visible table/figure result for
  the same method and regime. Resolve that visible result to its source
  JSON/manifest with
  `python3 agent_guidance/skills/shared/scripts/resolve_visible_settings.py ...`,
  reuse its algorithmic settings, change only the requested variable, and fail
  closed if provenance is missing.
- For Paper-I replay reruns that affect Hubbard--Holstein Qiskit cost tables,
  cost-vs-iteration plots, or Table-III resource provenance, agents must start
  from the existing Paper-I Qiskit cost-table provenance and its linked source
  map/support JSON. Reuse the recorded route/profile, pool, cutoff pair,
  optimizer/SPSA settings, seeds, backend/cost settings, source
  scaffold/current pointers, and replay/depth contract. The only intended
  algorithmic changes for the current repair line are the newly implemented
  prune and batching controls explicitly named by the user, plus output
  paths/run labels; fail closed on any additional settings drift.
- Paper promotion and demotion decisions are user-only. Agents may report source
  facts, validation outcomes, missing evidence, failed gates, and risk notes, but
  must not tell the user that a run/artifact/table value/report is
  paper-promotable or not paper-promotable. Ask the user what to promote, defer,
  rerun, or edit.
- Paper IV and Paper V do not yet have dedicated run/results skills. When Paper
  IV or V work invokes Paper I SNAKE/static ADAPT, Paper II AP-McLachlan
  dynamics, or Paper III QSE machinery, follow the corresponding method skill
  gate. If a Paper IV/V request has no matching gate, report the missing
  contract instead of inventing one.
- Reviewing, suggesting edits, drafting replacement prose, or editing any active
  paper manuscript or related titles, abstracts, captions, literature addenda,
  figures, table prose/cells, `.tex`, or PDF-facing assets must use
  `$journal-math-manuscript-refiner` and then
  `MATH/paper_facing/shared/journal_math_skill_supplement.md` before proposing or
  applying changes. Chat-only proposed wording, paper-edit plans, literature-positioning
  suggestions, and review-only responses are manuscript-facing work; the skill is
  required before those proposals are given, not only before file edits.
- `$pedagogical-math-primer` is not an editor for the active Paper I, Paper II,
  or Paper III journal manuscripts. Use it for separate teaching notes,
  derivation companions, equation explanations, and explicitly pedagogical
  artifacts. If pedagogical derivation work informs a manuscript correction,
  translate the result back into journal prose under
  `$journal-math-manuscript-refiner` and the shared manuscript contracts.
- Paper-I GPT-Pro/Atlas review loops use `$journal-math-manuscript-refiner` for
  manuscript review and `$gpt-pro-handoff` only for requested exports. No
  repo-local `paper-i-gpt-pro-review` skill exists in this checkout. GPT Pro is
  advisory only; repo agents must triage its output into candidate changes,
  ask the user for approval, and then follow the manuscript, table, citation,
  or run gates that each approved change actually triggers.
- Paper-I SNAKE noise-model appendix work, scalar/gate-noise model math,
  noise diagnostic plots, dense math-only primer PDFs, and noise-model GPT
  handoffs must use `agent_guidance/skills/paper-i-noise-model-primer/SKILL.md`.
  This skill routes agents through only the applicable manuscript, pedagogy,
  PDF, or GPT-handoff workflow while preserving the noise-model
  source-of-truth artifacts.
- No repo-local `paper-i-hh-cost-convergence-plot` skill exists in this
  checkout. Paper-I Hubbard--Holstein Table-III cost/iteration overlays follow
  the explicit target plotting/report contract and `paper-i-run` only when a
  run or evidence report is actually requested. Questions about exact versus
  reconstructed per-prefix hardware cost are evidence/provenance questions and
  do not trigger a plotting skill unless a plot is requested.
- Reading a required skill to follow it is mandatory; editing any file under
  `agent_guidance/skills/` is forbidden unless the user explicitly asks in the current
  request to change a skill.

The supplement is the active repo-local override for draft-mode placeholder
handling, SNAKE naming, paper split, and author preferences.

## Manuscript Agent Contract

- Ask questions and make an explicit plan before implementing manuscript,
  table, run-skill, or paper-facing evidence changes when the requested behavior
  affects claims, data provenance, table values, route identity, or artifact
  semantics. Prefer a short plan and one clarifying question over guessing when
  intent is ambiguous.
- When explaining issues to the user, use ubiquitous mathematical and physical
  language first: Hamiltonians, ansatz manifolds, gradients, optimizer
  stationarity, cutoff error, reference energy, target tolerance, and compiled
  resource metrics.  Avoid repo-provenance shorthand, artifact names, route
  labels, cluster labels, or source-map jargon unless the user explicitly asks
  for implementation provenance.  Define claims through equations, variables,
  and manuscript-facing concepts before mentioning local files.
- Do not edit, rewrite, or update any skill file under `agent_guidance/skills/` unless the
  user explicitly asks in the current request to change a skill. Reading a skill
  to follow it is required; changing a skill is not authorized by ordinary
  manuscript, table, run, or report work.
- Paper-facing table data is preservation-first. Agents must not delete, blank,
  demote, or replace existing completed/numeric table cells, source-backed
  non-hit cells, or completed-status cells with `running`, `NR`, `n/a`,
  placeholders, blanks, or any lower-evidence status unless the user explicitly
  requests that specific destructive change in the current turn.
- New running, queued, failed, missing, or incomplete evidence is additive status
  metadata, not permission to erase prior displayed evidence. Preserve the old
  displayed value, report the new status separately, and ask before destructive
  replacement.

After every edit to a paper manuscript `.tex` source in `MATH/paper_details/`,
regenerate the corresponding PDF before reporting the edit complete. Run the
build from `MATH/paper_details/` with `latexmk` when available, or with
`tectonic --keep-logs --reruns 2 <paper>.tex` as the approved fallback when
`latexmk` is absent. A missing `latexmk` binary is not a blocker if `tectonic`
passes. If the selected builder fails, report the exact failure and do not claim
the TeX/PDF pair is synchronized.

Supporting documentation lives primarily in the paper-support bridge above:
`MATH/paper_facing/`, `MATH/paper_details/literature_addenda/`, `MATH/notes/`,
and any journal-target or writing-guide files created for paper construction.
Use these documents to track literature boundaries, journal style, claim
provenance, and required evidence before inspecting manuscripts. Reader-facing
manuscript prose should remain standalone: avoid local labels such as "Paper I"
or "companion paper" unless they are internal planning notes.

## Future Repo-Local Skill Candidates — Inactive

The following are planning placeholders only. They are not mandatory gates, are
not discoverable skills, and must not be treated as active instructions unless a
future explicit user request creates or updates a real skill file:

- paper-navigation helper for choosing the correct paper-support folder;
- claim-audit helper for checking literature-backed, our-data-backed, and
  definition/design claims against the shared claim-source contract;
- table-reference helper for maintaining table-reference notes after a
  paper-specific results skill identifies a stable need.

Do not create or edit skills as part of ordinary paper, run, table, or report
work.

## Target Math Directory Organization

Keep the root focused on source-of-truth math, build helpers, and classified support folders:

- Legacy-only background artifacts: `MATH/Math.md` and `MATH/paper_details/Math.tex`/PDF. Do not route active-paper work to them.
- `MATH/paper_details/adaptive_selection_staged_continuation.tex` and its PDF.
- `MATH/paper_details/main_condensed.tex` and its PDF.
- Active paper manuscripts, PDFs, and bib files:
  - `MATH/paper_details/Paper_I.tex` / `Paper_I.pdf` / `Paper_INotes.bib`;
  - `MATH/paper_details/time_dynamics_paper_II.tex` / `.pdf` / `time_dynamics_paper_IINotes.bib`;
  - `MATH/paper_details/excited_spectra_dynamics_paper_III.tex` / `.pdf` / `excited_spectra_dynamics_paper_IIINotes.bib`;
  - `MATH/paper_details/molecular_vibronic_h2o_paper_IV.tex` / `.pdf` when present;
  - `MATH/paper_details/paper_V_high_u_gkba.tex` / `.pdf` when present.
- `MATH/17A_symbol_guide.tex` and its PDF when present.
- `MATH/AGENTS.md`.

Move or classify everything else by role during cleanup:

- Build helpers: `MATH/build/`.
- Scratch/generated LaTeX files: `MATH/generated/`, not source files.
- LaTeX build byproducts (`.aux`, `.bbl`, `.blg`, `.fls`, `.fdb_latexmk`, `.log`, `.out`, `.toc`) belong in generated/archive locations, but active paper PDFs stay beside their `.tex` sources in `MATH/paper_details/`.
- Old backups, inactive paper drafts, and archaic repo math: `MATH/archive/`.
- Still-useful planning notes and run notes: `MATH/notes/`.

## Route Identity

- The ordinary Paper-I adaptive method is RA-ADAPT. Its only ordinary
  execution interface is `run_ra_adapt(problem, request=None)`, owned by
  `agent_guidance/static-adapt/AGENTS.md` and its canonical run guide. Typed
  requests and resolved protocols are the executable authority; this file does
  not duplicate their settings, and a folder or materialized bundle is not
  execution authorization.
- Paper-I provenance contains three historically named SNAKE families.
  JR-SNAKE, FM-SNAKE, and SR-SNAKE remain exact compatibility/research
  identities for preserved artifacts. Do not use `Route 4`, `historical route`,
  or the legacy `route_a` umbrella as a method name:

  | Display name | Stable family id | Defining controller structure |
  |---|---|---|
  | **JR-SNAKE** | `joint_response_snake` | Macro-to-Pauli-child funnel with a full active-plus-batch joint-response model and batch proposal/admission semantics. |
  | **FM-SNAKE** | `formal_manifold_snake` | Query-closed formal-manifold phase models with branch-local propagated manifold/curvature state and formal-manifold reoptimization; it does not invoke the JR selector. |
  | **SR-SNAKE** | `singleton_response_snake` | Phase-I/II parent records in physical lanes; exact-cardinality-one Pauli-child forwarding into Phase III with symmetry/padding enforcement; lane-free Phase III with a full active-plus-one-candidate response before supported-rank reduction; Phase 0 and batching disabled by default. |

- The preserved Paper-I **SR-SNAKE** baseline is the source-locked historical
  route used by the active Paper-I Hubbard--Holstein provenance:
  `sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1`,
  resolved as
  `supported_projected_generalized_source_metric_no_overlap_trust_full_response_symmetric_cost_no_prune_v1`
  with contract digest
  `fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2`.
  It uses full active-plus-singleton Phase-III response geometry, a
  source-metric no-endpoint-overlap trust update, and a complete
  supported-FS-whitened accepted refit. Its preserved baseline has batching,
  pruning, beam, Phase 0, and phase-live hysteresis off.
- The provenance source for this preserved baseline is
  `MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723/paper_i_hh_macro_common_accuracy_20260723_provenance.json`
  and its hash-locked tracker. Current Markdown, unqualified aliases, registry
  defaults, and old planning notes do not override that provenance. Explicit
  user-approved changes must be recorded as new typed policy identities rather
  than silently relabeling preserved results.
- Current RA-ADAPT route contracts retain the preserved profile and digest
  through their serialized `lineage_authority`; do not rename or overwrite the
  historical identity.
- Historical whitened-selector, Phase-II-whitening, overlap-trust, hysteresis,
  beam, pruning, and versioned SR profiles remain explicit compatibility or
  ablation identities. They are not enumerated or considered by an unqualified
  canonical Paper-I request.
- Preserve cost/resource weighting, shortlist/funnel semantics, symmetry and
  padding guards, admission, pruning, batching, beam, insertion, stopping, and
  accepted-refit behavior as explicit resolved policy fields. Do not infer a
  family or canonical identity from one feature flag, final batch size,
  `static_route_id=route_a`, or an unqualified legacy alias.
- `route_a` and `paper_i_production_v1` remain compatibility/provenance labels
  for preserved manifests. They are not sufficient to distinguish JR-SNAKE,
  FM-SNAKE, and SR-SNAKE. Only for an explicit historical provenance request, use
  `agent_guidance/static-adapt/route-identities.md` for the detailed registry
  and executable-field map.
- HH realtime/time-dynamics planning defaults to Paper-II AP-McLachlan:
  append-prune McLachlan support-patch control over time points/time
  iterations.
- Do not use "checkpoint controller" as the Paper-II route identity. Existing
  checkpoint-named code, flags, and artifact fields are legacy compatibility
  surfaces until they are migrated behind tested aliases.
- Secant/tangent-secant terms are diagnostics/proposals only, not the default route identity.
- Archival `secant_lead*` results must not set route selection, run defaults, or code defaults.

## ADAPT And Pools

- New ADAPT and time-dynamics implementations default to `phase3_v1` logic as defined in `MATH/paper_details/adaptive_selection_staged_continuation.tex`.
- `full_meta` means the problem-local mega pool: all implemented operator families for the resolved problem, including reusable generic operators, Hamiltonian quadratures, Hamiltonian blocks, and family-specific HVA terms when available. Do not redefine this global pool name.
- For canonical Hubbard--Holstein Paper-I adaptive work, use unfiltered
  `full_meta` with HVA families included. Phase I and Phase II operate through
  the physical macro-family lanes; shortlisted macros are projected into
  exact-cardinality-one Pauli children for Phase III, and the fixed-sector and
  binary-padding symmetry guard is mandatory. `full_meta_minus_hva` is an
  explicitly requested pool ablation only and cannot define canonical defaults.
  Intact-macro admission is one explicit controlled comparison, not another
  ordinary canonical route.
- The ordinary Paper-I adaptive method is RA-ADAPT. Append-ADAPT has a separate
  explicitly named comparator interface; its locked artifacts remain frozen
  comparator/replay sources, while an explicitly named comparator study may
  invoke the method. Geo-ADAPT remains an explicitly named Paper-I-local
  benchmark. JR-SNAKE, FM-SNAKE, QEB, HEA,
  family-informed, TETRIS, and other historical paths are compatibility,
  research, or ablation surfaces that never enter canonical route resolution
  implicitly.
- For explicit historical comparison or provenance work, resolve every
  Paper-I HH SNAKE result as JR-SNAKE, FM-SNAKE, SR-SNAKE, or unresolved before
  comparing it. `route_a` / `paper_i_production_v1` is a legacy compatibility
  umbrella, not the method identity. For the preserved SR-SNAKE profile,
  preserve the provenance-locked macro-to-singleton pool
  funnel, mandatory hard symmetry guard, full Phase-III response, no-overlap
  trust, whitened full accepted refit, and cost/resource weighting. Record
  exact pruning, batching, beam, insertion, shortlist, stopping, and optimizer
  fields from the source lock or an explicit typed override. Only in that
  historical provenance workflow, use
  `agent_guidance/static-adapt/route-identities.md` for family resolution and
  `agent_guidance/static-adapt/route-a-language.md` only for legacy field
  translation. Record observed fields explicitly instead of translating them
  into one stale Route-A flag.
- For Paper-I phonon-bearing ADAPT work, the variational and exact-reference
  energies must use the same phonon cutoff. Report the single working cutoff
  `n_ph_work` and compute `abs(E_alg(n_ph_work) - E_exact(n_ph_work))`. Do not
  require, infer, or report a distinct higher `n_ph_ref` unless the user
  explicitly requests a separate cutoff-sensitivity study; such a study is not
  part of the Paper-I FM/JR/SNAKE model-comparison accuracy coordinate.
