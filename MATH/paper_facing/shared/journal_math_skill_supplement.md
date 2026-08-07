# Repo-Local Supplement for `$journal-math-manuscript-refiner`

Created: 2026-05-09  
Purpose: local instructions for future agents using `$journal-math-manuscript-refiner` on Paper I / Paper II / Paper III.

## Why this exists

This supplement is repo-local author configuration. Future agents should load it alongside `$journal-math-manuscript-refiner` whenever working on Paper I, Paper II, or Paper III.

Recommended instruction to future agents:

> Use `$journal-math-manuscript-refiner`, then read `MATH/paper_facing/shared/journal_math_skill_supplement.md` before drafting or editing Paper I/Paper II/Paper III.

## Mandatory load order

0. `MATH/paper_facing/two_paper_strategy.md` for split, overlap, venue, and submission-sequencing rules.

1. `MATH/AGENTS.md`.
2. `MATH/Math.md` sections relevant to the target paper.
3. Target paper draft only if manuscript editing is explicitly requested.
4. Shared writing guides:
   - `MATH/paper_facing/shared/claim_source_types.md`.
   - `MATH/paper_facing/shared/repo_to_journal_translation.md`.
   - `MATH/paper_facing/shared/ai_manuscript_style_guardrails.md`.
5. Target paper folder:
   - Paper I: `MATH/paper_facing/paper_I_static_scaffold/`.
   - Paper II: `MATH/paper_facing/paper_II_dynamics/`.
   - Paper III: `MATH/paper_facing/paper_III_spectra/` when activated.
6. Within the target folder, read `literature_addendum.md`, `reference_map.md`, `claim_boundaries.md`, `papers_to_cite.md`, and `papers_to_mimic.md`.
7. Target journal dossier in `MATH/paper_facing/journal_targets/` if a journal is named.

## Automatic behavior expected from future agents

- Check `MATH/Math.md` before writing mathematical formalism.
- Check the literature addendum and reference map before related-work prose.
- Check the target journal dossier before changing article structure or style.
- Classify claims as literature-backed, our-data-backed, or definition/design.
- Translate repo-native language using the translation guide.
- Do not tell the author how to frame a paper unless the user explicitly asks
  for framing, venue positioning, abstract/introduction strategy, or
  final-polish review. For method-source/specification work, review equations,
  notation, default algorithmic choices, and implementation contracts.
- Do not turn a method-spec review into caveats about current benchmark
  maturity or publishability. Results may be intentionally deleted or out of
  scope while the method is being defined.
- Keep Paper I, Paper II, and Paper III method scopes separate.
- Enforce the non-overlap rule from `MATH/paper_facing/two_paper_strategy.md`: Paper I owns ADAPT ansatz construction; Paper II owns checkpoint trajectory maintenance; Paper III owns spectra, QSE excitation manifolds, transition observables, and driven excited-state response.
- Refuse to inflate benchmark claims without locked evidence.
- Prefer journal-style structure over generic AI prose.
- Avoid manuscript edits when the user asks only for addenda/support docs.
- After every edit to a Paper I/Paper II/Paper III manuscript `.tex` file, regenerate the corresponding PDF before reporting completion. Run the build from `MATH/paper_details/` with `latexmk` when available, or with `tectonic --keep-logs --reruns 2 <paper>.tex` as the approved fallback when `latexmk` is absent. A missing `latexmk` binary is not a blocker if `tectonic` passes. If the selected builder fails, report the exact error and treat the TeX/PDF pair as unsynchronized.

## Draft-mode versus final-polish mode

Default to **draft-mode editorial support** unless the user explicitly asks for a final submission polish, camera-ready cleanup, or placeholder-removal pass.

In draft mode:

- Preserve `\placeholder{...}`, `\tentative{...}`, red text, and intentionally provisional table cells. These are working insertion points for later benchmark fills, not defects to remove.
- Keep visible result placeholders close to the final table/caption/prose locations so locked numbers can be inserted with minimal structural churn.
- Do not recommend deleting red/provisional result blocks merely because they look unfinished. Instead, label which cells require locked artifacts and which prose claims must remain conditional.
- Preserve `\structlabel{...}` and other concise structural labels when they guide the author's drafting workflow. In Paper I draft mode, these labels may render as red placeholders when the author wants them visible; they are not prose headings and must not be blended into reader-facing sentences.
- Avoid replacing the author's draft with generic four-part abstract templates, stock introduction rewrites, or broad related-work insertions unless requested.
- Treat placeholder management as claim hygiene: keep the slot, constrain the claim, and record the evidence needed to fill it.

In final-polish mode:

- Convert or remove structural labels.
- Replace red placeholders with locked values, neutral prose, or explicit omissions.
- Remove remaining provisional color and draft-only comments before submission.

## Current Paper I author preferences

Until the user changes these preferences:

- Use **SNAKE** throughout Paper I after defining it once. Define it as **Selection--Novel ADAPT Kost Evaluator**. Do not advise dropping the acronym or replacing it with purely descriptive method language by default.
- Do not proactively retitle Paper I, rewrite the abstract into a generic template, or restructure the introduction/related-work section unless the user asks for that specific pass.
- Focus first on clear required fixes: Paper II/Paper III leakage, mathematical consistency, unsupported claim boundaries, typo/grammar cleanup, and journal tone at the sentence level.
- Preserve the working red placeholders and tentative result commands so final benchmark numbers can be inserted later.

## Terminology discipline before drafting

Future agents must preserve this complaint as an active writing constraint:

- **Controller** and **scaffold** are not canonical ADAPT terms.
- Canonical ADAPT vocabulary is **operator pool**, **ansatz**, **adaptive sequence**, **operator selection**, and **parameter optimization**.
- Paper I should not be made to sound like a closed-loop adaptive dynamical decision system. It is an ADAPT ansatz-construction paper whose strongest framing is **budgeted adaptive-ansatz acquisition by geometry- and cost-aware candidate-position selection**. Do not over-specify "static" in reader-facing prose merely to distinguish Paper I from dynamics work.
- In Paper I manuscript prose, prefer **adaptive ansatz**, **operator sequence**, **generator sequence**, **operator support**, and **variational manifold**. Do not use **scaffold** in reader-facing Paper I prose.
- Paper II may use **controller** only after defining a checkpoint diagnostics-to-action policy such as

\[
\pi:\mathcal D_k\mapsto u_k,
\qquad
u_k\in\{\mathrm{stay},\mathrm{append},\mathrm{prune},\mathrm{branch}\}.
\]

Then **controller** means the checkpoint decision policy, not generic ADAPT. Before that point, prefer **adaptive McLachlan evolution**, **dynamic ansatz refinement**, **append--prune update rule**, or **checkpoint-maintained variational manifold**.

## Paper split reminder

Paper I:

- ADAPT ansatz-construction paper; use canonical manuscript vocabulary and do not over-specify "static" in reader-facing prose.
- Recommended title: **Geometry- and Cost-Aware ADAPT Ansatz Construction for Mixed Fermion--Boson Systems**.
- Core phrase: **budgeted adaptive-ansatz acquisition by geometry- and cost-aware candidate-position selection**.
- Novelty sentence to adapt: "SNAKE selects candidate-position records under a joint geometric and hardware-cost objective, reranks them by reduced-window Schur relaxation, and removes stale generators by rollback-safe generator ablation."
- Mention dynamics only as downstream use of compact operator supports.

Paper II:

- Real-time checkpoint-adaptive McLachlan dynamics paper.
- Recommended title: **Bidirectional Checkpoint-Adaptive McLachlan Dynamics for Mixed Fermion--Boson Systems**.
- Core phrase: **bidirectional checkpoint-local manifold maintenance**.
- Novelty sentence to adapt: "AP-McLachlan maintains a checkpoint-local variational manifold by admitting zero-amplitude Schur-confirmed tangent blocks, pruning verified redundant generators, and allowing exchange patches without breaking trajectory continuity."
- Treat ADAPT ansatz construction as companion-paper initialization, not as Paper II method space.
- In Paper-II method-source or implementation-spec sections, repo-visible
  runtime labels are allowed after the mathematical object is defined. Examples:
  `parameterization_mode`, `per_pauli_term`, `logical_shared`, support atom,
  support patch, append ladder, prune ladder, exchange, solve repair, and
  drive-aligned ansatz augmentation. Do not flag these labels as defects when
  the section is intentionally binding the paper equations to implementation.

Paper III:

- Spectra and excited-state response paper.
- Recommended title: **Geometry-Aware Quantum Subspace Expansion and Excited-State McLachlan Dynamics for Mixed Fermion--Boson Hamiltonians** unless a shorter journal-specific title is chosen.
- Core phrase: **geometry-selected spectral-manifold acquisition**.
- Novelty sentence to adapt: "The method selects excitation records by probe-transition gain, overlap-metric novelty, residual-Schur spectral gain, conditioning, and measurement cost, then tests whether the selected nonorthogonal manifold remains closed under driven propagation."
- Treat Paper I as a seed-state source and Paper II as live-propagation lineage, not as Paper III method space.

## Refusal / downgrade rules

Downgrade or refuse a sentence if it:

- claims first-of-kind geometry-aware ADAPT without handling Geo-ADAPT-VQE;
- claims adaptive dynamics is new without handling AVQDS/adaptive pVQD;
- turns exact diagnostic references into controller-decision inputs;
- says hardware-ready without compiled/noisy/finite-shot evidence;
- cites external literature for our own benchmark outcome;
- uses route names or artifacts as manuscript prose;
- overuses **controller** or **scaffold** where canonical ADAPT terms would be clearer;
- claims QSE, qEOM, VQD, SSVQE, or adaptive dynamics are new in Paper III without the required boundary citations;
- reports Paper III spectral, transition-strength, conditioning, or matrix-measurement gains without locked tables/figures.
