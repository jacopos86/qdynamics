# Paper Construction Agent Brief

Created: 2026-05-09  
Purpose: implementation handoff for a separate agent that will build the paper-construction support system before any major Paper I / Paper II manuscript rewrite.

## 1. Objective

Create the repository support materials and skill updates needed to make Paper I and Paper II journal-ready with AI assistance, while preventing the usual AI failure modes:

- importing repo-native prose into manuscripts;
- overclaiming beyond benchmark evidence;
- treating every sentence as a literature claim;
- reverse-engineering mathematical formalism from code instead of using the math source of truth;
- ignoring journal-specific article structure, style, and evidence expectations;
- blurring the two-paper split between static ansatz construction and checkpoint-adaptive dynamics.

The immediate implementation target is not to rewrite the papers. The immediate target is to build the paper-construction infrastructure: reference maps, journal target/style dossiers, claim-source rules, repo-to-journal translation rules, and an update to the paper-writing skill so future agents draft in the right style.

## 2. Non-negotiable scope boundary

Do not edit the paper manuscripts in this pass unless the user explicitly asks.

Do not edit:

- `MATH/paper_details/static_adapt_paper_I.tex`
- `MATH/paper_details/time_dynamics_paper_II.tex`
- generated PDFs

Do edit or create support documents only.

The likely support paths are:

```text
MATH/paper_details/literature_addenda/
  static_adapt_paper_I_literature_addendum.md
  time_dynamics_paper_II_literature_addendum.md
  paper_I_reference_map.md
  paper_II_reference_map.md

MATH/notes/journal_targets/
  journal_target_matrix.md
  prx_quantum_style.md
  quantum_journal_style.md
  physical_review_a_style.md
  physical_review_research_style.md
  npj_quantum_information_style.md
  physical_review_b_style.md

MATH/notes/writing_guides/
  ai_manuscript_style_guardrails.md
  repo_to_journal_translation.md
  claim_source_types.md
  paper_I_claim_boundaries.md
  paper_II_claim_boundaries.md
```

If a paper-writing skill is updated, inspect the relevant skill first. The current manuscript-related skill is expected to be the journal/math manuscript skill, likely named `journal-math-manuscript-refiner`. If direct modification of a global user skill is inappropriate for the implementation context, create a repo-local instruction file that the skill can consume or that future agents can reference explicitly.

## 3. Source-of-truth hierarchy for paper construction

The mathematical formalism comes from:

```text
MATH/Math.md
```

The math should not be reconstructed from code. The repo implementation may be used to verify whether a claim is implemented or whether benchmark artifacts exist, but equations, notation, phase definitions, pruning laws, and controller structure should be taken from `MATH/Math.md` and the active generated math documents.

Use this hierarchy:

1. `MATH/Math.md` for mathematical definitions, algorithms, notation, and source-of-truth formalism.
2. Paper-specific addenda for literature positioning and claim boundaries.
3. Active paper drafts for current prose structure and missing-citation locations.
4. Benchmark outputs and result artifacts only for our-data-backed performance claims.
5. Code only to verify implementation status, not to invent manuscript formalism.

This point is important: the paper-writing workflow should not map the repo to math. It should map `MATH/Math.md` to journal-ready exposition, then cite prior work and our benchmark evidence appropriately.

## 3A. Reader-facing independence and terminology rules

Internal planning documents may use local labels such as Paper I, Paper II, and Paper III. Manuscript-facing prose should not. A reader should not need to know the local paper sequence to understand a sentence. Replace local ordinal references with the scientific object:

- "Paper I" -> "the ground-state ADAPT ansatz-construction method" or "the optimized ground-state variational ansatz";
- "Paper II" -> "the checkpoint-adaptive McLachlan dynamics route";
- "Paper III" -> "the geometry-aware QSE and excited-state dynamics method";
- "companion paper" -> a citation or a one-clause description of the method, depending on publication status.

Similarly, "scaffold" is repo shorthand and should not be the default journal term. Use "ansatz," "variational ansatz," "operator sequence," "selected operator basis," "selected nonorthogonal basis," or "variational manifold" unless a manuscript deliberately defines "scaffold" as a technical term.

Before any manuscript-facing pass, scan the edited text for:

```text
Paper I
Paper II
Paper One
first paper
second paper
companion paper
scaffold
```

and replace or locally define the term. The default is replacement, not definition.

## 4. Current two-paper split

Paper I and Paper II should be cleanly separated.

### Paper I

Target draft:

```text
MATH/paper_details/static_adapt_paper_I.tex
```

Addendum:

```text
MATH/paper_details/literature_addenda/static_adapt_paper_I_literature_addendum.md
```

Core identity:

Paper I is the static ansatz-construction paper. It should be framed as geometry- and cost-aware ADAPT ansatz construction for mixed fermion--boson systems.

The central object is not a new operator pool. It is a candidate-position acquisition policy:

- candidate-position records rather than bare generator labels;
- Fubini--Study / state-space geometry;
- lower-confidence trust-region signal;
- tangent-span novelty;
- reduced-window Schur reranking;
- compiled-resource and measurement-cost pressure;
- batching and beam continuation;
- rollback-safe static pruning.

Paper I may mention that compact ansätze can seed dynamics, but the online McLachlan stay/append/prune controller belongs to Paper II.

Recommended title direction:

```text
Geometry- and Cost-Aware ADAPT Ansatz Construction for Mixed Fermion--Boson Systems
```

### Paper II

Target draft:

```text
MATH/paper_details/time_dynamics_paper_II.tex
```

Addendum:

```text
MATH/paper_details/literature_addenda/time_dynamics_paper_II_literature_addendum.md
```

Core identity:

Paper II is the real-time variational-dynamics paper. It should be framed as checkpoint-adaptive McLachlan dynamics with bidirectional manifold maintenance:

- compact offline ansatz only as initializer;
- checkpoint-local stay / append / prune / repair lane map;
- zero-initialized tangent-block append;
- Schur-complement tangent-gain append confirmation;
- rollback-safe prune with projection, shadow verification, persistence, cooldown, and exact rollback;
- stabilized projective McLachlan solves;
- measurement-compatible decision data;
- trajectory and resource Pareto benchmarking.

Paper II should not spend method space rederiving the full three-phase static selector. In manuscript-facing prose it should either cite the static ansatz-construction method or describe it as the optimized ground-state ADAPT ansatz; it should not rely on the phrase "companion paper."

Recommended title direction:

```text
Bidirectional Checkpoint-Adaptive McLachlan Dynamics for Mixed Fermion--Boson Systems
```

## 5. Conversation decisions to preserve

This brief records the relevant decisions from the discussion so the implementation agent does not restart the argument.

1. We want AI assistance for writing the papers, but AI tends to introduce repo prose and non-journal-standard terms.
2. The solution is to build agent-facing support documents before rewriting the papers.
3. These support documents should include paper-dependent scholarly reference maps.
4. They should also include journal target dossiers and style dossiers based on official journal guidance and representative articles.
5. The journal dossiers are high priority because the papers must follow the structure and evidence expectations of the target venue.
6. Not every AI-generated claim should be traceable to prior literature. Some claims are backed by our own data and benchmarks; some are method definitions from `MATH/Math.md`.
7. Claim provenance should be classified into literature-backed claims, our-data-backed claims, and definition/design claims.
8. Literature-backed claims cite prior work.
9. Our-data-backed claims cite our tables, figures, benchmark protocols, and ablations.
10. Definition/design claims cite equations, algorithms, and definitions from `MATH/Math.md`.
11. The math formalism should come from `MATH/Math.md`, not from reverse-engineering repo code.
12. The implementation pass should create multiple support files, not one monolithic note.
13. The implementation pass should update the paper-writing skill or provide repo-local skill instructions if directly editing the global skill is not appropriate.
14. The user expects to use RepoPrompt MCP for the eventual large-context paper editing pass, because it is better suited to selecting large context from `MATH/Math.md`, addenda, and drafts.
15. The current task is to prepare that future pass.

## 6. Claim-source taxonomy

Create:

```text
MATH/notes/writing_guides/claim_source_types.md
```

It should define at least these buckets.

### Literature-backed claims

Examples:

- "ADAPT-VQE grows an ansatz by gradient-selected operators."
- "AVQDS uses McLachlan-distance information to adaptively build dynamics ansätze."
- "Geo-ADAPT-VQE uses the quantum information metric / natural gradient for operator selection."

Required support:

- prior-work citation;
- boundary sentence explaining how our work differs.

### Our-data-backed claims

Examples:

- "The method reduces two-qubit depth at fixed error on a benchmark suite."
- "Pruning improves resource Pareto behavior in calm intervals."
- "Ablating Schur reranking worsens the final ansatz."

Required support:

- result table, figure, benchmark manifest, ablation, or locked sweep;
- no citation padding as if the result came from prior literature.

### Definition/design claims

Examples:

- "The Phase-II novelty is a window-span Fubini residual."
- "A dynamic prune is accepted only after live projection/shadow/persistence guards pass."
- "The controller separates structural expressivity miss from numerical under-realization."

Required support:

- `MATH/Math.md` equation, algorithm, or section reference;
- possibly a manuscript method equation;
- not necessarily external literature.

### Prohibited behavior

The writing guide should explicitly forbid:

- citing every sentence merely to look scholarly;
- converting our benchmark claims into literature claims;
- converting design intentions into demonstrated results;
- using repo artifact names as evidence in manuscript prose;
- letting exact benchmark/reference data appear as controller-decision input in QPU-faithful method descriptions.

## 7. Repo-to-journal translation guide

Create:

```text
MATH/notes/writing_guides/repo_to_journal_translation.md
```

This guide should prevent repo-native prose from appearing in the papers.

Include a banned/replacement table.

Examples:

| Repo-native phrase | Journal replacement |
|---|---|
| route | benchmark protocol, controller variant, algorithmic configuration |
| manifest | parameter record, benchmark metadata, reproducibility record |
| phase3_v1 | final reduced-window geometric selector |
| full_meta | problem-local operator pool |
| run | benchmark instance, numerical experiment |
| artifact | result file, benchmark output, supplemental data |
| decision row | controller decision record |
| exact_v1 | exact-assisted diagnostic controller, if that is what it is |
| checkpoint_controller_reference_mode | controller exact-input mode / measurement-compatible decision data |

Also include positive style rules:

- Use "controller", "selector", "candidate-position record", "variational tangent manifold", "compiled-resource burden", "measurement-compatible estimates", "trajectory Pareto", and "rollback-safe ablation" where appropriate.
- Do not mention file paths in journal prose unless discussing reproducibility supplements.
- Translate implementation facts into equations, algorithms, benchmark protocols, figures, or tables.

## 8. AI manuscript style guardrails

Create:

```text
MATH/notes/writing_guides/ai_manuscript_style_guardrails.md
```

Required contents:

1. Do not write generic background paragraphs.
2. Do not overuse "novel", "framework", "pipeline", "robust", "seamless", "leverage", or vague "we propose" language.
3. Every paragraph should have a technical job: problem, prior limitation, method object, decision rule, benchmark evidence, limitation, or conclusion.
4. Keep Paper I and Paper II asymmetric:
   - Paper I: static ansatz acquisition.
   - Paper II: checkpoint-adaptive live dynamics.
5. Use exact comparator names where needed.
6. Avoid "better ADAPT" and "adaptive dynamics is new."
7. Use the stronger boundaries:
   - Paper I: candidate-position acquisition under geometry and cost.
   - Paper II: bidirectional checkpoint-local manifold maintenance.
8. In equations and method prose, use `MATH/Math.md` as the authority.
9. In result prose, cite our figures/tables/benchmark protocols.
10. In related-work prose, cite external literature and distinguish overlap precisely.

## 9. Paper-specific reference maps

Create:

```text
MATH/paper_details/literature_addenda/paper_I_reference_map.md
MATH/paper_details/literature_addenda/paper_II_reference_map.md
```

Each entry should use this schema:

```md
### CitationKey

Full reference:

Primary link:

Use in:

Supports:

Overlap risk:

Boundary sentence:

Do not claim:

Manuscript placement:

Notes for agent:
```

### Required Paper I entries

At minimum include:

- Hubbard 1963
- Holstein 1959
- Weiße--Fehske ED
- Troyer--Wiese sign problem
- Schollwöck DMRG/MPS
- Peruzzo VQE
- McClean hybrid variational algorithms
- Kandala hardware-efficient VQE
- McClean barren plateaus
- Grimsley ADAPT-VQE
- Tang qubit-ADAPT
- Yordanov QEB-ADAPT
- Feniou Overlap-ADAPT
- Anastasiou TETRIS-ADAPT
- Ramôa CEO-ADAPT
- Ramôa Hessian recycling
- Stadelmann gradient troughs, if kept
- Provost--Vallée Fubini--Study geometry
- Stokes quantum natural gradient
- Sohail--Koike-Akino Geo-ADAPT-VQE, arXiv:2603.10325
- Verteletskyi/Yen/Izmaylov measurement grouping
- Macridin electron-phonon simulation
- Sawaya d-level encodings
- Li variational basis state encoder
- Denner hybrid electron-phonon method

### Required Paper II entries

At minimum include:

- McLachlan 1964
- Dirac/Frenkel only if explicitly invoked
- Li--Benjamin variational quantum simulator
- Yuan variational quantum simulation
- Endo variational general processes
- Benedetti hardware-efficient time evolution
- Miessen--Ollitrault--Tavernelli spin-boson McLachlan dynamics, arXiv:2108.04258
- Barison pVQD
- Yao AVQDS
- Linteau adaptive pVQD
- Zhang compressed AVQDS
- Suzuki product formulas
- Childs--Su product-formula simulation
- Childs--Ostrander--Su randomized product formulas
- Campbell qDRIFT
- Macridin electron-phonon simulation
- Sawaya d-level encodings
- Li variational basis state encoder
- Denner hybrid electron-phonon method

## 10. Journal target dossiers

Create:

```text
MATH/notes/journal_targets/journal_target_matrix.md
MATH/notes/journal_targets/prx_quantum_style.md
MATH/notes/journal_targets/quantum_journal_style.md
MATH/notes/journal_targets/physical_review_a_style.md
MATH/notes/journal_targets/physical_review_research_style.md
MATH/notes/journal_targets/npj_quantum_information_style.md
MATH/notes/journal_targets/physical_review_b_style.md
```

The implementation agent must browse current official journal pages for this. Use official journal pages first. If representative article style is assessed, use recent published articles from that journal as examples, but do not copy prose.

Each style file should include:

- official scope;
- article types;
- length or formatting expectations if specified;
- abstract style;
- introduction style;
- related-work density;
- methods/results balance;
- equation tolerance;
- figure/table conventions;
- data/code availability expectations;
- cover-letter guidance;
- why Paper I fits or does not fit;
- why Paper II fits or does not fit;
- what evidence threshold would make the journal realistic;
- what claims would be too strong for that journal;
- links and access date.

The journal target matrix should include a compact ranking for each paper:

| Journal | Paper I fit | Paper II fit | Evidence needed | Risk |
|---|---|---|---|---|

Initial target set:

- PRX Quantum
- Quantum
- Physical Review A
- Physical Review Research
- npj Quantum Information
- Physical Review B as conditional, especially if Paper II becomes model-Hamiltonian-heavy

Optional/conditional:

- QCE for engineering-facing conference exposure;
- QIP only if the work is reframed as theory-heavy quantum information;
- NeurIPS only if a real learned policy or ML/search contribution is introduced.

Do not rely on stale venue deadlines. Search online for current guidance and record the access date.

## 11. Journal style extraction

The goal is not to copy sentences. The goal is to extract journal-standard structure.

For each target journal, identify patterns such as:

- how the abstract opens;
- how quickly the introduction names prior work;
- whether the paper foregrounds method or result first;
- how comparator tables are used;
- how detailed method sections are;
- how limitations are phrased;
- how broad the conclusion is;
- whether the paper uses assertive or conservative claims.

Create reusable guidance like:

```md
For PRX Quantum-style drafting:
- Lead with field-level bottleneck, not implementation detail.
- State the method object by paragraph 2.
- Use a broad-interest result hook before deep notation.
- Keep evidence claims tied to figures.
```

The exact guidance must be based on current official pages and representative articles.

## 12. Paper-specific claim boundary files

Create:

```text
MATH/notes/writing_guides/paper_I_claim_boundaries.md
MATH/notes/writing_guides/paper_II_claim_boundaries.md
```

### Paper I boundaries

Safe:

- candidate-position records extend ADAPT selection beyond bare generator labels;
- geometry and cost enter acquisition before the circuit is purchased;
- mixed fermion--boson systems make encoding/cutoff/layout part of the resource surface.

Needs data:

- any percentage improvement;
- any superiority over all ADAPT variants;
- any shot-cost reduction;
- any hardware-readiness claim.

Avoid:

- "first geometry-aware ADAPT method" because Geo-ADAPT-VQE exists;
- "better ADAPT" without specifying which ADAPT baseline and metric;
- "solves electron-phonon simulation on NISQ devices."

### Paper II boundaries

Safe:

- McLachlan propagation as regularized least-squares projection;
- checkpoint-local controller;
- append/prune/stay/repair lanes;
- exact reference trajectories are diagnostic outputs, not controller inputs in QPU-faithful routes.

Needs data:

- improvement over pVQD, AVQDS, adaptive pVQD, product formulas, or qDRIFT;
- broad utility of pruning;
- material benefit from Euler/RK4 switching;
- hardware-readiness beyond compiled/noisy simulation.

Avoid:

- "adaptive variational dynamics is new";
- "product formulas are infeasible";
- "reduced gate count implies practical advantage" without noise/resource evidence, especially given spin-boson McLachlan benchmark literature.

## 13. Skill update requirements

The paper-writing skill should be updated so future agents automatically:

1. Check `MATH/Math.md` before writing mathematical formalism.
2. Check the paper-specific literature addendum.
3. Check the paper-specific reference map.
4. Check the target journal dossier if a target journal is named.
5. Use claim-source taxonomy before making claims.
6. Translate repo-native language using the repo-to-journal guide.
7. Keep Paper I and Paper II method scopes separate.
8. Refuse to inflate benchmark claims without locked evidence.
9. Prefer journal-style structure over generic AI prose.
10. Avoid manuscript edits when the user asks only for addenda/support docs.

If the existing skill is global and should not be modified directly, create a repo-local skill supplement such as:

```text
MATH/notes/writing_guides/journal_math_skill_supplement.md
```

and add instructions in the implementation summary telling future agents to load it with the manuscript skill.

## 14. RepoPrompt MCP guidance for the later large-context pass

The user plans to use RepoPrompt MCP for the later paper-editing pass. The implementation agent should prepare context selections around:

- `MATH/Math.md` sections relevant to Paper I:
  - static ADAPT selector;
  - Phase I/II/III scoring;
  - pruning;
  - batching/beam;
  - mixed fermion--boson benchmark definitions.
- `MATH/Math.md` sections relevant to Paper II:
  - projective McLachlan controller;
  - checkpoint append/prune/stay/repair;
  - measurement-compatible decision data;
  - dynamics benchmark definitions.
- paper drafts:
  - `MATH/paper_details/static_adapt_paper_I.tex`
  - `MATH/paper_details/time_dynamics_paper_II.tex`
- addenda and new support docs.

The RepoPrompt pass should not select broad generated artifacts, run outputs, or irrelevant implementation files unless needed for a specific benchmark-result claim.

## 15. Literature scanning instructions

The implementation agent should search for more relevant literature beyond the current addenda.

Use primary sources:

- journal pages;
- arXiv pages;
- DOI pages;
- official code repositories only when discussing implementation availability.

Do not rely on memory for post-2024 or arXiv-preprint status. Verify current title, authors, venue, DOI, and publication status.

Already identified must-cite additions:

- `arXiv:2108.04258`: Miessen, Ollitrault, Tavernelli, spin-boson McLachlan dynamics performance study. Required for Paper II.
- `arXiv:2603.10325`: Sohail and Koike-Akino, Geo-ADAPT-VQE. Required for Paper I.

Search directions:

Paper I:

- geometry-aware ADAPT;
- natural-gradient ADAPT;
- quantum information metric VQE/ADAPT;
- overlap/covariance/correlation guided ADAPT;
- measurement-cost-aware ADAPT;
- Hessian-reuse ADAPT;
- batching/disjoint-layer ADAPT;
- electron-phonon VQE or adaptive ansatz construction.

Paper II:

- adaptive variational quantum dynamics;
- AVQDS variants;
- adaptive pVQD;
- compressed or tiled adaptive dynamics;
- McLachlan dynamics under noise;
- spin-boson variational dynamics;
- variational dynamics for electron-phonon / mixed fermion-boson systems;
- online circuit pruning or compression during dynamics.

## 16. Deliverables for the implementation agent

The implementation pass is complete when these exist:

```text
MATH/paper_details/literature_addenda/paper_I_reference_map.md
MATH/paper_details/literature_addenda/paper_II_reference_map.md
MATH/notes/journal_targets/journal_target_matrix.md
MATH/notes/journal_targets/prx_quantum_style.md
MATH/notes/journal_targets/quantum_journal_style.md
MATH/notes/journal_targets/physical_review_a_style.md
MATH/notes/journal_targets/physical_review_research_style.md
MATH/notes/journal_targets/npj_quantum_information_style.md
MATH/notes/journal_targets/physical_review_b_style.md
MATH/notes/writing_guides/ai_manuscript_style_guardrails.md
MATH/notes/writing_guides/repo_to_journal_translation.md
MATH/notes/writing_guides/claim_source_types.md
MATH/notes/writing_guides/paper_I_claim_boundaries.md
MATH/notes/writing_guides/paper_II_claim_boundaries.md
```

And either:

- the paper-writing skill is updated, or
- a repo-local skill supplement exists and the summary explains how future agents should load it.

## 17. Acceptance criteria

The implementation is acceptable when:

- all support docs are present;
- journal target files cite official source URLs and access dates;
- reference maps distinguish what each article supports and what not to claim;
- the claim-source taxonomy explicitly separates literature-backed, our-data-backed, and design/formalism claims;
- `MATH/Math.md` is named as the mathematical source of truth;
- repo-to-journal translation rules prevent repo-native phrasing;
- Paper I and Paper II are kept independent and asymmetrically dependent;
- no paper `.tex` manuscript is edited during this support-doc pass unless explicitly requested;
- no generated PDFs are rebuilt during this support-doc pass unless explicitly requested.
