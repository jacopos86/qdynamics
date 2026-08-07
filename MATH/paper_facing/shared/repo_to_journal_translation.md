# Repo-to-Journal Translation Guide

Created: 2026-05-09  
Scope: Paper I / Paper II journal prose and addenda.  
Authority: translate from `MATH/Math.md` plus benchmark evidence, not from code names.

## Rule

Implementation language may help agents find evidence, but manuscript prose must speak in equations, algorithms, benchmark protocols, figures, tables, and reproducibility records. Do not paste repo-native names into a journal paragraph unless the paragraph is explicitly about supplemental reproducibility.

Exception for method-source/specification work: when a Paper-II section is being
used as the implementation source of truth, repo-visible option labels may
appear after the mathematical object is defined. Examples include
`parameterization_mode`, `per_pauli_term`, `logical_shared`, support atom,
support patch, append ladder, prune ladder, exchange, solve repair, and
drive-aligned ansatz augmentation. These are allowed because they bind equations
to runtime behavior. Do not flag them as prose defects merely because they are
also code labels.

For Paper-I SNAKE feature/provenance language, first use `MATH/paper_facing/paper_I_static_scaffold/snake_ubiquitous_language.md` to map repo fields such as `phase3_novelty_ablation_mode`, `phase1_prune_enabled`, `phase2_frontier_ratio`, and `phase3_selector_geometry_mode` to the corresponding Unicode math symbols and support prose. Then translate the result again for reader-facing manuscript text.

For Paper-II AP-McLachlan/time-dynamics language, first use
`agent_guidance/time-dynamics/ubiquitous-language.md` to map repo phrases such as
ANZATS seed handoff, `psi_ref`, `psi_initial`, executor, drive profile versus
drive operator, drive-aligned ansatz augmentation, support patch, fixed
McLachlan comparator, and exact-reference data flow to the corresponding
mathematical objects and support prose. Then translate the result again for
reader-facing manuscript text.

## Banned / replacement table

| Repo-native phrase | Journal replacement |
|---|---|
| route | benchmark protocol; controller variant; algorithmic configuration |
| manifest | parameter record; benchmark metadata; reproducibility record |
| phase3_v1 | final reduced-window geometric selector |
| full_meta | problem-local operator pool |
| pareto_lean | compact resource-aware pool/configuration, if scientifically defined |
| run | benchmark instance; numerical experiment; trajectory experiment |
| artifact | result file; benchmark output; supplemental data |
| decision row | controller decision record |
| exact_v1 | exact-assisted diagnostic controller, if that is what it is |
| checkpoint_controller_reference_mode | controller exact-input mode |
| checkpoint_controller_exact_input_mode=off | measurement-compatible decision data; no exact target input to decisions |
| FakeNighthawk / FakeMarrakesh | calibrated fake-backend compilation/noise model, if relevant |
| raw JSON | machine-readable result record; supplemental data table |
| pipeline | benchmark harness; analysis workflow; algorithm implementation, depending on context |
| hardcoded | fixed benchmark configuration; frozen protocol |
| replay | rerun of a frozen benchmark configuration; post hoc reproduction |
| scratch / smoke | development check; not manuscript evidence |
| decision telemetry | controller diagnostic record |
| reference_mode | diagnostic exact-reference mode, unless it truly controls decisions |
| controller exact input | exact target/reference data used as controller input; avoid in QPU-faithful routes |
| controller | checkpoint decision policy; use only after defining diagnostics-to-action map |
| static ADAPT controller | static candidate-position selector; adaptive ansatz-construction procedure |
| scaffold | operator support; generator sequence; adaptive ansatz; variational manifold |
| live scaffold | evolving operator support; checkpoint-maintained variational manifold |
| closed-loop adaptive dynamical decision system | adaptive McLachlan evolution with a checkpoint decision policy |

## Positive vocabulary

Use these terms where they are technically correct:

- adaptive ansatz;
- operator sequence;
- generator sequence;
- operator support;
- selector;
- candidate-position record;
- checkpoint decision policy;
- controller, only after a formal diagnostics-to-action policy is defined;
- variational tangent manifold;
- Fubini--Study displacement;
- tangent-span novelty;
- Schur-complement gain;
- compiled-resource burden;
- measurement-compatible estimates;
- trajectory Pareto surface;
- rollback-safe ablation;
- parameter record;
- benchmark protocol;
- reproducibility record.

## Terminology bifurcation

Use canonical ADAPT language externally unless the paper has explicitly crossed into sequential structural decisions.

External Paper I language should usually be:

- adaptive ansatz;
- operator pool;
- operator sequence;
- candidate-position record;
- operator selection;
- parameter optimization;
- variational manifold.

Avoid making Paper I sound like a closed-loop dynamical decision system. Standard ADAPT is greedy variational construction: select an operator, append it, and reoptimize. Paper I extends the selection object and scoring law, but it should still sound grounded in ADAPT vocabulary.

Paper II can use control language after it defines a checkpoint policy

\[
\pi:\mathcal D_k\mapsto u_k,
\qquad
u_k\in\{\mathrm{stay},\mathrm{append},\mathrm{prune},\mathrm{branch}\},
\]

or an equivalent update law for \((\mathcal O_k,\theta_k)\). In that context, **controller** is mathematically justified. Even there, keep **adaptive McLachlan evolution**, **dynamic ansatz refinement**, and **checkpoint-maintained variational manifold** available as less jargon-heavy alternatives.

## Translation examples

Repo-native:

> `phase3_v1` route selects from `full_meta` and writes a manifest.

Journal-native:

> The final selector reranks a reduced candidate window from the problem-local operator pool using local tangent geometry, Schur-complement gain, and compiled-resource pressure, and records the benchmark parameters for reproducibility.

Repo-native:

> `checkpoint_controller_exact_input_mode=off` with `benchmark_exact` overlays.

Journal-native:

> Controller decisions are made from measurement-compatible estimates of the prepared circuit state; exact diagonalization is reserved for post-run diagnostic overlays.

Repo-native:

> A prune artifact passed replay.

Journal-native:

> The proposed deletion passed a rollback-safe projection/refit check and was reproduced under the frozen benchmark protocol.

## Style rules

- Do not mention file paths in journal prose unless discussing reproducibility supplements.
- Do not mention route names, CLI flags, or internal field names in final-polish
  main-manuscript prose unless the local paragraph is explicitly a
  method-specification or reproducibility-contract paragraph. In Paper-II
  source-of-truth method sections, implementation labels may be included after
  the mathematical object is defined.
- Translate implementation facts into one of: equation, algorithm step, benchmark protocol, figure, table, or supplemental reproducibility record.
- Keep QPU-faithful Paper II language strict: exact reference data may appear as diagnostics, never as decision inputs.
- Prefer "measurement-compatible estimates" over vague "observable data" when discussing controller decisions.
- Prefer "compiled two-qubit burden" or "compiled-resource burden" over raw internal cost proxies unless the proxy is explicitly defined.
- Keep Paper I and Paper II asymmetric:
  - Paper I: static adaptive ansatz / operator-support acquisition and pruning.
  - Paper II: live checkpoint-local append/prune/stay/repair maintenance.
- Treat **scaffold** as support-doc shorthand. In manuscript prose, prefer **operator support**, **generator sequence**, **adaptive ansatz**, or **variational manifold** unless a term is explicitly defined.
