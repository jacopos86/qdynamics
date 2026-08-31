# PRX Quantum Style Dossier

Created: 2026-05-09  
Access date for links: 2026-05-09.  
Role: journal-style and venue-fit guidance for Paper I / Paper II support work.

## Official sources checked

- Scope/about: https://journals.aps.org/prxquantum/scope
- Author guidelines: https://journals.aps.org/prxquantum/authors
- Additional shared guidance: APS length/style/data pages where applicable; Nature content/data pages where applicable.

## Representative articles / article pages checked

- https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.6.010317
- https://journals.aps.org/prxquantum/abstract/10.1103/yyln-q22s

These examples were used for structure/style patterns only. Do not copy prose.

## Official scope

PRX Quantum covers the broad quantum information science and technology community across physics, computer science, mathematics, chemistry, materials, engineering, and technology. Its scope includes quantum computation, simulation, algorithms, hardware, benchmarking, many-body contexts, and quantum algorithms for chemical calculations.

## Article types and formatting expectations

Research Articles have no length limit. Perspectives have a 7,500-word guideline, Tutorials 37,500 words, and Comments/Replies 3,500 words. APS/REVTeX style and Physical Review data-availability policy apply. The manuscript should be written for broad QIST readers.

## Abstract style

Open with a field-level bottleneck and the central advance, not with implementation detail. Abstract claims must be broad-interest, specific, and evidence-tied.

## Introduction style

Name the bottleneck quickly, then identify prior method families and the method object by paragraph 2. Avoid a long generic VQE tutorial.

## Related-work density

Moderate-to-high, but only for real prior-work claims. PRX Quantum expects contextualization for a broad QIST audience; cite modern overlap papers explicitly.

## Methods/results balance

Balanced: method novelty must be clear, but results/figures must prove significance. A pure implementation paper will be weak.

## Equation tolerance

High tolerance for equations when they clarify algorithmic novelty. Keep notation introduced before deep derivations; push exhaustive derivations to supplement.

## Figure/table conventions

Figures should tell the broad story: method schematic, comparator table, Pareto fronts, ablations. Tables should be readable across subcommunities.

## Data/code availability expectations

APS data availability statement required. Public data/code are encouraged; if not public, state why and make supporting data available privately on reasonable request.

## Cover-letter or submission guidance

Cover letter should explain journal fit, context/significance, related Physical Review submission history or joint submissions, and recommended/excluded referees.

## Why Paper I fits / does not fit

Paper I fits only if the candidate-position selector is shown to be a broadly useful quantum-algorithm advance beyond ADAPT variants, not just a repo benchmark. It does not fit if evidence is limited to one HH configuration or internal resource proxies.

## Why Paper II fits / does not fit

Paper II fits only if bidirectional checkpoint maintenance is clearly beyond AVQDS/adaptive pVQD and supported by trajectory/resource/noise or finite-shot evidence. It does not fit if it reads as exact-assisted simulation tooling.

## Evidence threshold that would make this journal realistic

Modern baselines, cross-family benchmarks, phase/lane ablations, measurement-compatible decision audit, and compiled/noisy resource evidence for any hardware-facing claim.

## Claims too strong for this journal without more evidence

"First geometry-aware ADAPT," "adaptive variational dynamics is new," "hardware-ready," or "practical advantage" without stringent evidence.

## Reusable drafting guidance

- Lead with field-level bottleneck.
- State the method object by paragraph 2.
- Use a broad-interest result hook before notation.
- Tie every performance claim to a figure/table.
- Keep exact diagnostic references out of QPU-faithful decision prose.

## Links

- https://journals.aps.org/prxquantum/scope
- https://journals.aps.org/prxquantum/authors
- Representative examples above.
