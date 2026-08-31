# Quantum Style Dossier

Created: 2026-05-09  
Access date for links: 2026-05-09.  
Role: journal-style and venue-fit guidance for Paper I / Paper II support work.

## Official sources checked

- Scope/about: https://quantum-journal.org/about/
- Author guidelines: https://quantum-journal.org/instructions/authors/
- Additional shared guidance: APS length/style/data pages where applicable; Nature content/data pages where applicable.

## Representative articles / article pages checked

- https://quantum-journal.org/papers/
- https://doi.org/10.22331/q-2021-07-28-512

These examples were used for structure/style patterns only. Do not copy prose.

## Official scope

Quantum is an open, community-run journal for quantum science and related fields. It publishes original research, reviews, and software-showcase papers across theoretical, experimental, numerical, and software contributions.

## Article types and formatting expectations

Submission requires an arXiv preprint posted to or cross-listed with quant-ph. There are no typesetting, format, or length constraints, but the main results and assumptions should appear clearly in the first pages. Cover letters are not necessary. Quantum LaTeX class is encouraged for final version.

## Abstract style

Abstract can be technical, but the first pages must make contributions and assumptions clear. Avoid broad hype; be precise and verifiable.

## Introduction style

Aim the introduction at expert editors closest to the topic. State contributions early, preferably as a concise contribution list or paragraph.

## Related-work density

High and precise. Quantum values relation to literature, technical correctness, significance, reproducibility, and honest limitations.

## Methods/results balance

Method detail can be substantial. Results should demonstrate state-of-the-art improvement or conceptual/technical significance, not merely correctness.

## Equation tolerance

High. Quantum is comfortable with mathematical and algorithmic detail, but assumptions must be explicit and checkable.

## Figure/table conventions

Figures should support the argument, but dense proofs/derivations are acceptable if organized. Include reproducibility-friendly tables and appendices.

## Data/code availability expectations

Strongly encourages data and code under FOSS licenses; editors/referees must receive anonymous/free access when needed. Software showcases need community access.

## Cover-letter or submission guidance

No cover letter necessary; submit arXiv reference, keywords, suggested editors/referees/referee exclusions. No submission fee.

## Why Paper I fits / does not fit

Paper I fits if it is a technically precise contribution above incremental ADAPT variants, with exact assumptions and reproducible baselines. It does not fit if the selector is only empirically tuned without clear algorithmic abstraction.

## Why Paper II fits / does not fit

Paper II fits if the checkpoint controller is formalized cleanly and compared honestly against pVQD/AVQDS/adaptive pVQD. It does not fit if exact-reference leakage clouds the method definition.

## Evidence threshold that would make this journal realistic

Clear contribution/assumption summary in first pages; code/data availability; exact baselines; ablations proving which components matter.

## Claims too strong for this journal without more evidence

"State of the art" without reproducible comparison; hidden assumptions; missing limitations; broad claims unsupported by data.

## Reusable drafting guidance

- Put contributions and assumptions in first two pages.
- Be technically direct; avoid broad-interest padding.
- Include limitations and failure modes honestly.
- Make reproducibility assets easy to locate.
- Use `MATH/Math.md` definitions rather than repo prose.

## Links

- https://quantum-journal.org/about/
- https://quantum-journal.org/instructions/authors/
- Representative examples above.
