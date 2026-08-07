# Paper I Journal Recommendations

Created: 2026-05-09  
Updated: 2026-07-10
Companion strategy memo: `MATH/paper_facing/two_paper_strategy.md`

## Active target

**PRX Quantum** is the author-selected target for the SNAKE manuscript.
The prospective editorial route is exceptional capability: a transferable
method for resource-constrained adaptive variational-manifold acquisition.
Treat this as the target and audit criterion, not as a conclusion that the
current manuscript or evidence already satisfies the journal's threshold.

Use `journal_variants/prx_quantum.md` for the Paper-I venue overlay.

## Fallback targets

1. **Quantum** — strongest overall fit if the algorithmic abstraction is clean, self-contained, reproducible, and honest about scope/limitations.
2. **Physical Review A** — best acceptance-likelihood fallback if the mixed fermion--boson physics and compiled-resource consequences are foregrounded.
3. **npj Quantum Information** — strong if the paper reads as hardware-conscious quantum-algorithm methodology with realistic measurement/compilation accounting.

## Conditional targets

- **Physical Review Research** as a broad physics-methods fallback.
- **Physical Review B** only if Hubbard--Holstein/electron--phonon physics conclusions become central.

## Not first-route targets

- PRL, PRX, or Nature Communications unless the finished result becomes much stronger than the current support state.
- QIP unless a formal theory contribution is introduced.
- NeurIPS unless genuine ML/search learning content is introduced.

## Paper I-specific venue concerns

Likely reviewer concerns:

- Is this one crisp principle or a bundle of heuristics?
- Are measured gains benchmark-specific?
- Are the many knobs reproducible?
- Does candidate-position Schur reranking generalize beyond this pool/model choice?

Preempt with:

- one dominant claim: candidate-position Schur-reranked admission under cost;
- held-out sweeps and paired ablations;
- released controller settings / benchmark registry;
- proxy-vs-final-resource correlation plots;
- encoding/layout panels that materially affect conclusions.

## Evidence needed before routing upward

- Full baseline table against ADAPT-VQE and feasible modern variants.
- Phase-by-phase ablations.
- Proxy-vs-final compiled cost validation.
- Encoding/cutoff/layout stress tests.
- Frozen protocol separating tuning from evaluation.

## First journal variants to maintain

- `journal_variants/prx_quantum.md`
- `journal_variants/quantum.md`
- `journal_variants/pra.md`

Do not make full `.tex` duplicates until a target journal is selected.
