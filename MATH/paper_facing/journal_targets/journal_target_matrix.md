# Journal Target Matrix for Paper I / Paper II

Created: 2026-05-09  
Purpose: compact venue-routing guide before major manuscript rewrites.  
Sources: current official journal pages and representative article pages listed in the individual dossiers.  
Rule: do not rely on stale venue deadlines; recheck official pages before submission.

| Journal | Paper I fit | Paper II fit | Evidence needed | Risk |
|---|---|---|---|---|
| PRX Quantum | Ambitious if the static selector is framed as a broad quantum-algorithm contribution with modern ADAPT comparators and strong Pareto/ablation evidence. | Ambitious if checkpoint bidirectional dynamics is broadly benchmarked, measurement-compatible, and clearly beyond AVQDS/adaptive pVQD. | Cross-family benchmark matrix, modern baselines, finite-shot/compiled/noisy evidence, clear broad-interest hook. | Desk rejection if contribution looks incremental, small-system, or repo-specific. |
| Quantum | Strong candidate for technically precise algorithmic work if assumptions, contributions, and limitations are clear in first pages. | Strong candidate if controller mechanics and comparisons are technically complete and reproducible. | Exact comparator names, reproducible data/code, honest scope, contribution summary early. | Correct but incremental work is below threshold; long review times. |
| Physical Review A | Good if pitched as quantum-algorithm / quantum-simulation method rather than materials physics. | Good for variational dynamics, AMO/QIS-adjacent simulation, and method evidence. | Solid methods, exact baselines, no overbroad quantum-advantage claims. | May be less ideal if Hubbard--Holstein material physics dominates. |
| Physical Review Research | Good broad physics-methods fallback with room for longer method and benchmark detail. | Good if the paper is a broadly useful dynamics method with transparent evidence. | Complete benchmark package and clear novelty; open-access/APC planning. | Less selective branding than PRX Quantum; still needs significance beyond implementation. |
| npj Quantum Information | Good if the story emphasizes near-term quantum information, resource pressure, reproducibility, and code/data availability. | Good if trajectory/resource Pareto and measurement-compatible decisions are hardware-facing. | Nature-style concise Article structure, data/code availability, strong figures. | Less natural if the paper reads as narrow condensed-matter model benchmarking. |
| Physical Review B | Conditional: fit improves if mixed fermion--boson physics and Hubbard--Holstein benchmarks dominate over algorithmic generality. | Conditional: strongest if model-Hamiltonian dynamics/observables become the central contribution. | Material/model physics relevance, comparison to condensed-matter methods, careful PRB section choice. | Algorithm-only ADAPT/VQA framing may be out of scope or less compelling. |

## Corrected routing recommendation

Portfolio framing:

> **Paper 1 constructs compact mixed fermion--boson variational scaffolds under geometric and hardware budgets. Paper 2 maintains those scaffolds during real-time McLachlan propagation by reversible append--prune checkpoint control.**

Paper I routing:

- Strongest realistic targets: **Physical Review A** or **Quantum**.
- **PRX Quantum** only if the final Pareto story is much broader than a few diagnostic cases: pool-matched baselines, held-out sweeps, encoding/layout stress tests, real measurement allocation, and convincing resource reductions against modern ADAPT variants.
- **npj Quantum Information** is plausible if the resource/accounting angle is strong and the paper reads as broadly useful quantum-computing methodology rather than a narrow VQE variant.
- Do not route upward while the static draft still has unfinished benchmark placeholders or incomplete baseline tables.

Paper II routing:

- Best ambitious target: **PRX Quantum**, but only with a strong final benchmark matrix: cross-family mixed fermion--boson results, fixed-scaffold controls, AVQDS/adaptive-pVQD-style baselines, append/no-append and prune/no-prune ablations, finite-shot perturbations, and preferably fake-backend or hardware-facing validation.
- Best realistic target: **Physical Review A**, because it is a natural venue for detailed quantum-science methods papers and is less dependent on a field-shifting claim than PRX Quantum.
- **Quantum** is also strong if the algorithmic structure is clean, self-contained, reproducible, and honest about baselines.
- If pruning only works in one calm diagnostic case, route to PRA/Quantum rather than PRX Quantum. If append and prune both give robust Pareto gains across driven fermionic, bosonic, and mixed systems with finite-shot/noise stress, try PRX Quantum first.

General strategy:

- Finish Paper I first if it is numerically closer.
- Treat Paper II as the higher-upside submission once append--prune evidence is real.
- Use PRR as a broad physics-methods fallback when the evidence is complete but the paper lacks PRX Quantum-level breadth.
- Use PRB only if model-Hamiltonian physics and electron--phonon conclusions become central.

## Venues to avoid as first route unless the results change materially

- **PRL**: avoid unless the finished paper becomes a short, broadly important result rather than a detailed adaptive-variational methods paper.
- **PRX**: avoid unless there is a breakthrough/paradigm-shifting physics result, not just a strong algorithmic method.
- **Nature Communications**: avoid unless the result is significant to specialists across a broad field, with evidence beyond the current draft state.

## Optional/conditional conference venues

- **QIP**: not the natural first venue unless one paper is reframed as a theory-heavy quantum-information contribution with a clean formal result. QIP has no published proceedings and targets outstanding recent work in quantum information/computation.
- **IEEE Quantum Week / QCE**: useful as a conference companion for implementation, benchmarking, hardware accounting, or software workflow. As of 2026-05-09, the official QCE26 technical-paper deadline had passed on 2026-04-27; poster, workshop-paper, and panel routes had later deadlines, so recheck the current track before planning.
- **NeurIPS**: weak unless genuine machine-learning content is added: learned acquisition policies, generalization, regret, differentiable controller training, or a reusable combinatorial-search formulation. A heuristic adaptive quantum algorithm alone is not enough.

## Links

- PRX Quantum: https://journals.aps.org/prxquantum/scope; https://journals.aps.org/prxquantum/authors
- Quantum: https://quantum-journal.org/about/; https://quantum-journal.org/instructions/authors/
- Physical Review A: https://journals.aps.org/pra/about; https://journals.aps.org/pra/authors
- Physical Review Research: https://journals.aps.org/prresearch/scope; https://journals.aps.org/prresearch/authors
- npj Quantum Information: https://www.nature.com/npjqi/aims; https://www.nature.com/npjqi/for-authors-and-referees/guide-to-authors
- Physical Review B: https://journals.aps.org/prb/about; https://journals.aps.org/prb/authors
