# Paper III novelty claims boundary (2026-08-18)

Scope: read this whenever editing `MATH/paper_details/excited_spectra_dynamics_paper_III.tex`
or writing Paper III claim, abstract, introduction, or referee-response text.
It distills two deep-research novelty audits (search cutoff 2026-08-17/18) and
is the authoritative boundary between claimable and conceded territory. It is
lane-scoped guidance: agents not working on Paper III have no reason to load it.

## Verdicts (final, post-audit-v2)

| Claim | Verdict | Most damaging prior work |
|---|---|---|
| C1: QSE on Hubbard--Holstein | PARTIALLY NOVEL | Backes et al., PRB 107, 165155 (2023) — HH quantum-Krylov spectral calculation on hardware (`Backes2023HHDMFT`) |
| C2: cost-weighted QSE basis construction | **NOT NOVEL generically** | Utama & Dipojono A-CASE, arXiv:2608.00560 (2026, preprint) — utility/cost acquisition with a multiplicative measurement-cost denominator (`UtamaDipojono2026ACASE`) |
| C3: exchange-maintained (delete--add) QSE bases | PARTIALLY NOVEL | Patra et al. PIGen-SQD, arXiv:2512.06858 (preprint) — drop/blacklist/regenerate in a measured SQD subspace (`Patra2025PIGenSQD`); classically Wu--Simon thick-restart Lanczos (`WuSimon2000ThickRestart`) |
| C4: QSE-root handoff to adaptive McLachlan dynamics | NOVEL, NARROWLY | Gomes, Williams-Young & de Jong, JCTC 19 (2023) — adaptive propagation of particle-added response branches (`Gomes2023AVQDSGreen`) |

## Never claim (conceded; each has a bib key in the manuscript)

- first HH quantum-Krylov or measured-subspace calculation (`Backes2023HHDMFT`);
- first electron--phonon VQD/qEOM treatment (`ZhouShang2024EPhQC`);
- first mixed fermion--boson excited-state quantum method (`PavosevicFlick2021QEDEOM`, `SawayaHuh2019VibronicSpectra`);
- first quantized electron--phonon quantum simulation (`Macridin2018ElectronPhonon`);
- utility-per-measurement-cost subspace growth, or a multiplicative cost
  denominator in the acquisition score (`UtamaDipojono2026ACASE`,
  `UtamaDipojono2026DACASE`);
- QSE excitation-pool pruning or compaction (`LiuDeng2026CSQSE`,
  `Getelina2024HardwareNoiseQSE`);
- resource-aware quantum-subspace design or measurement optimization
  (`Zhang2024MEQKSD`, `Nakamura2024AdaptiveMeasurementQSE`);
- quantum-subspace restart or discard-and-regrow iteration
  (`Rammal2026OQKD`, `OLeary2025PartitionedQSE`, `WuSimon2000ThickRestart`,
  `Sorensen1992IRA`);
- add--remove maintenance of a measured diagonalization subspace
  (`Patra2025PIGenSQD`, `RobledoMoreno2025SQD`);
- adaptive excited-response or eigensolver-to-dynamics propagation
  (`Yao2021AVQDS`, `Gomes2023AVQDSGreen`, `Mootz2024AdaptiveGreen`,
  `Gandon2024NonadiabaticQSE`, `Sambasivam2026TIMESADAPT`,
  `Berthusen2024MRQDDynamics`).

## Claimable (sharpest audit-approved sentences)

- **C1:** "To our knowledge, this is the first operator-based quantum subspace
  expansion calculation of low-lying Ritz roots and transition strengths for
  an explicit Hubbard--Holstein lattice Hamiltonian with quantized phonon
  registers."
- **C2 (specialization only, never the generic law):** "We specialize
  cost-aware adaptive quantum-subspace construction to an explicit mixed
  fermion--boson response problem by ranking Hubbard--Holstein QSE directions
  against their incremental compiled two-qubit measurement-circuit cost and
  reporting the resulting gate-cost--spectral-accuracy frontier."
- **C3 (requires the implemented joint patch, not prune-then-append):** "To
  our knowledge, we introduce the first certified cost-gated joint
  delete--add move for a measured, nonorthogonal QSE response basis: a
  candidate patch is committed atomically only after recomputation of the
  projected pencil verifies improvement of the declared target-root objective
  and compiled hardware cost while satisfying overlap-conditioning and
  statistical-stability guards."
- **C4 (secondary):** "To our knowledge, this is the first explicit workflow
  in which a selected QSE Ritz root is prepared as the initial state of
  McLachlan real-time dynamics and the variational support continues to
  adapt along the resulting excited-state trajectory."

Lead framing: the **conjunction** — a compiled-cost-selected,
exchange-maintained QSE response manifold for an explicit mixed fermion--boson
lattice Hamiltonian, with the QSE-root-to-adaptive-McLachlan handoff as a
secondary workflow contribution. Accuracy checks (gap error 6.3e-8, eight
matched roots, fidelity 0.999999983, 0.9879-vs-0.9209 propagation fidelities)
are implementation validation, never novelty.

## Standing obligations

1. C2 language must always say *compiled two-qubit measurement-circuit cost*
   — never generic "hardware cost" or "measurement cost" — and must cite
   A-CASE/DA-CASE as concurrent preprint work in the same design space.
2. C3 survives only as an atomic joint patch with before/after pencil
   certification; if the implementation degrades to sequential prune + append,
   the claim degrades with it (Paper II selector dependency).
3. C2 claims require the implemented cost ledger and frontier data, which
   exist: see the evidence inventory below.
4. Preprint-only sources (A-CASE, DA-CASE, PIGen-SQD, OQKD, TIMES-ADAPT) are
   marked "preprint" in the bibliography; keep that marking at submission and
   re-verify their publication status.

## Evidence inventory (statevector diagnostics, committed drivers)

- Frontier arms + oracle cross-check:
  `output/diagnostics/paper_iii_cost_frontier_arms_20260818_v1/` (graph-span),
  `..._transpile_20260818_v2/` (full transpile; identical selections;
  Spearman 0.9999 in `oracle_agreement_2q.json`).
- Comparator arms (fixed-class QSE, kicked real-time Krylov):
  `comparator_arms_summary.json` in the v1 directory; driver
  `pipelines/exact_bench/paper_iii_qse_comparator_arms.py`.
- Multi-regime sweep (six HH regimes, exact-sector references, alpha sweep):
  `output/diagnostics/paper_iii_regime_frontier_sweep_20260818_v1/`; driver
  `pipelines/exact_bench/paper_iii_qse_regime_frontier_sweep.py`. Notable:
  at u=8 the full 158-operator manifold itself misses the sector gap
  (limit error 5.2e-1) — report as a pool limitation, never a selection
  failure; selection reaches the manifold optimum at the lowest cost there.
- Exchange maintenance (C3 implementation):
  `pipelines/qse_spectra/exchange_maintenance.py` — certified joint
  delete--add patches with atomic commit; evidence in
  `exchange_maintenance_evidence.json` (v1 evidence dir): geometry support
  improved 6.8e-5 -> 4.0e-6 at identical 48 2Q via four certified patches;
  budgeted variant reaches 6.8e-5 at 12 2Q from the cheapest-first support.
  C3's standing obligation (joint patch, not prune-then-append) is satisfied
  by this module; keep the budgeted variant described as budgeted, not as
  plain dominance.
- Machinery: `pipelines/qse_spectra/compiled_costs.py` (2Q-only preset
  `two_qubit_only_v1`), selection modes in
  `pipelines/qse_spectra/record_selection.py`.
