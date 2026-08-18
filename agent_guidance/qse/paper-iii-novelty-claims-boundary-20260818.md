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
- adaptive enlargement of a quantum Rayleigh--Ritz subspace, or adaptive
  operator-pool selection over a nonorthogonal generalized eigenproblem —
  this lineage rests on ESTABLISHED, refereed work, not on A-CASE:
  residual-guided QDavidson (`Tkachenko2024QDavidson`, 2022/2024) and
  gradient-based ADAPT-GCIM over UCC generator pools
  (`Zheng2024ADAPTGCIM`, arXiv:2312.07691, npj QI 2024). A-CASE itself
  concedes "neither adaptive subspace growth nor replacing variational
  optimization by a generalized eigenproblem is new";
- utility-per-measurement-cost subspace growth, or a multiplicative cost
  denominator in the acquisition score (`UtamaDipojono2026ACASE`,
  `UtamaDipojono2026DACASE`) — this NARROW cost-normalization point is
  the only place A-CASE is the operative collision. Framing rule
  (user-directed 2026-08-19): concede adaptive construction to
  QDavidson/ADAPT-GCIM, cite A-CASE only for the cost denominator; never
  present A-CASE as the principal threat to adaptive-basis novelty.
  Judge A-CASE solely by its public disclosure — authorship/provenance
  impressions must not enter any priority argument;
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
  `pipelines/exact_bench/paper_iii_qse_regime_frontier_sweep.py`. Two
  corrections locked in on 2026-08-18: (a) the u=8 "pool limitation" first
  recorded was a reference artifact — the true sector E1 is the spin
  triplet (dE=0.475, exactly degenerate with its S_z=+-1 partners in the
  (2,0)/(0,2) sectors), which expectation-based sector filtering dropped;
  with the exact sector-projected reference the u=8 manifold limit is
  2.4e-9 and all selection arms reach 5e-6 or better. Sector references
  must always come from the exact sector-restricted eigenproblem. (b) The
  genuinely truncation-limited regime at nph3 is weak_strong (manifold
  limit 4.3e-5), motivating the canonical nph7 strong-phonon pools.
- Exchange repair of the stuck strong-phonon supports
  (`pipelines/exact_bench/paper_iii_qse_exchange_repair.py` ->
  `output/diagnostics/paper_iii_exchange_repair_20260818_v1/`): from the
  stalled alpha=1 supports, certified exchange reaches weak_strong
  2.8e-5 @ 218 2Q (one dominance patch; beats the complete class 9.4e-4 @
  412 in both coordinates) and intermediate_strong 1.1e-4 @ 182
  (dominance) / 8.5e-6 @ 254 (budgeted at class parity, vs class 4.8e-5 @
  412). Select-then-exchange dominates the fixed class in accuracy and
  compiled cost in every tested regime — this is the paper's central
  results narrative.
- Exchange maintenance (C3 implementation):
  `pipelines/qse_spectra/exchange_maintenance.py` — certified joint
  delete--add patches with atomic commit; evidence in
  `exchange_maintenance_evidence.json` (v1 evidence dir): geometry support
  improved 6.8e-5 -> 4.0e-6 at identical 48 2Q via four certified patches;
  budgeted variant reaches 6.8e-5 at 12 2Q from the cheapest-first support.
  C3's standing obligation (joint patch, not prune-then-append) is satisfied
  by this module; keep the budgeted variant described as budgeted, not as
  plain dominance.
- Paper-I-convention sweep (nph3 weak / nph7 strong phonon sectors,
  canonical u/g from stationary-core artifacts, per-regime full_meta pools):
  `pipelines/exact_bench/paper_iii_qse_paper_i_convention_sweep.py` ->
  `output/diagnostics/paper_iii_paper_i_convention_sweep_20260818_v1/`.
  The sweep's raw selection arms show a mixed strong-phonon picture
  (alpha=1 dominates strong_strong_u8; the complete linear-response class
  wins at weak_strong/intermediate_strong), which the exchange repair
  resolves — cite the raw sweep as motivation for
  exchange maintenance and multi-root objectives, never hide it.
- Child-granularity study (Paper II atom coordinate):
  `pipelines/exact_bench/paper_iii_qse_child_granularity.py` ->
  `output/diagnostics/paper_iii_child_granularity_20260818_v1/`. The child
  span is numerically exact (1e-15) in the studied regimes — the earlier
  weak_strong "nph3 truncation limit" (4.3e-5) was macro-span limitation,
  not phonon truncation; children require exact sector projection in
  general (number non-conservation); macro granularity wins at low budget,
  children beyond ~200 2Q.
- Multi-root evidence (2026-08-19 overnight): Ky Fan trace objective
  (`target_root_count`) in exchange and multi-root residual scoring in
  geometry selection (`geometry_target_roots`, bounded discount floor
  `geometry_cost_discount_floor` after the zero-cost pathology). At budget
  60 select-then-exchange resolves all six excitations to
  (near-)manifold-limit accuracy in every Paper I regime
  (`paper_iii_multiroot_sweep_20260818_v1/multiroot_sweep_b60.json`); the
  complete fixed class fails several higher roots in five regimes and,
  being complete, cannot be repaired by budget. Transition strengths
  (`paper_iii_transition_strengths_20260818_v1/`): fixed class collapses
  at nph7 (relative errors up to ~10) while exchange supports hold
  1e-3..1e-9 — this is the C1 "and transition strengths" evidence.
- L=3 pilot (2026-08-19, `pipelines/exact_bench/paper_iii_qse_l3_pilot.py`
  -> `output/diagnostics/paper_iii_l3_pilot_20260819_v1/`): at L=3/nph1
  the 200-element full pool collapses to retained rank 72 (overlap
  condition ~1e10) and loses roots 4-6; the selected 60-support resolves
  all six roots to 4e-9..2e-3 at ~290 2Q. Selection beats the FULL POOL,
  the fixed class, and input order — the scaling/conditioning argument
  for the paper. Spin-boson family is registry-ready for the generality
  arm; Peierls-Hubbard would be a new problem family (user decision).
- Peierls-Hubbard pilot (2026-08-19,
  `pipelines/exact_bench/paper_iii_qse_peierls_pilot.py` ->
  `paper_iii_peierls_pilot_20260819_v1/`): second e-ph family (bond
  coupling). Fixed linear-response class fails catastrophically (0.5-6.8,
  cannot resolve six roots); selection matches the full-pool manifold at
  ~4x lower cost. peierls_v1 pool span limits several roots (needs higher
  displacement powers); registry promotion pending user architecture
  review. Supports the generality claim: the right class is
  coupling-structure-dependent, selection finds it automatically.
- Retained-frame selection correction (2026-08-19, algorithmically
  important): geometry novelty is now measured against the numerically
  RETAINED principal directions of the selected images under the solver's
  relative overlap cutoff, not the exact span — span-based novelty counts
  directions carried below the pencil's retention weight and permanently
  floors the operators that could re-supply them (found via the Peierls
  triplet sector). With the fix the Peierls pilot resolves all six roots
  to 1e-10..1e-15 in every regime; HH/L=3 evidence rerun (`*_retained.json`,
  consolidation repointed). Manuscript point: the selection metric must
  share the pencil's stabilization scale.
- Machinery: `pipelines/qse_spectra/compiled_costs.py` (2Q-only preset
  `two_qubit_only_v1`), selection modes in
  `pipelines/qse_spectra/record_selection.py`.
- Driven-dynamics matrix (2026-08-19,
  `pipelines/exact_bench/paper_iii_qse_driven_dynamics.py` ->
  `paper_iii_driven_dynamics_20260819_v1/driven_dynamics.json`): frozen
  QSE propagation of the selected first-excitation root under the
  staggered-density gaussian-sinusoid drive (A=0.2, tbar=4, T=8, 160
  midpoint steps), frequency swept on a QSE-anchored grid. Finding: the
  manuscript escape FRACTION saturates at 1 for near-eigenstate initial
  states (variance = squared static residual); the informative,
  measurement-compatible scalar is the escape FLUX (unnormalized
  numerator, sqrt). Baseline flux = static Ritz residual (< eps=1e-3
  everywhere); drive amplifies it 40-240x in the four responsive
  regimes, tracking fidelity loss (min 0.937-0.981) and observable
  error. Both u=8 regimes are drive-dark (first excitation = sector
  spin triplet, one electron/site, annihilated by staggered density):
  zero response, flux flat at baseline, frozen propagation exact — the
  diagnostic certifies dynamical closure without exact references.
- Adaptive-growth negative result (2026-08-19,
  `pipelines/exact_bench/paper_iii_qse_adaptive_dynamics.py` ->
  `adaptive_dynamics.json`): admitting the best drift-aligned pool
  record at escape checkpoints (support rebuild + state re-injection,
  losses <=1e-8) does NOT improve driven fidelity: admitted records
  align up to ~60% with the instantaneous drift at isolated
  checkpoints (few percent at most others), yet peak flux and fidelity
  are unchanged at the 16-addition cap — the drive regenerates
  out-of-pool directions faster than records can supply them. Driven
  escape is POOL-LIMITED — the missing directions are not any pool
  record applied to the reference — so
  record-level adaptivity cannot close it and live circuit refinement
  (AP-McLachlan handoff, Paper II exchange route) is the principled
  third tier. Claim boundary: do NOT claim adaptive-QSE growth closes
  driven escape; DO claim the escalation trigger (flux + alignment
  scan) is fully premeasurable. Per-regime promoted-AP handoff is NOT
  yet evidence (July demo is locked to the weak_weak advisor
  artifacts); do not cite it as regime-wide.
- Three-tier handoff validation (2026-08-19,
  `pipelines/excited_dynamics/paper_iii_regime_handoff.py` ->
  `paper_iii_regime_handoff_20260819_v1{,_exchange}/`): weak_weak,
  worst-escape omega=1.25. Compact refit 32 Paulis @ 2.1e-14; promotion
  at numerical precision. Min fidelity vs exact: frozen pencil (73
  records) 0.958 > AP append-only adaptive 0.639 > AP exchange recipe
  0.431 (locally certified deletions accumulate; drive never quiets;
  params 30->12) > fixed-support circuit 0.143. CLAIM BOUNDARY: C4
  remains a WORKFLOW claim only — never claim live McLachlan refinement
  outperforms frozen-QSE propagation (measured: it does not, at this
  scale with a generic 30-atom pool); DO claim mechanism validation and
  that structural adaptation dominates fixed circuits (3-4.5x). Forward
  path (unevidenced — do not claim): QSE-informed live pools (selected
  records' Pauli children). Ops lesson pinned: the demo-era
  `_run_ap_grid` config lacked the route doc's mandatory guards
  (max_joint_patch_evaluations / max_certification_attempts_per_level /
  max_structural_pool_size) — without them the post-purge selector
  grinds (~8 CPU-min/step) and accumulates memoized solves (~500
  MB/min); with them, ~3 s/step at flat 250 MB.
