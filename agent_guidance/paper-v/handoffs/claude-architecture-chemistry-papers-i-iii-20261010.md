# Claude architecture reply: molecular electronic structure into Papers I–III (Program A, A3)

Date: 2026-10-10. Author role: Claude architecture (persistent agent for Program A).
Answers: the Codex groundwork report
`agent_guidance/shared/handoffs/chemistry-papers-i-iii-groundwork-20261010.md`
(branch `codex/chemistry-papers-i-iii-groundwork-20261010`, commit `d3c2b27f`).
Status: architecture decisions and Codex handoff skeletons. No source, test,
contract or manuscript was edited; nothing was committed.

| Anchor | Value |
|---|---|
| Source read | `/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`, branch `claude/paper-v-routes-20260914`, HEAD `0de9685a` (main `d1436ab1` merged in) |
| Source drift vs Codex report | `git diff a264ca23 0de9685a -- src pipelines test` is empty; every `path:line` below is valid at both commits |
| Graph index | `/Users/lordskyler/Holstein_test` at `d1436ab1`, indexed 2026-09-22; `git diff d1436ab1 HEAD -- src pipelines test` touches one Paper-V test only, so the graph is current for every symbol named here |
| Python | `/Users/lordskyler/Holstein_test/.venv/bin/python` |
| Upstream chemistry code | not selected (GitHub user `doronwu` verified, no repository). Everything below is an input contract, independent of the upstream format |

---

## 0. Decisions and negative results (read first)

1. **Negative: a 20-qubit molecule cannot run through Paper I or Paper II today.**
   Paper II's online McLachlan geometry multiplies a dense `2^n x 2^n` H(t)
   (`pipelines/time_dynamics/ap_mclachlan/geometry_eval.py:72-77`, matrix built at
   `ap_mclachlan/hamiltonian.py:50-56,89`). Paper I resolves a dense exact
   energy on every run without an exact-ED stop
   (`pipelines/static_adapt/sr_snake/_context.py:781-787` ->
   `builders/problem_setup.py:629-635` -> `src/quantum/vqe_latex_python_pairs.py:2342`,
   dense builder `:2293-2305`). One dense complex matrix is `16 * 4^n` bytes:
   0.27 GB at n=12, 4.3 GB at n=14, 69 GB at n=16. Paper III's QSE core is
   matrix-free (`pipelines/qse_spectra/core.py:777`), but its providers compute
   the reporting spectrum densely (`pipelines/qse_spectra/paper_iii_problem.py:233-235`).
   **Decision:** first molecular fixtures are capped at 12 qubits (6 active
   spatial orbitals); the matrix-free increments in §6 lift the cap.
2. **The admissibility seam already exists below the facade.** The shared
   static method admits a non-HH problem when the request's candidate adapter
   declares `application_family_key == problem.family_key` and a route override
   is supplied (`sr_snake/_context.py:752-766`); the H2O adapters declare it
   (`ra_adapt/adapters.py:57,360,377,406`). The only Hubbard--Holstein lock that
   must change is the facade if-chain in `run_ra_adapt`
   (`ra_adapt/engine.py:5742-5755`) plus the four H2O-style touchpoints listed
   in §2.1. **Decision:** admit `molecular_restricted_closed_shell` as a named
   application keyed on `application_family_key`, added as new branches after
   the existing ones so every Hubbard--Holstein byte is unchanged.
3. **Measured: every molecular UCCSD generator is one sector-safe block.** In
   `build_molecular_uccsd_pool` (`src/quantum/chemistry/molecular_uccsd.py:9-26`,
   excitations `src/quantum/vqe_latex_python_pairs.py:797-905`) the
   Jordan--Wigner words of each generator mutually commute and share one X/Y
   flip support: for 4 spatial orbitals with (2,2) electrons, 26 of 26
   generators; for 6 with (3,3), 117 of 117; zero noncommuting word pairs; every
   `execution_mode` is `termwise_product`. Consequences: (a) the H2O
   sector-complete Pauli-block split (`ra_adapt/pools.py:1140`) returns the
   whole generator, so the Paper-I molecular candidate is the macro generator
   and the raw single-word representation is inadmissible; (b) per-word
   execution reproduces grouped execution exactly, so Codex's grouped-execution
   incompatibility with the AP structural cache (report §4) does not arise for
   spin-orbital UCCSD; it remains real for H2O vibronic blocks and for any
   spin-adapted generator that sums several excitations.
4. **The three reported baseline failures reproduce and are pre-existing
   records, not new defects** (§5.3). None sits on the molecular path; two
   matter for Paper II's seed contract.
5. **Evidence ownership.** The application identity is named by problem family,
   not by paper. Whether any molecular result becomes Paper I, II or III
   evidence, and which lane owns the resolved molecular problem, stays with the
   user (campaign plan §2, "Reserved for the user").

---

## 1. Input contract every lane needs from any upstream chemistry code

The lanes build the qubit Hamiltonian themselves from integrals; an upstream
Pauli-string export is used only as an independent conversion witness after
re-indexing to the convention below. This keeps mapping conventions out of the
lane boundary. Present interchange object:
`RestrictedClosedShellMolecularProblem` (`src/quantum/chemistry/psi4_adapter.py:19-31`),
loaded by `restricted_closed_shell_problem_from_jsonable` (`:59-128`).

### 1.1 Common to Papers I, II, III

| # | Quantity (repository field) | Units | Convention the lane assumes | Enforced where |
|---|---|---|---|---|
| 1 | `n_spatial_orbitals` = m (active) | count | `request.num_sites` must equal m | `builders/problem_registry.py:815-820` |
| 2 | `n_alpha`, `n_beta` | count | equal (closed shell) | `psi4_adapter.py:83-86` |
| 3 | `reference`, `multiplicity` | label | `"rhf"`, 1; open shell and UHF/ROHF are rejected | `psi4_adapter.py:78-83` |
| 4 | `one_body_integrals_mo` h_pq | Hartree | real, shape (m,m), orthonormal spatial MOs; if a core is frozen, h is the core-contracted h^eff (formula in the Codex report §2, producer `generate_h2o_linear_fd_fixture.py:527`) | shape `psi4_adapter.py:87-102`; real-only via `_real_scalar` in `molecular_hamiltonian.py` |
| 5 | `two_body_integrals_mo` (pq\|rs) | Hartree | real, chemists' notation, spatial (not spin-orbital, not antisymmetrized), 8-fold symmetric; builder reads `eri[p,r,q,s]` with prefactor 1/2 | `molecular_hamiltonian.py:131-137` |
| 6 | `nuclear_repulsion_energy` (the scalar slot) | Hartree | E_nuc plus any frozen-core constant, added once as the identity coefficient | `molecular_hamiltonian.py:163`; **the loader silently defaults a missing key to 0.0** (`psi4_adapter.py:123`) — the lane must require it present |
| 7 | `hf_energy` | Hartree | must equal ⟨HF\|H\|HF⟩ from items 4–6 (conversion witness) | loader defaults missing to 0.0 (`psi4_adapter.py:122`); no check today |
| 8 | Orbital order | — | occupied orbitals first inside the active space; the reference fills spatial orbitals 0..n_alpha−1 in each spin block | `src/quantum/hartree_fock_reference_state.py:84` |
| 9 | Spin-orbital to qubit | — | blocked: alpha 0..m−1, beta m..2m−1; qubit index = spin-orbital index; Pauli words left-to-right q_(n−1)…q_0 | `agent_guidance/shared/scientific-invariants.md:27-28`; `pipelines/contracts/problem.py:62`; ordering check `problem_registry.py:811-814` |
| 10 | Fermion-to-qubit mapping | — | Jordan--Wigner, built in-lane from items 4–6 | `molecular_hamiltonian.py:151+` |
| 11 | Orbital-basis provenance | — | geometry with units (Å or bohr stated), basis-set name, charge, SCF type, point group used (c1 or declared irreps), active/frozen/deleted orbital index lists | identity only (`geometry_spec`, `basis`, `charge` fields) |
| 12 | Content digest | sha256 | digest of the exact JSON bytes consumed; seed artifacts today store only the path (`pipelines/scaffold/runtime_loader.py:1381`) | not enforced today |
| 13 | Reporting reference (optional) | Hartree | active-space FCI ground energy and lowest R roots, from the upstream code, for reporting and conversion checks only | never an online input (`agent_guidance/shared/scientific-invariants.md`, exact-reference isolation) |

Real-valued integrals are mandatory: the loader casts to `float`
(`psi4_adapter.py:68-69`). Complex integrals (spin–orbit, magnetic fields) or
open-shell references are a new problem family, not a parameter.

### 1.2 Paper II additions

| # | Quantity | Units | Convention |
|---|---|---|---|
| 14 | Dipole integrals d^ξ_pq, ξ ∈ {x,y,z} | e·a0 | same active MOs as item 4; core-contracted constant and nuclear dipole stated separately (an identity term cancels in the horizontal drift b̄ but stays in the energy) |
| 15 | Time and field units | atomic units | t in ħ/E_h; field in E_h/(e·a0); drive waveform `c(t)=A sin(ωt+φ) exp(−t²/(2 t̄²))` (`src/quantum/drives_time_potential.py:119`) with A, ω, t̄, φ in these units |
| 16 | Basis motion | — | fixed orbital basis declared; moving-basis transport is out of Program A |

### 1.3 Paper III additions

| # | Quantity | Convention |
|---|---|---|
| 17 | Target root count R | declared per molecule; not the HH constant 6 (`paper_iii_problem.py:176`) |
| 18 | Spin multiplicity and irrep of each reference root (optional, reporting) | needed to interpret sorted-index matching (§4.2) |
| 19 | Transition observables | dipole components, item 14 |
| 20 | Charged-sector request (Green functions only) | declared N±1 sector |

---

## 2. Paper I — Resource-Aware ADAPT (`agent_guidance/static-adapt/`)

### 2.1 Admissibility seam

**Where the problem enters today.** `ProblemRequest`
(`pipelines/contracts/problem.py:19-39`, lattice-shaped fields plus
`molecular_problem_json`) -> `resolve_problem_context`
(`builders/problem_registry.py:1963`) -> family spec
`molecular_restricted_closed_shell` (`:1731-1746`) -> resolver
`_resolve_molecular_problem_context` (`:804-932`) -> `ResolvedProblemContext`
(`pipelines/contracts/problem.py:156-178`): blocked layout, JW Hamiltonian,
sector `n_up`/`n_dn` fixed counts (`problem_registry.py:880-891`), RHF
reference, dense exact target, pools `uccsd, full_hamiltonian,
hamiltonian_blocks, hva, family_max, full_meta` with default `uccsd`
(`:1460-1467`). Resolution works today: an H2 STO-3G fixture resolves to 4
qubits, sector `closed_shell_fixed_number_sector`, default pool `uccsd`.

**What rejects it.** Five places, all Hubbard--Holstein or H2O specific:

| # | Site | What it checks |
|---|---|---|
| 1 | `ra_adapt/engine.py:5742-5755` (`run_ra_adapt`) | family ∈ {hh L=2, hh L=3, hubbard L∈{2,4}, H2O linear-FD}; molecular raises "locked to the canonical Hubbard--Holstein L=2 problem" (reproduced) |
| 2 | `ra_adapt/engine.py:514-547` (`_h2o_application_active`, called at `:5801`, `:1461`, `build_resolved_ra_protocol` region `:2680-2690`) | H2O adapter types and method policy (singleton admission, plateau or append-only insertion, pruning off, beam off) |
| 3 | `ra_adapt/engine.py:1461-1482`, `:1673-1684`, `:1959-1990` (`_repaired_route_contract`) | chooses the parent route contract; H2O uses `canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract` (`sr_snake_route_profile.py:3363-3409`, settings `:1100-1109`, `phase3_backend_cost_mode: proxy`) |
| 4 | `ra_adapt/pools.py:524-601` (`_parent_pool_spec`) and `:352-366` (`_require_supported_singleton_problem`) | returns the pool spec per family; falls through to H2O for any non-HH family |
| 5 | `ra_adapt/pools.py:271-281` (`_require_paper_i_problem`) inside `build_executable_macro_pool` (`:897-972`, which also hard-codes `problem_key="hh"` at `:927` and HH removed-count expectations) | macro pool locked to HH L=2 |

Below the facade the method is already family-agnostic: the SR runtime takes
the resolved Hamiltonian and family key from the context
(`pipelines/static_adapt/adapt_pipeline.py:30071-30076`), and
`_resolve_execution_context` admits any family whose adapter declares
`application_family_key` (`sr_snake/_context.py:752-766`).

**Smallest change (decided).** A named molecular application, mirroring the
H2O application but with one predicate instead of five scattered ones:

- one lane-owned predicate (suggested module
  `pipelines/static_adapt/ra_adapt/molecular_application.py`) that returns true
  only for family `molecular_restricted_closed_shell` **and** a request whose
  adapter is the new named macro adapter **and** singleton admission,
  append-only or plateau insertion, pruning off, beam off;
- a named macro adapter in `ra_adapt/adapters.py` declaring
  `application_family_key = "molecular_restricted_closed_shell"`,
  `application_pool_key = "uccsd"`, whose `executable_pool` builds the parent
  inventory through `_parent_records` and fails closed on any grouped or
  execution sector violation (pattern: `build_h2o_symmetry_complete_generator_pool`,
  `pools.py:1040-1100`), Phase-II exposure = retained macro identity (no child
  split, by decision 0.3);
- one added branch at each of sites 1–4, placed after the existing branches;
  site 5 is bypassed (the adapter supplies its own `executable_pool`, as H2O
  does);
- a parent route contract built like the H2O v3 overlay on
  `canonical_sr_snake_no_prune_symmetric_cost_v1_contract` with execution
  overrides `problem`, `adapt_pool: uccsd`, `phase3_backend_cost_mode: proxy`,
  and the horizon taken from the request (the parent leaves the horizon to
  the caller, `sr_snake_route_profile.py:697-703`; do not invent a constant).

The ordinary request (`request=None` -> default single-word adapter) on a
molecular problem stays rejected. Accepted-state resume stays HH/pure-Hubbard
only (`sr_snake/_resume.py:1455-1476`); fresh starts only.

**Regression pin for Hubbard--Holstein (bitwise).** Computed at `0de9685a`
with the test helpers `_hh_problem()` and
`RAAdaptRequest(execution=_execution(rounds=1))` from
`test/test_ra_adapt_facade.py:115-136`:

| Pinned value | At `0de9685a` |
|---|---|
| `protocol.algorithm_id` | `paper_i_ra_adapt_nonstationary_incremental_full_response_v2` |
| `protocol.sha256` | `fc25b5c22637fd6e52a11cda8cff08db59559e342c96f879a9a108e5ef698cc7` |
| `protocol.route_contract["sha256"]` | `0312875b4f190d94a77a37cd6c245bd678764077d0c0e365354b1e099060941a` |
| `test/test_ra_adapt_refactor_parity.py` (3-round accepted trajectories, energies at 10 decimals, protocol digests) | 2 passed, 9.4 s |

The existing facade digest test cannot serve as the pin: it is red at
baseline (§5.3).

### 2.2 Method assumptions about the Hamiltonian

| Assumption | Code site | Verdict for a molecular Hamiltonian |
|---|---|---|
| Operator pool: HH `full_meta` (lifted UCCSD, PAOP, HVA, Hamiltonian blocks) | `problem_registry.py:1507+`; `pools.py:589-596` | **Parameter.** Pool key `uccsd` exists for the family: spin-orbital singles and doubles relative to the RHF occupation (`vqe_latex_python_pairs.py:822-905`). Which pool is the scientific choice is GPT A2's. |
| Candidate representation: ordinary route stages single Pauli words | `ra_adapt/contracts.py:238-247`; `adapters.py:216` | **New implementation** (named macro adapter). Single words leave the (N_alpha, N_beta) sector at finite angle (`pools.py:1040-1050` docstring); decision 0.3 makes the macro generator the finest sector-safe candidate. |
| Sector audit of candidates | `pipelines/static_adapt/sector_invariants.py:58-115` | **Unchanged.** `n_up`/`n_dn` resolve to the blocked alpha/beta qubit groups (`:79-104`). |
| Cost model: Phase-I proxy, 2(w−1) CX per active word of weight w, single-qubit and shift-span terms | `pipelines/scaffold/hh_continuation_scoring.py:2039-2115`; H2O precedent sets `proxy` (`engine.py:1676-1684`) | **Unchanged.** Hamiltonian-independent; its values depend on the blocked ordering, which places alpha–beta doubles' Z strings across up to 2m−1 qubits. |
| `S_alg = N_H_outer + N_H_refit + N_grad + N_metric` | `agent_guidance/static-adapt/AGENTS.md:83-84` | **Unchanged as defined.** It counts occurrences, not Pauli terms; a molecular H with far more terms has the same `S_alg` per evaluation. Whether that remains the intended burden is a GPT A2 question. |
| Phase-III score `ΔE_TR/(1+K_3)` | `test/test_ra_adapt_cost_scope.py:53` | **Unchanged.** |
| Reference state | `hartree_fock_reference_state.py:84`; `problem_registry.py:836-844` | **Unchanged**, given input item 8. |
| Energy-dimensioned stop threshold `SRStopPolicy.gradient_tolerance` | `sr_snake/contracts.py:651-654`; H2O used 5e-7 (`ra_adapt/h2o_application.py:66`) | **Parameter**, in Hartree, declared per application. Trust radii in the Fubini–Study chart are dimensionless and unchanged. |
| Exact target (reporting) | `problem_setup.py:629-635` -> dense `vqe_latex_python_pairs.py:2342` | **New implementation above 12 qubits** (sector-sparse). Exact-ED stop (`agent_guidance/static-adapt/policies/stopping.md`) is opt-in by the user; the H2O launcher uses it (`h2o_application.py:143-165`) — do not copy that into the molecular application. |

---

## 3. Paper II — AP-McLachlan dynamics (`agent_guidance/time-dynamics/`)

### 3.1 Admissibility seam

**Where the problem enters today.** A seed artifact ->
`load_scaffold_runtime_input_from_payload` (`pipelines/scaffold/runtime_loader.py:2073-2114`),
which sends every non-`hh` family to `_load_generic_runtime_input_from_payload`
(`:2086-2095`, `:1043-1110`) -> `ScaffoldRuntimeInput`
(`pipelines/contracts/scaffold.py:31-67`) ->
`run_append_ap_mclachlan_from_runtime_input`
(`pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py:143-300`).
The AP runner has no family gate.

**What rejects or degrades it.**

| # | Site | Effect on a molecular seed |
|---|---|---|
| 1 | `runtime_loader.py:1379` | missing `settings.problem` silently resolves to `hh` |
| 2 | `runtime_loader.py:1075-1081`, modes `:43-47` | a seed with a parameterization layout gets a selected-only pool unless it declares `replay_candidate_pool_mode` ∈ {`diagnostic_replay_family_pool`, `family_pool`, `replay_family_pool`}; then `can_structural_edit` is false (`contracts/scaffold.py:66-67`) and generalized exchange cannot insert |
| 3 | `adapters/drive_terms.py:758-809` | `resolve_realtime_drive_model` raises "no drive-term seam" because `_MOLECULAR_CAPABILITIES.drive_operator_kind` is `None` (`problem_registry.py:1586-1594`) |
| 4 | `ap_mclachlan/geometry_eval.py:72-77` | dense H(t) per geometry evaluation; infeasible above about 13 qubits |
| 5 | `run_lock.py:121-126` | physics fingerprint binds seed bytes, family and `n_ph_max`; the integrals are bound only through a path string |

**Smallest change (decided).** Undriven admission needs no source change:
an explicit `settings.problem`, `settings.molecular_problem_json`, and
`replay_candidate_pool_mode = "family_pool"` in the seed. Driven admission
needs a new drive kind (site 3), registered in `_MOLECULAR_CAPABILITIES`; the
H2 vibronic dH/dR kind (`drive_terms.py:711-755`) is the template for
reading an operator from `runtime_data`. Scale needs a matrix-free H(t)ψ
(site 4) as a named option with dense as the default. Content binding needs
an integral digest added to the run-lock physics only when present, so every
existing fingerprint is unchanged (site 5).

**Regression pin.** `test/test_ap_mclachlan_route_parity.py` (decisions,
support evolution, energies) unchanged; at `0de9685a` the batch
`test_ap_mclachlan_route_parity.py test_ap_mclachlan_state_hamiltonian.py
test_realtime_drive_terms.py test_time_dynamics_run_lock.py
test_scaffold_runtime_loader.py` gives 69 passed, 5 failed in 37.6 s; all
five failures are in `docs/reports/FULL_SUITE_BASELINE_20260831_failed.txt`
(`:575-577`, `:591-592`).

### 3.2 Method assumptions about the Hamiltonian

| Assumption | Code site | Verdict |
|---|---|---|
| Generalized exchange: patch pair (D, I) on one frozen ray, admissible faces, realized-L² acceptance | `pipelines/time_dynamics/generalized_exchange.py:29` | **Unchanged** (ansatz- and Hamiltonian-independent). |
| Tangent metric K, force f, V, captured drift `Q = 2fᵀv − vᵀKv` | `geometry_eval.py:63-79,172`; `inverse.py:81,137,218` | **Unchanged mathematically.** H enters only through b̄ = −i(H(t)−E)ψ; the dense product at `geometry_eval.py:72-77` is the **new implementation** needed for scale. Ridge `1e-7` acts on K, which is dimensionless, so it carries over. |
| Coordinates: one per Pauli word by default | `ap_append_from_adapt_artifact.py:155`; `state.py:832-860` | **Unchanged.** Seed parity holds for UCCSD (decision 0.3). Each per-word coordinate leaves the (N_alpha, N_beta) sector at nonzero angle — the same structure as the HH fermionic words — so sector leakage must be reported per step. The alternative, one coordinate per excitation (`logical_shared`), is admitted by the structural cache for `termwise_product` terms (`structural_cache.py:72-79,113-120`) and needs `theta_logical` in the seed (`state.py:915-933`). Which coordinates Paper II uses is a GPT A2 question. |
| Drive `H(t) = H_static + c(t)·D`, one component | `ap_mclachlan/hamiltonian.py:22-121`; `drive_terms.py:758-809` | **New implementation**: a dipole drive kind from input item 14. An undriven run needs a nonstationary prepared state. |
| Insertion/deletion cost: Paper-I proxy denominators, shot component zero | `exchange_integration.py:30-31,153-160`; `append_cost.py:13-26,202,217` | **Unchanged.** |
| Motion controls `|dt|·sqrt(vᵀKv)`, `|dt|·max|v_j|` | `fixed_step.py:436` | **Parameter**: time grid, cuts and drive values in atomic units (input item 15). |
| Seed contract and run lock | sites 1, 2, 5 above | **Parameter** (seed fields) and **small implementation** (digest). |

---

## 4. Paper III — cost-weighted operator-selected QSE (`agent_guidance/qse/`)

### 4.1 Admissibility seam

**Where the problem enters today.** The static cell resolves a declared
problem kind in `_resolve_problem` (`pipelines/exact_bench/paper_iii_static_qse_cell.py:178-202`, dispatch `:184-193`):
`siam` -> `load_siam_problem` (`pipelines/qse_spectra/siam_static_adapter.py:547-638`),
`hh` -> `load_problem` (`pipelines/qse_spectra/paper_iii_problem.py:170-260`),
anything else raises `unknown problem kind` (`:193`). A third family already
has a provider outside the cell: Peierls (`pipelines/qse_spectra/peierls_static_adapter.py:197`).

**What a molecular provider must return** (the duck type the cell uses,
`SIAMStaticProblem` at `siam_static_adapter.py:458`): `hamiltonian`, ordered
`basis` of `QSEBasisElement` (`core.py:45-65`, Pauli strings or polynomials),
`costs` and `resources` from `annotate_basis_with_compiled_costs`
(`compiled_costs.py:169-215`, preset `two_qubit_only_v1`, `:55-72`),
`ground` plus `ground_provenance` (imported artifact or diagnostic fixture,
mutually exclusive, `siam_static_adapter.py:571-595`), `spectrum`,
`pool_digest`, and `num_qubits`; `assert_uniform_granularity`
(`paper_iii_problem.py:135-150`) applied.

**Smallest change (decided).** A molecular provider module beside the SIAM
and Peierls adapters, one `kind == "molecular"` branch in `_resolve_problem`,
and a molecular physics key beside `hh_physics_key` and
`siam_physics_key_typed` (`pipelines/qse_spectra/static_campaign.py:222-251`)
whose `case_id` carries the integral content digest (the key's `parameters`
are floats, `:200`, and cannot hold a digest). Existing HH and SIAM keys and
digests are untouched.

**Regression pin.** `test/test_paper_iii_static_qse_cell.py
test_paper_iii_siam_static_adapter.py test_paper_iii_static_campaign.py`:
51 passed in 7.3 s at `0de9685a`.

### 4.2 Method assumptions about the Hamiltonian

| Assumption | Code site | Verdict |
|---|---|---|
| QSE basis from a declared ordered operator pool (the campaign plan's "operator-selected Krylov space") | HH `paper_iii_problem.py:200`; SIAM `siam_static_adapter.py:171` | **New implementation**: a molecular pool. The QSE core is unchanged. The pool's content (neutral number-conserving excitations, spin adaptation) is GPT A2's. |
| Cost-weighted selection: full-pencil Ky Fan gain over the lowest R roots, novelty floor `1e-12`, no additive reward, gain divided by `max(0.05, K_i/K_max)^α` | `pipelines/qse_spectra/selection_policy.py:16-30`; `record_selection.py:769,845` | **Unchanged.** |
| Record cost K_i: price of inserting exp(−iθB_i) at the circuit tail (Marrakesh graph-span) | `compiled_costs.py:81,169-215` | **Unchanged if every B_i is a Hermitian Pauli polynomial.** A non-Hermitian a†_p a_q record has no defined price under this oracle; supply Hermitian combinations. |
| Reference projection Q0, raw projected vectors | `paper_iii_static_qse_cell.py:307-312`; `core.py:663-680` | **Unchanged.** |
| Sector: only identity projection; `sector_label` is metadata | `core.py` (Codex report §5) | **Unchanged** for number-conserving B_i on a fixed-sector root state; Green functions need input item 20. |
| Target window: lowest R roots, ascending index matching | `cell.target_roots`; growth-trace matching (Codex §5) | **Parameter** R (input item 17). S_z-conserving operators reach singlet and triplet (M_S = 0) states, so the lowest R roots can mix multiplicities, and spatial degeneracy breaks index matching; input item 18 or spin-adapted records are needed before reporting matched roots. |
| Reference spectrum (reporting) | dense, `paper_iii_problem.py:233-235` | **New implementation above 12 qubits** (sector-sparse), or the 12-qubit cap. |
| Completion certificate `Q0(H − ε_ν)Ψ_ν` | `method_completion.py:127-129` | **Unchanged**; Codex's qualification for approximate seeds stands. |

---

## 5. Review of the Codex groundwork report

### 5.1 Candidate sites: agree, correct, add

| Codex item | Verdict | Evidence |
|---|---|---|
| `ProblemRequest` -> resolver -> `ResolvedProblemContext` is the data boundary (§3) | **Agree.** | `pipelines/contracts/problem.py:19,156`; `problem_registry.py:804` |
| Facade "rejects non-HH" vs source admits H2O (§3) | **Agree, and wider.** Source also admits HH L=3 (named Page-12 application) and Hubbard L=2 and L=4 (named applications), `engine.py:5743-5791,5816-5831`. `agent_guidance/static-adapt/AGENTS.md:79-80` stays true for the ordinary request; `agent_guidance/molecular-vibronic/AGENTS.md:22-25` ("Paper-IV execution … remains deferred") is stale against source. Authority order (root `AGENTS.md`, "Historical documentation never overrides current code, tests, typed settings"): source governs. Correction is Paper I increment 4. | |
| Private SR context override `_context.py:742` is "a guarded compatibility route, not the proposed public provider API" (§3) | **Correct.** `_resolve_execution_context` is the shared method's general admission rule for lane-owned applications (`:752-766`) and is what the molecular application should use, through a named adapter — not bypass. | impact: LOW, 3 callers (`run_ra_adapt`, `_execute_sr_snake`) |
| Grouped-generator limit in AP (§4) | **Correct the scope.** Real for H2O blocks and spin-adapted sums; absent for spin-orbital UCCSD (decision 0.3, measured). | `state.py:832-860`; `structural_cache.py:72-79,113-120` |
| Runtime loader defaults missing problem to HH; selected-only pool for parameterized artifacts (§4) | **Agree.** These are seed-contract requirements (§3.1 sites 1–2). | `runtime_loader.py:1379,1075-1081` |
| Run lock binds seed bytes, not upstream tensors (§4) | **Agree;** the seed stores a path (`runtime_loader.py:1381`). | `run_lock.py:121-126` |
| QSE docs vs locked selector (§5) | **Agree; source governs.** The executable lock `35d5fa99` (2026-08-28) post-dates the acquisition contract's last change `e246c5b7` (2026-08-27); the static cell uses the fixed-pool trace (`paper_iii_static_qse_cell.py:447`). Reconciling the Paper-III docs is that lane's task and does not block Program A. | `agent_guidance/qse/paper-iii-static-results-acquisition-contract.md:18-22`; `selection_policy.py:16-18` |
| QSE provider candidate: "model provider analogous to HH/SIAM" (§5) | **Agree; add** that Peierls is a third precedent (`peierls_static_adapter.py:197`) and that the cell dispatch is the only gate. | |
| **Add:** dense-matrix scale limits in Papers I and II | Not in the Codex report. Decision 0.1. | `geometry_eval.py:72`; `_context.py:781-787`; `vqe_latex_python_pairs.py:2293-2305` |
| **Add:** Paper I has five family touchpoints, not one | §2.1 table | |
| **Add:** missing scalar and HF energy default to 0.0 silently | input contract rows 6–7 | `psi4_adapter.py:122-123` |

### 5.2 Impact analysis (graph at `d1436ab1`, upstream, depth 1, tests excluded)

| Symbol (file) | Risk | Direct callers | Note |
|---|---|---|---|
| `run_ra_adapt` (`ra_adapt/engine.py`) | **CRITICAL** | 191; 50 processes, 20 modules | mostly `chtc/**/run_cell.py`/`worker.py`; edit only by adding a branch after existing ones |
| `build_resolved_ra_protocol` (engine) | **CRITICAL** | 85; 41 processes | same rule |
| `_repaired_route_contract` (engine) | **HIGH** | 9 (engine, `ra_adapt/bundles.py`, `pipelines/reporting/paper_i_run_summary.py`, a chtc arm) | |
| `_h2o_application_active` (engine) | LOW | 3, all in engine | not edited; pattern only |
| `_parent_pool_spec` (`ra_adapt/pools.py`) | LOW | 2 (`_parent_records`, `_build_guarded_single_pauli_pool`) | |
| `_require_supported_singleton_problem` (pools) | LOW | 3 | not edited |
| `_require_paper_i_problem` (pools) | LOW | 1 (`build_executable_macro_pool`) | not edited |
| `build_h2o_sector_complete_pauli_block_pool` (pools) | LOW | 1 | not edited |
| `H2OLinearFDSectorCompletePauliBlockCandidateAdapter` (adapters) | **CRITICAL** | 40 files import it | not edited; new sibling class only |
| `canonical_sr_snake_h2o_derivative_resolved_paper_i_v3_contract` | **HIGH** | 7 | not edited; template |
| `_resolve_execution_context` (`sr_snake/_context.py`) | LOW | 3 | not edited |
| `resolve_problem_context` (`problem_registry.py`) | **CRITICAL** | 40 files, 174 impacted | not edited |
| `_resolve_molecular_problem_context` | **UNKNOWN** | 0 resolved | confirmed by grep: reached only through `_context_resolver=` at `problem_registry.py:1745`, dispatched by `_problem_family_spec_resolve` (`:149-155`); its real upstream is `resolve_problem_context` (CRITICAL). Not edited |
| `load_restricted_closed_shell_problem_from_json` (`psi4_adapter.py`) | **HIGH** | 4 | a required-field check here changes H2O and H2 vibronic producers too — do it in the new application, not the loader |
| `build_restricted_closed_shell_molecular_hamiltonian` | **HIGH** | 7 (H2O and H2 vibronic producers) | not edited |
| `build_molecular_uccsd_pool` | **HIGH** | 7 | not edited |
| `hartree_fock_statevector` | **HIGH** | 5 | not edited |
| `exact_ground_energy_sector` (`vqe_latex_python_pairs.py`) | **HIGH** | 5 (incl. pure-Hubbard fixed VQE) | not edited; add a sparse sibling |
| `_exact_gs_energy_for_problem` (`problem_setup.py`) | MEDIUM | 5 | molecular branch only |
| `resolve_fixed_count_qubit_groups` | LOW | 3 | not edited |
| `_load_generic_runtime_input_from_payload` (`runtime_loader.py`) | LOW | 1 | not edited |
| `_problem_request_from_payload` | LOW | 4 | not edited |
| `resolve_realtime_drive_model` (`drive_terms.py`) | **HIGH** | 3 (`hamiltonian.py`, `fixed_vqe_conditioning.py`, benchmarks `common.py`) | one new branch before the final raise |
| `time_dependent_hamiltonian_from_runtime_input` | **CRITICAL** | 11 | not edited |
| `TimeDependentHamiltonian` (`hamiltonian.py`) | **HIGH** | 20 files | additive method only |
| `evaluate_mclachlan_geometry` (`geometry_eval.py`) | **HIGH** | 9 (trajectory RHS, certification, AVQDS, parity probe) | dense default must stay bitwise |
| `state_from_scaffold_runtime_input` | **CRITICAL** | 14 | not edited |
| `_terms_for_ap_parameterization` | LOW | 1 | not edited |
| `build_run_lock` | LOW | 1 | key added only when present |
| `_resolve_problem` (`paper_iii_static_qse_cell.py`) | LOW | 4 (stage functions) | one new branch |
| `load_siam_problem` | LOW | 3 | template only |
| `StaticQSEPhysicsKey` (`static_campaign.py`) | MEDIUM | 8 | not edited; new constructor function beside it |
| `assert_uniform_granularity` | LOW | 3 | not edited |
| `annotate_basis_with_compiled_costs` | **CRITICAL** | 31; 4 processes | not edited |
| `build_qse_matrices` | LOW | 1 | not edited |

HIGH and CRITICAL symbols above are either untouched or receive only an
additive branch placed after the existing ones; each handoff names the test
that would catch a change to the existing branches.

### 5.3 The three baseline failures

Reproduced at `0de9685a`: `3 failed in 2.83 s` (command in the Codex report
§7). The Codex coordinator also reproduced and documented them (report §7,
"Reproducible baseline failures"). Verdict:

| Failure | Recorded | Cause | Bearing on Program A |
|---|---|---|---|
| `test/test_ra_adapt_facade.py:2009` (frozen ordinary-v1 plateau route digest) | `docs/reports/FULL_SUITE_BASELINE_20260831_failed.txt:547`; classification row 6, "Pinned route/protocol digests & profiles stale", fix path a user decision (re-pin vs restore byte layout) | expected digest set at `6442fbb5` and never re-pinned; later root-settings edits (for example `a33a2af9` deleting `phase3_shadow_damping_policy`) move it | none on the molecular path; the HH pin must be the new literal digests in §2.1, not this test |
| `test/test_scaffold_runtime_loader.py:645` (spin-boson replay expects a complete pool) | `:592`; classification rows 16–17, incomplete graft (`e0e6d6cd` grafted the test, not the loader block) | generic replay of a parameterized artifact returns `selected_terms_only` (`runtime_loader.py:1075-1081`) | **material**: a molecular seed takes this exact path; Paper II handoff requires `replay_candidate_pool_mode` |
| `test/test_scaffold_runtime_loader.py:679` (message regex) | `:591`; row 16 | the invalid state is still rejected; only the message changed | none |

Additional pre-existing reds seen while taking the lane baselines, all in the
same record: `test_hamiltonian_adapter_capabilities.py` ×2 (`:356-357`;
missing fixture `test_support/molecular_vibronic_h2_sto3g_nph1_binary.json`),
`test_realtime_drive_terms.py` ×3 (`:575-577`).

---

## 6. Codex handoffs (form of `agent_guidance/shared/agent-handoff-contract.md`)

### 6.0 Shared to all three

- **Checkout.** New worktree per lane from `0de9685a` (or a later `main` with
  an empty `git diff 0de9685a -- src pipelines test` outside `paper_5/`),
  under `/Users/lordskyler/Holstein_test.worktrees/`; branches
  `codex/chemistry-paper-i-admissibility-20261010`,
  `codex/chemistry-paper-ii-admissibility-20261010`,
  `codex/chemistry-paper-iii-admissibility-20261010`. Stage explicit paths
  only.
- **Python.** `/Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q`.
- **Fixture.** The in-repository H2 STO-3G closed-shell JSON writer
  `_write_molecular_problem_json` (`test/test_static_adapt_builders.py:88-125`,
  2 spatial orbitals, 4 qubits); for 8- and 12-qubit checks build random real
  integrals with the 8-fold symmetry in the test itself. No upstream code, no
  Psi4, no molecule choice.
- **Scope out.** Manuscripts; evidence, run registries and campaign modules
  (`pipelines/time_dynamics/paper_ii_runs.py`, `campaigns/`); CHTC; Paper-IV
  H2O application and its tests (pattern only); `src/quantum/**` operator core
  (`src/quantum/AGENTS.md:33`); `psi4_adapter.py` loader (HIGH, shared with
  H2O/H2 producers). If the work appears to need any of them, stop and report.
- **Machine.** 24 GB; run tests directly, no RAM guard
  (`agent_guidance/shared/memory-budget.md`). Nothing above 12 qubits.
- **Evidence standards.** Impact analysis before each function edit
  (`node /Users/lordskyler/Holstein_test/.gitnexus/run.cjs impact "<symbol>"
  --direction upstream --repo /Users/lordskyler/Holstein_test --file <path>`);
  `detect_changes` before each commit (worktree is unindexed: zero means
  unseen); grep to confirm UNKNOWN. Every new behaviour is a named option or a
  named application with a default-path test.
- **Traps.** (1) A molecular `ProblemRequest` still needs the lattice fields
  (`t, u, dv, omega0, g_ep, n_ph_max`); pass placeholders as the H2O launcher
  does (`ra_adapt/h2o_application.py:106-125`) with `n_ph_max=0` and
  `include_zero_point` irrelevant to the electronic family. (2) `num_sites` must equal the
  number of spatial orbitals. (3) A missing `nuclear_repulsion_energy` loads
  as 0.0 without error. (4) Exact energies are reporting inputs; never use one
  in a controller decision or as a stop unless the user names the exact-ED
  stop. (5) The facade digest test and the two loader tests are red at
  baseline; do not "fix" them in these increments.
- **Autonomy.** Proceed through all increments of a lane without pausing;
  commit at each increment boundary; stop on any stop condition.
- **Report back.** One block per increment:

```markdown
## Increment N — <goal>
Status: done | blocked | skipped
Commands run: <exact>
Result: <numbers, decisions, diffs>
Verification: <test counts, before/after values>
Deviations: <anything done differently, and why>
```

### 6.1 Paper I — named molecular application (first)

**Baseline.** `test/test_ra_adapt_refactor_parity.py`: 2 passed (9.4 s).
`test/test_ra_adapt_h2o_application.py test/test_hamiltonian_adapter_capabilities.py test/test_static_adapt_builders.py`:
62 passed, 2 failed (`FULL_SUITE_BASELINE_20260831_failed.txt:356-357`), 72.8 s.
Facade suite: one known red (`:547`).

**Increment 1 — pin today's behaviour (tests only, under 15 minutes).**
- Goal: the Hubbard--Holstein protocol identity, today's molecular rejection,
  and the UCCSD block structure are pinned before any source edit.
- Work: new `test/test_ra_adapt_problem_family_admissibility.py` with
  (a) the three literal values of §2.1 for the ordinary HH L=2 request;
  (b) `run_ra_adapt(molecular_problem, None)` raises `ValueError` matching
  `locked to the canonical Hubbard--Holstein L=2 problem`;
  (c) for UCCSD pools at (m, electrons) = (4, (2,2)) and (6, (3,3)): 26 and 117
  generators, zero noncommuting word pairs inside any generator, one X/Y flip
  support per generator, every `execution_mode == "termwise_product"`.
- Command: `pytest -q test/test_ra_adapt_problem_family_admissibility.py test/test_ra_adapt_refactor_parity.py`
- Expected: all new tests pass; parity 2 passed.
- Stop: a digest differs from §2.1 at the starting commit -> stop and report
  the starting SHA and both values (the start differs or the protocol is not
  deterministic); do not re-pin.

**Increment 2 — admit the family (about 20 minutes).**
- Goal: `run_ra_adapt` executes `molecular_restricted_closed_shell` only
  through the named macro application; every increment-1 pin is unchanged.
- Work, in this order, impact analysis first on each symbol: the predicate
  module; the named adapter (`ra_adapt/adapters.py`) and its fail-closed
  sector-audited `uccsd` generator pool (`ra_adapt/pools.py`, pattern
  `:1040-1100`); the molecular branch in `_parent_pool_spec` placed before its
  `_require_supported_singleton_problem` call; the parent route contract
  (`sr_snake_route_profile.py`, pattern `:3363-3409`); branches in
  `run_ra_adapt` (`engine.py:5742-5755`, policy check beside `:5801`),
  `_repaired_route_contract` (beside `:1461`, `:1673`, `:1959`) and
  `build_resolved_ra_protocol` (beside `:2680`). Gradient tolerance in Hartree
  is a required field of the application request, no default.
- Tests: (a) H2 STO-3G named application, 2 controller rounds, fresh start:
  completes, accepted energies non-increasing, final accepted energy ≥ the
  sector exact energy − 1e-10 (exact used only in the assertion);
  (b) `request=None` still rejected (increment-1 test unchanged);
  (c) `SinglePauliWordCandidateAdapter` on the molecular problem rejected;
  (d) the named adapter on the HH problem rejected.
- Command: increment-1 command plus `test/test_ra_adapt_h2o_application.py test/test_static_adapt_builders.py`.
- Expected: new tests pass; increment-1 pins identical; parity 2 passed;
  62 passed / 2 known reds unchanged.
- Stop: any pin or parity change -> revert the increment and report; if
  `_resolve_execution_context` rejects the application, report — do not edit
  `_context.py`.

**Increment 3 — reporting exact energy above 12 qubits (about 20 minutes).**
- Goal: the molecular exact target no longer builds a `2^n x 2^n` matrix.
- Work: a sector-restricted sparse ground-energy helper beside
  `exact_ground_energy_sector` (do not edit that HIGH-risk function); the
  molecular branch of `_exact_gs_energy_for_problem` (`problem_setup.py:629-635`)
  uses it only above 12 qubits, so every existing molecular result at ≤ 12
  qubits is bitwise unchanged.
- Tests: helper equals dense to 1e-10 at 4 and 8 qubits; a 16-qubit random
  closed-shell instance (sector dimension C(8,4)² = 4,900) resolves in under
  a minute without a dense matrix.
- Stop: disagreement above 1e-10 -> stop and report.

**Increment 4 — contract text (under 10 minutes).**
- Goal: `agent_guidance/molecular-vibronic/AGENTS.md:22-25` and
  `agent_guidance/static-adapt/AGENTS.md:79-80` state the named applications
  the facade actually admits, including the molecular one, with the commit
  SHA. Visible diff, no other wording changes.

### 6.2 Paper II — molecular seed, matrix-free geometry, content lock

**Baseline.** §3.1 batch: 69 passed, 5 failed (all recorded), 37.6 s.

**Increment 1 — characterize the molecular runtime path (tests only, under 15 minutes).**
- Goal: the AP path is shown to accept a molecular runtime input built in
  memory, and today's drive rejection is pinned.
- Work: new `test/test_ap_mclachlan_molecular_runtime.py` that builds a
  `ScaffoldRuntimeInput` directly (`contracts/scaffold.py:31`) for H2 STO-3G:
  resolved molecular problem; `psi_ref` = HF; selected terms = the UCCSD
  generators with small nonzero angles; `psi_initial` prepared by their
  product; candidate pool = full `uccsd` pool marked
  `CandidatePoolSource("resolved_pool", "uccsd", "complete")`. Assert:
  (a) `state_from_scaffold_runtime_input` reproduces `psi_initial` to 1e-10 in
  `per_pauli_term` and, with `theta_logical`, in `logical_shared`;
  (b) `time_dependent_hamiltonian_from_runtime_input(..., drive_config=...)`
  raises `no drive-term seam for family 'molecular_restricted_closed_shell'`;
  (c) an undriven `run_append_ap_mclachlan_from_runtime_input` over
  `times=(0.0, 0.05)` completes with finite energies and a run lock whose
  `family_key` is `molecular_restricted_closed_shell`. Report the energy
  change over that horizon and the largest per-step sector leakage
  (norm of the state outside (N_alpha, N_beta) = (1,1)) as numbers; no
  threshold is asserted on them.
- Expected: new tests pass; §3.1 batch unchanged.
- Stop: (a) fails -> stop and report the reconstruction error (contradicts
  decision 0.3).

**Increment 2 — matrix-free H(t)ψ as a named option (about 20 minutes).**
- Goal: geometry evaluation can apply H(t) through compiled Pauli actions
  (`src/quantum/compiled_polynomial.py:40,85`) instead of a dense matrix;
  dense remains the default.
- Work: an additive `TimeDependentHamiltonian` method applying
  `H_static ψ + c(t)·Dψ`, selected by a named provider option; the default
  branch at `geometry_eval.py:72-77` keeps the identical dense expression.
- Tests: b̄, K, f, V agree to 1e-12 between options on the HH L=2 n_ph_max=1
  seed and on H2; `test_ap_mclachlan_route_parity.py` unchanged; a 16-qubit
  random closed-shell runtime input completes one geometry evaluation with the
  new option and no dense matrix.
- Stop: route parity changes -> revert and report.

**Increment 3 — seed contract and content lock (about 20 minutes).**
- Goal: a molecular seed loads through the generic replay path with a complete
  pool, and the run lock binds the integrals' bytes.
- Work: a test seed payload with explicit `settings.problem`,
  `settings.molecular_problem_json`, a parameterization layout and
  `replay_candidate_pool_mode="family_pool"`; assert `can_structural_edit`.
  Add an integral-content digest to the run-lock physics only when the family
  carries a molecular JSON, so HH fingerprints are byte-identical.
- Tests: the seed loads complete; HH run-lock fingerprints in
  `test_time_dynamics_run_lock.py` unchanged; changing one integral changes the
  molecular fingerprint.

**Increment 4 — dipole drive kind.** Blocked until A1 confirms input item 14
and GPT A2 states the coupling sign and polarization convention. Template:
`drive_terms.py:711-755`; registration in `_MOLECULAR_CAPABILITIES`
(`problem_registry.py:1586-1594`); `A=0` must equal the undriven run
(`scientific-invariants.md:36-37`).

### 6.3 Paper III — molecular static provider

**Baseline.** §4.1 batch: 51 passed, 7.3 s.

**Increment 1 — provider (under 20 minutes).**
- Goal: a molecular provider returns the problem shape the static cell
  consumes.
- Work: `pipelines/qse_spectra/molecular_static_adapter.py` mirroring
  `load_siam_problem` (`siam_static_adapter.py:547-638`): Hamiltonian from the
  shared resolver; basis = `QSEBasisElement(kind="pauli_polynomial")` from the
  UCCSD generator polynomials (Hermitian; a mechanical choice to exercise the
  seam, not the scientific pool, which is GPT A2's); `assert_uniform_granularity`;
  costs via `annotate_basis_with_compiled_costs` with `two_qubit_only_v1`;
  ground from an imported artifact or the labelled diagnostic fixture
  (mutually exclusive); reporting spectrum by dense sector diagonalization,
  failing closed above 12 qubits; `pool_digest` over ordered names.
- Tests: on H2 STO-3G with the diagnostic ground, `compute_qse_spectra` under
  the Q0 / raw-projected policy (`paper_iii_static_qse_cell.py:307-312`)
  returns Ritz values each ≥ the matching exact excited eigenvalue − 1e-10
  (interlacing in the complement of the exact ground state).
- Expected: new tests pass; 51 passed unchanged.

**Increment 2 — cell admission and physics key (about 20 minutes).**
- Goal: the static cell accepts `kind == "molecular"`.
- Work: one branch in `_resolve_problem` (`paper_iii_static_qse_cell.py:184-193`);
  a molecular physics-key constructor beside `siam_physics_key_typed`
  (`static_campaign.py:234-251`) with the integral content digest in
  `case_id` and the sector written as SIAM does (`:247`).
- Tests: HH and SIAM physics-key digests unchanged (existing tests); the
  molecular cell runs its ground-state import and trace stages on H2.

**Increment 3 — observables and pool.** Blocked on A1 (dipole integrals) and
GPT A2 (neutral-excitation pool, spin adaptation, root matching).

---

## 7. Unresolved, with owner

| Item | Owner |
|---|---|
| Upstream repository, license, export format; whether it delivers real RHF active-space integrals (§1.1) and dipole integrals (§1.2) | coordinator / Codex A1 |
| Paper I: whether `S_alg` per evaluation remains the intended burden for a Hamiltonian with many more Pauli terms | GPT A2 |
| Paper II: per-word coordinates (sector leakage) or one coordinate per excitation (`logical_shared`); dipole coupling sign and polarization | GPT A2 |
| Paper III: molecular operator pool, spin adaptation, root matching under degeneracy | GPT A2 |
| Which molecules; any evidence adoption; whether a fourth manuscript is implied; who owns the resolved molecular problem | user |
| 12-qubit cap until §6.1 increment 3 and §6.2 increment 2 land | user (scope); recommended yes |

Files to edit (by Codex, per §6): `pipelines/static_adapt/ra_adapt/{engine.py,adapters.py,pools.py}`, a new `ra_adapt` predicate module, `pipelines/static_adapt/sr_snake_route_profile.py`, `pipelines/static_adapt/builders/problem_setup.py`, `pipelines/time_dynamics/ap_mclachlan/{hamiltonian.py,geometry_eval.py}`, `pipelines/time_dynamics/run_lock.py`, `pipelines/time_dynamics/adapters/drive_terms.py` (blocked increment), `pipelines/qse_spectra/molecular_static_adapter.py` (new), `pipelines/qse_spectra/static_campaign.py`, `pipelines/exact_bench/paper_iii_static_qse_cell.py`, two lane contracts, and new tests.
