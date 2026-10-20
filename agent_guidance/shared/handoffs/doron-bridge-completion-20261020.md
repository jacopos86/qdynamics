# Doron chemistry–quantum bridge — completion record (2026-10-20)

Thread `doron-chemistry-quantum-bridge-20261019`. On 2026-10-20 Jake retired the
external Codex/Claude review loop ("finish everything yourself"). This file is the
completion record: what now runs on Doron Wu's actual system through all three
methods, how each result was measured, what was verified and by whom, and what
remains a limit or an unselected scientific decision. Earlier records stay where they
are: `doron-chemistry-quantum-bridge-20261019.md` (original handoff),
`doron-bridge-corrections-20261019-002.md` (review cycles 002–004, §1–§10),
`doron-bridge-h2o-reproduction-20261019.md` (running his exporter).

## 0. Pins

| item | value |
|---|---|
| worktree | `/Users/lordskyler/Holstein_test.worktrees/claude-doron-bridge-20261019` |
| branch | `claude/doron-chemistry-quantum-bridge-20261019` (never merged into `main`, never pushed; `main` is `d1436ab1`) |
| last reviewed pin | `acffb79b` (packet 004 record; reviewer recheck-004 was requested and is moot) |
| completion commit | `01bf17ca` — RA-ADAPT named molecular application, Hermitian dynamics pool, sector-complete QSE bases, `ra` method key, CLI flags, tests |
| merge of the H2O reproduction | `c6dfcc48` (merges `claude/doron-h2o-export-20261019` at `ca38b3f2`, no code on that branch) |
| receipts + this record | appended in §9 after the commit |
| Doron's producer | `jacopos86/qdynamics` branch `doron-chemistry` commit `e5ec5bc45d02169e77ce93b05126c9804a03cdb6`, run here with pyscf 2.14.0 (origin `simulated`: his code, our run, his inferred geometry; his own bytes and declarations are still absent) |
| system | H2O/STO-3G, frozen-core CAS(8e,6o), 12 qubits, `n_up = n_dn = 4`, E_SCF = −74.96302344338558 Ha, exact sector E0 = −75.01250192475737 Ha |

## 1. Status per method on Doron's system (final receipts: `doron-bridge-h2o-20261019/receipts/complete_20261020/`)

| method (lane) | route | CPU | GPU (Apple MPS, float32) | result |
|---|---|---|---|---|
| mapping witness | tensor-level witnesses of the mapped operator | `cpu_verified` | `gpu_verified` | HF energy of the mapped operator equals the intake determinant energy |
| exact sector reference | 225-dimensional sector block, `eigvalsh` on CPU | `cpu_verified` | `gpu_verified` (float32 block, 4.2e-5 Ha from float64) | E0 = −75.0125019248 Ha |
| QSE (Paper III) | repository `compute_qse_spectra`; backend pencil parity | `cpu_verified` | `gpu_verified` | CISD 93: lowest Ritz +0.705 mHa above E0; CISDTQ 225 = sector: every root exact to 2.6e-13 Ha |
| ADAPT selection gradients (Paper I) | `resolve_pool_plan` full_meta + backend commutator gradients | `cpu_verified` | `gpu_verified` | top |g| `uccsd_dbl(ab:1,7->5,11)` 0.30498 |
| **RA-ADAPT/SNAKE construction (Paper I)** | `run_ra_adapt` through the named molecular application | `cpu_verified` | CPU-only route (not attempted) | 24 plateau rounds: E − E0 = 36.88 → 0.169 mHa; chemical accuracy at round 17 |
| **AP-McLachlan (Paper II)** | `run_append_mclachlan_trajectory`, exchange controller | `cpu_verified` | CPU-only route (not attempted) | trajectory moves: support 2 → 20 over t ∈ [0, 0.5], fidelity vs exact 0.9903 at t = 0.5, leakage 0 |

Exit policy unchanged: 0 all verified; 1 failed check or exception after intake; 2 usage; 3 skipped/blocked.

## 2. What changed and why (commit `01bf17ca`)

### 2.1 Paper I: the typed RA-ADAPT facade now executes the molecular family

Before: `run_ra_adapt` rejected every family except HH L=2/3, pure Hubbard via named
adapters, and the molecular-vibronic H2O lane; the bridge recorded `admitted=False`
and ran selection gradients only. A read-only reader (Opus) mapped the admission
chain and proved in scratch that a named adapter with `application_family_key` and
lane-owned pools runs to completion once the gate is widened.

Now, in the shape of the existing named applications (`paper_i_hubbard_l4.py`,
`h2o_application.py`):

* `pipelines/static_adapt/ra_adapt/molecular_closed_shell.py` —
  `MolecularClosedShellMacroCandidateAdapter` (fixed identity, `application_family_key =
  molecular_restricted_closed_shell`, `pool_key` `uccsd` by default; parent and executable
  inventories built from `resolve_pool_plan` + the repository sector audit so the
  executable labels equal the runtime labels, 92 = 92 on H2O with 0 removals),
  `require_molecular_closed_shell_problem` (fail-closed: family, blocked ordering, no
  bosons, closed-shell partially filled sector, strict-intake receipt present),
  `is_molecular_closed_shell_application` (singleton admission, plateau-commutation or
  append-only insertion, pruning off, beam off — the compositions that were probed),
  route overlays (executed `problem` = the molecular family, `phase3_backend_cost_mode =
  proxy` instead of the HH Marrakesh graph-span oracle, lane invariants, HH-only
  invariants dropped), `build_molecular_closed_shell_ra_request` (no `ExactEDStop` unless
  asked: the exact reference stays a reporting input).
* `engine.py` — additive branches gated on that adapter: the facade gate admits the
  family only with the named adapter (default request still raises "locked"), post-parse
  and `build_resolved_ra_protocol` fail closed, `_repaired_route_contract` applies the
  overlays and the profile prefix `doron_molecular_ra_adapt__`.
  `adapt_pipeline.py`: the route-profile gate accepts that prefix.
* bridge method key `ra` → receipt `ra_adapt_construction` (energies per round,
  operators and insertion positions, stop receipt, executed settings, lane invariants,
  integrity receipt). Checks: energies non-increasing from HF; final energy not below
  the exact sector ground energy; executed problem/pool/cost/profile name this lane.
  The horizon is capped at the executable pool size (on a 3-generator witness the
  state is exact after two rounds and the engine's canonical v2 candidate-gain receipt
  cannot close at round 4; requested and executed horizons are both recorded). An
  engine receipt-closure `RuntimeError` becomes a `failed` receipt, exit 1.
* CLI: `--ra-rounds` (default 8), `--ra-insertion`, `--ra-pool`; validated in phase 1.

### 2.2 Paper II: the AP-McLachlan trajectory moves

Why it never moved: from the real HF reference, every repository UCCSD generator
`i(T − T†)` has odd-`Y` words with real coefficients, so `exp(−iθG)` is real
orthogonal, the tangent is real, the McLachlan force `Im⟨∂ψ|H|ψ⟩` vanishes
identically, and θ̇ = 0; the only even-`Y` generators the audit admitted from
`full_meta` were diagonal `Z` words, which only add a phase. Measured: `|v| = 0`
exactly for all 92 UCCSD generators; the exchange controller then scored 0 gain for
every candidate it could see (first 8 in frozen pool order).

Now: `pipelines/chemistry_bridge/molecular_pools.py` offers the bridge-local pool
`hermitian_excitations`: Hermitian number-conserving singles `a†_a a_p + h.c.` (2
words) and doubles `T + T†` (8 words) over the UCCSD index set. Every word is even-`Y`
with a real coefficient (real symmetric block → complex exponential), the words of one
block mutually commute (shared-angle termwise product = exact exponential, preserves
`(n_up, n_dn)`; verified by the executed-unitary audit at run time, not assumed). The
pool is bridge-local on purpose: `resolve_pool_plan` and `build_molecular_uccsd_pool`
are CRITICAL in the impact graph (49 / 38 upstream dependents). Seed rule `auto` picks
the eligible generator of largest initial velocity `|Re⟨Gψ|(H − E)ψ⟩|`
(`auto_max_initial_velocity_v1`; class-rank fallback, recorded as such, when all
velocities vanish — the earlier `full_meta` receipts are reproduced that way), and
candidates are offered in decreasing velocity order because the controller scores only
the first `max_structural_pool_size` candidates in offered order. Receipts record the
velocity table, the dynamics-capable count, and the frozen-reference fidelity floor
`|⟨HF|exact(t)⟩|²`. CLI: `--apm-support-config` (JSON overrides of
`SupportPatchControllerConfig`, validated in phase 1), default `--apm-pool
hermitian_excitations`; repository keys are still accepted and the frozen behaviour is
pinned as a control in the tests.

Brillouin check on H2O: Hermitian singles have `|v| ≤ 3.6e-11` (canonical RHF
orbitals), Hermitian doubles up to 0.1525 (`herm_dbl(ab:1,7->5,11)`, the same orbital
quartet the ADAPT gradient ranks first).

### 2.3 Paper III: QSE bases that span the sector

`run_qse` basis kinds `uccsd_sdt_products` and `uccsd_sdtq_products`: singles and doubles
plus products of commuting UCCSD generators with disjoint occupied and disjoint
virtual spin-orbital sets (double × single = triple, double × double = quadruple), one
representative per excited determinant. Whether the pencil spans the fixed-count
sector is measured through its rank, never assumed. CLI `--qse-basis`. On H2O the
CISDTQ basis has 1 + 16 + 76 + 96 + 36 = 225 elements = the sector dimension and
reproduces every exact sector root; the CISD and CISDT Ritz values are upper bounds
(0.705 and 0.614 mHa above E0).

## 3. Measured on Doron's H2O (CPU float64 unless stated; all numbers from the receipts in §1)

RA-ADAPT, `uccsd` pool (92 generators, 0 audit removals), plateau-commutation
insertion, proxy cost, 24 rounds, 247 s:

| round | 1 | 2 | 4 | 8 | 12 | 16 | 17 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|---|
| E − E0 (mHa) | 36.878 | 33.596 | 26.109 | 15.380 | 6.296 | 2.303 | 1.460 | 0.846 | 0.169 |

First picks `uccsd_dbl(ab:1,7->5,11)`, `uccsd_dbl(ab:2,8->4,10)`; 24 accepted
operators; stop `maximum_controller_rounds`; HF energy equals the intake witness to
0.0; the exact reference was not passed to the controller.

AP-McLachlan, `hermitian_excitations`, `logical_shared`, rk4, default exchange
controller (`residual_ratio` gate 0.02, `max_structural_pool_size` 8), t = 0 … 0.5 step
0.025, seed `herm_dbl(ab:1,7->5,11)` (|v| 0.1525), candidates velocity-ordered:

| t | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| support | 6 | 10 | 14 | 18 | 20 |
| fidelity vs exact sector propagation | 0.99948 | 0.99844 | 0.99661 | 0.99432 | 0.99032 |
| frozen-reference floor `|⟨HF|exact(t)⟩|²` | 0.99902 | 0.99613 | 0.99149 | 0.98533 | 0.97796 |
| variational E − E0 (mHa) | 50.29 | 53.39 | 62.67 | 72.95 | 86.66 |

At t = 0.5 the moving ansatz leaves 0.439 of the frozen reference's infidelity
(`final_infidelity_ratio_vs_static_reference`); 30 of the 92 offered generators have a
non-zero initial velocity at HF (all 92 are dynamics-capable by construction).

Exact E − E0 = 49.48 mHa at every time (static H, zero drift); the variational energy
drifts by +37 mHa over the horizon — McLachlan on a real-parameter manifold does not
conserve energy, and this is reported, not hidden. Leakage out of the sector: 0.0.
Controls on the same system and horizon: the same controller with pool-order
candidates kept support 2 and reached fidelity 0.98187 at t = 0.5; the
`mclachlan_l2` / pool-128 configuration did not finish in 27 min and was stopped
(no receipt). Run time of the final receipt: 50 s for the trajectory, 66 s for the
exact reference, 252 s for the 24 RA rounds (CPU, numpy float64).

QSE: CISD 93 × 93 (rank 93) lowest Ritz +0.70496 mHa; CISDT 189 (rank 189) +0.614 mHa;
CISDTQ 225 (rank 225 = sector) |Ritz − exact| ≤ 2.6e-13 Ha for all 225 roots. MPS float32
pencil parity holds for both the 93- and the 225-element bases (`gpu_verified`; on MPS
the Ritz roots are the CPU core's and are compared against the float32 block's exact
roots, hence the recorded 4.2e-5 Ha gap there, inside the scaled float32 gate).

## 4. Verification (no external reviewer)

* Tests on the merged branch `c6dfcc48` (repository venv): `test_chemistry_bridge_{export,backend,methods,cli,gpu_parity}.py`
  + `test_ra_adapt_molecular_closed_shell_application.py` + `test_molecular_diagnostic_intake.py`
  — 206 passed, 18 skipped (torch absent there by design). GPU venv (`~/.venvs/doron-bridge-gpu`,
  torch 2.14.1 / h5py 3.16.0): the bridge and application suites — 137 passed, 9 skipped
  (MPS float64 refused by the device; CUDA absent).
* Existing RA-ADAPT lock / facade / named-lane suites (`test_ra_adapt_lock_*`,
  `test_ra_adapt_facade.py`, `test_ra_adapt_h2o_application.py`,
  `test_ra_adapt_pure_hubbard_noise_page12.py`, `test_ra_adapt_l3_page12_application.py`,
  `test_ra_adapt_pools_and_policies.py`, `test_ra_adapt_pool_selection.py`,
  `test_ra_adapt_physics_invariants.py`): 103 passed, 7 failed. The same 7 fail on the
  untouched `main` checkout `d1436ab1` (verified by running them there): singleton
  trajectory lock (`plateau_v1` vs `v2` profile), frozen ordinary v1 protocol, three
  pure-Hubbard noise page-12 tests, two L=3 page-12 tests. Pre-existing; not touched.
* GitNexus (worktree index rebuilt): `run_ra_adapt` / `_repaired_route_contract` /
  `build_resolved_ra_protocol` CRITICAL (541 / 545 / 591 upstream), `resolve_pool_plan`
  49, `build_molecular_uccsd_pool` 38 (both left untouched; the dynamics pool is
  bridge-local), `run_ap_mclachlan` / `resolve_molecular_pool` LOW;
  `_validated_ra_adapt_route_contract` not in the index (caller confirmed by text).
  `detect-changes --scope staged` before `01bf17ca`: 9 files, 97 symbols, risk high.
  Mitigation: every engine edit is an additive branch gated on the named adapter.
* Independent adversarial verification: four Opus refuters with fresh context, each
  told to break one claim with its own dense-numpy computation (Hermitian pool physics
  and audit; RA lane isolation incl. route-contract sha256 equality for the HH L=2 and
  H2O lanes between `main` and this branch; QSE product bases and interlacing; the RA
  construction's energies recomputed from the receipt's operators and parameters).
  Verdicts in §9. This is not the Codex 6.1 Ultra check; it is what replaces it.

## 5. Limits and unselected decisions (honest)

* Doron's own export bytes and declarations are still absent; everything is origin
  `simulated` (his exporter at `e5ec5bc4`, our run, his inferred geometry). Real-data
  acceptance stays blocked on a pinned file from him.
* Pool choice for RA-ADAPT (`uccsd`; `full_meta` resolves to 170 executable generators
  and was not run end to end), the plateau/shortlist thresholds inherited from the
  HH-calibrated macro parent route, and the proxy Phase-III cost are lane choices made
  here, not validated for molecules.
* AP-McLachlan: the pool (`hermitian_excitations`), seed rule, candidate order and the
  controller configuration are bridge choices; the energy drift of the variational
  trajectory is real; no drive (the family declares none).
* GPU: Apple MPS float32 only; no CUDA, no float64 on MPS; RA-ADAPT and AP-McLachlan
  are CPU-only routes.
* Intake scope cap 12 qubits; the full-space H2O (14 qubits) stays out of scope.
* Seven pre-existing RA-ADAPT lock/lane test failures on `main` are not addressed.

## 6. Reproduce

```sh
cd /Users/lordskyler/Holstein_test.worktrees/claude-doron-bridge-20261019
export PYTHONPATH=. STATIC_ADAPT_HH_POOL_CACHE=off PYTHONDONTWRITEBYTECODE=1 PYDEPHASING_TESTING=1
M=agent_guidance/shared/handoffs/doron-bridge-h2o-20261019/h2o_cas8e6o_problem.manifest.json
# all six methods on CPU (24 RA rounds ~4 min, dynamics to t=0.5 ~1 min)
/Users/lordskyler/Holstein_test/.venv/bin/python -m pipelines.chemistry_bridge run --manifest $M --out-dir /tmp/doron_cpu --methods mapping,exact,qse,adapt,ra,apm --ra-rounds 24 --apm-times 0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5
# sector-complete QSE
/Users/lordskyler/Holstein_test/.venv/bin/python -m pipelines.chemistry_bridge run --manifest $M --out-dir /tmp/doron_qse --methods exact,qse --qse-basis uccsd_sdtq_products
# GPU parity (Apple MPS)
/Users/lordskyler/.venvs/doron-bridge-gpu/bin/python -m pipelines.chemistry_bridge run --manifest $M --out-dir /tmp/doron_mps --backend torch:mps:float32 --methods mapping,exact,qse,adapt
# tests
/Users/lordskyler/Holstein_test/.venv/bin/python -m pytest -q -p no:randomly test/test_chemistry_bridge_export.py test/test_chemistry_bridge_backend.py test/test_chemistry_bridge_methods.py test/test_chemistry_bridge_cli.py test/test_chemistry_bridge_gpu_parity.py test/test_ra_adapt_molecular_closed_shell_application.py
```

## 7. Independent adversarial verification — verdicts and what they changed

Four Opus refuters (fresh context, read-only, own dense-numpy recomputation, told to
default to "refuted"); transcript `subagents/workflows/wf_947618d7-1d0`.

| claim | verdict | what the refuter established | discrepancies and the fix |
|---|---|---|---|
| Hermitian dynamics pool | confirmed | every block real symmetric, words even-Y and pairwise commuting (0 anticommuting pairs on random6 and on all 92 H2O blocks); termwise product = `expm` to 8.9e-16 / 4.4e-16; out-of-sector norm ≤ 8.3e-17; UCCSD velocities exactly 0.0 for 8/8 and 92/92; Hermitian doubles non-zero (H2O top 0.152491, `herm_dbl(ab:1,7->5,11)`, argmax unique); Hermitian singles ≤ 3.62e-11 on H2O; audit passes 92/92 and 8/8 | docstring over-claimed that the run-time audit verifies product exactness — it verifies fixed-count preservation; wording corrected (`molecular_pools.py`). Brillouin holds only for canonical HF orbitals: on random tensors Hermitian singles have non-zero velocity (stated now). |
| QSE product bases | confirmed | random8: 27/35/36 elements and ranks, CISDTQ reproduces all 36 sector roots (4.4e-14), H2O all 225 (5.7e-13), CISD 0.704959 mHa, CISDT 0.61438 mHa; product elements are single, distinct, in-sector determinants; interlacing holds to roundoff (−1.4e-13 worst) | the regression compared only the lowest 8 roots while its comment said "every root"; it now asks the exact reference for all 36 and asserts `exact_roots_compared == 36`. The receipts on H2O compare the lowest 16 (2.6e-13); the all-225 statement rests on the refuter's recomputation. |
| RA construction receipt | confirmed | final energy recomputed from the receipt's operators and parameters with dense `expm`: 4.4e-16 from the receipt (random6, 4 rounds), 0.0 (8 rounds), 1.3e-15 (contract4); every generator equals an independently built `i(T − T†)`; stop receipt carries no exact target; integrity receipt reporting-only, controller influence false; the inherited HH thresholds listed (`plateau_prior_mean_decrease_ratio_threshold` 1e-4, `physical_lane_shortlist_aggressiveness` 3, shortlist sizes 24/12, …) | `route_contract_sha256` was empty (the contract has no such key) — now computed with the engine's `_route_sha256`; the JSON stop receipt drops `None` fields, so `exact_reference_used_online` is now an explicit boolean; the horizon cap is a bridge-side workaround, not a natural limit (generators are re-admitted at later rounds) — comment and this record say so; contract4 is exact after round 1, not 2. |
| RA lane isolation | partially confirmed | route contracts and their sha256 for the canonical HH L=2 request and for the H2O lane are bitwise identical between untouched `main` and this branch (`6ed3f6a4…` for H2O; the only protocol-level difference in a first run was the fixture's absolute path in the test setup, not code); default / foreign requests on a molecular context raise messages containing "locked"; executed molecular route names the family, pool `uccsd`, cost `proxy`, lane invariants | (i) the pipeline route-profile gate accepted any contract with the molecular prefix without a family cross-check — it now cross-checks the resolved problem family, executed problem, proxy cost, admissible executed pool, and the lane invariants (`adapt_pipeline.py`, same shape as the H2O cross-check, with the failing condition named in the error); (ii) the second molecular gate in `run_ra_adapt` was unreachable — removed; (iii) two ungated imports (engine module-level, pipeline function-local) have no behavioural effect (measured identical); (iv) `phase3_backend_name = FakeMarrakesh` is inherited although the cost mode is `proxy` — same as the H2O lane; left as is and stated here. |

After these fixes the application, bridge and CLI RA tests pass again; the final test
counts and the lock-suite rerun are in §9. The CPU receipt was regenerated with the fixed
code so the committed receipt carries the route-contract hash and the explicit flag.

## 8. Test counts after the refuter-driven fixes

| suite | result |
|---|---|
| repository venv: bridge export/backend/methods/cli/gpu-parity + application + strict intake | 206 passed, 18 skipped |
| GPU venv (torch 2.14.1, h5py 3.16.0): bridge suites + application | 137 passed, 9 skipped |
| RA-ADAPT lock / facade / named-lane set (10 files) | 103 passed, 7 failed — identical set fails on untouched `main` `d1436ab1` |

## 9. Record (appended after the commits)

| item | value |
|---|---|
| completion code | `01bf17ca` |
| merge of the H2O reproduction branch | `c6dfcc48` |
| first receipts + this handoff | `a53cec52` |
| refuter-driven fixes + regenerated CPU receipt + §7/§8 | `af04ec7fb2e31eaac6a9c2fecb6eebe4f95decff` |
| this record | the next commit |
| `main` | `d1436ab1`, untouched; branch not merged, not pushed |
| final CPU receipt clock | 2026-10-20T09:02:25Z – 09:08:21Z (mapping 0.8 s, exact 66 s, QSE 1.3 s, ADAPT gradients, RA 24 rounds 252 s, APM 21 checkpoints 50 s) |
| session clock | work began 2026-10-20T07:5xZ after Jake's instruction; fixes committed 2026-10-20T09:13:46Z |
