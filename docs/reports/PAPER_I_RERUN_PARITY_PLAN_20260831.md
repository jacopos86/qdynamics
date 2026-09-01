# Paper I Rerun-Parity Plan — weak-HH + pure-Hubbard producers — 2026-08-31

> **Scope amendment (user directive, 2026-08-31, supersedes §2's full-horizon
> schedule):** weak-HH parity runs are capped at **k=20** — sufficient because
> every weak-cell k\* ≤ 11 and `kstar_tables.json` `spot_check` records
> `identical_ordering: true` (truncated replay == full-run prefix), and the
> saved per-iteration evidence exists on main (`b3` per-round `[k, dE]`
> traces; `aavqe_full_hist_tolmatch_9665004.json`). Executed scope = §1.2
> Tier-0 screens at k=20 for all 21 weak cells + Template D (L=3 Hubbard
> noise, Table II). ALL k=50/75 full-horizon rows (§1.3) are deferred until
> the user schedules them (or the CHTC fetch supersedes the need). Cells
> whose comparison would require rounds >20 defer to post-fetch.

Goal: RE-EXECUTE the Paper-I producers locally and check they reproduce the published numbers.
Scope (user): weak-Holstein-sector Hubbard–Holstein cells + pure-Hubbard runs. This plan is the
execution contract; it complements (does not redo) `docs/reports/EVIDENCE_PARITY_AUDIT_20260831.md`
§2.1, which already verified the STORED data matches the manuscript.

Anchors
- Manuscript: `/Users/lordskyler/ADAPT---Paper-I/Paper_I_overleaf_20260818/Paper_I.tex`
- Evidence: `chtc/paper_i_ra_adapt_repair_20260727/results_tooling/kstar_tables.json`
  sha256 `060e7524…cada0c` (digest-matched to Fig 2 provenance)
- Corrections: `/Users/lordskyler/ADAPT---Paper-I/Paper_I_overleaf_20260818/figures/paper_i_hh_hierarchical_route_appendix_20260823/{b3_phase_reuse_corrections_20260821.json, hh_flat_vs_hierarchical_comparison_tables.json}`
  and `/Users/lordskyler/ADAPT---Paper-I/Paper_I_overleaf_20260818/figures/ra_optional_controls_main/estimator_reuse_corrections_20260821.json`
- Golden: `agent_guidance/static-adapt/golden/` (MANIFEST 10/10 gate)
- ED reference: `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_ed_cutoff_reference_six_regime_20260727.json`
  (E_ED weak_weak M=3 = -0.9183809199948856; L=3 Hubbard E_ED = -2.1414935642714075)

CRITICAL LINEAGE RULING (probe-verified 2026-08-31)
- main @ dafbda5e is NOT the evidence lineage for the HH cells: route contract v2 vs archived v4
  (missing cost_backend/phase3_shadow_damping fields), builder no longer accepts
  `compile_cost_policy`, `appendix_demos` module absent, sealed-worker source-inventory sha drifted
  (live ff2c6c96… vs recorded 6f43505e…) so the CHTC worker replay fails closed.
- All HH reruns MUST run from a git-archive export of `origin/paper-i-integration-20260831`
  (B3-MR protocol rematerializes BIT-EXACT, sha `19bff49dce0f…`; B3 flat protocols identical except
  source-lock/implementation-inventory hashes — record both shas per run, §6.3 code-lineage caveat).
- The L=3 noise campaign MUST run from a git-archive export of `c2cd9379`
  ("preserve Paper I appendix worktree state"); it is rejected on main AND on the integration branch.
- Governance: the executing session loads the `paper-i-run` skill before launching
  (MATH/AGENTS.md:92–95 run-skill gate). Reruns are parity diagnostics only — no promotion language,
  no manuscript/table edits, git stays read-only, all outputs under `$SCRATCH` (never repo paths).

---

## 1. Run matrix

### 1.0 Setup (S0) and pre-flight gates (G1–G3) — run once, ~2 min total

```bash
REPO=/Users/lordskyler/Holstein_test
PY="$REPO/.venv/bin/python"
RG="$REPO/pipelines/shell/ram_guard.py"
SCRATCH=<any non-repo scratch dir>          # e.g. the session scratchpad
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
       PYTHONHASHSEED=0 STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off STATIC_ADAPT_HH_POOL_CACHE=off

mkdir -p "$SCRATCH/pi_integration" "$SCRATCH/appendix_demos_tree"
git -C "$REPO" archive origin/paper-i-integration-20260831 -- pipelines src docs | tar -x -C "$SCRATCH/pi_integration"
git -C "$REPO" archive c2cd9379 -- pipelines src docs chtc/ra_adapt_appendix_noise_20260821_v1 | tar -x -C "$SCRATCH/appendix_demos_tree"
```

| Gate | Command | Must show |
|---|---|---|
| G1 versions | `"$PY" -c "import qiskit,numpy,scipy;print(qiskit.__version__,numpy.__version__,scipy.__version__)"` | `2.3.1 2.3.5 1.16.3` (== requirements pins; else tuple mismatches become ENVIRONMENT findings, not physics FAILs) |
| G2 golden | `cd "$REPO/agent_guidance/static-adapt/golden" && shasum -a 256 -c MANIFEST.sha256` | 10/10 OK (must run FROM the golden dir) |
| G3 anchor | `shasum -a 256 "$REPO/chtc/paper_i_ra_adapt_repair_20260727/results_tooling/kstar_tables.json"` | `060e7524…cada0c` |

Re-run G2+G3 AFTER all reruns (proves reruns touched no stored inputs).

### 1.1 Command templates (all ram_guard-wrapped; env from S0 is inherited)

**Template A — AVQE comparator (fresh full-history to r75)**, cwd `$SCRATCH/pi_integration`:

```bash
cd "$SCRATCH/pi_integration" && REGIME=<regime> "$PY" "$RG" --limit-mb 4000 -- "$PY" -c "
import os,sys,json; sys.path.insert(0,os.getcwd())
from pipelines.static_adapt.ra_adapt import run_append_adapt, build_paper_i_ra_hh_regime_problem
from pipelines.static_adapt.ra_adapt.contracts import AppendAdaptRequest
from pipelines.static_adapt.sr_snake.contracts import SRExecutionPolicy, SRStopPolicy
R=os.environ['REGIME']; p=build_paper_i_ra_hh_regime_problem(R)
r=run_append_adapt(p, AppendAdaptRequest(execution=SRExecutionPolicy(stop=SRStopPolicy(maximum_controller_rounds=75))))
json.dump(r.to_dict(), open(f'aavqe_{R}_r75.json','w'), default=str)"
```

Notes: integration branch natively carries HH Powell xtol=1e-4/ftol=1e-8 (tolmatch folded in — no
monkeypatch). Alternative exact replay: the in-git r50→r75 continuation worker
`chtc/paper_i_ra_adapt_repair_20260727/paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2_chtc/worker.py`
with its sha-gated `anchors/` (carries a 1e-14 expected_prefix_energy gate; weak_weak
-0.9180092370322384 — if that gate trips on this machine it is a reportable determinism finding).

**Template B — B3-MR hierarchical (canonical `run_ra_adapt`, bare request)**, cwd `$SCRATCH/pi_integration`:

```bash
cd "$SCRATCH/pi_integration" && REGIME=<regime> ARM=<append|plateau|always> "$PY" "$RG" --limit-mb 8000 -- "$PY" -c "
import os,sys,json; from pathlib import Path; sys.path.insert(0,os.getcwd())
from pipelines.static_adapt.ra_adapt import build_paper_i_ra_hh_regime_problem, run_ra_adapt, RAAdaptRequest, RAAdaptOperationalControls
from pipelines.static_adapt.sr_snake.contracts import AppendOnlyInsertion, PlateauCommutationInsertion, AlwaysCommutationReducedInsertion, SRExecutionPolicy, SRMethodPolicy, SRStopPolicy, FreshStart, SRObservationPolicy, CheckpointObservation, EstimatorLedgerObservation
R=os.environ['REGIME']; A=os.environ['ARM']
ins={'append':AppendOnlyInsertion,'plateau':PlateauCommutationInsertion,'always':AlwaysCommutationReducedInsertion}[A]()
p=build_paper_i_ra_hh_regime_problem(R)
req=RAAdaptRequest(method=SRMethodPolicy(insertion=ins), execution=SRExecutionPolicy(stop=SRStopPolicy(maximum_controller_rounds=50)))
out=Path(f'b3mr_{R}_{A}')
res=run_ra_adapt(p, req, operational_controls=RAAdaptOperationalControls(maximum_controller_rounds=50, resume=FreshStart(), observation=SRObservationPolicy(checkpoint=CheckpointObservation(path=out/'checkpoints/current.json', every_controller_rounds=1, keep_history_tail=50), estimator_ledger=EstimatorLedgerObservation(path=out/'result/estimator_ledger.json'), resource_rounds=tuple(range(1,51)))))
json.dump(res.to_dict(), open(out/'result.json','w'), default=str)"
```

**Template C — B3 flat singleton (materialized study protocol — MANDATORY: a bare request raises
"Study policies require a validated bundle protocol materialization authority")**, cwd `$SCRATCH/pi_integration`:

```bash
cd "$SCRATCH/pi_integration" && REGIME=<regime> ARM=<append|plateau|always> "$PY" "$RG" --limit-mb 8000 -- "$PY" -c "
import os,sys,json; from pathlib import Path; sys.path.insert(0,os.getcwd())
from pipelines.static_adapt.ra_adapt import build_paper_i_ra_hh_regime_problem, build_paper_i_ra_all_phase_adaptive_forced_k50_request, build_paper_i_ra_all_phase_position_adaptive_forced_k50_request, materialize_paper_i_ra_semantic_protocol, run_ra_adapt, RAAdaptOperationalControls
from pipelines.static_adapt.sr_snake.contracts import FreshStart, SRObservationPolicy, CheckpointObservation, EstimatorLedgerObservation
R=os.environ['REGIME']; A=os.environ['ARM']
p=build_paper_i_ra_hh_regime_problem(R)
if A=='append':
    req=build_paper_i_ra_all_phase_adaptive_forced_k50_request(insertion_policy='append_only', maximum_controller_rounds=50, compile_cost_policy='explicit_depth_qiskit_cost_v1')
else:
    req=build_paper_i_ra_all_phase_position_adaptive_forced_k50_request(insertion_policy={'plateau':'plateau_commutation','always':'always_commutation_reduced'}[A], maximum_controller_rounds=50, compile_cost_policy='explicit_depth_qiskit_cost_v1')
proto=materialize_paper_i_ra_semantic_protocol(p, req)
out=Path(f'b3_{R}_{A}')
res=run_ra_adapt(p, proto, operational_controls=RAAdaptOperationalControls(maximum_controller_rounds=50, resume=FreshStart(), observation=SRObservationPolicy(checkpoint=CheckpointObservation(path=out/'checkpoints/current.json', every_controller_rounds=1, keep_history_tail=50), estimator_ledger=EstimatorLedgerObservation(path=out/'result/estimator_ledger.json'), resource_rounds=tuple(range(1,51)))))
json.dump(res.to_dict(), open(out/'result.json','w'), default=str)"
```

Record the materialized protocol sha in the receipt: expected to differ from the archived
`protocols/sixregime__*.json` ONLY via source-lock/implementation-inventory/bundle-manifest hashes
(route_contract, request, pool sha `c1b03515b02f8daf…`/948 labels, seeds {adapt:7, transpiler:7},
powell/200, horizon 50 must be IDENTICAL — else void the cell, fail closed).

**Template D — L=3 pure-Hubbard noise campaign (sealed runner, no args; sets its own science env)**:

```bash
cd "$SCRATCH/appendix_demos_tree" && "$PY" "$RG" --limit-mb 4000 -- "$PY" chtc/ra_adapt_appendix_noise_20260821_v1/run_local_ra_adapt_appendix_noise_l3_20260821_v1.py
```

Writes `output/local_runs/ra_adapt_appendix_noise_l3_20260821_v1` + `chtc/.../results` INSIDE the
export tree (not the repo). Do NOT substitute the un-promoted 2026-08-24 noise-branch rerun or the
adaptive-shortlist variant (exhausts its positive-score population).

### 1.2 Tier-0 star-value screens (RECOMMENDED FIRST PASS, ~30–90 min total, all 21 HH cells)

Same templates A/B/C with `maximum_controller_rounds` set to the cell's k\* (11 ww / 9 iw / 10 sw)
and `resource_rounds=tuple(range(1,k*+1))`. Each screen: ~1–5 min, <500 MB. This checks every
Table I / Table III star quantity (k\*, dE\_star, salg\_star, tuple\_star) before committing to the
long full-horizon runs — valid because `kstar_tables.json` spot_check records
`identical_ordering: true` (truncated replay == full-run prefix). A Tier-0 failure fires the stop
condition for that cell immediately at minutes-scale cost.

### 1.3 Full-horizon matrix — cheapest first (all HH cells: regimes at U/t ∈ {0.25, 1.25, 8},
λ=0.25, g=0.3535533905932738, ω0=1, t=1, L=2 OBC dimer, M=3, 8 qubits, 948-singleton pool,
seeds adapt=7/transpiler=7; tuples compiled FakeMarrakesh/opt-1/seed_transpiler=7)

Regime keys: ww = weak_weak, iw = intermediate_weak, sw = strong_weak_u8.

**Block 1 — AVQE comparator (Template A; Table I/III AVQE columns, Fig 2/5 AVQE curves).**
Expected from kstar_tables `aavqe`; per-round history vs golden
`bundle3_surviving_inputs/aavqe_full_hist_tolmatch_9665004.json` (sha `a05789300cd9…4271`).
Est 10–45 min each, <1.5 GB, guard 4000.

| ID | Cell | k\* | dE\_star | salg\_star | k\_term | dE\_term | salg\_term | dE\_45 | tuple\_star N2q/D2q/Dc |
|---|---|---|---|---|---|---|---|---|---|
| R01 | REGIME=weak_weak | 11 | 4.594932538475316e-4 | 13538 | 75 | 3.7166032517021996e-4 | 109710 | 3.716855362658755e-4 | 102/84/238 |
| R02 | REGIME=intermediate_weak | 9 | 2.128817926291826e-4 | 10910 | 75 | 1.723995773584841e-4 | 108736 | 1.7244521303438898e-4 | 35/30/101 |
| R03 | REGIME=strong_weak_u8 | 10 | 1.5929813839132834e-6 | 11845 | 75 | 1.388709197414073e-6 | 108357 | 1.396718688040366e-6 | 65/47/152 |

Tolerance: integers/tuples EXACT; energies Tier A/B (§3). Probe anchor: rounds 1–5 matched golden to
~1e-12 (E1=0.117217781462682 … E5=-0.905934119253846). EST-08 caveat: Powell callback-count-derived
quantities are cross-machine fragile — do not score them as exact.

**Block 2 — B3-MR hierarchical append (Template B, ARM=append; Table III rows / Fig 5). STRONGEST
parity target: protocol rematerializes sha-exact (`19bff49dce0f…`); probe reproduced stored trace
rounds 1–6 to every printed digit.** Expected from kstar_tables `b5` (Table III uses
salg_star_corrected). Est 15–45 min each, <2 GB, guard 8000.

| ID | Cell | k\* | dE\_star | salg\_star corr (raw) | dE\_term | salg\_term corr | tuple\_star |
|---|---|---|---|---|---|---|---|
| R04 | ww append | 11 | 6.604014228903665e-4 | 6277 (6409) | 1.12260529760988e-8 | 103493 | 39/32/112 |
| R05 | iw append | 9 | 5.05197874511365e-4 | 5599 (5707) | 9.223886543452409e-9 | 107600 | 35/25/83 |
| R06 | sw append | 10 | 3.2046935170004787e-6 | 5790 (5910) | 1.3804587777332955e-6 | 87578 | 37/28/101 |

Trace anchor (ww append, dE rounds 1–6): 2.1683809199948216, 1.0355987014575028,
0.0433809199948223, 0.03300017329550031, 0.022571755033389707, 0.012446800746102826.
Cross-check: `hh_flat_vs_hierarchical_comparison_tables.json` `*.hierarchical_RA`.

**Block 3 — L=3 noise campaign (Template D; Fig 4 + Table II, 6 published cells in one command).**
Est 45–120 min total, <1.5 GB (probe: one h10 RA cell = 155.5 s / 977 MB), guard 4000.

| ID | Cell | Expected (Table II verbatim @ k=10, AVQE/RA per level low, high, extreme) |
|---|---|---|
| R07 | full campaign (RA×3 + AVQE×3, h=20) | N2q 268/112, 268/160, 181/42; D2q 255/93, 253/144, 153/29; Dc 1049/380, 1044/591, 604/124; E(1e3) 10.9/6.79, 13.7/6.50, 6.58/4.77. RA S_alg corrected k10 = 6794/6497/4768 (raw 7034/6737/5011); k20 = 23541/21750/18910 (raw 24033/22233/19399). Fig 4 claim: RA lower at low/extreme; AVQE terminal error 14.4% lower at high. |

Sources: `estimator_reuse_corrections_20260821.json` (noise.k10/k20), NOISE_RESOURCE_TABLE in the
manuscript-repo `plot_ra_optional_controls_main.py`, per-cell shas in
`ra_optional_controls_l3_main_provenance.json` (post-run sha verification). E_ED = -2.1414935642714075.

**Block 4 — B3 flat append (Template C, ARM=append; Table I RA-Append column / Fig 2).**
Expected kstar_tables `b3` + corrections prefixes `b_depth_append_ra|<regime>`.
Est 0.5–2 h each, <2 GB, guard 8000.

| ID | Cell | k\* | dE\_star | salg\_star raw→corr | dE\_term (k50) | salg\_term | tuple\_star |
|---|---|---|---|---|---|---|---|
| R08 | ww append | 11 | 3.8402516319224755e-4 | 15150→14814 | 1.2306511365522965e-10 | 142500 | 46/38/134 |
| R09 | iw append | 9 | 2.128818283578804e-4 | 11936→11712 | 7.825538916872432e-8 | 167510 | 38/24/75 |
| R10 | sw append | 10 | 1.5925504381897326e-6 | 13860→13584 | 1.383672606047881e-6 | 130024 | 34/22/86 |

Trace anchor (ww append): dE(k=1)=1.035598701457568, dE(k=2)=0.04338091999489757 (probe matched ~1e-11).

**Block 5 — B3-MR hierarchical plateau + always (Template B; Table III / Fig 5).**
Est 1–3 h each, <2 GB, guard 8000.

| ID | Cell | salg\_star corr (raw) | salg\_term corr | tuple\_star | Table III printed E(1e3) |
|---|---|---|---|---|---|
| R11 | ww plateau | 8736 (8912) | 275384 | 39/32/112 | 8.74 |
| R12 | iw plateau | 5599 (5707) | 277200 | 35/25/83 | 5.60 |
| R13 | sw plateau | 10294 (10478) | 291497 | 36/24/87 | 10.3 |
| R14 | ww always | 17776 (18168) | 298653 | 45/40/154 | 17.8 |
| R15 | iw always | 12358 (12632) | 309779 | 23/17/69 | 12.4 |
| R16 | sw always | 14031 (14343) | 300914 | 38/27/101 | 14.0 |

k\* = 11/9/10 per regime as in Block 2; dE\_star/dE\_term per kstar_tables `b5` cells.

**Block 6 — B3 flat plateau + always (Template C; Table I RA-Plateau/RA-Insert columns). The
expensive half. Archived cells were partial-archive (exit-held): comparison basis is the recorded
stream trace + checkpoint S_alg receipts embedded in kstar_tables. Insertion-position replay adds
FP-sensitive position scoring — a position-tie flip is the most likely cross-machine divergence.**
Est 3–8 h each, <2–3 GB, guard 8000.

| ID | Cell | k\* | dE\_star | salg\_star raw→corr | tuple\_star | dE\_term (k50) | salg\_term |
|---|---|---|---|---|---|---|---|
| R17 | ww plateau | 11 | 5.225026718932435e-4 | 15150→14814 | 46/38/134 | 5.928590951498336e-14 | 681482 |
| R18 | iw plateau | 9 | 4.3477075966658063e-4 | 11936→11712 | 38/24/75 | 3.902433931557425e-14 | 703294 |
| R19 | sw plateau | 10 | 1.7645136867194822e-6 | 25823→25547 | 56/38/126 | 8.011725545209458e-7 | 696874 |
| R20 | ww always | 11 | 5.16826159910222e-4 | 38145→37697 | 69/51/177 | 6.106226635438361e-14 | 696344 |
| R21 | iw always | 9 | 6.781227491718944e-4 | 25769→25419 | 39/34/130 | 3.880229471064922e-14 | 696117 |
| R22 | sw always | 10 | 7.290394488768115e-6 | 31497→31173 | 46/32/120 | 4.440892098500626e-16 | 694393 |

Table I printed E(1e3), plateau/insert: 14.8/37.7 (ww), 11.7/25.4 (iw), 25.5/31.2 (sw).
Cross-check: `hh_flat_vs_hierarchical_comparison_tables.json` `weak_insertion_policies[].flat_pool_RA`.

---

## 2. Total wall-clock and execution rule

| Block | Cells | Est per cell | Block subtotal |
|---|---|---|---|
| Pre-flight + Tier-0 screens | gates + 21 prefix runs | 1–5 min | 0.5–1.5 h |
| 1 AVQE ×3 | R01–R03 | 10–45 min | 0.5–2.25 h |
| 2 B3-MR append ×3 | R04–R06 | 15–45 min | 0.75–2.25 h |
| 3 Noise campaign | R07 | 45–120 min | 0.75–2 h |
| 4 B3 flat append ×3 | R08–R10 | 0.5–2 h | 1.5–6 h |
| 5 B3-MR plateau/always ×6 | R11–R16 | 1–3 h | 6–18 h |
| 6 B3 flat plateau/always ×6 | R17–R22 | 3–8 h | 18–48 h |
| **TOTAL (serial)** | 22 runs (27 published cells) | — | **~28–79 h (~3–5 overnight days)** |

Execution rule: STRICTLY SERIAL, one run at a time, starting only AFTER the currently-running full
test suite completes (one heavy job per machine). Every run ram_guard-wrapped at the limit in its
row. Blocks execute in matrix order (cheapest first) so failures surface early; Tier-0 screens run
before any full-horizon block. Blocks 1–4 (star values for every Table I/III row + all of
Table II) complete in one overnight (~4–15 h); Block 6 is deferrable/optional depth — the star
quantities it validates are already covered by its Tier-0 screens, only the k50 terminal
(dE_term/salg_term) values require the full runs. RAM ceiling everywhere <3 GB observed-basis;
guards at 4000/8000 give headroom without risking the machine.

---

## 3. Comparison rubric (mirrors the repo's own parity/lock tests)

EXACT — zero tolerance, ever (plain `==`):
- k\*, k_term, the kstar map {ww:11, iw:9, sw:10}, accepted operator sequences, insertion
  positions, controller_round indices (lock-test standard, `test_ra_adapt_lock_singleton_trajectory.py:140–147`).
- S_alg and its four components with the internal identity S_alg == N_H_outer + N_H_refit +
  N_grad + N_metric (lane invariant); salg_star/salg_term; phase-reuse-corrected values must
  recompute exactly from raw integers via the corrections JSONs.
- Compiled tuples N2q/D2q/Dc/W1q + compile identity fields (FakeMarrakesh, opt level 1,
  seed_transpiler=7, routing_included) — exact GIVEN qiskit==2.3.1 (G1); version drift reclassifies
  a tuple mismatch as ENVIRONMENT, not FAIL. Proxy tuples: exact, no caveat.
- Protocol seeds {adapt:7, transpiler:7}; lineage digest `fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2` (never relabel).

FLOATS — three tiers (committed standards):
- Tier A PASS: abs diff ≤ 2.0e-12 (trajectory-lock) or equal after 10-decimal rounding (refactor-parity).
- Tier B PASS-WITH-NOTE (M4-BLAS ulp class): diff ≤ 1.0e-8 (comparator-parity ceiling) AND all
  integer selections identical AND value identical at manuscript printed precision.
- Tier C FAIL: > 1.0e-8, or any float drift accompanied by an integer flip.

Integer mismatch = FAIL, written up as a REPORTABLE FP EVENT (Paper III ±2 terminal-K precedent,
audit §2.3) — never absorbed into a tolerance, never silently rerun-until-match.

DIGESTS: byte-shas gate STORED inputs (G2/G3, before AND after) and fetched CHTC artifacts only.
For REGENERATED outputs byte-sha equality is sufficient-but-not-necessary (Paper II precedent:
physics ≤6.8e-15 with npz byte drift). Judge reruns by normalized field-level compare, exactly as
`test_ra_adapt_refactor_parity.py`: strip volatile keys (wallclock/timestamp/time_s/duration/
elapsed/hostname/path/pid), round floats to 10 decimals, sort keys, require exact equality.

PROTOCOL IDENTITY (fail-closed): B3-MR must rematerialize sha-exact; B3 flat may differ ONLY in
source-lock/implementation-inventory/bundle-manifest hashes — record archived + live shas in every
receipt as the pending §6.3 code-lineage caveat. Any route_contract/request/pool/seed delta voids
the cell and all downstream comparisons.

PRINTED-PRECISION tier runs only after raw-value rows pass; a printed match atop a Tier-C raw
failure is still FAIL. Per-cell receipt: command, REGIME/ARM, protocol shas, G1 versions,
wallclock, ram_guard peak RSS, field diff verdicts.

---

## 4. Exclusions (with reasons)

1. **Pure-Hubbard L=4 matched-k75 block** (Paper_I.tex ~1490–1516; k=60 9.93e-2 vs 1.83e-13;
   −39.9–50.2% circuits, +140.2% estimator): CHTC-scale — EXCLUDED from local rerun per task scope.
   Status: fetch-verified path only — CHTC fetch **P0** (audit §3 item 2): RA `result.tar.gz`
   sha256 `1d0ac8416d124ba6a85118f8a1a228b4f754c5567c0eb341db515426fdfca614`, member
   `./ra_l3_matched_k75/result.json`, receipts
   `chtc/ra_adapt_l4_qiskit_proxy_extensions_20260825/submission_receipt_v3.json` (cluster 9697719).
   AVQE k75 result.json (sha 41126e55…) local-only-and-gone → user-scheduled rerun from 53bba1c2.
   Corrected RA S_alg k60: 300584 − 892 = 299692 recomputes exactly (audit-verified).
2. **Fig 3 L=3 optional controls (AVQE/RA-Append/batch/beam/ablation, k=11 tuples, batch 1e-6 @ k=7)**:
   BLOCKED, no command. (a) The exact campaign driver (`paper_i_l3_phase0_qiskit_adaptive_20260826`)
   is in NO git ref — any rerun is a reconstruction requiring user authorization (audit §2.1, do not
   redo). (b) The batch arm is the ONLY in-scope target routed through the batch-admission
   controller, which on main fails 8/28 canonical-interface tests (ValueError at
   `pipelines/static_adapt/sr_snake/_selection.py:1650`) — gated on the pending §6.3 code-lineage
   decision. FLAGGED as batch-admission-dependent.
3. **Strong-Holstein sector** (weak_strong, intermediate_strong, strong_strong_u8; M=7, k\*=75/75/27,
   S_alg~1e6, CHTC 40 GB requests; insertion arms explicitly incomplete-prefix evidence): outside
   user scope ("weak HH") and CHTC-scale. Note: weak_strong/intermediate_strong contain the word
   "weak" but are strong-sector — do not rerun them by name confusion.
4. **Sealed CHTC worker replay route** (b3/b3mr `worker.py` + execution_authorization): fails closed
   on current code — job `source_inventory_sha256` `6f43505e8b4b…` vs live `ff2c6c96d3ec…`
   (59 static_adapt files changed since 551a9fe0). Route exclusion only; the cells themselves run
   via the integration-branch facade (Templates B/C), which is protocol-verified against the same
   archives. All 21 weak-HH cells are SingletonAdmission — zero batch-controller exposure.
5. **AVQE r0→50 original producer package**: absent from git; substituted by the fresh full-history
   Template A run (probe-verified round-for-round vs golden) with the in-git r50→75 continuation
   worker as the sealed alternative.
6. **Any HH rerun from main @ dafbda5e**: prohibited — wrong lineage (route contract v2 vs archived
   v4; costexp identity not constructible; appendix_demos absent).
7. **Legacy Table-III source-lock family**: user decision pending (audit §2.1 row) — not a rerun target.

---

## 5. Stop conditions

1. **A mismatched cell = record and report, do not debug-loop.** Write the per-cell verdict
   (Tier + rubric class: FP event / environment / FAIL), keep the rerun output for diffing, and move
   to the NEXT cell. No settings tweaks, no code edits, no retry ladders. At most ONE identical
   same-command repeat is permitted for cells estimated <1 h (to separate machine flake from
   deterministic drift); zero repeats for longer cells.
2. **Fail-closed at materialization**: if a protocol's route_contract, request, pool sha, or seeds
   differ from the archived cell (beyond the documented source-lock hash delta), VOID the cell,
   skip the rest of that family, report — comparisons downstream of a wrong protocol are meaningless.
3. **Resource stops**: ram_guard kill, or wall-clock exceeding 2× the row's upper estimate → abort
   the cell, record partial trace + peak RSS, continue.
4. **Family stop**: two Tier-C failures in one family (A/B/C) → halt that family, report; the
   pattern (which arms/regimes flip) is the finding.
5. **Scheduling**: never start while the full test suite (or any other heavy job) is running;
   re-check before each block. One run at a time, always ram_guard-wrapped.
6. **Integrity**: G2+G3 re-run at the end must still show 10/10 OK and `060e7524…` — any change is
   an immediate full-stop and report.
7. **Governance**: no promotion/demotion language in the resulting report (user-only per
   MATH/AGENTS.md); git read-only; the only new repo file from execution is the final parity report
   in `docs/reports/`; all run outputs stay in `$SCRATCH`.
