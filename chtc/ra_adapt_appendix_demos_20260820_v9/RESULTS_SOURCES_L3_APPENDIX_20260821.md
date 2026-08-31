# L=3 appendix results: source-of-truth map

**For:** the manuscript agents. Every number and curve in the L=3 appendix
figures traces to one of the artifact roots below. Update plots from these
paths, not from chat transcripts or memory. When a campaign is rerun its
runtime root is replaced wholesale; the provenance JSONs carry the digests
that tell you whether anything changed.

Repo: `/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820`
Branch `appendix-demos-20260820`. This file written at commit `3458156f0a77ee32c49c57b2b8160470a24f9ead`.
Physics for every table: half-filled L=3 Hubbard chain, open boundary, t=1,
U/t=1.25, 6 qubits, HF reference, E_ED = -2.1414935642714075. Pool:
macro_generator_v1, 24 sector-guarded intact generators, identical everywhere
including AAVQE. Cost tuples (N2q, D2q, Dc, ...) are FakeMarrakesh, opt 1,
seed 7, full circuit incl. reference preparation.

---

## 1. Optional-control figures (in the manuscript now)

Campaign `ra_adapt_appendix_demos_l3_20260820_v9` — the v4 appendix route:
measured_residual_response_v1, joint-minus-active-only gain, native chart,
FS-preconditioned Powell directions, zero start, Powell 200/1e-5/1e-12,
horizon 15.

- Figures + provenance JSON (all caption numbers):
  `chtc/ra_adapt_appendix_demos_20260820/results_l3_v9/figures/`
  (`ra_appendix_l3_v9_provenance.json`, sha256 c782aef881e9...)
- Signed cells: `output/local_runs/ra_adapt_appendix_demos_l3_20260820_v9/cells/`
- Regenerate: `chtc/ra_adapt_appendix_demos_20260820_v9/plot_l3_v9_appendix_figures.py`
  and `plot_beam_branching.py`
- Manuscript figure dir: `ADAPT---Paper-I/Paper_I_overleaf_20260818/figures/ra_adapt_appendix_v9/`

| curve | mk | N2q | D2q | Dc | E | err@15 |
|---|---|---|---|---|---|---|
| AAVQE | 8 | 216/508 | 204/493 | 837/2024 | 6510/8522 | 1.64e-14 |
| RA-append | 8 | 104/127 | 88/109 | 363/441 | 4124/9547 | 7.11e-15 |
| + ablation (gated) | 8 | 104/112 | 88/93 | 363/378 | 4124/10658 | 8.88e-16 |
| + beam (Pareto) | 8 | 88/111 | 71/81 | 298/330 | 4508/10185 | 2.22e-15 |
| + batch | 3 | 200/229 | 188/206 | 776/853 | 2683/11823 | 8.44e-15 |

(cells are marker-round / horizon values; E is branch-local.)

## 2. Noise panel (in the manuscript now)

Probe runs at commit 8c7845e0 (RA noise arm: Singleton/off/off overlay on the
same physics; AAVQE comparator with the same noise oracle). Horizon 20.

- Figure: `chtc/ra_adapt_appendix_noise_20260821_v1/figures/ra_appendix_l3_noise_v9.{pdf,png}`
- Generators: `plot_l3_noise_figure.py`, `probe_noise.py`, `probe_aavqe.py` (same dir)
- Signed campaign (receipts): `chtc/ra_adapt_appendix_noise_20260821_v1/run_local_ra_adapt_appendix_noise_l3_20260821_v1.py`
  writes `output/local_runs/ra_adapt_appendix_noise_l3_20260821_v1/` — RA cells
  complete and digest-matched to the figure; AAVQE cells were mid-run when the
  campaign was last stopped, so the figure's AAVQE numbers currently trace to
  the probe runs; rerun the campaign to completion before claiming receipts.
- Floors (k=20): RA 5.82e-06 / 2.53e-03 / 1.48e-01; AAVQE 2.44e-05 / 2.21e-03 / 4.67e-01
  (low/high/extreme). Dotted lines are measured plateau medians (k>=10), NOT
  analytic floors — keep captions honest about that.

## 3. Affine-chart ablation (NOT yet in the manuscript)

Campaign `ra_adapt_appendix_demos_l3_20260820_v10_whitened` — identical to
v9 except accepted-refit chart supported_fs_whitened_fixed_v1 (and its base
chart); zero start retained. Cells:
`output/local_runs/ra_adapt_appendix_demos_l3_20260820_v10_whitened/cells/`

| arm | mk | N2q | D2q | Dc | E | err@15 |
|---|---|---|---|---|---|---|
| RA-append | 8 | 137 | 110 | 458 | 9403 | 3.11e-15 |
| + batch | 3 | 229 | 206 | 840 | 11352 | 6.22e-15 |
| + ablation | 8 | 108 | 87 | 359 | 10825 | 8.88e-16 |
| + beam | 8 | 112 | 80 | 335 | 9822 | 4.44e-16 |

Reading: chart is a small, mostly favourable effect (E down 1.5-4% in three
arms, error down in three); marker rounds unchanged. HH already runs this
chart, so this argues against re-running HH on the chart axis.

## 4. HH-algorithm arms at L=3 (in progress; scratchpad, no receipts yet)

Macro semantic-closure route, forced variant: stationary_source_response_v1,
raw joint gain, affine chart, no initialization, Powell 200/**1e-4/1e-8**
(HH's own tolerances — differ from the appendix arms), horizon 15 forced.
Algorithm id `paper_i_ra_macro_append_endpoint_gradient_adaptive_phase123_adaptive_qiskit_forced_semantic_closure_v2`.

Runner: scratchpad `macro_hh_arm.py` (to be promoted to a campaign once all
four arms pass). Results so far:

Results live under `chtc/ra_adapt_hh_route_l3_20260821/results/<arm>/result.json`
(runner scripts beside them).  Horizon 15 forced; batch ends earlier by its
own declared policy (see notes).

| arm | rounds | first<=1e-6 | err@final | status |
|---|---|---|---|---|
| RA-append | 15 | 9 | 3.11e-15 | done |
| + ablation (gated) | 15 | 9 | 3.11e-15 | done — identical trajectory to baseline; no deletion separated it |
| + beam (Pareto) | 15 | **8** | 8.88e-16 | done — one round earlier than baseline; siblings walk the adaptive prefix |
| + batch | 15 | **6** | 7.99e-15 | done — converges 3 rounds early; forced to the full horizon via the size-1 rank-gate waiver (the champion macro generator is admitted even when its tangent adds no metric rank, mirroring the forced single-admission path; multi-member subsets keep the gate) |

Batch semantics on this route (decided during the extension, flag to the
author): the adaptive champion must be admitted every round; the batch may
extend the admission with jointly-feasible companions from its ranked window
(commit order phase2-ranked, receipt order score-ranked).  The beam needs
`compile_cost_policy="explicit_depth_qiskit_cost_v1"` so the route declares
the 0.30/0.30/0.25 pricing identity.

Natural-terminal (unforced) baseline for reference: stops itself at round 9,
err 7.4e-11 — that stop is the route's own convergence criterion.

## 4b. Deferred-repair ablation (new variant, both routes)

`metric_ablation_gated_norefit`: identical gated screens; acceptance is
still decided by the canonical measured delete-and-refit trial, and ONLY the
commit changes -- accepted deletions leave the survivors at their inherited
angles and the next round's admission refit repairs.  Declared as
`prune_commit_policy = deferred_repair_unrefitted_survivors_v1`.  (An earlier
form of this variant also replaced the acceptance test with the raw deletion
energy; that version was retired as a confounder, and its 9455-query result
is superseded by the table below.)

Appendix (v4) route, h15 -- results at
`chtc/ra_adapt_appendix_demos_20260820/results_l3_v9_extra/metric_ablation_gated_norefit/`:

| variant | N2q/D2q/Dc | E | err@15 |
|---|---|---|---|
| RA-append | 127/109/441 | 9547 | 7.1e-15 |
| ablation (gated, refit) | 112/93/378 | 10658 | 8.9e-16 |
| ablation (gated, deferred repair) | 112/93/378 | **9455** | 1.8e-15 |

Same circuit reduction and the same six deletions, 11.3% cheaper than the
refitting prune, and cheaper than the baseline itself: deferring repair to
the admission refit that runs anyway makes ablation estimator-free.

HH route, h15 (`chtc/ra_adapt_hh_route_l3_20260821/results/metric_ablation_gated_norefit/`):
first<=1e-6 at 9, err@15 3.11e-15 -- indistinguishable from that route's
baseline, as with every ablation variant there.

## 5. Route axes (why these configurations differ)

| axis | HH route (semantic_closure_v2) | appendix route (full_response_v2) |
|---|---|---|
| active response | stationary (assumed) | measured residual |
| candidate gain | raw joint | joint minus active-only baseline |
| refit chart | affine FS-whitened | native + FS-preconditioned directions |
| refit initialization | none (concept absent) | off (explicit) |
| Powell tol | xtol 1e-4, ftol 1e-8 | xtol 1e-5, ftol 1e-12 |
| Phase-III exhaustion | natural terminal (forced variant added) | runs to horizon |

The math handoff (`HANDOFF_MANUSCRIPT_MATH_BATCH_BEAM_PRUNE_20260821.md`,
same directory) carries the exact code and divergences for batch/beam/prune.

