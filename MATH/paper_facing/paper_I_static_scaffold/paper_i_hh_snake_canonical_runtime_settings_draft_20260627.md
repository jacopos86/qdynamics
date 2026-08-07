# Paper I HH SNAKE Result Settings Audit
Created: 2026-06-27
Updated: 2026-07-09

Scope: historical Paper-I Hubbard-Holstein SNAKE runtime rows plus the current
visible-row recovery correction below.  For the active POWELL visible-row
recovery/candidate line, use the narrower settings lock:

```text
MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_powell_visible_recovery_candidate_settings_20260706.md
```

The older source table here began from
the 2026-06-24 Schur warm-start update in
`MATH/paper_details/static_adapt_paper_I.tex`; the current visible Page-8 /
Powell pool-exposure row must be resolved from its support CSV/JSON and
result/effective-command JSON before recovery.
Physics/Hamiltonian parameters are intentionally omitted here. This file records
SNAKE algorithmic settings and mismatches between the manuscript cost notation
and the implemented/result-producing parameterization. The `Suggested canonical`
column is a draft prescriptive run surface.
In settings tables, `SAME` means all six historical regime values match the
suggested canonical value; `Discrep.` means at least one regime differs, is
missing, or the suggested value remains unresolved. Matrix choices belong in
the `Suggested canonical` column, for example `2 vs 8 parity check; use 8 if
equivalent`. Long continuous Optuna settings are rounded for display in the
tables below; the full-precision values remain in the source audit/provenance.

## Visible-Row Provenance Correction (2026-07-05)

Do not use this draft settings table by itself to answer what the currently
rendered Paper-I HH row used.  For recovery of an existing displayed row, first
resolve the manuscript/PDF row to its support CSV/JSON and then to the
result/effective-command JSON.  The current Page-8 / Powell pool-exposure
visible SNAKE weak-weak row is a `visible_row` with:

- `--adapt-inner-optimizer POWELL`;
- unfiltered `--adapt-pool full_meta` with no class filter, hence HVA included;
- native Phase-III archival Pauli-child split;
- `hard_guard` route label/effective command, with the historical child-set
  representative forwarded as missing-spec/skip-pass metadata rather than a
  newly enforced child-set hard guard;
- `--phase3-runtime-split-max-subset-size 1`;
- maxiter/refit `200` and depth cap `30`;
- no `--phase3-source-lock-preferred-sequence`.

The older SPSA/subset-size-3 row that appears in command-audit ancestry is a
`parent_source_lock`, not the default visible-row fact.  The active 2026-07-06
POWELL visible-row recovery/candidate line keeps cap `1`, adds
`--adapt-beam-lambda 0.005`, and uses the metric-regularized prune route.  Cap
`3` is historical/diagnostic unless the user explicitly reopens a
child-subset-size study.
See
`MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_visible_row_provenance_layers_20260705.md`
for the reusable provenance-layer rule and current anchor paths.

Recovery acceptance note: for the current weak-weak visible-row anchor, exact
selected-label identity is not required.  A bounded adaptive rerun without
`--phase3-source-lock-preferred-sequence` recovered the selected-prefix energy
to below `1e-12` absolute difference while admitting some spin-degenerate or
near-degenerate child-set labels.  Treat that as the baseline recovery gate for
subsequent one-variable prune, beam, and batch perturbations unless strict
label identity is explicitly requested.

## Canonical Full-Geometry / Full-Refit Correction (2026-07-09)

Do not use the historical `windowed` plus every-eight full-refit overlay as the
forward canonical Hubbard-Holstein SNAKE policy.  Historical artifacts that
actually ran with `--adapt-reopt-policy windowed`, `--adapt-full-refit-every 8`,
and `--adapt-final-full-refit true` should remain recorded as historical
provenance, but that combination is not the desired canonical route.

The forward canonical route uses full-coordinate reoptimization after each
accepted ansatz update, full Phase-II/III geometry windows, physical
operator-type lanes, no batching, and the metric-regularized prune nomination
route:

```text
--adapt-reopt-policy full
--adapt-window-size 99
--adapt-window-topk 0
--adapt-maxiter 200
--adapt-full-refit-every 1
--adapt-final-full-refit true
--adapt-insertion-mode always
--phase1-probe-max-positions 999999
--phase1-trough-margin-ratio 1.0
--phase3-geometry-window-size 99
--phase3-novelty-ablation-mode off
--phase1-prune-schur-nomination-route metric_regularized_v1
--phase1-prune-metric-schur-mu 0.01
--phase1-prune-metric-schur-cost-weighting ansatz_entry_denominator_v1
--static-lane-route physical_operator_type
--physical-lane-shortlist-aggressiveness 3
--phase1-maturity-cap-min 999999
--phase1-maturity-cap-max 999999
--phase2-maturity-cap-min 999999
--phase2-maturity-cap-max 999999
--phase3-maturity-cap-min 999999
--phase3-maturity-cap-max 999999
--phase-maturity-shot-min 1
--phase-maturity-shot-max 1
--phase1-maturity-shot-cap 1
--phase2-maturity-shot-cap 1
--phase3-maturity-shot-cap 1
--phase-live-hysteresis-disabled
--phase2-no-batching
--phase3-no-batching
```

Here `full` means all current ansatz parameters are reoptimized after every
accepted iteration, with the usual `--adapt-maxiter 200` budget.  The values
`--adapt-window-size 99` and `--adapt-window-topk 0` are retained as explicit
full-ansatz guards for any window-derived helper path; under
`--adapt-reopt-policy full`, they are not a separate local-window
approximation.  The candidate insertion-position domain is also full-ansatz:
`--adapt-insertion-mode always` explicitly enables insertion-position
evaluation at every accepted iteration, `--phase1-probe-max-positions 999999`
keeps all insertion positions available at the Paper-I HH depth scale, and
`--phase1-trough-margin-ratio 1.0` keeps the existing full-probe trigger
convention.  The value
`--phase3-geometry-window-size 99` makes the Phase-II and Phase-III geometry
windows cover the full existing ansatz for the Paper-I depth scale.  The final
full-refit flag is source-parity metadata under the full policy: the accepted
iterations already use full-coordinate reoptimization.

The forward canonical route does not use maturity scheduling as an additional
adaptive record-budget route.  The maturity cap min/max flags are therefore set
to a nonbinding sentinel.  In the physical-lane implementation these sentinels
are clamped to the effective shortlist caps, so the retained-record budget is
controlled by the explicit shortlist policy rather than by a separate
runway/maturity schedule.  Phase-live hysteresis is disabled, and maturity-shot
settings are frozen at one probe per phase so they do not alter phase liveness.

This is a configuration-level full-geometry policy.  It does not yet implement
a single shared measured Gram cache between Phase-II/III scoring and
post-admission metric-prune nomination.  The canonical baseline keeps the
Phase-II/III metric/novelty terms active: `--phase3-novelty-ablation-mode off`
means no novelty ablation is applied.  Mechanism-ablation rows should compare
against this active baseline, and run telemetry should show the metric-prune
route as active (for example `metric_prune_route_active=true` or
`metric_regularized_active=true`) when the metric-regularized prune route is in
scope.  The metric-prune route and Phase-II/III scoring use the same
tangent-overlap/Fubini--Study Gram construction, but a future QPU-faithful
shared-measurement cache is a separate implementation change.

## Current Comparison Contract

Current user-selected contract for the active POWELL visible-row recovery pass:

- All three methods use unfiltered `full_meta`; do not apply the
  `full_meta_minus_hva` class filter.
- Geo-ADAPT and append-only ADAPT use singleton child splitting from their
  comparator candidate pool:
  `generic_adapt_runtime_split_mode=shortlist_pauli_children_v1`,
  `generic_adapt_runtime_split_symmetry_policy=hard_guard`, and
  `generic_adapt_runtime_split_max_subset_size=1`.
- SNAKE uses the recovered Paper-I Phase-III archival child-set split path:
  `--phase3-runtime-split-mode shortlist_pauli_children_v1`,
  `--allow-archival-phase3-runtime-split`,
  `--phase3-runtime-split-selection-mode archival_child_set_forward_v1`, and
  `--phase3-runtime-split-child-set-symmetry-policy hard_guard` with
  `--phase3-runtime-split-max-subset-size 1` for the active 2026-07-06 POWELL
  visible-row recovery/candidate line.  The
  selected child-set representative must preserve the historical
  `symmetry_spec=None` / `runtime_split_symmetry_spec_missing` metadata.
- The active forward canonical SNAKE line sets full reoptimization,
  full-ansatz position/refit/geometry windows (`--phase1-probe-max-positions
  999999`, `--adapt-window-size 99`, `--adapt-window-topk 0`,
  `--phase3-geometry-window-size 99`), physical operator-type lanes,
  active metric/novelty scoring (`--phase3-novelty-ablation-mode off`),
  `--phase1-prune-schur-nomination-route metric_regularized_v1`,
  `--phase1-prune-metric-schur-mu 0.01`, and disables Phase-II/III batching.
- SNAKE shared/global child-pool expansion remains off for this contract.

This supersedes the earlier draft surface that used `full_meta_minus_hva` and a
Phase-0/shared Pauli-child pool.

## Physical-Lane And Batch Correction (2026-07-08)

The current Paper-I manuscript route no longer interprets the staged lane
partition as support- or commutation-defined algebraic lanes.  For
Hubbard--Holstein, the lane partition is physical operator type:

```text
electronic UCCSD
electronic hopping/current
phonon cloud
phonon relaxation
dressed electron-phonon
Hamiltonian-block / HVA
```

Older table rows below that mention algebraic shortlisting or
`algebraic_nested_v1` should be read as historical selector-surface labels
unless a specific result artifact proves support/commutation lanes were used.
The current physical-lane route shortlists within the six physical operator
types above.  Support and commutation diagnostics may still be logged or used as
optional compatibility telemetry, but they do not define the lane route.

The forward canonical Paper-I Hubbard--Holstein SNAKE line disables Phase-II
and Phase-III batching.  Do not use the older reduced-plane target/cap `8/16`
rows below, and do not use the interim maxB=1 reduced-plane route unless the
user explicitly reopens batching as an ablation.

## Current Matrix Selection Note

The earlier SPSA/canonical-settings matrix tested collapse witness `off/on` and phase
liveness/hysteresis `default/off` under the full-meta Phase-III child-set route
above. The completed weak-weak, intermediate-weak, and strong-weak
default-hysteresis cells show no meaningful difference between collapse
witness `off` and `on`; the completed weak-weak and intermediate-weak
hysteresis cells show negligible difference between `default` and `off`.

Prescriptive choice for the next canonical runs:

- Collapse witness: `off`.
- Phase liveness/hysteresis: `off`.

The enabled/default variants remain diagnostic alternatives to mention in the
matrix report, but they are not the default command surface for the next
SNAKE runs unless later matrix cells reverse this conclusion.

## Source Set

Common prefix:
\path{raw_outputs/paper_i_hh_schur_warm_start_native200_depth30_20260623_v1/}

Each row appends `/json/result.json`.

| Regime | Run directory under common prefix |
| --- | --- |
| WW | \path{paper_i_hh_native200_forced_depth30_noearlystop_20260619_v2__weak_weak__snake__native_forced__maxiter200__depth30_noearlystop} |
| IW | \path{paper_i_hh_native200_forced_depth30_noearlystop_20260619_v2__intermediate_weak__snake__native_forced__maxiter200__depth30_noearlystop} |
| SW | \path{paper_i_hh_native200_forced_depth30_noearlystop_20260619_v2__strong_weak__snake__native_forced__maxiter200__depth30_noearlystop} |
| WS | \path{paper_i_hh_native200_forced_depth30_noearlystop_20260619_v2__weak_strong__snake__native_forced__maxiter200__depth30_noearlystop} |
| IS | \path{paper_i_hh_native200_forced_depth30_noearlystop_20260619_v2__intermediate_strong__snake__native_forced__maxiter200__depth30_noearlystop} |
| SS | \path{paper_i_hh_native200_forced_depth30_noearlystop_20260619_v2__strong_strong__snake__native_forced__maxiter200__depth30_noearlystop__full_schur_fromscratch_8workers_20260623} |

Each result directory also contains the exact executed command in
`effective_command.json`.

## Cost Term Map

Paper I writes the phase-indexed cost penalty as

$$
K_t(r)
=
1+\lambda_{2q}\bar C_{2q}(r;t)
+\lambda_d\bar C_d(r;t)
+\lambda_{1q}\bar C_{1q}(r;t)
+\lambda_\theta\bar C_\theta(r;t)
+\lambda_{\rm shot}\bar C_{\rm shot}(r;t).
$$

For the visible SNAKE rows, explicit `phase*_lambda_*` fields were not supplied
for the five Paper-I lambdas. The code used a compatibility alias map.

| Paper symbol | Phase-I value used | Phase-II/III value used | Repo source |
| --- | ---: | ---: | --- |
| \path{lambda_2q} | `0.05` | `0.20` | Phase I: `phase1_lambda_compile`; Phase II/III: `phase2-w-depth * compile_cx_proxy_weight` |
| \path{lambda_d} | `0.05` | `0.20` | Phase I: `phase1_lambda_compile`; Phase II/III: `phase2-w-depth` |
| \path{lambda_1q} | `0.025` | `0.10` | Phase I: `phase1_lambda_compile * 0.5`; Phase II/III: `phase2-w-depth * 0.5` |
| \path{lambda_theta} | `0` historically; canonical rerun `0.001` | `0.10` | Phase I: historical rows omitted explicit `--phase1-lambda-theta`; canonical rerun sets it directly. Phase II/III: `phase2-w-optdim` |
| \path{lambda_shot} | `0.02` | `0.15` | Phase I: `phase1_lambda_measure`; Phase II/III: `max(phase2-w-shot, phase2-w-group, phase2-w-reuse)` |

Thus the historical Phase-I rows used $\lambda_{\theta,1}=0$, while the
canonical rerun surface sets $\lambda_{\theta,1}=0.001$. The intended canonical
cost formulas are

$$
K_1(r)
=
1
+0.05\bar C_{2q}
+0.05\bar C_d
+0.025\bar C_{1q}
+0.001\bar C_\theta
+0.02\bar C_{\rm shot},
$$

$$
K_2(r)
=
1
+0.20\bar C_{2q}
+0.20\bar C_d
+0.10\bar C_{1q}
+0.10\bar C_\theta
+0.15\bar C_{\rm shot},
$$

$$
K_3(r)
=
1
+0.20\bar C_{2q}
+0.20\bar C_d
+0.10\bar C_{1q}
+0.10\bar C_\theta
+0.15\bar C_{\rm shot}.
$$

The distinction between `K_2` and `K_3` is the phase record family and the
candidate features entering the barred costs. The lambdas were the same for
Phases II and III in these result rows.

## Cost And Backend Settings Used

| Flag / effective field | Discrep. | Paper-I LaTeX symbol / role | WW | IW | SW | WS | IS | SS | Suggested canonical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \path{phase1_lambda_F} | SAME | $\Delta E_1(r)$: Phase-I gain scale before $S_{1,\ell}(r)=\Delta E_1(r)\mathcal N_1(r)/K_1(r)$; implementation alias, not one of the five $K_t$ lambdas | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |
| \path{phase1_lambda_compile} | SAME | $\lambda_{2q},\lambda_d,\lambda_{1q}$ in $K_1(r)$: legacy Phase-I compile-cost source | `0.05` | `0.05` | `0.05` | `0.05` | `0.05` | `0.05` | `0.05` |
| \path{phase1_lambda_measure} | SAME | $\lambda_{\rm shot}$ in $K_1(r)$: legacy Phase-I measurement-cost source | `0.02` | `0.02` | `0.02` | `0.02` | `0.02` | `0.02` | `0.02` |
| \path{--phase1-lambda-theta} | New canonical | $\lambda_{\theta,1}\bar C_\theta(r;1)$ in $K_1(r)$: tiny Phase-I parameter-count burden | `-` | `-` | `-` | `-` | `-` | `-` | `0.001` |
| \path{phase1_lambda_leak} | SAME | No Paper-I symbol: implementation-only leakage guard; no displayed term in $K_t(r)$ | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| \path{--phase2-w-depth} | SAME | $\lambda_d,\lambda_{2q},\lambda_{1q}$ in $K_2(r),K_3(r)$: legacy Phase-II/III compile-depth source | `0.2` | `0.2` | `0.2` | `0.2` | `0.2` | `0.2` | `0.2` |
| \path{--phase2-w-group} | SAME | $\bar C_{\rm shot}(r;t)$: measurement-group part of the shot-cost feature; enters effective $\lambda_{\rm shot}$ by alias max rule | `0.15` | `0.15` | `0.15` | `0.15` | `0.15` | `0.15` | `0.15` |
| \path{--phase2-w-optdim} | SAME | $\lambda_\theta\bar C_\theta(r;t)$ in $K_2(r),K_3(r)$: parameter-count burden source | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` |
| \path{--phase2-w-reuse} | SAME | $\bar C_{\rm shot}(r;t)$: measurement-cache reuse part of the shot-cost feature; enters effective $\lambda_{\rm shot}$ by alias max rule | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` |
| \path{--phase2-w-lifetime} | SAME | No Paper-I symbol: implementation horizon/lifetime multiplier on resource burden; no separate displayed term in $K_t(r)$ | `0.05` | `0.05` | `0.05` | `0.05` | `0.05` | `0.05` | `0.05` |
| \path{--phase2-w-shot} | Discrep. | $\bar C_{\rm shot}(r;t)$: direct new-shot part of the shot-cost feature; would be $\lambda_{\rm shot}$ only without the alias max rule | `0.08` | `0.04` | `0.15` | `0.02` | `0.02` | `0.08` | `0.05` |
| effective `lambda_shot` for `K_2,K_3` | SAME | $\lambda_{\rm shot}$ in $K_2(r),K_3(r)$: actual coefficient of $\bar C_{\rm shot}(r;t)$ after `max(w_shot,w_group,w_reuse)` | `0.15` | `0.15` | `0.15` | `0.15` | `0.15` | `0.15` | `0.15` |
| \path{--phase3-backend-cost-mode} | SAME | $\widehat C_{2q}(r),\widehat C_d(r)$: backend oracle used to estimate compile-cost features before $K_3(r)$ normalization | \path{marrakesh_graph_span_v1} | \path{marrakesh_graph_span_v1} | \path{marrakesh_graph_span_v1} | \path{marrakesh_graph_span_v1} | \path{marrakesh_graph_span_v1} | \path{marrakesh_graph_span_v1} | \path{marrakesh_graph_span_v1} |
| \path{--phase3-backend-name} | SAME | $G$ and $\widehat C_d(r)$: hardware graph used in the Marrakesh compile-cost proxy | `FakeMarrakesh` | `FakeMarrakesh` | `FakeMarrakesh` | `FakeMarrakesh` | `FakeMarrakesh` | `FakeMarrakesh` | `FakeMarrakesh` |
| \path{--phase3-backend-w-2q} | SAME | $\widehat C_{2q}(r)$ feature weight: backend-oracle internal weight, not $\lambda_{2q}$ itself | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |
| \path{--phase3-backend-w-depth} | Discrep. | $\widehat C_d(r)$ feature weight: backend-oracle internal layout-span/depth weight, not $\lambda_d$ itself | `0.25` | `0.1` | `0.25` | `0.25` | `0.25` | `0.1` | `0.15` |
| \path{--phase3-backend-w-size} | SAME | No Paper-I symbol: backend-oracle internal size term feeding compile burden | `0.01` | `0.01` | `0.01` | `0.01` | `0.01` | `0.01` | `0.01` |
| \path{--phase3-backend-transpile-seed} | SAME | No Paper-I symbol: implementation seed for backend-cost oracle | `7` | `7` | `7` | `7` | `7` | `7` | `7` |
| \path{--phase3-backend-optimization-level} | SAME | No Paper-I symbol: implementation transpiler setting for backend-cost oracle | `1` | `1` | `1` | `1` | `1` | `1` | `1` |

`phase3-backend-w-*` are backend-oracle compile weights. They are not the same
objects as the Paper-I `lambda_*` coefficients in `K_t`; they shape the backend
cost feature before or alongside the normalized cost denominator.

## Route, Pool, Child Policy

| Setting | Discrep. | Paper-I LaTeX symbol / role | WW | IW | SW | WS | IS | SS | Suggested canonical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \path{--static-route-id} | SAME | $\mathcal R_{t-1}\to\mathcal R_t$ and $U_k(\theta_k)\oplus\mathcal B\ominus\mathcal D$: route identity for the SNAKE staged map and update rule | `route_a` | `route_a` | `route_a` | `route_a` | `route_a` | `route_a` | `route_a` |
| \path{--static-meta-feature-profile} | SAME | $\Delta E_t(r),\mathcal N_t(r),K_t(r)$: implementation bundle selecting feature definitions | \path{paper_i_production_v1} | \path{paper_i_production_v1} | \path{paper_i_production_v1} | \path{paper_i_production_v1} | \path{paper_i_production_v1} | \path{paper_i_production_v1} | \path{paper_i_production_v1} |
| \path{--adapt-continuation-mode} | SAME | $t=0,1,2,3$: enables the four-phase SNAKE restriction sequence | `phase3_v1` | `phase3_v1` | `phase3_v1` | `phase3_v1` | `phase3_v1` | `phase3_v1` | `phase3_v1` |
| \path{--adapt-pool} | SAME | $\mathcal R_0$ and $r=(m,p)$: initial candidate generator universe and position records | `full_meta` | `full_meta` | `full_meta` | `full_meta` | `full_meta` | `full_meta` | `full_meta` |
| \path{--adapt-pool-class-filter-json} | SAME | $\mathcal R_0$ restriction: pool filter before forming candidate records; current contract applies no `full_meta_minus_hva` filter | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| \path{--phase3-runtime-split-mode} | SAME | $C_{\rm split}(m)$: SNAKE Phase-III-only Pauli-child split after macro-generator shortlisting | \path{shortlist_pauli_children_v1} | \path{shortlist_pauli_children_v1} | \path{shortlist_pauli_children_v1} | \path{shortlist_pauli_children_v1} | \path{shortlist_pauli_children_v1} | \path{shortlist_pauli_children_v1} | \path{shortlist_pauli_children_v1} |
| \path{--allow-archival-phase3-runtime-split} | SAME | $C_{\rm split}(m)$: permits the archival Phase-III split route used by the recovered historical rows | `true` | `true` | `true` | `true` | `true` | `true` | `true` |
| \path{--phase3-runtime-split-selection-mode} | Discrep. | $C_{\rm split}(m)$: child-set representative policy inside the SNAKE Phase-III archival split | \path{proxy_child_set_preselection} | \path{proxy_child_set_preselection} | \path{proxy_child_set_preselection} | \path{proxy_child_set_preselection} | \path{proxy_child_set_preselection} | \path{proxy_child_set_preselection} | \path{archival_child_set_forward_v1} |
| \path{--phase3-runtime-split-max-subset-size} | Discrep. | $C_{\rm split}(m)$: maximum Pauli-child subset size produced by the SNAKE Phase-III split; active 2026-07-06 POWELL visible-row recovery keeps the resolved visible-row cap `1`; cap `3` is historical/diagnostic only unless explicitly reopened | `3` | `3` | `3` | `3` | `3` | `3` | `1` |
| \path{--phase3-runtime-split-child-set-symmetry-policy} | Discrep. | $C_{\rm split}(m)$: symmetry policy requested for SNAKE Phase-III child-set construction. Current visible-row recovery uses CLI `hard_guard`, while selected child-set representatives preserve historical missing-spec/skip-pass metadata. | \path{parent} | \path{parent} | \path{parent} | \path{parent} | \path{parent} | \path{parent} | \path{hard_guard} |
| \path{--adapt-child-pool-expansion-mode} | SAME | $C_{\rm split}(m)$: alternative pre-Phase-I child-pool expansion policy; off because SNAKE child splitting is Phase-III only | `-` | `-` | `-` | `-` | `-` | `-` | `off` |
| \path{--shared-pauli-pool-mode} | SAME | $\mathcal R_0$: shared/global Pauli-child pool route; off because this contract uses SNAKE Phase-III split and comparator runtime split | `-` | `-` | `-` | `-` | `-` | `-` | `off` |
| \path{--shared-pauli-pool-symmetry-policy} | SAME | $\mathcal R_0$: shared/global Pauli-child pool symmetry policy; unused when shared pool is off | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| \path{--shared-pauli-pool-max-subset-size} | SAME | $C_{\rm split}(m)$: shared/global child-set size cap; unused when shared pool is off | `-` | `-` | `-` | `-` | `-` | `-` | `-` |
| \path{generic_adapt_runtime_split_mode} | New comparator contract | $C_{\rm split}(m)$: Geo/append comparator Pauli-child split policy applied to their candidate pool | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | \path{shortlist_pauli_children_v1} |
| \path{generic_adapt_runtime_split_symmetry_policy} | New comparator contract | $C_{\rm split}(m)$: Geo/append comparator hard-guard symmetry policy for child splitting | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | \path{hard_guard} |
| \path{generic_adapt_runtime_split_max_subset_size} | New comparator contract | $C_{\rm split}(m)$: Geo/append comparator singleton child-set cap | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `1` |

## Optimizer, Beam, Stopping

| Setting | Discrep. | Paper-I LaTeX symbol / role | WW | IW | SW | WS | IS | SS | Suggested canonical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \path{--adapt-inner-optimizer} | Discrep. | $\theta_{k+1}$: inner reoptimization after admitting $\mathcal B$ and pruning $\mathcal D$ | `SPSA` | `SPSA` | `SPSA` | `SPSA` | `SPSA` | `SPSA` | `optimizer overlay` |
| \path{--adapt-maxiter} | SAME | $\min_\theta E(\theta)$: budget for inner variational minimization after each update | `200` | `200` | `200` | `200` | `200` | `200` | `200` |
| \path{--adapt-final-refit-maxiter} | SAME | $\theta$: budget for final full-coordinate refit at the reported ansatz | `200` | `200` | `200` | `200` | `200` | `200` | `200` |
| \path{--adapt-max-depth} | SAME | $k$: outer-loop cap on accepted ansatz depth or prefix length | `30` | `30` | `30` | `30` | `30` | `30` | `30` |
| \path{--adapt-eps-grad} | SAME | $g$: gradient-plateau stopping threshold for candidate gradients | `5e-7` | `5e-7` | `5e-7` | `5e-7` | `5e-7` | `5e-7` | `5e-7` |
| \path{--adapt-eps-energy} | SAME | $E(\theta)$: energy-change stopping threshold for the outer adaptive loop | `1e-9` | `1e-9` | `1e-9` | `1e-9` | `1e-9` | `1e-9` | `1e-9` |
| \path{--adapt-seed} | SAME | No Paper-I symbol: stochastic implementation seed | `7` | `7` | `7` | `7` | `7` | `7` | `7` |
| \path{--adapt-state-backend} | SAME | $g,F,h,\mathbf R$: state/estimator backend used to evaluate gradients, metric scales, curvatures, and Schur blocks | `compiled` | `compiled` | `compiled` | `compiled` | `compiled` | `compiled` | `compiled` |
| \path{--adapt-beam-live-branches} | SAME | $N$: beam branch cap for near-degenerate candidate or deletion branches | `3` | `3` | `3` | `3` | `3` | `3` | `3` |
| \path{--adapt-beam-children-per-parent} | SAME | $N_{\rm child}$: child branch count generated from each live parent branch | `2` | `2` | `2` | `2` | `2` | `2` | `2` |
| \path{--adapt-beam-terminated-keep} | SAME | $N_{\rm term}$: cap on retained terminated branches in beam bookkeeping | `3` | `3` | `3` | `3` | `3` | `3` | `3` |
| \path{--adapt-drop-floor} | SAME | No Paper-I symbol: implementation plateau/drop gate | `-1` | `-1` | `-1` | `-1` | `-1` | `-1` | `-1` |
| \path{--adapt-drop-patience} | SAME | No Paper-I symbol: implementation plateau/drop patience related to stopping paragraph | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| \path{--adapt-drop-min-depth} | SAME | $k$: minimum depth before implementation drop gate may act | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| \path{--adapt-grad-floor} | SAME | $g$: implementation gradient floor in stopping logic | `-1` | `-1` | `-1` | `-1` | `-1` | `-1` | `-1` |
| \path{--adapt-no-repeats} | Discrep. | $\mathcal R_t$: duplicate-record policy inside candidate sets | `true` | `true` | `true` | `-` | `-` | `-` | `true` |
| \path{--phase3-enable-rescue} | SAME | No Paper-I symbol: implementation rescue fallback around Phase-III selection | `true` | `true` | `true` | `true` | `true` | `true` | `true` |
| \path{--phase3-lifetime-cost-mode} | SAME | $K_t(r)$: enables/disables implementation lifetime multiplier attached to cost penalty; off in these rows | `off` | `off` | `off` | `off` | `off` | `off` | `off` |

## SPSA Settings

| Setting | Discrep. | Paper-I LaTeX symbol / role | WW | IW | SW | WS | IS | SS | Suggested canonical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \path{--adapt-spsa-a} | SAME | $\theta$: SPSA gain-sequence scale for inner optimization; no SNAKE scoring symbol | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` |
| \path{--adapt-spsa-c} | SAME | $\theta$: SPSA perturbation scale for inner optimization; no SNAKE scoring symbol | `0.02` | `0.02` | `0.02` | `0.02` | `0.02` | `0.02` | `0.02` |
| \path{--adapt-spsa-alpha} | SAME | $\theta$: SPSA gain decay exponent for inner optimization | `0.602` | `0.602` | `0.602` | `0.602` | `0.602` | `0.602` | `0.602` |
| \path{--adapt-spsa-gamma} | SAME | $\theta$: SPSA perturbation decay exponent for inner optimization | `0.101` | `0.101` | `0.101` | `0.101` | `0.101` | `0.101` | `0.101` |
| \path{--adapt-spsa-A} | SAME | $\theta$: SPSA stability offset for inner optimization | `5.0` | `5.0` | `5.0` | `5.0` | `5.0` | `5.0` | `5.0` |
| \path{--adapt-spsa-avg-last} | SAME | $\theta$: implementation averaging over final SPSA iterates; no SNAKE scoring symbol | `0` | `0` | `0` | `0` | `0` | `0` | `0` |
| \path{--adapt-spsa-eval-repeats} | SAME | $S$: repeated objective evaluations per SPSA probe; contributes to estimator work if counted | `1` | `1` | `1` | `1` | `1` | `1` | `1` |
| \path{--adapt-spsa-eval-agg} | SAME | No Paper-I symbol: aggregation rule for repeated SPSA objective evaluations | `mean` | `mean` | `mean` | `mean` | `mean` | `mean` | `mean` |
| \path{--adapt-spsa-callback-every} | SAME | No Paper-I symbol: logging/callback cadence | `5` | `5` | `5` | `5` | `5` | `5` | `5` |
| \path{--adapt-spsa-parallel-evaluations} | Discrep. | No Paper-I score symbol: parallelized SPSA objective probes affect runtime, not $S_{t,\ell}(r)$ | `2` | `2` | `2` | `4` | `4` | `8` | `2 vs 8 parity check; use 8 if equivalent` |

## Shortlists And Phase Scoring

| Setting | Discrep. | Paper-I LaTeX symbol / role | WW | IW | SW | WS | IS | SS | Suggested canonical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \path{--phase0-pilot-max-records} | SAME | $S_{0,\ell}(r)$ and $\mathcal R_1$: cap on Phase-0 survivors entering Phase I | `96` | `96` | `96` | `96` | `96` | `96` | `96` |
| \path{--phase1-shortlist-size} | SAME | $\mathcal R_1$: cap on records retained after Phase I threshold or fallback | `24` | `24` | `24` | `24` | `24` | `24` | `24` |
| \path{--phase2-shortlist-fraction} | SAME | $\mathcal R_2$: fractional cap for the Phase-II candidate set after Phase-I ranking | `0.25` | `0.25` | `0.25` | `0.25` | `0.25` | `0.25` | `0.25` |
| \path{--phase2-shortlist-size} | SAME | $\mathcal R_2$: absolute cap for the Phase-II candidate set | `12` | `12` | `12` | `12` | `12` | `12` | `12` |
| \path{--adapt-insertion-mode} | New canonical | $p$ in $r=(m,p)$: forces candidate insertion-position evaluation every accepted iteration | `-` | `-` | `-` | `-` | `-` | `-` | \path{always} |
| \path{--phase1-probe-max-positions} | SAME | $r=(m,p)$: candidate-position domain cap for insertion index $p$ | `999999` | `999999` | `999999` | `999999` | `999999` | `999999` | `999999` |
| \path{--phase1-trough-margin-ratio} | SAME | $p$ in $r=(m,p)$: implementation trigger for probing more positions; no displayed Paper-I symbol | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |
| \path{--phase2-rho} | Discrep. | $\rho$: trust-region radius in $\Delta E_1(r)$, $\Delta E_2(r)$, and $\Delta E_3(r)$ | `0.25` | `0.25` | `0.5` | `0.25` | `0.25` | `0.5` | `0.5` |
| \path{--phase2-lambda-H} | SAME | $\lambda_H$ and $\mathbf R_\lambda=\mathbf R+\lambda_H I$: Schur/Hessian damping | `1e-6` | `1e-6` | `1e-6` | `1e-6` | `1e-6` | `1e-6` | `1e-6` |
| \path{--phase2-gamma-N} | SAME | $\mathcal N_2(r),\mathcal N_3(r)$: exponent applied to novelty factor in implementation | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |
| \path{--phase2-frontier-ratio} | SAME | $\tau_{2,\ell}$: implementation approximation to Phase-II lane threshold | `0.9` | `0.9` | `0.9` | `0.9` | `0.9` | `0.9` | `0.9` |
| \path{--phase3-frontier-ratio} | SAME | $\tau_{3,\ell}$: implementation approximation to Phase-III lane threshold | `0.9` | `0.9` | `0.9` | `0.9` | `0.9` | `0.9` | `0.9` |
| \path{--phase2-novelty-mode} | SAME | $\mathcal N_2(r)=1-q^\top G^+q/F$: collective-span residual novelty | `collective_span_v1` | `collective_span_v1` | `collective_span_v1` | `collective_span_v1` | `collective_span_v1` | `collective_span_v1` | `collective_span_v1` |
| \path{--hardware-resolution-mode} | SAME | $\widehat C_x(r)$ and $K_t(r)$: mode for resolving hardware-cost proxies before normalization | `ideal` | `ideal` | `ideal` | `ideal` | `ideal` | `ideal` | `ideal` |

## Windows And Schur Geometry

| Setting | Discrep. | Paper-I LaTeX symbol / role | WW | IW | SW | WS | IS | SS | Suggested canonical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \path{--adapt-reopt-policy} | Discrep. | $W_r$: chooses local refit-coordinate set for inner reoptimization | `windowed` | `windowed` | `windowed` | `windowed` | `windowed` | `windowed` | `full` |
| \path{--adapt-window-size} | Discrep. | $W_r\subseteq\{1,\ldots,\lvert\boldsymbol\theta_k\rvert\}$: explicit full-ansatz guard for any window-derived helper path | `4` | `4` | `999999` | `4` | `4` | `16` | `99` |
| \path{--adapt-window-topk} | Discrep. | $W_r$: top-coupled coordinates retained for local refit and Schur work when a windowed policy is active | `4` | `4` | `999999` | `4` | `4` | `16` | `0` |
| \path{--adapt-full-refit-every} | Discrep. | $\theta$: explicit full-refit cadence guard retained for command parity; full policy already reoptimizes all coordinates each accepted iteration | `8` | `8` | `8` | `8` | `8` | `8` | `1` |
| \path{--adapt-final-full-refit} | SAME | $\theta$: final full-coordinate refit flag; inert under `full` but retained for source parity with current diagnostic commands | `true` | `true` | `true` | `true` | `true` | `true` | `true` |
| \path{--phase3-selector-policy} | SAME | $\mathcal R_3$ and $B^\star$: Phase-III selector producing retained records and admitted batch | `algebraic_nested_v1` | `algebraic_nested_v1` | `algebraic_nested_v1` | `algebraic_nested_v1` | `algebraic_nested_v1` | `algebraic_nested_v1` | `algebraic_nested_v1` |
| \path{--phase3-selector-geometry-mode} | SAME | $F^\star,h^\star,q^\star,\mathcal N_3(r)$: enables Schur-reduced geometry | `reduced` | `reduced` | `reduced` | `reduced` | `reduced` | `reduced` | `reduced` |
| \path{--phase3-window-relaxation-mode} | SAME | $\delta\theta^\star=-\alpha\mathbf R^{-1}c$: controls reduced-window solve over $W_r$ | `reduced` | `reduced` | `reduced` | `reduced` | `reduced` | `reduced` | `reduced` |
| \path{--phase3-geometry-window-size} | Discrep. | $W_r$: candidate-specific Schur geometry window size; `99` covers the full existing ansatz at the Paper-I HH depth scale | `4` | `4` | `0` | `4` | `4` | `16` | `99` |
| \path{--adapt-schur-warm-start-mode} | SAME | $\delta\theta^\star$: uses append/prune Schur responses as inner-optimizer warm starts | `append-prune` | `append-prune` | `append-prune` | `append-prune` | `append-prune` | `append-prune` | `append-prune` |

## Batching

Canonical forward runs disable Phase-II and Phase-III batching.  Batch-specific
selector modes, target sizes, caps, near-degeneracy ratios, rank tolerances, and
additivity tolerances in the table below are therefore inactive for canonical
runs and should be read as diagnostic defaults only if the user explicitly
reopens a batching ablation.

| Setting | Discrep. | Paper-I LaTeX symbol / role | WW | IW | SW | WS | IS | SS | Suggested canonical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \path{--phase2-enable-batching} | Discrep. | $\mathcal B\subseteq\mathcal R_3$: allows construction of candidate batch sets from Phase-II survivors | `true` | `true` | `true` | `true` | `true` | `true` | `false` via \path{--phase2-no-batching} |
| \path{--phase3-enable-batching} | Discrep. | $B^\star\in\arg\max_{B\subseteq\mathcal R_3}\Delta E_3(B)/K_3(B)$: enables final batch argmax approximation | `true` | `true` | `true` | `true` | `true` | `true` | `false` via \path{--phase3-no-batching} |
| \path{--phase2-batch-target-size} | Discrep. | $\mathcal B$: target number of records in greedy batch construction | `8` | `8` | `8` | `8` | `8` | `8` | `off` |
| \path{--phase2-batch-size-cap} | Discrep. | $\mathcal B$: hard cap on number of records in a batch | `16` | `16` | `16` | `16` | `16` | `16` | `off` |
| \path{--phase3-batch-selection-mode} | SAME | $G_B^\star,H_B^\star,\Delta E_3(B)$: batch geometry mode | `reduced_plane` | `reduced_plane` | `reduced_plane` | `reduced_plane` | `reduced_plane` | `reduced_plane` | `reduced_plane` |
| \path{--phase3-batch-prefilter-mode} | SAME | $\mathcal B$: optional conservative prefilter on eligible batches; off here | `off` | `off` | `off` | `off` | `off` | `off` | `off` |
| \path{--phase2-batch-near-degenerate-ratio} | Discrep. | $S_{2,\ell}(r)$ and $\tau_{2,\ell}$: score-degeneracy threshold for considering additional records in $\mathcal B$ | `0.99824` | `0.91435` | `0.99824` | `0.98` | `0.98` | `0.91435` | `0.98` |
| \path{--phase3-batch-near-degenerate-ratio} | Discrep. | $S_{3,\ell}(r)$ and $\tau_{3,\ell}$: Phase-III score-degeneracy threshold for batch or beam candidates | `0.99824` | `0.91435` | `0.99824` | `0.98` | `0.98` | `0.91435` | `0.98` |
| \path{--phase2-batch-rank-rel-tol} | Discrep. | $G_B^\star$: relative-rank tolerance for reduced batch tangent matrix | `1.366e-4` | `7.703e-7` | `1.366e-4` | `1.910e-5` | `1.910e-5` | `7.703e-7` | `0.25` |
| \path{--phase3-batch-rank-rel-tol} | Discrep. | $G_B^\star$: Phase-III rank tolerance for reduced batch geometry | `1.366e-4` | `7.703e-7` | `1.366e-4` | `1.910e-5` | `1.910e-5` | `7.703e-7` | `0.25` |
| \path{--phase2-batch-additivity-tol} | Discrep. | $\Delta E_3(B)$: implementation compatibility tolerance comparing joint and single-record gains | `0.6663` | `0.01028` | `0.6663` | `0.09993` | `0.09993` | `0.01028` | `0.25` |
| \path{--phase3-batch-additivity-tol} | Discrep. | $B^\star$: Phase-III compatibility tolerance for greedy approximation to the batch argmax | `0.6663` | `0.01028` | `0.6663` | `0.09993` | `0.09993` | `0.01028` | `0.25` |

## Pruning

| Setting | Discrep. | Paper-I LaTeX symbol / role | WW | IW | SW | WS | IS | SS | Suggested canonical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \path{--phase1-prune-enabled} | SAME | $\mathcal D$ in $U_k(\theta_k)\oplus\mathcal B\ominus\mathcal D$: enables ablation-admissible deletion set | `true` | `true` | `true` | `true` | `true` | `true` | `true` |
| \path{--phase1-prune-policy} | SAME | $W_s,\Lambda_s,s_j^\star$: uses recoverability ladder for deletion screening | \path{recoverability_ladder_v1} | \path{recoverability_ladder_v1} | \path{recoverability_ladder_v1} | \path{recoverability_ladder_v1} | \path{recoverability_ladder_v1} | \path{recoverability_ladder_v1} | \path{recoverability_ladder_v1} |
| \path{--phase1-prune-mode} | SAME | $d_j\in U_k(\theta_k)$: runs generator ablation live and at final checkpoint | `both` | `both` | `both` | `both` | `both` | `both` | `both` |
| \path{--phase1-prune-schur-nomination-route} | Discrep. | $\mathcal D$: Schur surrogate used to nominate deletion candidates before rollback-safe testing | historical default | historical default | historical default | historical default | historical default | historical default | \path{metric_regularized_v1} |
| \path{--phase1-prune-metric-schur-mu} | Discrep. | $\mu$: ridge parameter for metric-regularized prune nomination | historical default | historical default | historical default | historical default | historical default | historical default | `0.01` |
| \path{--phase1-prune-metric-schur-cost-weighting} | Discrep. | $\mathcal D$: cost denominator for metric-regularized prune nomination | historical default | historical default | historical default | historical default | historical default | historical default | \path{ansatz_entry_denominator_v1} |
| \path{--phase1-prune-fraction} | Discrep. | $\mathcal D$: implementation cap on deletion-candidate nomination before testing membership | `0.41019` | `0.19310` | `0.41019` | `0.33923` | `0.33923` | `0.19310` | `0.4` |
| \path{--phase1-prune-max-candidates} | SAME | $d_j$: maximum deletion candidates tested per prune pass | `6` | `6` | `6` | `6` | `6` | `6` | `6` |
| \path{--phase1-prune-max-regression} | SAME | $\epsilon_{\rm del}$ and $\Delta E(d_j)\le\epsilon_{\rm del}$: measured deletion-loss tolerance | `1e-8` | `1e-8` | `1e-8` | `1e-8` | `1e-8` | `1e-8` | `1e-8` |
| \path{--phase1-prune-amplitude-witness-optional} | SAME | $\chi(d_j)$: optional amplitude-history eligibility diagnostic, not deletion authority here | `true` | `true` | `true` | `true` | `true` | `true` | `true` |
| \path{--phase1-prune-collapse-peak-abs-min} | Discrep. | $\chi(d_j)$: optional amplitude-collapse peak threshold | `-` | `-` | `2e-3` | `-` | `-` | `-` | `off`; `2e-3` diagnostic |
| \path{--phase1-prune-collapse-current-abs-max} | Discrep. | $\chi(d_j)$: optional current-amplitude threshold | `-` | `-` | `5e-4` | `-` | `-` | `-` | `off`; `5e-4` diagnostic |
| \path{--phase1-prune-collapse-ratio} | Discrep. | $\chi(d_j)$: optional amplitude-collapse ratio | `-` | `-` | `0.2` | `-` | `-` | `-` | `off`; `0.2` diagnostic |
| \path{--phase1-prune-collapse-min-abs-drop} | Discrep. | $\chi(d_j)$: optional minimum amplitude drop | `-` | `-` | `2e-3` | `-` | `-` | `-` | `off`; `2e-3` diagnostic |
| \path{--phase1-prune-collapse-min-observations} | Discrep. | $\chi(d_j)$: minimum history length for optional amplitude-collapse diagnostic | `-` | `-` | `4` | `-` | `-` | `-` | `off`; `4` diagnostic |

## Phase Maturity And Hysteresis

| Setting | Discrep. | Paper-I LaTeX symbol / role | WW | IW | SW | WS | IS | SS | Suggested canonical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \path{--phase1-maturity-cap-min} | Discrep. | $\mathcal R_1$: nonbinding maturity cap sentinel; physical-lane route clamps to the effective Phase-I shortlist cap | `8` | `12` | `12` | `8` | `8` | `24` | `999999` |
| \path{--phase1-maturity-cap-max} | Discrep. | $\mathcal R_1$: nonbinding maturity cap sentinel; physical-lane route clamps to the effective Phase-I shortlist cap | `24` | `32` | `32` | `24` | `24` | `64` | `999999` |
| \path{--phase2-maturity-cap-min} | Discrep. | $\mathcal R_2$: nonbinding maturity cap sentinel; physical-lane route clamps to the effective Phase-II shortlist cap | `6` | `8` | `8` | `6` | `6` | `12` | `999999` |
| \path{--phase2-maturity-cap-max} | Discrep. | $\mathcal R_2$: nonbinding maturity cap sentinel; physical-lane route clamps to the effective Phase-II shortlist cap | `16` | `24` | `24` | `16` | `16` | `48` | `999999` |
| \path{--phase3-maturity-cap-min} | Discrep. | $\mathcal R_3$: nonbinding maturity cap sentinel; physical-lane route clamps to the effective Phase-III shortlist cap | `4` | `4` | `4` | `4` | `4` | `8` | `999999` |
| \path{--phase3-maturity-cap-max} | SAME | $\mathcal R_3$: nonbinding maturity cap sentinel; physical-lane route clamps to the effective Phase-III shortlist cap | `10` | `10` | `10` | `10` | `10` | `10` | `999999` |
| \path{--phase-maturity-shot-min} | Discrep. | No Paper-I symbol: minimum liveness/probe count for implementation phase-retirement logic | `1` | `2` | `1` | `1` | `1` | `1` | `1` |
| \path{--phase-maturity-shot-max} | Discrep. | No Paper-I symbol: maximum liveness/probe count for implementation phase-retirement logic | `2` | `8` | `4` | `1` | `1` | `4` | `1` |
| \path{--phase1-maturity-shot-cap} | Discrep. | $\mathcal R_1$: Phase-I liveness/probe cap controlling activity updates | `2` | `4` | `2` | `1` | `1` | `2` | `1` |
| \path{--phase2-maturity-shot-cap} | Discrep. | $\mathcal R_2$: Phase-II liveness/probe cap controlling activity updates | `2` | `8` | `4` | `1` | `1` | `4` | `1` |
| \path{--phase3-maturity-shot-cap} | Discrep. | $\mathcal R_3$: Phase-III liveness/probe cap controlling activity updates | `2` | `8` | `4` | `1` | `1` | `4` | `1` |
| \path{--phase-live-hysteresis-enabled} | SAME/defaulted | No Paper-I symbol: implementation hysteresis for whether Phase-II/III remains live; `-` means code default | `true` | `true` | `-` | `-` | `-` | `-` | `false` via \path{--phase-live-hysteresis-disabled} |
| \path{--phase2-null-nrem-high-threshold} | SAME/defaulted | $\mathcal R_2$: implementation threshold on normalized remaining records for nulling Phase II; `-` means code default | `0.0` | `0.0` | `-` | `-` | `-` | `-` | `off`; default diagnostic |
| \path{--phase2-live-nrem-low-threshold} | SAME/defaulted | $\mathcal R_2$: implementation threshold on normalized remaining records for reactivating Phase II; `-` means code default | `0.25` | `0.25` | `-` | `-` | `-` | `-` | `off`; default diagnostic |
| \path{--phase3-null-nrem-high-threshold} | SAME/defaulted | $\mathcal R_3$: implementation threshold on normalized remaining records for nulling Phase III; `-` means code default | `0.75` | `0.75` | `-` | `-` | `-` | `-` | `off`; default diagnostic |
| \path{--phase3-live-nrem-low-threshold} | SAME/defaulted | $\mathcal R_3$: implementation threshold on normalized remaining records for reactivating Phase III; `-` means code default | `1.25` | `1.25` | `-` | `-` | `-` | `-` | `off`; default diagnostic |
| \path{--phase2-hysteresis-steps} | SAME/defaulted | $\mathcal R_2$: consecutive liveness decisions needed before changing Phase-II status; `-` means code default | `2` | `2` | `-` | `-` | `-` | `-` | `off`; default diagnostic |
| \path{--phase3-hysteresis-steps} | SAME/defaulted | $\mathcal R_3$: consecutive liveness decisions needed before changing Phase-III status; `-` means code default | `1` | `1` | `-` | `-` | `-` | `-` | `off`; default diagnostic |

## Implementation Versus Paper-I Cost Description

| Item | Finding |
| --- | --- |
| Cost lambdas | The manuscript names explicit `lambda_2q, lambda_d, lambda_1q, lambda_theta, lambda_shot`. The result commands did not set those explicit lambdas; the implementation used compatibility aliases. |
| Phase-I cost | Phase I used `lambda_compile=0.05` and `lambda_measure=0.02`, giving a different effective `K_1` from `K_2,K_3`. If Paper I reads as using one common lambda vector across phases, that is not what the result settings did. |
| Shot coefficient | The per-regime `--phase2-w-shot` values did not become the effective `lambda_shot` when below `0.15`; the alias rule used `max(phase2-w-shot, phase2-w-group, phase2-w-reuse)`, making `lambda_shot=0.15` for all six rows. |
| Backend depth weight | `--phase3-backend-w-depth` is a backend-oracle compile feature weight, not the Paper-I `lambda_d` coefficient in `K_t`. |
| Pool parity | Current comparison contract uses unfiltered `full_meta` for SNAKE, Geo-ADAPT, and append-only ADAPT. Do not apply the `full_meta_minus_hva` class filter in this pass. |
| Pauli children | Current comparison contract uses the recovered Paper-I SNAKE Phase-III archival child-set split with subset cap `1`. Geo-ADAPT and append-only ADAPT still use generic comparator runtime split unless their comparator cap is separately reopened. Child-subset cap `3` is historical/diagnostic ancestry only, and canonical SNAKE disables Phase-II/III batching unless the user explicitly reopens a batching ablation. |
| Shot proxy history | These artifacts predate the later manuscript-faithful fixed-precision shot-proxy code change. Do not assume the historical selector shot term exactly equals the current Paper-I formula unless the run artifact/code version is separately audited. |
