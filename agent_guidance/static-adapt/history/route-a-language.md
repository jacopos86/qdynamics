# Route-A Compatibility Language Map

Purpose: translate the legacy Paper-I `route_a` compatibility surface into the
repo fields that must be checked in source JSONs, run manifests, and table
provenance. Read `route-identities.md` first for the active JR-SNAKE,
FM-SNAKE, and SR-SNAKE method-family registry.

Scope: Paper-I/static ADAPT artifacts that still report Route A /
`paper_i_production_v1`. This file is a translation and verification aid;
`route_a` alone does not identify JR-SNAKE, FM-SNAKE, or SR-SNAKE. Executable
policy still lives in source settings and locked artifacts.

## Read This First

- Use **Pauli-child / child-set exploration** for the child/split mechanism.
- Resolve the stable family first using `route-identities.md`; then use this
  map to interpret the remaining Route-A-era fields.
- Map in this direction: **symbolic math term -> math-paper phrase -> repo field/artifact field**.
- Do not infer field-level evidence from manuscript prose alone. Verify the source artifact when a table cell or report claim depends on a mechanism.
- Compact promotion JSONs may record only a route contract. Field-level proof usually comes from the underlying `current.json`, `summary.json`, or run manifest.

## Route-A Visual Map

This diagram is a field-translation overview for the legacy umbrella. It is
not the algorithm diagram for all three active families: SR-SNAKE has Phase 0
off and singleton admission, JR-SNAKE owns the joint batch funnel, and
FM-SNAKE owns its query-closed formal-manifold route.

```mermaid
flowchart TD
    A[HH problem and n_ph^work] --> B[Problem-local full_meta adaptive pool]
    B --> C[Candidate-position records r=(m,p)]
    C --> D[Phase 0 pilot and physical operator-type lanes]
    D --> E[Phase I gain / resource score]
    E --> F[Phase II collective-span novelty]
    F --> G[Pauli-child / child-set and single or ordered-batch beam exploration]
    G --> H[Phase III reduced-window Schur geometry]
    H --> I[algebraic_nested_v1 selector]
    I --> J[Append selected record or child set]
    J --> K[Local refit]
    K --> L[Generator ablation: rollback-safe prune]
    L --> M[Recoverability checks; amplitude shortlist/witness optional]
    M --> N[Manifest / source JSON / table evidence]

    X[Not current Route A unless explicitly ablation] --> X1[old winning or reduced pools]
    X --> X2[legacy pairwise novelty]
    X --> X3[missing algebraic_nested_v1 evidence]
    X --> X4[missing recoverability_ladder_v1 evidence]
    X --> X5[forecaster/exact-assisted decision logic]
```

## Core Translation Table

| Symbolic math term | Math-paper phrase | Repo field / artifact field | Route-A status |
|---|---|---|---|
| `r=(m,p)` | candidate-position record; generator plus insertion position | candidate record / `CandidateFeatures`; selected labels with position metadata | Required. Do not collapse to bare generator scoring. |
| `m` | generator | pool/operator label, candidate generator metadata | Required as part of `r=(m,p)`. |
| `p` | insertion position | position id / insertion metadata; position-domain controls | Required as part of `r=(m,p)`. |
| `R_k(t)`, `S_k(t)` | staged candidate universe and shortlists | `phase0`, `phase1`, `phase2`, `phase3`; shortlist/frontier fields | Required. |
| `Delta E_k(r;t)` | phase-resolved gain proxy | phase score components; energy/gain fields in candidate records | Required. |
| `N_2`, `N_3` | tangent novelty / reduced-window novelty | `phase2_novelty_mode=collective_span_v1`; `phase3_novelty_ablation_mode=off`; exponent-style novelty fields should remain `1` | Required unless novelty ablation. Do not set a novelty exponent to `0` to preserve novelty; that neutralizes multiplicative novelty. |
| `K_k(r;t)` | resource-cost burden | compile/depth/shot/resource weights and score denominators | On by default, but user may request no-cost/no-resource mode, especially for HH. Optuna must keep the chosen cost mode fixed. |
| `F_r^*`, `h_r^*`, `q_r^*` | Phase-III reduced-window Schur geometry | `phase3_selector_geometry_mode=reduced`; `phase3_window_relaxation_mode=reduced` | Required unless reduced-geometry ablation. |
| `ell in L` | physical operator-type lane metadata / lane-wise shortlist for HH | `static_lane_route=physical_operator_type`; physical lane fields | Required for the current Paper-I HH Route-A line. Support/commutation algebraic lanes are diagnostic unless explicitly reopened. |
| `C_split(m)` | Pauli-child split family / child-set candidates | `phase3_runtime_split_mode=shortlist_pauli_children_v1`; `child_set`; `child_generator_ids` | Feature surface. Report exact observed fields. |
| `B_child` | beam children per parent | `adapt_beam_children_per_parent`; `adapt_beam_live_branches`; `adapt_beam_terminated_keep` | Feature surface; branch caps may be Optuna-tuned. |
| `G_B` or reduced batch plane | reduced-plane batching / joint admission geometry | `phase3_enable_batching`; `phase3_batch_selection_mode=reduced_plane`; batch target/cap fields; commutation-gate fields if present | Feature surface for batching ablations. The current Paper-I HH canonical route disables Phase-II/III batching. |
| batch admission path | SR-SNAKE singleton admission versus JR-SNAKE ordered-batch proposal/admission | `beam_structural_mode`; `batch_selection_mode`; `phase2_batch_selection_mode`; `phase3_batch_selection_mode`; `batch_size`; `beam_cost_K`; `survival_policy_version`; `prune_key_version` | Feature surface. `maxB=1` only proves no multi-record batch was admitted; it does not establish SR-SNAKE if `beam_structural_mode=ordered_batch_admission`. |
| `d_j in O_t` | existing ansatz coordinate/generator selected for ablation | prune candidate / selected ansatz entry | Required for generator-ablation language. |
| `Lambda_j^ominus`, `Delta E_j^{ominus,refit}` | rollback-safe generator ablation; prune by recoverability | `phase1_prune_enabled=true`; `phase1_prune_policy=recoverability_ladder_v1`; `phase1_prune_mode=both` | Required for current Route A unless no-prune ablation. |
| amplitude shortlist / witness | optional prune shortlist or amplitude-collapse witness | `phase1_prune_amplitude_witness_required`, amplitude shortlist fields if present | Not required in the preferred profile unless the user explicitly asks for that ablation/profile. |
| `n_ph^work` | algorithmic working phonon cutoff | `n_ph_work`, `boson_cutoff`, `n_ph_max` | Always report for HH. |
| `n_ph^ref`, `n_ph^ED`, `n_ph^eval` | compatibility names for the exact-reference/evaluation cutoff | `n_ph_ref`, `exact_reference_boson_cutoff`, `n_ph_eval` | Paper-I model-comparison accuracy is same-cutoff: every present compatibility field must resolve to `n_ph_work`. A higher cutoff belongs only to a separately requested cutoff-sensitivity study. |
| `full_meta` | problem-local full-meta operator pool, including HVA when available | `adapt_pool=full_meta`; `route_base_pool_key=full_meta` | Global pool definition. Do not redefine this name. |
| `full_meta_minus_hva` | Historical Paper-I HH adaptive pool profile: full-meta base with HVA excluded from adaptive selection | `adapt_pool=full_meta`; `adapt_pool_class_filter_json=agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json`; `selected_logical_route=standard` | Historical/diagnostic for the current HH route. Current Paper-I HH SNAKE, Geo-ADAPT, and append-only ADAPT use unfiltered `full_meta` with HVA included unless a visible source row explicitly says otherwise. |
| forecast / future exact quantities | diagnostic forecaster, not route identity | `exact_step_forecast`, `forecast_*`, exact future fields | Not Route A decision evidence; diagnostic/ablation only. |

## SR-SNAKE Stage-Language Rule

Do not summarize SR-SNAKE as `R_2=S_1` or as a batch-size-one JR route. Its
source-locked stage language is:

1. Phase I scores parent-position records in physical operator-type lanes.
2. Each retained parent produces only exact-cardinality-one Pauli children.
3. The hard fixed-sector guard runs before child scoring.
4. Exact binary-padding projection rejects zero directions and may create a
   grouped legal polynomial.
5. Deterministic projective normalization and parent-position deduplication
   retain one representative for each legal direction.
6. The archival bridge forwards one singleton representative per retained
   parent.
7. Phase II evaluates the child while inheriting the parent's physical lane.
8. Phase III receives a lane-free child population and applies the full
   active-plus-singleton response.
9. Exactly one candidate-position record is admitted.

The raw split subset cardinality is one. Padding projection can turn that raw
Pauli child into a grouped legal polynomial without converting it into a
multi-child batch. Preserve raw coefficients and split lineage in evidence.
The current repaired compatibility spelling for the forwarding bridge is
`phase3_runtime_split_selection_mode=archival_child_set_forward_v1`; the name
is archival, but the resolved method family is SR-SNAKE.

## Route-A Required Evidence Checklist

For a Paper-I SNAKE production row, verify these in the source artifact whenever possible:

```text
meta_feature_profile = paper_i_production_v1
selected_logical_route = standard
pool / route_base_pool_key = full_meta
HH adaptive class filter = None for the current unfiltered full_meta route
HVA adaptive pool membership = included in the current unfiltered full_meta route unless an explicit ablation says otherwise
phase3_selector_policy = algebraic_nested_v1
static_lane_route = physical_operator_type for HH
phase3_selector_geometry_mode = reduced
phase3_geometry_window_size = 99 for the current HH canonical route
phase3_window_relaxation_mode = reduced
phase3_novelty_ablation_mode = off
novelty exponent field, if present = 1
cost/resource mode = on by default; no-cost only when the user asks
adapt_reopt_policy = full
adapt_window_size = 99
adapt_full_refit_every = 1
phase1_prune_enabled = true
phase1_prune_policy = recoverability_ladder_v1
phase1_prune_mode = both
phase1_prune_schur_nomination_route = metric_regularized_v1
phase1_prune_metric_schur_mu = 0.01
phase1_prune_amplitude_witness_required = false/not required unless explicitly requested
phase2_enable_batching = false for the current HH canonical route
phase3_enable_batching = false for the current HH canonical route
batch commutation requirement = false/not required unless explicitly requested
beam/child exploration = enabled
```

For Pauli-child / child-set / batch/beam behavior, report the observed fields literally instead of reducing them to one slogan. Beam should be enabled in the preferred profile:

```text
phase3_runtime_split_mode
runtime_split_mode
phase3_enable_batching
phase3_batch_selection_mode
adapt_beam_children_per_parent
adapt_beam_live_branches
adapt_beam_terminated_keep
beam_structural_mode
batch_selection_mode
batch_size
beam_cost_K
survival_policy_version
prune_key_version
child_set / chosen_representation / child_generator_ids
```

## Ordered-Batch Route Labels

Use these labels only when the source artifact records the listed fields. Do
not infer them from final `batch_size` alone.

| Support label | Required observed evidence | Meaning |
|---|---|---|
| SR-SNAKE singleton admission | `beam_structural_mode=single_admission` or `stop_or_single_admission`; Phase 0 disabled; Phase-II/III batching disabled; observed `batch_size=1`; full active-plus-singleton Phase-III response recorded | The controller selects exactly one candidate-position record through the Singleton-Response SNAKE path. Do not call this route old or historical merely because compatibility fields retain those words. |
| metric-prune greedy ordered batch | metric-prune fields active; `beam_structural_mode=ordered_batch_admission`; `batch_selection_mode=greedy_reduced_plane` | A batch-ablation route constructs ordered reduced-plane batch proposals greedily before beam survival/admission. Not the current HH canonical no-batching route. |
| metric-prune combinatorial ordered batch | metric-prune fields active; `beam_structural_mode=ordered_batch_admission`; `batch_selection_mode=combinatorial_reduced_plane` | A batch-ablation route constructs ordered reduced-plane batch proposals combinatorially before beam survival/admission. Not the current HH canonical no-batching route. |
| cost-weighted beam | `beam_cost_K` present and a nontrivial beam-cost/survival comparator is recorded, such as `lambda_beam` or `survival_policy_version` | Beam survival uses a cost-aware branch comparator. If these fields are absent or zero, say ordered-batch beam, not cost-weighted beam. |

Agent audit rule:

```text
maxB=1 blocks multi-record admission, but it does not by itself establish
SR-SNAKE. Verify beam_structural_mode and
batch_selection_mode before claiming a batch/prune-only ablation.
```

Operational implication:

```text
B_max = 1 controls the final admitted batch cardinality.
beam_structural_mode and batch_selection_mode control how the singleton
proposal was generated, ranked, and passed through beam survival.
```

Therefore, a `B_max=1` run can still differ from SR-SNAKE if it
uses `beam_structural_mode=ordered_batch_admission`. The final admitted set has
one record, but that one record may have been selected from a different ordered
proposal frontier than the SR-SNAKE singleton-response route.

For weak-weak reproduction diagnostics, isolate variables in this order:

1. **SR-SNAKE admission plus new beam**: keep the SR-SNAKE
   singleton-response admission route and vary only the beam survival
   comparator;
2. **new ordered batch plus old beam**: keep the old/no-cost beam survival
   comparator and vary only `beam_structural_mode` plus
   `batch_selection_mode`;
3. **metric prune only**: keep both the SR-SNAKE singleton admission route and the
   old beam comparator, and vary only the prune/recoverability machinery;
4. **combined new route**: metric prune plus ordered-batch admission plus
   cost-weighted beam.

Each row must be source-locked against the same Paper-I visible/source row, and
any non-requested setting drift makes the row diagnostic rather than a
one-variable ablation.

## Evidence Wording Rules

Use these distinctions in run notes, table comments, and agent replies:

- **Route contract present**: a compact file says `profile=paper_i_production_v1` or `route_contract`. This is useful but may not prove every field.
- **Field-level evidence present**: the source `current.json`, `summary.json`, or manifest includes the exact fields above.
- **Mechanism active**: the source fields show the mechanism enabled for that run or prefix.
- **Mechanism available but not active**: the profile supports the mechanism, but source fields show it off, absent, or unused.
- **Ablation/control**: a mechanism is intentionally disabled under matched Hamiltonian, seed, cutoff, encoding, backend, optimizer, and resource settings.

## Preferred Phrases

| Avoid | Use |
|---|---|
| vague child-policy shorthand | Pauli-child / child-set exploration |
| nested selector | current Phase-III selector, `algebraic_nested_v1` |
| recoverability ladder | rollback-safe generator ablation / prune-recoverability check |
| cheap route | resource-cost burden or cost-aware selector |
| full meta means winning pool | problem-local full-meta pool; do not reduce to historical winning pools |
| full_meta means no HVA everywhere | `full_meta` includes HVA when available; the current Paper-I HH route uses unfiltered `full_meta` unless an explicit ablation says otherwise |
| exact forecast route | diagnostic forecast/ablation; not decision evidence |

## Minimal Agent Reply Template

When asked whether a Table III SNAKE row used a mechanism, answer in this order:

1. State the route contract if known.
2. State the exact source artifact checked.
3. Quote the relevant field names and values, not a prose paraphrase.
4. If only a compact promotion file is local, say field-level evidence requires the underlying source JSON.

Example:

```text
The row has Route-A contract evidence, but the compact promotion JSON may not embed prune fields. Field-level proof requires the underlying current.json; look for phase1_prune_enabled=true, phase1_prune_policy=recoverability_ladder_v1, phase1_prune_mode=both, and whether phase1_prune_amplitude_witness_required is false/off/not required unless the user requested that profile.
```
