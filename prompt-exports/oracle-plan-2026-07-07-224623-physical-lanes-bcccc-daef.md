## Final Prompt
<taskname="Physical Lanes"/>
<task>
Implement a new Paper-I SNAKE/static-ADAPT route variant that preserves the existing algebraic support/commutation lane route and adds an alternate physical-operator-type lane route for phase shortlisting. The new route should define shortlist lanes by physical operator family/type: lifted UCCSD/electronic correlation, electronic current, phonon displacement, phonon squeeze/relaxation, dressed phonon/correlation/cloud/hopdrag, HVA/Hamiltonian blocks, and fallback/other. Use those lanes for phase shortlist budgeting between phases. Make the new route 3x more aggressive by default than current shortlist settings while preserving source/baseline settings otherwise. Keep the existing algebraic route as the default and do not weaken current Route A tests/semantics.
</task>

<architecture>
- `pipelines/static_adapt/algebraic_metadata.py` owns the current four algebraic lanes (`flat`, `curv`, `disj`, `mix`), exact support/commutation metadata, phase0 weak lane payloads, quota-pressure budget allocation, and phase0/phase1/phase2 lane shortlist helpers. Unknown lanes currently collapse to `mix`, so physical lanes need either generalized helpers or a separate route-aware wrapper.
- `pipelines/static_adapt/phase_shortlists.py` wraps generic phase shortlisting. `PhaseShortlistRuntime` is currently algebraic-specific (`algebraic_lane_policy_active`, algebraic pressures/thresholds). This is the main place to generalize lane key/vocabulary/telemetry while keeping current algebraic behavior.
- `pipelines/static_adapt/adapt_pipeline.py` wires CLI kwargs, pool registry/family IDs, algebraic payload attachment, phase2 inheritance/cache payloads, runtime construction, beam/SNAKE and non-beam phase1/phase2 shortlist calls, route identity, and final telemetry.
- `pipelines/scaffold/hh_continuation_types.py` defines `CandidateFeatures`; add physical lane fields here if the route needs persistent per-candidate metadata. `adapt_candidate_record_cache.py` serializes `CandidateFeatures.__dict__`, so new fields are naturally persisted but cache semantics may need a version/key review.
- `pipelines/contracts/static_provenance.py` already classifies HH full_meta labels (`classify_hh_full_meta_label`) into source-visible classes. Use it as the stable base for physical lane assignment rather than parsing labels ad hoc in the pipeline.
- `pipelines/static_adapt/cli_config.py` and `pipelines/static_adapt/optimization/phase3_policy_optuna.py` expose CLI/policy surfaces for route/profile, phase0 lane mode, phase1/phase2 shortlist caps, algebraic pressures, and phase3 selector policy. Add explicit physical-lane route flags here without changing defaults.
- `pipelines/static_adapt/route_identity.py` encodes Route A required components and production profile metadata. Add route-variant telemetry carefully; do not make the existing algebraic Route A contract mean something different.
</architecture>

<selected_context>
- `pipelines/static_adapt/algebraic_metadata.py`: current algebraic lane constants, assign/summarize logic, phase0 weak payloads, and lane quota/shortlist helpers.
- `pipelines/static_adapt/phase_shortlists.py`: full runtime/wrapper implementation for phase1 and phase2 lane shortlists.
- `pipelines/static_adapt/adapt_pipeline.py`: sliced import/kwarg validation, pool registry, algebraic payload, phase2 inheritance/cache, runtime construction, phase0 hook, route identity, beam/SNAKE phase1/phase2 shortlist calls, non-beam phase1/phase2 calls, final telemetry.
- `pipelines/static_adapt/cli_config.py`: sliced CLI option and kwargs mapping surfaces.
- `pipelines/static_adapt/optimization/phase3_policy_optuna.py`: sliced policy defaults, `StaticPolicy`, policy-to-CLI, default trial params, CLI param parsing, sanitization.
- `pipelines/contracts/static_provenance.py`: full HH full_meta class taxonomy and label classifier.
- `pipelines/scaffold/hh_continuation_types.py`: `CandidateFeatures` fields including current algebraic and phase0 algebraic fields.
- `pipelines/static_adapt/route_identity.py`: Route A required components, observed component payloads, validation/rendering helpers.
- `MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_snake_nobatch_duplicate_promotion_20260707.json`: named promoted copy baseline manifest; source says POWELL, no batching, `full_meta_unfiltered_hva_included`, Pauli-child split, children per parent 2, beam lambda 0.005, live branches 3, subset size 1, metric prune enabled.
- `MATH/paper_facing/paper_I_static_scaffold/hh_fixed_scaffold_motif_inventory_20260614.md`: concrete motifs for electronic current, phonon displacement/squeeze, UCC correlation, and dressed phonon/correlation/cloud/hopdrag lanes.
- `agent_guidance/skills/shared/scripts/resolve_visible_settings.py`, `paper-i-run/SKILL.md`, `source-locked-sensitivity/SKILL.md`: source-lock and visible-settings resolver requirements.
- Tests selected: `test_static_adapt_phase_shortlists.py`, `test_static_adapt_algebraic_metadata.py`, `test_hh_full_meta_class_filter.py`, `test/optimization/test_phase3_policy_optuna.py`.
</selected_context>

<implementation_plan>
1. Define explicit physical lane taxonomy, preferably near the existing source-visible classifier in `pipelines/contracts/static_provenance.py` or a small new static-ADAPT metadata module that imports it. Suggested constants: `electronic_correlation`, `electronic_current`, `phonon_displacement`, `phonon_squeeze_relaxation`, `dressed_phonon_correlation`, `hva_hamiltonian_blocks`, `other`.
2. Map `classify_hh_full_meta_label(label)` classes to physical lanes. Suggested mapping: `uccsd_sing`/`uccsd_dbl` -> electronic correlation; `hh_fermionic_reusable` labels containing `current` -> electronic current, other reusable electronic hop/exchange/correlation labels -> electronic correlation; `hh_phonon_linear` and `paop_disp` -> phonon displacement; `hh_phonon_quadratic` and `hh_vlf_sq` -> phonon squeeze/relaxation; PAOP cloud/dbl/hopdrag/curdrag/hop2 and UCCSD-PAOP product classes -> dressed phonon/correlation; `hva_layer`, `hh_hamiltonian_block`, `hh_termwise_unit`, `hh_termwise_quadrature` -> HVA/Hamiltonian blocks; otherwise `other`.
3. Generalize lane shortlist helpers in `algebraic_metadata.py` so lane allocation accepts `lanes`, `lane_key`, and `fallback_lane`. Preserve public algebraic wrappers and old defaults exactly, including unknown algebraic lane -> `mix`; physical unknowns should fallback to `other`.
4. Extend `PhaseShortlistRuntime`/wrappers in `phase_shortlists.py` with route-neutral lane settings (`lane_policy_name`, `lane_key`, `lane_budget_keys`, fallback lane, summary name) or equivalent. Keep existing algebraic fields/telemetry populated for the current route.
5. Add `CandidateFeatures` physical fields, for example `physical_operator_lane`, `physical_operator_class`, `physical_operator_lane_source`, plus phase2 health/live fields if the lane-health shortlist writes them. If phase0 physical screening is implemented, add phase0 physical equivalents.
6. In `adapt_pipeline.py`, add `_run_hardcoded_adapt_vqe` kwargs and validation for proposed flags `static_lane_route="algebraic|physical_operator_type"` and `physical_lane_shortlist_aggressiveness=2|3` (default 3 when physical route is selected). Keep `static_lane_route="algebraic"` as default. Derive effective physical caps from source/base caps by factor 3: phase1 64 -> 22, phase2 12 -> 4, phase2 fraction 0.2 -> 0.0666667. Record both base and effective settings in telemetry.
7. In `adapt_pipeline.py`, add a physical payload builder from `candidate_label`/generator metadata/classifier. Attach it next to `_attach_phase1_selector_metadata`, inherit it in `_phase2_inherited_selector_updates`, add it to `_feature_base_scoring_cache_payload`, and route both beam/SNAKE and non-beam phase1/phase2 shortlist calls through the selected lane key/vocabulary.
8. Add CLI flags in `cli_config.py`: `--static-lane-route {algebraic,physical_operator_type}` and `--physical-lane-shortlist-aggressiveness {2,3}`; map them into `_run_hardcoded_adapt_vqe` kwargs. Mirror these in `phase3_policy_optuna.py` (`StaticPolicy`, defaults, `policy_to_cli_args`, `trial_params_from_cli_command`, sanitization). Do not reuse `_CANONICAL_LANE_FAMILIES`; that is a benchmark grouping, not shortlist lane taxonomy.
9. Add route/result telemetry: selected lane route, physical lane vocabulary, aggressiveness factor, base/effective shortlist caps, per-lane budgets/counts, and classifier version. If using route identity, add variant telemetry without changing existing Route A required components.
</implementation_plan>

<tests_and_runs>
Run targeted unit tests after implementation:
```bash
python3 -m pytest test/test_hh_full_meta_class_filter.py test/test_static_adapt_algebraic_metadata.py test/test_static_adapt_phase_shortlists.py test/optimization/test_phase3_policy_optuna.py
```

Add/extend tests:
- `test_hh_full_meta_class_filter.py`: physical classifier mappings for UCCSD, current, phonon displacement, squeeze/relaxation, PAOP/cloud/hopdrag/product, HVA/ham blocks, fallback.
- `test_static_adapt_algebraic_metadata.py`: generalized lane budgeting works for physical lanes; existing algebraic wrappers still pass unchanged.
- `test_static_adapt_phase_shortlists.py`: runtime can shortlist with `physical_operator_lane` and writes physical lane health while algebraic mode remains unchanged.
- `test/optimization/test_phase3_policy_optuna.py`: new flags default to algebraic route, round-trip from CLI and policy-to-CLI, sanitize invalid values.

Resolve source-visible settings before any paper-facing run; fail closed if the source map/axis/method does not match:
```bash
python3 agent_guidance/skills/shared/scripts/resolve_visible_settings.py \
  --source-map output/pdf/paper_i_hh_powell_visible_batchroute_nobatch_vs_paper1_snake_20260707/paper_i_hh_powell_visible_batchroute_nobatch_vs_paper1_snake_20260707_provenance.json \
  --target-axis regimes \
  --target-key weak_weak \
  --method snake_nobatch \
  --output-json artifacts/json/physical_lane_weak_weak_visible_settings_trace.json
```

After source lock is resolved, smoke the new route with source/base shortlist settings and physical 3x effective caps:
```bash
python3 -m pipelines.static_adapt.adapt_pipeline \
  --problem hh --adapt-pool full_meta \
  --static-route-id route_a --static-meta-feature-profile paper_i_production_v1 \
  --phase3-selector-policy algebraic_nested_v1 \
  --static-lane-route physical_operator_type \
  --physical-lane-shortlist-aggressiveness 3 \
  --phase1-shortlist-size 64 --phase2-shortlist-size 12 --phase2-shortlist-fraction 0.2 \
  --phase3-no-batching \
  --adapt-beam-children-per-parent 2 --adapt-beam-lambda 0.005 --adapt-beam-live-branches 3 \
  --runtime-split-mode shortlist_pauli_children_v1 --runtime-split-max-subset-size 1 \
  --adapt-max-depth 1 --adapt-maxiter 2 \
  --output-json artifacts/json/physical_lane_smoke.json
```
Use the resolver output to fill in the exact HH regime parameters and any baseline flags not shown here; do not substitute generic defaults for the named promoted-copy source settings.
</tests_and_runs>

<relationships>
- CLI args (`cli_config.py`) -> `_run_hardcoded_adapt_vqe` (`adapt_pipeline.py`) -> `PhaseShortlistRuntime` (`phase_shortlists.py`) -> lane helpers (`algebraic_metadata.py`).
- Full_meta label -> `classify_hh_full_meta_label` (`static_provenance.py`) -> physical lane payload -> `CandidateFeatures` -> phase1/phase2 shortlist budget allocation.
- Phase1 candidate metadata -> `_phase2_inherited_selector_updates` -> phase2 lane health shortlist -> final continuation telemetry.
- `phase3_policy_optuna.py` policy fields -> CLI args -> same `adapt_pipeline.py` route path.
</relationships>

<ambiguities>
- Whether to represent the physical route as a new `static_route_id` or as a lane-route variant under the current Route A pipeline is not explicitly settled. Prefer an explicit `static_lane_route` submode with default `algebraic`, plus telemetry/variant fields, unless tests/docs require a new route id.
- The selected context includes the main and beam/SNAKE shortlist call sites, but not every duplicate helper around them. Keep beam and non-beam behavior synchronized.
- The baseline source-map path/method may need adjustment after inspecting the provenance JSON. Source-lock rules require resolving the visible source settings first and failing closed if they cannot be resolved.
</ambiguities>

## Selection
- Files: 18 total (9 full, 9 slice)
- Total tokens: 111180 (Auto view)
- Token breakdown: full 30606, slice 80574

### Files
### Selected Files
/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/
├── MATH/
│   └── paper_facing/
│       └── paper_I_static_scaffold/
│           ├── hh_fixed_scaffold_motif_inventory_20260614.md — 1,852 tokens (full)
│           └── paper_i_hh_snake_nobatch_duplicate_promotion_20260707.json — 4,889 tokens (full)
├── agent_guidance/
│   └── skills/
│       ├── paper-i-run/
│       │   └── SKILL.md — 5,386 tokens (lines 94-313 (Paper-I visible-run gate, source-visible settings resolver requirement, smoke policy, and current promoted HH Powell SNAKE no-batch contract.), 362-451 (Local/Optuna speed path caveat; distinguish older full_meta_minus_hva defaults from the named promoted unfiltered full_meta HVA-included baseline.))
│       ├── shared/
│       │   └── scripts/
│       │       └── resolve_visible_settings.py — 3,743 tokens (full)
│       └── source-locked-sensitivity/
│           └── SKILL.md — 2,143 tokens (full)
├── pipelines/
│   ├── contracts/
│   │   └── static_provenance.py — 3,156 tokens (full)
│   ├── scaffold/
│   │   └── hh_continuation_types.py — 3,401 tokens (lines 1-330 (CandidateFeatures dataclass and related typed payloads; edit point for adding physical-lane fields and preserving serialization compatibility.))
│   └── static_adapt/
│       ├── optimization/
│       │   └── phase3_policy_optuna.py — 8,947 tokens (lines 80-119 (Existing canonical_lane_families benchmark grouping; distinguish from new physical operator-type shortlist lanes.), 220-349 (Policy defaults/choices for phase0 algebraic lane mode and algebraic lane quota/threshold controls; add explicit physical lane route/aggressiveness choices here if policy-visible.), 520-640 (StaticPolicy dataclass route/shortlist fields; add persistent policy fields for physical lane route and aggressiveness.), 1640-1775 (policy_to_cli_args emission for phase0/algebraic/phase3 selector controls; emit new lane-route flags here.), 4962-5155 (default_trial_params and CLI-derived trial params for route/shortlist/algebraic controls; add physical lane route parameters.), 5301-5415 (Trial parameter sanitization and CLI-category normalization for relevant route/lane choices.))
│       ├── adapt_candidate_record_cache.py — 2,042 tokens (full)
│       ├── adapt_pipeline.py — 31,934 tokens (lines 120-220 (Core imports for CandidateFeatures, provenance, and static ADAPT utilities; add physical classifier imports here.), 500-590 (Phase shortlist runtime/wrapper imports; generalized lane route imports belong here.), 1450-1730 (_run_hardcoded_adapt_vqe signature/defaults and early validation for route/profile/shortlist controls; add lane-route and aggressiveness kwargs without changing algebraic defaults.), 2880-3095 (Pool registry/family-id construction after pool resolution; physical lanes can derive from candidate labels and generator metadata here.), 3410-3600 (Active algebraic metadata setup, _algebraic_payload_for_candidate, and _attach_phase1_selector_metadata; parallel physical payload should preserve existing route behavior.), 3860-3990 (Phase2 inherited selector updates and feature cache payload; include new physical-lane fields so Phase2 shortlists and cached records stay consistent.), 4700-4825 (PhaseShortlistRuntime construction with algebraic pressures/thresholds; route selected lane key/vocabulary into runtime here.), 4860-4945 (Phase0 pilot setup and feature creation context; include only enough to understand phase0 lane metadata hook if physical phase0 screening is enabled.), 5400-5605 (Static route identity payload construction and observed components; add lane-route/variant telemetry without mutating existing algebraic Route A contract.), 10840-11090 (Beam/SNAKE Phase1 feature records, selector metadata attachment, and lane shortlist call; required for promoted no-batch SNAKE baseline path.), 12360-12510 (Beam/SNAKE Phase2 lane-health shortlist call; required for promoted no-batch SNAKE baseline path.), 15930-16140 (Main Phase1 feature records, selector metadata attachment, and lane shortlist call; keep non-beam path consistent.), 17180-17310 (Main Phase2 lane-health shortlist call; keep non-beam path consistent.), 22780-22970 (Final continuation/result telemetry including phase0 and algebraic_lane_policy payloads; add physical lane route telemetry/provenance.))
│       ├── algebraic_metadata.py — 7,441 tokens (lines 1-180 (Imports, algebraic lane constants (flat/curv/disj/mix), normalization helpers, and public lane vocabulary preserved for existing route.), 690-930 (Exact expansion context summarization and assign_lane path for current support/commutation route; keep behavior intact while adding physical route elsewhere.), 1080-1478 (Phase0 weak algebraic payload plus lane quota/shortlist helpers for phase0/phase1/phase2; likely place to generalize lane vocab/key while retaining current wrappers.))
│       ├── cli_config.py — 8,358 tokens (lines 1-120 (CLI imports/constants including route/profile choices; add lane-route choices here if local to cli_config.), 1240-1380 (Static route/profile CLI options; nearby placement for a Paper-I physical-lane variant flag.), 1500-1695 (Phase0/phase1/phase2 shortlist and algebraic lane CLI options; edit point for physical-lane route/aggressiveness controls.), 3660-3845 (_build_run_hardcoded_adapt_vqe_kwargs mapping for route/profile/phase0/phase1/phase2/algebraic controls; pass new lane route/aggressiveness settings.))
│       ├── phase_shortlists.py — 4,839 tokens (full)
│       └── route_identity.py — 6,513 tokens (lines 1-250 (Route/profile constants, Route A required components, and production profile metadata; preserve existing algebraic Route A contract.), 300-560 (StaticRouteIdentityConfig and observed/required component payload construction; add lane-route/variant telemetry carefully here if needed.), 620-723 (Validation/summary helpers and static route identity rendering used by tests and result telemetry.))
└── test/
    ├── optimization/
    │   └── test_phase3_policy_optuna.py — 4,831 tokens (lines 1328-1505 (Optuna default/canonical lane family and trial parameter surface tests; extend expected defaults without confusing benchmark lane families with physical shortlist lanes.), 1980-2190 (CLI command parsing and policy-to-CLI/apply tests around algebraic lane and phase3 selector controls; extend for new lane route flag.))
    ├── test_hh_full_meta_class_filter.py — 4,600 tokens (full)
    ├── test_static_adapt_algebraic_metadata.py — 3,763 tokens (lines 1-230 (Current exact algebra/lane assignment tests and phase0 weak algebraic index expectations; protect existing support/commutation route.), 270-409 (Quota-pressure and phase1/phase2 shortlist tests over current four algebraic lanes; likely pattern for generalized/physical lane tests.))
    └── test_static_adapt_phase_shortlists.py — 3,342 tokens (full)


---

## Generated Plan

## Chat Send ✅
- **Chat**: `physical-lanes-BCCCCA` | **Mode**: plan

### Response
## 1. Summary

Implement this as a targeted route-variant change, not a broad ADAPT refactor: keep existing Route A/algebraic support-commutation behavior as the default, generalize the existing lane shortlist machinery, and add an opt-in `static_lane_route=physical_operator_type` that budgets Phase-1 and Phase-2 shortlists by HH physical operator family. The physical route changes only lane assignment and effective shortlist aggressiveness, defaulting to 3x tighter caps/fractions; all baseline/source settings remain recorded separately and otherwise unchanged. No CHTC operation is required for this implementation plan.

---

## 2. Current-state analysis

### Existing lane ownership and flow

- `pipelines/static_adapt/algebraic_metadata.py`
  - Owns the current lane vocabulary:
    - `flat`, `curv`, `disj`, `mix`
  - Computes exact support/commutation relations from serialized Pauli expansions.
  - Provides:
    - `assign_lane(...)`
    - `phase0_weak_lane_payload(...)`
    - `algebraic_lane_quota_pressure_budgets(...)`
    - `phase1_lane_shortlist_records(...)`
    - `phase2_lane_health_shortlist_records(...)`
  - Current helpers assume `LANES_PHASE1` and unknown lanes collapse to `mix`.

- `pipelines/static_adapt/phase_shortlists.py`
  - Wraps generic and algebraic lane-aware shortlist selection.
  - `PhaseShortlistRuntime` is currently algebraic-specific:
    - `algebraic_lane_policy_active`
    - `algebraic_metadata_summary`
    - algebraic quota pressures / rel threshold
  - Phase-1 and Phase-2 wrappers choose between:
    - generic `phase_shortlist_records(...)`, or
    - algebraic lane quota/health helpers.

- `pipelines/static_adapt/adapt_pipeline.py`
  - Receives CLI kwargs from `cli_config.py`.
  - Resolves pool and `pool_generator_registry`.
  - Builds exact algebraic metadata when `phase3_selector_policy='algebraic_nested_v1'`.
  - Adds algebraic payloads to candidate records/CandidateFeatures.
  - Sends Phase-1 and Phase-2 candidate records through `PhaseShortlistRuntime`.
  - Copies inherited selector metadata into Phase-2 full-feature records and candidate-cache keys.
  - Emits final continuation telemetry including `algebraic_lane_policy`.

- `pipelines/contracts/static_provenance.py`
  - Already provides stable HH full-meta label classification via `classify_hh_full_meta_label(label)`.
  - This should be reused as the physical lane classifier base; do not parse labels ad hoc in `adapt_pipeline.py`.

### Blocking constraints

- Existing Route A tests and semantics must remain unchanged:
  - `static_route_id=route_a`
  - `phase3_selector_policy=algebraic_nested_v1`
  - `algebraic_shortlisting_enabled=True`
  - existing four algebraic lanes and `mix` fallback.
- Physical lanes require a different lane key/vocabulary:
  - `physical_operator_lane`
  - fallback `other`, not `mix`.
- Candidate-record caching can reuse old records unless the route/lane metadata is included in the cache key or the cache version is bumped.
- Source-lock rules mean paper-facing runs must record base settings separately from effective physical-route settings.

---

## 3. Design

### 3.1 Physical operator lane taxonomy

Add the physical taxonomy in `pipelines/contracts/static_provenance.py` because it already owns the stable HH full-meta class vocabulary.

Define additive constants:

```py
HH_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION = "hh_physical_operator_lanes_v1"

PHYS_LANE_ELECTRONIC_CORRELATION = "electronic_correlation"
PHYS_LANE_ELECTRONIC_CURRENT = "electronic_current"
PHYS_LANE_PHONON_DISPLACEMENT = "phonon_displacement"
PHYS_LANE_PHONON_SQUEEZE_RELAXATION = "phonon_squeeze_relaxation"
PHYS_LANE_DRESSED_PHONON_CORRELATION = "dressed_phonon_correlation"
PHYS_LANE_HVA_HAMILTONIAN_BLOCKS = "hva_hamiltonian_blocks"
PHYS_LANE_OTHER = "other"
```

Lane order:

```py
HH_PHYSICAL_OPERATOR_LANES = (
    electronic_correlation,
    electronic_current,
    phonon_displacement,
    phonon_squeeze_relaxation,
    dressed_phonon_correlation,
    hva_hamiltonian_blocks,
    other,
)
```

Classification contract:

- Inputs:
  - `label: str`
  - optional already-resolved HH class if available.
- Output payload:
  - `physical_operator_class`
  - `physical_operator_lane`
  - `physical_operator_classifier_version`
  - fallback reason/source.

Mapping:

| `classify_hh_full_meta_label` class | Physical lane |
|---|---|
| `uccsd_sing`, `uccsd_dbl` | `electronic_correlation` |
| `hh_fermionic_reusable` with `"current"` in label | `electronic_current` |
| other `hh_fermionic_reusable` | `electronic_correlation` |
| `hh_phonon_linear`, `paop_disp` | `phonon_displacement` |
| `hh_phonon_quadratic`, `hh_vlf_sq` | `phonon_squeeze_relaxation` |
| `paop_cloud_p`, `paop_cloud_x`, `paop_dbl`, `paop_hopdrag`, `paop_dbl_p`, `paop_dbl_x`, `paop_curdrag`, `paop_hop2`, `uccsd_paop_product`, `uccsd_paop_product_seq_ferm`, `uccsd_paop_product_seq_motif` | `dressed_phonon_correlation` |
| `hva_layer`, `hh_hamiltonian_block`, `hh_termwise_unit`, `hh_termwise_quadrature` | `hva_hamiltonian_blocks` |
| `paop_other`, unclassified, missing label | `other` |

For runtime-split child labels, classification should try labels in this order:

1. candidate label;
2. metadata parent label fields such as `runtime_split_parent_label`, `parent_label`, `source_parent_label`, `template_label`, `base_label`;
3. fallback to candidate label as `other`.

This prevents Pauli-child representatives from collapsing to `other` when parent metadata is available.

---

### 3.2 Static lane route constants

Add a small pure module:

`pipelines/static_adapt/lane_routes.py`

Purpose: centralize CLI/policy/runtime lane-route choices without coupling CLI to `adapt_pipeline.py`.

Define:

```py
STATIC_LANE_ROUTE_ALGEBRAIC = "algebraic"
STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE = "physical_operator_type"
STATIC_LANE_ROUTE_CHOICES = (
    STATIC_LANE_ROUTE_ALGEBRAIC,
    STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
)
PHYSICAL_LANE_SHORTLIST_AGGRESSIVENESS_CHOICES = (2, 3)
```

Also define a small lane spec shape:

```py
StaticShortlistLaneSpec:
    route: str
    lane_key: str
    lanes: tuple[str, ...]
    fallback_lane: str
    health_key_prefix: str
```

Resolution:

- `algebraic`:
  - key `algebraic_lane`
  - lanes `LANES_PHASE1`
  - fallback `mix`
  - health prefix `algebraic`
- `physical_operator_type`:
  - key `physical_operator_lane`
  - lanes `HH_PHYSICAL_OPERATOR_LANES`
  - fallback `other`
  - health prefix `physical_operator`

---

### 3.3 Generalized lane shortlist helpers

Modify `pipelines/static_adapt/algebraic_metadata.py` by adding generalized helpers while preserving existing algebraic wrappers exactly.

New generalized helpers:

```py
lane_quota_pressure_budgets(
    records,
    *,
    lanes,
    lane_key,
    fallback_lane,
    score_key,
    threshold,
    cap,
    pressure,
    lane_rel_threshold=0.0,
    tie_break_score_key=None,
) -> dict[str, int]
```

```py
lane_phase1_shortlist_records(
    records,
    *,
    lanes,
    lane_key,
    fallback_lane,
    ...
) -> list[dict[str, Any]]
```

```py
lane_phase2_health_shortlist_records(
    records,
    *,
    lanes,
    lane_key,
    fallback_lane,
    health_key_prefix,
    ...
) -> list[dict[str, Any]]
```

Keep current public wrappers as compatibility wrappers:

- `algebraic_lane_quota_pressure_budgets(...)`
- `phase1_lane_shortlist_records(...)`
- `phase2_lane_health_shortlist_records(...)`

These wrappers must pass:

- `lanes=LANES_PHASE1`
- `lane_key='algebraic_lane'`
- `fallback_lane=LANE_MIX`
- `health_key_prefix='algebraic'`

Behavioral invariants:

- Existing algebraic tests must pass unchanged.
- Unknown algebraic lane still normalizes to `mix`.
- Unknown physical lane normalizes to `other`.
- Budget dict always contains every lane in the selected vocabulary.
- Duplicate candidate identity behavior remains unchanged.

---

### 3.4 Phase shortlist runtime generalization

Modify `pipelines/static_adapt/phase_shortlists.py`.

Extend `PhaseShortlistRuntime` additively:

```py
shortlist_lane_route: str = "algebraic"
shortlist_lane_key: str = "algebraic_lane"
shortlist_lanes: tuple[str, ...] = LANES_PHASE1
shortlist_fallback_lane: str = "mix"
shortlist_lane_health_prefix: str = "algebraic"
shortlist_lane_summary: MutableMapping[str, Any] | None = None
```

Preserve existing fields:

- `algebraic_lane_policy_active`
- `algebraic_metadata_summary`
- `algebraic_phase1_lane_quota_pressure`
- `algebraic_phase2_lane_quota_pressure`
- `algebraic_phase2_lane_rel_threshold`

Wrapper behavior:

- If lane policy inactive, keep generic shortlist behavior.
- If active:
  - use `runtime.shortlist_lane_key`;
  - use `runtime.shortlist_lanes`;
  - use `runtime.shortlist_fallback_lane`;
  - write health fields using `runtime.shortlist_lane_health_prefix`.

For physical route, Phase-2 health fields should be:

- `physical_operator_lane_health`
- `physical_operator_lane_relative_health`
- `physical_operator_lane_live`

Keep `_record_algebraic_shortlist_runtime(...)` for compatibility, but internally make it route-aware. For algebraic route it writes the same keys currently used by tests. For physical route it writes into `runtime.shortlist_lane_summary`.

---

### 3.5 CandidateFeatures persistence

Modify `pipelines/scaffold/hh_continuation_types.py`.

Add optional fields to `CandidateFeatures`:

```py
physical_operator_lane: str | None = None
physical_operator_class: str | None = None
physical_operator_classifier_label: str | None = None
physical_operator_lane_source: str = "unassigned"
physical_operator_lane_quality: str = "unassigned"
physical_operator_lane_health: float | None = None
physical_operator_lane_relative_health: float | None = None
physical_operator_lane_live: bool | None = None
```

Compatibility:

- Old result JSONs lacking these fields still deserialize through defaults.
- New result JSONs are additive; old consumers ignoring extra keys remain compatible.

---

### 3.6 Adapt pipeline route wiring

Modify `pipelines/static_adapt/adapt_pipeline.py`.

#### New kwargs

Add to `_run_hardcoded_adapt_vqe(...)`:

```py
static_lane_route: str = "algebraic"
physical_lane_shortlist_aggressiveness: int = 3
```

Validation:

- Normalize `static_lane_route`.
- `physical_lane_shortlist_aggressiveness` must be `2` or `3`.
- If `static_lane_route == physical_operator_type`, require `problem_key == "hh"`.
- Do not require a new `static_route_id`; this is a route variant under the existing base route identity.

#### Effective shortlist aggressiveness

Keep base/source values unchanged:

- `phase1_shortlist_size_base = phase1_shortlist_size`
- `phase2_shortlist_size_base = phase2_shortlist_size`
- `phase2_shortlist_fraction_base = phase2_shortlist_fraction`

If `static_lane_route == physical_operator_type`:

```text
factor = physical_lane_shortlist_aggressiveness
effective_cap = 0 if base <= 0 else max(1, ceil(base / factor))
effective_fraction = base_fraction / factor
```

Default physical behavior:

- Phase-1 `64 -> 22`
- Phase-2 `12 -> 4`
- Phase-2 fraction `0.2 -> 0.0666666667`

If `static_lane_route == algebraic`, factor is effectively `1` and existing values are untouched.

Telemetry must record both base and effective values.

#### Physical payload builder

Add a local helper near `_algebraic_payload_for_candidate(...)`:

Inputs:

- `candidate_term`
- `pool_generator_registry`
- candidate index when available.

Output keys:

```py
{
    "physical_operator_lane": ...,
    "physical_operator_class": ...,
    "physical_operator_classifier_label": ...,
    "physical_operator_lane_source": ...,
    "physical_operator_lane_quality": ...,
}
```

Use `classify_hh_physical_operator_lane(...)` from `static_provenance.py`, trying parent labels from generator metadata before fallback.

Attach this payload wherever algebraic selector metadata is currently attached to Phase-1 candidate records.

#### Phase-2 inheritance/cache

Extend `_phase2_inherited_selector_updates(...)` to copy:

- `physical_operator_lane`
- `physical_operator_class`
- `physical_operator_classifier_label`
- `physical_operator_lane_source`
- `physical_operator_lane_quality`
- `physical_operator_lane_health`
- `physical_operator_lane_relative_health`
- `physical_operator_lane_live`

Extend `_feature_base_scoring_cache_payload(...)` with the same physical lane identity fields so physical/algebraic cached records cannot be confused.

Also bump `_CANDIDATE_RECORD_CACHE_CODE_VERSION` in `adapt_candidate_record_cache.py` because CandidateFeatures semantics and cache identity change.

#### Runtime construction

When building `PhaseShortlistRuntime`:

- Resolve lane spec from `static_lane_route`.
- For algebraic route:
  - summary remains `algebraic_metadata_summary`.
- For physical route:
  - create `physical_operator_lane_summary`.
  - pass it as `shortlist_lane_summary`.
  - keep `algebraic_metadata_summary` intact for exact algebraic metadata/Route A telemetry.

#### Final continuation telemetry

Add:

```py
"static_lane_route": "algebraic" | "physical_operator_type",
"shortlist_lane_route": {...},
"physical_operator_lane_policy": {...},
```

Keep existing:

```py
"algebraic_lane_policy": dict(algebraic_metadata_summary)
```

Physical summary should include:

- schema/version;
- lane vocabulary;
- classifier version;
- fallback lane;
- aggressiveness factor;
- base/effective shortlist caps/fractions;
- last per-lane budgets/counts from Phase-1 and Phase-2.

---

### 3.7 CLI surface

Modify `pipelines/static_adapt/cli_config.py`.

Add near `--static-route-id` / `--static-meta-feature-profile`:

```text
--static-lane-route {algebraic,physical_operator_type}
default: algebraic

--physical-lane-shortlist-aggressiveness {2,3}
default: 3
```

Map both through `_build_run_hardcoded_adapt_vqe_kwargs(...)`.

Important behavior:

- `--static-lane-route algebraic` is the default and must not change current runs.
- The aggressiveness flag is ignored unless physical route is selected, but still recorded.

---

### 3.8 Route identity telemetry

Modify `pipelines/static_adapt/route_identity.py`.

Do **not** add physical lane route to `ROUTE_A_REQUIRED_COMPONENTS`.

Add optional/additive fields to `StaticRouteIdentityConfig`:

```py
static_lane_route: Any = "algebraic"
physical_lane_shortlist_aggressiveness: int | None = None
```

After normal route validation, append telemetry:

```py
"static_lane_route": ...,
"static_lane_route_is_route_identity": False,
"route_variant_id": (
    "route_a_physical_operator_lanes_v1"
    if route_id == route_a and static_lane_route == physical_operator_type
    else None
)
```

This preserves Route A validation semantics while making the physical route variant machine-visible.

---

### 3.9 Optuna/policy surface

Modify `pipelines/static_adapt/optimization/phase3_policy_optuna.py`.

Add fields to `StaticPolicy`:

```py
static_lane_route: str = "algebraic"
physical_lane_shortlist_aggressiveness: int = 3
```

Update:

- `default_trial_params()`
  - include `static_lane_route="algebraic"`
  - include `physical_lane_shortlist_aggressiveness=3`
- `policy_to_cli_args(...)`
  - emit the new flags from policy.
- `trial_params_from_cli_command(...)`
  - parse both new flags.
- policy sanitization/normalization
  - invalid route -> `algebraic`
  - invalid aggressiveness -> `3`
- roundtrip audit
  - add rows for both knobs.
  - runtime lookup should read:
    - `continuation.static_lane_route`
    - `continuation.physical_operator_lane_policy.aggressiveness_factor`

Do not reuse `_CANONICAL_LANE_FAMILIES`; that is benchmark-suite grouping, not shortlist lane taxonomy.

---

## 4. File-by-file impact

### `pipelines/contracts/static_provenance.py`

- Add physical lane constants and classifier version.
- Add physical lane classifier function using `classify_hh_full_meta_label`.
- Export new constants/functions in `__all__`.
- Rationale: stable HH source-visible taxonomy already lives here.

### `pipelines/static_adapt/lane_routes.py` — new file

- Add static lane route constants, choices, normalizer, and lane spec resolver.
- Depends on:
  - algebraic lane constants from `algebraic_metadata.py`
  - physical lane constants from `static_provenance.py`
- Rationale: shared lightweight route constants for CLI, policy, and runtime.

### `pipelines/static_adapt/algebraic_metadata.py`

- Add generalized lane budget/shortlist helpers.
- Rewrite existing algebraic wrappers to call generalized helpers with old defaults.
- Preserve Phase-0 weak algebraic behavior unchanged.
- Dependency: physical route wrappers in `phase_shortlists.py`.

### `pipelines/static_adapt/phase_shortlists.py`

- Extend `PhaseShortlistRuntime`.
- Make Phase-1/Phase-2 lane wrappers route-neutral.
- Preserve exported compatibility names.
- Add route-aware runtime telemetry recording.

### `pipelines/scaffold/hh_continuation_types.py`

- Add physical lane fields to `CandidateFeatures`.
- No constructor call sites need mandatory changes because all fields have defaults.

### `pipelines/static_adapt/adapt_candidate_record_cache.py`

- Bump `_CANDIDATE_RECORD_CACHE_CODE_VERSION`.
- Rationale: candidate feature identity and inherited selector metadata change.

### `pipelines/static_adapt/adapt_pipeline.py`

- Add kwargs and validation for static lane route/aggressiveness.
- Compute base/effective shortlist settings.
- Build and attach physical lane payloads.
- Copy physical fields through Phase-2 inheritance.
- Include physical fields in cache payload.
- Construct route-aware `PhaseShortlistRuntime`.
- Add final telemetry.
- Add route identity variant fields via updated config.

### `pipelines/static_adapt/cli_config.py`

- Add CLI arguments.
- Pass through to `_run_hardcoded_adapt_vqe` kwargs.

### `pipelines/static_adapt/route_identity.py`

- Add additive variant telemetry fields only.
- Do not alter `ROUTE_A_REQUIRED_COMPONENTS`.

### `pipelines/static_adapt/optimization/phase3_policy_optuna.py`

- Add policy/default/CLI parsing/audit support.
- Keep default route algebraic.

### Tests

- `test/test_hh_full_meta_class_filter.py`
  - Add physical classifier mapping tests.
- `test/test_static_adapt_algebraic_metadata.py`
  - Add generalized physical lane budget tests.
  - Existing algebraic tests must remain unchanged.
- `test/test_static_adapt_phase_shortlists.py`
  - Add physical route runtime tests.
  - Assert physical health fields are written.
- `test/optimization/test_phase3_policy_optuna.py`
  - Add default, CLI parse, policy-to-CLI, and audit coverage.

---

## 5. Risks and migration

- **Cache migration:** bumping candidate-record cache version invalidates old candidate-record cache entries. This is intentional and safer than mixing algebraic and physical lane records.
- **Source-lock audits:** physical route intentionally changes `static_lane_route` and derived effective shortlist settings. Paper-facing run notes must list these as changed settings; all other settings must come from the visible source resolver.
- **Route identity:** physical route is a Route A variant, not a replacement for Route A. Existing Route A validation must remain strict and unchanged.
- **Fallback-heavy pools:** if runtime-split metadata lacks parent labels, many child candidates may fall into `other`. Telemetry should expose per-lane counts so this is visible.

---

## 6. Implementation order

1. Add physical lane taxonomy/classifier in `static_provenance.py` and tests for mappings.
2. Add `lane_routes.py` with route constants/spec resolver.
3. Generalize lane helpers in `algebraic_metadata.py`; confirm existing algebraic tests still pass.
4. Extend `PhaseShortlistRuntime` and wrappers in `phase_shortlists.py`; add physical runtime tests.
5. Add `CandidateFeatures` physical fields and bump candidate-record cache version.
6. Wire `adapt_pipeline.py`:
   - kwargs;
   - validation;
   - effective shortlist factor;
   - physical payload;
   - inheritance/cache;
   - runtime construction;
   - final telemetry.
7. Add CLI flags and kwargs mapping in `cli_config.py`.
8. Add route identity variant telemetry.
9. Add Optuna/policy support and roundtrip tests.
10. Run targeted tests:

```bash
python3 -m pytest \
  test/test_hh_full_meta_class_filter.py \
  test/test_static_adapt_algebraic_metadata.py \
  test/test_static_adapt_phase_shortlists.py \
  test/optimization/test_phase3_policy_optuna.py
```

11. Only after tests pass, use the visible-settings resolver before any paper-facing physical-lane run; do not substitute generic defaults for promoted-copy source settings.


> 💡 Continue this plan conversation with ask_oracle(chat_id: "physical-lanes-BCCCCA", new_chat: false)