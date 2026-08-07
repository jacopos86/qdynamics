## Final Prompt
<taskname="Hubbard HVA Lane"/>

<task>
Add a provenance-safe Paper-I Hubbard weak physical-lane SNAKE pool that combines existing Hubbard UCCSD + QEB/qubit-excitation operators with Hamiltonian-variational/HVA block terms as an additional physical lane, then run Hubbard weak to max depth 10. Preserve the recent Hubbard-only `uccsd_qeb` implementation and extend it minimally; do not touch manuscript files, do not duplicate unrelated jobs, and do not drift settings except the new pool/lane bundle and explicit `--adapt-max-depth 10`. A prelaunch audit over final emitted pool labels must pass with `other_count == 0` and no exact `other` labels before running.
</task>

<architecture>
- Pool entry: `cli_config.py --adapt-pool` choices come from `problem_registry.available_adapt_pool_keys()` via `_HUBBARD_POOL_KEYS`; add a new distinct Hubbard key such as `uccsd_qeb_hva`/`uccsd_qeb_hva_blocks`, not a replacement for `uccsd_qeb` or default `uccsd`.
- Pool resolution: `pool_resolution.resolve_pool_plan()` maps pool keys to `AnsatzTerm` pools and provenance (`method_name`, `pool_stage_family`, `pool_family_ids`). Current Hubbard `uccsd_qeb` branch uses `_build_hubbard_uccsd_qeb_pool()` and `_hubbard_uccsd_qeb_family_id_for_label()`.
- Pool construction: `primitive_pools.py` already owns `_build_uccsd_pool()`, shared QEB construction (`build_qeb_pool_specs()`, `_build_qeb_pool()`), structural dedup (`_polynomial_signature()`, `_deduplicate_pool_terms()`), and `build_runtime_pool_terms()` support for `uccsd_qeb`.
- HVA context: `_build_hva_pool()` is HH-specific and emits HH layer labels from `HubbardHolsteinLayerwiseAnsatz` (`hop_layer`, `onsite_layer`, `potential_layer`, plus phonon/eph/drive). `_build_family_hva_pool()` currently excludes plain `hubbard`. Plain `HubbardLayerwiseAnsatz` has labels `hop_layer`, `onsite_layer`, `potential_layer`, but its `base_terms` are representative single terms, not full grouped polynomials. Full plain-Hubbard block polynomials are available from `build_hubbard_kinetic()`, `build_hubbard_onsite()`, and `build_hubbard_potential()`; existing termwise grouped labels from `_build_hamiltonian_blocks_pool(problem_key="hubbard")` are `ham_block::hop(...)`, `ham_block::onsite(...)`, `ham_block::pot(...)`.
- Physical lanes: `static_provenance.py` owns Hubbard lane constants, classifier version, `HUBBARD_PHYSICAL_OPERATOR_LANES`, `_classify_hubbard_physical_operator_lane()`, and `summarize_static_physical_operator_pool_labels()`. Current Hubbard lanes are `uccsd_single`, `uccsd_double`, `qeb_excitation`, `other`.
- Route metadata: `lane_routes.py` maps `problem="hubbard"` to route id `route_a_hubbard_physical_operator_lanes_v2_uccsd_qeb_split`; bump this when adding the HVA lane.
- Runtime gate: `adapt_pipeline.py` builds/expands the pool, computes `prelaunch_pool_audit` for physical lane route, and currently blocks only `problem_key == "hubbard" and pool_key == "uccsd_qeb"` unless physical route is active, audit has no other labels, and QEB survives.
</architecture>

<selected_context>
- `pipelines/static_adapt/builders/primitive_pools.py`: full implementation for UCCSD/QEB pools, HVA/HH builders, Hamiltonian block builders, dedup helpers, runtime pool materialization, and exports. Main likely edit target for a combined Hubbard UCCSD+QEB+HVA builder and family-id helper.
- `pipelines/static_adapt/builders/pool_resolution.py`: full resolver. Add the new Hubbard pool branch, keep metadata distinct, and extend special `pool_family_ids` handling for the combined pool.
- `pipelines/static_adapt/builders/problem_registry.py`: full registry and `_HUBBARD_POOL_KEYS`; add the new key without changing default pool.
- `pipelines/contracts/static_provenance.py`: full lane constants/classifiers/audit. Add/reuse a Hubbard HVA lane string (likely `hva_hamiltonian_blocks`), classify exact emitted HVA labels, and bump Hubbard classifier/schema/version.
- `pipelines/static_adapt/lane_routes.py`: full route contract; bump Hubbard route variant id to include HVA lane semantics.
- `pipelines/static_adapt/adapt_pipeline.py`: slices for imports, pool resolution/expansion, physical-lane policy, prelaunch audit/gate, candidate classification, current.json monitoring, and final continuation provenance.
- `pipelines/static_adapt/cli_config.py`: slices showing pool choices are registry-driven and result settings persist `adapt_pool`, physical lane aggressiveness, and `max_depth`.
- `src/quantum/hubbard_latex_python_pairs.py`: slices with grouped plain-Hubbard polynomial builders (`build_hubbard_kinetic`, `build_hubbard_onsite`, `build_hubbard_potential`).
- `src/quantum/vqe_latex_python_pairs.py`: slices for `AnsatzTerm`, `HubbardTermwiseAnsatz`, `HubbardLayerwiseAnsatz`, and HH `_poly_group` behavior.
- Tests: full `test_static_adapt_algebraic_metadata.py` and `test_static_adapt_lane_routes.py`; slices in `test_static_adapt_builders.py`, `test_adapt_vqe_integration.py`, and `test_generic_static_adapt_variants.py` for parser, pool, classifier, route, and QEB-label coverage.
</selected_context>

<relationships>
- `--adapt-pool <new_key>` -> `available_adapt_pool_keys()` -> `_HUBBARD_POOL_KEYS` -> `resolve_pool_plan()` -> combined deduped `AnsatzTerm` pool -> final labels audited in `adapt_pipeline`.
- UCCSD labels `uccsd_sing(...)` / `uccsd_dbl(...)` -> Hubbard classifier -> `uccsd_single` / `uccsd_double`.
- QEB labels `qeb_pair(i,j)` / `qeb_double(i,j->a,b)` -> Hubbard classifier -> `qeb_excitation`.
- HVA/block labels must be defined by the chosen builder and classifier together. Expected candidates: grouped `hop_layer`, `onsite_layer`, `potential_layer` if building plain-Hubbard HVA blocks, or `ham_block::hop(...)`, `ham_block::onsite(...)`, `ham_block::pot(...)` if reusing termwise Hamiltonian blocks. Whichever is emitted must classify to the new Hubbard HVA lane and must not be `other`.
- `--static-lane-route physical_operator_type` -> `resolve_static_shortlist_lane_spec(problem="hubbard")` -> `physical_operator_lanes_for_problem("hubbard")` -> `PhaseShortlistRuntime` lane budgets/health.
- Final provenance path: `adapt_pipeline` prelaunch audit -> `physical_operator_lane_summary` -> `adapt_vqe.continuation.physical_operator_lane_policy` in final result.
</relationships>

<implementation_notes>
- Keep `uccsd_qeb` working unchanged. Add a new provenance-clear combined key and method name, e.g. `hardcoded_adapt_vqe_uccsd_qeb_hva_hubbard`.
- Dedup structurally after concatenating UCCSD + QEB + HVA/block terms. Make label/source retention policy explicit; lane counts depend on surviving final labels.
- Do not enable generic `pool='hva'` for Hubbard unless deliberately chosen; the request is for a combined Paper-I Hubbard weak lane pool.
- Update `build_runtime_pool_terms()` if prelaunch/debug tooling should materialize the new pool outside `resolve_pool_plan()`.
- Extend the runtime gate from `uccsd_qeb` to the new combined key: require physical lane route, `other_count == 0`, no `exact_other_labels`, positive QEB count, and positive HVA lane count.
- Add focused tests: classifier for emitted HVA labels, Hubbard lane list/route id includes HVA lane, parser accepts new key, pool builder dedups and keeps UCCSD/QEB/HVA survivors, resolve/audit over Hubbard weak settings has zero `other`.
</implementation_notes>

<run_plan>
After patch and tests, run only Hubbard weak to max depth 10. Use the prior Hubbard weak `uccsd_qeb` run/settings as baseline if visible in the filesystem (reported plateau at iteration 2, abs delta_e about 0.003902), changing only the pool/lane bundle and `--adapt-max-depth 10`. Before launching the ADAPT loop, run a cheap pool audit: resolve the intended Hubbard weak problem and new pool, classify every final label with `classify_static_physical_operator_lane(..., problem="hubbard")`, print lane counts/classifier version/route id, and stop if any label is `other`.

Candidate command shape to adapt from the baseline source settings:
`python -m pipelines.static_adapt.adapt_pipeline --problem hubbard --adapt-pool <new_key> --static-lane-route physical_operator_type --physical-lane-shortlist-aggressiveness 3 --phase1-shortlist-size 42 --phase2-shortlist-size 21 --phase2-shortlist-fraction 0.4375 --phase2-no-batching --phase3-no-batching --adapt-reopt-policy full --adapt-full-refit-every 0 --adapt-final-full-refit false --adapt-max-depth 10 --adapt-benchmark-target-abs-delta-e 1e-5 ...`
</run_plan>

<ambiguities>
- “HVA/Hamiltonian-variational ansatz block terms” could mean grouped HVA layer blocks (`hop_layer`/`onsite_layer`/`potential_layer`) or existing termwise Hamiltonian blocks (`ham_block::...`). The selected code shows both; choose one explicitly and keep builder labels/classifier/tests aligned.
- The exact prior raw output root/settings are not selected here; inspect targeted filesystem provenance before running, and do not infer optimizer/seed/output layout from memory.
- Working tree is dirty with recent `uccsd_qeb` lane changes; preserve unrelated changes and do not revert them.
</ambiguities>

## Selection
- Files: 17 total (9 full, 8 slice)
- Total tokens: 110819 (Auto view)
- Token breakdown: full 83822, slice 26997

### Files
### Selected Files
/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/
├── pipelines/
│   ├── contracts/
│   │   └── static_provenance.py — 8,046 tokens (full)
│   ├── exact_bench/
│   │   └── generic_static_adapt_variants.py — 991 tokens (lines 1-70 (Exact-bench QEB comparator imports build_qeb_pool_specs from primitive_pools; confirms QEB construction is now shared.), 1458-1476 (build_pairwise_qubit_excitation_pool delegates to build_qeb_pool_specs and emits qeb_pair/qeb_double labels.))
│   └── static_adapt/
│       ├── builders/
│       │   ├── __init__.py — 1,038 tokens (full)
│       │   ├── pool_resolution.py — 17,255 tokens (full)
│       │   ├── primitive_pools.py — 24,185 tokens (full)
│       │   └── problem_registry.py — 20,696 tokens (full)
│       ├── adapt_pipeline.py — 16,435 tokens (lines 300-419 (Imports for static physical lane classifier/audit, primitive pool helpers, and pool resolution used by runtime gate.), 720-799 (Selected-parameterization resolver; grouped_exact execution mode can force logical_shared if HVA blocks are represented as grouped generators.), 3038-3268 (Pool resolution call, child/shared pool expansion mutations before prelaunch audit, and pool-built telemetry; establishes final emitted pool seen by the lane audit.), 3440-3619 (Static/physical lane policy construction plus prelaunch pool audit and current Hubbard uccsd_qeb launch blocker; likely needs extension to new HVA pool key.), 3760-3929 (Candidate physical-lane payload assignment and lane count telemetry; labels are classified from parent/metadata/candidate labels during SNAKE selection.), 10180-10409 (current.json checkpoint payload with settings/adapt_vqe fields used for monitoring depth, pool_type, energy, abs_delta_e, and operators.), 23190-23419 (Final result continuation payload embedding static_lane_policy and physical_operator_lane_policy for post-run provenance checks.))
│       ├── cli_config.py — 1,812 tokens (lines 1208-1262 (--adapt-pool CLI option uses available_adapt_pool_keys(); adding the new Hubbard pool key to registry makes parser accept it.), 3578-3622 (Settings serialization includes adapt_pool, physical lane aggressiveness, model parameters, and max_depth for result provenance.), 3918-3957 (CLI kwargs mapping around pool filters and Phase-III settings; confirms no separate flag is needed for the new pool key.))
│       ├── lane_routes.py — 1,232 tokens (full)
│       └── phase_shortlists.py — 5,467 tokens (full)
├── src/
│   └── quantum/
│       ├── hubbard_latex_python_pairs.py — 1,352 tokens (lines 730-899 (Plain Hubbard grouped-polynomial sources build_hubbard_kinetic/onsite/potential used to construct HVA block/layer generators safely.))
│       └── vqe_latex_python_pairs.py — 3,940 tokens (lines 540-749 (AnsatzTerm plus HubbardTermwiseAnsatz and HubbardLayerwiseAnsatz label behavior; plain Hubbard layerwise base_terms are representative, not full grouped polynomials.), 1460-1639 (HH HubbardHolsteinLayerwiseAnsatz _poly_group implementation showing grouped-polynomial labels hop_layer/onsite_layer/potential_layer/phonon/eph for HVA semantics.))
└── test/
    ├── test_adapt_vqe_integration.py — 554 tokens (lines 1588-1642 (Parser tests for adapt pool choices, including the existing Hubbard uccsd_qeb parse test to mirror for the new pool key.))
    ├── test_generic_static_adapt_variants.py — 196 tokens (lines 704-720 (QEB convention test for qeb_pair labels and repo exyz qubit ordering, useful if QEB helper changes accidentally.))
    ├── test_static_adapt_algebraic_metadata.py — 5,488 tokens (full)
    ├── test_static_adapt_builders.py — 1,717 tokens (lines 1-90 (Imports and assertion helpers for builder tests, including primitive_pools and HubbardTermwiseAnsatz comparison helpers.), 480-531 (Existing Hubbard hamiltonian_blocks test showing labels ham_block::hop/onsite/pot from HubbardTermwiseAnsatz.), 1900-1952 (Existing uccsd_qeb pool test to extend for combined UCCSD+QEB+HVA dedup and surviving lane labels.))
    └── test_static_adapt_lane_routes.py — 415 tokens (full)


---

## Generated Plan

## Chat Send ✅
- **Chat**: `hubbard-hva-lane-A352F4` | **Mode**: plan

### Response
## 1. **Summary**

Add a new additive Hubbard-only pool key, `uccsd_qeb_hva_blocks`, that preserves the existing `uccsd_qeb` path and extends it with grouped plain-Hubbard HVA block generators built from full `build_hubbard_kinetic`, `build_hubbard_onsite`, and `build_hubbard_potential` polynomials. Add a new Hubbard physical lane, `hva_hamiltonian_blocks`, bump Hubbard physical-lane provenance versions/route id, and extend the prelaunch gate so this pool can only run under `--static-lane-route physical_operator_type` after a final emitted-label audit proves `other_count == 0`, no exact `other` labels, at least one QEB survivor, and at least one HVA-block survivor. After implementation and targeted tests, run only the prior Hubbard weak configuration with the new pool/lane bundle and explicit `--adapt-max-depth 10`.

---

## 2. **Current-state analysis**

### Pool and CLI flow

Current CLI pool choices flow as:

```text
cli_config.py --adapt-pool choices
  → problem_registry.available_adapt_pool_keys()
  → _HUBBARD_POOL_KEYS
  → pool_resolution.resolve_pool_plan()
  → pool: list[AnsatzTerm]
  → adapt_pipeline pool expansion / audit / ADAPT loop
```

`cli_config.py` does not hardcode individual pool names; adding a key to `_HUBBARD_POOL_KEYS` is enough for parser acceptance.

### Existing Hubbard `uccsd_qeb`

`primitive_pools.py` already owns:

- `_build_uccsd_pool(...)`
- `build_qeb_pool_specs(...)`
- `_build_qeb_pool(...)`
- `_build_hubbard_uccsd_qeb_pool(...)`
- `_hubbard_uccsd_qeb_family_id_for_label(...)`
- `_deduplicate_pool_terms(...)`
- `_polynomial_signature(...)`

`_build_hubbard_uccsd_qeb_pool()` concatenates UCCSD then QEB, structurally deduplicates, and preserves first-surviving labels. That should remain unchanged.

### HVA/block source options

There are two relevant existing concepts:

1. `HubbardLayerwiseAnsatz` emits `hop_layer`, `onsite_layer`, `potential_layer`, but its `base_terms` are representative single terms, not full grouped polynomials.
2. Full plain-Hubbard grouped polynomials already exist in `src/quantum/hubbard_latex_python_pairs.py`:
   - `build_hubbard_kinetic(...)`
   - `build_hubbard_onsite(...)`
   - `build_hubbard_potential(...)`

Use option 2. Do **not** reuse `HubbardLayerwiseAnsatz.base_terms` for this pool.

### Physical-lane flow

Physical lanes are owned by `pipelines/contracts/static_provenance.py`.

Current Hubbard lanes:

```python
(
    "uccsd_single",
    "uccsd_double",
    "qeb_excitation",
    "other",
)
```

Current classifier behavior:

- `uccsd_sing(...)` / `uccsd_ferm_lifted::uccsd_sing(...)` → `uccsd_single`
- `uccsd_dbl(...)` / `uccsd_ferm_lifted::uccsd_dbl(...)` → `uccsd_double`
- `qeb_pair(...)` / `qeb_double(...)` → `qeb_excitation`
- everything else → `other`

`adapt_pipeline.py` already computes a `prelaunch_pool_audit` after pool resolution and expansion when `static_lane_route == physical_operator_type`. It currently enforces the audit only for `problem_key == "hubbard" and pool_key == "uccsd_qeb"`.

### Blocking gaps

To satisfy the request:

- The new combined pool key does not exist.
- Hubbard physical lanes do not include an HVA lane.
- Hubbard classifier would classify HVA/block labels as `other`.
- Route id still describes only UCCSD/QEB split.
- Runtime prelaunch gate does not protect the new pool.
- Tests do not cover combined UCCSD+QEB+HVA provenance.

---

## 3. **Design**

### 3.1 New pool key

Use:

```text
uccsd_qeb_hva_blocks
```

Rationale: it is distinct from `uccsd_qeb`, explicitly Hubbard physical-lane oriented, and names the added grouped HVA block component.

Do **not** change:

- default Hubbard pool: remains `uccsd`
- existing `uccsd_qeb` builder
- existing `uccsd_qeb` method name/behavior

New method name:

```text
hardcoded_adapt_vqe_uccsd_qeb_hva_blocks_hubbard
```

---

### 3.2 New grouped Hubbard HVA block builder

Add in `pipelines/static_adapt/builders/primitive_pools.py`.

#### New helper

Shape:

```python
def _build_hubbard_hva_blocks_pool(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    ordering: str,
    boundary: str,
) -> list[AnsatzTerm]:
    ...
```

It builds up to three grouped block generators:

| Label | Polynomial source | Execution mode |
|---|---|---|
| `hva_block::hop_layer` | `build_hubbard_kinetic(...)` | `grouped_exact` |
| `hva_block::onsite_layer` | `build_hubbard_onsite(...)` | `grouped_exact` |
| `hva_block::potential_layer` | `build_hubbard_potential(...)` | `grouped_exact` |

Rules:

- Use `repr_mode="JW"`.
- Use `pbc = boundary.strip().lower() == "periodic"` for kinetic edges.
- Use the same scalar `dv` convention as existing Hubbard block builders.
- Clean each polynomial with `_clean_real_pool_polynomial(...)`.
- Skip any cleaned polynomial with no terms.
- Set `execution_mode="grouped_exact"` on each HVA block `AnsatzTerm`.

Do **not** use `HubbardLayerwiseAnsatz.base_terms`, because those are representative, not grouped full polynomials.

#### New combined builder

Shape:

```python
def _build_hubbard_uccsd_qeb_hva_blocks_pool(
    *,
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
    t: float,
    u: float,
    dv: float,
    boundary: str,
) -> list[AnsatzTerm]:
    ...
```

Algorithm:

1. Build `raw_uccsd = _build_uccsd_pool(...)`.
2. Build `raw_qeb = _build_qeb_pool(2 * num_sites)`.
3. Build `raw_hva = _build_hubbard_hva_blocks_pool(...)`.
4. Concatenate in this exact order:

   ```text
   raw_uccsd + raw_qeb + raw_hva
   ```

5. Deduplicate with `_deduplicate_pool_terms(...)`.

The first-occurrence retention policy is intentional: it preserves existing `uccsd_qeb` UCCSD/QEB precedence and only adds HVA labels when structurally new.

#### New family-id helper

Add:

```python
def _hubbard_uccsd_qeb_hva_family_id_for_label(label: str) -> str:
    ...
```

Mapping:

| Label pattern | Family id |
|---|---|
| `uccsd_sing(...)`, `uccsd_dbl(...)` | existing `_hubbard_uccsd_qeb_family_id_for_label(...)` behavior |
| `qeb_pair(...)`, `qeb_double(...)` | `qeb_excitation` |
| `hva_block::hop_layer`, `hva_block::onsite_layer`, `hva_block::potential_layer` | `hva_hamiltonian_blocks` |
| fallback | `uccsd_qeb_hva_blocks` |

---

### 3.3 Pool resolution

In `pool_resolution.resolve_pool_plan()` add a Hubbard branch:

```python
elif pool_key == "uccsd_qeb_hva_blocks":
    if problem_key != "hubbard":
        raise ValueError(...)
    pool = _build_hubbard_uccsd_qeb_hva_blocks_pool(...)
    method_name = "hardcoded_adapt_vqe_uccsd_qeb_hva_blocks_hubbard"
```

Then extend `pool_family_ids` assignment:

```python
if problem_key == "hubbard" and pool_key == "uccsd_qeb":
    ...
elif problem_key == "hubbard" and pool_key == "uccsd_qeb_hva_blocks":
    pool_family_ids = [
        _hubbard_uccsd_qeb_hva_family_id_for_label(str(term.label))
        for term in pool
    ]
else:
    ...
```

For staged non-HH modes, keep existing behavior:

```python
phase1_core_limit = len(pool)
phase1_residual_indices = set()
```

No selected-logical or HH symmetry filtering changes are needed.

---

### 3.4 Runtime materialization support

Extend `build_runtime_pool_terms()` in `primitive_pools.py` with key `uccsd_qeb_hva_blocks` for debug/prelaunch tooling parity.

Metadata should include:

```python
{
    "family": "uccsd_qeb_hva_blocks",
    "problem_key": "hubbard",
    "candidate_pool_complete": True,
    "raw_uccsd_count": ...,
    "raw_qeb_count": ...,
    "raw_hva_block_count": ...,
    "dedup_total": ...,
    "dedup_policy": "uccsd_qeb_first_then_hva_structural_signature_v1",
    "surviving_qeb_count": ...,
    "surviving_hva_block_count": ...,
}
```

This is additive and does not affect `resolve_pool_plan()` behavior.

---

### 3.5 Hubbard physical-lane provenance

Modify `pipelines/contracts/static_provenance.py`.

#### New lane constant

Add an explicit Hubbard alias:

```python
HUBBARD_PHYSICAL_OPERATOR_LANE_HVA_HAMILTONIAN_BLOCKS = "hva_hamiltonian_blocks"
```

It may reuse the existing HH string value, but should have a Hubbard-named export for provenance clarity.

#### Bump version

Before:

```python
HUBBARD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION =
    "hubbard_physical_operator_lanes_v2_uccsd_qeb_split"
```

After:

```python
"hubbard_physical_operator_lanes_v3_uccsd_qeb_hva_blocks"
```

#### Update lane tuple

After:

```python
HUBBARD_PHYSICAL_OPERATOR_LANES = (
    HH_PHYSICAL_OPERATOR_LANE_UCCSD_SINGLE,
    HH_PHYSICAL_OPERATOR_LANE_UCCSD_DOUBLE,
    HUBBARD_PHYSICAL_OPERATOR_LANE_QEB_EXCITATION,
    HUBBARD_PHYSICAL_OPERATOR_LANE_HVA_HAMILTONIAN_BLOCKS,
    HH_PHYSICAL_OPERATOR_LANE_OTHER,
)
```

#### Update classifier

In `_classify_hubbard_physical_operator_lane(label)` classify HVA/block labels before fallback:

Recognized as `hva_hamiltonian_blocks`:

- `hva_block::hop_layer`
- `hva_block::onsite_layer`
- `hva_block::potential_layer`
- optionally existing block labels:
  - `ham_block::hop(...)`
  - `ham_block::onsite(...)`
  - `ham_block::pot(...)`
- optionally bare layer labels:
  - `hop_layer`
  - `onsite_layer`
  - `potential_layer`

Bump returned schema to:

```text
hubbard_physical_operator_lane_classification_v3
```

Do not import pool builders into `static_provenance.py`; it must remain implementation-free.

---

### 3.6 Route metadata

Modify `pipelines/static_adapt/lane_routes.py`.

Only bump the Hubbard route id:

```python
"route_a_hubbard_physical_operator_lanes_v3_uccsd_qeb_hva_blocks"
```

Leave HH, spin_boson, and bose_hubbard route ids unchanged.

`resolve_static_shortlist_lane_spec("physical_operator_type", problem="hubbard")` will automatically include the new lane through `physical_operator_lanes_for_problem("hubbard")`.

---

### 3.7 Runtime prelaunch gate

Modify `adapt_pipeline.py`.

Current gate protects only:

```python
problem_key == "hubbard" and pool_key == "uccsd_qeb"
```

Replace with protected-pool logic:

```python
protected_hubbard_physical_lane_pools = {
    "uccsd_qeb",
    "uccsd_qeb_hva_blocks",
}
```

For both protected pools require:

1. `static_lane_route_key == STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE`
2. `prelaunch_pool_audit["other_count"] == 0`
3. `prelaunch_pool_audit["exact_other_labels"]` empty
4. `lane_counts["qeb_excitation"] > 0`

For `uccsd_qeb_hva_blocks` additionally require:

```python
lane_counts["hva_hamiltonian_blocks"] > 0
```

Error messages must include:

- pool key
- route key
- `other_count`
- `exact_other_labels`
- `lane_counts`

This gate runs after pool resolution and child/shared-pool mutations, so it audits the final emitted labels that the ADAPT loop will use.

---

### 3.8 Parameterization behavior

Do not add a new global parameterization override unless tests reveal the existing execution-mode path is bypassed.

The new HVA block `AnsatzTerm`s must carry:

```python
execution_mode="grouped_exact"
```

`adapt_pipeline._ansatz_terms_require_logical_shared_parameterization(...)` already detects this and forces `logical_shared` when a grouped HVA block is selected. Existing `uccsd_qeb` terms remain unchanged.

---

## 4. **File-by-file impact**

### `pipelines/contracts/static_provenance.py`

Changes:

- Add `HUBBARD_PHYSICAL_OPERATOR_LANE_HVA_HAMILTONIAN_BLOCKS`.
- Bump `HUBBARD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION`.
- Add HVA lane to `HUBBARD_PHYSICAL_OPERATOR_LANES`.
- Extend `_classify_hubbard_physical_operator_lane(...)`.
- Bump returned Hubbard schema string to v3.
- Export the new constant in `__all__`.

Depends on: chosen emitted HVA labels from `primitive_pools.py`.

---

### `pipelines/static_adapt/lane_routes.py`

Changes:

- Update `PHYSICAL_LANE_ROUTE_VARIANT_IDS_BY_PROBLEM["hubbard"]` to v3.

Depends on: new Hubbard lane/version in `static_provenance.py`.

---

### `pipelines/static_adapt/builders/primitive_pools.py`

Changes:

- Import `build_hubbard_kinetic`, `build_hubbard_onsite`, `build_hubbard_potential`.
- Add `_build_hubbard_hva_blocks_pool(...)`.
- Add `_build_hubbard_uccsd_qeb_hva_blocks_pool(...)`.
- Add `_hubbard_uccsd_qeb_hva_family_id_for_label(...)`.
- Extend `build_runtime_pool_terms(...)` for `uccsd_qeb_hva_blocks`.
- Export new helper symbols in `__all__`.

Depends on: no classifier dependency; labels must match classifier tests.

---

### `pipelines/static_adapt/builders/__init__.py`

Changes:

- Re-export:
  - `_build_hubbard_hva_blocks_pool`
  - `_build_hubbard_uccsd_qeb_hva_blocks_pool`
  - `_hubbard_uccsd_qeb_hva_family_id_for_label`

Depends on: primitive pool additions.

---

### `pipelines/static_adapt/builders/problem_registry.py`

Changes:

- Add `uccsd_qeb_hva_blocks` to `_HUBBARD_POOL_KEYS`, adjacent to `uccsd_qeb`.
- Do not change `default_pool_key="uccsd"`.

Depends on: resolver support should land atomically with this, otherwise parser accepts an unresolved pool.

---

### `pipelines/static_adapt/builders/pool_resolution.py`

Changes:

- Import new combined builder and family-id helper.
- Add resolver branch for `pool_key == "uccsd_qeb_hva_blocks"` under Hubbard.
- Extend `pool_family_ids` handling for the new key.

Depends on: primitive pool helper and registry key.

---

### `pipelines/static_adapt/adapt_pipeline.py`

Changes:

- Import Hubbard lane constants if desired for gate clarity.
- Replace single `uccsd_qeb` gate with protected-pool set.
- Add HVA survivor requirement for `uccsd_qeb_hva_blocks`.

No change required to current checkpoint/final result serialization; `adapt_pool`, `max_depth`, `static_lane_policy`, and `physical_operator_lane_policy` are already persisted.

Depends on: classifier and route updates.

---

### `pipelines/static_adapt/cli_config.py`

No production change required.

Parser choices are registry-driven, so adding `_HUBBARD_POOL_KEYS` is sufficient.

---

### Tests

Update/add focused tests in:

#### `test/test_static_adapt_algebraic_metadata.py`

Add assertions that Hubbard classifier maps:

- `hva_block::hop_layer`
- `hva_block::onsite_layer`
- `hva_block::potential_layer`
- optionally `ham_block::hop(...)`

to `hva_hamiltonian_blocks`.

Extend audit test so a mixed list of UCCSD, QEB, and HVA labels has:

```python
other_count == 0
exact_other_labels == []
lane_counts["hva_hamiltonian_blocks"] > 0
```

#### `test/test_static_adapt_lane_routes.py`

Update Hubbard route-id expectation to v3 and assert the new lane appears in `spec.lanes`.

#### `test/test_static_adapt_builders.py`

Add builder tests:

1. HVA block builder emits grouped labels and `execution_mode="grouped_exact"`.
2. Combined pool deduplicates structurally.
3. Combined pool retains at least one UCCSD label, one QEB label, and one HVA-block label.
4. `summarize_static_physical_operator_pool_labels(..., problem="hubbard")` over combined labels has zero `other`.

#### `test/test_adapt_vqe_integration.py`

Add parser acceptance test:

```text
--problem hubbard --adapt-pool uccsd_qeb_hva_blocks
```

---

## 5. **Risks and migration**

- This is additive for pool resolution and CLI behavior; existing `uccsd_qeb` remains unchanged.
- Hubbard physical-lane classifier version and route id intentionally change. Old result JSONs are not migrated; new runs will carry v3 provenance.
- If child/shared pool expansion is enabled in the copied baseline settings and it rewrites final labels into unclassified forms, the new prelaunch gate will fail before the ADAPT loop. That is desired for provenance safety.
- If all HVA blocks deduplicate away for a pathological tiny/zero Hamiltonian setting, the new pool gate will fail because `hva_hamiltonian_blocks` has no survivor.

---

## 6. **Implementation order**

1. **Static provenance first**
   - Add Hubbard HVA lane, bump classifier/schema, extend classifier.
   - Update route id.
   - Add/update classifier and route tests.

2. **Pool builder**
   - Add grouped HVA block builder.
   - Add combined UCCSD+QEB+HVA builder.
   - Add runtime-pool materialization support.
   - Add builder/export tests.

3. **Registry and resolver**
   - Add `uccsd_qeb_hva_blocks` to `_HUBBARD_POOL_KEYS`.
   - Add `resolve_pool_plan()` branch and family-id handling.
   - Add parser test.

4. **Runtime gate**
   - Extend Hubbard physical-lane prelaunch gate to protect both `uccsd_qeb` and `uccsd_qeb_hva_blocks`.
   - Require HVA survivor only for the new pool.

5. **Targeted tests**
   - Run focused tests for static provenance, lane routes, builder pool, parser acceptance, and resolver/audit.

6. **Prelaunch audit before run**
   - Resolve the intended Hubbard weak problem with:
     - `adapt_pool=uccsd_qeb_hva_blocks`
     - `static_lane_route=physical_operator_type`
     - copied baseline settings otherwise
   - Classify final emitted labels.
   - Stop unless:
     - `other_count == 0`
     - `exact_other_labels == []`
     - `lane_counts["qeb_excitation"] > 0`
     - `lane_counts["hva_hamiltonian_blocks"] > 0`

7. **Run Hubbard weak depth 10**
   - Locate the prior Hubbard weak `uccsd_qeb` settings/result.
   - Reuse all optimizer, seed, route, output-layout, batching, and benchmark settings.
   - Change only:
     - `--adapt-pool uccsd_qeb_hva_blocks`
     - physical lane bundle if not already present:
       - `--static-lane-route physical_operator_type`
       - `--physical-lane-shortlist-aggressiveness 3`
     - `--adapt-max-depth 10`
   - Do not launch duplicate unrelated jobs and do not edit manuscript files.


> 💡 Continue this plan conversation with ask_oracle(chat_id: "hubbard-hva-lane-A352F4", new_chat: false)