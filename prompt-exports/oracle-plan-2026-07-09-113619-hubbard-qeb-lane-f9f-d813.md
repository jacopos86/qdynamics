## Final Prompt
<taskname="Hubbard QEB Lane"/>

<task>
Implement only after user approval/Oracle gate: add a provenance-safe Paper-I Hubbard weak rerun path using physical operator lanes with a new QEB/qubit-excitation lane. The first target is Hubbard weak only, combining the existing Hubbard UCCSD pool with existing repo QEB/qubit-excitation operators, structurally deduplicated. The lane classifier must classify final real pool labels as `uccsd_single`, `uccsd_double`, `qeb_excitation`, or fallback `other`, and launch is blocked unless a pre-launch audit proves `other_count == 0` and lists no exact `other` labels. Do not touch manuscript files and do not duplicate unrelated jobs.
</task>

<architecture>
- Pool entry path: `cli_config.py --adapt-pool` choices come from `problem_registry.available_adapt_pool_keys()`; `problem_registry._HUBBARD_POOL_KEYS` controls whether a new combined Hubbard pool key is CLI-visible; `pool_resolution.resolve_pool_plan()` maps that key to concrete `AnsatzTerm` pools and provenance `method_name`/`pool_stage_family`/`pool_family_ids`.
- UCCSD source: `primitive_pools._build_uccsd_pool()` returns `AnsatzTerm` labels emitted by `HardcodedUCCSDAnsatz`: `uccsd_sing(...)` and `uccsd_dbl(...)`.
- QEB source: `generic_static_adapt_variants.build_pairwise_qubit_excitation_pool()` builds benchmark-local QEB singles/doubles in repo exyz convention, emitting `_PoolCandidate` labels `qeb_pair(i,j)` and `qeb_double(i,j->a,b)` with `PauliPolynomial`s. Reuse/lift this construction carefully; avoid making the static ADAPT runtime depend on benchmark comparator plumbing unless that dependency is explicitly accepted.
- Structural dedup: `primitive_pools._polynomial_signature()`, `_polynomial_signature_digest()`, `_deduplicate_pool_terms()`, and `_deduplicate_pool_terms_lightweight()` already normalize real Pauli-polynomial content. Preserve clear label policy when a duplicate is dropped, because lane audit sees final emitted labels.
- Physical lanes: `static_provenance.py` owns Hubbard classifier version/lanes and `classify_static_physical_operator_lane()`. `lane_routes.py` exposes the problem-specific physical route variant id and lane list to runtime provenance.
- Runtime provenance: `adapt_pipeline.py` scales raw 42/21/0.4375 by `--physical-lane-shortlist-aggressiveness 3`, records `static_lane_policy` and `physical_operator_lane_policy`, increments `other_count`, writes current checkpoints, and embeds final lane policy in `adapt_vqe.continuation`.
</architecture>

<selected_context>
- `pipelines/contracts/static_provenance.py`: full classifier constants and dispatch. Current Hubbard lanes are only UCCSD single/double/other; add `qeb_excitation`, bump classifier/schema/version as needed, and classify exactly emitted QEB labels.
- `pipelines/static_adapt/lane_routes.py`: full physical route contract and Hubbard route variant id; update the Hubbard variant id to reflect QEB lane semantics.
- `pipelines/static_adapt/builders/primitive_pools.py`: full primitive pool builders, UCCSD builder, polynomial signature/dedup helpers, runtime pool materialization, `__all__`; likely home for a combined UCCSD+QEB builder or shared QEB construction.
- `pipelines/static_adapt/builders/pool_resolution.py`: slices for imports/dataclasses and `resolve_pool_plan()` non-HH branch; map the new Hubbard pool key and ensure metadata remains distinct from plain `uccsd`.
- `pipelines/static_adapt/builders/problem_registry.py`: slices for request/context resolution, `_HUBBARD_POOL_KEYS`, registry entries, `available_adapt_pool_keys()`, and default pool resolution.
- `pipelines/static_adapt/builders/__init__.py`: full builder reexports; update only if the new helper should be exported.
- `pipelines/static_adapt/cli_config.py`: slices for parser flags and kwargs mapping for pool, physical lanes, raw caps/fraction, no batching, full reopt, refit toggles, target threshold.
- `pipelines/static_adapt/adapt_pipeline.py`: slices for cap scaling, pool resolution call, lane policy payloads, candidate classifier payload, current.json monitoring, final continuation payload.
- `pipelines/static_adapt/phase_shortlists.py`: full generic lane shortlist implementation; no expected change if lane list/classifier are correct.
- `pipelines/static_adapt/output_artifacts.py`: slices for result JSON settings persistence and output logging.
- `pipelines/exact_bench/generic_static_adapt_variants.py`: QEB generator source and comparator-local provenance warning.
- `src/quantum/pauli_polynomial_class.py`, `qubitization_module.py`, `pauli_words.py`, `vqe_latex_python_pairs.py` slices: Pauli/QEB/UCCSD term data structures and label conventions.
- Tests selected: `test_static_adapt_algebraic_metadata.py`, `test_static_adapt_lane_routes.py`, `test_static_adapt_phase_shortlists.py`, `test_generic_static_adapt_variants.py` slices, `test_adapt_vqe_integration.py` slices, `test_static_adapt_builders.py` slices.
- Run-gate docs selected: `agent_guidance/skills/paper-i-run/SKILL.md` slices and `agent_guidance/skills/source-locked-sensitivity/SKILL.md` full.
</selected_context>

<relationships>
- `--adapt-pool <new_hubbard_uccsd_qeb_key>` -> `available_adapt_pool_keys()` -> `_HUBBARD_POOL_KEYS` -> `resolve_pool_plan()` -> combined `AnsatzTerm` pool -> `adapt_pipeline` candidate labels.
- UCCSD labels `uccsd_sing(...)` / `uccsd_dbl(...)` -> `classify_static_physical_operator_lane(problem="hubbard")` -> `uccsd_single` / `uccsd_double`.
- QEB labels `qeb_pair(...)` / `qeb_double(...)` -> new Hubbard classifier rule -> `qeb_excitation`.
- `--static-lane-route physical_operator_type` -> `resolve_static_shortlist_lane_spec(problem="hubbard")` -> lanes from `physical_operator_lanes_for_problem("hubbard")` -> `PhaseShortlistRuntime` quota/health logic.
- Raw shortlist settings 42/21/0.4375 + aggressiveness 3 -> effective 14/7/0.14583333333333334 in `adapt_pipeline.py`.
- `current.json` monitoring -> `checkpoint.depth`, `adapt_vqe.energy`, `adapt_vqe.abs_delta_e`, `operators`, `pool_type`, target-threshold fields; final result -> `adapt_vqe.continuation.physical_operator_lane_policy.other_count`.
</relationships>

<implementation_notes>
- Minimal patch surface is likely `primitive_pools.py`, `pool_resolution.py`, `problem_registry.py`, `static_provenance.py`, `lane_routes.py`, and focused tests. `cli_config.py` should not need a new flag if the new pool key is added to `_HUBBARD_POOL_KEYS`.
- Choose a clear pool key (for example `uccsd_qeb` or similar) and keep it distinct from plain `uccsd` in `method_name`, `pool_stage_family`, and result `adapt_pool` provenance.
- Convert QEB `_PoolCandidate` objects to `AnsatzTerm(label=candidate.label, polynomial=candidate.polynomial)` for the static runtime. Do not preserve benchmark-only run-loop semantics such as QEB comparator replacement policy unless explicitly needed.
- Deduplicate structurally after concatenating UCCSD + QEB. Tests should assert no duplicate `_polynomial_signature()` values in the combined pool and that both UCCSD and QEB labels remain when not duplicates.
- Add classifier tests for `qeb_pair(0,1)` and `qeb_double(0,3->1,2)`, lane-list tests for Hubbard including `qeb_excitation`, parse tests for the new pool key, and resolve-pool tests that classify every final label with `other_count == 0` for the intended Hubbard weak settings.
</implementation_notes>

<run_plan>
Do not launch without explicit user approval. Before any code or run change, satisfy the user’s Oracle/run-gate request if available.

1. Inspect the existing target root directly if filesystem access is available: `raw_outputs/paper_i_alt_hamiltonian_physical_operator_lanes_1p75_fullreopt_agent_repair1_20260709`. RepoPrompt did not expose this root, so do not infer its exact weak-regime args, optimizer, seed, or output layout. Reuse its effective command/settings as the baseline and change only the pool/lane addition.
2. After patch/tests, run a pre-launch pool audit without a full ADAPT loop: resolve the Hubbard weak problem and combined pool, collect all final emitted labels, classify each with `classify_static_physical_operator_lane(label, problem="hubbard")`, summarize lane counts/classifier version/route variant, and stop if any label maps to `other`.
3. Candidate command shape after approval, with exact weak-regime parameters filled from the inspected source root:
   `python -m pipelines.static_adapt.adapt_pipeline --problem hubbard --adapt-pool <new_pool_key> --static-lane-route physical_operator_type --physical-lane-shortlist-aggressiveness 3 --phase1-shortlist-size 42 --phase2-shortlist-size 21 --phase2-shortlist-fraction 0.4375 --phase2-no-batching --phase3-no-batching --adapt-reopt-policy full --adapt-full-refit-every 0 --adapt-final-full-refit false --adapt-max-depth 30 --adapt-benchmark-target-abs-delta-e 1e-5 ...`
4. Use a new, distinct local output root for Hubbard weak only. Do not enqueue Hubbard strong, HH, spin-boson/Rabi, Bose-Hubbard, or unrelated duplicate jobs.
5. Monitor local tmux via targeted `current.json`/result paths. Success threshold is final `abs_delta_e < 1e-5`; report status, depth, energy, abs energy error, lane-policy `other_count`, and source/result paths.
</run_plan>

<ambiguities>
- The named existing raw output root is not visible in RepoPrompt search, so exact source command/settings and the reported ~0.003902 plateau were not independently verified here.
- The final pool key name is not fixed; pick a provenance-clear key and keep all labels/classifier route ids consistent with it.
- If structural dedup drops a QEB term as duplicate of a UCCSD term or vice versa, the label/source policy affects lane counts. Make that policy explicit and test it before launch.
- `test_adapt_vqe_integration.py` references a historical `pipelines/hardcoded/adapt_pipeline.py` path that RepoPrompt did not expose as a file; selected context uses the visible `pipelines/static_adapt/adapt_pipeline.py` implementation.
</ambiguities>

## Selection
- Files: 23 total (12 full, 11 slice)
- Total tokens: 108265 (Auto view)
- Token breakdown: full 51736, slice 56529

### Files
### Selected Files
/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/
├── agent_guidance/
│   └── skills/
│       ├── paper-i-run/
│       │   └── SKILL.md — 1,802 tokens (lines 106-175 (Paper-I run gates: visible/source settings, source-lock baseline, no bulk output scans, run-class discipline, no manuscript table update for smoke.), 697-724 (Reporting/monitoring expectations for current/result JSON, source paths, abs energy error, and current.json caveats.))
│       └── source-locked-sensitivity/
│           └── SKILL.md — 2,143 tokens (full)
├── pipelines/
│   ├── contracts/
│   │   └── static_provenance.py — 7,504 tokens (full)
│   ├── exact_bench/
│   │   └── generic_static_adapt_variants.py — 4,700 tokens (lines 1-140 (Module provenance and imports showing QEB comparator is benchmark-local and uses PauliPolynomial/PauliTerm/AnsatzTerm.), 300-430 (_PoolCandidate dataclass and QEB variant metadata/source naming context.), 640-670 (QEB comparator pool naming/construction strings: qubit_excitation_singles_doubles_pool.), 1400-1549 (Existing QEB/qubit-excitation singles+doubles generator, emitted labels qeb_pair()/qeb_double(), exyz convention, PauliPolynomial construction.))
│   └── static_adapt/
│       ├── builders/
│       │   ├── __init__.py — 982 tokens (full)
│       │   ├── pool_resolution.py — 8,963 tokens (lines 1-120 (Imports, primitive pool builder imports, PoolFilterResolution/PoolResolution dataclasses; add combined pool import here.), 694-760 (resolve_pool_plan signature and initialization, including pool_key/default resolution setup.), 887-1410 (Non-HH pool resolution branches, including Hubbard UCCSD branch where a UCCSD+QEB pool key should be mapped and method_name assigned.; Post-build pool_stage_family/pool_family_ids, legal/filter handling, staged phase limits, and PoolResolution return metadata.))
│       │   ├── primitive_pools.py — 22,300 tokens (full)
│       │   └── problem_registry.py — 8,644 tokens (lines 1-260 (Imports, ProblemRequest.from_namespace, generic ProblemFamilySpec.resolve, and Hubbard layout setup used by CLI/problem context resolution.), 1400-2004 (Physical family resolver tail plus _HUBBARD_POOL_KEYS and other pool-key tuples; add combined UCCSD+QEB key here for Hubbard CLI availability.; Capabilities and _PROBLEM_REGISTRY entries, including Hubbard family default/admissible pool config and neighboring family patterns.; Public registry helpers: available_adapt_pool_keys(), get_problem_family_spec(), resolve_runtime_default_pool_key(), resolve_problem_context(), and __all__.))
│       ├── adapt_pipeline.py — 12,513 tokens (lines 300-439 (Imports of static provenance classifiers, lane routes, pool builders, and resolve_pool_plan used by physical-lane runtime.), 1475-1535 (_run_hardcoded_adapt_vqe signature showing adapt_current_json, target threshold, pool/problem, and runtime control kwargs.), 1930-2020 (Physical-lane route normalization and effective cap/fraction scaling: raw caps divided by aggressiveness factor.), 3010-3095 (resolve_pool_plan call and unpacking of pool_key/method/pool metadata before lane classification.), 3485-3565 (static_lane_policy and physical_operator_lane_policy summaries including classifier version, route variant, lane list, other_count, and effective caps.), 3780-3857 (_physical_payload_for_candidate chooses labels, calls classify_static_physical_operator_lane(), updates lane_counts and other_count.), 4218-4265 (Candidate cache/provenance fields recording static_lane_route, physical lane classifier version, effective caps, and per-candidate lane labels.), 5038-5077 (PhaseShortlistRuntime wiring for generic physical lane shortlisting using lane key/list/fallback/summary.), 10099-10380 (current.json checkpoint writer fields for monitoring depth, energy, target threshold, pool type, selected operators, history tail, and continuation metadata.), 23192-23210 (Final continuation payload embeds static_lane_policy and physical_operator_lane_policy in result JSON.))
│       ├── cli_config.py — 6,295 tokens (lines 1130-1305 (Parser setup for core Hubbard/problem args, --adapt-pool choices from available_adapt_pool_keys(), and physical lane route/aggressiveness flags.), 1535-1585 (Reoptimization/full-refit CLI flags: --adapt-reopt-policy, --adapt-full-refit-every, --adapt-final-full-refit.), 1740-1760 (Phase-II raw shortlist cap/fraction CLI flags used for 42/21/0.4375 settings.), 2050-2095 (No-batching CLI aliases --phase2-no-batching and --phase3-no-batching plus batch-mode context.), 3458-3568 (_build_run_hardcoded_adapt_vqe_kwargs maps core problem, pool, lane route, aggressiveness, max depth, and target threshold args into runtime kwargs.), 3600-3670 (Runtime kwargs mapping for reopt policy, full refit cadence/final refit, continuation, and beam toggles.), 3918-3978 (Runtime kwargs mapping for phase1/phase2 shortlist caps/fraction and no-batching effective flags.))
│       ├── lane_routes.py — 1,231 tokens (full)
│       ├── output_artifacts.py — 2,703 tokens (lines 483-652 (build_output_payload settings block with adapt_pool, continuation, max depth, exact reference, and optimizer settings persisted into result JSON.), 1020-1069 (persist_output_artifacts writes result JSON/PDF and logs output path, energy, stop reason, depth, abs_delta_e.))
│       └── phase_shortlists.py — 5,467 tokens (full)
├── src/
│   └── quantum/
│       ├── pauli_polynomial_class.py — 1,969 tokens (full)
│       ├── pauli_words.py — 38 tokens (full)
│       ├── qubitization_module.py — 578 tokens (full)
│       └── vqe_latex_python_pairs.py — 3,003 tokens (lines 400-425 (half_filled_num_particles helper used by Hubbard UCCSD pool resolution/tests.), 540-619 (AnsatzTerm dataclass target for converting QEB pool candidates into static ADAPT runtime terms.), 740-930 (HardcodedUCCSDAnsatz construction and UCCSD single/double label emission used by _build_uccsd_pool and Hubbard lane classifier.))
└── test/
    ├── test_adapt_vqe_integration.py — 4,778 tokens (lines 1-245 (Integration-test imports, dynamic pipeline import surface, resolve_pool_plan helper, and selected-ops reconstruction.), 1568-1652 (Parser tests for --adapt-pool choices; natural place to add combined Hubbard UCCSD+QEB pool parse coverage.), 2708-2762 (Pool builder sanity tests including UCCSD L=2 AnsatzTerm shape; natural place for combined-pool smoke test.), 4428-4512 (Hubbard invalid/unsupported pool edge-case tests to preserve HH-only pool rejection and unsupported-pool errors.))
    ├── test_generic_static_adapt_variants.py — 2,087 tokens (lines 680-742 (QEB generator tests for qeb_pair labels, exyz qubit ordering, and basic compiled-depth behavior.), 1688-1762 (QEB double-label usage and duplicate/replacement policy tests, including qeb_double(0,3->1,2).), 2144-2185 (QEB duplicate-polynomial test pattern showing alternate labels can share the same polynomial.))
    ├── test_static_adapt_algebraic_metadata.py — 5,227 tokens (full)
    ├── test_static_adapt_builders.py — 1,041 tokens (lines 1-70 (Builder test imports for primitive_pools, hardcoded/static adaptor surfaces, and Hubbard helpers.), 1390-1430 (Reexport test for extracted helpers; update if a new combined pool helper is exported.))
    ├── test_static_adapt_lane_routes.py — 273 tokens (full)
    └── test_static_adapt_phase_shortlists.py — 4,024 tokens (full)


---

## Generated Plan

## Chat Send ✅
- **Chat**: `hubbard-qeb-lane-F9F345` | **Mode**: plan

### Response
## 1. **Summary**

Use a targeted static-ADAPT extension, not a broad refactor: add a new Hubbard-only pool key `uccsd_qeb` that builds the existing Hubbard UCCSD pool plus repo-native QEB/qubit-excitation generators, structurally deduplicates them, and routes final labels through an updated Hubbard physical-operator lane classifier with a new `qeb_excitation` lane. Add a pre-launch physical-lane audit for this path that blocks execution unless the final emitted pool has `other_count == 0`, no exact `other` labels, and at least one surviving QEB label. Do not touch manuscript files or launch any run until the user/Oracle gate approves the source-locked Hubbard weak rerun.

---

## 2. **Current-state analysis**

### Pool and CLI flow today

- `pipelines/static_adapt/cli_config.py`
  - `--adapt-pool` choices come from `available_adapt_pool_keys()`.
- `pipelines/static_adapt/builders/problem_registry.py`
  - `_HUBBARD_POOL_KEYS` controls Hubbard-admissible pool keys.
  - `available_adapt_pool_keys()` unions all family pool keys for CLI visibility.
- `pipelines/static_adapt/builders/pool_resolution.py`
  - `resolve_pool_plan()` maps `adapt_pool` to concrete `list[AnsatzTerm]`.
  - For non-HH Hubbard:
    - `uccsd` calls `_build_uccsd_pool()`.
    - `cse`, `full_hamiltonian`, `hamiltonian_blocks`, `full_meta` are supported.
  - After construction, non-HH pools currently default:
    - `pool_stage_family = [pool_key] * len(pool)`
    - `pool_family_ids = [pool_key] * len(pool)`

### Existing UCCSD and QEB sources

- UCCSD source:
  - `primitive_pools._build_uccsd_pool()` wraps `HardcodedUCCSDAnsatz.base_terms`.
  - Labels are:
    - `uccsd_sing(...)`
    - `uccsd_dbl(...)`
- QEB source:
  - `pipelines/exact_bench/generic_static_adapt_variants.py`
  - `build_pairwise_qubit_excitation_pool()` emits benchmark-local `_PoolCandidate` objects:
    - `qeb_pair(i,j)`
    - `qeb_double(i,j->a,b)`
  - It already uses repo `PauliPolynomial` / `PauliTerm` and correct exyz qubit-0-LSB convention.
  - Static ADAPT should not import this benchmark module directly; instead extract the QEB construction into static pool builders and have the benchmark wrapper delegate back.

### Physical lane flow today

- `pipelines/contracts/static_provenance.py`
  - Hubbard lanes currently include only:
    - `uccsd_single`
    - `uccsd_double`
    - `other`
  - `_classify_hubbard_physical_operator_lane()` only recognizes UCCSD labels.
  - QEB labels would currently classify as `other`.
- `pipelines/static_adapt/lane_routes.py`
  - `resolve_static_shortlist_lane_spec(..., problem="hubbard")` gets lane list from `physical_operator_lanes_for_problem("hubbard")`.
  - Hubbard route variant id currently reflects only UCCSD split semantics.
- `pipelines/static_adapt/adapt_pipeline.py`
  - `--static-lane-route physical_operator_type` activates physical lane shortlist routing.
  - Raw caps/fraction are divided by `--physical-lane-shortlist-aggressiveness`.
  - Candidate-level physical lane classification happens during feature generation.
  - Final `adapt_vqe.continuation.physical_operator_lane_policy` is persisted.
  - There is no current pre-launch full-pool audit proving all final labels avoid `other`.

### Blocking issues

1. No CLI-visible Hubbard pool exists for UCCSD + QEB.
2. QEB construction is isolated in `exact_bench`, which should not become a runtime dependency of static ADAPT.
3. Hubbard physical lane classifier would send QEB labels to `other`.
4. The final pool is not audited before launch, so a run could start with unclassified labels.

---

## 3. **Design**

### A. New Hubbard pool key: `uccsd_qeb`

Use `uccsd_qeb` as the canonical pool key.

Behavior:

- Valid only for `problem="hubbard"`.
- Builds:
  1. `_build_uccsd_pool(num_sites, num_particles, ordering)`
  2. QEB pool over `num_qubits = 2 * num_sites`
- Concatenates UCCSD first, QEB second.
- Deduplicates structurally with existing `_deduplicate_pool_terms()` / `_polynomial_signature()`.
- Duplicate policy:
  - First occurrence wins.
  - Therefore UCCSD labels are preserved if a QEB generator is structurally identical to UCCSD.
  - QEB lane counts reflect only surviving structurally distinct QEB labels.

Additive helper shapes in `primitive_pools.py`:

```py
@dataclass(frozen=True)
class _QEBPoolTermSpec:
    label: str
    polynomial: PauliPolynomial
    support: tuple[int, ...]
    pauli_labels_exyz: tuple[str, ...]
    construction: str
```

```py
def _build_qeb_pool_specs(
    num_qubits: int,
    *,
    max_terms: int | None = None,
) -> tuple[_QEBPoolTermSpec, ...]:
    ...
```

```py
def _build_qeb_pool(
    num_qubits: int,
    *,
    max_terms: int | None = None,
) -> list[AnsatzTerm]:
    ...
```

```py
def _build_hubbard_uccsd_qeb_pool(
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> list[AnsatzTerm]:
    ...
```

```py
def _hubbard_uccsd_qeb_family_id_for_label(label: str) -> str:
    # uccsd_sing / uccsd_dbl -> "uccsd"
    # qeb_pair / qeb_double -> "qeb_excitation"
    # otherwise -> "uccsd_qeb"
```

### B. Pool resolution

In `resolve_pool_plan()`:

- Add branch for `pool_key == "uccsd_qeb"` in the non-HH path.
- Require `problem_key == "hubbard"`.
- Set:
  - `method_name = "hardcoded_adapt_vqe_uccsd_qeb_hubbard"`
  - `pool_stage_family = ["uccsd_qeb"] * len(pool)`
  - `pool_family_ids = [_hubbard_uccsd_qeb_family_id_for_label(term.label) ...]`

Do not change Hubbard default pool; it remains `uccsd`.

### C. Hubbard physical lane classifier update

In `static_provenance.py`:

Add:

```py
HUBBARD_PHYSICAL_OPERATOR_LANE_QEB_EXCITATION = "qeb_excitation"
HUBBARD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION = (
    "hubbard_physical_operator_lanes_v2_uccsd_qeb_split"
)
```

Update `HUBBARD_PHYSICAL_OPERATOR_LANES` to:

```py
(
    "uccsd_single",
    "uccsd_double",
    "qeb_excitation",
    "other",
)
```

Classifier behavior:

- `uccsd_sing(...)` → `uccsd_single`
- `uccsd_dbl(...)` → `uccsd_double`
- exact canonical QEB labels:
  - `qeb_pair(<int>,<int>)`
  - `qeb_double(<int>,<int>-><int>,<int>)`
  → `qeb_excitation`
- Anything else → `other`

Use anchored matching so labels like `qeb_pair_alt` remain `other`.

### D. Pre-launch pool audit and launch block

Add a label-only audit helper in `static_provenance.py`:

```py
def summarize_static_physical_operator_pool_labels(
    labels: Sequence[Any],
    *,
    problem: Any,
) -> dict[str, Any]:
    ...
```

Audit payload shape:

```json
{
  "schema": "static_physical_operator_pool_lane_audit_v1",
  "problem": "hubbard",
  "classifier_version": "hubbard_physical_operator_lanes_v2_uccsd_qeb_split",
  "classified_count": 0,
  "lane_counts": {
    "uccsd_single": 0,
    "uccsd_double": 0,
    "qeb_excitation": 0,
    "other": 0
  },
  "other_count": 0,
  "exact_other_labels": [],
  "require_no_other_pass": true
}
```

In `adapt_pipeline.py`, after all final pool mutation/expansion steps and before candidate scoring:

- If `static_lane_route_key == "physical_operator_type"`, attach audit to:
  - `physical_operator_lane_summary["prelaunch_pool_audit"]`
- If `problem_key == "hubbard"` and `pool_key == "uccsd_qeb"`:
  - require `static_lane_route_key == "physical_operator_type"`
  - require `other_count == 0`
  - require `exact_other_labels == []`
  - require `lane_counts["qeb_excitation"] > 0`
  - otherwise raise `ValueError` before ADAPT starts.

This is the code-level launch block for the new path.

### E. Route variant id

In `lane_routes.py`, update Hubbard physical lane variant id to:

```py
"route_a_hubbard_physical_operator_lanes_v2_uccsd_qeb_split"
```

HH, spin-boson, and bose-hubbard route ids remain unchanged.

### F. Source-lock and Paper-I run gate

No run is launched by this implementation.

Before the Hubbard weak rerun:

1. Inspect the existing source root:
   - `raw_outputs/paper_i_alt_hamiltonian_physical_operator_lanes_1p75_fullreopt_agent_repair1_20260709`
2. Resolve exact weak-regime command/settings.
3. Generate a machine-readable source-lock audit.
4. Only changed fields should be the user-approved pool/lane bundle:
   - `adapt_pool: uccsd_qeb`
   - classifier/route variant version change
   - any physical-lane route fields only if not already present in source
5. Run the pre-launch pool audit.
6. Block if any final label maps to `other`.

---

## 4. **File-by-file impact**

### `pipelines/static_adapt/builders/primitive_pools.py`

- Add QEB construction helpers extracted from `generic_static_adapt_variants.py`.
- Add `_QEBPoolTermSpec`.
- Add `_build_qeb_pool_specs()`.
- Add `_build_qeb_pool()`.
- Add `_build_hubbard_uccsd_qeb_pool()`.
- Add `_hubbard_uccsd_qeb_family_id_for_label()`.
- Export new helpers in `__all__`.
- Optionally add `build_runtime_pool_terms()` support for `pool_key == "uccsd_qeb"` when `problem_key == "hubbard"`.

### `pipelines/exact_bench/generic_static_adapt_variants.py`

- Replace local QEB construction internals with delegation to `_build_qeb_pool_specs()`.
- Preserve existing `_PoolCandidate` output shape and labels.
- Existing QEB comparator semantics remain unchanged.

### `pipelines/static_adapt/builders/pool_resolution.py`

- Import `_build_hubbard_uccsd_qeb_pool` and family-id helper.
- Add `uccsd_qeb` branch for Hubbard.
- Raise `ValueError` if requested for non-Hubbard.
- Assign distinct `method_name`, `pool_stage_family`, and `pool_family_ids`.

### `pipelines/static_adapt/builders/problem_registry.py`

- Add `"uccsd_qeb"` to `_HUBBARD_POOL_KEYS`, preferably immediately after `"uccsd"`.
- Do not change default pool.

### `pipelines/static_adapt/builders/__init__.py`

- Reexport new primitive pool helper(s) only if the repository convention is to expose all primitive builders here.
- At minimum, reexport `_build_hubbard_uccsd_qeb_pool` if tests or compatibility wrappers need it.

### `pipelines/contracts/static_provenance.py`

- Add `HUBBARD_PHYSICAL_OPERATOR_LANE_QEB_EXCITATION`.
- Bump Hubbard classifier version.
- Add QEB lane to `HUBBARD_PHYSICAL_OPERATOR_LANES`.
- Update `_classify_hubbard_physical_operator_lane()`.
- Add `summarize_static_physical_operator_pool_labels()`.
- Export new constant/helper in `__all__`.

### `pipelines/static_adapt/lane_routes.py`

- Update Hubbard physical route variant id to QEB-aware v2 string.
- No behavior change to generic lane routing.

### `pipelines/static_adapt/adapt_pipeline.py`

- Import the new audit helper.
- Attach `prelaunch_pool_audit` to `physical_operator_lane_summary`.
- Block `hubbard + uccsd_qeb` unless:
  - physical lane route is active,
  - `other_count == 0`,
  - no `exact_other_labels`,
  - at least one `qeb_excitation` label survives dedup.

### Tests

Update/add focused tests in:

- `test/test_static_adapt_algebraic_metadata.py`
  - QEB labels classify as `qeb_excitation`.
  - malformed/noncanonical QEB-like labels classify as `other`.
  - audit helper reports `other_count == 0` for UCCSD/QEB labels.
- `test/test_static_adapt_lane_routes.py`
  - Hubbard physical route lanes include `qeb_excitation`.
  - Hubbard route variant id is updated.
- `test/test_static_adapt_builders.py`
  - Combined pool is nonempty.
  - Contains UCCSD and QEB labels after dedup.
  - `_polynomial_signature()` values are unique.
- `test/test_adapt_vqe_integration.py`
  - Parser accepts `--problem hubbard --adapt-pool uccsd_qeb`.
  - `resolve_pool_plan()` returns `pool_key == "uccsd_qeb"`.
  - All final labels classify non-`other`.
  - Non-Hubbard use raises.
- `test/test_generic_static_adapt_variants.py`
  - Existing QEB tests should pass unchanged after delegation.

---

## 5. **Risks and migration**

- No persisted schema migration is required.
- New result JSONs may contain:
  - `settings.adapt_pool = "uccsd_qeb"`
  - Hubbard physical classifier version v2.
  - `qeb_excitation` in physical lane lists/counts.
  - `prelaunch_pool_audit` under `adapt_vqe.continuation.physical_operator_lane_policy`.
- Old code will not understand the new pool key; rollback means using older artifacts or rerunning with `--adapt-pool uccsd`.
- Main implementation risk is QEB helper extraction changing exact-bench QEB ordering. Preserve `pauli_labels_exyz` order from the current benchmark implementation.
- QEB doubles scale combinatorially in qubit count; this path is only intended for the approved Hubbard weak target, not broad suite expansion.

---

## 6. **Implementation order**

1. Add QEB spec builder in `primitive_pools.py` by extracting current benchmark-local QEB construction.
2. Refactor `generic_static_adapt_variants.build_pairwise_qubit_excitation_pool()` to delegate to the new QEB spec builder.
3. Add `_build_hubbard_uccsd_qeb_pool()` with UCCSD-first structural dedup.
4. Add Hubbard `qeb_excitation` lane, classifier version bump, and pool-label audit helper.
5. Add `uccsd_qeb` to `_HUBBARD_POOL_KEYS`.
6. Wire `resolve_pool_plan()` for `uccsd_qeb`.
7. Update `lane_routes.py` Hubbard physical route variant id.
8. Add `adapt_pipeline.py` pre-launch audit/guard for `hubbard + uccsd_qeb`.
9. Add/update focused tests.
10. After tests pass, stop. Do not launch.
11. Only after explicit user/Oracle approval:
    - inspect the source root,
    - create the source-lock settings audit,
    - run the pre-launch pool audit,
    - launch Hubbard weak only if both audits pass.


> 💡 Continue this plan conversation with ask_oracle(chat_id: "hubbard-qeb-lane-F9F345", new_chat: false)