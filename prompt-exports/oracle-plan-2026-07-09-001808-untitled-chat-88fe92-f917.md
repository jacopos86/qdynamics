# Oracle Plan

## 1. Summary

Use a targeted extension, not a broad refactor. The existing `physical_operator_type` route already has the right lane-shortlist machinery; it is HH-only because the lane contract, summary, and classifier dispatch are hard-coded to HH. Add problem-specific label classifiers for `hubbard`, `spin_boson`, and `bose_hubbard`, dispatch by `problem_key`, and leave HH constants/classifier behavior unchanged. Do not modify pool builders; reuse the existing labels emitted by `HardcodedUCCSDAnsatz`, `build_spin_boson_full_meta_terms`, and `build_bose_hubbard_full_meta_terms`.

---

## 2. Current-state analysis

### Existing route flow

1. CLI parses:
   - `--static-lane-route {algebraic,physical_operator_type}`
   - `--physical-lane-shortlist-aggressiveness {2,3}`

2. `adapt_pipeline.py` normalizes:
   - `static_lane_route_key = normalize_static_lane_route(...)`
   - `physical_lane_shortlist_aggressiveness_val = normalize_physical_lane_shortlist_aggressiveness(...)`
   - `shortlist_lane_spec = resolve_static_shortlist_lane_spec(static_lane_route_key)`

3. For `physical_operator_type`, current code:
   - Uses HH lane constants from `static_provenance.py`
   - Applies the aggressiveness factor to effective Phase I / Phase II shortlist caps
   - Raises unless `problem_key == "hh"`:
     ```py
     if static_lane_route_key == STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE and problem_key != "hh":
         raise ValueError(...)
     ```

4. Candidate rows get lane metadata through `_physical_payload_for_candidate(...)`, which currently calls HH-only:
   ```py
   _classify_hh_physical_operator_lane(label)
   ```

5. Existing lane shortlist calls are already generic:
   - `_phase1_lane_shortlist_with_legacy_hook(...)`
   - `_phase2_lane_health_shortlist_with_legacy_hook(...)`

So the blocking pieces are only the HH guard, HH-only lane contract, and HH-only classifier call.

---

## 3. Recommended lane taxonomy

### A. Hubbard / UCCSD-only route

Use existing `HardcodedUCCSDAnsatz.base_terms` labels from `src/quantum/vqe_latex_python_pairs.py`.

| Lane | Match existing label prefixes |
|---|---|
| `uccsd_single` | `uccsd_sing(alpha:`, `uccsd_sing(beta:` |
| `uccsd_double` | `uccsd_dbl(aa:`, `uccsd_dbl(bb:`, `uccsd_dbl(ab:` |
| `other` | fallback |

Recommended run pool for Hubbard: `--adapt-pool uccsd`.

Do **not** rely on `--adapt-pool full_meta` for the Hubbard lane route unless additional Hubbard full-meta label classifiers are added later.

---

### B. Spin-boson / Rabi full-meta route

Use existing labels emitted in `src/quantum/operator_pools/spin_boson.py`, then wrapped by `primitive_pools.py` as:

```text
full_meta::{label}
```

Recommended lanes:

| Lane | Match existing label prefixes |
|---|---|
| `emitter_matter` | `full_meta::emitter_flip`, `full_meta::emitter_imbalance`, `full_meta::emitter_y` |
| `boson_linear` | `full_meta::boson_number`, `full_meta::boson_displacement`, `full_meta::boson_momentum` |
| `boson_nonlinear` | `full_meta::boson_x_sq`, `full_meta::boson_p_sq`, `full_meta::boson_n_sq`, `full_meta::boson_squeeze_x`, `full_meta::boson_xp_sym`, bare `full_meta::n_x`, bare `full_meta::n_p` |
| `transverse_coupling` | `full_meta::transverse_x`, `full_meta::transverse_p`, `full_meta::number_weighted_flip`, `full_meta::x_sq_flip`, `full_meta::p_sq_flip`, `full_meta::n_sq_flip`, `full_meta::n_x_flip`, `full_meta::n_p_flip`, `full_meta::transverse_coupling` |
| `longitudinal_coupling` | `full_meta::longitudinal_x`, `full_meta::longitudinal_p`, `full_meta::number_weighted_imbalance`, `full_meta::x_sq_imbalance`, `full_meta::p_sq_imbalance`, `full_meta::n_sq_imbalance`, `full_meta::n_x_imbalance`, `full_meta::n_p_imbalance`, `full_meta::longitudinal_coupling` |
| `emitter_y_correlation` | `full_meta::x_sq_emitter_y`, `full_meta::p_sq_emitter_y` |
| `other` | fallback |

Classifier order matters: match longer coupling prefixes like `n_x_imbalance` / `n_x_flip` before bare `n_x`.

---

### C. Bose-Hubbard full-meta route

Use existing labels emitted in `src/quantum/operator_pools/boson_chains.py`, then wrapped by `primitive_pools.py` as:

```text
full_meta::{label}
```

Recommended lanes:

| Lane | Match existing label prefixes |
|---|---|
| `number_density_interaction` | `full_meta::n_<site>`, `full_meta::n_sq_<site>`, `full_meta::number_<site>`, `full_meta::interaction_<site>`, `full_meta::staggered_number_<site>`, `full_meta::nn_<i>_<j>` |
| `onsite_quadrature` | `full_meta::x_<site>`, `full_meta::p_<site>`, `full_meta::x_sq_<site>`, `full_meta::p_sq_<site>`, `full_meta::squeeze_x_<site>`, `full_meta::squeeze_p_<site>`, `full_meta::n_x_<site>`, `full_meta::n_p_<site>`, `full_meta::n_x_sq_<site>`, `full_meta::n_p_sq_<site>` |
| `single_particle_transport` | `full_meta::hop_<i>_<j>`, `full_meta::current_<i>_<j>` |
| `intersite_quadrature` | `full_meta::xx_<i>_<j>`, `full_meta::pp_<i>_<j>` |
| `density_assisted_transport` | `full_meta::density_hop_<i>_<j>_left`, `full_meta::density_hop_<i>_<j>_right`, `full_meta::density_current_<i>_<j>_left`, `full_meta::density_current_<i>_<j>_right` |
| `pair_transport` | `full_meta::pair_hop_<i>_<j>`, `full_meta::pair_current_<i>_<j>` |
| `other` | fallback |

---

## 4. Minimal code-change plan

### `pipelines/contracts/static_provenance.py`

Add problem-specific constants and classifiers while leaving all existing HH symbols unchanged.

Add classifier versions:

```py
HUBBARD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION = "hubbard_physical_operator_lanes_v1_uccsd_split"
SPIN_BOSON_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION = "spin_boson_physical_operator_lanes_v1_full_meta"
BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION = "bose_hubbard_physical_operator_lanes_v1_full_meta"
```

Add lane tuples:

```py
HUBBARD_PHYSICAL_OPERATOR_LANES = (
    "uccsd_single",
    "uccsd_double",
    "other",
)

SPIN_BOSON_PHYSICAL_OPERATOR_LANES = (
    "emitter_matter",
    "boson_linear",
    "boson_nonlinear",
    "transverse_coupling",
    "longitudinal_coupling",
    "emitter_y_correlation",
    "other",
)

BOSE_HUBBARD_PHYSICAL_OPERATOR_LANES = (
    "number_density_interaction",
    "onsite_quadrature",
    "single_particle_transport",
    "intersite_quadrature",
    "density_assisted_transport",
    "pair_transport",
    "other",
)
```

Add additive dispatch helpers:

```py
def normalize_static_physical_operator_problem(problem: Any) -> str: ...

def physical_operator_lanes_for_problem(problem: Any) -> tuple[str, ...]: ...

def physical_operator_classifier_version_for_problem(problem: Any) -> str: ...

def classify_static_physical_operator_lane(
    label: str,
    *,
    problem: Any,
    hh_full_meta_class: str | None = None,
) -> dict[str, str | None]: ...
```

Dispatch behavior:

- `problem == "hh"` delegates to existing `classify_hh_physical_operator_lane(...)` unchanged.
- `problem == "hubbard"` uses UCCSD label prefixes.
- `problem == "spin_boson"` uses spin-boson full-meta prefixes.
- `problem == "bose_hubbard"` uses Bose-Hubbard full-meta prefixes.
- Unsupported problem raises early only when physical route is requested.

---

### `pipelines/static_adapt/lane_routes.py`

Modify `resolve_static_shortlist_lane_spec` to accept problem context:

Before:

```py
def resolve_static_shortlist_lane_spec(route: Any) -> StaticShortlistLaneSpec:
```

After:

```py
def resolve_static_shortlist_lane_spec(
    route: Any,
    *,
    problem: Any = "hh",
) -> StaticShortlistLaneSpec:
```

For algebraic route, behavior is unchanged.

For physical route:

```py
return StaticShortlistLaneSpec(
    route=STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
    lane_key="physical_operator_lane",
    lanes=physical_operator_lanes_for_problem(problem),
    fallback_lane="other",
    health_key_prefix="physical_operator",
)
```

Keep existing:

```py
PHYSICAL_LANE_ROUTE_VARIANT_ID = "route_a_physical_operator_lanes_v2_uccsd_split"
```

as the HH variant ID.

Add problem-specific variant IDs:

```py
PHYSICAL_LANE_ROUTE_VARIANT_IDS_BY_PROBLEM = {
    "hh": PHYSICAL_LANE_ROUTE_VARIANT_ID,
    "hubbard": "route_a_hubbard_physical_operator_lanes_v1_uccsd_split",
    "spin_boson": "route_a_spin_boson_physical_operator_lanes_v1_full_meta",
    "bose_hubbard": "route_a_bose_hubbard_physical_operator_lanes_v1_full_meta",
}
```

---

### `pipelines/static_adapt/adapt_pipeline.py`

Move/duplicate `problem_key` normalization before lane spec resolution:

```py
problem_key = str(problem).strip().lower()
static_lane_route_key = normalize_static_lane_route(static_lane_route)
shortlist_lane_spec = resolve_static_shortlist_lane_spec(
    static_lane_route_key,
    problem=problem_key,
)
```

Replace the HH-only guard:

Before:

```py
if static_lane_route_key == STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE and problem_key != "hh":
    raise ValueError(...)
```

After:

```py
if static_lane_route_key == STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE:
    normalize_static_physical_operator_problem(problem_key)
```

This raises for unsupported problems but allows:

```text
hh
hubbard
spin_boson
bose_hubbard
```

Update physical lane summary initialization to use `shortlist_lane_spec.lanes` instead of `_HH_PHYSICAL_OPERATOR_LANES`.

Keep HH schema unchanged:

```py
"schema": (
    "hh_physical_operator_lane_policy_v1"
    if problem_key == "hh"
    else "static_physical_operator_lane_policy_v1"
)
```

Replace classifier call inside `_physical_payload_for_candidate(...)`:

Before:

```py
payload = _classify_hh_physical_operator_lane(label)
```

After:

```py
payload = classify_static_physical_operator_lane(
    label,
    problem=problem_key,
)
```

Keep the existing label-source priority unchanged:

1. `runtime_split_parent_label`
2. metadata parent/source/template/base labels
3. feature candidate label
4. `candidate_term.label`

This is important for runtime-split candidates.

No changes are needed at the Phase I / Phase II shortlist call sites.

---

### `pipelines/static_adapt/cli_config.py`

Only update help text. Do not change parser choices.

Current help says physical route uses HH operator families. Update to say:

- HH physical families remain supported.
- Hubbard supports UCCSD singles/doubles.
- Spin-boson/Rabi and Bose-Hubbard support full-meta physical lanes.

No parser change is needed for the “1.75-less-aggressive” route; keep using integer `--physical-lane-shortlist-aggressiveness 3` with scaled base caps.

---

## 5. Smoke tests / dry runs

### A. Classifier-only smoke test

After implementation:

```bash
cd /Users/jakestrobel/local_repos/Holstein_test_fullclone_3

python - <<'PY'
from pipelines.contracts.static_provenance import classify_static_physical_operator_lane

samples = [
    ("hubbard", "uccsd_sing(alpha:0->2)"),
    ("hubbard", "uccsd_dbl(ab:0,4->2,6)"),
    ("spin_boson", "full_meta::emitter_flip"),
    ("spin_boson", "full_meta::longitudinal_x"),
    ("spin_boson", "full_meta::n_x_flip"),
    ("bose_hubbard", "full_meta::hop_0_1"),
    ("bose_hubbard", "full_meta::density_current_0_1_left"),
    ("bose_hubbard", "full_meta::pair_hop_0_1"),
]

for problem, label in samples:
    print(problem, label, "=>", classify_static_physical_operator_lane(label, problem=problem)["physical_operator_lane"])
PY
```

Expected: no sample maps to `other`.

---

### B. Builder-label coverage smoke test

Hubbard UCCSD:

```bash
python - <<'PY'
from collections import Counter
from src.quantum.vqe_latex_python_pairs import HardcodedUCCSDAnsatz
from pipelines.contracts.static_provenance import classify_static_physical_operator_lane

terms = HardcodedUCCSDAnsatz(
    dims=4,
    num_particles=(2, 2),
    reps=1,
    repr_mode="JW",
    indexing="blocked",
).base_terms

counts = Counter(
    classify_static_physical_operator_lane(t.label, problem="hubbard")["physical_operator_lane"]
    for t in terms
)
print(counts)
assert counts["other"] == 0
PY
```

Spin-boson full-meta:

```bash
python - <<'PY'
from collections import Counter
from src.quantum.operator_pools.spin_boson import build_spin_boson_full_meta_terms
from pipelines.contracts.static_provenance import classify_static_physical_operator_lane

labels = [
    f"full_meta::{label}"
    for label, _ in build_spin_boson_full_meta_terms(
        num_sites=1,
        t=1.0,
        u=0.5,
        dv=0.1,
        omega0=1.0,
        g_ep=0.2,
        n_ph_max=2,
        boson_encoding="binary",
    )
]

counts = Counter(
    classify_static_physical_operator_lane(label, problem="spin_boson")["physical_operator_lane"]
    for label in labels
)
print(counts)
assert counts["other"] == 0
PY
```

Bose-Hubbard full-meta:

```bash
python - <<'PY'
from collections import Counter
from src.quantum.operator_pools.boson_chains import build_bose_hubbard_full_meta_terms
from pipelines.contracts.static_provenance import classify_static_physical_operator_lane

labels = [
    f"full_meta::{label}"
    for label, _ in build_bose_hubbard_full_meta_terms(
        num_sites=2,
        t=1.0,
        u=0.5,
        dv=0.1,
        omega0=1.0,
        n_ph_max=2,
        boson_encoding="binary",
        boundary="open",
    )
]

counts = Counter(
    classify_static_physical_operator_lane(label, problem="bose_hubbard")["physical_operator_lane"]
    for label in labels
)
print(counts)
assert counts["other"] == 0
PY
```

---

## 6. Run-configuration notes

For the requested “1.75-less-aggressive” route, keep parser behavior unchanged and use:

```text
--static-lane-route physical_operator_type
--physical-lane-shortlist-aggressiveness 3
--phase1-shortlist-size 42
--phase2-shortlist-size 21
--phase2-shortlist-fraction 0.4375
```

This yields effective:

```text
Phase I cap: ceil(42 / 3) = 14
Phase II cap: ceil(21 / 3) = 7
Phase II fraction: 0.4375 / 3 = 0.145833...
```

Full-reoptimization overlay:

```text
--adapt-reopt-policy full
--adapt-full-refit-every 0
--adapt-final-full-refit false
--adapt-max-depth 30
```

Use no batching, matching the existing HH workflow flag names already used in the current run script.

Recommended pools:

| Problem | Recommended pool |
|---|---|
| `hubbard` | `--adapt-pool uccsd` |
| `spin_boson` / Rabi family | `--adapt-pool full_meta` |
| `bose_hubbard` | `--adapt-pool full_meta` |

---

## 7. Concerns / compatibility

1. **Current code cannot run non-HH physical lanes** until the HH-only guard and classifier dispatch are changed.
2. **Hubbard with `--adapt-pool full_meta` is not recommended** for this minimal change. The selected context only guarantees clean singles/doubles labels for the UCCSD pool.
3. **Spin-boson and Bose-Hubbard full-meta pools look supported** in `primitive_pools.py`, because both have explicit `full_meta` construction branches.
4. **HH behavior can remain unchanged** by delegating `problem == "hh"` to the existing HH classifier and keeping the existing HH lane constants/schema.
5. **Oracle note:** in this text-only response I cannot actually query Oracle. The taxonomy above is derived directly from the existing builder labels and is the minimal implementation candidate to submit for Oracle semantic review before launching the non-HH local jobs.