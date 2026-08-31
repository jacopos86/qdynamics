# Hubbard–Holstein ADAPT→Replay Continuation Follow-on Implementation Specification

## Purpose

This document is the **post-`phase3_v1` follow-on specification**.
It assumes the repository already contains the opt-in `phase3_v1` landing described in the implementation summary:

- generator metadata ownership exists;
- motif extraction/tiling exists in an initial form;
- shared symmetry metadata exists with **verify-only** mitigation hooks;
- overlap-guided rescue plumbing exists and is **simulator-only**;
- lifetime-cost weighting exists inside the shortlist/full-score architecture;
- legacy, `phase1_v1`, and `phase2_v1` remain explicit and stable.

The purpose of this document is to convert the **remaining deferred items** into a new, cohesive, implementation-ready plan for a coding agent.

This document is intentionally written as a **control/architecture spec** rather than as repo-specific patch instructions. It is meant to be used as a **series of agent prompts**. The agent should audit the current codebase before editing, bind the generic interfaces here to the actual repository surfaces, and stop on any policy/spec/code mismatch.

---

## What This Document Covers

This follow-on specification covers four workstreams:

1. **Regression closure and invariants** after `phase3_v1`.
2. **Live selective macro-splitting**, including runtime split-event production.
3. **Active symmetry mitigation beyond verify-only**, using a backend-safe first implementation.
4. **Richer motif transfer**, especially boundary-mode handling and multi-source motif growth.

This document does **not** reopen the already-completed `phase3_v1` items except where the new work depends on them.

---

## What This Document Does Not Cover

The following are deliberately **out of scope** unless a later design document explicitly reopens them:

1. rewriting the operator algebra core;
2. replacing the current statevector-centric production path with a different quantum runtime stack;
3. introducing a hardware-path rescue fallback;
4. turning every macro-generator into a permanently split pool representation;
5. forcing active symmetry mitigation onto backends that cannot safely support it.

In particular:

- **rescue mode remains simulator-only by default**;
- **macro-generators remain the default primitive**;
- **hard symmetry preservation in ansatz construction remains primary**;
- mitigation is an **additional estimator/post-processing layer**, not a substitute for symmetry-preserving ansatz design.

---

## Executive Summary

The remaining work should be implemented in the following order:

### Step 0 — Close the verification gap
Run the omitted transitive tests, confirm backward compatibility of newly added metadata, and freeze the `phase3_v1` baseline.

### Step 1 — Add runtime selective macro-splitting
Use macro-generators by default, but allow **shortlist-only split probing** when a parent macro-generator appears too coarse, too expensive, or too geometrically misaligned.

### Step 2 — Add backend-safe active symmetry mitigation
The first implementation should be **simpler than the full research vision**:

- keep `verify_only` as-is;
- add **bitstring postselection** where the symmetry is directly checkable from sampled outcomes;
- add **projector-renormalized estimators** only where the measured symmetry operators and the observable groupings make this safe.

This is enough to move beyond metadata-only mitigation while keeping the architecture open for later symmetry-expansion estimators.

### Step 3 — Strengthen motif transfer
The first motif transfer extension should focus on:

- explicit boundary modes;
- canonical motif embedding rules;
- multi-source motif provenance;
- safe lift/tile seeding for larger systems.

### Step 4 — Revalidate the combined path
After Steps 1–3 are landed, rerun integration tests, confirm mode gating, and verify that all new logic remains opt-in.

---

## Baseline Assumptions

The coding agent should assume the following are already true:

1. **Scaffolded replay** is the default continuation pattern.
2. **Freeze-then-unfreeze replay** already exists.
3. **Windowed refits** and periodic full-prefix refits already exist.
4. **Shortlist/full-score architecture** already exists.
5. **Generator metadata ownership** already exists.
6. **Verify-only symmetry hooks** already exist.
7. **Motif extraction/tiling** exists, but with limited boundary richness.
8. **Overlap-guided rescue plumbing** exists and is simulator-only.
9. **Lifetime-cost weighting** exists inside the score stack.

The new work must extend this baseline without destabilizing prior modes.

---

## Repository Guardrails Carried Forward

The following guardrails remain non-negotiable.

### Core mathematical/representation guardrails
- Internal Pauli conventions remain unchanged.
- The canonical JW construction remains the source of truth.
- New behavior should wrap existing algebra/execution surfaces rather than rewrite them.

### Replay/path guardrails
- Replay remains provenance-aware.
- Prepared-state vs reference-state semantics remain explicit.
- Residual blocks remain zero-initialized unless a future spec explicitly changes this.

### Continuation guardrails
- The score must evaluate the **same continuation protocol the optimizer will actually use**.
- Window-refit-aware logic must use the actual unlocked window, not a hypothetical one.
- Lifetime burden remains **additive effective cost**, not a product of unrelated penalties.

### Mode guardrails
- `legacy`, `phase1_v1`, `phase2_v1`, and `phase3_v1` must remain explicit.
- Any new mode must be opt-in and must not silently perturb older modes.

### Safety/architecture guardrails
- Do not add hardware rescue fallback under the guise of “completing” rescue mode.
- Do not claim symmetry mitigation bias reduction on paths that only verify sector consistency.
- Do not expand the full pool by eagerly splitting all macros.

---

## Implementation Modes Suggested by This Document

This document assumes the codebase may add one or more new opt-in modes/flags, for example:

- `phase3_v2_split`
- `phase3_v2_symm_simple`
- `phase3_v2_motif`
- `phase3_v2_full`

The exact naming is a repo decision. The important property is:

> each new behavior must be individually gateable and composable, so it can be tested in isolation and then combined.

---

## Workstream 0 — Regression Closure and Baseline Freeze

### Objective
Before adding new functionality, close the verification gap left by the `phase3_v1` pass.

### Why this matters
The terminal summary reported that the following tests were **not** run during the `phase3_v1` pass:

- `test_vqe_hh_integration.py`
- `test_spsa_optimizer.py`
- `test_vqe_minimize_spsa_smoke.py`

Even when those files were not directly edited, transitive effects can still occur because replay, handoff, scoring, and HH orchestration surfaces were changed.

### Required actions
1. Run the omitted tests against the current `phase3_v1` branch.
2. Confirm backward compatibility for serialized handoff/export artifacts that predate the new metadata.
3. Confirm that turning off `phase3_v1` still reproduces prior behavior.
4. Confirm that `phase3_v1` fields deserialize with safe defaults when absent.

### Acceptance criteria
- all omitted tests pass;
- backward-compatible deserialization is confirmed;
- no mode gating regressions appear;
- no metadata field becomes mandatory on older artifacts.

### Likely touchpoints to audit
- orchestration entry points;
- handoff/serialization bundle types;
- replay controller;
- replay scoring entry point;
- integration tests around HH replay.

### Agent prompt

```text
Audit the current post-phase3_v1 branch without changing behavior first.

Goals:
1. Run the omitted regression tests:
   - test/test_vqe_hh_integration.py
   - test/test_spsa_optimizer.py
   - test/test_vqe_minimize_spsa_smoke.py
2. Audit backward compatibility of handoff/export artifacts after the added phase3_v1 metadata.
3. Confirm that legacy, phase1_v1, phase2_v1, and phase3_v1 remain explicitly gated and stable.
4. If any field added in phase3_v1 became mandatory during load/roundtrip for older artifacts, fix that with backward-safe defaults.

Rules:
- Do not add new functionality in this pass.
- Do not rewrite core algebra or replay semantics.
- Stop and report any spec/code/policy mismatch before proceeding.

Deliverables:
- concise audit summary;
- exact failing tests if any;
- minimal compatibility patch if needed;
- no behavioral changes beyond backward-safe compatibility fixes.
```

---

## Workstream 1 — Live Selective Macro-Splitting

## Objective
Introduce a **runtime selective-splitting engine** that can emit live split events and decide, on a shortlist-only basis, whether a macro-generator should remain intact or be replaced by a small admissible child set.

This is not a license to atomize the pool. The design principle remains:

> keep macro-generators by default; split only when splitting materially improves predicted useful energy drop per lifetime cost while preserving required symmetries.

### Why this is the right follow-on
`phase3_v1` already made the schema/provenance split-aware, but did not implement live split-event production. That means the repository can *record* splits but cannot yet *decide* them dynamically during candidate selection.

The missing piece is a runtime decision layer.

---

## Mathematical Model for Selective Splitting

Suppose a macro-generator is represented by

\[
A^{(M)} = \sum_{r=1}^{R} \nu_r A_r,
\]

where the child generators \(A_r\) are the finest admissible pieces exposed by the existing split-policy surfaces.

The parent macro-generator has a position-aware score

\[
S_M(p),
\]

and each child has its own score

\[
S_r(p_r).
\]

A split candidate set \(B \subseteq \{1,\dots,R\}\) should be evaluated by a split objective of the form

\[
S_{\mathrm{split}}(B)
=
\sum_{r\in B} S_r
- \sum_{r<s\in B} \Pi_{rs}
- \Lambda_{\mathrm{split}}(B),
\]

where:

- \(\Pi_{rs}\) is a compatibility penalty for pairwise overlap, non-layerability, or strong cross-curvature;
- \(\Lambda_{\mathrm{split}}(B)\) is an explicit split-complexity penalty, which may account for:
  - extra optimizer dimension,
  - additional compiled burden,
  - extra new measurement groups,
  - provenance churn.

A split should be accepted only if

\[
S_{\mathrm{split}}(B^\star)
\ge
(1+\delta_{\mathrm{split}})
S_M,
\]

with a positive margin \(\delta_{\mathrm{split}}>0\), and only when all hard admissibility checks pass.

### Hard admissibility checks
A split may only be considered when all of the following hold:

1. **hard symmetry preservation**: every child in the admissible set preserves the required conserved quantities, or the split policy explicitly marks the children as sector-safe;
2. **support admissibility**: children remain valid on the current lattice/system embedding;
3. **template realizability**: children can be instantiated by the existing execution templates;
4. **bounded search width**: only a small number of child subsets may be explored;
5. **position admissibility**: insertion positions used for split scoring are from the same allowed-position policy already used in the continuation stack.

### Strong recommendation for the first runtime version
The first runtime split engine should be **strictly narrower than the fully general problem**:

- probe splits only on **shortlisted macro candidates**;
- only use **existing child decompositions** already available from the split-policy surfaces;
- only explore subsets of size at most **2 or 3**;
- only consider children that either commute or are known to layer well after compilation;
- do not recurse: **macro → children** is enough for the first live engine.

This is the correct first production version. It captures the value of selective splitting without exploding pool size.

---

## Trigger Conditions for Split Probing

A shortlisted macro-generator should be probed for splitting only if one or more of the following is true:

1. **cost trigger**
   - the parent’s compiled burden increment is high relative to its score;
2. **geometry trigger**
   - the parent tangent appears to contain multiple distinct useful directions, e.g. one child has much higher novelty than the macro as a whole;
3. **layering trigger**
   - the children can be compiled into a shallower or more parallel block than the parent template;
4. **position trigger**
   - the best parent score occurs at a position suggesting only part of the parent’s support is actually useful;
5. **stability trigger**
   - the parent repeatedly reaches the shortlist but fails final admission because its score is diluted by mixed useful/useless pieces.

A simple trigger rule is:

\[
\text{probe-split}(M)=1
\iff
\Bigl(
\frac{S_M}{K_M}<\tau_{\mathrm{macro}}
\Bigr)
\;
\text{or}
\;
\Bigl(
\max_r S_r^{\mathrm{proxy}} > (1+\kappa_{\mathrm{probe}}) S_M^{\mathrm{proxy}}
\Bigr),
\]

where the child proxies are cheap screening scores, not full final scores.

---

## Runtime Decision Protocol

### Stage 1 — Macro-first screening
1. score the active pool using the current screening path;
2. shortlist the top macro candidates;
3. do **not** split the full pool.

### Stage 2 — Split probe on shortlisted parents only
For each shortlisted parent macro-generator:

1. obtain child decomposition from existing split-policy surfaces;
2. reject immediately if hard symmetry or execution-template checks fail;
3. evaluate child screening scores at allowed positions;
4. enumerate a tiny admissible child set family \(B\) of size 1–3;
5. compute \(S_{\mathrm{split}}(B)\);
6. accept either the parent or the best child set, whichever wins by the split margin.

### Stage 3 — Emit a split event
Whenever the engine decides between parent and children, emit a structured event that records:

- parent id;
- child ids;
- reason for probing;
- reason for final choice;
- parent score;
- child scores;
- selected subset;
- compile-cost comparison;
- symmetry checks;
- insertion positions used.

This event should exist even if the parent wins. The split engine is an observable subsystem, not a silent heuristic.

---

## Data/Schema Additions for Selective Splitting

The exact field names are a repo decision, but the runtime schema should support the following concepts.

### Generator metadata additions
- `is_macro_generator`
- `parent_generator_id`
- `child_generator_ids`
- `split_family_id`
- `split_depth`
- `split_policy_tag`
- `supports_runtime_split`

### Split probe record
- `probe_trigger`
- `parent_screen_score`
- `child_proxy_scores`
- `admissible_child_subsets`
- `chosen_representation` (`parent` or `child_set`)
- `chosen_child_ids`
- `split_margin`
- `symmetry_gate_results`
- `compiled_cost_parent`
- `compiled_cost_children`

### Replay/handoff persistence
- selected representation at each admitted step;
- split provenance graph;
- ability to reconstruct the executable scaffold in the chosen representation.

---

## Pseudocode for Selective Splitting

```python
class SplitProbeResult:
    parent_id: str
    child_ids: list[str]
    parent_score: float
    child_scores: dict[str, float]
    best_child_subset: tuple[str, ...]
    split_objective_best: float
    chosen_mode: str   # 'parent' or 'children'
    trigger: str
    symmetry_ok: bool
    compile_parent: float
    compile_children: float


def should_probe_split(parent_feat, cfg):
    if not parent_feat.supports_runtime_split:
        return False
    if parent_feat.parent_cost_over_score < cfg.split_probe_cost_ratio_floor:
        return True
    if parent_feat.best_child_proxy > (1.0 + cfg.split_probe_margin) * parent_feat.parent_proxy:
        return True
    if parent_feat.shortlist_recurrence_count >= cfg.split_probe_patience:
        return True
    return False


def enumerate_child_subsets(children, cfg):
    # first version: bounded width, no recursion
    valid = []
    for subset in all_subsets_up_to_k(children, k=cfg.max_split_subset_size):
        if subset_is_pairwise_compatible(subset, cfg):
            valid.append(subset)
    return valid


def split_objective(child_features, subset, cfg):
    reward = sum(child_features[c].score for c in subset)
    pair_pen = 0.0
    for a, b in all_pairs(subset):
        pair_pen += compatibility_penalty(child_features[a], child_features[b], cfg)
    split_pen = split_complexity_penalty(subset, child_features, cfg)
    return reward - pair_pen - split_pen


def probe_selective_split(parent, position_policy, cfg):
    if not should_probe_split(parent, cfg):
        return None

    children = materialize_children_from_existing_split_policy(parent)
    if not children:
        return None

    if not all(child_preserves_required_symmetry(c, cfg) for c in children):
        return SplitProbeResult(chosen_mode='parent', symmetry_ok=False, ...)

    child_features = {}
    for child in children:
        best = None
        for p in allowed_positions(child, position_policy):
            feat = evaluate_candidate_features(child, p, cfg)
            if best is None or feat.score > best.score:
                best = feat
        if best is not None:
            child_features[child.id] = best

    subsets = enumerate_child_subsets(list(child_features.keys()), cfg)
    if not subsets:
        return SplitProbeResult(chosen_mode='parent', symmetry_ok=True, ...)

    best_subset = None
    best_value = float('-inf')
    for subset in subsets:
        val = split_objective(child_features, subset, cfg)
        if val > best_value:
            best_value = val
            best_subset = subset

    parent_value = parent.full_score
    choose_children = best_value >= (1.0 + cfg.split_accept_margin) * parent_value

    return SplitProbeResult(
        parent_id=parent.id,
        child_ids=list(child_features.keys()),
        parent_score=parent_value,
        child_scores={k: v.score for k, v in child_features.items()},
        best_child_subset=tuple(best_subset) if best_subset else tuple(),
        split_objective_best=best_value,
        chosen_mode='children' if choose_children else 'parent',
        trigger=parent.split_probe_trigger,
        symmetry_ok=True,
        compile_parent=parent.compiled_burden,
        compile_children=sum(child_features[c].compiled_burden for c in (best_subset or [])),
    )
```

---

## Testing Requirements for Selective Splitting

### Unit tests
1. a macro with no valid children never probes;
2. symmetry-violating children are rejected;
3. parent wins when child gains do not clear the split margin;
4. children win when split objective is materially larger;
5. split events are emitted for both parent-win and child-win cases;
6. replay/handoff roundtrip preserves parent/child provenance.

### Integration tests
1. older modes remain unchanged when split probing is disabled;
2. the active pool does not blow up in size due to eager splitting;
3. a known macro-child example chooses the child set under opt-in split mode;
4. pruning and replay can consume a scaffold containing admitted child sets.

### Anti-patterns
- do not recursively split children in the first version;
- do not score every child of every macro across the full pool;
- do not let child selection bypass the usual symmetry/cost gates;
- do not let split provenance disappear in export/handoff.

### Agent prompt

```text
Implement live selective macro-splitting as a shortlist-only runtime extension.

Baseline assumptions:
- phase3_v1 already supports generator metadata ownership and schema-level split provenance;
- existing split-policy surfaces already define parent/child decompositions;
- macro-generators remain the default primitive.

What to implement:
1. Add a runtime split probe that activates only on shortlisted macro candidates.
2. Use existing child decompositions only; do not introduce recursive splitting.
3. Allow child subsets of size at most 2 or 3, with compatibility penalties.
4. Emit structured split events whether the parent or the child set wins.
5. Preserve backward compatibility and make the whole feature opt-in.

Scoring rule:
- compare the parent score against a child-set split objective
  sum(child scores) - pair penalties - split complexity penalty
- accept children only if they beat the parent by a positive margin.

Hard rules:
- preserve hard symmetry constraints;
- do not split the full pool eagerly;
- do not rewrite core operator algebra;
- do not regress older modes.

Deliverables:
- implementation;
- focused unit tests;
- at least one integration test showing parent-win and one showing child-win;
- concise summary of actual files touched and any spec/code mismatches.
```

---

## Workstream 2 — Active Symmetry Mitigation Beyond Verify-Only

## Objective
Extend the existing symmetry subsystem beyond metadata and verification so that the repository can support **real, opt-in mitigation behavior** on backends where this is safe.

The first implementation should be **deliberately simpler** than the full research picture.

### Recommended first mitigation tier
Implement two new mitigation modes:

1. **bitstring postselection** for symmetries that are directly computable from sampled outcomes in the measurement basis;
2. **projector-renormalized estimation** for observable/symmetry groupings where the required quantities can be estimated safely from grouped measurements.

Keep more aggressive symmetry-expansion estimators as a later extension.

This is the correct first production implementation because it turns symmetry metadata into actual estimator behavior while avoiding unsupported backend assumptions.

---

## Mathematical Model for Symmetry Mitigation

Let \(\{S_a\}\) be mutually commuting symmetry operators with target eigenvalues \(s_a^\star\).
Assume the target sector projector is

\[
\Pi_\star = \prod_a \Pi_a,
\qquad
\Pi_a = \frac{I + s_a^\star S_a}{2},
\]

for involutory symmetries \(S_a^2=I\). The same design principle extends to more general sector projectors, but the first backend-safe implementation should preferentially target the involutory/diagonal case.

### Verification quantities
Even without active mitigation, the following are useful diagnostics:

\[
v_a = \langle S_a \rangle,
\qquad
p_\star = \langle \Pi_\star \rangle.
\]

Here:

- \(v_a\) measures sector consistency for a single symmetry;
- \(p_\star\) is the probability weight in the target sector.

### Postselected estimator
When sampled measurement outcomes \(x\) admit a direct sector test,

\[
\chi_\star(x)=
\begin{cases}
1,& x\in \mathcal X_\star,\\
0,& \text{otherwise},
\end{cases}
\]

we can define the postselected estimator for a diagonal observable readout \(O(x)\):

\[
\widehat{\langle O \rangle}_{\mathrm{post}}
=
\frac{\sum_x \chi_\star(x) w_x O(x)}{\sum_x \chi_\star(x) w_x},
\]

where \(w_x\) is the shot count or sample weight.

This reduces symmetry-violating contamination at the cost of increased variance when \(p_\star\) is small.

### Projector-renormalized estimator
For compatible grouped-measurement settings, define

\[
\langle O \rangle_{\Pi}
:=
\frac{\langle \Pi_\star O \Pi_\star \rangle}{\langle \Pi_\star \rangle}.
\]

When \([O,\Pi_\star]=0\) and the measurement strategy supports the needed correlators, this may simplify to

\[
\langle O \rangle_{\Pi}
=
\frac{\langle O \Pi_\star \rangle}{\langle \Pi_\star \rangle}.
\]

This estimator is more general than shot-level postselection, but it requires safe grouped measurement support and careful variance accounting.

---

## What the First Symmetry-Mitigation Version Should Actually Implement

The first active implementation should **not** attempt maximal generality.

### Implement now
1. keep `verify_only` exactly as-is;
2. add `postselect_diag_v1` for diagonal or bitstring-checkable symmetries;
3. add `projector_renorm_v1` only when:
   - the symmetry projector is expressible in the measured operator language,
   - the grouped-measurement plan supports the required moments,
   - the variance blow-up is tracked.

### Defer
- generic non-diagonal postselection;
- full symmetry-expansion estimator families;
- automatic backend-specific mitigation tuning;
- mitigation on backends that do not expose the required measurement information.

This phased approach is correct. It adds real mitigation without promising more than the runtime can safely deliver.

---

## Required Architecture for Symmetry Mitigation

### Separation of responsibilities
1. **ansatz construction layer**
   - preserves hard symmetries whenever required;
2. **verification layer**
   - measures sector diagnostics but does not modify estimators;
3. **mitigation layer**
   - optionally transforms estimator outputs using postselection or projector renormalization;
4. **telemetry layer**
   - reports retained-shot fraction, sector probability, mitigation mode, and variance effects.

### Important design rule
Mitigation must be implemented as an **estimator/post-processing service**, not embedded ad hoc into the ansatz-growth logic.

That keeps the continuation logic clean.

---

## Data/Schema Additions for Symmetry Mitigation

### Symmetry specification
- symmetry operator id(s);
- target sector label/value(s);
- whether the symmetry is hard-gated in ansatz construction;
- whether the symmetry is bitstring-checkable in the chosen measurement basis;
- whether projector-based mitigation is supported.

### Mitigation result payload
- mitigation mode used;
- sector diagnostics \(v_a\);
- target-sector probability estimate \(p_\star\);
- retained-shot fraction after postselection;
- mitigation-adjusted energy/observable estimate;
- variance estimate / confidence interval;
- whether the mitigation fell back to verification only.

### Failure/fallback semantics
If a requested mitigation mode cannot be safely realized on the active backend/measurement plan, the code must either:

1. raise a clear configuration error, or
2. explicitly fall back to `verify_only` and report that fallback.

Silent degradation is not acceptable.

---

## Pseudocode for Symmetry Mitigation

```python
class SymmetryMitigationResult:
    mode: str
    sector_values: dict[str, float]
    sector_probability: float | None
    retained_fraction: float | None
    observable_value: float
    observable_variance: float | None
    fell_back_to_verify_only: bool


def verify_sector(grouped_measurements, symmetry_spec):
    sector_values = {}
    for sym in symmetry_spec.symmetry_ops:
        sector_values[sym.id] = estimate_expectation(grouped_measurements, sym.operator)
    p_star = estimate_target_sector_probability(grouped_measurements, symmetry_spec)
    return sector_values, p_star


def postselect_diag(bitstrings, observable_decoder, symmetry_spec):
    kept = []
    for sample in bitstrings:
        if sample_in_target_sector(sample, symmetry_spec):
            kept.append(sample)
    if len(kept) == 0:
        raise MitigationFailure("No samples remained after symmetry postselection")
    value = weighted_average(observable_decoder(s) for s in kept)
    retained_fraction = len(kept) / max(len(bitstrings), 1)
    variance = weighted_variance(observable_decoder(s) for s in kept)
    return value, retained_fraction, variance


def projector_renorm(grouped_measurements, observable, symmetry_spec):
    p_star = estimate_target_sector_probability(grouped_measurements, symmetry_spec)
    if p_star <= 0:
        raise MitigationFailure("Target-sector probability is zero or unresolved")
    numerator = estimate_projected_observable(grouped_measurements, observable, symmetry_spec)
    value = numerator / p_star
    variance = estimate_ratio_variance(numerator, p_star, grouped_measurements)
    return value, p_star, variance


def apply_symmetry_mitigation(raw_payload, observable, symmetry_cfg):
    sector_values, p_star = verify_sector(raw_payload.grouped_measurements, symmetry_cfg.spec)

    if symmetry_cfg.mode == 'verify_only':
        return SymmetryMitigationResult(
            mode='verify_only',
            sector_values=sector_values,
            sector_probability=p_star,
            retained_fraction=None,
            observable_value=raw_payload.observable_value,
            observable_variance=raw_payload.observable_variance,
            fell_back_to_verify_only=False,
        )

    if symmetry_cfg.mode == 'postselect_diag_v1':
        if not symmetry_cfg.spec.supports_bitstring_postselection:
            return fallback_verify_only(...)
        value, retained, variance = postselect_diag(
            raw_payload.bitstrings,
            raw_payload.observable_decoder,
            symmetry_cfg.spec,
        )
        return SymmetryMitigationResult(
            mode='postselect_diag_v1',
            sector_values=sector_values,
            sector_probability=p_star,
            retained_fraction=retained,
            observable_value=value,
            observable_variance=variance,
            fell_back_to_verify_only=False,
        )

    if symmetry_cfg.mode == 'projector_renorm_v1':
        if not symmetry_cfg.spec.supports_projector_renorm:
            return fallback_verify_only(...)
        value, p_star, variance = projector_renorm(
            raw_payload.grouped_measurements,
            observable,
            symmetry_cfg.spec,
        )
        return SymmetryMitigationResult(
            mode='projector_renorm_v1',
            sector_values=sector_values,
            sector_probability=p_star,
            retained_fraction=None,
            observable_value=value,
            observable_variance=variance,
            fell_back_to_verify_only=False,
        )

    raise ValueError(f"Unknown symmetry mitigation mode: {symmetry_cfg.mode}")
```

---

## Testing Requirements for Symmetry Mitigation

### Unit tests
1. `verify_only` path remains unchanged;
2. diagonal bitstring postselection keeps only target-sector shots;
3. projector renormalization matches a known analytic/simulator case;
4. unsupported mitigation modes fall back or error explicitly;
5. mitigation telemetry reports retained fraction and sector probability;
6. zero-retained-shot cases are handled explicitly.

### Integration tests
1. opt-in mitigation changes estimator behavior only when requested;
2. hard symmetry-gated ansatz construction remains unchanged;
3. replay and ADAPT scoring are not silently altered by mitigation metadata;
4. mitigation can consume cached measurement data where appropriate.

### Anti-patterns
- do not silently treat verification as mitigation;
- do not postselect on symmetries that are not actually checkable in the sampled basis;
- do not ignore variance blow-up after strong postselection;
- do not let mitigation override hard symmetry admission gates.

### Agent prompt

```text
Implement the first active symmetry-mitigation tier beyond verify-only.

Baseline assumptions:
- shared symmetry metadata already exists;
- verify-only hooks already exist;
- hard symmetry preservation in ansatz construction remains primary.

Implement now:
1. keep verify_only unchanged;
2. add postselect_diag_v1 for bitstring-checkable symmetries;
3. add projector_renorm_v1 only where the measurement/grouping plan safely supports it;
4. add telemetry for sector values, target-sector probability, retained-shot fraction, and mitigation fallback.

Hard rules:
- mitigation is opt-in;
- if a requested mitigation mode is unsupported on the active backend, fall back explicitly or raise clearly;
- do not claim mitigation benefit on verify-only paths;
- do not rewrite ansatz construction or operator algebra.

Deliverables:
- implementation;
- unit tests for postselection and projector renormalization;
- integration test confirming no behavior change when mitigation is disabled;
- concise summary of actual backends/measurement paths supported by the first version.
```

---

## Workstream 3 — Richer Motif Transfer and Boundary Modes

## Objective
Upgrade the current motif subsystem so that motifs can be transferred more reliably across lattice sizes, source families, and boundary conditions.

The current `phase3_v1` motif support is a good first step, but the follow-on work must make motif transfer a stronger scaling tool.

---

## Mathematical/Structural Model for Motifs

A motif should be treated as a transferable local continuation primitive.

Represent a motif family by a tuple

\[
\mathcal M = (\mathcal G, \Delta, \tau, \beta, \Sigma, \mathcal P),
\]

where:

- \(\mathcal G\): ordered local generator family/template ids;
- \(\Delta\): relative support offsets;
- \(\tau\): boundary mode;
- \(\beta\): parameter template/statistics;
- \(\Sigma\): symmetry metadata;
- \(\mathcal P\): provenance over source systems/runs.

### Boundary mode
The boundary mode \(\tau\) determines where the motif may be legally embedded.
Examples:

- `interior_only`
- `left_edge`
- `right_edge`
- `both_edges`
- `periodic_wrap`
- `interface_only`
- `forbid_truncation`
- `allow_safe_truncation`

### Embedding rule
A motif embedding into a target system of size \(L_t\) is valid only if:

1. the translated support is legal under the target geometry;
2. the boundary mode permits the placement;
3. the embedded generators preserve required symmetries;
4. the chosen generator templates are executable on the target compiled scaffold;
5. duplicated or overlapping embeddings are resolved deterministically.

---

## What the First Motif-Transfer Extension Should Implement

The first extension should again be **simpler than the fully general problem**.

### Implement now
1. explicit boundary modes in the motif schema;
2. canonical embedding checks for OBC/PBC-style boundaries actually supported by the repo;
3. multi-source provenance for a motif family;
4. deterministic deduplication when multiple motifs map to the same target support;
5. parameter initialization rules based on robust source statistics;
6. safe pool/scaffold seeding from tiled motifs.

### Defer
- learned nonlinear size-scaling laws for parameters;
- aggressive cross-family motif fusion;
- automatic motif discovery from heterogeneous objectives;
- nonlocal or graph-irregular motif transport unless the repo already supports such geometry.

This is the correct first strengthening pass.

---

## Parameter Initialization for Tiled Motifs

The target of motif transfer is not perfect parameter prediction. It is **search entropy reduction**.

A good first initialization rule is:

\[
\theta^{(t)}_{j,\mathrm{init}}
=
\mathrm{clip}\Bigl(
\operatorname{median}_{s\in \mathcal P_j}(\theta^{(s)}_j),
-\rho_j,
\rho_j
\Bigr),
\]

with optional sign-consistency filtering. Here:

- \(\mathcal P_j\) is the set of source occurrences for generator slot \(j\);
- \(\rho_j\) is a trust cap from recent admitted magnitudes or stage-specific initialization limits.

A slightly richer but still safe variant is:

\[
\theta^{(t)}_{j,\mathrm{init}}
=
\mathrm{clip}(a_j + b_j \theta^{(s)}_j, -\rho_j, \rho_j),
\]

with \((a_j,b_j)\) fitted only if enough source examples exist.

The first implementation may stay with the median-and-clip rule. That is fully acceptable and more robust.

---

## Required Motif Control Logic

### Extraction
From solved source systems, extract motifs with:

- canonical local support representation;
- ordered generator family ids;
- relative offsets;
- boundary classification;
- parameter statistics;
- source system metadata;
- associated symmetry metadata.

### Canonicalization
Two motif instances should merge into one family if they are related by allowed translations/reflections and have compatible boundary classification.

### Tiling
When seeding a larger system:

1. generate admissible embeddings for each motif family;
2. reject embeddings that violate boundary rules;
3. reject embeddings that violate symmetry or support rules;
4. resolve collisions deterministically;
5. emit seeded pool/scaffold candidates with provenance links back to the source motifs.

### Collision resolution
When two embeddings compete for the same support or nearly the same continuation role, pick by a deterministic priority order, for example:

1. exact boundary match over relaxed match;
2. more source support count;
3. lower compiled cost estimate;
4. higher historical usefulness score.

---

## Data/Schema Additions for Motifs

### Motif family payload
- motif family id;
- ordered generator/template ids;
- support offsets;
- allowed boundary modes;
- source system sizes;
- source family tags;
- parameter summary statistics;
- symmetry metadata;
- canonical orientation/reflection tag.

### Embedding payload
- target system id/size;
- embedding translation/reflection;
- boundary classification in target;
- collision-resolution outcome;
- seeded parameter vector;
- seed mode (`pool`, `scaffold`, `replay_family`).

### Provenance requirements
The seeded target object must retain provenance back to:

- motif family id;
- source runs;
- source generator families.

---

## Pseudocode for Motif Transfer

```python
class MotifFamily:
    id: str
    generator_templates: list[str]
    relative_offsets: list[tuple[int, ...]]
    boundary_modes: set[str]
    parameter_stats: dict[str, float]
    symmetry_spec: dict
    source_runs: list[str]
    source_sizes: list[int]


class MotifEmbedding:
    motif_id: str
    target_size: int
    translation: tuple[int, ...]
    reflection: str | None
    boundary_mode: str
    seeded_params: list[float]
    support_sites: list[int]


def classify_boundary_mode(local_support, system_geometry):
    # repo-specific implementation
    ...


def extract_motif_family(source_scaffold, cfg):
    motifs = []
    for block in detect_local_repeating_blocks(source_scaffold, cfg):
        motifs.append(
            MotifFamily(
                id=make_motif_id(block),
                generator_templates=block.generator_templates,
                relative_offsets=canonicalize_offsets(block.supports),
                boundary_modes={classify_boundary_mode(block.supports, block.geometry)},
                parameter_stats=robust_parameter_stats(block.parameters),
                symmetry_spec=block.symmetry_spec,
                source_runs=[block.run_id],
                source_sizes=[block.system_size],
            )
        )
    return merge_equivalent_motifs(motifs, cfg)


def admissible_embeddings(motif, target_geometry, cfg):
    out = []
    for placement in all_allowed_translations_and_reflections(motif, target_geometry, cfg):
        boundary = classify_embedding_boundary(motif, placement, target_geometry)
        if boundary not in motif.boundary_modes and not relaxed_boundary_match(boundary, motif, cfg):
            continue
        if not embedding_preserves_required_symmetry(motif, placement, cfg):
            continue
        out.append((placement, boundary))
    return out


def initialize_motif_params(motif, cfg):
    params = []
    for slot in motif.generator_templates:
        med = motif.parameter_stats[slot]['median']
        rho = cfg.motif_param_trust_cap.get(slot, cfg.default_param_cap)
        params.append(float(np.clip(med, -rho, rho)))
    return params


def tile_motifs_to_target(motif_families, target_geometry, cfg):
    candidates = []
    for motif in motif_families:
        for placement, boundary in admissible_embeddings(motif, target_geometry, cfg):
            emb = MotifEmbedding(
                motif_id=motif.id,
                target_size=target_geometry.size,
                translation=placement.translation,
                reflection=placement.reflection,
                boundary_mode=boundary,
                seeded_params=initialize_motif_params(motif, cfg),
                support_sites=placement.support_sites,
            )
            candidates.append(emb)
    return deterministic_collision_resolve(candidates, cfg)
```

---

## Testing Requirements for Motif Transfer

### Unit tests
1. motif families canonicalize correctly under allowed translations;
2. boundary classification is stable for supported geometries;
3. invalid boundary embeddings are rejected;
4. deterministic collision resolution is stable;
5. seeded parameters respect trust caps;
6. motif provenance survives serialization and replay handoff.

### Integration tests
1. seeding a larger target from a smaller source produces admissible pool/scaffold candidates;
2. turning motif seeding off restores baseline behavior;
3. multiple source runs merge into one motif family when appropriate;
4. motifs do not inject symmetry-violating generator instances.

### Anti-patterns
- do not hardcode one source size as globally canonical;
- do not seed motifs without checking target boundary legality;
- do not infer parameter scaling laws from insufficient data;
- do not allow motif seeding to bypass normal admission/symmetry gates.

### Agent prompt

```text
Implement the next motif-transfer extension focused on boundary handling and multi-source transfer.

Baseline assumptions:
- phase3_v1 already has motif extraction/tiling in an initial form;
- generator metadata ownership and symmetry metadata already exist.

Implement now:
1. explicit motif boundary modes;
2. canonical embedding legality checks for the repo’s supported geometries/boundaries;
3. multi-source motif provenance and family merging;
4. deterministic collision resolution;
5. robust parameter initialization using source statistics with trust caps.

Rules:
- keep the feature opt-in;
- do not let motif seeding bypass normal symmetry or admission gates;
- do not introduce a speculative global scaling-law fit unless enough source data exists;
- preserve provenance into handoff/replay artifacts.

Deliverables:
- implementation;
- unit tests for boundary legality and collision resolution;
- at least one integration test showing larger-system seeding from smaller-system motifs;
- concise summary of actual geometries/boundaries supported in the first version.
```

---

## Workstream 4 — Combined Integration, Telemetry, and Acceptance

## Objective
After the three feature workstreams are implemented, validate the combined system as a coherent extension rather than as a pile of independent patches.

### Required combined checks
1. all new features remain opt-in;
2. `legacy`, `phase1_v1`, `phase2_v1`, and `phase3_v1` behavior remain stable when the new features are off;
3. selective splitting, symmetry mitigation, and motif transfer can coexist without corrupting provenance;
4. exports/handoffs roundtrip all new metadata and event histories;
5. replay/scaffold reconstruction remains deterministic.

### Telemetry that should exist by the end
- split events and outcomes;
- mitigation mode and fallback state;
- sector diagnostics and retained fractions;
- motif family ids and target embeddings used;
- explicit mode flags for every run.

### Acceptance criteria for the full follow-on pass
The follow-on pass is complete when:

1. all omitted regression tests from Workstream 0 pass;
2. new feature tests pass;
3. all new behaviors are opt-in and backward-compatible;
4. split provenance, symmetry mitigation payloads, and motif provenance survive export/handoff/replay;
5. no unsupported backend path silently claims active mitigation.

### Agent prompt

```text
Perform the final combined integration pass after the new split/symmetry/motif work lands.

Goals:
1. Verify that all new behaviors remain opt-in.
2. Re-run the earlier omitted regressions plus the new feature tests.
3. Confirm that provenance survives export/handoff/replay for:
   - runtime split events,
   - symmetry mitigation payloads,
   - motif family/embedding metadata.
4. Confirm deterministic scaffold reconstruction.
5. Report any places where combined mode interactions required schema changes.

Rules:
- do not broaden default behavior;
- do not silently fallback on unsupported mitigation modes;
- stop and report if any combined interaction violates earlier mode stability.

Deliverables:
- combined test result summary;
- final file-touch summary;
- list of any remaining deliberate deferrals.
```

---

## Suggested Prompt Order for Your Agents

Use the workstreams in this order.

### Prompt 1
Regression closure and backward compatibility audit.

### Prompt 2
Selective macro-splitting runtime engine.

### Prompt 3
Backend-safe active symmetry mitigation (`postselect_diag_v1`, `projector_renorm_v1`).

### Prompt 4
Motif boundary modes and multi-source transfer.

### Prompt 5
Combined integration, telemetry, and acceptance pass.

This ordering is important:

- Prompt 1 freezes the baseline.
- Prompt 2 adds the most structurally invasive runtime decision logic.
- Prompt 3 extends the estimator layer without perturbing ansatz construction.
- Prompt 4 strengthens scaling/seeding after provenance and symmetry plumbing are stable.
- Prompt 5 validates the combined system.

---

## Likely Repo Touchpoints to Audit Before Editing

The exact locations must be confirmed by the repo-aware agent, but the terminal summary suggests the following surfaces are likely relevant:

- `adapt_pipeline.py`
- `hubbard_pipeline.py`
- `hh_vqe_from_adapt_family.py`
- `handoff_state_bundle.py`
- `hh_continuation_types.py`
- `hh_continuation_scoring.py`
- `hh_continuation_replay.py`
- `hh_continuation_generators.py`
- `hh_continuation_motifs.py`
- `hh_continuation_symmetry.py`
- `hh_continuation_rescue.py`
- the existing ADAPT/replay/staged-export test files

The agent should treat that list as **likely touchpoints**, not as permission to edit without audit.

---

## Simplification Policy

This document explicitly allows simpler first implementations for the new follow-on work, provided the simplification is documented.

### Allowed simplifications
1. **selective splitting**
   - shortlist-only;
   - bounded child subset size;
   - no recursive split search.

2. **symmetry mitigation**
   - diagonal/bitstring-checkable postselection first;
   - projector renormalization only on supported measurement paths;
   - defer more general symmetry expansion.

3. **motif transfer**
   - median-and-clip parameter initialization;
   - supported boundary families only;
   - no speculative nonlinear scaling models.

### Required discipline
Each simplification must be recorded as:

- intentionally limited first version;
- current capabilities;
- current exclusions;
- future extension points.

---

## Global Anti-Patterns to Avoid

1. Do not silently broaden default behavior.
2. Do not turn macro-generators into eagerly split pool atoms.
3. Do not implement mitigation where the measurement/backend path cannot support it safely.
4. Do not let motif seeding bypass ordinary scoring/admission/symmetry gates.
5. Do not lose provenance when parent/child representations change.
6. Do not couple replay-state reconstruction to transient runtime-only objects.
7. Do not claim hardware rescue support.

---

## Final Deliverable Standard

A completed follow-on pass should leave the repository with:

1. a stable `phase3_v1` baseline;
2. an opt-in live selective-splitting path;
3. an opt-in active symmetry-mitigation path beyond verification;
4. richer motif transfer with boundary-aware tiling;
5. complete provenance/telemetry for all three;
6. regression coverage showing older modes remain stable.

That is the correct next architectural step after the current `phase3_v1` landing.