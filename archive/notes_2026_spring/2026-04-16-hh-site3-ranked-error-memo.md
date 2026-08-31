# HH site3 ranked error memo

## Purpose
This memo is the tight working ranking of the most plausible errors behind the remaining HH \(L=2\) high-amplitude site3 controller failure after the guarded d-shape repairs and after the newer fidelity-first barrier branch. It is intentionally narrower than the full investigation report: the goal is to keep one short, decision-oriented list of what is most likely wrong, what is merely confounding interpretation, and what should be checked next.

## Executive summary
The strongest current split is now stable:

- **Best immediate local bug/seam:** a real **two-surface commit-law mismatch**.
- **Best global-cause seam:** the controller still appears to settle into the **wrong late stay/baseline basin**.
- **Strongest hidden confound:** `fidelity_first_barrier_v1` is **not** running on the same search surface as the guarded d-shape lane because it loses the preferred-site helper machinery.
- **Current Optuna study:** useful as a **feasibility alarm**, not yet trustworthy as a ranking surface.

That means the right mental model is not “late rescue family is gone, so proposal generation must be broken.” The current evidence instead says:
1. a candidate can be good enough under one guarded comparison surface and then be canceled under another;
2. even when early path dependence is removed, the late failing branch still reconverges into the wrong stay basin; and
3. the fidelity-first branch comparison is confounded because it is not just a score-law swap.

## Ranked error list

### 1. Two-surface commit-law mismatch (**confirmed local defect**)
**What it is**
The guarded below-floor / componentwise-aspiration path can admit a candidate using a guarded **core-only** comparison, but the final override path later compares that candidate to stay using full `_forecast_tracking_score(...)`. In fidelity-first mode, that full score adds the live barrier penalty back in.

**Why it matters**
This is the cleanest policy inconsistency now visible in the artifacts. It means the runtime can tentatively say “this append wins” and then effectively judge the same append under a different law at final commit.

**Observed evidence**
Checkpoint probe replay found explicit contradiction rows at checkpoints `11,15,31,35,67`:
- `admission_reason = below_floor_componentwise_aspiration`
- candidate admitted = yes
- final action = `stay`
- `decision_override_reason = exact_forecast_nonimproving_tracking_score`

Those are not abstract code-level worries; they show up in the live packet.

**Key seam references**
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:4236-4267`
  - componentwise aspiration uses guarded comparison logic and material-gain tests.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:3818-3898`
  - fidelity-first total score = fidelity-first core score + live d-barrier penalty.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:4430-4458`
  - final override compares full `_forecast_tracking_score(selected)` vs `_forecast_tracking_score(stay)`.
- `notes/2026-04-16-hh-site3-checkpoint-probe-summary.md`
  - contradiction table at checkpoints `11/15/31/35/67`.

**Current verdict**
Confirmed. This is the best immediate local seam.

---

### 2. Hidden fidelity-first search-surface fork (**confirmed confound, probably important**) 
**What it is**
`fidelity_first_barrier_v1` is not merely “same search, different score.” Preferred-site helper logic is gated on the d-shape barrier ranking path, so fidelity-first loses preferred-site baseline-scale extension, selected-candidate scale extension, repeated preferred-turn reopen, and preferred-site shortlist injection.

**Why it matters**
Any “v18 vs v19” interpretation is partially contaminated until this is acknowledged. A worse fidelity-first result does not cleanly imply the fidelity-first score is worse; some of the late recovery machinery is not even being offered on equal terms.

**Observed evidence**
Parallel audits converged that fidelity-first loses:
- preferred-site index activation,
- baseline-scale extension,
- selected-candidate scale extension,
- repeated preferred-turn reopen,
- preferred-site shortlist injection.

This is a search-surface fork, not just a scoring-surface fork.

**Key seam references**
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:7398-7417`
  - `_exact_v1_preferred_site_index_at_time(...)` returns `None` unless d-shape barrier ranking is active.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:5018-5057`
  - preferred-site baseline-step-scale extension only opens when preferred-site index exists.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:7231-7276`
  - candidate pool reopening logic depends on preferred-site state.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:7420-7491`
  - shortlist preferred-site injection path.

**Current verdict**
Confirmed. This is the biggest hidden confound in branch comparison.

---

### 3. Wrong late stay/baseline basin (**best current global cause**) 
**What it is**
The stay path selected by `_select_exact_v1_baseline_step_scale()` still looks like the main upstream failure basin. It searches proposal kind, blend, step, and gain using rollout score, and by the late window the failing branch appears structurally committed to a bad stay family.

**Why it matters**
This is the best explanation for why removing early `v19` appends does not recover the late rescue window. If the wrong stay basin is already chosen by checkpoint `102`, repaired late admission logic arrives too late.

**Observed evidence**
- In the force-stay counterfactual, forcing away early `v19` appends at `3,7,61` changes the middle path but **does not** restore late `v18`-style rescue.
- By `102+`, both `v19` actual-prefix and `v19 force_stay_3_7_61` reconverge into `no_tracking_win_vs_stay -> stay`.
- Baseline margins are tie-like early, but become clearly nontrivial in the late failing window for `v19` (`115 ~ 9.22e-2`, `120 ~ 1.60e-1`), suggesting late hardening rather than mere tie noise.
- Representative late stay choices are anti-drive-looking, e.g. `v19@102 = norm_locked_blend_v1, blend -0.5, step 0.0, gain 1.0`.

**Key seam references**
- `notes/2026-04-16-hh-site3-checkpoint-probe-summary.md`
  - late counterfactual findings and baseline tuples.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:5728-6157`
  - baseline step-scale / gain-scale selection path.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:2853-2942`
  - rollout horizon scoring.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:5208-5302`
  - tangent-secant proposal path.
- tie/search-order evidence from the baseline-geometry audit:
  - tie-breaking prefers lower blend, then lower step, then lower gain;
  - gain search is downstream of the current winner, not global.

**Current verdict**
Highly likely. This remains the best global-cause candidate.

---

### 4. Fidelity-first branch is hybrid/confounded (**confirmed interpretation hazard**) 
**What it is**
The fidelity-first branch is not a clean alternate controller law. It swaps the guarded core score, but still inherits the live d-barrier penalty and large portions of the guarded turn-window / protected-horizon structure.

**Why it matters**
This branch should not be treated as a decisive verdict on “the fidelity-first idea.” It changes multiple coupled mechanisms at once and disables some d-shape-only helpers while retaining other guarded machinery.

**Observed evidence**
Audits found that fidelity-first:
- still adds the live d-barrier penalty on top of the fidelity-first core,
- still uses shared guarded admission/screening structure,
- loses d-shape-only preferred-site behavior,
- leaves some tuned knobs effectively dead/no-op in practice.

**Key seam references**
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:3818-3898`
- `pipelines/time_dynamics/optimization/hh_realtime_optuna.py:215-227`, `365-410`, `592-652`
- `test/test_hh_realtime_checkpoint_controller.py:9622-9694`

**Current verdict**
Confirmed. Treat `v19` as a hybrid probe, not a clean alternate-law verdict.

---

### 5. Late rescue-family demotion before/within shortlist-confirm (**likely, not yet fully settled**) 
**What it is**
The late `v18` rescue family is absent from the recorded `v19` confirmed top candidate set at `102,115,117,118,119,120`, but that only proves absence from the recorded confirmed top-k, not full-pool absence.

**Why it matters**
This determines whether the next deeper diagnostic should target proposal-family generation or shortlist/ranking mechanics.

**Observed evidence**
At each late rescue checkpoint, the `v18` winning label is absent from both:
- `v19` actual-prefix confirmed top set,
- `v19` force-stay confirmed top set.

But the probe code only emits a limited confirmed top set, so the strongest honest interpretation is **demotion/shortlist divergence**, not proven pool absence.

**Key seam references**
- `notes/2026-04-16-hh-site3-checkpoint-probe-summary.md`
  - late-window candidate-family mismatch table.
- `pipelines/time_dynamics/hh_realtime_checkpoint_probe.py:24-95`
  - confirmed top-k telemetry boundary.
- `pipelines/time_dynamics/hh_realtime_checkpoint_controller.py:8788-8811`
  - confirm/selection telemetry seam.

**Current verdict**
Open, but now narrower. Best current read is demotion before or within shortlist/confirm.

---

### 6. Optuna feasibility collapse masks ranking signal (**confirmed study-level problem**) 
**What it is**
The fidelity-first site3 prefsite/autobaseline Optuna study is currently dominated by hard validity gates rather than by useful objective ranking.

**Observed evidence**
Current audit result:
- `24/24` invalid
- `feasible_trial_count = 0`
- `best_objective_trial = null`

Also, the density-first feasible profile has:
- objective weights on energy/fidelity = `0.0`
- but hard validity gates requiring:
  - mean energy error `<= 0.25`
  - final energy error `<= 0.25`
  - fidelity `>= 0.90`
  - mean total occupation abs error `<= 0.05`

So energy/fidelity do not shape objective ranking continuously here; they mainly delete trials.

**Key seam references**
- `pipelines/time_dynamics/optimization/hh_realtime_optuna.py:206-253`
- `artifacts/agent_runs/20260415_hh_l2_high_amp_fidelity_first_site3_prefsite_autobaseline_optuna_v1/logs/command.sh`
- Optuna audit summary from the finished second-wave subagent pass.

**Current verdict**
Confirmed. This study is an alarm bell first, optimizer second.

## Current ordering for intervention / investigation
1. **First seam to interrogate conceptually:** single-surface commit law.
2. **Second seam to parity-check:** preferred-site/search-surface parity for fidelity-first vs guarded d-shape.
3. **Third seam to instrument if needed:** full `pool -> scout -> shortlist -> confirm` audit at `102/115/117/118/119/120`.
4. **Only after that:** consider larger stay-score redesign or proposal-family expansion.

## What not to do next
- Do **not** treat `v19` as a clean fidelity-first falsification.
- Do **not** start from another generic pre-turn threshold relaxation or outside-turn stall tweak.
- Do **not** use the current Optuna objective table as trustworthy model ranking.
- Do **not** promote append-family insufficiency back to rank #1 unless the late full-pool audit actually proves the rescue family is missing from the pool, not merely demoted.

## Bottom line
If we want the smallest, highest-information next conceptual seam, it is still this:

> The same comparison surface that says a candidate is good enough to admit should also decide final commit versus stay.

If that seam proves unimportant after parity-clean comparison, the evidence shifts more decisively toward the **wrong late stay/baseline basin** as the main cause.
