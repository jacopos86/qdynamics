# Realtime Checkpoint Controller: Agent Handoff

## Purpose

This note is for future coding agents working on the HH adaptive realtime checkpoint controller. It records what actually worked in code, what was a dead end, what the current algorithm is, and how to sync that back into the math manuscripts.

This note is not the final source of truth. The source-of-truth manuscript targets are:

- `MATH/adaptive_selection_staged_continuation.tex`
- `MATH/adaptive_selection_and_mclachlan_time_dynamics.tex`

The immediate repo implementation lives primarily in:

- `pipelines/hardcoded/hh_realtime_checkpoint_controller.py`
- `pipelines/hardcoded/hh_realtime_checkpoint_types.py`
- `pipelines/hardcoded/hh_staged_cli_args.py`
- `pipelines/hardcoded/hh_staged_workflow.py`

## Executive Status

We now have a driven exact-v1 controller that can:

- leave the old "static until append" failure mode,
- recover the correct time-alignment / temporal shape on the old benchmark,
- tune amplitude by exact forecast terms,
- slightly overshoot or undershoot exact energy when pushed.

However, the benchmark used for most of this tuning is now known to be the wrong optimization surface for QPU-facing work:

- old tuning point: `A=1.5`, `omega=2.0`, `tbar=2.5`, `t0=5.0`, `t_final=10.0`, `num_times=201`
- exact energy span on that window is only about `5.9e-3`
- this is too close to a real QPU noise floor of about `1e-3`

So the current algorithm work is scientifically useful, but future tuning should move to a heavier perturbative drive with a larger and longer-lived exact response.

## Current Best Fine-Benchmark Setting

On the old fine-resolution benchmark, the current best tradeoff is:

- horizon steps: `2`
- horizon weights: uniform
- energy slope weight: `500`
- energy curvature weight: `200`
- full blend ladder: `[0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0]`
- gain ladder: `[1.0, 1.05, 1.1, 1.15, 1.2, 1.25]`
- excursion under-weight: `900`
- excursion over-weight: `120`
- excursion relative tolerance: `0.03`

This was labeled:

- `band_u900_o120_t03`

Its role is not "final QPU candidate". Its role is "best current proof that the algorithm can recover correct driven time-dynamics shape and tune amplitude with exact forecast terms".

## What Actually Worked

### 1. The stay manifold was genuinely blind

The original problem was not just plotting or drive plumbing. The fixed scaffold on the canonical replay seed was nearly first-order blind to the drive on the stay manifold.

Evidence:

- stay-only flow produced almost no visible motion
- finite-difference checks showed the first-order stay gradient was effectively zero on the bad checkpoints

Practical consequence:

- pure fixed-scaffold McLachlan on this scaffold can look static even when the exact driven reference is moving

### 2. Drive-aligned density augmentation fixed the shape problem

The first real algorithmic win was to augment the stay baseline with a drive-aligned density direction, then let exact forecast choose a blend.

Important implementation detail:

- do not simply add the raw drive-only direction
- first remove the component parallel to the baseline McLachlan flow
- then renormalize back to the baseline quadratic norm

This changed the direction of motion without making the blended vector artificially larger just because of norm inflation.

Net effect:

- time-alignment improved substantially
- the controller stopped looking "too smooth and in the wrong phase"

### 3. Horizon scoring helped, but only modestly

Moving from horizon `h=1` to `h=2` improved shape a little. `h=3` helped slightly more, but the gain was small relative to extra complexity.

Practical lesson:

- `h=2` is the best default compromise
- horizon alone is not enough to fix amplitude

### 4. Energy slope / curvature terms were useful, but not enough by themselves

Adding energy-slope and energy-curvature mismatch terms to the exact forecast score made the controller less naively pointwise.

Useful settings:

- slope weight around `500`
- curvature weight around `200`

Important caveat:

- curvature for `h=2` needed the previous-step anchor to become active
- before that fix, the curvature term at `h=2` was effectively inert

### 5. Full blend ladder was better than low-blend caps

We compared:

- full blend ladder
- restricted ladders such as `blend <= 0.4`

The low-blend caps sometimes matched energy amplitude a bit more closely, but they gave up too much of the temporal shape. The full ladder gave the best overall driven shape.

### 6. Joint gain + step-scale search was a dead end

We tried adding a gain ladder above `1.0`, but the first implementation let gain and step-scale cancel each other. That made the experiment expensive and mostly meaningless.

Fix:

- choose blend + step-scale first
- then apply post-step gain as a genuine amplitude knob

Even after that correction, gain ladder alone did not give enough improvement to justify its extra cost as the main tuning lever.

### 7. One-sided excursion-underresponse term was the first major amplitude fix

The first genuinely strong amplitude improvement came from penalizing under-response in signed energy excursion over the forecast horizon.

This lifted the controller trace without destroying the shape recovered by full blend.

### 8. Mild symmetric excursion band was better than pure under-lift

The current best fine-benchmark result came from replacing pure under-lift with a narrow amplitude band around the exact signed excursion:

- penalize under-response strongly
- penalize over-response weakly
- allow a small relative tolerance band before either penalty activates

Best fine-benchmark setting so far:

- under `900`
- over `120`
- tolerance `0.03`

This kept the `exc800` shape but moved the energy closer to exact and reduced kinking.

### 9. The controller can overshoot

This is now known explicitly.

On the same fine benchmark, a stronger pure under-only setting already produced slight positive signed energy crossings:

- `exc1000` had positive signed energy difference at some times
- so the algorithm is not hard-clipped below exact

That matters because it means the remaining amplitude issue is tuning, not a structural "cannot overshoot" limitation.

## What Did Not Work

### Empty candidate pool was a red herring for the staged exact-v1 path

The legacy fixed-manifold runner has a structurally locked pool. That is real, but it is not the explanation for the staged exact-v1 controller we were debugging.

Do not confuse:

- legacy fixed-manifold replay lock
- current staged adaptive controller family pool

### One-step weight retuning alone

Simple normalization / one-step weight changes did not produce the desired shape-amplitude fix. Some of those experiments made the trajectory worse.

### Baseline step refinement rounds

Densifying the local step-scale ladder around the best scale looked sensible, but in practice it did not help. It made the trace worse on the fine benchmark.

### Stronger excursion-band penalties

Once the band became too strong, the controller fell back downward again and lost the desired lift. Mild banding worked; aggressive banding did not.

## Troubleshooting Lessons

### If the controller looks static

Check:

1. Is the exact/reference branch actually moving?
2. Is the stay manifold first-order blind on this scaffold?
3. Is the drive-aligned density direction present and being found by label?

One real bug we hit:

- the runtime labels had replay suffixes like `__r0`
- matching only the bare drive label made the blend path silently inert

### If the controller shape is right but amplitude is low

Check:

1. full blend ladder is active
2. horizon is at least `2`
3. slope / curvature terms are active
4. excursion-under / excursion-band terms are active

### If gain ladder seems to do nothing

Check whether gain and step-scale are being jointly optimized. If they are, they can cancel and make the experiment misleading.

### If `h=2` curvature seems useless

Check that the score includes the previous-step anchor. Without that anchor, second differences are not actually being formed.

### If the benchmark "looks good" but the drive is too small

Do not keep tuning it.

That is the current situation with the old `A=1.5`, `t0=5.0`, `tbar=2.5` benchmark window. It was a useful proving ground, but it is now the wrong target because the exact response is too close to the QPU noise scale.

## Current Algorithm in Math Form

### Continuous within-scaffold motion

On the current scaffold, solve projective McLachlan on the fixed tangent plane:

$$
\dot\theta_{\mathrm{base}}
\in
\arg\min_v \left\{ \lVert \bar T v - \bar b \rVert^2 + v^\top \Lambda v \right\}.
$$

### Drive-only residual direction

Restrict to the runtime coordinates associated with the drive-aligned density block, solve the reduced McLachlan system on that subspace, and embed back to the full runtime coordinate vector:

$$
\dot\theta_{\mathrm{drive}}
=
\iota_{\mathrm{drive}}
\left(K_{\mathrm{drive}}^{+} f_{\mathrm{drive}}\right).
$$

Then remove the component already explained by the baseline:

$$
r_{\mathrm{drive}}
=
\dot\theta_{\mathrm{drive}}
-
\frac{\langle \dot\theta_{\mathrm{base}}, \dot\theta_{\mathrm{drive}} \rangle_G}
{\langle \dot\theta_{\mathrm{base}}, \dot\theta_{\mathrm{base}} \rangle_G}
\dot\theta_{\mathrm{base}}.
$$

Renormalize the residual to the baseline quadratic norm, then blend:

$$
\dot\theta_{\mathrm{blend}}(w)
=
\operatorname{Renorm}_G
\left(
\dot\theta_{\mathrm{base}} + w \, r_{\mathrm{drive}}
\right),
\qquad
w \in \mathcal W_{\mathrm{blend}}.
$$

### Step scale and gain

For each candidate blend, evaluate a stay-step scale and then a post-step gain:

$$
\dot\theta_{\mathrm{cand}}(w,s,g)
=
g \, s \, \dot\theta_{\mathrm{blend}}(w),
\qquad
s \in \mathcal S_{\mathrm{step}},
\quad
g \in \mathcal S_{\mathrm{gain}}.
$$

Operationally, the important point is:

- choose blend and step first
- then choose gain as a genuine post-step amplitude knob

### Horizon forecast score

Let the exact forecast horizon be \(j=1,\dots,H\), with weights \(\omega_j\). The base pointwise score is

$$
S_{\mathrm{pt}}
=
\frac{1}{\sum_j \omega_j}
\sum_{j=1}^{H}
\omega_j
\left[
\delta_{\mathrm{fid},j}
+
e_{\mathrm{stag},j}
+
e_{\mathrm{dbl},j}
+
e_{\mathrm{occ},j}
+
e_{E,j}
\right],
$$

where

$$
\delta_{\mathrm{fid},j} = \max(0, 1 - F_j),
$$

and the other terms are the absolute exact-reference errors at the forecast point.

### Shape terms

For horizon scoring, add energy-slope and energy-curvature mismatch terms:

$$
S_{\mathrm{shape}}
=
\lambda_{\mathrm{slope}} \, \overline{\Delta_E^{(1)}}
+
\lambda_{\mathrm{curv}} \, \overline{\Delta_E^{(2)}}.
$$

The current implementation uses the previous-step anchor so curvature is active even at `h=2`.

### Excursion band terms

Let the signed controller and exact energy excursions be measured relative to the previous-step anchor:

$$
\epsilon^{\mathrm{ctrl}}_j
=
\operatorname{sign}(\epsilon^{\mathrm{exact}}_j)
\left(E^{\mathrm{ctrl}}_j - E^{\mathrm{ctrl}}_{\mathrm{anchor}}\right),
\qquad
\epsilon^{\mathrm{exact}}_j
=
E^{\mathrm{exact}}_j - E^{\mathrm{exact}}_{\mathrm{anchor}}.
$$

Let the exact excursion magnitude be \(a_j = |\epsilon^{\mathrm{exact}}_j|\), and define a relative tolerance band

$$
[a_j(1-\tau_{\mathrm{rel}}), \, a_j(1+\tau_{\mathrm{rel}})].
$$

Then the one-sided penalties are

$$
p^{\mathrm{under}}_j
=
\max\{0, a_j(1-\tau_{\mathrm{rel}}) - \epsilon^{\mathrm{ctrl}}_j\},
$$

$$
p^{\mathrm{over}}_j
=
\max\{0, \epsilon^{\mathrm{ctrl}}_j - a_j(1+\tau_{\mathrm{rel}})\}.
$$

The final excursion contribution is

$$
S_{\mathrm{exc}}
=
\lambda_{\mathrm{under}}
\frac{\sum_j \omega_j p^{\mathrm{under}}_j}{\sum_j \omega_j}
+
\lambda_{\mathrm{over}}
\frac{\sum_j \omega_j p^{\mathrm{over}}_j}{\sum_j \omega_j}.
$$

So the total stay forecast score is

$$
S_{\mathrm{forecast}}
=
S_{\mathrm{pt}} + S_{\mathrm{shape}} + S_{\mathrm{exc}}.
$$

## What Should Be Reflected in the Manuscripts

### `MATH/adaptive_selection_staged_continuation.tex`

This file should reflect:

- that the staged continuation/controller surface now includes an exact-v1 driven stay-direction policy, not just append/prune shortlist logic
- that the controller can use a drive-aligned residual direction inside the stay lane
- that amplitude control is now forecast-based rather than only append-based
- that the controller has a small family of tunable exact-forecast policy weights:
  - horizon
  - slope
  - curvature
  - excursion under
  - excursion over
  - excursion tolerance band

In other words, continuation is no longer just "select records and continue"; it also includes a runtime policy for selecting the within-scaffold motion law used between append/prune events.

### `MATH/adaptive_selection_and_mclachlan_time_dynamics.tex`

This file should reflect, explicitly:

1. the drive-aligned residual blend construction,
2. the stay-lane exact forecast horizon score,
3. the energy slope / curvature penalties,
4. the signed excursion band penalties,
5. the fact that gain is a post-step amplitude choice, not a redundant joint magnitude with step-scale.

The current Section 17 already has the right chapter-level framing, but it needs to be updated from the older "miss opens append lane" picture to the more accurate runtime statement:

- stay itself is now a nontrivial forecast-selected policy surface
- the controller can recover driven dynamics without opening append
- append/prune are no longer the only meaningful controller actions

## Heavier-Drive Retarget: Next Correct Optimization Surface

Future tuning should move to a heavier perturbative drive whose exact response is well above the QPU noise floor and remains active longer in the plotted window.

Working target from earlier retarget experiments:

- `drive_A ≈ 0.5 - 0.6`
- `drive_omega = 2.0`
- `drive_tbar ≈ 4.0`
- `drive_t0 = 0.0`
- `t_final ≈ 8.0`

Why this target:

- exact energy span is in the rough `1e-2` to `1e-1` window rather than around `6e-3`
- the pulse is not mostly spent by `t ≈ 4`
- the exact dynamics remain visually active through much more of the plotted interval

This is the right next optimization target for QPU-preparatory tuning.

## Practical Advice to the Next Agent

1. Do not restart from the old static/frozen diagnosis.
2. Do not retune the old fine benchmark forever.
3. Keep the full blend ladder.
4. Start heavier-drive retuning from the current fine-benchmark winner:
   - horizon `2`
   - slope `500`
   - curvature `200`
   - under `900`
   - over `120`
   - tolerance `0.03`
5. Re-evaluate those values on the heavier drive before changing the algorithm again.
6. Only after the heavier-drive benchmark is established should you decide whether amplitude still needs more lift or whether the score should become more symmetric.
