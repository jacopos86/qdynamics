# Critique: Paper I SNAKE Noise Robustness Run Plan

## 1. Top 3 under-specified seams
1. **Noise seed / replicate policy.** The plan requires `seed policy` in records/manifests (`docs/plans/paper-i-snake-noise-robustness-run-plan-2026-05-31.md:58`, `:116`, `:149`, `:182`) but never says whether noise seeds are shared across `N_eff`, regimes, trials, or repeated replicates. This affects statistical credibility and record generation order.
2. **Stop-gate signal definition.** The SNR gate says “best available Phase3 gradient signal” with `max(abs(gradient)/gradient_stderr)` (`docs/plans/...:99`) and the runway gate allows `n_rem_high <= 1.0` or `useful_horizon <= 1.0` (`:98`), but it does not define which candidate set/phase/window supplies the gradient, whether `sigma_hat` or `gradient_stderr` wins, or how “or” is adjudicated. Implementers would guess at the semantic boundary between telemetry and termination.
3. **Record-generator ownership.** Item 1 asks for “one record/manifest schema or generator option” (`docs/plans/...:116`), while Item 4 says add a mode or path reusing the three-model Route-A generator (`:146-151`). The export preserved the same ambiguity (“mode or companion generator,” `prompt-exports/oracle-plan-2026-05-31-112853-snake-noise-plan-212-99c3.md:213-215`). The plan should choose whether this is an HH-only companion generator/contract or a gated extension of the three-model generator.

## 2. Specificity balance
- **Over-specified:** Initial `sigma0_abs=1.0e-3`, five `N_eff` levels, `snr_stop_threshold=2.0`, and `n_rem_high/useful_horizon <= 1.0` (`docs/plans/...:82-100`, `:193`) are useful calibration seeds, but the work items risk turning them into implementation constants.
- **Useful framing dropped:** The export named exact HH case IDs (`prompt-exports/...:158-162`); the plan downgrades this to regime names (`docs/plans/...:65`, `:107`). That weakens generator validation. The export also included `selector_measurement_proxy.py` for shot/work accounting (`prompt-exports/...:42`, `:115`), but the final plan omits it from key files/references despite making shot-equivalent metadata auditable.

## 3. Contradictions or missing dependencies
- The plan cites prior caution that noise overlays were deferred until ideal/noiseless SNAKE was clean (`docs/plans/...:48-49`) but has no explicit precondition to verify current clean HH baseline status before the noisy candidate grid.
- `Open Questions: None blocking` (`docs/plans/...:249-250`) is too strong given unresolved seed policy, stop-signal definition, generator ownership, and calibration-to-candidate selection rules.
- Item 9 requires selected `N_eff` levels after calibration (`docs/plans/...:204`, `:208`) but no dependency defines the selection criterion or CHTC budget cap.

## 4. Risk of over-planning
The plan is heavier than necessary for an implementation handoff: Items 1, 3, 4, and 7 all restate manifest/routing/preservation guardrails, and the final “Files to edit” repeats per-item key files (`docs/plans/...:113-184`, `:236-245`). The long Background is valuable provenance but should not become an implementation checklist; the actionable plan could stay at metadata/routing, stop gate, tests/reporting, then smoke/calibration/candidate.

## 5. Questions whose answers change implementation order
1. Should a clean/noiseless HH SNAKE baseline be verified or repaired before any noise implementation/run work?
2. What is the required noise-seed/replicate policy for candidate evidence?
3. Should `noise_floor_agreement_v1` be telemetry-only during calibration, with terminal stopping enabled only after calibration passes?
4. Should the HH noise grid be a new HH-only generator/manifest contract or an extension of the existing three-model Route-A generator?
5. After calibration, should the candidate grid run all five `N_eff` levels or only a selected subset under an explicit CHTC budget?
