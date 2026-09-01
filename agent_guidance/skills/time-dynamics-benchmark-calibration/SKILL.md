---
name: time-dynamics-benchmark-calibration
description: Pre-execution Paper-II dynamics benchmark calibration audit. Run before CHTC submission, aggregation, evidence-PDF generation, or a visible table update, and fail closed on ERROR. Emits a machine-readable calibration receipt for paper-ii-results. Do not use for run execution, ordinary dynamics implementation, mathematical explanation, or manuscript prose.
---

# Time-dynamics benchmark calibration

Recreated 2026-09-01 as a repo-local, script-backed skill (the original was a
user-level `$time-dynamics-benchmark-calibration` adapter lost with the previous
machine — `RECOVERY-AUDIT.md` §8, recreation spec
`docs/reports/EVIDENCE_PARITY_AUDIT_20260831.md` §4 item 2). It is repo-tracked
so it survives machine loss.

This is a **pre-execution** audit, not a post-hoc report. `ERROR` means stop:
`agent_guidance/skills/paper-ii-run/SKILL.md` gates CHTC submission,
aggregation, evidence-PDF generation, and table updates on it, and
`agent_guidance/skills/paper-ii-results/SKILL.md` refuses to touch a visible
Paper-II cell while a calibration `ERROR` stands **or a required check was never
run**.

## Runner

```bash
PYTHONPATH=. .venv/bin/python -m pipelines.time_dynamics.diagnostics.paper_ii_benchmark_calibration \
  --artifact MATH/paper_facing/paper_II_dynamics/<frontier_or_promotion_receipt>.json \
  --run-root output/<campaign>/ \
  --receipt-json output/diagnostics/paper_ii_calibration/<stamp>/calibration_receipt.json
```

- `--self-test` exercises every check against the restored lock and synthetic
  pass/fail fixtures. Run it after any change to the runner.
- Receipt schema `paper_ii_benchmark_calibration_receipt_v1`; status is one of
  `ERROR` / `WARN` / `INCOMPLETE` / `OK`. Exit `1` only on `ERROR`.
- `INCOMPLETE` means one or more checks had no local evidence. It is never a
  pass. Most Paper-II raw trajectories are CHTC-side (`EVIDENCE_PARITY_AUDIT_20260831.md`
  §3), so `INCOMPLETE` is the expected state until a fetch lands.
- Hand the receipt path and its status to `paper-ii-results` in the evidence
  handoff, alongside the class-settings lock record and the QPU-faithful status.

## What the runner checks mechanically

| Check | Verdict source |
|---|---|
| `promoted_class_settings_lock` | `chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_lock_v1.json` exists, digests, declares `dynamics_class_settings_lock_manifest_v1`, `lock_status: locked`, `require_canonical_controller_classes`, and every settings entry is `class_tuned_result_locked` + `strict_online_feedback_exact_free`. Absent lock, or only the smoke variant, is `ERROR`. |
| `benchmark_point_sameness` | Groups rows by benchmark point (seed/regime, drive, horizon, controller) and requires every method inside one point to agree on static seed hash, drive, time grid, phonon cutoff, reference/diagnostic policy, observable set, diagnostic exact reference, and compile target. Axes the rows do not declare are reported as **undeclared**, never as passing. |
| `seed_provenance` | No `staged`, `fallback`, `pending`, or `recovery` seed artifact backs the evidence. Markers are matched against the seed artifact's own basename and explicit seed-status fields; parent directories are excluded on purpose, because the canonical campaign directory is literally `paper_ii_staged_accuracy_euler_v1`. Also reports every distinct `seed_sha256` across the artifact set. |
| `trajectory_freeze_invalid_rows` | The `theta_dot_l2=0`, `rho_num=0`, `rho_miss≈1` signature over trajectory rows, at the `1e-12` motion tolerance already used by `pipelines/time_dynamics/benchmarks/common.py`. Also surfaces any embedded `non_frozen_when_exact_diagnostic_moves` sidecar failure. |
| `qpu_faithful_decision_separation` | `uses_reference_for_decision`, `uses_future_exact_forecast_for_decision`, `exact_reference_controller_inputs` across `runtime_contract` / `summary` / `route_config` / `provenance`. Truthy is leakage; **undeclared is also `ERROR`** — absence is not separation. |
| `matched_accuracy_policy_binding` | The live campaign constants in `pipelines/time_dynamics/campaigns/paper_ii_staged_accuracy_v1.py` still equal the protocol's 5% individual response target, 15% response guard, factor-of-two matched burden, and 1e-4 mean-infidelity ceiling. |
| `response_normalized_matched_accuracy` | Applies the **live** `AccuracyPolicy` object (not a reimplementation) per row — `qualifies`, `comparison_eligible`, `accuracy_burden`, response components — and per complete APM/AVQDS pair via `pair_matches`. |

No threshold in the runner is invented. Every constant is read from the lock
manifest, the live campaign module, or `benchmarks/common.py`.

## What routes to documents, not to the script

- **Admission and comparison reasoning.**
  `agent_guidance/time-dynamics/paper-ii-comparison-protocol.md` is the
  authority for why comparison is at fixed accuracy rather than fixed
  threshold, what the factorial axes are, rung selection and its selection
  bias, and the reporting contract. The runner scores rows against the policy;
  it does not decide what a scored row means for the paper.
- **Author overrides.** When an artifact declares an explicit override (e.g.
  `t20_provisional_author_override_receipt.json`), rows or pairs that miss the
  policy are reported as `WARN` naming the override, never as an automated
  pass. Promotion under an override is the user's decision.
- **Declaring the undeclared axes.** Observable set, compile target, time grid,
  and phonon cutoff are frequently absent from receipt rows and present only in
  the run payloads those rows name. Resolve them against the run payloads or
  the class-settings lock; do not treat the runner's `undeclared` count as a
  pass.
- **Promoting a newer lock.** `paper-ii-run` SKILL.md owns that; the runner only
  enforces whichever lock it is pointed at.
- **Exact-reference store absence.** The canonical policy construction verifies
  the tracked response-scale sidecar against
  `output/paper_ii_staged/exact_references/`, which is regenerable-local and
  routinely missing (`EVIDENCE_PARITY_AUDIT_20260831.md` §2.2). Regenerate it
  with `pipelines/time_dynamics/diagnostics/exact_reference_store.py`, or pass
  `--allow-unverified-response-scales` to score against the sidecar's recorded
  scales — that degraded mode never reports `OK`.

## Order of operations

1. Confirm the promoted lock and the target visible claim before anything runs.
2. Run the calibration runner over the artifacts the campaign will consume.
3. On `ERROR`, stop and report the failing check; do not submit, aggregate,
   build an evidence PDF, or edit a table.
4. On `INCOMPLETE`, either supply the missing evidence (fetch or regenerate) or
   carry the unrun-check list explicitly into the handoff — `paper-ii-results`
   treats it as its own blocker.
5. On `OK` or an override-explained `WARN`, hand the receipt path, status, and
   unrun-check list to `paper-ii-results` with the rest of the evidence
   handoff.

## Related

- `agent_guidance/skills/paper-ii-run/SKILL.md` — gates on this skill.
- `agent_guidance/skills/paper-ii-results/SKILL.md` — consumes this receipt.
- `agent_guidance/time-dynamics/paper-ii-comparison-protocol.md` — the policy.
- `agent_guidance/time-dynamics/run-guide.md` and `AGENTS.md` — lane contracts.
