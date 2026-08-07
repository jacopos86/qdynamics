# Oracle Review

## Summary

The proposed route is directionally consistent with the current code surface: `--static-lane-route physical_operator_type` is wired through CLI parsing, lane normalization, physical classifiers, candidate metadata, generic shortlist machinery, and final `adapt_vqe.continuation` provenance. The plan correctly prioritizes HH first, requires no launch without approval, uses full reoptimization/no refit overlay, and treats `other` labels as stop conditions. The remaining gaps are mainly provenance/launch-safety gates that must be closed before proposing concrete tmux commands.

## Findings

### P1 — Should Fix

1. **Missing duplicate-run / overwrite gate before tmux launch**  
   **Ref:** Generated plan: “Local tmux run plan”; `adapt_pipeline.py` current checkpoint/output paths  
   The plan does not explicitly require checking existing tmux sessions, running local Python processes, or live `current.json` files under the target roots before assigning output paths. This risks duplicate HH jobs or overwriting a partial run.  
   **Suggestion:** Before command proposal, inspect existing tmux sessions, local static-ADAPT processes, and each target run directory for `current.json`, `result.json`, logs, and completion status. If any matching run is active or incomplete, stop and ask whether to monitor, resume, or use a new root.

2. **20260709 HH target-root inspection is a hard blocker**  
   **Ref:** `raw_outputs/paper_i_hh_physical_operator_lanes_integer_caps_fullreopt_scan_20260709`; `raw_outputs/paper_i_cross_model_physical_operator_lanes_1p75_fullreopt_20260709`  
   The plan acknowledges these roots were unavailable, but command reconstruction must not fall back to 20260708 anchors as if they were equivalent.  
   **Suggestion:** Treat missing or unreadable 20260709 roots as blocked. Inspect `current.json`, `result.json`, `run_command.*`, `effective_command.json`, `commands.json`, `source_lock_manifest.json`, and any manifest sidecars before deriving HH commands.

3. **Batching semantics conflict must be resolved explicitly**  
   **Ref:** `paper_i_hh_powell_visible_recovery_candidate_settings_20260706.md`; route facts requesting `--phase2-no-batching` / `--phase3-no-batching`; `generate_paper_i_hh_weak_weak_snake_mechanism_ablation_records.py`  
   Some nearby docs/generators describe maxB=1 or cap-3 batch routes, while the requested 20260709 physical-lane route uses no batching unless target provenance says otherwise.  
   **Suggestion:** Add a settings-diff audit field for batching. Do not mix maxB=1/cap-3 CHTC mechanism-ablation anchors into the no-batching local route without explicit approval.

4. **Full-reoptimization/no-overlay must be proven from effective command, not inferred**  
   **Ref:** `cli_config.py`; `output_artifacts.py`; `adapt_pipeline.py`  
   `adapt_final_full_refit` may appear as a string in top-level `settings`, and `current.json` may not expose all reoptimization settings.  
   **Suggestion:** Verify these from `run_command.json`, `effective_command.json`, or `commands.json` first:  
   ```text
   --adapt-reopt-policy full
   --adapt-full-refit-every 0
   --adapt-final-full-refit false
   ```  
   Do not resume or continue an old `windowed` / every-8 / final-refit artifact in place.

5. **`other` audit needs exact labels, not only summary counts**  
   **Ref:** `static_provenance.py`; `adapt_pipeline.py` physical-lane summary  
   Runtime summaries include `other_count`, but may not preserve exact `other` labels in the policy summary, and counts can reflect repeated candidate classifications rather than unique pool labels.  
   **Suggestion:** Prelaunch audit must emit unique exact labels, classifier payloads, lane counts, and `other_labels`. If nonempty, stop and ask with the label list.

6. **Read-gate checklist is incomplete before command proposal**  
   **Ref:** `agent_guidance/skills/paper-i-run/SKILL.md`; `source-locked-sensitivity/SKILL.md`  
   The plan references key Paper-I docs but should explicitly require the visible/support source chain before commands.  
   **Suggestion:** Before proposing commands, inspect root/MATH run guidance if available, visible HH support CSV/JSON, source result/effective-command sidecars, and the requested raw-output roots. Keep CHTC manifests as provenance references only.

7. **Do not allow “narrow plumbing fixes” before approval**  
   **Ref:** Generated plan / audit gates  
   The user request says no edit or launch without explicit approval. Any implied permission for plumbing fixes should be removed.  
   **Suggestion:** Treat missing top-level provenance duplication, command-sidecar gaps, or audit-script needs as blockers/questions. Do not patch until the user approves that exact code change.

8. **Spin-boson/Rabi command naming needs care**  
   **Ref:** `problem_registry.py`; `static_provenance.py`  
   The classifier aliases `rabi` to `spin_boson`, but the CLI problem registry exposes `spin_boson`, not necessarily `rabi`.  
   **Suggestion:** Present lane semantics as spin-boson/Rabi, but use/audit `--problem spin_boson` in any future command template unless the CLI is verified to accept `rabi`.

### P2 — Consider

1. **Do not expect live `current.json` to contain final lane policy**  
   **Ref:** `adapt_pipeline.py` current checkpoint writer  
   Final `result.json` carries `adapt_vqe.continuation.static_lane_policy` and `physical_operator_lane_policy`; live checkpoints may not.  
   **Suggestion:** Use live `current.json` for depth/energy/operators, and final/result/candidate rows for lane-policy verification.

2. **Top-level `settings` may omit lane route details**  
   **Ref:** `output_artifacts.py`  
   Lane route provenance is primarily in `adapt_vqe.continuation`, not necessarily top-level `settings`.  
   **Suggestion:** Audits should read both top-level `settings` and `adapt_vqe.continuation`, plus command sidecars.

3. **Hubbard must be explicit UCCSD**  
   **Ref:** `problem_registry.py`; `static_provenance.py`  
   Hubbard defaults to `uccsd`, but source-locked/provenance-safe commands should not rely on defaults.  
   **Suggestion:** Set and audit `--adapt-pool uccsd`; verify `settings.adapt_pool` and `adapt_pool_requested`.

4. **CHTC artifacts are not launch authorization**  
   **Ref:** `chtc/phase3_optuna/...` manifests/generator  
   Nearby CHTC records are useful source anchors but target work is local tmux only.  
   **Suggestion:** Do not submit, fetch, or monitor CHTC for this route unless separately approved.