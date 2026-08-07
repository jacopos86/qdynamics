# Time Dynamics CHTC Storage Sources

Last updated: 2026-06-14.

Purpose: give future Paper-II/time-dynamics agents a durable pointer to the CHTC storage surfaces checked during the U8 HH cleanup, without requiring broad remote scans.

## Remote root

Default CHTC root for this repository:

```text
~/Holstein_phase3_optuna_chtc
```

## Time-dynamics storage surfaces checked

On 2026-06-14, the following targeted paths were checked on CHTC:

| Remote path | Size observed | Cleanup action |
|---|---:|---|
| `raw_outputs/generic_time_dynamics_table` | `341M` | Removed as generated raw output after confirming no active/recent dynamics jobs. |
| `chtc/generic_time_dynamics_table/input` | `264M` | Kept; staged input/source material for generic Paper-II table batches. |
| `chtc/time_dynamics_optuna/input` | `323M` | Kept; staged input/source material for time-dynamics Optuna/profile batches. |
| `logs/generic_time_dynamics_table` | missing | None. |
| `logs/time_dynamics_optuna` | missing | None. |

Remote cleanup note created by the CHTC session:

```text
~/cleanup_time_dynamics_space_20260614T144533Z.txt
```

## Guidance for future agents

- Treat `raw_outputs/generic_time_dynamics_table` as generated/reconstructable CHTC output unless a current Paper-II source map explicitly points to it.
- Do not delete `chtc/generic_time_dynamics_table/input` or `chtc/time_dynamics_optuna/input` for storage hygiene without checking whether their staged seed artifacts, class settings, or record manifests are still needed.
- Before deleting any time-dynamics evidence, check active jobs with a targeted query:

```bash
condor_q jsstrobel -nobatch -af ClusterId ProcId JobBatchName JobStatus RemoteHost HoldReason \
  | grep -Ei 'time_dynamics|generic_time|paper_ii|dynamics' || true
```

- For Paper-II evidence/table work, still follow `agent_guidance/skills/paper-ii-run/SKILL.md`, `$time-dynamics-benchmark-calibration` when available, and `agent_guidance/skills/paper-ii-results/SKILL.md`.
