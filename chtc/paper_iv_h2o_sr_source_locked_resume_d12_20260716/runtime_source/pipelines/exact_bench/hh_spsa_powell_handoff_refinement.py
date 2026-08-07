#!/usr/bin/env python3
"""Launch a narrow SPSA replay refinement from the best Powell HH artifacts.

This utility is designed to sit *behind* a broad SPSA Optuna study.  It waits
for the live broad-study summary, checks whether that study actually found any
in-band feasible SPSA point, and only then launches a tighter replay sweep that
starts from the best Powell incumbents rather than re-running broad ADAPT from
scratch.

The refinement surface is `pipelines/scaffold/hh_vqe_from_adapt_family.py`
with compile-scout post-processing from `pipelines/scaffold/adapt_circuit_cost.py`.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WAIT_SUMMARY = REPO_ROOT / "artifacts/agent_runs/20260414_hh_l2_current_focus_spsa_optuna_v1/summary.json"
DEFAULT_WAIT_PROGRESS = REPO_ROOT / "artifacts/agent_runs/20260414_hh_l2_current_focus_spsa_optuna_v1/progress.json"
DEFAULT_OUTPUT_TAG = "20260414_hh_l2_powell_manifold_spsa_refine_v1"
DEFAULT_EPSILON = 6.2e-5
DEFAULT_POLL_SECONDS = 60
DEFAULT_MAX_POLLS = 0
DEFAULT_RESTARTS = 8
DEFAULT_MAXITER = 2400
DEFAULT_COMPILE_BACKEND = "FakeMarrakesh"
DEFAULT_COMPILE_OPT_LEVEL = 1
DEFAULT_COMPILE_SEED = 7
DEFAULT_CURRENT_SUMMARY = REPO_ROOT / "artifacts/agent_runs/20260414_hh_l2_current_focus_spsa_optuna_v1/summary.json"
DEFAULT_LEGACY_SUMMARY = REPO_ROOT / "artifacts/agent_runs/20260414_hh_l2_legacy_focus_spsa_optuna_v1/summary.json"
DEFAULT_LEGACY_TRIAL_ROOT = REPO_ROOT / "artifacts/agent_runs/20260414_hh_l2_legacy_focus_spsa_optuna_v1/legacy/eps_6.200em05"
DEFAULT_MONITOR_STATE_JSON = REPO_ROOT / "artifacts/agent_runs/20260414_hh_l2_math_top2_monitor_v4/state.json"

DEFAULT_POLICIES = ("auto", "tile_adapt")
DEFAULT_SEED_ORDER = ("current_98", "current_118", "legacy_75", "legacy_81")

VQE_REPLAY_ENTRYPOINT = REPO_ROOT / "pipelines/scaffold/hh_vqe_from_adapt_family.py"
COMPILE_SCOUT_ENTRYPOINT = REPO_ROOT / "pipelines/scaffold/adapt_circuit_cost.py"
MATH_MONITOR_ENTRYPOINT = REPO_ROOT / "pipelines/exact_bench/hh_math_top2_monitor.py"


@dataclass(frozen=True)
class SeedSpec:
    name: str
    lane: str
    input_json: Path
    expected_delta_abs: float
    expected_two_qubit_count: int
    expected_depth: int
    handoff_state_kind: str
    notes: str


@dataclass(frozen=True)
class CaseSpec:
    lane: str
    lane_trial_index: int
    seed: SeedSpec
    replay_seed_policy: str
    replay_continuation_mode: str = "phase3_v1"

    @property
    def trial_name(self) -> str:
        return f"trial_{self.lane_trial_index:04d}"

    @property
    def case_name(self) -> str:
        return f"{self.seed.name}__{self.replay_seed_policy}"


@dataclass
class CaseOutcome:
    lane: str
    case_name: str
    case_dir: str
    params: dict[str, Any]
    abs_delta_e: float | None
    compiled_count_2q: int | None
    compiled_depth: int | None
    logical_operator_count: int | None
    runtime_parameter_count: int | None
    feasible: bool
    constraints: list[float]
    result_json: str
    compile_json: str
    returncode: int
    compile_returncode: int | None
    pipeline_elapsed_s: float
    compile_elapsed_s: float | None
    total_elapsed_s: float
    invalid_reasons: list[str]
    seed_name: str
    seed_notes: str


SEED_LIBRARY: dict[str, SeedSpec] = {
    "current_98": SeedSpec(
        name="current_98",
        lane="current",
        input_json=REPO_ROOT / "artifacts/agent_runs/20260414_hh_l2_bridge_diag_focus_optuna_v1/global/eps_6.200em05/trial_0024/json/result.json",
        expected_delta_abs=5.617823464465488e-05,
        expected_two_qubit_count=98,
        expected_depth=267,
        handoff_state_kind="reference_state",
        notes="Best Powell current-route incumbent from bridge-focused proxy-reduced search.",
    ),
    "current_118": SeedSpec(
        name="current_118",
        lane="current",
        input_json=REPO_ROOT / "artifacts/agent_runs/20260414_hh_l2_bridge_diag_focus_optuna_v1/global/eps_6.200em05/trial_0005/json/result.json",
        expected_delta_abs=5.617823464487692e-05,
        expected_two_qubit_count=118,
        expected_depth=351,
        handoff_state_kind="reference_state",
        notes="Runner-up Powell current-route incumbent from bridge-focused proxy-reduced search.",
    ),
    "legacy_75": SeedSpec(
        name="legacy_75",
        lane="legacy",
        input_json=REPO_ROOT / "artifacts/agent_runs/20260414_hh_l2_legacy_focus_optuna_v1/legacy/eps_6.200em05/trial_0016/json/result.json",
        expected_delta_abs=5.617823464446059e-05,
        expected_two_qubit_count=75,
        expected_depth=178,
        handoff_state_kind="prepared_state",
        notes="Best Powell legacy incumbent from the focused legacy search.",
    ),
    "legacy_81": SeedSpec(
        name="legacy_81",
        lane="legacy",
        input_json=REPO_ROOT / "artifacts/agent_runs/20260409_hh_l2_hist81_legacy_current_compare_d16_v3/legacy_20260322/json/result.json",
        expected_delta_abs=5.617823464482141e-05,
        expected_two_qubit_count=81,
        expected_depth=151,
        handoff_state_kind="prepared_state",
        notes="Frozen March 22 Powell legacy oracle.",
    ),
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_shell_command(path: Path, cmd: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(shlex.quote(str(x)) for x in cmd) + "\n"
    path.write_text(rendered, encoding="utf-8")
    path.chmod(0o755)


def _study_has_feasible_points(summary_payload: dict[str, Any]) -> bool:
    for study in summary_payload.get("studies", []):
        try:
            if int(study.get("feasible_count", 0)) > 0:
                return True
        except Exception:
            continue
        for obs in study.get("observations", []):
            if bool(obs.get("feasible", False)):
                return True
    return False


def _wait_for_summary(summary_path: Path, progress_path: Path | None, poll_seconds: int, max_polls: int) -> dict[str, Any]:
    polls = 0
    while True:
        if summary_path.exists():
            return _read_json(summary_path)
        if progress_path is not None and progress_path.exists():
            progress_payload = _read_json(progress_path)
            if bool(progress_payload.get("done", False)) and summary_path.exists():
                return _read_json(summary_path)
        polls += 1
        if int(max_polls) > 0 and polls >= int(max_polls):
            raise TimeoutError(f"Timed out waiting for summary: {summary_path}")
        time.sleep(float(max(1, poll_seconds)))


def _build_cases(seed_names: Sequence[str], policies: Sequence[str]) -> list[CaseSpec]:
    current_specs = [SEED_LIBRARY[name] for name in seed_names if SEED_LIBRARY[name].lane == "current"]
    legacy_specs = [SEED_LIBRARY[name] for name in seed_names if SEED_LIBRARY[name].lane == "legacy"]
    cases: list[CaseSpec] = []
    for lane, specs in (("current", current_specs), ("legacy", legacy_specs)):
        lane_idx = 0
        for spec in specs:
            for policy in policies:
                cases.append(
                    CaseSpec(
                        lane=lane,
                        lane_trial_index=lane_idx,
                        seed=spec,
                        replay_seed_policy=str(policy),
                    )
                )
                lane_idx += 1
    return cases


def _best_compile_row(compile_payload: dict[str, Any]) -> dict[str, Any] | None:
    rows = list(compile_payload.get("rows", []))
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            int(row.get("compiled_count_2q", 10**9)),
            int(row.get("compiled_depth", 10**9)),
            int(row.get("compiled_size", 10**9)),
        ),
    )


def _result_abs_delta(payload: dict[str, Any]) -> float | None:
    for path in (
        ("vqe", "abs_delta_e"),
        ("adapt_vqe", "abs_delta_e"),
    ):
        cur: Any = payload
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur is not None:
            try:
                return float(cur)
            except Exception:
                continue
    return None


def _logical_operator_count(payload: dict[str, Any]) -> int | None:
    for path in (
        ("replay_contract", "adapt_depth"),
        ("adapt_vqe", "ansatz_depth"),
        ("seed_baseline", "logical_num_parameters"),
    ):
        cur: Any = payload
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur is not None:
            try:
                return int(cur)
            except Exception:
                continue
    return None


def _runtime_parameter_count(payload: dict[str, Any]) -> int | None:
    for path in (
        ("vqe", "num_parameters"),
        ("adapt_vqe", "num_parameters"),
        ("replay_contract", "derived_num_parameters"),
    ):
        cur: Any = payload
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur is not None:
            try:
                return int(cur)
            except Exception:
                continue
    return None


def _build_replay_command(
    case: CaseSpec,
    result_json: Path,
    output_csv: Path,
    output_md: Path,
    output_log: Path,
    restarts: int,
    maxiter: int,
) -> list[str]:
    return [
        "python",
        "-u",
        str(VQE_REPLAY_ENTRYPOINT),
        "--adapt-input-json",
        str(case.seed.input_json),
        "--output-json",
        str(result_json),
        "--output-csv",
        str(output_csv),
        "--output-md",
        str(output_md),
        "--output-log",
        str(output_log),
        "--tag",
        f"{case.lane}_{case.case_name}",
        "--replay-continuation-mode",
        str(case.replay_continuation_mode),
        "--replay-seed-policy",
        str(case.replay_seed_policy),
        "--phase3-symmetry-mitigation-mode",
        "verify_only",
        "--method",
        "SPSA",
        "--restarts",
        str(int(restarts)),
        "--maxiter",
        str(int(maxiter)),
        "--progress-every-s",
        "30",
        "--energy-backend",
        "one_apply_compiled",
        "--spsa-a",
        "0.1",
        "--spsa-c",
        "0.02",
        "--spsa-alpha",
        "0.602",
        "--spsa-gamma",
        "0.101",
        "--spsa-A",
        "5.0",
        "--spsa-avg-last",
        "0",
        "--spsa-eval-repeats",
        "1",
        "--spsa-eval-agg",
        "mean",
    ]


def _build_compile_command(result_json: Path, compile_json: Path, backend_name: str, optimization_level: int, seed_transpiler: int) -> list[str]:
    return [
        "python",
        "-u",
        str(COMPILE_SCOUT_ENTRYPOINT),
        "--artifact-json",
        str(result_json),
        "--backend-name",
        str(backend_name),
        "--optimization-level",
        str(int(optimization_level)),
        "--seed-transpiler",
        str(int(seed_transpiler)),
        "--output-json",
        str(compile_json),
    ]


def _run_logged_command(cmd: Sequence[str], stdout_path: Path, stderr_path: Path) -> tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        proc = subprocess.run(
            list(cmd),
            cwd=str(REPO_ROOT),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    return int(proc.returncode), float(time.perf_counter() - t0)


def _run_case(
    case: CaseSpec,
    output_root: Path,
    epsilon_abs_delta_e: float,
    restarts: int,
    maxiter: int,
    compile_backend: str,
    compile_opt_level: int,
    compile_seed: int,
) -> CaseOutcome:
    case_dir = output_root / case.lane / case.trial_name
    logs_dir = case_dir / "logs"
    json_dir = case_dir / "json"
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    result_json = json_dir / "result.json"
    output_csv = json_dir / "result.csv"
    output_md = json_dir / "result.md"
    output_log = logs_dir / "replay.log"
    compile_json = json_dir / "compile_scout_fake_marrakesh.json"

    replay_cmd = _build_replay_command(case, result_json, output_csv, output_md, output_log, restarts, maxiter)
    _write_shell_command(logs_dir / "command.sh", replay_cmd)
    replay_returncode, replay_elapsed_s = _run_logged_command(
        replay_cmd,
        logs_dir / "stdout.log",
        logs_dir / "stderr.log",
    )

    compile_returncode: int | None = None
    compile_elapsed_s: float | None = None
    if replay_returncode == 0 and result_json.exists():
        compile_cmd = _build_compile_command(
            result_json=result_json,
            compile_json=compile_json,
            backend_name=compile_backend,
            optimization_level=compile_opt_level,
            seed_transpiler=compile_seed,
        )
        _write_shell_command(logs_dir / "compile_command.sh", compile_cmd)
        compile_returncode, compile_elapsed_s = _run_logged_command(
            compile_cmd,
            logs_dir / "compile_stdout.log",
            logs_dir / "compile_stderr.log",
        )

    invalid_reasons: list[str] = []
    result_payload = _read_json(result_json) if result_json.exists() else {}
    compile_payload = _read_json(compile_json) if compile_json.exists() else {}
    delta_abs = _result_abs_delta(result_payload)
    logical_count = _logical_operator_count(result_payload)
    runtime_count = _runtime_parameter_count(result_payload)
    best_compile = _best_compile_row(compile_payload) if compile_json.exists() else None
    compiled_count_2q = int(best_compile.get("compiled_count_2q")) if isinstance(best_compile, dict) and best_compile.get("compiled_count_2q") is not None else None
    compiled_depth = int(best_compile.get("compiled_depth")) if isinstance(best_compile, dict) and best_compile.get("compiled_depth") is not None else None

    if replay_returncode != 0 or not result_json.exists():
        invalid_reasons.append("pipeline_failed")
    if delta_abs is None:
        invalid_reasons.append("missing_delta")
    elif float(delta_abs) > float(epsilon_abs_delta_e):
        invalid_reasons.append("energy_band_failed")
    if compile_returncode is None or compile_returncode != 0 or compiled_count_2q is None or compiled_depth is None:
        invalid_reasons.append("compile_failed")

    feasible = len(invalid_reasons) == 0
    constraints = [
        float((delta_abs if delta_abs is not None else 1.0) - float(epsilon_abs_delta_e)),
        0.0 if (compile_returncode == 0 and compiled_count_2q is not None and compiled_depth is not None) else 1.0,
    ]
    total_elapsed_s = float(replay_elapsed_s + (compile_elapsed_s or 0.0))
    params = {
        "base_preset": str(case.seed.name),
        "selector_geometry_mode": "powell_handoff_phase3_v1",
        "runtime_split_mode": "phase3_v1_replay",
        "repeats_mode": "inherited",
        "phase1_prune_mode": "inherited",
        "replay_seed_policy": str(case.replay_seed_policy),
        "seed_handoff_state_kind": str(case.seed.handoff_state_kind),
    }

    return CaseOutcome(
        lane=str(case.lane),
        case_name=str(case.case_name),
        case_dir=str(case_dir.relative_to(REPO_ROOT)),
        params=params,
        abs_delta_e=None if delta_abs is None else float(delta_abs),
        compiled_count_2q=compiled_count_2q,
        compiled_depth=compiled_depth,
        logical_operator_count=logical_count,
        runtime_parameter_count=runtime_count,
        feasible=bool(feasible),
        constraints=[float(x) for x in constraints],
        result_json=str(result_json),
        compile_json=str(compile_json),
        returncode=int(replay_returncode),
        compile_returncode=compile_returncode,
        pipeline_elapsed_s=float(replay_elapsed_s),
        compile_elapsed_s=None if compile_elapsed_s is None else float(compile_elapsed_s),
        total_elapsed_s=float(total_elapsed_s),
        invalid_reasons=list(invalid_reasons),
        seed_name=str(case.seed.name),
        seed_notes=str(case.seed.notes),
    )


def _lane_payload(lane: str, epsilon_abs_delta_e: float, outcomes: Sequence[CaseOutcome]) -> dict[str, Any]:
    lane_obs = [outcome for outcome in outcomes if outcome.lane == lane]
    return {
        "study_name": f"{lane}_powell_handoff_eps_{epsilon_abs_delta_e:.3e}",
        "lane": str(lane),
        "epsilon_abs_delta_e": float(epsilon_abs_delta_e),
        "completed_trial_count": int(len(lane_obs)),
        "feasible_count": int(sum(1 for obs in lane_obs if obs.feasible)),
        "observations": [asdict(obs) for obs in lane_obs],
    }


def _best_feasible_by_lane(outcomes: Sequence[CaseOutcome]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for lane in ("current", "legacy"):
        feasible = [
            obs for obs in outcomes
            if obs.lane == lane and obs.feasible and obs.compiled_count_2q is not None and obs.compiled_depth is not None and obs.abs_delta_e is not None
        ]
        if not feasible:
            continue
        best = min(
            feasible,
            key=lambda obs: (
                int(obs.compiled_count_2q),
                int(obs.compiled_depth),
                float(obs.abs_delta_e),
            ),
        )
        out[lane] = asdict(best)
    return out


def _write_progress(output_root: Path, outcomes: Sequence[CaseOutcome], total_cases: int, waiting_done: bool) -> None:
    payload = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_spsa_powell_handoff_refinement_v1",
        "waiting_done": bool(waiting_done),
        "completed_trial_count": int(len(outcomes)),
        "total_trial_count": int(total_cases),
        "lane_completed_counts": {
            "current": int(sum(1 for obs in outcomes if obs.lane == "current")),
            "legacy": int(sum(1 for obs in outcomes if obs.lane == "legacy")),
        },
        "lane_feasible_counts": {
            "current": int(sum(1 for obs in outcomes if obs.lane == "current" and obs.feasible)),
            "legacy": int(sum(1 for obs in outcomes if obs.lane == "legacy" and obs.feasible)),
        },
    }
    _write_json(output_root / "progress.json", payload)


def _write_summary(output_root: Path, trigger_summary_path: Path, trigger_payload: dict[str, Any], epsilon_abs_delta_e: float, outcomes: Sequence[CaseOutcome]) -> None:
    payload = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_spsa_powell_handoff_refinement_v1",
        "tag": str(output_root.name),
        "output_dir": str(output_root),
        "trigger_summary_path": str(trigger_summary_path),
        "trigger_has_feasible_broad_spsa": bool(_study_has_feasible_points(trigger_payload)),
        "epsilon_abs_delta_e": float(epsilon_abs_delta_e),
        "studies": [
            _lane_payload("current", epsilon_abs_delta_e, outcomes),
            _lane_payload("legacy", epsilon_abs_delta_e, outcomes),
        ],
        "best_feasible_by_lane": _best_feasible_by_lane(outcomes),
    }
    _write_json(output_root / "summary.json", payload)
    _write_json(
        output_root / "current_summary.json",
        {
            **payload,
            "studies": [_lane_payload("current", epsilon_abs_delta_e, outcomes)],
        },
    )


def _refresh_math(output_root: Path) -> int:
    current_summaries = ",".join(
        [
            str(DEFAULT_CURRENT_SUMMARY),
            str(output_root / "current_summary.json"),
        ]
    )
    legacy_trial_roots = ",".join(
        [
            str(DEFAULT_LEGACY_TRIAL_ROOT),
            str(output_root / "legacy"),
        ]
    )
    legacy_summaries = str(DEFAULT_LEGACY_SUMMARY)
    cmd = [
        "python",
        str(MATH_MONITOR_ENTRYPOINT),
        "--once",
        "--current-summaries",
        current_summaries,
        "--legacy-trial-roots",
        legacy_trial_roots,
        "--legacy-summaries",
        legacy_summaries,
        "--state-json",
        str(DEFAULT_MONITOR_STATE_JSON),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return int(proc.returncode)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wait for a broad SPSA HH study to finish, then launch Powell-manifold SPSA replay refinement if broad SPSA stayed infeasible.")
    p.add_argument("--wait-summary", type=Path, default=DEFAULT_WAIT_SUMMARY)
    p.add_argument("--wait-progress", type=Path, default=DEFAULT_WAIT_PROGRESS)
    p.add_argument("--output-tag", type=str, default=DEFAULT_OUTPUT_TAG)
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    p.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    p.add_argument("--max-polls", type=int, default=DEFAULT_MAX_POLLS, help="0 means wait indefinitely.")
    p.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS)
    p.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    p.add_argument("--compile-backend", type=str, default=DEFAULT_COMPILE_BACKEND)
    p.add_argument("--compile-opt-level", type=int, default=DEFAULT_COMPILE_OPT_LEVEL)
    p.add_argument("--compile-seed", type=int, default=DEFAULT_COMPILE_SEED)
    p.add_argument("--seed-names", type=str, default=",".join(DEFAULT_SEED_ORDER))
    p.add_argument("--policies", type=str, default=",".join(DEFAULT_POLICIES))
    p.add_argument("--refresh-math", action="store_true")
    p.add_argument("--no-refresh-math", dest="refresh_math", action="store_false")
    p.set_defaults(refresh_math=True)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = REPO_ROOT / "artifacts" / "agent_runs" / str(args.output_tag)
    output_root.mkdir(parents=True, exist_ok=True)

    seed_names = [item.strip() for item in str(args.seed_names).split(",") if item.strip()]
    policies = [item.strip() for item in str(args.policies).split(",") if item.strip()]
    missing = [name for name in seed_names if name not in SEED_LIBRARY]
    if missing:
        raise SystemExit(f"Unknown seed names: {missing}")

    _write_progress(output_root, outcomes=[], total_cases=len(_build_cases(seed_names, policies)), waiting_done=False)
    trigger_payload = _wait_for_summary(
        summary_path=Path(args.wait_summary),
        progress_path=Path(args.wait_progress) if args.wait_progress else None,
        poll_seconds=int(args.poll_seconds),
        max_polls=int(args.max_polls),
    )
    decision_payload = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_spsa_powell_handoff_refinement_v1",
        "trigger_summary_path": str(Path(args.wait_summary)),
        "trigger_has_feasible_broad_spsa": bool(_study_has_feasible_points(trigger_payload)),
        "action": "skip" if _study_has_feasible_points(trigger_payload) else "launch",
    }
    _write_json(output_root / "decision.json", decision_payload)

    if _study_has_feasible_points(trigger_payload):
        _write_summary(
            output_root=output_root,
            trigger_summary_path=Path(args.wait_summary),
            trigger_payload=trigger_payload,
            epsilon_abs_delta_e=float(args.epsilon),
            outcomes=[],
        )
        _write_progress(output_root, outcomes=[], total_cases=0, waiting_done=True)
        if bool(args.refresh_math):
            _refresh_math(output_root)
        return 0

    cases = _build_cases(seed_names, policies)
    outcomes: list[CaseOutcome] = []
    for case in cases:
        outcome = _run_case(
            case=case,
            output_root=output_root,
            epsilon_abs_delta_e=float(args.epsilon),
            restarts=int(args.restarts),
            maxiter=int(args.maxiter),
            compile_backend=str(args.compile_backend),
            compile_opt_level=int(args.compile_opt_level),
            compile_seed=int(args.compile_seed),
        )
        outcomes.append(outcome)
        _write_summary(
            output_root=output_root,
            trigger_summary_path=Path(args.wait_summary),
            trigger_payload=trigger_payload,
            epsilon_abs_delta_e=float(args.epsilon),
            outcomes=outcomes,
        )
        _write_progress(output_root, outcomes=outcomes, total_cases=len(cases), waiting_done=True)
        if bool(args.refresh_math):
            _refresh_math(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
