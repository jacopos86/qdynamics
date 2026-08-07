#!/usr/bin/env python3
"""Generate historical warm-start ledger for Route-A Phase0 nph=1 Optuna records."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optuna  # noqa: E402

import pipelines.static_adapt.optimization.phase3_policy_optuna as p3opt  # noqa: E402
from pipelines.exact_bench.table_i_canonical_cases import (  # noqa: E402
    TABLE_I_STANDARD_PROFILE,
    table_i_executable_specs,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_ID = "routeA_phase0_nph1_oracle_v1"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "input" / BATCH_ID
WARM_START_LEDGER = "phase0_nph1_best_warm_start_ledger.json"
WARM_START_LEDGER_REPO_PATH = f"chtc/phase3_optuna/input/{BATCH_ID}/{WARM_START_LEDGER}"

# These are intentionally exact-source rows only. Do not add synthetic
# Hamiltonian-family baselines here unless the artifact was actually run at that
# physics point; apply_historical_ledger_to_specs uses these rows as baselines.
_SOURCE_ROWS: tuple[dict[str, Any], ...] = (
    {
        "benchmark_id": "hubbard_L2",
        "family": "hubbard",
        "summary_json": "raw_outputs/chtc_phase3_optuna/tripartite_spsa_A_current_collective_fermionic_smallrobust_target5e5_v1/summary.json",
        "trial_number": 31,
        "artifact_json": "raw_outputs/chtc_phase3_optuna/tripartite_spsa_A_current_collective_fermionic_smallrobust_target5e5_v1/run/trial_0031/hubbard_L2/json/result.json",
        "notes": "fermionic Route-A SPSA canonical source; same-family u6 variants may reuse params but not baseline",
    },
    {
        "benchmark_id": "bose_hubbard_L2",
        "family": "bose_hubbard",
        "summary_json": "raw_outputs/chtc_phase3_optuna/routeA_spsa_bosonic_L1L2_nph1_bh_hk_spinboson_warm_smallrobust_target5e5_v1/summary.json",
        "trial_number": 31,
        "artifact_json": "raw_outputs/chtc_phase3_optuna/routeA_spsa_bosonic_L1L2_nph1_bh_hk_spinboson_warm_smallrobust_target5e5_v1/run/trial_0031/bose_hubbard_L2/json/result.json",
        "notes": "bosonic Route-A SPSA canonical source",
    },
    {
        "benchmark_id": "harmonic_kerr_chain_L2",
        "family": "harmonic_kerr_chain",
        "summary_json": "artifacts/agent_runs/20260513_hk_l2_nph1_accuracy_ceiling_optuna/run/summary.json",
        "trial_number": 1,
        "artifact_json": "artifacts/agent_runs/20260513_hk_l2_nph1_accuracy_ceiling_optuna/run/trial_0001/result.json",
        "notes": "harmonic Kerr nph=1 accuracy ceiling Route-A source",
    },
    {
        "benchmark_id": "spin_boson_L1",
        "family": "spin_boson",
        "summary_json": "raw_outputs/chtc_phase3_optuna/routeA_spsa_bosonic_L1L2_nph1_bh_hk_spinboson_warm_smallrobust_target5e5_v1/summary.json",
        "trial_number": 0,
        "artifact_json": "raw_outputs/chtc_phase3_optuna/routeA_spsa_bosonic_L1L2_nph1_bh_hk_spinboson_warm_smallrobust_target5e5_v1/run/trial_0000/spin_boson_L1/json/result.json",
        "notes": "spin-boson bosonic-lane source",
    },
    {
        "benchmark_id": "hh_L2",
        "family": "hh",
        "summary_json": "artifacts/agent_runs/20260508_hh_l2_nph1_spsa_algebraic_warmstart_v1/hh_L2/summary.json",
        "trial_number": 13,
        "artifact_json": "artifacts/agent_runs/20260508_hh_l2_nph1_spsa_algebraic_warmstart_v1/hh_L2/trial_0013/hh_L2/json/result.json",
        "notes": "strict Route-A SPSA HH source; excludes legacy weak/fluke HH records",
    },
)


def table_i_phase0_nph1_specs() -> tuple[p3opt.HamiltonianBenchmarkSpec, ...]:
    specs = table_i_executable_specs(TABLE_I_STANDARD_PROFILE)
    if not specs:
        raise ValueError("Table-I standard executable spec set is empty")
    return tuple(specs)


def _load_json(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads((REPO_ROOT / Path(path)).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _path_get(root: Any, path: str) -> Any:
    node = root
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _adapt_payload(result: Mapping[str, Any]) -> Mapping[str, Any]:
    adapt = result.get("adapt_vqe")
    return adapt if isinstance(adapt, Mapping) else result


def _result_metric(result: Mapping[str, Any], *paths: str) -> Any:
    adapt = _adapt_payload(result)
    for path in paths:
        value = _path_get(result, path)
        if value is not None:
            return value
        if path.startswith("adapt_vqe."):
            value = _path_get(adapt, path[len("adapt_vqe.") :])
            if value is not None:
                return value
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def _summary_trial(summary: Mapping[str, Any], trial_number: int) -> Mapping[str, Any]:
    trials = summary.get("trials")
    if not isinstance(trials, Sequence) or isinstance(trials, (str, bytes)):
        raise ValueError("summary has no trials array")
    for trial in trials:
        if isinstance(trial, Mapping) and int(trial.get("number", -1)) == int(trial_number):
            return trial
    raise ValueError(f"trial {trial_number} not found in summary")


def _trial_params(summary: Mapping[str, Any], trial_number: int) -> dict[str, Any]:
    try:
        trial = _summary_trial(summary, trial_number)
    except ValueError:
        if "best_params" in summary and isinstance(summary["best_params"], Mapping):
            return dict(summary["best_params"])
        raise
    params = trial.get("params")
    if isinstance(params, Mapping) and params:
        return dict(params)
    if "best_params" in summary and isinstance(summary["best_params"], Mapping):
        return dict(summary["best_params"])
    raise ValueError(f"trial {trial_number} has no params")


def _emitted_cli_args(summary: Mapping[str, Any], trial_number: int) -> list[str] | None:
    try:
        trial = _summary_trial(summary, trial_number)
    except ValueError:
        return None
    user_attrs = trial.get("user_attrs")
    audit = user_attrs.get("policy_roundtrip_audit") if isinstance(user_attrs, Mapping) else None
    if isinstance(audit, Mapping):
        args = audit.get("emitted_cli_args")
        if isinstance(args, Sequence) and not isinstance(args, (str, bytes)) and args:
            return [str(arg) for arg in args]
    return None


def _command_from_params(spec: p3opt.HamiltonianBenchmarkSpec, params: Mapping[str, Any]) -> str:
    clean_params = p3opt._sanitize_trial_params({**p3opt.default_trial_params(), **dict(params)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        policy = p3opt.sample_policy_from_trial(optuna.trial.FixedTrial(clean_params), policy_search_profile="default")
    args = p3opt.apply_policy_to_pipeline_args(spec.base_pipeline_args, policy, spec)
    return "python -m pipelines.static_adapt.adapt_pipeline " + shlex.join(str(arg) for arg in args)


def _setting_args(settings: Mapping[str, Any]) -> list[str]:
    args: list[str] = []
    toggles = {
        "adapt_allow_repeats": ("--adapt-allow-repeats", "--adapt-no-repeats"),
        "phase2_enable_batching": ("--phase2-enable-batching", "--phase2-no-batching"),
        "phase1_prune_enabled": ("--phase1-prune-enabled", "--phase1-no-prune"),
        "phase1_prune_amplitude_witness_required": (
            "--phase1-prune-amplitude-witness-required",
            "--phase1-prune-amplitude-witness-optional",
        ),
        "phase0_pilot_enabled": ("--phase0-pilot-enabled", "--phase0-no-pilot"),
        "phase_live_hysteresis_enabled": ("--phase-live-hysteresis-enabled", "--phase-live-hysteresis-disabled"),
    }
    for key, value in sorted(settings.items()):
        if value is None or isinstance(value, (Mapping, list, tuple)):
            continue
        if key in toggles and isinstance(value, bool):
            args.append(toggles[key][0] if value else toggles[key][1])
            continue
        if isinstance(value, bool):
            if value:
                args.append("--" + key.replace("_", "-"))
            continue
        args.extend(["--" + key.replace("_", "-"), str(value)])
    return args


def _command_from_result_settings(spec: p3opt.HamiltonianBenchmarkSpec, result: Mapping[str, Any]) -> str | None:
    settings = result.get("settings")
    if not isinstance(settings, Mapping) or not settings:
        return None
    args = [str(arg) for arg in spec.base_pipeline_args]
    args.extend(_setting_args(settings))
    return "python -m pipelines.static_adapt.adapt_pipeline " + shlex.join(str(arg) for arg in args)


def _command_for_source(
    spec: p3opt.HamiltonianBenchmarkSpec,
    summary: Mapping[str, Any],
    result: Mapping[str, Any],
    trial_number: int,
) -> str:
    emitted = _emitted_cli_args(summary, trial_number)
    if emitted:
        return "python -m pipelines.static_adapt.adapt_pipeline " + shlex.join(str(arg) for arg in emitted)
    from_settings = _command_from_result_settings(spec, result)
    if from_settings:
        return from_settings
    return _command_from_params(spec, _trial_params(summary, trial_number))


def _ledger_row(source: Mapping[str, Any], specs_by_id: Mapping[str, p3opt.HamiltonianBenchmarkSpec]) -> dict[str, Any]:
    benchmark_id = str(source["benchmark_id"])
    spec = specs_by_id[benchmark_id]
    summary = _load_json(str(source["summary_json"]))
    result = _load_json(str(source["artifact_json"]))
    trial_number = int(source["trial_number"])
    command = _command_for_source(spec, summary, result, trial_number)
    parsed_params = p3opt.trial_params_from_cli_command(command)
    summary_params = _trial_params(summary, trial_number)
    trial_params = p3opt._sanitize_trial_params({**parsed_params, **dict(summary_params)})
    params, audit = p3opt._nondefault_trial_params_with_audit(
        trial_params,
        source="generated_ledger.trial_params",
        has_command=True,
    )
    if params is None:
        raise ValueError(f"{benchmark_id}: historical command does not parse as non-default warm start: {audit}")
    abs_delta_e = _float_or_none(
        _result_metric(
            result,
            "adapt_vqe.abs_delta_e",
            "adapt_vqe.exact_abs_delta_e_from_final_state",
            "abs_delta_e",
        )
    )
    if abs_delta_e is None:
        raise ValueError(f"{benchmark_id}: could not extract abs_delta_e from {source['artifact_json']}")
    selected_backend = _result_metric(result, "adapt_vqe.backend_compile_cost_summary.selected_backend")
    if not isinstance(selected_backend, Mapping):
        selected_backend = {}
    count_2q = _int_or_none(selected_backend.get("compiled_count_2q"))
    depth_2q = _int_or_none(selected_backend.get("compiled_depth_2q"))
    circuit_depth = _int_or_none(selected_backend.get("compiled_depth"))
    adapt = _adapt_payload(result)
    return {
        "schema": "phase3_historical_warm_start_row_v1",
        "record_id": f"{benchmark_id}:routeA_phase0_nph1_warm_start",
        "benchmark_id": benchmark_id,
        "problem": spec.family,
        "family": spec.family,
        "L": int(spec.features.L),
        "n_ph_max": int(p3opt._pipeline_arg_value(spec.base_pipeline_args, "--n-ph-max") or 1),
        "pool": "full_meta",
        "continuation": "phase3_v1",
        "inner_optimizer": "SPSA",
        "warm_start_eligible": True,
        "artifact_json": str(source["artifact_json"]),
        "source_summary_json": str(source["summary_json"]),
        "source_trial_number": trial_number,
        "source_notes": str(source.get("notes") or ""),
        "abs_delta_e": abs_delta_e,
        "count_2q": count_2q,
        "depth_2q": depth_2q,
        "circuit_depth": circuit_depth,
        "ansatz_depth": _int_or_none(adapt.get("ansatz_depth")),
        "num_parameters": _int_or_none(adapt.get("num_parameters")),
        "trial_params": trial_params,
        "command": command,
        "command_parse_audit": audit,
    }


def build_ledger() -> dict[str, Any]:
    specs = table_i_phase0_nph1_specs()
    specs_by_id = {spec.benchmark_id: spec for spec in specs}
    missing = [source["benchmark_id"] for source in _SOURCE_ROWS if source["benchmark_id"] not in specs_by_id]
    if missing:
        raise ValueError(f"warm-start sources missing from Table-I specs: {missing}")
    rows = [_ledger_row(source, specs_by_id) for source in _SOURCE_ROWS]
    by_problem: dict[str, Any] = {}
    by_size_cutoff: dict[str, Any] = {}
    for row in rows:
        by_problem[str(row["family"])] = row
        by_size_cutoff[f"{row['family']}|L={row['L']}|nph={row['n_ph_max']}"] = row
    return {
        "schema": "routeA_phase0_nph1_best_warm_start_ledger_v1",
        "generated_by": "chtc/phase3_optuna/generate_routeA_phase0_nph1_warm_start_ledger.py",
        "suite_profile": TABLE_I_STANDARD_PROFILE,
        "static_route_id": "route_a",
        "working_n_ph_max": 1,
        "policy_search_profile": "default",
        "best_warm_start_by_problem": by_problem,
        "best_warm_start_by_problem_size_cutoff": by_size_cutoff,
        "source_rows": list(_SOURCE_ROWS),
    }


def render_artifact() -> str:
    return json.dumps(build_ledger(), indent=2, sort_keys=True) + "\n"


def write_artifact(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    path = Path(output_dir) / WARM_START_LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_artifact(), encoding="utf-8")
    return path


def check_artifact(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[str]:
    path = Path(output_dir) / WARM_START_LEDGER
    try:
        actual = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [f"missing generated artifact: {path}"]
    expected = render_artifact()
    if actual != expected:
        return [f"generated artifact is stale: {path}"]
    return []


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        errors = check_artifact(args.output_dir)
        if errors:
            for error in errors:
                print(error)
            return 1
        print("Route-A Phase0 nph1 warm-start ledger is current")
        return 0
    if args.write:
        path = write_artifact(args.output_dir)
        print(f"wrote Route-A Phase0 nph1 warm-start ledger: {path}")
        return 0
    print(render_artifact(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
