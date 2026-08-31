"""Paper-III P7b local pre-CHTC HH science-pilot harness.

This module is intentionally report/orchestration-only.  It may launch the
strict HH realtime command as a subprocess, then validates the resulting JSON by
plain data inspection.  It must not import realtime/controller, Qiskit, CHTC,
Optuna, or exact-benchmark modules.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.excited_dynamics.io import load_json, sha256_file, write_json


PAPER_III_LOCAL_SCIENCE_PILOT_REPORT_SCHEMA_VERSION = "paper_iii_local_science_pilot_report_v1"
PAPER_III_LOCAL_SCIENCE_PILOT_PIPELINE = "paper_iii_local_science_pilot"
MODULE_NAME = "pipelines.excited_dynamics.paper_iii_local_science_pilot"
DEFAULT_OUTPUT_DIR = Path("artifacts/agent_runs/paper_iii_p7b_local_science_pilot")
DEFAULT_HH_ARTIFACT_JSON = Path(
    "artifacts/json/adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_with_ansatz_input_20260321T214822Z.json"
)
DEFAULT_T_FINAL = 1.0
DEFAULT_NUM_TIMES = 9
DEFAULT_TIMEOUT_SECONDS = 1800
_TIME_TOLERANCE = 1.0e-9
_FORBIDDEN_RAW_MARKERS = (
    "amplitudes_qn_to_q0",
    "raw_physical_state",
    "basis_matrix_vectors",
    "exact_target_trajectories",
    "exact_step_forecast",
    "state_at(",
)
_NONFINITE_OR_NONJSON_MARKER = "nonfinite_or_nonserializable_json_value"
_ALLOWED_STRICT_DECISION_DATA_FLOWS = {"ideal_observable_estimator", "measurement_oracle"}


class PaperIIILocalSciencePilotError(ValueError):
    """Raised when the local P7b report cannot be emitted safely."""


@dataclass(frozen=True)
class PaperIIILocalSciencePilotConfig:
    artifact_json: Path = DEFAULT_HH_ARTIFACT_JSON
    output_dir: Path = DEFAULT_OUTPUT_DIR
    t_final: float = DEFAULT_T_FINAL
    num_times: int = DEFAULT_NUM_TIMES
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    run_tag: str = "paper_iii_p7b_hh_strict_realtime_pilot"
    report_only_existing_output: bool = False
    progress_json: Path | None = None
    partial_payload_json: Path | None = None
    existing_strict_output_json: Path | None = None

    @property
    def strict_output_json(self) -> Path:
        return Path(self.output_dir) / "hh_strict_realtime_pilot.json"

    @property
    def report_json(self) -> Path:
        return Path(self.output_dir) / "paper_iii_local_science_pilot_report.json"

    @property
    def report_md(self) -> Path:
        return Path(self.output_dir) / "paper_iii_local_science_pilot_report.md"

    @property
    def command_log_md(self) -> Path:
        return Path(self.output_dir) / "command_log.md"

    @property
    def run_manifest_json(self) -> Path:
        return Path(self.output_dir) / "run_manifest.json"

    @property
    def logs_dir(self) -> Path:
        return Path(self.output_dir) / "logs"

    @property
    def command_sh(self) -> Path:
        return self.logs_dir / "command.sh"

    @property
    def stdout_log(self) -> Path:
        return self.logs_dir / "stdout.log"

    @property
    def stderr_log(self) -> Path:
        return self.logs_dir / "stderr.log"


@dataclass(frozen=True)
class _CommandResult:
    status: str
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    error: str | None = None
    started_utc: str | None = None
    finished_utc: str | None = None
    wallclock_seconds: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "error": self.error,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "wallclock_seconds": self.wallclock_seconds,
        }


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PaperIIILocalSciencePilotError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PaperIIILocalSciencePilotError(f"{name} must be a sequence")
    return value


def _display_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _optional_display_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return _display_path(path)


def _optional_path_exists(path: str | Path | None) -> bool:
    return bool(path is not None and Path(path).exists())


def _shell_join(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _sha256_or_none(path: Path) -> str | None:
    return sha256_file(path) if Path(path).exists() else None


def build_strict_hh_command(config: PaperIIILocalSciencePilotConfig) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pipelines.time_dynamics.runners.hh_from_adapt_artifact",
        "--artifact-json",
        str(Path(config.artifact_json)),
        "--output-json",
        str(config.strict_output_json),
        "--run-tag",
        str(config.run_tag),
        "--t-final",
        str(float(config.t_final)),
        "--num-times",
        str(int(config.num_times)),
        "--checkpoint-controller-mode",
        "oracle_v1",
        "--checkpoint-controller-reference-mode",
        "off",
        "--checkpoint-controller-exact-input-mode",
        "off",
        "--diagnostic-exact-reference-mode",
        "benchmark_exact",
        "--checkpoint-controller-strict-qpu-faithful",
        "--checkpoint-controller-noise-mode",
        "ideal",
        "--compile-audit-mode",
        "off",
    ]
    if config.progress_json is not None:
        command.extend(["--progress-json", str(Path(config.progress_json))])
    if config.partial_payload_json is not None:
        command.extend(["--partial-payload-json", str(Path(config.partial_payload_json))])
    return command


def _strict_payload_source_json(config: PaperIIILocalSciencePilotConfig) -> Path:
    if config.existing_strict_output_json is not None and not config.report_only_existing_output:
        raise PaperIIILocalSciencePilotError(
            "existing_strict_output_json is only valid with report_only_existing_output=True"
        )
    if config.existing_strict_output_json is not None:
        return Path(config.existing_strict_output_json)
    return config.strict_output_json


def _write_command_logs(config: PaperIIILocalSciencePilotConfig, *, command: Sequence[str], result: _CommandResult) -> None:
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    config.command_sh.write_text(
        "#!/bin/sh\nset -eu\n" + _shell_join(command) + "\n",
        encoding="utf-8",
    )
    config.stdout_log.write_text(result.stdout, encoding="utf-8")
    config.stderr_log.write_text(result.stderr, encoding="utf-8")


def _decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _timed_command_result(
    *,
    status: str,
    exit_code: int | None,
    timed_out: bool,
    stdout: str,
    stderr: str,
    error: str | None,
    started_utc: str,
    started_perf: float,
) -> _CommandResult:
    return _CommandResult(
        status=status,
        exit_code=exit_code,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        error=error,
        started_utc=started_utc,
        finished_utc=_utc_now(),
        wallclock_seconds=max(0.0, float(time.perf_counter() - started_perf)),
    )


def _report_only_command_result(config: PaperIIILocalSciencePilotConfig) -> _CommandResult:
    started_utc = _utc_now()
    started_perf = time.perf_counter()
    source_json = _strict_payload_source_json(config)
    if source_json.exists():
        return _timed_command_result(
            status="existing_output",
            exit_code=None,
            timed_out=False,
            stdout="",
            stderr="",
            error=None,
            started_utc=started_utc,
            started_perf=started_perf,
        )
    return _timed_command_result(
        status="blocked_missing_existing_strict_output",
        exit_code=None,
        timed_out=False,
        stdout="",
        stderr=f"existing strict HH output missing: {source_json}\n",
        error=f"existing strict HH output missing: {source_json}",
        started_utc=started_utc,
        started_perf=started_perf,
    )


def _execute_strict_hh_command(config: PaperIIILocalSciencePilotConfig, *, command: Sequence[str]) -> _CommandResult:
    started_utc = _utc_now()
    started_perf = time.perf_counter()
    source = Path(config.artifact_json)
    if not source.exists():
        result = _timed_command_result(
            status="blocked_missing_source_artifact",
            exit_code=None,
            timed_out=False,
            stdout="",
            stderr=f"source artifact missing: {source}\n",
            error=f"source artifact missing: {source}",
            started_utc=started_utc,
            started_perf=started_perf,
        )
        _write_command_logs(config, command=command, result=result)
        return result

    if config.strict_output_json.exists():
        config.strict_output_json.unlink()
    for sidecar_path in (config.progress_json, config.partial_payload_json):
        if sidecar_path is not None:
            sidecar = Path(sidecar_path)
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            if sidecar.exists():
                sidecar.unlink()

    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=int(config.timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        result = _timed_command_result(
            status="timed_out",
            exit_code=None,
            timed_out=True,
            stdout=_decode_timeout_stream(exc.stdout),
            stderr=_decode_timeout_stream(exc.stderr) + f"\nTIMEOUT after {int(config.timeout_seconds)} seconds\n",
            error=f"timeout after {int(config.timeout_seconds)} seconds",
            started_utc=started_utc,
            started_perf=started_perf,
        )
        _write_command_logs(config, command=command, result=result)
        return result

    status = "completed" if completed.returncode == 0 else "failed_nonzero_exit"
    result = _timed_command_result(
        status=status,
        exit_code=int(completed.returncode),
        timed_out=False,
        stdout=str(completed.stdout or ""),
        stderr=str(completed.stderr or ""),
        error=None if completed.returncode == 0 else f"exit code {completed.returncode}",
        started_utc=started_utc,
        started_perf=started_perf,
    )
    _write_command_logs(config, command=command, result=result)
    return result


def _blocked_hh_stub(
    *,
    config: PaperIIILocalSciencePilotConfig,
    command: Sequence[str],
    blockers: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": "paper_iii_local_science_pilot_blocked_hh_run_v1",
        "status": "blocked",
        "controller_output_available": False,
        "paper_iii_science_benchmark": False,
        "artifact_json": _display_path(config.artifact_json),
        "output_json": _display_path(config.strict_output_json),
        "command": _shell_join(command),
        "t_final": float(config.t_final),
        "num_times": int(config.num_times),
        "blockers": list(blockers),
    }


def _find_true_key(value: Any, *, key: str, prefix: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            path = f"{prefix}.{raw_key}" if prefix else str(raw_key)
            if raw_key == key and raw_value is True:
                hits.append(path)
            hits.extend(_find_true_key(raw_value, key=key, prefix=path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            hits.extend(_find_true_key(item, key=key, prefix=path))
    return hits


def _find_key_values(value: Any, *, key: str, prefix: str = "") -> list[tuple[str, Any]]:
    hits: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            path = f"{prefix}.{raw_key}" if prefix else str(raw_key)
            if raw_key == key:
                hits.append((path, raw_value))
            hits.extend(_find_key_values(raw_value, key=key, prefix=path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            hits.extend(_find_key_values(item, key=key, prefix=path))
    return hits


def _forbidden_marker_hits(value: Any) -> list[str]:
    try:
        serialized = json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return [_NONFINITE_OR_NONJSON_MARKER]
    return [marker for marker in _FORBIDDEN_RAW_MARKERS if marker in serialized]


def _is_physical_trajectory_row(row: Mapping[str, Any]) -> bool:
    return row.get("trajectory_sample_kind", "state_sample") != "repair_event" and row.get("advances_time", True) is not False


def _trajectory_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = payload.get("trajectory", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [row for row in raw if isinstance(row, Mapping)]


def _ledger_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = payload.get("ledger", [])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [row for row in raw if isinstance(row, Mapping)]


def _finite_series(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = _finite_or_none(row.get(key))
        if value is not None:
            out.append(value)
    return out


def _last_finite(rows: Sequence[Mapping[str, Any]], *keys: str) -> float | None:
    for row in reversed(rows):
        for key in keys:
            value = _finite_or_none(row.get(key))
            if value is not None:
                return value
    return None


def _series_metric_block(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = _finite_series(rows, key)
    return {
        f"final_{key}": _first_not_none(_finite_or_none(summary.get(f"final_{key}")), values[-1] if values else None),
        f"mean_{key}": _first_not_none(
            _finite_or_none(summary.get(f"mean_{key}")),
            float(sum(values) / len(values)) if values else None,
        ),
        f"max_{key}": _first_not_none(_finite_or_none(summary.get(f"max_{key}")), max(values) if values else None),
        f"{key}_sample_count": int(len(values)),
    }


def _summarize_metrics(
    *,
    payload: Mapping[str, Any],
    expected_t_final: float,
    expected_num_times: int,
) -> dict[str, Any]:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), Mapping) else {}
    trajectory = _trajectory_rows(payload)
    ledger = _ledger_rows(payload)
    physical_rows = [row for row in trajectory if _is_physical_trajectory_row(row)]
    repair_event_count = int(len(trajectory) - len(physical_rows))
    final_time = _last_finite(physical_rows, "time")
    time_tol = max(_TIME_TOLERANCE, _TIME_TOLERANCE * abs(float(expected_t_final)))
    full_horizon_gate_passed = final_time is not None and final_time + time_tol >= float(expected_t_final)

    fidelity_values = _finite_series(physical_rows, "fidelity_exact")
    if fidelity_values:
        fidelity = {
            "fidelity_exact_status": "computed",
            "final_fidelity_exact": float(fidelity_values[-1]),
            "min_fidelity_exact": float(min(fidelity_values)),
            "one_minus_min_fidelity_exact": float(1.0 - min(fidelity_values)),
            "fidelity_exact_sample_count": int(len(fidelity_values)),
        }
    else:
        fidelity = {
            "fidelity_exact_status": "missing_not_computed",
            "final_fidelity_exact": None,
            "min_fidelity_exact": None,
            "one_minus_min_fidelity_exact": None,
            "fidelity_exact_sample_count": 0,
        }

    strict_route = {
        "mode": summary.get("mode"),
        "decision_path_kind": summary.get("decision_path_kind"),
        "decision_backend": summary.get("decision_backend"),
        "decision_data_flow": summary.get("decision_data_flow"),
        "controller_exact_input_mode": summary.get("controller_exact_input_mode"),
        "diagnostic_exact_reference_mode": summary.get("diagnostic_exact_reference_mode"),
        "strict_qpu_faithful": summary.get("strict_qpu_faithful"),
        "strict_qpu_hh": summary.get("strict_qpu_hh"),
        "strict_measurement_oracle_certified": summary.get("strict_measurement_oracle_certified"),
        "qpu_faithful_decisions_passed": summary.get("qpu_faithful_decisions_passed"),
        "strict_decision_contract_passed": summary.get("strict_decision_contract_passed"),
        "exact_decision_checkpoints": int(summary.get("exact_decision_checkpoints", 0) or 0),
        "uses_reference_for_decision": bool(summary.get("uses_reference_for_decision", False)),
        "uses_future_exact_forecast_for_decision": bool(summary.get("uses_future_exact_forecast_for_decision", False)),
        "uses_statevector_as_ideal_observable_estimator": bool(
            summary.get("uses_statevector_as_ideal_observable_estimator", False)
        ),
    }

    return {
        "row_counts": {
            "raw_trajectory_row_count": int(len(trajectory)),
            "trajectory_state_sample_count": int(len(physical_rows)),
            "repair_event_row_count": repair_event_count,
            "ledger_row_count": int(len(ledger)),
        },
        "horizon": {
            "expected_t_final": float(expected_t_final),
            "expected_num_times": int(expected_num_times),
            "final_physical_time": final_time,
            "physical_row_count": int(len(physical_rows)),
            "full_horizon_gate_passed": bool(full_horizon_gate_passed),
        },
        "strict_route": strict_route,
        "energy": {
            **_series_metric_block(summary, physical_rows, "abs_energy_total_error"),
            "final_energy_total_controller": _last_finite(physical_rows, "energy_total_controller", "energy_total"),
            "final_energy_total_exact": _last_finite(physical_rows, "energy_total_exact"),
            "final_abs_energy_total_error_to_ed_ground": _first_not_none(
                _finite_or_none(summary.get("final_abs_energy_total_error_to_ed_ground")),
                _last_finite(physical_rows, "abs_energy_total_error_to_ed_ground"),
            ),
        },
        "observables": {
            **_series_metric_block(summary, physical_rows, "abs_primary_density_error"),
            **_series_metric_block(summary, physical_rows, "site_occupations_abs_error_max"),
            **_series_metric_block(summary, physical_rows, "abs_staggered_error"),
            **_series_metric_block(summary, physical_rows, "abs_doublon_error"),
        },
        "fidelity": fidelity,
    }


def validate_strict_hh_payload(
    payload: Mapping[str, Any],
    *,
    expected_t_final: float,
    expected_num_times: int,
) -> dict[str, Any]:
    violations: list[str] = []
    if not isinstance(payload, Mapping):
        return {"passed": False, "violations": ["hh payload must be a JSON object"], "violation_count": 1}

    route_config = payload.get("route_config") if isinstance(payload.get("route_config"), Mapping) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    reference = payload.get("reference") if isinstance(payload.get("reference"), Mapping) else {}
    diagnostic_reference = (
        payload.get("diagnostic_reference") if isinstance(payload.get("diagnostic_reference"), Mapping) else {}
    )
    trajectory = _trajectory_rows(payload)
    ledger = _ledger_rows(payload)
    physical_rows = [row for row in trajectory if _is_physical_trajectory_row(row)]

    if payload.get("status") == "blocked":
        violations.append("hh strict route output is a blocked stub")
    if route_config.get("problem_family") != "hh":
        violations.append("route_config.problem_family must be 'hh'")
    if route_config.get("strict_qpu_faithful") is not True:
        violations.append("route_config.strict_qpu_faithful must be true")
    if route_config.get("strict_qpu_hh") is not True:
        violations.append("route_config.strict_qpu_hh must be true")
    if route_config.get("diagnostic_exact_reference_mode") != "benchmark_exact":
        violations.append("route_config.diagnostic_exact_reference_mode must be benchmark_exact")
    if route_config.get("strict_measurement_oracle_certified") is not True:
        violations.append("route_config.strict_measurement_oracle_certified must be true")
    if route_config.get("decision_data_flow") not in _ALLOWED_STRICT_DECISION_DATA_FLOWS:
        violations.append(
            "route_config.decision_data_flow must be ideal_observable_estimator or measurement_oracle"
        )

    if summary.get("strict_qpu_faithful") is not True:
        violations.append("summary.strict_qpu_faithful must be true")
    if summary.get("strict_qpu_hh") is not True:
        violations.append("summary.strict_qpu_hh must be true")
    if summary.get("diagnostic_exact_reference_mode") != "benchmark_exact":
        violations.append("summary.diagnostic_exact_reference_mode must be benchmark_exact")
    if summary.get("strict_measurement_oracle_certified") is not True:
        violations.append("summary.strict_measurement_oracle_certified must be true")
    if summary.get("decision_backend") != "oracle":
        violations.append("summary.decision_backend must be oracle")
    if summary.get("decision_data_flow") not in _ALLOWED_STRICT_DECISION_DATA_FLOWS:
        violations.append("summary.decision_data_flow must be ideal_observable_estimator or measurement_oracle")

    for context, record in (("route_config", route_config), ("summary", summary)):
        if record.get("controller_exact_input_mode") != "off":
            violations.append(f"{context}.controller_exact_input_mode must be off")
    for path, value in _find_key_values(payload, key="controller_exact_input_mode"):
        if value != "off":
            violations.append(f"{path} must be off")

    if summary.get("reference_enabled") is True:
        violations.append("summary.reference_enabled must be false")
    if summary.get("controller_reference_enabled") is True:
        violations.append("summary.controller_reference_enabled must be false")
    if reference.get("reference_enabled") is True:
        violations.append("reference.reference_enabled must be false")

    for key in (
        "uses_reference_for_decision",
        "uses_future_exact_forecast_for_decision",
        "feeds_controller_decisions",
    ):
        hits = _find_true_key(payload, key=key)
        if hits:
            violations.append(f"{key}=true at {hits}")

    exact_checkpoint_values = _find_key_values(payload, key="exact_decision_checkpoints")
    for path, value in exact_checkpoint_values:
        try:
            count = int(value or 0)
        except Exception:
            violations.append(f"{path} must be an integer")
            continue
        if count > 0:
            violations.append(f"{path}={count}")

    if summary.get("qpu_faithful_decisions_passed") is not True:
        violations.append("summary.qpu_faithful_decisions_passed must be true")
    if summary.get("strict_decision_contract_passed") is False:
        violations.append("summary.strict_decision_contract_passed must not be false")
    if summary.get("strict_fail_closed") is True:
        violations.append("summary.strict_fail_closed must be false")
    if summary.get("status") == "strict_fail_closed":
        violations.append("summary.status must not be strict_fail_closed")

    for row_group_name, rows in (("trajectory", trajectory), ("ledger", ledger)):
        for idx, row in enumerate(rows):
            if row.get("decision_backend") == "exact":
                violations.append(f"{row_group_name}[{idx}].decision_backend must not be exact")
            if "decision_data_flow" in row and row.get("decision_data_flow") not in _ALLOWED_STRICT_DECISION_DATA_FLOWS:
                violations.append(
                    f"{row_group_name}[{idx}].decision_data_flow must be ideal_observable_estimator or measurement_oracle"
                )
    for path, value in _find_key_values(payload, key="decision_backend"):
        if value == "exact":
            violations.append(f"{path}.decision_backend must not be exact")
    for path, value in _find_key_values(payload, key="decision_data_flow"):
        if value not in _ALLOWED_STRICT_DECISION_DATA_FLOWS:
            violations.append(f"{path}.decision_data_flow must be ideal_observable_estimator or measurement_oracle")

    if diagnostic_reference:
        for key in (
            "feeds_controller_decisions",
            "uses_reference_for_decision",
            "uses_future_exact_forecast_for_decision",
        ):
            if diagnostic_reference.get(key) is True:
                violations.append(f"diagnostic_reference.{key} must be false")

    if not physical_rows:
        violations.append("trajectory must contain physical state-sample rows")
    final_time = _last_finite(physical_rows, "time")
    time_tol = max(_TIME_TOLERANCE, _TIME_TOLERANCE * abs(float(expected_t_final)))
    if final_time is None:
        violations.append("final physical trajectory time is missing")
    elif final_time + time_tol < float(expected_t_final):
        violations.append(f"final physical trajectory time {final_time} is short of t_final {float(expected_t_final)}")
    if len(physical_rows) < int(expected_num_times):
        violations.append(f"physical trajectory row count {len(physical_rows)} is less than num_times {int(expected_num_times)}")

    marker_hits = _forbidden_marker_hits(payload)
    if marker_hits:
        if _NONFINITE_OR_NONJSON_MARKER in marker_hits:
            violations.append("hh output contains non-finite or non-serializable JSON values")
        else:
            violations.append(f"hh output contains {len(marker_hits)} forbidden raw/reference payload marker(s)")

    deduped = list(dict.fromkeys(violations))
    return {
        "passed": len(deduped) == 0,
        "violations": deduped,
        "violation_count": int(len(deduped)),
        "physical_row_count": int(len(physical_rows)),
        "final_physical_time": final_time,
    }


def _assert_no_raw_payload(value: Any, *, context: str) -> None:
    hits = _forbidden_marker_hits(value)
    if hits:
        raise PaperIIILocalSciencePilotError(f"{context} would emit forbidden raw/reference payload markers: {hits}")


def _build_report(
    *,
    config: PaperIIILocalSciencePilotConfig,
    command: Sequence[str],
    command_result: _CommandResult,
    strict_payload_source_json: Path,
    hh_payload: Mapping[str, Any],
    generated_utc: str,
) -> dict[str, Any]:
    validation = validate_strict_hh_payload(
        hh_payload,
        expected_t_final=float(config.t_final),
        expected_num_times=int(config.num_times),
    )
    metrics = _summarize_metrics(
        payload=hh_payload,
        expected_t_final=float(config.t_final),
        expected_num_times=int(config.num_times),
    )
    command_completed = command_result.status in {"completed", "existing_output"} and command_result.timed_out is False
    command_blockers: list[str] = []
    if command_result.status == "blocked_missing_source_artifact":
        command_blockers.append("latest_phase3_source_artifact_missing_locally")
    elif command_result.status == "blocked_missing_existing_strict_output":
        command_blockers.append("existing_strict_output_missing")
    elif command_result.status == "timed_out":
        command_blockers.append(f"strict_hh_command_timeout_after_{int(config.timeout_seconds)}s")
    elif command_result.status == "failed_nonzero_exit":
        command_blockers.append(f"strict_hh_command_exit_code_{command_result.exit_code}")
    elif command_result.status not in {"completed", "existing_output"}:
        command_blockers.append(f"strict_hh_command_status_{command_result.status}")

    raw_payload_blockers = hh_payload.get("blockers", [])
    payload_blockers = (
        [str(blocker) for blocker in raw_payload_blockers]
        if isinstance(raw_payload_blockers, Sequence)
        and not isinstance(raw_payload_blockers, (str, bytes, bytearray))
        else []
    )
    validation_blockers = [f"strict_validation:{violation}" for violation in validation["violations"]]
    blockers = [*command_blockers, *payload_blockers, *validation_blockers]
    success = command_completed and bool(validation["passed"])

    report: dict[str, Any] = {
        "schema_version": PAPER_III_LOCAL_SCIENCE_PILOT_REPORT_SCHEMA_VERSION,
        "pipeline": PAPER_III_LOCAL_SCIENCE_PILOT_PIPELINE,
        "generated_utc": generated_utc,
        "report_kind": "paper_iii_p7b_local_pre_chtc_science_pilot",
        "controller_usable": False,
        "feeds_controller_decisions": False,
        "reference_comparisons_feed_controller_decisions": False,
        "paper_iii_science_benchmark": bool(success),
        "production_claim": False,
        "controller_boundary": {
            "controller_usable": False,
            "feeds_controller_decisions": False,
            "reference_comparisons_feed_controller_decisions": False,
            "report_only": True,
            "decision_path_allowed": False,
            "controller_exact_input_mode": "off",
            "diagnostic_exact_reference_mode": "benchmark_exact_post_run_only",
        },
        "case": {
            "case_id": "paper_iii_p7b_hh_L2_strict_local_pilot",
            "family": "hubbard_holstein",
            "route_family_key": "hh",
            "local_pilot": True,
            "pre_chtc": True,
            "downscaled_local_pilot": True,
            "chtc_used": False,
            "optuna_used": False,
            "fresh_adapt_run": False,
            "t_final": float(config.t_final),
            "num_times": int(config.num_times),
        },
        "runs": {
            "strict_hh_runtime_dynamics": {
                "status": "completed" if success else "blocked",
                "command_status": command_result.status,
                "command_completed": bool(command_completed),
                "paper_iii_science_benchmark": bool(success),
                "source_artifact_json": _display_path(config.artifact_json),
                "source_artifact_sha256": _sha256_or_none(Path(config.artifact_json)),
                "output_json": _display_path(strict_payload_source_json),
                "output_sha256": _sha256_or_none(strict_payload_source_json),
                "nominal_output_json": _display_path(config.strict_output_json),
                "strict_payload_source_json": _display_path(strict_payload_source_json),
                "strict_payload_source_sha256": _sha256_or_none(strict_payload_source_json),
                "report_only_existing_output": bool(config.report_only_existing_output),
                "existing_strict_output_json": _optional_display_path(config.existing_strict_output_json),
                "progress_json": _optional_display_path(config.progress_json),
                "progress_json_exists": _optional_path_exists(config.progress_json),
                "partial_payload_json": _optional_display_path(config.partial_payload_json),
                "partial_payload_json_exists": _optional_path_exists(config.partial_payload_json),
                "command_started_utc": command_result.started_utc,
                "command_finished_utc": command_result.finished_utc,
                "command_wallclock_seconds": command_result.wallclock_seconds,
                "command": _shell_join(command),
                "command_argv": [str(part) for part in command],
                "command_result": command_result.to_json(),
                "exit_code": command_result.exit_code,
                "timed_out": command_result.timed_out,
                "strict_validation": validation,
                "metrics": metrics,
            }
        },
        "post_run_comparisons": {
            "post_run_only": True,
            "feeds_controller_decisions": False,
            "reference_comparisons_feed_controller_decisions": False,
            "energy": metrics["energy"],
            "observables": metrics["observables"],
            "fidelity": metrics["fidelity"],
        },
        "evidence_classification": {
            "local_science_pilot": bool(success),
            "paper_iii_science_benchmark": bool(success),
            "production_claim": False,
            "chtc_ready": bool(success),
            "p5b_p6a_p7a_reclassified_as_science": False,
        },
        "calibration_gate": {
            "local_pre_chtc_only": True,
            "final_table_batch": False,
            "chtc_submission_attempted": False,
            "optuna_used": False,
            "same_seed_group_scope": "single_strict_hh_run",
            "latest_phase3_source_artifact_missing_locally": not Path(config.artifact_json).exists(),
            "exact_reference_role": "diagnostic_post_run_only",
            "strict_controller_exact_inputs_off": True,
        },
        "blockers": blockers,
        "scope_guardrails": {
            "report_only": True,
            "source_artifacts_modified": False,
            "chtc_used": False,
            "optuna_used": False,
            "fresh_adapt_run": False,
            "realtime_or_controller_route_changed": False,
            "adapt_static_defaults_changed": False,
            "mclachlan_realtime_defaults_changed": False,
            "physical_vectors_emitted_in_report": False,
            "p5b_reclassified_as_science": False,
            "p6a_reclassified_as_science": False,
            "p7a_reclassified_as_science": False,
        },
    }
    _assert_no_raw_payload(report, context="paper_iii_local_science_pilot_report")
    return report


def _build_markdown(report: Mapping[str, Any]) -> str:
    run = _mapping(_mapping(report.get("runs"), name="report.runs").get("strict_hh_runtime_dynamics"), name="strict_hh_runtime_dynamics")
    metrics = _mapping(run.get("metrics"), name="strict_run.metrics")
    horizon = _mapping(metrics.get("horizon"), name="strict_run.metrics.horizon")
    energy = _mapping(_mapping(report.get("post_run_comparisons"), name="report.post_run_comparisons").get("energy"), name="energy")
    fidelity = _mapping(_mapping(report.get("post_run_comparisons"), name="report.post_run_comparisons").get("fidelity"), name="fidelity")
    lines = [
        "# Paper III P7b Local Pre-CHTC Science Pilot",
        "",
        f"Generated UTC: `{report.get('generated_utc')}`",
        "",
        "## Classification",
        "",
        f"- paper_iii_science_benchmark: `{str(report.get('paper_iii_science_benchmark')).lower()}`",
        f"- production_claim: `{str(report.get('production_claim')).lower()}`",
        f"- command_status: `{run.get('command_status')}`",
        f"- command_wallclock_seconds: `{run.get('command_wallclock_seconds')}`",
        f"- strict_validation_passed: `{str(_mapping(run.get('strict_validation'), name='strict_validation').get('passed')).lower()}`",
        "",
        "## Strict HH pilot metrics",
        "",
        f"- physical_row_count: `{horizon.get('physical_row_count')}`",
        f"- final_physical_time: `{horizon.get('final_physical_time')}`",
        f"- full_horizon_gate_passed: `{str(horizon.get('full_horizon_gate_passed')).lower()}`",
        f"- final_abs_energy_total_error: `{energy.get('final_abs_energy_total_error')}`",
        f"- max_abs_energy_total_error: `{energy.get('max_abs_energy_total_error')}`",
        f"- fidelity_exact_status: `{fidelity.get('fidelity_exact_status')}`",
        f"- progress_json_exists: `{str(run.get('progress_json_exists')).lower()}`",
        f"- partial_payload_json_exists: `{str(run.get('partial_payload_json_exists')).lower()}`",
        "",
        "## Scope guardrails",
        "",
    ]
    for key, value in _mapping(report.get("scope_guardrails"), name="report.scope_guardrails").items():
        lines.append(f"- {key}: `{str(value).lower() if isinstance(value, bool) else value}`")
    blockers = list(report.get("blockers", []))
    lines.extend(["", "## Blockers", ""])
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- `none`")
    return "\n".join(lines) + "\n"


def _build_command_log(
    *,
    config: PaperIIILocalSciencePilotConfig,
    report: Mapping[str, Any],
    command_result: _CommandResult,
) -> str:
    run = _mapping(_mapping(report.get("runs"), name="report.runs").get("strict_hh_runtime_dynamics"), name="strict_hh_runtime_dynamics")
    metrics = _mapping(run.get("metrics"), name="strict_run.metrics")
    horizon = _mapping(metrics.get("horizon"), name="strict_run.metrics.horizon")
    energy = _mapping(_mapping(report.get("post_run_comparisons"), name="report.post_run_comparisons").get("energy"), name="energy")
    blockers = list(report.get("blockers", []))
    lines = [
        "# P7b Local Science Pilot Command Log",
        "",
        f"Generated UTC: `{report.get('generated_utc')}`",
        "",
        "## Strict HH command",
        "",
        f"`{run.get('command')}`",
        "",
        f"Status: `{command_result.status}`",
        f"Exit code: `{command_result.exit_code}`",
        f"Timed out: `{str(command_result.timed_out).lower()}`",
        f"Started UTC: `{command_result.started_utc}`",
        f"Finished UTC: `{command_result.finished_utc}`",
        f"Wall-clock seconds: `{command_result.wallclock_seconds}`",
        "",
        "## Outputs",
        "",
        f"- strict HH output: `{run.get('output_json')}`",
        f"- report JSON: `{_display_path(config.report_json)}`",
        f"- report markdown: `{_display_path(config.report_md)}`",
        f"- stdout log: `{_display_path(config.stdout_log)}`",
        f"- stderr log: `{_display_path(config.stderr_log)}`",
        f"- progress JSON: `{run.get('progress_json')}` (exists: `{str(run.get('progress_json_exists')).lower()}`)",
        f"- partial payload JSON: `{run.get('partial_payload_json')}` (exists: `{str(run.get('partial_payload_json_exists')).lower()}`)",
        "",
        "## Key metrics",
        "",
        f"- paper_iii_science_benchmark: `{str(report.get('paper_iii_science_benchmark')).lower()}`",
        f"- physical_row_count: `{horizon.get('physical_row_count')}`",
        f"- final_physical_time: `{horizon.get('final_physical_time')}`",
        f"- final_abs_energy_total_error: `{energy.get('final_abs_energy_total_error')}`",
        f"- max_abs_energy_total_error: `{energy.get('max_abs_energy_total_error')}`",
        "",
        "## Blockers",
        "",
    ]
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- `none`")
    return "\n".join(lines) + "\n"


def _build_run_manifest(
    *,
    config: PaperIIILocalSciencePilotConfig,
    report: Mapping[str, Any],
    command_result: _CommandResult,
) -> dict[str, Any]:
    run = _mapping(_mapping(report.get("runs"), name="report.runs").get("strict_hh_runtime_dynamics"), name="strict_hh_runtime_dynamics")
    metrics = _mapping(run.get("metrics"), name="strict_run.metrics")
    return {
        "schema_version": "agent_run_manifest_v1",
        "slice": "paper_iii_p7b_local_science_pilot",
        "generated_utc": report.get("generated_utc"),
        "local_pre_chtc_science_pilot": True,
        "paper_iii_science_benchmark": bool(report.get("paper_iii_science_benchmark")),
        "production_claim": False,
        "pipeline": PAPER_III_LOCAL_SCIENCE_PILOT_PIPELINE,
        "commands": {
            "strict_hh_runtime_dynamics": {
                "command": run.get("command"),
                "exit_code": command_result.exit_code,
                "status": command_result.status,
                "timed_out": command_result.timed_out,
                "started_utc": command_result.started_utc,
                "finished_utc": command_result.finished_utc,
                "wallclock_seconds": command_result.wallclock_seconds,
                "result": "completed" if report.get("paper_iii_science_benchmark") else "blocked",
            }
        },
        "artifacts": {
            "strict_hh_output_json": run.get("output_json"),
            "nominal_strict_hh_output_json": _display_path(config.strict_output_json),
            "output_json": _display_path(config.report_json),
            "output_md": _display_path(config.report_md),
            "command_log_md": _display_path(config.command_log_md),
            "run_manifest_json": _display_path(config.run_manifest_json),
            "command_sh": _display_path(config.command_sh),
            "stdout_log": _display_path(config.stdout_log),
            "stderr_log": _display_path(config.stderr_log),
            "source_artifact_json": _display_path(config.artifact_json),
            "progress_json": run.get("progress_json"),
            "progress_json_exists": run.get("progress_json_exists"),
            "partial_payload_json": run.get("partial_payload_json"),
            "partial_payload_json_exists": run.get("partial_payload_json_exists"),
        },
        "controller_boundary": dict(_mapping(report.get("controller_boundary"), name="report.controller_boundary")),
        "output_summary": {
            "schema_version": report.get("schema_version"),
            "pipeline": report.get("pipeline"),
            "paper_iii_science_benchmark": bool(report.get("paper_iii_science_benchmark")),
            "controller_usable": False,
            "feeds_controller_decisions": False,
            "reference_comparisons_feed_controller_decisions": False,
            "physical_row_count": metrics["horizon"]["physical_row_count"],
            "final_physical_time": metrics["horizon"]["final_physical_time"],
            "final_abs_energy_total_error": metrics["energy"].get("final_abs_energy_total_error"),
            "max_abs_energy_total_error": metrics["energy"].get("max_abs_energy_total_error"),
            "fidelity_exact_status": metrics["fidelity"].get("fidelity_exact_status"),
            "blocker_count": len(report.get("blockers", [])),
            "command_started_utc": command_result.started_utc,
            "command_finished_utc": command_result.finished_utc,
            "command_wallclock_seconds": command_result.wallclock_seconds,
            "progress_json_exists": run.get("progress_json_exists"),
            "partial_payload_json_exists": run.get("partial_payload_json_exists"),
        },
        "scope_guardrails": dict(_mapping(report.get("scope_guardrails"), name="report.scope_guardrails")),
        "blockers": list(report.get("blockers", [])),
    }


def run_paper_iii_local_science_pilot(config: PaperIIILocalSciencePilotConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    payload_source_json = _strict_payload_source_json(config)
    command = build_strict_hh_command(config)

    if config.report_only_existing_output:
        command_result = _report_only_command_result(config)
        _write_command_logs(config, command=command, result=command_result)
    else:
        command_result = _execute_strict_hh_command(config, command=command)
        payload_source_json = _strict_payload_source_json(config)

    blockers: list[str] = []
    if command_result.error:
        blockers.append(command_result.error)
    if not payload_source_json.exists():
        stub = _blocked_hh_stub(config=config, command=command, blockers=blockers or ["strict HH output missing"])
        write_json(config.strict_output_json, stub)
        payload_source_json = config.strict_output_json

    try:
        hh_payload = load_json(payload_source_json)
    except Exception as exc:
        hh_payload = {
            "schema_version": "paper_iii_local_science_pilot_blocked_hh_run_v1",
            "status": "blocked",
            "controller_output_available": False,
            "paper_iii_science_benchmark": False,
            "blockers": [f"strict_output_json_parse_error:{type(exc).__name__}:{exc}"],
        }
        write_json(config.strict_output_json, hh_payload)
        payload_source_json = config.strict_output_json
    if not isinstance(hh_payload, Mapping):
        hh_payload = {
            "schema_version": "paper_iii_local_science_pilot_blocked_hh_run_v1",
            "status": "blocked",
            "controller_output_available": False,
            "paper_iii_science_benchmark": False,
            "blockers": ["strict HH output was not a JSON object"],
        }
        write_json(config.strict_output_json, hh_payload)
        payload_source_json = config.strict_output_json

    generated_utc = _utc_now()
    report = _build_report(
        config=config,
        command=command,
        command_result=command_result,
        strict_payload_source_json=payload_source_json,
        hh_payload=hh_payload,
        generated_utc=generated_utc,
    )
    markdown = _build_markdown(report)
    command_log = _build_command_log(config=config, report=report, command_result=command_result)
    manifest = _build_run_manifest(config=config, report=report, command_result=command_result)

    _assert_no_raw_payload(markdown, context="paper_iii_local_science_pilot_markdown")
    _assert_no_raw_payload(command_log, context="paper_iii_local_science_pilot_command_log")
    _assert_no_raw_payload(manifest, context="paper_iii_local_science_pilot_run_manifest")

    write_json(config.report_json, report)
    config.report_md.write_text(markdown, encoding="utf-8")
    config.command_log_md.write_text(command_log, encoding="utf-8")
    write_json(config.run_manifest_json, manifest)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run/report the Paper-III P7b local pre-CHTC strict HH science pilot.")
    parser.add_argument("--artifact-json", default=str(DEFAULT_HH_ARTIFACT_JSON), help="Existing HH ADAPT seed artifact JSON")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for P7b pilot artifacts")
    parser.add_argument("--t-final", type=float, default=DEFAULT_T_FINAL)
    parser.add_argument("--num-times", type=int, default=DEFAULT_NUM_TIMES)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--run-tag", default="paper_iii_p7b_hh_strict_realtime_pilot")
    parser.add_argument(
        "--report-only-existing-output",
        action="store_true",
        help="Do not launch the HH subprocess; validate/report an existing hh_strict_realtime_pilot.json.",
    )
    parser.add_argument(
        "--existing-strict-output-json",
        default=None,
        help=(
            "Explicit existing strict HH JSON to validate in report-only mode. "
            "Reports are written to --output-dir without modifying the source JSON."
        ),
    )
    parser.add_argument("--progress-json", default=None, help="Optional strict HH progress JSON path to forward.")
    parser.add_argument(
        "--partial-payload-json",
        default=None,
        help="Optional strict HH partial-payload JSON path to forward.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_paper_iii_local_science_pilot(
        PaperIIILocalSciencePilotConfig(
            artifact_json=Path(args.artifact_json),
            output_dir=Path(args.output_dir),
            t_final=float(args.t_final),
            num_times=int(args.num_times),
            timeout_seconds=int(args.timeout_seconds),
            run_tag=str(args.run_tag),
            report_only_existing_output=bool(args.report_only_existing_output),
            progress_json=Path(args.progress_json) if args.progress_json else None,
            partial_payload_json=Path(args.partial_payload_json) if args.partial_payload_json else None,
            existing_strict_output_json=Path(args.existing_strict_output_json)
            if args.existing_strict_output_json
            else None,
        )
    )
    return 0


__all__ = [
    "MODULE_NAME",
    "PAPER_III_LOCAL_SCIENCE_PILOT_PIPELINE",
    "PAPER_III_LOCAL_SCIENCE_PILOT_REPORT_SCHEMA_VERSION",
    "PaperIIILocalSciencePilotConfig",
    "PaperIIILocalSciencePilotError",
    "build_parser",
    "build_strict_hh_command",
    "main",
    "run_paper_iii_local_science_pilot",
    "validate_strict_hh_payload",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
