#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.route_identity import (
    ROUTE_ID_UNSPECIFIED,
    STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
    STATIC_META_FEATURE_PROFILE_SAFE_CORE_V1,
    normalize_static_meta_feature_profile,
    normalize_static_route_id,
    route_identity_payload,
    static_route_id_from_record,
)

HH_ROUTE_FAITHFULNESS_LADDER_SCHEMA = "paper_i_hh_snake_noise_route_faithfulness_ladder_v1"
HH_SELECTED_ENERGY_DIAGNOSTIC_ZERO_NOISE_RUNG_IDS = frozenset(
    {
        "selected_energy_oracle_zero_noise_control",
        "both_oracle_surfaces_zero_noise_control",
    }
)
HH_ROUTE_FAITHFULNESS_RUNG_IDS = frozenset(
    {
        "original_route_clean_control",
        "checkpoint_replay_zero_noise_control",
        "oracle_gradient_zero_noise_control",
        "selected_energy_oracle_zero_noise_control",
        "both_oracle_surfaces_zero_noise_control",
        "shot_proxy_target_hit",
        "shot_proxy_noise_floor_stop",
        "shot_proxy_confidence_scoring",
        "gate_synthetic_depolarizing_target_hit",
        "gate_synthetic_depolarizing_noise_floor_stop",
        "gate_synthetic_depolarizing_confidence_scoring",
        "combined_noise_target_hit",
        "combined_noise_noise_floor_stop",
        "combined_noise_confidence_scoring",
    }
)
HH_BLOCKED_SELECTED_ENERGY_INNER_OBJECTIVE_MODE = "noisy_v1"
HH_BLOCKED_NOISE_RUNG_REASON = "blocked_until_selected_energy_zero_noise_control_passes_route_faithfulness_v1"
HH_NOISY_V1_TARGET_HIT_RUNG_IDS = frozenset(
    {
        "shot_proxy_target_hit",
        "gate_synthetic_depolarizing_target_hit",
        "combined_noise_target_hit",
    }
)
SPIN_BOSON_ROUTE_FAITHFULNESS_LADDER_SCHEMA = "paper_i_spin_boson_snake_noise_route_faithfulness_ladder_v1"
SPIN_BOSON_ZERO_NOISE_ADAPTIVE_RUNG_ID = "same_driver_zero_noise_adaptive_control"
SPIN_BOSON_SCALAR_VALUE_NOISE_RUNG_ID = "scalar_value_noise_target_hit"
SPIN_BOSON_ZERO_NOISE_PASS_EVIDENCE_SCHEMA = "paper_i_spin_boson_zero_noise_adaptive_pass_evidence_v1"


def _hh_noise_evidence_module() -> Any:
    """Import HH noise evidence helpers only for noise-ladder rows.

    Normal Paper-I HH recovery/preflight rows do not need this module.  Keeping
    it lazy avoids pulling the heavy noise-evidence path into unrelated submit
    preflights.
    """

    from chtc.phase3_optuna import paper_i_hh_noise_evidence as module  # noqa: WPS433

    return module


def _hubbard_recalibration_module() -> Any:
    """Import Hubbard recalibration helpers only for recalibration rows."""

    from chtc.phase3_optuna import paper_i_hubbard_snake_recalibration as module  # noqa: WPS433

    return module


def _noise_oracle_defaults_helpers() -> tuple[Any, Any]:
    from pipelines.exact_bench.noise_oracle_defaults import (  # noqa: WPS433
        gate_tuple_to_cli_value,
        normalize_gate_name_tuple,
    )

    return gate_tuple_to_cli_value, normalize_gate_name_tuple


def _clean_ladder_contract_module() -> Any:
    from chtc.phase3_optuna import paper_i_clean_ladder_contract as module  # noqa: WPS433

    return module


def _table_i_audit_escalation_module() -> Any:
    from chtc.phase3_optuna import paper_i_table_i_audit_escalation as module  # noqa: WPS433

    return module


def normalize_paper_i_ladder_stage(value: str) -> Any:
    return _clean_ladder_contract_module().normalize_paper_i_ladder_stage(value)


def validate_clean_ladder_common_fields(*args: Any, **kwargs: Any) -> Any:
    return _clean_ladder_contract_module().validate_clean_ladder_common_fields(*args, **kwargs)


def validate_clean_ladder_source_metadata(*args: Any, **kwargs: Any) -> Any:
    return _clean_ladder_contract_module().validate_clean_ladder_source_metadata(*args, **kwargs)


def validate_candidate_row_authorization(*args: Any, **kwargs: Any) -> Any:
    return _table_i_audit_escalation_module().validate_candidate_row_authorization(*args, **kwargs)


def _routea_selected_recovery_contract_present(row: Mapping[str, str]) -> bool:
    fields = (
        "paper_i_cutoff_ladder_stage",
        "source_lock_reference_json",
        "source_lock_command_audit_json",
        "source_lock_result_json",
        "source_lock_status",
        "selected_logical_source_json",
        "selected_logical_source_manifest_json",
    )
    return any(str(row.get(field) or "").strip() for field in fields)


def validate_routea_selected_recovery_contract(row: Mapping[str, str], *args: Any, **kwargs: Any) -> Any:
    if not _routea_selected_recovery_contract_present(row):
        return []
    return _clean_ladder_contract_module().validate_routea_selected_recovery_contract(row, *args, **kwargs)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_positive_float_env(
    name: str,
    *,
    default: float,
    observability_errors: list[str],
    minimum: float | None = None,
) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        value = float(str(raw).strip())
    except Exception:
        observability_errors.append(f"env:{name}:invalid_float:{raw!r};using_default={default}")
        return float(default)
    if not math.isfinite(value) or value <= 0:
        observability_errors.append(f"env:{name}:nonpositive_or_nonfinite:{raw!r};using_default={default}")
        return float(default)
    if minimum is not None and value < minimum:
        observability_errors.append(f"env:{name}:below_minimum:{value};using_minimum={minimum}")
        return float(minimum)
    return float(value)


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_nonnegative_float_env(
    name: str,
    *,
    default: float,
    observability_errors: list[str],
) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        value = float(str(raw).strip())
    except Exception:
        observability_errors.append(f"env:{name}:invalid_float:{raw!r};using_default={default}")
        return float(default)
    if not math.isfinite(value) or value < 0:
        observability_errors.append(f"env:{name}:negative_or_nonfinite:{raw!r};using_default={default}")
        return float(default)
    return float(value)


def _heartbeat_interval_s(observability_errors: list[str]) -> float:
    return _read_positive_float_env(
        "PHASE3_HEARTBEAT_INTERVAL_SEC",
        default=60.0,
        observability_errors=observability_errors,
        minimum=5.0,
    )


def _progress_stale_after_s(observability_errors: list[str]) -> float:
    return _read_positive_float_env(
        "PHASE3_PROGRESS_STALE_AFTER_SEC",
        default=300.0,
        observability_errors=observability_errors,
    )


def _require_first_progress_within_s(observability_errors: list[str]) -> float:
    return _read_nonnegative_float_env(
        "PHASE3_REQUIRE_FIRST_PROGRESS_WITHIN_SEC",
        default=0.0,
        observability_errors=observability_errors,
    )


def _split_words(value: str) -> list[str]:
    value = (value or "").strip()
    return [] if not value else value.split()


def _add_option(cmd: list[str], flag: str, value: str | None) -> None:
    if value is None:
        return
    value = str(value).strip()
    if value == "" or value.lower() == "none":
        return
    cmd.extend([flag, value])


def _canonical_gate_cli_value(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and str(value).strip().lower() in {"", "none"}:
        return None
    gate_tuple_to_cli_value, normalize_gate_name_tuple = _noise_oracle_defaults_helpers()
    gates = normalize_gate_name_tuple(value, default=None, field_name=field_name)
    if not gates:
        return None
    return gate_tuple_to_cli_value(gates, field_name=field_name)


def _gate_list_values(value: Any, *, field_name: str) -> list[str]:
    canonical = _canonical_gate_cli_value(value, field_name=field_name)
    if canonical is None:
        return []
    _, normalize_gate_name_tuple = _noise_oracle_defaults_helpers()
    return list(normalize_gate_name_tuple(canonical, default=None, field_name=field_name))


def _objective_2q_depth_weight(row: Mapping[str, str]) -> str | None:
    """Prefer explicit D2q weight, but keep legacy depth-weight records hardware-aware."""

    for key in ("objective_2q_depth_weight", "objective_depth_2q_weight", "objective_depth_weight"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "none":
            return text
    return None


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _falsey(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "n", "off"}


_PHASE0_ROW_FIELDS = (
    "phase0_pilot_enabled",
    "phase0_pilot_alpha",
    "phase0_pilot_threshold",
    "phase0_pilot_max_records",
    "phase0_lane_quota_pressure",
    "phase0_algebraic_lane_mode",
)
_ORACLE_WARM_START_REQUIREMENT_FIELDS = (
    "oracle_required_static_route_id",
    "oracle_required_suite_profile",
    "oracle_require_phase0_aware",
    "oracle_require_compatible_warm_starts",
)


def _row_subset(row: Mapping[str, str], fields: Sequence[str]) -> dict[str, str]:
    return {field: str(row.get(field) or "") for field in fields}


def _fixed_inner_optimizer(row: Mapping[str, str], *, static_route_id: str) -> str:
    raw = str(row.get("fixed_inner_optimizer") or "").strip().upper()
    if normalize_static_route_id(static_route_id) != ROUTE_ID_UNSPECIFIED and not raw:
        raise ValueError("declared static Route A/B records must explicitly set fixed_inner_optimizer")
    return raw or "SPSA"


def _validate_canonical_class_record(row: Mapping[str, str], *, record_id: str) -> None:
    lane = str(row.get("canonical_lane") or "").strip().lower().replace("-", "_")
    rid = str(record_id or row.get("record_id") or "").strip().lower()
    families = {part.strip().lower() for part in str(row.get("families") or "").split() if part.strip()}
    looks_mixed = lane == "mixed" or "_mixed_" in rid or rid.startswith("mixed_")
    if not looks_mixed:
        return
    if "spin_boson" in families:
        raise ValueError(
            f"{record_id}: spin_boson is bosonic/spin-oscillator, not mixed fermion-boson; "
            "do not submit legacy mixed records that average HH with spin_boson."
        )
    if families and not any(family.startswith("molecular_vibronic") for family in families):
        raise ValueError(
            f"{record_id}: mixed fermion-boson canonical records must include a molecular-vibronic "
            "benchmark in addition to HH; HH-only mixed tuning is intentionally blocked."
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def load_record(records_path: Path, record_id: str) -> dict[str, str]:
    rows = list(csv.DictReader(records_path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    row = next((r for r in rows if r.get("record_id") == record_id), None)
    if row is None:
        raise ValueError(f"record_id {record_id!r} not found in {records_path}")
    return dict(row)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _resolve_under_repo(path_value: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    raw = str(path_value or "").strip()
    path = Path(os.path.expanduser(raw))
    return path if path.is_absolute() else Path(repo_root) / path


def _safe_relative(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    try:
        return str(Path(path).resolve(strict=False).relative_to(Path(repo_root).resolve(strict=False)))
    except ValueError:
        return str(path)


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve(strict=False).relative_to(Path(root).resolve(strict=False))
        return True
    except ValueError:
        return False


def parse_submit_contract(submit_path: Path) -> dict[str, Any]:
    """Parse the small HTCondor submit contract needed for local preflight."""

    submit_path = Path(submit_path)
    fields: dict[str, str] = {}
    queue_record_id_file = ""
    for raw_line in submit_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("queue "):
            queue_body = line[len("queue ") :].strip()
            lower_body = queue_body.lower()
            from_index = lower_body.find(" from ")
            if from_index >= 0:
                queue_vars = queue_body[:from_index]
                queue_file = queue_body[from_index + len(" from ") :].strip()
                queue_var_names = [
                    item.strip()
                    for chunk in queue_vars.split(",")
                    for item in chunk.split()
                    if item.strip()
                ]
                if "record_id" in queue_var_names:
                    queue_record_id_file = queue_file
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    transfer_input_files = [
        part.strip()
        for part in fields.get("transfer_input_files", "").split(",")
        if part.strip()
    ]
    raw_env = fields.get("environment", "").strip()
    if (raw_env.startswith('"') and raw_env.endswith('"')) or (raw_env.startswith("'") and raw_env.endswith("'")):
        raw_env = raw_env[1:-1]
    env: dict[str, str] = {}
    for token in shlex.split(raw_env):
        if "=" in token:
            key, value = token.split("=", 1)
            env[key] = value
    return {
        "schema": "phase3_submit_contract_v1",
        "submit_path": str(submit_path),
        "executable": fields.get("executable", "").strip(),
        "arguments": fields.get("arguments", "").strip(),
        "arguments": fields.get("arguments", "").strip(),
        "transfer_input_files": transfer_input_files,
        "transfer_output_files": [
            part.strip()
            for part in fields.get("transfer_output_files", "").split(",")
            if part.strip()
        ],
        "environment": env,
        "records_path": env.get("PHASE3_RECORDS_PATH", ""),
        "argument_records_path": (
            shlex.split(fields.get("arguments", "").strip())[1]
            if len(shlex.split(fields.get("arguments", "").strip())) >= 2
            and shlex.split(fields.get("arguments", "").strip())[0] == "$(record_id)"
            else ""
        ),
        "queue_record_id_file": queue_record_id_file,
        "job_batch_name": fields.get("+JobBatchName", "").strip().strip('"'),
        "request_cpus": fields.get("request_cpus", "").strip().strip('"'),
        "request_memory": fields.get("request_memory", "").strip().strip('"'),
        "request_disk": fields.get("request_disk", "").strip().strip('"'),
        "requirements": fields.get("requirements", "").strip(),
        "max_runtime": fields.get("+MaxRuntime", "").strip().strip('"'),
        "should_transfer_files": fields.get("should_transfer_files", "").strip(),
        "when_to_transfer_output": fields.get("when_to_transfer_output", "").strip(),
        "transfer_executable": fields.get("transfer_executable", "").strip(),
        "preserve_relative_paths": fields.get("preserve_relative_paths", "").strip(),
    }


def _is_sandbox_visible(path_value: str | Path, *, transfer_input_files: Sequence[str], repo_root: Path = REPO_ROOT) -> bool:
    path = _resolve_under_repo(path_value, repo_root=repo_root)
    repo = Path(repo_root).resolve(strict=False)
    if not _path_is_under(path, repo):
        return False
    for item in transfer_input_files:
        root = _resolve_under_repo(item, repo_root=repo_root)
        if _path_is_under(path, root):
            return True
    return False


def _source_artifact_values(row: Mapping[str, str]) -> list[tuple[str, str]]:
    fields = (
        "historical_ledger",
        "selected_logical_source_json",
        "selected_logical_source_manifest_json",
        "molecular_problem_json",
        "hardware_resolution_profile_json",
        "trial_param_overrides_json",
        "enqueue_trial_params_json",
        "selected_energy_zero_noise_pass_evidence_json",
        "oracle_summary_root",
        "reference_energy_cache_json",
        "paper_i_hubbard_snake_recalibration_candidate_manifest_json",
        "paper_i_hubbard_snake_recalibration_source_audit_json",
    )
    out: list[tuple[str, str]] = []
    for field in fields:
        value = str(row.get(field) or "").strip()
        if value and value.lower() != "none":
            out.append((field, value))
    return out


def _int_words(value: str | None) -> tuple[int, ...] | None:
    words = _split_words(value or "")
    return None if not words else tuple(int(part) for part in words)


def _positive_int_or_none(value: str | None) -> int | None:
    try:
        parsed = int(float(str(value or "").strip()))
    except Exception:
        return None
    return parsed if parsed >= 0 else None


def _candidate_specs_for_row(row: Mapping[str, str]) -> tuple[Any, ...]:
    import pipelines.static_adapt.optimization.phase3_policy_optuna as p3opt
    from pipelines.exact_bench.table_i_canonical_cases import table_i_executable_specs, table_i_suite_profile

    calibration_profile = str(row.get("calibration_profile") or "off").strip().lower().replace("-", "_")
    if calibration_profile and calibration_profile != "off":
        return tuple(p3opt.calibration_static_benchmark_specs(calibration_profile))

    requested_benchmark_ids = tuple(_split_words(row.get("benchmark_ids", "")))
    suite_profile_raw = str(row.get("suite_profile") or "").strip()
    if suite_profile_raw and requested_benchmark_ids:
        profile_key = table_i_suite_profile(suite_profile_raw)
        specs = tuple(table_i_executable_specs(profile_key))
        requested = set(requested_benchmark_ids)
        selected = tuple(spec for spec in specs if str(spec.benchmark_id) in requested)
        if selected:
            return selected

    families = p3opt.canonical_lane_families(row.get("canonical_lane")) or tuple(_split_words(row.get("families", ""))) or None
    sizes = _int_words(row.get("sizes"))
    boson_cutoffs = _int_words(row.get("boson_cutoffs"))
    boson_cutoff = None
    if not boson_cutoffs and str(row.get("boson_cutoff") or "").strip():
        boson_cutoff = int(str(row.get("boson_cutoff")).strip())
    raw_ref = str(row.get("exact_reference_boson_cutoff") or "4").strip()
    exact_reference_boson_cutoff = None if raw_ref and int(raw_ref) <= 0 else int(raw_ref or "4")
    specs = p3opt.filter_static_benchmark_suite(
        families=families,
        sizes=sizes,
        molecular_problem_json=(row.get("molecular_problem_json") or None),
        boson_cutoff=boson_cutoff,
        boson_cutoffs=boson_cutoffs,
        exact_reference_boson_cutoff=exact_reference_boson_cutoff,
        physics_grid_profile=row.get("physics_grid_profile") or "canonical",
    )
    return tuple(
        p3opt.filter_canonical_lane_specs(
            specs,
            lane=row.get("canonical_lane"),
            stage=row.get("canonical_lane_stage") or "train",
        )
    )


def _float_field(row: Mapping[str, str], field: str) -> float | None:
    try:
        value = float(str(row.get(field) or ""))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _validate_reference_cache_record(
    *,
    cache_records: Mapping[str, Any],
    key_hash: str,
    expected_nph: int,
    expected_energy: float | None,
    label: str,
) -> list[str]:
    blockers: list[str] = []
    if not key_hash:
        blockers.append(f"paper_i_ladder_{label}_energy_key_missing")
        return blockers
    record = cache_records.get(key_hash)
    if not isinstance(record, Mapping):
        blockers.append(f"paper_i_ladder_reference_cache_key_missing:{key_hash}")
        return blockers
    key = record.get("key") if isinstance(record.get("key"), Mapping) else {}
    try:
        actual_nph = int(key.get("n_ph_max"))
    except Exception:
        actual_nph = None
    if actual_nph != int(expected_nph):
        blockers.append(f"paper_i_ladder_{label}_cache_nph_mismatch:{actual_nph}:{expected_nph}")
    try:
        actual_energy = float(record.get("exact_energy"))
    except Exception:
        actual_energy = None
    if expected_energy is None:
        blockers.append(f"paper_i_ladder_{label}_row_energy_missing")
    elif actual_energy is None or not math.isclose(float(actual_energy), float(expected_energy), rel_tol=0.0, abs_tol=1e-10):
        blockers.append(f"paper_i_ladder_{label}_cache_energy_mismatch:{actual_energy}:{expected_energy}")
    return blockers


def _pipeline_arg_value(args: Sequence[Any], flag: str) -> str | None:
    tokens = tuple(str(part) for part in args)
    try:
        return tokens[tokens.index(str(flag)) + 1]
    except (ValueError, IndexError):
        return None


def _paper_i_cutoff_floor_preflight(
    row: Mapping[str, str],
    *,
    expected_specs: Sequence[Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Reject paper-facing rows whose exact cutoff floor already misses tau."""

    if str(row.get("required_target_profile") or "").strip() != "paper_i_phys_v1":
        return [], []
    n_ph_work_row = str(row.get("n_ph_work") or "").strip()
    n_ph_ref_row = str(row.get("n_ph_ref") or "").strip()
    if not n_ph_work_row or not n_ph_ref_row:
        return [], []

    tau_raw = str(row.get("tau_phys") or row.get("tau_tight") or "0.0002").strip()
    try:
        tau = float(tau_raw)
    except Exception:
        tau = 0.0002

    requested = set(_split_words(row.get("benchmark_ids", "")))
    blockers: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    for spec in expected_specs:
        benchmark_id = str(getattr(spec, "benchmark_id", ""))
        if requested and benchmark_id not in requested:
            continue
        if not bool(getattr(getattr(spec, "features", None), "bosonic", False)):
            continue
        try:
            work_nph = int(n_ph_work_row)
            ref_nph = int(n_ph_ref_row)
        except Exception:
            blockers.append(f"paper_i_cutoff_floor_nph_fields_invalid:{benchmark_id}:{n_ph_work_row}:{n_ph_ref_row}")
            continue
        if ref_nph <= work_nph:
            blockers.append(f"paper_i_cutoff_floor_reference_not_above_work:{benchmark_id}:{work_nph}:{ref_nph}")
            continue
        spec_ref = getattr(spec, "exact_reference_n_ph_max", None)
        if spec_ref is not None and int(spec_ref) != int(ref_nph):
            blockers.append(f"paper_i_cutoff_floor_spec_ref_mismatch:{benchmark_id}:{spec_ref}:{ref_nph}")
            continue
        spec_work_raw = _pipeline_arg_value(getattr(spec, "base_pipeline_args", ()), "--n-ph-max")
        if spec_work_raw is not None and int(spec_work_raw) != int(work_nph):
            blockers.append(f"paper_i_cutoff_floor_spec_work_mismatch:{benchmark_id}:{spec_work_raw}:{work_nph}")
            continue
        try:
            from pipelines.exact_bench.static_reference_metrics import exact_energy_for_spec

            same_energy, _same_key, _same_payload = exact_energy_for_spec(spec, n_ph_max=work_nph)
            ref_energy, _ref_key, _ref_payload = exact_energy_for_spec(spec, n_ph_max=ref_nph)
        except Exception as exc:
            blockers.append(f"paper_i_cutoff_floor_exact_energy_failed:{benchmark_id}:{type(exc).__name__}:{exc}")
            continue
        cutoff_floor = abs(float(same_energy) - float(ref_energy))
        diagnostic = {
            "benchmark_id": benchmark_id,
            "n_ph_work": int(work_nph),
            "n_ph_ref": int(ref_nph),
            "same_cutoff_exact_energy": float(same_energy),
            "reference_cutoff_exact_energy": float(ref_energy),
            "exact_cutoff_floor_abs_delta_e": float(cutoff_floor),
            "tau_phys": float(tau),
            "status": "fail" if cutoff_floor > tau else "pass",
        }
        diagnostics.append(diagnostic)
        if cutoff_floor > tau:
            blockers.append(
                "paper_i_cutoff_floor_exceeds_tau:"
                f"{benchmark_id}:n_ph_work={work_nph}:n_ph_ref={ref_nph}:"
                f"cutoff_floor={cutoff_floor:.12g}:tau={tau:.12g}"
            )
    return blockers, diagnostics


def _validate_routea_clean_ladder_preflight(
    row: Mapping[str, str],
    *,
    record_id: str,
    expected_benchmark_ids: Sequence[str],
    expected_specs: Sequence[Any],
    transfer_inputs: Sequence[str],
    submit_contract: Mapping[str, Any] | None,
    repo_root: Path,
) -> tuple[list[str], dict[str, Any] | None]:
    stage_raw = str(row.get("paper_i_cutoff_ladder_stage") or "").strip()
    if not stage_raw:
        return [], None
    blockers: list[str] = []
    try:
        stage, config = normalize_paper_i_ladder_stage(stage_raw)
    except ValueError:
        return [f"paper_i_ladder_unknown_stage:{stage_raw}"], {"stage": stage_raw, "validation_status": "fail"}
    blockers.extend(validate_clean_ladder_common_fields(row, stage=stage, stage_config=config, row_kind="routea_phase0"))
    blockers.extend(
        validate_routea_selected_recovery_contract(
            row,
            submit_contract=submit_contract,
            repo_root=repo_root,
        )
    )
    benchmark_ids = _split_words(row.get("benchmark_ids", ""))
    if len(benchmark_ids) != 1:
        blockers.append(f"paper_i_ladder_requires_single_benchmark_id:{benchmark_ids}")
    elif benchmark_ids[0] not in set(expected_benchmark_ids):
        blockers.append(f"paper_i_ladder_non_executable_benchmark_id:{benchmark_ids[0]}")
    blockers.extend(
        validate_clean_ladder_source_metadata(
            row,
            stage=stage,
            stage_config=config,
            target_case_id=benchmark_ids[0] if len(benchmark_ids) == 1 else None,
        )
    )
    if config.requires_prior_failure:
        for field in ("paper_i_ladder_candidate_manifest_json", "paper_i_ladder_source_audit_json"):
            value = str(row.get(field) or "").strip()
            if not value:
                continue
            resolved = _resolve_under_repo(value, repo_root=repo_root)
            if not resolved.exists():
                blockers.append(f"source_artifact_missing:{field}:{value}")
            if transfer_inputs and not _is_sandbox_visible(value, transfer_input_files=transfer_inputs, repo_root=repo_root):
                blockers.append(f"source_artifact_not_transferred:{field}:{value}")
        blockers.extend(
            validate_candidate_row_authorization(
                row,
                target_case_id=benchmark_ids[0] if len(benchmark_ids) == 1 else "",
                lane="snake",
                algorithm_id="static_family_native_adapt_phase3",
                repo_root=repo_root,
            )
        )
    if any("spin_boson_L1" in bid for bid in benchmark_ids):
        blockers.append("paper_i_ladder_legacy_spin_boson_L1_forbidden")
    selected_specs = [spec for spec in expected_specs if str(getattr(spec, "benchmark_id", "")) in set(benchmark_ids)]
    if len(selected_specs) == 1:
        try:
            import pipelines.static_adapt.optimization.phase3_policy_optuna as p3opt

            target = p3opt.resolve_paper_i_phys_v1_target(selected_specs[0])
            if int(target.get("N_phys")) != 2:
                blockers.append(f"paper_i_ladder_paper_i_phys_N_phys_mismatch:{target.get('N_phys')}:2")
            if not math.isclose(float(target.get("tau_phys")), 0.0002, rel_tol=0.0, abs_tol=1e-15):
                blockers.append(f"paper_i_ladder_paper_i_phys_tau_phys_mismatch:{target.get('tau_phys')}:0.0002")
            if not math.isclose(float(target.get("tau_tight")), 0.0002, rel_tol=0.0, abs_tol=1e-15):
                blockers.append(f"paper_i_ladder_paper_i_phys_tau_tight_mismatch:{target.get('tau_tight')}:0.0002")
            if config.n_ph_work is not None and int(target.get("phonon_cutoff_work")) != int(config.n_ph_work):
                blockers.append(f"paper_i_ladder_paper_i_phys_work_cutoff_mismatch:{target.get('phonon_cutoff_work')}:{config.n_ph_work}")
            if config.n_ph_ref is not None and int(target.get("phonon_cutoff_eval_reference")) != int(config.n_ph_ref):
                blockers.append(f"paper_i_ladder_paper_i_phys_ref_cutoff_mismatch:{target.get('phonon_cutoff_eval_reference')}:{config.n_ph_ref}")
            if str(target.get("accuracy_gate_metric") or "") != "abs_error_reference_cutoff":
                blockers.append(f"paper_i_ladder_paper_i_phys_accuracy_metric_mismatch:{target.get('accuracy_gate_metric')}")
        except Exception as exc:
            blockers.append(f"paper_i_ladder_paper_i_phys_target_failed:{type(exc).__name__}:{exc}")
    if str(row.get("required_target_benchmark_ids") or "").strip() != " ".join(benchmark_ids):
        blockers.append("paper_i_ladder_required_target_benchmark_mismatch")
    if str(row.get("required_target_profile") or "").strip() != "paper_i_phys_v1":
        blockers.append("paper_i_ladder_required_target_profile_mismatch")
    if str(row.get("discovery_objective_mode") or "").strip() != "discovery_first_crossing":
        blockers.append("paper_i_ladder_discovery_objective_mode_mismatch")
    if str(row.get("target_abs_delta_e") or "").strip():
        blockers.append("paper_i_ladder_legacy_target_abs_delta_e_must_be_blank")
    if str(row.get("required_target_abs_delta_e") or "").strip():
        blockers.append("paper_i_ladder_required_target_abs_delta_e_must_be_blank")
    if str(row.get("physics_grid_profile") or "").strip() != "paper_i_clean":
        blockers.append("paper_i_ladder_physics_grid_profile_mismatch")
    if str(row.get("static_route_id") or "").strip() != "route_a":
        blockers.append("paper_i_ladder_static_route_id_mismatch")
    if str(row.get("fixed_inner_optimizer") or "").strip().upper() != "SPSA":
        blockers.append("paper_i_ladder_fixed_inner_optimizer_mismatch")
    if str(row.get("paper_i_ladder_snake_policy") or "").strip() not in _clean_ladder_contract_module().PAPER_I_LADDER_SNAKE_POLICIES:
        blockers.append("paper_i_ladder_snake_policy_missing_or_invalid")
    if config.n_ph_work is not None and str(row.get("boson_cutoff") or "").strip() != str(config.n_ph_work):
        blockers.append("paper_i_ladder_boson_cutoff_mismatch")
    if config.n_ph_ref is not None and str(row.get("exact_reference_boson_cutoff") or "").strip() != str(config.n_ph_ref):
        blockers.append("paper_i_ladder_exact_reference_boson_cutoff_mismatch")
    if str(row.get("reference_energy_status") or "").strip() != "ok":
        blockers.append("paper_i_ladder_reference_energy_status_mismatch")

    cache_value = str(row.get("reference_energy_cache_json") or "").strip()
    same_key = str(row.get("same_cutoff_reference_energy_key") or "").strip()
    ref_key = str(row.get("reference_cutoff_energy_key") or "").strip()
    cache_records: Mapping[str, Any] = {}
    if not cache_value:
        blockers.append("paper_i_ladder_reference_energy_cache_missing")
    else:
        cache_path = _resolve_under_repo(cache_value, repo_root=repo_root)
        if transfer_inputs and not _is_sandbox_visible(cache_value, transfer_input_files=transfer_inputs, repo_root=repo_root):
            blockers.append(f"source_artifact_not_transferred:reference_energy_cache_json:{cache_value}")
        try:
            cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if str(cache_payload.get("schema") or "") != "static_reference_energy_cache_v1":
                blockers.append(f"paper_i_ladder_reference_cache_schema_mismatch:{cache_payload.get('schema')}")
            raw_records = cache_payload.get("records")
            if not isinstance(raw_records, Mapping) or not raw_records:
                blockers.append("paper_i_ladder_reference_cache_records_missing_or_empty")
                cache_records = {}
            else:
                cache_records = dict(raw_records)
        except Exception as exc:
            blockers.append(f"paper_i_ladder_reference_cache_unreadable:{type(exc).__name__}:{exc}")
    if config.n_ph_work is not None:
        blockers.extend(
            _validate_reference_cache_record(
                cache_records=cache_records,
                key_hash=same_key,
                expected_nph=int(config.n_ph_work),
                expected_energy=_float_field(row, "same_cutoff_exact_gs_energy"),
                label="same_cutoff",
            )
        )
    if config.n_ph_ref is not None:
        blockers.extend(
            _validate_reference_cache_record(
                cache_records=cache_records,
                key_hash=ref_key,
                expected_nph=int(config.n_ph_ref),
                expected_energy=_float_field(row, "exact_reference_energy"),
                label="reference_cutoff",
            )
        )
    payload = {
        "stage": stage,
        "n_ph_work": config.n_ph_work,
        "n_ph_ref": config.n_ph_ref,
        "snake_policy": row.get("paper_i_ladder_snake_policy"),
        "validation_status": "fail" if blockers else "pass",
        "reference_energy_cache_json": cache_value,
        "same_cutoff_reference_energy_key": same_key,
        "reference_cutoff_energy_key": ref_key,
    }
    return blockers, payload


def _is_hubbard_snake_recalibration_row(row: Mapping[str, str]) -> bool:
    return (
        _truthy(row.get("paper_i_hubbard_snake_recalibration"))
        or bool(str(row.get("paper_i_hubbard_snake_recalibration_candidate_manifest_json") or "").strip())
        or bool(str(row.get("paper_i_hubbard_snake_recalibration_candidate_key") or "").strip())
    )


def _validate_hubbard_snake_recalibration_preflight(
    row: Mapping[str, str],
    *,
    expected_benchmark_ids: Sequence[str],
    repo_root: Path,
) -> tuple[list[str], dict[str, Any] | None]:
    if not _is_hubbard_snake_recalibration_row(row):
        return [], None
    hubbard_recalibration = _hubbard_recalibration_module()
    blockers: list[str] = []
    benchmark_ids = _split_words(row.get("benchmark_ids", ""))
    expected_cases = set(hubbard_recalibration.EXPECTED_CASE_ID_BY_REGIME.values())
    target_case_id = benchmark_ids[0] if len(benchmark_ids) == 1 else ""
    if len(benchmark_ids) != 1:
        blockers.append(f"paper_i_hubbard_snake_recalibration_requires_single_benchmark_id:{benchmark_ids}")
    elif target_case_id not in expected_cases:
        blockers.append(f"paper_i_hubbard_snake_recalibration_unexpected_benchmark_id:{target_case_id}")
    elif target_case_id not in set(expected_benchmark_ids):
        blockers.append(f"paper_i_hubbard_snake_recalibration_non_executable_benchmark_id:{target_case_id}")
    if _split_words(row.get("families", "")) != ["hubbard"]:
        blockers.append(f"paper_i_hubbard_snake_recalibration_family_mismatch:{row.get('families')}")
    if str(row.get("suite_profile") or "").strip() != hubbard_recalibration.TABLE_I_CLEAN_NPH3_REF4_PROFILE:
        blockers.append(
            "paper_i_hubbard_snake_recalibration_suite_profile_mismatch:"
            f"{row.get('suite_profile')}:{hubbard_recalibration.TABLE_I_CLEAN_NPH3_REF4_PROFILE}"
        )
    if str(row.get("static_route_id") or "").strip() != "route_a":
        blockers.append("paper_i_hubbard_snake_recalibration_static_route_id_mismatch")
    if str(row.get("fixed_inner_optimizer") or "").strip().upper() != "SPSA":
        blockers.append("paper_i_hubbard_snake_recalibration_fixed_inner_optimizer_mismatch")
    if str(row.get("required_target_profile") or "").strip() != "paper_i_phys_v1":
        blockers.append("paper_i_hubbard_snake_recalibration_required_target_profile_mismatch")
    if str(row.get("discovery_objective_mode") or "").strip() != "discovery_first_crossing":
        blockers.append("paper_i_hubbard_snake_recalibration_discovery_objective_mode_mismatch")
    if str(row.get("required_target_benchmark_ids") or "").strip() != " ".join(benchmark_ids):
        blockers.append("paper_i_hubbard_snake_recalibration_required_target_benchmark_mismatch")
    if str(row.get("target_abs_delta_e") or "").strip():
        blockers.append("paper_i_hubbard_snake_recalibration_legacy_target_abs_delta_e_must_be_blank")
    if str(row.get("required_target_abs_delta_e") or "").strip():
        blockers.append("paper_i_hubbard_snake_recalibration_required_target_abs_delta_e_must_be_blank")
    if str(row.get("physics_grid_profile") or "").strip() != "paper_i_clean":
        blockers.append("paper_i_hubbard_snake_recalibration_physics_grid_profile_mismatch")
    if str(row.get("exact_reference_boson_cutoff") or "").strip() != "0":
        blockers.append("paper_i_hubbard_snake_recalibration_exact_reference_boson_cutoff_mismatch")
    if str(row.get("boson_cutoff") or "").strip() or str(row.get("boson_cutoffs") or "").strip():
        blockers.append("paper_i_hubbard_snake_recalibration_phonon_cutoff_fields_must_be_blank")
    if str(row.get("paper_i_cutoff_ladder_stage") or "").strip():
        blockers.append("paper_i_hubbard_snake_recalibration_must_not_use_phonon_ladder_stage")
    blockers.extend(
        hubbard_recalibration.validate_candidate_row_authorization(
            row,
            target_case_id=target_case_id,
            repo_root=repo_root,
        )
    )
    payload = {
        "validation_status": "fail" if blockers else "pass",
        "target_case_id": target_case_id,
        "candidate_manifest_json": row.get("paper_i_hubbard_snake_recalibration_candidate_manifest_json"),
        "candidate_key": row.get("paper_i_hubbard_snake_recalibration_candidate_key"),
        "source_audit_json": row.get("paper_i_hubbard_snake_recalibration_source_audit_json"),
        "reason": row.get("paper_i_hubbard_snake_recalibration_reason"),
    }
    return blockers, payload


def build_phase3_preflight_manifest(
    row: Mapping[str, str],
    *,
    record_id: str,
    records_path: Path,
    submit_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Local CHTC fail-closed checks before a record is submitted.

    The checks intentionally duplicate the runner's benchmark selection and the
    submit file's transfer contract so missing provenance files and impossible
    benchmark IDs fail locally instead of burning queue cycles.
    """

    blockers: list[str] = []
    warnings: list[str] = []
    submit_contract: dict[str, Any] | None = None
    transfer_inputs: list[str] = []
    if submit_path is not None:
        submit_contract = parse_submit_contract(Path(submit_path))
        transfer_inputs = list(submit_contract.get("transfer_input_files") or [])

    records_path_resolved = _resolve_under_repo(records_path, repo_root=repo_root)
    if not records_path_resolved.exists():
        blockers.append(f"records_path_missing:{_safe_relative(records_path_resolved, repo_root=repo_root)}")
    if submit_contract is not None:
        submit_records = str(submit_contract.get("records_path") or "").strip()
        if submit_records and _safe_relative(records_path_resolved, repo_root=repo_root) != submit_records:
            blockers.append(
                "records_path_mismatch:"
                f"submit={submit_records}:preflight={_safe_relative(records_path_resolved, repo_root=repo_root)}"
            )
        if submit_records and not _is_sandbox_visible(submit_records, transfer_input_files=transfer_inputs, repo_root=repo_root):
            blockers.append(f"submit_records_not_transferred:{submit_records}")
        queue_file = str(submit_contract.get("queue_record_id_file") or "").strip()
        if queue_file:
            queue_path = _resolve_under_repo(queue_file, repo_root=repo_root)
            if not queue_path.exists():
                blockers.append(f"queue_record_id_file_missing:{queue_file}")
            elif not _is_sandbox_visible(queue_file, transfer_input_files=transfer_inputs, repo_root=repo_root):
                blockers.append(f"queue_record_id_file_not_transferred:{queue_file}")

    command: list[str] = []
    command_error = ""
    try:
        command = build_phase3_command(
            row,
            record_id=record_id,
            out_root=Path("raw_outputs") / "preflight" / record_id,
            run_root=Path("raw_outputs") / "preflight" / record_id / "run",
            progress_dir=Path("raw_outputs") / "preflight" / record_id / "progress",
        )
    except Exception as exc:
        command_error = f"{type(exc).__name__}: {exc}"
        blockers.append(f"command_build_failed:{command_error}")

    source_artifacts: list[dict[str, Any]] = []
    for field, value in _source_artifact_values(row):
        resolved = _resolve_under_repo(value, repo_root=repo_root)
        exists = resolved.exists()
        sandbox_visible = (
            None
            if not transfer_inputs
            else _is_sandbox_visible(value, transfer_input_files=transfer_inputs, repo_root=repo_root)
        )
        entry = {
            "field": field,
            "value": value,
            "resolved_path": str(resolved),
            "repo_relative_path": _safe_relative(resolved, repo_root=repo_root),
            "exists": bool(exists),
            "is_dir": bool(resolved.is_dir()) if exists else False,
            "sandbox_visible": sandbox_visible,
            "sha256": sha256_file(resolved) if exists and resolved.is_file() else None,
        }
        source_artifacts.append(entry)
        if not exists:
            blockers.append(f"source_artifact_missing:{field}:{value}")
        if sandbox_visible is False:
            blockers.append(f"source_artifact_not_transferred:{field}:{value}")

    benchmark_error = ""
    expected_specs: tuple[Any, ...] = ()
    expected_benchmark_ids: list[str] = []
    requested_benchmark_ids = _split_words(row.get("benchmark_ids", ""))
    try:
        expected_specs = _candidate_specs_for_row(row)
        expected_benchmark_ids = [str(spec.benchmark_id) for spec in expected_specs]
        allowed = set(expected_benchmark_ids)
        missing_ids = [bid for bid in requested_benchmark_ids if bid not in allowed]
        for bid in missing_ids:
            blockers.append(f"non_executable_benchmark_id:{bid}")
    except Exception as exc:
        benchmark_error = f"{type(exc).__name__}: {exc}"
        blockers.append(f"benchmark_resolution_failed:{benchmark_error}")

    if _truthy(row.get("oracle_require_compatible_warm_starts")):
        oracle_root = str(row.get("oracle_summary_root") or "").strip()
        if not oracle_root:
            blockers.append("oracle_compatible_warm_start_required_but_root_missing")
        elif expected_specs:
            try:
                import pipelines.static_adapt.optimization.phase3_policy_optuna as p3opt

                records_by_benchmark, skips_by_benchmark = p3opt.oracle_summary_warm_start_records_for_specs(
                    (oracle_root,),
                    tuple(expected_specs),
                    limit_per_benchmark=_positive_int_or_none(row.get("oracle_enqueue_limit")),
                    required_static_route_id=row.get("oracle_required_static_route_id"),
                    required_suite_profile=row.get("oracle_required_suite_profile"),
                    require_phase0_aware=_truthy(row.get("oracle_require_phase0_aware")),
                    require_compatible_warm_starts=True,
                )
                for spec in expected_specs:
                    if requested_benchmark_ids and str(spec.benchmark_id) not in set(requested_benchmark_ids):
                        continue
                    if not records_by_benchmark.get(str(spec.benchmark_id)):
                        reasons = sorted({str(skip.reason) for skip in skips_by_benchmark.get(str(spec.benchmark_id), ())})
                        detail = ",".join(reasons) if reasons else "no_summary_candidate"
                        blockers.append(f"oracle_compatible_warm_start_missing:{spec.benchmark_id}:{detail}")
            except Exception as exc:
                blockers.append(f"oracle_compatible_warm_start_preflight_failed:{type(exc).__name__}:{exc}")

    ladder_blockers, ladder_payload = _validate_routea_clean_ladder_preflight(
        row,
        record_id=record_id,
        expected_benchmark_ids=expected_benchmark_ids,
        expected_specs=expected_specs,
        transfer_inputs=transfer_inputs,
        submit_contract=submit_contract,
        repo_root=repo_root,
    )
    blockers.extend(ladder_blockers)
    cutoff_floor_blockers, cutoff_floor_payload = _paper_i_cutoff_floor_preflight(
        row,
        expected_specs=expected_specs,
    )
    blockers.extend(cutoff_floor_blockers)
    hubbard_recalibration_blockers, hubbard_recalibration_payload = _validate_hubbard_snake_recalibration_preflight(
        row,
        expected_benchmark_ids=expected_benchmark_ids,
        repo_root=repo_root,
    )
    blockers.extend(hubbard_recalibration_blockers)

    status = "fail" if blockers else "pass"
    command_text = " ".join(shlex.quote(str(part)) for part in command)
    return {
        "schema": "phase3_chtc_preflight_manifest_v1",
        "generated_utc": _now_utc(),
        "status": status,
        "ok": status == "pass",
        "record_id": record_id,
        "records_path": _safe_relative(records_path_resolved, repo_root=repo_root),
        "submit_contract": submit_contract,
        "transfer_input_files": transfer_inputs,
        "blocking_reasons": blockers,
        "warnings": warnings,
        "source_artifacts": source_artifacts,
        "paper_i_cutoff_ladder": ladder_payload,
        "paper_i_cutoff_floor": cutoff_floor_payload,
        "paper_i_hubbard_snake_recalibration": hubbard_recalibration_payload,
        "benchmark_selection": {
            "calibration_profile": row.get("calibration_profile"),
            "families": _split_words(row.get("families", "")),
            "sizes": _split_words(row.get("sizes", "")),
            "boson_cutoff": row.get("boson_cutoff"),
            "boson_cutoffs": _split_words(row.get("boson_cutoffs", "")),
            "exact_reference_boson_cutoff": row.get("exact_reference_boson_cutoff"),
            "physics_grid_profile": row.get("physics_grid_profile"),
            "requested_benchmark_ids": requested_benchmark_ids,
            "expected_benchmark_ids": expected_benchmark_ids,
            "resolution_error": benchmark_error,
        },
        "command": command,
        "command_sha256": hashlib.sha256(command_text.encode("utf-8")).hexdigest() if command else None,
        "command_error": command_error,
    }


def _validate_hh_route_faithfulness_trial_overrides(row: Mapping[str, str], *, record_id: str) -> None:
    if str(row.get("route_faithfulness_ladder_schema") or "") != HH_ROUTE_FAITHFULNESS_LADDER_SCHEMA:
        return
    if not str(row.get("trial_param_overrides_json") or "").strip():
        raise ValueError(
            f"{record_id}: HH route-faithfulness rows must carry trial_param_overrides_json "
            "so Optuna-sampled static policy fields are clamped to the source-lock route"
        )
    if not str(row.get("source_lock_trial_param_overrides_sha256") or "").strip():
        raise ValueError(f"{record_id}: HH route-faithfulness rows must record source_lock_trial_param_overrides_sha256")


def _validate_hh_novelty_surface_trial_overrides(row: Mapping[str, str], *, record_id: str) -> None:
    if str(row.get("policy_search_profile") or "").strip().lower().replace("-", "_") != "hh_novelty_surface_v1":
        return
    override_value = str(row.get("trial_param_overrides_json") or "").strip()
    if not override_value:
        raise ValueError(f"{record_id}: hh_novelty_surface_v1 rows must carry hard-bound trial_param_overrides_json")
    override_path = _resolve_under_repo(override_value)
    if not override_path.exists():
        raise ValueError(f"{record_id}: trial_param_overrides_json_missing:{override_value}")
    payload = _read_json_object(override_path)
    raw_fields = payload.get("trial_param_overrides", payload)
    if not isinstance(raw_fields, Mapping):
        raise ValueError(f"{record_id}: trial_param_overrides payload must be an object")
    sampled_fields = {
        "novelty_bonus",
        "phase2_motif_bonus_weight",
        "phase2_gamma_N",
        "phase2_gamma_N_schedule_mode",
        "phase2_gamma_N_schedule_start",
        "phase2_gamma_N_schedule_end",
    }
    forbidden = sorted(field for field in sampled_fields if field in raw_fields)
    if forbidden:
        raise ValueError(
            f"{record_id}: hh_novelty_surface_v1 forbids sampled novelty fields in trial_param_overrides_json: "
            + ",".join(forbidden)
        )
    if str(raw_fields.get("phase3_novelty_ablation_mode", "off")).strip().lower() != "off":
        raise ValueError(f"{record_id}: hh_novelty_surface_v1 requires phase3_novelty_ablation_mode=off")


def _validate_hh_noise_target_hit_predecessor_evidence(row: Mapping[str, str], *, record_id: str, rung_id: str) -> None:
    hh_noise_evidence = _hh_noise_evidence_module()
    status = str(row.get("noise_target_hit_pass_evidence_status") or "").strip()
    if status != hh_noise_evidence.NOISE_TARGET_HIT_PASS_EVIDENCE_STATUS:
        raise ValueError(
            f"{record_id}: HH noise rung {rung_id} uses phase3_oracle_inner_objective_mode=noisy_v1, "
            "which remains blocked until same-noise target-hit evidence validates; "
            f"status={status or 'missing'}; reason={HH_BLOCKED_NOISE_RUNG_REASON}"
        )
    if str(row.get("noise_target_hit_pass_evidence_schema") or "").strip() != hh_noise_evidence.NOISE_TARGET_HIT_PASS_EVIDENCE_SCHEMA:
        raise ValueError(f"{record_id}: HH target-hit evidence schema mismatch")
    if str(row.get("noise_target_hit_pass_evidence_validation_errors") or "").strip():
        raise ValueError(f"{record_id}: HH target-hit evidence carries validation errors")
    if not str(row.get("noise_target_hit_pass_evidence_json") or "").strip():
        raise ValueError(f"{record_id}: HH target-hit evidence path missing")
    if not str(row.get("noise_target_hit_pass_evidence_sha256") or "").strip():
        raise ValueError(f"{record_id}: HH target-hit evidence sha256 missing")
    evidence_rung = str(row.get("noise_target_hit_pass_evidence_rung_id") or "").strip()
    required_predecessors = set(_split_words(row.get("ladder_rung_required_predecessors", "")))
    if evidence_rung not in {pred for pred in required_predecessors if pred.endswith("_target_hit")}:
        raise ValueError(
            f"{record_id}: HH target-hit evidence rung {evidence_rung!r} is not a required target-hit predecessor"
        )
    evidence_case = str(row.get("noise_target_hit_pass_evidence_case_id") or "").strip()
    row_case = str(row.get("benchmark_ids") or "").strip()
    if evidence_case != row_case:
        raise ValueError(f"{record_id}: HH target-hit evidence case mismatch: {evidence_case!r}!={row_case!r}")
    if str(row.get("noise_target_hit_pass_evidence_stop_reason") or "").strip() != "benchmark_abs_delta_e_target":
        raise ValueError(f"{record_id}: HH target-hit evidence did not stop by benchmark_abs_delta_e_target")
    row_n_eff = str(row.get("noise_n_eff") or "").strip()
    if row_n_eff and row_n_eff != "off":
        evidence_n_eff = str(row.get("noise_target_hit_pass_evidence_n_eff") or "").strip()
        if not math.isclose(float(row_n_eff), float(evidence_n_eff), rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"{record_id}: HH target-hit evidence N_eff mismatch: {evidence_n_eff}!={row_n_eff}")
        row_sigma0 = str(row.get("noise_sigma0_abs") or "").strip()
        evidence_sigma0 = str(row.get("noise_target_hit_pass_evidence_sigma0_abs") or "").strip()
        if not math.isclose(float(row_sigma0), float(evidence_sigma0), rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"{record_id}: HH target-hit evidence sigma0_abs mismatch: {evidence_sigma0}!={row_sigma0}")


def _validate_hh_route_faithfulness_ladder_semantics(row: Mapping[str, str], *, record_id: str) -> None:
    schema = str(row.get("route_faithfulness_ladder_schema") or "").strip()
    rung_id = str(row.get("ladder_rung_id") or "").strip()
    row_record_id = str(row.get("record_id") or record_id or "").strip()
    is_hh_ladder_row = (
        schema == HH_ROUTE_FAITHFULNESS_LADDER_SCHEMA
        or rung_id in HH_ROUTE_FAITHFULNESS_RUNG_IDS
        or row_record_id.startswith("routeA_paper_i_hh_noise_robustness")
    )
    if not is_hh_ladder_row:
        return
    inner_objective_mode = str(row.get("phase3_oracle_inner_objective_mode") or "").strip()
    if inner_objective_mode == HH_BLOCKED_SELECTED_ENERGY_INNER_OBJECTIVE_MODE and rung_id not in HH_SELECTED_ENERGY_DIAGNOSTIC_ZERO_NOISE_RUNG_IDS:
        if rung_id not in HH_NOISY_V1_TARGET_HIT_RUNG_IDS or str(row.get("ladder_stage") or "") != "noise_surface_target_hit":
            _validate_hh_noise_target_hit_predecessor_evidence(row, record_id=record_id, rung_id=rung_id)
        hh_noise_evidence = _hh_noise_evidence_module()
        try:
            hh_noise_evidence.validate_evidence_row_provenance(
                row,
                baseline_reference_json=str(row.get("source_lock_reference_json") or "").strip() or None,
                repo_root=REPO_ROOT,
            )
        except hh_noise_evidence.EvidenceValidationError as exc:
            raise ValueError(
                f"{record_id}: HH noise rung {rung_id} uses phase3_oracle_inner_objective_mode=noisy_v1, "
                "which remains blocked because the zero-noise selected-energy oracle control failed route-faithfulness before guard-fix validation; "
                "selected-energy zero-noise pass evidence is required and must validate; "
                f"reason={HH_BLOCKED_NOISE_RUNG_REASON}; evidence_error={exc}"
            ) from exc
    if rung_id in HH_SELECTED_ENERGY_DIAGNOSTIC_ZERO_NOISE_RUNG_IDS and schema != HH_ROUTE_FAITHFULNESS_LADDER_SCHEMA:
        raise ValueError(f"{record_id}: {rung_id} diagnostic row must carry HH route-faithfulness ladder schema")
    predecessor_ids = set(_split_words(row.get("ladder_rung_required_predecessors", "")))
    forbidden_predecessors = sorted(predecessor_ids & HH_SELECTED_ENERGY_DIAGNOSTIC_ZERO_NOISE_RUNG_IDS)
    if forbidden_predecessors:
        raise ValueError(
            f"{record_id}: selected-energy diagnostic zero-noise rungs must not be prerequisites: "
            f"{forbidden_predecessors}"
        )
    if rung_id not in HH_SELECTED_ENERGY_DIAGNOSTIC_ZERO_NOISE_RUNG_IDS:
        return
    if str(row.get("route_faithfulness_prerequisite") or "").strip().lower() != "false":
        raise ValueError(f"{record_id}: {rung_id} is diagnostic-only and must explicitly set route_faithfulness_prerequisite=false")
    if str(row.get("target_hit_required") or "").strip().lower() != "false":
        raise ValueError(f"{record_id}: {rung_id} is diagnostic-only and must explicitly set target_hit_required=false")
    if str(row.get("ladder_stage") or "") != "pre_noise_diagnostic":
        raise ValueError(f"{record_id}: {rung_id} must use ladder_stage=pre_noise_diagnostic")
    if str(row.get("route_faithfulness_prerequisite_status") or "") != "diagnostic_only_not_route_faithfulness_prerequisite":
        raise ValueError(f"{record_id}: {rung_id} must declare diagnostic-only prerequisite status")
    if not str(row.get("route_faithfulness_prerequisite_reason") or "").strip():
        raise ValueError(f"{record_id}: {rung_id} must explain why it is not a prerequisite")


def _is_spin_boson_route_faithfulness_row(row: Mapping[str, str], *, record_id: str) -> bool:
    schema = str(row.get("route_faithfulness_ladder_schema") or "").strip()
    rung_id = str(row.get("ladder_rung_id") or "").strip()
    row_record_id = str(row.get("record_id") or record_id or "").strip()
    return (
        schema == SPIN_BOSON_ROUTE_FAITHFULNESS_LADDER_SCHEMA
        or rung_id in {SPIN_BOSON_ZERO_NOISE_ADAPTIVE_RUNG_ID, SPIN_BOSON_SCALAR_VALUE_NOISE_RUNG_ID}
        or row_record_id.startswith("routeA_paper_i_spin_boson_noise_robustness")
    )


def _validate_spin_boson_route_faithfulness_trial_overrides(row: Mapping[str, str], *, record_id: str) -> None:
    if not _is_spin_boson_route_faithfulness_row(row, record_id=record_id):
        return
    if str(row.get("route_faithfulness_ladder_schema") or "").strip() != SPIN_BOSON_ROUTE_FAITHFULNESS_LADDER_SCHEMA:
        raise ValueError(f"{record_id}: spin-boson route-faithfulness rows must carry the spin-boson ladder schema")
    if not str(row.get("trial_param_overrides_json") or "").strip():
        raise ValueError(
            f"{record_id}: spin-boson route-faithfulness rows must carry trial_param_overrides_json "
            "so the visible trial 0737 static policy is source-locked"
        )
    if not str(row.get("source_lock_trial_param_overrides_sha256") or "").strip():
        raise ValueError(
            f"{record_id}: spin-boson route-faithfulness rows must record source_lock_trial_param_overrides_sha256"
        )


def _require_spin_boson_field(row: Mapping[str, str], *, record_id: str, field: str, expected: str) -> None:
    if str(row.get(field) or "").strip() != str(expected):
        raise ValueError(f"{record_id}: {field} must be {expected!r} for spin-boson source-locked noise rows")


def _require_spin_boson_zero_float(row: Mapping[str, str], *, record_id: str, field: str) -> None:
    value = _float_field(row, field)
    if value != 0.0:
        raise ValueError(f"{record_id}: {field} must be exactly 0.0 for spin-boson source-locked noise rows")


def _validate_spin_boson_route_faithfulness_ladder_semantics(row: Mapping[str, str], *, record_id: str) -> None:
    if not _is_spin_boson_route_faithfulness_row(row, record_id=record_id):
        return
    _require_spin_boson_field(row, record_id=record_id, field="route_faithfulness_ladder_schema", expected=SPIN_BOSON_ROUTE_FAITHFULNESS_LADDER_SCHEMA)
    _require_spin_boson_field(row, record_id=record_id, field="families", expected="spin_boson")
    _require_spin_boson_field(row, record_id=record_id, field="phase3_oracle_gradient_mode", expected="ideal")
    gradient_step = _float_field(row, "phase3_oracle_gradient_step")
    if gradient_step is None or gradient_step <= 0.0:
        raise ValueError(f"{record_id}: phase3_oracle_gradient_step must be finite and > 0 for spin-boson source-locked noise rows")
    _require_spin_boson_field(row, record_id=record_id, field="phase3_oracle_execution_surface", expected="expectation_v1")
    _require_spin_boson_field(row, record_id=record_id, field="phase3_oracle_inner_objective_mode", expected="noisy_v1")
    if str(row.get("phase3_oracle_backend_name") or "").strip():
        raise ValueError(f"{record_id}: spin-boson source-locked rows must not set phase3_oracle_backend_name")
    if _truthy(row.get("phase3_oracle_use_fake_backend")):
        raise ValueError(f"{record_id}: spin-boson source-locked rows must not use fake backend metadata")
    _require_spin_boson_zero_float(row, record_id=record_id, field="phase3_oracle_synthetic_depolarizing_1q_error")
    _require_spin_boson_zero_float(row, record_id=record_id, field="phase3_oracle_synthetic_depolarizing_2q_error")
    if str(row.get("phase3_oracle_synthetic_coherent_1q_angle_std") or "").strip():
        _require_spin_boson_zero_float(row, record_id=record_id, field="phase3_oracle_synthetic_coherent_1q_angle_std")
    if str(row.get("phase3_oracle_synthetic_coherent_2q_angle_std") or "").strip():
        _require_spin_boson_zero_float(row, record_id=record_id, field="phase3_oracle_synthetic_coherent_2q_angle_std")
    if str(row.get("phase3_oracle_synthetic_depolarizing_1q_gates") or "").strip():
        raise ValueError(f"{record_id}: spin-boson source-locked scalar lane must not set synthetic 1q gates")
    if str(row.get("phase3_oracle_synthetic_depolarizing_2q_gates") or "").strip():
        raise ValueError(f"{record_id}: spin-boson source-locked scalar lane must not set synthetic 2q gates")
    if str(row.get("phase3_oracle_synthetic_coherent_1q_gates") or "").strip():
        raise ValueError(f"{record_id}: spin-boson source-locked scalar lane must not set coherent 1q gates")
    if str(row.get("phase3_oracle_synthetic_coherent_2q_gates") or "").strip():
        raise ValueError(f"{record_id}: spin-boson source-locked scalar lane must not set coherent 2q gates")
    if str(row.get("physical_shots_unchanged") or "").strip().lower() != "true":
        raise ValueError(f"{record_id}: spin-boson scalar proxy rows must declare physical_shots_unchanged=true")
    if str(row.get("physical_shot_count_claimed") or "").strip().lower() != "false":
        raise ValueError(f"{record_id}: spin-boson scalar proxy rows must declare physical_shot_count_claimed=false")
    if str(row.get("fixed_gate_error_reduction_claimed") or "").strip().lower() != "false":
        raise ValueError(f"{record_id}: spin-boson scalar proxy rows must declare fixed_gate_error_reduction_claimed=false")
    rung_id = str(row.get("ladder_rung_id") or "").strip()
    if rung_id == SPIN_BOSON_ZERO_NOISE_ADAPTIVE_RUNG_ID:
        _require_spin_boson_field(row, record_id=record_id, field="phase3_oracle_value_noise_model", expected="off")
        _require_spin_boson_zero_float(row, record_id=record_id, field="phase3_oracle_value_noise_std")
        if str(row.get("noise_n_eff") or "").strip() != "off":
            raise ValueError(f"{record_id}: zero-noise spin-boson control must use noise_n_eff=off")
        if str(row.get("route_faithfulness_prerequisite") or "").strip().lower() != "true":
            raise ValueError(f"{record_id}: zero-noise spin-boson control must be the route-faithfulness prerequisite")
        if str(row.get("target_hit_required") or "").strip().lower() != "true":
            raise ValueError(f"{record_id}: zero-noise spin-boson control must require target hit")
    elif rung_id == SPIN_BOSON_SCALAR_VALUE_NOISE_RUNG_ID:
        _require_spin_boson_field(row, record_id=record_id, field="phase3_oracle_value_noise_model", expected="gaussian_iid_v1")
        n_eff = _float_field(row, "phase3_oracle_value_noise_n_eff")
        std = _float_field(row, "phase3_oracle_value_noise_std")
        if n_eff is None or n_eff <= 0.0:
            raise ValueError(f"{record_id}: scalar spin-boson value-noise rows require positive N_eff")
        if std is None or std <= 0.0:
            raise ValueError(f"{record_id}: scalar spin-boson value-noise rows require positive value-noise std")
        if str(row.get("zero_noise_pass_evidence_status") or "").strip() != "pass":
            raise ValueError(f"{record_id}: scalar spin-boson value-noise rows require zero-noise pass evidence")
        if not str(row.get("zero_noise_pass_evidence_json") or "").strip():
            raise ValueError(f"{record_id}: scalar spin-boson value-noise rows require zero_noise_pass_evidence_json")
    else:
        raise ValueError(f"{record_id}: unknown spin-boson route-faithfulness rung {rung_id!r}")


def build_phase3_command(row: Mapping[str, str], *, record_id: str, out_root: Path, run_root: Path, progress_dir: Path) -> list[str]:
    _validate_canonical_class_record(row, record_id=record_id)
    _validate_hh_route_faithfulness_trial_overrides(row, record_id=record_id)
    _validate_hh_novelty_surface_trial_overrides(row, record_id=record_id)
    _validate_hh_route_faithfulness_ladder_semantics(row, record_id=record_id)
    _validate_spin_boson_route_faithfulness_trial_overrides(row, record_id=record_id)
    _validate_spin_boson_route_faithfulness_ladder_semantics(row, record_id=record_id)
    selected_recovery_blockers = validate_routea_selected_recovery_contract(row)
    if selected_recovery_blockers:
        raise ValueError(f"{record_id}: selected recovery validation failed: {selected_recovery_blockers}")
    static_route_id = static_route_id_from_record(
        row,
        record_id=record_id,
        fail_on_route_named_missing=True,
    )
    fixed_inner_optimizer = _fixed_inner_optimizer(row, static_route_id=static_route_id)
    fixed_phase2_novelty_mode = (row.get("phase2_novelty_mode") or "collective_span_v1").strip().lower()
    cmd: list[str] = [
        "python", "-u", "-m", "pipelines.static_adapt.optimization.phase3_policy_optuna",
        "--fixed-inner-optimizer", fixed_inner_optimizer,
        "--fixed-phase2-novelty-mode", fixed_phase2_novelty_mode,
        "--static-route-id", static_route_id,
        "--mode", row.get("mode", "oracle-grid") or "oracle-grid",
        "--output-dir", str(run_root),
        "--progress-dir", str(progress_dir),
        "--study-prefix", f"chtc_phase3_optuna/{record_id}",
        "--storage", f"sqlite:///{(out_root / 'study.sqlite3').resolve()}",
        "--telemetry-record-id", str(record_id),
    ]
    _add_option(cmd, "--trial-param-overrides-json", row.get("trial_param_overrides_json"))
    _add_option(cmd, "--enqueue-trial-params-json", row.get("enqueue_trial_params_json"))
    _add_option(cmd, "--canonical-lane", row.get("canonical_lane"))
    _add_option(cmd, "--canonical-lane-stage", row.get("canonical_lane_stage"))
    if "suite_profile" in row and not str(row.get("suite_profile") or "").strip():
        # phase3_policy_optuna defaults --suite-profile to "standard".  Noncanonical
        # repair records intentionally leave suite_profile blank and select specs via
        # family/cutoff filters, so pass an explicit empty value to suppress the
        # Table-I standard suite default.
        cmd.extend(["--suite-profile", ""])
    else:
        _add_option(cmd, "--suite-profile", row.get("suite_profile"))
    for benchmark_id in _split_words(row.get("benchmark_ids", "")):
        cmd.extend(["--benchmark-id", benchmark_id])
    _add_option(cmd, "--policy-search-profile", row.get("policy_search_profile"))
    _add_option(cmd, "--meta-feature-profile", row.get("meta_feature_profile"))
    _add_option(cmd, "--calibration-profile", row.get("calibration_profile"))
    _add_option(cmd, "--required-target-profile", row.get("required_target_profile"))
    _add_option(cmd, "--discovery-objective-mode", row.get("discovery_objective_mode"))
    for required_id in _split_words(row.get("required_target_benchmark_ids", "")):
        cmd.extend(["--required-target-benchmark-id", required_id])
    _add_option(cmd, "--required-target-abs-delta-e", row.get("required_target_abs_delta_e"))
    _add_option(cmd, "--required-target-penalty", row.get("required_target_penalty"))
    families = _split_words(row.get("families", ""))
    if families:
        cmd.extend(["--families", *families])
    sizes = _split_words(row.get("sizes", ""))
    if sizes:
        cmd.extend(["--sizes", *sizes])
    _add_option(cmd, "--boson-cutoff", row.get("boson_cutoff"))
    cutoffs = _split_words(row.get("boson_cutoffs", ""))
    if cutoffs:
        cmd.extend(["--boson-cutoffs", *cutoffs])
    _add_option(cmd, "--exact-reference-boson-cutoff", row.get("exact_reference_boson_cutoff"))
    if _truthy(row.get("force_same_cutoff_objective")):
        cmd.append("--force-same-cutoff-objective")
    _add_option(cmd, "--physics-grid-profile", row.get("physics_grid_profile"))
    _add_option(cmd, "--molecular-problem-json", row.get("molecular_problem_json"))
    _add_option(cmd, "--historical-ledger", row.get("historical_ledger"))
    _add_option(cmd, "--selected-logical-route", row.get("selected_logical_route"))
    _add_option(cmd, "--selected-logical-source-json", row.get("selected_logical_source_json"))
    _add_option(cmd, "--selected-logical-transfer-mode", row.get("selected_logical_transfer_mode"))
    _add_option(cmd, "--oracle-summary-root", row.get("oracle_summary_root"))
    _add_option(cmd, "--oracle-enqueue-limit", row.get("oracle_enqueue_limit"))
    _add_option(cmd, "--objective-profile", row.get("objective_profile"))
    _add_option(cmd, "--target-abs-delta-e", row.get("target_abs_delta_e"))
    _add_option(cmd, "--objective-energy-weight", row.get("objective_energy_weight"))
    _add_option(cmd, "--objective-2q-weight", row.get("objective_2q_weight"))
    _add_option(cmd, "--objective-2q-depth-weight", _objective_2q_depth_weight(row))
    _add_option(cmd, "--objective-depth-weight", row.get("objective_depth_weight"))
    _add_option(cmd, "--objective-parameter-weight", row.get("objective_parameter_weight"))
    _add_option(cmd, "--objective-shot-weight", row.get("objective_shot_weight"))
    _add_option(cmd, "--objective-weight-preset", row.get("objective_weight_preset"))
    _add_option(cmd, "--objective-family-weights", row.get("objective_family_weights"))
    _add_option(cmd, "--objective-benchmark-weights", row.get("objective_benchmark_weights"))
    _add_option(cmd, "--hardware-resolution-mode", row.get("hardware_resolution_mode"))
    _add_option(cmd, "--hardware-resolution-profile-json", row.get("hardware_resolution_profile_json"))
    _add_option(cmd, "--hardware-resolution-profile-name", row.get("hardware_resolution_profile_name"))
    _add_option(cmd, "--phase3-adapt-parallel-gradient-workers", row.get("phase3_adapt_parallel_gradient_workers"))
    _add_option(cmd, "--phase3-adapt-beam-parent-workers", row.get("phase3_adapt_beam_parent_workers"))
    _add_option(cmd, "--phase3-oracle-gradient-mode", row.get("phase3_oracle_gradient_mode"))
    _add_option(cmd, "--phase3-oracle-backend-name", row.get("phase3_oracle_backend_name"))
    if _truthy(row.get("phase3_oracle_use_fake_backend")):
        cmd.append("--phase3-oracle-use-fake-backend")
    _add_option(cmd, "--phase3-oracle-shots", row.get("phase3_oracle_shots"))
    _add_option(cmd, "--phase3-oracle-repeats", row.get("phase3_oracle_repeats"))
    _add_option(cmd, "--phase3-oracle-aggregate", row.get("phase3_oracle_aggregate"))
    _add_option(cmd, "--phase3-oracle-seed", row.get("phase3_oracle_seed"))
    _add_option(cmd, "--phase3-oracle-gradient-step", row.get("phase3_oracle_gradient_step"))
    _add_option(cmd, "--phase3-oracle-execution-surface", row.get("phase3_oracle_execution_surface"))
    _add_option(cmd, "--phase3-oracle-inner-objective-mode", row.get("phase3_oracle_inner_objective_mode"))
    _add_option(cmd, "--phase3-oracle-value-noise-model", row.get("phase3_oracle_value_noise_model"))
    _add_option(cmd, "--phase3-oracle-value-noise-std", row.get("phase3_oracle_value_noise_std"))
    _add_option(cmd, "--phase3-oracle-value-noise-seed", row.get("phase3_oracle_value_noise_seed"))
    _add_option(cmd, "--phase3-oracle-value-noise-sigma0-abs", row.get("phase3_oracle_value_noise_sigma0_abs"))
    _add_option(cmd, "--phase3-oracle-value-noise-n-eff", row.get("phase3_oracle_value_noise_n_eff"))
    _add_option(cmd, "--phase3-oracle-synthetic-depolarizing-1q-error", row.get("phase3_oracle_synthetic_depolarizing_1q_error"))
    _add_option(cmd, "--phase3-oracle-synthetic-depolarizing-2q-error", row.get("phase3_oracle_synthetic_depolarizing_2q_error"))
    _add_option(
        cmd,
        "--phase3-oracle-synthetic-depolarizing-1q-gates",
        _canonical_gate_cli_value(
            row.get("phase3_oracle_synthetic_depolarizing_1q_gates"),
            field_name="phase3_oracle_synthetic_depolarizing_1q_gates",
        ),
    )
    _add_option(
        cmd,
        "--phase3-oracle-synthetic-depolarizing-2q-gates",
        _canonical_gate_cli_value(
            row.get("phase3_oracle_synthetic_depolarizing_2q_gates"),
            field_name="phase3_oracle_synthetic_depolarizing_2q_gates",
        ),
    )
    _add_option(cmd, "--phase3-oracle-synthetic-coherent-1q-angle-std", row.get("phase3_oracle_synthetic_coherent_1q_angle_std"))
    _add_option(cmd, "--phase3-oracle-synthetic-coherent-2q-angle-std", row.get("phase3_oracle_synthetic_coherent_2q_angle_std"))
    _add_option(cmd, "--phase3-oracle-synthetic-coherent-seed", row.get("phase3_oracle_synthetic_coherent_seed"))
    _add_option(cmd, "--phase3-oracle-synthetic-coherent-generator-mode", row.get("phase3_oracle_synthetic_coherent_generator_mode"))
    _add_option(
        cmd,
        "--phase3-oracle-synthetic-coherent-1q-gates",
        _canonical_gate_cli_value(
            row.get("phase3_oracle_synthetic_coherent_1q_gates"),
            field_name="phase3_oracle_synthetic_coherent_1q_gates",
        ),
    )
    _add_option(
        cmd,
        "--phase3-oracle-synthetic-coherent-2q-gates",
        _canonical_gate_cli_value(
            row.get("phase3_oracle_synthetic_coherent_2q_gates"),
            field_name="phase3_oracle_synthetic_coherent_2q_gates",
        ),
    )
    _add_option(cmd, "--phase3-oracle-value-noise-seed-policy", row.get("phase3_oracle_value_noise_seed_policy"))
    _add_option(cmd, "--phase3-oracle-value-noise-base-seed", row.get("phase3_oracle_value_noise_base_seed"))
    _add_option(cmd, "--phase3-oracle-value-noise-replicate-id", row.get("phase3_oracle_value_noise_replicate_id"))
    _add_option(cmd, "--adapt-noise-floor-stop-policy", row.get("adapt_noise_floor_stop_policy"))
    _add_option(cmd, "--adapt-noise-floor-snr-threshold", row.get("adapt_noise_floor_snr_threshold"))
    _add_option(cmd, "--adapt-noise-floor-n-rem-high-threshold", row.get("adapt_noise_floor_n_rem_high_threshold"))
    _add_option(cmd, "--adapt-noise-floor-useful-horizon-threshold", row.get("adapt_noise_floor_useful_horizon_threshold"))
    phase0_enabled = row.get("phase0_pilot_enabled")
    if str(phase0_enabled or "").strip():
        cmd.append("--phase0-no-pilot" if _falsey(phase0_enabled) else "--phase0-pilot-enabled")
    _add_option(cmd, "--phase0-pilot-alpha", row.get("phase0_pilot_alpha"))
    _add_option(cmd, "--phase0-pilot-threshold", row.get("phase0_pilot_threshold"))
    _add_option(cmd, "--phase0-pilot-max-records", row.get("phase0_pilot_max_records"))
    _add_option(cmd, "--phase0-lane-quota-pressure", row.get("phase0_lane_quota_pressure"))
    _add_option(cmd, "--phase0-algebraic-lane-mode", row.get("phase0_algebraic_lane_mode"))
    _add_option(cmd, "--phase1-score-z-alpha", row.get("phase1_score_z_alpha"))
    _add_option(cmd, "--phase2-score-z-alpha", row.get("phase2_score_z_alpha"))
    _add_option(cmd, "--phase3-selector-policy", row.get("phase3_selector_policy"))
    _add_option(cmd, "--phase3-selector-geometry-mode", row.get("phase3_selector_geometry_mode"))
    _add_option(cmd, "--phase3-novelty-ablation-mode", row.get("phase3_novelty_ablation_mode"))
    _add_option(cmd, "--phase3-window-relaxation-mode", row.get("phase3_window_relaxation_mode"))
    _add_option(cmd, "--phase3-batch-selection-mode", row.get("phase3_batch_selection_mode"))
    _add_option(cmd, "--phase3-batch-prefilter-mode", row.get("phase3_batch_prefilter_mode"))
    _add_option(cmd, "--phase2-rho", row.get("phase2_rho"))
    phase3_batching = row.get("phase3_enable_batching") or row.get("phase2_enable_batching")
    if str(phase3_batching or "").strip():
        cmd.append("--phase3-no-batching" if _falsey(phase3_batching) else "--phase3-enable-batching")
    phase1_prune = row.get("phase1_prune_enabled")
    if str(phase1_prune or "").strip():
        cmd.append("--phase1-no-prune" if _falsey(phase1_prune) else "--phase1-prune-enabled")
    adapt_allow_repeats = row.get("phase3_adapt_allow_repeats")
    if str(adapt_allow_repeats or "").strip():
        cmd.append("--adapt-no-repeats" if _falsey(adapt_allow_repeats) else "--adapt-allow-repeats")
    _add_option(cmd, "--oracle-required-static-route-id", row.get("oracle_required_static_route_id"))
    _add_option(cmd, "--oracle-required-suite-profile", row.get("oracle_required_suite_profile"))
    if _truthy(row.get("oracle_require_phase0_aware")):
        cmd.append("--oracle-require-phase0-aware")
    if _truthy(row.get("oracle_require_compatible_warm_starts")):
        cmd.append("--oracle-require-compatible-warm-starts")
    _add_option(cmd, "--robustness-gate", row.get("robustness_gate"))
    lanes = _split_words(row.get("robustness_gate_lanes", ""))
    if lanes:
        cmd.extend(["--robustness-gate-lanes", *lanes])
    _add_option(cmd, "--robustness-gate-target-abs-delta-e", row.get("robustness_gate_target_abs_delta_e"))
    _add_option(cmd, "--n-trials", row.get("n_trials"))
    _add_option(cmd, "--n-jobs", row.get("n_jobs"))
    _add_option(cmd, "--benchmarks-per-trial-jobs", row.get("benchmarks_per_trial_jobs"))
    _add_option(cmd, "--seed", row.get("seed"))
    _add_option(cmd, "--trial-timeout-sec", row.get("trial_timeout_sec"))
    _add_option(cmd, "--trial-prune-depth", row.get("trial_prune_depth"))
    _add_option(cmd, "--trial-prune-abs-delta-e", row.get("trial_prune_abs_delta_e"))
    _add_option(cmd, "--trial-prune-metric", row.get("trial_prune_metric"))
    _add_option(cmd, "--compile-timeout-sec", row.get("compile_timeout_sec"))
    if str(row.get("enqueue_default", "true")).strip().lower() in {"0", "false", "no"}:
        cmd.append("--no-enqueue-default")
    if str(row.get("enqueue_historical", "true")).strip().lower() in {"0", "false", "no"}:
        cmd.append("--no-enqueue-historical")
    return cmd


def _phase3_env(row: Mapping[str, str]) -> dict[str, str]:
    static_route_id = static_route_id_from_record(row)
    fixed_inner_optimizer = _fixed_inner_optimizer(row, static_route_id=static_route_id)
    fixed_phase2_novelty_mode = (row.get("phase2_novelty_mode") or "collective_span_v1").strip().lower()
    env = os.environ.copy()
    env["PHASE3_POLICY_INNER_OPTIMIZER"] = fixed_inner_optimizer
    env["PHASE3_POLICY_PHASE2_NOVELTY_MODE"] = fixed_phase2_novelty_mode
    env["PHASE3_POLICY_STATIC_ROUTE_ID"] = static_route_id
    if str(row.get("record_id") or "").strip():
        env["PHASE3_RECORD_ID"] = str(row.get("record_id") or "").strip()
    return env


def _record_meta_feature_profile(row: Mapping[str, str]) -> tuple[str, str]:
    raw = str(row.get("meta_feature_profile") or "").strip()
    if raw:
        return normalize_static_meta_feature_profile(raw), "row_field"
    return STATIC_META_FEATURE_PROFILE_SAFE_CORE_V1, "legacy_missing_row_field_default"


def _meta_feature_policy_contract(profile: str) -> dict[str, Any]:
    if profile == STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1:
        return {
            "schema": "phase3_meta_feature_policy_contract_v1",
            "profile": profile,
            "production_contract": "paper_i_clean_table_i_static_snake_v1",
            "candidate_position_route": "r=(m,p)",
            "post_score_position_prior": "forbidden",
            "hard_identity_features": {
                "phase0_pilot_enabled": True,
                "algebraic_lanes_enabled": True,
                "phase3_selector_policy": "algebraic_nested_v1",
                "phase3_selector_geometry_mode": "reduced",
                "phase3_novelty_ablation_mode": "off",
                "phase3_window_relaxation_mode": "reduced",
                "phase1_prune_enabled": True,
                "phase1_prune_amplitude_witness_required": True,
                "phase_live_hysteresis_enabled": False,
            },
            "forced_zero_terms": {
                "compile_position_shift_weight": 0.0,
                "family_repeat_penalty": 0.0,
                "motif_bonus_weight": 0.0,
            },
            "feature_bundle_may_toggle": [
                "phase3_batching_enabled",
                "adapt_allow_repeats",
            ],
            "tunable_pressure_knobs": [
                "position_domain_and_shortlist_caps",
                "phase0_scalar_thresholds",
                "algebraic_lane_thresholds_and_quotas",
                "frontier_ratios",
                "batch_caps_and_gates",
                "prune_pressures_and_tolerances",
                "optimizer_and_refit_budgets",
            ],
        }
    return {
        "schema": "phase3_meta_feature_policy_contract_v1",
        "profile": profile,
        "safe_core_preserves_hard_snake_identity": True,
        "feature_bundle_may_toggle": [
            "phase3_batching_enabled",
            "phase1_prune_enabled",
            "phase1_prune_amplitude_witness_required",
            "phase0_pilot_enabled",
            "adapt_allow_repeats",
            "adapt_reopt_policy",
            "adapt_insertion_mode",
            "beam_budget_knobs",
        ],
    }


def _route_faithfulness_ladder_manifest(row: Mapping[str, str]) -> dict[str, Any] | None:
    schema = str(row.get("route_faithfulness_ladder_schema") or "").strip()
    if not schema:
        return None
    return {
        "schema": schema,
        "rung_id": row.get("ladder_rung_id"),
        "rung_order": _positive_int_or_none(row.get("ladder_rung_order")),
        "stage": row.get("ladder_stage"),
        "generation_gate": row.get("ladder_rung_gate"),
        "required_predecessors": _split_words(row.get("ladder_rung_required_predecessors", "")),
        "same_noise_surface_key": row.get("same_noise_surface_key"),
        "oracle_surface_class": row.get("oracle_surface_class"),
        "noise_surface_class": row.get("noise_surface_class"),
        "target_hit_required": _truthy(row.get("target_hit_required")),
        "target_hit_success_stop_reason": row.get("target_hit_success_stop_reason"),
        "route_faithfulness_prerequisite": _truthy(row.get("route_faithfulness_prerequisite")),
        "route_faithfulness_prerequisite_status": row.get("route_faithfulness_prerequisite_status"),
        "route_faithfulness_prerequisite_reason": row.get("route_faithfulness_prerequisite_reason"),
        "diagnostic_non_hit_allowed": _truthy(row.get("diagnostic_non_hit_allowed")),
        "diagnostic_non_hit_stop_reasons": _split_words(row.get("diagnostic_non_hit_stop_reasons", "")),
        "pre_run_rung_pass_status": "not_evaluated",
        "rung_pass_source": "record_manifest_pre_run_not_completed",
        "original_route_knobs_preserved": _truthy(row.get("original_route_knobs_preserved")),
        "confidence_score_penalties_changed": _truthy(row.get("confidence_score_penalties_changed")),
        "confidence_scoring_semantic": row.get("confidence_scoring_semantic"),
        "stop_policy": row.get("adapt_noise_floor_stop_policy"),
        "route_preservation_fields": {
            "phase1_score_z_alpha": row.get("phase1_score_z_alpha"),
            "phase2_score_z_alpha": row.get("phase2_score_z_alpha"),
        },
        "source_lock": {
            "status": row.get("source_lock_status"),
            "baseline_contract_status": row.get("baseline_contract_status"),
            "reference_json": row.get("source_lock_reference_json"),
            "reference_sha256": row.get("source_lock_reference_sha256"),
            "command_json": row.get("source_lock_command_json"),
            "command_sha256": row.get("source_lock_command_sha256"),
            "trial_param_overrides_json": row.get("source_lock_trial_param_overrides_json") or row.get("trial_param_overrides_json"),
            "trial_param_overrides_sha256": row.get("source_lock_trial_param_overrides_sha256"),
            "trial_param_override_field_count": _positive_int_or_none(row.get("source_lock_trial_param_override_field_count")),
            "trial_param_override_fields": _split_words(row.get("source_lock_trial_param_override_fields", "")),
            "case_id": row.get("source_lock_case_id"),
            "stop_reason": row.get("source_lock_stop_reason"),
            "energy": row.get("source_lock_energy"),
            "abs_delta_e": row.get("source_lock_abs_delta_e"),
            "ansatz_depth": row.get("source_lock_ansatz_depth"),
            "n_ph_work": row.get("source_lock_n_ph_work"),
            "n_ph_ref": row.get("source_lock_n_ph_ref"),
            "validation_errors": [
                part
                for part in str(row.get("source_lock_validation_errors") or "").split(";")
                if part
            ],
            "tableiii_source_map_json": row.get("source_lock_tableiii_source_map_json"),
            "tableiii_provenance_txt": row.get("source_lock_tableiii_provenance_txt"),
        },
        "value_noise_contract": {
            "semantic": row.get("phase3_oracle_value_noise_semantic") or row.get("noise_semantic"),
            "model": row.get("phase3_oracle_value_noise_model") or "off",
            "std": _float_field(row, "phase3_oracle_value_noise_std"),
            "sigma0_abs": _float_field(row, "phase3_oracle_value_noise_sigma0_abs"),
            "N_eff": _float_field(row, "phase3_oracle_value_noise_n_eff"),
            "seed": row.get("phase3_oracle_value_noise_seed"),
            "seed_policy": row.get("phase3_oracle_value_noise_seed_policy"),
            "base_seed": row.get("phase3_oracle_value_noise_base_seed"),
            "replicate_id": row.get("phase3_oracle_value_noise_replicate_id"),
            "zero_noise_off_model_required": str(row.get("noise_surface_class") or "") == "zero_noise",
        },
        "synthetic_depolarizing_contract": {
            "gradient_mode": row.get("phase3_oracle_gradient_mode") or "off",
            "gradient_step": _float_field(row, "phase3_oracle_gradient_step"),
            "one_qubit_error": _float_field(row, "phase3_oracle_synthetic_depolarizing_1q_error"),
            "two_qubit_error": _float_field(row, "phase3_oracle_synthetic_depolarizing_2q_error"),
            "one_qubit_gates": _gate_list_values(
                row.get("phase3_oracle_synthetic_depolarizing_1q_gates"),
                field_name="phase3_oracle_synthetic_depolarizing_1q_gates",
            ),
            "two_qubit_gates": _gate_list_values(
                row.get("phase3_oracle_synthetic_depolarizing_2q_gates"),
                field_name="phase3_oracle_synthetic_depolarizing_2q_gates",
            ),
        },
        "synthetic_coherent_contract": {
            "gradient_mode": row.get("phase3_oracle_gradient_mode") or "off",
            "gradient_step": _float_field(row, "phase3_oracle_gradient_step"),
            "one_qubit_angle_std": _float_field(row, "phase3_oracle_synthetic_coherent_1q_angle_std"),
            "two_qubit_angle_std": _float_field(row, "phase3_oracle_synthetic_coherent_2q_angle_std"),
            "seed": row.get("phase3_oracle_synthetic_coherent_seed"),
            "generator_mode": row.get("phase3_oracle_synthetic_coherent_generator_mode"),
            "one_qubit_gates": _gate_list_values(
                row.get("phase3_oracle_synthetic_coherent_1q_gates"),
                field_name="phase3_oracle_synthetic_coherent_1q_gates",
            ),
            "two_qubit_gates": _gate_list_values(
                row.get("phase3_oracle_synthetic_coherent_2q_gates"),
                field_name="phase3_oracle_synthetic_coherent_2q_gates",
            ),
        },
        "selected_energy_zero_noise_pass_evidence": {
            "schema": row.get("selected_energy_zero_noise_pass_evidence_schema"),
            "status": row.get("selected_energy_zero_noise_pass_evidence_status"),
            "evidence_json": row.get("selected_energy_zero_noise_pass_evidence_json"),
            "evidence_sha256": row.get("selected_energy_zero_noise_pass_evidence_sha256"),
            "baseline_json": row.get("selected_energy_zero_noise_pass_evidence_baseline_json"),
            "baseline_sha256": row.get("selected_energy_zero_noise_pass_evidence_baseline_sha256"),
            "case_id": row.get("selected_energy_zero_noise_pass_evidence_case_id"),
            "stop_reason": row.get("selected_energy_zero_noise_pass_evidence_stop_reason"),
            "energy": row.get("selected_energy_zero_noise_pass_evidence_energy"),
            "abs_delta_e": row.get("selected_energy_zero_noise_pass_evidence_abs_delta_e"),
            "ansatz_depth": row.get("selected_energy_zero_noise_pass_evidence_ansatz_depth"),
            "operator_count": row.get("selected_energy_zero_noise_pass_evidence_operator_count"),
            "sequence_sha256": row.get("selected_energy_zero_noise_pass_evidence_sequence_sha256"),
            "baseline_sequence_sha256": row.get("selected_energy_zero_noise_pass_evidence_baseline_sequence_sha256"),
            "requested_inner_objective_mode": row.get(
                "selected_energy_zero_noise_pass_evidence_requested_inner_objective_mode"
            ),
            "effective_inner_objective_mode": row.get(
                "selected_energy_zero_noise_pass_evidence_effective_inner_objective_mode"
            ),
            "runtime_guard_reason": row.get("selected_energy_zero_noise_pass_evidence_runtime_guard_reason"),
            "validation_errors": [
                part
                for part in str(row.get("selected_energy_zero_noise_pass_evidence_validation_errors") or "").split(";")
                if part
            ],
        },
    }


def write_record_manifest(row: Mapping[str, str], *, record_id: str, out_root: Path, run_root: Path, progress_dir: Path, command: Sequence[str]) -> None:
    _validate_hh_route_faithfulness_ladder_semantics(row, record_id=record_id)
    _validate_spin_boson_route_faithfulness_ladder_semantics(row, record_id=record_id)
    static_route_id = static_route_id_from_record(
        row,
        record_id=record_id,
        fail_on_route_named_missing=True,
    )
    meta_feature_profile, meta_feature_profile_source = _record_meta_feature_profile(row)
    route_identity = route_identity_payload(
        {
            "base_pool_key": row.get("route_base_pool_key") or row.get("pool_key"),
            "continuation_mode": row.get("continuation_mode"),
            "phase2_novelty_mode": row.get("phase2_novelty_mode"),
            "phase3_selector_policy": row.get("phase3_selector_policy"),
            "phase3_selector_geometry_mode": row.get("phase3_selector_geometry_mode"),
            "algebraic_shortlisting_enabled": row.get("algebraic_shortlisting_enabled"),
            "hardware_resolution_schema": row.get("hardware_resolution_schema") or "gradient_resolution_v1",
            "hardware_resolution_mode": row.get("hardware_resolution_mode") or "ideal",
            "phase2_raw_score_formula": row.get("phase2_raw_score_formula"),
            "canonical_score_formula": row.get("canonical_score_formula"),
            "primary_selector_score_key": row.get("primary_selector_score_key"),
            "auxiliary_terms_primary_mode": row.get("auxiliary_terms_primary_mode"),
            "phase3_novelty_ablation_mode": row.get("phase3_novelty_ablation_mode"),
            "phase3_window_relaxation_mode": row.get("phase3_window_relaxation_mode"),
            "phase2_enable_batching": row.get("phase2_enable_batching"),
            "phase3_enable_batching": row.get("phase3_enable_batching") or row.get("phase2_enable_batching"),
            "phase3_batch_selection_mode": row.get("phase3_batch_selection_mode") or row.get("phase2_batch_selection_mode") or "reduced_plane",
            "phase3_batch_prefilter_mode": row.get("phase3_batch_prefilter_mode") or "off",
            "phase3_nested_window_application": (
                row.get("phase3_nested_window_application")
                or (
                    "composed_batch_window_v1"
                    if str(row.get("phase3_selector_policy") or "").strip() == "algebraic_nested_v1"
                    else "legacy_reopt_policy"
                )
            ),
            "phase1_prune_enabled": row.get("phase1_prune_enabled"),
            "phase1_prune_policy": row.get("phase1_prune_policy"),
            "phase1_prune_mode": row.get("phase1_prune_mode"),
            "phase1_prune_amplitude_witness_required": row.get("phase1_prune_amplitude_witness_required"),
            "meta_feature_profile": meta_feature_profile,
        },
        declared_route_id=static_route_id,
        optimizer_lane=_fixed_inner_optimizer(row, static_route_id=static_route_id),
    )
    if normalize_static_route_id(static_route_id) != ROUTE_ID_UNSPECIFIED and not bool(route_identity.get("valid", False)):
        raise ValueError(
            f"record {record_id!r} declares static_route_id={static_route_id!r} but route identity is invalid: "
            + "; ".join(str(x) for x in route_identity.get("noncanonical_reasons", ()))
        )
    ladder_payload: dict[str, Any] | None = None
    if str(row.get("paper_i_cutoff_ladder_stage") or "").strip():
        try:
            stage, config = normalize_paper_i_ladder_stage(str(row.get("paper_i_cutoff_ladder_stage") or ""))
            ladder_payload = {
                "stage": stage,
                "n_ph_work": config.n_ph_work,
                "n_ph_ref": config.n_ph_ref,
                "snake_policy": row.get("paper_i_ladder_snake_policy"),
                "acceptance_threshold": row.get("paper_i_ladder_acceptance_threshold"),
                "requires_prior_failure": row.get("paper_i_ladder_requires_prior_failure"),
                "escalation_reason": row.get("paper_i_ladder_escalation_reason"),
                "allow_ref5": row.get("paper_i_ladder_allow_ref5"),
                "tau_phys": row.get("tau_phys"),
                "tau_tight": row.get("tau_tight"),
            }
        except Exception:
            ladder_payload = {"stage": row.get("paper_i_cutoff_ladder_stage"), "validation_status": "unparsed"}
    reference_energy_payload = None
    if str(row.get("reference_energy_cache_json") or "").strip():
        reference_energy_payload = {
            "reference_energy_cache_json": row.get("reference_energy_cache_json"),
            "same_cutoff_reference_energy_key": row.get("same_cutoff_reference_energy_key"),
            "reference_cutoff_energy_key": row.get("reference_cutoff_energy_key"),
            "same_cutoff_exact_gs_energy": row.get("same_cutoff_exact_gs_energy"),
            "exact_reference_energy": row.get("exact_reference_energy"),
            "exact_reference_n_ph_max": row.get("exact_reference_n_ph_max"),
            "reference_energy_status": row.get("reference_energy_status"),
        }
    route_faithfulness_ladder_payload = _route_faithfulness_ladder_manifest(row)
    payload = {
        "schema": "phase3_chtc_record_manifest_v1",
        "record_id": record_id,
        "generated_utc": _now_utc(),
        "mode": row.get("mode"),
        "suite_profile": row.get("suite_profile"),
        "benchmark_ids": _split_words(row.get("benchmark_ids", "")),
        "families": _split_words(row.get("families", "")),
        "sizes": _split_words(row.get("sizes", "")),
        "boson_cutoff": row.get("boson_cutoff"),
        "boson_cutoffs": _split_words(row.get("boson_cutoffs", "")),
        "exact_reference_boson_cutoff": row.get("exact_reference_boson_cutoff"),
        "force_same_cutoff_objective": _truthy(row.get("force_same_cutoff_objective")),
        "trial_prune_gate": {
            "depth": row.get("trial_prune_depth"),
            "abs_delta_e": row.get("trial_prune_abs_delta_e"),
            "metric": row.get("trial_prune_metric") or "same_cutoff_abs_delta_e",
            "comparator_label": row.get("trial_prune_comparator_label"),
        },
        "physics_grid_profile": row.get("physics_grid_profile"),
        "calibration_profile": row.get("calibration_profile"),
        "hardware_resolution": {
            "schema": row.get("hardware_resolution_schema") or "gradient_resolution_v1",
            "mode": row.get("hardware_resolution_mode") or "ideal",
            "profile_json": row.get("hardware_resolution_profile_json"),
            "profile_name": row.get("hardware_resolution_profile_name"),
        },
        "fixed_inner_optimizer": _fixed_inner_optimizer(row, static_route_id=static_route_id),
        "static_route_id": static_route_id,
        "static_route_identity": route_identity,
        "phase2_novelty_mode": row.get("phase2_novelty_mode"),
        "meta_feature_profile": meta_feature_profile,
        "meta_feature_profile_source": meta_feature_profile_source,
        "meta_feature_policy_contract": _meta_feature_policy_contract(meta_feature_profile),
        "algorithm_variant": row.get("algorithm_variant"),
        "trial_param_overrides_json": row.get("trial_param_overrides_json"),
        "objective_profile": row.get("objective_profile"),
        "discovery_objective_mode": row.get("discovery_objective_mode") or "terminal_proxy",
        "objective_weights": {
            "energy": row.get("objective_energy_weight"),
            "count_2q": row.get("objective_2q_weight"),
            "depth_2q": _objective_2q_depth_weight(row),
            "depth": row.get("objective_depth_weight"),
            "parameters": row.get("objective_parameter_weight"),
            "shot": row.get("objective_shot_weight"),
            "target_abs_delta_e": row.get("target_abs_delta_e"),
        },
        "objective_weighting": {
            "preset": row.get("objective_weight_preset"),
            "family_weights": row.get("objective_family_weights"),
            "benchmark_weights": row.get("objective_benchmark_weights"),
        },
        "phase0": {
            "schema": "phase0_optuna_chtc_row_v1",
            "phase0_aware": any(str(row.get(field) or "").strip() for field in _PHASE0_ROW_FIELDS),
            "phase0_is_route_identity": False,
            "row_fields": _row_subset(row, _PHASE0_ROW_FIELDS),
            "defaults_source": "pipelines.static_adapt.optimization.phase3_policy_optuna.PHASE0_OPTUNA_DEFAULTS",
            "max_records_semantics": "0_uncapped; generated Phase0 nph1 studies use capped defaults",
        },
        "oracle_warm_start_requirements": {
            "schema": "oracle_warm_start_requirements_v1",
            "row_fields": _row_subset(row, _ORACLE_WARM_START_REQUIREMENT_FIELDS),
            "required_static_route_id": row.get("oracle_required_static_route_id"),
            "required_suite_profile": row.get("oracle_required_suite_profile"),
            "require_phase0_aware": _truthy(row.get("oracle_require_phase0_aware")),
            "require_compatible_warm_starts": _truthy(row.get("oracle_require_compatible_warm_starts")),
        },
        "route_faithfulness_ladder": route_faithfulness_ladder_payload,
        "phase3_oracle": {
            "gradient_mode": row.get("phase3_oracle_gradient_mode") or "off",
            "backend_name": row.get("phase3_oracle_backend_name"),
            "use_fake_backend": _truthy(row.get("phase3_oracle_use_fake_backend")),
            "shots": row.get("phase3_oracle_shots"),
            "repeats": row.get("phase3_oracle_repeats"),
            "aggregate": row.get("phase3_oracle_aggregate"),
            "seed": row.get("phase3_oracle_seed"),
            "gradient_step": row.get("phase3_oracle_gradient_step"),
            "execution_surface": row.get("phase3_oracle_execution_surface"),
            "inner_objective_mode": row.get("phase3_oracle_inner_objective_mode") or "exact",
            "value_noise_model": row.get("phase3_oracle_value_noise_model") or "off",
            "value_noise_std": row.get("phase3_oracle_value_noise_std"),
            "value_noise_seed": row.get("phase3_oracle_value_noise_seed"),
            "value_noise_sigma0_abs": row.get("phase3_oracle_value_noise_sigma0_abs"),
            "value_noise_n_eff": row.get("phase3_oracle_value_noise_n_eff"),
            "value_noise_seed_policy": row.get("phase3_oracle_value_noise_seed_policy"),
            "value_noise_base_seed": row.get("phase3_oracle_value_noise_base_seed"),
            "value_noise_replicate_id": row.get("phase3_oracle_value_noise_replicate_id"),
            "value_noise_semantic": row.get("phase3_oracle_value_noise_semantic") or row.get("noise_semantic"),
            "synthetic_depolarizing_1q_error": row.get("phase3_oracle_synthetic_depolarizing_1q_error"),
            "synthetic_depolarizing_2q_error": row.get("phase3_oracle_synthetic_depolarizing_2q_error"),
            "synthetic_depolarizing_1q_gates": _canonical_gate_cli_value(
                row.get("phase3_oracle_synthetic_depolarizing_1q_gates"),
                field_name="phase3_oracle_synthetic_depolarizing_1q_gates",
            ) or "",
            "synthetic_depolarizing_2q_gates": _canonical_gate_cli_value(
                row.get("phase3_oracle_synthetic_depolarizing_2q_gates"),
                field_name="phase3_oracle_synthetic_depolarizing_2q_gates",
            ) or "",
            "synthetic_coherent_1q_angle_std": row.get("phase3_oracle_synthetic_coherent_1q_angle_std"),
            "synthetic_coherent_2q_angle_std": row.get("phase3_oracle_synthetic_coherent_2q_angle_std"),
            "synthetic_coherent_seed": row.get("phase3_oracle_synthetic_coherent_seed"),
            "synthetic_coherent_generator_mode": row.get("phase3_oracle_synthetic_coherent_generator_mode"),
            "synthetic_coherent_1q_gates": _canonical_gate_cli_value(
                row.get("phase3_oracle_synthetic_coherent_1q_gates"),
                field_name="phase3_oracle_synthetic_coherent_1q_gates",
            ) or "",
            "synthetic_coherent_2q_gates": _canonical_gate_cli_value(
                row.get("phase3_oracle_synthetic_coherent_2q_gates"),
                field_name="phase3_oracle_synthetic_coherent_2q_gates",
            ) or "",
            "physical_shots_unchanged": _truthy(row.get("physical_shots_unchanged")) if str(row.get("physical_shots_unchanged") or "").strip() else True,
            "fixed_gate_error_reduction_claimed": _truthy(row.get("fixed_gate_error_reduction_claimed")),
        },
        "noise_robustness": {
            "schema": "phase3_noise_robustness_record_v2",
            "run_class": row.get("robustness_run_class") or row.get("run_class"),
            "promotion_status": row.get("robustness_promotion_status"),
            "noise_level_id": row.get("noise_level_id"),
            "sigma0_abs": row.get("noise_sigma0_abs") or row.get("phase3_oracle_value_noise_sigma0_abs"),
            "N_eff": row.get("noise_n_eff") or row.get("phase3_oracle_value_noise_n_eff"),
            "derived_std": row.get("noise_derived_std") or row.get("phase3_oracle_value_noise_std"),
            "std_formula": row.get("noise_std_formula"),
            "semantic": row.get("noise_semantic"),
            "replicate_id": row.get("noise_replicate_id") or row.get("phase3_oracle_value_noise_replicate_id"),
            "pair_group_id": row.get("noise_pair_group_id"),
            "base_seed": row.get("noise_base_seed") or row.get("phase3_oracle_value_noise_base_seed"),
            "seed_derivation": row.get("noise_seed_derivation"),
            "ladder_rung_id": row.get("ladder_rung_id"),
            "ladder_stage": row.get("ladder_stage"),
            "oracle_surface_class": row.get("oracle_surface_class"),
            "noise_surface_class": row.get("noise_surface_class"),
            "generation_gate": row.get("ladder_rung_gate"),
            "required_predecessors": _split_words(row.get("ladder_rung_required_predecessors", "")),
            "same_noise_surface_key": row.get("same_noise_surface_key"),
            "confidence_score_penalties_changed": _truthy(row.get("confidence_score_penalties_changed")),
            "confidence_scoring_semantic": row.get("confidence_scoring_semantic"),
            "phase1_score_z_alpha": row.get("phase1_score_z_alpha"),
            "phase2_score_z_alpha": row.get("phase2_score_z_alpha"),
            "target_hit_required": _truthy(row.get("target_hit_required")),
            "target_hit_success_stop_reason": row.get("target_hit_success_stop_reason"),
            "route_faithfulness_prerequisite": _truthy(row.get("route_faithfulness_prerequisite")),
            "route_faithfulness_prerequisite_status": row.get("route_faithfulness_prerequisite_status"),
            "route_faithfulness_prerequisite_reason": row.get("route_faithfulness_prerequisite_reason"),
            "diagnostic_non_hit_allowed": _truthy(row.get("diagnostic_non_hit_allowed")),
            "diagnostic_non_hit_stop_reasons": _split_words(row.get("diagnostic_non_hit_stop_reasons", "")),
            "pre_run_rung_pass_status": "not_evaluated" if route_faithfulness_ladder_payload else None,
            "source_lock_status": row.get("source_lock_status"),
            "physical_shots_unchanged": _truthy(row.get("physical_shots_unchanged")) if str(row.get("physical_shots_unchanged") or "").strip() else True,
            "fixed_gate_error_reduction_claimed": _truthy(row.get("fixed_gate_error_reduction_claimed")),
            "adapt_noise_floor_stop_policy": row.get("adapt_noise_floor_stop_policy"),
            "fixed_hardware_diagnostic": {
                "enabled": _truthy(row.get("fixed_hardware_diagnostic_enabled")),
                "semantic": row.get("fixed_hardware_diagnostic_semantic"),
                "backend_name": row.get("fixed_hardware_diagnostic_backend_name"),
                "mode": row.get("fixed_hardware_diagnostic_mode"),
            },
        },
        "paper_i_cutoff_ladder": ladder_payload,
        "reference_energy_metadata": reference_energy_payload,
        "objective_provenance": {
            "schema": "phase3_chtc_objective_provenance_v1",
            "objective_weight_preset": row.get("objective_weight_preset"),
            "discovery_objective_mode": row.get("discovery_objective_mode") or "terminal_proxy",
            "objective_final_score_noise_mode": "exact_noiseless_v1",
            "phase3_oracle_inner_objective_mode": row.get("phase3_oracle_inner_objective_mode") or "exact",
            "phase3_oracle_value_noise_model": row.get("phase3_oracle_value_noise_model") or "off",
            "phase3_oracle_value_noise_std": row.get("phase3_oracle_value_noise_std"),
            "phase3_oracle_value_noise_sigma0_abs": row.get("phase3_oracle_value_noise_sigma0_abs"),
            "phase3_oracle_value_noise_n_eff": row.get("phase3_oracle_value_noise_n_eff"),
            "phase3_oracle_value_noise_seed_policy": row.get("phase3_oracle_value_noise_seed_policy"),
            "phase3_oracle_value_noise_base_seed": row.get("phase3_oracle_value_noise_base_seed"),
            "phase3_oracle_value_noise_replicate_id": row.get("phase3_oracle_value_noise_replicate_id"),
            "phase3_oracle_synthetic_depolarizing_1q_error": row.get("phase3_oracle_synthetic_depolarizing_1q_error"),
            "phase3_oracle_synthetic_depolarizing_2q_error": row.get("phase3_oracle_synthetic_depolarizing_2q_error"),
            "phase3_oracle_synthetic_depolarizing_1q_gates": _canonical_gate_cli_value(
                row.get("phase3_oracle_synthetic_depolarizing_1q_gates"),
                field_name="phase3_oracle_synthetic_depolarizing_1q_gates",
            ) or "",
            "phase3_oracle_synthetic_depolarizing_2q_gates": _canonical_gate_cli_value(
                row.get("phase3_oracle_synthetic_depolarizing_2q_gates"),
                field_name="phase3_oracle_synthetic_depolarizing_2q_gates",
            ) or "",
            "phase3_oracle_synthetic_coherent_1q_angle_std": row.get("phase3_oracle_synthetic_coherent_1q_angle_std"),
            "phase3_oracle_synthetic_coherent_2q_angle_std": row.get("phase3_oracle_synthetic_coherent_2q_angle_std"),
            "phase3_oracle_synthetic_coherent_seed": row.get("phase3_oracle_synthetic_coherent_seed"),
            "phase3_oracle_synthetic_coherent_generator_mode": row.get("phase3_oracle_synthetic_coherent_generator_mode"),
            "phase3_oracle_synthetic_coherent_1q_gates": _canonical_gate_cli_value(
                row.get("phase3_oracle_synthetic_coherent_1q_gates"),
                field_name="phase3_oracle_synthetic_coherent_1q_gates",
            ) or "",
            "phase3_oracle_synthetic_coherent_2q_gates": _canonical_gate_cli_value(
                row.get("phase3_oracle_synthetic_coherent_2q_gates"),
                field_name="phase3_oracle_synthetic_coherent_2q_gates",
            ) or "",
            "adapt_noise_floor_stop_policy": row.get("adapt_noise_floor_stop_policy"),
        },
        "runtime": {
            "phase3_adapt_parallel_gradient_workers": row.get("phase3_adapt_parallel_gradient_workers"),
            "phase3_adapt_beam_parent_workers": row.get("phase3_adapt_beam_parent_workers"),
            "n_jobs": row.get("n_jobs"),
            "benchmarks_per_trial_jobs": row.get("benchmarks_per_trial_jobs"),
        },
        "n_trials": row.get("n_trials"),
        "timeouts": {
            "trial_timeout_sec": row.get("trial_timeout_sec"),
            "compile_timeout_sec": row.get("compile_timeout_sec"),
        },
        "output_root": str(out_root),
        "run_root": str(run_root),
        "progress_dir": str(progress_dir),
        "command": [str(part) for part in command],
    }
    _write_json(out_root / "record_manifest.json", payload)


def _command_long_option_value(command: Sequence[str], option: str) -> str | None:
    for index, token in enumerate(command):
        if str(token) == option:
            if index + 1 >= len(command):
                return ""
            return str(command[index + 1])
        prefix = f"{option}="
        if str(token).startswith(prefix):
            return str(token)[len(prefix) :]
    return None


def _values_equivalent(expected: Any, actual: Any) -> bool:
    if expected is None and actual is None:
        return True
    if isinstance(expected, bool) or isinstance(actual, bool):
        def _boolish(value: Any) -> bool | None:
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {"true", "1", "yes", "y", "on"}:
                return True
            if text in {"false", "0", "no", "n", "off"}:
                return False
            return None
        return _boolish(expected) == _boolish(actual)
    try:
        exp = float(expected)
        act = float(actual)
        if math.isfinite(exp) and math.isfinite(act):
            return abs(exp - act) <= max(1e-12, 1e-10 * max(abs(exp), abs(act), 1.0))
    except (TypeError, ValueError):
        pass
    return str(expected).strip() == str(actual).strip()


_SOURCE_LOCK_RESULT_PREFERRED_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "adapt_beam_live_branches": ("adapt_beam_live_branches_requested",),
    "adapt_beam_children_per_parent": ("adapt_beam_children_per_parent_requested",),
    "adapt_beam_terminated_keep": ("adapt_beam_terminated_keep_requested",),
}

_SOURCE_LOCK_RESULT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "spsa_a": ("adapt_spsa_a",),
    "spsa_c": ("adapt_spsa_c",),
    "spsa_A": ("adapt_spsa_A",),
    "spsa_alpha": ("adapt_spsa_alpha",),
    "spsa_gamma": ("adapt_spsa_gamma",),
    "lambda_compile": ("phase1_lambda_compile", "phase2_lambda_compile"),
    "lambda_measure": ("phase1_lambda_measure", "phase2_lambda_measure"),
    "lambda_leak": ("phase1_lambda_leak", "phase2_lambda_leak"),
    "lambda_1q": ("phase1_lambda_1q", "phase2_lambda_1q"),
    "lambda_2q": ("phase1_lambda_2q", "phase2_lambda_2q"),
    "lambda_d": ("phase1_lambda_d", "phase2_lambda_d"),
    "lambda_theta": ("phase1_lambda_theta", "phase2_lambda_theta"),
    "lambda_shot": ("phase1_lambda_shot", "phase2_lambda_shot"),
    "compile_cx_weight": ("phase1_compile_cx_proxy_weight", "phase2_compile_cx_proxy_weight"),
    "compile_sq_weight": ("phase1_compile_sq_proxy_weight", "phase2_compile_sq_proxy_weight"),
    "compile_rotation_step_weight": ("phase1_compile_rotation_step_weight", "phase2_compile_rotation_step_weight"),
    "compile_position_shift_weight": ("phase1_compile_position_shift_weight", "phase2_compile_position_shift_weight"),
    "compile_refit_active_weight": ("phase1_compile_refit_active_weight", "phase2_compile_refit_active_weight"),
    "measure_groups_weight": ("phase1_measure_groups_weight", "phase2_measure_groups_weight"),
    "measure_shots_weight": ("phase1_measure_shots_weight", "phase2_measure_shots_weight"),
    "measure_reuse_weight": ("phase1_measure_reuse_weight", "phase2_measure_reuse_weight"),
    "opt_dim_cost_scale": ("phase1_opt_dim_cost_scale", "phase2_opt_dim_cost_scale"),
    "family_repeat_penalty": ("phase1_family_repeat_cost_scale", "phase2_family_repeat_cost_scale"),
}


def _source_lock_result_observations(result_payload: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    adapt_vqe = result_payload.get("adapt_vqe")
    containers: tuple[tuple[str, Mapping[str, Any]], ...] = tuple(
        (name, container)
        for name, container in (
            ("settings", result_payload.get("settings")),
            ("adapt_vqe", adapt_vqe),
        )
        if isinstance(container, Mapping)
    )
    preferred_aliases = _SOURCE_LOCK_RESULT_PREFERRED_FIELD_ALIASES.get(key, ())
    for container_name, container in containers:
        for alias in preferred_aliases:
            if alias in container:
                observations.append(
                    {
                        "container": container_name,
                        "field": alias,
                        "value": container.get(alias),
                    }
                )
    if observations:
        return observations
    aliases = (key, *(_SOURCE_LOCK_RESULT_FIELD_ALIASES.get(key, ())))
    for container_name, container in containers:
        for alias in aliases:
            if alias in container:
                observations.append(
                    {
                        "container": container_name,
                        "field": alias,
                        "value": container.get(alias),
                    }
                )
    return observations


def _selected_generator_history(result_payload: Mapping[str, Any]) -> list[str]:
    adapt_vqe = result_payload.get("adapt_vqe", result_payload)
    if not isinstance(adapt_vqe, Mapping):
        return []
    rows = adapt_vqe.get("history") or adapt_vqe.get("history_tail") or []
    history: list[str] = []
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if isinstance(row, Mapping):
                value = row.get("selected_op") or row.get("operator") or row.get("selected_operator")
                if value is not None:
                    history.append(str(value))
    if history:
        return history
    operators = adapt_vqe.get("operators")
    if isinstance(operators, Sequence) and not isinstance(operators, (str, bytes)):
        return [str(op) for op in operators]
    return history


def _sequence_sha256(history: Sequence[str]) -> str:
    return _sha256_text(json.dumps(list(history), separators=(",", ":"), ensure_ascii=False))


def _hh_source_lock_result_paths(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    return sorted(run_root.glob("*/trial_*/**/json/result.json"))


def _write_hh_source_lock_audit(
    row: Mapping[str, str],
    *,
    record_id: str,
    out_root: Path,
    run_root: Path,
    command: Sequence[str],
) -> dict[str, Any] | None:
    if str(row.get("route_faithfulness_ladder_schema") or "").strip() != HH_ROUTE_FAITHFULNESS_LADDER_SCHEMA:
        return None
    errors: list[str] = []
    warnings: list[str] = []
    override_path_value = str(row.get("trial_param_overrides_json") or row.get("source_lock_trial_param_overrides_json") or "").strip()
    command_override_value = _command_long_option_value(command, "--trial-param-overrides-json")
    override_payload: dict[str, Any] = {}
    override_fields: dict[str, Any] = {}
    override_path: Path | None = None
    override_sha256: str | None = None
    expected_override_sha256 = str(row.get("source_lock_trial_param_overrides_sha256") or "").strip() or None
    if not override_path_value:
        errors.append("missing_trial_param_overrides_json")
    else:
        override_path = _resolve_under_repo(override_path_value)
        if not override_path.exists():
            errors.append(f"trial_param_overrides_json_missing:{override_path_value}")
        else:
            try:
                override_payload = _read_json_object(override_path)
                raw_fields = override_payload.get("trial_param_overrides", override_payload)
                if isinstance(raw_fields, Mapping):
                    override_fields = dict(raw_fields)
                else:
                    errors.append("trial_param_overrides_payload_not_object")
                override_sha256 = sha256_file(override_path)
                if expected_override_sha256 and override_sha256 != expected_override_sha256:
                    errors.append("source_lock_trial_param_overrides_sha256_mismatch")
            except Exception as exc:
                errors.append(f"trial_param_overrides_json_unreadable:{type(exc).__name__}:{exc}")
    command_path_match = bool(command_override_value) and str(command_override_value) == override_path_value
    if not command_path_match:
        errors.append("command_trial_param_overrides_json_mismatch")
    expected_count = _positive_int_or_none(row.get("source_lock_trial_param_override_field_count"))
    if expected_count is not None and override_fields and int(expected_count) != len(override_fields):
        errors.append("source_lock_trial_param_override_field_count_mismatch")
    required_fields = tuple(str(field) for field in override_payload.get("required_route_critical_fields", ()) or ())
    missing_required = [field for field in required_fields if field not in override_fields]
    if missing_required:
        errors.append("missing_required_route_critical_override_fields")
    summary_path = out_root / "summary.json"
    summary_payload: dict[str, Any] = {}
    summary_overrides: dict[str, Any] = {}
    summary_match = False
    if summary_path.exists():
        try:
            summary_payload = _read_json_object(summary_path)
            raw_summary = summary_payload.get("trial_param_overrides", {})
            if isinstance(raw_summary, Mapping):
                summary_overrides = dict(raw_summary)
                summary_match = set(summary_overrides) == set(override_fields) and all(
                    _values_equivalent(override_fields[key], summary_overrides.get(key)) for key in override_fields
                )
                if not summary_match:
                    errors.append("summary_trial_param_overrides_mismatch")
            else:
                errors.append("summary_trial_param_overrides_not_object")
        except Exception as exc:
            errors.append(f"summary_json_unreadable:{type(exc).__name__}:{exc}")
    else:
        warnings.append("summary_json_missing")
    reference_history: list[str] = []
    reference_path_value = str(row.get("source_lock_reference_json") or "").strip()
    if reference_path_value:
        reference_path = _resolve_under_repo(reference_path_value)
        if reference_path.exists():
            try:
                reference_history = _selected_generator_history(_read_json_object(reference_path))
            except Exception as exc:
                warnings.append(f"source_lock_reference_json_unreadable:{type(exc).__name__}:{exc}")
        else:
            warnings.append("source_lock_reference_json_missing")
    trials: list[dict[str, Any]] = []
    result_paths = _hh_source_lock_result_paths(run_root)
    if not result_paths:
        warnings.append("no_result_jsons_found")
    for result_path in result_paths:
        try:
            result_payload = _read_json_object(result_path)
        except Exception as exc:
            trials.append(
                {
                    "result_json": _safe_relative(result_path),
                    "status": "unreadable",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            errors.append("result_json_unreadable")
            continue
        adapt_vqe = result_payload.get("adapt_vqe", result_payload)
        if not isinstance(adapt_vqe, Mapping):
            adapt_vqe = {}
        field_results: dict[str, Any] = {}
        serialized_match_count = 0
        serialized_mismatch_count = 0
        not_serialized_count = 0
        for key, expected in sorted(override_fields.items()):
            observations = _source_lock_result_observations(result_payload, key)
            if not observations:
                status = "not_serialized_in_result_json"
                not_serialized_count += 1
            else:
                matches = [obs for obs in observations if _values_equivalent(expected, obs.get("value"))]
                status = "match" if len(matches) == len(observations) else "mismatch"
                if status == "match":
                    serialized_match_count += 1
                else:
                    serialized_mismatch_count += 1
            field_results[key] = {
                "expected": expected,
                "status": status,
                "observations": observations,
            }
        if serialized_mismatch_count:
            errors.append(f"result_serialized_source_lock_field_mismatch:{_safe_relative(result_path)}")
        history = _selected_generator_history(result_payload)
        sequence_status = "no_reference_sequence"
        first_mismatch: dict[str, Any] | None = None
        if reference_history:
            if reference_history[: len(history)] == history:
                sequence_status = "source_lock_prefix_match"
            else:
                sequence_status = "source_lock_sequence_changed"
                for idx, (expected_op, actual_op) in enumerate(zip(reference_history, history), start=1):
                    if expected_op != actual_op:
                        first_mismatch = {
                            "admission": idx,
                            "source_lock": expected_op,
                            "result": actual_op,
                        }
                        break
                if first_mismatch is None and len(history) != len(reference_history):
                    first_mismatch = {
                        "admission": min(len(history), len(reference_history)) + 1,
                        "source_lock": reference_history[min(len(history), len(reference_history))] if len(reference_history) > len(history) else None,
                        "result": history[min(len(history), len(reference_history))] if len(history) > len(reference_history) else None,
                    }
        trials.append(
            {
                "result_json": _safe_relative(result_path),
                "stop_reason": adapt_vqe.get("stop_reason"),
                "ansatz_depth": adapt_vqe.get("ansatz_depth"),
                "energy": adapt_vqe.get("energy"),
                "abs_delta_e": adapt_vqe.get("abs_delta_e"),
                "target_hit": str(adapt_vqe.get("stop_reason") or "") == str(row.get("target_hit_success_stop_reason") or "benchmark_abs_delta_e_target"),
                "source_lock_field_counts": {
                    "expected": len(override_fields),
                    "serialized_match": serialized_match_count,
                    "serialized_mismatch": serialized_mismatch_count,
                    "not_serialized_in_result_json": not_serialized_count,
                },
                "sequence_status": sequence_status,
                "sequence_sha256": _sequence_sha256(history),
                "source_lock_sequence_sha256": _sequence_sha256(reference_history) if reference_history else None,
                "first_sequence_mismatch": first_mismatch,
                "field_results": field_results,
            }
        )
    lock_status = "fail" if errors else "pass"
    result_serialization_status = (
        "no_results"
        if not trials
        else (
            "complete"
            if all((trial.get("source_lock_field_counts", {}).get("not_serialized_in_result_json", 0) == 0) for trial in trials if trial.get("status") != "unreadable")
            else "partial"
        )
    )
    payload = {
        "schema": "paper_i_hh_source_lock_audit_v1",
        "record_id": record_id,
        "generated_utc": _now_utc(),
        "status": lock_status,
        "ok": lock_status == "pass",
        "errors": errors,
        "warnings": warnings,
        "lock_basis": "command_and_summary_trial_param_overrides_match_source_lock_payload",
        "trial_param_overrides_json": override_path_value,
        "trial_param_overrides_json_resolved": _safe_relative(override_path) if override_path is not None else None,
        "trial_param_overrides_sha256": override_sha256,
        "expected_trial_param_overrides_sha256": expected_override_sha256,
        "command_trial_param_overrides_json": command_override_value,
        "command_trial_param_overrides_json_match": command_path_match,
        "summary_json": _safe_relative(summary_path),
        "summary_trial_param_overrides_match": summary_match,
        "override_field_count": len(override_fields),
        "expected_override_field_count": expected_count,
        "required_route_critical_fields": list(required_fields),
        "missing_required_route_critical_fields": missing_required,
        "result_field_serialization_status": result_serialization_status,
        "result_jsons_inspected": len(result_paths),
        "source_lock_reference_json": reference_path_value,
        "source_lock_reference_sequence_sha256": _sequence_sha256(reference_history) if reference_history else None,
        "trials": trials,
    }
    _write_json(out_root / "source_lock_audit.json", payload)
    return payload


def _spin_boson_result_target_error(result_payload: Mapping[str, Any], adapt_vqe: Mapping[str, Any]) -> float | None:
    cutoff = result_payload.get("cutoff_diagnostics") if isinstance(result_payload.get("cutoff_diagnostics"), Mapping) else {}
    for value in (
        cutoff.get("primary_error") if isinstance(cutoff, Mapping) else None,
        cutoff.get("abs_error_reference_cutoff") if isinstance(cutoff, Mapping) else None,
        adapt_vqe.get("abs_delta_e"),
    ):
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _latest_progress_current_json_path(progress_dir: Path | None) -> Path | None:
    if progress_dir is None or not progress_dir.exists():
        return None
    candidates = [progress_dir / "current.json"]
    try:
        candidates.extend(path for path in progress_dir.glob("*/current.json") if path.parent != progress_dir)
    except Exception:
        pass
    existing = [path for path in candidates if path.exists() and path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _write_spin_boson_source_lock_audit(
    row: Mapping[str, str],
    *,
    record_id: str,
    out_root: Path,
    run_root: Path,
    command: Sequence[str],
    progress_dir: Path | None = None,
) -> dict[str, Any] | None:
    if str(row.get("route_faithfulness_ladder_schema") or "").strip() != SPIN_BOSON_ROUTE_FAITHFULNESS_LADDER_SCHEMA:
        return None
    errors: list[str] = []
    warnings: list[str] = []
    override_path_value = str(row.get("trial_param_overrides_json") or row.get("source_lock_trial_param_overrides_json") or "").strip()
    command_override_value = _command_long_option_value(command, "--trial-param-overrides-json")
    override_payload: dict[str, Any] = {}
    override_fields: dict[str, Any] = {}
    override_path: Path | None = None
    override_sha256: str | None = None
    expected_override_sha256 = str(row.get("source_lock_trial_param_overrides_sha256") or "").strip() or None
    if not override_path_value:
        errors.append("missing_trial_param_overrides_json")
    else:
        override_path = _resolve_under_repo(override_path_value)
        if not override_path.exists():
            errors.append(f"trial_param_overrides_json_missing:{override_path_value}")
        else:
            try:
                override_payload = _read_json_object(override_path)
                raw_fields = override_payload.get("trial_param_overrides", override_payload)
                if isinstance(raw_fields, Mapping):
                    override_fields = dict(raw_fields)
                else:
                    errors.append("trial_param_overrides_payload_not_object")
                override_sha256 = sha256_file(override_path)
                if expected_override_sha256 and override_sha256 != expected_override_sha256:
                    errors.append("source_lock_trial_param_overrides_sha256_mismatch")
            except Exception as exc:
                errors.append(f"trial_param_overrides_json_unreadable:{type(exc).__name__}:{exc}")
    command_path_match = bool(command_override_value) and str(command_override_value) == override_path_value
    if not command_path_match:
        errors.append("command_trial_param_overrides_json_mismatch")
    expected_count = _positive_int_or_none(row.get("source_lock_trial_param_override_field_count"))
    if expected_count is not None and override_fields and int(expected_count) != len(override_fields):
        errors.append("source_lock_trial_param_override_field_count_mismatch")
    required_fields = tuple(str(field) for field in override_payload.get("required_route_critical_fields", ()) or ())
    missing_required = [field for field in required_fields if field not in override_fields]
    if missing_required:
        errors.append("missing_required_route_critical_override_fields")
    summary_path = out_root / "summary.json"
    summary_overrides: dict[str, Any] = {}
    summary_match = False
    if summary_path.exists():
        try:
            summary_payload = _read_json_object(summary_path)
            raw_summary = summary_payload.get("trial_param_overrides", {})
            if isinstance(raw_summary, Mapping):
                summary_overrides = dict(raw_summary)
                summary_match = set(summary_overrides) == set(override_fields) and all(
                    _values_equivalent(override_fields[key], summary_overrides.get(key)) for key in override_fields
                )
                if not summary_match:
                    errors.append("summary_trial_param_overrides_mismatch")
            else:
                errors.append("summary_trial_param_overrides_not_object")
        except Exception as exc:
            errors.append(f"summary_json_unreadable:{type(exc).__name__}:{exc}")
    else:
        warnings.append("summary_json_missing")
    reference_history: list[str] = []
    reference_path_value = str(row.get("source_lock_reference_json") or "").strip()
    if reference_path_value:
        reference_path = _resolve_under_repo(reference_path_value)
        if reference_path.exists():
            try:
                reference_history = _selected_generator_history(_read_json_object(reference_path))
            except Exception as exc:
                errors.append(f"source_lock_reference_json_unreadable:{type(exc).__name__}:{exc}")
        else:
            errors.append("source_lock_reference_json_missing")
    else:
        errors.append("source_lock_reference_json_missing")
    target_threshold = (
        _float_field(row, "source_lock_target_abs_delta_e")
        or _float_field(row, "target_abs_delta_e")
        or _float_field(row, "required_target_abs_delta_e")
        or 2.0e-4
    )
    trials: list[dict[str, Any]] = []
    result_paths = _hh_source_lock_result_paths(run_root)
    if not result_paths:
        errors.append("no_result_jsons_found")
    for result_path in result_paths:
        try:
            result_payload = _read_json_object(result_path)
        except Exception as exc:
            trials.append(
                {
                    "result_json": _safe_relative(result_path),
                    "status": "unreadable",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            errors.append("result_json_unreadable")
            continue
        adapt_vqe = result_payload.get("adapt_vqe", result_payload)
        if not isinstance(adapt_vqe, Mapping):
            adapt_vqe = {}
        field_results: dict[str, Any] = {}
        serialized_match_count = 0
        serialized_mismatch_count = 0
        not_serialized_count = 0
        for key, expected in sorted(override_fields.items()):
            observations = _source_lock_result_observations(result_payload, key)
            if not observations:
                status = "not_serialized_in_result_json"
                not_serialized_count += 1
            else:
                matches = [obs for obs in observations if _values_equivalent(expected, obs.get("value"))]
                status = "match" if len(matches) == len(observations) else "mismatch"
                if status == "match":
                    serialized_match_count += 1
                else:
                    serialized_mismatch_count += 1
            field_results[key] = {
                "expected": expected,
                "status": status,
                "observations": observations,
            }
        if serialized_mismatch_count:
            errors.append(f"result_serialized_source_lock_field_mismatch:{_safe_relative(result_path)}")
        history = _selected_generator_history(result_payload)
        sequence_exact_match = bool(reference_history) and list(history) == list(reference_history)
        first_mismatch: dict[str, Any] | None = None
        if reference_history and not sequence_exact_match:
            for idx, (expected_op, actual_op) in enumerate(zip(reference_history, history), start=1):
                if expected_op != actual_op:
                    first_mismatch = {"admission": idx, "source_lock": expected_op, "result": actual_op}
                    break
            if first_mismatch is None and len(history) != len(reference_history):
                idx = min(len(history), len(reference_history))
                first_mismatch = {
                    "admission": idx + 1,
                    "source_lock": reference_history[idx] if idx < len(reference_history) else None,
                    "result": history[idx] if idx < len(history) else None,
                }
        if not sequence_exact_match:
            errors.append(f"source_lock_sequence_mismatch:{_safe_relative(result_path)}")
        stop_reason = adapt_vqe.get("stop_reason")
        target_hit = str(stop_reason or "") == str(row.get("target_hit_success_stop_reason") or "benchmark_abs_delta_e_target")
        if not target_hit:
            errors.append(f"target_hit_stop_reason_mismatch:{_safe_relative(result_path)}")
        target_error = _spin_boson_result_target_error(result_payload, adapt_vqe)
        if target_error is None:
            errors.append(f"target_error_missing:{_safe_relative(result_path)}")
        elif target_error > float(target_threshold):
            errors.append(f"target_error_above_threshold:{_safe_relative(result_path)}:{target_error}>{target_threshold}")
        trials.append(
            {
                "result_json": _safe_relative(result_path),
                "stop_reason": stop_reason,
                "ansatz_depth": adapt_vqe.get("ansatz_depth"),
                "energy": adapt_vqe.get("energy"),
                "abs_delta_e": adapt_vqe.get("abs_delta_e"),
                "target_error": target_error,
                "target_threshold": float(target_threshold),
                "target_hit": target_hit,
                "source_lock_field_counts": {
                    "expected": len(override_fields),
                    "serialized_match": serialized_match_count,
                    "serialized_mismatch": serialized_mismatch_count,
                    "not_serialized_in_result_json": not_serialized_count,
                },
                "sequence_status": "source_lock_exact_match" if sequence_exact_match else "source_lock_sequence_changed",
                "sequence_exact_match": sequence_exact_match,
                "sequence_sha256": _sequence_sha256(history),
                "source_lock_sequence_sha256": _sequence_sha256(reference_history) if reference_history else None,
                "first_sequence_mismatch": first_mismatch,
                "field_results": field_results,
            }
        )
    lock_status = "fail" if errors else "pass"
    result_serialization_status = (
        "no_results"
        if not trials
        else (
            "complete"
            if all((trial.get("source_lock_field_counts", {}).get("not_serialized_in_result_json", 0) == 0) for trial in trials if trial.get("status") != "unreadable")
            else "partial"
        )
    )
    payload = {
        "schema": "paper_i_spin_boson_source_lock_audit_v1",
        "record_id": record_id,
        "generated_utc": _now_utc(),
        "status": lock_status,
        "ok": lock_status == "pass",
        "errors": errors,
        "warnings": warnings,
        "lock_basis": "command_and_summary_trial_param_overrides_match_source_lock_payload_plus_exact_sequence_target_hit",
        "trial_param_overrides_json": override_path_value,
        "trial_param_overrides_json_resolved": _safe_relative(override_path) if override_path is not None else None,
        "trial_param_overrides_sha256": override_sha256,
        "expected_trial_param_overrides_sha256": expected_override_sha256,
        "command_trial_param_overrides_json": command_override_value,
        "command_trial_param_overrides_json_match": command_path_match,
        "summary_json": _safe_relative(summary_path),
        "summary_trial_param_overrides_match": summary_match,
        "override_field_count": len(override_fields),
        "expected_override_field_count": expected_count,
        "required_route_critical_fields": list(required_fields),
        "missing_required_route_critical_fields": missing_required,
        "result_field_serialization_status": result_serialization_status,
        "result_jsons_inspected": len(result_paths),
        "source_lock_reference_json": reference_path_value,
        "source_lock_reference_sequence_sha256": _sequence_sha256(reference_history) if reference_history else None,
        "zero_noise_adaptive_pass_evidence_json": None,
        "zero_noise_adaptive_pass_evidence_sha256": None,
        "trials": trials,
    }
    audit_path = out_root / "source_lock_audit.json"
    _write_json(audit_path, payload)
    if lock_status == "pass" and str(row.get("ladder_rung_id") or "").strip() == SPIN_BOSON_ZERO_NOISE_ADAPTIVE_RUNG_ID:
        pass_trial = next((trial for trial in trials if trial.get("sequence_exact_match") and trial.get("target_hit")), None)
        if pass_trial is not None:
            current_path = _latest_progress_current_json_path(progress_dir)
            evidence_path = out_root / "zero_noise_adaptive_pass_evidence.json"
            evidence = {
                "schema": SPIN_BOSON_ZERO_NOISE_PASS_EVIDENCE_SCHEMA,
                "record_id": record_id,
                "generated_utc": _now_utc(),
                "status": "pass",
                "case_id": str(row.get("source_lock_case_id") or row.get("benchmark_ids") or ""),
                "result_json": pass_trial.get("result_json"),
                "current_json": _safe_relative(current_path) if current_path is not None else None,
                "source_lock_audit_json": _safe_relative(audit_path),
                "source_lock_reference_json": reference_path_value,
                "source_lock_reference_sha256": str(
                    row.get("source_lock_reference_sha256")
                    or row.get("source_lock_expected_reference_sha256")
                    or ""
                ),
                "source_lock_sequence_sha256": pass_trial.get("source_lock_sequence_sha256"),
                "sequence_sha256": pass_trial.get("sequence_sha256"),
                "stop_reason": pass_trial.get("stop_reason"),
                "target_error": pass_trial.get("target_error"),
                "target_abs_delta_e": float(target_threshold),
                "abs_delta_e": pass_trial.get("abs_delta_e"),
                "energy": pass_trial.get("energy"),
                "ansatz_depth": pass_trial.get("ansatz_depth"),
                "target_hit_success_stop_reason": row.get("target_hit_success_stop_reason") or "benchmark_abs_delta_e_target",
                "route_faithfulness_ladder_schema": SPIN_BOSON_ROUTE_FAITHFULNESS_LADDER_SCHEMA,
                "ladder_rung_id": SPIN_BOSON_ZERO_NOISE_ADAPTIVE_RUNG_ID,
                "source_lock_audit_status": lock_status,
            }
            _write_json(evidence_path, evidence)
            payload["zero_noise_adaptive_pass_evidence_json"] = _safe_relative(evidence_path)
            payload["zero_noise_adaptive_pass_evidence_sha256"] = sha256_file(evidence_path)
            _write_json(audit_path, payload)
    return payload


def _latest_trial_dir(run_root: Path) -> str | None:
    if not run_root.exists():
        return None
    trials = [path for path in run_root.glob("trial_*") if path.is_dir()]
    if not trials:
        return None
    latest = max(trials, key=lambda path: path.stat().st_mtime)
    try:
        return str(latest.relative_to(run_root.parent))
    except ValueError:
        return str(latest)


def _progress_source_candidates(progress_dir: Path, filename: str) -> list[Path]:
    paths = [progress_dir / filename]
    try:
        nested = [
            path
            for path in progress_dir.glob(f"*/{filename}")
            if path.parent != progress_dir
        ]
    except Exception:
        nested = []
    nested.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)
    paths.extend(nested)
    return paths


def _progress_source_label(progress_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(progress_dir))
    except ValueError:
        return str(path)


def _jsonl_tail(path: Path, *, max_lines: int = 40, max_bytes: int = 256 * 1024) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - int(max_bytes)), os.SEEK_SET)
            text = fh.read().decode("utf-8", errors="replace")
    except Exception:
        return []
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, Mapping):
            events.append(dict(payload))
    return events[-max(1, int(max_lines)):]


def _latest_trial_event_tail(progress_dir: Path, *, max_lines: int = 40) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in _progress_source_candidates(progress_dir, "trial_events.jsonl"):
        events.extend(_jsonl_tail(path, max_lines=max_lines))
    events.sort(key=lambda event: str(event.get("timestamp_utc") or event.get("updated_utc") or ""))
    return events[-max(1, int(max_lines)):]


def _compact_child_progress(payload: Mapping[str, Any]) -> dict[str, Any]:
    child = payload.get("last_child_heartbeat")
    if not isinstance(child, Mapping):
        return {}
    progress = child.get("progress")
    if not isinstance(progress, Mapping):
        progress = {}
    keys = (
        "depth",
        "energy",
        "delta_abs_current",
        "delta_e",
        "abs_delta_e",
        "max_grad",
        "stop_reason_so_far",
        "selected_generator",
        "selected_position",
    )
    progress_summary = {key: progress.get(key) for key in keys if key in progress}
    return {
        "status": child.get("status"),
        "elapsed_s": child.get("elapsed_s"),
        "pid": child.get("pid"),
        "last_ai_log_event": child.get("last_ai_log_event"),
        "progress": progress_summary,
    }


def _compact_progress_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "state",
        "trial_number",
        "value",
        "mode",
        "study_name",
        "benchmark_id",
        "family",
        "abs_delta_e",
        "delta_e",
        "primary_error",
        "best_trial_number",
        "best_value",
        "best_primary_error",
        "best_delta_e",
        "best_first_crossing",
        "best_resource_score",
        "pareto_front_size",
        "active_child_count",
        "child_label",
        "child_benchmark_id",
        "child_trial_number",
    )
    out = {key: payload.get(key) for key in keys if key in payload}
    child = _compact_child_progress(payload)
    if child:
        out["child"] = child
    return out


def _latest_iteration_event(events: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(list(events)):
        if event.get("event") == "child_ai_log" or "depth" in event:
            progress = event.get("progress") if isinstance(event.get("progress"), Mapping) else {}
            return {
                "timestamp_utc": event.get("timestamp_utc"),
                "trial_number": event.get("trial_number"),
                "benchmark_id": event.get("benchmark_id"),
                "depth": event.get("depth") if event.get("depth") is not None else progress.get("depth"),
                "energy": event.get("energy") if event.get("energy") is not None else progress.get("energy"),
                "delta_abs_current": event.get("delta_abs_current") if event.get("delta_abs_current") is not None else progress.get("delta_abs_current"),
                "max_grad": event.get("max_grad") if event.get("max_grad") is not None else progress.get("max_grad"),
                "stop_reason_so_far": event.get("stop_reason_so_far") if event.get("stop_reason_so_far") is not None else progress.get("stop_reason_so_far"),
            }
    return None


def _latest_trial_completed_event(events: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(list(events)):
        if event.get("event") == "trial_completed":
            return dict(event)
    return None


def _write_live_status(
    out_root: Path,
    *,
    heartbeat: Mapping[str, Any],
    current_payload: Mapping[str, Any],
    current_best_payload: Mapping[str, Any],
    current_parseable: bool,
    current_best_parseable: bool,
) -> None:
    progress_dir = out_root / "progress"
    events = _latest_trial_event_tail(progress_dir, max_lines=40)
    compact = {
        "schema": "phase3_live_status_v1",
        "record_id": heartbeat.get("record_id"),
        "timestamp_utc": heartbeat.get("timestamp_utc"),
        "state": heartbeat.get("state"),
        "host": heartbeat.get("host"),
        "elapsed_s": heartbeat.get("elapsed_s"),
        "phase3_pid": heartbeat.get("phase3_pid"),
        "phase3_returncode": heartbeat.get("phase3_returncode"),
        "latest_trial_dir": heartbeat.get("latest_trial_dir"),
        "current_trial_number": heartbeat.get("current_trial_number"),
        "current_benchmark_id": heartbeat.get("current_benchmark_id"),
        "current_parseable": bool(current_parseable),
        "current_best_parseable": bool(current_best_parseable),
        "current_path": heartbeat.get("current_path"),
        "current_best_path": heartbeat.get("current_best_path"),
        "trial_events_path": heartbeat.get("trial_events_path"),
        "progress_freshness": heartbeat.get("progress_freshness"),
        "current_summary": _compact_progress_payload(current_payload),
        "current_best_summary": _compact_progress_payload(current_best_payload),
        "latest_iteration_event": _latest_iteration_event(events),
        "latest_trial_completed_event": _latest_trial_completed_event(events),
        "trial_event_tail": events,
    }
    full_payload = {
        **compact,
        "raw_current": dict(current_payload) if isinstance(current_payload, Mapping) else {},
        "raw_current_best": dict(current_best_payload) if isinstance(current_best_payload, Mapping) else {},
    }
    _write_json(progress_dir / "live_status.json", full_payload)
    _append_jsonl(progress_dir / "live_status.jsonl", compact)


def _latest_progress_json_mapping(
    progress_dir: Path,
    filename: str,
    *,
    source_label: str,
    observability_errors: list[str],
) -> tuple[Mapping[str, Any], bool, Path | None]:
    latest_payload: Mapping[str, Any] = {}
    latest_path: Path | None = None
    latest_timestamp: datetime | None = None
    parseable_any = False
    for path in _progress_source_candidates(progress_dir, filename):
        path_source_label = _progress_source_label(progress_dir, path)
        payload, parseable = _read_json_mapping(
            path,
            source_label=path_source_label,
            observability_errors=observability_errors,
        )
        if not parseable:
            continue
        parseable_any = True
        timestamp = _progress_timestamp(
            payload,
            path,
            source_label=path_source_label,
            observability_errors=observability_errors,
        )
        if latest_timestamp is None or (timestamp is not None and timestamp > latest_timestamp):
            latest_payload = payload
            latest_path = path
            latest_timestamp = timestamp
    return latest_payload, parseable_any, latest_path


def _read_current(progress_dir: Path) -> Mapping[str, Any]:
    errors: list[str] = []
    payload, parseable, _path = _latest_progress_json_mapping(
        progress_dir,
        "current.json",
        source_label="current.json",
        observability_errors=errors,
    )
    return payload if parseable else {}


def _read_json_mapping(path: Path, *, source_label: str, observability_errors: list[str]) -> tuple[Mapping[str, Any], bool]:
    if not path.exists():
        return {}, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        observability_errors.append(f"progress_source:{source_label}:json:{type(exc).__name__}:{exc}")
        return {}, False
    if not isinstance(payload, Mapping):
        observability_errors.append(f"progress_source:{source_label}:not_mapping")
        return {}, False
    return payload, True


def _trial_numbers_equal(left: Any, right: Any) -> bool:
    try:
        return int(left) == int(right)
    except Exception:
        return str(left) == str(right)


def _current_best_not_clobbered(current_best: Mapping[str, Any], *, parseable: bool) -> bool:
    if not parseable:
        return False
    best_trial_number = current_best.get("best_trial_number")
    if best_trial_number is None:
        return False
    if "trial_number" not in current_best:
        return True
    return _trial_numbers_equal(current_best.get("trial_number"), best_trial_number)


def _progress_timestamp(
    payload: Mapping[str, Any],
    path: Path,
    *,
    source_label: str,
    observability_errors: list[str],
) -> datetime | None:
    for key in ("timestamp_utc", "updated_utc", "generated_utc", "completed_utc"):
        parsed = _parse_utc_timestamp(payload.get(key))
        if parsed is not None:
            return parsed
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception as exc:
        observability_errors.append(f"progress_source:{source_label}:mtime:{type(exc).__name__}:{exc}")
        return None


def _latest_trial_event_timestamp(path: Path, *, observability_errors: list[str]) -> datetime | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 64 * 1024), os.SEEK_SET)
            text = fh.read().decode("utf-8", errors="replace")
    except Exception as exc:
        observability_errors.append(f"progress_source:trial_events.jsonl:read:{type(exc).__name__}:{exc}")
        return None
    for line in reversed(text.splitlines()):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        parsed = _parse_utc_timestamp(payload.get("timestamp_utc") or payload.get("updated_utc"))
        if parsed is not None:
            return parsed
    if text.strip():
        observability_errors.append("progress_source:trial_events.jsonl:no_parseable_timestamp_in_tail")
    return None


def _latest_trial_event_source(
    progress_dir: Path,
    *,
    observability_errors: list[str],
) -> tuple[datetime | None, Path | None]:
    latest_timestamp: datetime | None = None
    latest_path: Path | None = None
    for path in _progress_source_candidates(progress_dir, "trial_events.jsonl"):
        timestamp = _latest_trial_event_timestamp(path, observability_errors=observability_errors)
        if timestamp is None:
            continue
        if latest_timestamp is None or timestamp > latest_timestamp:
            latest_timestamp = timestamp
            latest_path = path
    return latest_timestamp, latest_path


def _file_present_nonempty(path: Path, *, source_label: str, observability_errors: list[str]) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except Exception as exc:
        observability_errors.append(f"progress_source:{source_label}:stat:{type(exc).__name__}:{exc}")
        return False


def _progress_file_present_nonempty(
    progress_dir: Path,
    filename: str,
    *,
    observability_errors: list[str],
) -> tuple[bool, Path | None]:
    for path in _progress_source_candidates(progress_dir, filename):
        present = _file_present_nonempty(
            path,
            source_label=_progress_source_label(progress_dir, path),
            observability_errors=observability_errors,
        )
        if present:
            return True, path
    return False, None


def _progress_freshness_payload(
    progress_dir: Path,
    *,
    now: datetime,
    stale_after_s: float,
    heartbeat_interval_s: float,
    observability_errors: list[str],
) -> dict[str, Any]:
    source_timestamps: dict[str, datetime] = {}
    source_paths: dict[str, str] = {}
    for source_label in ("current.json", "current_best.json", "status_snapshot.json"):
        payload, parseable, path = _latest_progress_json_mapping(
            progress_dir,
            source_label,
            source_label=source_label,
            observability_errors=observability_errors,
        )
        if not parseable:
            continue
        if path is None:
            continue
        timestamp = _progress_timestamp(
            payload,
            path,
            source_label=_progress_source_label(progress_dir, path),
            observability_errors=observability_errors,
        )
        if timestamp is not None:
            source_timestamps[source_label] = timestamp
            source_paths[source_label] = _progress_source_label(progress_dir, path)
    trial_events_ts, trial_events_path = _latest_trial_event_source(
        progress_dir,
        observability_errors=observability_errors,
    )
    if trial_events_ts is not None:
        source_timestamps["trial_events.jsonl"] = trial_events_ts
        if trial_events_path is not None:
            source_paths["trial_events.jsonl"] = _progress_source_label(progress_dir, trial_events_path)

    def _age(timestamp: datetime | None) -> float | None:
        if timestamp is None:
            return None
        return round(max(0.0, float((now - timestamp).total_seconds())), 3)

    source_ages = {source: _age(timestamp) for source, timestamp in source_timestamps.items()}
    if source_timestamps:
        last_source, last_timestamp = max(source_timestamps.items(), key=lambda item: item[1])
        progress_age_s = _age(last_timestamp)
        last_progress_update_utc = last_timestamp.isoformat()
        last_progress_source = last_source
    else:
        progress_age_s = None
        last_progress_update_utc = None
        last_progress_source = None
    progress_stale = bool(progress_age_s is not None and progress_age_s > float(stale_after_s))
    return {
        "schema": "phase3_progress_freshness_v1",
        "stale_after_s": float(stale_after_s),
        "heartbeat_interval_s": float(heartbeat_interval_s),
        "progress_stale": progress_stale,
        "progress_age_s": progress_age_s,
        "source_ages_s": source_ages,
        "source_timestamps_utc": {source: timestamp.isoformat() for source, timestamp in source_timestamps.items()},
        "source_paths": source_paths,
        "current_age_s": source_ages.get("current.json"),
        "current_best_age_s": source_ages.get("current_best.json"),
        "status_snapshot_age_s": source_ages.get("status_snapshot.json"),
        "trial_events_age_s": source_ages.get("trial_events.jsonl"),
        "last_progress_update_utc": last_progress_update_utc,
        "last_progress_source": last_progress_source,
        "last_progress_path": source_paths.get(last_progress_source) if last_progress_source is not None else None,
    }


def snapshot_sqlite(out_root: Path) -> list[str]:
    errors: list[str] = []
    src = out_root / "study.sqlite3"
    dest = out_root / "progress" / "study_snapshot.sqlite3"
    if not src.exists() or src.stat().st_size <= 0:
        return errors
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".sqlite3.tmp")
    try:
        if tmp.exists():
            tmp.unlink()
        with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as source, sqlite3.connect(tmp) as target:
            source.backup(target)
        tmp.replace(dest)
        return errors
    except Exception as exc:
        errors.append(f"sqlite_backup:{type(exc).__name__}:{exc}")
        try:
            shutil.copy2(src, dest)
        except Exception as copy_exc:
            errors.append(f"sqlite_copy:{type(copy_exc).__name__}:{copy_exc}")
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
    return errors


def write_heartbeat(
    out_root: Path,
    *,
    record_id: str,
    state: str,
    started: float,
    phase3_proc: subprocess.Popen[str] | None = None,
    phase3_returncode: int | None = None,
    observability_errors: Sequence[str] = (),
) -> dict[str, Any]:
    progress_dir = out_root / "progress"
    run_root = out_root / "run"
    current = _read_current(progress_dir)
    errors = list(observability_errors)
    now = datetime.now(timezone.utc)
    heartbeat_interval_s = _heartbeat_interval_s(errors)
    stale_after_s = _progress_stale_after_s(errors)
    current_payload, current_parseable, current_path = _latest_progress_json_mapping(
        progress_dir,
        "current.json",
        source_label="current.json",
        observability_errors=errors,
    )
    current_best_payload, current_best_parseable, current_best_path = _latest_progress_json_mapping(
        progress_dir,
        "current_best.json",
        source_label="current_best.json",
        observability_errors=errors,
    )
    trial_events_present, trial_events_path = _progress_file_present_nonempty(
        progress_dir,
        "trial_events.jsonl",
        observability_errors=errors,
    )
    freshness = _progress_freshness_payload(
        progress_dir,
        now=now,
        stale_after_s=stale_after_s,
        heartbeat_interval_s=heartbeat_interval_s,
        observability_errors=errors,
    )
    sqlite_path = out_root / "study.sqlite3"
    heartbeat = {
        "schema": "phase3_chtc_heartbeat_v1",
        "record_id": record_id,
        "timestamp_utc": now.isoformat(),
        "heartbeat_present": True,
        "host": socket.gethostname(),
        "wrapper_pid": os.getpid(),
        "phase3_pid": None if phase3_proc is None else int(phase3_proc.pid),
        "phase3_returncode": phase3_returncode,
        "elapsed_s": round(float(time.monotonic() - started), 3),
        "state": state,
        "summary_exists": (out_root / "summary.json").exists() or (run_root / "summary.json").exists(),
        "study_sqlite_exists": sqlite_path.exists(),
        "study_sqlite_size": sqlite_path.stat().st_size if sqlite_path.exists() else 0,
        "latest_trial_dir": _latest_trial_dir(run_root),
        "current_trial_number": current.get("trial_number"),
        "current_benchmark_id": current.get("benchmark_id"),
        "trial_events_present": bool(trial_events_present),
        "current_parseable": bool(current_parseable),
        "current_best_parseable": bool(current_best_parseable),
        "current_path": None if current_path is None else _progress_source_label(progress_dir, current_path),
        "current_best_path": None if current_best_path is None else _progress_source_label(progress_dir, current_best_path),
        "trial_events_path": None if trial_events_path is None else _progress_source_label(progress_dir, trial_events_path),
        "current_best_not_clobbered": _current_best_not_clobbered(
            current_best_payload,
            parseable=current_best_parseable,
        ),
        "stale_failure_detection_latency_s": round(float(stale_after_s + heartbeat_interval_s), 3),
        "progress_freshness": freshness,
        "active_child_count": 0 if phase3_proc is None or phase3_proc.poll() is not None else 1,
        "observability_errors": errors,
    }
    try:
        _write_live_status(
            out_root,
            heartbeat=heartbeat,
            current_payload=current_payload,
            current_best_payload=current_best_payload,
            current_parseable=current_parseable,
            current_best_parseable=current_best_parseable,
        )
    except Exception as exc:
        heartbeat["observability_errors"] = [
            *list(heartbeat.get("observability_errors") or []),
            f"live_status:{type(exc).__name__}:{exc}",
        ]
    _write_json(out_root / "heartbeat.json", heartbeat)
    _write_json(progress_dir / "heartbeat_snapshot.json", heartbeat)
    return heartbeat


def _find_summary(run_root: Path) -> Path | None:
    direct = run_root / "summary.json"
    if direct.exists():
        return direct
    nested = sorted(run_root.glob("**/summary.json"))
    return nested[0] if nested else None


def copy_summary_if_available(out_root: Path, run_root: Path) -> bool:
    summary = _find_summary(run_root)
    if summary is None:
        return False
    shutil.copyfile(summary, out_root / "summary.json")
    return True


def run_phase3_with_heartbeat(command: Sequence[str], *, out_root: Path, record_id: str, env: Mapping[str, str]) -> int:
    progress_dir = out_root / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    errors: list[str] = []
    interval = _heartbeat_interval_s(errors)
    terminate_on_stale_progress = _truthy_env("PHASE3_TERMINATE_ON_STALE_PROGRESS")
    require_first_progress_within_s = _require_first_progress_within_s(errors)
    _append_jsonl(progress_dir / "wrapper_events.jsonl", {"schema": "phase3_wrapper_event_v1", "timestamp_utc": _now_utc(), "record_id": record_id, "event": "phase3_starting"})
    write_heartbeat(out_root, record_id=record_id, state="starting", started=started)
    proc = subprocess.Popen([str(part) for part in command], env=dict(env), text=True, start_new_session=True)
    _append_jsonl(
        progress_dir / "wrapper_events.jsonl",
        {
            "schema": "phase3_wrapper_event_v1",
            "timestamp_utc": _now_utc(),
            "record_id": record_id,
            "event": "phase3_started",
            "pid": int(proc.pid),
        },
    )
    returncode: int | None = None
    terminate_signal: int | None = None
    previous_handlers: dict[int, Any] = {}

    def _forward_signal(signum: int, _frame: Any) -> None:
        nonlocal terminate_signal
        terminate_signal = int(signum)
        _append_jsonl(
            progress_dir / "wrapper_events.jsonl",
            {
                "schema": "phase3_wrapper_event_v1",
                "timestamp_utc": _now_utc(),
                "record_id": record_id,
                "event": "phase3_signal_forwarded",
                "pid": int(proc.pid),
                "signal": int(signum),
            },
        )
        try:
            os.killpg(int(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception as exc:
            errors.append(f"forward_signal:{int(signum)}:{type(exc).__name__}:{exc}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[int(signum)] = signal.getsignal(signum)
            signal.signal(signum, _forward_signal)
        except Exception as exc:  # pragma: no cover - platform dependent
            errors.append(f"install_signal:{int(signum)}:{type(exc).__name__}:{exc}")
    try:
        while True:
            try:
                returncode = proc.wait(timeout=interval)
                break
            except subprocess.TimeoutExpired:
                errors.extend(snapshot_sqlite(out_root))
                state = "terminating" if terminate_signal is not None else "running"
                heartbeat = write_heartbeat(out_root, record_id=record_id, state=state, started=started, phase3_proc=proc, observability_errors=errors)
                freshness = heartbeat.get("progress_freshness", {})
                progress_stale = bool(isinstance(freshness, Mapping) and freshness.get("progress_stale"))
                no_first_progress_timeout = (
                    terminate_on_stale_progress
                    and float(require_first_progress_within_s) > 0.0
                    and isinstance(freshness, Mapping)
                    and freshness.get("last_progress_update_utc") is None
                    and float(time.monotonic() - started) > float(require_first_progress_within_s)
                )
                if terminate_on_stale_progress and proc.poll() is None and (progress_stale or no_first_progress_timeout):
                    reason = "stale_progress" if progress_stale else "missing_initial_progress"
                    _append_jsonl(
                        progress_dir / "wrapper_events.jsonl",
                        {
                            "schema": "phase3_wrapper_event_v1",
                            "timestamp_utc": _now_utc(),
                            "record_id": record_id,
                            "event": "phase3_monitor_termination_requested",
                            "pid": int(proc.pid),
                            "reason": reason,
                            "progress_freshness": freshness,
                        },
                    )
                    try:
                        os.killpg(int(proc.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    except Exception as exc:
                        errors.append(f"terminate_for_{reason}:{type(exc).__name__}:{exc}")
                    try:
                        proc.wait(timeout=30.0)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(int(proc.pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        except Exception as exc:
                            errors.append(f"sigkill_for_{reason}:{type(exc).__name__}:{exc}")
                        proc.wait()
                    returncode = 124
                    break
                if terminate_signal is not None and proc.poll() is None:
                    try:
                        returncode = proc.wait(timeout=30.0)
                        break
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(int(proc.pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        except Exception as exc:
                            errors.append(f"sigkill_after_signal:{type(exc).__name__}:{exc}")
                        returncode = proc.wait()
                        break
    except BaseException:
        if proc.poll() is None:
            try:
                os.killpg(int(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        errors.extend(snapshot_sqlite(out_root))
        write_heartbeat(out_root, record_id=record_id, state="terminated", started=started, phase3_proc=proc, observability_errors=errors)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except Exception:
                pass
    errors.extend(snapshot_sqlite(out_root))
    if terminate_signal is not None:
        returncode = 128 + int(terminate_signal)
    state = "completed" if int(returncode or 0) == 0 else ("terminated" if terminate_signal is not None else "failed")
    write_heartbeat(
        out_root,
        record_id=record_id,
        state=state,
        started=started,
        phase3_proc=proc,
        phase3_returncode=int(returncode or 0),
        observability_errors=errors,
    )
    _append_jsonl(
        progress_dir / "wrapper_events.jsonl",
        {
            "schema": "phase3_wrapper_event_v1",
            "timestamp_utc": _now_utc(),
            "record_id": record_id,
            "event": "phase3_exited",
            "pid": int(proc.pid),
            "returncode": int(returncode or 0),
        },
    )
    return int(returncode or 0)


def write_final_status(out_root: Path, payload: Mapping[str, Any]) -> None:
    progress_dir = out_root / "progress"
    _write_json(progress_dir / "final_status.json", payload)
    _write_json(out_root / "chtc_status.json", payload)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-id", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    record_id = str(args.record_id)
    records_path = Path(args.records)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    progress_dir = out_root / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    run_root = out_root / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status: dict[str, Any] = {
        "schema": "phase3_chtc_status_v1",
        "record_id": record_id,
        "records_path": str(records_path),
        "output_root": str(out_root),
        "started_utc": _now_utc(),
        "state": "starting",
        "returncode": None,
        "summary_exists": False,
    }
    try:
        row = load_record(records_path, record_id)
        env = _phase3_env(row)
        command = build_phase3_command(row, record_id=record_id, out_root=out_root, run_root=run_root, progress_dir=progress_dir)
        (out_root / "record.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out_root / "command.sh").write_text(" ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8")
        write_record_manifest(row, record_id=record_id, out_root=out_root, run_root=run_root, progress_dir=progress_dir, command=command)
        print("RUN", " ".join(shlex.quote(part) for part in command), flush=True)
        rc = run_phase3_with_heartbeat(command, out_root=out_root, record_id=record_id, env=env)
        summary_exists = copy_summary_if_available(out_root, run_root)
        snapshot_errors = snapshot_sqlite(out_root)
        source_lock_audit_status = None
        source_lock_audit_errors: list[str] = []
        try:
            source_lock_audit = _write_hh_source_lock_audit(
                row,
                record_id=record_id,
                out_root=out_root,
                run_root=run_root,
                command=command,
            )
            if source_lock_audit is None:
                source_lock_audit = _write_spin_boson_source_lock_audit(
                    row,
                    record_id=record_id,
                    out_root=out_root,
                    run_root=run_root,
                    command=command,
                    progress_dir=progress_dir,
                )
            if source_lock_audit is not None:
                source_lock_audit_status = str(source_lock_audit.get("status") or "")
                source_lock_audit_errors = [str(item) for item in source_lock_audit.get("errors", ()) or ()]
        except Exception as exc:
            source_lock_audit_status = "failed_to_write"
            source_lock_audit_errors = [f"{type(exc).__name__}: {exc}"]
            snapshot_errors = [*snapshot_errors, f"source_lock_audit:{type(exc).__name__}:{exc}"]
        if rc == 0 and not summary_exists:
            status["state"] = "failed"
            status["error"] = f"no summary.json found under {run_root}"
            rc = 1
        else:
            status["state"] = "completed" if rc == 0 else ("terminated" if rc >= 128 else "failed")
        status.update(
            {
                "returncode": int(rc),
                "completed_utc": _now_utc(),
                "elapsed_s": round(float(time.monotonic() - started), 3),
                "summary_exists": bool((out_root / "summary.json").exists()),
                "study_sqlite_exists": bool((out_root / "study.sqlite3").exists()),
                "sqlite_snapshot_exists": bool((progress_dir / "study_snapshot.sqlite3").exists()),
                "snapshot_errors": snapshot_errors,
                "source_lock_audit_exists": bool((out_root / "source_lock_audit.json").exists()),
                "source_lock_audit_status": source_lock_audit_status,
                "source_lock_audit_errors": source_lock_audit_errors,
                "zero_noise_adaptive_pass_evidence_exists": bool((out_root / "zero_noise_adaptive_pass_evidence.json").exists()),
            }
        )
        write_heartbeat(out_root, record_id=record_id, state=status["state"], started=started, phase3_returncode=int(rc), observability_errors=snapshot_errors)
        write_final_status(out_root, status)
        return int(rc)
    except BaseException as exc:
        status.update(
            {
                "state": "failed",
                "returncode": 1,
                "completed_utc": _now_utc(),
                "elapsed_s": round(float(time.monotonic() - started), 3),
                "summary_exists": bool((out_root / "summary.json").exists()),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        try:
            write_heartbeat(out_root, record_id=record_id, state="failed", started=started, observability_errors=[status["error"]])
            write_final_status(out_root, status)
        finally:
            print(status["error"], file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
