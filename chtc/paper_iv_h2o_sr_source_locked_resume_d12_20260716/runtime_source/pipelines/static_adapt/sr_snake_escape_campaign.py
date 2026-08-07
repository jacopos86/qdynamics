#!/usr/bin/env python3
"""Materialize source-locked SR-SNAKE disabled/saddle command pairs.

This module is intentionally a command *generator*, not a runner.  It reads
a preserved physical-operator-lane command, changes only the
explicit SR profile and execution-path controls listed below, and fails closed
if its normalized argv audit observes any other difference.  It never starts,
stops, or inspects a process.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping, Sequence

from pipelines.static_adapt.cli_config import _build_adapt_arg_parser

from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    SR_ESCAPE_DISABLED,
    SR_ESCAPE_SADDLE_ONLY,
    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
    SR_ROUTE_FAMILY,
    sr_route_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_COMMANDS_JSON = (
    REPO_ROOT
    / "raw_outputs"
    / "paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708"
    / "commands.json"
)

CAMPAIGN_SCHEMA = "paper_i_hh_sr_snake_escape_command_campaign_v1"
COMMAND_RECORD_SCHEMA = "paper_i_hh_sr_snake_escape_command_record_v1"
ARGV_AUDIT_SCHEMA = "paper_i_hh_sr_snake_escape_argv_audit_v1"
PAIR_AUDIT_SCHEMA = "paper_i_hh_sr_snake_escape_pair_audit_v1"
RUNTIME_SOURCE_LOCK_SCHEMA = "paper_i_hh_sr_snake_escape_runtime_source_lock_v1"
RUNTIME_SOURCE_LOCK_VERIFICATION_SCHEMA = (
    "paper_i_hh_sr_snake_escape_runtime_source_lock_verification_v1"
)
EFFECTIVE_SETTINGS_AUDIT_SCHEMA = (
    "paper_i_hh_sr_snake_escape_effective_settings_audit_v1"
)
EXECUTION_CONTRACT_SCHEMA = "paper_i_hh_sr_snake_escape_execution_contract_v1"
ENVIRONMENT_CONTRACT_SCHEMA = (
    "paper_i_hh_sr_snake_escape_environment_contract_v1"
)
PRELAUNCH_VERIFICATION_SCHEMA = (
    "paper_i_hh_sr_snake_escape_prelaunch_verification_v1"
)
EXECUTION_DEPENDENCY_SCHEMA = (
    "paper_i_hh_sr_snake_escape_execution_dependency_v1"
)
SOURCE_ANCHOR_REFERENCE_SCHEMA = (
    "paper_i_hh_sr_snake_escape_source_anchor_reference_v1"
)
ANCHOR_VALIDATION_SCHEMA = (
    "paper_i_hh_sr_snake_escape_anchor_validation_v1"
)
CURRENT_REVISION_CONTROL_VALIDATION_SCHEMA = (
    "paper_i_hh_sr_snake_escape_current_revision_control_validation_v1"
)

ANCHOR_ENERGY_ABS_TOLERANCE = 1e-9
ANCHOR_EXACT_ENERGY_ABS_TOLERANCE = 1e-12

RUNTIME_SOURCE_LOCK_PATHS = (
    "pipelines/scaffold/hh_continuation_generators.py",
    "pipelines/scaffold/hh_continuation_pruning.py",
    "pipelines/scaffold/hh_continuation_scoring.py",
    "pipelines/static_adapt/adapt_candidate_record_cache.py",
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/builders/hh_pool_presets.py",
    "pipelines/static_adapt/cli_config.py",
    "pipelines/static_adapt/engine_support.py",
    "pipelines/static_adapt/joint_linear_solve.py",
    "pipelines/static_adapt/joint_step_warm_start.py",
    "pipelines/static_adapt/output_artifacts.py",
    "pipelines/static_adapt/route_a_funnel.py",
    "pipelines/static_adapt/route_a_schur_selector.py",
    "pipelines/static_adapt/route_a_trust_region.py",
    "pipelines/static_adapt/route_identity.py",
    "pipelines/static_adapt/resume_scaffold.py",
    "pipelines/static_adapt/selector_query_closure.py",
    "pipelines/static_adapt/sr_snake_escape_campaign.py",
    "pipelines/static_adapt/sr_snake_escape_controller.py",
    "src/quantum/ansatz_parameterization.py",
)

ADAPTIVE_TRUST_POLICY = "displacement_calibrated_unbounded_v2"
EXACT_PROJECTED_GROUPED_PADDING = "exact_projected_grouped_v1"

# These variables can alter Python import resolution, deterministic hashing, or
# numerical threading.  They are safe to record and are the only environment
# values locked by this generator.  In particular, the generator never copies
# the full parent environment (which may contain credentials) into a manifest.
LOCKED_EXECUTION_ENVIRONMENT_VARIABLES = (
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "PYTHONHOME",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
RECORD_TARGET_PATH_FIELDS = (
    "output_json",
    "current_json",
    "estimator_call_ledger_json",
    "log_path",
)
RECORD_CACHE_ENVIRONMENT_VARIABLES = (
    "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR",
    "STATIC_ADAPT_HH_POOL_CACHE_DIR",
    "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR",
)

PROFILE_FLAGS = frozenset(
    {
        "--phase0-pilot-enabled",
        "--phase0-no-pilot",
        "--phase2-enable-batching",
        "--phase2-no-batching",
        "--phase3-enable-batching",
        "--phase3-no-batching",
        "--phase3-runtime-split-max-subset-size",
        "--phase3-runtime-split-subset-sizes",
        "--phase3-runtime-split-child-padding-policy",
        "--historical-singleton-coordinate-solve-policy",
        "--historical-singleton-trust-region-update-policy",
        "--sr-powell-coordinate-chart-policy",
        "--sr-escape-mode",
    }
)
EXECUTION_FLAGS = frozenset(
    {
        "--output-json",
        "--adapt-current-json",
        "--adapt-estimator-call-ledger-json",
        "--skip-trajectory",
    }
)
SOURCE_TO_PROFILE_ALLOWED_FLAGS = PROFILE_FLAGS | EXECUTION_FLAGS
PAIR_ALLOWED_FLAGS = frozenset(
    {
        "--historical-singleton-coordinate-solve-policy",
        "--sr-escape-mode",
        "--output-json",
        "--adapt-current-json",
        "--adapt-estimator-call-ledger-json",
    }
)

PROFILE_SPECS: tuple[dict[str, str], ...] = (
    {
        "profile_key": SR_ESCAPE_DISABLED,
        "sr_escape_mode": SR_ESCAPE_DISABLED,
        "coordinate_solve_policy": (
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
        ),
        "trust_region_update_policy": ADAPTIVE_TRUST_POLICY,
    },
    {
        "profile_key": SR_ESCAPE_SADDLE_ONLY,
        "sr_escape_mode": SR_ESCAPE_SADDLE_ONLY,
        "coordinate_solve_policy": (
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
        ),
        "trust_region_update_policy": ADAPTIVE_TRUST_POLICY,
    },
)


def _campaign_route_profile(escape_mode: str) -> str:
    return sr_route_profile(
        escape_mode,
        powell_coordinate_chart_policy=(
            SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
        ),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_reference_number(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"source anchor {name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"source anchor {name} must be finite")
    return result


def build_source_anchor_reference(source_result_path: Path) -> dict[str, Any]:
    """Extract the strict disabled-route regression target from a source result."""

    path = Path(source_result_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("source anchor result must be a JSON object")
    adapt_vqe = payload.get("adapt_vqe")
    if not isinstance(adapt_vqe, Mapping):
        raise ValueError("source anchor result is missing adapt_vqe")
    history = adapt_vqe.get("history")
    operators = adapt_vqe.get("operators")
    if not isinstance(history, list):
        raise ValueError("source anchor adapt_vqe.history must be an array")
    if not isinstance(operators, list):
        raise ValueError("source anchor adapt_vqe.operators must be an array")
    success = adapt_vqe.get("success")
    if not isinstance(success, bool):
        raise ValueError("source anchor adapt_vqe.success must be boolean")
    stop_reason = adapt_vqe.get("stop_reason")
    if not isinstance(stop_reason, str) or not stop_reason:
        raise ValueError("source anchor adapt_vqe.stop_reason must be nonempty")
    ansatz_depth_raw = adapt_vqe.get("ansatz_depth")
    if isinstance(ansatz_depth_raw, bool):
        raise ValueError("source anchor adapt_vqe.ansatz_depth must be an integer")
    try:
        ansatz_depth = int(ansatz_depth_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "source anchor adapt_vqe.ansatz_depth must be an integer"
        ) from exc
    if ansatz_depth < 0 or float(ansatz_depth) != float(ansatz_depth_raw):
        raise ValueError(
            "source anchor adapt_vqe.ansatz_depth must be a nonnegative integer"
        )
    abs_delta_e = _finite_reference_number(
        "adapt_vqe.abs_delta_e", adapt_vqe.get("abs_delta_e")
    )
    if abs_delta_e < 0.0:
        raise ValueError("source anchor adapt_vqe.abs_delta_e must be nonnegative")
    return {
        "schema": SOURCE_ANCHOR_REFERENCE_SCHEMA,
        "source_result_json": str(path),
        "source_result_sha256": file_sha256(path),
        "success": success,
        "stop_reason": stop_reason,
        "history_length": len(history),
        "ansatz_depth": ansatz_depth,
        "operator_count": len(operators),
        "operator_sequence_sha256": payload_sha256(operators),
        "energy": _finite_reference_number(
            "adapt_vqe.energy", adapt_vqe.get("energy")
        ),
        "exact_gs_energy": _finite_reference_number(
            "adapt_vqe.exact_gs_energy", adapt_vqe.get("exact_gs_energy")
        ),
        "abs_delta_e": abs_delta_e,
        "energy_abs_tolerance": float(ANCHOR_ENERGY_ABS_TOLERANCE),
        "exact_energy_abs_tolerance": float(
            ANCHOR_EXACT_ENERGY_ABS_TOLERANCE
        ),
        "operator_sequence_match_required": True,
    }


def build_runtime_source_lock(
    runtime_repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Hash the exact current-code surfaces shared by every campaign row."""

    root = Path(runtime_repo_root).expanduser().resolve()
    files: list[dict[str, str]] = []
    for relative_path in RUNTIME_SOURCE_LOCK_PATHS:
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"runtime source-lock file is missing: {path}")
        files.append(
            {
                "path": str(relative_path),
                "sha256": file_sha256(path),
            }
        )
    aggregate_payload = {
        str(row["path"]): str(row["sha256"]) for row in files
    }
    return {
        "schema": RUNTIME_SOURCE_LOCK_SCHEMA,
        "runtime_repo_root": str(root),
        "files": files,
        "aggregate_sha256": payload_sha256(aggregate_payload),
    }


def verify_runtime_source_lock(
    expected_lock: Mapping[str, Any],
    *,
    runtime_repo_root: Path | None = None,
) -> dict[str, Any]:
    """Recompute and strictly verify a recorded runtime source lock.

    This API is intended to run immediately before a generated command.  It
    checks the same explicit dependency surface used at generation time; it
    does not claim to hash the entire Python dependency closure.
    """

    if str(expected_lock.get("schema", "")) != RUNTIME_SOURCE_LOCK_SCHEMA:
        raise ValueError("runtime source lock has an unsupported schema")
    expected_root_raw = expected_lock.get("runtime_repo_root")
    if not isinstance(expected_root_raw, str) or not expected_root_raw:
        raise ValueError("runtime source lock is missing runtime_repo_root")
    expected_root = Path(expected_root_raw).expanduser().resolve()
    root = (
        expected_root
        if runtime_repo_root is None
        else Path(runtime_repo_root).expanduser().resolve()
    )
    if root != expected_root:
        raise ValueError(
            "runtime source-lock root mismatch: "
            f"expected {expected_root}, found {root}"
        )

    expected_files_raw = expected_lock.get("files")
    if not isinstance(expected_files_raw, list):
        raise ValueError("runtime source lock files must be an array")
    expected_files: dict[str, str] = {}
    for raw in expected_files_raw:
        if not isinstance(raw, Mapping):
            raise ValueError("runtime source lock contains a non-object file row")
        relative_path = str(raw.get("path", ""))
        sha256 = str(raw.get("sha256", ""))
        if not relative_path or relative_path in expected_files:
            raise ValueError("runtime source-lock paths must be nonempty and unique")
        if len(sha256) != 64:
            raise ValueError(
                f"runtime source-lock hash is invalid for {relative_path}"
            )
        expected_files[relative_path] = sha256
    if tuple(expected_files) != RUNTIME_SOURCE_LOCK_PATHS:
        raise ValueError(
            "runtime source-lock dependency surface differs from the current "
            "declared dependency surface"
        )

    current = build_runtime_source_lock(root)
    current_files = {
        str(row["path"]): str(row["sha256"]) for row in current["files"]
    }
    mismatches = [
        relative_path
        for relative_path in RUNTIME_SOURCE_LOCK_PATHS
        if expected_files.get(relative_path) != current_files.get(relative_path)
    ]
    expected_aggregate = str(expected_lock.get("aggregate_sha256", ""))
    aggregate_equal = expected_aggregate == str(current["aggregate_sha256"])
    if mismatches or not aggregate_equal:
        raise ValueError(
            "runtime source-lock verification failed: "
            f"changed_files={mismatches}, aggregate_equal={aggregate_equal}"
        )
    return {
        "schema": RUNTIME_SOURCE_LOCK_VERIFICATION_SCHEMA,
        "status": "pass",
        "runtime_repo_root": str(root),
        "aggregate_sha256": str(current["aggregate_sha256"]),
        "verified_file_count": len(current_files),
        "changed_files": [],
        "dependency_surface_scope": "explicit_declared_runtime_files",
    }


def _build_environment_contract() -> dict[str, Any]:
    return {
        "schema": ENVIRONMENT_CONTRACT_SCHEMA,
        "policy": "parent_environment_with_locked_safe_variables_v1",
        "inherit_parent_environment": True,
        "secret_values_recorded": False,
        "locked_safe_variables": {
            name: os.environ.get(name)
            for name in LOCKED_EXECUTION_ENVIRONMENT_VARIABLES
        },
        "unlisted_environment_values_recorded": False,
        "verification_requirement": (
            "locked_safe_variables_must_match_immediately_prelaunch"
        ),
    }


def _build_execution_contract(runtime_repo_root: Path) -> dict[str, Any]:
    launch_cwd = Path(runtime_repo_root).expanduser().resolve()
    python_executable = Path(sys.executable).expanduser().resolve()
    if not launch_cwd.is_dir():
        raise ValueError(f"launch cwd is missing: {launch_cwd}")
    if not python_executable.is_file():
        raise ValueError(f"runtime Python executable is missing: {python_executable}")
    return {
        "schema": EXECUTION_CONTRACT_SCHEMA,
        "launch_cwd": str(launch_cwd),
        "runtime_python_executable": str(python_executable),
        "environment": _build_environment_contract(),
        "generator_launches_processes": False,
        "execution_authorized": False,
    }


def _verify_environment_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if str(contract.get("schema", "")) != ENVIRONMENT_CONTRACT_SCHEMA:
        raise ValueError("execution environment contract has an unsupported schema")
    if contract.get("secret_values_recorded") is not False:
        raise ValueError("execution environment contract may not record secrets")
    expected = contract.get("locked_safe_variables")
    if not isinstance(expected, Mapping):
        raise ValueError("execution environment contract is missing locked variables")
    if (
        len(expected) != len(LOCKED_EXECUTION_ENVIRONMENT_VARIABLES)
        or set(expected) != set(LOCKED_EXECUTION_ENVIRONMENT_VARIABLES)
    ):
        raise ValueError("execution environment locked-variable surface changed")
    changed = [
        name
        for name in LOCKED_EXECUTION_ENVIRONMENT_VARIABLES
        if expected.get(name) != os.environ.get(name)
    ]
    if changed:
        raise ValueError(
            "execution environment verification failed for locked variables: "
            f"{changed}"
        )
    return {
        "schema": ENVIRONMENT_CONTRACT_SCHEMA,
        "status": "pass",
        "verified_variables": list(LOCKED_EXECUTION_ENVIRONMENT_VARIABLES),
        "changed_variables": [],
        "secret_values_recorded": False,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_argv(argv: Sequence[str]) -> tuple[tuple[str, ...], dict[str, str | None]]:
    """Parse the narrow direct-module CLI representation used by the source.

    Unknown flags are accepted and retained.  Duplicate flags and ``--x=y``
    spellings are rejected because either would make the normalized diff
    ambiguous.
    """

    tokens = [str(token) for token in argv]
    if not tokens:
        raise ValueError("command argv must be nonempty")
    first_flag = next(
        (index for index, token in enumerate(tokens) if token.startswith("--")),
        len(tokens),
    )
    prefix = tuple(tokens[:first_flag])
    if not prefix:
        raise ValueError("command argv must retain an executable/module prefix")
    options: dict[str, str | None] = {}
    index = first_flag
    while index < len(tokens):
        flag = tokens[index]
        if not flag.startswith("--") or flag == "--":
            raise ValueError(f"ambiguous command token at argv[{index}]: {flag!r}")
        if "=" in flag:
            raise ValueError(f"--flag=value spelling is unsupported: {flag!r}")
        if flag in options:
            raise ValueError(f"duplicate command flag is unsupported: {flag}")
        value: str | None = None
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            value = tokens[index + 1]
            index += 1
        options[flag] = value
        index += 1
    return prefix, options


def _adapt_cli_flag_tokens(argv: Sequence[str]) -> list[str]:
    prefix, _options = _parse_argv(argv)
    if prefix[1:] != ("-m", "pipelines.static_adapt.adapt_pipeline"):
        raise ValueError(
            "command must use the direct adapt_pipeline module prefix; "
            f"found {list(prefix)!r}"
        )
    return [str(token) for token in argv[len(prefix) :]]


def _normalize_setting_value(value: Any, *, launch_cwd: Path) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            label = "nan"
        elif value > 0.0:
            label = "positive_infinity"
        else:
            label = "negative_infinity"
        return {"nonfinite_float": label}
    if isinstance(value, Path):
        path = value.expanduser()
        if not path.is_absolute():
            path = launch_cwd / path
        return {"absolute_path": str(path.resolve())}
    if isinstance(value, (list, tuple)):
        return [
            _normalize_setting_value(item, launch_cwd=launch_cwd)
            for item in value
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_setting_value(item, launch_cwd=launch_cwd)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise ValueError(
        "adapt CLI parser produced an unsupported normalized setting type: "
        f"{type(value).__name__}"
    )


def _parse_normalized_effective_settings(
    argv: Sequence[str], *, launch_cwd: Path
) -> tuple[dict[str, Any], argparse.ArgumentParser]:
    """Resolve every adapt CLI default with the production parser.

    ``parse_args`` (rather than ``parse_known_args``) is deliberate: unknown
    flags and trailing tokens fail closed.  Parser diagnostics are captured so
    generation remains a library-safe ValueError API rather than exiting the
    process.
    """

    parser = _build_adapt_arg_parser(adapt_gradient_parity_rtol=1e-8)
    tokens = _adapt_cli_flag_tokens(argv)
    parser_error = io.StringIO()
    try:
        with contextlib.redirect_stderr(parser_error):
            namespace = parser.parse_args(tokens)
    except SystemExit as exc:
        diagnostic = parser_error.getvalue().strip()
        raise ValueError(
            "adapt CLI parser rejected command argv"
            + (f": {diagnostic}" if diagnostic else "")
        ) from exc

    expected_destinations = {
        str(action.dest)
        for action in parser._actions
        if action.dest != argparse.SUPPRESS
        and action.default != argparse.SUPPRESS
    }
    raw_settings = vars(namespace)
    missing = sorted(expected_destinations - set(raw_settings))
    unknown = sorted(set(raw_settings) - expected_destinations)
    if missing or unknown:
        raise ValueError(
            "adapt CLI parser settings could not be completely resolved: "
            f"missing={missing}, unknown={unknown}"
        )
    normalized = {
        destination: _normalize_setting_value(
            raw_settings[destination], launch_cwd=launch_cwd
        )
        for destination in sorted(expected_destinations)
    }
    return normalized, parser


def audit_effective_settings_diff(
    source_argv: Sequence[str],
    effective_argv: Sequence[str],
    *,
    launch_cwd: Path,
    allowed_flags: set[str] | frozenset[str] = SOURCE_TO_PROFILE_ALLOWED_FLAGS,
    schema: str = EFFECTIVE_SETTINGS_AUDIT_SCHEMA,
) -> dict[str, Any]:
    """Audit parser-resolved settings, including defaults absent from argv."""

    source_settings, parser = _parse_normalized_effective_settings(
        source_argv, launch_cwd=launch_cwd
    )
    effective_settings, _ = _parse_normalized_effective_settings(
        effective_argv, launch_cwd=launch_cwd
    )
    allowed_destinations: set[str] = set()
    unresolved_allowed_flags: list[str] = []
    for flag in sorted(allowed_flags):
        action = parser._option_string_actions.get(flag)
        if action is None or action.dest == argparse.SUPPRESS:
            unresolved_allowed_flags.append(flag)
            continue
        allowed_destinations.add(str(action.dest))
    if unresolved_allowed_flags:
        raise ValueError(
            "effective-settings audit cannot resolve allowed CLI flags: "
            f"{unresolved_allowed_flags}"
        )

    changed_fields: list[dict[str, Any]] = []
    for destination in sorted(set(source_settings) | set(effective_settings)):
        source_value = source_settings.get(destination)
        effective_value = effective_settings.get(destination)
        if source_value != effective_value:
            changed_fields.append(
                {
                    "field": destination,
                    "source_value": source_value,
                    "effective_value": effective_value,
                }
            )
    changed_destinations = [str(row["field"]) for row in changed_fields]
    unexpected = sorted(set(changed_destinations) - allowed_destinations)
    if unexpected:
        raise ValueError(
            "effective-settings audit observed unexpected resolved changes: "
            f"{unexpected}"
        )
    return {
        "schema": str(schema),
        "status": "pass",
        "parser_factory": (
            "pipelines.static_adapt.cli_config._build_adapt_arg_parser"
        ),
        "parser_adapt_gradient_parity_rtol": 1e-8,
        "all_defaults_resolved": True,
        "unknown_cli_tokens": [],
        "unresolved_allowed_flags": [],
        "launch_cwd": str(Path(launch_cwd).expanduser().resolve()),
        "allowed_flags": sorted(allowed_flags),
        "allowed_destinations": sorted(allowed_destinations),
        "changed_destinations": changed_destinations,
        "unexpected_changed_destinations": [],
        "changed_fields": changed_fields,
        "source_settings": source_settings,
        "source_settings_sha256": payload_sha256(source_settings),
        "effective_settings": effective_settings,
        "effective_settings_sha256": payload_sha256(effective_settings),
    }


def _remove_flags(argv: Sequence[str], flags: set[str] | frozenset[str]) -> list[str]:
    tokens = [str(token) for token in argv]
    _parse_argv(tokens)
    output: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in flags:
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                index += 2
            else:
                index += 1
            continue
        output.append(token)
        index += 1
    return output


def _set_value_flag(argv: Sequence[str], flag: str, value: str) -> list[str]:
    tokens = [str(token) for token in argv]
    _, options = _parse_argv(tokens)
    if flag not in options:
        return [*tokens, flag, str(value)]
    index = tokens.index(flag)
    if options[flag] is None:
        raise ValueError(f"cannot replace valueless flag with a value: {flag}")
    output = list(tokens)
    output[index + 1] = str(value)
    return output


def _set_bool_flag(argv: Sequence[str], flag: str) -> list[str]:
    tokens = [str(token) for token in argv]
    _, options = _parse_argv(tokens)
    if flag in options:
        if options[flag] is not None:
            raise ValueError(f"expected a valueless boolean flag: {flag}")
        return tokens
    return [*tokens, flag]


def _option_diff(
    left_argv: Sequence[str], right_argv: Sequence[str]
) -> tuple[bool, list[dict[str, Any]]]:
    left_prefix, left_options = _parse_argv(left_argv)
    right_prefix, right_options = _parse_argv(right_argv)
    rows: list[dict[str, Any]] = []
    for flag in sorted(set(left_options) | set(right_options)):
        left_present = flag in left_options
        right_present = flag in right_options
        left_value = left_options.get(flag)
        right_value = right_options.get(flag)
        if left_present != right_present or left_value != right_value:
            rows.append(
                {
                    "flag": flag,
                    "source_present": left_present,
                    "effective_present": right_present,
                    "source_value": left_value,
                    "effective_value": right_value,
                }
            )
    return left_prefix == right_prefix, rows


def audit_argv_diff(
    source_argv: Sequence[str],
    effective_argv: Sequence[str],
    *,
    allowed_flags: set[str] | frozenset[str] = SOURCE_TO_PROFILE_ALLOWED_FLAGS,
    schema: str = ARGV_AUDIT_SCHEMA,
) -> dict[str, Any]:
    """Return an exact semantic argv audit and fail closed on any other diff."""

    prefix_equal, changes = _option_diff(source_argv, effective_argv)
    changed_flags = [str(row["flag"]) for row in changes]
    changed_set = set(changed_flags)
    unexpected = sorted(changed_set - set(allowed_flags))
    profile_diffs = sorted(changed_set & set(PROFILE_FLAGS))
    nonprofile = sorted(changed_set - set(PROFILE_FLAGS))
    allowed_execution_diffs = sorted(set(nonprofile) & set(EXECUTION_FLAGS))
    unexpected_nonprofile = sorted(set(nonprofile) - set(EXECUTION_FLAGS))
    status = (
        "pass"
        if prefix_equal and not unexpected and not unexpected_nonprofile
        else "blocked"
    )
    audit = {
        "schema": str(schema),
        "status": status,
        "source_argv_sha256": payload_sha256(list(source_argv)),
        "effective_argv_sha256": payload_sha256(list(effective_argv)),
        "executable_prefix_equal": bool(prefix_equal),
        "allowed_diff_flags": sorted(allowed_flags),
        "changed_flags": changed_flags,
        "changed_fields": changes,
        "profile_diff_flags": profile_diffs,
        "allowed_execution_diff_flags": allowed_execution_diffs,
        "unexpected_diff_flags": unexpected,
        "nonprofile_diff_flags": nonprofile,
        "unexpected_nonprofile_diff_flags": unexpected_nonprofile,
        "unexpected_nonprofile_diff_empty": not unexpected_nonprofile,
    }
    if status != "pass":
        raise ValueError(
            "source-lock argv audit failed: "
            f"prefix_equal={prefix_equal}, unexpected={unexpected}, "
            f"unexpected_nonprofile={unexpected_nonprofile}"
        )
    return audit


def _require_flag(
    options: Mapping[str, str | None], flag: str, expected: str | None
) -> None:
    if flag not in options:
        raise ValueError(f"source command is missing required flag {flag}")
    if options[flag] != expected:
        raise ValueError(
            f"source command {flag} mismatch: expected {expected!r}, "
            f"found {options[flag]!r}"
        )


def _validate_source_profile(argv: Sequence[str]) -> None:
    """Verify that a row is the intended July-8 singleton source family."""

    _prefix, options = _parse_argv(argv)
    for flag, expected in (
        ("--problem", "hh"),
        ("--adapt-continuation-mode", "phase3_v1"),
        ("--static-route-id", "route_a"),
        ("--static-lane-route", "physical_operator_type"),
        ("--phase2-no-batching", None),
        ("--phase3-no-batching", None),
        ("--phase3-runtime-split-mode", "shortlist_pauli_children_v1"),
        (
            "--phase3-runtime-split-selection-mode",
            "archival_child_set_forward_v1",
        ),
        ("--phase3-runtime-split-max-subset-size", "1"),
    ):
        _require_flag(options, flag, expected)
    for forbidden in (
        "--phase2-enable-batching",
        "--phase3-enable-batching",
    ):
        if forbidden in options:
            raise ValueError(f"source command enables forbidden batching flag {forbidden}")


def _resolve_source_result_path(
    raw_path: str, *, source_commands_json: Path, repo_root: Path
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    repo_candidate = (repo_root / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    local_candidate = (source_commands_json.parent / path).resolve()
    if local_candidate.exists():
        return local_candidate
    return repo_candidate


def _normalize_regime(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def _load_source_rows(
    source_commands_json: Path, *, default_single_regime: str | None = None
) -> list[dict[str, Any]]:
    payload = json.loads(source_commands_json.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        if not default_single_regime:
            raise ValueError(
                "a single source command object requires exactly one requested regime"
            )
        payload_rows: list[Any] = [payload]
    elif isinstance(payload, list) and payload:
        payload_rows = payload
    else:
        raise ValueError(
            "source commands JSON must be a nonempty array or one command object"
        )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in payload_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("source commands JSON contains a non-object row")
        row = dict(raw)
        raw_regime = row.get("regime")
        regime = _normalize_regime(
            str(
                default_single_regime
                if raw_regime is None and len(payload_rows) == 1
                else raw_regime or ""
            )
        )
        if not regime or regime in seen:
            raise ValueError("source command regimes must be nonempty and unique")
        if (
            raw_regime is not None
            and default_single_regime is not None
            and len(payload_rows) == 1
            and regime != _normalize_regime(default_single_regime)
        ):
            raise ValueError(
                "single source command regime disagrees with requested regime"
            )
        seen.add(regime)
        argv = row.get("argv")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise ValueError(f"source row {regime} has invalid argv")
        _validate_source_profile(argv)
        _prefix, options = _parse_argv(argv)
        source_output = options.get("--output-json")
        if not source_output:
            raise ValueError(f"source row {regime} is missing --output-json")
        if row.get("output_json") is None:
            row["output_json"] = source_output
        elif row.get("output_json") != source_output:
            raise ValueError(
                f"source row {regime} output_json does not match argv --output-json"
            )
        if row.get("shell") is not None and shlex.split(str(row["shell"])) != argv:
            raise ValueError(f"source row {regime} shell does not reproduce argv")
        row["_source_command_row_sha256"] = payload_sha256(dict(raw))
        row["regime"] = regime
        rows.append(row)
    return rows


def _profile_argv(
    source_argv: Sequence[str],
    *,
    coordinate_policy: str,
    escape_mode: str,
    output_json: Path,
    current_json: Path,
    estimator_call_ledger_json: Path,
) -> list[str]:
    argv = _remove_flags(
        source_argv,
        frozenset(
            {
                "--phase0-pilot-enabled",
                "--phase2-enable-batching",
                "--phase3-enable-batching",
            }
        ),
    )
    argv = _set_bool_flag(argv, "--phase0-no-pilot")
    argv = _set_bool_flag(argv, "--phase2-no-batching")
    argv = _set_bool_flag(argv, "--phase3-no-batching")
    argv = _set_value_flag(
        argv, "--phase3-runtime-split-max-subset-size", "1"
    )
    argv = _set_value_flag(argv, "--phase3-runtime-split-subset-sizes", "1")
    argv = _set_value_flag(
        argv,
        "--phase3-runtime-split-child-padding-policy",
        EXACT_PROJECTED_GROUPED_PADDING,
    )
    argv = _set_value_flag(
        argv, "--historical-singleton-coordinate-solve-policy", coordinate_policy
    )
    argv = _set_value_flag(
        argv,
        "--historical-singleton-trust-region-update-policy",
        ADAPTIVE_TRUST_POLICY,
    )
    argv = _set_value_flag(
        argv,
        "--sr-powell-coordinate-chart-policy",
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
    )
    argv = _set_value_flag(argv, "--sr-escape-mode", escape_mode)
    argv = _set_value_flag(argv, "--output-json", str(output_json))
    argv = _set_value_flag(argv, "--adapt-current-json", str(current_json))
    argv = _set_value_flag(
        argv,
        "--adapt-estimator-call-ledger-json",
        str(estimator_call_ledger_json),
    )
    argv = _set_bool_flag(argv, "--skip-trajectory")
    return argv


def _validate_effective_profile(
    argv: Sequence[str], *, coordinate_policy: str, escape_mode: str
) -> None:
    _prefix, options = _parse_argv(argv)
    required = (
        ("--phase0-no-pilot", None),
        ("--phase2-no-batching", None),
        ("--phase3-no-batching", None),
        ("--phase3-runtime-split-max-subset-size", "1"),
        ("--phase3-runtime-split-subset-sizes", "1"),
        (
            "--phase3-runtime-split-child-padding-policy",
            EXACT_PROJECTED_GROUPED_PADDING,
        ),
        ("--historical-singleton-coordinate-solve-policy", coordinate_policy),
        (
            "--historical-singleton-trust-region-update-policy",
            ADAPTIVE_TRUST_POLICY,
        ),
        (
            "--sr-powell-coordinate-chart-policy",
            SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
        ),
        ("--sr-escape-mode", escape_mode),
        ("--skip-trajectory", None),
    )
    for flag, expected in required:
        _require_flag(options, flag, expected)
    for forbidden in (
        "--phase0-pilot-enabled",
        "--phase2-enable-batching",
        "--phase3-enable-batching",
    ):
        if forbidden in options:
            raise ValueError(f"effective command retains conflicting flag {forbidden}")


def _record_target_paths(
    record: Mapping[str, Any], *, campaign_root: Path
) -> dict[str, Path]:
    root = Path(campaign_root).expanduser().resolve()
    targets: dict[str, Path] = {}
    for field in RECORD_TARGET_PATH_FIELDS:
        raw = record.get(field)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"campaign record is missing target path {field}")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ValueError(f"campaign target path must be absolute: {field}={raw!r}")
        path = path.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"campaign target path escapes campaign root: {field}={path}"
            ) from exc
        targets[field] = path
    cache_directories = record.get("launch_environment_overrides")
    if not isinstance(cache_directories, Mapping):
        raise ValueError("campaign record is missing launch cache overrides")
    if set(cache_directories) != set(RECORD_CACHE_ENVIRONMENT_VARIABLES):
        raise ValueError("campaign record launch cache override surface changed")
    for variable in RECORD_CACHE_ENVIRONMENT_VARIABLES:
        raw = cache_directories.get(variable)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"campaign cache override is invalid: {variable}")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ValueError(
                f"campaign cache override path must be absolute: {variable}"
            )
        path = path.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"campaign cache path escapes campaign root: {variable}={path}"
            ) from exc
        targets[f"cache:{variable}"] = path
    if len(set(targets.values())) != len(targets):
        raise ValueError("campaign record target paths must be distinct")
    return targets


def prepare_record_parent_directories(
    record: Mapping[str, Any], *, campaign_root: Path
) -> list[str]:
    """Create only the isolated parents required by one generated record.

    The helper is intentionally separate from generation and first enforces the
    same no-clobber target check as prelaunch verification.
    """

    targets = _record_target_paths(record, campaign_root=campaign_root)
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise ValueError(f"campaign no-clobber check failed: existing={existing}")
    parent_paths = sorted({path.parent for path in targets.values()}, key=str)
    for parent in parent_paths:
        parent.mkdir(parents=True, exist_ok=True)
    return [str(path) for path in parent_paths]


def verify_campaign_manifest_prelaunch(
    manifest: Mapping[str, Any],
    *,
    record_id: str,
    current_cwd: Path | None = None,
) -> dict[str, Any]:
    """Verify source lock, runtime contract, and no-clobber immediately prelaunch.

    Passing this check is a preflight receipt only.  It deliberately does not
    change the manifest's ``execution_authorized=false`` state and does not
    release the saddle dependency gate.
    """

    if str(manifest.get("schema", "")) != CAMPAIGN_SCHEMA:
        raise ValueError("campaign manifest has an unsupported schema")
    if manifest.get("generator_only") is not True:
        raise ValueError("campaign manifest is not generator-only")
    if manifest.get("execution_authorized") is not False:
        raise ValueError("campaign manifest execution_authorized must remain false")
    execution_contract = manifest.get("execution_contract")
    if not isinstance(execution_contract, Mapping):
        raise ValueError("campaign manifest is missing execution_contract")
    if str(execution_contract.get("schema", "")) != EXECUTION_CONTRACT_SCHEMA:
        raise ValueError("campaign execution contract has an unsupported schema")
    launch_cwd = Path(str(execution_contract.get("launch_cwd", ""))).resolve()
    observed_cwd = (
        Path.cwd().resolve()
        if current_cwd is None
        else Path(current_cwd).expanduser().resolve()
    )
    if observed_cwd != launch_cwd:
        raise ValueError(
            f"prelaunch cwd mismatch: expected {launch_cwd}, found {observed_cwd}"
        )
    expected_python = str(execution_contract.get("runtime_python_executable", ""))
    observed_python = str(Path(sys.executable).expanduser().resolve())
    if expected_python != observed_python:
        raise ValueError(
            "prelaunch Python executable mismatch: "
            f"expected {expected_python}, found {observed_python}"
        )
    environment = execution_contract.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("campaign execution contract is missing environment")
    environment_receipt = _verify_environment_contract(environment)

    expected_lock = manifest.get("runtime_source_lock")
    if not isinstance(expected_lock, Mapping):
        raise ValueError("campaign manifest is missing runtime_source_lock")
    source_lock_receipt = verify_runtime_source_lock(
        expected_lock, runtime_repo_root=launch_cwd
    )
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("campaign manifest records must be an array")
    matches = [
        raw
        for raw in records
        if isinstance(raw, Mapping) and str(raw.get("record_id", "")) == record_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"campaign record_id must resolve exactly once: {record_id!r}"
        )
    record = matches[0]
    if record.get("execution_authorized") is not False:
        raise ValueError("campaign record execution_authorized must remain false")
    if record.get("runtime_source_lock_sha256") != expected_lock.get(
        "aggregate_sha256"
    ):
        raise ValueError("campaign record runtime source-lock hash mismatch")
    launch_argv = record.get("launch_argv")
    if (
        not isinstance(launch_argv, list)
        or not launch_argv
        or not all(isinstance(item, str) for item in launch_argv)
    ):
        raise ValueError("campaign record launch_argv is invalid")
    if launch_argv[0] != expected_python:
        raise ValueError("campaign record launch_argv Python executable mismatch")
    if payload_sha256(launch_argv) != record.get("launch_argv_sha256"):
        raise ValueError("campaign record launch_argv hash mismatch")
    audit = record.get("effective_settings_audit")
    if not isinstance(audit, Mapping) or audit.get("status") != "pass":
        raise ValueError("campaign record effective-settings audit is not pass")

    campaign_root = Path(str(manifest.get("campaign_root", ""))).resolve()
    targets = _record_target_paths(record, campaign_root=campaign_root)
    required_parents = sorted({str(path.parent) for path in targets.values()})
    if record.get("required_parent_directories") != required_parents:
        raise ValueError("campaign record required parent-directory contract mismatch")
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise ValueError(f"campaign no-clobber check failed: existing={existing}")
    return {
        "schema": PRELAUNCH_VERIFICATION_SCHEMA,
        "status": "pass",
        "record_id": str(record_id),
        "preflight_only": True,
        "execution_authorized": False,
        "launch_cwd": str(launch_cwd),
        "runtime_python_executable": observed_python,
        "runtime_source_lock_verification": source_lock_receipt,
        "environment_verification": environment_receipt,
        "target_paths_absent": True,
        "verified_target_paths": {
            field: str(path) for field, path in targets.items()
        },
        "required_parent_directories": required_parents,
        "dependency_gate_status": str(
            record.get("dependency_gate_status", "unresolved")
        ),
    }


def validate_disabled_anchor_result(
    manifest: Mapping[str, Any], *, record_id: str
) -> dict[str, Any]:
    """Validate a completed disabled record before any saddle launch.

    This is a strict regression gate against the exact source result.  It does
    not mutate the campaign or authorize the dependent saddle record.
    """

    if str(manifest.get("schema", "")) != CAMPAIGN_SCHEMA:
        raise ValueError("campaign manifest has an unsupported schema")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("campaign manifest records must be an array")
    matches = [
        raw
        for raw in records
        if isinstance(raw, Mapping) and str(raw.get("record_id", "")) == record_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"campaign record_id must resolve exactly once: {record_id!r}"
        )
    record = matches[0]
    if str(record.get("profile_key", "")) != SR_ESCAPE_DISABLED:
        raise ValueError("anchor validation requires a disabled profile record")
    if str(record.get("dependency_gate_role", "")) != "disabled_anchor":
        raise ValueError("anchor validation requires the disabled anchor role")
    reference = record.get("source_anchor_reference")
    if not isinstance(reference, Mapping):
        raise ValueError("disabled record is missing source_anchor_reference")
    if str(reference.get("schema", "")) != SOURCE_ANCHOR_REFERENCE_SCHEMA:
        raise ValueError("source anchor reference has an unsupported schema")
    if reference.get("source_result_sha256") != record.get("source_result_sha256"):
        raise ValueError("source anchor reference hash disagrees with the record")

    expected_lock = manifest.get("runtime_source_lock")
    if not isinstance(expected_lock, Mapping):
        raise ValueError("campaign manifest is missing runtime_source_lock")
    execution_contract = manifest.get("execution_contract")
    if not isinstance(execution_contract, Mapping):
        raise ValueError("campaign manifest is missing execution_contract")
    launch_cwd = Path(str(execution_contract.get("launch_cwd", ""))).resolve()
    source_lock_receipt = verify_runtime_source_lock(
        expected_lock, runtime_repo_root=launch_cwd
    )

    result_raw = record.get("output_json")
    if not isinstance(result_raw, str) or not result_raw:
        raise ValueError("disabled record is missing output_json")
    result_path = Path(result_raw).expanduser().resolve()
    if not result_path.is_file():
        raise ValueError(f"disabled anchor result is missing: {result_path}")
    observed = build_source_anchor_reference(result_path)

    checks = {
        "success": bool(observed["success"] is True)
        and bool(reference.get("success") is True),
        "stop_reason": observed["stop_reason"] == reference.get("stop_reason"),
        "history_length": observed["history_length"]
        == reference.get("history_length"),
        "ansatz_depth": observed["ansatz_depth"] == reference.get("ansatz_depth"),
        "operator_count": observed["operator_count"]
        == reference.get("operator_count"),
        "operator_sequence": observed["operator_sequence_sha256"]
        == reference.get("operator_sequence_sha256"),
        "energy": abs(float(observed["energy"]) - float(reference["energy"]))
        <= float(reference["energy_abs_tolerance"]),
        "exact_gs_energy": abs(
            float(observed["exact_gs_energy"])
            - float(reference["exact_gs_energy"])
        )
        <= float(reference["exact_energy_abs_tolerance"]),
        "abs_delta_e": abs(
            float(observed["abs_delta_e"]) - float(reference["abs_delta_e"])
        )
        <= float(reference["energy_abs_tolerance"]),
        "observed_energy_consistency": abs(
            abs(
                float(observed["energy"])
                - float(observed["exact_gs_energy"])
            )
            - float(observed["abs_delta_e"])
        )
        <= float(reference["exact_energy_abs_tolerance"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "disabled anchor validation failed; saddle gate remains closed: "
            f"failed_checks={failed}"
        )
    return {
        "schema": ANCHOR_VALIDATION_SCHEMA,
        "status": "pass",
        "record_id": str(record_id),
        "saddle_gate_release_authority": "external_user_or_agent_after_receipt",
        "execution_authorized": False,
        "source_reference": dict(reference),
        "observed_result": observed,
        "checks": checks,
        "failed_checks": [],
        "runtime_source_lock_verification": source_lock_receipt,
    }


def validate_current_revision_disabled_result(
    manifest: Mapping[str, Any], *, record_id: str
) -> dict[str, Any]:
    """Bind one completed disabled control to its frozen current revision.

    Unlike :func:`validate_disabled_anchor_result`, this diagnostic gate makes
    no historical-equivalence comparison.  It verifies the immutable campaign
    pair, current source lock, disabled result integrity, and the still-unused
    paired saddle target.  The returned receipt does not mutate the manifest or
    authorize execution; an external user or agent must make the scoped launch
    decision after preserving the receipt.
    """

    if str(manifest.get("schema", "")) != CAMPAIGN_SCHEMA:
        raise ValueError("campaign manifest has an unsupported schema")
    if manifest.get("generator_only") is not True:
        raise ValueError("campaign manifest is not generator-only")
    if manifest.get("execution_authorized") is not False:
        raise ValueError(
            "campaign manifest execution_authorized must remain false"
        )
    expected_profile_order = [
        str(spec["profile_key"]) for spec in PROFILE_SPECS
    ]
    if manifest.get("profile_order") != expected_profile_order:
        raise ValueError(
            "current-revision validation only supports the disabled/saddle-only "
            "profile order"
        )
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("campaign manifest records must be an array")
    records_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("campaign manifest records must all be objects")
        raw_record_id = raw.get("record_id")
        if not isinstance(raw_record_id, str) or not raw_record_id:
            raise ValueError("campaign record_id values must be nonempty strings")
        if raw_record_id in records_by_id:
            raise ValueError(
                "campaign record_id values must be unique: "
                f"duplicate={raw_record_id!r}"
            )
        records_by_id[raw_record_id] = raw
    record = records_by_id.get(str(record_id))
    if record is None:
        raise ValueError(
            f"campaign record_id must resolve exactly once: {record_id!r}"
        )
    if str(record.get("schema", "")) != COMMAND_RECORD_SCHEMA:
        raise ValueError("disabled control record has an unsupported schema")
    if str(record.get("profile_key", "")) != SR_ESCAPE_DISABLED:
        raise ValueError(
            "current-revision control validation requires a disabled profile"
        )
    if str(record.get("sr_escape_mode", "")) != SR_ESCAPE_DISABLED:
        raise ValueError("disabled control record must use disabled escape mode")
    if str(record.get("dependency_gate_role", "")) != "disabled_anchor":
        raise ValueError(
            "current-revision control validation requires the disabled anchor role"
        )
    if record.get("generator_only") is not True:
        raise ValueError("disabled control record must remain generator-only")
    if record.get("execution_authorized") is not False:
        raise ValueError("campaign record execution_authorized must remain false")

    expected_lock = manifest.get("runtime_source_lock")
    if not isinstance(expected_lock, Mapping):
        raise ValueError("campaign manifest is missing runtime_source_lock")
    execution_contract = manifest.get("execution_contract")
    if not isinstance(execution_contract, Mapping):
        raise ValueError("campaign manifest is missing execution_contract")
    launch_cwd = Path(str(execution_contract.get("launch_cwd", ""))).resolve()
    runtime_python_executable = str(
        execution_contract.get("runtime_python_executable", "")
    )
    source_lock_receipt = verify_runtime_source_lock(
        expected_lock, runtime_repo_root=launch_cwd
    )
    aggregate_lock = str(expected_lock.get("aggregate_sha256", ""))
    if str(record.get("runtime_source_lock_sha256", "")) != aggregate_lock:
        raise ValueError("disabled control runtime source-lock hash mismatch")
    campaign_root = Path(str(manifest.get("campaign_root", ""))).resolve()

    def _validated_profile_record(
        raw_record: Mapping[str, Any],
        *,
        profile_key: str,
        escape_mode: str,
        coordinate_policy: str,
    ) -> dict[str, Any]:
        if str(raw_record.get("schema", "")) != COMMAND_RECORD_SCHEMA:
            raise ValueError(
                f"{profile_key} campaign record has an unsupported schema"
            )
        if str(raw_record.get("profile_key", "")) != profile_key:
            raise ValueError(f"paired record is not the {profile_key} profile")
        if str(raw_record.get("sr_escape_mode", "")) != escape_mode:
            raise ValueError(
                f"{profile_key} campaign record has the wrong escape mode"
            )
        if str(raw_record.get("route_family", "")) != SR_ROUTE_FAMILY:
            raise ValueError(f"{profile_key} campaign route family mismatch")
        if raw_record.get("route_profile") != _campaign_route_profile(escape_mode):
            raise ValueError(f"{profile_key} campaign route profile mismatch")
        if str(
            raw_record.get("sr_powell_coordinate_chart_policy", "")
        ) != SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1:
            raise ValueError(
                f"{profile_key} campaign Powell chart policy mismatch"
            )
        if str(
            raw_record.get("historical_singleton_coordinate_solve_policy", "")
        ) != coordinate_policy:
            raise ValueError(
                f"{profile_key} campaign coordinate-solve policy mismatch"
            )
        if str(
            raw_record.get(
                "historical_singleton_trust_region_update_policy", ""
            )
        ) != ADAPTIVE_TRUST_POLICY:
            raise ValueError(
                f"{profile_key} campaign trust-update policy mismatch"
            )
        if raw_record.get("generator_only") is not True:
            raise ValueError(f"{profile_key} campaign record is not generator-only")
        if raw_record.get("execution_authorized") is not False:
            raise ValueError(
                f"{profile_key} campaign record execution_authorized must remain false"
            )
        if str(raw_record.get("runtime_source_lock_sha256", "")) != aggregate_lock:
            raise ValueError(f"{profile_key} runtime source-lock hash mismatch")

        source_argv = raw_record.get("source_argv")
        if (
            not isinstance(source_argv, list)
            or not source_argv
            or not all(isinstance(item, str) for item in source_argv)
        ):
            raise ValueError(f"{profile_key} source_argv is invalid")
        if payload_sha256(source_argv) != raw_record.get(
            "source_command_argv_sha256"
        ):
            raise ValueError(f"{profile_key} source_argv hash mismatch")
        _validate_source_profile(source_argv)

        argv = raw_record.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) for item in argv)
        ):
            raise ValueError(f"{profile_key} argv is invalid")
        if payload_sha256(argv) != raw_record.get("argv_sha256"):
            raise ValueError(f"{profile_key} argv hash mismatch")
        _validate_effective_profile(
            argv,
            coordinate_policy=coordinate_policy,
            escape_mode=escape_mode,
        )

        launch_argv = raw_record.get("launch_argv")
        if (
            not isinstance(launch_argv, list)
            or not launch_argv
            or not all(isinstance(item, str) for item in launch_argv)
        ):
            raise ValueError(f"{profile_key} launch_argv is invalid")
        if payload_sha256(launch_argv) != raw_record.get("launch_argv_sha256"):
            raise ValueError(f"{profile_key} launch_argv hash mismatch")
        expected_launch_argv = [runtime_python_executable, *argv[1:]]
        if launch_argv != expected_launch_argv:
            raise ValueError(
                f"{profile_key} launch_argv does not reproduce the audited argv"
            )
        _adapt_cli_flag_tokens(launch_argv)

        targets = _record_target_paths(raw_record, campaign_root=campaign_root)
        _prefix, options = _parse_argv(argv)
        for flag, field in (
            ("--output-json", "output_json"),
            ("--adapt-current-json", "current_json"),
            (
                "--adapt-estimator-call-ledger-json",
                "estimator_call_ledger_json",
            ),
        ):
            if options.get(flag) != str(targets[field]):
                raise ValueError(
                    f"{profile_key} {flag} does not match its record target"
                )

        stored_argv_audit = raw_record.get("allowed_diff_audit")
        if not isinstance(stored_argv_audit, Mapping):
            raise ValueError(f"{profile_key} argv audit is missing")
        recomputed_argv_audit = audit_argv_diff(source_argv, argv)
        if payload_sha256(stored_argv_audit) != payload_sha256(
            recomputed_argv_audit
        ):
            raise ValueError(f"{profile_key} argv audit does not recompute")

        stored_effective_audit = raw_record.get("effective_settings_audit")
        if not isinstance(stored_effective_audit, Mapping):
            raise ValueError(f"{profile_key} effective-settings audit is missing")
        recomputed_effective_audit = audit_effective_settings_diff(
            source_argv,
            argv,
            launch_cwd=launch_cwd,
        )
        if payload_sha256(stored_effective_audit) != payload_sha256(
            recomputed_effective_audit
        ):
            raise ValueError(
                f"{profile_key} effective-settings audit does not recompute"
            )
        return {
            "argv": argv,
            "launch_argv": launch_argv,
            "targets": targets,
            "effective_settings_audit": recomputed_effective_audit,
        }

    disabled_contract = _validated_profile_record(
        record,
        profile_key=SR_ESCAPE_DISABLED,
        escape_mode=SR_ESCAPE_DISABLED,
        coordinate_policy=(
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
        ),
    )

    regime = str(record.get("regime", ""))
    pair_audits = manifest.get("pair_audits")
    if not isinstance(pair_audits, list):
        raise ValueError("campaign manifest pair_audits must be an array")
    if not all(isinstance(raw, Mapping) for raw in pair_audits):
        raise ValueError("campaign manifest pair_audits must all be objects")
    matching_pairs = [
        raw
        for raw in pair_audits
        if isinstance(raw, Mapping)
        and str(raw.get("regime", "")) == regime
        and str(raw.get("baseline_record_id", "")) == str(record_id)
    ]
    if len(matching_pairs) != 1:
        raise ValueError("disabled control does not resolve to one passing pair audit")
    pair_audit = matching_pairs[0]
    if (
        str(pair_audit.get("schema", "")) != PAIR_AUDIT_SCHEMA
        or pair_audit.get("status") != "pass"
    ):
        raise ValueError("disabled control pair audit is not a passing v1 audit")
    saddle_record_id = str(pair_audit.get("saddle_record_id", ""))
    saddle_record = records_by_id.get(saddle_record_id)
    if saddle_record is None:
        raise ValueError("disabled control pair is missing its saddle record")
    if (
        str(saddle_record.get("schema", "")) != COMMAND_RECORD_SCHEMA
        or str(saddle_record.get("regime", "")) != regime
        or str(saddle_record.get("profile_key", ""))
        != SR_ESCAPE_SADDLE_ONLY
        or str(saddle_record.get("sr_escape_mode", ""))
        != SR_ESCAPE_SADDLE_ONLY
        or str(saddle_record.get("dependency_gate_role", ""))
        != "saddle_after_disabled_anchor"
        or str(saddle_record.get("runtime_source_lock_sha256", ""))
        != aggregate_lock
        or saddle_record.get("execution_dependencies") != [str(record_id)]
        or saddle_record.get("generator_only") is not True
        or saddle_record.get("execution_authorized") is not False
    ):
        raise ValueError("paired saddle record contract is invalid")
    same_regime_records = [
        raw for raw in records if str(raw.get("regime", "")) == regime
    ]
    if len(same_regime_records) != 2 or {
        str(raw.get("profile_key", "")) for raw in same_regime_records
    } != {SR_ESCAPE_DISABLED, SR_ESCAPE_SADDLE_ONLY}:
        raise ValueError(
            "current-revision release requires exactly one disabled/saddle-only "
            "pair in the regime"
        )

    saddle_contract = _validated_profile_record(
        saddle_record,
        profile_key=SR_ESCAPE_SADDLE_ONLY,
        escape_mode=SR_ESCAPE_SADDLE_ONLY,
        coordinate_policy=(
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
        ),
    )
    for field in (
        "source_commands_json_sha256",
        "source_command_row_sha256",
        "source_command_argv_sha256",
        "source_result_sha256",
    ):
        if saddle_record.get(field) != record.get(field):
            raise ValueError(f"paired records disagree on source binding {field}")

    recomputed_pair_argv_audit = audit_argv_diff(
        disabled_contract["argv"],
        saddle_contract["argv"],
        allowed_flags=PAIR_ALLOWED_FLAGS,
        schema=PAIR_AUDIT_SCHEMA,
    )
    recomputed_pair_effective_audit = audit_effective_settings_diff(
        disabled_contract["argv"],
        saddle_contract["argv"],
        launch_cwd=launch_cwd,
        allowed_flags=PAIR_ALLOWED_FLAGS,
        schema=(
            "paper_i_hh_sr_snake_escape_pair_effective_settings_audit_v1"
        ),
    )
    expected_pair_audit = {
        "schema": PAIR_AUDIT_SCHEMA,
        "status": "pass",
        "regime": regime,
        "baseline_record_id": str(record_id),
        "saddle_record_id": saddle_record_id,
        "allowed_profile_pair_diff_flags": sorted(PAIR_ALLOWED_FLAGS),
        "unexpected_nonprofile_diff_flags": recomputed_pair_argv_audit[
            "unexpected_nonprofile_diff_flags"
        ],
        "nonprofile_diff_flags": recomputed_pair_argv_audit[
            "nonprofile_diff_flags"
        ],
        "unexpected_nonprofile_diff_empty": recomputed_pair_argv_audit[
            "unexpected_nonprofile_diff_empty"
        ],
        "argv_audit": recomputed_pair_argv_audit,
        "effective_settings_audit": recomputed_pair_effective_audit,
    }
    if payload_sha256(pair_audit) != payload_sha256(expected_pair_audit):
        raise ValueError("disabled/saddle-only pair audit does not recompute")

    saddle_prelaunch_receipt = verify_campaign_manifest_prelaunch(
        manifest,
        record_id=saddle_record_id,
    )

    result_raw = record.get("output_json")
    if not isinstance(result_raw, str) or not result_raw:
        raise ValueError("disabled control is missing output_json")
    result_path = Path(result_raw).expanduser().resolve()
    if not result_path.is_file():
        raise ValueError(f"disabled control result is missing: {result_path}")
    observed = build_source_anchor_reference(result_path)
    if observed.get("success") is not True:
        raise ValueError("disabled current-revision control did not succeed")
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    settings = (
        result_payload.get("settings", {})
        if isinstance(result_payload, Mapping)
        else {}
    )
    if not isinstance(settings, Mapping):
        raise ValueError("disabled control result settings must be an object")
    result_profile_checks = {
        "sr_escape_mode": str(settings.get("sr_escape_mode", ""))
        == SR_ESCAPE_DISABLED,
        "coordinate_solve_policy": str(
            settings.get("historical_singleton_coordinate_solve_policy", "")
        )
        == str(record.get("historical_singleton_coordinate_solve_policy", "")),
        "trust_update_policy": str(
            settings.get(
                "historical_singleton_trust_region_update_policy", ""
            )
        )
        == str(
            record.get("historical_singleton_trust_region_update_policy", "")
        ),
        "powell_coordinate_chart_policy": str(
            settings.get("sr_powell_coordinate_chart_policy", "")
        )
        == str(record.get("sr_powell_coordinate_chart_policy", "")),
    }
    failed_profile_checks = [
        name for name, passed in result_profile_checks.items() if not passed
    ]
    if failed_profile_checks:
        raise ValueError(
            "disabled current-revision control profile mismatch: "
            f"failed_checks={failed_profile_checks}"
        )

    return {
        "schema": CURRENT_REVISION_CONTROL_VALIDATION_SCHEMA,
        "status": "pass",
        "record_id": str(record_id),
        "regime": regime,
        "validation_mode": "current_revision_diagnostic_control_v1",
        "historical_equivalence_claimed": False,
        "historical_anchor_status": "not_used_for_this_receipt",
        "saddle_record_id": saddle_record_id,
        "saddle_gate_release_scope": (
            f"{regime}_current_revision_disabled_to_saddle_only_pair"
        ),
        "saddle_gate_release_authority": (
            "external_user_or_agent_after_receipt"
        ),
        "execution_authorized": False,
        "combined_mode_authorized": False,
        "runtime_source_lock_verification": source_lock_receipt,
        "runtime_source_lock_sha256": aggregate_lock,
        "disabled_launch_argv_sha256": str(
            record.get("launch_argv_sha256", "")
        ),
        "paired_saddle_launch_argv_sha256": str(
            saddle_record.get("launch_argv_sha256", "")
        ),
        "pair_audit_sha256": payload_sha256(pair_audit),
        "paired_saddle_prelaunch_verification": saddle_prelaunch_receipt,
        "observed_disabled_result": observed,
        "observed_disabled_result_sha256": file_sha256(result_path),
        "result_profile_checks": result_profile_checks,
        "paired_saddle_output_absent": True,
        "paired_saddle_targets_absent": True,
    }


def build_campaign(
    *,
    source_commands_json: Path = DEFAULT_SOURCE_COMMANDS_JSON,
    regimes: Sequence[str],
    campaign_root: Path,
    repo_root: Path = REPO_ROOT,
    runtime_repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build a validated two-profile command manifest without launching runs."""

    source_commands_json = Path(source_commands_json).expanduser().resolve()
    campaign_root = Path(campaign_root).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    runtime_repo_root = Path(runtime_repo_root).expanduser().resolve()
    runtime_source_lock = build_runtime_source_lock(runtime_repo_root)
    execution_contract = _build_execution_contract(runtime_repo_root)
    launch_cwd = Path(str(execution_contract["launch_cwd"]))
    runtime_python_executable = str(
        execution_contract["runtime_python_executable"]
    )
    if not source_commands_json.is_file():
        raise ValueError(f"source commands JSON is missing: {source_commands_json}")
    requested = [_normalize_regime(value) for value in regimes]
    if not requested:
        raise ValueError("at least one regime must be requested")
    if any(not value for value in requested) or len(requested) != len(set(requested)):
        raise ValueError("requested regimes must be nonempty and unique")

    source_rows = _load_source_rows(
        source_commands_json,
        default_single_regime=(requested[0] if len(requested) == 1 else None),
    )
    source_by_regime = {str(row["regime"]): row for row in source_rows}
    missing = [value for value in requested if value not in source_by_regime]
    if missing:
        raise ValueError(f"requested regimes are absent from source commands: {missing}")

    commands_sha256 = file_sha256(source_commands_json)
    command_records: list[dict[str, Any]] = []
    pair_audits: list[dict[str, Any]] = []
    for regime in requested:
        source_row = source_by_regime[regime]
        source_argv = [str(token) for token in source_row["argv"]]
        source_output = str(source_row["output_json"])
        source_result_path = _resolve_source_result_path(
            source_output,
            source_commands_json=source_commands_json,
            repo_root=repo_root,
        )
        if not source_result_path.is_file():
            raise ValueError(
                f"source result JSON for {regime} is missing: {source_result_path}"
            )
        source_result_hash = file_sha256(source_result_path)
        source_anchor_reference = build_source_anchor_reference(
            source_result_path
        )
        if source_anchor_reference["source_result_sha256"] != source_result_hash:
            raise ValueError("source anchor changed while the campaign was built")
        source_row_hash = str(source_row["_source_command_row_sha256"])
        source_argv_hash = payload_sha256(source_argv)
        raw_source_environment = source_row.get("environment", {})
        if not isinstance(raw_source_environment, Mapping):
            raise ValueError(f"source row {regime} environment must be an object")
        source_environment = {
            str(key): str(value) for key, value in raw_source_environment.items()
        }
        unexpected_source_environment = sorted(
            set(source_environment) - set(RECORD_CACHE_ENVIRONMENT_VARIABLES)
        )
        if unexpected_source_environment:
            raise ValueError(
                "source command contains unhandled execution environment keys: "
                f"{unexpected_source_environment}"
            )
        regime_slug = regime.replace("-", "_")
        paired: dict[str, dict[str, Any]] = {}
        for spec in PROFILE_SPECS:
            profile_key = str(spec["profile_key"])
            run_dir = campaign_root / regime_slug / profile_key
            output_json = run_dir / "json" / "result.json"
            current_json = run_dir / "current.json"
            estimator_call_ledger_json = (
                run_dir / "json" / "estimator_call_ledger.json"
            )
            log_path = run_dir / "run.log"
            cache_root = run_dir / "cache"
            launch_environment_overrides = {
                "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR": str(
                    cache_root / "candidate_records"
                ),
                "STATIC_ADAPT_HH_POOL_CACHE_DIR": str(cache_root / "hh_pool"),
                "STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR": str(
                    cache_root / "hh_generators"
                ),
            }
            effective_argv = _profile_argv(
                source_argv,
                coordinate_policy=str(spec["coordinate_solve_policy"]),
                escape_mode=str(spec["sr_escape_mode"]),
                output_json=output_json,
                current_json=current_json,
                estimator_call_ledger_json=estimator_call_ledger_json,
            )
            _validate_effective_profile(
                effective_argv,
                coordinate_policy=str(spec["coordinate_solve_policy"]),
                escape_mode=str(spec["sr_escape_mode"]),
            )
            audit = audit_argv_diff(source_argv, effective_argv)
            effective_settings_audit = audit_effective_settings_diff(
                source_argv,
                effective_argv,
                launch_cwd=launch_cwd,
            )
            launch_argv = [runtime_python_executable, *effective_argv[1:]]
            target_paths = {
                "output_json": output_json,
                "current_json": current_json,
                "estimator_call_ledger_json": estimator_call_ledger_json,
                "log_path": log_path,
            }
            cache_paths = {
                key: Path(value)
                for key, value in launch_environment_overrides.items()
            }
            required_parent_directories = sorted(
                {
                    *(str(path.parent) for path in target_paths.values()),
                    *(str(path.parent) for path in cache_paths.values()),
                }
            )
            no_clobber_checks = " && ".join(
                f"test ! -e {shlex.quote(str(path))}"
                for path in [*target_paths.values(), *cache_paths.values()]
            )
            parent_preparation = "mkdir -p " + " ".join(
                shlex.quote(path) for path in required_parent_directories
            )
            launch_shell = shlex.join(
                [
                    "env",
                    *(
                        f"{key}={value}"
                        for key, value in launch_environment_overrides.items()
                    ),
                    *launch_argv,
                ]
            )
            shell_with_log_redirect = (
                f"{no_clobber_checks} && {parent_preparation} && "
                f"(set -C; exec {launch_shell} > "
                f"{shlex.quote(str(log_path))} 2>&1)"
            )
            baseline_record_id = f"{regime_slug}__sr_{SR_ESCAPE_DISABLED}"
            if profile_key == SR_ESCAPE_DISABLED:
                dependencies: list[str] = []
                dependency_gate_role = "disabled_anchor"
                dependency_gate_status = "pending_external_launch_decision"
            else:
                dependencies = [baseline_record_id]
                dependency_gate_role = "saddle_after_disabled_anchor"
                dependency_gate_status = (
                    "blocked_pending_disabled_anchor_completion_and_validation"
                )
            record = {
                "schema": COMMAND_RECORD_SCHEMA,
                "record_id": f"{regime_slug}__sr_{profile_key}",
                "regime": regime,
                "route_family": SR_ROUTE_FAMILY,
                "route_profile": _campaign_route_profile(
                    str(spec["sr_escape_mode"])
                ),
                "profile_key": profile_key,
                "sr_escape_mode": str(spec["sr_escape_mode"]),
                "historical_singleton_coordinate_solve_policy": str(
                    spec["coordinate_solve_policy"]
                ),
                "historical_singleton_trust_region_update_policy": str(
                    spec["trust_region_update_policy"]
                ),
                "sr_powell_coordinate_chart_policy": (
                    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
                ),
                "source_commands_json": str(source_commands_json),
                "source_commands_json_sha256": commands_sha256,
                "source_command_row_sha256": source_row_hash,
                "source_command_argv_sha256": source_argv_hash,
                "source_result_json": str(source_result_path),
                "source_result_sha256": source_result_hash,
                "source_anchor_reference": source_anchor_reference,
                "source_environment": source_environment,
                "runtime_source_lock_sha256": str(
                    runtime_source_lock["aggregate_sha256"]
                ),
                "source_argv": source_argv,
                "argv": effective_argv,
                "argv_sha256": payload_sha256(effective_argv),
                "launch_argv": launch_argv,
                "launch_argv_sha256": payload_sha256(launch_argv),
                "runtime_python_executable": runtime_python_executable,
                "launch_cwd": str(launch_cwd),
                "launch_environment_overrides": launch_environment_overrides,
                "launch_environment_audit": {
                    "schema": (
                        "paper_i_hh_sr_snake_escape_launch_environment_audit_v1"
                    ),
                    "status": "pass",
                    "source_environment": source_environment,
                    "allowed_override_variables": list(
                        RECORD_CACHE_ENVIRONMENT_VARIABLES
                    ),
                    "effective_environment_overrides": (
                        launch_environment_overrides
                    ),
                    "unexpected_source_environment_variables": [],
                    "secret_values_recorded": False,
                },
                "shell": launch_shell,
                "shell_with_log_redirect": shell_with_log_redirect,
                "output_json": str(output_json),
                "current_json": str(current_json),
                "estimator_call_ledger_json": str(
                    estimator_call_ledger_json
                ),
                "log_path": str(log_path),
                "required_parent_directories": required_parent_directories,
                "parent_directory_preparation_helper": (
                    "pipelines.static_adapt.sr_snake_escape_campaign."
                    "prepare_record_parent_directories"
                ),
                "no_clobber_target_fields": list(RECORD_TARGET_PATH_FIELDS),
                "no_clobber_cache_directories": launch_environment_overrides,
                "execution_path_audit": {
                    "schema": "paper_i_hh_sr_snake_escape_execution_path_audit_v1",
                    "status": "pass",
                    "allowed_path_fields": [
                        "output_json",
                        "current_json",
                        "estimator_call_ledger_json",
                        "log_path",
                    ],
                    "source_output_json": source_output,
                    "source_current_json": _parse_argv(source_argv)[1].get(
                        "--adapt-current-json"
                    ),
                    "source_estimator_call_ledger_json": _parse_argv(
                        source_argv
                    )[1].get("--adapt-estimator-call-ledger-json"),
                    "source_log_path": source_row.get("log_path"),
                    "effective_output_json": str(output_json),
                    "effective_current_json": str(current_json),
                    "effective_estimator_call_ledger_json": str(
                        estimator_call_ledger_json
                    ),
                    "effective_log_path": str(log_path),
                    "unexpected_path_fields": [],
                },
                "allowed_diff_audit": audit,
                "effective_settings_audit": effective_settings_audit,
                "execution_dependencies": dependencies,
                "dependency_gate_role": dependency_gate_role,
                "dependency_gate_status": dependency_gate_status,
                "generator_only": True,
                "execution_authorized": False,
            }
            command_records.append(record)
            paired[profile_key] = record

        baseline = paired[SR_ESCAPE_DISABLED]
        saddle = paired[SR_ESCAPE_SADDLE_ONLY]
        pair_audit = audit_argv_diff(
            baseline["argv"],
            saddle["argv"],
            allowed_flags=PAIR_ALLOWED_FLAGS,
            schema=PAIR_AUDIT_SCHEMA,
        )
        pair_effective_settings_audit = audit_effective_settings_diff(
            baseline["argv"],
            saddle["argv"],
            launch_cwd=launch_cwd,
            allowed_flags=PAIR_ALLOWED_FLAGS,
            schema=(
                "paper_i_hh_sr_snake_escape_pair_effective_settings_audit_v1"
            ),
        )
        pair_audits.append(
            {
                "schema": PAIR_AUDIT_SCHEMA,
                "status": "pass",
                "regime": regime,
                "baseline_record_id": baseline["record_id"],
                "saddle_record_id": saddle["record_id"],
                "allowed_profile_pair_diff_flags": sorted(PAIR_ALLOWED_FLAGS),
                "unexpected_nonprofile_diff_flags": pair_audit[
                    "unexpected_nonprofile_diff_flags"
                ],
                "nonprofile_diff_flags": pair_audit["nonprofile_diff_flags"],
                "unexpected_nonprofile_diff_empty": pair_audit[
                    "unexpected_nonprofile_diff_empty"
                ],
                "argv_audit": pair_audit,
                "effective_settings_audit": pair_effective_settings_audit,
            }
        )

    dependency_gates = []
    records_by_id = {
        str(record["record_id"]): record for record in command_records
    }
    for regime in requested:
        regime_slug = regime.replace("-", "_")
        baseline_id = f"{regime_slug}__sr_{SR_ESCAPE_DISABLED}"
        saddle_id = f"{regime_slug}__sr_{SR_ESCAPE_SADDLE_ONLY}"
        if baseline_id not in records_by_id or saddle_id not in records_by_id:
            raise ValueError(f"campaign dependency records are incomplete for {regime}")
        dependency_gates.append(
            {
                "schema": EXECUTION_DEPENDENCY_SCHEMA,
                "regime": regime,
                "ordering_policy": "disabled_anchor_before_saddle_v1",
                "disabled_anchor_record_id": baseline_id,
                "saddle_record_id": saddle_id,
                "disabled_anchor_must_complete_first": True,
                "disabled_anchor_requires_external_validation": True,
                "disabled_anchor_validation_schema": ANCHOR_VALIDATION_SCHEMA,
                "disabled_anchor_validation_helper": (
                    "pipelines.static_adapt.sr_snake_escape_campaign."
                    "validate_disabled_anchor_result"
                ),
                "current_revision_control_validation_schema": (
                    CURRENT_REVISION_CONTROL_VALIDATION_SCHEMA
                ),
                "current_revision_control_validation_helper": (
                    "pipelines.static_adapt.sr_snake_escape_campaign."
                    "validate_current_revision_disabled_result"
                ),
                "source_anchor_reference": records_by_id[baseline_id][
                    "source_anchor_reference"
                ],
                "saddle_initial_gate_status": (
                    "blocked_pending_disabled_anchor_completion_and_validation"
                ),
                "generator_releases_gate": False,
                "execution_authorized": False,
            }
        )

    return {
        "schema": CAMPAIGN_SCHEMA,
        "status": "pass",
        "run_class": "diagnostic",
        "runner_mode": "direct_current_code_command",
        "wrapper_used": False,
        "generator_only": True,
        "execution_authorized": False,
        "source_commands_json": str(source_commands_json),
        "source_commands_json_sha256": commands_sha256,
        "runtime_source_lock": runtime_source_lock,
        "execution_contract": execution_contract,
        "campaign_root": str(campaign_root),
        "requested_regimes": requested,
        "profile_order": [str(spec["profile_key"]) for spec in PROFILE_SPECS],
        "allowed_source_diff_flags": sorted(SOURCE_TO_PROFILE_ALLOWED_FLAGS),
        "records": command_records,
        "pair_audits": pair_audits,
        "execution_dependency_gates": dependency_gates,
        "launch_count": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate source-locked current-code SR-SNAKE disabled/saddle command "
            "pairs. This command never launches the generated argv vectors."
        )
    )
    parser.add_argument(
        "--source-commands-json",
        type=Path,
        default=DEFAULT_SOURCE_COMMANDS_JSON,
    )
    parser.add_argument(
        "--regime",
        action="append",
        help="Source regime; repeat for multiple command-list rows.",
    )
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument(
        "--verify-manifest",
        type=Path,
        help=(
            "Recompute the runtime lock and no-clobber checks for one record; "
            "does not launch or authorize it."
        ),
    )
    parser.add_argument(
        "--validate-anchor-manifest",
        type=Path,
        help=(
            "Validate one completed disabled result against its exact source "
            "anchor; does not launch or authorize the saddle record."
        ),
    )
    parser.add_argument(
        "--validate-current-control-manifest",
        type=Path,
        help=(
            "Validate one completed disabled result as a source-locked "
            "current-revision diagnostic control; makes no historical "
            "equivalence claim and does not authorize the saddle record."
        ),
    )
    parser.add_argument(
        "--record-id",
        help="Record to check with --verify-manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        verification_modes = sum(
            value is not None
            for value in (
                args.verify_manifest,
                args.validate_anchor_manifest,
                args.validate_current_control_manifest,
            )
        )
        if verification_modes > 1:
            raise ValueError(
                "manifest verification modes are mutually exclusive"
            )
        if any(
            value is not None
            for value in (
                args.verify_manifest,
                args.validate_anchor_manifest,
                args.validate_current_control_manifest,
            )
        ):
            if not args.record_id:
                raise ValueError("manifest verification requires --record-id")
            if args.regime or args.campaign_root or args.output_manifest:
                raise ValueError(
                    "verification mode cannot be combined with generation options"
                )
            selected_manifest = next(
                value
                for value in (
                    args.verify_manifest,
                    args.validate_anchor_manifest,
                    args.validate_current_control_manifest,
                )
                if value is not None
            )
            manifest_path = Path(selected_manifest).expanduser().resolve()
            manifest_payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if not isinstance(manifest_payload, Mapping):
                raise ValueError("campaign manifest must be a JSON object")
            if args.verify_manifest is not None:
                receipt = verify_campaign_manifest_prelaunch(
                    manifest_payload,
                    record_id=str(args.record_id),
                )
            elif args.validate_anchor_manifest is not None:
                receipt = validate_disabled_anchor_result(
                    manifest_payload,
                    record_id=str(args.record_id),
                )
            else:
                receipt = validate_current_revision_disabled_result(
                    manifest_payload,
                    record_id=str(args.record_id),
                )
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 0
        if args.record_id:
            raise ValueError("--record-id requires a manifest verification mode")
        if not args.regime or args.campaign_root is None or args.output_manifest is None:
            raise ValueError(
                "generation requires --regime, --campaign-root, and --output-manifest"
            )
        campaign = build_campaign(
            source_commands_json=args.source_commands_json,
            regimes=args.regime,
            campaign_root=args.campaign_root,
        )
        _atomic_write_json(args.output_manifest, campaign)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(campaign, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
