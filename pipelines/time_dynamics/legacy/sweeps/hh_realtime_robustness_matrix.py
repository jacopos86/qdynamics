#!/usr/bin/env python3
"""Deterministic Math17A robustness matrix for HH L=2, n_ph_max=1.

This entrypoint is a fixed-case reproducibility surface, not an Optuna study.  It
wraps the canonical artifact-seeded Chapter 17A realtime route and reuses the
existing evaluation/spectra helpers owned by the realtime Optuna harness.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.time_dynamics.legacy.checkpoint_types import (  # noqa: E402
    HIGH_MISS_NO_ADMIT_POLICY_DEFAULT,
    normalize_high_miss_no_admit_policy,
    physical_trajectory_rows,
)
from pipelines.time_dynamics.runners.generic_from_adapt_artifact import (  # noqa: E402
    build_parser as build_realtime_parser,
    run_from_args as run_realtime_from_args,
)
from pipelines.time_dynamics.legacy.checkpoint_route_defaults import (  # noqa: E402
    DRIVE_PATTERN_DEFAULT,
    DRIVE_T0_DEFAULT,
    DRIVE_TIME_SAMPLING_DEFAULT,
    EXACT_STEPS_MULTIPLIER_DEFAULT,
    NUM_TIMES_DEFAULT,
    T_FINAL_DEFAULT,
)
from pipelines.time_dynamics.legacy.analysis.hh_time_dynamics_spectra import render_spectrum_pdf  # noqa: E402
from pipelines.time_dynamics.optimization.hh_realtime_optuna import (  # noqa: E402
    BaseRunConfig,
    TrialParams,
    ValidityGates,
    _build_realtime_tokens,
    _invalid_reasons,
    _trial_metrics_from_payload,
    _write_json,
)

_PIPELINE_NAME = "hh_realtime_robustness_matrix_v1"
_SCOPE_NOTE = "HH-only deterministic Math17A matrix; fixed L=2, n_ph_max=1 artifact surface."
DEFAULT_ARTIFACT_JSON = Path(
    "artifacts/json/"
    "adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_with_ansatz_input_20260321T214822Z.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/time_dynamics/hh_realtime_robustness_matrix")
DEFAULT_TAG = "hh_l2_nph1_math17a_robustness_matrix"
DEFAULT_COMPILE_AUDIT_PREFERRED_FAKES = "FakeMarrakesh,FakeNighthawk,FakeFez"
CONTROLLER_PROFILE_DEFAULT = "default"
CONTROLLER_PROFILE_HH_L2_T8_ANCHOR_V1 = "hh_l2_t8_anchor_v1"
_CONTROLLER_PROFILE_CHOICES = (
    CONTROLLER_PROFILE_DEFAULT,
    CONTROLLER_PROFILE_HH_L2_T8_ANCHOR_V1,
)
_HH_L2_T8_ANCHOR_PROFILE = {
    "high_miss_no_admit_policy": "bounded_stay_advance",
    "confirm_score_mode": "exact_gain_ratio",
    "miss_persistence_spec": "3:3",
    "append_margin_abs": 1.0e-5,
    "drive_t0": 4.0,
}
_STAGE0 = 0
_STAGE1 = 1
_STAGE2 = 2


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    stage: int
    enable_drive: bool
    drive_A: float | None = None
    drive_omega: float | None = None
    drive_tbar: float | None = None
    drive_phi: float | None = None
    purpose: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatrixCaseObservation:
    case_id: str
    stage: int
    status: str
    case: dict[str, Any]
    metrics: dict[str, Any]
    high_signal: dict[str, Any]
    invalid_reasons: list[str]
    result_json: str | None
    spectra_json: str | None
    spectra_pdf: str | None
    progress_json: str | None
    partial_payload_json: str | None
    input_tokens_json: str | None
    case_summary_json: str | None
    realtime_command: str | None
    elapsed_s: float | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactScope:
    artifact_json: str
    problem: str
    L: int
    n_ph_max: int
    sources: dict[str, str]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


_STAGE_MATRIX_BUILT_MATH = (
    "Built Math: v(t)=A sin(omega t + phi) exp(-(t-t0)^2/(2 tbar^2)); "
    "matrix = fixed list of A,omega,tbar,phi cases."
)
def stage0_cases() -> tuple[MatrixCase, ...]:
    return (
        MatrixCase(
            case_id="static_no_drive",
            stage=_STAGE0,
            enable_drive=False,
            purpose="canonical static/no-drive Chapter 17A baseline",
        ),
        MatrixCase(
            case_id="drive_A0_safe",
            stage=_STAGE0,
            enable_drive=True,
            drive_A=0.0,
            drive_omega=1.0,
            drive_tbar=4.0,
            drive_phi=math.pi / 2.0,
            purpose="drive plumbing safe-test: drive enabled with neutral A=0",
        ),
    )


def stage1_cases() -> tuple[MatrixCase, ...]:
    phi = math.pi / 2.0
    return (
        MatrixCase("drive_A0p6_w1_tbar4", _STAGE1, True, 0.6, 1.0, 4.0, phi, "moderate signal scout"),
        MatrixCase("drive_A1p2_w1_tbar4", _STAGE1, True, 1.2, 1.0, 4.0, phi, "first stronger signal scout"),
        MatrixCase("drive_A2p0_w1_tbar4", _STAGE1, True, 2.0, 1.0, 4.0, phi, "visible occupation/energy scout"),
        MatrixCase("drive_A3p0_w1_tbar4", _STAGE1, True, 3.0, 1.0, 4.0, phi, "high-signal bounded scout"),
        MatrixCase("drive_A2p0_w0p5_tbar6", _STAGE1, True, 2.0, 0.5, 6.0, phi, "low-frequency charge-transfer scout"),
        MatrixCase("drive_A3p0_w0p5_tbar6", _STAGE1, True, 3.0, 0.5, 6.0, phi, "strong low-frequency scout"),
    )


def stage2_cases() -> tuple[MatrixCase, ...]:
    phi = math.pi / 2.0
    return (
        MatrixCase("drive_A4p0_w0p5_tbar6", _STAGE2, True, 4.0, 0.5, 6.0, phi, "conditional escalation: low frequency A=4"),
        MatrixCase("drive_A4p0_w1_tbar4", _STAGE2, True, 4.0, 1.0, 4.0, phi, "conditional escalation: unit frequency A=4"),
        MatrixCase("drive_A6p0_w0p5_tbar6_guarded", _STAGE2, True, 6.0, 0.5, 6.0, phi, "guarded final escalation after stable A=4 misses"),
    )


def default_cases(*, include_stage2: bool = False) -> tuple[MatrixCase, ...]:
    cases = [*stage0_cases(), *stage1_cases()]
    if bool(include_stage2):
        cases.extend(stage2_cases())
    return tuple(cases)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_get(mapping: Mapping[str, Any], path: Sequence[str]) -> tuple[Any, str] | None:
    node: Any = mapping
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node, ".".join(path)


def _find_first_key(value: Any, keys: set[str], *, prefix: str = "", depth: int = 0) -> tuple[Any, str] | None:
    if depth > 5 or not isinstance(value, Mapping):
        return None
    skip_keys = {
        "trajectory",
        "ledger",
        "raw_traces",
        "spectra",
        "statevector",
        "statevectors",
        "operator_pool",
        "candidate_pool",
        "ansatz_terms",
    }
    for key, child in value.items():
        key_text = str(key)
        path = key_text if not prefix else f"{prefix}.{key_text}"
        if key_text in keys:
            return child, path
        if key_text in skip_keys:
            continue
        found = _find_first_key(child, keys, prefix=path, depth=depth + 1)
        if found is not None:
            return found
    return None


def _first_metadata_value(
    payload: Mapping[str, Any],
    *,
    paths: Sequence[Sequence[str]],
    fallback_keys: set[str],
) -> tuple[Any, str] | None:
    for path in paths:
        found = _path_get(payload, path)
        if found is not None:
            return found
    return _find_first_key(payload, fallback_keys)


def _normalize_problem(raw: Any) -> str | None:
    if isinstance(raw, Mapping):
        for key in ("problem", "name", "family", "model", "model_name", "model_family"):
            if key in raw:
                normalized = _normalize_problem(raw[key])
                if normalized is not None:
                    return normalized
        return None
    text = str(raw).strip().lower().replace("-", "_").replace(" ", "_") if raw is not None else ""
    if text in {"hh", "hubbard_holstein", "hubbardholstein"}:
        return "hh"
    if text == "hubbard":
        return "hubbard"
    return text or None


def _as_int_or_none(raw: Any) -> int | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        numeric = float(raw)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    rounded = int(round(numeric))
    return rounded if abs(float(rounded) - float(numeric)) <= 1.0e-12 else None


def extract_artifact_scope_metadata(payload: Mapping[str, Any]) -> ArtifactScope:
    problem_found = _first_metadata_value(
        payload,
        paths=(
            ("settings", "problem"),
            ("metadata", "problem"),
            ("run_settings", "problem"),
            ("config", "problem"),
            ("cfg", "problem"),
            ("physics", "problem"),
            ("model", "problem"),
            ("model", "family"),
            ("model", "name"),
            ("model_family",),
            ("model_name",),
            ("problem",),
        ),
        fallback_keys={"problem", "problem_type", "model_family", "model_name"},
    )
    L_found = _first_metadata_value(
        payload,
        paths=(
            ("settings", "L"),
            ("settings", "dims"),
            ("settings", "num_sites"),
            ("metadata", "L"),
            ("run_settings", "L"),
            ("config", "L"),
            ("cfg", "L"),
            ("physics", "L"),
            ("model", "L"),
            ("L",),
        ),
        fallback_keys={"L", "dims", "num_sites", "n_sites"},
    )
    nph_found = _first_metadata_value(
        payload,
        paths=(
            ("settings", "n_ph_max"),
            ("settings", "nph_max"),
            ("metadata", "n_ph_max"),
            ("run_settings", "n_ph_max"),
            ("config", "n_ph_max"),
            ("cfg", "n_ph_max"),
            ("physics", "n_ph_max"),
            ("model", "n_ph_max"),
            ("n_ph_max",),
        ),
        fallback_keys={"n_ph_max", "nph_max", "n_phonon_max", "phonon_max", "max_phonon"},
    )

    missing = []
    if problem_found is None:
        missing.append("problem")
    if L_found is None:
        missing.append("L")
    if nph_found is None:
        missing.append("n_ph_max")
    if missing:
        raise ValueError(
            "HH realtime robustness matrix requires artifact metadata for "
            f"problem='hh', L=2, n_ph_max=1; missing: {', '.join(missing)}"
        )

    problem_raw, problem_source = problem_found
    L_raw, L_source = L_found
    nph_raw, nph_source = nph_found
    problem = _normalize_problem(problem_raw)
    L_value = _as_int_or_none(L_raw)
    nph_value = _as_int_or_none(nph_raw)
    if problem is None or L_value is None or nph_value is None:
        raise ValueError(
            "HH realtime robustness matrix could not parse artifact scope metadata: "
            f"problem={problem_raw!r} from {problem_source}, L={L_raw!r} from {L_source}, "
            f"n_ph_max={nph_raw!r} from {nph_source}"
        )
    return ArtifactScope(
        artifact_json="",
        problem=str(problem),
        L=int(L_value),
        n_ph_max=int(nph_value),
        sources={"problem": str(problem_source), "L": str(L_source), "n_ph_max": str(nph_source)},
    )


def validate_artifact_scope(artifact_json: Path | str) -> ArtifactScope:
    path = Path(artifact_json).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Artifact scope preflight failed: artifact JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Artifact scope preflight failed: artifact JSON is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Artifact scope preflight failed: expected object JSON at {path}")
    scope = extract_artifact_scope_metadata(payload)
    expected = {"problem": "hh", "L": 2, "n_ph_max": 1}
    actual = {"problem": scope.problem, "L": scope.L, "n_ph_max": scope.n_ph_max}
    if actual != expected:
        raise ValueError(
            "Artifact scope preflight failed for HH realtime robustness matrix: "
            f"expected {expected}, got {actual}; sources={scope.sources}; artifact={path}"
        )
    return ArtifactScope(
        artifact_json=str(path),
        problem=scope.problem,
        L=scope.L,
        n_ph_max=scope.n_ph_max,
        sources=dict(scope.sources),
    )


def _controller_profile_settings(args: argparse.Namespace) -> dict[str, Any]:
    profile = str(getattr(args, "controller_profile", CONTROLLER_PROFILE_DEFAULT))
    if profile == CONTROLLER_PROFILE_HH_L2_T8_ANCHOR_V1:
        return dict(_HH_L2_T8_ANCHOR_PROFILE)
    return {}


def _normalize_miss_persistence_spec(raw: str | None) -> str | None:
    if raw in {None, ""}:
        return None
    text = str(raw).strip().replace("/", ":")
    if ":" not in text:
        raise ValueError("miss persistence must use WINDOW:COUNT, e.g. 3:3.")
    raw_window, raw_count = text.split(":", 1)
    try:
        window = int(raw_window)
        count = int(raw_count)
    except Exception as exc:
        raise ValueError("miss persistence WINDOW and COUNT must be integers.") from exc
    if window < 1:
        raise ValueError("miss persistence WINDOW must be >= 1.")
    if count < 1:
        raise ValueError("miss persistence COUNT must be >= 1.")
    if count > window:
        raise ValueError("miss persistence COUNT must be <= WINDOW.")
    return f"{window}:{count}"


def _miss_persistence_window_count(spec: str | None) -> tuple[int, int]:
    normalized = _normalize_miss_persistence_spec(spec)
    if normalized is None:
        return 1, 1
    raw_window, raw_count = normalized.split(":", 1)
    return int(raw_window), int(raw_count)


def _resolved_trial_param_value(
    args: argparse.Namespace,
    key: str,
    explicit_value: Any,
    *,
    default: Any = None,
) -> Any:
    if explicit_value not in {None, ""}:
        return explicit_value
    profile_value = _controller_profile_settings(args).get(str(key), None)
    if profile_value not in {None, ""}:
        return profile_value
    return default


def _resolved_case_drive_t0(args: argparse.Namespace, *, enable_drive: bool) -> float:
    explicit = getattr(args, "drive_t0", None)
    if explicit is not None:
        return float(explicit)
    if bool(enable_drive):
        profile_value = _controller_profile_settings(args).get("drive_t0", None)
        if profile_value is not None:
            return float(profile_value)
    return float(DRIVE_T0_DEFAULT)


def _resolved_summary_drive_t0(args: argparse.Namespace) -> float:
    explicit = getattr(args, "drive_t0", None)
    if explicit is not None:
        return float(explicit)
    profile_value = _controller_profile_settings(args).get("drive_t0", None)
    return float(DRIVE_T0_DEFAULT if profile_value is None else profile_value)


def _resolved_append_pool_family(args: argparse.Namespace) -> str:
    explicit = getattr(args, "append_pool_family", None)
    if explicit not in {None, ""}:
        return str(explicit)
    profile_value = _controller_profile_settings(args).get("append_pool_family", None)
    if profile_value not in {None, ""}:
        return str(profile_value)
    return "match_replay"


def _case_base_config(args: argparse.Namespace, case: MatrixCase) -> BaseRunConfig:
    return BaseRunConfig(
        artifact_json=Path(args.artifact_json),
        study_profile="hh_l2_nph1_math17a_robustness_matrix_v1",
        loader_mode=str(args.loader_mode),
        generator_family=str(args.generator_family),
        fallback_family=str(args.fallback_family),
        append_pool_family=_resolved_append_pool_family(args),
        lock_fixed_manifold=bool(args.lock_fixed_manifold),
        allow_repeats=bool(args.allow_repeats),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        enable_drive=bool(case.enable_drive),
        disable_drive=False,
        drive_A=float(0.0 if case.drive_A is None else case.drive_A),
        drive_omega=float(1.0 if case.drive_omega is None else case.drive_omega),
        drive_tbar=float(4.0 if case.drive_tbar is None else case.drive_tbar),
        drive_phi=float(math.pi / 2.0 if case.drive_phi is None else case.drive_phi),
        drive_pattern=str(args.drive_pattern),
        drive_custom_weights=str(args.drive_custom_weights),
        drive_include_identity=bool(args.drive_include_identity),
        drive_time_sampling=str(args.drive_time_sampling),
        drive_t0=_resolved_case_drive_t0(args, enable_drive=bool(case.enable_drive)),
        exact_steps_multiplier=int(args.exact_steps_multiplier),
        pair=str(args.pair),
        spectra_detrend=str(args.spectra_detrend),
        spectra_window=str(args.spectra_window),
        max_peaks=int(args.max_peaks),
        max_harmonic=int(args.max_harmonic),
    )


def _matrix_trial_params(args: argparse.Namespace) -> TrialParams:
    confirm_score_mode = _resolved_trial_param_value(
        args,
        "confirm_score_mode",
        getattr(args, "checkpoint_controller_confirm_score_mode", None),
    )
    miss_persistence_spec = _normalize_miss_persistence_spec(
        _resolved_trial_param_value(
            args,
            "miss_persistence_spec",
            getattr(args, "checkpoint_controller_miss_persistence", None),
        )
    )
    append_margin_abs = _resolved_trial_param_value(
        args,
        "append_margin_abs",
        getattr(args, "checkpoint_controller_append_margin_abs", None),
    )
    return TrialParams(
        include_tangent_secant_proposal=False,
        high_miss_no_admit_policy=normalize_high_miss_no_admit_policy(
            args.checkpoint_controller_high_miss_no_admit_policy
        ),
        repair_retry_max_attempts=int(args.checkpoint_controller_repair_retry_max_attempts),
        repair_retry_escalation_mode=str(args.checkpoint_controller_repair_retry_escalation_mode),
        repair_retry_admission_policy=str(args.checkpoint_controller_repair_retry_admission_policy),
        repair_retry_rescue_min_gain_ratio=float(
            args.checkpoint_controller_repair_retry_rescue_min_gain_ratio
        ),
        repair_retry_rescue_attempt=str(args.checkpoint_controller_repair_retry_rescue_attempt),
        miss_persistence_spec=miss_persistence_spec,
        append_margin_abs=None if append_margin_abs is None else float(append_margin_abs),
        confirm_score_mode=None if confirm_score_mode in {None, ""} else str(confirm_score_mode),
    )


def _append_compile_audit_tokens(tokens: list[str], args: argparse.Namespace) -> list[str]:
    out = list(tokens)
    out.extend(["--compile-audit-mode", str(args.compile_audit_mode)])
    if str(args.compile_audit_mode) != "off":
        out.extend(
            [
                "--compile-audit-backend-name",
                str(args.compile_audit_backend_name),
                "--compile-audit-seed-transpiler",
                str(int(args.compile_audit_seed_transpiler)),
                "--compile-audit-optimization-level",
                str(int(args.compile_audit_optimization_level)),
                "--compile-audit-preferred-fake-backends",
                str(args.compile_audit_preferred_fake_backends),
            ]
        )
    return out


def validate_matrix_realtime_contract(tokens: Sequence[str], args: argparse.Namespace) -> dict[str, Any]:
    realtime_args = build_realtime_parser().parse_args([str(token) for token in tokens])
    trial_params = _matrix_trial_params(args)
    miss_window, miss_count = _miss_persistence_window_count(trial_params.miss_persistence_spec)
    contract = {
        "t_final": float(realtime_args.t_final),
        "num_times": int(realtime_args.num_times),
        "exact_steps_multiplier": int(realtime_args.exact_steps_multiplier),
        "drive_t0": float(realtime_args.drive_t0),
        "checkpoint_controller_mode": str(realtime_args.checkpoint_controller_mode),
        "checkpoint_controller_reference_mode": str(realtime_args.checkpoint_controller_reference_mode),
        "high_miss_no_admit_policy": normalize_high_miss_no_admit_policy(
            realtime_args.checkpoint_controller_high_miss_no_admit_policy
        ),
        "confirm_score_mode": str(realtime_args.checkpoint_controller_confirm_score_mode),
        "miss_persistence_window": int(realtime_args.checkpoint_controller_miss_persistence_window),
        "miss_persistence_count": int(realtime_args.checkpoint_controller_miss_persistence_count),
        "append_margin_abs": float(realtime_args.checkpoint_controller_append_margin_abs),
        "append_pool_family": str(realtime_args.append_pool_family),
        "repair_retry_max_attempts": int(realtime_args.checkpoint_controller_repair_retry_max_attempts),
        "repair_retry_escalation_mode": str(realtime_args.checkpoint_controller_repair_retry_escalation_mode),
        "repair_retry_admission_policy": str(realtime_args.checkpoint_controller_repair_retry_admission_policy),
        "repair_retry_rescue_min_gain_ratio": float(
            realtime_args.checkpoint_controller_repair_retry_rescue_min_gain_ratio
        ),
        "repair_retry_rescue_attempt": str(realtime_args.checkpoint_controller_repair_retry_rescue_attempt),
        "compile_audit_mode": str(realtime_args.compile_audit_mode),
        "compile_audit_backend_name": str(realtime_args.compile_audit_backend_name),
        "compile_audit_seed_transpiler": int(realtime_args.compile_audit_seed_transpiler),
        "compile_audit_optimization_level": int(realtime_args.compile_audit_optimization_level),
        "compile_audit_preferred_fake_backends": str(realtime_args.compile_audit_preferred_fake_backends),
    }
    expected = {
        "t_final": float(args.t_final),
        "num_times": int(args.num_times),
        "exact_steps_multiplier": int(args.exact_steps_multiplier),
        "drive_t0": _resolved_case_drive_t0(args, enable_drive=bool(realtime_args.enable_drive)),
        "checkpoint_controller_mode": "exact_v1",
        "checkpoint_controller_reference_mode": "benchmark_exact",
        "high_miss_no_admit_policy": normalize_high_miss_no_admit_policy(
            args.checkpoint_controller_high_miss_no_admit_policy
        ),
        "confirm_score_mode": str(trial_params.confirm_score_mode or "compressed_whitened_v1"),
        "miss_persistence_window": int(miss_window),
        "miss_persistence_count": int(miss_count),
        "append_margin_abs": float(
            1.0e-6 if trial_params.append_margin_abs is None else trial_params.append_margin_abs
        ),
        "append_pool_family": _resolved_append_pool_family(args),
        "repair_retry_max_attempts": int(args.checkpoint_controller_repair_retry_max_attempts),
        "repair_retry_escalation_mode": str(args.checkpoint_controller_repair_retry_escalation_mode),
        "repair_retry_admission_policy": str(args.checkpoint_controller_repair_retry_admission_policy),
        "repair_retry_rescue_min_gain_ratio": float(
            args.checkpoint_controller_repair_retry_rescue_min_gain_ratio
        ),
        "repair_retry_rescue_attempt": str(args.checkpoint_controller_repair_retry_rescue_attempt),
        "compile_audit_mode": str(args.compile_audit_mode),
        "compile_audit_backend_name": str(args.compile_audit_backend_name),
        "compile_audit_seed_transpiler": int(args.compile_audit_seed_transpiler),
        "compile_audit_optimization_level": int(args.compile_audit_optimization_level),
        "compile_audit_preferred_fake_backends": str(args.compile_audit_preferred_fake_backends),
    }
    mismatches = {
        key: {"expected": expected[key], "actual": contract.get(key)}
        for key in expected
        if contract.get(key) != expected[key]
    }
    if mismatches:
        raise ValueError(f"Matrix realtime contract drift: {mismatches}")
    return contract


def build_case_realtime_tokens(
    *,
    args: argparse.Namespace,
    case: MatrixCase,
    case_dir: Path,
) -> tuple[list[str], BaseRunConfig, Path, Path, Path]:
    result_json = case_dir / "result.json"
    progress_json = case_dir / "realtime_progress.json"
    partial_payload_json = case_dir / "partial_payload.json"
    base_cfg = _case_base_config(args, case)
    tokens = _build_realtime_tokens(
        base_cfg=base_cfg,
        params=_matrix_trial_params(args),
        output_json=result_json,
        run_tag=f"{args.tag}__{case.case_id}",
        progress_json=progress_json,
        partial_payload_json=partial_payload_json,
    )
    tokens = _append_compile_audit_tokens(tokens, args)
    validate_matrix_realtime_contract(tokens, args)
    return (
        tokens,
        base_cfg,
        result_json,
        progress_json,
        partial_payload_json,
    )


def _realtime_command(tokens: Sequence[str]) -> str:
    return "python -m pipelines.time_dynamics.runners.hh_from_adapt_artifact " + shlex.join(
        [str(token) for token in tokens]
    )


def _as_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if math.isfinite(numeric) else None


def _series_from_rows(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray | None:
    values: list[float] = []
    for row in rows:
        numeric = _as_finite_float(row.get(key))
        if numeric is None:
            return None
        values.append(float(numeric))
    return np.asarray(values, dtype=float) if values else None


def _site_matrix_from_rows(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray | None:
    matrix: list[list[float]] = []
    width: int | None = None
    for row in rows:
        raw = row.get(key)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return None
        vals: list[float] = []
        for item in raw:
            numeric = _as_finite_float(item)
            if numeric is None:
                return None
            vals.append(float(numeric))
        if width is None:
            width = len(vals)
        if len(vals) != int(width):
            return None
        matrix.append(vals)
    return np.asarray(matrix, dtype=float) if matrix else None


def _array_from_raw_traces(raw: Mapping[str, Any], key: str) -> np.ndarray | None:
    if key not in raw:
        return None
    try:
        arr = np.asarray(raw.get(key), dtype=float)
    except Exception:
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


_HIGH_SIGNAL_BUILT_MATH = (
    "Built Math: signal(case)=1[max_site_occ >= tau_n or span(E_exact) >= tau_E]; "
    "span(x)=max_t x(t)-min_t x(t)."
)
def high_signal_fields(
    *,
    payload: Mapping[str, Any],
    analysis: Mapping[str, Any] | None = None,
    site_threshold: float = 1.80,
    exact_energy_span_threshold: float = 1.0,
) -> dict[str, Any]:
    raw = analysis.get("raw_traces", {}) if isinstance(analysis, Mapping) and isinstance(analysis.get("raw_traces", {}), Mapping) else {}
    raw_rows = [dict(row) for row in payload.get("trajectory", []) if isinstance(row, Mapping)]
    rows = physical_trajectory_rows(raw_rows, fallback_to_raw=False)
    if len(rows) != len(raw_rows):
        raw = {}

    site = _array_from_raw_traces(raw, "site_occupations")
    if site is None:
        site = _site_matrix_from_rows(rows, "site_occupations")
    site_exact = _array_from_raw_traces(raw, "site_occupations_exact")
    if site_exact is None:
        site_exact = _site_matrix_from_rows(rows, "site_occupations_exact")
    energy = _array_from_raw_traces(raw, "energy_total")
    if energy is None:
        energy = _series_from_rows(rows, "energy_total")
    energy_exact = _array_from_raw_traces(raw, "energy_total_exact")
    if energy_exact is None:
        energy_exact = _series_from_rows(rows, "energy_total_exact")

    def _max(arr: np.ndarray | None) -> float | None:
        return None if arr is None or arr.size == 0 else float(np.max(arr))

    def _span(arr: np.ndarray | None) -> float | None:
        return None if arr is None or arr.size == 0 else float(np.ptp(arr.reshape(-1)))

    def _excursion(arr: np.ndarray | None) -> float | None:
        if arr is None or arr.size == 0:
            return None
        flat = arr.reshape(-1)
        return float(np.max(np.abs(flat - float(flat[0]))))

    max_site_occupation = _max(site)
    max_site_occupation_exact = _max(site_exact)
    max_site_for_signal = max(
        [x for x in (max_site_occupation, max_site_occupation_exact) if x is not None],
        default=None,
    )
    raw_total_energy_span = _span(energy)
    exact_total_energy_span = _span(energy_exact)
    raw_total_energy_excursion_from_initial = _excursion(energy)
    exact_total_energy_excursion_from_initial = _excursion(energy_exact)

    reasons: list[str] = []
    if max_site_for_signal is not None and max_site_for_signal >= float(site_threshold):
        reasons.append("max_site_occupation")
    exact_span_hit = exact_total_energy_span is not None and exact_total_energy_span >= float(
        exact_energy_span_threshold
    )
    if exact_span_hit:
        reasons.append("exact_total_energy_span")

    return {
        "is_high_signal": bool(reasons),
        "reasons": list(reasons),
        "thresholds": {
            "max_site_occupation": float(site_threshold),
            "exact_total_energy_span": float(exact_energy_span_threshold),
        },
        "max_site_occupation": max_site_occupation,
        "max_site_occupation_exact": max_site_occupation_exact,
        "max_site_occupation_for_signal": max_site_for_signal,
        "raw_total_energy_span": raw_total_energy_span,
        "exact_total_energy_span": exact_total_energy_span,
        "raw_total_energy_excursion_from_initial": raw_total_energy_excursion_from_initial,
        "exact_total_energy_excursion_from_initial": exact_total_energy_excursion_from_initial,
    }


def _evaluate_case(
    *,
    args: argparse.Namespace,
    case: MatrixCase,
    run_root: Path,
    validity_gates: ValidityGates,
) -> MatrixCaseObservation:
    started = time.time()
    case_dir = run_root / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    spectra_json = case_dir / "spectra.json"
    spectra_pdf = case_dir / "spectra.pdf"
    case_summary_json = case_dir / "case_summary.json"
    input_tokens_json = case_dir / "input_tokens.json"
    tokens, base_cfg, result_json, progress_json, partial_payload_json = build_case_realtime_tokens(
        args=args,
        case=case,
        case_dir=case_dir,
    )
    command = _realtime_command(tokens)
    _write_json(
        input_tokens_json,
        {
            "generated_utc": _now_utc(),
            "pipeline": _PIPELINE_NAME,
            "case": case.to_json(),
            "tokens": list(tokens),
            "realtime_command": command,
            "progress_json": str(progress_json),
            "partial_payload_json": str(partial_payload_json),
            "base_config": asdict(base_cfg),
            "trial_params": asdict(_matrix_trial_params(args)),
        },
    )
    try:
        realtime_args = build_realtime_parser().parse_args(tokens)
        payload = run_realtime_from_args(realtime_args)
        if not result_json.exists():
            _write_json(result_json, payload)
        analysis, metrics = _trial_metrics_from_payload(
            payload=payload,
            output_json=result_json,
            base_cfg=base_cfg,
        )
        _write_json(spectra_json, analysis)
        spectra_pdf_path: Path | None = None
        if not bool(getattr(args, "skip_spectra_pdf", False)):
            render_spectrum_pdf(
                analysis,
                output_pdf=spectra_pdf,
                max_harmonic=int(base_cfg.max_harmonic),
            )
            spectra_pdf_path = spectra_pdf
        invalid_reasons = _invalid_reasons(metrics, gates=validity_gates)
        high_signal = high_signal_fields(
            payload=payload,
            analysis=analysis,
            site_threshold=float(args.high_signal_site_occupation),
            exact_energy_span_threshold=float(args.high_signal_exact_energy_span),
        )
        status = "invalid" if invalid_reasons else "completed"
        obs = MatrixCaseObservation(
            case_id=str(case.case_id),
            stage=int(case.stage),
            status=status,
            case=case.to_json(),
            metrics=dict(metrics),
            high_signal=dict(high_signal),
            invalid_reasons=list(invalid_reasons),
            result_json=str(result_json),
            spectra_json=str(spectra_json),
            spectra_pdf=None if spectra_pdf_path is None else str(spectra_pdf_path),
            progress_json=str(progress_json) if progress_json.exists() else str(progress_json),
            partial_payload_json=str(partial_payload_json) if partial_payload_json.exists() else str(partial_payload_json),
            input_tokens_json=str(input_tokens_json),
            case_summary_json=str(case_summary_json),
            realtime_command=command,
            elapsed_s=float(time.time() - started),
        )
    except SystemExit as exc:
        obs = MatrixCaseObservation(
            case_id=str(case.case_id),
            stage=int(case.stage),
            status="failed",
            case=case.to_json(),
            metrics={},
            high_signal={"is_high_signal": False, "reasons": []},
            invalid_reasons=["system_exit"],
            result_json=str(result_json),
            spectra_json=None,
            spectra_pdf=None,
            progress_json=str(progress_json),
            partial_payload_json=str(partial_payload_json),
            input_tokens_json=str(input_tokens_json),
            case_summary_json=str(case_summary_json),
            realtime_command=command,
            elapsed_s=float(time.time() - started),
            error=f"SystemExit: {exc}",
        )
    except Exception as exc:
        obs = MatrixCaseObservation(
            case_id=str(case.case_id),
            stage=int(case.stage),
            status="failed",
            case=case.to_json(),
            metrics={},
            high_signal={"is_high_signal": False, "reasons": []},
            invalid_reasons=["exception"],
            result_json=str(result_json),
            spectra_json=None,
            spectra_pdf=None,
            progress_json=str(progress_json),
            partial_payload_json=str(partial_payload_json),
            input_tokens_json=str(input_tokens_json),
            case_summary_json=str(case_summary_json),
            realtime_command=command,
            elapsed_s=float(time.time() - started),
            error=str(exc),
        )
    _write_json(case_summary_json, obs.to_json())
    return obs


def _status_counts(observations: Sequence[MatrixCaseObservation]) -> dict[str, int]:
    return {
        "completed": int(sum(1 for obs in observations if obs.status == "completed")),
        "invalid": int(sum(1 for obs in observations if obs.status == "invalid")),
        "failed": int(sum(1 for obs in observations if obs.status == "failed")),
    }


def build_matrix_summary(
    *,
    args: argparse.Namespace,
    run_root: Path,
    observations: Sequence[MatrixCaseObservation],
    planned_cases: Sequence[MatrixCase],
    done: bool,
    artifact_scope: ArtifactScope | None = None,
) -> dict[str, Any]:
    counts = _status_counts(observations)
    high_signal_cases = [
        obs.case_id
        for obs in observations
        if bool(obs.high_signal.get("is_high_signal")) and obs.status == "completed"
    ]
    full_horizon_observed = [
        obs for obs in observations if "full_horizon_gate_passed" in dict(obs.metrics)
    ]
    full_horizon_failed_cases = [
        {
            "case_id": str(obs.case_id),
            "status": str(obs.status),
            "reason": str(obs.metrics.get("full_horizon_gate_reason", "failed")),
            "final_time": obs.metrics.get("full_horizon_final_time"),
            "expected_t_final": obs.metrics.get("full_horizon_expected_t_final"),
            "observed_row_count": obs.metrics.get("full_horizon_observed_row_count"),
            "expected_row_count": obs.metrics.get("full_horizon_expected_row_count"),
        }
        for obs in full_horizon_observed
        if obs.metrics.get("full_horizon_gate_passed") is False
    ]

    def _sum_metric(key: str) -> int:
        return int(sum(int(obs.metrics.get(key, 0) or 0) for obs in observations))

    def _merge_metric_counts(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for obs in observations:
            counts = obs.metrics.get(key, {})
            if not isinstance(counts, Mapping):
                continue
            for count_key, value in counts.items():
                out[str(count_key)] = int(out.get(str(count_key), 0)) + int(value)
        return dict(sorted(out.items()))

    first_bad_high_miss_no_admit_case = None
    for obs in observations:
        diag = obs.metrics.get("first_bad_high_miss_no_admit_checkpoint_diagnostic")
        if isinstance(diag, Mapping):
            first_bad_high_miss_no_admit_case = {
                "case_id": str(obs.case_id),
                "diagnostic": dict(diag),
            }
            break
    trial_params = _matrix_trial_params(args)
    miss_window, miss_count = _miss_persistence_window_count(trial_params.miss_persistence_spec)
    high_miss_policy = normalize_high_miss_no_admit_policy(
        args.checkpoint_controller_high_miss_no_admit_policy
    )
    repair_retry_active = str(high_miss_policy) == "repair_retry"
    repair_retry_rescue_configured = (
        str(args.checkpoint_controller_repair_retry_admission_policy)
        == "rescue_best_confirmed_append_v1"
    )
    repair_retry_rescue_active = bool(repair_retry_active and repair_retry_rescue_configured)
    return {
        "schema_version": _PIPELINE_NAME,
        "generated_utc": _now_utc(),
        "done": bool(done),
        "pipeline": _PIPELINE_NAME,
        "scope": {
            "note": _SCOPE_NOTE,
            "problem": "hh" if artifact_scope is None else str(artifact_scope.problem),
            "L": 2 if artifact_scope is None else int(artifact_scope.L),
            "n_ph_max": 1 if artifact_scope is None else int(artifact_scope.n_ph_max),
            "artifact_scope_sources": {} if artifact_scope is None else dict(artifact_scope.sources),
            "real_runtime_or_qpu": False,
            "repair_retry": bool(repair_retry_active),
            "repair_retry_rescue": bool(repair_retry_rescue_active),
            "repair_retry_rescue_configured": bool(repair_retry_rescue_configured),
            "repair_retry_rescue_active": bool(repair_retry_rescue_active),
            "high_miss_no_admit_soft_fallback_default": bool(
                high_miss_policy == HIGH_MISS_NO_ADMIT_POLICY_DEFAULT
            ),
            "scaffold_adapter": False,
        },
        "provenance": {
            "artifact_json": str(args.artifact_json),
            "validated_artifact_json": None if artifact_scope is None else str(artifact_scope.artifact_json),
            "output_dir": str(args.output_dir),
            "run_root": str(run_root),
            "tag": str(args.tag),
            "matrix_argv": list(getattr(args, "_matrix_argv", [])),
        },
        "defaults": {
            "controller_profile": str(args.controller_profile),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "exact_steps_multiplier": int(args.exact_steps_multiplier),
            "checkpoint_controller_mode": "exact_v1",
            "checkpoint_controller_reference_mode": "benchmark_exact",
            "high_miss_no_admit_policy": str(high_miss_policy),
            "confirm_score_mode": str(trial_params.confirm_score_mode or "compressed_whitened_v1"),
            "miss_persistence_window": int(miss_window),
            "miss_persistence_count": int(miss_count),
            "append_margin_abs": float(
                1.0e-6 if trial_params.append_margin_abs is None else trial_params.append_margin_abs
            ),
            "append_pool_family": _resolved_append_pool_family(args),
            "repair_retry_max_attempts": int(args.checkpoint_controller_repair_retry_max_attempts),
            "repair_retry_escalation_mode": str(args.checkpoint_controller_repair_retry_escalation_mode),
            "repair_retry_admission_policy": str(args.checkpoint_controller_repair_retry_admission_policy),
            "repair_retry_rescue_min_gain_ratio": float(
                args.checkpoint_controller_repair_retry_rescue_min_gain_ratio
            ),
            "repair_retry_rescue_attempt": str(args.checkpoint_controller_repair_retry_rescue_attempt),
            "compile_audit_mode": str(args.compile_audit_mode),
            "compile_audit_backend_name": str(args.compile_audit_backend_name),
            "compile_audit_seed_transpiler": int(args.compile_audit_seed_transpiler),
            "compile_audit_optimization_level": int(args.compile_audit_optimization_level),
            "compile_audit_preferred_fake_backends": str(args.compile_audit_preferred_fake_backends),
            "drive_pattern": str(args.drive_pattern),
            "drive_time_sampling": str(args.drive_time_sampling),
            "drive_t0": _resolved_summary_drive_t0(args),
            "drive_t0_explicit": bool(getattr(args, "drive_t0", None) is not None),
            "skip_spectra_pdf": bool(getattr(args, "skip_spectra_pdf", False)),
        },
        "stage2": {
            "mode": str(args.stage2_mode),
            "include_stage2_flag": bool(getattr(args, "include_stage2", False)),
            "available_cases": [case.to_json() for case in stage2_cases()],
        },
        "high_signal_criteria": {
            "max_site_occupation_gte": float(args.high_signal_site_occupation),
            "exact_total_energy_span_gte": float(args.high_signal_exact_energy_span),
        },
        "planned_case_count": int(len(planned_cases)),
        "observed_case_count": int(len(observations)),
        "status_counts": counts,
        "completed_case_count": int(counts["completed"]),
        "invalid_case_count": int(counts["invalid"]),
        "failed_case_count": int(counts["failed"]),
        "high_signal_case_count": int(len(high_signal_cases)),
        "high_signal_cases": list(high_signal_cases),
        "full_horizon_observed_case_count": int(len(full_horizon_observed)),
        "full_horizon_passed_case_count": int(
            sum(1 for obs in full_horizon_observed if obs.metrics.get("full_horizon_gate_passed") is True)
        ),
        "full_horizon_failed_case_count": int(len(full_horizon_failed_cases)),
        "full_horizon_all_observed_cases_passed": bool(
            full_horizon_observed and not full_horizon_failed_cases
        ),
        "full_horizon_failed_cases": list(full_horizon_failed_cases),
        "high_miss_diagnostics": {
            "high_miss_count": _sum_metric("high_miss_count"),
            "high_miss_no_admit_count": _sum_metric("high_miss_no_admit_count"),
            "append_no_harm_veto_count": _sum_metric("append_no_harm_veto_count"),
            "high_miss_no_admit_reason_counts": _merge_metric_counts(
                "high_miss_no_admit_reason_counts"
            ),
            "high_miss_no_admit_resolution_counts": _merge_metric_counts(
                "high_miss_no_admit_resolution_counts"
            ),
            "append_no_harm_veto_reason_counts": _merge_metric_counts(
                "append_no_harm_veto_reason_counts"
            ),
            "first_bad_high_miss_no_admit_case": first_bad_high_miss_no_admit_case,
        },
        "planned_cases": [case.to_json() for case in planned_cases],
        "cases": [obs.to_json() for obs in observations],
    }


def _write_summary(
    *,
    args: argparse.Namespace,
    run_root: Path,
    observations: Sequence[MatrixCaseObservation],
    planned_cases: Sequence[MatrixCase],
    done: bool,
    artifact_scope: ArtifactScope | None = None,
) -> dict[str, Any]:
    summary = build_matrix_summary(
        args=args,
        run_root=run_root,
        observations=observations,
        planned_cases=planned_cases,
        done=done,
        artifact_scope=artifact_scope,
    )
    summary_json = Path(args.summary_json) if args.summary_json else run_root / "summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(summary_json, summary)
    return summary


def _stage1_has_completed_high_signal(observations: Sequence[MatrixCaseObservation]) -> bool:
    return any(
        int(obs.stage) == _STAGE1
        and obs.status == "completed"
        and bool(obs.high_signal.get("is_high_signal"))
        for obs in observations
    )


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    _normalize_args(args)
    artifact_scope = validate_artifact_scope(args.artifact_json)
    run_root = Path(args.output_dir) / str(args.tag)
    run_root.mkdir(parents=True, exist_ok=True)
    observations: list[MatrixCaseObservation] = []
    planned: list[MatrixCase] = [*stage0_cases(), *stage1_cases()]
    if str(args.stage2_mode) == "all":
        planned.extend(stage2_cases())
    validity_gates = ValidityGates()

    for case in [*stage0_cases(), *stage1_cases()]:
        obs = _evaluate_case(args=args, case=case, run_root=run_root, validity_gates=validity_gates)
        observations.append(obs)
        _write_summary(
            args=args,
            run_root=run_root,
            observations=observations,
            planned_cases=planned,
            done=False,
            artifact_scope=artifact_scope,
        )
        if bool(args.stop_on_failure) and obs.status == "failed":
            return _write_summary(
                args=args,
                run_root=run_root,
                observations=observations,
                planned_cases=planned,
                done=True,
                artifact_scope=artifact_scope,
            )

    if str(args.stage2_mode) == "if-miss" and not _stage1_has_completed_high_signal(observations):
        planned.extend(stage2_cases())
        for case in stage2_cases():
            obs = _evaluate_case(args=args, case=case, run_root=run_root, validity_gates=validity_gates)
            observations.append(obs)
            _write_summary(
                args=args,
                run_root=run_root,
                observations=observations,
                planned_cases=planned,
                done=False,
                artifact_scope=artifact_scope,
            )
            if obs.status != "completed" or bool(obs.high_signal.get("is_high_signal")):
                break
    elif str(args.stage2_mode) == "all":
        for case in stage2_cases():
            obs = _evaluate_case(args=args, case=case, run_root=run_root, validity_gates=validity_gates)
            observations.append(obs)
            _write_summary(
                args=args,
                run_root=run_root,
                observations=observations,
                planned_cases=planned,
                done=False,
                artifact_scope=artifact_scope,
            )
            if bool(args.stop_on_failure) and obs.status == "failed":
                break

    return _write_summary(
        args=args,
        run_root=run_root,
        observations=observations,
        planned_cases=planned,
        done=True,
        artifact_scope=artifact_scope,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic HH L=2,n_ph_max=1 Math17A robustness matrix."
    )
    parser.add_argument("--artifact-json", default=str(DEFAULT_ARTIFACT_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument(
        "--controller-profile",
        choices=_CONTROLLER_PROFILE_CHOICES,
        default=CONTROLLER_PROFILE_DEFAULT,
        help=(
            "Reusable controller knob bundle. hh_l2_t8_anchor_v1 sets the validated L=2 t=8 "
            "anchor knobs while leaving explicit CLI overrides authoritative."
        ),
    )
    parser.add_argument("--loader-mode", default="replay_family")
    parser.add_argument("--generator-family", default="match_adapt")
    parser.add_argument("--fallback-family", default="full_meta")
    parser.add_argument(
        "--append-pool-family",
        "--candidate-pool-family",
        dest="append_pool_family",
        default=None,
        help=(
            "Explicit append/candidate-pool family. If omitted, all profiles use match_replay; "
            "full_meta append remains an explicit diagnostic/tuning override."
        ),
    )
    parser.add_argument("--lock-fixed-manifold", action="store_true")
    parser.add_argument("--allow-repeats", action="store_true")
    parser.add_argument("--t-final", type=float, default=T_FINAL_DEFAULT)
    parser.add_argument("--num-times", type=int, default=NUM_TIMES_DEFAULT)
    parser.add_argument("--exact-steps-multiplier", type=int, default=EXACT_STEPS_MULTIPLIER_DEFAULT)
    parser.add_argument("--drive-pattern", default=DRIVE_PATTERN_DEFAULT)
    parser.add_argument("--drive-custom-weights", default="")
    parser.add_argument("--drive-include-identity", action="store_true")
    parser.add_argument("--drive-time-sampling", default=DRIVE_TIME_SAMPLING_DEFAULT)
    parser.add_argument(
        "--drive-t0",
        type=float,
        default=None,
        help=(
            "Explicit drive time offset. If omitted, hh_l2_t8_anchor_v1 uses 4.0 for drive cases; "
            "other profiles use the route default."
        ),
    )
    parser.add_argument("--pair", default="auto")
    parser.add_argument("--spectra-detrend", choices=("constant", "linear", "none"), default="linear")
    parser.add_argument("--spectra-window", choices=("hann", "none"), default="hann")
    parser.add_argument("--max-peaks", type=int, default=5)
    parser.add_argument("--max-harmonic", type=int, default=3)
    parser.add_argument(
        "--stage2-mode",
        choices=("off", "if-miss", "all"),
        default="off",
        help="Stage 2 is opt-in. 'if-miss' runs it only if Stage 1 has no completed high-signal case.",
    )
    parser.add_argument(
        "--include-stage2",
        action="store_true",
        help="Alias for --stage2-mode if-miss when --stage2-mode is left at off.",
    )
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--high-signal-site-occupation", type=float, default=1.80)
    parser.add_argument("--high-signal-exact-energy-span", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-controller-high-miss-no-admit-policy",
        choices=("bounded_stay_advance", "legacy_advance_stay", "repair_stop", "repair_retry"),
        default=HIGH_MISS_NO_ADMIT_POLICY_DEFAULT,
        help=(
            "Default bounded_stay_advance advances a physical state_sample with loud telemetry; "
            "repair_stop is explicit strict diagnostic mode; repair_retry is experimental opt-in."
        ),
    )
    parser.add_argument("--checkpoint-controller-repair-retry-max-attempts", type=int, default=2)
    parser.add_argument(
        "--checkpoint-controller-repair-retry-escalation-mode",
        choices=("append_budget_then_stabilize_v1",),
        default="append_budget_then_stabilize_v1",
    )
    parser.add_argument(
        "--checkpoint-controller-repair-retry-admission-policy",
        choices=("strict", "rescue_best_confirmed_append_v1"),
        default="strict",
    )
    parser.add_argument(
        "--checkpoint-controller-repair-retry-rescue-min-gain-ratio",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--checkpoint-controller-repair-retry-rescue-attempt",
        choices=("terminal_attempt_only",),
        default="terminal_attempt_only",
    )
    parser.add_argument(
        "--checkpoint-controller-confirm-score-mode",
        choices=("exact_gain_ratio", "compressed_whitened_v1"),
        default=None,
        help="Explicitly forward the controller confirm-score mode; profile/default may also set it.",
    )
    parser.add_argument(
        "--checkpoint-controller-miss-persistence",
        default=None,
        help="Explicit high-miss persistence as WINDOW:COUNT or WINDOW/COUNT, e.g. 3:3.",
    )
    parser.add_argument(
        "--checkpoint-controller-append-margin-abs",
        type=float,
        default=None,
        help="Explicit append absolute-improvement margin. Unset leaves legacy route default unless a profile sets it.",
    )
    parser.add_argument(
        "--compile-audit-mode",
        choices=("off", "final_scaffold"),
        default="final_scaffold",
        help="Matrix default is the Phase-1 final-scaffold local fake compile audit.",
    )
    parser.add_argument("--compile-audit-backend-name", default="FakeMarrakesh")
    parser.add_argument("--compile-audit-seed-transpiler", type=int, default=7)
    parser.add_argument("--compile-audit-optimization-level", type=int, default=2)
    parser.add_argument(
        "--compile-audit-preferred-fake-backends",
        default=DEFAULT_COMPILE_AUDIT_PREFERRED_FAKES,
    )
    parser.add_argument(
        "--skip-spectra-pdf",
        action="store_true",
        help="Write spectra JSON only; skip per-case spectra PDF rendering.",
    )
    return parser


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if bool(args.include_stage2) and str(args.stage2_mode) == "off":
        args.stage2_mode = "if-miss"
    if str(getattr(args, "controller_profile", CONTROLLER_PROFILE_DEFAULT)) not in _CONTROLLER_PROFILE_CHOICES:
        raise ValueError(f"Unsupported controller profile {args.controller_profile!r}.")
    args.checkpoint_controller_high_miss_no_admit_policy = normalize_high_miss_no_admit_policy(
        args.checkpoint_controller_high_miss_no_admit_policy
    )
    args.checkpoint_controller_miss_persistence = _normalize_miss_persistence_spec(
        getattr(args, "checkpoint_controller_miss_persistence", None)
    )
    if getattr(args, "checkpoint_controller_append_margin_abs", None) is not None:
        append_margin = float(args.checkpoint_controller_append_margin_abs)
        if not math.isfinite(append_margin) or append_margin < 0.0:
            raise ValueError("--checkpoint-controller-append-margin-abs must be finite and nonnegative.")
    if getattr(args, "drive_t0", None) is not None:
        drive_t0 = float(args.drive_t0)
        if not math.isfinite(drive_t0):
            raise ValueError("--drive-t0 must be finite.")
    retry_attempts = int(args.checkpoint_controller_repair_retry_max_attempts)
    if retry_attempts < 0 or retry_attempts > 2:
        raise ValueError("--checkpoint-controller-repair-retry-max-attempts must be between 0 and 2.")
    rescue_min_gain = float(args.checkpoint_controller_repair_retry_rescue_min_gain_ratio)
    if not math.isfinite(rescue_min_gain) or rescue_min_gain < 0.0:
        raise ValueError(
            "--checkpoint-controller-repair-retry-rescue-min-gain-ratio must be finite and nonnegative."
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args._matrix_argv = list(sys.argv[1:] if argv is None else argv)
    _normalize_args(args)
    run_matrix(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
