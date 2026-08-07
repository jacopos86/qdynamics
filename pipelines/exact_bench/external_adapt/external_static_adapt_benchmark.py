#!/usr/bin/env python3
"""Benchmark-local public-code external ADAPT static benchmark adapter.

This module is the exact-bench home for executable external ADAPT competitor
rows.  It intentionally imports third-party CEO code lazily from the allowlisted
external checkout and never calls the project Phase3 controller as a substitute
for CEO/TETRIS/Overlap logic.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.benchmark_metrics_proxy import write_proxy_sidecars
from pipelines.exact_bench.comparator_provenance import comparator_source_fields
from pipelines.exact_bench.external_adapt.provenance import (
    CEO_ADAPT_VQE_PINNED_COMMIT,
    external_algorithm_manifest_metadata,
    get_external_reference_spec,
)
from pipelines.exact_bench.external_adapt.repository_manager import checkout_dir_for

SCHEMA_VERSION = "external_static_adapt_benchmark_v1"
_RUNNER_MODULE = "pipelines.exact_bench.external_adapt.external_static_adapt_benchmark"
_CEO_WORKER_SCHEMA = "ceo_public_code_worker_v1"
_CEO_REFERENCE_ID = "ceo_adapt_vqe"
_CEO_METHOD_ID = "static_ceo_adapt_phase3"
_TETRIS_METHOD_ID = "static_tetris_adapt_phase3"
_CEO_DISPATCH = "external_static_adapt_ceo_public_code"
_TETRIS_DISPATCH = "external_static_adapt_tetris_public_code"
_EXTERNAL_ADAPT_DISPATCH_BY_ALGORITHM = {
    _CEO_METHOD_ID: _CEO_DISPATCH,
    _TETRIS_METHOD_ID: _TETRIS_DISPATCH,
}
_EXTERNAL_ADAPT_WORKER_MODE_BY_ALGORITHM = {
    _CEO_METHOD_ID: "ceo",
    _TETRIS_METHOD_ID: "tetris",
}
_EXTERNAL_ADAPT_PYTHON_ENV = "HOLSTEIN_EXTERNAL_ADAPT_PYTHON"
_DEFAULT_CEO_VENV_PYTHON = Path("~/.cache/holstein_external_competitors/ceo_adapt_vqe/.venv/bin/python").expanduser()
_DEFAULT_NUMBA_CACHE_DIR = Path("/tmp/holstein_external_adapt_numba_cache")


class ExternalAdaptUnavailable(RuntimeError):
    """Raised when the external public-code adapter cannot run in this env."""


class ExternalAdaptProvenanceMismatch(RuntimeError):
    """Raised when the materialized external checkout is not at the required pin."""


class ExternalAdaptWorkerFailed(RuntimeError):
    """Raised when the isolated public-code worker returns a runtime failure."""

    def __init__(self, message: str, *, exception_type: str = "ExternalAdaptWorkerFailed") -> None:
        super().__init__(message)
        self.exception_type = exception_type


@dataclass(frozen=True)
class ExternalAdaptHubbardCaseSettings:
    """Parameter payload passed to the pinned public-code Hubbard worker."""

    case_profile: str
    x_dim: int
    y_dim: int
    t: float
    u: float
    periodic: bool
    particle_hole_symmetry: bool
    threshold: float
    max_adapt_iter: int
    max_opt_iter: int

    @property
    def L(self) -> int:
        return int(self.x_dim) * int(self.y_dim)

    def to_worker_args(self) -> tuple[str, ...]:
        return (
            "--case-profile",
            str(self.case_profile),
            "--x-dim",
            str(int(self.x_dim)),
            "--y-dim",
            str(int(self.y_dim)),
            "--t",
            repr(float(self.t)),
            "--u",
            repr(float(self.u)),
            "--periodic",
            "true" if self.periodic else "false",
            "--particle-hole-symmetry",
            "true" if self.particle_hole_symmetry else "false",
            "--threshold",
            repr(float(self.threshold)),
            "--max-adapt-iter",
            str(int(self.max_adapt_iter)),
            "--max-opt-iter",
            str(int(self.max_opt_iter)),
        )


_EXTERNAL_ADAPT_HUBBARD_L2_DEFAULT_SETTINGS = ExternalAdaptHubbardCaseSettings(
    case_profile="external_hubbard_L2_public_code_default",
    x_dim=2,
    y_dim=1,
    t=1.0,
    u=4.0,
    periodic=True,
    particle_hole_symmetry=False,
    threshold=1e-3,
    max_adapt_iter=6,
    max_opt_iter=300,
)
_EXTERNAL_ADAPT_HUBBARD_TETRIS_CASE_SETTINGS: dict[str, ExternalAdaptHubbardCaseSettings] = {
    "hubbard_L2": _EXTERNAL_ADAPT_HUBBARD_L2_DEFAULT_SETTINGS,
    "hubbard_L2_three_model_weak": ExternalAdaptHubbardCaseSettings(
        case_profile="paper_i_hubbard_L2_three_model_weak_tetris_diagnostic",
        x_dim=2,
        y_dim=1,
        t=1.0,
        u=0.5,
        periodic=True,
        particle_hole_symmetry=False,
        threshold=1e-8,
        max_adapt_iter=80,
        max_opt_iter=300,
    ),
    "hubbard_L2_three_model_strong": ExternalAdaptHubbardCaseSettings(
        case_profile="paper_i_hubbard_L2_three_model_strong_tetris_diagnostic",
        x_dim=2,
        y_dim=1,
        t=1.0,
        u=1.5,
        periodic=True,
        particle_hole_symmetry=False,
        threshold=1e-8,
        max_adapt_iter=20,
        max_opt_iter=300,
    ),
}
_EXTERNAL_ADAPT_CASE_SETTINGS_BY_ALGORITHM_FAMILY: dict[
    str, dict[str, dict[str, ExternalAdaptHubbardCaseSettings]]
] = {
    _CEO_METHOD_ID: {"hubbard": {"hubbard_L2": _EXTERNAL_ADAPT_HUBBARD_L2_DEFAULT_SETTINGS}},
    _TETRIS_METHOD_ID: {"hubbard": _EXTERNAL_ADAPT_HUBBARD_TETRIS_CASE_SETTINGS},
}
_EXTERNAL_ADAPT_CASE_IDS_BY_ALGORITHM_FAMILY = {
    algorithm_id: {
        family: tuple(settings_by_case)
        for family, settings_by_case in cases_by_family.items()
    }
    for algorithm_id, cases_by_family in _EXTERNAL_ADAPT_CASE_SETTINGS_BY_ALGORITHM_FAMILY.items()
}


@dataclass(frozen=True)
class ExternalAdaptRunSummary:
    """Small normalized summary returned by the public-code execution layer."""

    energy: float
    exact_energy: float | None
    initial_energy: float | None
    selected_operator_count: int
    num_parameters: int
    nfev: int | None
    ngev: int | None
    nit: int | None
    adapt_iterations: int
    adapt_success: bool
    adapt_stop_reason: str
    pool_name: str
    pool_size: int | None
    selected_indices: tuple[int, ...]
    coefficients: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    selected_gradients: tuple[float, ...]
    nfevs_by_iteration: tuple[tuple[int, ...], ...] = ()
    ngevs_by_iteration: tuple[tuple[int, ...], ...] = ()
    nits_by_iteration: tuple[tuple[int, ...], ...] = ()
    energy_history: tuple[float, ...] = ()
    adapt_history: tuple[dict[str, Any], ...] = ()
    raw_stdout_tail: str = ""
    worker_mode: str = "ceo"
    tetris_enabled: bool = False
    tetris_batching_enabled: bool = False
    tetris_progressive_opt: bool = False
    tetris_candidate_window: str | None = None
    tetris_screening_rule: str | None = None
    operators_added_per_iteration: tuple[int, ...] = ()
    max_operators_added_per_iteration: int | None = None
    batch_iterations: int | None = None
    selected_indices_by_iteration: tuple[tuple[int, ...], ...] = ()
    coefficients_by_iteration: tuple[tuple[float, ...], ...] = ()
    worker_python: str = ""
    worker_python_source: str = ""
    worker_schema: str = ""
    worker_returncode: int | None = None
    external_case_profile: str | None = None
    hubbard_x_dim: int | None = None
    hubbard_y_dim: int | None = None
    hubbard_t: float | None = None
    hubbard_u: float | None = None
    hubbard_periodic: bool | None = None
    hubbard_particle_hole_symmetry: bool | None = None
    adapt_threshold: float | None = None
    adapt_max_adapt_iter: int | None = None
    adapt_max_opt_iter: int | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            return str(value)
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sum_nested(values: Any) -> int | None:
    if values is None:
        return None
    if isinstance(values, (int, np.integer)):
        return int(values)
    if isinstance(values, (float, np.floating)):
        return int(values)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        total = 0
        saw = False
        for item in values:
            part = _sum_nested(item)
            if part is None:
                continue
            total += int(part)
            saw = True
        return total if saw else None
    return None


def _float_tuple(values: Any) -> tuple[float, ...]:
    if values is None:
        return ()
    out: list[float] = []
    for item in list(values):
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            out.extend(_float_tuple(item))
        else:
            try:
                out.append(float(item))
            except Exception:
                pass
    return tuple(out)


def _int_tuple(values: Any) -> tuple[int, ...]:
    if values is None:
        return ()
    out: list[int] = []
    for item in list(values):
        try:
            out.append(int(item))
        except Exception:
            pass
    return tuple(out)


def _tuple_of_int_tuples(values: Any) -> tuple[tuple[int, ...], ...]:
    if values is None or isinstance(values, (str, bytes, bytearray)):
        return ()
    out: list[tuple[int, ...]] = []
    try:
        iterator = iter(values)
    except TypeError:
        return ()
    for item in iterator:
        if item is None or isinstance(item, (str, bytes, bytearray)):
            out.append(())
        elif isinstance(item, (int, np.integer)):
            out.append((int(item),))
        else:
            out.append(_int_tuple(item))
    return tuple(out)


def _tuple_of_float_tuples(values: Any) -> tuple[tuple[float, ...], ...]:
    if values is None or isinstance(values, (str, bytes, bytearray)):
        return ()
    out: list[tuple[float, ...]] = []
    try:
        iterator = iter(values)
    except TypeError:
        return ()
    for item in iterator:
        if item is None or isinstance(item, (str, bytes, bytearray)):
            out.append(())
        elif isinstance(item, (int, float, np.integer, np.floating)):
            out.append((float(item),))
        else:
            out.append(_float_tuple(item))
    return tuple(out)


def _tuple_of_dicts(values: Any) -> tuple[dict[str, Any], ...]:
    if values is None or isinstance(values, (str, bytes, bytearray)):
        return ()
    try:
        iterator = iter(values)
    except TypeError:
        return ()
    out: list[dict[str, Any]] = []
    for item in iterator:
        if isinstance(item, Mapping):
            out.append(dict(item))
    return tuple(out)


def _license_files(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    return tuple(
        sorted(
            child.name
            for child in path.iterdir()
            if child.is_file() and child.name.lower().startswith(("license", "copying"))
        )
    )


def default_external_static_adapt_case_ids(family: str, algorithm_id: str) -> tuple[str, ...]:
    """Return first-slice external ADAPT cases with executable adapters."""

    family_key = str(family).strip()
    algorithm_key = str(algorithm_id).strip()
    cases_by_family = _EXTERNAL_ADAPT_CASE_IDS_BY_ALGORITHM_FAMILY.get(algorithm_key, {})
    return tuple(cases_by_family.get(family_key, ()))


def external_static_adapt_dispatch_for_algorithm(algorithm_id: str) -> str | None:
    """Return the concrete benchmark-local dispatch label for a wired external ADAPT row."""

    return _EXTERNAL_ADAPT_DISPATCH_BY_ALGORITHM.get(str(algorithm_id).strip())


def _dispatch_for_algorithm(algorithm_id: str) -> str:
    return external_static_adapt_dispatch_for_algorithm(algorithm_id) or "external_static_adapt_scaffold"


def _worker_mode_for_algorithm(algorithm_id: str) -> str | None:
    return _EXTERNAL_ADAPT_WORKER_MODE_BY_ALGORITHM.get(str(algorithm_id).strip())


def _case_settings_for(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
) -> ExternalAdaptHubbardCaseSettings | None:
    return (
        _EXTERNAL_ADAPT_CASE_SETTINGS_BY_ALGORITHM_FAMILY.get(str(algorithm_id).strip(), {})
        .get(str(family).strip(), {})
        .get(str(case_id).strip())
    )


def _is_promoted_tetris_row(*, family: str, case_id: str, algorithm_id: str) -> bool:
    return str(algorithm_id).strip() == _TETRIS_METHOD_ID and str(case_id).strip() in default_external_static_adapt_case_ids(
        str(family).strip(),
        _TETRIS_METHOD_ID,
    )


def _is_promoted_external_row(*, family: str, case_id: str, algorithm_id: str) -> bool:
    return str(case_id).strip() in default_external_static_adapt_case_ids(str(family).strip(), str(algorithm_id).strip())


def _resolved_git_commit(checkout_dir: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(checkout_dir),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _validate_ceo_checkout(checkout_dir: Path | None = None) -> tuple[Path, str]:
    checkout = Path(checkout_dir) if checkout_dir is not None else checkout_dir_for(_CEO_REFERENCE_ID)
    checkout = checkout.expanduser()
    if not checkout.exists():
        raise ExternalAdaptUnavailable(
            f"CEO reference checkout is missing at {checkout}; fetch/materialize {_CEO_REFERENCE_ID} first"
        )
    if not (checkout / ".git").exists():
        raise ExternalAdaptUnavailable(f"CEO reference checkout at {checkout} is not a git checkout")
    try:
        resolved = _resolved_git_commit(checkout)
    except Exception as exc:  # pragma: no cover - exercised through controlled skip path in tests
        raise ExternalAdaptUnavailable(f"could not resolve CEO checkout commit at {checkout}: {exc}") from exc
    if resolved != CEO_ADAPT_VQE_PINNED_COMMIT:
        raise ExternalAdaptProvenanceMismatch(
            f"CEO checkout commit {resolved} does not match required {CEO_ADAPT_VQE_PINNED_COMMIT}"
        )
    return checkout, resolved


def _text_tail(text: str, *, max_lines: int = 20) -> str:
    lines = str(text or "").splitlines()
    return "\n".join(lines[-max_lines:])


def _worker_script_path() -> Path:
    return Path(__file__).with_name("ceo_public_code_worker.py")


def _worker_environment() -> dict[str, str]:
    env = os.environ.copy()
    # quimb imports numba-jitted helpers with cache=True.  When the external venv
    # itself lives under ~/.cache, numba can fail to derive a source-file cache
    # locator unless an explicit normal cache directory is supplied.
    env.setdefault("NUMBA_CACHE_DIR", str(_DEFAULT_NUMBA_CACHE_DIR))
    return env


def _resolve_external_adapt_python() -> tuple[Path, str]:
    """Return the Python interpreter used for isolated external ADAPT execution."""

    raw_env = os.environ.get(_EXTERNAL_ADAPT_PYTHON_ENV)
    if raw_env:
        python = Path(raw_env).expanduser()
        source = f"env:{_EXTERNAL_ADAPT_PYTHON_ENV}"
    elif _DEFAULT_CEO_VENV_PYTHON.exists():
        python = _DEFAULT_CEO_VENV_PYTHON
        source = "default_ceo_cache_venv"
    else:
        python = Path(sys.executable).expanduser()
        source = "current_python"

    if not python.exists():
        raise ExternalAdaptUnavailable(f"external ADAPT Python interpreter from {source} is missing at {python}")
    if python.is_dir():
        raise ExternalAdaptUnavailable(f"external ADAPT Python interpreter from {source} is a directory, not an executable: {python}")
    if not os.access(python, os.X_OK):
        raise ExternalAdaptUnavailable(f"external ADAPT Python interpreter from {source} is not executable: {python}")
    return python, source


def _worker_protocol_error(message: str, *, stderr: str = "") -> ExternalAdaptWorkerFailed:
    if stderr:
        message = f"{message}; worker stderr tail: {_text_tail(stderr)}"
    return ExternalAdaptWorkerFailed(message, exception_type="WorkerProtocolError")


def _parse_worker_payload(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    stdout = str(completed.stdout or "").strip()
    if not stdout:
        raise _worker_protocol_error("CEO public-code worker did not emit JSON", stderr=str(completed.stderr or ""))
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise _worker_protocol_error(
            f"CEO public-code worker did not return strict JSON ({exc})",
            stderr=str(completed.stderr or ""),
        ) from exc
    if not isinstance(payload, dict):
        raise _worker_protocol_error("CEO public-code worker JSON was not an object", stderr=str(completed.stderr or ""))
    if payload.get("schema") != _CEO_WORKER_SCHEMA:
        raise _worker_protocol_error(
            f"CEO public-code worker schema mismatch: {payload.get('schema')!r}",
            stderr=str(completed.stderr or ""),
        )
    return payload


def _summary_from_worker_result(
    result: Mapping[str, Any],
    *,
    worker_python: Path,
    worker_python_source: str,
    worker_returncode: int,
) -> ExternalAdaptRunSummary:
    try:
        return ExternalAdaptRunSummary(
            energy=float(result["energy"]),
            exact_energy=float(result["exact_energy"]) if result.get("exact_energy") is not None else None,
            initial_energy=float(result["initial_energy"]) if result.get("initial_energy") is not None else None,
            selected_operator_count=int(result["selected_operator_count"]),
            num_parameters=int(result["num_parameters"]),
            nfev=int(result["nfev"]) if result.get("nfev") is not None else None,
            ngev=int(result["ngev"]) if result.get("ngev") is not None else None,
            nit=int(result["nit"]) if result.get("nit") is not None else None,
            nfevs_by_iteration=_tuple_of_int_tuples(result.get("nfevs_by_iteration", ())),
            ngevs_by_iteration=_tuple_of_int_tuples(result.get("ngevs_by_iteration", ())),
            nits_by_iteration=_tuple_of_int_tuples(result.get("nits_by_iteration", ())),
            adapt_iterations=int(result["adapt_iterations"]),
            adapt_success=bool(result["adapt_success"]),
            adapt_stop_reason=str(result["adapt_stop_reason"]),
            pool_name=str(result.get("pool_name", "OVP_CEO")),
            pool_size=int(result["pool_size"]) if result.get("pool_size") is not None else None,
            selected_indices=_int_tuple(result.get("selected_indices", ())),
            coefficients=_float_tuple(result.get("coefficients", ())),
            gradient_norms=_float_tuple(result.get("gradient_norms", ())),
            selected_gradients=_float_tuple(result.get("selected_gradients", ())),
            energy_history=_float_tuple(result.get("energy_history", ())),
            adapt_history=_tuple_of_dicts(result.get("adapt_history", ())),
            raw_stdout_tail=str(result.get("raw_stdout_tail", "")),
            worker_mode=str(result.get("worker_mode", "ceo")),
            tetris_enabled=bool(result.get("tetris_enabled", False)),
            tetris_batching_enabled=bool(result.get("tetris_batching_enabled", False)),
            tetris_progressive_opt=bool(result.get("tetris_progressive_opt", False)),
            tetris_candidate_window=(
                str(result["tetris_candidate_window"]) if result.get("tetris_candidate_window") is not None else None
            ),
            tetris_screening_rule=(
                str(result["tetris_screening_rule"]) if result.get("tetris_screening_rule") is not None else None
            ),
            operators_added_per_iteration=_int_tuple(result.get("operators_added_per_iteration", ())),
            max_operators_added_per_iteration=(
                int(result["max_operators_added_per_iteration"])
                if result.get("max_operators_added_per_iteration") is not None
                else None
            ),
            batch_iterations=int(result["batch_iterations"]) if result.get("batch_iterations") is not None else None,
            selected_indices_by_iteration=_tuple_of_int_tuples(result.get("selected_indices_by_iteration", ())),
            coefficients_by_iteration=_tuple_of_float_tuples(result.get("coefficients_by_iteration", ())),
            worker_python=str(worker_python),
            worker_python_source=worker_python_source,
            worker_schema=_CEO_WORKER_SCHEMA,
            worker_returncode=int(worker_returncode),
            external_case_profile=(
                str(result["external_case_profile"]) if result.get("external_case_profile") is not None else None
            ),
            hubbard_x_dim=int(result["hubbard_x_dim"]) if result.get("hubbard_x_dim") is not None else None,
            hubbard_y_dim=int(result["hubbard_y_dim"]) if result.get("hubbard_y_dim") is not None else None,
            hubbard_t=float(result["hubbard_t"]) if result.get("hubbard_t") is not None else None,
            hubbard_u=float(result["hubbard_u"]) if result.get("hubbard_u") is not None else None,
            hubbard_periodic=bool(result["hubbard_periodic"]) if result.get("hubbard_periodic") is not None else None,
            hubbard_particle_hole_symmetry=(
                bool(result["hubbard_particle_hole_symmetry"])
                if result.get("hubbard_particle_hole_symmetry") is not None
                else None
            ),
            adapt_threshold=float(result["adapt_threshold"]) if result.get("adapt_threshold") is not None else None,
            adapt_max_adapt_iter=(
                int(result["adapt_max_adapt_iter"]) if result.get("adapt_max_adapt_iter") is not None else None
            ),
            adapt_max_opt_iter=(
                int(result["adapt_max_opt_iter"]) if result.get("adapt_max_opt_iter") is not None else None
            ),
        )
    except Exception as exc:
        raise ExternalAdaptWorkerFailed(
            f"CEO public-code worker returned invalid completed result JSON ({type(exc).__name__}: {exc})",
            exception_type="WorkerProtocolError",
        ) from exc


def _run_hubbard_l2_public_code(
    *,
    checkout_dir: Path,
    worker_mode: str,
    case_settings: ExternalAdaptHubbardCaseSettings,
) -> ExternalAdaptRunSummary:
    """Execute the pinned public external ADAPT code in an isolated Python subprocess."""

    mode = str(worker_mode).strip().lower()
    if mode not in {"ceo", "tetris"}:
        raise ExternalAdaptWorkerFailed(
            f"unsupported external ADAPT worker mode {worker_mode!r}",
            exception_type="WorkerProtocolError",
        )
    python, python_source = _resolve_external_adapt_python()
    worker = _worker_script_path()
    if not worker.exists():
        raise ExternalAdaptWorkerFailed(
            f"CEO public-code worker module is missing at {worker}",
            exception_type="WorkerProtocolError",
        )
    command = [
        str(python),
        str(worker),
        "--checkout-dir",
        str(checkout_dir),
        "--worker-mode",
        mode,
        *case_settings.to_worker_args(),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(checkout_dir),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_worker_environment(),
        )
    except OSError as exc:
        raise ExternalAdaptUnavailable(
            f"could not launch external ADAPT Python interpreter {python} from {python_source}: {exc}"
        ) from exc

    payload = _parse_worker_payload(completed)
    status = str(payload.get("status", ""))
    reason = str(payload.get("reason") or "")
    worker_context = f"mode={mode} interpreter={python} source={python_source} returncode={completed.returncode}"
    if status == "skipped_optional_dependency":
        raise ExternalAdaptUnavailable(f"{reason}; {worker_context}")
    if status == "failed":
        raise ExternalAdaptWorkerFailed(
            f"{reason}; {worker_context}",
            exception_type=str(payload.get("exception_type") or "ExternalWorkerError"),
        )
    if status != "completed":
        raise ExternalAdaptWorkerFailed(
            f"external ADAPT public-code worker returned unexpected status {status!r}; {worker_context}",
            exception_type="WorkerProtocolError",
        )
    if completed.returncode != 0:
        raise ExternalAdaptWorkerFailed(
            f"external ADAPT public-code worker completed JSON with non-zero return code; {worker_context}",
            exception_type="WorkerProtocolError",
        )
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise ExternalAdaptWorkerFailed(
            f"external ADAPT public-code worker completed without object result; {worker_context}",
            exception_type="WorkerProtocolError",
        )
    return _summary_from_worker_result(
        result,
        worker_python=python,
        worker_python_source=python_source,
        worker_returncode=int(completed.returncode),
    )


def _run_ceo_hubbard_l2_public_code(
    *,
    checkout_dir: Path,
    case_settings: ExternalAdaptHubbardCaseSettings,
) -> ExternalAdaptRunSummary:
    """Execute the pinned public CEO code path in an isolated Python subprocess."""

    return _run_hubbard_l2_public_code(checkout_dir=checkout_dir, worker_mode="ceo", case_settings=case_settings)


def _run_tetris_hubbard_l2_public_code(
    *,
    checkout_dir: Path,
    case_settings: ExternalAdaptHubbardCaseSettings,
) -> ExternalAdaptRunSummary:
    """Execute the pinned public TETRIS mode path in an isolated Python subprocess."""

    return _run_hubbard_l2_public_code(checkout_dir=checkout_dir, worker_mode="tetris", case_settings=case_settings)


def _source_fields(
    algorithm_id: str,
    *,
    checkout_dir: Path | None,
    resolved_commit: str | None,
) -> dict[str, Any]:
    return comparator_source_fields(
        str(algorithm_id),
        runner_module=_RUNNER_MODULE,
        external_reference_resolved_commit=resolved_commit,
        external_reference_cache_root=checkout_dir,
        external_reference_license_status=(
            "license_files_listed_in_external_reference_payload" if checkout_dir is not None else "not_checked_checkout_missing"
        ),
    )


def _reference_payload(*, checkout_dir: Path | None, resolved_commit: str | None) -> dict[str, Any]:
    spec = get_external_reference_spec(_CEO_REFERENCE_ID)
    return {
        "reference_id": _CEO_REFERENCE_ID,
        "display_name": spec.display_name,
        "url": spec.url,
        "clone_url": spec.clone_url,
        "checkout_dir": None if checkout_dir is None else str(checkout_dir),
        "required_commit": CEO_ADAPT_VQE_PINNED_COMMIT,
        "resolved_commit": resolved_commit,
        "license_files": [] if checkout_dir is None else list(_license_files(checkout_dir)),
        "reference_tier": spec.reference_tier,
        "availability": spec.availability,
    }


def _base_row(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    status: str,
    checkout_dir: Path | None,
    resolved_commit: str | None,
    reason: str = "",
) -> dict[str, Any]:
    promoted_tetris = _is_promoted_tetris_row(family=family, case_id=case_id, algorithm_id=algorithm_id)
    case_settings = _case_settings_for(family=family, case_id=case_id, algorithm_id=algorithm_id)
    worker_mode = _worker_mode_for_algorithm(algorithm_id) if _is_promoted_external_row(
        family=family,
        case_id=case_id,
        algorithm_id=algorithm_id,
    ) else None
    tetris_enabled = bool(promoted_tetris)
    return {
        "run_id": f"{case_id}::{algorithm_id}",
        "schema": SCHEMA_VERSION,
        "family": family,
        "problem": family,
        "L": case_settings.L if case_settings is not None else None,
        "hamiltonian_id": case_id,
        "case_id": case_id,
        "status": status,
        "method_id": algorithm_id,
        "method_kind": "adapt_selector_variant",
        "ansatz_name": (
            "public_code_ceo_adapt_vqe_ovp_ceo_tetris"
            if tetris_enabled
            else "public_code_ceo_adapt_vqe_ovp_ceo"
        ),
        "pool_name": "OVP_CEO",
        "algorithm_origin": (
            "external_public_code_ceo_adapt_vqe_tetris"
            if tetris_enabled
            else "external_public_code_ceo_adapt_vqe"
        ),
        "external_reference_id": _CEO_REFERENCE_ID,
        "external_reference_commit": resolved_commit,
        "external_reference_required_commit": CEO_ADAPT_VQE_PINNED_COMMIT,
        "external_checkout_dir": None if checkout_dir is None else str(checkout_dir),
        "reason": reason,
        "uses_exact_for_decision": False,
        "exact_reference_usage": "public_code_exact_energy_for_error_reporting_only_not_selection",
        "phase3_controller_called": False,
        "external_adapt_policy": "do_not_emulate_through_phase3_controller",
        **_source_fields(algorithm_id, checkout_dir=checkout_dir, resolved_commit=resolved_commit),
        "worker_mode": worker_mode,
        "tetris_enabled": tetris_enabled,
        "tetris_batching_enabled": tetris_enabled,
        "tetris_progressive_opt": False if tetris_enabled else None,
        "tetris_screening_rule": "disjoint_qubit_support_via_pool_get_qubits" if tetris_enabled else None,
        "pauli_ordering": "external OpenFermion/public-code ordering; energy invariant for Hubbard L2 reporting",
        "internal_pauli_alphabet": "external_openfermion_public_code_boundary",
        "external_case_profile": case_settings.case_profile if case_settings is not None else None,
        "hubbard_x_dim": case_settings.x_dim if case_settings is not None else None,
        "hubbard_y_dim": case_settings.y_dim if case_settings is not None else None,
        "hubbard_t": case_settings.t if case_settings is not None else None,
        "hubbard_u": case_settings.u if case_settings is not None else None,
        "hubbard_periodic": case_settings.periodic if case_settings is not None else None,
        "hubbard_particle_hole_symmetry": (
            case_settings.particle_hole_symmetry if case_settings is not None else None
        ),
        "adapt_threshold": case_settings.threshold if case_settings is not None else None,
        "adapt_max_adapt_iter": case_settings.max_adapt_iter if case_settings is not None else None,
        "adapt_max_opt_iter": case_settings.max_opt_iter if case_settings is not None else None,
    }


def _payload_common(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    status: str,
    row: Mapping[str, Any],
    checkout_dir: Path | None,
    resolved_commit: str | None,
    reason: str = "",
    exception_type: str | None = None,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    promoted_external = _is_promoted_external_row(family=family, case_id=case_id, algorithm_id=algorithm_id)
    dispatch = _dispatch_for_algorithm(algorithm_id) if promoted_external else "external_static_adapt_scaffold"
    guardrails = {
        "uses_exact_for_decision": False,
        "exact_reference_usage": row.get("exact_reference_usage"),
        "phase3_controller_called": False,
        "external_adapt_policy": "do_not_emulate_through_phase3_controller",
        "tetris_row_promoted": _is_promoted_tetris_row(family=family, case_id=case_id, algorithm_id=algorithm_id),
        "overlap_row_promoted": False,
    }
    source_fields = _source_fields(algorithm_id, checkout_dir=checkout_dir, resolved_commit=resolved_commit)
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "family": family,
        "case_id": case_id,
        "algorithm_id": algorithm_id,
        "method_id": algorithm_id,
        "status": status,
        "reason": reason,
        "exception_type": exception_type,
        "runner": "pipelines.exact_bench.external_adapt.external_static_adapt_benchmark.run_external_static_adapt_single",
        "dispatch": dispatch,
        "external_reference": _reference_payload(checkout_dir=checkout_dir, resolved_commit=resolved_commit),
        "metadata": external_algorithm_manifest_metadata(
            algorithm_id,
            status=status,
            dispatch=dispatch,
        ),
        "comparator_source": source_fields,
        "table_i": {
            "tex_label": "tab:benchmark_suite",
            "first_slice": str(case_id).strip() == "hubbard_L2",
            "sweep_complete": False,
        },
        "guardrails": guardrails,
        "rows": [dict(row)],
        "result": result if result is not None else dict(row),
        "finished_utc": _utc_now(),
    }
    return payload


def _write_normalized_artifacts(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(payload.get("rows", []))
    _write_json(output_dir / "result.json", payload)
    _write_json(output_dir / "rows.json", {"schema": f"{SCHEMA_VERSION}_rows", "rows": rows})
    _write_json(
        output_dir / "manifest.json",
        {"schema": f"{SCHEMA_VERSION}_manifest", **{k: v for k, v in payload.items() if k != "schema"}},
    )
    _write_json(output_dir / "generic_static_single.json", payload)
    if str(payload.get("status", "")).startswith("skipped"):
        _write_json(output_dir / "external_static_adapt_skip.json", payload)
    write_proxy_sidecars(
        rows,
        output_dir,
        summary_extras={
            "schema_source": SCHEMA_VERSION,
            "external_reference_id": _CEO_REFERENCE_ID,
            "external_reference_commit": payload.get("external_reference", {}).get("resolved_commit"),
        },
    )


def _skip_payload(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    output_dir: Path,
    status: str,
    reason: str,
    checkout_dir: Path | None = None,
    resolved_commit: str | None = None,
) -> dict[str, Any]:
    row = _base_row(
        family=family,
        case_id=case_id,
        algorithm_id=algorithm_id,
        status=status,
        checkout_dir=checkout_dir,
        resolved_commit=resolved_commit,
        reason=reason,
    )
    row.update({"adapt_stop_reason": reason})
    payload = _payload_common(
        family=family,
        case_id=case_id,
        algorithm_id=algorithm_id,
        status=status,
        row=row,
        checkout_dir=checkout_dir,
        resolved_commit=resolved_commit,
        reason=reason,
    )
    _write_normalized_artifacts(output_dir, payload)
    return payload


def _failure_payload(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    output_dir: Path,
    reason: str,
    exception_type: str,
    checkout_dir: Path | None = None,
    resolved_commit: str | None = None,
) -> dict[str, Any]:
    row = _base_row(
        family=family,
        case_id=case_id,
        algorithm_id=algorithm_id,
        status="failed",
        checkout_dir=checkout_dir,
        resolved_commit=resolved_commit,
        reason=reason,
    )
    row.update({"exception_type": exception_type, "adapt_stop_reason": "external_runtime_error"})
    payload = _payload_common(
        family=family,
        case_id=case_id,
        algorithm_id=algorithm_id,
        status="failed",
        row=row,
        checkout_dir=checkout_dir,
        resolved_commit=resolved_commit,
        reason=reason,
        exception_type=exception_type,
    )
    _write_normalized_artifacts(output_dir, payload)
    return payload


def _completed_payload(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    output_dir: Path,
    checkout_dir: Path,
    resolved_commit: str,
    summary: ExternalAdaptRunSummary,
    started_utc: str,
    runtime_s: float,
) -> dict[str, Any]:
    delta = None if summary.exact_energy is None else abs(float(summary.energy) - float(summary.exact_energy))
    row = _base_row(
        family=family,
        case_id=case_id,
        algorithm_id=algorithm_id,
        status="ok",
        checkout_dir=checkout_dir,
        resolved_commit=resolved_commit,
    )
    row.update(
        {
            "energy": float(summary.energy),
            "exact_energy": summary.exact_energy,
            "exact_gs_energy": summary.exact_energy,
            "initial_energy": summary.initial_energy,
            "delta_E_abs": delta,
            "abs_delta_e": delta,
            "num_parameters": int(summary.num_parameters),
            "selected_operator_count": int(summary.selected_operator_count),
            "adapt_depth_reached": int(summary.selected_operator_count),
            "adapt_iterations": int(summary.adapt_iterations),
            "adapt_success": bool(summary.adapt_success),
            "adapt_stop_reason": summary.adapt_stop_reason,
            "nfev": summary.nfev,
            "ngev": summary.ngev,
            "nit": summary.nit,
            "nfevs_by_iteration": [list(values) for values in summary.nfevs_by_iteration],
            "ngevs_by_iteration": [list(values) for values in summary.ngevs_by_iteration],
            "nits_by_iteration": [list(values) for values in summary.nits_by_iteration],
            "runtime_s": float(runtime_s),
            "started_utc": started_utc,
            "finished_utc": _utc_now(),
            "pool_name": summary.pool_name,
            "pool_size": summary.pool_size,
            "selected_indices": list(summary.selected_indices),
            "coefficients": list(summary.coefficients),
            "gradient_norms": list(summary.gradient_norms),
            "selected_gradients": list(summary.selected_gradients),
            "energy_history": list(summary.energy_history),
            "adapt_history": [dict(entry) for entry in summary.adapt_history],
            "selected_indices_by_iteration": [list(indices) for indices in summary.selected_indices_by_iteration],
            "coefficients_by_iteration": [list(coefficients) for coefficients in summary.coefficients_by_iteration],
            "operators_added_per_iteration": list(summary.operators_added_per_iteration),
            "max_operators_added_per_iteration": summary.max_operators_added_per_iteration,
            "batch_iterations": summary.batch_iterations,
            "worker_mode": summary.worker_mode,
            "tetris_enabled": bool(summary.tetris_enabled),
            "tetris_batching_enabled": bool(summary.tetris_batching_enabled),
            "tetris_progressive_opt": bool(summary.tetris_progressive_opt),
            "tetris_candidate_window": summary.tetris_candidate_window,
            "tetris_screening_rule": summary.tetris_screening_rule,
            "raw_stdout_tail": summary.raw_stdout_tail,
            "external_adapt_python": summary.worker_python or None,
            "external_adapt_python_source": summary.worker_python_source or None,
            "external_adapt_worker_schema": summary.worker_schema or None,
            "external_adapt_worker_returncode": summary.worker_returncode,
            "external_case_profile": summary.external_case_profile or row.get("external_case_profile"),
            "hubbard_x_dim": summary.hubbard_x_dim if summary.hubbard_x_dim is not None else row.get("hubbard_x_dim"),
            "hubbard_y_dim": summary.hubbard_y_dim if summary.hubbard_y_dim is not None else row.get("hubbard_y_dim"),
            "hubbard_t": summary.hubbard_t if summary.hubbard_t is not None else row.get("hubbard_t"),
            "hubbard_u": summary.hubbard_u if summary.hubbard_u is not None else row.get("hubbard_u"),
            "hubbard_periodic": (
                summary.hubbard_periodic if summary.hubbard_periodic is not None else row.get("hubbard_periodic")
            ),
            "hubbard_particle_hole_symmetry": (
                summary.hubbard_particle_hole_symmetry
                if summary.hubbard_particle_hole_symmetry is not None
                else row.get("hubbard_particle_hole_symmetry")
            ),
            "adapt_threshold": (
                summary.adapt_threshold if summary.adapt_threshold is not None else row.get("adapt_threshold")
            ),
            "adapt_max_adapt_iter": (
                summary.adapt_max_adapt_iter
                if summary.adapt_max_adapt_iter is not None
                else row.get("adapt_max_adapt_iter")
            ),
            "adapt_max_opt_iter": (
                summary.adapt_max_opt_iter if summary.adapt_max_opt_iter is not None else row.get("adapt_max_opt_iter")
            ),
        }
    )
    result = asdict(summary)
    result["delta_E_abs"] = delta
    payload = _payload_common(
        family=family,
        case_id=case_id,
        algorithm_id=algorithm_id,
        status="completed",
        row=row,
        checkout_dir=checkout_dir,
        resolved_commit=resolved_commit,
        result=result,
    )
    _write_normalized_artifacts(output_dir, payload)
    return payload


def run_external_static_adapt_single(
    *,
    family: str,
    case_id: str,
    algorithm_id: str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Run or explicitly skip one external ADAPT static benchmark row."""

    family_key = str(family).strip()
    case_key = str(case_id).strip()
    algorithm_key = str(algorithm_id).strip()
    output = Path(output_dir)

    if case_key not in default_external_static_adapt_case_ids(family_key, algorithm_key):
        return _skip_payload(
            family=family_key,
            case_id=case_key,
            algorithm_id=algorithm_key,
            output_dir=output,
            status="skipped_not_implemented",
            reason="external reference is cataloged, but this algorithm/family/case has no conformance-tested adapter",
        )
    case_settings = _case_settings_for(family=family_key, case_id=case_key, algorithm_id=algorithm_key)
    if case_settings is None:
        return _skip_payload(
            family=family_key,
            case_id=case_key,
            algorithm_id=algorithm_key,
            output_dir=output,
            status="skipped_not_implemented",
            reason="external reference case is listed, but no parameter settings payload is registered",
        )

    checkout: Path | None = None
    resolved_commit: str | None = None
    try:
        checkout, resolved_commit = _validate_ceo_checkout()
    except ExternalAdaptProvenanceMismatch as exc:
        return _skip_payload(
            family=family_key,
            case_id=case_key,
            algorithm_id=algorithm_key,
            output_dir=output,
            status="skipped_provenance_mismatch",
            reason=str(exc),
        )
    except ExternalAdaptUnavailable as exc:
        return _skip_payload(
            family=family_key,
            case_id=case_key,
            algorithm_id=algorithm_key,
            output_dir=output,
            status="skipped_optional_dependency",
            reason=str(exc),
        )

    started = _utc_now()
    t0 = time.perf_counter()
    try:
        if algorithm_key == _TETRIS_METHOD_ID:
            summary = _run_tetris_hubbard_l2_public_code(checkout_dir=checkout, case_settings=case_settings)
        else:
            summary = _run_ceo_hubbard_l2_public_code(checkout_dir=checkout, case_settings=case_settings)
    except ExternalAdaptUnavailable as exc:
        return _skip_payload(
            family=family_key,
            case_id=case_key,
            algorithm_id=algorithm_key,
            output_dir=output,
            status="skipped_optional_dependency",
            reason=str(exc),
            checkout_dir=checkout,
            resolved_commit=resolved_commit,
        )
    except ExternalAdaptWorkerFailed as exc:
        return _failure_payload(
            family=family_key,
            case_id=case_key,
            algorithm_id=algorithm_key,
            output_dir=output,
            reason=str(exc),
            exception_type=exc.exception_type,
            checkout_dir=checkout,
            resolved_commit=resolved_commit,
        )
    except Exception as exc:
        return _failure_payload(
            family=family_key,
            case_id=case_key,
            algorithm_id=algorithm_key,
            output_dir=output,
            reason=str(exc),
            exception_type=type(exc).__name__,
            checkout_dir=checkout,
            resolved_commit=resolved_commit,
        )
    return _completed_payload(
        family=family_key,
        case_id=case_key,
        algorithm_id=algorithm_key,
        output_dir=output,
        checkout_dir=checkout,
        resolved_commit=resolved_commit,
        summary=summary,
        started_utc=started,
        runtime_s=float(time.perf_counter() - t0),
    )


__all__ = [
    "ExternalAdaptProvenanceMismatch",
    "ExternalAdaptRunSummary",
    "ExternalAdaptUnavailable",
    "ExternalAdaptWorkerFailed",
    "SCHEMA_VERSION",
    "default_external_static_adapt_case_ids",
    "external_static_adapt_dispatch_for_algorithm",
    "run_external_static_adapt_single",
]
