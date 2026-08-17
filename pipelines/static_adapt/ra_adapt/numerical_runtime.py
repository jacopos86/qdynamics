"""Fail-closed numerical-runtime identity for matched Append/RA evidence.

The container image is necessary but not sufficient for a floating-point
comparison.  This module closes the remaining execution inputs that can alter
the realized route: CPU allocation, numerical-library thread policy, Python
and dependency versions, BLAS implementation, and host CPU features.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from importlib import metadata
import io
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

from pipelines.static_adapt.ra_adapt.contracts import canonical_sha256


NUMERICAL_RUNTIME_CONTRACT_SCHEMA = (
    "paper_i_append_ra_numerical_runtime_contract_v1"
)
NUMERICAL_RUNTIME_RECEIPT_SCHEMA = (
    "paper_i_append_ra_numerical_runtime_receipt_v1"
)
NUMERICAL_RUNTIME_PARITY_SCHEMA = (
    "paper_i_append_ra_numerical_runtime_parity_v1"
)

CANONICAL_NUMERICAL_THREAD_ENVIRONMENT: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "PYTHONHASHSEED": "0",
}

OBSERVED_CONTAINER_IMAGE_SHA256_ENV = (
    "STATIC_ADAPT_CONTAINER_IMAGE_SHA256"
)

_REQUIRED_DEPENDENCIES = ("numpy", "scipy", "qiskit")
_PAIR_METHODS = ("append_adapt", "ra_adapt")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class NumericalRuntimeContractError(RuntimeError):
    """Raised when a numerical runtime is missing or differs across methods."""


def _digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise NumericalRuntimeContractError(f"{label} must be a SHA-256 digest.")
    return text


def build_numerical_runtime_contract(
    *,
    container_image_sha256: str,
    request_cpus: int,
) -> dict[str, Any]:
    """Build the one runtime contract shared by a matched Append/RA pair."""

    cpus = int(request_cpus)
    if cpus < 1:
        raise NumericalRuntimeContractError("request_cpus must be positive.")
    return _digested(
        {
            "schema": NUMERICAL_RUNTIME_CONTRACT_SCHEMA,
            "container_image_sha256": _require_sha256(
                container_image_sha256,
                label="container image",
            ),
            "container_entry": "apptainer_exec_cleanenv_v1",
            "clean_environment": True,
            "execution_device": "cpu",
            "request_cpus": cpus,
            "request_gpus": 0,
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "thread_environment": dict(
                CANONICAL_NUMERICAL_THREAD_ENVIRONMENT
            ),
            "observed_container_image_sha256_environment_variable": (
                OBSERVED_CONTAINER_IMAGE_SHA256_ENV
            ),
            "runtime_receipt_required": True,
            "pairing_scope": "matched_pair_exact_runtime_receipts_v1",
            "comparison_policy": "reject_before_comparison_on_runtime_drift_v1",
        }
    )


def normalize_numerical_runtime_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NumericalRuntimeContractError(
            "numerical runtime contract must be a mapping."
        )
    supplied_sha = value.get("sha256")
    try:
        request_cpus = int(value.get("request_cpus", 0))
    except (TypeError, ValueError) as exc:
        raise NumericalRuntimeContractError(
            "request_cpus must be a positive integer."
        ) from exc
    normalized = build_numerical_runtime_contract(
        container_image_sha256=str(value.get("container_image_sha256", "")),
        request_cpus=request_cpus,
    )
    expected_fields = dict(normalized)
    if dict(value) != expected_fields:
        raise NumericalRuntimeContractError(
            "numerical runtime contract fields drifted from the canonical contract."
        )
    if supplied_sha != normalized["sha256"]:
        raise NumericalRuntimeContractError(
            "numerical runtime contract digest drifted."
        )
    return normalized


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise NumericalRuntimeContractError(f"{label} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise NumericalRuntimeContractError(
            f"{label} must be a positive integer."
        ) from exc
    if result < 1:
        raise NumericalRuntimeContractError(f"{label} must be a positive integer.")
    return result


def _normalize_observed_runtime(
    observed: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(observed, Mapping):
        raise NumericalRuntimeContractError(
            "observed numerical runtime must be a mapping."
        )
    image_sha = _require_sha256(
        observed.get("container_image_sha256"),
        label="observed container image",
    )
    if image_sha != contract["container_image_sha256"]:
        raise NumericalRuntimeContractError(
            "container image differs from the paired numerical runtime contract."
        )

    thread_environment = observed.get("thread_environment")
    if thread_environment != contract["thread_environment"]:
        raise NumericalRuntimeContractError(
            "numerical thread environment differs from the paired contract."
        )

    cpu = observed.get("cpu")
    if not isinstance(cpu, Mapping):
        raise NumericalRuntimeContractError("CPU runtime receipt is missing.")
    allocated_cpus = _positive_int(
        cpu.get("allocated_cpus"), label="CPU allocation"
    )
    if allocated_cpus != int(contract["request_cpus"]):
        raise NumericalRuntimeContractError(
            "CPU allocation differs from the paired numerical runtime contract."
        )
    affinity_cpus = _positive_int(
        cpu.get("affinity_cpus"), label="CPU affinity count"
    )
    if affinity_cpus < allocated_cpus:
        raise NumericalRuntimeContractError(
            "CPU affinity count is smaller than the allocated CPU count."
        )
    model_name = str(cpu.get("model_name") or "").strip()
    if model_name.lower() in {"", "unavailable", "unknown", "x86_64", "amd64"}:
        raise NumericalRuntimeContractError(
            "CPU model identity is missing or generic."
        )
    flags_sha = _require_sha256(
        cpu.get("flags_sha256"), label="CPU feature flags"
    )
    if flags_sha == _EMPTY_SHA256:
        raise NumericalRuntimeContractError(
            "CPU feature flags are missing."
        )

    python = observed.get("python")
    if not isinstance(python, Mapping) or any(
        not str(python.get(field) or "").strip()
        for field in ("implementation", "version", "executable")
    ):
        raise NumericalRuntimeContractError("Python runtime identity is incomplete.")
    dependencies = observed.get("dependencies")
    if not isinstance(dependencies, Mapping) or any(
        not str(dependencies.get(name) or "").strip()
        for name in _REQUIRED_DEPENDENCIES
    ):
        raise NumericalRuntimeContractError(
            "NumPy/SciPy/Qiskit dependency identity is incomplete."
        )
    numpy_configuration_sha256 = _require_sha256(
        observed.get("numpy_configuration_sha256"),
        label="NumPy configuration",
    )

    platform_receipt = observed.get("platform")
    if not isinstance(platform_receipt, Mapping):
        raise NumericalRuntimeContractError("platform runtime identity is missing.")
    libc = str(platform_receipt.get("libc") or "").strip()
    if (
        str(platform_receipt.get("system", "")) != contract["platform_system"]
        or str(platform_receipt.get("machine", ""))
        != contract["platform_machine"]
        or not libc
        or libc in {"-", "unknown", "unavailable"}
        or libc.endswith("-")
    ):
        raise NumericalRuntimeContractError(
            "platform architecture differs from the paired numerical runtime contract."
        )

    threadpools = observed.get("threadpools")
    if not isinstance(threadpools, list) or not threadpools:
        raise NumericalRuntimeContractError(
            "BLAS/LAPACK threadpool identity is missing."
        )
    normalized_threadpools: list[dict[str, Any]] = []
    observed_blas = False
    for row in threadpools:
        if not isinstance(row, Mapping):
            raise NumericalRuntimeContractError(
                "BLAS/LAPACK threadpool receipt is malformed."
            )
        num_threads = _positive_int(
            row.get("num_threads"), label="threadpool thread count"
        )
        if num_threads != 1:
            raise NumericalRuntimeContractError(
                "threadpool thread count is not pinned to one."
            )
        user_api = str(row.get("user_api") or "").strip()
        internal_api = str(row.get("internal_api") or "").strip()
        filepath = str(row.get("filepath") or "").strip()
        if not user_api or not internal_api or not filepath:
            raise NumericalRuntimeContractError(
                "BLAS/LAPACK threadpool identity is incomplete."
            )
        if user_api == "blas":
            observed_blas = True
            if not str(row.get("version") or "").strip():
                raise NumericalRuntimeContractError(
                    "BLAS implementation version is missing."
                )
        normalized_threadpools.append(
            {
                str(key): row.get(key)
                for key in (
                    "user_api",
                    "internal_api",
                    "prefix",
                    "filepath",
                    "version",
                    "threading_layer",
                    "architecture",
                    "num_threads",
                )
            }
        )
    if not observed_blas:
        raise NumericalRuntimeContractError(
            "No loaded BLAS implementation was observed."
        )
    normalized_threadpools.sort(
        key=lambda row: (
            str(row.get("internal_api")),
            str(row.get("filepath")),
        )
    )

    return {
        "container_image_sha256": image_sha,
        "python": {str(key): python.get(key) for key in sorted(python)},
        "dependencies": {
            name: str(dependencies[name]) for name in _REQUIRED_DEPENDENCIES
        },
        "numpy_configuration_sha256": numpy_configuration_sha256,
        "platform": {
            "system": str(platform_receipt["system"]),
            "machine": str(platform_receipt["machine"]),
            "libc": libc,
        },
        "cpu": {
            "allocated_cpus": allocated_cpus,
            "affinity_cpus": affinity_cpus,
            "model_name": model_name,
            "flags_sha256": flags_sha,
        },
        "thread_environment": dict(thread_environment),
        "threadpools": normalized_threadpools,
    }


def _read_cpu_identity() -> tuple[str, str]:
    cpuinfo = Path("/proc/cpuinfo")
    model_name = ""
    flags: set[str] = set()
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            key, separator, raw_value = line.partition(":")
            if not separator:
                continue
            if key.strip() in {"model name", "Hardware"} and not model_name:
                model_name = raw_value.strip()
            if key.strip() in {"flags", "Features"}:
                flags.update(raw_value.split())
    if not model_name:
        model_name = (
            platform.processor().strip()
            or platform.machine().strip()
            or "unavailable"
        )
    flags_payload = " ".join(sorted(flags)).encode("utf-8")
    return model_name, hashlib.sha256(flags_payload).hexdigest()


def _allocated_cpu_count() -> int:
    job_ad = str(os.environ.get("_CONDOR_JOB_AD", "")).strip()
    if job_ad and Path(job_ad).is_file():
        for line in Path(job_ad).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            key, separator, raw_value = line.partition("=")
            if separator and key.strip() == "RequestCpus":
                raw = raw_value.strip()
                if raw.isdecimal() and int(raw) > 0:
                    return int(raw)
                raise NumericalRuntimeContractError(
                    "Condor RequestCpus is not a positive integer."
                )
    for name in (
        "STATIC_ADAPT_ALLOCATED_CPUS",
        "CONDOR_REQUEST_CPUS",
        "REQUEST_CPUS",
    ):
        raw = str(os.environ.get(name, "")).strip()
        if raw.isdecimal() and int(raw) > 0:
            return int(raw)
        if raw:
            raise NumericalRuntimeContractError(
                f"{name} is not a positive integer."
            )
    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        return max(1, len(affinity(0)))
    return int(os.cpu_count() or 1)


def capture_numerical_runtime(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture the runtime after entering the pinned container."""

    normalized_contract = normalize_numerical_runtime_contract(contract)
    try:
        import numpy as np
        import scipy.linalg as scipy_linalg
        from threadpoolctl import threadpool_info
    except ImportError as exc:
        raise NumericalRuntimeContractError(
            "NumPy and threadpoolctl are required for the runtime receipt."
        ) from exc

    # Force numerical libraries to load before observing threadpool state.
    np.dot(np.ones((2, 2), dtype=float), np.ones((2, 2), dtype=float))
    scipy_linalg.eigh(np.eye(2, dtype=float), check_finite=False)
    config_stream = io.StringIO()
    with redirect_stdout(config_stream):
        np.show_config()
    numpy_configuration_sha256 = hashlib.sha256(
        config_stream.getvalue().encode("utf-8")
    ).hexdigest()
    libc_name, libc_version = platform.libc_ver()
    model_name, flags_sha256 = _read_cpu_identity()
    affinity = getattr(os, "sched_getaffinity", None)
    affinity_cpus = (
        len(affinity(0))
        if callable(affinity)
        else int(os.cpu_count() or 1)
    )
    observed = {
        "container_image_sha256": os.environ.get(
            OBSERVED_CONTAINER_IMAGE_SHA256_ENV
        ),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "dependencies": {
            name: metadata.version(name) for name in _REQUIRED_DEPENDENCIES
        },
        "numpy_configuration_sha256": numpy_configuration_sha256,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "libc": f"{libc_name}-{libc_version}",
        },
        "cpu": {
            "allocated_cpus": _allocated_cpu_count(),
            "affinity_cpus": int(affinity_cpus),
            "model_name": model_name,
            "flags_sha256": flags_sha256,
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in CANONICAL_NUMERICAL_THREAD_ENVIRONMENT
        },
        "threadpools": threadpool_info(),
    }
    return _normalize_observed_runtime(
        observed, contract=normalized_contract
    )


def build_numerical_runtime_receipt(
    contract: Mapping[str, Any],
    *,
    method: str,
    observed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if method not in _PAIR_METHODS:
        raise NumericalRuntimeContractError(
            "numerical runtime receipt method must be append_adapt or "
            "ra_adapt."
        )
    normalized_contract = normalize_numerical_runtime_contract(contract)
    normalized_observed = (
        capture_numerical_runtime(normalized_contract)
        if observed is None
        else _normalize_observed_runtime(
            observed, contract=normalized_contract
        )
    )
    fingerprint_sha256 = canonical_sha256(normalized_observed)
    return _digested(
        {
            "schema": NUMERICAL_RUNTIME_RECEIPT_SCHEMA,
            "method": method,
            "contract_sha256": normalized_contract["sha256"],
            "runtime_fingerprint_sha256": fingerprint_sha256,
            "runtime": normalized_observed,
            "status": "passed",
        }
    )


def _normalize_runtime_receipt(
    value: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    expected_method: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NumericalRuntimeContractError(
            "numerical runtime receipt must be a mapping."
        )
    if value.get("schema") != NUMERICAL_RUNTIME_RECEIPT_SCHEMA:
        raise NumericalRuntimeContractError(
            "numerical runtime receipt schema drifted."
        )
    if value.get("method") != expected_method:
        raise NumericalRuntimeContractError(
            "numerical runtime receipt method drifted."
        )
    if value.get("contract_sha256") != contract["sha256"]:
        raise NumericalRuntimeContractError(
            "numerical runtime receipt contract differs from the expected "
            "paired contract."
        )
    runtime = value.get("runtime")
    if not isinstance(runtime, Mapping):
        raise NumericalRuntimeContractError(
            "numerical runtime receipt has no runtime fingerprint payload."
        )
    normalized_runtime = _normalize_observed_runtime(
        runtime, contract=contract
    )
    expected_fingerprint = canonical_sha256(normalized_runtime)
    if value.get("runtime_fingerprint_sha256") != expected_fingerprint:
        raise NumericalRuntimeContractError(
            "runtime fingerprint digest drifted."
        )
    expected_receipt = _digested(
        {
            "schema": NUMERICAL_RUNTIME_RECEIPT_SCHEMA,
            "method": expected_method,
            "contract_sha256": _require_sha256(
                value.get("contract_sha256"), label="runtime contract"
            ),
            "runtime_fingerprint_sha256": expected_fingerprint,
            "runtime": normalized_runtime,
            "status": "passed",
        }
    )
    if dict(value) != expected_receipt:
        raise NumericalRuntimeContractError(
            "numerical runtime receipt digest or fields drifted."
        )
    return expected_receipt


def assert_numerical_runtime_parity(
    receipts: Mapping[str, Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Require Append and RA to share one exact observed runtime."""

    if set(receipts) != set(_PAIR_METHODS):
        raise NumericalRuntimeContractError(
            "runtime parity requires exactly append_adapt and ra_adapt receipts."
        )
    normalized_contract = normalize_numerical_runtime_contract(contract)
    normalized = {
        method: _normalize_runtime_receipt(
            receipts[method],
            contract=normalized_contract,
            expected_method=method,
        )
        for method in _PAIR_METHODS
    }
    fingerprints = {
        receipt["runtime_fingerprint_sha256"]
        for receipt in normalized.values()
    }
    if len(fingerprints) != 1:
        raise NumericalRuntimeContractError(
            "Append and RA runtime fingerprints differ."
        )
    return _digested(
        {
            "schema": NUMERICAL_RUNTIME_PARITY_SCHEMA,
            "status": "passed",
            "methods": list(_PAIR_METHODS),
            "contract_sha256": normalized_contract["sha256"],
            "runtime_fingerprint_sha256": next(iter(fingerprints)),
        }
    )
