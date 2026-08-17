from __future__ import annotations

import copy

import pytest

from pipelines.static_adapt.ra_adapt.numerical_runtime import (
    CANONICAL_NUMERICAL_THREAD_ENVIRONMENT,
    NumericalRuntimeContractError,
    OBSERVED_CONTAINER_IMAGE_SHA256_ENV,
    assert_numerical_runtime_parity,
    build_numerical_runtime_contract,
    build_numerical_runtime_receipt,
)


IMAGE_SHA256 = "a" * 64


def _contract() -> dict[str, object]:
    return build_numerical_runtime_contract(
        container_image_sha256=IMAGE_SHA256,
        request_cpus=4,
    )


def _observed_runtime() -> dict[str, object]:
    return {
        "container_image_sha256": IMAGE_SHA256,
        "python": {
            "implementation": "CPython",
            "version": "3.12.4",
            "executable": "/usr/bin/python3",
        },
        "dependencies": {
            "numpy": "2.3.5",
            "scipy": "1.16.3",
            "qiskit": "2.3.1",
        },
        "numpy_configuration_sha256": "b" * 64,
        "platform": {
            "system": "Linux",
            "machine": "x86_64",
            "libc": "glibc-2.39",
        },
        "cpu": {
            "allocated_cpus": 4,
            "affinity_cpus": 4,
            "model_name": "test-cpu",
            "flags_sha256": "c" * 64,
        },
        "thread_environment": dict(
            CANONICAL_NUMERICAL_THREAD_ENVIRONMENT
        ),
        "threadpools": [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "prefix": "libopenblas",
                "filepath": "/usr/lib/libopenblas.so",
                "version": "0.3.29",
                "threading_layer": "pthreads",
                "architecture": "SkylakeX",
                "num_threads": 1,
            }
        ],
    }


def test_contract_closes_image_cpu_device_and_thread_policy() -> None:
    contract = _contract()

    assert contract["container_image_sha256"] == IMAGE_SHA256
    assert contract["request_cpus"] == 4
    assert contract["request_gpus"] == 0
    assert contract["execution_device"] == "cpu"
    assert contract["clean_environment"] is True
    assert contract["thread_environment"] == (
        CANONICAL_NUMERICAL_THREAD_ENVIRONMENT
    )
    assert contract["runtime_receipt_required"] is True
    assert contract["pairing_scope"] == (
        "matched_pair_exact_runtime_receipts_v1"
    )
    assert contract[
        "observed_container_image_sha256_environment_variable"
    ] == OBSERVED_CONTAINER_IMAGE_SHA256_ENV
    assert contract["sha256"]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("container_image_sha256",), "d" * 64, "image"),
        (("cpu", "allocated_cpus"), 1, "CPU allocation"),
        (("thread_environment", "OPENBLAS_NUM_THREADS"), "4", "thread"),
        (("threadpools", 0, "num_threads"), 4, "threadpool"),
        (("threadpools", 0, "version"), None, "version"),
        (("cpu", "model_name"), "x86_64", "CPU model"),
        (
            ("cpu", "flags_sha256"),
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855",
            "feature flags",
        ),
        (("platform", "libc"), "-", "platform"),
    ],
)
def test_runtime_receipt_fails_closed_on_numerical_stack_drift(
    path: tuple[object, ...],
    value: object,
    message: str,
) -> None:
    observed = copy.deepcopy(_observed_runtime())
    cursor: object = observed
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]

    with pytest.raises(NumericalRuntimeContractError, match=message):
        build_numerical_runtime_receipt(
            _contract(), method="append_adapt", observed=observed
        )


def test_append_and_ra_receipts_must_have_one_exact_runtime_fingerprint() -> None:
    append = build_numerical_runtime_receipt(
        _contract(), method="append_adapt", observed=_observed_runtime()
    )
    ra = build_numerical_runtime_receipt(
        _contract(), method="ra_adapt", observed=_observed_runtime()
    )

    parity = assert_numerical_runtime_parity(
        {"append_adapt": append, "ra_adapt": ra},
        contract=_contract(),
    )
    assert parity["status"] == "passed"
    assert parity["runtime_fingerprint_sha256"] == append[
        "runtime_fingerprint_sha256"
    ]

    drifted = copy.deepcopy(ra)
    drifted["runtime_fingerprint_sha256"] = "e" * 64
    with pytest.raises(
        NumericalRuntimeContractError, match="runtime fingerprint"
    ):
        assert_numerical_runtime_parity(
            {"append_adapt": append, "ra_adapt": drifted},
            contract=_contract(),
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("dependencies", "numpy"), "2.3.6"),
        (("cpu", "model_name"), "other-test-cpu"),
        (("cpu", "flags_sha256"), "d" * 64),
        (("threadpools", 0, "version"), "0.3.30"),
    ],
)
def test_parity_rejects_self_consistent_runtime_identity_drift(
    path: tuple[object, ...],
    value: object,
) -> None:
    append = build_numerical_runtime_receipt(
        _contract(), method="append_adapt", observed=_observed_runtime()
    )
    ra_runtime = copy.deepcopy(_observed_runtime())
    cursor: object = ra_runtime
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    ra = build_numerical_runtime_receipt(
        _contract(), method="ra_adapt", observed=ra_runtime
    )

    with pytest.raises(
        NumericalRuntimeContractError, match="runtime fingerprints"
    ):
        assert_numerical_runtime_parity(
            {"append_adapt": append, "ra_adapt": ra},
            contract=_contract(),
        )


def test_threadpool_observation_requires_a_real_blas_identity() -> None:
    observed = _observed_runtime()
    observed["threadpools"] = [
        {
            "user_api": "openmp",
            "internal_api": "openmp",
            "prefix": "libgomp",
            "filepath": "/usr/lib/libgomp.so",
            "version": None,
            "threading_layer": None,
            "architecture": None,
            "num_threads": 1,
        }
    ]

    with pytest.raises(
        NumericalRuntimeContractError, match="No loaded BLAS"
    ):
        build_numerical_runtime_receipt(
            _contract(), method="append_adapt", observed=observed
        )


def test_threadpool_order_does_not_change_runtime_fingerprint() -> None:
    first = _observed_runtime()
    first["threadpools"].append(  # type: ignore[union-attr]
        {
            "user_api": "openmp",
            "internal_api": "openmp",
            "prefix": "libgomp",
            "filepath": "/usr/lib/libgomp.so",
            "version": None,
            "threading_layer": None,
            "architecture": None,
            "num_threads": 1,
        }
    )
    second = copy.deepcopy(first)
    second["threadpools"].reverse()  # type: ignore[union-attr]

    first_receipt = build_numerical_runtime_receipt(
        _contract(), method="append_adapt", observed=first
    )
    second_receipt = build_numerical_runtime_receipt(
        _contract(), method="append_adapt", observed=second
    )
    assert first_receipt["runtime_fingerprint_sha256"] == second_receipt[
        "runtime_fingerprint_sha256"
    ]
