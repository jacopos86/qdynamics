from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.exact_bench.generic_static_benchmark import run_single


def test_run_single_threads_opt_in_powell_maxiter_cap_policy_only_to_generic_append(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pipelines.exact_bench.generic_static_adapt_variants as variants

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "schema": "generic_static_adapt_variants_v4",
            "status": "completed",
            "metadata": {},
            "rows": [{"status": "ok"}],
        }

    monkeypatch.setattr(variants, "run_generic_static_adapt_variant_single", _fake_runner)
    monkeypatch.setenv("GENERIC_STATIC_TABLE_GENERIC_ADAPT_STOP_POLICY", "fixed_horizon_no_target_v1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAX_DEPTH", "30")
    monkeypatch.setenv(
        "GENERIC_STATIC_TABLE_POWELL_MAXITER_CAP_POLICY",
        "accept_finite_nonincreasing_v1",
    )

    payload = run_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / "append_cap",
    )

    assert captured["powell_maxiter_cap_policy"] == "accept_finite_nonincreasing_v1"
    assert payload["powell_maxiter_cap_policy"] == "accept_finite_nonincreasing_v1"
    assert payload["metadata"]["powell_maxiter_cap_policy"] == "accept_finite_nonincreasing_v1"


def test_run_single_rejects_powell_cap_policy_on_wrong_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "GENERIC_STATIC_TABLE_POWELL_MAXITER_CAP_POLICY",
        "accept_finite_nonincreasing_v1",
    )

    with pytest.raises(ValueError, match="powell_maxiter_cap_policy env overlay is only valid"):
        run_single(
            family="hubbard",
            case_id="hubbard_L2_three_model_strong",
            algorithm_id="static_hea_qiskit_vqe",
            output_dir=tmp_path / "bad_cap_dispatch",
        )
