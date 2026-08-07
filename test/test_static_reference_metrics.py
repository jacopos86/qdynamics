from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipelines.exact_bench import static_reference_metrics as metrics
from pipelines.exact_bench.table_i_canonical_cases import (
    TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE,
    table_i_executable_specs,
)


def test_odd_l_exact_reference_uses_canonical_half_filled_sector(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(metrics, "build_problem_hamiltonian", lambda **_kwargs: object())

    def _fake_exact(_hamiltonian, **kwargs):
        captured.update(kwargs)
        return -2.5

    monkeypatch.setattr(metrics, "_exact_gs_energy_for_problem", _fake_exact)
    metrics._exact_energy_cached.cache_clear()
    spec = SimpleNamespace(
        benchmark_id="hubbard_L3_scaling_weak",
        family="hubbard",
        base_pipeline_args=(
            "--problem",
            "hubbard",
            "--L",
            "3",
            "--t",
            "1",
            "--u",
            "0.25",
            "--n-ph-max",
            "1",
        ),
    )

    energy, _key_hash, _key = metrics.exact_energy_for_spec(spec, n_ph_max=1)

    assert energy == -2.5
    assert captured["num_particles"] == (2, 1)
    assert _key["num_particles"] == (2, 1)
    assert _key["num_particles_source"].endswith("half_filled_num_particles")
    metrics._exact_energy_cached.cache_clear()


@pytest.mark.parametrize(
    ("case_id", "expected_energy"),
    (
        ("hubbard_L3_scaling_weak", -2.6758452676191875),
        ("hubbard_L3_scaling_strong", -0.7077115708616566),
    ),
)
def test_hubbard_l3_scaling_exact_energy_matches_run_sector(case_id: str, expected_energy: float) -> None:
    spec = next(
        item
        for item in table_i_executable_specs(TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE)
        if item.benchmark_id == case_id
    )

    energy, _key_hash, _key = metrics.exact_energy_for_spec(spec, n_ph_max=1)

    assert energy == pytest.approx(expected_energy, abs=1.0e-12)


@pytest.mark.parametrize(
    ("case_id", "expected_policy"),
    (
        ("spin_boson_L3_nph3_scaling_weak", "single_emitter_truncated_boson_register"),
        ("bose_hubbard_L3_nph3_scaling_weak", "unrestricted_truncated_boson_register"),
    ),
)
def test_pure_boson_exact_key_does_not_claim_fermion_particles(
    case_id: str,
    expected_policy: str,
) -> None:
    spec = next(
        item
        for item in table_i_executable_specs(TABLE_I_PAPER_I_SCALING_MATRIX_PROFILE)
        if item.benchmark_id == case_id
    )

    key = metrics.reference_energy_key(spec, n_ph_max=int(spec.exact_reference_n_ph_max or 1))

    assert key["num_particles"] is None
    assert key["num_particles_source"] is None
    assert key["exact_sector_policy"] == expected_policy
