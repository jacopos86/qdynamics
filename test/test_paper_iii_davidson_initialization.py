"""Deterministic, non-oracular initialization locks for direct Davidson."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.qse_spectra.davidson_initialization import (
    ORIGIN_COMPUTATIONAL_PROBE,
    ORIGIN_FIXED_RECORD,
    build_direct_davidson_start,
    target_sector_indices,
)
from pipelines.qse_spectra.siam_static_adapter import siam_physics_key
from src.quantum.siam_model import sector_indices as siam_sector_indices


def _hh_sector_indices(nq: int) -> tuple[int, ...]:
    return tuple(
        index
        for index in range(1 << nq)
        if sum((index >> q) & 1 for q in (0, 1)) == 1
        and sum((index >> q) & 1 for q in (2, 3)) == 1
    )


def test_hh_low_diagonal_probes_use_stable_energy_then_index_order() -> None:
    nq = 6
    sector = _hh_sector_indices(nq)
    diagonal = np.full(1 << nq, 100.0)
    diagonal[sector[0]] = -2.0
    diagonal[sector[1]] = -2.0
    diagonal[sector[2]] = -3.0
    fixed_seed = object()
    problem = SimpleNamespace(
        num_qubits=nq,
        n_ph_max=1,
        hamiltonian=np.diag(diagonal),
        basis=(fixed_seed,),
        resources=({"c_hat_2q": 5.0, "c_hat_d": 7.0, "c_hat_1q": 11.0},),
    )

    start = build_direct_davidson_start(
        problem,
        target_root_count=2,
        fixed_record_indices=(0,),
        computational_probe_count=3,
    )

    assert [seed.origin for seed in start.seed_directions] == [
        ORIGIN_FIXED_RECORD,
        ORIGIN_COMPUTATIONAL_PROBE,
        ORIGIN_COMPUTATIONAL_PROBE,
        ORIGIN_COMPUTATIONAL_PROBE,
    ]
    assert start.receipt["computational_probe_indices"] == [
        sector[2],
        sector[0],
        sector[1],
    ]
    assert start.seed_directions[0].resources == {
        "n2q": 5.0,
        "d2q": 7.0,
        "dc": 18.0,
    }
    for seed in start.seed_directions[1:]:
        index = int(seed.label.rsplit("::", 1)[1])
        assert seed.resources == {
            "n2q": 0.0,
            "d2q": 0.0,
            "dc": float(index.bit_count()),
        }


def test_siam_probes_stay_inside_the_declared_particle_sector() -> None:
    key = siam_physics_key("siam_nb2_fixture")
    dim = 1 << key.n_qubits
    problem = SimpleNamespace(
        key=key,
        num_qubits=key.n_qubits,
        hamiltonian=np.diag(np.arange(dim, dtype=float)),
        basis=(),
        resources=(),
    )

    assert target_sector_indices(problem) == siam_sector_indices(key.n_bath)
    start = build_direct_davidson_start(
        problem,
        target_root_count=6,
        fixed_record_indices=(),
    )

    probes = start.receipt["computational_probe_indices"]
    assert len(probes) == 7
    assert set(probes).issubset(set(siam_sector_indices(key.n_bath)))
    assert start.receipt["uses_exact_reference"] is False


def test_robust_start_breaks_the_hubbard_u8_invariant_subspace_ceiling(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipelines.exact_bench.paper_iii_growth_trace_campaign import (
        trace_adaptive_qse,
    )
    from pipelines.qse_spectra import paper_iii_problem as provider

    cache_key = ("hubbard_u8_seed_regression", 8.0, 0.0, 1, 6)
    provider._PROBLEM_CACHE.pop(cache_key, None)
    monkeypatch.setattr(provider, "REFERENCE_STORE", tmp_path / "reference_store")

    problem = provider.load_problem(
        regime="hubbard_u8_seed_regression",
        u=8.0,
        g_ep=0.0,
        n_ph_max=1,
        target_roots=6,
    )
    rows = trace_adaptive_qse(problem, k_max=64)

    assert rows[-1]["max_root_abs_error"] < 1.0e-10
    assert rows[-1]["seed_computational_probe_count"] >= 1
    assert rows[-1]["initialization_receipt"]["uses_exact_reference"] is False
    provider._PROBLEM_CACHE.pop(cache_key, None)


def test_direct_davidson_fails_before_silently_truncating_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pipelines.exact_bench import paper_iii_growth_trace_campaign as campaign

    problem = SimpleNamespace(
        basis=(),
        hamiltonian=np.eye(4),
        ground=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=complex),
        num_qubits=2,
        references=(1.0,),
    )
    start = SimpleNamespace(
        seed_directions=tuple(SimpleNamespace() for _ in range(4)),
    )
    monkeypatch.setattr(
        campaign, "build_direct_davidson_start", lambda *_a, **_k: start
    )

    with pytest.raises(ValueError, match="DAVIDSON_INITIALIZATION_CAP_INSUFFICIENT"):
        campaign.trace_adaptive_qse(problem, k_max=4, target_root_count=1)
