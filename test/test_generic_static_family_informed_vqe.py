#!/usr/bin/env python3
"""Tests for the generic family-informed fixed VQE benchmark row."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.exact_bench import generic_static_family_informed_vqe as family_vqe
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
from pipelines.exact_bench.table_i_canonical_cases import TABLE_I_CANONICAL_CASE_IDS_BY_FAMILY
from pipelines.exact_bench.generic_static_adapt_variants import build_full_meta_candidate_pool
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


class _TinyLayout:
    def __init__(self, total_qubits: int = 2) -> None:
        self.total_qubits = int(total_qubits)
        self.fermion_qubits = int(total_qubits)

    def block(self, name: str):  # noqa: ANN201
        if name in {"fermion", "full"}:
            return SimpleNamespace(start_qubit=0, stop_qubit=self.total_qubits)
        return None


def _fake_spec() -> SimpleNamespace:
    return SimpleNamespace(
        benchmark_id="hubbard_L2",
        family="hubbard",
        base_pipeline_args=("--problem", "hubbard", "--L", "2"),
        split="train",
        tags=(),
        features=None,
    )


def _poly(labels: tuple[str, ...] = ("xy", "yx")) -> PauliPolynomial:
    return PauliPolynomial("JW", [PauliTerm(len(labels[0]), ps=label, pc=0.5) for label in labels])


def _candidate(label: str, labels: tuple[str, ...] = ("xy", "yx")) -> SimpleNamespace:
    return SimpleNamespace(
        label=label,
        polynomial=_poly(labels),
        pauli_labels_exyz=labels,
        support=tuple(i for i, ch in enumerate(reversed(labels[0])) if ch != "e"),
    )


def _fake_context(events: list[str], *, num_qubits: int = 2) -> SimpleNamespace:
    xx = ("e" * max(0, int(num_qubits) - 2)) + "xx"
    yy = ("e" * max(0, int(num_qubits) - 2)) + "yy"
    hamiltonian = PauliPolynomial(
        "JW",
        [PauliTerm(num_qubits, ps=xx, pc=0.5), PauliTerm(num_qubits, ps=yy, pc=0.5)],
    )

    def _resolve_energy(ai_log=None):  # noqa: ANN001
        assert events == ["optimizer"]
        events.append("exact")
        return -1.0

    return SimpleNamespace(
        request=SimpleNamespace(num_sites=2, ordering="blocked"),
        layout=_TinyLayout(num_qubits),
        hamiltonian=hamiltonian,
        reference_state=SimpleNamespace(build_state=lambda: np.eye(1 << int(num_qubits), dtype=complex)[1]),
        exact_target=SimpleNamespace(resolve_energy=_resolve_energy),
        sector=SimpleNamespace(constraints=()),
    )


def test_default_static_family_informed_vqe_case_ids_cover_table_i_canonical_suite() -> None:
    for family, case_ids in TABLE_I_CANONICAL_CASE_IDS_BY_FAMILY.items():
        assert family_vqe.default_static_family_informed_vqe_case_ids(family) == tuple(case_ids)

def test_policy_matcher_allows_only_declared_label_classes() -> None:
    assert family_vqe._family_informed_policy_match("hubbard", "uccsd_dbl(ab:0,2->1,3)").policy_class == "uccsd_dbl"
    assert family_vqe._family_informed_policy_match("hubbard", "hop(i=0,j=1,spin=0)") is None
    assert family_vqe._family_informed_policy_match("spinless_tv", "ham_quad::hop_nn(0,1)").policy_class == "spinless_hamiltonian_quadrature"
    assert family_vqe._family_informed_policy_match("spinless_tv", "ham_term(xx)") is None
    assert family_vqe._family_informed_policy_match("bose_hubbard", "full_meta::current_0_1").policy_class == "bosonic_full_meta_generator"
    assert family_vqe._family_informed_policy_match("bose_hubbard", "ham_term(xx)") is None
    assert family_vqe._family_informed_policy_match("bose_hubbard", "full_meta::ham_term(xx)") is None
    assert family_vqe._family_informed_policy_match("bose_hubbard", "full_meta::ham_unit_term(xx)") is None
    assert family_vqe._family_informed_policy_match("bose_hubbard", "full_meta::generic_xy") is None
    assert family_vqe._family_informed_policy_match("spin_boson", "full_meta::boson_displacement").policy_class == "spin_boson_full_meta_generator"
    assert family_vqe._family_informed_policy_match("spin_boson", "full_meta::ham_term(xx)") is None
    assert family_vqe._family_informed_policy_match("spin_boson", "full_meta::generic_xy") is None
    assert family_vqe._family_informed_policy_match("hh", "uccsd_ferm_lifted::uccsd_sing(alpha:0->1)").policy_class == "uccsd_sing"
    assert family_vqe._family_informed_policy_match("hh", "hh_termwise_ham_unit_term(eeee)") is None
    assert family_vqe._family_informed_policy_match("hh", "paop_unknown") is None


def test_policy_selection_rejects_generic_fallback_labels() -> None:
    pool = (
        _candidate("ham_full"),
        _candidate("ham_term(xx)"),
        _candidate("generic_xy"),
        _candidate("hop(i=0,j=1,spin=0)"),
    )
    selection = family_vqe._select_family_informed_candidates("hubbard", pool, max_terms=4)

    assert selection.selected == ()
    assert selection.approved_pool_count == 0
    assert selection.rejected_pool_count == 3
    assert selection.dropped_ham_full_count == 1


def test_actual_full_meta_canonical_selection_is_policy_approved_for_all_table_i_families() -> None:
    for family, case_ids in TABLE_I_CANONICAL_CASE_IDS_BY_FAMILY.items():
        for case_id in case_ids:
            context = family_vqe._resolve_context_from_spec(family_vqe._spec_by_case_id(family, case_id))
            pool = build_full_meta_candidate_pool(context)
            selection = family_vqe._select_family_informed_candidates(family, pool, max_terms=12)
            assert selection.selected, (family, case_id)
            assert selection.selected_matches, (family, case_id)
            assert len(selection.selected) == len(selection.selected_matches)
            for candidate, match in zip(selection.selected, selection.selected_matches, strict=True):
                assert family_vqe._family_informed_policy_match(family, str(candidate.label)) == match

def test_no_policy_match_emits_resource_guard_without_exact(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    monkeypatch.setattr(family_vqe, "has_scipy_minimize_support", lambda: True)
    monkeypatch.setattr(family_vqe, "_import_scipy_minimize", lambda: None)
    monkeypatch.setattr(family_vqe, "_spec_by_case_id", lambda family, case_id: _fake_spec())
    monkeypatch.setattr(family_vqe, "_resolve_context_from_spec", lambda spec: _fake_context(events))
    monkeypatch.setattr(family_vqe, "build_full_meta_candidate_pool", lambda context, *, max_terms=family_vqe._POOL_TERM_CAP: (_candidate("generic_xy"),))

    payload = family_vqe.run_static_family_informed_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
    )

    assert events == []
    assert payload["status"] == "skipped_resource_guard"
    assert payload["resource_guard"]["resource_guard_kind"] == "family_informed_vqe_no_policy_match"
    assert payload["rows"][0]["family_informed_policy_match_status"] == "no_policy_match"
    assert (tmp_path / "generic_static_single.json").exists()


def test_full_meta_cap_guard_writes_normalized_resource_skip(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    monkeypatch.setattr(family_vqe, "has_scipy_minimize_support", lambda: True)
    monkeypatch.setattr(family_vqe, "_import_scipy_minimize", lambda: None)
    monkeypatch.setattr(family_vqe, "_spec_by_case_id", lambda family, case_id: _fake_spec())
    monkeypatch.setattr(family_vqe, "_resolve_context_from_spec", lambda spec: _fake_context(events))

    def _raise_cap(context, *, max_terms=family_vqe._POOL_TERM_CAP):  # noqa: ANN001
        raise ValueError("full_meta pool exceeds cap: 257 > 256")

    monkeypatch.setattr(family_vqe, "build_full_meta_candidate_pool", _raise_cap)

    payload = family_vqe.run_static_family_informed_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
    )

    assert events == []
    assert payload["status"] == "skipped_resource_guard"
    assert payload["resource_guard"]["resource_guard_kind"] == "family_informed_vqe_full_meta_pool_term_cap"
    assert payload["rows"][0]["compiled_circuit_stats_status"] == "not_applicable_not_completed"


def test_qubit_cap_guard_writes_normalized_resource_skip(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    monkeypatch.setattr(family_vqe, "has_scipy_minimize_support", lambda: True)
    monkeypatch.setattr(family_vqe, "_import_scipy_minimize", lambda: None)
    monkeypatch.setattr(family_vqe, "_spec_by_case_id", lambda family, case_id: _fake_spec())
    monkeypatch.setattr(
        family_vqe,
        "_resolve_context_from_spec",
        lambda spec: _fake_context(events, num_qubits=family_vqe._QUBIT_CAP + 1),
    )

    payload = family_vqe.run_static_family_informed_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
    )

    assert events == []
    assert payload["status"] == "skipped_resource_guard"
    assert payload["resource_guard"]["resource_guard_kind"] == "family_informed_vqe_qubit_cap"


def test_missing_scipy_writes_controlled_skip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(family_vqe, "has_scipy_minimize_support", lambda: False)

    payload = family_vqe.run_static_family_informed_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
    )

    assert payload["status"] == "skipped_optional_dependency"
    assert payload["rows"][0]["phase3_controller_called"] is False
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "rows.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "metrics_proxy_summary.json").exists()


def test_runner_resolves_exact_after_optimizer_and_emits_required_fields(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []

    def _fake_minimize(objective, x0, method=None, options=None):  # noqa: ANN001, ANN003, ANN201
        events.append("optimizer")
        x = np.asarray(x0, dtype=float).reshape(-1) + 0.1
        return SimpleNamespace(x=x, fun=float(objective(x)), nfev=2, nit=1, success=True, message="ok")

    def _fake_sector(context, psi):  # noqa: ANN001
        assert events == ["optimizer", "exact"]
        events.append("sector")
        return {"sector_probability": 1.0, "truncation_constraints_evaluated": []}

    monkeypatch.setattr(family_vqe, "has_scipy_minimize_support", lambda: True)
    monkeypatch.setattr(family_vqe, "_import_scipy_minimize", lambda: _fake_minimize)
    monkeypatch.setattr(family_vqe, "_spec_by_case_id", lambda family, case_id: _fake_spec())
    monkeypatch.setattr(family_vqe, "_resolve_context_from_spec", lambda spec: _fake_context(events))
    monkeypatch.setattr(family_vqe, "build_full_meta_candidate_pool", lambda context, *, max_terms=family_vqe._POOL_TERM_CAP: (_candidate("uccsd_sing(alpha:0->1)"),))
    monkeypatch.setattr(family_vqe, "_sector_or_unavailable", _fake_sector)

    payload = family_vqe.run_static_family_informed_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
        optimizer_maxiter=5,
    )

    assert events == ["optimizer", "exact", "sector"]
    assert payload["status"] == "completed"
    row = payload["rows"][0]
    assert row["method_id"] == "static_family_informed_vqe"
    assert row["method_label"] == "family-informed VQE"
    assert row["taxonomy_role"] == "fixed_ansatz_comparator"
    assert row["pool_name"] == "family_informed_full_meta_subset"
    assert row["phase3_controller_called"] is False
    assert row["phase3_emulation"] is False
    assert row["uses_exact_for_decision"] is False
    assert row["exact_reference_usage"] == "reporting_only_after_optimization"
    assert row["family_informed_policy_match_version"] == "family_informed_explicit_policy_v1"
    assert row["family_informed_policy_match_status"] == "approved_policy_labels"
    assert row["selected_operator_policy_classes"] == ["uccsd_sing"]
    assert row["optimizer"] == "scipy.optimize.minimize:BFGS"
    assert row["optimizer_kind"] == "bfgs"
    assert row["optimizer_profile"] is None
    assert row["shots_total"] > 0
    assert row["compiled_circuit_stats_status"] in {"deterministic_pauli_rotation_proxy", "ok"}


def test_family_informed_spsa_does_not_require_scipy_and_records_budget(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []

    def _fake_spsa(fun, x0, **kwargs):  # noqa: ANN001, ANN003, ANN201
        events.append("optimizer")
        assert kwargs["maxiter"] == 3
        assert kwargs["seed"] == 123
        assert kwargs["a"] == pytest.approx(0.05)
        assert kwargs["c"] == pytest.approx(0.02)
        assert kwargs["alpha"] == pytest.approx(0.601)
        assert kwargs["gamma"] == pytest.approx(0.11)
        assert kwargs["A"] == pytest.approx(8.0)
        assert kwargs["eval_repeats"] == 2
        assert kwargs["avg_last"] == 3
        x = np.asarray(x0, dtype=float).reshape(-1) + 0.1
        return SimpleNamespace(x=x, fun=float(fun(x)), nfev=2, nit=1, success=True, message="spsa ok")

    def _fake_sector(context, psi):  # noqa: ANN001
        assert events == ["optimizer", "exact"]
        events.append("sector")
        return {"sector_probability": 1.0, "truncation_constraints_evaluated": []}

    monkeypatch.setattr(family_vqe, "has_scipy_minimize_support", lambda: False)
    monkeypatch.setattr(family_vqe, "_import_scipy_minimize", lambda: (_ for _ in ()).throw(AssertionError("no scipy")))
    monkeypatch.setattr(family_vqe, "spsa_minimize", _fake_spsa)
    monkeypatch.setattr(family_vqe, "_spec_by_case_id", lambda family, case_id: _fake_spec())
    monkeypatch.setattr(family_vqe, "_resolve_context_from_spec", lambda spec: _fake_context(events))
    monkeypatch.setattr(family_vqe, "build_full_meta_candidate_pool", lambda context, *, max_terms=family_vqe._POOL_TERM_CAP: (_candidate("uccsd_sing(alpha:0->1)"),))
    monkeypatch.setattr(family_vqe, "_sector_or_unavailable", _fake_sector)

    payload = family_vqe.run_static_family_informed_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
        optimizer_profile=PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
        optimizer_profile_source="env",
        family_informed_optimizer="spsa",
        family_informed_spsa_maxiter=3,
        family_informed_spsa_seed=123,
        family_informed_spsa_a=0.05,
        family_informed_spsa_c=0.02,
        family_informed_spsa_alpha=0.601,
        family_informed_spsa_gamma=0.11,
        family_informed_spsa_big_a=8.0,
        family_informed_spsa_eval_repeats=2,
        family_informed_spsa_avg_last=3,
        optimizer_overlay_source="test",
    )

    assert events == ["optimizer", "exact", "sector"]
    assert payload["status"] == "completed"
    row = payload["rows"][0]
    assert row["optimizer"] == "repo_native_spsa:spsa_minimize"
    assert row["optimizer_kind"] == "spsa"
    assert row["optimizer_profile"] == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
    assert row["optimizer_profile_source"] == "env"
    assert row["optimizer_overlay_source"] == "test"
    assert row["optimizer_maxiter"] == 3
    assert row["family_informed_spsa_maxiter"] == 3
    assert row["family_informed_spsa_seed"] == 123
    assert row["family_informed_spsa_a"] == pytest.approx(0.05)
    assert row["family_informed_spsa_c"] == pytest.approx(0.02)
    assert row["family_informed_spsa_alpha"] == pytest.approx(0.601)
    assert row["family_informed_spsa_gamma"] == pytest.approx(0.11)
    assert row["family_informed_spsa_big_a"] == pytest.approx(8.0)
    assert row["family_informed_spsa_eval_repeats"] == 2
    assert row["family_informed_spsa_avg_last"] == 3
    assert row["family_informed_spsa_schedule_sources"]["family_informed_spsa_a"] == "explicit"
    assert row["family_informed_spsa_schedule_sources"]["family_informed_spsa_avg_last"] == "explicit"
    assert row["spsa_seed"] == 123
    assert row["spsa_a"] == pytest.approx(0.05)
    assert row["spsa_c"] == pytest.approx(0.02)
    assert row["spsa_alpha"] == pytest.approx(0.601)
    assert row["spsa_gamma"] == pytest.approx(0.11)
    assert row["spsa_A"] == pytest.approx(8.0)
    assert row["spsa_big_a"] == pytest.approx(8.0)
    assert row["spsa_eval_repeats"] == 2
    assert row["spsa_avg_last"] == 3
    assert row["energy"] == pytest.approx(row["optimizer_reported_energy"])


def test_family_informed_spsa_blank_schedule_records_native_defaults_without_kwargs() -> None:
    settings = family_vqe._normalize_optimizer_settings(
        optimizer_maxiter=3,
        seed=123,
        family_informed_optimizer="spsa",
    )

    assert family_vqe._native_spsa_kwargs_from_settings(settings) == {}
    assert settings["family_informed_spsa_a"] == pytest.approx(0.2)
    assert settings["family_informed_spsa_c"] == pytest.approx(0.1)
    assert settings["family_informed_spsa_alpha"] == pytest.approx(0.602)
    assert settings["family_informed_spsa_gamma"] == pytest.approx(0.101)
    assert settings["family_informed_spsa_big_a"] == pytest.approx(10.0)
    assert settings["family_informed_spsa_eval_repeats"] == 1
    assert settings["family_informed_spsa_avg_last"] == 0
    assert set(settings["family_informed_spsa_schedule_sources"].values()) == {"native_default"}


def test_family_informed_spsa_schedule_requires_spsa(tmp_path: Path) -> None:
    payload = family_vqe.run_static_family_informed_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
        family_informed_optimizer="bfgs",
        family_informed_spsa_a=0.05,
    )

    assert payload["status"] == "failed"
    assert payload["exception_type"] == "ValueError"
    assert "require optimizer=spsa" in payload["reason"]


def test_decision_noise_affects_family_informed_objective_but_final_energy_is_exact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def _fake_minimize(objective, x0, method=None, options=None):  # noqa: ANN001, ANN003, ANN201
        events.append("optimizer")
        x = np.asarray(x0, dtype=float).reshape(-1) + 0.1
        decision_value = float(objective(x))
        return SimpleNamespace(x=x, fun=decision_value, nfev=1, nit=1, success=True, message="ok")

    def _fake_sector(context, psi):  # noqa: ANN001
        events.append("sector")
        return {"sector_probability": 1.0, "truncation_constraints_evaluated": []}

    monkeypatch.setattr(family_vqe, "has_scipy_minimize_support", lambda: True)
    monkeypatch.setattr(family_vqe, "_import_scipy_minimize", lambda: _fake_minimize)
    monkeypatch.setattr(family_vqe, "_spec_by_case_id", lambda family, case_id: _fake_spec())
    monkeypatch.setattr(family_vqe, "_resolve_context_from_spec", lambda spec: _fake_context(events))
    monkeypatch.setattr(family_vqe, "build_full_meta_candidate_pool", lambda context, *, max_terms=family_vqe._POOL_TERM_CAP: (_candidate("uccsd_sing(alpha:0->1)"),))
    monkeypatch.setattr(family_vqe, "_sector_or_unavailable", _fake_sector)

    payload = family_vqe.run_static_family_informed_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
        optimizer_maxiter=5,
        benchmark_decision_noise_config={
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "0.5",
            "benchmark_decision_noise_seed": "20260515",
        },
    )

    row = payload["rows"][0]
    meta = row["benchmark_decision_noise"]
    first_draw = meta["trace_preview"][0]
    assert payload["benchmark_decision_noise_status"] == "ok"
    assert row["benchmark_decision_noise_status"] == "ok"
    assert meta["semantic"] == "benchmark_decision_value_noise_not_physical_shots_v1"
    assert meta["draw_count_total"] >= 1
    assert meta["surfaces_affected"] == ["family_informed_objective"]
    assert row["optimizer_decision_energy"] == pytest.approx(first_draw["value_decision"])
    assert row["energy"] == pytest.approx(first_draw["value_ideal"])
    assert row["delta_E_abs"] == pytest.approx(abs(row["energy"] - row["exact_energy"]))
    assert row["shots_total"] > 0
    rows_payload = json.loads((tmp_path / "rows.json").read_text(encoding="utf-8"))
    assert rows_payload["benchmark_decision_noise_status"] == "ok"


def test_failure_path_emits_normalized_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(family_vqe, "_spec_by_case_id", lambda family, case_id: (_ for _ in ()).throw(RuntimeError("boom")))

    payload = family_vqe.run_static_family_informed_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
    )

    assert payload["status"] == "failed"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["guardrails"]["phase3_controller_called"] is False
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "rows.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "metrics_proxy_summary.json").exists()
