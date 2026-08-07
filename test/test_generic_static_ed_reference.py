#!/usr/bin/env python3
"""Tests for the generic exact-bench static ED reference row."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pipelines.exact_bench import generic_static_ed_reference as edref


_NON_HH_STATIC_ED_REFERENCE_FAMILIES = (
    "hubbard",
    "ionic_hubbard",
    "extended_hubbard",
    "ttprime_hubbard",
    "spinless_tv",
    "spin_boson",
    "bose_hubbard",
    "harmonic_kerr_chain",
    "molecular_restricted_closed_shell",
)


def _fake_spec(
    *,
    family: str = "hubbard",
    benchmark_id: str = "hubbard_L2",
    L: int = 2,
    n_qubits: int = 4,
    dense_eigh_max_dim: int = 1024,
) -> SimpleNamespace:
    base_args = (
        "--problem",
        family,
        "--L",
        str(L),
        "--t",
        "1.0",
        "--u",
        "4.0",
        "--dv",
        "0.0",
        "--omega0",
        "1.0",
        "--g-ep",
        "0.5",
        "--n-ph-max",
        "1",
        "--boson-encoding",
        "binary",
        "--ordering",
        "blocked",
        "--boundary",
        "open",
    )
    if family == "molecular_restricted_closed_shell":
        base_args = (
            "--problem",
            family,
            "--molecular-problem-json",
            "fake-molecule.json",
            "--L",
            str(L),
            "--ordering",
            "blocked",
            "--boundary",
            "open",
            "--dense-eigh-max-dim",
            str(dense_eigh_max_dim),
        )
    return SimpleNamespace(
        benchmark_id=benchmark_id,
        family=family,
        features=SimpleNamespace(problem=family, L=L, n_qubits=n_qubits, molecular=(family == "molecular_restricted_closed_shell")),
        base_pipeline_args=base_args,
        split="train",
        tags=("static_phase3",),
    )


def _fake_context(events: list[str]) -> SimpleNamespace:
    class _ExactTarget:
        kind = "exact_ground_energy_sector"
        comparison_space_label = "full_register"

        def resolve_energy(self, *, ai_log=None):  # noqa: ANN001, ANN201 - exact target protocol
            events.append("exact_target.resolve_energy")
            assert ai_log is None
            return -1.75

    return SimpleNamespace(
        request=SimpleNamespace(num_sites=2),
        layout=SimpleNamespace(total_qubits=4),
        sector=SimpleNamespace(label="half_filled_spin_sector", comparison_space_label="full_register"),
        exact_target=_ExactTarget(),
        exact_comparison_space_label="full_register",
    )


def test_default_static_ed_reference_case_ids_cover_non_hh_train_suite() -> None:
    all_ids = []
    for family in _NON_HH_STATIC_ED_REFERENCE_FAMILIES:
        ids = edref.default_static_ed_reference_case_ids(family)
        assert ids, family
        assert all(str(case_id).startswith(family) for case_id in ids)
        all_ids.extend(ids)

    assert len(all_ids) == 17
    assert edref.default_static_ed_reference_case_ids("hh") == ()
    assert "hubbard_L2" in all_ids
    assert "spin_boson_L1" in all_ids
    assert "molecular_restricted_closed_shell_L2" in all_ids


def test_success_path_resolves_existing_exact_target_and_writes_artifacts(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    spec = _fake_spec()
    context = _fake_context(events)

    monkeypatch.setattr(edref, "_spec_by_case_id", lambda family, case_id: spec)
    monkeypatch.setattr(edref, "_resolve_context_from_spec", lambda resolved_spec: context)

    payload = edref.run_static_ed_reference_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
    )

    assert events == ["exact_target.resolve_energy"]
    assert payload["status"] == "completed"
    assert payload["guardrails"]["phase3_controller_called"] is False
    row = payload["rows"][0]
    assert row["status"] == "ok"
    assert row["energy"] == -1.75
    assert row["exact_energy"] == -1.75
    assert row["delta_E_abs"] == 0.0
    assert row["method_id"] == "static_ed_reference"
    assert row["method_kind"] == "classical_reference"
    assert row["uses_exact_for_decision"] is False
    assert row["phase3_controller_called"] is False
    assert row["exact_target_kind"] == "exact_ground_energy_sector"
    assert row["qiskit_boundary"] == "not_used"

    for name in (
        "result.json",
        "rows.json",
        "manifest.json",
        "generic_static_single.json",
        "metrics_proxy_runs.csv",
        "metrics_proxy_runs.jsonl",
        "metrics_proxy_summary.json",
    ):
        assert (tmp_path / name).exists(), name

    summary = json.loads((tmp_path / "metrics_proxy_summary.json").read_text(encoding="utf-8"))
    assert summary["status_counts"] == {"ok": 1}
    assert summary["schema_source"] == edref.SCHEMA_VERSION


def test_failure_path_emits_normalized_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        edref,
        "_spec_by_case_id",
        lambda family, case_id: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    payload = edref.run_static_ed_reference_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
    )

    assert payload["status"] == "failed"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["guardrails"]["phase3_controller_called"] is False
    assert payload["rows"][0]["status"] == "failed"
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "rows.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "metrics_proxy_summary.json").exists()


def test_molecular_dense_guard_skips_without_resolving_context(monkeypatch, tmp_path: Path) -> None:
    spec = _fake_spec(
        family="molecular_restricted_closed_shell",
        benchmark_id="molecular_restricted_closed_shell_lih_sto3g_L6",
        L=6,
        n_qubits=12,
        dense_eigh_max_dim=1024,
    )

    monkeypatch.setattr(edref, "_spec_by_case_id", lambda family, case_id: spec)

    def _forbidden_resolve_context(resolved_spec):  # noqa: ANN001, ANN202
        raise AssertionError("resource-guarded molecular case must not resolve context")

    monkeypatch.setattr(edref, "_resolve_context_from_spec", _forbidden_resolve_context)

    payload = edref.run_static_ed_reference_single(
        family="molecular_restricted_closed_shell",
        case_id="molecular_restricted_closed_shell_lih_sto3g_L6",
        output_dir=tmp_path,
    )

    assert payload["status"] == "skipped_resource_guard"
    assert payload["guardrails"]["phase3_controller_called"] is False
    assert payload["resource_guard"]["dense_hilbert_dimension"] == 4096
    assert payload["resource_guard"]["dense_eigh_max_dim"] == 1024
    row = payload["rows"][0]
    assert row["status"] == "skipped_resource_guard"
    assert row["resource_guard"] is True
    assert row["exact_reference_usage"] == "not_resolved_resource_guard"
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "metrics_proxy_summary.json").exists()
