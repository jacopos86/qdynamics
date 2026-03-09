#!/usr/bin/env python3
"""Tests for hardcoded hubbard_pipeline internal ADAPT reference source wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hubbard_latex_python_pairs import boson_qubits_per_site

import pipelines.hardcoded.hubbard_pipeline as hp


def _hh_state_dim(L: int, n_ph_max: int, boson_encoding: str) -> int:
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    nq_total = 2 * int(L) + int(L) * qpb
    return 1 << int(nq_total)


def _basis0(dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=complex)
    out[0] = 1.0
    return out


class TestParseArgsAdaptRefSource:
    def test_parse_accepts_adapt_ref_source_and_composite_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "hubbard_pipeline.py",
                "--L", "2",
                "--adapt-pool", "uccsd_paop_lf_full",
                "--adapt-ref-source", "vqe",
            ],
        )
        args = hp.parse_args()
        assert str(args.adapt_pool) == "uccsd_paop_lf_full"
        assert str(args.adapt_ref_source) == "vqe"

    def test_parse_accepts_full_meta_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "hubbard_pipeline.py",
                "--L", "2",
                "--adapt-pool", "full_meta",
                "--adapt-ref-source", "vqe",
            ],
        )
        args = hp.parse_args()
        assert str(args.adapt_pool) == "full_meta"
        assert str(args.adapt_ref_source) == "vqe"

    def test_adapt_ref_source_default_is_hf(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "argv", ["hubbard_pipeline.py", "--L", "2"])
        args = hp.parse_args()
        assert str(args.adapt_ref_source) == "hf"


class TestAdaptRefSourceVQEPath:
    def test_internal_adapt_uses_vqe_override_when_requested(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        dim = _hh_state_dim(2, 1, "binary")
        psi_vqe = _basis0(dim)
        captured: dict[str, object] = {}

        def _fake_run_hardcoded_vqe(**kwargs):
            return {
                "success": True,
                "method": "mock_vqe",
                "ansatz": "hh_hva_ptw",
                "energy": -1.0,
                "exact_filtered_energy": -1.1,
                "num_particles": {"n_up": 1, "n_dn": 1},
            }, np.array(psi_vqe, copy=True)

        def _fake_run_internal_adapt_paop(**kwargs):
            captured["psi_ref_override"] = kwargs.get("psi_ref_override")
            return {
                "success": True,
                "pool_type": "uccsd_paop_lf_full",
                "ansatz_depth": 3,
                "num_parameters": 3,
                "energy": -1.05,
                "abs_delta_e": 0.05,
                "stop_reason": "eps_grad",
                "elapsed_s": 0.01,
                "allow_repeats": True,
            }, _basis0(dim)

        def _fake_simulate_trajectory(**kwargs):
            return ([{"time": 0.0, "fidelity": 1.0}], [])

        monkeypatch.setattr(hp, "_run_hardcoded_vqe", _fake_run_hardcoded_vqe)
        monkeypatch.setattr(hp, "_run_internal_adapt_paop", _fake_run_internal_adapt_paop)
        monkeypatch.setattr(hp, "_simulate_trajectory", _fake_simulate_trajectory)

        out_json = tmp_path / "hc_hh_adapt_ref_source_vqe.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "hubbard_pipeline.py",
                "--L", "2",
                "--problem", "hh",
                "--omega0", "1.0",
                "--g-ep", "0.5",
                "--n-ph-max", "1",
                "--boson-encoding", "binary",
                "--vqe-ansatz", "hh_hva_ptw",
                "--adapt-pool", "uccsd_paop_lf_full",
                "--adapt-ref-source", "vqe",
                "--skip-qpe",
                "--skip-pdf",
                "--output-json", str(out_json),
            ],
        )
        hp.main()

        psi_ref_override = captured.get("psi_ref_override")
        assert isinstance(psi_ref_override, np.ndarray)
        assert int(psi_ref_override.size) == dim
        assert np.isclose(float(np.linalg.norm(psi_ref_override)), 1.0)

        payload = json.loads(out_json.read_text(encoding="utf-8"))
        settings = payload.get("settings", {})
        adapt_internal = payload.get("adapt_internal", {})
        assert settings.get("adapt_pool") == "uccsd_paop_lf_full"
        assert settings.get("adapt_ref_source") == "vqe"
        assert adapt_internal.get("pool_type") == "uccsd_paop_lf_full"

    def test_adapt_ref_source_vqe_errors_cleanly_when_vqe_failed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        dim = _hh_state_dim(2, 1, "binary")
        calls = {"internal_adapt_called": False}

        def _fake_run_hardcoded_vqe(**kwargs):
            return {
                "success": False,
                "method": "mock_vqe",
                "ansatz": "hh_hva_ptw",
                "energy": None,
                "exact_filtered_energy": None,
                "error": "forced failure",
            }, _basis0(dim)

        def _fake_run_internal_adapt_paop(**kwargs):
            calls["internal_adapt_called"] = True
            raise AssertionError("internal ADAPT should not run when adapt_ref_source=vqe and VQE fails")

        monkeypatch.setattr(hp, "_run_hardcoded_vqe", _fake_run_hardcoded_vqe)
        monkeypatch.setattr(hp, "_run_internal_adapt_paop", _fake_run_internal_adapt_paop)

        out_json = tmp_path / "hc_hh_adapt_ref_source_vqe_fail.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "hubbard_pipeline.py",
                "--L", "2",
                "--problem", "hh",
                "--omega0", "1.0",
                "--g-ep", "0.5",
                "--n-ph-max", "1",
                "--boson-encoding", "binary",
                "--vqe-ansatz", "hh_hva_ptw",
                "--initial-state-source", "hf",
                "--adapt-pool", "uccsd_paop_lf_full",
                "--adapt-ref-source", "vqe",
                "--skip-qpe",
                "--skip-pdf",
                "--output-json", str(out_json),
            ],
        )
        with pytest.raises(RuntimeError, match="--adapt-ref-source vqe"):
            hp.main()
        assert calls["internal_adapt_called"] is False


class TestInternalHHAdaptTerminationSemantics:
    def test_internal_hh_phase3_disables_eps_energy_hard_stop(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        dim = _hh_state_dim(2, 1, "binary")

        def _fake_run_hardcoded_vqe(**kwargs):
            return {
                "success": True,
                "method": "mock_vqe",
                "ansatz": "hh_hva_ptw",
                "energy": -1.0,
                "exact_filtered_energy": -1.1,
                "num_particles": {"n_up": 1, "n_dn": 1},
            }, _basis0(dim)

        def _fake_simulate_trajectory(**kwargs):
            return ([{"time": 0.0, "fidelity": 1.0}], [])

        monkeypatch.setattr(hp, "_run_hardcoded_vqe", _fake_run_hardcoded_vqe)
        monkeypatch.setattr(hp, "_simulate_trajectory", _fake_simulate_trajectory)

        out_json = tmp_path / "hc_hh_internal_adapt_eps_energy_semantics.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "hubbard_pipeline.py",
                "--L", "2",
                "--problem", "hh",
                "--omega0", "1.0",
                "--g-ep", "0.5",
                "--n-ph-max", "1",
                "--boson-encoding", "binary",
                "--vqe-ansatz", "hh_hva_ptw",
                "--adapt-pool", "paop_lf_std",
                "--adapt-continuation-mode", "phase3_v1",
                "--adapt-max-depth", "3",
                "--adapt-eps-grad", "-1",
                "--adapt-eps-energy", "1e9",
                "--adapt-maxiter", "5",
                "--skip-qpe",
                "--skip-pdf",
                "--output-json", str(out_json),
            ],
        )
        hp.main()

        payload = json.loads(out_json.read_text(encoding="utf-8"))
        adapt_internal = payload.get("adapt_internal", {})
        assert bool(adapt_internal.get("eps_energy_termination_enabled")) is False
        assert bool(adapt_internal.get("eps_grad_termination_enabled")) is False
        assert bool(adapt_internal.get("adapt_drop_policy_enabled")) is True
        assert adapt_internal.get("adapt_drop_floor_resolved") == pytest.approx(5e-4)
        assert int(adapt_internal.get("adapt_drop_patience_resolved")) == 3
        assert int(adapt_internal.get("adapt_drop_min_depth_resolved")) == 12
        assert adapt_internal.get("adapt_grad_floor_resolved") == pytest.approx(2e-2)
        assert adapt_internal.get("adapt_drop_policy_source") == "auto_hh_staged"
        assert str(adapt_internal.get("stop_reason")) in {"max_depth", "pool_exhausted"}
        assert str(adapt_internal.get("stop_reason")) != "eps_energy"
        assert str(adapt_internal.get("stop_reason")) != "eps_grad"


# ────────────────────────────────────────────────────────────────────
#  P2 — windowed reopt wrapper plumbing (hubbard_pipeline)
# ────────────────────────────────────────────────────────────────────


class TestHubbardPipelineWindowedCLI:
    """CLI arg-parsing for windowed reopt knobs via hubbard_pipeline."""

    def test_parse_accepts_windowed_policy(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["hubbard_pipeline.py", "--L", "2",
             "--adapt-reopt-policy", "windowed"],
        )
        args = hp.parse_args()
        assert args.adapt_reopt_policy == "windowed"

    def test_parse_windowed_knob_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["hubbard_pipeline.py", "--L", "2"])
        args = hp.parse_args()
        assert args.adapt_reopt_policy == "append_only"
        assert args.adapt_window_size == 3
        assert args.adapt_window_topk == 0
        assert args.adapt_full_refit_every == 0
        assert args.adapt_final_full_refit == "true"

    def test_parse_windowed_knob_overrides(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            [
                "hubbard_pipeline.py", "--L", "2",
                "--adapt-reopt-policy", "windowed",
                "--adapt-window-size", "4",
                "--adapt-window-topk", "2",
                "--adapt-full-refit-every", "3",
                "--adapt-final-full-refit", "false",
            ],
        )
        args = hp.parse_args()
        assert args.adapt_window_size == 4
        assert args.adapt_window_topk == 2
        assert args.adapt_full_refit_every == 3
        assert args.adapt_final_full_refit == "false"


class TestHubbardPipelineWindowedPassthrough:
    """Verify windowed knobs are forwarded from main() to _run_internal_adapt_paop."""

    def test_windowed_knobs_reach_internal_adapt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        dim = _hh_state_dim(2, 1, "binary")
        captured: dict[str, object] = {}

        def _fake_run_hardcoded_vqe(**kwargs):
            return {
                "success": True, "method": "mock", "ansatz": "hh_hva_ptw",
                "energy": -1.0, "exact_filtered_energy": -1.1,
                "num_particles": {"n_up": 1, "n_dn": 1},
            }, _basis0(dim)

        def _fake_run_internal_adapt_paop(**kwargs):
            captured["adapt_reopt_policy"] = kwargs.get("adapt_reopt_policy")
            captured["adapt_window_size"] = kwargs.get("adapt_window_size")
            captured["adapt_window_topk"] = kwargs.get("adapt_window_topk")
            captured["adapt_full_refit_every"] = kwargs.get("adapt_full_refit_every")
            captured["adapt_final_full_refit"] = kwargs.get("adapt_final_full_refit")
            return {
                "success": True, "pool_type": "uccsd", "ansatz_depth": 1,
                "num_parameters": 1, "energy": -1.0, "abs_delta_e": 0.1,
                "stop_reason": "max_depth", "elapsed_s": 0.01,
                "allow_repeats": True,
            }, _basis0(dim)

        def _fake_simulate_trajectory(**kwargs):
            return ([{"time": 0.0, "fidelity": 1.0}], [])

        monkeypatch.setattr(hp, "_run_hardcoded_vqe", _fake_run_hardcoded_vqe)
        monkeypatch.setattr(hp, "_run_internal_adapt_paop", _fake_run_internal_adapt_paop)
        monkeypatch.setattr(hp, "_simulate_trajectory", _fake_simulate_trajectory)

        out_json = tmp_path / "hc_windowed_passthrough.json"
        monkeypatch.setattr(
            sys, "argv",
            [
                "hubbard_pipeline.py",
                "--L", "2",
                "--problem", "hh",
                "--omega0", "1.0", "--g-ep", "0.5",
                "--n-ph-max", "1", "--boson-encoding", "binary",
                "--vqe-ansatz", "hh_hva_ptw",
                "--adapt-pool", "uccsd",
                "--adapt-reopt-policy", "windowed",
                "--adapt-window-size", "4",
                "--adapt-window-topk", "2",
                "--adapt-full-refit-every", "5",
                "--adapt-final-full-refit", "false",
                "--skip-qpe", "--skip-pdf",
                "--output-json", str(out_json),
            ],
        )
        hp.main()

        assert captured["adapt_reopt_policy"] == "windowed"
        assert captured["adapt_window_size"] == 4
        assert captured["adapt_window_topk"] == 2
        assert captured["adapt_full_refit_every"] == 5
        assert captured["adapt_final_full_refit"] is False
