"""Round-trip test: staged export -> canonical replay metadata validation.

Verifies that state bundles emitted by the upgraded staged exporter satisfy
the canonical replay parser contract in ``hh_vqe_from_adapt_family.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_vqe_from_adapt_family import (
    _extract_adapt_operator_theta_sequence,
    _extract_payload_parameterization_layout,
    _infer_handoff_state_kind,
    _resolve_family_from_metadata,
)
from pipelines.hardcoded.handoff_state_bundle import (
    HandoffStateBundleConfig,
    write_handoff_state_bundle,
)
from src.quantum.ansatz_parameterization import build_parameter_layout, serialize_layout
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


# ---------------------------------------------------------------------------
# Helper: build a minimal staged-export payload that mirrors what
# write_handoff_state_bundle now emits for a final stage export.
# ---------------------------------------------------------------------------

def _make_staged_export_payload(
    *,
    L: int = 2,
    operators: list[str] | None = None,
    optimal_point: list[float] | None = None,
    logical_optimal_point: list[float] | None = None,
    logical_num_parameters: int | None = None,
    parameterization: dict | None = None,
    pool_type: str | None = None,
) -> dict:
    if operators is None:
        operators = ["op_a", "op_b", "op_c"]
    if optimal_point is None:
        optimal_point = [0.1, -0.2, 0.3]
    if pool_type is None:
        pool_type = "pool_a"

    nq = 2 * L + L  # minimal: 2L fermion + L boson bits for n_ph_max=1/binary
    dim = 1 << nq
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    amps = {}
    for idx in range(dim):
        amp = psi[idx]
        if abs(amp) > 1e-14:
            amps[format(idx, f"0{nq}b")] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}

    adapt_vqe = {
        "energy": -1.0,
        "abs_delta_e": 0.01,
        "relative_error_abs": 0.001,
        "operators": operators,
        "optimal_point": optimal_point,
        "ansatz_depth": len(operators),
        "num_parameters": len(optimal_point),
        "pool_type": pool_type,
    }
    if logical_optimal_point is not None:
        adapt_vqe["logical_optimal_point"] = logical_optimal_point
    if logical_num_parameters is not None:
        adapt_vqe["logical_num_parameters"] = int(logical_num_parameters)
    if parameterization is not None:
        adapt_vqe["parameterization"] = dict(parameterization)

    return {
        "generated_utc": "2026-03-06T00:00:00Z",
        "settings": {
            "L": L,
            "problem": "hh",
            "t": 1.0,
            "u": 4.0,
            "dv": 0.0,
            "omega0": 1.0,
            "g_ep": 0.5,
            "n_ph_max": 1,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
            "sector_n_up": (L + 1) // 2,
            "sector_n_dn": L // 2,
        },
        "adapt_vqe": adapt_vqe,
        "initial_state": {
            "source": "A_probe_final",
            "nq_total": nq,
            "amplitudes_qn_to_q0": amps,
            "amplitude_cutoff": 1e-14,
            "norm": 1.0,
            "handoff_state_kind": "prepared_state",
        },
        "exact": {"E_exact_sector": -2.0},
    }


def _make_runtime_layout() -> tuple[dict[str, object], list[float], list[float]]:
    terms = [
        AnsatzTerm(
            label="op_x",
            polynomial=PauliPolynomial(
                "JW",
                [PauliTerm(2, ps="xx", pc=1.0), PauliTerm(2, ps="zz", pc=0.5)],
            ),
        ),
        AnsatzTerm(
            label="op_y",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xy", pc=1.0)]),
        ),
    ]
    layout = build_parameter_layout(terms)
    logical_theta = [0.125, -0.2]
    runtime_theta = [0.1, 0.15, -0.2]
    return serialize_layout(layout), logical_theta, runtime_theta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCanonicalReplayFieldsPresent:
    """Verify exported payload satisfies the canonical replay parser."""

    def test_extract_succeeds_for_staged_export(self) -> None:
        payload = _make_staged_export_payload()
        labels, theta = _extract_adapt_operator_theta_sequence(payload)
        assert labels == ["op_a", "op_b", "op_c"]
        assert np.allclose(theta, [0.1, -0.2, 0.3])

    def test_operators_and_optimal_point_length_match(self) -> None:
        payload = _make_staged_export_payload(operators=["a", "b"], optimal_point=[0.1, 0.2])
        labels, theta = _extract_adapt_operator_theta_sequence(payload)
        assert len(labels) == len(theta)

    def test_parameterized_runtime_vector_can_exceed_operator_length(self) -> None:
        parameterization, logical_theta, runtime_theta = _make_runtime_layout()
        payload = _make_staged_export_payload(
            operators=["op_x", "op_y"],
            optimal_point=runtime_theta,
            logical_optimal_point=logical_theta,
            logical_num_parameters=2,
            parameterization=parameterization,
        )
        labels, theta = _extract_adapt_operator_theta_sequence(payload)
        layout = _extract_payload_parameterization_layout(payload)
        assert labels == ["op_x", "op_y"]
        assert np.allclose(theta, runtime_theta)
        assert layout is not None
        assert int(layout.logical_parameter_count) == 2
        assert int(layout.runtime_parameter_count) == 3

    def test_pool_type_resolves_pool_a(self) -> None:
        payload = _make_staged_export_payload(pool_type="pool_a")
        fam, src = _resolve_family_from_metadata(payload)
        assert fam == "pool_a"
        assert src == "adapt_vqe.pool_type"

    def test_pool_type_resolves_pool_b(self) -> None:
        payload = _make_staged_export_payload(pool_type="pool_b")
        fam, src = _resolve_family_from_metadata(payload)
        assert fam == "pool_b"
        assert src == "adapt_vqe.pool_type"

    def test_ansatz_depth_matches_operators(self) -> None:
        ops = ["x", "y", "z", "w"]
        payload = _make_staged_export_payload(operators=ops, optimal_point=[0.1] * 4)
        assert payload["adapt_vqe"]["ansatz_depth"] == len(ops)
        assert payload["adapt_vqe"]["num_parameters"] == len(ops)

    def test_settings_has_required_hh_keys(self) -> None:
        for L in [2, 3, 4, 5]:
            payload = _make_staged_export_payload(L=L)
            settings = payload["settings"]
            for key in ("L", "t", "u", "omega0", "g_ep", "n_ph_max",
                        "boson_encoding", "ordering", "boundary",
                        "sector_n_up", "sector_n_dn"):
                assert key in settings, f"Missing settings key {key} for L={L}"
            assert settings["L"] == L

    def test_initial_state_has_amplitudes(self) -> None:
        payload = _make_staged_export_payload()
        assert "amplitudes_qn_to_q0" in payload["initial_state"]
        assert "nq_total" in payload["initial_state"]


class TestStagedExportRejectsIncomplete:
    """Verify the replay parser rejects incomplete staged exports."""

    def test_missing_operators_rejected(self) -> None:
        payload = _make_staged_export_payload()
        del payload["adapt_vqe"]["operators"]
        with pytest.raises(ValueError, match="adapt_vqe.operators"):
            _extract_adapt_operator_theta_sequence(payload)

    def test_missing_optimal_point_rejected(self) -> None:
        payload = _make_staged_export_payload()
        del payload["adapt_vqe"]["optimal_point"]
        with pytest.raises(ValueError, match="adapt_vqe.optimal_point"):
            _extract_adapt_operator_theta_sequence(payload)

    def test_length_mismatch_rejected(self) -> None:
        payload = _make_staged_export_payload(operators=["a"], optimal_point=[0.1, 0.2])
        with pytest.raises(ValueError, match="Length mismatch"):
            _extract_adapt_operator_theta_sequence(payload)


class TestArbitraryLRoundTrip:
    """Verify round-trip for L=2,3,4,5."""

    @pytest.mark.parametrize("L", [2, 3, 4, 5])
    def test_roundtrip_for_L(self, L: int) -> None:
        n_ops = L + 1
        ops = [f"op_{i}" for i in range(n_ops)]
        theta = [float(i) * 0.1 for i in range(n_ops)]
        payload = _make_staged_export_payload(L=L, operators=ops, optimal_point=theta)

        labels, theta_out = _extract_adapt_operator_theta_sequence(payload)
        assert labels == ops
        assert np.allclose(theta_out, theta)

        fam, src = _resolve_family_from_metadata(payload)
        assert fam == "pool_a"

    @pytest.mark.parametrize("L", [2, 3, 4, 5])
    def test_settings_sector_half_filling(self, L: int) -> None:
        payload = _make_staged_export_payload(L=L)
        s = payload["settings"]
        assert s["sector_n_up"] == (L + 1) // 2
        assert s["sector_n_dn"] == L // 2


class TestWriteStateBundleRoundTrip:
    """Test the active handoff bundle writer produces replay-compatible JSON."""

    def test_write_and_read_back(self, tmp_path: Path) -> None:
        """Write a state bundle via write_handoff_state_bundle and verify it round-trips."""
        cfg = HandoffStateBundleConfig(
            L=2,
            t=1.0,
            U=4.0,
            dv=0.0,
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            sector_n_up=1,
            sector_n_dn=1,
        )

        nq = 2 * 2 + 2  # L=2, n_ph_max=1, binary
        psi = np.zeros(1 << nq, dtype=complex)
        psi[0] = 1.0

        parameterization, logical_theta, runtime_theta = _make_runtime_layout()

        out_path = tmp_path / "stage_export.json"
        write_handoff_state_bundle(
            path=out_path,
            psi_state=psi,
            cfg=cfg,
            source="A_probe_final",
            exact_energy=-2.0,
            energy=-1.9,
            delta_E_abs=0.1,
            relative_error_abs=0.05,
            meta={"run_id": "A_probe", "budget_name": "probe"},
            adapt_operators=["op_x", "op_y"],
            adapt_optimal_point=runtime_theta,
            adapt_logical_optimal_point=logical_theta,
            adapt_parameterization=parameterization,
            adapt_logical_num_parameters=2,
            adapt_pool_type="pool_a",
            continuation_mode="phase1_v1",
            continuation_scaffold={"num_parameters": 3, "post_prune": True},
            continuation_details={
                "score_version": "phase3_reduced_rerank_v1",
                "runtime_split_summary": {"mode": "shortlist_pauli_children_v1", "selected_child_count": 1},
                "phase1_feature_rows": [{"should_not": "override_reserved_fields"}],
            },
            optimizer_memory={"version": "phase2_optimizer_memory_v1", "parameter_count": 3, "available": True},
            selected_generator_metadata=[
                {
                    "generator_id": "gen:1",
                    "family_id": "paop_lf_std",
                    "template_id": "paop_lf_std|macro|terms2|sites0,1|bos0|ferm1",
                    "candidate_label": "op_x",
                    "support_qubits": [0, 1],
                    "support_sites": [0, 1],
                    "support_site_offsets": [0, 1],
                    "is_macro_generator": True,
                    "split_policy": "preserve",
                }
            ],
            motif_library={
                "library_version": "phase3_motif_library_v1",
                "source_tag": "unit_test",
                "source_num_sites": 2,
                "ordering": "blocked",
                "boson_encoding": "binary",
                "records": [
                    {
                        "motif_id": "motif:1",
                        "family_id": "paop_lf_std",
                        "template_id": "paop_lf_std|macro|terms2|sites0,1|bos0|ferm1",
                        "source_num_sites": 2,
                        "relative_order": 0,
                        "support_site_offsets": [0, 1],
                        "mean_theta": 0.1,
                        "mean_abs_theta": 0.1,
                        "sign_hint": 1,
                        "generator_ids": ["gen:1"],
                    }
                ],
            },
            motif_usage={"source_tag": "unit_test", "selected_count": 1},
            symmetry_mitigation={"mode": "verify_only", "executed": True, "passed": True},
            rescue_history=[{"enabled": False, "triggered": False, "reason": "disabled"}],
            pre_prune_scaffold={"operators": ["op_x", "op_y", "op_z"]},
            prune_summary={"executed": True, "accepted_count": 1},
            ansatz_input_state=np.array(psi, copy=True),
            ansatz_input_state_source="warm_start_hva",
            ansatz_input_state_handoff_state_kind="prepared_state",
        )

        # Read back and validate
        payload = json.loads(out_path.read_text(encoding="utf-8"))

        # Canonical replay fields
        labels, theta = _extract_adapt_operator_theta_sequence(payload)
        layout = _extract_payload_parameterization_layout(payload)
        assert labels == ["op_x", "op_y"]
        assert np.allclose(theta, runtime_theta)
        assert layout is not None
        assert int(layout.logical_parameter_count) == 2
        assert int(layout.runtime_parameter_count) == 3
        assert payload["adapt_vqe"]["ansatz_depth"] == 2
        assert payload["adapt_vqe"]["num_parameters"] == 3
        assert payload["adapt_vqe"]["logical_num_parameters"] == 2
        assert np.allclose(payload["adapt_vqe"]["logical_optimal_point"], logical_theta)
        assert payload["adapt_vqe"]["pool_type"] == "pool_a"

        # Family resolution
        fam, src = _resolve_family_from_metadata(payload)
        assert fam == "pool_a"

        # Settings
        assert payload["settings"]["L"] == 2
        assert payload["settings"]["sector_n_up"] == 1
        assert payload["settings"]["sector_n_dn"] == 1

        # Initial state
        assert "amplitudes_qn_to_q0" in payload["initial_state"]
        assert payload["initial_state"]["nq_total"] == nq
        assert payload["ansatz_input_state"]["source"] == "warm_start_hva"
        assert payload["ansatz_input_state"]["handoff_state_kind"] == "prepared_state"
        assert payload["ansatz_input_state"]["nq_total"] == nq
        assert payload["continuation"]["mode"] == "phase1_v1"
        assert payload["continuation"]["optimizer_memory"]["parameter_count"] == 3
        assert payload["continuation"]["selected_generator_metadata"][0]["generator_id"] == "gen:1"
        assert payload["continuation"]["motif_library"]["records"][0]["motif_id"] == "motif:1"
        assert payload["continuation"]["symmetry_mitigation"]["mode"] == "verify_only"
        assert payload["continuation"]["rescue_history"][0]["reason"] == "disabled"
        assert payload["continuation"]["score_version"] == "phase3_reduced_rerank_v1"
        assert payload["continuation"]["runtime_split_summary"]["selected_child_count"] == 1
        assert payload["continuation"]["phase1_feature_rows"][0]["should_not"] == "override_reserved_fields"
        assert payload["adapt_vqe"]["pre_prune_scaffold"]["operators"] == ["op_x", "op_y", "op_z"]
        assert payload["adapt_vqe"]["prune_summary"]["executed"] is True

    def test_write_sparse_bundle_without_continuation_preserves_legacy_load(self, tmp_path: Path) -> None:
        cfg = HandoffStateBundleConfig(
            L=2,
            t=1.0,
            U=4.0,
            dv=0.0,
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            sector_n_up=1,
            sector_n_dn=1,
        )

        nq = 2 * 2 + 2
        psi = np.zeros(1 << nq, dtype=complex)
        psi[0] = 1.0

        out_path = tmp_path / "legacy_shape.json"
        write_handoff_state_bundle(
            path=out_path,
            psi_state=psi,
            cfg=cfg,
            source="A_probe_final",
            exact_energy=-2.0,
            energy=-1.9,
            delta_E_abs=0.1,
            relative_error_abs=0.05,
            adapt_operators=["op_x"],
            adapt_optimal_point=[0.1],
            adapt_pool_type="pool_a",
        )

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert "continuation" not in payload
        assert "ansatz_input_state" not in payload

        labels, theta = _extract_adapt_operator_theta_sequence(payload)
        assert labels == ["op_x"]
        assert np.allclose(theta, [0.1])

        fam, src = _resolve_family_from_metadata(payload)
        assert fam == "pool_a"
        assert src == "adapt_vqe.pool_type"

        kind, provenance = _infer_handoff_state_kind(payload)
        assert kind == "prepared_state"
        assert provenance == "inferred_source"


# ---------------------------------------------------------------------------
# Provenance tests
# ---------------------------------------------------------------------------


class TestStagedExportProvenance:
    """Verify staged exports carry explicit handoff_state_kind provenance."""

    def test_staged_export_has_prepared_state_kind(self) -> None:
        payload = _make_staged_export_payload()
        assert payload["initial_state"]["handoff_state_kind"] == "prepared_state"

    def test_provenance_is_explicit(self) -> None:
        payload = _make_staged_export_payload()
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "prepared_state"
        assert src == "explicit"

    @pytest.mark.parametrize("L", [2, 3, 4, 5])
    def test_provenance_present_for_all_L(self, L: int) -> None:
        payload = _make_staged_export_payload(L=L)
        assert "handoff_state_kind" in payload["initial_state"]
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "prepared_state"
        assert src == "explicit"

    def test_legacy_payload_without_provenance_infers_from_source(self) -> None:
        """Old payloads without handoff_state_kind can be inferred from source."""
        payload = _make_staged_export_payload()
        del payload["initial_state"]["handoff_state_kind"]
        kind, src = _infer_handoff_state_kind(payload)
        # Source is "A_probe_final" which ends with "_final" -> prepared_state
        assert kind == "prepared_state"
        assert src == "inferred_source"

    def test_write_state_bundle_stamps_provenance(self, tmp_path: Path) -> None:
        """Verify write_handoff_state_bundle stamps handoff_state_kind."""
        cfg = HandoffStateBundleConfig(
            L=2,
            t=1.0,
            U=4.0,
            dv=0.0,
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            sector_n_up=1,
            sector_n_dn=1,
        )

        nq = 2 * 2 + 2
        psi = np.zeros(1 << nq, dtype=complex)
        psi[0] = 1.0

        out_path = tmp_path / "provenance_test.json"
        write_handoff_state_bundle(
            path=out_path,
            psi_state=psi,
            cfg=cfg,
            source="A_probe_final",
            exact_energy=-2.0,
            energy=-1.9,
            delta_E_abs=0.1,
            relative_error_abs=0.05,
            handoff_state_kind="prepared_state",
            adapt_operators=["op_x"],
            adapt_optimal_point=[0.1],
            adapt_pool_type="pool_a",
        )

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["initial_state"]["handoff_state_kind"] == "prepared_state"
        kind, src = _infer_handoff_state_kind(payload)
        assert kind == "prepared_state"
        assert src == "explicit"
