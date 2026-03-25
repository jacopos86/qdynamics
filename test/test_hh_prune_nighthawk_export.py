from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.hardcoded.hh_prune_nighthawk as prune
from src.quantum.ansatz_parameterization import build_parameter_layout, project_runtime_theta_block_mean


_SOURCE_JSON = REPO_ROOT / "artifacts/json/hh_prune_nighthawk_aggressive_5op.json"


def _source_payload() -> dict[str, object]:
    return prune._load_payload(_SOURCE_JSON)


def test_build_variant_scaffold_source_order_contract() -> None:
    payload = _source_payload()
    spec = prune._FIXED_SCAFFOLD_VARIANTS[0]

    scaffold_ops, theta_runtime, runtime_labels = prune._build_variant_scaffold(
        source_payload=payload,
        spec=spec,
    )

    assert [op.label for op in scaffold_ops] == [
        "uccsd_ferm_lifted::uccsd_sing(alpha:0->1)",
        "uccsd_ferm_lifted::uccsd_sing(beta:2->3)",
        "paop_lf_full:paop_dbl_p(site=1->phonon=1)",
        "paop_full:paop_hopdrag(0,1)::child_set[0,2]",
        "paop_full:paop_disp(site=0)",
    ]
    assert runtime_labels == [
        "eeeexy",
        "eeyxee",
        "yeeeee",
        "yezeze",
        "yeyyee",
        "eyeeez",
        "eyezee",
    ]
    assert theta_runtime.shape == (7,)
    assert not np.allclose(theta_runtime, 0.0)


def test_build_variant_scaffold_disp_first_contract() -> None:
    payload = _source_payload()
    spec = prune._FIXED_SCAFFOLD_VARIANTS[1]

    scaffold_ops, theta_runtime, runtime_labels = prune._build_variant_scaffold(
        source_payload=payload,
        spec=spec,
    )

    assert [op.label for op in scaffold_ops] == [
        "paop_full:paop_disp(site=0)",
        "paop_lf_full:paop_dbl_p(site=1->phonon=1)",
        "paop_full:paop_hopdrag(0,1)::child_set[0,2]",
        "uccsd_ferm_lifted::uccsd_sing(alpha:0->1)",
        "uccsd_ferm_lifted::uccsd_sing(beta:2->3)",
    ]
    assert runtime_labels == [
        "eyeeez",
        "eyezee",
        "yeeeee",
        "yezeze",
        "yeyyee",
        "eeeexy",
        "eeyxee",
    ]
    assert list(spec.source_order_runtime_indices) == [5, 6, 2, 3, 4, 0, 1]
    assert theta_runtime.shape == (7,)


def test_build_export_payload_marks_fixed_scaffold_metadata() -> None:
    payload = _source_payload()
    spec = prune._FIXED_SCAFFOLD_VARIANTS[1]
    scaffold_ops, theta_runtime, runtime_labels = prune._build_variant_scaffold(
        source_payload=payload,
        spec=spec,
    )
    layout = build_parameter_layout(scaffold_ops, sort_terms=False)
    theta_logical = project_runtime_theta_block_mean(theta_runtime, layout)
    optimization = prune.LockedScaffoldOptimizationResult(
        layout=layout,
        theta_runtime=[float(x) for x in theta_runtime.tolist()],
        theta_logical=[float(x) for x in theta_logical.tolist()],
        energy=0.159,
        exact_energy=0.1587,
        abs_delta_e=3.0e-4,
        success=True,
        message="ok",
        nfev=12,
        method="POWELL",
        elapsed_s=0.5,
    )

    export_payload = prune._build_export_payload(
        source_payload=payload,
        spec=spec,
        settings=dict(payload.get("settings", {})),
        exact_energy=0.1587,
        optimization=optimization,
        scaffold_ops=scaffold_ops,
        runtime_labels=runtime_labels,
        source_artifact_json=_SOURCE_JSON,
    )

    adapt_vqe = export_payload["adapt_vqe"]
    assert adapt_vqe["pool_type"] == "fixed_scaffold_locked"
    assert adapt_vqe["structure_locked"] is True
    assert adapt_vqe["fixed_scaffold_kind"] == "hh_nighthawk_circuit_optimized_7term_v1"
    assert len(adapt_vqe["optimal_point"]) == 7
    assert len(adapt_vqe["logical_optimal_point"]) == 5
    assert adapt_vqe["fixed_scaffold_metadata"]["term_order_id"] == "disp_first"
    assert adapt_vqe["fixed_scaffold_metadata"]["source_order_runtime_indices"] == [5, 6, 2, 3, 4, 0, 1]
    assert adapt_vqe["fixed_scaffold_metadata"]["runtime_term_labels_exyz"] == runtime_labels
    assert export_payload["ansatz_input_state"]["handoff_state_kind"] == "reference_state"
    assert export_payload["initial_state"]["handoff_state_kind"] == "prepared_state"
