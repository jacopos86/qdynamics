from __future__ import annotations

import numpy as np
import pytest

from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
)
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    GeneratorParameterBlock,
    RotationTermSpec,
)


def _layout(labels: list[str]) -> AnsatzParameterLayout:
    return AnsatzParameterLayout(
        mode="per_pauli_term_v1",
        term_order="sorted",
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        blocks=tuple(
            GeneratorParameterBlock(
                candidate_label=label,
                logical_index=index,
                runtime_start=index,
                terms=(
                    RotationTermSpec(
                        pauli_exyz="x",
                        coeff_real=1.0,
                        nq=1,
                    ),
                ),
            )
            for index, label in enumerate(labels)
        ),
    )


def _batch3_summary() -> dict[str, object]:
    return {
        "joint_linear_solve_policy_effective": (
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
        ),
        "geometry_workspace": {"active_indices": [0]},
        "G_AA_raw": [[1.0]],
        "G_BB_raw": np.eye(3, dtype=float).tolist(),
        "selected_subset": [
            {"position_id": 1},
            {"position_id": 1},
            {"position_id": 1},
        ],
    }


def test_batch3_accepted_step_uses_certified_subset_when_commit_list_is_partial() -> None:
    realized = adapt_pipeline._accepted_sr_v2_joint_coordinate_step(
        selector_summary=_batch3_summary(),
        pre_parameter_count=1,
        positions_in_commit_order=[1],
        theta_before_refit=np.asarray([0.2, 0.0, 0.0, 0.0]),
        theta_after_refit=np.asarray([0.3, 0.1, -0.2, 0.4]),
        post_layout=_layout(["old", "a", "b", "c"]),
    )

    assert realized == pytest.approx([0.1, 0.1, -0.2, 0.4])


def test_batch3_accepted_step_rejects_complete_order_drift() -> None:
    with pytest.raises(RuntimeError, match="commit order disagrees"):
        adapt_pipeline._accepted_sr_v2_joint_coordinate_step(
            selector_summary=_batch3_summary(),
            pre_parameter_count=1,
            positions_in_commit_order=[0, 1, 1],
            theta_before_refit=np.asarray([0.2, 0.0, 0.0, 0.0]),
            theta_after_refit=np.asarray([0.3, 0.1, -0.2, 0.4]),
            post_layout=_layout(["old", "a", "b", "c"]),
        )


def test_singleton_accepted_step_restores_serialized_zero_active_gram() -> None:
    realized = adapt_pipeline._accepted_sr_v2_joint_coordinate_step(
        selector_summary={
            "joint_linear_solve_policy_effective": (
                JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
            ),
            "geometry_workspace": {"active_indices": []},
            "G_AA_raw": [],
            "G_BB_raw": [[1.0]],
            "selected_subset": [{"position_id": 0}],
        },
        pre_parameter_count=0,
        positions_in_commit_order=[0],
        theta_before_refit=np.asarray([0.0]),
        theta_after_refit=np.asarray([0.125]),
        post_layout=_layout(["candidate"]),
    )

    assert realized == pytest.approx([0.125])
