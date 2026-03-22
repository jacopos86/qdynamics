from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.ansatz_parameterization import (
    build_parameter_layout,
    deserialize_layout,
    expand_legacy_logical_theta,
    project_runtime_theta_block_mean,
    runtime_indices_for_logical_indices,
    runtime_insert_position,
    serialize_layout,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


def _sample_terms() -> list[AnsatzTerm]:
    return [
        AnsatzTerm(
            label="g0",
            polynomial=PauliPolynomial(
                "JW",
                [PauliTerm(2, ps="xx", pc=1.0), PauliTerm(2, ps="zz", pc=0.5)],
            ),
        ),
        AnsatzTerm(
            label="g1",
            polynomial=PauliPolynomial(
                "JW",
                [PauliTerm(2, ps="ee", pc=1.0), PauliTerm(2, ps="xy", pc=1.0)],
            ),
        ),
    ]


def test_build_parameter_layout_tracks_logical_and_runtime_indices() -> None:
    layout = build_parameter_layout(_sample_terms())

    assert int(layout.logical_parameter_count) == 2
    assert int(layout.runtime_parameter_count) == 3
    assert int(layout.blocks[0].runtime_start) == 0
    assert int(layout.blocks[0].runtime_count) == 2
    assert int(layout.blocks[1].runtime_start) == 2
    assert int(layout.blocks[1].runtime_count) == 1
    assert runtime_insert_position(layout, 0) == 0
    assert runtime_insert_position(layout, 1) == 2
    assert runtime_insert_position(layout, 2) == 3
    assert runtime_indices_for_logical_indices(layout, [0, 1]) == [0, 1, 2]


def test_expand_and_project_runtime_theta_block_means() -> None:
    layout = build_parameter_layout(_sample_terms())

    theta_runtime = expand_legacy_logical_theta([0.3, -0.4], layout)
    assert np.allclose(theta_runtime, [0.3, 0.3, -0.4])

    theta_logical = project_runtime_theta_block_mean([0.2, 0.4, -0.1], layout)
    assert np.allclose(theta_logical, [0.3, -0.1])


def test_serialize_deserialize_layout_roundtrip() -> None:
    layout = build_parameter_layout(_sample_terms())
    restored = deserialize_layout(serialize_layout(layout))

    assert restored.mode == "per_pauli_term_v1"
    assert restored.term_order == layout.term_order
    assert restored.ignore_identity is True
    assert int(restored.logical_parameter_count) == int(layout.logical_parameter_count)
    assert int(restored.runtime_parameter_count) == int(layout.runtime_parameter_count)
    assert [block.candidate_label for block in restored.blocks] == ["g0", "g1"]
    assert [spec.pauli_exyz for spec in restored.blocks[0].terms] == ["xx", "zz"]
    assert [spec.pauli_exyz for spec in restored.blocks[1].terms] == ["xy"]
