from __future__ import annotations

from dataclasses import asdict

from pipelines.scaffold.hh_continuation_generators import build_generator_metadata
from pipelines.static_adapt.route_a_child_padding import (
    ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
    RouteAChildPaddingConfig,
)
from pipelines.static_adapt.selector_measurement_proxy import (
    ControllerMeasurementWorkRecordRuntime,
    _common_exposure_probe_payload_for_records,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def test_common_exposure_applies_active_padding_projection_before_counting() -> None:
    polynomial = PauliPolynomial(
        "JW",
        [
            PauliTerm(8, ps="eeexeeee", pc=1.0),
            PauliTerm(8, ps="exeeeeee", pc=1.0),
        ],
    )
    parent = AnsatzTerm(label="parent", polynomial=polynomial)
    symmetry_spec = {
        "particle_number_mode": "preserving",
        "spin_sector_mode": "preserving",
        "phonon_number_mode": "not_conserved",
        "hard_guard": True,
    }
    metadata = asdict(
        build_generator_metadata(
            label="parent",
            polynomial=polynomial,
            family_id="test",
            num_sites=2,
            ordering="blocked",
            qpb=2,
            symmetry_spec=symmetry_spec,
            fixed_num_particles=(1, 1),
        )
    )
    runtime = ControllerMeasurementWorkRecordRuntime(
        pool=[parent],
        pool_generator_registry={"parent": metadata},
        phase3_enabled=True,
        pool_symmetry_specs=[symmetry_spec],
        problem_key="hh",
        num_sites=2,
        ordering="blocked",
        qpb=2,
        phase3_runtime_split_mode_key="shortlist_pauli_children_v1",
        phase3_runtime_split_selection_mode_key="archival_child_set_forward_v1",
        phase3_runtime_split_child_set_symmetry_policy_key="hard_guard",
        phase3_runtime_split_max_subset_size_value=1,
        phase3_runtime_split_subset_sizes_value=(1,),
        fixed_num_particles=(1, 1),
        child_padding_config=RouteAChildPaddingConfig(
            policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
            problem_key="hh",
            num_sites=2,
            n_ph_max=2,
            boson_encoding="binary",
            total_register_width=8,
        ),
    )

    payload = _common_exposure_probe_payload_for_records(
        [{"candidate_term": parent, "candidate_pool_index": 0, "position_id": 0}],
        runtime=runtime,
        expand_runtime_split=True,
    )

    assert payload["child_padding_policy"] == ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1
    assert payload["child_padding_projection_input_count"] == 2
    assert payload["child_padding_projection_output_count"] == 2
    assert payload["common_parent_candidate_count"] == 1
    assert payload["common_expanded_candidate_count"] == 3
    assert payload["common_exposure_operator_probe_count"] == 3
