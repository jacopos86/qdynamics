from pipelines.time_dynamics.diagnostics.ap_terminal_qiskit_cost import (
    _structural_occurrence_base_label,
)


def test_structural_occurrence_base_label_strips_exchange_child_suffix() -> None:
    label = "hop_layer::r0::eeeexx::insr6c3o0::r0::eeeexx"
    assert _structural_occurrence_base_label(label) == "hop_layer::r0::eeeexx"


def test_structural_occurrence_base_label_strips_avqds_child_suffix() -> None:
    label = "phonon_layer::r0::ezeeee::avqds2c17o0::r0::ezeeee"
    assert _structural_occurrence_base_label(label) == "phonon_layer::r0::ezeeee"


def test_structural_occurrence_base_label_preserves_ordinary_label() -> None:
    label = "phonon_layer::r0::ezeeee"
    assert _structural_occurrence_base_label(label) == label
