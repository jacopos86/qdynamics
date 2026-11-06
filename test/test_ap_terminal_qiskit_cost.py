from pipelines.time_dynamics.diagnostics.ap_terminal_qiskit_cost import (
    _drive_aligned_enabled,
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


def test_drive_aligned_reconstruction_uses_requested_route_on_resume() -> None:
    payload = {
        "drive_aligned_ansatz": {
            "applied": False,
            "reason": "resume_terminal_support_already_contains_drive_alignment",
        }
    }
    metadata = {"drive_aligned_ansatz_requested": True}

    assert _drive_aligned_enabled(payload, metadata) is True


def test_drive_aligned_reconstruction_respects_explicit_disabled_route() -> None:
    payload = {"drive_aligned_ansatz": {"applied": True}}
    metadata = {"drive_aligned_ansatz_requested": False}

    assert _drive_aligned_enabled(payload, metadata) is False
