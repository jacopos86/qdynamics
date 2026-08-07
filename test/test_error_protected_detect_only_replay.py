from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.error_protected import adapt_detect_only_replay as replay
from pipelines.error_protected.contracts import (
    DetectionReplayInput,
    ErrorDetectionConfig,
    RawObservableBundle,
)
from pipelines.error_protected.raw_acquisition import validate_detect_only_request
from pipelines.error_protected.reporting import build_comparison_payload
from pipelines.error_protected.shot_postprocess import (
    sector_filter_unavailable,
    spin_orbital_index_sets,
    summarize_sector_audit,
)


def _valid_backend_scheduled_request(tmp_path: Path, **kwargs) -> DetectionReplayInput:
    detection = kwargs.pop("detection", ErrorDetectionConfig(mode="off"))
    values = {
        "artifact_json": tmp_path / "adapt.json",
        "output_json": tmp_path / "sidecar.json",
        "noise_mode": "backend_scheduled",
        "backend_name": "fake_backend",
        "use_fake_backend": True,
        "detection": detection,
    }
    values.update(kwargs)
    return DetectionReplayInput(**values)


def _fake_runtime_input(*, ordering: str = "blocked", sector: tuple[int, int] = (1, 1)):
    request = SimpleNamespace(
        problem_key="hh",
        num_sites=2,
        ordering=ordering,
        t=1.0,
        u=4.0,
        dv=0.0,
        omega0=1.0,
        g_ep=1.0,
        n_ph_max=1,
        boson_encoding="binary",
    )
    layout = SimpleNamespace(
        total_qubits=4,
        fermion_qubits=4,
        boson_qubits=0,
        ordering=ordering,
    )
    sector_obj = SimpleNamespace(
        num_particles=sector,
        label="half_filled_fermion_sector",
        comparison_space_label="fermion_register",
    )
    resolved_problem = SimpleNamespace(
        family_key="hh",
        request=request,
        layout=layout,
        sector=sector_obj,
        default_num_particles=sector,
    )
    base_layout = SimpleNamespace(runtime_parameter_count=0, logical_parameter_count=0)
    psi_ref = np.zeros(2**4, dtype=complex)
    psi_ref[0] = 1.0
    return SimpleNamespace(
        resolved_problem=resolved_problem,
        psi_ref=psi_ref,
        base_layout=base_layout,
        theta_runtime=np.array([], dtype=float),
        exact_energy=-1.25,
    )


def test_sector_audit_counts_blocked_hh_sector() -> None:
    record = SimpleNamespace(
        basis_label="ZZZZ",
        num_qubits=4,
        measured_logical_qubits=(0, 1, 2, 3),
        counts={"0101": 3, "1111": 2},
        shots_completed=5,
    )

    estimate = summarize_sector_audit(
        (record,),
        num_sites=2,
        ordering="blocked",
        sector_n_up=1,
        sector_n_dn=1,
        min_accepted_shots=1,
        strict=True,
    )

    assert estimate.status == "ok"
    assert estimate.total_shots == 5
    assert estimate.accepted_shots == 3
    assert estimate.rejected_shots == 2
    assert estimate.acceptance_rate == pytest.approx(0.6)
    assert estimate.per_group["sector_counts"] == {"1,1": 3, "2,2": 2}


def test_sector_ordering_must_be_supported() -> None:
    with pytest.raises(ValueError, match="Unsupported spin-orbital ordering"):
        spin_orbital_index_sets(2, "guessed")


def test_sector_audit_empty_records_are_missing_observability() -> None:
    estimate = summarize_sector_audit(
        (),
        num_sites=2,
        ordering="blocked",
        sector_n_up=1,
        sector_n_dn=1,
        strict=True,
    )

    comparisons = build_comparison_payload(
        energy_bundle=None,
        sector_audit_estimate=estimate,
        exact_energy=None,
    )

    assert estimate.status == "failed_missing_detector_observability"
    assert estimate.total_shots == 0
    assert comparisons["sector_acceptance_rate"] is None


def test_sector_filter_unavailable_carries_no_corrected_energy() -> None:
    estimate = sector_filter_unavailable(strict=True)

    assert estimate.name == "sector_filter"
    assert estimate.status == "failed_missing_detector_observability"
    assert estimate.accepted_mean is None
    assert estimate.corrected_mean is None
    assert estimate.per_group["applied"] is False
    assert estimate.per_group["postselection_applied"] is False
    assert estimate.per_group["correction_applied"] is False


def test_validate_request_rejects_sidecar_path_collision(tmp_path: Path) -> None:
    request = _valid_backend_scheduled_request(tmp_path, output_json=tmp_path / "adapt.json")

    with pytest.raises(ValueError, match="output_json must not equal"):
        validate_detect_only_request(request)


def test_validate_request_rejects_non_off_correction(tmp_path: Path) -> None:
    request = _valid_backend_scheduled_request(
        tmp_path,
        detection=ErrorDetectionConfig(
            mode="off",
            postprocess_correction_mode="lookup",  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(ValueError, match="correction mode 'off'"):
        validate_detect_only_request(request)


def test_validate_request_rejects_negative_min_accepted_shots(tmp_path: Path) -> None:
    request = _valid_backend_scheduled_request(
        tmp_path,
        detection=ErrorDetectionConfig(mode="off", min_accepted_shots=-1),
    )

    with pytest.raises(ValueError, match="min_accepted_shots"):
        validate_detect_only_request(request)


def test_cli_sector_detect_alias_maps_to_sector_audit(tmp_path: Path) -> None:
    args = replay.build_parser().parse_args(
        [
            "--artifact-json",
            str(tmp_path / "adapt.json"),
            "--output-json",
            str(tmp_path / "sidecar.json"),
            "--noise-mode",
            "backend_scheduled",
            "--execution-surface",
            "raw_measurement_v1",
            "--backend-name",
            "fake_backend",
            "--use-fake-backend",
            "--detection-mode",
            "sector_detect",
            "--raw-grouping-mode",
            "qwc_basis_cover_reuse",
            "--min-acceptance-rate",
            "0.25",
        ]
    )

    request = replay.resolve_detection_replay_input(args)

    assert request.detection.mode == "sector_audit"
    assert request.detection.min_acceptance_rate == pytest.approx(0.25)
    assert request.execution_surface == "raw_measurement_v1"
    assert request.raw_grouping_mode == "qwc_basis_cover_reuse"


def test_resolve_detection_context_artifact_metadata_wins_and_conflicts_fail_closed() -> None:
    runtime_input = _fake_runtime_input(ordering="blocked", sector=(1, 1))
    detection = ErrorDetectionConfig(
        mode="sector_audit",
        sector_n_up=2,
        sector_n_dn=1,
        ordering="interleaved",
    )

    context = replay.resolve_detection_context(runtime_input, detection)

    assert context.ordering == "blocked"
    assert context.ordering_source == "artifact"
    assert (context.sector_n_up, context.sector_n_dn) == (1, 1)
    assert context.sector_source == "artifact"
    assert "cli_ordering_conflicts_with_artifact_ordering" in context.errors
    assert "cli_sector_conflicts_with_artifact_sector" in context.errors


def test_sector_filter_replay_preserves_audit_and_raw_energy(monkeypatch, tmp_path: Path) -> None:
    artifact = tmp_path / "adapt.json"
    artifact.write_text('{"canonical": true}\n', encoding="utf-8")
    output = tmp_path / "detect.json"
    runtime_input = _fake_runtime_input(ordering="blocked", sector=(1, 1))
    plan = SimpleNamespace(nq=4, plan_digest="plan", structure_digest="structure")

    def fake_acquire_raw_bundle(*, observable_family: str, **_kwargs):
        if observable_family == "energy_raw":
            energy_record = SimpleNamespace(
                basis_label="XX",
                num_qubits=4,
                measured_logical_qubits=(0, 1),
                counts={"00": 6},
                shots_completed=6,
            )
            return RawObservableBundle(
                observable_family="energy_raw",
                mean=-1.0,
                stderr=0.1,
                total_shots=6,
                record_count=1,
                records=(energy_record,),
                diagnostics={"source": "test"},
            )
        sector_record = SimpleNamespace(
            basis_label="ZZZZ",
            num_qubits=4,
            measured_logical_qubits=(0, 1, 2, 3),
            counts={"0101": 5, "1111": 1},
            shots_completed=6,
        )
        return RawObservableBundle(
            observable_family="sector_full_register_z",
            mean=None,
            stderr=None,
            total_shots=6,
            record_count=1,
            records=(sector_record,),
            diagnostics={"source": "test"},
        )

    monkeypatch.setattr(replay, "load_replay_runtime_input", lambda *_args, **_kwargs: runtime_input)
    monkeypatch.setattr(replay, "build_parameterized_plan_from_runtime_input", lambda *_args: plan)
    monkeypatch.setattr(replay, "build_energy_observable", lambda *_args: object())
    monkeypatch.setattr(replay, "build_full_register_z_observable", lambda *_args: object())
    monkeypatch.setattr(replay, "acquire_raw_bundle", fake_acquire_raw_bundle)

    request = DetectionReplayInput(
        artifact_json=artifact,
        output_json=output,
        noise_mode="backend_scheduled",
        backend_name="fake_backend",
        use_fake_backend=True,
        detection=ErrorDetectionConfig(
            mode="sector_filter",
            strict=True,
            min_accepted_shots=1,
        ),
    )

    summary = replay.run_detect_only_replay(request)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert artifact.read_text(encoding="utf-8") == '{"canonical": true}\n'
    assert summary["estimates"]["energy_raw"]["raw_mean"] == -1.0
    assert payload["raw_summary"]["total_shots"] == 6
    assert payload["raw_summary"]["record_count"] == 1
    assert payload["raw_summary"]["observable_families"] == ["energy_raw"]
    assert payload["estimates"]["energy_raw"]["raw_mean"] == -1.0
    assert payload["comparisons"]["accepted_minus_raw"] is None
    assert payload["comparisons"]["accepted_minus_exact_reference"] is None
    assert payload["estimates"]["sector_audit"]["status"] == "ok"
    assert payload["estimates"]["sector_audit"]["accepted_shots"] == 5
    assert payload["estimates"]["sector_filter"]["status"] == "failed_missing_detector_observability"
    assert payload["estimates"]["sector_filter"]["accepted_mean"] is None
    assert payload["estimates"]["sector_filter"]["corrected_mean"] is None
    assert payload["estimates"]["sector_filter"]["per_group"]["filter_application_enabled"] is False
    assert payload["diagnostics"]["correction"]["applied"] is False
