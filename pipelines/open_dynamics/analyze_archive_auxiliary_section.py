#!/usr/bin/env python3
"""Audit the reciprocal archive section on stored Paper V development states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from paper5.stability.archive_auxiliary_memory import (
    ArchiveAuxiliaryFrame,
    build_archive_auxiliary_frame,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.krylov_memory_closure import (
    build_krylov_closure_construction,
)
from paper5.stability.matrix_reference import (
    pauli_repaired_closed_scalar_rhs,
)

RUN_ID = "paper_v_archive_auxiliary_section_cutoff12_16_20260804_v1"
DEFAULT_INPUT = Path(
    "output/local_runs/"
    "paper_v_mixed_tangent_closure_identifiability_cutoff16_t20_20260804_v2/"
    "mixed_tangent_closure_identifiability.npz"
)
DEFAULT_OUTPUT = Path("output/local_runs") / RUN_ID


class _FixedDriveParameters:
    """Delegate physical parameters while fixing one sampled drive value."""

    def __init__(self, parameters: DimerParameters, drive_value: float) -> None:
        self._parameters = parameters
        self._drive_value = float(drive_value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parameters, name)

    def drive_difference(self, time: float) -> float:
        del time
        return self._drive_value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _block_diagnostics(frame: ArchiveAuxiliaryFrame) -> dict[str, object]:
    return {
        name: {
            "resolved_skew_residual": blocks.resolved_skew_residual,
            "hidden_skew_residual": blocks.hidden_skew_residual,
            "reciprocity_residual": blocks.reciprocity_residual,
        }
        for name, blocks in (
            ("static", frame.static_blocks),
            ("drive", frame.drive_blocks),
        )
    }


def _path_summary(
    raw_lift_relative: np.ndarray,
    section_relative: np.ndarray,
    centered_relative: np.ndarray,
    centered_absolute: np.ndarray,
    centered_source_norm: np.ndarray,
    coupling_rank: np.ndarray,
    hidden_section_norm: np.ndarray,
    *,
    algebraic_noise_floor: float,
) -> dict[str, object]:
    floored_relative = centered_absolute / np.maximum(
        centered_source_norm,
        algebraic_noise_floor,
    )
    return {
        "maximum_raw_lift_relative_residual": float(np.max(raw_lift_relative)),
        "maximum_raw_section_relative_residual": float(np.max(section_relative)),
        "maximum_centered_section_relative_residual": float(
            np.max(centered_relative)
        ),
        "maximum_centered_section_absolute_residual": float(
            np.max(centered_absolute)
        ),
        "maximum_noise_floored_centered_section_relative_residual": float(
            np.max(floored_relative)
        ),
        "rms_noise_floored_centered_section_relative_residual": float(
            np.sqrt(np.mean(floored_relative**2))
        ),
        "minimum_centered_section_source_norm": float(
            np.min(centered_source_norm)
        ),
        "maximum_centered_section_source_norm": float(
            np.max(centered_source_norm)
        ),
        "coupling_ranks": sorted({int(value) for value in coupling_rank}),
        "maximum_hidden_section_norm": float(np.max(hidden_section_norm)),
    }


def run(
    input_path: Path,
    output_directory: Path,
    *,
    cutoffs: tuple[int, ...] = (12, 16),
    rank_tolerances: tuple[float, ...] = (1e-10, 1e-12),
    algebraic_noise_floor: float = 1e-12,
) -> dict[str, object]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty output directory {output_directory}"
        )
    if algebraic_noise_floor <= 0.0:
        raise ValueError("algebraic_noise_floor must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)

    payload = np.load(input_path, allow_pickle=False)
    labels = np.asarray(payload["labels"]).astype(str)
    times = np.asarray(payload["times"], dtype=float)
    states = np.asarray(payload["closed_coordinates"], dtype=float)
    drives = np.asarray(payload["drive_difference"], dtype=float).squeeze(-1)
    if states.shape != (labels.size, times.size, 31):
        raise ValueError("stored closed-coordinate array has an unexpected shape")
    if drives.shape != (labels.size, times.size):
        raise ValueError("stored drive array has an unexpected shape")

    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    plan: dict[str, object] = {
        "schema": "paper_v_archive_auxiliary_section_plan_v1",
        "run_id": RUN_ID,
        "classification": "offline_stored_state_algebraic_diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "scientific_question": (
            "Can the Pauli-repaired archive field be embedded as a minimum-"
            "norm section of the first reciprocal Hilbert--Schmidt force shell?"
        ),
        "input": str(input_path),
        "paths": labels.tolist(),
        "sample_count_per_path": int(times.size),
        "cutoffs": list(cutoffs),
        "rank_tolerances": list(rank_tolerances),
        "algebraic_noise_floor": algebraic_noise_floor,
        "archive_field": "pauli_repaired_31_coordinate_moment_eom",
        "online_exact_reference_used": False,
        "autonomous_rollout_executed": False,
    }
    _write_json_atomic(output_directory / "plan.json", plan)

    started = time.time()
    audit_summaries: list[dict[str, object]] = []
    array_payload: dict[str, np.ndarray] = {
        "labels": labels,
        "times": times,
    }
    for cutoff in cutoffs:
        for rank_tolerance in rank_tolerances:
            print(
                f"cutoff={cutoff} rank_tolerance={rank_tolerance:.1e}",
                flush=True,
            )
            construction = build_krylov_closure_construction(
                parameters,
                phonon_cutoff=cutoff,
                shell_count=1,
                rank_tolerance=rank_tolerance,
            )
            frame = build_archive_auxiliary_frame(construction, order=1)
            shape = (labels.size, times.size)
            raw_lift_relative = np.empty(shape, dtype=float)
            section_relative = np.empty(shape, dtype=float)
            centered_relative = np.empty(shape, dtype=float)
            centered_absolute = np.empty(shape, dtype=float)
            centered_source_norm = np.empty(shape, dtype=float)
            coupling_rank = np.empty(shape, dtype=int)
            hidden_section_norm = np.empty(shape, dtype=float)
            smallest_coupling_ratio = np.empty(shape, dtype=float)

            for path_index in range(labels.size):
                for time_index, time_value in enumerate(times):
                    drive_value = float(drives[path_index, time_index])
                    protocol_parameters = _FixedDriveParameters(
                        parameters,
                        drive_value,
                    )
                    archive_velocity = pauli_repaired_closed_scalar_rhs(
                        float(time_value),
                        states[path_index, time_index],
                        protocol_parameters,  # type: ignore[arg-type]
                    )
                    certificate = frame.section(
                        states[path_index, time_index],
                        archive_velocity,
                        drive_value=drive_value,
                        relative_tolerance=rank_tolerance,
                    )
                    raw_lift_relative[path_index, time_index] = (
                        certificate.raw_lift_relative_residual
                    )
                    section_relative[path_index, time_index] = (
                        certificate.section_relative_residual
                    )
                    centered_relative[path_index, time_index] = (
                        certificate.centered_section_relative_residual
                    )
                    centered_absolute[path_index, time_index] = float(
                        np.linalg.norm(certificate.centered_incompatibility)
                    )
                    centered_source_norm[path_index, time_index] = float(
                        np.linalg.norm(
                            certificate.centering_jacobian
                            @ certificate.section_source
                        )
                    )
                    coupling_rank[path_index, time_index] = (
                        certificate.coupling_rank
                    )
                    hidden_section_norm[path_index, time_index] = float(
                        np.linalg.norm(certificate.hidden_section)
                    )
                    retained_values = certificate.coupling_singular_values[
                        : certificate.coupling_rank
                    ]
                    smallest_coupling_ratio[path_index, time_index] = float(
                        retained_values[-1] / retained_values[0]
                    )

            key = f"cutoff{cutoff}_tol{rank_tolerance:.0e}".replace("-", "m")
            array_payload[f"{key}_raw_lift_relative"] = raw_lift_relative
            array_payload[f"{key}_centered_section_relative"] = centered_relative
            array_payload[f"{key}_centered_section_absolute"] = centered_absolute
            array_payload[f"{key}_coupling_rank"] = coupling_rank
            path_summaries = {
                str(label): _path_summary(
                    raw_lift_relative[path_index],
                    section_relative[path_index],
                    centered_relative[path_index],
                    centered_absolute[path_index],
                    centered_source_norm[path_index],
                    coupling_rank[path_index],
                    hidden_section_norm[path_index],
                    algebraic_noise_floor=algebraic_noise_floor,
                )
                for path_index, label in enumerate(labels)
            }
            maximum_floored = float(
                np.max(
                    centered_absolute
                    / np.maximum(centered_source_norm, algebraic_noise_floor)
                )
            )
            audit_summaries.append(
                {
                    "phonon_cutoff": cutoff,
                    "rank_tolerance": rank_tolerance,
                    "hidden_dimension": frame.hidden_dimension,
                    "coupling_ranks": sorted(
                        {int(value) for value in coupling_rank.reshape(-1)}
                    ),
                    "minimum_supported_coupling_singular_ratio": float(
                        np.min(smallest_coupling_ratio)
                    ),
                    "maximum_raw_lift_relative_residual": float(
                        np.max(raw_lift_relative)
                    ),
                    "maximum_noise_floored_centered_section_relative_residual": (
                        maximum_floored
                    ),
                    "maximum_centered_section_absolute_residual": float(
                        np.max(centered_absolute)
                    ),
                    "component_block_diagnostics": _block_diagnostics(frame),
                    "paths": path_summaries,
                }
            )
            print(
                "  hidden="
                f"{frame.hidden_dimension} ranks="
                f"{sorted({int(value) for value in coupling_rank.reshape(-1)})} "
                f"max lift={np.max(raw_lift_relative):.3e} "
                f"max section={maximum_floored:.3e}",
                flush=True,
            )

    np.savez_compressed(
        output_directory / "archive_auxiliary_section.npz",
        **array_payload,
    )
    maximum_lift = max(
        float(audit["maximum_raw_lift_relative_residual"])
        for audit in audit_summaries
    )
    maximum_section = max(
        float(audit["maximum_noise_floored_centered_section_relative_residual"])
        for audit in audit_summaries
    )
    ranks_stable = all(
        audit["coupling_ranks"] == [19] for audit in audit_summaries
    )
    gate_passed = maximum_lift < 1e-9 and maximum_section < 1e-9 and ranks_stable
    summary: dict[str, object] = {
        "schema": "paper_v_archive_auxiliary_section_summary_v1",
        "run_id": RUN_ID,
        "classification": "offline_stored_state_algebraic_diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "status": "complete",
        "gate_passed": gate_passed,
        "decision": (
            "continue_to_reachable_observable_frame_and_autonomous_memory_law"
            if gate_passed
            else "extend_operator_envelope_before_autonomous_memory_law"
        ),
        "interpretation": (
            "Passing means the archive field is an algebraic section of the "
            "first reciprocal force shell on these stored states. It does not "
            "establish an accurate autonomous auxiliary rollout."
        ),
        "maximum_raw_lift_relative_residual": maximum_lift,
        "maximum_noise_floored_centered_section_relative_residual": (
            maximum_section
        ),
        "coupling_rank_stable_at_19": ranks_stable,
        "audits": audit_summaries,
        "exact_trajectory_used_online": False,
        "autonomous_rollout_executed": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(output_directory / "summary.json", summary)

    repo_root = Path(__file__).resolve().parents[2]
    source_paths = (
        Path(__file__).resolve(),
        repo_root
        / "paper_5/src/paper5/stability/archive_auxiliary_memory.py",
        repo_root / "paper_5/src/paper5/stability/krylov_memory_closure.py",
        repo_root / "paper_5/src/paper5/stability/matrix_reference.py",
    )
    artifact_paths = (
        output_directory / "plan.json",
        output_directory / "summary.json",
        output_directory / "archive_auxiliary_section.npz",
    )
    manifest: dict[str, object] = {
        "schema": "paper_v_archive_auxiliary_section_manifest_v1",
        "run_id": RUN_ID,
        "status": "complete",
        "classification": "offline_stored_state_algebraic_diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "python": sys.version,
        "platform": platform.platform(),
        "input_hash": _sha256(input_path),
        "source_hashes": {
            str(path.relative_to(repo_root)): _sha256(path)
            for path in source_paths
        },
        "artifact_hashes": {
            path.name: _sha256(path) for path in artifact_paths
        },
    }
    _write_json_atomic(output_directory / "runtime_manifest.json", manifest)
    return summary


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(","))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cutoffs", type=_parse_csv_ints, default=(12, 16))
    parser.add_argument(
        "--rank-tolerances",
        type=_parse_csv_floats,
        default=(1e-10, 1e-12),
    )
    parser.add_argument("--algebraic-noise-floor", type=float, default=1e-12)
    args = parser.parse_args()
    summary = run(
        args.input,
        args.output_directory,
        cutoffs=args.cutoffs,
        rank_tolerances=args.rank_tolerances,
        algebraic_noise_floor=args.algebraic_noise_floor,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
