from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.quantum.chemistry.generate_h2o_linear_fd_fixture import (
    AMU_TO_ELECTRON_MASS,
    BACKEND_RECORD_SCHEMA,
)


def _mode_vector(mode_index: int) -> list[list[float]]:
    vector = np.zeros((3, 3), dtype=float)
    vector[int(mode_index), int(mode_index)] = 1.0
    return vector.tolist()


def synthetic_three_mode_h2o_linear_fd_backend_record() -> dict[str, object]:
    mode_labels = ("bend", "symmetric_stretch", "antisymmetric_stretch")
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.4, 1.1], [0.0, -1.4, 1.1]],
        dtype=float,
    )
    one_body = np.array([[-1.0, 0.02], [0.02, 0.4]], dtype=float)
    two_body = np.zeros((2, 2, 2, 2), dtype=float)

    normal_modes: list[dict[str, object]] = []
    aligned_tensors: list[dict[str, object]] = []
    for mode_index, label in enumerate(mode_labels):
        q_step = 0.1
        q_step_alt = 0.05
        derivative = 1.0e-4 * float(mode_index + 1)
        normal_modes.append(
            {
                "label": label,
                "frequency_hartree": 0.01 + 0.002 * mode_index,
                "frequency_cm1": (0.01 + 0.002 * mode_index) * 219474.63136320,
                "mass_weighted_eigenvector": _mode_vector(mode_index),
                "q_step_au": q_step,
                "q_step_alt_au": q_step_alt,
            }
        )
        for step_kind, step in (("primary", q_step), ("alt", q_step_alt)):
            for sign in (-1, 1):
                displacement_id = f"{label}_{step_kind}_{sign:+d}"
                tensor_id = f"aligned_{displacement_id}"
                aligned_tensors.append(
                    {
                        "aligned_tensor_id": tensor_id,
                        "source_snapshot_id": f"snapshot_{displacement_id}",
                        "displacement_id": displacement_id,
                        "geometry_id": f"geometry_{displacement_id}",
                        "mode_label": label,
                        "sign": sign,
                        "step_kind": step_kind,
                        "q_displacement_au": float(sign) * float(step),
                        "coordinates_bohr": coordinates.tolist(),
                        "scalar_energy_hartree": float(sign) * float(step) * derivative,
                        "one_body_integrals": one_body.tolist(),
                        "two_body_integrals": two_body.tolist(),
                        "alignment": {
                            "alignment_id": f"align_{displacement_id}",
                            "active_rotation": np.eye(2).tolist(),
                            "singular_values": [1.0, 1.0],
                            "min_singular_value": 1.0,
                            "alignment_residual_fro": 0.0,
                            "active_to_external_leakage_fro": 0.0,
                            "external_to_active_leakage_fro": 0.0,
                            "passed": True,
                        },
                    }
                )

    return {
        "schema": BACKEND_RECORD_SCHEMA,
        "backend": {
            "name": "synthetic",
            "method": "scf",
            "basis": "synthetic",
            "reference": "rhf",
        },
        "system": {
            "charge": 0,
            "multiplicity": 1,
            "optimized": True,
        },
        "center_snapshot_id": "synthetic_center",
        "geometry": {
            "geometry_id": "synthetic_h2o_center",
            "symbols": ["O", "H", "H"],
            "coordinates_bohr": coordinates.tolist(),
            "masses_me": (
                np.array([15.99491461957, 1.00782503223, 1.00782503223])
                * AMU_TO_ELECTRON_MASS
            ).tolist(),
            "provenance": {"source": "unit_test"},
        },
        "active_space": {
            "active_space_kind": "synthetic_active2_three_mode_variant",
            "frozen_core_indices": [],
            "active_indices_center": [0, 1],
            "external_indices": [],
            "n_spatial_orbitals": 2,
            "num_particles": [1, 1],
            "scalar_energy_hartree": 0.0,
            "one_body_integrals": one_body.tolist(),
            "two_body_integrals": two_body.tolist(),
        },
        "normal_modes": normal_modes,
        "aligned_tensors": aligned_tensors,
        "report_summary": {
            "paper_iv_active_space_variant": "synthetic_active2_three_mode_test"
        },
    }


def write_synthetic_three_mode_h2o_linear_fd_backend_record_json(
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(synthetic_three_mode_h2o_linear_fd_backend_record(), indent=2)
        + "\n",
        encoding="utf-8",
    )
    return output
