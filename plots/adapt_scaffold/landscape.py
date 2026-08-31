"""Final-scaffold trajectory utilities and optional surrogate energy support."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from docs.reports.pdf_utils import get_plt, require_matplotlib
from plots.adapt_scaffold.artifact_reader import AdaptPlotArtifact, FinalScaffoldRecord
from pipelines.scaffold.hh_continuation_generators import rebuild_polynomial_from_serialized_terms
from pipelines.scaffold.hh_vqe_from_adapt_family import PoolTermwiseAnsatz
from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, hamiltonian_matrix


@dataclass(frozen=True)
class ScaffoldTrajectoryDataset:
    projected_points: np.ndarray
    final_scaffold_records: tuple[FinalScaffoldRecord, ...]
    prefix_runtime_points: np.ndarray
    actual_step_energies: np.ndarray | None
    block_l2_norms: np.ndarray
    warnings: tuple[str, ...]
    projection_method: str
    projection_axes: np.ndarray | None


@dataclass(frozen=True)
class FinalScaffoldEnergyModel:
    layout_runtime_count: int
    ansatz: PoolTermwiseAnsatz
    hmat: np.ndarray
    psi_ref: np.ndarray

    def energy(self, theta_runtime: np.ndarray) -> float:
        psi = np.asarray(self.ansatz.prepare_state(np.asarray(theta_runtime, dtype=float), self.psi_ref), dtype=complex)
        return float(np.real(np.vdot(psi, self.hmat @ psi)))


def trajectory_prerequisite_failure_reason(artifact: AdaptPlotArtifact) -> str | None:
    if artifact.layout is None:
        return "missing_parameterization_layout"
    if artifact.final_theta_runtime is None:
        return "missing_final_theta_runtime"
    if int(artifact.layout.runtime_parameter_count) <= 0:
        return "empty_runtime_parameterization"
    if not artifact.final_scaffold_records:
        return "missing_final_scaffold_records"
    return None


def _project_prefix_points(prefix_runtime_points: np.ndarray) -> tuple[np.ndarray, str, np.ndarray | None]:
    runtime_dim = int(prefix_runtime_points.shape[1])
    if runtime_dim <= 0:
        return np.zeros((int(prefix_runtime_points.shape[0]), 2), dtype=float), "empty_runtime_dim", None
    if runtime_dim == 1:
        return np.column_stack([prefix_runtime_points[:, 0], np.zeros(prefix_runtime_points.shape[0])]), "runtime_dim_1", None
    centered = prefix_runtime_points - np.mean(prefix_runtime_points, axis=0, keepdims=True)
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular_values > 1.0e-12))
    axes = np.zeros((min(2, vt.shape[0]), runtime_dim), dtype=float)
    if vt.size > 0:
        axes[0, :] = vt[0, :]
    if vt.shape[0] > 1:
        axes[1, :] = vt[1, :]
    projected = centered @ axes[:2, :].T
    if projected.shape[1] == 1:
        projected = np.column_stack([projected[:, 0], np.zeros(projected.shape[0])])
    return projected, f"centered_svd_rank_{rank}", axes


def build_scaffold_trajectory_dataset(artifact: AdaptPlotArtifact) -> ScaffoldTrajectoryDataset:
    reason = trajectory_prerequisite_failure_reason(artifact)
    if reason is not None:
        raise ValueError(reason)
    assert artifact.layout is not None
    assert artifact.final_theta_runtime is not None

    runtime_count = int(artifact.layout.runtime_parameter_count)
    prefix_points = np.zeros((int(artifact.layout.logical_parameter_count) + 1, runtime_count), dtype=float)
    block_norms = np.zeros(int(artifact.layout.logical_parameter_count), dtype=float)
    for idx, block in enumerate(artifact.layout.blocks, start=1):
        prefix_points[idx] = prefix_points[idx - 1]
        prefix_points[idx, block.runtime_start:block.runtime_stop] = artifact.final_theta_runtime[
            block.runtime_start:block.runtime_stop
        ]
        block_slice = prefix_points[idx, block.runtime_start:block.runtime_stop]
        block_norms[idx - 1] = float(np.linalg.norm(block_slice))

    projected_points, projection_method, projection_axes = _project_prefix_points(prefix_points)
    energy_lookup = {
        int(step.step_index): float(step.energy_after_opt)
        for step in artifact.history_steps
    }
    actual_step_energies = np.full(int(prefix_points.shape[0]), np.nan, dtype=float)
    for idx, record in enumerate(artifact.final_scaffold_records, start=1):
        if record.admission_step_index is not None and record.admission_step_index in energy_lookup:
            actual_step_energies[idx] = energy_lookup[int(record.admission_step_index)]

    return ScaffoldTrajectoryDataset(
        projected_points=np.asarray(projected_points, dtype=float),
        final_scaffold_records=tuple(artifact.final_scaffold_records),
        prefix_runtime_points=np.asarray(prefix_points, dtype=float),
        actual_step_energies=np.asarray(actual_step_energies, dtype=float),
        block_l2_norms=np.asarray(block_norms, dtype=float),
        warnings=tuple(artifact.warnings),
        projection_method=str(projection_method),
        projection_axes=(None if projection_axes is None else np.asarray(projection_axes, dtype=float)),
    )


def maybe_build_final_scaffold_energy_model(artifact: AdaptPlotArtifact) -> FinalScaffoldEnergyModel | None:
    if artifact.layout is None or artifact.final_theta_runtime is None:
        return None
    if artifact.ansatz_input_state_vector is None:
        return None
    if int(artifact.layout.logical_parameter_count) != int(len(artifact.final_operator_labels)):
        return None

    terms: list[AnsatzTerm] = []
    for block in artifact.layout.blocks:
        serialized_terms = [
            {
                "pauli_exyz": str(term.pauli_exyz),
                "coeff_re": float(term.coeff_real),
                "coeff_im": 0.0,
                "nq": int(term.nq),
            }
            for term in block.terms
        ]
        poly = rebuild_polynomial_from_serialized_terms(serialized_terms)
        terms.append(AnsatzTerm(label=str(block.candidate_label), polynomial=poly))

    settings = artifact.settings
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=int(settings.get("L", 0)),
        J=float(settings.get("t", 1.0)),
        U=float(settings.get("u", 0.0)),
        omega0=float(settings.get("omega0", 1.0)),
        g=float(settings.get("g_ep", 0.0)),
        n_ph_max=int(settings.get("n_ph_max", 0)),
        boson_encoding=str(settings.get("boson_encoding", "binary")),
        v_t=float(settings.get("dv", 0.0)),
        v0=None,
        t_eval=None,
        repr_mode="JW",
        indexing=str(settings.get("ordering", "blocked")),
        pbc=(str(settings.get("boundary", "open")).strip().lower() == "periodic"),
        include_zero_point=bool(settings.get("include_zero_point", True)),
    )
    hmat = np.asarray(hamiltonian_matrix(h_poly), dtype=complex)
    psi_ref = np.asarray(artifact.ansatz_input_state_vector, dtype=complex).reshape(-1)
    nq = int(round(np.log2(int(psi_ref.size))))
    ansatz = PoolTermwiseAnsatz(
        terms=terms,
        reps=1,
        nq=nq,
        parameterization_layout=artifact.layout,
        parameterization_mode="per_pauli_term",
    )
    return FinalScaffoldEnergyModel(
        layout_runtime_count=int(artifact.layout.runtime_parameter_count),
        ansatz=ansatz,
        hmat=hmat,
        psi_ref=psi_ref,
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_scaffold_trajectory_outputs(
    artifact: AdaptPlotArtifact,
    *,
    output_png: Path,
    output_json: Path,
    surrogate_energy_model: FinalScaffoldEnergyModel | None = None,
) -> dict[str, Any]:
    dataset = build_scaffold_trajectory_dataset(artifact)
    surrogate_prefix_energies: np.ndarray | None = None
    if surrogate_energy_model is not None:
        surrogate_prefix_energies = np.asarray(
            [surrogate_energy_model.energy(point) for point in dataset.prefix_runtime_points],
            dtype=float,
        )

    require_matplotlib()
    plt = get_plt()
    _ensure_parent(output_png)
    _ensure_parent(output_json)

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    points = np.asarray(dataset.projected_points, dtype=float)
    color_values = (
        dataset.actual_step_energies
        if dataset.actual_step_energies is not None and np.isfinite(dataset.actual_step_energies[1:]).any()
        else np.arange(points.shape[0], dtype=float)
    )
    scatter = ax.scatter(points[:, 0], points[:, 1], c=color_values, cmap="viridis", s=80, zorder=3)
    fig.colorbar(scatter, ax=ax, shrink=0.85, label="Recorded step energy / point index")
    ax.scatter(points[0, 0], points[0, 1], color="black", marker="*", s=130, label="Origin", zorder=4)
    for idx, record in enumerate(dataset.final_scaffold_records, start=1):
        start = points[idx - 1]
        end = points[idx]
        width = 1.0 + 2.5 * (
            float(dataset.block_l2_norms[idx - 1]) / max(float(np.max(dataset.block_l2_norms)), 1.0e-12)
        )
        ax.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops={"arrowstyle": "->", "linewidth": width, "alpha": 0.85, "color": "tab:blue"},
            zorder=2,
        )
        ax.text(end[0], end[1], str(record.final_order_index), fontsize=8, ha="left", va="bottom")
    ax.set_title("Final scaffold trajectory (runtime_prefix_final_theta)")
    ax.set_xlabel("Projected axis 1")
    ax.set_ylabel("Projected axis 2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    json_payload: dict[str, Any] = {
        "path_mode": "runtime_prefix_final_theta",
        "projection": {
            "method": dataset.projection_method,
            "runtime_dimension": int(dataset.prefix_runtime_points.shape[1]),
        },
        "warnings": list(dataset.warnings),
        "points": [],
        "rays": [],
    }
    for idx, point in enumerate(points):
        json_payload["points"].append(
            {
                "point_index": int(idx),
                "final_order_index": (None if idx == 0 else int(dataset.final_scaffold_records[idx - 1].final_order_index)),
                "x": float(point[0]),
                "y": float(point[1]),
                "actual_energy_after_opt": (
                    None
                    if dataset.actual_step_energies is None or not np.isfinite(dataset.actual_step_energies[idx])
                    else float(dataset.actual_step_energies[idx])
                ),
                "surrogate_prefix_energy": (
                    None
                    if surrogate_prefix_energies is None or not np.isfinite(surrogate_prefix_energies[idx])
                    else float(surrogate_prefix_energies[idx])
                ),
            }
        )
    for idx, record in enumerate(dataset.final_scaffold_records, start=1):
        json_payload["rays"].append(
            {
                "final_order_index": int(record.final_order_index),
                "generator_label": str(record.generator_label),
                "generator_id": record.generator_id,
                "runtime_start": record.runtime_start,
                "runtime_stop": record.runtime_stop,
                "block_l2_norm": float(dataset.block_l2_norms[idx - 1]),
                "from_point_index": int(idx - 1),
                "to_point_index": int(idx),
                "admission_order_index": record.admission_order_index,
                "admission_step_index": record.admission_step_index,
                "admission_record_index": record.admission_record_index,
                "actual_energy_after_opt": (
                    None
                    if dataset.actual_step_energies is None or not np.isfinite(dataset.actual_step_energies[idx])
                    else float(dataset.actual_step_energies[idx])
                ),
            }
        )
    output_json.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    return {
        "status": "written",
        "path_png": str(output_png),
        "path_json": str(output_json),
        "point_count": int(points.shape[0]),
        "surrogate_energy": "written" if surrogate_prefix_energies is not None else "not_requested_or_unavailable",
    }


__all__ = [
    "FinalScaffoldEnergyModel",
    "ScaffoldTrajectoryDataset",
    "build_scaffold_trajectory_dataset",
    "maybe_build_final_scaffold_energy_model",
    "trajectory_prerequisite_failure_reason",
    "write_scaffold_trajectory_outputs",
]
