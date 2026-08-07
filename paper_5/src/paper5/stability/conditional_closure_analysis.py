"""Exact-state gate for the electron-conditioned Gaussian moment closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .conditional_closure import (
    ELECTRONIC_CONDITIONED_GAUSSIAN_CLOSURE,
)
from .exact_reference import exact_holstein_moment_hierarchy_trajectory
from .hubbard_dimer import DimerParameters
from .moment_hierarchy import (
    ZERO_CUMULANT_CLOSURE,
    MomentHierarchy,
    MomentKey,
    TerminalMomentClosure,
    _commutator,
    _hamiltonian_terms,
    moment_hierarchy,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_times(final_time: float, sample_step: float) -> np.ndarray:
    if final_time <= 0.0 or sample_step <= 0.0:
        raise ValueError("final_time and sample_step must be positive")
    intervals = int(round(final_time / sample_step))
    if not np.isclose(intervals * sample_step, final_time, atol=1e-12):
        raise ValueError("final_time must be an integer multiple of sample_step")
    return np.linspace(0.0, final_time, intervals + 1)


def _degree_indices(
    hierarchy: MomentHierarchy,
    degree: int,
) -> np.ndarray:
    return np.asarray(
        [
            index + 2
            for index, key in enumerate(hierarchy.moment_keys)
            if key.degree == degree
        ],
        dtype=int,
    )


def _vector_metrics(
    defect: np.ndarray,
    exact: np.ndarray,
) -> dict[str, float]:
    defect_norm = np.linalg.norm(defect, axis=1)
    exact_norm = np.linalg.norm(exact, axis=1)
    defect_rms = float(np.sqrt(np.mean(defect_norm**2)))
    exact_rms = float(np.sqrt(np.mean(exact_norm**2)))
    return {
        "component_rms": float(np.sqrt(np.mean(defect**2))),
        "vector_rms": defect_rms,
        "maximum_norm": float(np.max(defect_norm)),
        "exact_vector_rms": exact_rms,
        "relative_vector_rms": float(
            defect_rms / max(exact_rms, np.finfo(float).tiny)
        ),
    }


def _required_terminal_keys(
    hierarchy: MomentHierarchy,
    parameters: DimerParameters,
) -> tuple[MomentKey, ...]:
    keys: set[MomentKey] = set()
    for observable in hierarchy.moment_keys:
        for _, hamiltonian_key in _hamiltonian_terms(0.37, parameters):
            keys.update(
                key
                for key in _commutator(hamiltonian_key, observable)
                if key.degree == hierarchy.maximum_degree + 1
            )
    return tuple(sorted(keys))


def _evaluate_closure(
    hierarchy: MomentHierarchy,
    closure: TerminalMomentClosure,
    parameters: DimerParameters,
    times: np.ndarray,
    coordinates: np.ndarray,
    terminal_keys: tuple[MomentKey, ...],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    derivatives = np.empty_like(coordinates)
    predicted_terminal = np.empty((times.size, len(terminal_keys)))
    diagnostics: list[dict[str, float | int]] = []
    for index, (time, state) in enumerate(
        zip(times, coordinates, strict=True)
    ):
        derivatives[index] = hierarchy.rhs(
            float(time),
            state,
            parameters,
            closure=closure,
        )
        _, moments = hierarchy.unpack(state)
        resolver = closure.prepare(moments, hierarchy.maximum_degree)
        predicted_terminal[index] = [
            resolver.moment(key) for key in terminal_keys
        ]
        resolver_diagnostics = getattr(resolver, "diagnostics", None)
        if resolver_diagnostics is not None:
            diagnostics.append(dict(resolver_diagnostics))
    return derivatives, predicted_terminal, diagnostics


def _write_plot(
    path: Path,
    times: np.ndarray,
    exact_derivative: np.ndarray,
    terminal_indices: np.ndarray,
    zero_derivative: np.ndarray,
    adapted_derivative: np.ndarray,
    exact_terminal_moments: np.ndarray,
    zero_terminal_moments: np.ndarray,
    adapted_terminal_moments: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    exact_norm = np.linalg.norm(
        exact_derivative[:, terminal_indices],
        axis=1,
    )
    zero_defect = np.linalg.norm(
        zero_derivative[:, terminal_indices]
        - exact_derivative[:, terminal_indices],
        axis=1,
    )
    adapted_defect = np.linalg.norm(
        adapted_derivative[:, terminal_indices]
        - exact_derivative[:, terminal_indices],
        axis=1,
    )
    axes[0].plot(
        times,
        exact_norm,
        color="#555555",
        label="exact degree-four velocity",
    )
    axes[0].plot(
        times,
        zero_defect,
        color="#D28E2B",
        linestyle="--",
        label="zero fifth cumulant",
    )
    axes[0].plot(
        times,
        adapted_defect,
        color="#2378B5",
        label="conditioned Gaussian",
    )
    axes[0].set_xlabel(r"dimensionless time $t\,t_{\rm hop}$")
    axes[0].set_ylabel(r"degree-four $\ell_2$ norm")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=7.5)

    axes[1].plot(
        times,
        np.linalg.norm(
            zero_terminal_moments - exact_terminal_moments,
            axis=1,
        ),
        color="#D28E2B",
        linestyle="--",
        label="zero fifth cumulant",
    )
    axes[1].plot(
        times,
        np.linalg.norm(
            adapted_terminal_moments - exact_terminal_moments,
            axis=1,
        ),
        color="#2378B5",
        label="conditioned Gaussian",
    )
    axes[1].set_xlabel(r"dimensionless time $t\,t_{\rm hop}$")
    axes[1].set_ylabel(r"required fifth-moment defect $\ell_2$ norm")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=7.5)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_conditional_closure_analysis(
    run_directory: Path,
    *,
    parameters: DimerParameters,
    final_time: float = 4.0,
    sample_step: float = 0.01,
    phonon_cutoff: int = 20,
    convergence_cutoffs: tuple[int, ...] = (12, 16, 20),
    convergence_time: float = 0.5,
    eigensolver_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
    maximum_step: float = 0.01,
    terminal_relative_defect_threshold: float = 0.1,
) -> dict[str, Any]:
    """Compare zero-cumulant and conditioned-Gaussian fifth-moment rules."""

    if phonon_cutoff not in convergence_cutoffs:
        raise ValueError("convergence_cutoffs must include phonon_cutoff")
    if not 0.0 < convergence_time <= final_time:
        raise ValueError("convergence_time must lie in (0, final_time]")
    if terminal_relative_defect_threshold <= 0.0:
        raise ValueError("terminal threshold must be positive")

    run_directory.mkdir(parents=True, exist_ok=True)
    retained = moment_hierarchy(4)
    oracle = moment_hierarchy(5)
    if oracle.moment_keys[: len(retained.moment_keys)] != retained.moment_keys:
        raise RuntimeError("moment hierarchy ordering is not nested")
    times = _sample_times(final_time, sample_step)
    exact = exact_holstein_moment_hierarchy_trajectory(
        parameters,
        hierarchy=oracle,
        sample_times=times,
        phonon_cutoff=phonon_cutoff,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    coordinates = exact.coordinates[:, : retained.coordinate_count]
    exact_derivative = exact.coordinate_derivatives[
        :, : retained.coordinate_count
    ]
    terminal_keys = _required_terminal_keys(retained, parameters)
    oracle_index = {
        key: index + 2 for index, key in enumerate(oracle.moment_keys)
    }
    exact_terminal_moments = exact.coordinates[
        :, [oracle_index[key] for key in terminal_keys]
    ]

    zero_derivative, zero_terminal_moments, _ = _evaluate_closure(
        retained,
        ZERO_CUMULANT_CLOSURE,
        parameters,
        exact.times,
        coordinates,
        terminal_keys,
    )
    adapted_derivative, adapted_terminal_moments, diagnostics = (
        _evaluate_closure(
            retained,
            ELECTRONIC_CONDITIONED_GAUSSIAN_CLOSURE,
            parameters,
            exact.times,
            coordinates,
            terminal_keys,
        )
    )
    terminal_indices = _degree_indices(retained, 4)
    exact_terminal_derivative = exact_derivative[:, terminal_indices]
    zero_derivative_defect = (
        zero_derivative[:, terminal_indices] - exact_terminal_derivative
    )
    adapted_derivative_defect = (
        adapted_derivative[:, terminal_indices] - exact_terminal_derivative
    )
    zero_metrics = _vector_metrics(
        zero_derivative_defect,
        exact_terminal_derivative,
    )
    adapted_metrics = _vector_metrics(
        adapted_derivative_defect,
        exact_terminal_derivative,
    )
    zero_moment_metrics = _vector_metrics(
        zero_terminal_moments - exact_terminal_moments,
        exact_terminal_moments,
    )
    adapted_moment_metrics = _vector_metrics(
        adapted_terminal_moments - exact_terminal_moments,
        exact_terminal_moments,
    )

    direction = adapted_derivative[:, terminal_indices] - zero_derivative[
        :, terminal_indices
    ]
    direction_norm_squared = float(np.sum(direction**2))
    optimal_blend = float(
        -np.sum(zero_derivative_defect * direction)
        / max(direction_norm_squared, np.finfo(float).tiny)
    )
    blended_defect = zero_derivative_defect + optimal_blend * direction
    blend_metrics = _vector_metrics(
        blended_defect,
        exact_terminal_derivative,
    )

    cutoff_metrics: dict[str, Any] = {}
    for cutoff in convergence_cutoffs:
        if cutoff == phonon_cutoff:
            sample_index = int(round(convergence_time / sample_step))
            cutoff_times = exact.times[[0, sample_index]]
            cutoff_coordinates = coordinates[[0, sample_index]]
            cutoff_derivative = exact_derivative[[0, sample_index]]
        else:
            cutoff_exact = exact_holstein_moment_hierarchy_trajectory(
                parameters,
                hierarchy=oracle,
                sample_times=np.asarray([0.0, convergence_time]),
                phonon_cutoff=cutoff,
                eigensolver_tolerance=eigensolver_tolerance,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                maximum_step=maximum_step,
            )
            cutoff_times = cutoff_exact.times
            cutoff_coordinates = cutoff_exact.coordinates[
                :, : retained.coordinate_count
            ]
            cutoff_derivative = cutoff_exact.coordinate_derivatives[
                :, : retained.coordinate_count
            ]
        cutoff_adapted, _, cutoff_diagnostics = _evaluate_closure(
            retained,
            ELECTRONIC_CONDITIONED_GAUSSIAN_CLOSURE,
            parameters,
            cutoff_times,
            cutoff_coordinates,
            terminal_keys,
        )
        cutoff_defect = (
            cutoff_adapted[:, terminal_indices]
            - cutoff_derivative[:, terminal_indices]
        )
        cutoff_metrics[str(cutoff)] = {
            **_vector_metrics(
                cutoff_defect,
                cutoff_derivative[:, terminal_indices],
            ),
            "maximum_jordan_relative_residual": max(
                float(item["maximum_jordan_relative_residual"])
                for item in cutoff_diagnostics
            ),
        }

    control_parameters = DimerParameters(
        hopping=parameters.hopping,
        gamma=parameters.gamma,
        lambda_ep=0.0,
        drive_amplitude=parameters.drive_amplitude,
        pulse_width=parameters.pulse_width,
    )
    control = exact_holstein_moment_hierarchy_trajectory(
        control_parameters,
        hierarchy=oracle,
        sample_times=np.linspace(0.0, 2.0, 9),
        phonon_cutoff=3,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    control_coordinates = control.coordinates[:, : retained.coordinate_count]
    control_derivative, _, _ = _evaluate_closure(
        retained,
        ELECTRONIC_CONDITIONED_GAUSSIAN_CLOSURE,
        control_parameters,
        control.times,
        control_coordinates,
        terminal_keys,
    )
    control_maximum_defect = float(
        np.max(
            np.abs(
                control_derivative
                - control.coordinate_derivatives[
                    :, : retained.coordinate_count
                ]
            )
        )
    )

    terminal_rms = np.sqrt(
        np.mean(adapted_derivative_defect**2, axis=0)
    )
    top_terminal_derivative_components = [
        {
            "name": retained.state_names[int(terminal_indices[index])],
            "rms": float(terminal_rms[index]),
            "maximum_abs": float(
                np.max(np.abs(adapted_derivative_defect[:, index]))
            ),
        }
        for index in np.argsort(terminal_rms)[::-1][:10]
    ]
    moment_rms = np.sqrt(
        np.mean(
            (adapted_terminal_moments - exact_terminal_moments) ** 2,
            axis=0,
        )
    )
    top_required_moments = [
        {
            "name": (
                f"moment_{terminal_keys[index].spin_up.lower()}"
                f"{terminal_keys[index].spin_down.lower()}"
                f"_x{terminal_keys[index].x_power}"
                f"_p{terminal_keys[index].p_power}"
            ),
            "rms": float(moment_rms[index]),
            "maximum_abs": float(
                np.max(
                    np.abs(
                        adapted_terminal_moments[:, index]
                        - exact_terminal_moments[:, index]
                    )
                )
            ),
        }
        for index in np.argsort(moment_rms)[::-1][:10]
    ]

    adapted_passed = (
        adapted_metrics["relative_vector_rms"]
        <= terminal_relative_defect_threshold
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            **asdict(parameters),
            "coupling": parameters.coupling,
            "phonon_cutoff": phonon_cutoff,
            "final_time": final_time,
            "sample_step": sample_step,
            "maximum_step": maximum_step,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
        },
        "closure": {
            "name": ELECTRONIC_CONDITIONED_GAUSSIAN_CLOSURE.name,
            "construction": (
                "operator-valued electronic-conditioned displacement and "
                "covariance with a Jordan-product Gaussian recurrence"
            ),
            "required_fifth_moment_count": len(terminal_keys),
        },
        "terminal_derivative_metrics": {
            "zero_cumulant": zero_metrics,
            "electronic_conditioned_gaussian": adapted_metrics,
        },
        "required_fifth_moment_metrics": {
            "zero_cumulant": zero_moment_metrics,
            "electronic_conditioned_gaussian": adapted_moment_metrics,
        },
        "offline_optimal_scalar_blend_diagnostic": {
            "adapted_weight": optimal_blend,
            "metrics": blend_metrics,
            "online_authorized": False,
        },
        "closure_domain_diagnostics": {
            "support_ranks": sorted(
                {int(item["support_rank"]) for item in diagnostics}
            ),
            "minimum_density_eigenvalue": min(
                float(item["density_minimum_eigenvalue"])
                for item in diagnostics
            ),
            "maximum_jordan_relative_residual": max(
                float(item["maximum_jordan_relative_residual"])
                for item in diagnostics
            ),
        },
        "cutoff_convergence_at_t_0_and_t_0_5": cutoff_metrics,
        "decoupled_control_maximum_component_defect": (
            control_maximum_defect
        ),
        "top_terminal_derivative_components": (
            top_terminal_derivative_components
        ),
        "top_required_fifth_moments": top_required_moments,
        "validation_gate": {
            "terminal_relative_defect_threshold": (
                terminal_relative_defect_threshold
            ),
            "adapted_closure_passed": bool(adapted_passed),
            "raw_propagation_authorized": bool(adapted_passed),
            "decision": (
                "proceed to autonomous propagation"
                if adapted_passed
                else (
                    "defer autonomous propagation; electronic conditioning "
                    "reduces the terminal defect but does not make the "
                    "fourth-order velocity accurate"
                )
            ),
        },
    }

    prefix = "electronic_conditioned_gaussian_closure_gate"
    trajectory_path = run_directory / f"{prefix}.npz"
    np.savez_compressed(
        trajectory_path,
        times=exact.times,
        exact_coordinates=coordinates,
        exact_coordinate_derivatives=exact_derivative,
        zero_cumulant_coordinate_derivatives=zero_derivative,
        adapted_coordinate_derivatives=adapted_derivative,
        exact_required_fifth_moments=exact_terminal_moments,
        zero_cumulant_required_fifth_moments=zero_terminal_moments,
        adapted_required_fifth_moments=adapted_terminal_moments,
    )
    plot_path = run_directory / f"{prefix}.png"
    _write_plot(
        plot_path,
        exact.times,
        exact_derivative,
        terminal_indices,
        zero_derivative,
        adapted_derivative,
        exact_terminal_moments,
        zero_terminal_moments,
        adapted_terminal_moments,
    )
    summary_path = run_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("conditional_closure.py"),
        Path(__file__).with_name("moment_hierarchy.py"),
        Path(__file__).with_name("exact_reference.py"),
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "source_hashes": {
            str(path.resolve()): _sha256(path) for path in source_paths
        },
        "artifact_hashes": {
            path.name: _sha256(path)
            for path in (summary_path, trajectory_path, plot_path)
        },
    }
    (run_directory / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--sample-step", type=float, default=0.01)
    parser.add_argument("--phonon-cutoff", type=int, default=20)
    parser.add_argument("--maximum-step", type=float, default=0.01)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_conditional_closure_analysis(
        args.run_directory,
        parameters=DimerParameters(
            lambda_ep=args.lambda_ep,
            gamma=args.gamma,
            drive_amplitude=args.drive,
        ),
        final_time=args.final_time,
        sample_step=args.sample_step,
        phonon_cutoff=args.phonon_cutoff,
        maximum_step=args.maximum_step,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
