"""Frozen, reference-blind contract for the multi-coherent holdout.

This module contains only model-side settings and audits.  It does not import
or invoke an exact driven propagator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import sqrt
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping

import numpy as np
import scipy

from .hubbard_dimer import DimerParameters, GaussianSineDrive
from .conditional_packets import (
    electron_relative_product_to_local_state,
    electron_relative_state,
)
from .exact_reference import (
    _build_exact_dimer_model,
    _contract_matrix_state,
    _ground_state,
)
from .matrix_reference import matrix_state_to_closed_scalar_coordinates
from .moment_hierarchy import moment_hierarchy
from .multi_coherent import (
    multi_coherent_state,
    normalized_diagonal_kick_generator,
    retract_multi_coherent_parameters,
    relative_state_closed_coordinates,
    symmetric_projected_generator_kick,
)
from .multi_coherent_scores import (
    closed_coordinate_distance,
    development_coordinate_scales,
)


_PREPARED_SCHEMA = "paper5.multi_coherent.blind_inputs.v1"
_PREPARED_MANIFEST = "pre_model_manifest.json"
_PREPARED_ARRAYS = "frozen_initial_conditions.npz"
_MODEL_BATCH_MANIFEST = "model_batch_manifest.json"
_MODEL_COST_MANIFEST = "model_cost_manifest.json"
_SCORE_CONSUMPTION_RECEIPT = "score_consumption_receipt.json"


@dataclass(frozen=True)
class MultiCoherentHoldoutSettings:
    """Prospective settings for the one double-pulse model holdout."""

    final_time: float = 100.0
    score_interval: tuple[float, float] = (8.0, 20.0)
    sensitivity_interval: tuple[float, float] = (40.0, 100.0)
    phonon_cutoff: int = 16
    initial_packets_per_electronic_branch: int = 4
    maximum_packets_per_electronic_branch: int = 6
    maximum_geometric_tangent_rank: int = 96
    geometric_gram_relative_threshold: float = 1e-10
    tangent_regularization: str = "tikhonov"
    relative_damping: float = 3e-4
    segment_length: float = 0.5
    output_sample_step: float = 0.05
    coarse_maximum_step: float = 0.02
    fine_maximum_step: float = 0.01
    coarse_relative_tolerance: float = 1e-7
    coarse_absolute_tolerance: float = 1e-9
    fine_relative_tolerance: float = 1e-8
    fine_absolute_tolerance: float = 1e-10
    spawn_relative_residual_threshold: float = 5e-2
    spawn_absolute_residual_threshold: float = 2e-2
    spawn_fit_maximum_iterations: int = 40
    spawn_fit_population_size: int = 6
    spawn_seed: int = 260804
    kick_size: float = 1e-4
    work_residual_ceiling: float = 1e-3
    fidelity_floor: float = 0.99
    electron_trace_distance_ceiling: float = 0.025
    block_rms_ceiling: float = 0.05
    block_maximum_ceiling: float = 0.10
    sensitivity_ratio_ceiling: float = 2.0
    amplification_uncertainty_ceiling: float = 0.05
    numerical_resolution_fraction: float = 0.10
    exact_mutual_infidelity_ceiling: float = 1e-9
    exact_moment_disagreement_ceiling: float = 1e-6
    exact_midpoint_step: float = 0.005
    exact_midpoint_refined_step: float = 0.0025
    exact_exponential_action_tolerance: float = 1e-13
    exact_dop853_relative_tolerance: float = 1e-10
    exact_dop853_absolute_tolerance: float = 1e-12
    exact_dop853_maximum_step: float = 0.005
    exact_dop853_refined_relative_tolerance: float = 1e-11
    exact_dop853_refined_absolute_tolerance: float = 1e-13
    exact_dop853_refined_maximum_step: float = 0.0025
    cost_repeat_count: int = 3
    model_to_direct_cost_ratio_ceiling: float = 0.5

    def __post_init__(self) -> None:
        if self.final_time <= 0.0:
            raise ValueError("final_time must be positive")
        for name, interval in (
            ("score_interval", self.score_interval),
            ("sensitivity_interval", self.sensitivity_interval),
        ):
            if not 0.0 <= interval[0] < interval[1] <= self.final_time:
                raise ValueError(f"{name} must lie inside the holdout horizon")
        if self.phonon_cutoff < 2:
            raise ValueError("phonon_cutoff must be at least two")
        if self.initial_packets_per_electronic_branch < 1:
            raise ValueError("initial packet count must be positive")
        if (
            self.maximum_packets_per_electronic_branch
            < self.initial_packets_per_electronic_branch
        ):
            raise ValueError("maximum packet count cannot be below the initial count")
        if self.maximum_raw_coordinate_count != 96:
            raise ValueError("the frozen relative-mode raw-coordinate cap is 96")
        if self.tangent_regularization != "tikhonov":
            raise ValueError("the frozen holdout uses Tikhonov regularization")
        positive_fields = (
            "maximum_geometric_tangent_rank",
            "geometric_gram_relative_threshold",
            "relative_damping",
            "segment_length",
            "output_sample_step",
            "coarse_maximum_step",
            "fine_maximum_step",
            "coarse_relative_tolerance",
            "coarse_absolute_tolerance",
            "fine_relative_tolerance",
            "fine_absolute_tolerance",
            "spawn_fit_maximum_iterations",
            "spawn_fit_population_size",
            "kick_size",
            "work_residual_ceiling",
            "fidelity_floor",
            "electron_trace_distance_ceiling",
            "block_rms_ceiling",
            "block_maximum_ceiling",
            "sensitivity_ratio_ceiling",
            "amplification_uncertainty_ceiling",
            "numerical_resolution_fraction",
            "exact_mutual_infidelity_ceiling",
            "exact_moment_disagreement_ceiling",
            "exact_midpoint_step",
            "exact_midpoint_refined_step",
            "exact_exponential_action_tolerance",
            "exact_dop853_relative_tolerance",
            "exact_dop853_absolute_tolerance",
            "exact_dop853_maximum_step",
            "exact_dop853_refined_relative_tolerance",
            "exact_dop853_refined_absolute_tolerance",
            "exact_dop853_refined_maximum_step",
            "cost_repeat_count",
            "model_to_direct_cost_ratio_ceiling",
        )
        for name in positive_fields:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        samples = int(round(self.segment_length / self.output_sample_step))
        if samples < 1 or not np.isclose(
            samples * self.output_sample_step,
            self.segment_length,
            atol=1e-12,
        ):
            raise ValueError(
                "segment_length must be divisible by output_sample_step"
            )
        if not 0.0 < self.fidelity_floor <= 1.0:
            raise ValueError("fidelity_floor must lie in (0, 1]")
        if not 0.0 < self.numerical_resolution_fraction < 1.0:
            raise ValueError(
                "numerical_resolution_fraction must lie between zero and one"
            )
        if not 0.0 < self.model_to_direct_cost_ratio_ceiling < 1.0:
            raise ValueError(
                "model_to_direct_cost_ratio_ceiling must lie between zero and one"
            )

    @property
    def maximum_total_branch_packets(self) -> int:
        """Count packets after the exact-center, four-branch reduction."""

        return 4 * self.maximum_packets_per_electronic_branch

    @property
    def maximum_raw_coordinate_count(self) -> int:
        """Count coefficient/displacement real coordinates in this chart."""

        return 16 * self.maximum_packets_per_electronic_branch

    def drive_protocol(self, parameters: DimerParameters) -> GaussianSineDrive:
        """Return the frozen causal pulse at zero and its delayed repeat."""

        return GaussianSineDrive.from_parameters(
            parameters,
            delays=(0.0, 8.0),
        )


@dataclass(frozen=True)
class BlindModelAudit:
    """Reference-blind acceptance status for one stored model trajectory."""

    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class HoldoutInitialConditions:
    """Central and symmetric model-chart preparations frozen before scoring."""

    central_parameters: np.ndarray
    plus_parameters: np.ndarray
    minus_parameters: np.ndarray
    closed_coordinates: np.ndarray
    initial_distance: float
    projected_kick_norm: float
    parameter_direction_norm: float
    kick_projection_relative_residual: float


@dataclass(frozen=True)
class FrozenHoldoutInputs:
    """Verified frozen model inputs loaded without opening a driven reference."""

    settings: MultiCoherentHoldoutSettings
    parameters: DimerParameters
    initial_conditions: HoldoutInitialConditions
    exact_initial_state_vectors: np.ndarray
    exact_initial_closed_coordinates: np.ndarray
    exact_initial_distance: float
    initial_distance_relative_disagreement: float
    pair_local_retained_norms: np.ndarray
    coordinate_scales: np.ndarray
    manifest_sha256: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class FrozenModelBatch:
    """Verified six-run model batch frozen before reference access."""

    prepared: FrozenHoldoutInputs
    directory: Path
    manifest_sha256: str
    manifest: dict[str, Any]


def build_holdout_initial_conditions(
    initial_parameters: np.ndarray,
    coordinate_scales: np.ndarray,
    *,
    settings: MultiCoherentHoldoutSettings,
    parameters: DimerParameters,
) -> HoldoutInitialConditions:
    """Construct the reference-blind central state and symmetric chart pair."""

    relative_dimension = 2 * settings.phonon_cutoff + 1
    expected_shape = (
        16 * settings.initial_packets_per_electronic_branch,
    )
    if np.asarray(initial_parameters).shape != expected_shape:
        raise ValueError(
            "initial_parameters do not match the frozen packets-per-branch count"
        )
    central = retract_multi_coherent_parameters(
        np.asarray(initial_parameters, dtype=float),
        relative_dimension=relative_dimension,
    )
    generator = normalized_diagonal_kick_generator(
        central,
        relative_dimension=relative_dimension,
    )
    kick = symmetric_projected_generator_kick(
        central,
        generator,
        relative_dimension=relative_dimension,
        step=settings.kick_size,
        regularization=settings.tangent_regularization,
        relative_damping=settings.relative_damping,
    )
    hierarchy = moment_hierarchy(4)
    center_amplitude = -sqrt(2.0) * parameters.coupling / parameters.omega_ph
    prepared = (
        central,
        kick.plus_parameters,
        kick.minus_parameters,
    )
    closed = np.asarray(
        [
            relative_state_closed_coordinates(
                multi_coherent_state(
                    state,
                    relative_dimension=relative_dimension,
                ),
                hierarchy,
                center_amplitude=center_amplitude,
            )
            for state in prepared
        ]
    )
    initial_distance = closed_coordinate_distance(
        closed[1],
        closed[2],
        coordinate_scales,
    )
    if initial_distance < 1e-6:
        raise ValueError("the symmetric holdout signal is unresolved")
    return HoldoutInitialConditions(
        central_parameters=central,
        plus_parameters=kick.plus_parameters,
        minus_parameters=kick.minus_parameters,
        closed_coordinates=closed,
        initial_distance=initial_distance,
        projected_kick_norm=kick.projected_direction_norm,
        parameter_direction_norm=kick.parameter_direction_norm,
        kick_projection_relative_residual=(
            kick.projection_relative_residual
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _frozen_source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    names = (
        "hubbard_dimer.py",
        "conditional_packets.py",
        "cone_correction.py",
        "exact_reference.py",
        "matrix_reference.py",
        "moment_hierarchy.py",
        "multi_coherent.py",
        "multi_coherent_holdout.py",
        "multi_coherent_holdout_cli.py",
        "multi_coherent_long_horizon.py",
        "multi_coherent_propagation.py",
        "multi_coherent_sealed_score.py",
        "multi_coherent_scores.py",
    )
    return {
        str((directory / name).resolve()): _sha256(directory / name)
        for name in names
    }


def _frozen_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def freeze_blind_multi_coherent_inputs(
    directory: Path,
    *,
    initial_parameters: np.ndarray,
    development_times: np.ndarray,
    development_closed_coordinates: np.ndarray,
    settings: MultiCoherentHoldoutSettings,
    parameters: DimerParameters,
    input_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Atomically freeze model inputs without evaluating the holdout reference."""

    target = Path(directory)
    if target.exists():
        raise FileExistsError(f"prepared directory already exists: {target}")
    times = np.asarray(development_times, dtype=float)
    development = np.asarray(development_closed_coordinates, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("development_times must contain at least two samples")
    if development.shape != (times.size, 31):
        raise ValueError("development coordinates must have shape (times, 31)")
    if not np.all(np.diff(times) > 0.0) or not np.isclose(times[0], 0.0):
        raise ValueError("development_times must increase strictly from zero")
    if times[-1] < settings.score_interval[1]:
        raise ValueError("development reference does not cover the score interval")
    scales = development_coordinate_scales(
        development,
        phonon_cutoff=settings.phonon_cutoff,
    )
    initial = build_holdout_initial_conditions(
        initial_parameters,
        scales,
        settings=settings,
        parameters=parameters,
    )
    exact_model = _build_exact_dimer_model(
        parameters,
        phonon_cutoff=settings.phonon_cutoff,
    )
    _, exact_ground_state = _ground_state(
        exact_model,
        eigensolver_tolerance=1e-12,
    )
    ground_factors = electron_relative_state(
        exact_ground_state,
        phonon_cutoff=settings.phonon_cutoff,
    )
    relative_dimension = 2 * settings.phonon_cutoff + 1
    relative_pair = (
        multi_coherent_state(
            initial.plus_parameters,
            relative_dimension=relative_dimension,
        ),
        multi_coherent_state(
            initial.minus_parameters,
            relative_dimension=relative_dimension,
        ),
    )
    pair_embeddings = tuple(
        electron_relative_product_to_local_state(
            state,
            ground_factors.center_state,
            phonon_cutoff=settings.phonon_cutoff,
        )
        for state in relative_pair
    )
    exact_initial_states = np.asarray(
        [
            exact_ground_state,
            pair_embeddings[0].state,
            pair_embeddings[1].state,
        ]
    )
    exact_initial_closed = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(
                _contract_matrix_state(exact_model, state)
            )
            for state in exact_initial_states
        ]
    )
    exact_initial_distance = closed_coordinate_distance(
        exact_initial_closed[1],
        exact_initial_closed[2],
        scales,
    )
    distance_disagreement = abs(
        exact_initial_distance - initial.initial_distance
    ) / initial.initial_distance
    if distance_disagreement > 0.01:
        raise ValueError(
            "independent model/local contractions disagree on the initial signal"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}-",
        dir=target.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        arrays_path = temporary / _PREPARED_ARRAYS
        np.savez_compressed(
            arrays_path,
            central_parameters=initial.central_parameters,
            plus_parameters=initial.plus_parameters,
            minus_parameters=initial.minus_parameters,
            initial_closed_coordinates=initial.closed_coordinates,
            exact_initial_state_vectors=exact_initial_states,
            exact_initial_closed_coordinates=exact_initial_closed,
            coordinate_scales=scales,
            development_times=times,
        )
        manifest: dict[str, Any] = {
            "schema": _PREPARED_SCHEMA,
            "status": "model_inputs_frozen_reference_unopened",
            "settings": asdict(settings),
            "parameters": {
                **asdict(parameters),
                "coupling": parameters.coupling,
            },
            "capacity": {
                "counting_convention": (
                    "exact_center_removed_four_electronic_branches"
                ),
                "initial_packets_per_electronic_branch": (
                    settings.initial_packets_per_electronic_branch
                ),
                "maximum_packets_per_electronic_branch": (
                    settings.maximum_packets_per_electronic_branch
                ),
                "maximum_total_branch_packets": (
                    settings.maximum_total_branch_packets
                ),
                "maximum_raw_coordinate_count": (
                    settings.maximum_raw_coordinate_count
                ),
            },
            "initialization": {
                "initial_distance": initial.initial_distance,
                "projected_kick_norm": initial.projected_kick_norm,
                "parameter_direction_norm": initial.parameter_direction_norm,
                "kick_projection_relative_residual": (
                    initial.kick_projection_relative_residual
                ),
                "exact_initial_distance": exact_initial_distance,
                "initial_distance_relative_disagreement": distance_disagreement,
                "center_relative_factorization": (
                    ground_factors.center_factorization
                ),
                "pair_local_retained_norms": [
                    embedding.retained_norm for embedding in pair_embeddings
                ],
            },
            "source_hashes": _frozen_source_hashes(),
            "input_hashes": dict(input_hashes or {}),
            "artifact_hashes": {_PREPARED_ARRAYS: _sha256(arrays_path)},
            "environment": _frozen_environment(),
        }
        manifest["manifest_sha256"] = _manifest_digest(manifest)
        (temporary / _PREPARED_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    return manifest


def load_frozen_multi_coherent_inputs(directory: Path) -> FrozenHoldoutInputs:
    """Verify and load a reference-unopened model-input manifest."""

    target = Path(directory)
    manifest_path = target / _PREPARED_MANIFEST
    arrays_path = target / _PREPARED_ARRAYS
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError("frozen multi-coherent inputs are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != _PREPARED_SCHEMA:
        raise ValueError("unknown frozen multi-coherent input schema")
    expected_digest = _manifest_digest(manifest)
    if manifest.get("manifest_sha256") != expected_digest:
        raise ValueError("frozen input manifest digest mismatch")
    if manifest.get("status") != "model_inputs_frozen_reference_unopened":
        raise ValueError("frozen input manifest has the wrong status")
    if manifest.get("environment") != _frozen_environment():
        raise ValueError("frozen execution environment changed")
    if manifest["artifact_hashes"].get(_PREPARED_ARRAYS) != _sha256(
        arrays_path
    ):
        raise ValueError("frozen initial-condition hash mismatch")
    for name, digest in manifest["source_hashes"].items():
        path = Path(name)
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"frozen source hash changed: {name}")

    settings_values = dict(manifest["settings"])
    settings_values["score_interval"] = tuple(
        settings_values["score_interval"]
    )
    settings_values["sensitivity_interval"] = tuple(
        settings_values["sensitivity_interval"]
    )
    settings = MultiCoherentHoldoutSettings(**settings_values)
    parameter_values = dict(manifest["parameters"])
    parameter_values.pop("coupling", None)
    parameters = DimerParameters(**parameter_values)
    with np.load(arrays_path) as arrays:
        coordinate_scales = np.asarray(
            arrays["coordinate_scales"],
            dtype=float,
        )
        initial = HoldoutInitialConditions(
            central_parameters=np.asarray(
                arrays["central_parameters"],
                dtype=float,
            ),
            plus_parameters=np.asarray(arrays["plus_parameters"], dtype=float),
            minus_parameters=np.asarray(
                arrays["minus_parameters"],
                dtype=float,
            ),
            closed_coordinates=np.asarray(
                arrays["initial_closed_coordinates"],
                dtype=float,
            ),
            initial_distance=float(manifest["initialization"]["initial_distance"]),
            projected_kick_norm=float(
                manifest["initialization"]["projected_kick_norm"]
            ),
            parameter_direction_norm=float(
                manifest["initialization"]["parameter_direction_norm"]
            ),
            kick_projection_relative_residual=float(
                manifest["initialization"][
                    "kick_projection_relative_residual"
                ]
            ),
        )
        exact_initial_state_vectors = np.asarray(
            arrays["exact_initial_state_vectors"],
            dtype=complex,
        )
        exact_initial_closed_coordinates = np.asarray(
            arrays["exact_initial_closed_coordinates"],
            dtype=float,
        )
    return FrozenHoldoutInputs(
        settings=settings,
        parameters=parameters,
        initial_conditions=initial,
        exact_initial_state_vectors=exact_initial_state_vectors,
        exact_initial_closed_coordinates=exact_initial_closed_coordinates,
        exact_initial_distance=float(
            manifest["initialization"]["exact_initial_distance"]
        ),
        initial_distance_relative_disagreement=float(
            manifest["initialization"][
                "initial_distance_relative_disagreement"
            ]
        ),
        pair_local_retained_norms=np.asarray(
            manifest["initialization"]["pair_local_retained_norms"],
            dtype=float,
        ),
        coordinate_scales=coordinate_scales,
        manifest_sha256=expected_digest,
        manifest=manifest,
    )


def run_frozen_multi_coherent_model_trajectory(
    prepared_directory: Path,
    run_directory: Path,
    *,
    member: str,
    resolution: str,
) -> dict[str, Any]:
    """Run one frozen, reference-blind member of the model batch."""

    from .multi_coherent_long_horizon import (
        run_segmented_multi_coherent_horizon,
    )

    prepared = load_frozen_multi_coherent_inputs(prepared_directory)
    if member not in ("central", "plus", "minus"):
        raise ValueError("member must be central, plus, or minus")
    if resolution not in ("coarse", "fine"):
        raise ValueError("resolution must be coarse or fine")
    target = Path(run_directory)
    if target.exists():
        raise FileExistsError(f"model run directory already exists: {target}")
    settings = prepared.settings
    initial_parameters = {
        "central": prepared.initial_conditions.central_parameters,
        "plus": prepared.initial_conditions.plus_parameters,
        "minus": prepared.initial_conditions.minus_parameters,
    }[member]
    if resolution == "coarse":
        maximum_step = settings.coarse_maximum_step
        relative_tolerance = settings.coarse_relative_tolerance
        absolute_tolerance = settings.coarse_absolute_tolerance
    else:
        maximum_step = settings.fine_maximum_step
        relative_tolerance = settings.fine_relative_tolerance
        absolute_tolerance = settings.fine_absolute_tolerance
    summary = run_segmented_multi_coherent_horizon(
        target,
        gate_directory=None,
        parameters=prepared.parameters,
        final_time=settings.final_time,
        segment_length=settings.segment_length,
        output_sample_step=settings.output_sample_step,
        segment_timeout_seconds=60.0,
        maximum_step=maximum_step,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        phonon_cutoff=settings.phonon_cutoff,
        packet_count=settings.initial_packets_per_electronic_branch,
        tangent_regularization=settings.tangent_regularization,
        relative_damping=settings.relative_damping,
        adaptive_capacity=True,
        maximum_packet_count=(
            settings.maximum_packets_per_electronic_branch
        ),
        spawn_relative_residual_threshold=(
            settings.spawn_relative_residual_threshold
        ),
        spawn_absolute_residual_threshold=(
            settings.spawn_absolute_residual_threshold
        ),
        spawn_fit_maximum_iterations=(
            settings.spawn_fit_maximum_iterations
        ),
        spawn_fit_population_size=settings.spawn_fit_population_size,
        spawn_seed=settings.spawn_seed,
        compare_exact=False,
        drive_protocol=settings.drive_protocol(prepared.parameters),
        initial_parameters_override=initial_parameters,
    )
    audit = audit_blind_multi_coherent_summary(summary, settings)
    audit_payload = {
        "schema": "paper5.multi_coherent.blind_model_audit.v1",
        "prepared_manifest_sha256": prepared.manifest_sha256,
        "member": member,
        "resolution": resolution,
        "passed": audit.passed,
        "failures": list(audit.failures),
    }
    (target / "blind_model_audit.json").write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _validated_blind_model_run(
    prepared: FrozenHoldoutInputs,
    run_directory: Path,
    *,
    member: str,
    resolution: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify one stored blind run against its frozen semantic contract."""

    summary_path = run_directory / "summary.json"
    arrays_path = run_directory / "segmented_horizon.npz"
    runtime_manifest_path = run_directory / "runtime_manifest.json"
    audit_path = run_directory / "blind_model_audit.json"
    required = (
        summary_path,
        arrays_path,
        runtime_manifest_path,
        audit_path,
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"blind model run is incomplete: {run_directory.name}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = audit_blind_multi_coherent_summary(summary, prepared.settings)
    if not audit.passed:
        raise RuntimeError(
            f"blind model run failed its audit: {run_directory.name}: "
            + ", ".join(audit.failures)
        )
    settings = prepared.settings
    expected_numerics = {
        "maximum_step": (
            settings.coarse_maximum_step
            if resolution == "coarse"
            else settings.fine_maximum_step
        ),
        "relative_tolerance": (
            settings.coarse_relative_tolerance
            if resolution == "coarse"
            else settings.fine_relative_tolerance
        ),
        "absolute_tolerance": (
            settings.coarse_absolute_tolerance
            if resolution == "coarse"
            else settings.fine_absolute_tolerance
        ),
    }
    for name, expected in expected_numerics.items():
        if summary["parameters"].get(name) != expected:
            raise ValueError(
                f"{run_directory.name} has a {resolution} "
                f"numerical-setting mismatch: {name}"
            )
    if summary["parameters"].get("output_sample_step") != (
        settings.output_sample_step
    ):
        raise ValueError(f"{run_directory.name} has the wrong output sample step")
    expected_initial = {
        "central": prepared.initial_conditions.central_parameters,
        "plus": prepared.initial_conditions.plus_parameters,
        "minus": prepared.initial_conditions.minus_parameters,
    }[member]
    if summary["initialization"].get("parameter_sha256") != _array_digest(
        expected_initial
    ):
        raise ValueError(f"{run_directory.name} has the wrong initial state")

    with np.load(arrays_path) as arrays:
        forbidden = {
            "state_fidelity",
            "exact_closed_coordinates",
            "closed_coordinate_relative_error",
        }
        if forbidden.intersection(arrays.files):
            raise ValueError(
                f"{run_directory.name} contains exact-reference score arrays"
            )

    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        not audit_payload.get("passed")
        or audit_payload.get("prepared_manifest_sha256")
        != prepared.manifest_sha256
        or audit_payload.get("member") != member
        or audit_payload.get("resolution") != resolution
    ):
        raise ValueError(f"{run_directory.name} audit provenance mismatch")

    runtime = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    if runtime.get("status") != "complete":
        raise ValueError(f"{run_directory.name} runtime manifest is incomplete")
    runtime_sources = runtime.get("source_hashes", {})
    frozen_sources = prepared.manifest.get("source_hashes", {})
    if not runtime_sources or any(
        frozen_sources.get(name) != digest
        for name, digest in runtime_sources.items()
    ):
        raise ValueError(f"{run_directory.name} runtime source provenance mismatch")
    for file_name, digest in runtime.get("artifact_hashes", {}).items():
        path = run_directory / file_name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(
                f"{run_directory.name} runtime artifact hash mismatch: {file_name}"
            )

    return summary, {
        "summary_sha256": _sha256(summary_path),
        "arrays_sha256": _sha256(arrays_path),
        "runtime_manifest_sha256": _sha256(runtime_manifest_path),
        "blind_audit_sha256": _sha256(audit_path),
    }


def _validated_model_cost_manifest(
    prepared: FrozenHoldoutInputs,
    path: Path,
) -> dict[str, Any]:
    """Verify the sealed three-repeat model cost record and its artifacts."""

    if not path.is_file():
        raise FileNotFoundError("the three-repeat model cost manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "paper5.multi_coherent.model_cost.v1":
        raise ValueError("unknown model cost manifest schema")
    if manifest.get("manifest_sha256") != _manifest_digest(manifest):
        raise ValueError("model cost manifest digest mismatch")
    if (
        manifest.get("status") != "model_cost_frozen_reference_unopened"
        or manifest.get("prepared_manifest_sha256")
        != prepared.manifest_sha256
    ):
        raise ValueError("model cost manifest provenance mismatch")
    if manifest.get("warmup") != "fine_central":
        raise ValueError("model cost manifest has the wrong warmup")
    if manifest.get("source_hashes") != prepared.manifest.get("source_hashes"):
        raise ValueError("model cost manifest source provenance mismatch")

    repeat_count = prepared.settings.cost_repeat_count
    expected_repeats = {
        f"cost_model_repeat_{repeat}" for repeat in range(1, repeat_count + 1)
    }
    repeats = manifest.get("repeats", {})
    walls = np.asarray(manifest.get("wall_seconds", []), dtype=float)
    residents = np.asarray(
        manifest.get("maximum_resident_set_bytes", []),
        dtype=float,
    )
    if (
        set(repeats) != expected_repeats
        or walls.shape != (repeat_count,)
        or residents.shape != (repeat_count,)
        or not np.all(np.isfinite(walls))
        or not np.all(np.isfinite(residents))
        or np.any(walls <= 0.0)
        or np.any(residents <= 0.0)
    ):
        raise ValueError("model cost manifest has invalid repetitions")
    if not np.isclose(
        manifest.get("median_wall_seconds", np.nan),
        np.median(walls),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("model cost manifest has the wrong wall-time median")
    if manifest.get("maximum_resident_set_bytes_over_repeats") != int(
        np.max(residents)
    ):
        raise ValueError("model cost manifest has the wrong resident-set maximum")

    file_keys = {
        "summary_sha256": "summary.json",
        "arrays_sha256": "segmented_horizon.npz",
        "runtime_manifest_sha256": "runtime_manifest.json",
        "blind_audit_sha256": "blind_model_audit.json",
    }
    for repeat_name, record in repeats.items():
        for digest_key, file_name in file_keys.items():
            artifact = path.parent / repeat_name / file_name
            if not artifact.is_file() or _sha256(artifact) != record.get(
                digest_key
            ):
                raise ValueError(
                    "model cost artifact hash mismatch: "
                    f"{repeat_name}/{file_name}"
                )
    return manifest


def seal_frozen_multi_coherent_model_batch(
    prepared_directory: Path,
    batch_directory: Path,
) -> dict[str, Any]:
    """Verify all six blind model runs and freeze them before exact scoring."""

    prepared = load_frozen_multi_coherent_inputs(prepared_directory)
    batch = Path(batch_directory)
    manifest_path = batch / _MODEL_BATCH_MANIFEST
    if manifest_path.exists():
        raise FileExistsError("model batch is already sealed")
    model_cost_path = batch / _MODEL_COST_MANIFEST
    _validated_model_cost_manifest(prepared, model_cost_path)
    run_records: dict[str, Any] = {}
    for resolution in ("coarse", "fine"):
        for member in ("central", "plus", "minus"):
            run_name = f"{resolution}_{member}"
            run_directory = batch / run_name
            _, run_records[run_name] = _validated_blind_model_run(
                prepared,
                run_directory,
                member=member,
                resolution=resolution,
            )
    manifest: dict[str, Any] = {
        "schema": "paper5.multi_coherent.blind_model_batch.v1",
        "status": "model_outputs_frozen_reference_unopened",
        "prepared_manifest_sha256": prepared.manifest_sha256,
        "model_cost_manifest_sha256": _sha256(model_cost_path),
        "runs": run_records,
        "source_hashes": _frozen_source_hashes(),
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def seal_frozen_multi_coherent_model_cost(
    prepared_directory: Path,
    batch_directory: Path,
) -> dict[str, Any]:
    """Freeze three independent fine-central model timing repetitions."""

    prepared = load_frozen_multi_coherent_inputs(prepared_directory)
    batch = Path(batch_directory)
    manifest_path = batch / _MODEL_COST_MANIFEST
    if manifest_path.exists():
        raise FileExistsError("model cost repetitions are already sealed")
    wall_seconds: list[float] = []
    resident_bytes: list[int] = []
    repeats: dict[str, Any] = {}
    for repeat in range(1, prepared.settings.cost_repeat_count + 1):
        name = f"cost_model_repeat_{repeat}"
        directory = batch / name
        summary, repeats[name] = _validated_blind_model_run(
            prepared,
            directory,
            member="central",
            resolution="fine",
        )
        usage = summary.get("resource_usage", {})
        wall = float(usage.get("wall_seconds", 0.0))
        resident = int(usage.get("maximum_resident_set_bytes", 0))
        if wall <= 0.0 or resident <= 0:
            raise ValueError(f"model cost repetition lacks resource data: {name}")
        wall_seconds.append(wall)
        resident_bytes.append(resident)
    manifest: dict[str, Any] = {
        "schema": "paper5.multi_coherent.model_cost.v1",
        "status": "model_cost_frozen_reference_unopened",
        "prepared_manifest_sha256": prepared.manifest_sha256,
        "warmup": "fine_central",
        "wall_seconds": wall_seconds,
        "maximum_resident_set_bytes": resident_bytes,
        "median_wall_seconds": float(np.median(wall_seconds)),
        "maximum_resident_set_bytes_over_repeats": max(resident_bytes),
        "repeats": repeats,
        "source_hashes": _frozen_source_hashes(),
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_frozen_multi_coherent_model_batch(
    prepared_directory: Path,
    batch_directory: Path,
    *,
    require_unconsumed: bool = False,
) -> FrozenModelBatch:
    """Verify every blind model artifact and its source-locked batch manifest."""

    prepared = load_frozen_multi_coherent_inputs(prepared_directory)
    batch = Path(batch_directory)
    manifest_path = batch / _MODEL_BATCH_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError("model batch manifest is missing")
    if require_unconsumed and (batch / _SCORE_CONSUMPTION_RECEIPT).exists():
        raise RuntimeError("the frozen model batch has already been consumed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "paper5.multi_coherent.blind_model_batch.v1":
        raise ValueError("unknown blind model batch schema")
    digest = _manifest_digest(manifest)
    if manifest.get("manifest_sha256") != digest:
        raise ValueError("blind model batch manifest digest mismatch")
    if manifest.get("status") != "model_outputs_frozen_reference_unopened":
        raise ValueError("blind model batch has the wrong status")
    if manifest.get("prepared_manifest_sha256") != prepared.manifest_sha256:
        raise ValueError("blind model batch refers to the wrong prepared inputs")
    model_cost_path = batch / _MODEL_COST_MANIFEST
    if manifest.get("model_cost_manifest_sha256") != _sha256(model_cost_path):
        raise ValueError("blind model batch has the wrong cost manifest")
    _validated_model_cost_manifest(prepared, model_cost_path)
    for name, expected_digest in manifest.get("source_hashes", {}).items():
        path = Path(name)
        if not path.is_file() or _sha256(path) != expected_digest:
            raise ValueError(f"frozen batch source hash changed: {name}")
    file_keys = {
        "summary_sha256": "summary.json",
        "arrays_sha256": "segmented_horizon.npz",
        "runtime_manifest_sha256": "runtime_manifest.json",
        "blind_audit_sha256": "blind_model_audit.json",
    }
    expected_runs = {
        f"{resolution}_{member}"
        for resolution in ("coarse", "fine")
        for member in ("central", "plus", "minus")
    }
    if set(manifest.get("runs", {})) != expected_runs:
        raise ValueError("blind model batch does not contain exactly six runs")
    for run_name, record in manifest["runs"].items():
        for digest_key, file_name in file_keys.items():
            path = batch / run_name / file_name
            if not path.is_file() or _sha256(path) != record.get(digest_key):
                raise ValueError(
                    f"blind model artifact hash mismatch: {run_name}/{file_name}"
                )
    return FrozenModelBatch(
        prepared=prepared,
        directory=batch,
        manifest_sha256=digest,
        manifest=manifest,
    )


def audit_blind_multi_coherent_summary(
    summary: Mapping[str, Any],
    settings: MultiCoherentHoldoutSettings,
) -> BlindModelAudit:
    """Audit completion, frozen settings, capacity, and structural outputs."""

    failures: list[str] = []
    parameters = summary.get("parameters", {})
    initialization = summary.get("initialization", {})
    capacity = summary.get("capacity", {})
    tangent = summary.get("tangent_diagnostics", {})
    physicality = summary.get("physicality", {})
    work = summary.get("work_balance", {})

    if summary.get("status") != "complete":
        failures.append("model_trajectory_incomplete")
    expected_parameters = {
        "target_final_time": settings.final_time,
        "phonon_cutoff": settings.phonon_cutoff,
        "packet_count": settings.initial_packets_per_electronic_branch,
        "maximum_packet_count": settings.maximum_packets_per_electronic_branch,
        "tangent_regularization": settings.tangent_regularization,
        "relative_damping": settings.relative_damping,
    }
    for name, expected in expected_parameters.items():
        if parameters.get(name) != expected:
            failures.append(f"frozen_parameter_mismatch:{name}")
    expected_drive = {
        "amplitude": 1.0,
        "pulse_width": 1.0,
        "delays": [0.0, 8.0],
    }
    if parameters.get("drive_protocol") != expected_drive:
        failures.append("frozen_parameter_mismatch:drive_protocol")
    if (
        summary.get("offline_exact_comparison") is not None
        or initialization.get(
            "exact_reference_used_after_t0_by_model_rhs",
            True,
        )
        or capacity.get("online_exact_reference_used", True)
    ):
        failures.append("exact_reference_was_opened")
    if capacity.get("final_total_branch_packets", np.inf) > (
        settings.maximum_total_branch_packets
    ):
        failures.append("total_branch_packet_cap_exceeded")
    if capacity.get("final_raw_coordinate_count", np.inf) > (
        settings.maximum_raw_coordinate_count
    ):
        failures.append("raw_coordinate_cap_exceeded")
    if tangent.get("geometric_gram_relative_threshold") != (
        settings.geometric_gram_relative_threshold
    ):
        failures.append("geometric_rank_threshold_mismatch")
    geometric_ranks = tangent.get("geometric_ranks", [])
    if not geometric_ranks:
        failures.append("geometric_tangent_rank_missing")
    elif max(geometric_ranks) > settings.maximum_geometric_tangent_rank:
        failures.append("geometric_tangent_rank_cap_exceeded")
    if physicality.get("maximum_norm_drift", np.inf) > 1e-10:
        failures.append("normalized_state_drift")
    if physicality.get("minimum_electron_density_eigenvalue", -np.inf) < -1e-10:
        failures.append("electronic_physicality_failure")
    if physicality.get("minimum_relative_uncertainty_margin", -np.inf) < -1e-10:
        failures.append("bosonic_physicality_failure")
    if work.get("maximum_normalized_residual", np.inf) > (
        settings.work_residual_ceiling
    ):
        failures.append("work_residual_ceiling_exceeded")
    ordered = tuple(sorted(set(failures)))
    return BlindModelAudit(passed=not ordered, failures=ordered)


__all__ = [
    "BlindModelAudit",
    "FrozenHoldoutInputs",
    "FrozenModelBatch",
    "HoldoutInitialConditions",
    "MultiCoherentHoldoutSettings",
    "audit_blind_multi_coherent_summary",
    "build_holdout_initial_conditions",
    "freeze_blind_multi_coherent_inputs",
    "load_frozen_multi_coherent_inputs",
    "load_frozen_multi_coherent_model_batch",
    "run_frozen_multi_coherent_model_trajectory",
    "seal_frozen_multi_coherent_model_batch",
    "seal_frozen_multi_coherent_model_cost",
]
