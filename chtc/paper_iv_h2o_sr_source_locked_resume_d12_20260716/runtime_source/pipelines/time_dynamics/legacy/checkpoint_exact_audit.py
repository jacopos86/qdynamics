#!/usr/bin/env python3
"""Exact benchmark/audit helper for HH realtime checkpoint control."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from pipelines.time_dynamics.adapters.observables import observable_family_key
from src.quantum.drives_time_potential import reference_method_name
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


@dataclass(frozen=True)
class _ExactReferenceMetadata:
    kind: str
    reference_method: str | None
    reference_steps_multiplier: int
    projection_time_sampling: str
    geometry_sample_time_policy: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "reference_method": (None if self.reference_method is None else str(self.reference_method)),
            "reference_steps_multiplier": int(self.reference_steps_multiplier),
            "projection_time_sampling": str(self.projection_time_sampling),
            "geometry_sample_time_policy": str(self.geometry_sample_time_policy),
        }


@dataclass(frozen=True)
class _ExactReferenceArtifacts:
    metadata: _ExactReferenceMetadata
    reference_states: tuple[np.ndarray, ...] | None = None
    exact_evals: np.ndarray | None = None
    exact_evecs: np.ndarray | None = None
    exact_coeffs0: np.ndarray | None = None


def _freeze_array(arr: np.ndarray | Sequence[Any], *, dtype: Any | None = None) -> np.ndarray:
    frozen = np.ascontiguousarray(np.asarray(arr, dtype=dtype)).copy()
    frozen.setflags(write=False)
    return frozen


def _exact_array_fingerprint(arr: np.ndarray | Sequence[Any], *, dtype: Any | None = None) -> dict[str, Any]:
    frozen = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    digest = hashlib.sha1(frozen.view(np.uint8).tobytes()).hexdigest()
    return {
        "shape": [int(x) for x in frozen.shape],
        "dtype": str(frozen.dtype),
        "sha1": str(digest),
    }


def _drive_config_payload(drive_config: Any | None) -> dict[str, Any] | None:
    if drive_config is None:
        return None
    return {
        "enabled": bool(drive_config.enabled),
        "n_sites": int(drive_config.n_sites),
        "ordering": str(drive_config.ordering),
        "drive_A": float(drive_config.drive_A),
        "drive_omega": float(drive_config.drive_omega),
        "drive_tbar": float(drive_config.drive_tbar),
        "drive_phi": float(drive_config.drive_phi),
        "drive_pattern": str(drive_config.drive_pattern),
        "drive_custom_weights": (
            None
            if drive_config.drive_custom_weights is None
            else [float(x) for x in drive_config.drive_custom_weights]
        ),
        "drive_include_identity": bool(drive_config.drive_include_identity),
        "drive_time_sampling": str(drive_config.drive_time_sampling),
        "drive_t0": float(drive_config.drive_t0),
        "exact_steps_multiplier": int(drive_config.exact_steps_multiplier),
    }


def _polynomial_identity_payload(poly: Any, *, tol: float = 1.0e-12) -> list[dict[str, Any]]:
    if poly is None:
        return []
    coeff_map: dict[str, complex] = {}
    for term in tuple(poly.return_polynomial()):
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        coeff_map[label] = coeff_map.get(label, 0.0 + 0.0j) + coeff
    return [
        {
            "label": str(label),
            "real": float(np.real(coeff)),
            "imag": float(np.imag(coeff)),
        }
        for label, coeff in sorted(coeff_map.items())
        if abs(complex(coeff)) > float(tol)
    ]


def _drive_model_identity_payload(drive_model: Any | None) -> dict[str, Any] | None:
    if drive_model is None:
        return None
    profile_payload = getattr(drive_model, "profile_payload", None)
    return {
        "family_key": str(getattr(drive_model, "family_key", "")),
        "operator_label": str(getattr(drive_model, "operator_label", "")),
        "drive_term_count": int(getattr(drive_model, "drive_term_count", 0)),
        "spatial_weights": [
            float(x)
            for x in tuple(getattr(drive_model, "spatial_weights", ()))
        ],
        "profile_payload": (
            None if profile_payload is None else dict(profile_payload)
        ),
        "drive_poly": _polynomial_identity_payload(getattr(drive_model, "drive_poly", None)),
    }


def _projection_sample_time(*, time_start: float, time_stop: float, sampling: str) -> float:
    sampling_norm = str(sampling).strip().lower()
    if sampling_norm not in {"left", "midpoint", "right"}:
        raise ValueError(f"Unsupported drive_time_sampling {sampling!r}.")
    if sampling_norm == "left":
        return float(time_start)
    if sampling_norm == "midpoint":
        return 0.5 * (float(time_start) + float(time_stop))
    return float(time_stop)


def _evolve_state_under_constant_hamiltonian(
    psi_state: np.ndarray,
    hmat: np.ndarray,
    dt: float,
) -> np.ndarray:
    psi_arr = np.asarray(psi_state, dtype=complex).reshape(-1)
    if abs(float(dt)) <= 1.0e-15:
        return np.asarray(psi_arr, dtype=complex).reshape(-1).copy()
    evals, evecs = np.linalg.eigh(np.asarray(hmat, dtype=complex))
    coeffs0 = np.asarray(np.conjugate(evecs).T @ psi_arr, dtype=complex).reshape(-1)
    phases = np.exp(-1.0j * np.asarray(evals, dtype=float) * float(dt))
    return np.asarray(evecs @ (phases * coeffs0), dtype=complex).reshape(-1)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(out) else float(out)


def _abs_error_or_none(lhs: Any, rhs: Any) -> float | None:
    lhs_f = _finite_or_none(lhs)
    rhs_f = _finite_or_none(rhs)
    if lhs_f is None or rhs_f is None:
        return None
    return float(abs(float(lhs_f) - float(rhs_f)))


def _matching_vector_or_none(
    snapshot: Mapping[str, Any],
) -> tuple[list[float], str | None, list[str] | None]:
    try:
        values = np.asarray(snapshot.get("site_occupations", ()), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        values = np.asarray([], dtype=float)
    payload_values = (
        [float(x) for x in values.tolist()]
        if values.size > 0 and np.all(np.isfinite(values))
        else []
    )
    label_raw = snapshot.get("site_occupations_label", None)
    label = None if label_raw in {None, ""} else str(label_raw)
    component_labels_raw = snapshot.get("site_occupations_component_labels", None)
    component_labels = (
        [str(x) for x in component_labels_raw]
        if isinstance(component_labels_raw, Sequence)
        and not isinstance(component_labels_raw, (str, bytes, bytearray))
        else None
    )
    return payload_values, label, component_labels


def _vector_error_max_or_none(lhs: Sequence[float], rhs: Sequence[float]) -> float | None:
    lhs_arr = np.asarray(lhs, dtype=float).reshape(-1)
    rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
    if (
        lhs_arr.size <= 0
        or lhs_arr.size != rhs_arr.size
        or not np.all(np.isfinite(lhs_arr))
        or not np.all(np.isfinite(rhs_arr))
    ):
        return None
    return float(np.max(np.abs(lhs_arr - rhs_arr)))


def _vector_abs_error_or_none(lhs: Sequence[float], rhs: Sequence[float]) -> list[float] | None:
    lhs_arr = np.asarray(lhs, dtype=float).reshape(-1)
    rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
    if (
        lhs_arr.size <= 0
        or lhs_arr.size != rhs_arr.size
        or not np.all(np.isfinite(lhs_arr))
        or not np.all(np.isfinite(rhs_arr))
    ):
        return None
    return [float(x) for x in np.abs(lhs_arr - rhs_arr).tolist()]


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        value = _finite_or_none(row.get(key))
        if value is not None:
            values.append(float(value))
    return values


def _summary_stats_for_values(values: Sequence[float], *, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {}
    return {
        f"initial_{prefix}": float(arr[0]),
        f"final_{prefix}": float(arr[-1]),
        f"mean_{prefix}": float(np.mean(arr)),
        f"max_{prefix}": float(np.max(arr)),
    }


def _benchmark_snapshot_payload(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(snapshot)
    payload["doublon"] = _finite_or_none(payload.get("doublon"))
    payload["staggered"] = _finite_or_none(payload.get("staggered"))
    site_values, label, component_labels = _matching_vector_or_none(payload)
    payload["site_occupations"] = [float(x) for x in site_values]
    if label is not None:
        payload["site_occupations_label"] = str(label)
    if component_labels is not None:
        payload["site_occupations_component_labels"] = [str(x) for x in component_labels]
    return payload


class RealtimeExactAuditHelper:
    def __init__(
        self,
        *,
        h_poly: Any,
        hmat: np.ndarray,
        psi_initial: np.ndarray,
        times: np.ndarray,
        drive_config: Any | None,
        drive_profile: Mapping[str, Any] | None,
        drive_coeff_provider_exyz: Any | None,
        drive_model: Any | None,
        exact_reference_cache: MutableMapping[str, object] | None = None,
    ) -> None:
        self.h_poly = h_poly
        self.hmat = np.asarray(hmat, dtype=complex)
        self.psi_initial = np.asarray(psi_initial, dtype=complex).reshape(-1)
        self.times = np.asarray(times, dtype=float).reshape(-1)
        self.drive_config = drive_config
        self.drive_profile = None if drive_profile is None else dict(drive_profile)
        self.drive_coeff_provider_exyz = drive_coeff_provider_exyz
        self.drive_model = drive_model
        self._exact_reference_cache = exact_reference_cache
        self._reference_states: tuple[np.ndarray, ...] | None = None
        self._exact_evals: np.ndarray | None = None
        self._exact_evecs: np.ndarray | None = None
        self._exact_coeffs0: np.ndarray | None = None
        self._exact_reference_metadata: _ExactReferenceMetadata | None = None
        self._exact_reference_cache_key: str | None = None

    def _build_exact_reference_metadata(self) -> _ExactReferenceMetadata:
        if self.drive_config is None:
            return _ExactReferenceMetadata(
                kind="static_exact_reference_from_replay_seed",
                reference_method=None,
                reference_steps_multiplier=1,
                projection_time_sampling="left",
                geometry_sample_time_policy="checkpoint_time",
            )
        time_sampling = str(self.drive_config.drive_time_sampling)
        return _ExactReferenceMetadata(
            kind="driven_piecewise_constant_reference_from_replay_seed",
            reference_method=str(reference_method_name(time_sampling)),
            reference_steps_multiplier=int(self.drive_config.exact_steps_multiplier),
            projection_time_sampling=str(time_sampling),
            geometry_sample_time_policy=(
                f"interval_{str(time_sampling).strip().lower()}_plus_t0_with_final_endpoint_fallback"
            ),
        )

    def _build_exact_reference_cache_key(self, metadata: _ExactReferenceMetadata) -> str:
        payload = {
            "metadata": metadata.to_payload(),
            "hmat": _exact_array_fingerprint(self.hmat, dtype=complex),
            "psi_initial": _exact_array_fingerprint(self.psi_initial, dtype=complex),
            "times": _exact_array_fingerprint(self.times, dtype=float),
            "drive_config": _drive_config_payload(self.drive_config),
            "drive_model_identity": _drive_model_identity_payload(self.drive_model),
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return str(hashlib.sha1(text.encode("utf-8")).hexdigest())

    def _build_driven_reference_states_from_drive_model(self) -> tuple[np.ndarray, ...]:
        if self.drive_config is None or self.drive_model is None:
            raise ValueError("Driven exact reference from drive_model requires drive_config and drive_model.")
        family_key = str(getattr(self.drive_model, "family_key", "")).strip().lower()
        if not family_key:
            raise ValueError("Driven exact reference from drive_model requires a non-empty family_key.")
        if self.times.size <= 0:
            return tuple()
        substeps = int(self.drive_config.exact_steps_multiplier)
        if substeps < 1:
            raise ValueError("Driven exact reference requires exact_steps_multiplier >= 1.")
        sampling = str(self.drive_config.drive_time_sampling)
        drive_t0 = float(self.drive_config.drive_t0)
        hmat_static = np.asarray(self.hmat, dtype=complex)
        hmat_drive = np.asarray(hamiltonian_matrix(self.drive_model.drive_poly), dtype=complex)
        if hmat_drive.shape != hmat_static.shape:
            raise ValueError("Driven exact reference drive operator shape mismatch.")
        psi_current = np.asarray(self.psi_initial, dtype=complex).reshape(-1).copy()
        states = [_freeze_array(psi_current, dtype=complex)]
        for interval_idx in range(int(self.times.size) - 1):
            time_start = float(self.times[int(interval_idx)])
            time_stop = float(self.times[int(interval_idx) + 1])
            dt = (float(time_stop) - float(time_start)) / float(substeps)
            for substep_idx in range(substeps):
                sub_start = float(time_start) + float(substep_idx) * float(dt)
                sub_stop = float(time_start) + float(substep_idx + 1) * float(dt)
                sample_time = _projection_sample_time(
                    time_start=float(sub_start),
                    time_stop=float(sub_stop),
                    sampling=str(sampling),
                )
                physical_time = float(sample_time) + float(drive_t0)
                drive_coeff = float(self.drive_model.coefficient_at(float(physical_time)))
                hmat_step = (
                    hmat_static
                    if abs(float(drive_coeff)) <= 1.0e-15
                    else np.asarray(hmat_static + (float(drive_coeff) * hmat_drive), dtype=complex)
                )
                psi_current = _evolve_state_under_constant_hamiltonian(
                    psi_current,
                    hmat_step,
                    float(dt),
                )
            states.append(_freeze_array(psi_current, dtype=complex))
        return tuple(states)

    def _build_exact_reference_artifacts(self, metadata: _ExactReferenceMetadata) -> _ExactReferenceArtifacts:
        if self.drive_config is not None:
            if self.drive_model is not None:
                return _ExactReferenceArtifacts(
                    metadata=metadata,
                    reference_states=self._build_driven_reference_states_from_drive_model(),
                )
            from pipelines.hardcoded.hh_fixed_manifold_measured import (
                FixedManifoldDriveConfig,
                FixedManifoldMeasuredConfig,
                _build_driven_reference_states,
            )

            reference_states = _build_driven_reference_states(
                psi_initial=np.asarray(self.psi_initial, dtype=complex),
                times=self.times,
                hmat_static=np.asarray(self.hmat, dtype=complex),
                h_poly_static=self.h_poly,
                drive_coeff_provider_exyz=self.drive_coeff_provider_exyz,
                drive_cfg=FixedManifoldDriveConfig(
                    enable_drive=True,
                    drive_A=float(self.drive_config.drive_A),
                    drive_omega=float(self.drive_config.drive_omega),
                    drive_tbar=float(self.drive_config.drive_tbar),
                    drive_phi=float(self.drive_config.drive_phi),
                    drive_pattern=str(self.drive_config.drive_pattern),
                    drive_custom_s=None,
                    drive_include_identity=bool(self.drive_config.drive_include_identity),
                    drive_time_sampling=str(self.drive_config.drive_time_sampling),
                    drive_t0=float(self.drive_config.drive_t0),
                    exact_steps_multiplier=int(self.drive_config.exact_steps_multiplier),
                ),
                geom_cfg=FixedManifoldMeasuredConfig(),
            )
            return _ExactReferenceArtifacts(
                metadata=metadata,
                reference_states=tuple(
                    _freeze_array(np.asarray(state, dtype=complex).reshape(-1), dtype=complex)
                    for state in reference_states
                ),
            )

        exact_evals, exact_evecs = np.linalg.eigh(self.hmat)
        exact_evals_arr = _freeze_array(np.asarray(exact_evals, dtype=float).reshape(-1), dtype=float)
        exact_evecs_arr = _freeze_array(np.asarray(exact_evecs, dtype=complex), dtype=complex)
        exact_coeffs0_arr = _freeze_array(
            np.asarray(exact_evecs_arr.conj().T @ self.psi_initial, dtype=complex).reshape(-1),
            dtype=complex,
        )
        return _ExactReferenceArtifacts(
            metadata=metadata,
            exact_evals=exact_evals_arr,
            exact_evecs=exact_evecs_arr,
            exact_coeffs0=exact_coeffs0_arr,
        )

    def _install_exact_reference_artifacts(
        self,
        artifacts: _ExactReferenceArtifacts,
        *,
        expected_metadata: _ExactReferenceMetadata,
    ) -> None:
        if artifacts.metadata != expected_metadata:
            raise ValueError("Exact reference cache metadata mismatch.")
        dim = int(self.psi_initial.size)
        self._exact_reference_metadata = expected_metadata
        if expected_metadata.kind == "driven_piecewise_constant_reference_from_replay_seed":
            if artifacts.reference_states is None:
                raise ValueError("Driven exact reference cache entry is missing states.")
            if len(artifacts.reference_states) != len(self.times):
                raise ValueError("Driven exact reference cache length mismatch.")
            for state in artifacts.reference_states:
                arr = np.asarray(state, dtype=complex).reshape(-1)
                if arr.shape != (dim,):
                    raise ValueError("Driven exact reference cache state shape mismatch.")
            self._reference_states = tuple(artifacts.reference_states)
            self._exact_evals = None
            self._exact_evecs = None
            self._exact_coeffs0 = None
            return

        if artifacts.exact_evals is None or artifacts.exact_evecs is None or artifacts.exact_coeffs0 is None:
            raise ValueError("Static exact reference cache entry is incomplete.")
        if np.asarray(artifacts.exact_evals).shape != (dim,):
            raise ValueError("Static exact reference eigenvalue shape mismatch.")
        if np.asarray(artifacts.exact_evecs).shape != (dim, dim):
            raise ValueError("Static exact reference eigenvector shape mismatch.")
        if np.asarray(artifacts.exact_coeffs0).shape != (dim,):
            raise ValueError("Static exact reference coefficient shape mismatch.")
        self._reference_states = None
        self._exact_evals = artifacts.exact_evals
        self._exact_evecs = artifacts.exact_evecs
        self._exact_coeffs0 = artifacts.exact_coeffs0

    def ensure_ready(self) -> None:
        if self._exact_reference_metadata is not None and (
            self._reference_states is not None or self._exact_evals is not None
        ):
            return
        metadata = self._build_exact_reference_metadata()
        self._exact_reference_metadata = metadata
        cache_key = self._build_exact_reference_cache_key(metadata)
        self._exact_reference_cache_key = str(cache_key)
        artifacts: _ExactReferenceArtifacts
        if self._exact_reference_cache is None:
            artifacts = self._build_exact_reference_artifacts(metadata)
        else:
            cached = self._exact_reference_cache.get(cache_key)
            if cached is None:
                artifacts = self._build_exact_reference_artifacts(metadata)
            else:
                if not isinstance(cached, _ExactReferenceArtifacts):
                    raise ValueError("Exact reference cache entry has invalid type.")
                artifacts = cached
        self._install_exact_reference_artifacts(artifacts, expected_metadata=metadata)
        if self._exact_reference_cache is not None and self._exact_reference_cache.get(cache_key) is None:
            self._exact_reference_cache[cache_key] = artifacts

    def state_at(self, time_value: float) -> np.ndarray:
        self.ensure_ready()
        if self._reference_states is not None:
            if len(self._reference_states) == 0:
                return np.asarray(self.psi_initial, dtype=complex).reshape(-1)
            idx = int(np.argmin(np.abs(self.times - float(time_value))))
            return np.asarray(self._reference_states[int(idx)], dtype=complex).reshape(-1)
        phases = np.exp(-1.0j * np.asarray(self._exact_evals, dtype=float) * float(time_value))
        return np.asarray(self._exact_evecs @ (phases * self._exact_coeffs0), dtype=complex)

    def reference_payload(self) -> dict[str, Any]:
        self.ensure_ready()
        metadata = self._exact_reference_metadata
        return {
            "reference_mode": "benchmark_exact",
            "reference_enabled": True,
            "kind": (None if metadata is None else str(metadata.kind)),
            "initial_state": "stage_result.psi_final",
            "times": [float(x) for x in self.times.tolist()],
            "drive_profile": (None if self.drive_profile is None else dict(self.drive_profile)),
            "reference_method": (None if metadata is None else metadata.reference_method),
            "reference_steps_multiplier": (
                None if metadata is None else int(metadata.reference_steps_multiplier)
            ),
            "projection_time_sampling": (
                None if metadata is None else str(metadata.projection_time_sampling)
            ),
            "geometry_sample_time_policy": (
                None if metadata is None else str(metadata.geometry_sample_time_policy)
            ),
        }


def _strict_controller_exact_audit_reason(controller: Any) -> str | None:
    if bool(getattr(controller, "strict_qpu_faithful", False)) or bool(
        getattr(controller, "strict_qpu_hh", False)
    ):
        return "strict QPU-faithful controllers must not use exact-audit decision helpers"
    if getattr(controller, "hmat", None) is None:
        return "exact-audit helpers require a dense Hamiltonian matrix"
    return None


def build_exact_audit_helper_for_controller(
    controller: Any,
    *,
    exact_reference_cache: MutableMapping[str, object] | None = None,
) -> RealtimeExactAuditHelper:
    strict_reason = _strict_controller_exact_audit_reason(controller)
    if strict_reason is not None:
        raise ValueError(
            f"{strict_reason}; attach exact references only as post-run diagnostics."
        )
    return RealtimeExactAuditHelper(
        h_poly=controller.h_poly,
        hmat=np.asarray(controller.hmat, dtype=complex),
        psi_initial=np.asarray(controller.psi_initial, dtype=complex),
        times=np.asarray(controller.times, dtype=float),
        drive_config=getattr(controller, "_drive_config", None),
        drive_profile=getattr(controller, "_drive_profile", None),
        drive_coeff_provider_exyz=getattr(controller, "_drive_coeff_provider_exyz", None),
        drive_model=getattr(controller, "_drive_model", None),
        exact_reference_cache=exact_reference_cache,
    )


def exact_v1_pre_action_snapshot(
    controller: Any,
    exact_helper: RealtimeExactAuditHelper,
    *,
    checkpoint_index: int,
) -> dict[str, Any]:
    if str(controller.cfg.mode) != "exact_v1":
        raise ValueError("exact_v1_pre_action_snapshot requires cfg.mode='exact_v1'")
    idx = int(checkpoint_index)
    if idx < 0 or idx >= int(len(controller.times)):
        raise ValueError(f"checkpoint_index {idx} out of range for {len(controller.times)} checkpoints.")
    time_value = float(controller.times[idx])
    time_stop = None if int(idx) + 1 >= int(len(controller.times)) else float(controller.times[int(idx) + 1])
    physical_time = float(controller._projection_sample_time(float(time_value), time_stop))
    step_hamiltonian = controller._step_hamiltonian_artifacts(float(physical_time))
    psi_current = np.asarray(
        controller.current_executor.prepare_state(controller.current_theta, controller.replay_context.psi_ref),
        dtype=complex,
    ).reshape(-1)
    psi_exact = np.asarray(exact_helper.state_at(float(time_value)), dtype=complex).reshape(-1)
    current_obs = _benchmark_snapshot_payload(controller._observable_snapshot(psi_current))
    exact_obs = _benchmark_snapshot_payload(controller._observable_snapshot(psi_exact))
    current_sites, current_label, current_components = _matching_vector_or_none(current_obs)
    exact_sites, exact_label, exact_components = _matching_vector_or_none(exact_obs)
    site_error = _vector_error_max_or_none(current_sites, exact_sites)
    primary_density_current = _finite_or_none(
        controller._primary_density_value_from_snapshot(current_obs)
    )
    primary_density_exact = _finite_or_none(
        controller._primary_density_value_from_snapshot(exact_obs)
    )
    primary_density_error = _abs_error_or_none(
        primary_density_current,
        primary_density_exact,
    )
    energy_current = float(
        np.real(np.vdot(psi_current, np.asarray(step_hamiltonian.hmat, dtype=complex) @ psi_current))
    )
    energy_exact = float(
        np.real(np.vdot(psi_exact, np.asarray(step_hamiltonian.hmat, dtype=complex) @ psi_exact))
    )
    return {
        "checkpoint_index": int(idx),
        "time": float(time_value),
        "time_stop": (None if time_stop is None else float(time_stop)),
        "physical_time": float(physical_time),
        "h_poly_step": step_hamiltonian.h_poly,
        "hmat_step": np.asarray(step_hamiltonian.hmat, dtype=complex),
        "drive_term_count": int(step_hamiltonian.drive_term_count),
        "terms": list(controller.current_terms),
        "layout": controller.current_layout,
        "executor": controller.current_executor,
        "theta_runtime": np.asarray(controller.current_theta, dtype=float).reshape(-1).copy(),
        "psi_ref": np.asarray(controller.replay_context.psi_ref, dtype=complex).reshape(-1).copy(),
        "psi_current": np.asarray(psi_current, dtype=complex).reshape(-1),
        "psi_exact": np.asarray(psi_exact, dtype=complex).reshape(-1).copy(),
        "observable_family": str(
            current_obs.get(
                "observable_family",
                observable_family_key(getattr(controller, "resolved_problem", None)),
            )
        ),
        "current_observables": dict(current_obs),
        "exact_observables": dict(exact_obs),
        "site_occupations_label": (
            str(current_label)
            if current_label is not None
            else (None if exact_label is None else str(exact_label))
        ),
        "site_occupations_component_labels": (
            [str(x) for x in current_components]
            if current_components is not None
            else (
                None
                if exact_components is None
                else [str(x) for x in exact_components]
            )
        ),
        "primary_density_mode": str(controller._exact_forecast_primary_density_target_mode()),
        "fidelity_exact": float(abs(np.vdot(psi_exact, psi_current)) ** 2),
        "energy_current": float(energy_current),
        "energy_exact": float(energy_exact),
        "abs_energy_total_error": float(abs(energy_current - energy_exact)),
        "site_occupations_abs_error_max": site_error,
        "abs_primary_density_error": primary_density_error,
        "scaffold_labels": controller._current_scaffold_labels(),
        "logical_block_count": int(controller.current_layout.logical_parameter_count),
        "runtime_parameter_count": int(controller.current_layout.runtime_parameter_count),
        "num_sites": int(controller._num_sites),
        "ordering": str(controller._ordering),
    }


def exact_step_forecast(
    controller: Any,
    exact_helper: RealtimeExactAuditHelper,
    *,
    time_stop: float,
    executor: Any,
    theta_runtime: np.ndarray | Sequence[float],
) -> dict[str, Any]:
    psi_pred = np.asarray(
        executor.prepare_state(
            np.asarray(theta_runtime, dtype=float).reshape(-1),
            controller.replay_context.psi_ref,
        ),
        dtype=complex,
    ).reshape(-1)
    psi_exact = np.asarray(exact_helper.state_at(float(time_stop)), dtype=complex).reshape(-1)
    step_hamiltonian = controller._step_hamiltonian_artifacts(float(time_stop))
    energy_total_controller_next = float(
        np.real(np.vdot(psi_pred, step_hamiltonian.hmat @ psi_pred))
    )
    energy_total_exact_next = float(
        np.real(np.vdot(psi_exact, step_hamiltonian.hmat @ psi_exact))
    )
    pred_obs = _benchmark_snapshot_payload(controller._observable_snapshot(psi_pred))
    exact_obs = _benchmark_snapshot_payload(controller._observable_snapshot(psi_exact))
    pred_site, pred_label, pred_components = _matching_vector_or_none(pred_obs)
    exact_site, exact_label, exact_components = _matching_vector_or_none(exact_obs)
    site_occ_abs_error_max = _vector_error_max_or_none(pred_site, exact_site)
    primary_density_controller_next = _finite_or_none(
        controller._primary_density_value_from_snapshot(pred_obs)
    )
    primary_density_exact_next = _finite_or_none(
        controller._primary_density_value_from_snapshot(exact_obs)
    )
    return {
        "time_stop_next": float(time_stop),
        "fidelity_exact_next": float(abs(np.vdot(psi_exact, psi_pred)) ** 2),
        "observable_family": str(
            pred_obs.get(
                "observable_family",
                observable_family_key(getattr(controller, "resolved_problem", None)),
            )
        ),
        "primary_density_mode": str(controller._exact_forecast_primary_density_target_mode()),
        "primary_density_controller_next": primary_density_controller_next,
        "primary_density_exact_next": primary_density_exact_next,
        "site_occupations_controller_next": [float(x) for x in pred_site],
        "site_occupations_exact_next": [float(x) for x in exact_site],
        "site_occupations_label": (
            str(pred_label)
            if pred_label is not None
            else (None if exact_label is None else str(exact_label))
        ),
        "site_occupations_component_labels": (
            [str(x) for x in pred_components]
            if pred_components is not None
            else (
                None
                if exact_components is None
                else [str(x) for x in exact_components]
            )
        ),
        "doublon_controller_next": pred_obs.get("doublon"),
        "doublon_exact_next": exact_obs.get("doublon"),
        "staggered_controller_next": pred_obs.get("staggered"),
        "staggered_exact_next": exact_obs.get("staggered"),
        "energy_total_controller_next": float(energy_total_controller_next),
        "energy_total_exact_next": float(energy_total_exact_next),
        "abs_energy_total_error_next": float(
            abs(float(energy_total_controller_next) - float(energy_total_exact_next))
        ),
        "abs_primary_density_error_next": _abs_error_or_none(
            primary_density_controller_next,
            primary_density_exact_next,
        ),
        "abs_staggered_error_next": _abs_error_or_none(
            pred_obs.get("staggered"),
            exact_obs.get("staggered"),
        ),
        "abs_doublon_error_next": _abs_error_or_none(
            pred_obs.get("doublon"),
            exact_obs.get("doublon"),
        ),
        "site_occupations_abs_error_max_next": site_occ_abs_error_max,
    }


class RealtimeExactAuditObserver:
    _ED_GROUND_ENERGY_MATCH_ATOL = 1.0e-8

    def __init__(
        self,
        controller: Any,
        exact_helper: RealtimeExactAuditHelper,
        *,
        ed_ground_exact_energy: float | None = None,
        ed_ground_exact_energy_source: str | None = None,
    ) -> None:
        self.controller = controller
        self.exact_helper = exact_helper
        self._ed_ground_energy: float | None = _finite_or_none(ed_ground_exact_energy)
        self._ed_ground_energy_source: str = (
            str(ed_ground_exact_energy_source)
            if ed_ground_exact_energy_source not in {None, ""}
            else (
                "artifact_static_ed_ground_energy"
                if self._ed_ground_energy is not None
                else "controller_hmat_lowest_eigenpair"
            )
        )
        self._ed_ground_full_hilbert_energy: float | None = None
        self._ed_ground_state: np.ndarray | None = None
        self._ed_ground_helper: RealtimeExactAuditHelper | None = None
        self._ed_ground_init_error: str | None = None
        self._ed_ground_state_disabled_reason: str | None = None
        try:
            evals, evecs = np.linalg.eigh(np.asarray(controller.hmat, dtype=complex))
            idx = int(np.argmin(np.asarray(evals, dtype=float)))
            full_hilbert_ground_energy = float(np.real(evals[idx]))
            self._ed_ground_full_hilbert_energy = float(full_hilbert_ground_energy)
            if self._ed_ground_energy is None:
                self._ed_ground_energy = float(full_hilbert_ground_energy)
                self._ed_ground_energy_source = "controller_hmat_lowest_eigenpair"
            energy_mismatch = abs(
                float(full_hilbert_ground_energy) - float(self._ed_ground_energy)
            )
            if energy_mismatch > self._ED_GROUND_ENERGY_MATCH_ATOL:
                self._ed_ground_state_disabled_reason = (
                    "full_hilbert_ground_energy_mismatch_with_artifact_target"
                )
                return
            self._ed_ground_state = np.asarray(evecs[:, idx], dtype=complex).reshape(-1)
            self._ed_ground_helper = RealtimeExactAuditHelper(
                h_poly=controller.h_poly,
                hmat=np.asarray(controller.hmat, dtype=complex),
                psi_initial=np.asarray(self._ed_ground_state, dtype=complex),
                times=np.asarray(controller.times, dtype=float),
                drive_config=getattr(controller, "_drive_config", None),
                drive_profile=getattr(controller, "_drive_profile", None),
                drive_coeff_provider_exyz=getattr(controller, "_drive_coeff_provider_exyz", None),
                drive_model=getattr(controller, "_drive_model", None),
                exact_reference_cache=getattr(exact_helper, "_exact_reference_cache", None),
            )
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            self._ed_ground_init_error = f"{type(exc).__name__}: {exc}"

    def on_checkpoint(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        time_value = float(payload["time"])
        step_hamiltonian = payload["step_hamiltonian"]
        psi_current = np.asarray(payload["psi_current"], dtype=complex).reshape(-1)
        controller_obs = _benchmark_snapshot_payload(
            payload.get("controller_obs") or self.controller._observable_snapshot(psi_current)
        )
        site_occ_controller, site_label, site_components = _matching_vector_or_none(
            controller_obs
        )
        primary_density_controller = _finite_or_none(
            self.controller._primary_density_value_from_snapshot(controller_obs)
        )
        energy_controller = float(payload["energy_total_controller"])

        psi_exact = np.asarray(self.exact_helper.state_at(float(time_value)), dtype=complex).reshape(-1)
        energy_exact = float(
            np.real(
                np.vdot(
                    np.asarray(psi_exact, dtype=complex).reshape(-1),
                    np.asarray(step_hamiltonian.hmat, dtype=complex)
                    @ np.asarray(psi_exact, dtype=complex).reshape(-1),
                )
            )
        )
        fidelity_exact = float(abs(np.vdot(psi_exact, psi_current)) ** 2)
        exact_obs = _benchmark_snapshot_payload(
            self.controller._observable_snapshot(np.asarray(psi_exact, dtype=complex).reshape(-1))
        )
        site_occ_exact, exact_site_label, exact_site_components = _matching_vector_or_none(
            exact_obs
        )
        site_occ_abs_error = _vector_abs_error_or_none(site_occ_controller, site_occ_exact)
        primary_density_exact = _finite_or_none(
            self.controller._primary_density_value_from_snapshot(exact_obs)
        )

        trajectory_update: dict[str, Any] = {
            "observable_family": str(
                controller_obs.get(
                    "observable_family",
                    observable_family_key(getattr(self.controller, "resolved_problem", None)),
                )
            ),
            "primary_density_mode": str(
                self.controller._exact_forecast_primary_density_target_mode()
            ),
            "site_occupations": [float(x) for x in site_occ_controller],
            "site_occupations_label": (
                str(site_label)
                if site_label is not None
                else (None if exact_site_label is None else str(exact_site_label))
            ),
            "site_occupations_component_labels": (
                [str(x) for x in site_components]
                if site_components is not None
                else (
                    None
                    if exact_site_components is None
                    else [str(x) for x in exact_site_components]
                )
            ),
            "doublon": controller_obs.get("doublon"),
            "staggered": controller_obs.get("staggered"),
            "energy_total_exact": float(energy_exact),
            "abs_energy_total_error": float(abs(float(energy_controller) - float(energy_exact))),
            "fidelity_exact": float(fidelity_exact),
            "fidelity_initial_exact": float(
                abs(
                    np.vdot(
                        np.asarray(self.controller.psi_initial, dtype=complex).reshape(-1),
                        np.asarray(psi_exact, dtype=complex).reshape(-1),
                    )
                )
                ** 2
            ),
            "primary_density_exact": primary_density_exact,
            "abs_primary_density_error": _abs_error_or_none(
                primary_density_controller,
                primary_density_exact,
            ),
            "staggered_exact": exact_obs.get("staggered"),
            "abs_staggered_error": _abs_error_or_none(
                controller_obs.get("staggered"),
                exact_obs.get("staggered"),
            ),
            "doublon_exact": exact_obs.get("doublon"),
            "abs_doublon_error": _abs_error_or_none(
                controller_obs.get("doublon"),
                exact_obs.get("doublon"),
            ),
            "site_occupations_exact": [float(x) for x in site_occ_exact],
            "site_occupations_up_exact": [
                float(x) for x in np.asarray(exact_obs["n_up_site"], dtype=float).tolist()
            ],
            "site_occupations_dn_exact": [
                float(x) for x in np.asarray(exact_obs["n_dn_site"], dtype=float).tolist()
            ],
            "site_occupations_abs_error": (
                None if site_occ_abs_error is None else [float(x) for x in site_occ_abs_error]
            ),
            "site_occupations_abs_error_max": _vector_error_max_or_none(
                site_occ_controller,
                site_occ_exact,
            ),
        }
        if self._ed_ground_helper is not None:
            psi_ed_ground = np.asarray(
                self._ed_ground_helper.state_at(float(time_value)), dtype=complex
            ).reshape(-1)
            energy_ed_ground = float(
                np.real(
                    np.vdot(
                        psi_ed_ground,
                        np.asarray(step_hamiltonian.hmat, dtype=complex) @ psi_ed_ground,
                    )
                )
            )
            ed_obs = _benchmark_snapshot_payload(
                self.controller._observable_snapshot(psi_ed_ground)
            )
            site_occ_ed, _ed_site_label, _ed_site_components = _matching_vector_or_none(
                ed_obs
            )
            site_occ_abs_error_ed = _vector_abs_error_or_none(
                site_occ_controller,
                site_occ_ed,
            )
            primary_density_ed = _finite_or_none(
                self.controller._primary_density_value_from_snapshot(ed_obs)
            )
            trajectory_update.update(
                {
                    "energy_total_ed_ground": float(energy_ed_ground),
                    "abs_energy_total_error_to_ed_ground": float(
                        abs(float(energy_controller) - float(energy_ed_ground))
                    ),
                    "fidelity_ed_ground": float(abs(np.vdot(psi_ed_ground, psi_current)) ** 2),
                    "primary_density_ed_ground": primary_density_ed,
                    "abs_primary_density_error_to_ed_ground": _abs_error_or_none(
                        primary_density_controller,
                        primary_density_ed,
                    ),
                    "staggered_ed_ground": ed_obs.get("staggered"),
                    "abs_staggered_error_to_ed_ground": _abs_error_or_none(
                        controller_obs.get("staggered"),
                        ed_obs.get("staggered"),
                    ),
                    "doublon_ed_ground": ed_obs.get("doublon"),
                    "abs_doublon_error_to_ed_ground": _abs_error_or_none(
                        controller_obs.get("doublon"),
                        ed_obs.get("doublon"),
                    ),
                    "site_occupations_ed_ground": [float(x) for x in site_occ_ed],
                    "site_occupations_abs_error_to_ed_ground": (
                        None
                        if site_occ_abs_error_ed is None
                        else [float(x) for x in site_occ_abs_error_ed]
                    ),
                    "site_occupations_abs_error_to_ed_ground_max": _vector_error_max_or_none(
                        site_occ_controller,
                        site_occ_ed,
                    ),
                }
            )
        elif self._ed_ground_energy is not None:
            # Sector-constrained artifacts (for example Hubbard) may carry a
            # filtered exact target energy whose eigenvector is not the lowest
            # vector of the full matrix.  Keep the energy target, but do not
            # fabricate fidelity/observable targets from the wrong state.
            energy_ed_ground = float(self._ed_ground_energy)
            trajectory_update.update(
                {
                    "energy_total_ed_ground": float(energy_ed_ground),
                    "abs_energy_total_error_to_ed_ground": float(
                        abs(float(energy_controller) - float(energy_ed_ground))
                    ),
                }
            )
        ledger_update = {
            "energy_total_exact": float(energy_exact),
            "abs_energy_total_error": float(abs(float(energy_controller) - float(energy_exact))),
            "fidelity_exact": float(fidelity_exact),
        }
        if "energy_total_ed_ground" in trajectory_update:
            ledger_update.update(
                {
                    "energy_total_ed_ground": trajectory_update.get("energy_total_ed_ground"),
                    "abs_energy_total_error_to_ed_ground": trajectory_update.get(
                        "abs_energy_total_error_to_ed_ground"
                    ),
                }
            )
            if "fidelity_ed_ground" in trajectory_update:
                ledger_update["fidelity_ed_ground"] = trajectory_update.get("fidelity_ed_ground")
        post_prune_psi = payload.get("post_prune_psi")
        if post_prune_psi is not None:
            reduced_psi = np.asarray(post_prune_psi, dtype=complex).reshape(-1)
            reduced_obs = _benchmark_snapshot_payload(self.controller._observable_snapshot(reduced_psi))
            reduced_site_occ, _reduced_label, _reduced_components = _matching_vector_or_none(
                reduced_obs
            )
            reduced_site_occ_abs_error = _vector_abs_error_or_none(
                reduced_site_occ,
                site_occ_exact,
            )
            trajectory_update.update(
                {
                    "post_prune_fidelity_exact": float(abs(np.vdot(psi_exact, reduced_psi)) ** 2),
                    "post_prune_abs_energy_total_error": float(
                        abs(
                            float(
                                np.real(
                                    np.vdot(
                                        reduced_psi,
                                        np.asarray(step_hamiltonian.hmat, dtype=complex) @ reduced_psi,
                                    )
                                )
                            )
                            - float(energy_exact)
                        )
                    ),
                    "post_prune_abs_staggered_error": _abs_error_or_none(
                        reduced_obs.get("staggered"),
                        exact_obs.get("staggered"),
                    ),
                    "post_prune_abs_doublon_error": _abs_error_or_none(
                        reduced_obs.get("doublon"),
                        exact_obs.get("doublon"),
                    ),
                    "post_prune_site_occupations_abs_error": (
                        None
                        if reduced_site_occ_abs_error is None
                        else [float(x) for x in reduced_site_occ_abs_error]
                    ),
                    "post_prune_site_occupations_abs_error_max": _vector_error_max_or_none(
                        reduced_site_occ,
                        site_occ_exact,
                    ),
                }
            )
        return {
            "trajectory_update": trajectory_update,
            "ledger_update": ledger_update,
        }

    def finalize(
        self,
        *,
        summary: Mapping[str, Any],
        reference: Mapping[str, Any],
        trajectory: Sequence[Mapping[str, Any]],
        ledger: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        del reference, ledger
        reference_payload = dict(self.exact_helper.reference_payload())
        trajectory_first = trajectory[0] if len(trajectory) > 0 else None
        reference_payload["observable_family"] = str(
            (
                trajectory_first.get("observable_family")
                if isinstance(trajectory_first, Mapping)
                and trajectory_first.get("observable_family") not in {None, ""}
                else summary.get(
                    "final_observable_family",
                    observable_family_key(getattr(self.controller, "resolved_problem", None)),
                )
            )
        )
        reference_payload["primary_density_mode"] = str(
            self.controller._exact_forecast_primary_density_target_mode()
        )
        energy_target_enabled = self._ed_ground_energy is not None
        state_target_enabled = self._ed_ground_helper is not None
        drive_enabled = getattr(self.controller, "_drive_config", None) is not None
        if state_target_enabled:
            ed_ground_kind = "exact_trajectory_from_static_hamiltonian_ed_ground_state"
            drive_scope = "same_time_dependent_hamiltonian" if drive_enabled else "static_hamiltonian"
            initial_state = "static_hamiltonian_ed_ground_state"
        elif energy_target_enabled:
            ed_ground_kind = "artifact_static_ed_ground_energy"
            drive_scope = (
                "static_ground_energy_only; state trajectory target unavailable"
                if drive_enabled
                else "static_hamiltonian_energy_only"
            )
            initial_state = None
        else:
            ed_ground_kind = "unavailable_static_ed_ground_energy"
            drive_scope = "unavailable"
            initial_state = None
        ed_ground_payload: dict[str, Any] = {
            "enabled": bool(energy_target_enabled),
            "state_target_enabled": bool(state_target_enabled),
            "kind": str(ed_ground_kind),
            "static_ground_energy": self._ed_ground_energy,
            "exact_energy": self._ed_ground_energy,
            "source": str(self._ed_ground_energy_source),
            "full_hilbert_ground_energy": self._ed_ground_full_hilbert_energy,
            "energy_match_tolerance": float(self._ED_GROUND_ENERGY_MATCH_ATOL),
            "reference_method": None,
            "reference_steps_multiplier": None,
            "initial_state": initial_state,
            "drive_scope": drive_scope,
        }
        if self._ed_ground_state_disabled_reason is not None:
            ed_ground_payload["state_target_disabled_reason"] = str(
                self._ed_ground_state_disabled_reason
            )
        if self._ed_ground_helper is not None:
            target_reference_payload = dict(self._ed_ground_helper.reference_payload())
            ed_ground_payload.update(
                {
                    "reference_method": target_reference_payload.get("reference_method"),
                    "reference_steps_multiplier": target_reference_payload.get(
                        "reference_steps_multiplier"
                    ),
                    "projection_time_sampling": target_reference_payload.get(
                        "projection_time_sampling"
                    ),
                    "geometry_sample_time_policy": target_reference_payload.get(
                        "geometry_sample_time_policy"
                    ),
                    "drive_profile": target_reference_payload.get("drive_profile"),
                }
            )
        elif self._ed_ground_init_error is not None:
            ed_ground_payload["error"] = str(self._ed_ground_init_error)

        summary_update: dict[str, Any] = {
            "reference_mode": "benchmark_exact",
            "reference_enabled": True,
            "ed_ground_energy_target_enabled": bool(energy_target_enabled),
            "ed_ground_energy_target_state_enabled": bool(state_target_enabled),
            "ed_ground_energy_target_kind": str(ed_ground_payload["kind"]),
            "ed_ground_energy_target_exact_energy": self._ed_ground_energy,
            "ed_ground_energy_target_source": str(ed_ground_payload["source"]),
        }
        if self._ed_ground_state_disabled_reason is not None:
            summary_update["ed_ground_energy_target_state_disabled_reason"] = str(
                self._ed_ground_state_disabled_reason
            )
        summary_update.update(
            _summary_stats_for_values(
                _finite_values(trajectory, "abs_energy_total_error_to_ed_ground"),
                prefix="abs_energy_total_error_to_ed_ground",
            )
        )
        summary_update.update(
            _summary_stats_for_values(
                _finite_values(trajectory, "abs_primary_density_error_to_ed_ground"),
                prefix="abs_primary_density_error_to_ed_ground",
            )
        )
        summary_update.update(
            _summary_stats_for_values(
                _finite_values(trajectory, "site_occupations_abs_error_to_ed_ground_max"),
                prefix="site_occupations_abs_error_to_ed_ground_max",
            )
        )
        summary_update.update(
            _summary_stats_for_values(
                _finite_values(trajectory, "fidelity_ed_ground"),
                prefix="fidelity_ed_ground",
            )
        )
        if self._ed_ground_init_error is not None:
            summary_update["ed_ground_energy_target_error"] = str(self._ed_ground_init_error)

        reference_payload["ed_ground_energy_target"] = ed_ground_payload
        return {
            "summary_update": summary_update,
            "reference": reference_payload,
        }


def run_controller_with_exact_audit(
    controller: Any,
    exact_helper: RealtimeExactAuditHelper | None = None,
    *,
    ed_ground_exact_energy: float | None = None,
    ed_ground_exact_energy_source: str | None = None,
):
    strict_reason = _strict_controller_exact_audit_reason(controller)
    if strict_reason is not None:
        raise ValueError(
            f"{strict_reason}; attach exact references only as post-run diagnostics."
        )
    helper = exact_helper or build_exact_audit_helper_for_controller(controller)
    observer = RealtimeExactAuditObserver(
        controller,
        helper,
        ed_ground_exact_energy=ed_ground_exact_energy,
        ed_ground_exact_energy_source=ed_ground_exact_energy_source,
    )
    return controller.run(checkpoint_observer=observer)


__all__ = [
    "RealtimeExactAuditHelper",
    "RealtimeExactAuditObserver",
    "build_exact_audit_helper_for_controller",
    "exact_step_forecast",
    "exact_v1_pre_action_snapshot",
    "run_controller_with_exact_audit",
]
