from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from pipelines.scaffold.hh_vqe_from_adapt_family import ReplayScaffoldContext
from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input
from pipelines.static_adapt.builders.primitive_pools import build_runtime_pool_terms
from pipelines.time_dynamics.adapters.hh import HH_REALTIME_ADAPTER
from pipelines.time_dynamics.legacy.checkpoint_route_policy import (
    strict_qpu_faithful_requested,
    validate_realtime_route_request,
)
from src.quantum.ansatz_parameterization import serialize_layout
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


def RealtimeCheckpointController(**kwargs: Any) -> Any:
    """Compatibility factory for old generic-runner monkeypatch surfaces."""

    return HH_REALTIME_ADAPTER.create_controller(**kwargs)


def build_exact_audit_helper_for_controller(
    controller: Any,
    *,
    exact_reference_cache: dict[str, object] | None = None,
) -> Any:
    return HH_REALTIME_ADAPTER.build_exact_audit_helper_for_controller(
        controller,
        exact_reference_cache=exact_reference_cache,
    )


def _attach_diagnostic_exact_reference(*, args: Any, controller: Any, result: Any) -> Any:
    return HH_REALTIME_ADAPTER.attach_diagnostic_exact_reference(
        args=args,
        controller=controller,
        result=result,
    )


@dataclass(frozen=True)
class RealtimeControllerSeed:
    loaded: Any
    runtime_input: Any
    cfg: Any
    oracle_config: Any
    drive_config: Any
    replay_context: ReplayScaffoldContext
    h_poly: Any
    hmat: np.ndarray | None


def _resolved_family(runtime_input: Any) -> str:
    family = runtime_input.provenance.get("resolved_family", None)
    if family not in {None, ""}:
        return str(family)
    pool_key = getattr(runtime_input.candidate_pool_source, "pool_key", None)
    if pool_key not in {None, ""}:
        return str(pool_key)
    return str(runtime_input.resolved_problem.family_key)


def _statevector_sha256(psi: Any) -> str:
    arr = np.asarray(psi, dtype=np.complex128).reshape(-1)
    payload = np.ascontiguousarray(
        np.stack([arr.real, arr.imag], axis=1).astype("<f8", copy=False)
    )
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _complex_vector_payload(vec: Any) -> list[dict[str, float | None]]:
    arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
    return [
        {"re": _finite_float_or_none(z.real), "im": _finite_float_or_none(z.imag)}
        for z in arr
    ]


def _payload_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _payload_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _payload_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_payload_safe(item) for item in value]
    if isinstance(value, complex):
        return {"re": _finite_float_or_none(value.real), "im": _finite_float_or_none(value.imag)}
    if isinstance(value, (np.floating, float)):
        return _finite_float_or_none(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _serialized_pauli_terms(poly: Any) -> list[dict[str, Any]]:
    if poly is None or not hasattr(poly, "return_polynomial"):
        return []
    out: list[dict[str, Any]] = []
    for term in poly.return_polynomial():
        coeff = complex(getattr(term, "p_coeff"))
        out.append(
            {
                "pauli_exyz": str(term.pw2strng()).lower(),
                "coeff_re": _finite_float_or_none(coeff.real),
                "coeff_im": _finite_float_or_none(coeff.imag),
            }
        )
    return out


class FixedScaffoldQiskitParityObserver:
    """Post-run fixed-scaffold parity capture for the generic benchmark route.

    The observer serializes the prepared scaffold state, fixed layout, theta, and
    sampled Hamiltonian terms after checkpoint telemetry is produced.  It is a
    diagnostic/export hook only and has no path back into controller decisions.
    """

    schema = "fixed_scaffold_qiskit_post_run_payload_v1"

    def __init__(self, *, runtime_input: Any) -> None:
        self._psi_ref = np.asarray(runtime_input.psi_ref, dtype=np.complex128).reshape(-1)
        self._fixed_layout_payload: dict[str, Any] | None = None
        self._layout_stable = True
        self._checkpoints: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []

    def on_checkpoint(self, payload: Mapping[str, Any]) -> None:
        checkpoint_index = int(payload.get("checkpoint_index", len(self._checkpoints)))
        layout = payload.get("layout_at_checkpoint")
        theta = payload.get("theta_runtime_at_checkpoint")
        native_state = payload.get("psi_current")
        if layout is None or theta is None or native_state is None:
            self._errors.append(
                {
                    "checkpoint_index": checkpoint_index,
                    "reason": "missing_layout_theta_or_native_state",
                    "has_layout": layout is not None,
                    "has_theta": theta is not None,
                    "has_native_state": native_state is not None,
                }
            )
            return None

        layout_payload = serialize_layout(layout)
        checkpoint_payload: dict[str, Any] = {
            "checkpoint_index": checkpoint_index,
            "time": _finite_float_or_none(payload.get("time")),
            "time_stop": _finite_float_or_none(payload.get("time_stop")),
            "physical_time": _finite_float_or_none(payload.get("physical_time")),
            "theta_runtime": [
                _finite_float_or_none(x)
                for x in np.asarray(theta, dtype=float).reshape(-1).tolist()
            ],
            "native_state": _complex_vector_payload(native_state),
            "native_state_source": "controller_prepared_scaffold_state_for_current_theta",
            "energy_total_controller": _finite_float_or_none(
                payload.get("energy_total_controller")
            ),
            "controller_observables": _payload_safe(payload.get("controller_obs", {})),
            "trajectory_energy_total": _finite_float_or_none(
                (payload.get("trajectory_row", {}) or {}).get("energy_total")
                if isinstance(payload.get("trajectory_row", {}), Mapping)
                else None
            ),
        }
        if self._fixed_layout_payload is None:
            self._fixed_layout_payload = layout_payload
        elif layout_payload != self._fixed_layout_payload:
            self._layout_stable = False
            checkpoint_payload["layout"] = layout_payload

        step_hamiltonian = payload.get("step_hamiltonian")
        terms = _serialized_pauli_terms(getattr(step_hamiltonian, "h_poly", None))
        if terms:
            checkpoint_payload["hamiltonian_terms_exyz"] = terms
            checkpoint_payload["hamiltonian_terms_source"] = "step_hamiltonian_artifacts.h_poly"
        self._checkpoints.append(checkpoint_payload)
        return None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "role": "post_run_benchmark_local_qiskit_parity_input",
            "data_flow": "serialized_after_checkpoint_telemetry_not_controller_input",
            "controller_decisions_modified": False,
            "exact_reference_controller_inputs": False,
            "qiskit_used_in_online_controller": False,
            "psi_ref": _complex_vector_payload(self._psi_ref),
            "psi_ref_sha256": _statevector_sha256(self._psi_ref),
            "fixed_layout": self._fixed_layout_payload,
            "layout_stable": bool(self._layout_stable),
            "checkpoint_count": int(len(self._checkpoints)),
            "checkpoints": list(self._checkpoints),
            "errors": list(self._errors),
        }


def _strict_state_prep_contract_from_runtime_input(runtime_input: Any) -> dict[str, Any]:
    """Describe the seed state-prep boundary used by strict realtime routes.

    This is provenance only: it identifies the ansatz input state and prepared
    ADAPT seed state that the controller may prepare/measure.  It is not an
    ED target/reference trajectory and must not be used as exact decision data.
    """

    extensions = getattr(runtime_input, "extensions", {})
    summary = (
        extensions.get("generic_loader_summary", {})
        if isinstance(extensions, dict)
        else {}
    )
    if not isinstance(summary, dict):
        summary = {}
    ref_spec = getattr(getattr(runtime_input, "resolved_problem", None), "reference_state", None)
    ref_allowlist = sorted(
        {
            str(item)
            for item in (
                getattr(ref_spec, "kind", None),
                getattr(ref_spec, "source_label", None),
            )
            if item not in {None, ""}
        }
    )
    ansatz_source = summary.get(
        "ansatz_input_state_source",
        getattr(ref_spec, "source_label", "unknown"),
    )
    ansatz_kind = summary.get(
        "ansatz_input_state_handoff_state_kind",
        getattr(ref_spec, "state_kind", "reference_state"),
    )
    ansatz_location = summary.get(
        "ansatz_input_state_source_location",
        summary.get("reference_state_source", "runtime_input.psi_ref"),
    )
    initial_source = summary.get(
        "initial_state_payload_source",
        summary.get("initial_state_source", "unknown"),
    )
    initial_kind = summary.get("initial_state_handoff_state_kind", "prepared_state")
    initial_location = summary.get(
        "initial_state_source_location",
        summary.get("initial_state_source", "runtime_input.psi_initial"),
    )
    reconstruction_error = summary.get("prepared_state_reconstruction_error", None)
    return {
        "version": "strict_state_prep_v1",
        "role": "prepared_seed_state_only",
        "feeds_controller_decisions": "prepared_ansatz_observables_only",
        "exact_target_or_reference_trajectory": False,
        "ansatz_input_state": {
            "role": "ansatz_input_state",
            "source_location": str(ansatz_location),
            "source": str(ansatz_source),
            "source_allowlist": list(ref_allowlist),
            "handoff_state_kind": str(ansatz_kind),
            "state_sha256": _statevector_sha256(runtime_input.psi_ref),
        },
        "initial_state": {
            "role": "prepared_ansatz_state",
            "source_location": str(initial_location),
            "source": str(initial_source),
            "source_allowlist": ["adapt_vqe", "reconstructed_from_scaffold"],
            "handoff_state_kind": str(initial_kind),
            "state_sha256": _statevector_sha256(runtime_input.psi_initial),
        },
        "prepared_state_reconstruction_error": (
            None if reconstruction_error is None else float(reconstruction_error)
        ),
    }


"Built Math: replay_context := {psi_ref, selected_terms, candidate_pool}; controller(seed) := McLachlan(H, theta_0, psi_0)."
def _replay_context_from_runtime_input(
    runtime_input: Any,
    *,
    append_pool_family: str,
) -> ReplayScaffoldContext:
    request = runtime_input.resolved_problem.request
    resolved_family = _resolved_family(runtime_input)
    family_pool = (
        tuple(runtime_input.candidate_pool_terms)
        if len(tuple(runtime_input.candidate_pool_terms)) > 0
        else tuple(runtime_input.selected_terms)
    )
    family_info = {
        "requested": runtime_input.provenance.get("requested_family", resolved_family),
        "resolved": resolved_family,
        "resolution_source": runtime_input.provenance.get("resolution_source", "runtime_loader"),
        "fallback_used": bool(runtime_input.extensions.get("generic_family_info", {}).get("fallback_used", False)),
        "warning": runtime_input.extensions.get("generic_family_info", {}).get("warning", None),
    }
    pool_meta = dict(runtime_input.extensions.get("generic_pool_meta", {}))
    state_prep_contract = _strict_state_prep_contract_from_runtime_input(runtime_input)
    pool_meta.update(
        {
            "candidate_pool_complete": bool(runtime_input.candidate_pool_source.candidate_pool_complete),
            "structure_locked": bool(runtime_input.structure_locked),
            "family_pool_origin": "runtime_input.candidate_pool_terms",
            "strict_state_prep_contract": dict(state_prep_contract),
        }
    )
    append_family_info = None
    append_family_pool = None
    append_pool_meta = None
    append_family_terms_count = None
    append_request = str(append_pool_family).strip().lower()
    pool_complete = bool(runtime_input.candidate_pool_source.candidate_pool_complete)
    if append_request in {"", "match_replay"}:
        append_family_info = {
            "requested": "match_replay" if append_request == "" else append_request,
            "resolved": str(resolved_family),
            "resolution_source": "replay_family",
            "fallback_used": False,
            "uses_replay_pool": True,
        }
        append_family_pool = tuple(family_pool)
        append_pool_meta = dict(pool_meta)
        append_pool_meta.update(
            {
                "candidate_pool_complete": bool(pool_complete),
                "append_pool_source": "replay_family_pool",
                "replay_family": str(resolved_family),
                "append_family": str(resolved_family),
            }
        )
        append_family_terms_count = int(len(family_pool))
    elif append_request == str(resolved_family).strip().lower() and bool(pool_complete):
        append_family_info = {
            "requested": append_request,
            "resolved": str(resolved_family),
            "resolution_source": "append_pool_family",
            "fallback_used": False,
            "uses_replay_pool": False,
        }
        append_family_pool = tuple(family_pool)
        append_pool_meta = dict(pool_meta)
        append_pool_meta.update(
            {
                "candidate_pool_complete": True,
                "append_pool_source": "reused_complete_replay_pool_same_family",
                "replay_family": str(resolved_family),
                "append_family": str(resolved_family),
            }
        )
        append_family_terms_count = int(len(family_pool))
    else:
        if str(runtime_input.resolved_problem.family_key) == "hh":
            raise ValueError(
                "Neutral realtime entrypoint only supports --append-pool-family match_replay for HH; "
                "use pipelines.time_dynamics.runners.hh_from_adapt_artifact for HH-specific append families."
            )
        append_terms, explicit_append_meta = build_runtime_pool_terms(
            pool_key=str(append_request),
            problem_key=str(runtime_input.resolved_problem.family_key),
            h_poly=runtime_input.h_poly,
            **dict(runtime_input.candidate_pool_source.pool_build_kwargs),
        )
        append_family_info = {
            "requested": append_request,
            "resolved": append_request,
            "resolution_source": "append_pool_family",
            "fallback_used": False,
            "uses_replay_pool": False,
        }
        append_family_pool = tuple(append_terms)
        append_pool_meta = dict(explicit_append_meta)
        append_pool_meta.update(
            {
                "candidate_pool_complete": True,
                "append_pool_source": "explicit_runtime_pool",
                "replay_family": str(resolved_family),
                "append_family": str(append_request),
            }
        )
        append_family_terms_count = int(len(append_terms))

    theta_logical = runtime_input.theta_logical
    if theta_logical is None:
        theta_logical = np.zeros(
            int(runtime_input.base_layout.logical_parameter_count),
            dtype=float,
        )
    return ReplayScaffoldContext(
        cfg=SimpleNamespace(
            L=int(request.num_sites),
            ordering=str(request.ordering),
            reps=1,
        ),
        h_poly=runtime_input.h_poly,
        psi_ref=np.asarray(runtime_input.psi_ref, dtype=complex).reshape(-1),
        payload_in={
            "settings": {
                "problem": str(runtime_input.resolved_problem.family_key),
                "L": int(request.num_sites),
                "ordering": str(request.ordering),
            },
            "adapt_vqe": {
                "pool_type": str(resolved_family),
            },
        },
        family_info=dict(family_info),
        family_pool=tuple(family_pool),
        pool_meta=dict(pool_meta),
        replay_terms=tuple(runtime_input.selected_terms),
        base_layout=runtime_input.base_layout,
        adapt_theta_runtime=np.asarray(runtime_input.theta_runtime, dtype=float).reshape(-1),
        adapt_theta_logical=np.asarray(theta_logical, dtype=float).reshape(-1),
        adapt_depth=int(len(tuple(runtime_input.selected_terms))),
        handoff_state_kind=str(runtime_input.provenance.get("handoff_state_kind", "prepared_state")),
        provenance_source=str(runtime_input.provenance.get("provenance_source", "runtime_loader")),
        family_terms_count=int(len(tuple(family_pool))),
        append_family_info=append_family_info,
        append_family_pool=append_family_pool,
        append_pool_meta=append_pool_meta,
        append_family_terms_count=append_family_terms_count,
    )


"Built Math: seed(args) := {runtime_input, replay_context, cfg, drive, oracle, H}; seed is reusable before controller instantiation."
def build_controller_seed_from_args(
    args: argparse.Namespace,
    *,
    cfg: Any | None = None,
) -> RealtimeControllerSeed:
    artifact_json = Path(args.artifact_json).expanduser().resolve()
    runtime_input = load_scaffold_runtime_input(
        artifact_json,
        loader_mode=str(args.loader_mode),
        tag=str(args.run_tag),
        generator_family=str(args.generator_family),
        fallback_family=str(args.fallback_family),
    )
    family_key = str(runtime_input.resolved_problem.family_key)
    drive_requested = bool(getattr(args, "enable_drive", False))
    resolved_cfg = HH_REALTIME_ADAPTER.build_controller_config(args) if cfg is None else cfg
    primary_density_mode = str(
        getattr(resolved_cfg, "exact_forecast_primary_density_target_mode", "auto")
    ).strip().lower()
    route = validate_realtime_route_request(
        family_key=family_key,
        controller_mode=str(resolved_cfg.mode),
        reference_mode=str(resolved_cfg.reference_mode),
        drive_requested=drive_requested,
        strict_qpu_faithful=strict_qpu_faithful_requested(args),
        append_pool_family=getattr(args, "append_pool_family", "match_replay"),
        num_sites=int(runtime_input.resolved_problem.request.num_sites),
        drive_include_identity=bool(getattr(args, "drive_include_identity", False)),
        primary_density_mode=str(primary_density_mode),
    )
    effective_append_pool_family = str(route.effective_append_pool_family)
    replay_context = _replay_context_from_runtime_input(
        runtime_input,
        append_pool_family=str(effective_append_pool_family),
    )
    request = runtime_input.resolved_problem.request
    drive_config = HH_REALTIME_ADAPTER.build_drive_config(
        args,
        n_sites=int(request.num_sites),
        ordering=str(request.ordering),
    )
    oracle_config = HH_REALTIME_ADAPTER.build_oracle_config(args)
    h_poly = runtime_input.h_poly
    strict_qpu_faithful = strict_qpu_faithful_requested(args)
    hmat = (
        None
        if strict_qpu_faithful
        else np.asarray(hamiltonian_matrix(h_poly), dtype=complex)
    )
    loaded = SimpleNamespace(
        replay_context=replay_context,
        runtime_input=runtime_input,
        psi_initial=np.asarray(runtime_input.psi_initial, dtype=complex).reshape(-1),
    )
    return RealtimeControllerSeed(
        loaded=loaded,
        runtime_input=runtime_input,
        cfg=resolved_cfg,
        oracle_config=oracle_config,
        drive_config=drive_config,
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
    )


"Built Math: controller(seed, cfg) := McLachlan(seed.H, psi_0, theta_0, cfg); exact_helper is attached only for benchmark_exact."
def finalize_controller_bundle_from_seed(
    args: argparse.Namespace,
    *,
    seed: RealtimeControllerSeed,
    exact_reference_cache: dict[str, object] | None = None,
) -> dict[str, Any]:
    strict_qpu_faithful = strict_qpu_faithful_requested(args)
    controller = RealtimeCheckpointController(
        cfg=seed.cfg,
        replay_context=seed.replay_context,
        h_poly=seed.h_poly,
        hmat=seed.hmat,
        psi_initial=np.asarray(seed.runtime_input.psi_initial, dtype=complex),
        best_theta=np.asarray(seed.runtime_input.theta_runtime, dtype=float),
        allow_repeats=bool(args.allow_repeats),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        drive_config=seed.drive_config,
        oracle_base_config=seed.oracle_config,
        progress_path=getattr(args, "progress_json", None),
        partial_payload_path=getattr(args, "partial_payload_json", None),
        exact_reference_cache=exact_reference_cache,
        resolved_problem=seed.runtime_input.resolved_problem,
        strict_qpu_faithful=strict_qpu_faithful,
    )
    exact_helper = (
        None
        if strict_qpu_faithful or str(seed.cfg.reference_mode) != "benchmark_exact"
        else build_exact_audit_helper_for_controller(
            controller,
            exact_reference_cache=exact_reference_cache,
        )
    )
    resolved_drive_config = getattr(controller, "_drive_config", seed.drive_config)
    return {
        "loaded": seed.loaded,
        "runtime_input": seed.runtime_input,
        "cfg": seed.cfg,
        "oracle_config": seed.oracle_config,
        "drive_config": resolved_drive_config,
        "controller": controller,
        "exact_helper": exact_helper,
        "strict_qpu_faithful": strict_qpu_faithful,
        "strict_qpu_hh": bool(
            strict_qpu_faithful
            and str(seed.runtime_input.resolved_problem.family_key) == "hh"
        ),
    }


"Built Math: bundle(args) := seed(args) + controller(seed); non-HH drive is narrowed to spin_boson plus exact_v1 benchmark lattice/boson-chain routes."
def build_controller_bundle_from_args(
    args: argparse.Namespace,
    *,
    exact_reference_cache: dict[str, object] | None = None,
) -> dict[str, Any]:
    seed = build_controller_seed_from_args(args)
    return finalize_controller_bundle_from_seed(
        args,
        seed=seed,
        exact_reference_cache=exact_reference_cache,
    )


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    output_json = Path(args.output_json).expanduser().resolve()
    bundle = build_controller_bundle_from_args(args)
    loaded = bundle["loaded"]
    cfg = bundle["cfg"]
    oracle_config = bundle["oracle_config"]
    resolved_drive_config = bundle["drive_config"]
    controller = bundle["controller"]
    exact_helper = bundle.get("exact_helper")
    fixed_scaffold_parity_observer = (
        FixedScaffoldQiskitParityObserver(runtime_input=bundle["runtime_input"])
        if bool(getattr(args, "emit_fixed_scaffold_qiskit_parity_payload", False))
        else None
    )
    ed_ground_exact_energy = bundle["runtime_input"].exact_energy
    ed_ground_exact_energy_source = "runtime_input.exact_energy"
    if fixed_scaffold_parity_observer is not None and exact_helper is not None:
        raise ValueError(
            "fixed scaffold Qiskit parity observer is only supported on the repo-native "
            "controller path, not exact-audit wrapper runs"
        )
    if exact_helper is None:
        result = (
            controller.run()
            if fixed_scaffold_parity_observer is None
            else controller.run(checkpoint_observer=fixed_scaffold_parity_observer)
        )
    else:
        result = HH_REALTIME_ADAPTER.run_controller_with_exact_audit(
            controller,
            exact_helper,
            ed_ground_exact_energy=ed_ground_exact_energy,
            ed_ground_exact_energy_source=ed_ground_exact_energy_source,
        )
    result, diagnostic_reference = _attach_diagnostic_exact_reference(
        args=args,
        controller=controller,
        result=result,
    )
    compile_audit_config = HH_REALTIME_ADAPTER.build_compile_audit_config_from_args(args)
    compile_audit = None
    if str(compile_audit_config.mode) != "off":
        compile_audit = HH_REALTIME_ADAPTER.run_final_scaffold_compile_audit(
            controller=controller,
            config=compile_audit_config,
        )
        compile_audit = dict(compile_audit)
        compile_audit["prune_event_audit"] = HH_REALTIME_ADAPTER.run_prune_event_compile_audit(
            controller=controller,
            config=compile_audit_config,
        )
    payload = HH_REALTIME_ADAPTER.build_output_payload(
        args=args,
        loaded=loaded,
        cfg=cfg,
        drive_config=resolved_drive_config,
        oracle_config=oracle_config,
        result=result,
        compile_audit=compile_audit,
        ed_ground_exact_energy=ed_ground_exact_energy,
        ed_ground_exact_energy_source=ed_ground_exact_energy_source,
        diagnostic_reference=diagnostic_reference,
        problem_family=str(bundle["runtime_input"].resolved_problem.family_key),
    )
    if fixed_scaffold_parity_observer is not None:
        payload["fixed_scaffold_parity_payload"] = fixed_scaffold_parity_observer.to_payload()
    payload["runtime_contract"] = {
        "problem_family": str(bundle["runtime_input"].resolved_problem.family_key),
        "hamiltonian_capabilities": asdict(
            bundle["runtime_input"].resolved_problem.capabilities
        ),
        "structure_locked": bool(bundle["runtime_input"].structure_locked),
        "candidate_pool_complete": bool(bundle["runtime_input"].candidate_pool_source.candidate_pool_complete),
        "selected_term_count": int(len(tuple(bundle["runtime_input"].selected_terms))),
        "candidate_pool_term_count": int(len(tuple(bundle["runtime_input"].candidate_pool_terms))),
        "controller_exact_input_mode": payload.get("summary", {}).get(
            "controller_exact_input_mode",
            payload.get("route_config", {}).get("controller_exact_input_mode", "off"),
        ),
        "diagnostic_exact_reference_mode": payload.get("summary", {}).get(
            "diagnostic_exact_reference_mode",
            payload.get("route_config", {}).get("diagnostic_exact_reference_mode", "off"),
        ),
        "decision_data_flow": payload.get("summary", {}).get(
            "decision_data_flow",
            payload.get("route_config", {}).get("decision_data_flow", "unknown"),
        ),
        "uses_reference_for_decision": bool(
            payload.get("summary", {}).get(
                "uses_reference_for_decision",
                payload.get("route_config", {}).get("uses_reference_for_decision", False),
            )
        ),
        "uses_future_exact_forecast_for_decision": bool(
            payload.get("summary", {}).get(
                "uses_future_exact_forecast_for_decision",
                payload.get("route_config", {}).get(
                    "uses_future_exact_forecast_for_decision", False
                ),
            )
        ),
        "uses_statevector_as_ideal_observable_estimator": bool(
            payload.get("summary", {}).get(
                "uses_statevector_as_ideal_observable_estimator",
                payload.get("route_config", {}).get(
                    "uses_statevector_as_ideal_observable_estimator", False
                ),
            )
        ),
        "strict_measurement_oracle_certified": bool(
            payload.get("summary", {}).get(
                "strict_measurement_oracle_certified",
                payload.get("route_config", {}).get(
                    "strict_measurement_oracle_certified", False
                ),
            )
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = HH_REALTIME_ADAPTER.build_parser()
    parser.description = "Run generic realtime checkpoint controller from an ADAPT artifact."
    parser.add_argument(
        "--emit-fixed-scaffold-qiskit-parity-payload",
        action="store_true",
        help=(
            "Emit post-run fixed-scaffold layout/theta/state data for benchmark-local "
            "Qiskit parity. This is diagnostic only and is not used by controller decisions."
        ),
    )
    parser.set_defaults(checkpoint_controller_mode="off")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_from_args(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
