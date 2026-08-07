"""Neutral scaffold runtime loader for legacy HH and generic artifact inputs."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.hh_vqe_from_adapt_family import build_replay_scaffold_context
from pipelines.scaffold.runtime_contract import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.static_adapt.builders.primitive_pools import build_runtime_pool_terms
from pipelines.contracts.problem import ProblemRequest, canonical_problem_key
from pipelines.static_adapt.builders.problem_registry import resolve_problem_context
from pipelines.static_adapt.builders.problem_setup import _resolve_exact_energy_from_payload
from pipelines.scaffold.hh_fixed_manifold_loader import (
    FixedManifoldRunSpec,
    LoadedRunContext,
    _make_replay_run_cfg,
    _statevector_from_named_payload_state,
    _validate_prepared_state_consistency,
    build_fixed_scaffold_context_from_payload,
    normalize_replay_payload,
)
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    build_parameter_layout,
    deserialize_layout,
    expand_legacy_logical_theta,
    project_runtime_theta_block_mean,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


PREPARED_STATE_RECONSTRUCTION_ATOL = 1.0e-10
DIAGNOSTIC_REPLAY_FAMILY_POOL_MODES = {
    "diagnostic_replay_family_pool",
    "family_pool",
    "replay_family_pool",
}
LEGAL_SUBSPACE_APPEND_GUARD_SCHEMA_V1 = "ap_candidate_pool_legal_subspace_hard_guard_v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object payload at {path}.")
    return dict(payload)


def _parse_cli_args_for_settings(args: Sequence[Any]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    tokens = [str(x) for x in args]
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if not token.startswith("--"):
            idx += 1
            continue
        raw_key = token[2:]
        raw_value: Any = True
        if "=" in raw_key:
            raw_key, raw_value = raw_key.split("=", 1)
        elif idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
            raw_value = tokens[idx + 1]
            idx += 1
        key = str(raw_key).replace("-", "_")
        parsed[key] = raw_value
        idx += 1
    return parsed


def _statevector_to_payload(
    vec: np.ndarray,
    *,
    source: str,
    handoff_state_kind: str,
    cutoff: float = 1.0e-14,
) -> dict[str, Any]:
    arr = np.asarray(vec, dtype=complex).reshape(-1)
    if arr.size <= 0 or arr.size & (arr.size - 1):
        raise ValueError("Statevector length must be a positive power of two.")
    nq = int(round(math.log2(int(arr.size))))
    amplitudes: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(arr):
        if abs(complex(amp)) <= float(cutoff):
            continue
        amplitudes[format(idx, f"0{nq}b")] = {
            "re": float(np.real(amp)),
            "im": float(np.imag(amp)),
        }
    if not amplitudes:
        raise ValueError("Statevector payload would be empty.")
    return {
        "source": str(source),
        "handoff_state_kind": str(handoff_state_kind),
        "nq_total": int(nq),
        "amplitudes_qn_to_q0": amplitudes,
    }


def _settings_from_static_comparator_wrapper(payload: Mapping[str, Any]) -> dict[str, Any]:
    spec = payload.get("spec", {})
    if not isinstance(spec, Mapping):
        spec = {}
    base_args = spec.get("base_pipeline_args", None)
    if not isinstance(base_args, Sequence) or isinstance(base_args, (str, bytes)):
        raise ValueError("Static comparator wrapper missing spec.base_pipeline_args.")
    parsed = _parse_cli_args_for_settings(base_args)
    guardrails = payload.get("guardrails", {})
    if not isinstance(guardrails, Mapping):
        guardrails = {}
    settings: dict[str, Any] = {
        "problem": parsed.get("problem", payload.get("family", "hh")),
        "L": parsed.get("L", parsed.get("num_sites", spec.get("features", {}).get("L", 0) if isinstance(spec.get("features", {}), Mapping) else 0)),
        "t": parsed.get("t", 1.0),
        "u": parsed.get("u", parsed.get("U", 0.0)),
        "dv": parsed.get("dv", 0.0),
        "omega0": parsed.get("omega0", 1.0),
        "g_ep": parsed.get("g_ep", 0.0),
        "n_ph_max": parsed.get("n_ph_max", 0),
        "boson_encoding": parsed.get("boson_encoding", "binary"),
        "ordering": parsed.get("ordering", "blocked"),
        "boundary": parsed.get("boundary", "open"),
        "v_nn": parsed.get("v_nn", 0.0),
        "t_prime": parsed.get("t_prime", 0.0),
        "include_zero_point": parsed.get("include_zero_point", True),
        "adapt_pool": guardrails.get("pool_name", parsed.get("adapt_pool", "full_meta")),
    }
    for optional_key in (
        "sector_n_up",
        "sector_n_dn",
        "paop_r",
        "paop_split_paulis",
        "paop_prune_eps",
        "paop_normalization",
    ):
        if optional_key in parsed:
            settings[optional_key] = parsed[optional_key]
    return settings


def _adapt_vqe_from_static_comparator_wrapper(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = payload.get("result", {})
    if not isinstance(result, Mapping):
        raise ValueError("Static comparator wrapper missing result object.")
    operators = result.get("selected_operators", None)
    theta = result.get("theta", None)
    if not isinstance(operators, Sequence) or isinstance(operators, (str, bytes)) or len(operators) == 0:
        raise ValueError("Static comparator wrapper missing result.selected_operators.")
    if not isinstance(theta, Sequence) or isinstance(theta, (str, bytes)) or len(theta) == 0:
        raise ValueError("Static comparator wrapper missing result.theta.")
    if len(operators) != len(theta):
        raise ValueError(
            "Static comparator wrapper selected-operator/theta length mismatch: "
            f"{len(operators)} vs {len(theta)}."
        )
    guardrails = payload.get("guardrails", {})
    if not isinstance(guardrails, Mapping):
        guardrails = {}
    exact_energy = (
        result.get("same_cutoff_exact_gs_energy", None)
        if result.get("same_cutoff_exact_gs_energy", None) is not None
        else result.get("exact_gs_energy", result.get("exact_energy", None))
    )
    adapt: dict[str, Any] = {
        "operators": [str(x) for x in operators],
        "optimal_point": [float(x) for x in theta],
        "logical_optimal_point": [float(x) for x in theta],
        "pool_type": str(guardrails.get("pool_name", result.get("adapt_pool", "full_meta"))),
        "structure_locked": False,
    }
    if exact_energy is not None:
        adapt["exact_gs_energy"] = float(exact_energy)
    return adapt


def _normalize_static_comparator_wrapper_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Lift generic static comparator wrappers into the scaffold runtime shape.

    Paper-I comparator records such as ``generic_static_single.json`` are
    provenance/report wrappers.  Their selected scaffold is still complete, but
    it lives under ``result`` rather than the top-level ``settings`` and
    ``adapt_vqe`` keys consumed by the neutral runtime contract.
    """

    if isinstance(payload.get("settings", None), Mapping) and isinstance(payload.get("adapt_vqe", None), Mapping):
        return dict(payload)
    schema = str(payload.get("schema", "")).strip()
    if schema not in {
        "generic_static_adapt_variant_single_v1",
        "generic_static_single_v1",
        "generic_static_family_informed_vqe_single_v1",
    } and not (
        isinstance(payload.get("result", None), Mapping)
        and payload.get("runtime_seed_schema", None) == "paper_ii_static_seed_runtime_payload_v1"
    ):
        return dict(payload)

    out = dict(payload)
    out["settings"] = _settings_from_static_comparator_wrapper(payload)
    out["adapt_vqe"] = _adapt_vqe_from_static_comparator_wrapper(payload)
    out["scaffold_runtime_normalization"] = {
        "source": "generic_static_comparator_wrapper",
        "source_schema": schema,
        "runtime_seed_json": payload.get("runtime_seed_json", None),
        "runtime_seed_schema": payload.get("runtime_seed_schema", None),
    }

    if not isinstance(out.get("ansatz_input_state", None), Mapping):
        request = _problem_request_from_payload(out)
        resolved_problem = resolve_problem_context(request)
        psi_ref = np.asarray(resolved_problem.reference_state.build_state(), dtype=complex).reshape(-1)
        out["ansatz_input_state"] = _statevector_to_payload(
            psi_ref,
            source="resolved_problem.reference_state",
            handoff_state_kind="reference_state",
        )
    return out


def _boolish(raw: Any, default: bool) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return bool(raw)
    if isinstance(raw, str):
        lowered = str(raw).strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(raw)


def _extract_continuation_block(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    top = payload.get("continuation", None)
    if isinstance(top, Mapping):
        return dict(top)
    adapt_vqe = payload.get("adapt_vqe", None)
    if isinstance(adapt_vqe, Mapping):
        nested = adapt_vqe.get("continuation", None)
        if isinstance(nested, Mapping):
            return dict(nested)
    return {}


def _selected_generator_term_map(payload: Mapping[str, Any]) -> dict[str, AnsatzTerm]:
    continuation = _extract_continuation_block(payload)
    selected_meta = (
        continuation.get("selected_generator_metadata", [])
        if isinstance(continuation, Mapping)
        else []
    )
    if not isinstance(selected_meta, Sequence):
        return {}
    out: dict[str, AnsatzTerm] = {}
    for raw_meta in selected_meta:
        if not isinstance(raw_meta, Mapping):
            continue
        label = str(raw_meta.get("candidate_label", "")).strip()
        if label == "" or label in out:
            continue
        compile_meta = raw_meta.get("compile_metadata", None)
        serialized_terms = (
            compile_meta.get("serialized_terms_exyz", [])
            if isinstance(compile_meta, Mapping)
            else []
        )
        if not isinstance(serialized_terms, Sequence):
            continue
        poly = PauliPolynomial("JW")
        try:
            for term_info in serialized_terms:
                if not isinstance(term_info, Mapping):
                    raise ValueError("serialized_terms_exyz entries must be mappings.")
                label_exyz = str(term_info.get("pauli_exyz", "")).strip()
                nq = int(term_info.get("nq", len(label_exyz)))
                coeff = complex(
                    float(term_info.get("coeff_re", 0.0)),
                    float(term_info.get("coeff_im", 0.0)),
                )
                poly.add_term(PauliTerm(int(nq), ps=label_exyz, pc=coeff))
            poly._reduce()
        except Exception:
            continue
        out[label] = AnsatzTerm(label=label, polynomial=poly)
    return out


def _selected_generator_family_id(payload: Mapping[str, Any]) -> str | None:
    continuation = _extract_continuation_block(payload)
    selected_meta = (
        continuation.get("selected_generator_metadata", [])
        if isinstance(continuation, Mapping)
        else []
    )
    if not isinstance(selected_meta, Sequence):
        return None
    family_ids = [
        str(raw_meta.get("family_id", "")).strip().lower()
        for raw_meta in selected_meta
        if isinstance(raw_meta, Mapping) and str(raw_meta.get("family_id", "")).strip()
    ]
    if not family_ids:
        return None
    first = str(family_ids[0])
    if any(str(x) != first for x in family_ids[1:]):
        return None
    return first


def _ansatz_term_from_layout_block(block: Any) -> AnsatzTerm:
    poly = PauliPolynomial("JW")
    for spec in getattr(block, "terms", ()):
        poly.add_term(
            PauliTerm(
                int(spec.nq),
                ps=str(spec.pauli_exyz),
                pc=float(spec.coeff_real),
            )
        )
    poly._reduce()
    return AnsatzTerm(label=str(block.candidate_label), polynomial=poly)


def _selected_terms_from_layout(layout: AnsatzParameterLayout) -> tuple[AnsatzTerm, ...]:
    return tuple(_ansatz_term_from_layout_block(block) for block in layout.blocks)


def _resolve_selected_terms_from_labels(
    *,
    operator_labels: Sequence[str],
    candidate_pool_terms: Sequence[AnsatzTerm],
    payload_terms: Mapping[str, AnsatzTerm],
) -> tuple[AnsatzTerm, ...]:
    pool_matches: dict[str, list[AnsatzTerm]] = {}
    for term in candidate_pool_terms:
        pool_matches.setdefault(str(term.label), []).append(term)
    selected_terms: list[AnsatzTerm] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    for raw_label in operator_labels:
        label = str(raw_label)
        if label in payload_terms:
            selected_terms.append(payload_terms[label])
            continue
        matches = list(pool_matches.get(label, ()))
        if len(matches) == 1:
            selected_terms.append(matches[0])
            continue
        if len(matches) > 1:
            ambiguous.append(label)
            continue
        parent_label = label.split("::child_set")[0] if "::child_set" in label else label
        if parent_label in payload_terms:
            parent = payload_terms[parent_label]
            selected_terms.append(AnsatzTerm(label=label, polynomial=parent.polynomial))
            continue
        parent_matches = list(pool_matches.get(parent_label, ()))
        if len(parent_matches) == 1:
            selected_terms.append(
                AnsatzTerm(label=label, polynomial=parent_matches[0].polynomial)
            )
            continue
        if len(parent_matches) > 1:
            ambiguous.append(label)
            continue
        missing.append(label)
    if ambiguous:
        preview = ", ".join(ambiguous[:8])
        raise ValueError(
            "Ambiguous selected generator label(s) in runtime pool; artifact must carry "
            f"serialized parameterization or continuation metadata. Examples: {preview}"
        )
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"Could not reconstruct selected generator labels: {preview}")
    return tuple(selected_terms)


def _resolve_runtime_layout_and_theta(
    payload: Mapping[str, Any],
    selected_terms: Sequence[AnsatzTerm],
) -> tuple[AnsatzParameterLayout, np.ndarray]:
    adapt_vqe = payload.get("adapt_vqe", {})
    if not isinstance(adapt_vqe, Mapping):
        raise ValueError("Artifact payload missing adapt_vqe block required for runtime loading.")
    optimal_point = np.asarray(adapt_vqe.get("optimal_point", []), dtype=float).reshape(-1)
    parameterization = adapt_vqe.get("parameterization", None)
    if isinstance(parameterization, Mapping):
        layout = deserialize_layout(parameterization)
        if int(optimal_point.size) != int(layout.runtime_parameter_count):
            raise ValueError(
                "Runtime theta length "
                f"{optimal_point.size} does not match serialized layout runtime count "
                f"{layout.runtime_parameter_count}."
            )
        return layout, optimal_point
    layout = build_parameter_layout(
        list(selected_terms),
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    theta_runtime = expand_legacy_logical_theta(optimal_point, layout)
    return layout, np.asarray(theta_runtime, dtype=float).reshape(-1)


def _resolve_theta_logical(
    payload: Mapping[str, Any],
    *,
    layout: AnsatzParameterLayout,
    theta_runtime: np.ndarray,
) -> np.ndarray | None:
    adapt_vqe = payload.get("adapt_vqe", {})
    if isinstance(adapt_vqe, Mapping) and adapt_vqe.get("logical_optimal_point", None) is not None:
        theta_logical = np.asarray(
            adapt_vqe.get("logical_optimal_point", []),
            dtype=float,
        ).reshape(-1)
        if int(theta_logical.size) != int(layout.logical_parameter_count):
            raise ValueError(
                "logical_optimal_point length mismatch: got "
                f"{theta_logical.size}, expected {layout.logical_parameter_count}."
            )
        return theta_logical
    if int(layout.logical_parameter_count) <= 0:
        return np.zeros(0, dtype=float)
    return project_runtime_theta_block_mean(theta_runtime, layout)


def _reconstruct_prepared_state_from_runtime_input(
    *,
    selected_terms: Sequence[AnsatzTerm],
    layout: AnsatzParameterLayout,
    theta_runtime: np.ndarray,
    psi_ref: np.ndarray,
    theta_logical: np.ndarray | None = None,
    parameterization_mode: str = "per_pauli_term",
) -> np.ndarray:
    mode = str(parameterization_mode).strip().lower()
    if mode not in {"logical_shared", "per_pauli_term"}:
        raise ValueError(
            "Unsupported prepared-state reconstruction parameterization mode: "
            f"{parameterization_mode!r}."
        )
    executor = CompiledAnsatzExecutor(
        list(selected_terms),
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
        parameterization_mode=mode,
        parameterization_layout=layout,
    )
    theta_exec = np.asarray(theta_runtime, dtype=float).reshape(-1)
    if mode == "logical_shared":
        theta_exec = (
            project_runtime_theta_block_mean(theta_exec, layout)
            if theta_logical is None
            else np.asarray(theta_logical, dtype=float).reshape(-1)
        )
    return np.asarray(
        executor.prepare_state(
            theta_exec,
            np.asarray(psi_ref, dtype=complex).reshape(-1),
        ),
        dtype=complex,
    ).reshape(-1)


def _prepared_state_reconstruction_error(
    psi_reconstructed: np.ndarray,
    psi_initial: np.ndarray,
    *,
    tol: float = PREPARED_STATE_RECONSTRUCTION_ATOL,
) -> float:
    reconstructed = np.asarray(psi_reconstructed, dtype=complex).reshape(-1)
    initial = np.asarray(psi_initial, dtype=complex).reshape(-1)
    if int(reconstructed.size) != int(initial.size):
        raise ValueError(
            "Prepared-state parity check failed: reconstructed and payload "
            f"state dimensions differ ({reconstructed.size} != {initial.size})."
        )
    err = float(np.linalg.norm(reconstructed - initial))
    if not np.isfinite(err):
        raise ValueError("Prepared-state parity check failed: reconstruction error is not finite.")
    if err > float(tol):
        raise ValueError(
            "Prepared-state parity check failed: "
            f"||psi_reconstructed - psi_initial||={err:.3e} > {float(tol):.3e}."
        )
    return float(err)


def _pool_build_kwargs_from_request(
    request: ProblemRequest,
    *,
    default_num_particles: tuple[int, int],
    resolved_problem: Any | None = None,
) -> dict[str, Any]:
    runtime_data = (
        dict(getattr(resolved_problem, "runtime_data", {}) or {})
        if resolved_problem is not None and isinstance(getattr(resolved_problem, "runtime_data", {}), Mapping)
        else {}
    )
    return {
        "num_sites": int(request.num_sites),
        "t": float(request.t),
        "u": float(request.u),
        "dv": float(request.dv),
        "omega0": float(request.omega0),
        "g_ep": float(request.g_ep),
        "n_ph_max": int(request.n_ph_max),
        "boson_encoding": str(request.boson_encoding),
        "ordering": str(request.ordering),
        "boundary": str(request.boundary),
        "include_zero_point": bool(request.include_zero_point),
        "v_nn": float(request.v_nn),
        "t_prime": float(request.t_prime),
        "num_particles": tuple(int(x) for x in default_num_particles),
        "molecular_problem": runtime_data.get("molecular_problem"),
        "vibronic_h2_model": runtime_data.get("vibronic_h2_model"),
    }


def _canonical_pool_key(raw: Any) -> str | None:
    if raw in {None, ""}:
        return None
    text = str(raw).strip().lower()
    return None if text == "" else text


def _resolve_generic_family_info(
    payload: Mapping[str, Any],
    *,
    resolved_problem,
    generator_family: str,
    fallback_family: str,
) -> dict[str, Any]:
    admissible = {
        str(key).strip().lower() for key in getattr(resolved_problem, "admissible_pool_keys", ())
    }
    requested_raw = _canonical_pool_key(generator_family)
    if requested_raw not in {None, "match_adapt"}:
        if requested_raw not in admissible:
            raise ValueError(
                f"Explicit generator family {generator_family!r} is not admissible for "
                f"problem family {resolved_problem.family_key!r}."
            )
        return {
            "requested": requested_raw,
            "resolved": requested_raw,
            "resolution_source": "generator_family",
            "fallback_used": False,
            "warning": None,
        }

    requested_from_payload = _selected_generator_family_id(payload)
    resolution_source = None
    for source, candidate in (
        ("continuation.selected_generator_metadata.family_id", requested_from_payload),
        (
            "adapt_vqe.pool_type",
            payload.get("adapt_vqe", {}).get("pool_type", None)
            if isinstance(payload.get("adapt_vqe", None), Mapping)
            else None,
        ),
        (
            "settings.adapt_pool",
            payload.get("settings", {}).get("adapt_pool", None)
            if isinstance(payload.get("settings", None), Mapping)
            else None,
        ),
    ):
        key = _canonical_pool_key(candidate)
        if key is None:
            continue
        requested_from_payload = key
        resolution_source = str(source)
        if key in admissible:
            return {
                "requested": key,
                "resolved": key,
                "resolution_source": str(source),
                "fallback_used": False,
                "warning": None,
            }
        break

    for source, candidate in (
        ("resolved_problem.default_pool_key", getattr(resolved_problem, "default_pool_key", None)),
        ("fallback_family", fallback_family),
    ):
        key = _canonical_pool_key(candidate)
        if key is None:
            continue
        if key in admissible:
            warning = None
            if requested_from_payload is not None and requested_from_payload not in admissible:
                warning = (
                    f"Requested pool key {requested_from_payload!r} is not admissible for "
                    f"{resolved_problem.family_key!r}; fell back to {key!r}."
                )
            return {
                "requested": requested_from_payload,
                "resolved": key,
                "resolution_source": str(source if resolution_source is None else source),
                "fallback_used": True,
                "warning": warning,
            }

    raise ValueError(
        f"Could not resolve an admissible runtime pool key for problem family "
        f"{resolved_problem.family_key!r}."
    )


def _load_generic_runtime_input_from_payload(
    payload: Mapping[str, Any],
    *,
    artifact_json: Path,
    loader_mode: str | None,
    generator_family: str,
    fallback_family: str,
) -> ScaffoldRuntimeInput:
    resolved_loader_mode = "replay_family" if loader_mode in {None, ""} else str(loader_mode)
    if str(resolved_loader_mode) != "replay_family":
        raise ValueError(
            "Non-HH runtime loader currently supports loader_mode='replay_family' only."
        )
    request = _problem_request_from_payload(payload)
    resolved_problem = resolve_problem_context(request)
    family_info = _resolve_generic_family_info(
        payload,
        resolved_problem=resolved_problem,
        generator_family=str(generator_family),
        fallback_family=str(fallback_family),
    )
    pool_kwargs = _pool_build_kwargs_from_request(
        request,
        default_num_particles=tuple(
            int(x) for x in getattr(resolved_problem, "default_num_particles", (0, 0))
        ),
        resolved_problem=resolved_problem,
    )
    layout_parameterization = (
        payload.get("adapt_vqe", {}).get("parameterization", None)
        if isinstance(payload.get("adapt_vqe", None), Mapping)
        else None
    )
    candidate_pool_complete = not isinstance(layout_parameterization, Mapping)
    if candidate_pool_complete:
        candidate_pool_terms, candidate_pool_meta = build_runtime_pool_terms(
            pool_key=str(family_info["resolved"]),
            problem_key=str(resolved_problem.family_key),
            h_poly=resolved_problem.hamiltonian,
            **pool_kwargs,
        )
    else:
        candidate_pool_terms = ()
        candidate_pool_meta = {
            "pool_build_skipped": True,
            "reason": "serialized_parameterization_provides_selected_terms",
        }
    if isinstance(layout_parameterization, Mapping):
        layout, theta_runtime = _resolve_runtime_layout_and_theta(payload, ())
        selected_terms = _selected_terms_from_layout(layout)
    else:
        adapt_vqe = payload.get("adapt_vqe", {})
        if not isinstance(adapt_vqe, Mapping):
            raise ValueError("Artifact payload missing adapt_vqe block required for runtime loading.")
        operator_labels = [str(x) for x in adapt_vqe.get("operators", [])]
        if not operator_labels:
            raise ValueError(
                "Artifact payload missing adapt_vqe.operators required for non-HH runtime loading."
            )
        selected_terms = _resolve_selected_terms_from_labels(
            operator_labels=operator_labels,
            candidate_pool_terms=candidate_pool_terms,
            payload_terms=_selected_generator_term_map(payload),
        )
        layout, theta_runtime = _resolve_runtime_layout_and_theta(payload, selected_terms)
    theta_logical = _resolve_theta_logical(
        payload,
        layout=layout,
        theta_runtime=np.asarray(theta_runtime, dtype=float).reshape(-1),
    )
    ansatz_input_payload = payload.get("ansatz_input_state", None)
    if isinstance(ansatz_input_payload, Mapping):
        psi_ref = _statevector_from_named_payload_state(payload, "ansatz_input_state")
        psi_ref_source = "payload"
        psi_ref_source_location = "payload.ansatz_input_state"
        psi_ref_source_label = str(ansatz_input_payload.get("source", "payload"))
        psi_ref_handoff_state_kind = str(
            ansatz_input_payload.get("handoff_state_kind", "reference_state")
        )
    else:
        psi_ref = np.asarray(resolved_problem.reference_state.build_state(), dtype=complex).reshape(-1)
        psi_ref_source = "resolved_problem.reference_state"
        psi_ref_source_location = "resolved_problem.reference_state"
        psi_ref_source_label = str(
            getattr(resolved_problem.reference_state, "source_label", "reference_state")
        )
        psi_ref_handoff_state_kind = str(
            getattr(resolved_problem.reference_state, "state_kind", "reference_state")
        )
    psi_reconstructed = _reconstruct_prepared_state_from_runtime_input(
        selected_terms=selected_terms,
        layout=layout,
        theta_runtime=np.asarray(theta_runtime, dtype=float).reshape(-1),
        psi_ref=np.asarray(psi_ref, dtype=complex).reshape(-1),
        theta_logical=theta_logical,
        parameterization_mode=str(
            payload.get("adapt_vqe", {}).get(
                "parameterization_execution_mode",
                payload.get("adapt_vqe", {}).get(
                    "parameterization_mode",
                    "per_pauli_term",
                ),
            )
            if isinstance(payload.get("adapt_vqe", None), Mapping)
            else "per_pauli_term"
        ),
    )
    initial_state_payload = payload.get("initial_state", None)
    if isinstance(initial_state_payload, Mapping):
        psi_initial = _statevector_from_named_payload_state(payload, "initial_state")
        reconstruction_error = _prepared_state_reconstruction_error(
            psi_reconstructed,
            psi_initial,
            tol=PREPARED_STATE_RECONSTRUCTION_ATOL,
        )
        initial_state_source = "payload"
        initial_state_source_location = "payload.initial_state"
        initial_state_source_label = str(initial_state_payload.get("source", "payload"))
        initial_state_handoff_state_kind = str(
            initial_state_payload.get("handoff_state_kind", "prepared_state")
        )
    else:
        psi_initial = np.asarray(psi_reconstructed, dtype=complex).reshape(-1)
        reconstruction_error = 0.0
        initial_state_source = "reconstructed_from_scaffold"
        initial_state_source_location = "runtime_loader.reconstructed_from_scaffold"
        initial_state_source_label = "reconstructed_from_scaffold"
        initial_state_handoff_state_kind = "prepared_state"
    structure_locked = bool(
        payload.get("adapt_vqe", {}).get("structure_locked", False)
        if isinstance(payload.get("adapt_vqe", None), Mapping)
        else False
    )
    if candidate_pool_complete:
        candidate_pool_source = CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key=str(family_info["resolved"]),
            completeness="complete",
            pool_build_kwargs=_pool_build_kwargs_from_request(
                request,
                default_num_particles=tuple(
                    int(x) for x in getattr(resolved_problem, "default_num_particles", (0, 0))
                ),
                resolved_problem=resolved_problem,
            ),
            filter_payload={
                "resolution_source": family_info.get("resolution_source"),
                "fallback_used": bool(family_info.get("fallback_used", False)),
            },
        )
    else:
        candidate_pool_source = CandidatePoolSource(
            source_kind="selected_terms_only",
            pool_key=str(family_info["resolved"]),
            completeness="selected_only",
            pool_build_kwargs={},
            filter_payload={
                "resolution_source": family_info.get("resolution_source"),
                "fallback_used": bool(family_info.get("fallback_used", False)),
                "reason": "serialized_parameterization_provides_selected_terms",
            },
        )
    provenance = {
        "artifact_json": str(artifact_json),
        "loader_mode": "replay_family",
        "requested_family": family_info.get("requested", None),
        "resolved_family": family_info.get("resolved", None),
        "resolution_source": family_info.get("resolution_source", None),
        "handoff_state_kind": (
            payload.get("initial_state", {}).get("handoff_state_kind", None)
            if isinstance(payload.get("initial_state", None), Mapping)
            else None
        ),
        "provenance_source": "generic_runtime_loader",
    }
    extensions = {
        "generic_loader_summary": {
            "loader_mode": "replay_family",
            "input_artifact_json": str(artifact_json),
            "problem_family": str(resolved_problem.family_key),
            "candidate_pool_complete": bool(candidate_pool_complete),
            "family_pool_origin": (
                "build_runtime_pool_terms"
                if candidate_pool_complete
                else "serialized_parameterization_selected_terms_only"
            ),
            "logical_operator_count": int(layout.logical_parameter_count),
            "runtime_parameter_count": int(layout.runtime_parameter_count),
            "selected_term_count": int(len(selected_terms)),
            "candidate_pool_term_count": int(len(candidate_pool_terms)),
            "initial_state_source": str(initial_state_source),
            "reference_state_source": str(psi_ref_source),
            "ansatz_input_state_source_location": str(psi_ref_source_location),
            "ansatz_input_state_source": str(psi_ref_source_label),
            "ansatz_input_state_handoff_state_kind": str(psi_ref_handoff_state_kind),
            "initial_state_source_location": str(initial_state_source_location),
            "initial_state_payload_source": str(initial_state_source_label),
            "initial_state_handoff_state_kind": str(initial_state_handoff_state_kind),
            "prepared_state_reconstruction_error": float(reconstruction_error),
        },
        "generic_family_info": dict(family_info),
        "generic_pool_meta": dict(candidate_pool_meta),
    }
    return ScaffoldRuntimeInput(
        resolved_problem=resolved_problem,
        psi_ref=np.asarray(psi_ref, dtype=complex).reshape(-1),
        psi_initial=np.asarray(psi_initial, dtype=complex).reshape(-1),
        base_layout=layout,
        theta_runtime=np.asarray(theta_runtime, dtype=float).reshape(-1),
        theta_logical=(
            None
            if theta_logical is None
            else np.asarray(theta_logical, dtype=float).reshape(-1)
        ),
        structure_locked=structure_locked,
        exact_energy=_resolve_exact_energy_from_payload(payload),
        selected_terms=tuple(selected_terms),
        candidate_pool_terms=tuple(candidate_pool_terms),
        candidate_pool_source=candidate_pool_source,
        provenance=provenance,
        extensions=extensions,
    )


def _infer_legacy_loader_mode(payload: Mapping[str, Any]) -> str:
    adapt = payload.get("adapt_vqe", None)
    if not isinstance(adapt, Mapping):
        return "replay_family"
    fixed_meta = adapt.get("fixed_scaffold_metadata", None)
    pool_type = str(adapt.get("pool_type", "")).strip().lower()
    route_family = ""
    if isinstance(fixed_meta, Mapping):
        route_family = str(fixed_meta.get("route_family", "")).strip().lower()
    if (
        pool_type == "fixed_scaffold_locked"
        and bool(adapt.get("structure_locked", False))
        and route_family == "locked_imported_scaffold_v1"
    ):
        return "fixed_scaffold"
    return "replay_family"


def _problem_request_from_payload(payload: Mapping[str, Any]) -> ProblemRequest:
    settings = payload.get("settings", None)
    if not isinstance(settings, Mapping):
        raise ValueError("Artifact payload missing settings object required for runtime loading.")
    problem_key = canonical_problem_key(settings.get("problem", "hh"))
    raw_n_fermions = settings.get("n_fermions", None)
    raw_molecular_json = settings.get("molecular_problem_json", None)
    raw_molecular_h2_fixture_json = settings.get("molecular_vibronic_h2_fixture_json", None)
    raw_molecular_h2o_fixture_json = settings.get("molecular_vibronic_h2o_fixture_json", None)
    raw_molecular_h2o_linear_fd_fixture_json = settings.get(
        "molecular_vibronic_h2o_linear_fd_fixture_json",
        None,
    )
    return ProblemRequest(
        problem_key=problem_key,
        num_sites=int(settings.get("L", settings.get("num_sites", 0))),
        t=float(settings.get("t", 1.0)),
        u=float(settings.get("u", settings.get("U", 0.0))),
        dv=float(settings.get("dv", 0.0)),
        omega0=float(settings.get("omega0", 1.0)),
        g_ep=float(settings.get("g_ep", 0.0)),
        n_ph_max=int(settings.get("n_ph_max", 0)),
        boson_encoding=str(settings.get("boson_encoding", "binary")),
        ordering=str(settings.get("ordering", "blocked")),
        boundary=str(settings.get("boundary", "open")),
        include_zero_point=_boolish(settings.get("include_zero_point", True), True),
        molecular_problem_json=(
            None if raw_molecular_json in {None, ""} else str(Path(raw_molecular_json))
        ),
        molecular_vibronic_h2_fixture_json=(
            None
            if raw_molecular_h2_fixture_json in {None, ""}
            else str(Path(raw_molecular_h2_fixture_json))
        ),
        molecular_vibronic_h2o_fixture_json=(
            None
            if raw_molecular_h2o_fixture_json in {None, ""}
            else str(Path(raw_molecular_h2o_fixture_json))
        ),
        v_nn=float(settings.get("v_nn", 0.0)),
        t_prime=float(settings.get("t_prime", 0.0)),
        n_fermions=None if raw_n_fermions in {None, ""} else int(raw_n_fermions),
        molecular_vibronic_h2o_linear_fd_fixture_json=(
            None
            if raw_molecular_h2o_linear_fd_fixture_json in {None, ""}
            else str(Path(raw_molecular_h2o_linear_fd_fixture_json))
        ),
    )


def _reconstruct_prepared_state(
    loaded_context,
    *,
    use_logical_alias: bool = False,
) -> np.ndarray:
    layout = loaded_context.base_layout
    theta_runtime = np.asarray(
        loaded_context.adapt_theta_runtime,
        dtype=float,
    ).reshape(-1)
    if bool(use_logical_alias):
        theta_logical = getattr(loaded_context, "adapt_theta_logical", None)
        if theta_logical is None:
            raise ValueError(
                "Logical-shared prepared-state reconstruction requires "
                "adapt_theta_logical."
            )
        theta_runtime = np.asarray(
            expand_legacy_logical_theta(
                np.asarray(theta_logical, dtype=float).reshape(-1),
                layout,
            ),
            dtype=float,
        ).reshape(-1)
    executor = CompiledAnsatzExecutor(
        list(loaded_context.replay_terms),
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )
    return np.asarray(
        executor.prepare_state(
            theta_runtime,
            np.asarray(loaded_context.psi_ref, dtype=complex).reshape(-1),
        ),
        dtype=complex,
    ).reshape(-1)


def _resolve_prepared_state(
    payload: Mapping[str, Any],
    loaded_context,
) -> tuple[np.ndarray, str, float]:
    if isinstance(payload.get("initial_state", None), Mapping):
        psi_initial = _statevector_from_named_payload_state(payload, "initial_state")
        try:
            reconstruction_error = _validate_prepared_state_consistency(
                loaded_context,
                psi_initial,
                tol=PREPARED_STATE_RECONSTRUCTION_ATOL,
            )
            return np.asarray(psi_initial, dtype=complex).reshape(-1), "payload", float(
                reconstruction_error
            )
        except ValueError as runtime_exc:
            # HH full-meta execution can share one logical angle across every
            # Pauli factor while retaining the expanded runtime vector for
            # optimizer bookkeeping. Older checkpoints serialized both theta
            # views but not the execution mode. Accept the logical alias only
            # when it independently reproduces the persisted prepared state.
            try:
                psi_logical_alias = _reconstruct_prepared_state(
                    loaded_context,
                    use_logical_alias=True,
                )
                logical_alias_error = _prepared_state_reconstruction_error(
                    psi_logical_alias,
                    psi_initial,
                    tol=PREPARED_STATE_RECONSTRUCTION_ATOL,
                )
            except (TypeError, ValueError) as logical_alias_exc:
                raise ValueError(
                    "Prepared-state parity check failed; fallback to an unchecked "
                    "reconstructed state is disabled because neither the serialized "
                    "runtime coordinates nor their logical-shared alias reconstructs "
                    "the persisted state."
                ) from logical_alias_exc
            return (
                np.asarray(psi_initial, dtype=complex).reshape(-1),
                "payload_logical_shared_alias",
                float(logical_alias_error),
            )
    return _reconstruct_prepared_state(loaded_context), "reconstructed_from_scaffold", 0.0


def _load_legacy_loaded_context(
    payload: Mapping[str, Any],
    *,
    artifact_json: Path,
    loader_mode: str,
    tag: str,
    generator_family: str,
    fallback_family: str,
) -> LoadedRunContext:
    if str(loader_mode) == "replay_family":
        normalized = normalize_replay_payload(payload)
        continuation_lifted = (
            not isinstance(payload.get("continuation", None), Mapping)
            and isinstance(normalized.get("continuation", None), Mapping)
        )
        cfg = _make_replay_run_cfg(
            normalized,
            artifact_json=artifact_json,
            tag=tag,
            generator_family=str(generator_family),
            fallback_family=str(fallback_family),
        )
        psi_ref = _statevector_from_named_payload_state(normalized, "ansatz_input_state")
        replay_context = build_replay_scaffold_context(
            cfg,
            psi_ref=psi_ref,
            payload_in=normalized,
        )
        psi_initial, initial_state_source, reconstruction_error = _resolve_prepared_state(
            normalized,
            replay_context,
        )
        loader_summary = {
            "loader_mode": "replay_family",
            "input_artifact_json": str(artifact_json),
            "normalized_continuation_lifted": bool(continuation_lifted),
            "resolved_family": str(replay_context.family_info.get("resolved", "")),
            "resolution_source": str(replay_context.family_info.get("resolution_source", "")),
            "candidate_pool_complete": bool(
                replay_context.pool_meta.get("candidate_pool_complete", False)
            ),
            "fixed_manifold_locked": False,
            "lock_fixed_manifold_requested": False,
            "family_pool_origin": replay_context.pool_meta.get("family_pool_origin", None),
            "logical_operator_count": int(replay_context.base_layout.logical_parameter_count),
            "runtime_parameter_count": int(replay_context.base_layout.runtime_parameter_count),
            "initial_state_source": str(initial_state_source),
            "prepared_state_reconstruction_error": float(reconstruction_error),
        }
        payload_used = normalized
    elif str(loader_mode) == "fixed_scaffold":
        cfg = _make_replay_run_cfg(
            payload,
            artifact_json=artifact_json,
            tag=tag,
            generator_family="fixed_scaffold_locked",
            fallback_family=str(fallback_family),
        )
        replay_context = build_fixed_scaffold_context_from_payload(payload, cfg=cfg)
        psi_initial, initial_state_source, reconstruction_error = _resolve_prepared_state(
            payload,
            replay_context,
        )
        loader_summary = {
            "loader_mode": "fixed_scaffold",
            "input_artifact_json": str(artifact_json),
            "fixed_scaffold_kind": replay_context.pool_meta.get("fixed_scaffold_kind", None),
            "structure_locked": bool(replay_context.pool_meta.get("structure_locked", True)),
            "route_family": replay_context.pool_meta.get("route_family", None),
            "candidate_pool_complete": bool(
                replay_context.pool_meta.get("candidate_pool_complete", False)
            ),
            "fixed_manifold_locked": False,
            "lock_fixed_manifold_requested": False,
            "family_pool_origin": replay_context.pool_meta.get("family_pool_origin", None),
            "logical_operator_count": int(replay_context.base_layout.logical_parameter_count),
            "runtime_parameter_count": int(replay_context.base_layout.runtime_parameter_count),
            "initial_state_source": str(initial_state_source),
            "prepared_state_reconstruction_error": float(reconstruction_error),
        }
        payload_used = dict(payload)
    else:
        raise ValueError(
            f"Unsupported scaffold runtime loader_mode {loader_mode!r}; "
            "expected 'replay_family' or 'fixed_scaffold'."
        )

    return LoadedRunContext(
        spec=FixedManifoldRunSpec(
            name=str(artifact_json.stem),
            artifact_json=artifact_json,
            loader_mode=str(loader_mode),
            generator_family=str(generator_family),
            fallback_family=str(fallback_family),
        ),
        cfg=cfg,
        payload=dict(payload_used),
        replay_context=replay_context,
        psi_initial=np.asarray(psi_initial, dtype=complex).reshape(-1),
        loader_summary=dict(loader_summary),
    )


def _pool_build_kwargs(loaded: LoadedRunContext) -> dict[str, Any]:
    cfg = loaded.cfg
    return {
        "num_sites": int(cfg.L),
        "t": float(cfg.t),
        "u": float(cfg.u),
        "dv": float(cfg.dv),
        "omega0": float(cfg.omega0),
        "g_ep": float(cfg.g_ep),
        "n_ph_max": int(cfg.n_ph_max),
        "boson_encoding": str(cfg.boson_encoding),
        "ordering": str(cfg.ordering),
        "boundary": str(cfg.boundary),
        "num_particles": (int(cfg.sector_n_up), int(cfg.sector_n_dn)),
        "paop_r": int(cfg.paop_r),
        "paop_split_paulis": bool(cfg.paop_split_paulis),
        "paop_prune_eps": float(cfg.paop_prune_eps),
        "paop_normalization": str(cfg.paop_normalization),
    }


def _candidate_pool_source(loaded: LoadedRunContext, *, loader_mode: str) -> CandidatePoolSource:
    replay_context = loaded.replay_context
    family_info = dict(replay_context.family_info)
    pool_meta = dict(replay_context.pool_meta)

    filter_payload: dict[str, Any] = {}
    for key in (
        "selection_mode",
        "raw_sizes",
        "raw_total",
        "fixed_scaffold_kind",
        "route_family",
        "subject_kind",
        "family_pool_origin",
    ):
        if key in pool_meta and pool_meta.get(key, None) is not None:
            filter_payload[key] = pool_meta.get(key)

    if str(loader_mode) == "fixed_scaffold":
        return CandidatePoolSource(
            source_kind="selected_terms_only",
            pool_key="fixed_scaffold_locked",
            completeness="selected_only",
            pool_build_kwargs=_pool_build_kwargs(loaded),
            filter_payload=filter_payload,
        )

    requested_pool_mode = str(
        loaded.payload.get("replay_candidate_pool_mode", "") or ""
    ).strip()
    if requested_pool_mode in DIAGNOSTIC_REPLAY_FAMILY_POOL_MODES:
        family_pool = tuple(replay_context.family_pool)
        if not family_pool:
            raise ValueError(
                "Diagnostic replay-family append pool was requested, but the "
                "loaded replay context has an empty family_pool."
            )
        filter_payload["diagnostic_append_pool_override"] = {
            "enabled": True,
            "requested_mode": requested_pool_mode,
            "source": "replay_context.family_pool",
            "original_candidate_pool_complete": bool(
                pool_meta.get(
                    "candidate_pool_complete_before_payload_request",
                    pool_meta.get("candidate_pool_complete", False),
                )
            ),
            "original_selection_mode": pool_meta.get(
                "selection_mode_before_payload_request",
                pool_meta.get("selection_mode", None),
            ),
            "selected_term_count": int(len(tuple(replay_context.replay_terms))),
            "family_pool_term_count": int(len(family_pool)),
        }
        return CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key=(
                None
                if family_info.get("resolved", None) is None
                else str(family_info.get("resolved"))
            ),
            completeness="complete",
            pool_build_kwargs=_pool_build_kwargs(loaded),
            filter_payload=filter_payload,
        )

    candidate_pool_complete = bool(pool_meta.get("candidate_pool_complete", False))
    if candidate_pool_complete:
        source_kind = "resolved_pool"
        completeness = "complete"
    elif pool_meta.get("selection_mode", None) is not None:
        source_kind = "selected_terms_only"
        completeness = "selected_only"
    else:
        source_kind = "selected_terms_only"
        completeness = "partial"

    return CandidatePoolSource(
        source_kind=source_kind,
        pool_key=(
            None
            if family_info.get("resolved", None) is None
            else str(family_info.get("resolved"))
        ),
        completeness=completeness,
        pool_build_kwargs=_pool_build_kwargs(loaded),
        filter_payload=filter_payload,
    )


def _legal_subspace_dropped_labels(payload: Mapping[str, Any]) -> tuple[str, ...]:
    adapt = payload.get("adapt_vqe", None)
    if not isinstance(adapt, Mapping):
        return ()
    filter_meta = adapt.get("adapt_pool_legal_subspace_filter", None)
    if not isinstance(filter_meta, Mapping):
        return ()
    labels: list[str] = []
    seen: set[str] = set()
    for field in ("offender_labels", "component_risk_labels"):
        rows = filter_meta.get(field, ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("action", "")).strip().lower() != "dropped":
                continue
            label = str(row.get("label", "")).strip()
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append(label)
    return tuple(labels)


def _legal_subspace_no_pauli_split_parent_labels(payload: Mapping[str, Any]) -> tuple[str, ...]:
    adapt = payload.get("adapt_vqe", None)
    if not isinstance(adapt, Mapping):
        return ()
    filter_meta = adapt.get("adapt_pool_legal_subspace_filter", None)
    if not isinstance(filter_meta, Mapping):
        return ()
    rows = filter_meta.get("component_risk_labels", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ()
    labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        action = str(row.get("action", "")).strip().lower()
        if action == "dropped":
            continue
        try:
            leaking_count = int(row.get("termwise_component_leaking_term_count", 0) or 0)
        except (TypeError, ValueError):
            leaking_count = 0
        if leaking_count <= 0:
            continue
        label = str(row.get("label", "")).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return tuple(labels)


def _apply_legal_subspace_append_guard(
    candidate_pool_terms: Sequence[AnsatzTerm],
    *,
    selected_terms: Sequence[AnsatzTerm],
    payload: Mapping[str, Any],
) -> tuple[tuple[AnsatzTerm, ...], dict[str, Any] | None]:
    dropped = _legal_subspace_dropped_labels(payload)
    no_pauli_split = _legal_subspace_no_pauli_split_parent_labels(payload)
    if not dropped and not no_pauli_split:
        return tuple(candidate_pool_terms), None
    dropped_set = {str(label) for label in dropped}
    no_pauli_split_set = {str(label) for label in no_pauli_split}
    selected_labels = {
        str(getattr(term, "label", ""))
        for term in tuple(selected_terms)
        if str(getattr(term, "label", ""))
    }
    guarded: list[AnsatzTerm] = []
    dropped_present: list[str] = []
    for term in tuple(candidate_pool_terms):
        label = str(getattr(term, "label", ""))
        if label in dropped_set and label not in selected_labels:
            dropped_present.append(label)
            continue
        guarded.append(term)
    metadata = {
        "schema": LEGAL_SUBSPACE_APPEND_GUARD_SCHEMA_V1,
        "enabled": True,
        "source": "adapt_vqe.adapt_pool_legal_subspace_filter",
        "policy": "drop_static_legal_subspace_offenders_from_ap_append_pool",
        "dropped_label_count_declared": int(len(dropped_set)),
        "dropped_candidate_count": int(len(dropped_present)),
        "candidate_pool_count_before": int(len(tuple(candidate_pool_terms))),
        "candidate_pool_count_after": int(len(guarded)),
        "no_pauli_split_parent_count": int(len(no_pauli_split_set)),
        "no_pauli_split_parent_labels": [
            str(label) for label in tuple(no_pauli_split)
        ],
        "no_pauli_split_parent_labels_sample": [
            str(label) for label in tuple(no_pauli_split)[:12]
        ],
        "selected_label_collision_count": int(
            len(dropped_set.intersection(selected_labels))
        ),
        "dropped_labels_sample": [str(label) for label in dropped_present[:12]],
    }
    return tuple(guarded), metadata


def _scaffold_runtime_input_from_loaded(
    loaded: LoadedRunContext,
    *,
    artifact_json: Path,
    loader_mode: str,
) -> ScaffoldRuntimeInput:
    resolved_problem = resolve_problem_context(
        _problem_request_from_payload(loaded.payload),
        hamiltonian=loaded.replay_context.h_poly,
    )
    candidate_pool_source = _candidate_pool_source(loaded, loader_mode=loader_mode)

    replay_context = loaded.replay_context
    family_info = dict(replay_context.family_info)
    pool_meta = dict(replay_context.pool_meta)
    structure_locked = bool(
        pool_meta.get(
            "structure_locked",
            True if str(loader_mode) == "fixed_scaffold" else False,
        )
    )

    provenance = {
        "artifact_json": str(artifact_json),
        "loader_mode": str(loader_mode),
        "requested_family": family_info.get("requested", None),
        "resolved_family": family_info.get("resolved", None),
        "resolution_source": family_info.get("resolution_source", None),
        "handoff_state_kind": replay_context.handoff_state_kind,
        "provenance_source": replay_context.provenance_source,
    }
    extensions = {
        "legacy_loader_summary": dict(loaded.loader_summary),
        "legacy_family_info": family_info,
        "legacy_pool_meta": pool_meta,
    }
    if isinstance(loaded.payload.get("scaffold_runtime_normalization", None), Mapping):
        extensions["scaffold_runtime_normalization"] = dict(
            loaded.payload["scaffold_runtime_normalization"]
        )

    theta_logical = None
    if replay_context.adapt_theta_logical is not None:
        theta_logical = np.asarray(replay_context.adapt_theta_logical, dtype=float).reshape(-1)

    candidate_pool_terms: tuple[AnsatzTerm, ...]
    if candidate_pool_source.candidate_pool_complete:
        candidate_pool_terms = tuple(replay_context.family_pool)
    elif str(candidate_pool_source.completeness) == "selected_only":
        candidate_pool_terms = tuple(replay_context.replay_terms)
    else:
        candidate_pool_terms = tuple()
    guard_metadata = None
    if candidate_pool_source.candidate_pool_complete:
        candidate_pool_terms, guard_metadata = _apply_legal_subspace_append_guard(
            candidate_pool_terms,
            selected_terms=tuple(replay_context.replay_terms),
            payload=loaded.payload,
        )
        if guard_metadata is not None:
            filter_payload = dict(candidate_pool_source.filter_payload or {})
            filter_payload["legal_subspace_append_guard"] = dict(guard_metadata)
            candidate_pool_source = CandidatePoolSource(
                source_kind=candidate_pool_source.source_kind,
                pool_key=candidate_pool_source.pool_key,
                completeness=candidate_pool_source.completeness,
                pool_build_kwargs=candidate_pool_source.pool_build_kwargs,
                filter_payload=filter_payload,
            )
            extensions["legal_subspace_append_guard"] = dict(guard_metadata)

    return ScaffoldRuntimeInput(
        resolved_problem=resolved_problem,
        psi_ref=np.asarray(replay_context.psi_ref, dtype=complex).reshape(-1),
        psi_initial=np.asarray(loaded.psi_initial, dtype=complex).reshape(-1),
        base_layout=replay_context.base_layout,
        theta_runtime=np.asarray(replay_context.adapt_theta_runtime, dtype=float).reshape(-1),
        theta_logical=theta_logical,
        structure_locked=structure_locked,
        exact_energy=_resolve_exact_energy_from_payload(loaded.payload),
        selected_terms=tuple(replay_context.replay_terms),
        candidate_pool_terms=candidate_pool_terms,
        candidate_pool_source=candidate_pool_source,
        provenance=provenance,
        extensions=extensions,
    )


def load_scaffold_runtime_input_from_payload(
    payload: Mapping[str, Any],
    *,
    artifact_json: str | Path | None = None,
    loader_mode: str | None = None,
    tag: str | None = None,
    generator_family: str = "match_adapt",
    fallback_family: str = "full_meta",
) -> ScaffoldRuntimeInput:
    if not isinstance(payload, Mapping):
        raise ValueError("Scaffold runtime loader expects a JSON-object payload mapping.")
    payload_map = _normalize_static_comparator_wrapper_payload(payload)
    request = _problem_request_from_payload(payload_map)
    if str(request.problem_key) != "hh":
        return _load_generic_runtime_input_from_payload(
            payload_map,
            artifact_json=(
                Path("in_memory_artifact.json") if artifact_json is None else Path(artifact_json)
            ),
            loader_mode=loader_mode,
            generator_family=str(generator_family),
            fallback_family=str(fallback_family),
        )
    artifact_path = (
        Path("in_memory_artifact.json") if artifact_json is None else Path(artifact_json)
    )
    resolved_loader_mode = (
        _infer_legacy_loader_mode(payload_map) if loader_mode is None else str(loader_mode)
    )
    loaded = _load_legacy_loaded_context(
        payload_map,
        artifact_json=artifact_path,
        loader_mode=resolved_loader_mode,
        tag=(f"runtime_loader_{artifact_path.stem}" if tag is None else str(tag)),
        generator_family=str(generator_family),
        fallback_family=str(fallback_family),
    )
    return _scaffold_runtime_input_from_loaded(
        loaded,
        artifact_json=artifact_path,
        loader_mode=resolved_loader_mode,
    )


def load_scaffold_runtime_input(
    artifact_json: str | Path,
    *,
    loader_mode: str | None = None,
    tag: str | None = None,
    generator_family: str = "match_adapt",
    fallback_family: str = "full_meta",
) -> ScaffoldRuntimeInput:
    artifact_path = Path(artifact_json)
    payload = _read_json(artifact_path)
    return load_scaffold_runtime_input_from_payload(
        payload,
        artifact_json=artifact_path,
        loader_mode=loader_mode,
        tag=tag,
        generator_family=generator_family,
        fallback_family=fallback_family,
    )


__all__ = [
    "load_scaffold_runtime_input",
    "load_scaffold_runtime_input_from_payload",
]
