#!/usr/bin/env python3
"""Build diagnostic-only Paper-II AP/Qiskit smoke inputs.

This derives a short weak-weak HH smoke case from an existing seed-track case
manifest.  It preserves the static seed artifact and lock metadata, but marks
the derived case as diagnostic-only and shortens only the time grid.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
INPUT = Path("chtc/generic_time_dynamics_table/input")
SOURCE_CASE_MANIFEST = INPUT / "paper_ii_hh_seed_tracks_cases_v1.json"
CASE_MANIFEST = INPUT / "paper_ii_diagnostic_smoke_cases_v1.json"
RECORDS_TSV = INPUT / "paper_ii_diagnostic_smoke_records_v1.tsv"
RECORD_IDS = INPUT / "paper_ii_diagnostic_smoke_record_ids_v1.txt"
SMOKE_IDS = INPUT / "paper_ii_diagnostic_smoke_smoke_record_ids_v1.txt"
PREFIX_SEED_DIR = INPUT / "seed_artifacts_paper_ii_diagnostic_smoke_v1"

DIAGNOSTIC_PROFILE_ID = "paper_ii_qiskit_ap_hh_small_prefix_v1"
DEFAULT_SEED_PREFIX_LOGICAL_BLOCKS = 6
DEFAULT_DRIVE_ALIGN_PREFIX_SEED = True
DEFAULT_ALGORITHMS: tuple[str, ...] = (
    "dyn_qiskit_trotter_qrte",
    "dyn_qiskit_pvqd",
    "dyn_qiskit_varqrte",
    "dyn_controller_full",
)
DEFAULT_RUNTIME_BUDGETS: dict[str, int] = {
    "dyn_qiskit_trotter_qrte": 120,
    "dyn_qiskit_pvqd": 180,
    "dyn_qiskit_varqrte": 60,
    "dyn_controller_full": 300,
}


def _repo_path(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _read_json(root: Path, path: str | Path) -> Any:
    return json.loads(_repo_path(root, path).read_text(encoding="utf-8"))


def _write_json(root: Path, path: str | Path, payload: Any) -> None:
    out = _repo_path(root, path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ids(root: Path, path: str | Path, ids: Sequence[str]) -> None:
    out = _repo_path(root, path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def _manifest_cases(payload: Any) -> list[dict[str, Any]]:
    raw_cases = payload.get("cases", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise ValueError("source case manifest must contain a cases list")
    return [dict(item) for item in raw_cases if isinstance(item, dict)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_manifest_relative_path(root: Path, manifest_path: Path | str, artifact_path: str | Path) -> Path:
    artifact = Path(artifact_path)
    if artifact.is_absolute():
        return artifact
    manifest = _repo_path(root, manifest_path)
    return manifest.parent / artifact


def _truncate_seed_payload_for_diagnostic_prefix(
    payload: dict[str, Any],
    *,
    prefix_logical_blocks: int,
    source_seed_artifact: str,
    source_seed_artifact_sha256: str,
) -> dict[str, Any]:
    """Return a diagnostic-only prefix seed export preserving loader invariants."""

    prefix = int(prefix_logical_blocks)
    if prefix <= 0:
        raise ValueError("diagnostic seed prefix must keep at least one logical block")
    out = json.loads(json.dumps(payload))
    adapt = out.get("adapt_vqe", {})
    if not isinstance(adapt, dict):
        raise ValueError("seed payload missing adapt_vqe block")
    parameterization = adapt.get("parameterization", {})
    if not isinstance(parameterization, dict):
        raise ValueError("seed payload missing adapt_vqe.parameterization block")
    blocks = parameterization.get("blocks", [])
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("seed payload missing parameterization.blocks")
    if prefix > len(blocks):
        raise ValueError(
            f"diagnostic seed prefix {prefix} exceeds available logical blocks {len(blocks)}"
        )
    prefix_blocks = json.loads(json.dumps(blocks[:prefix]))
    runtime_count = int(
        sum(int(block.get("runtime_count", len(block.get("runtime_terms_exyz", [])))) for block in prefix_blocks)
    )
    expected_start = 0
    for logical_index, block in enumerate(prefix_blocks):
        runtime_terms = block.get("runtime_terms_exyz", [])
        runtime_block_count = int(block.get("runtime_count", len(runtime_terms)))
        block["logical_index"] = int(logical_index)
        block["runtime_start"] = int(expected_start)
        block["runtime_count"] = int(runtime_block_count)
        expected_start += int(runtime_block_count)
    if int(expected_start) != int(runtime_count):
        raise ValueError("internal diagnostic prefix runtime count mismatch")

    operators = list(adapt.get("operators", []))
    logical_theta = list(adapt.get("logical_optimal_point", []))
    runtime_theta = list(adapt.get("optimal_point", []))
    if len(operators) < prefix:
        raise ValueError("seed payload operators shorter than requested diagnostic prefix")
    if len(logical_theta) < prefix:
        raise ValueError("seed payload logical_optimal_point shorter than requested diagnostic prefix")
    if len(runtime_theta) < runtime_count:
        raise ValueError("seed payload optimal_point shorter than requested diagnostic prefix runtime count")

    adapt["operators"] = operators[:prefix]
    adapt["logical_optimal_point"] = logical_theta[:prefix]
    adapt["optimal_point"] = runtime_theta[:runtime_count]
    adapt["logical_num_parameters"] = int(prefix)
    adapt["num_parameters"] = int(runtime_count)
    adapt["energy"] = None
    adapt["exact_energy_from_final_state"] = None
    adapt["abs_delta_e"] = None
    adapt["diagnostic_prefix_truncated_from_logical_blocks"] = int(len(blocks))
    adapt["diagnostic_prefix_truncated_from_runtime_parameters"] = int(
        parameterization.get("runtime_parameter_count", len(runtime_theta))
    )
    adapt["diagnostic_prefix_logical_blocks"] = int(prefix)
    adapt["diagnostic_prefix_runtime_parameters"] = int(runtime_count)
    parameterization["blocks"] = prefix_blocks
    parameterization["logical_operator_count"] = int(prefix)
    parameterization["runtime_parameter_count"] = int(runtime_count)

    segment = out.get("adapt_segment", {})
    if isinstance(segment, dict):
        segment["diagnostic_prefix_truncation"] = True
        segment["source_final_depth"] = segment.get("final_depth")
        segment["source_final_runtime_parameter_count"] = segment.get("final_runtime_parameter_count")
        segment["final_depth"] = int(prefix)
        segment["final_runtime_parameter_count"] = int(runtime_count)
        segment["new_admission_records"] = int(prefix)
        segment["stop_reason"] = "diagnostic_prefix_truncation"

    for state_key in ("initial_state",):
        # The full-seed prepared state is no longer valid after truncation.
        # The runtime loader will reconstruct the prefix state from the HF
        # input, selected prefix layout, and prefix theta.
        out.pop(state_key, None)

    prefix_meta = {
        "schema": "paper_ii_diagnostic_seed_prefix_v1",
        "diagnostic_only_not_paper_evidence": True,
        "source_seed_artifact_json": str(source_seed_artifact),
        "source_seed_artifact_sha256": str(source_seed_artifact_sha256),
        "source_logical_operator_count": int(len(blocks)),
        "source_runtime_parameter_count": int(parameterization.get("diagnostic_source_runtime_parameter_count", 0) or adapt.get("diagnostic_prefix_truncated_from_runtime_parameters", 0)),
        "prefix_logical_operator_count": int(prefix),
        "prefix_runtime_parameter_count": int(runtime_count),
    }
    out["paper_ii_diagnostic_seed_prefix"] = dict(prefix_meta)
    seed_export = out.get("paper_ii_static_seed_export", {})
    if isinstance(seed_export, dict):
        seed_export["diagnostic_prefix_truncation"] = True
        seed_export["diagnostic_only_not_paper_evidence"] = True
        seed_export["source_seed_artifact_json"] = str(source_seed_artifact)
        seed_export["source_seed_artifact_sha256"] = str(source_seed_artifact_sha256)
        seed_export["source_logical_operator_count"] = seed_export.get("logical_operator_count")
        seed_export["source_runtime_parameter_count"] = seed_export.get("runtime_parameter_count")
        seed_export["logical_operator_count"] = int(prefix)
        seed_export["selected_term_count"] = int(prefix)
        seed_export["runtime_parameter_count"] = int(runtime_count)
        seed_export["static_parameter_count"] = int(runtime_count)
        seed_export["static_abs_delta_e"] = None
        seed_export["static_seed_display_label"] = f"SNAKE prefix {prefix}"
    lock = out.get("paper_ii_seed_lock", {})
    if isinstance(lock, dict):
        lock["diagnostic_prefix_truncation"] = True
        lock["diagnostic_only_not_paper_evidence"] = True
        lock["source_seed_artifact_json"] = str(source_seed_artifact)
        lock["source_seed_artifact_sha256"] = str(source_seed_artifact_sha256)
        lock["source_logical_operator_count"] = lock.get("runtime_logical_operator_count")
        lock["source_runtime_parameter_count"] = lock.get("runtime_parameter_count")
        lock["runtime_logical_operator_count"] = int(prefix)
        lock["runtime_selected_term_count"] = int(prefix)
        lock["runtime_parameter_count"] = int(runtime_count)
        lock["static_parameter_count"] = int(runtime_count)
        lock["static_abs_delta_e"] = None
        lock["static_seed_display_label"] = f"SNAKE prefix {prefix}"
    return out


def _parse_float_tuple(raw: Any) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return None
        parts = [part for item in text.split(",") for part in item.split()]
        return tuple(float(part) for part in parts if str(part).strip())
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return tuple(float(item) for item in raw)
    return (float(raw),)


def _num_qubits_from_state_dim(dim: int) -> int:
    if int(dim) <= 0:
        raise ValueError("state dimension must be positive")
    nq = int(round(math.log2(int(dim))))
    if (1 << nq) != int(dim):
        raise ValueError(f"state dimension is not a power of two: {dim}")
    return int(nq)


def _case_drive_payload(source_case: dict[str, Any]) -> dict[str, Any]:
    metadata = source_case.get("metadata", {}) if isinstance(source_case.get("metadata"), dict) else {}
    drive = metadata.get("drive", {}) if isinstance(metadata.get("drive"), dict) else {}
    return dict(drive)


def _maybe_augment_prefix_seed_with_drive_aligned_density(
    root: Path,
    *,
    artifact_relpath: Path,
    payload: dict[str, Any],
    source_case: dict[str, Any],
    prefix_logical_blocks: int,
) -> dict[str, Any]:
    """Mirror AP McLachlan's driven-HH zero tangent in the diagnostic seed."""

    drive = _case_drive_payload(source_case)
    if not bool(drive.get("enable_drive", False)):
        return payload
    drive_a = float(drive.get("A", 0.0))
    if abs(drive_a) <= 1.0e-12:
        return payload

    # Heavy HH imports are kept local so this builder remains cheap for
    # non-driven/non-augmented diagnostic manifests.
    import sys

    root_text = str(Path(root).resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input
    from pipelines.time_dynamics.ap_mclachlan.controller import (
        ControllerDriveConfig,
        _augment_replay_context_with_drive_aligned_density,
    )
    from pipelines.time_dynamics.runners.generic_from_adapt_artifact import (
        _replay_context_from_runtime_input,
    )
    from src.quantum.ansatz_parameterization import serialize_layout

    artifact_path = _repo_path(root, artifact_relpath)
    runtime_input = load_scaffold_runtime_input(
        artifact_path,
        loader_mode="replay_family",
        tag="paper_ii_diagnostic_drive_aligned_prefix_seed",
        generator_family="match_adapt",
        fallback_family="full_meta",
    )
    request = runtime_input.resolved_problem.request
    replay_context = _replay_context_from_runtime_input(
        runtime_input,
        append_pool_family="match_replay",
    )
    drive_config = ControllerDriveConfig(
        enabled=True,
        n_sites=int(request.num_sites),
        ordering=str(request.ordering),
        drive_A=float(drive_a),
        drive_omega=float(drive.get("omega", 1.0)),
        drive_tbar=float(drive.get("tbar", 1.0)),
        drive_phi=float(drive.get("phi", 0.0)),
        drive_pattern=str(drive.get("pattern", "staggered")),
        drive_custom_weights=_parse_float_tuple(drive.get("custom_weights")),
        drive_include_identity=bool(drive.get("include_identity", False)),
        drive_time_sampling=str(drive.get("time_sampling", "midpoint")),
        drive_t0=float(drive.get("t0", 0.0)),
        exact_steps_multiplier=1,
    )
    num_qubits = _num_qubits_from_state_dim(len(runtime_input.psi_ref))
    augmented_context, augmented_theta, active, label = _augment_replay_context_with_drive_aligned_density(
        replay_context,
        best_theta=runtime_input.theta_runtime,
        drive_config=drive_config,
        num_qubits=int(num_qubits),
    )
    if not bool(active):
        return payload

    out = json.loads(json.dumps(payload))
    layout_payload = serialize_layout(augmented_context.base_layout)
    logical_count = int(layout_payload["logical_operator_count"])
    runtime_count = int(layout_payload["runtime_parameter_count"])
    runtime_theta = [float(x) for x in list(augmented_theta)]
    logical_theta = [
        float(x) for x in list(augmented_context.adapt_theta_logical)
    ]
    if len(runtime_theta) != runtime_count:
        raise ValueError(
            "drive-aligned diagnostic seed theta length mismatch: "
            f"{len(runtime_theta)} != {runtime_count}"
        )
    if len(logical_theta) != logical_count:
        raise ValueError(
            "drive-aligned diagnostic seed logical theta length mismatch: "
            f"{len(logical_theta)} != {logical_count}"
        )
    term_labels = [str(term.label) for term in augmented_context.replay_terms]
    if len(term_labels) != logical_count:
        raise ValueError("drive-aligned diagnostic seed selected-term count mismatch")
    drive_block = dict(layout_payload["blocks"][-1])
    if str(drive_block.get("candidate_label")) != str(label):
        raise ValueError("drive-aligned diagnostic seed final block label mismatch")
    serialized_drive_terms = [
        dict(item) for item in drive_block.get("runtime_terms_exyz", [])
    ]

    adapt = out.setdefault("adapt_vqe", {})
    if not isinstance(adapt, dict):
        raise ValueError("seed payload missing adapt_vqe block")
    adapt["operators"] = term_labels
    adapt["logical_optimal_point"] = logical_theta
    adapt["optimal_point"] = runtime_theta
    adapt["logical_num_parameters"] = int(logical_count)
    adapt["num_parameters"] = int(runtime_count)
    adapt["parameterization"] = layout_payload
    adapt["diagnostic_drive_aligned_density_augmentation"] = True
    adapt["diagnostic_drive_aligned_density_label"] = str(label)
    adapt["diagnostic_seed_prefix_logical_blocks_before_drive_alignment"] = int(
        prefix_logical_blocks
    )
    adapt["diagnostic_seed_case_label"] = f"prefix{int(prefix_logical_blocks)}_drivealigned"
    drive_generator_metadata = {
        "candidate_label": str(label),
        "family_id": "diagnostic_drive_aligned_density",
        "source": "paper_ii_diagnostic_smoke_seed_alignment",
        "compile_metadata": {
            "serialized_terms_exyz": serialized_drive_terms,
            "num_polynomial_terms": int(len(serialized_drive_terms)),
            "signature_size": int(len(serialized_drive_terms)),
            "support_size": int(
                max(
                    (
                        sum(1 for ch in str(term.get("pauli_exyz", "")) if ch.lower() != "e")
                        for term in serialized_drive_terms
                    ),
                    default=0,
                )
            ),
            "has_boson_support": False,
            "has_fermion_support": True,
        },
    }
    continuation = adapt.get("continuation", {})
    if not isinstance(continuation, dict):
        continuation = {}
    selected_meta = continuation.get("selected_generator_metadata", [])
    if not isinstance(selected_meta, list):
        selected_meta = []
    selected_meta = [dict(item) for item in selected_meta if isinstance(item, dict)]
    selected_meta = [
        item for item in selected_meta if str(item.get("candidate_label", "")) != str(label)
    ]
    selected_meta.append(drive_generator_metadata)
    continuation["selected_generator_metadata"] = selected_meta
    adapt["continuation"] = continuation

    segment = out.get("adapt_segment", {})
    if isinstance(segment, dict):
        segment["diagnostic_drive_aligned_density_augmentation"] = True
        segment["diagnostic_drive_aligned_density_label"] = str(label)
        segment["diagnostic_seed_prefix_logical_blocks_before_drive_alignment"] = int(
            prefix_logical_blocks
        )
        segment["final_depth"] = int(logical_count)
        segment["final_runtime_parameter_count"] = int(runtime_count)
        segment["new_admission_records"] = int(logical_count)

    display_label = f"SNAKE prefix {int(prefix_logical_blocks)} + drive-aligned density"
    prefix_meta = out.get("paper_ii_diagnostic_seed_prefix", {})
    if isinstance(prefix_meta, dict):
        prefix_meta["drive_aligned_density_augmented"] = True
        prefix_meta["drive_aligned_density_label"] = str(label)
        prefix_meta["prefix_logical_operator_count_before_drive_alignment"] = int(
            prefix_logical_blocks
        )
        prefix_meta["runtime_logical_operator_count"] = int(logical_count)
        prefix_meta["runtime_selected_term_count"] = int(logical_count)
        prefix_meta["runtime_parameter_count"] = int(runtime_count)
        prefix_meta["diagnostic_seed_case_label"] = f"prefix{int(prefix_logical_blocks)}_drivealigned"
        prefix_meta["static_seed_display_label"] = display_label
        out["paper_ii_diagnostic_seed_prefix"] = prefix_meta
    for key in ("paper_ii_static_seed_export", "paper_ii_seed_lock"):
        block = out.get(key, {})
        if not isinstance(block, dict):
            continue
        block["diagnostic_drive_aligned_density_augmentation"] = True
        block["diagnostic_drive_aligned_density_label"] = str(label)
        block["diagnostic_seed_prefix_logical_blocks_before_drive_alignment"] = int(
            prefix_logical_blocks
        )
        block["logical_operator_count"] = int(logical_count)
        block["selected_term_count"] = int(logical_count)
        block["runtime_logical_operator_count"] = int(logical_count)
        block["runtime_selected_term_count"] = int(logical_count)
        block["runtime_parameter_count"] = int(runtime_count)
        block["static_parameter_count"] = int(runtime_count)
        block["static_seed_display_label"] = display_label
    for state_key in ("initial_state",):
        out.pop(state_key, None)
    return out


def _write_diagnostic_prefix_seed(
    root: Path,
    *,
    source_case_manifest: Path | str,
    source_case: dict[str, Any],
    prefix_logical_blocks: int,
) -> tuple[str, str, dict[str, Any]]:
    source_artifact = _resolve_manifest_relative_path(
        root,
        source_case_manifest,
        source_case["artifact_json"],
    )
    if not source_artifact.exists():
        raise FileNotFoundError(f"source seed artifact not found: {source_artifact}")
    source_sha = _sha256(source_artifact)
    payload = _read_json(root, source_artifact)
    prefix_payload = _truncate_seed_payload_for_diagnostic_prefix(
        payload,
        prefix_logical_blocks=int(prefix_logical_blocks),
        source_seed_artifact=str(source_case.get("artifact_json")),
        source_seed_artifact_sha256=str(source_sha),
    )
    metadata = source_case.get("metadata", {}) if isinstance(source_case.get("metadata"), dict) else {}
    seed_lock = metadata.get("seed_lock", {}) if isinstance(metadata.get("seed_lock"), dict) else {}
    regime = str(seed_lock.get("hh_regime_id", "hh"))
    track = str(seed_lock.get("seed_track", "seed"))
    drive = _case_drive_payload(source_case)
    suffix = (
        f"prefix{int(prefix_logical_blocks)}_drivealigned"
        if bool(drive.get("enable_drive", False))
        and abs(float(drive.get("A", 0.0))) > 1.0e-12
        and bool(DEFAULT_DRIVE_ALIGN_PREFIX_SEED)
        else f"prefix{int(prefix_logical_blocks)}"
    )
    out_rel = PREFIX_SEED_DIR / f"hh_{regime}_{track}_{suffix}_seed.json"
    _write_json(root, out_rel, prefix_payload)
    if bool(DEFAULT_DRIVE_ALIGN_PREFIX_SEED):
        prefix_payload = _maybe_augment_prefix_seed_with_drive_aligned_density(
            root,
            artifact_relpath=out_rel,
            payload=prefix_payload,
            source_case=source_case,
            prefix_logical_blocks=int(prefix_logical_blocks),
        )
        _write_json(root, out_rel, prefix_payload)
    out_abs = _repo_path(root, out_rel)
    return str(out_rel.relative_to(INPUT)), _sha256(out_abs), prefix_payload


def _drive_label(case: dict[str, Any]) -> str:
    metadata = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
    drive = metadata.get("drive", {}) if isinstance(metadata.get("drive"), dict) else {}
    amp = drive.get("A")
    if amp is None:
        return "Aunknown"
    text = ("%g" % float(amp)).replace(".", "p").replace("-", "m")
    return f"A{text}"


def _matches_case(
    case: dict[str, Any],
    *,
    family: str,
    hh_regime_id: str,
    seed_track: str,
    drive_label: str,
) -> bool:
    metadata = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
    seed_lock = metadata.get("seed_lock", {}) if isinstance(metadata.get("seed_lock"), dict) else {}
    return (
        str(case.get("family")) == str(family)
        and str(seed_lock.get("hh_regime_id")) == str(hh_regime_id)
        and str(seed_lock.get("seed_track")).lower() == str(seed_track).lower()
        and _drive_label(case) == str(drive_label)
    )


def _safe_id(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "p")
        .replace("-", "_")
    )


def _record_id(*, case_id: str, algorithm_id: str) -> str:
    return f"paper_ii_diag_smoke_v1_{_safe_id(case_id)}_{_safe_id(algorithm_id)}"


def _derived_case(
    source_case: dict[str, Any],
    *,
    source_manifest: Path,
    source_manifest_sha256: str,
    seed_artifact_json: str | None,
    seed_artifact_sha256: str | None,
    seed_prefix_logical_blocks: int | None,
    seed_prefix_runtime_parameters: int | None,
    seed_runtime_logical_operator_count: int | None,
    seed_runtime_selected_term_count: int | None,
    seed_runtime_parameter_count: int | None,
    seed_case_label: str | None,
    seed_display_label: str | None,
    t_final: float,
    num_times: int,
) -> dict[str, Any]:
    case = json.loads(json.dumps(source_case))
    metadata = case.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        case["metadata"] = metadata
    seed_lock = metadata.get("seed_lock", {}) if isinstance(metadata.get("seed_lock"), dict) else {}
    old_case_id = str(case["case_id"])
    old_group = str(
        metadata.get(
            "same_seed_comparator_group_id",
            seed_lock.get("same_seed_comparator_group_id", old_case_id),
        )
    )
    regime = str(seed_lock.get("hh_regime_id", "hh"))
    track = str(seed_lock.get("seed_track", "seed"))
    drive = _drive_label(case)
    t_label = ("%g" % float(t_final)).replace(".", "p").replace("-", "m")
    seed_label_fragment = (
        f"_{_safe_id(str(seed_case_label))}"
        if seed_case_label not in {None, ""}
        else (
            f"_prefix{int(seed_prefix_logical_blocks)}"
            if seed_prefix_logical_blocks is not None
            else ""
        )
    )
    runtime_logical_count = (
        int(seed_runtime_logical_operator_count)
        if seed_runtime_logical_operator_count is not None
        else seed_prefix_logical_blocks
    )
    runtime_selected_count = (
        int(seed_runtime_selected_term_count)
        if seed_runtime_selected_term_count is not None
        else runtime_logical_count
    )
    runtime_parameter_count = (
        int(seed_runtime_parameter_count)
        if seed_runtime_parameter_count is not None
        else seed_prefix_runtime_parameters
    )
    display_label = (
        str(seed_display_label)
        if seed_display_label not in {None, ""}
        else (
            f"SNAKE prefix {seed_prefix_logical_blocks}"
            if seed_prefix_logical_blocks is not None
            else str(seed_lock.get("static_seed_display_label", "seed"))
        )
    )
    new_case_id = f"diag_hh_{regime}_{track}{seed_label_fragment}_{drive}_t{t_label}_n{int(num_times)}_qiskit_ap_v1"
    new_group = f"diag_hh_{regime}_{track}{seed_label_fragment}_{drive}_t{t_label}_n{int(num_times)}_same_seed_v1"
    case["case_id"] = new_case_id
    case["description"] = (
        f"Diagnostic-only short Paper-II HH {track} seed matrix for AP McLachlan "
        "plus Qiskit-community comparators."
    )
    case["t_final"] = float(t_final)
    case["num_times"] = int(num_times)
    if seed_artifact_json is not None:
        case["artifact_json"] = str(seed_artifact_json)
    metadata.update(
        {
            "diagnostic_only_not_paper_evidence": True,
            "smoke_only_not_paper_evidence": True,
            "diagnostic_not_paper_facing": True,
            "diagnostic_short_grid": True,
            "diagnostic_profile_id": DIAGNOSTIC_PROFILE_ID,
            "paper_ii_calibration_gate_status": "not_run_diagnostic_only",
            "diagnostic_source_case_id": old_case_id,
            "diagnostic_source_same_seed_comparator_group_id": old_group,
            "diagnostic_source_case_manifest": str(source_manifest),
            "diagnostic_source_case_manifest_sha256": source_manifest_sha256,
            "same_seed_comparator_group_id": new_group,
            "diagnostic_seed_prefix_logical_blocks": seed_prefix_logical_blocks,
            "diagnostic_seed_prefix_runtime_parameters": seed_prefix_runtime_parameters,
            "diagnostic_seed_runtime_logical_operator_count": runtime_logical_count,
            "diagnostic_seed_runtime_selected_term_count": runtime_selected_count,
            "diagnostic_seed_runtime_parameter_count": runtime_parameter_count,
            "diagnostic_seed_case_label": seed_case_label,
            "diagnostic_seed_display_label": display_label,
            "diagnostic_seed_artifact_json": seed_artifact_json,
            "diagnostic_seed_artifact_sha256": seed_artifact_sha256,
            "diagnostic_runtime_budgets": {
                "schema": "paper_ii_diagnostic_runtime_budgets_v1",
                "enforced": True,
                "algorithm_timeout_seconds": dict(DEFAULT_RUNTIME_BUDGETS),
            },
            "qiskit_community_dynamics": {
                "qubit_cap": 12,
                "pvqd_optimizer_maxiter": 24,
                "varqrte_max_runtime_parameters": 64,
                "varqrte_max_qgt_entries": 4096,
                "trotter_num_timesteps_per_interval": 1,
                "varqrte_num_timesteps_per_interval": 1,
            },
        }
    )
    if isinstance(seed_lock, dict):
        seed_lock["same_seed_comparator_group_id"] = new_group
        if seed_artifact_json is not None:
            seed_lock["diagnostic_prefix_truncation"] = True
            seed_lock["diagnostic_only_not_paper_evidence"] = True
            seed_lock["static_seed_artifact_json"] = str(seed_artifact_json)
            seed_lock["static_seed_artifact_sha256"] = str(seed_artifact_sha256)
            seed_lock["normalized_seed_artifact_json"] = str(seed_artifact_json)
            seed_lock["normalized_seed_artifact_sha256"] = str(seed_artifact_sha256)
            seed_lock["seed_artifact_sha256"] = str(seed_artifact_sha256)
            seed_lock["runtime_logical_operator_count"] = runtime_logical_count
            seed_lock["runtime_selected_term_count"] = runtime_selected_count
            seed_lock["runtime_parameter_count"] = runtime_parameter_count
            seed_lock["static_parameter_count"] = runtime_parameter_count
            seed_lock["static_abs_delta_e"] = None
            seed_lock["static_seed_display_label"] = display_label
        metadata["seed_lock"] = seed_lock
    return case


def build_inputs(
    *,
    root: Path | str = ROOT,
    source_case_manifest: Path | str = SOURCE_CASE_MANIFEST,
    family: str = "hh",
    hh_regime_id: str = "weak_weak",
    seed_track: str = "snake",
    drive_label: str = "A0p2",
    t_final: float = 0.1,
    num_times: int = 3,
    seed_prefix_logical_blocks: int | None = DEFAULT_SEED_PREFIX_LOGICAL_BLOCKS,
    algorithms: Iterable[str] = DEFAULT_ALGORITHMS,
) -> dict[str, Any]:
    root_path = Path(root)
    source_path = _repo_path(root_path, source_case_manifest)
    source_payload = _read_json(root_path, source_case_manifest)
    source_cases = _manifest_cases(source_payload)
    matches = [
        case
        for case in source_cases
        if _matches_case(
            case,
            family=family,
            hh_regime_id=hh_regime_id,
            seed_track=seed_track,
            drive_label=drive_label,
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one source case for diagnostic smoke selection; "
            f"found {len(matches)} for family={family!r}, hh_regime_id={hh_regime_id!r}, "
            f"seed_track={seed_track!r}, drive_label={drive_label!r}"
        )
    source_sha = _sha256(source_path)
    seed_artifact_json: str | None = None
    seed_artifact_sha: str | None = None
    seed_prefix_runtime_parameters: int | None = None
    seed_runtime_logical_operator_count: int | None = None
    seed_runtime_selected_term_count: int | None = None
    seed_runtime_parameter_count: int | None = None
    seed_case_label: str | None = None
    seed_display_label: str | None = None
    if seed_prefix_logical_blocks is not None:
        seed_artifact_json, seed_artifact_sha, seed_payload = _write_diagnostic_prefix_seed(
            root_path,
            source_case_manifest=source_case_manifest,
            source_case=matches[0],
            prefix_logical_blocks=int(seed_prefix_logical_blocks),
        )
        prefix_meta = (
            seed_payload.get("paper_ii_diagnostic_seed_prefix", {})
            if isinstance(seed_payload.get("paper_ii_diagnostic_seed_prefix"), dict)
            else {}
        )
        seed_prefix_runtime_parameters = int(prefix_meta.get("prefix_runtime_parameter_count", 0))
        seed_runtime_logical_operator_count = int(
            prefix_meta.get(
                "runtime_logical_operator_count",
                prefix_meta.get("prefix_logical_operator_count", seed_prefix_logical_blocks),
            )
        )
        seed_runtime_selected_term_count = int(
            prefix_meta.get(
                "runtime_selected_term_count",
                seed_runtime_logical_operator_count,
            )
        )
        seed_runtime_parameter_count = int(
            prefix_meta.get(
                "runtime_parameter_count",
                prefix_meta.get("prefix_runtime_parameter_count", 0),
            )
        )
        seed_case_label = str(
            prefix_meta.get(
                "diagnostic_seed_case_label",
                f"prefix{int(seed_prefix_logical_blocks)}",
            )
        )
        seed_display_label = str(
            prefix_meta.get(
                "static_seed_display_label",
                f"SNAKE prefix {int(seed_prefix_logical_blocks)}",
            )
        )
    derived = _derived_case(
        matches[0],
        source_manifest=Path(source_case_manifest),
        source_manifest_sha256=source_sha,
        seed_artifact_json=seed_artifact_json,
        seed_artifact_sha256=seed_artifact_sha,
        seed_prefix_logical_blocks=seed_prefix_logical_blocks,
        seed_prefix_runtime_parameters=seed_prefix_runtime_parameters,
        seed_runtime_logical_operator_count=seed_runtime_logical_operator_count,
        seed_runtime_selected_term_count=seed_runtime_selected_term_count,
        seed_runtime_parameter_count=seed_runtime_parameter_count,
        seed_case_label=seed_case_label,
        seed_display_label=seed_display_label,
        t_final=float(t_final),
        num_times=int(num_times),
    )
    algorithm_ids = tuple(str(algorithm) for algorithm in algorithms)
    records: list[dict[str, str]] = []
    for algorithm_id in algorithm_ids:
        records.append(
            {
                "record_id": _record_id(case_id=str(derived["case_id"]), algorithm_id=algorithm_id),
                "kind": "benchmark",
                "family": str(derived["family"]),
                "tuning_class": str(derived.get("tuning_class", "")),
                "case_id": str(derived["case_id"]),
                "algorithm_id": algorithm_id,
                "variants": "diagnostic_smoke",
                "case_manifest": str(CASE_MANIFEST),
                "visible_table_method": "0",
                "diagnostic_only_not_paper_evidence": "1",
            }
        )
    manifest = {
        "schema": "paper_ii_diagnostic_smoke_cases_v1",
        "manifest_id": "paper_ii_diagnostic_smoke_cases_v1",
        "diagnostic_profile_id": DIAGNOSTIC_PROFILE_ID,
        "paper_facing": False,
        "paper_ii_calibration_gate_status": "not_run_diagnostic_only",
        "source_case_manifest": str(source_case_manifest),
        "source_case_manifest_sha256": source_sha,
        "source_case_id": str(matches[0]["case_id"]),
        "seed_prefix_logical_blocks": seed_prefix_logical_blocks,
        "seed_prefix_runtime_parameters": seed_prefix_runtime_parameters,
        "diagnostic_seed_artifact_json": seed_artifact_json,
        "diagnostic_seed_artifact_sha256": seed_artifact_sha,
        "case_count": 1,
        "record_count": len(records),
        "algorithms": list(algorithm_ids),
        "cases": [derived],
    }
    _write_json(root_path, CASE_MANIFEST, manifest)

    records_path = _repo_path(root_path, RECORDS_TSV)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "kind",
        "family",
        "tuning_class",
        "case_id",
        "algorithm_id",
        "variants",
        "case_manifest",
        "visible_table_method",
        "diagnostic_only_not_paper_evidence",
    ]
    with records_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    ids = [row["record_id"] for row in records]
    _write_ids(root_path, RECORD_IDS, ids)
    _write_ids(root_path, SMOKE_IDS, ids)
    return manifest


if __name__ == "__main__":
    result = build_inputs()
    print(
        json.dumps(
            {
                "case_count": result["case_count"],
                "record_count": result["record_count"],
                "case_manifest": str(CASE_MANIFEST),
                "records_tsv": str(RECORDS_TSV),
            },
            indent=2,
            sort_keys=True,
        )
    )
