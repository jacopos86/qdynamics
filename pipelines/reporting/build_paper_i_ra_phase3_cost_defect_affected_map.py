#!/usr/bin/env python3
"""Build the immutable-evidence map for the Paper-I RA Phase-III cost defect.

The builder deliberately reads only small protocol, manifest, and receipt JSON
files.  It never opens result/checkpoint/ledger payloads and it never rewrites a
historical artifact.  A record is called executed only when a passed execution
artifact binds the exact canonical protocol digest.  That classification is
independent of the stronger, currently empty, class "accepted trajectory proven
to change under corrected rescoring".
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT_RELATIVE = Path("chtc/paper_i_ra_adapt_repair_20260727")
LOCAL_RUN_ROOT_RELATIVE = Path("output/local_runs")
DEFAULT_OUTPUT_RELATIVE = Path(
    "agent_guidance/static-adapt/"
    "paper-i-ra-phase3-cost-defect-affected-map-20260816.json"
)

SCHEMA = "paper_i_ra_phase3_cost_defect_affected_map_v1"
INVENTORY_AS_OF = "2026-08-16"
MAX_SMALL_JSON_BYTES = 8 * 1024 * 1024
SIGNED_MODE = "zero_centered_signed_arctan_v1"
SCORING_MEMBER = "pipelines/scaffold/hh_continuation_scoring.py"

# Every affected package in this inventory locks this byte-identical consumer.
# Its historical Phase-III gate recognized only the older symmetric enum, so it
# silently replaced zero-centered signed factors with 1.0.
KNOWN_DEFECTIVE_SCORING_SHA256S = frozenset(
    {"fa6ab8a2700204d18f2f6b5550221355574042d3b06760bab75d50fd0b191e62"}
)

# Sealed as-of inventory.  This explicit boundary prevents a future corrected
# route that legitimately uses the same normalization mode from being swept
# into the historical defect map merely because it is materialized below the
# same campaign root.
AFFECTED_CONTAINER_IDS = (
    "paper_i_l3_weak_holstein_page12_append6_r50_20260812_v1_chtc",
    "paper_i_l3_weak_holstein_page12_append6_r50_20260812_v2_chtc",
    "paper_i_l3_weak_holstein_page12_append6_r50_20260812_v3_chtc",
    "paper_i_l3_weak_holstein_page12_append6_r50_20260812_v4_chtc",
    "paper_i_page10_strong_holstein_r70_accepted_continuations_20260809_v1_chtc",
    "paper_i_page10_strong_holstein_r70_accepted_continuations_20260809_v2_chtc",
    "paper_i_page12_matched_singleton12_r50_20260815_v1_local",
    "paper_i_page12_strong_holstein_r70_accepted_continuations_20260810_v1_local",
    "paper_i_page12_strong_holstein_r70_accepted_continuations_20260810_v2_local",
    "paper_i_page12_strong_holstein_r70_accepted_continuations_20260811_v1_chtc",
    "paper_i_page12_strong_holstein_r70_accepted_continuations_20260812_v2_chtc",
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_phase23_no_lanes_beam3x2_metric_prune_cap24_tau1em4_strong_weak_r30_20260811_v1_chtc",
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_phase23_no_lanes_beam3x2_metric_prune_cap24_tau1em4_strong_weak_r30_20260811_v2_chtc",
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_phase23_no_lanes_beam3x2_metric_prune_cap24_tau1em4_strong_weak_r30_20260811_v3_chtc",
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_phase23_no_lanes_cap24_tau1em4_r50_20260807_v1_chtc",
    "paper_i_ra_adapt_l3_intermediate_weak_page12_r50_20260811_v1_chtc",
    "paper_i_ra_adapt_l3_intermediate_weak_page12_r50_20260811_v2_chtc",
    "paper_i_ra_adapt_macro_gradient_phase0_then_singleton_phase123_qiskit_phase23_no_lanes_cap24_tau1em4_r50_20260807_v1_chtc",
    "paper_i_ra_adapt_macro_then_singleton_phase123_qiskit_phase23_no_lanes_tau1em4_r50_20260807_v1_chtc",
    "paper_i_ra_adapt_page12_insertion_comparators_r50_20260812_v1_chtc",
    "paper_i_ra_adapt_page16_insertion_comparators_k30_to_k50_20260813_v1_local_activation",
    "paper_i_ra_adapt_page16_insertion_comparators_k30_to_k50_20260813_v2_local_activation",
    "paper_i_ra_adapt_page16_insertion_comparators_weak50_strong30_20260812_v1_chtc",
    "paper_i_ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_beam3x2_metric_prune_cap24_tau1em4_r20_20260812_v1_chtc",
    "paper_i_ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_beam3x2_metric_prune_cap24_tau1em4_r20_20260812_v2_chtc",
    "paper_i_ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_cap24_tau1em4_weak50_strong30_20260811_v1_chtc",
    "paper_i_ra_adapt_pure_hubbard_page12_fullnoise6_r50_20260811_v1_chtc",
    "paper_i_ra_adapt_pure_hubbard_page12_fullnoise6_r50_20260811_v2_chtc",
    "paper_i_ra_adapt_pure_hubbard_page12_fullnoise6_r50_20260811_v3_chtc",
    "paper_i_ra_adapt_pure_hubbard_page12_lowhigh4_r20_20260811_v1_chtc",
    "paper_i_ra_adapt_pure_hubbard_page12_lowhigh4_r30_20260812_v1_chtc",
)

GROUP_LABELS = {
    "page12_global_singleton": "Page 12 global-singleton RA",
    "page16_intact_macro": "Page 16 intact-macro RA",
    "page16_beam3x2_metric_prune": "Page 16 intact-macro beam-3x2 metric-prune RA",
    "staged_macro_to_singleton": "Staged macro-to-singleton RA",
    "l3_continuation": "L=3 continuation RA",
    "pure_hubbard_diagnostics": "Pure-Hubbard RA diagnostics",
}
REQUIRED_GROUPS = frozenset(GROUP_LABELS)

REGIME_IDS = (
    "strong_strong_u8",
    "intermediate_strong",
    "strong_weak_u8",
    "intermediate_weak",
    "weak_strong",
    "weak_weak",
)


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path, *, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def read_small_json(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_SMALL_JSON_BYTES:
        raise ValueError(f"refusing to read non-small JSON ({size} bytes): {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_self_digest(payload: dict[str, Any], *, path: Path) -> str:
    declared = payload.get("sha256")
    if not isinstance(declared, str):
        raise ValueError(f"missing canonical self digest: {path}")
    actual = canonical_sha256(payload)
    if actual != declared:
        raise ValueError(
            f"canonical self digest mismatch for {path}: {actual} != {declared}"
        )
    return actual


def json_binding(
    path: Path,
    payload: dict[str, Any],
    *,
    repo_root: Path,
    require_self_digest: bool = True,
) -> dict[str, Any]:
    canonical = None
    if require_self_digest or isinstance(payload.get("sha256"), str):
        canonical = validate_self_digest(payload, path=path)
    binding: dict[str, Any] = {
        "path": repo_relative(path, repo_root=repo_root),
        "file_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "schema": payload.get("schema"),
        "status": payload.get("status"),
    }
    if canonical is not None:
        binding["canonical_sha256"] = canonical
    return binding


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _affected_protocol_paths(repo_root: Path) -> list[Path]:
    campaign_root = repo_root / CAMPAIGN_ROOT_RELATIVE
    paths: list[Path] = []
    for container_id in AFFECTED_CONTAINER_IDS:
        container = campaign_root / container_id
        if not container.is_dir():
            raise ValueError(f"sealed affected container is missing: {container}")
        for path in container.rglob("*.json"):
            if (
                path.parent.name != "protocols"
                or path.stat().st_size > MAX_SMALL_JSON_BYTES
            ):
                continue
            try:
                payload = read_small_json(path)
            except (json.JSONDecodeError, OSError, ValueError):
                continue
            mode = _nested(
                payload,
                "route_contract",
                "execution_settings",
                "phase3_hardware_cost_normalization_mode",
            )
            if mode == SIGNED_MODE:
                paths.append(path)
    return sorted(paths, key=lambda item: repo_relative(item, repo_root=repo_root))


def _container_id(protocol_path: Path, *, repo_root: Path) -> str:
    campaign_root = repo_root / CAMPAIGN_ROOT_RELATIVE
    return protocol_path.resolve().relative_to(campaign_root.resolve()).parts[0]


def _source_package_id(container_id: str, *, repo_root: Path) -> str:
    campaign_root = repo_root / CAMPAIGN_ROOT_RELATIVE
    container = campaign_root / container_id
    source_manifest = container / "source/source_archive_manifest.json"
    if source_manifest.is_file():
        return container_id
    activation_path = container / "activation_manifest.json"
    if not activation_path.is_file():
        raise ValueError(
            f"affected protocol container lacks a source lock or activation: {container_id}"
        )
    activation = read_small_json(activation_path)
    validate_self_digest(activation, path=activation_path)
    source_id = activation.get("source_package_id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError(f"activation lacks source_package_id: {activation_path}")
    return source_id


def _source_lock_binding(source_package_id: str, *, repo_root: Path) -> dict[str, Any]:
    campaign_root = repo_root / CAMPAIGN_ROOT_RELATIVE
    manifest_path = (
        campaign_root / source_package_id / "source/source_archive_manifest.json"
    )
    manifest = read_small_json(manifest_path)
    validate_self_digest(manifest, path=manifest_path)
    members = manifest.get("members")
    if not isinstance(members, list):
        raise ValueError(f"source archive manifest lacks members: {manifest_path}")
    matches = [
        member
        for member in members
        if isinstance(member, dict) and member.get("path") == SCORING_MEMBER
    ]
    if len(matches) != 1:
        raise ValueError(
            f"source archive must lock exactly one {SCORING_MEMBER}: {manifest_path}"
        )
    member = matches[0]
    scoring_sha = member.get("sha256")
    if scoring_sha not in KNOWN_DEFECTIVE_SCORING_SHA256S:
        raise ValueError(
            f"unrecognized scoring source is not proven affected: {scoring_sha}"
        )
    return {
        "source_package_id": source_package_id,
        "source_archive_manifest": json_binding(
            manifest_path, manifest, repo_root=repo_root
        ),
        "scoring_member": {
            "path": SCORING_MEMBER,
            "sha256": scoring_sha,
            "size_bytes": int(member["size_bytes"]),
            "known_defective_consumer": True,
        },
    }


def _group_id(container_id: str) -> str:
    if "pure_hubbard" in container_id:
        return "pure_hubbard_diagnostics"
    if "l3_" in container_id or "_l3" in container_id:
        return "l3_continuation"
    if "page16" in container_id and "beam3x2" in container_id:
        return "page16_beam3x2_metric_prune"
    if "page16" in container_id:
        return "page16_intact_macro"
    if (
        "page10" in container_id
        or "macro_then_singleton" in container_id
        or "macro_gradient_phase0_then_singleton" in container_id
    ):
        return "staged_macro_to_singleton"
    if "global_singleton" in container_id or "page12" in container_id:
        return "page12_global_singleton"
    raise ValueError(f"affected protocol container is not classified: {container_id}")


def _policy_label(insertion_kind: str) -> str:
    return {
        "plateau_commutation": "plateau",
        "always_commutation_reduced": "always_open",
        "append_only": "ra_append_only",
    }.get(insertion_kind, insertion_kind)


def _regime_id(execution_id: str) -> str | None:
    for regime in REGIME_IDS:
        if f"__{regime}__" in execution_id:
            return regime
    return None


def _execution_evidence_by_protocol(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    roots = [
        repo_root / CAMPAIGN_ROOT_RELATIVE,
        repo_root / LOCAL_RUN_ROOT_RELATIVE,
    ]

    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("execution_manifest.json"):
            if path.stat().st_size > MAX_SMALL_JSON_BYTES:
                continue
            payload = read_small_json(path)
            schema = str(payload.get("schema", ""))
            rounds = payload.get("controller_rounds_completed")
            protocol_sha = payload.get("protocol_sha256")
            if (
                "execution_manifest" not in schema
                or payload.get("status") != "passed"
                or not isinstance(rounds, int)
                or rounds <= 0
                or not isinstance(protocol_sha, str)
            ):
                continue
            binding = json_binding(path, payload, repo_root=repo_root)
            binding.update(
                {
                    "evidence_kind": "passed_execution_manifest",
                    "execution_id": payload.get("execution_id"),
                    "controller_rounds_completed": rounds,
                    "protocol_sha256": protocol_sha,
                }
            )
            evidence[protocol_sha].append(binding)

    closure_root = (
        repo_root
        / CAMPAIGN_ROOT_RELATIVE
        / "page12_insertion_comparator_closure_receipts"
    )
    if closure_root.is_dir():
        for path in sorted(closure_root.glob("*.json")):
            payload = read_small_json(path)
            protocol_sha = _nested(payload, "protocol", "canonical_sha256")
            checks = payload.get("authentication_checks")
            rounds = payload.get("controller_rounds_completed")
            if (
                not isinstance(protocol_sha, str)
                or payload.get("status")
                != "passed_authenticated_page12_insertion_comparator_closure"
                or not isinstance(checks, dict)
                or not checks
                or not all(value is True for value in checks.values())
                or not isinstance(rounds, int)
                or rounds <= 0
            ):
                continue
            binding = json_binding(path, payload, repo_root=repo_root)
            binding.update(
                {
                    "evidence_kind": "authenticated_archive_closure_receipt",
                    "execution_id": payload.get("run_id"),
                    "controller_rounds_completed": rounds,
                    "protocol_sha256": protocol_sha,
                    "cluster_id": payload.get("cluster_id"),
                    "proc_id": payload.get("proc_id"),
                }
            )
            evidence[protocol_sha].append(binding)

    campaign_root = repo_root / CAMPAIGN_ROOT_RELATIVE
    for path in campaign_root.rglob("*.json"):
        if path.parent.name != "terminal_chtc" or path.stat().st_size > MAX_SMALL_JSON_BYTES:
            continue
        payload = read_small_json(path)
        protocol_sha = payload.get("protocol_sha256")
        rounds = payload.get("controller_rounds_completed")
        if (
            not isinstance(protocol_sha, str)
            or payload.get("authenticated_full_sealed_closure") is not True
            or not str(payload.get("status", "")).startswith("passed_authenticated")
            or not isinstance(rounds, int)
            or rounds <= 0
        ):
            continue
        binding = json_binding(path, payload, repo_root=repo_root)
        binding.update(
            {
                "evidence_kind": "authenticated_terminal_receipt",
                "execution_id": payload.get("execution_id"),
                "controller_rounds_completed": rounds,
                "protocol_sha256": protocol_sha,
                "cluster_id": payload.get("cluster_id"),
                "proc_id": payload.get("proc_id"),
            }
        )
        evidence[protocol_sha].append(binding)

    for path in campaign_root.rglob("p4_packaged_numerical_receipt.json"):
        if path.stat().st_size > MAX_SMALL_JSON_BYTES:
            continue
        payload = read_small_json(path)
        protocol_sha = payload.get("protocol_sha256")
        rounds = payload.get("completed_controller_rounds")
        if (
            not isinstance(protocol_sha, str)
            or payload.get("status") != "passed"
            or payload.get("scientific_execution_performed") is not True
            or payload.get("source_locked_archive_validated") is not True
            or payload.get("source_locked_import_isolated") is not True
            or not isinstance(rounds, int)
            or rounds <= 0
        ):
            continue
        binding = json_binding(path, payload, repo_root=repo_root)
        binding.update(
            {
                "evidence_kind": "source_locked_packaged_numerical_witness",
                "execution_id": payload.get("execution_id"),
                "controller_rounds_completed": rounds,
                "protocol_sha256": protocol_sha,
                "paper_facing_result_allowed": False,
            }
        )
        evidence[protocol_sha].append(binding)

    for protocol_sha, bindings in evidence.items():
        unique = {item["path"]: item for item in bindings}
        evidence[protocol_sha] = [unique[key] for key in sorted(unique)]
    return dict(evidence)


def _submission_evidence_by_package(
    repo_root: Path, *, package_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    campaign_root = repo_root / CAMPAIGN_ROOT_RELATIVE
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in campaign_root.rglob("*submission_receipt*.json"):
        if path.stat().st_size > MAX_SMALL_JSON_BYTES:
            continue
        payload = read_small_json(path)
        cluster_id = payload.get("cluster_id") or _nested(payload, "submission", "cluster_id")
        submitted = payload.get("submitted") is True or isinstance(cluster_id, int)
        if not submitted:
            continue
        candidates: list[str] = []
        declared = payload.get("package_id")
        if isinstance(declared, str):
            candidates.append(declared)
        relative_parts = path.resolve().relative_to(campaign_root.resolve()).parts
        if relative_parts and relative_parts[0] in package_ids:
            candidates.append(relative_parts[0])
        for package_id in package_ids:
            if path.name.startswith(package_id):
                candidates.append(package_id)
        for package_id in sorted(set(candidates)):
            if package_id not in package_ids:
                continue
            binding = json_binding(
                path,
                payload,
                repo_root=repo_root,
                require_self_digest=False,
            )
            binding.update(
                {
                    "evidence_kind": "submission_receipt_not_completion_proof",
                    "cluster_id": cluster_id,
                }
            )
            evidence[package_id].append(binding)
    for package_id, bindings in evidence.items():
        unique = {item["path"]: item for item in bindings}
        evidence[package_id] = [unique[key] for key in sorted(unique)]
    return dict(evidence)


def source_input_paths(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Return every small historical input the builder can consume."""
    protocol_paths = _affected_protocol_paths(repo_root)
    source_packages = {
        _source_package_id(_container_id(path, repo_root=repo_root), repo_root=repo_root)
        for path in protocol_paths
    }
    inputs = set(protocol_paths)
    campaign_root = repo_root / CAMPAIGN_ROOT_RELATIVE
    for package_id in source_packages:
        inputs.add(campaign_root / package_id / "source/source_archive_manifest.json")
    for path in campaign_root.rglob("execution_manifest.json"):
        if path.stat().st_size <= MAX_SMALL_JSON_BYTES:
            inputs.add(path)
    local_root = repo_root / LOCAL_RUN_ROOT_RELATIVE
    if local_root.is_dir():
        for path in local_root.rglob("execution_manifest.json"):
            if path.stat().st_size <= MAX_SMALL_JSON_BYTES:
                inputs.add(path)
    closure_root = campaign_root / "page12_insertion_comparator_closure_receipts"
    if closure_root.is_dir():
        inputs.update(closure_root.glob("*.json"))
    inputs.update(campaign_root.rglob("p4_packaged_numerical_receipt.json"))
    inputs.update(campaign_root.rglob("*submission_receipt*.json"))
    for path in campaign_root.rglob("*.json"):
        if path.parent.name == "terminal_chtc" and path.stat().st_size <= MAX_SMALL_JSON_BYTES:
            inputs.add(path)
    for path in protocol_paths:
        container = campaign_root / _container_id(path, repo_root=repo_root)
        activation = container / "activation_manifest.json"
        if activation.is_file():
            inputs.add(activation)
    return sorted(inputs, key=lambda item: repo_relative(item, repo_root=repo_root))


def build_map(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    protocol_paths = _affected_protocol_paths(repo_root)
    if not protocol_paths:
        raise ValueError("no affected zero-centered Paper-I protocols found")

    execution_evidence = _execution_evidence_by_protocol(repo_root)
    prepared: list[tuple[Path, str, str, dict[str, Any]]] = []
    all_package_ids: set[str] = set()
    source_locks: dict[str, dict[str, Any]] = {}
    for path in protocol_paths:
        container_id = _container_id(path, repo_root=repo_root)
        source_package_id = _source_package_id(container_id, repo_root=repo_root)
        all_package_ids.update({container_id, source_package_id})
        source_locks.setdefault(
            source_package_id,
            _source_lock_binding(source_package_id, repo_root=repo_root),
        )
        prepared.append((path, container_id, source_package_id, read_small_json(path)))
    submission_evidence = _submission_evidence_by_package(
        repo_root, package_ids=all_package_ids
    )

    records: list[dict[str, Any]] = []
    for path, container_id, source_package_id, protocol in prepared:
        protocol_sha = validate_self_digest(protocol, path=path)
        settings = _nested(protocol, "route_contract", "execution_settings")
        invariants = _nested(protocol, "route_contract", "semantic_invariants")
        method = _nested(protocol, "request", "method")
        if not isinstance(settings, dict) or not isinstance(invariants, dict):
            raise ValueError(f"protocol lacks route contract settings/invariants: {path}")
        normalized_policy = invariants.get(
            "phase_ii_phase_iii_qiskit_population_normalization_policy"
        )
        if normalized_policy != SIGNED_MODE:
            raise ValueError(
                f"configured/normalized signed policy mismatch in affected protocol: {path}"
            )
        if not isinstance(method, dict):
            raise ValueError(f"protocol lacks typed method request: {path}")
        insertion_kind = _nested(method, "insertion", "kind")
        beam_kind = _nested(method, "beam", "kind")
        pruning_kind = _nested(method, "pruning", "kind")
        if not isinstance(insertion_kind, str):
            raise ValueError(f"protocol lacks typed insertion kind: {path}")
        execution_id = path.stem
        bound_execution = execution_evidence.get(protocol_sha, [])
        bound_submission = sorted(
            submission_evidence.get(container_id, [])
            + submission_evidence.get(source_package_id, []),
            key=lambda item: item["path"],
        )
        if bound_execution:
            evidence_status = "authenticated_executed_defective_consumer"
        elif bound_submission:
            evidence_status = "submitted_affected_no_execution_proof"
        else:
            evidence_status = "configured_source_locked_affected_no_execution_proof"
        problem = protocol.get("problem")
        if not isinstance(problem, dict):
            problem = {}
        stop_rounds = _nested(protocol, "request", "execution", "stop", "maximum_controller_rounds")
        records.append(
            {
                "group_id": _group_id(container_id),
                "container_id": container_id,
                "source_package_id": source_package_id,
                "execution_id": execution_id,
                "protocol": json_binding(path, protocol, repo_root=repo_root),
                "route_contract_sha256": _nested(protocol, "route_contract", "sha256"),
                "problem_key": problem.get("problem_key") or settings.get("problem"),
                "regime_id": _regime_id(execution_id),
                "n_ph_max": problem.get("n_ph_max"),
                "target_horizon": stop_rounds,
                "route_axes": {
                    "insertion_kind": insertion_kind,
                    "paper_i_policy_label": _policy_label(insertion_kind),
                    "beam_kind": beam_kind,
                    "pruning_kind": pruning_kind,
                    "candidate_representation_id": _nested(
                        protocol, "request", "adapter", "candidate_representation_id"
                    ),
                    "phase3_backend_cost_scope": settings.get(
                        "phase3_backend_cost_scope"
                    ),
                },
                "defect_configuration": {
                    "configured_normalization_mode": settings.get(
                        "phase3_hardware_cost_normalization_mode"
                    ),
                    "normalized_feature_policy": normalized_policy,
                    "configured_and_normalized_policy_match": True,
                    "source_lock": source_locks[source_package_id],
                    "defective_phase3_consumer_proven_in_source": True,
                },
                "evidence_status": evidence_status,
                "actually_executed_defective_phase3_consumer": bool(bound_execution),
                "execution_evidence": bound_execution,
                "submission_evidence": bound_submission,
                "accepted_trajectory_change": {
                    "proven": False,
                    "evidence": [],
                    "reason": (
                        "No authenticated corrected counterfactual replay or rescoring "
                        "receipt proves a changed winner or accepted prefix for this protocol."
                    ),
                },
            }
        )

    records.sort(key=lambda row: row["protocol"]["path"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(record)
    if set(grouped) != REQUIRED_GROUPS:
        missing = sorted(REQUIRED_GROUPS.difference(grouped))
        extra = sorted(set(grouped).difference(REQUIRED_GROUPS))
        raise ValueError(f"affected-map group closure failed; missing={missing}, extra={extra}")

    groups: list[dict[str, Any]] = []
    for group_id in GROUP_LABELS:
        group_records = grouped[group_id]
        statuses = Counter(row["evidence_status"] for row in group_records)
        policies = Counter(
            row["route_axes"]["paper_i_policy_label"] for row in group_records
        )
        groups.append(
            {
                "group_id": group_id,
                "label": GROUP_LABELS[group_id],
                "protocol_artifact_count": len(group_records),
                "distinct_protocol_digest_count": len(
                    {row["protocol"]["canonical_sha256"] for row in group_records}
                ),
                "container_count": len({row["container_id"] for row in group_records}),
                "authenticated_executed_protocol_digest_count": len(
                    {
                        row["protocol"]["canonical_sha256"]
                        for row in group_records
                        if row["actually_executed_defective_phase3_consumer"]
                    }
                ),
                "evidence_status_counts": dict(sorted(statuses.items())),
                "policy_counts": dict(sorted(policies.items())),
                "records": group_records,
            }
        )

    protocol_digests = {row["protocol"]["canonical_sha256"] for row in records}
    executed_digests = {
        row["protocol"]["canonical_sha256"]
        for row in records
        if row["actually_executed_defective_phase3_consumer"]
    }
    evidence_statuses = Counter(row["evidence_status"] for row in records)
    unsigned: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed_complete_lightweight_inventory",
        "inventory_as_of": INVENTORY_AS_OF,
        "scope": {
            "campaign_root": CAMPAIGN_ROOT_RELATIVE.as_posix(),
            "local_execution_manifest_root": LOCAL_RUN_ROOT_RELATIVE.as_posix(),
            "sealed_affected_container_ids": list(AFFECTED_CONTAINER_IDS),
            "maximum_json_bytes_read": MAX_SMALL_JSON_BYTES,
            "large_result_checkpoint_and_ledger_jsons_read": False,
            "historical_artifacts_mutated": False,
            "conventional_append_adapt_in_scope": False,
            "older_family_robust_symmetric_arctan_stationary_core_in_scope": False,
        },
        "defect_contract": {
            "configured_mode": SIGNED_MODE,
            "normalized_feature_policy_required_to_match": True,
            "defective_consumer_behavior": (
                "The locked historical Phase-III consumer recognized only the older "
                "symmetric enum and otherwise retained hardware_cost_score_factor=1.0."
            ),
            "scoring_member_path": SCORING_MEMBER,
            "known_defective_scoring_sha256s": sorted(
                KNOWN_DEFECTIVE_SCORING_SHA256S
            ),
            "classification_rule": (
                "configured affected = a self-digested protocol requests the signed mode "
                "and its source archive locks a known-defective consumer; actually executed "
                "= a passed small execution/closure/numerical receipt additionally binds that "
                "exact protocol digest."
            ),
        },
        "trajectory_change_proof_contract": {
            "required_evidence": (
                "An authenticated corrected counterfactual replay or complete rescoring "
                "receipt must demonstrate a changed selected winner or accepted prefix."
            ),
            "score_factor_mismatch_alone_is_sufficient": False,
            "configured_defect_or_execution_alone_is_sufficient": False,
        },
        "confirmed_score_factor_mismatch_observations": [
            {
                "route_family": "Page 12 global-singleton strong--weak plateau",
                "regime_id": "strong_weak_u8",
                "controller_round": 11,
                "phase_ii_signed_factor": 0.8645492142678116,
                "recorded_phase_iii_factor": 1.0,
                "score_factor_mismatch_proven": True,
                "accepted_winner_or_prefix_change_proven": False,
                "lightweight_authenticated_factor_trace_path_available": False,
                "interpretation": (
                    "This forensic observation proves producer/consumer score-factor "
                    "mismatch. It does not, without a corrected counterfactual ranking, "
                    "prove a different winner or accepted trajectory."
                ),
            }
        ],
        "minimal_reproducer": {
            "configured_phase_iii_input_factors": [1.25, 0.75],
            "defective_historical_phase_iii_factors": [1.0, 1.0],
            "corrected_phase_iii_factors": [1.25, 0.75],
            "accepted_trajectory_change_proven": False,
        },
        "summary": {
            "required_group_count": len(REQUIRED_GROUPS),
            "protocol_artifact_count": len(records),
            "distinct_protocol_digest_count": len(protocol_digests),
            "container_count": len({row["container_id"] for row in records}),
            "source_package_count": len({row["source_package_id"] for row in records}),
            "authenticated_executed_protocol_digest_count": len(executed_digests),
            "evidence_status_counts": dict(sorted(evidence_statuses.items())),
            "score_factor_mismatch_proven_count": 1,
            "accepted_trajectory_change_proven_count": 0,
        },
        "groups": groups,
    }
    output = dict(unsigned)
    output["sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return output


def encoded_map(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def write_map(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded_map(payload))
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / DEFAULT_OUTPUT_RELATIVE,
        help="affected-map JSON path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless the existing output is byte-identical to a fresh build",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_map()
    expected = encoded_map(payload)
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != expected:
            raise SystemExit(f"affected map is stale or missing: {args.output}")
        return 0
    write_map(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
