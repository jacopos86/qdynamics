"""Compare v10 and v11 stationary-core protocols without masking semantics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MATERIALIZATIONS = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations"
)
V10_ROOT = MATERIALIZATIONS / "ra_adapt_stationary_late_core_v10"
V11_ROOT = MATERIALIZATIONS / "ra_adapt_stationary_late_core_v11"
BUNDLE_ID = "ra_repair_stationary_late_core_v1"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "prompt-exports/"
    "paper-i-ra-adapt-v10-v11-semantic-comparison-20260729.json"
)

# These are the complete observed authority/digest seams. Every other protocol
# field must remain byte-equivalent as a canonical JSON value.
ALLOWED_AUTHORITY_POINTERS = (
    "/bundle_manifest_sha256",
    "/bundle_materialization/bundle_manifest_sha256",
    "/bundle_materialization/sha256",
    "/bundle_materialization/source_lock_refs_sha256",
    "/bundle_materialization/source_locks_sha256",
    "/sha256",
    "/source_locks/implementation_source_inventory_sha256",
    "/source_locks/source_locks_manifest_sha256",
)


class ComparisonError(RuntimeError):
    """Raised when v11 changes a v10 scientific protocol field."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComparisonError(f"Expected a JSON object: {path}")
    return value


def _binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    value = _load(path)
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "file_sha256": _file_sha256(path),
        "canonical_sha256": value.get("sha256"),
        "size_bytes": path.stat().st_size,
    }


def _differences(left: Any, right: Any, pointer: str = "") -> list[str]:
    if type(left) is not type(right):
        return [pointer]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{pointer}/{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_differences(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{pointer}/length"]
        paths = []
        for index, (left_item, right_item) in enumerate(
            zip(left, right, strict=True)
        ):
            paths.extend(
                _differences(
                    left_item,
                    right_item,
                    f"{pointer}/{index}",
                )
            )
        return paths
    return [] if left == right else [pointer]


def _without_authority(protocol: dict[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(protocol)
    for pointer in ALLOWED_AUTHORITY_POINTERS:
        tokens = pointer.lstrip("/").split("/")
        parent: Any = projected
        for token in tokens[:-1]:
            parent = parent[token]
        del parent[tokens[-1]]
    return projected


def _category_payload(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "physics": protocol["problem"],
        "method": {
            key: protocol[key]
            for key in (
                "algorithm_id",
                "adapter_id",
                "selector_identity",
                "candidate_representation",
                "active_gradient_policy",
                "derivative_chart_id",
                "estimator_accounting_convention",
                "resource_weighting_scope",
            )
        },
        "route": {
            "route_contract": protocol["route_contract"],
            "trust_policy_id": protocol["trust_policy_id"],
            "phase3_solver_id": protocol["phase3_solver_id"],
            "phase3_multiplier_contract": protocol[
                "phase3_multiplier_contract"
            ],
        },
        "cutoff": {
            "n_ph_max": protocol["problem"]["n_ph_max"],
            "problem_key": protocol["problem"]["problem_key"],
        },
        "seed": protocol["seeds"],
        "optimizer": {
            "optimizer": protocol["optimizer"],
            "optimizer_maxiter": protocol["optimizer_maxiter"],
        },
        "horizon": {
            "horizon": protocol["horizon"],
            "stopping_rule": protocol["stopping_rule"],
        },
        "pools": {
            "parent_inventory": protocol["parent_inventory"],
            "executable_pool": protocol["executable_pool"],
            "lineage_authority": protocol["lineage_authority"],
        },
        "settings": {
            "request": protocol["request"],
            "compile_identity": protocol["compile_identity"],
            "accepted_refit_base_chart_policy": protocol[
                "accepted_refit_base_chart_policy"
            ],
            "accepted_refit_coordinate_chart": protocol[
                "accepted_refit_coordinate_chart"
            ],
            "accepted_refit_scope": protocol["accepted_refit_scope"],
        },
    }


def compare() -> dict[str, Any]:
    v10_bundle = V10_ROOT / BUNDLE_ID
    v11_bundle = V11_ROOT / BUNDLE_ID
    v10_protocol_dir = v10_bundle / "protocols"
    v11_protocol_dir = v11_bundle / "protocols"
    v10_names = sorted(path.name for path in v10_protocol_dir.glob("*.json"))
    v11_names = sorted(path.name for path in v11_protocol_dir.glob("*.json"))
    if v10_names != v11_names or len(v10_names) != 48:
        raise ComparisonError("The v10/v11 48-protocol filename sets differ.")

    observed_difference_counts: dict[str, int] = {}
    category_pass_counts = {
        key: 0
        for key in (
            "physics",
            "method",
            "route",
            "cutoff",
            "seed",
            "optimizer",
            "horizon",
            "pools",
            "settings",
        )
    }
    rows: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []
    v10_semantic_matrix: dict[str, str] = {}
    v11_semantic_matrix: dict[str, str] = {}
    for name in v10_names:
        v10_protocol = _load(v10_protocol_dir / name)
        v11_protocol = _load(v11_protocol_dir / name)
        differences = _differences(v10_protocol, v11_protocol)
        for pointer in differences:
            observed_difference_counts[pointer] = (
                observed_difference_counts.get(pointer, 0) + 1
            )
        unexpected_pointers = sorted(
            set(differences) - set(ALLOWED_AUTHORITY_POINTERS)
        )
        if unexpected_pointers:
            unexpected.append(
                {
                    "protocol": name,
                    "json_pointers": unexpected_pointers,
                }
            )

        v10_semantic = _without_authority(v10_protocol)
        v11_semantic = _without_authority(v11_protocol)
        v10_semantic_sha256 = _canonical_sha256(v10_semantic)
        v11_semantic_sha256 = _canonical_sha256(v11_semantic)
        v10_categories = _category_payload(v10_protocol)
        v11_categories = _category_payload(v11_protocol)
        category_equal = {
            category: v10_categories[category] == v11_categories[category]
            for category in category_pass_counts
        }
        for category, equal in category_equal.items():
            category_pass_counts[category] += int(equal)
        cell_id = str(v10_protocol["bundle_materialization"]["cell_id"])
        v10_semantic_matrix[cell_id] = v10_semantic_sha256
        v11_semantic_matrix[cell_id] = v11_semantic_sha256
        rows.append(
            {
                "cell_id": cell_id,
                "protocol_path": f"protocols/{name}",
                "observed_difference_json_pointers": differences,
                "unexpected_difference_json_pointers": unexpected_pointers,
                "v10_semantic_sha256": v10_semantic_sha256,
                "v11_semantic_sha256": v11_semantic_sha256,
                "semantic_equal": v10_semantic == v11_semantic,
                "category_equal": category_equal,
            }
        )

    v10_source_input = _load(
        V10_ROOT / "source_materialization/source_locks_input.json"
    )
    v11_source_input = _load(
        V11_ROOT / "source_materialization/source_locks_input.json"
    )
    v10_baselines = _load(
        V10_ROOT / "source_materialization/problem_baselines.json"
    )
    v11_baselines = _load(
        V11_ROOT / "source_materialization/problem_baselines.json"
    )
    v10_final = _load(V10_ROOT / "final_publication_receipt.json")
    v11_final = _load(V11_ROOT / "final_publication_receipt.json")
    v10_inventory = v10_final["implementation_source_inventory"]
    v11_inventory = v11_final["implementation_source_inventory"]

    all_categories_pass = all(
        count == 48 for count in category_pass_counts.values()
    )
    semantic_matrix_equal = v10_semantic_matrix == v11_semantic_matrix
    source_inputs_equal = (
        v10_source_input == v11_source_input
        and v10_baselines == v11_baselines
    )
    passed = (
        not unexpected
        and semantic_matrix_equal
        and source_inputs_equal
        and all_categories_pass
        and v10_final["matrix"] == v11_final["matrix"]
        and v10_final["stationarity_selection"]
        == v11_final["stationarity_selection"]
        and v10_inventory["sha256"] != v11_inventory["sha256"]
    )
    if not passed:
        raise ComparisonError(
            "v10/v11 semantic comparison did not satisfy its assertions."
        )

    payload: dict[str, Any] = {
        "schema": (
            "paper_i_ra_adapt_stationary_core_v10_v11_"
            "semantic_comparison_v1"
        ),
        "status": "passed",
        "captured_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "scope": (
            "machine_comparison_only_not_execution_or_evidence_adoption"
        ),
        "v10": {
            "root": V10_ROOT.relative_to(REPO_ROOT).as_posix(),
            "final_receipt": _binding(
                V10_ROOT / "final_publication_receipt.json",
                relative_to=REPO_ROOT,
            ),
            "bundle_manifest": _binding(
                v10_bundle / "bundle_manifest.json",
                relative_to=REPO_ROOT,
            ),
            "implementation_source_inventory_sha256": v10_inventory[
                "sha256"
            ],
        },
        "v11": {
            "root": V11_ROOT.relative_to(REPO_ROOT).as_posix(),
            "final_receipt": _binding(
                V11_ROOT / "final_publication_receipt.json",
                relative_to=REPO_ROOT,
            ),
            "bundle_manifest": _binding(
                v11_bundle / "bundle_manifest.json",
                relative_to=REPO_ROOT,
            ),
            "implementation_source_inventory_sha256": v11_inventory[
                "sha256"
            ],
        },
        "source_inputs": {
            "source_locks_input_exact_equal": (
                v10_source_input == v11_source_input
            ),
            "source_locks_input_sha256": _canonical_sha256(
                v10_source_input
            ),
            "problem_baselines_exact_equal": v10_baselines == v11_baselines,
            "problem_baselines_sha256": _canonical_sha256(v10_baselines),
        },
        "protocol_matrix": {
            "protocol_count": 48,
            "filename_sets_equal": True,
            "allowed_authority_difference_json_pointers": list(
                ALLOWED_AUTHORITY_POINTERS
            ),
            "observed_difference_counts": dict(
                sorted(observed_difference_counts.items())
            ),
            "unexpected_difference_count": len(unexpected),
            "unexpected_differences": unexpected,
            "semantic_matrix_equal": semantic_matrix_equal,
            "v10_semantic_matrix_sha256": _canonical_sha256(
                v10_semantic_matrix
            ),
            "v11_semantic_matrix_sha256": _canonical_sha256(
                v11_semantic_matrix
            ),
            "category_pass_counts": category_pass_counts,
            "rows": rows,
        },
        "final_receipt_contracts": {
            "matrix_exact_equal": v10_final["matrix"] == v11_final["matrix"],
            "stationarity_selection_exact_equal": (
                v10_final["stationarity_selection"]
                == v11_final["stationarity_selection"]
            ),
        },
        "conclusion": {
            "scientific_protocol_semantics_exact_equal": True,
            "physics_equal": True,
            "method_equal": True,
            "route_equal": True,
            "cutoff_equal": True,
            "seed_equal": True,
            "optimizer_equal": True,
            "horizon_equal": True,
            "pools_equal": True,
            "settings_equal": True,
            "differences_confined_to_materialization_source_authority": True,
        },
    }
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = compare()
    with output.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": output.relative_to(REPO_ROOT).as_posix(),
                "sha256": payload["sha256"],
                "protocol_count": 48,
                "unexpected_difference_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
