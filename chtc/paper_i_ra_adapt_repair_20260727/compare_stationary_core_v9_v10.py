"""Compare v9 and v10 stationary-core protocols without masking semantics."""

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
V9_ROOT = MATERIALIZATIONS / "ra_adapt_stationary_late_core_v9"
V10_ROOT = MATERIALIZATIONS / "ra_adapt_stationary_late_core_v10"
BUNDLE_ID = "ra_repair_stationary_late_core_v1"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "prompt-exports/"
    "paper-i-ra-adapt-v9-v10-semantic-comparison-20260728.json"
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
    """Raised when v10 changes a v9 scientific protocol field."""


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
    v9_bundle = V9_ROOT / BUNDLE_ID
    v10_bundle = V10_ROOT / BUNDLE_ID
    v9_protocol_dir = v9_bundle / "protocols"
    v10_protocol_dir = v10_bundle / "protocols"
    v9_names = sorted(path.name for path in v9_protocol_dir.glob("*.json"))
    v10_names = sorted(path.name for path in v10_protocol_dir.glob("*.json"))
    if v9_names != v10_names or len(v9_names) != 48:
        raise ComparisonError("The v9/v10 48-protocol filename sets differ.")

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
    v9_semantic_matrix: dict[str, str] = {}
    v10_semantic_matrix: dict[str, str] = {}
    for name in v9_names:
        v9_protocol = _load(v9_protocol_dir / name)
        v10_protocol = _load(v10_protocol_dir / name)
        differences = _differences(v9_protocol, v10_protocol)
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

        v9_semantic = _without_authority(v9_protocol)
        v10_semantic = _without_authority(v10_protocol)
        v9_semantic_sha256 = _canonical_sha256(v9_semantic)
        v10_semantic_sha256 = _canonical_sha256(v10_semantic)
        v9_categories = _category_payload(v9_protocol)
        v10_categories = _category_payload(v10_protocol)
        category_equal = {
            category: v9_categories[category] == v10_categories[category]
            for category in category_pass_counts
        }
        for category, equal in category_equal.items():
            category_pass_counts[category] += int(equal)
        cell_id = str(v9_protocol["bundle_materialization"]["cell_id"])
        v9_semantic_matrix[cell_id] = v9_semantic_sha256
        v10_semantic_matrix[cell_id] = v10_semantic_sha256
        rows.append(
            {
                "cell_id": cell_id,
                "protocol_path": f"protocols/{name}",
                "observed_difference_json_pointers": differences,
                "unexpected_difference_json_pointers": unexpected_pointers,
                "v9_semantic_sha256": v9_semantic_sha256,
                "v10_semantic_sha256": v10_semantic_sha256,
                "semantic_equal": v9_semantic == v10_semantic,
                "category_equal": category_equal,
            }
        )

    v9_source_input = _load(
        V9_ROOT / "source_materialization/source_locks_input.json"
    )
    v10_source_input = _load(
        V10_ROOT / "source_materialization/source_locks_input.json"
    )
    v9_baselines = _load(
        V9_ROOT / "source_materialization/problem_baselines.json"
    )
    v10_baselines = _load(
        V10_ROOT / "source_materialization/problem_baselines.json"
    )
    v9_final = _load(V9_ROOT / "final_publication_receipt.json")
    v10_final = _load(V10_ROOT / "final_publication_receipt.json")
    v9_inventory = v9_final["implementation_source_inventory"]
    v10_inventory = v10_final["implementation_source_inventory"]

    all_categories_pass = all(
        count == 48 for count in category_pass_counts.values()
    )
    semantic_matrix_equal = v9_semantic_matrix == v10_semantic_matrix
    source_inputs_equal = (
        v9_source_input == v10_source_input
        and v9_baselines == v10_baselines
    )
    passed = (
        not unexpected
        and semantic_matrix_equal
        and source_inputs_equal
        and all_categories_pass
        and v9_final["matrix"] == v10_final["matrix"]
        and v9_final["stationarity_selection"]
        == v10_final["stationarity_selection"]
        and v9_inventory["sha256"] != v10_inventory["sha256"]
    )
    if not passed:
        raise ComparisonError(
            "v9/v10 semantic comparison did not satisfy its assertions."
        )

    payload: dict[str, Any] = {
        "schema": (
            "paper_i_ra_adapt_stationary_core_v9_v10_"
            "semantic_comparison_v1"
        ),
        "status": "passed",
        "captured_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "scope": (
            "machine_comparison_only_not_execution_or_evidence_adoption"
        ),
        "v9": {
            "root": V9_ROOT.relative_to(REPO_ROOT).as_posix(),
            "final_receipt": _binding(
                V9_ROOT / "final_publication_receipt.json",
                relative_to=REPO_ROOT,
            ),
            "bundle_manifest": _binding(
                v9_bundle / "bundle_manifest.json",
                relative_to=REPO_ROOT,
            ),
            "implementation_source_inventory_sha256": v9_inventory[
                "sha256"
            ],
        },
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
        "source_inputs": {
            "source_locks_input_exact_equal": (
                v9_source_input == v10_source_input
            ),
            "source_locks_input_sha256": _canonical_sha256(
                v9_source_input
            ),
            "problem_baselines_exact_equal": v9_baselines == v10_baselines,
            "problem_baselines_sha256": _canonical_sha256(v9_baselines),
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
            "v9_semantic_matrix_sha256": _canonical_sha256(
                v9_semantic_matrix
            ),
            "v10_semantic_matrix_sha256": _canonical_sha256(
                v10_semantic_matrix
            ),
            "category_pass_counts": category_pass_counts,
            "rows": rows,
        },
        "final_receipt_contracts": {
            "matrix_exact_equal": v9_final["matrix"] == v10_final["matrix"],
            "stationarity_selection_exact_equal": (
                v9_final["stationarity_selection"]
                == v10_final["stationarity_selection"]
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
