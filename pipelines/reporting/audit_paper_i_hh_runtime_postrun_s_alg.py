#!/usr/bin/env python3
"""Audit Paper-I SNAKE clean ``S_alg`` and runtime-ledger diagnostics.

For each selected first-hit prefix, the runtime checkpoint declares the final
occurrence sequence included in that prefix.  This audit streams the terminal
estimator-call sidecar to that exact boundary, independently re-deduplicates
physical primitive IDs, and requires equality with the runtime receipt.  It
separately reconstructs the clean algorithmic logical-invocation count from
the signed history and requires equality with the paper-facing target-prefix
row.  Runtime physical-identity uniqueness is a diagnostic, not ``S_alg``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any, Iterator, Mapping

from pipelines.reporting.build_paper_i_hh_comparator_tracking_summary import (
    _iter_named_json_array,
)
from pipelines.exact_bench.paper_i_s_alg_accounting import (
    PAPER_I_S_ALG_ACCOUNTING_SCHEMA,
    PAPER_I_S_ALG_CONTRACT,
    snake_clean_prefix_work,
)
from pipelines.static_adapt.estimator_call_ledger import (
    S_ALG_COMPONENTS,
    summarize_estimator_occurrence_prefix,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_JSON = REPO_ROOT / (
    "output/pdf/paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_"
    "tracking_20260715/target_energy_prefix_costs.json"
)
DEFAULT_OUTPUT_JSON = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_hh_sr_20260721/"
    "paper_i_no_overlap_runtime_postrun_s_alg_audit.json"
)
DEFAULT_ROUTE_ID = "no_overlap_trust_projected_phase3_nph3_7"
AUDIT_SCHEMA = "paper_i_runtime_postrun_s_alg_closure_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_path(source: Mapping[str, Any]) -> Path:
    path = Path(str(source.get("path") or ""))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _ledger_occurrences(
    archive_path: Path,
    *,
    member_name: str,
) -> Iterator[dict[str, Any]]:
    with tarfile.open(archive_path, "r|gz") as archive:
        for info in archive:
            if info.name != member_name:
                archive.members.clear()
                continue
            handle = archive.extractfile(info)
            if handle is None:
                raise RuntimeError(
                    f"cannot extract {member_name} from {archive_path}"
                )
            yield from _iter_named_json_array(handle, "occurrences")
            return
    raise RuntimeError(
        f"missing estimator-call ledger member {member_name} in {archive_path}"
    )


def _history_prefix(
    archive_path: Path,
    *,
    member_name: str,
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("history prefix count must be positive")
    rows: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r|gz") as archive:
        for info in archive:
            if info.name != member_name:
                archive.members.clear()
                continue
            handle = archive.extractfile(info)
            if handle is None:
                raise RuntimeError(
                    f"cannot extract {member_name} from {archive_path}"
                )
            for row in _iter_named_json_array(handle, "history"):
                if not isinstance(row, Mapping):
                    raise ValueError("result history contains a non-mapping row")
                rows.append(dict(row))
                if len(rows) == count:
                    return rows
            break
    raise RuntimeError(
        f"{member_name} contains only {len(rows)} history rows; "
        f"{count} were required"
    )


def _component_map(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a component mapping")
    components = {name: int(value.get(name, -1)) for name in S_ALG_COMPONENTS}
    if any(count < 0 for count in components.values()):
        raise ValueError(f"{label} lacks nonnegative component closure")
    return components


def _audit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    source = row.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("target-prefix row lacks a source receipt")
    archive_path = _source_path(source)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    observed_archive_sha256 = _sha256(archive_path)
    expected_archive_sha256 = str(source.get("sha256") or "")
    if (
        not expected_archive_sha256
        or observed_archive_sha256 != expected_archive_sha256
    ):
        raise ValueError(
            "target-prefix archive SHA-256 drift: "
            f"expected={expected_archive_sha256}, "
            f"observed={observed_archive_sha256}"
        )

    result_member = str(
        source.get("result_member") or source.get("member") or ""
    )
    if not result_member.endswith("/result.json"):
        raise ValueError("target-prefix source lacks an exact result member")
    history_position = int(row.get("history_position") or row.get("k_target") or 0)
    if history_position <= 0:
        raise ValueError("target-prefix row has no positive history position")
    history = _history_prefix(
        archive_path,
        member_name=result_member,
        count=history_position,
    )
    history_row = history[-1]
    checkpoint = history_row.get("active_prefix_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("selected history row lacks an active-prefix checkpoint")
    receipt = checkpoint.get("estimator_ledger_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("selected checkpoint lacks an estimator-ledger receipt")
    if (
        receipt.get("schema")
        != "paper_i_active_prefix_estimator_ledger_receipt_v1"
        or receipt.get("status") != "complete"
        or receipt.get("canonical_same_state_deduplication_active") is not True
        or receipt.get("raw_occurrences_preserved") is not True
    ):
        raise ValueError("selected runtime estimator receipt is not closed")
    outer_iteration = int(receipt.get("outer_iteration") or 0)
    if outer_iteration != int(row.get("outer_iteration") or 0):
        raise ValueError("runtime receipt outer iteration disagrees with target row")
    prefix_end = int(receipt.get("occurrence_sequence_end_inclusive") or 0)
    if prefix_end <= 0:
        raise ValueError("runtime receipt lacks a positive occurrence boundary")

    ledger_member = result_member.rsplit("/", 1)[0] + "/estimator_call_ledger.json"
    postrun = summarize_estimator_occurrence_prefix(
        _ledger_occurrences(archive_path, member_name=ledger_member),
        occurrence_sequence_end_inclusive=prefix_end,
    )
    runtime_raw = receipt.get("cumulative_raw_occurrences")
    runtime_unique = receipt.get("cumulative_unique_primitives")
    if not isinstance(runtime_raw, Mapping) or not isinstance(
        runtime_unique, Mapping
    ):
        raise ValueError("runtime receipt lacks cumulative accounting")
    runtime_raw_components = _component_map(
        runtime_raw.get("components"),
        label="runtime raw occurrences",
    )
    runtime_unique_components = _component_map(
        runtime_unique.get("components"),
        label="runtime unique primitives",
    )
    postrun_raw = postrun["cumulative_raw_occurrences"]
    postrun_unique = postrun["cumulative_unique_primitives"]
    postrun_raw_components = _component_map(
        postrun_raw.get("components"),
        label="post-run raw occurrences",
    )
    postrun_unique_components = _component_map(
        postrun_unique.get("components"),
        label="post-run unique primitives",
    )
    paper_components = _component_map(
        row.get("S_alg_components"),
        label="paper target-prefix components",
    )
    paper_receipt = row.get("S_alg_receipt")
    if (
        not isinstance(paper_receipt, Mapping)
        or paper_receipt.get("schema") != PAPER_I_S_ALG_ACCOUNTING_SCHEMA
        or paper_receipt.get("contract") != PAPER_I_S_ALG_CONTRACT
        or int(paper_receipt.get("accepted_prefix_length") or 0)
        != history_position
    ):
        raise ValueError("paper target-prefix row lacks a clean S_alg receipt")
    clean_recount = snake_clean_prefix_work(
        history=history,
        accepted_prefix_length=history_position,
        representation=str(paper_receipt.get("representation") or ""),
        estimator_ledger_receipt=receipt,
    )
    clean_components = _component_map(
        clean_recount.get("components"),
        label="clean algorithmic recount",
    )
    receipt_components = _component_map(
        paper_receipt.get("components"),
        label="paper clean S_alg receipt",
    )

    runtime_s_unique = int(
        runtime_unique.get("S_unique", runtime_unique.get("S_alg", -1))
    )
    postrun_s_unique = int(
        postrun_unique.get("S_unique", postrun_unique.get("S_alg", -1))
    )
    paper_s_alg = int(row.get("S_alg", -1))
    if (
        int(runtime_raw.get("total", -1)) != int(postrun_raw.get("total", -2))
        or runtime_raw_components != postrun_raw_components
        or runtime_unique_components != postrun_unique_components
        or runtime_s_unique != postrun_s_unique
        or clean_components != paper_components
        or receipt_components != paper_components
        or int(clean_recount.get("S_alg", -1)) != paper_s_alg
        or int(paper_receipt.get("S_alg", -1)) != paper_s_alg
        or sum(paper_components.values()) != paper_s_alg
    ):
        raise ValueError(
            "runtime/post-run diagnostics or clean paper S_alg recount disagree"
        )

    runtime_projection = {
        "occurrence_sequence_end_inclusive": prefix_end,
        "cumulative_raw_occurrences": {
            "components": runtime_raw_components,
            "total": int(runtime_raw["total"]),
        },
        "cumulative_unique_primitives": {
            "components": runtime_unique_components,
            "S_unique": runtime_s_unique,
        },
    }
    return {
        "route_id": str(row.get("route_id")),
        "regime": str(row.get("regime")),
        "history_position": history_position,
        "outer_iteration": outer_iteration,
        "status": "pass",
        "archive": {
            "path": _display_path(archive_path),
            "sha256": observed_archive_sha256,
        },
        "result_member": result_member,
        "ledger_member": ledger_member,
        "runtime_receipt_sha256": _canonical_sha256(runtime_projection),
        "postrun_summary_sha256": _canonical_sha256(postrun),
        "clean_recount_sha256": _canonical_sha256(clean_recount),
        "primitive_set_sha256": str(postrun["primitive_set_sha256"]),
        "S_alg": paper_s_alg,
        "components": paper_components,
        "raw_occurrence_count": int(postrun_raw["total"]),
        "runtime_unique_primitive_count": runtime_s_unique,
        "closure": {
            "runtime_occurrences_equal_postrun": True,
            "runtime_unique_identities_equal_postrun": True,
            "clean_recount_equals_paper_target_prefix": True,
            "clean_componentwise_equal": True,
            "source_archive_hash_matched": True,
        },
    }


def build_audit(
    *,
    target_json: Path,
    output_json: Path,
    route_id: str,
) -> dict[str, Any]:
    target_path = target_json.resolve()
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("rows", [])
        if isinstance(row, Mapping) and str(row.get("route_id")) == route_id
    ]
    if len(rows) != 6:
        raise ValueError(
            f"expected six completed target-prefix rows for {route_id}, got {len(rows)}"
        )
    audited = [_audit_row(row) for row in sorted(rows, key=lambda item: str(item["regime"]))]
    result = {
        "schema": AUDIT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "route_id": route_id,
        "target_prefix_source": {
            "path": _display_path(target_path),
            "sha256": _sha256(target_path),
        },
        "counting_contract": {
            "definition": (
                "reconstruct clean required logical estimator invocations "
                "from signed history; independently close raw occurrences "
                "and unique physical identities as runtime diagnostics"
            ),
            "components": list(S_ALG_COMPONENTS),
            "runtime_and_postrun_implementation": (
                "summarize_estimator_occurrence_prefix"
            ),
            "paper_recount_implementation": "snake_clean_prefix_work",
            "trusts_serialized_cumulative_totals": False,
        },
        "rows": audited,
        "summary": {
            "row_count": len(audited),
            "all_runtime_diagnostics_equal_postrun": True,
            "all_clean_recounts_equal_paper_target_prefix": True,
        },
    }
    output_path = output_json.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-json", type=Path, default=DEFAULT_TARGET_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--route-id", default=DEFAULT_ROUTE_ID)
    args = parser.parse_args()
    result = build_audit(
        target_json=args.target_json,
        output_json=args.output_json,
        route_id=str(args.route_id),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
