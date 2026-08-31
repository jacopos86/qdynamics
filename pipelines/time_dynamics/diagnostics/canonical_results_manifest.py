"""Bind the canonical Paper-II results PDF to its current verified inputs.

The visible campaign PDF is assembled from an established LaTeX layout and a
generated promotion-preview fragment.  This module owns the final fail-closed
manifest update: it verifies the ledger/preview linkage, refreshes all derived
counts, binds the compiled PDF and TeX digests, and records the active policy
transition without rediscovering raw runs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pypdf import PdfReader

from pipelines.time_dynamics.diagnostics.promotion_contract import (
    PROMOTION_PREVIEW_SCHEMA,
    REPORT_SCHEMA,
)
from pipelines.time_dynamics.staged_campaign import (
    CampaignLedger,
    METHOD_APM,
    METHOD_AVQDS,
    STATUS_ACCURACY_QUALIFIED,
    STATUS_REQUIRES_THRESHOLD_EXTENSION,
    STATUS_SAFEGUARD_TRIPPED,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return payload


def _verified_artifact(
    artifact: Mapping[str, Any], *, workspace_root: Path, label: str
) -> Path:
    try:
        path = Path(str(artifact["path"]))
        expected = str(artifact["sha256"])
    except KeyError as exc:
        raise ValueError(f"{label} lacks path/SHA-256") from exc
    target = path if path.is_absolute() else workspace_root / path
    target = target.resolve()
    if not target.is_file() or target.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {target}")
    observed = _sha256(target)
    if observed != expected:
        raise ValueError(
            f"{label} digest mismatch: expected {expected}, observed {observed}"
        )
    return target


def _slot_counts(preview: Mapping[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    tuples = list(preview.get("planned_tuples") or ())
    if int(preview.get("planned_tuple_count", -1)) != len(tuples):
        raise ValueError("promotion preview planned-tuple count is inconsistent")
    for item in tuples:
        methods = dict(item.get("methods") or {})
        if set(methods) != {METHOD_APM, METHOD_AVQDS}:
            raise ValueError("planned tuple does not contain exactly APM and AVQDS")
        for method in (METHOD_APM, METHOD_AVQDS):
            state = str(dict(methods[method]).get("paper_status") or "")
            if state not in {"qualified", "candidate", "pending", "unresolved"}:
                raise ValueError(f"unsupported planned-slot state {state!r}")
            counts[state] += 1
    return counts


def bind_canonical_results_manifest(
    *,
    ledger_path: Path,
    preview_receipt_path: Path,
    report_manifest_path: Path,
    report_tex_path: Path,
    report_pdf_path: Path,
    policy_transition_receipt_path: Path,
    workspace_root: Path = Path("."),
) -> dict[str, Any]:
    """Refresh one established report manifest after a routine rebuild."""

    workspace_root = workspace_root.resolve()
    ledger_path = ledger_path.resolve()
    preview_receipt_path = preview_receipt_path.resolve()
    report_manifest_path = report_manifest_path.resolve()
    report_tex_path = report_tex_path.resolve()
    report_pdf_path = report_pdf_path.resolve()
    policy_transition_receipt_path = policy_transition_receipt_path.resolve()

    ledger = CampaignLedger.read(ledger_path)
    preview = _read_object(preview_receipt_path, label="promotion preview")
    manifest = _read_object(report_manifest_path, label="campaign report manifest")
    transition = _read_object(
        policy_transition_receipt_path, label="policy transition receipt"
    )
    if preview.get("schema") != PROMOTION_PREVIEW_SCHEMA:
        raise ValueError("promotion preview does not use the canonical schema")
    if manifest.get("schema") != REPORT_SCHEMA:
        raise ValueError("campaign report manifest does not use the canonical schema")

    ledger_sha = _sha256(ledger_path)
    preview_ledger = dict(preview.get("ledger") or {})
    if preview_ledger.get("sha256") != ledger_sha:
        raise ValueError("promotion preview is stale against the canonical ledger")
    if int(preview_ledger.get("record_count", -1)) != len(ledger.records):
        raise ValueError("promotion preview record count differs from the ledger")
    preview_policy = dict(preview.get("qualification_policy") or {})
    if preview_policy != dict(ledger.qualification_policy):
        raise ValueError("promotion preview policy differs from the ledger")
    _verified_artifact(
        preview.get("tex_fragment") or {},
        workspace_root=workspace_root,
        label="promotion-preview TeX fragment",
    )
    if not report_tex_path.is_file() or not report_pdf_path.is_file():
        raise ValueError("compiled report TeX/PDF is missing")
    if not report_tex_path.stat().st_size or not report_pdf_path.stat().st_size:
        raise ValueError("compiled report TeX/PDF is empty")

    after = dict(transition.get("ledger_after") or {})
    if after.get("sha256") != ledger_sha or int(after.get("record_count", -1)) != len(
        ledger.records
    ):
        raise ValueError("policy transition does not bind the current ledger")
    policy_after = dict(transition.get("policy_after") or {})
    if policy_after.get("qualification_policy_sha256") != _mapping_sha256(
        ledger.qualification_policy
    ):
        raise ValueError("policy transition does not bind the exact ledger policy")
    _verified_artifact(
        policy_after.get("response_scale_sidecar") or {},
        workspace_root=workspace_root,
        label="response-scale sidecar",
    )
    for index, artifact in enumerate(transition.get("policy_source_artifacts") or ()):
        _verified_artifact(
            artifact,
            workspace_root=workspace_root,
            label=f"policy source artifact {index}",
        )

    records = tuple(ledger.records)
    t5 = tuple(record for record in records if record.horizon == "t5")
    t20 = tuple(record for record in records if record.horizon == "t20")
    t5_status = Counter(record.status for record in t5)
    t20_status = Counter(record.status for record in t20)
    slots = _slot_counts(preview)
    pairs = list(preview.get("pairs") or ())
    if int(preview.get("pair_count", -1)) != len(pairs):
        raise ValueError("promotion preview pair count is inconsistent")
    qualified_pairs = sum(
        all(
            str(dict(dict(pair.get("rows") or {})[method]).get("status"))
            == STATUS_ACCURACY_QUALIFIED
            for method in (METHOD_APM, METHOD_AVQDS)
        )
        for pair in pairs
    )
    candidate_pairs = len(pairs) - qualified_pairs
    reference_shas = {
        str(dict(item.get("exact_reference") or {}).get("sha256") or "")
        for item in preview.get("planned_tuples") or ()
    }
    if "" in reference_shas:
        raise ValueError("planned tuple lacks its exact-reference digest")

    preview_fragment = dict(preview["tex_fragment"])
    transition_before = dict(transition.get("ledger_before") or {})
    decision_handoff = dict(transition.get("decision_handoff") or {})
    manifest.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ledger_record_count": len(records),
            "verified_frontier_row_count": len(records),
            "source_ledger": str(ledger_path.relative_to(workspace_root)),
            "source_ledger_sha256": ledger_sha,
            "qualification_policy": preview_policy,
            "qualified_pair_count": qualified_pairs,
            "qualified_pair_horizon": "t20_response_normalized_or_matched_pair_v2",
            "pairs": pairs,
            "exact_reference_artifact_count": len(reference_shas),
            "t20_accuracy_qualified_method_arm_count": slots["qualified"],
            "t20_candidate_method_arm_count": slots["candidate"],
            "t20_pending_method_arm_count": slots["pending"],
            "t20_unresolved_method_arm_count": slots["unresolved"],
            "t20_qualified_pair_count": qualified_pairs,
            "t20_candidate_pair_count": candidate_pairs,
            "t20_evaluated_trajectory_count": sum(
                record.metrics is not None for record in t20
            ),
            "t20_requires_threshold_extension_count": t20_status[
                STATUS_REQUIRES_THRESHOLD_EXTENSION
            ],
            "t20_safeguard_count": t20_status[STATUS_SAFEGUARD_TRIPPED],
            "t5_accuracy_qualified_apm_count": sum(
                record.status == STATUS_ACCURACY_QUALIFIED
                and record.key.method == METHOD_APM
                for record in t5
            ),
            "t5_accuracy_qualified_avqds_count": sum(
                record.status == STATUS_ACCURACY_QUALIFIED
                and record.key.method == METHOD_AVQDS
                for record in t5
            ),
            "t5_requires_threshold_extension_count": t5_status[
                STATUS_REQUIRES_THRESHOLD_EXTENSION
            ],
            "t5_safeguard_count": t5_status[STATUS_SAFEGUARD_TRIPPED],
            "promotion_preview": {
                "pair_count": len(pairs),
                "planned_tuple_count": int(preview["planned_tuple_count"]),
                "receipt": str(preview_receipt_path.relative_to(workspace_root)),
                "receipt_sha256": _sha256(preview_receipt_path),
                "status": str(preview.get("status") or ""),
                "tex_fragment": str(
                    Path(str(preview_fragment["path"]))
                ),
                "tex_fragment_sha256": str(preview_fragment["sha256"]),
            },
            "policy_transition": {
                "status": str(transition.get("status") or ""),
                "receipt": str(
                    policy_transition_receipt_path.relative_to(workspace_root)
                ),
                "receipt_sha256": _sha256(policy_transition_receipt_path),
                "preserved_ledger": str(transition_before.get("path") or ""),
                "preserved_ledger_sha256": str(
                    transition_before.get("sha256") or ""
                ),
                "decision_handoff": decision_handoff,
                "source_revision": str(
                    transition.get("source_revision_before_uncommitted_repair") or ""
                ),
            },
            "tex": str(report_tex_path.relative_to(workspace_root)),
            "tex_sha256": _sha256(report_tex_path),
            "pdf": str(report_pdf_path.relative_to(workspace_root)),
            "pdf_bytes": report_pdf_path.stat().st_size,
            "pdf_sha256": _sha256(report_pdf_path),
            "page_count": len(PdfReader(report_pdf_path).pages),
        }
    )
    markdown_path = Path(str(manifest.get("markdown") or ""))
    markdown = (
        markdown_path
        if markdown_path.is_absolute()
        else workspace_root / markdown_path
    )
    if markdown.is_file() and markdown.stat().st_size > 0:
        manifest["markdown_sha256"] = _sha256(markdown)

    temporary = report_manifest_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(report_manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--preview-receipt", required=True, type=Path)
    parser.add_argument("--report-manifest", required=True, type=Path)
    parser.add_argument("--report-tex", required=True, type=Path)
    parser.add_argument("--report-pdf", required=True, type=Path)
    parser.add_argument("--policy-transition-receipt", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    manifest = bind_canonical_results_manifest(
        ledger_path=args.ledger,
        preview_receipt_path=args.preview_receipt,
        report_manifest_path=args.report_manifest,
        report_tex_path=args.report_tex,
        report_pdf_path=args.report_pdf,
        policy_transition_receipt_path=args.policy_transition_receipt,
        workspace_root=args.workspace_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["bind_canonical_results_manifest", "main"]
