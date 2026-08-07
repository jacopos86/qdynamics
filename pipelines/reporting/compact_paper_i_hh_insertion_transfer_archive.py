#!/usr/bin/env python3
"""Preserve the report-relevant content of one large Paper-I transfer archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tarfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import ijson


REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_SCALARS = frozenset(
    {
        "accepted",
        "accepted_admission",
        "delta_abs_current",
        "depth",
        "energy_after_opt",
        "logical_num_parameters_after_opt",
        "logical_parameters_added_this_step",
        "nfev_opt",
        "parameters_added_this_step",
        "selected_logical_op",
        "selected_op",
        "selected_position",
    }
)
RECEIPT_COMPONENTS = ("N_H_outer", "N_H_refit", "N_grad", "N_metric")


ARCHIVE_PATTERN = re.compile(
    r"^(?P<cluster>\d+)\.(?P<proc>\d+)__(?P<regime>.+)_transfer\.tar\.gz$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _member_name(archive: Path, regime: str) -> str:
    suffix = f"/{regime}/json/current.json"
    with tarfile.open(archive, "r:gz") as bundle:
        matches = [name for name in bundle.getnames() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"{archive.name}: expected one {suffix}, found {matches!r}")
    return matches[0]


def _extract_compact_current(
    *,
    archive: Path,
    member_name: str,
) -> dict[str, Any]:
    state_builder: Any | None = None
    state_depth = 0
    settings_builder: Any | None = None
    settings_depth = 0
    nested_builder: Any | None = None
    nested_depth = 0
    nested_target: str | None = None
    current_row: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    adapt_scalars: dict[str, Any] = {}
    state: dict[str, Any] | None = None
    settings: dict[str, Any] | None = None

    with tarfile.open(archive, "r:gz") as bundle:
        stream = bundle.extractfile(member_name)
        if stream is None:
            raise FileNotFoundError(member_name)
        for prefix, event, raw_value in ijson.parse(stream):
            value = _plain(raw_value)

            if state_builder is not None:
                state_builder.event(event, raw_value)
                if event in {"start_map", "start_array"}:
                    state_depth += 1
                elif event in {"end_map", "end_array"}:
                    state_depth -= 1
                    if state_depth == 0:
                        state = _plain(state_builder.value)
                        state_builder = None
                continue
            if prefix == "ansatz_input_state" and event == "start_map":
                state_builder = ijson.common.ObjectBuilder()
                state_builder.event(event, raw_value)
                state_depth = 1
                continue

            if settings_builder is not None:
                settings_builder.event(event, raw_value)
                if event in {"start_map", "start_array"}:
                    settings_depth += 1
                elif event in {"end_map", "end_array"}:
                    settings_depth -= 1
                    if settings_depth == 0:
                        settings = _plain(settings_builder.value)
                        settings_builder = None
                continue
            if prefix == "settings" and event == "start_map":
                settings_builder = ijson.common.ObjectBuilder()
                settings_builder.event(event, raw_value)
                settings_depth = 1
                continue

            if nested_builder is not None:
                nested_builder.event(event, raw_value)
                if event in {"start_map", "start_array"}:
                    nested_depth += 1
                elif event in {"end_map", "end_array"}:
                    nested_depth -= 1
                    if nested_depth == 0:
                        if current_row is None or nested_target is None:
                            raise RuntimeError("history nested-object state drift")
                        current_row[nested_target] = _plain(nested_builder.value)
                        nested_builder = None
                        nested_target = None
                continue

            if prefix == "adapt_vqe.history.item" and event == "start_map":
                if history:
                    history[-1].pop("_ordered_active_operators", None)
                current_row = {}
                continue
            if prefix == "adapt_vqe.history.item" and event == "end_map":
                if current_row is None:
                    raise RuntimeError("history row closed without opening")
                history.append(current_row)
                current_row = None
                continue
            if current_row is not None:
                base = "adapt_vqe.history.item."
                if prefix.startswith(base):
                    relative = prefix[len(base) :]
                    nested_key = (
                        "_ordered_active_operators"
                        if relative
                        == "active_prefix_checkpoint.ordered_active_operators"
                        else relative
                    )
                    if (
                        relative
                        in {
                            "selected_records",
                            "post_admission_prune",
                            "active_prefix_checkpoint.ordered_active_operators",
                        }
                        and event in {"start_array", "start_map"}
                    ):
                        nested_builder = ijson.common.ObjectBuilder()
                        nested_builder.event(event, raw_value)
                        nested_depth = 1
                        nested_target = nested_key
                        continue
                    if "." not in relative and relative in HISTORY_SCALARS and event in {
                        "boolean",
                        "number",
                        "string",
                        "null",
                    }:
                        current_row[relative] = value
                        continue
                    receipt_base = (
                        "active_prefix_checkpoint.estimator_ledger_receipt."
                    )
                    if relative.startswith(receipt_base) and event in {
                        "number",
                        "string",
                    }:
                        receipt_relative = relative[len(receipt_base) :]
                        receipt = current_row.setdefault(
                            "_compact_estimator_receipt",
                            {
                                "cumulative_raw_occurrences": {
                                    "components": {}
                                }
                            },
                        )
                        if receipt_relative in {"status", "outer_iteration"}:
                            receipt[receipt_relative] = value
                        component_base = "cumulative_raw_occurrences.components."
                        if receipt_relative.startswith(component_base):
                            component = receipt_relative[len(component_base) :]
                            if component in RECEIPT_COMPONENTS:
                                receipt["cumulative_raw_occurrences"]["components"][
                                    component
                                ] = value
                        elif receipt_relative == "cumulative_raw_occurrences.total":
                            receipt["cumulative_raw_occurrences"]["total"] = value
                continue

            if prefix.startswith("adapt_vqe.") and "." not in prefix[len("adapt_vqe.") :]:
                key = prefix[len("adapt_vqe.") :]
                if key in {
                    "abs_delta_e",
                    "adapt_beam_enabled",
                    "ansatz_depth",
                    "exact_gs_energy",
                    "method",
                    "success",
                } and event in {"boolean", "number", "string", "null"}:
                    adapt_scalars[key] = value

    if not history or state is None or settings is None:
        raise ValueError(f"{archive.name}: compact current extraction is incomplete")
    if int(adapt_scalars.get("ansatz_depth", -1)) != len(history):
        raise ValueError(f"{archive.name}: history/depth mismatch")
    return {
        "adapt_vqe": {
            **adapt_scalars,
            "history": history,
            "history_count": len(history),
            "history_tail_count": 0,
        },
        "ansatz_input_state": state,
        "settings": settings,
    }


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    archive = args.archive.resolve()
    match = ARCHIVE_PATTERN.match(archive.name)
    if match is None:
        raise ValueError(f"unexpected transfer archive name: {archive.name}")
    cluster_id = int(match.group("cluster"))
    proc_id = int(match.group("proc"))
    regime = match.group("regime")
    output = (
        args.output.resolve()
        if args.output is not None
        else archive.with_name(
            f"{cluster_id}.{proc_id}__{regime}_compact.json.gz"
        )
    )

    member = _member_name(archive, regime)
    payload = _extract_compact_current(
        archive=archive,
        member_name=member,
    )
    compact = {
        "schema": "paper_i_hh_insertion_compact_transfer_evidence_v1",
        "identity": {
            "cluster_id": cluster_id,
            "proc_id": proc_id,
            "regime": regime,
        },
        "source": {
            "archive": str(archive.relative_to(REPO_ROOT)),
            "archive_sha256": _sha256(archive),
            "member": member,
        },
        "payload_sha256": _canonical_sha256(payload),
        "payload": payload,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(compact, handle, sort_keys=True, separators=(",", ":"))

    with gzip.open(output, "rt", encoding="utf-8") as handle:
        observed = json.load(handle)
    if observed["payload_sha256"] != _canonical_sha256(observed["payload"]):
        raise ValueError("compact evidence payload hash mismatch")
    if observed["source"]["archive_sha256"] != _sha256(archive):
        raise ValueError("compact evidence source archive hash mismatch")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
