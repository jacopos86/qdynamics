#!/usr/bin/env python3
"""Motif extraction and tiling helpers for HH continuation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_types import MotifLibrary, MotifMetadata, MotifRecord
from src.quantum.ansatz_parameterization import deserialize_layout, project_runtime_theta_block_mean


def _boundary_behavior_from_sites(
    support_sites: Sequence[int],
    *,
    num_sites: int,
) -> str:
    if int(num_sites) <= 1 or not support_sites:
        return "interior_only"
    sites = sorted({int(x) for x in support_sites})
    touches_left = 0 in sites
    touches_right = (int(num_sites) - 1) in sites
    if touches_left and touches_right:
        return "both_edges"
    if touches_left:
        return "left_edge"
    if touches_right:
        return "right_edge"
    return "interior_only"


def _candidate_boundary_behavior(
    generator_metadata: Mapping[str, Any] | None,
    *,
    target_num_sites: int,
) -> str:
    if not isinstance(generator_metadata, Mapping):
        return "interior_only"
    sites = generator_metadata.get("support_sites", [])
    if not isinstance(sites, Sequence):
        return "interior_only"
    return _boundary_behavior_from_sites(
        [int(x) for x in sites],
        num_sites=int(target_num_sites),
    )


def _boundary_behavior_matches(
    source_behavior: str,
    target_behavior: str,
    *,
    transfer_mode: str,
) -> bool:
    mode_key = str(transfer_mode).strip().lower()
    if mode_key == "exact_match_v1":
        return True
    if mode_key != "boundary_v1":
        raise ValueError("transfer_mode must be one of {'exact_match_v1','boundary_v1'}.")
    src = str(source_behavior or "interior_only")
    tgt = str(target_behavior or "interior_only")
    if src == "both_edges":
        return bool(tgt == "both_edges")
    if src == "left_edge":
        return bool(tgt in {"left_edge", "both_edges"})
    if src == "right_edge":
        return bool(tgt in {"right_edge", "both_edges"})
    return bool(tgt == "interior_only")


def merge_motif_libraries(
    libraries: Sequence[Mapping[str, Any] | None],
) -> dict[str, Any] | None:
    valid = [dict(lib) for lib in libraries if isinstance(lib, Mapping)]
    if not valid:
        return None
    source_tags: list[str] = []
    ordering = str(valid[0].get("ordering", "blocked"))
    boson_encoding = str(valid[0].get("boson_encoding", "binary"))
    merged: dict[tuple[str, str, tuple[int, ...], str], dict[str, Any]] = {}
    for lib in valid:
        lib_ordering = str(lib.get("ordering", ordering))
        lib_boson_encoding = str(lib.get("boson_encoding", boson_encoding))
        if lib_ordering != ordering or lib_boson_encoding != boson_encoding:
            raise ValueError(
                "Cannot merge motif libraries with mismatched ordering/boson_encoding: "
                f"expected ({ordering}, {boson_encoding}), got ({lib_ordering}, {lib_boson_encoding})."
            )
        for tag in lib.get("source_tags", [lib.get("source_tag", "payload")]):
            tag_s = str(tag)
            if tag_s and tag_s not in source_tags:
                source_tags.append(tag_s)
        records = lib.get("records", [])
        if not isinstance(records, Sequence):
            continue
        for rec in records:
            if not isinstance(rec, Mapping):
                continue
            key = (
                str(rec.get("family_id", "")),
                str(rec.get("template_id", "")),
                tuple(int(x) for x in rec.get("support_site_offsets", [])),
                str(rec.get("boundary_behavior", "interior_only")),
            )
            bucket = merged.setdefault(
                key,
                {
                    "family_id": str(rec.get("family_id", "")),
                    "template_id": str(rec.get("template_id", "")),
                    "support_site_offsets": [int(x) for x in rec.get("support_site_offsets", [])],
                    "boundary_behavior": str(rec.get("boundary_behavior", "interior_only")),
                    "source_num_sites": int(rec.get("source_num_sites", lib.get("source_num_sites", 0))),
                    "relative_order": int(rec.get("relative_order", 0)),
                    "generator_ids": [],
                    "symmetry_spec": (
                        dict(rec.get("symmetry_spec", {}))
                        if isinstance(rec.get("symmetry_spec"), Mapping)
                        else None
                    ),
                    "source_tags": [],
                    "theta_vals": [],
                    "theta_abs_vals": [],
                },
            )
            bucket["relative_order"] = int(min(bucket["relative_order"], int(rec.get("relative_order", 0))))
            bucket["source_num_sites"] = int(max(bucket["source_num_sites"], int(rec.get("source_num_sites", lib.get("source_num_sites", 0)))))
            bucket["theta_vals"].append(float(rec.get("mean_theta", 0.0)))
            bucket["theta_abs_vals"].append(float(rec.get("mean_abs_theta", abs(float(rec.get("mean_theta", 0.0))))))
            for gid in rec.get("generator_ids", []):
                gid_s = str(gid)
                if gid_s and gid_s not in bucket["generator_ids"]:
                    bucket["generator_ids"].append(gid_s)
            for tag in rec.get("source_tags", [lib.get("source_tag", "payload")]):
                tag_s = str(tag)
                if tag_s and tag_s not in bucket["source_tags"]:
                    bucket["source_tags"].append(tag_s)
    records_out: list[MotifRecord] = []
    for idx, bucket in enumerate(
        sorted(
            merged.values(),
            key=lambda rec: (
                int(rec.get("relative_order", 0)),
                str(rec.get("family_id", "")),
                str(rec.get("template_id", "")),
                str(rec.get("boundary_behavior", "interior_only")),
            ),
        )
    ):
        theta_vals = [float(x) for x in bucket.pop("theta_vals", [])]
        theta_abs_vals = [float(x) for x in bucket.pop("theta_abs_vals", [])]
        mean_theta = float(sum(theta_vals) / max(1, len(theta_vals)))
        mean_abs_theta = float(sum(theta_abs_vals) / max(1, len(theta_abs_vals)))
        sign_hint = 0
        if mean_theta > 0.0:
            sign_hint = 1
        elif mean_theta < 0.0:
            sign_hint = -1
        digest = hashlib.sha1(
            (
                f"{bucket['family_id']}|{bucket['template_id']}|{bucket['support_site_offsets']}|"
                f"{bucket['boundary_behavior']}|{bucket['source_tags']}|{idx}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        records_out.append(
            MotifRecord(
                motif_id=f"motif:{digest}",
                family_id=str(bucket.get("family_id", "")),
                template_id=str(bucket.get("template_id", "")),
                source_num_sites=int(bucket.get("source_num_sites", 0)),
                relative_order=int(bucket.get("relative_order", 0)),
                support_site_offsets=[int(x) for x in bucket.get("support_site_offsets", [])],
                mean_theta=float(mean_theta),
                mean_abs_theta=float(mean_abs_theta),
                sign_hint=int(sign_hint),
                generator_ids=[str(x) for x in bucket.get("generator_ids", [])],
                symmetry_spec=(dict(bucket.get("symmetry_spec", {})) if isinstance(bucket.get("symmetry_spec"), Mapping) else None),
                boundary_behavior=str(bucket.get("boundary_behavior", "interior_only")),
                source_tags=[str(x) for x in bucket.get("source_tags", [])],
            )
        )
    merged_lib = MotifLibrary(
        library_version="phase3_motif_library_v2",
        source_tag=(str(source_tags[0]) if source_tags else str(valid[0].get("source_tag", "payload"))),
        source_num_sites=int(max(int(lib.get("source_num_sites", 0)) for lib in valid)),
        ordering=str(ordering),
        boson_encoding=str(boson_encoding),
        source_tags=[str(x) for x in source_tags],
        records=list(records_out),
    )
    return asdict(merged_lib)


def extract_motif_library(
    *,
    generator_metadata: Sequence[Mapping[str, Any]],
    theta: Sequence[float],
    source_num_sites: int,
    source_tag: str,
    ordering: str,
    boson_encoding: str,
) -> dict[str, Any]:
    records: list[MotifRecord] = []
    for idx, (meta, theta_val) in enumerate(zip(generator_metadata, theta)):
        family_id = str(meta.get("family_id", "unknown"))
        template_id = str(meta.get("template_id", "unknown"))
        support_site_offsets = [int(x) for x in meta.get("support_site_offsets", [])]
        support_sites = [int(x) for x in meta.get("support_sites", [])] if isinstance(meta.get("support_sites", []), Sequence) else []
        boundary_behavior = _boundary_behavior_from_sites(
            support_sites,
            num_sites=int(source_num_sites),
        )
        digest = hashlib.sha1(
            (
                f"{family_id}|{template_id}|{support_site_offsets}|{boundary_behavior}|{idx}|{source_num_sites}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        sign_hint = 0
        theta_f = float(theta_val)
        if theta_f > 0.0:
            sign_hint = 1
        elif theta_f < 0.0:
            sign_hint = -1
        records.append(
            MotifRecord(
                motif_id=f"motif:{digest}",
                family_id=str(family_id),
                template_id=str(template_id),
                source_num_sites=int(source_num_sites),
                relative_order=int(idx),
                support_site_offsets=[int(x) for x in support_site_offsets],
                mean_theta=float(theta_f),
                mean_abs_theta=float(abs(theta_f)),
                sign_hint=int(sign_hint),
                generator_ids=[str(meta.get("generator_id", ""))] if meta.get("generator_id") else [],
                symmetry_spec=(
                    dict(meta.get("symmetry_spec", {}))
                    if isinstance(meta.get("symmetry_spec"), Mapping)
                    else None
                ),
                boundary_behavior=str(boundary_behavior),
                source_tags=[str(source_tag)],
            )
        )
    library = MotifLibrary(
        library_version="phase3_motif_library_v1",
        source_tag=str(source_tag),
        source_num_sites=int(source_num_sites),
        ordering=str(ordering),
        boson_encoding=str(boson_encoding),
        source_tags=[str(source_tag)],
        records=list(records),
    )
    return asdict(library)


def load_motif_library_from_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    continuation = payload.get("continuation", None) if isinstance(payload, Mapping) else None
    if isinstance(continuation, Mapping):
        motif_library = continuation.get("motif_library", None)
        if isinstance(motif_library, Mapping):
            return dict(motif_library)
        if isinstance(motif_library, Sequence):
            return merge_motif_libraries([x for x in motif_library if isinstance(x, Mapping)])
        generator_metadata = continuation.get("selected_generator_metadata", None)
        adapt_block = payload.get("adapt_vqe", {}) if isinstance(payload.get("adapt_vqe", {}), Mapping) else {}
        optimal_point = adapt_block.get("optimal_point", None)
        logical_optimal_point = adapt_block.get("logical_optimal_point", None)
        parameterization = adapt_block.get("parameterization", None)
        settings = payload.get("settings", None)
        theta_logical: list[float] | None = None
        if isinstance(logical_optimal_point, Sequence):
            theta_logical = [float(x) for x in logical_optimal_point]
        elif isinstance(parameterization, Mapping) and isinstance(optimal_point, Sequence):
            try:
                layout = deserialize_layout(parameterization)
                theta_runtime = np.asarray([float(x) for x in optimal_point], dtype=float)
                theta_logical = [float(x) for x in project_runtime_theta_block_mean(theta_runtime, layout)]
            except Exception:
                theta_logical = None
        elif isinstance(optimal_point, Sequence):
            theta_logical = [float(x) for x in optimal_point]
        if isinstance(generator_metadata, Sequence) and theta_logical is not None and isinstance(settings, Mapping):
            return extract_motif_library(
                generator_metadata=[dict(x) for x in generator_metadata if isinstance(x, Mapping)],
                theta=theta_logical,
                source_num_sites=int(settings.get("L", 0)),
                source_tag=str(payload.get("generated_utc", "source_payload")),
                ordering=str(settings.get("ordering", "blocked")),
                boson_encoding=str(settings.get("boson_encoding", "binary")),
            )
    return None


def load_motif_library_from_json(path: str | Path) -> dict[str, Any] | None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_motif_library_from_payload(payload)


def _is_nonstring_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _int_list_or_empty(value: Any) -> list[int]:
    if not _is_nonstring_sequence(value):
        return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _str_list_or_empty(value: Any) -> list[str]:
    if not _is_nonstring_sequence(value):
        return []
    out: list[str] = []
    for item in value:
        item_s = _clean_optional_str(item)
        if item_s and item_s not in out:
            out.append(item_s)
    return out


_ABSENT_STRINGS = {"", "none", "null", "nan"}
_BROAD_POOL_FAMILY_IDS = {
    "full_meta",
    "pareto_lean",
    "pareto_lean_l2",
    "pareto_lean_l3",
    "pareto_lean_gate_pruned",
    "family_max",
}


def _clean_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _ABSENT_STRINGS:
        return None
    return text


def _label_operator_family_key(label: Any) -> str | None:
    """Return a transferable operator-family key from a concrete pool label.

    Historical-selected routing is meant to transfer *families* such as UCCSD,
    Hamiltonian quadratures, PAOP classes, or bosonic quadrature classes, not
    exact labels like ``uccsd_sing(alpha:0->1)``.  The labels are intentionally
    used here because several generic full_meta registries store only the broad
    pool key as ``family_id``.
    """

    raw = _clean_optional_str(label)
    if raw is None:
        return None
    if "::split[" in raw:
        raw = raw.split("::split[", 1)[0]
    if raw == "ham_full":
        return "ham_full"
    if raw.startswith("ham_term("):
        return "ham_term"
    if raw.startswith("hh_termwise_ham_unit_term("):
        return "hh_termwise_ham_unit_term"
    if raw.startswith("hh_termwise_ham_quadrature_term("):
        return "hh_termwise_ham_quadrature_term"
    if raw.startswith("uccsd_"):
        return "uccsd"
    for prefix in ("hop", "onsite", "pot"):
        if raw.startswith(f"{prefix}("):
            return prefix
    if "::" in raw:
        namespace, tail = raw.split("::", 1)
        tail = tail.split("(", 1)[0]
        if tail.startswith("uccsd_"):
            tail = "uccsd"
        else:
            # Drop site/index-only suffixes so x_0/x_1 and hop_0_1 transfer as
            # a single family, while semantic suffixes such as _left are kept.
            tail = re.sub(r"(?:_\d+)+$", "", tail)
        return f"{namespace}::{tail}" if tail else namespace
    if ":" in raw:
        namespace, tail = raw.split(":", 1)
        tail = tail.split("(", 1)[0]
        tail = re.sub(r"(?:_\d+)+$", "", tail)
        return f"{namespace}:{tail}" if tail else namespace
    stem = raw.split("(", 1)[0]
    stem = re.sub(r"(?:_\d+)+$", "", stem)
    return stem or None


def _specific_family_id(value: Any) -> str | None:
    family = _clean_optional_str(value)
    if family is None:
        return None
    if family.strip().lower() in _BROAD_POOL_FAMILY_IDS:
        return None
    return family


def _record_operator_family_ids(record: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("operator_family_id", "operator_family", "logical_family_id"):
        family = _clean_optional_str(record.get(key))
        if family and family not in out:
            out.append(family)
    family_id = _specific_family_id(record.get("family_id"))
    if family_id and family_id not in out:
        out.append(family_id)
    meta = record.get("generator_metadata")
    if isinstance(meta, Mapping):
        meta_family = _specific_family_id(meta.get("family_id"))
        if meta_family and meta_family not in out:
            out.append(meta_family)
        for label in _record_labels(meta):
            family = _label_operator_family_key(label)
            if family and family not in out:
                out.append(family)
    for label in _record_labels(record):
        family = _label_operator_family_key(label)
        if family and family not in out:
            out.append(family)
    return out


def _payload_source_tag(payload: Mapping[str, Any], default: str = "payload") -> str:
    for key in ("source_tag", "generated_utc", "artifact_json", "source_id", "record_id"):
        value = payload.get(key)
        text = _clean_optional_str(value)
        if text:
            return text
    return str(default)


def _selected_record_from_generator_metadata(
    meta: Mapping[str, Any],
    *,
    source_tag: str,
    source_kind: str,
    relative_order: int,
) -> dict[str, Any] | None:
    generator_id = _clean_optional_str(meta.get("generator_id"))
    label = (
        _clean_optional_str(meta.get("candidate_label"))
        or _clean_optional_str(meta.get("operator_label"))
        or _clean_optional_str(meta.get("label"))
    )
    family_id = _clean_optional_str(meta.get("family_id"))
    template_id = _clean_optional_str(meta.get("template_id"))
    offsets = _int_list_or_empty(meta.get("support_site_offsets", []))
    support_sites = _int_list_or_empty(meta.get("support_sites", []))
    if not any([generator_id, label, family_id, template_id, offsets]):
        return None
    boundary_behavior = _clean_optional_str(meta.get("boundary_behavior")) or ""
    if boundary_behavior == "" and support_sites:
        # Source-site count is optional in historical rows; when absent, the
        # structural matcher can still use family/template/offsets and treat the
        # boundary as interior-only.
        source_num_sites = meta.get("source_num_sites") or meta.get("num_sites")
        try:
            boundary_behavior = _boundary_behavior_from_sites(
                support_sites,
                num_sites=int(source_num_sites),
            )
        except Exception:
            boundary_behavior = "interior_only"
    if boundary_behavior == "":
        boundary_behavior = "interior_only"
    generator_ids = _str_list_or_empty(meta.get("generator_ids", []))
    if generator_id and generator_id not in generator_ids:
        generator_ids.insert(0, generator_id)
    operator_family = _record_operator_family_ids(
        {
            "generator_id": generator_id,
            "family_id": family_id,
            "template_id": template_id,
            "candidate_label": label,
            "operator_label": label,
            "generator_metadata": meta,
        }
    )
    return {
        "record_kind": "generator_metadata",
        "source_kind": str(source_kind),
        "source_tag": str(source_tag),
        "relative_order": int(relative_order),
        "generator_id": generator_id,
        "generator_ids": [str(x) for x in generator_ids],
        "family_id": family_id,
        "template_id": template_id,
        "operator_family_id": (operator_family[0] if operator_family else None),
        "operator_family_ids": [str(x) for x in operator_family],
        "candidate_label": label,
        "operator_label": label,
        "support_sites": [int(x) for x in support_sites],
        "support_site_offsets": [int(x) for x in offsets],
        "boundary_behavior": str(boundary_behavior),
        "generator_metadata": dict(meta),
    }


def _selected_record_from_label(
    label: Any,
    *,
    source_tag: str,
    source_kind: str,
    relative_order: int,
) -> dict[str, Any] | None:
    label_s = _clean_optional_str(label)
    if label_s is None:
        return None
    family = _label_operator_family_key(label_s)
    return {
        "record_kind": "operator_label",
        "source_kind": str(source_kind),
        "source_tag": str(source_tag),
        "relative_order": int(relative_order),
        "generator_id": None,
        "generator_ids": [],
        "family_id": None,
        "template_id": None,
        "operator_family_id": family,
        "operator_family_ids": ([family] if family else []),
        "candidate_label": str(label_s),
        "operator_label": str(label_s),
        "support_sites": [],
        "support_site_offsets": [],
        "boundary_behavior": "interior_only",
    }


def _selected_records_from_motif_library(
    motif_library: Mapping[str, Any],
    *,
    source_tag: str | None = None,
) -> list[dict[str, Any]]:
    records = motif_library.get("records", [])
    if not _is_nonstring_sequence(records):
        return []
    tag = str(source_tag or motif_library.get("source_tag") or "motif_library")
    out: list[dict[str, Any]] = []
    for idx, rec_raw in enumerate(records):
        if not isinstance(rec_raw, Mapping):
            continue
        generator_ids = _str_list_or_empty(rec_raw.get("generator_ids", []))
        generator_id = generator_ids[0] if generator_ids else None
        family_id = _clean_optional_str(rec_raw.get("family_id"))
        template_id = _clean_optional_str(rec_raw.get("template_id"))
        operator_family = _record_operator_family_ids(
            {
                "generator_id": generator_id,
                "family_id": family_id,
                "template_id": template_id,
                "candidate_label": rec_raw.get("candidate_label"),
                "operator_label": rec_raw.get("operator_label"),
            }
        )
        out.append(
            {
                "record_kind": "motif_record",
                "source_kind": "motif_library",
                "source_tag": str(tag),
                "relative_order": int(rec_raw.get("relative_order", idx)),
                "generator_id": generator_id,
                "generator_ids": [str(x) for x in generator_ids],
                "family_id": family_id,
                "template_id": template_id,
                "operator_family_id": (operator_family[0] if operator_family else None),
                "operator_family_ids": [str(x) for x in operator_family],
                "candidate_label": None,
                "operator_label": None,
                "support_sites": [],
                "support_site_offsets": _int_list_or_empty(rec_raw.get("support_site_offsets", [])),
                "boundary_behavior": str(rec_raw.get("boundary_behavior", "interior_only")),
                "motif_id": _clean_optional_str(rec_raw.get("motif_id")),
                "source_num_sites": int(rec_raw.get("source_num_sites", motif_library.get("source_num_sites", 0)) or 0),
                "motif_record": dict(rec_raw),
            }
        )
    return out


def _normalise_selected_records(
    records: Any,
    *,
    source_tag: str,
    source_kind: str,
) -> list[dict[str, Any]]:
    if not _is_nonstring_sequence(records):
        return []
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(records):
        rec: dict[str, Any] | None = None
        if isinstance(raw, Mapping):
            nested_meta = raw.get("generator_metadata")
            if isinstance(nested_meta, Mapping):
                rec = _selected_record_from_generator_metadata(
                    nested_meta,
                    source_tag=source_tag,
                    source_kind=source_kind,
                    relative_order=int(raw.get("relative_order", idx)),
                )
                if rec is not None:
                    for key in ("candidate_label", "operator_label", "selected_op", "label"):
                        if raw.get(key) not in {None, ""} and not rec.get("candidate_label"):
                            rec["candidate_label"] = str(raw.get(key))
                            rec["operator_label"] = str(raw.get(key))
                            break
            else:
                rec = _selected_record_from_generator_metadata(
                    raw,
                    source_tag=source_tag,
                    source_kind=source_kind,
                    relative_order=int(raw.get("relative_order", idx)),
                )
                if rec is None:
                    label = raw.get("candidate_label") or raw.get("operator_label") or raw.get("selected_op") or raw.get("label")
                    if label not in {None, ""}:
                        rec = _selected_record_from_label(
                            label,
                            source_tag=source_tag,
                            source_kind=source_kind,
                            relative_order=int(raw.get("relative_order", idx)),
                        )
        else:
            rec = _selected_record_from_label(
                raw,
                source_tag=source_tag,
                source_kind=source_kind,
                relative_order=idx,
            )
        if rec is not None:
            out.append(rec)
    return out


def _selected_logical_library_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    source_tag: str,
    source_kind: str,
    ordering: str | None = None,
    boson_encoding: str | None = None,
    source_num_sites: int | None = None,
) -> dict[str, Any] | None:
    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, rec_raw in enumerate(records):
        if not isinstance(rec_raw, Mapping):
            continue
        rec = dict(rec_raw)
        rec.setdefault("relative_order", idx)
        for key in (
            "generator_id",
            "family_id",
            "template_id",
            "operator_family_id",
            "candidate_label",
            "operator_label",
            "label",
        ):
            if key in rec:
                rec[key] = _clean_optional_str(rec.get(key))
        rec["generator_ids"] = _str_list_or_empty(rec.get("generator_ids", []))
        families = _record_operator_family_ids(rec)
        if families:
            rec["operator_family_id"] = families[0]
            rec["operator_family_ids"] = [str(x) for x in families]
        else:
            rec.setdefault("operator_family_ids", [])
        key = json.dumps(
            {
                "generator_ids": rec.get("generator_ids", []),
                "family_id": rec.get("family_id"),
                "template_id": rec.get("template_id"),
                "offsets": rec.get("support_site_offsets", []),
                "label": rec.get("candidate_label") or rec.get("operator_label"),
            },
            sort_keys=True,
            default=str,
        )
        if key in seen:
            continue
        seen.add(key)
        clean.append(rec)
    if not clean:
        return None
    source_tags: list[str] = []
    for rec in clean:
        tag = str(rec.get("source_tag") or source_tag)
        if tag and tag not in source_tags:
            source_tags.append(tag)
    return {
        "schema": "selected_logical_library_v1",
        "library_version": "selected_logical_library_v1",
        "source_kind": str(source_kind),
        "source_tag": str(source_tags[0] if source_tags else source_tag),
        "source_tags": [str(x) for x in source_tags],
        "ordering": (None if ordering in {None, ""} else str(ordering)),
        "boson_encoding": (None if boson_encoding in {None, ""} else str(boson_encoding)),
        "source_num_sites": (None if source_num_sites is None else int(source_num_sites)),
        "record_count": int(len(clean)),
        "records": [dict(x) for x in clean],
    }


def load_selected_logical_library_from_payload(payload: Any) -> dict[str, Any] | None:
    """Normalize selected logical generator/template evidence from an artifact.

    Accepted inputs include this helper's normalized ``selected_logical_library_v1``
    payloads, motif libraries, top-level or nested continuation selected-generator
    metadata, per-history selected feature rows, and legacy ``adapt_vqe.operators``
    label lists.  The normalized records are intentionally problem-generic; HH
    motif/template records are only one source of structural evidence.
    """

    if isinstance(payload, Mapping):
        source_tag = _payload_source_tag(payload)
        settings = payload.get("settings", {}) if isinstance(payload.get("settings", {}), Mapping) else {}
        ordering = settings.get("ordering") or payload.get("ordering")
        boson_encoding = settings.get("boson_encoding") or payload.get("boson_encoding")
        source_num_sites_raw = settings.get("L") or settings.get("num_sites") or payload.get("source_num_sites")
        source_num_sites = None
        try:
            if source_num_sites_raw not in {None, ""}:
                source_num_sites = int(source_num_sites_raw)
        except Exception:
            source_num_sites = None

        for key in ("selected_logical_payload", "selected_logical", "selected_logical_library"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                nested_lib = load_selected_logical_library_from_payload(nested)
                if nested_lib is not None:
                    return nested_lib

        schema = str(payload.get("schema") or payload.get("library_version") or "")
        if schema == "selected_logical_library_v1" and _is_nonstring_sequence(payload.get("records", [])):
            records = _normalise_selected_records(
                payload.get("records", []),
                source_tag=source_tag,
                source_kind=str(payload.get("source_kind", "selected_logical_library")),
            )
            return _selected_logical_library_from_records(
                records,
                source_tag=str(payload.get("source_tag", source_tag)),
                source_kind=str(payload.get("source_kind", "selected_logical_library")),
                ordering=ordering,
                boson_encoding=boson_encoding,
                source_num_sites=source_num_sites,
            )

        if str(payload.get("library_version", "")).startswith("phase3_motif_library") and _is_nonstring_sequence(payload.get("records", [])):
            records = _selected_records_from_motif_library(payload, source_tag=source_tag)
            return _selected_logical_library_from_records(
                records,
                source_tag=source_tag,
                source_kind="motif_library",
                ordering=ordering or payload.get("ordering"),
                boson_encoding=boson_encoding or payload.get("boson_encoding"),
                source_num_sites=source_num_sites,
            )

        selected_logical_records_raw = payload.get("selected_logical_records")
        if _is_nonstring_sequence(selected_logical_records_raw):
            records = _normalise_selected_records(
                selected_logical_records_raw,
                source_tag=source_tag,
                source_kind="selected_logical_records",
            )
            lib = _selected_logical_library_from_records(
                records,
                source_tag=source_tag,
                source_kind="selected_logical_records",
                ordering=ordering,
                boson_encoding=boson_encoding,
                source_num_sites=source_num_sites,
            )
            if lib is not None:
                return lib

        continuation = payload.get("continuation") if isinstance(payload.get("continuation"), Mapping) else None
        if isinstance(continuation, Mapping):
            motif_library = continuation.get("motif_library")
            if isinstance(motif_library, Mapping):
                records = _selected_records_from_motif_library(motif_library, source_tag=source_tag)
                lib = _selected_logical_library_from_records(
                    records,
                    source_tag=source_tag,
                    source_kind="continuation.motif_library",
                    ordering=ordering or motif_library.get("ordering"),
                    boson_encoding=boson_encoding or motif_library.get("boson_encoding"),
                    source_num_sites=source_num_sites,
                )
                if lib is not None:
                    return lib
            selected_meta = continuation.get("selected_generator_metadata")
            if _is_nonstring_sequence(selected_meta):
                records = _normalise_selected_records(
                    selected_meta,
                    source_tag=source_tag,
                    source_kind="continuation.selected_generator_metadata",
                )
                lib = _selected_logical_library_from_records(
                    records,
                    source_tag=source_tag,
                    source_kind="continuation.selected_generator_metadata",
                    ordering=ordering,
                    boson_encoding=boson_encoding,
                    source_num_sites=source_num_sites,
                )
                if lib is not None:
                    return lib

        adapt_block = payload.get("adapt_vqe") if isinstance(payload.get("adapt_vqe"), Mapping) else None
        if isinstance(adapt_block, Mapping):
            adapt_payload = dict(adapt_block)
            if "settings" not in adapt_payload and settings:
                adapt_payload["settings"] = dict(settings)
            if "source_tag" not in adapt_payload:
                adapt_payload["source_tag"] = source_tag
            nested_continuation = adapt_block.get("continuation")
            if isinstance(nested_continuation, Mapping):
                selected_meta = nested_continuation.get("selected_generator_metadata")
                if _is_nonstring_sequence(selected_meta):
                    records = _normalise_selected_records(
                        selected_meta,
                        source_tag=source_tag,
                        source_kind="adapt_vqe.continuation.selected_generator_metadata",
                    )
                    lib = _selected_logical_library_from_records(
                        records,
                        source_tag=source_tag,
                        source_kind="adapt_vqe.continuation.selected_generator_metadata",
                        ordering=ordering,
                        boson_encoding=boson_encoding,
                        source_num_sites=source_num_sites,
                    )
                    if lib is not None:
                        return lib
            history = adapt_block.get("history", [])
            records_from_history: list[dict[str, Any]] = []
            if _is_nonstring_sequence(history):
                for row_idx, row in enumerate(history):
                    if not isinstance(row, Mapping):
                        continue
                    feature_rows = row.get("selected_feature_rows", [])
                    if not _is_nonstring_sequence(feature_rows):
                        continue
                    for feat_idx, feat in enumerate(feature_rows):
                        if not isinstance(feat, Mapping):
                            continue
                        meta = feat.get("generator_metadata")
                        if isinstance(meta, Mapping):
                            rec = _selected_record_from_generator_metadata(
                                meta,
                                source_tag=source_tag,
                                source_kind="adapt_vqe.history.selected_feature_rows.generator_metadata",
                                relative_order=len(records_from_history),
                            )
                            if rec is not None:
                                rec["history_index"] = int(row_idx)
                                rec["selected_feature_row_index"] = int(feat_idx)
                                records_from_history.append(rec)
                lib = _selected_logical_library_from_records(
                    records_from_history,
                    source_tag=source_tag,
                    source_kind="adapt_vqe.history.selected_feature_rows.generator_metadata",
                    ordering=ordering,
                    boson_encoding=boson_encoding,
                    source_num_sites=source_num_sites,
                )
                if lib is not None:
                    return lib
            operators = adapt_block.get("operators", [])
            if _is_nonstring_sequence(operators):
                records = _normalise_selected_records(
                    operators,
                    source_tag=source_tag,
                    source_kind="adapt_vqe.operators",
                )
                lib = _selected_logical_library_from_records(
                    records,
                    source_tag=source_tag,
                    source_kind="adapt_vqe.operators",
                    ordering=ordering,
                    boson_encoding=boson_encoding,
                    source_num_sites=source_num_sites,
                )
                if lib is not None:
                    return lib

        selected_meta = payload.get("selected_generator_metadata")
        if _is_nonstring_sequence(selected_meta):
            records = _normalise_selected_records(
                selected_meta,
                source_tag=source_tag,
                source_kind="selected_generator_metadata",
            )
            return _selected_logical_library_from_records(
                records,
                source_tag=source_tag,
                source_kind="selected_generator_metadata",
                ordering=ordering,
                boson_encoding=boson_encoding,
                source_num_sites=source_num_sites,
            )
        return None

    if _is_nonstring_sequence(payload):
        records = _normalise_selected_records(
            payload,
            source_tag="payload_sequence",
            source_kind="selected_logical_sequence",
        )
        return _selected_logical_library_from_records(
            records,
            source_tag="payload_sequence",
            source_kind="selected_logical_sequence",
        )
    return None


def load_selected_logical_library_from_json(path: str | Path) -> dict[str, Any] | None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_selected_logical_library_from_payload(payload)


def _record_generator_ids(record: Mapping[str, Any]) -> list[str]:
    out = _str_list_or_empty(record.get("generator_ids", []))
    gen = _clean_optional_str(record.get("generator_id"))
    if gen and gen not in out:
        out.insert(0, gen)
    return out


def _record_labels(record: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("candidate_label", "operator_label", "label", "selected_op"):
        value = _clean_optional_str(record.get(key))
        if value and value not in out:
            out.append(value)
    return out


def _record_structural_key(record: Mapping[str, Any]) -> tuple[str, str, tuple[int, ...]] | None:
    family_id = _clean_optional_str(record.get("family_id"))
    template_id = _clean_optional_str(record.get("template_id"))
    if family_id is None or template_id is None:
        return None
    return (
        family_id,
        template_id,
        tuple(_int_list_or_empty(record.get("support_site_offsets", []))),
    )


def match_selected_logical_generators(
    *,
    selected_logical_library: Mapping[str, Any] | None,
    registry_by_label: Mapping[str, Mapping[str, Any]],
    target_num_sites: int,
    transfer_mode: str = "exact_match_v1",
) -> list[dict[str, Any]]:
    """Return registry rows selected by historical logical evidence.

    Match precedence is stable and intentional: exact ``generator_id`` first,
    then transferable family/template/support-offset motif matches, then exact
    candidate label fallback for older artifacts that lack metadata.
    """

    if not isinstance(selected_logical_library, Mapping):
        return []
    # Validate the mode through the shared boundary matcher.
    _boundary_behavior_matches("interior_only", "interior_only", transfer_mode=str(transfer_mode))
    records_raw = selected_logical_library.get("records", [])
    if not _is_nonstring_sequence(records_raw):
        return []
    records = [dict(x) for x in records_raw if isinstance(x, Mapping)]
    if not records:
        return []

    generator_id_to_record: dict[str, dict[str, Any]] = {}
    structural_records: list[dict[str, Any]] = []
    labels_to_record: dict[str, dict[str, Any]] = {}
    for rec in records:
        for gen_id in _record_generator_ids(rec):
            generator_id_to_record.setdefault(str(gen_id), rec)
        if _record_structural_key(rec) is not None:
            structural_records.append(rec)
        for label in _record_labels(rec):
            labels_to_record.setdefault(str(label), rec)

    matches: list[dict[str, Any]] = []
    for label, meta_raw in registry_by_label.items():
        meta = dict(meta_raw) if isinstance(meta_raw, Mapping) else {"candidate_label": str(label)}
        candidate_label = str(meta.get("candidate_label") or label)
        match_method: str | None = None
        matched_record: dict[str, Any] | None = None
        gen_id = meta.get("generator_id")
        if gen_id not in {None, ""} and str(gen_id) in generator_id_to_record:
            match_method = "generator_id"
            matched_record = generator_id_to_record[str(gen_id)]
        if match_method is None:
            candidate_key = _record_structural_key(meta)
            if candidate_key is not None:
                target_boundary = _candidate_boundary_behavior(
                    meta,
                    target_num_sites=int(target_num_sites),
                )
                for rec in structural_records:
                    if _record_structural_key(rec) != candidate_key:
                        continue
                    if not _boundary_behavior_matches(
                        str(rec.get("boundary_behavior", "interior_only")),
                        str(target_boundary),
                        transfer_mode=str(transfer_mode),
                    ):
                        continue
                    match_method = "template_support_offsets"
                    matched_record = rec
                    break
        if match_method is None:
            for label_key in (candidate_label, str(label)):
                if label_key in labels_to_record:
                    match_method = "exact_label"
                    matched_record = labels_to_record[label_key]
                    break
        if match_method is None or matched_record is None:
            continue
        matches.append(
            {
                "candidate_label": str(candidate_label),
                "registry_label": str(label),
                "generator_metadata": dict(meta),
                "match_method": str(match_method),
                "selected_logical_record": dict(matched_record),
                "source_tag": str(matched_record.get("source_tag") or selected_logical_library.get("source_tag", "payload")),
            }
        )
    return matches


def match_selected_logical_operator_families(
    *,
    selected_logical_library: Mapping[str, Any] | None,
    registry_by_label: Mapping[str, Mapping[str, Any]],
    target_num_sites: int,
    transfer_mode: str = "exact_match_v1",
) -> list[dict[str, Any]]:
    """Return registry rows selected by historical operator-family evidence.

    This is the production semantics for the ``historical-selected`` Optuna
    route: historical artifacts seed transferable operator families, not exact
    concrete labels. Exact matches are still computed first so legacy label-only
    sources can infer the target family before the closure is applied.
    """

    exact_matches = match_selected_logical_generators(
        selected_logical_library=selected_logical_library,
        registry_by_label=registry_by_label,
        target_num_sites=int(target_num_sites),
        transfer_mode=str(transfer_mode),
    )
    if not isinstance(selected_logical_library, Mapping):
        return []
    records_raw = selected_logical_library.get("records", [])
    if not _is_nonstring_sequence(records_raw):
        return exact_matches
    records = [dict(x) for x in records_raw if isinstance(x, Mapping)]

    family_ids: list[str] = []

    def add_family(value: Any) -> None:
        text = _clean_optional_str(value)
        if text and text not in family_ids:
            family_ids.append(text)

    for record in records:
        for family in _record_operator_family_ids(record):
            add_family(family)

    exact_by_label: dict[str, dict[str, Any]] = {}
    for match in exact_matches:
        label = _clean_optional_str(match.get("registry_label") or match.get("candidate_label"))
        if label:
            exact_by_label[label] = dict(match)
        meta = match.get("generator_metadata")
        if isinstance(meta, Mapping):
            for family in _record_operator_family_ids(meta):
                add_family(family)
        selected = match.get("selected_logical_record")
        if isinstance(selected, Mapping):
            for family in _record_operator_family_ids(selected):
                add_family(family)

    if not family_ids:
        return exact_matches

    matches: list[dict[str, Any]] = []
    for label, meta_raw in registry_by_label.items():
        meta = dict(meta_raw) if isinstance(meta_raw, Mapping) else {"candidate_label": str(label)}
        candidate_label = _clean_optional_str(meta.get("candidate_label")) or str(label)
        candidate_families = _record_operator_family_ids({**meta, "candidate_label": candidate_label})
        closure_family = next((family for family in candidate_families if family in family_ids), None)
        if closure_family is None:
            continue
        exact_match = exact_by_label.get(str(label)) or exact_by_label.get(candidate_label)
        selected_record = (
            dict(exact_match.get("selected_logical_record", {}))
            if isinstance(exact_match, Mapping) and isinstance(exact_match.get("selected_logical_record"), Mapping)
            else {
                "record_kind": "operator_family",
                "operator_family_id": str(closure_family),
                "operator_family_ids": [str(closure_family)],
                "source_kind": str(selected_logical_library.get("source_kind", "selected_logical_library")),
                "source_tag": str(selected_logical_library.get("source_tag", "payload")),
            }
        )
        match_method = (
            "operator_family_closure_from_exact"
            if isinstance(exact_match, Mapping)
            else "operator_family_closure"
        )
        matches.append(
            {
                "candidate_label": str(candidate_label),
                "registry_label": str(label),
                "generator_metadata": dict(meta),
                "match_method": str(match_method),
                "selected_logical_record": selected_record,
                "operator_family_id": str(closure_family),
                "operator_family_ids": [str(x) for x in candidate_families],
                "selected_operator_family_ids": [str(x) for x in family_ids],
                "source_tag": str(selected_record.get("source_tag") or selected_logical_library.get("source_tag", "payload")),
            }
        )
    return matches


def motif_bonus_for_generator(
    *,
    generator_metadata: Mapping[str, Any] | None,
    motif_library: Mapping[str, Any] | None,
    target_num_sites: int,
    transfer_mode: str = "exact_match_v1",
) -> tuple[float, dict[str, Any] | None]:
    if not isinstance(generator_metadata, Mapping) or not isinstance(motif_library, Mapping):
        return 0.0, None
    recs = motif_library.get("records", [])
    if not isinstance(recs, Sequence):
        return 0.0, None
    family_id = str(generator_metadata.get("family_id", ""))
    template_id = str(generator_metadata.get("template_id", ""))
    offsets = [int(x) for x in generator_metadata.get("support_site_offsets", [])]
    target_boundary_behavior = _candidate_boundary_behavior(
        generator_metadata,
        target_num_sites=int(target_num_sites),
    )
    motif_ids: list[str] = []
    motif_source_tags: list[str] = []
    boundary_behavior = None
    for rec in recs:
        if not isinstance(rec, Mapping):
            continue
        if str(rec.get("family_id", "")) != family_id:
            continue
        if str(rec.get("template_id", "")) != template_id:
            continue
        if [int(x) for x in rec.get("support_site_offsets", [])] != offsets:
            continue
        boundary_ok = _boundary_behavior_matches(
            str(rec.get("boundary_behavior", "interior_only")),
            str(target_boundary_behavior),
            transfer_mode=str(transfer_mode),
        )
        if not boundary_ok:
            continue
        boundary_behavior = str(rec.get("boundary_behavior", "interior_only"))
        motif_ids.append(str(rec.get("motif_id", "")))
        for tag in rec.get("source_tags", [motif_library.get("source_tag", "payload")]):
            tag_s = str(tag)
            if tag_s and tag_s not in motif_source_tags:
                motif_source_tags.append(tag_s)
    if not motif_ids:
        return 0.0, None
    meta = MotifMetadata(
        enabled=True,
        motif_tags=[str(family_id), str(template_id)],
        motif_ids=[str(x) for x in motif_ids[:4]],
        motif_source=(str(motif_source_tags[0]) if motif_source_tags else str(motif_library.get("source_tag", "payload"))),
        tiled_from_num_sites=int(motif_library.get("source_num_sites", 0)),
        target_num_sites=int(target_num_sites),
        boundary_behavior=str(boundary_behavior) if boundary_behavior is not None else None,
        transfer_mode=str(transfer_mode),
    )
    bonus = 0.1 + 0.02 * float(max(0, len(motif_ids) - 1))
    if str(transfer_mode).strip().lower() == "boundary_v1" and boundary_behavior is not None:
        bonus += 0.02
    return float(min(0.25, bonus)), asdict(meta)


def select_tiled_generators_from_library(
    *,
    motif_library: Mapping[str, Any] | None,
    registry_by_label: Mapping[str, Mapping[str, Any]],
    target_num_sites: int,
    excluded_labels: Sequence[str],
    max_seed: int,
    transfer_mode: str = "exact_match_v1",
) -> list[dict[str, Any]]:
    if not isinstance(motif_library, Mapping):
        return []
    excluded = {str(x) for x in excluded_labels}
    records = motif_library.get("records", [])
    if not isinstance(records, Sequence):
        return []
    seeded: list[dict[str, Any]] = []
    used_labels: set[str] = set()
    registry_rows = [dict(v) for _, v in sorted(registry_by_label.items(), key=lambda kv: str(kv[0]))]
    sorted_records = sorted(
        [dict(x) for x in records if isinstance(x, Mapping)],
        key=lambda rec: (
            int(rec.get("relative_order", 0)),
            str(rec.get("family_id", "")),
            str(rec.get("template_id", "")),
            str(rec.get("boundary_behavior", "interior_only")),
        ),
    )
    for rec in sorted_records:
        if len(seeded) >= int(max_seed):
            break
        family_id = str(rec.get("family_id", ""))
        template_id = str(rec.get("template_id", ""))
        offsets = [int(x) for x in rec.get("support_site_offsets", [])]
        for meta in registry_rows:
            label = str(meta.get("candidate_label", ""))
            if label in excluded or label in used_labels:
                continue
            if str(meta.get("family_id", "")) != family_id:
                continue
            if str(meta.get("template_id", "")) != template_id:
                continue
            if [int(x) for x in meta.get("support_site_offsets", [])] != offsets:
                continue
            bonus, motif_meta = motif_bonus_for_generator(
                generator_metadata=meta,
                motif_library=motif_library,
                target_num_sites=int(target_num_sites),
                transfer_mode=str(transfer_mode),
            )
            if bonus <= 0.0:
                continue
            seeded.append(
                {
                    "candidate_label": str(label),
                    "generator_metadata": dict(meta),
                    "motif_bonus": float(bonus),
                    "motif_metadata": dict(motif_meta) if isinstance(motif_meta, Mapping) else None,
                    "source_motif_id": str(rec.get("motif_id", "")),
                }
            )
            used_labels.add(label)
            break
    return seeded
