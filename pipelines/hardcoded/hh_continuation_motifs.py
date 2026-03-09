#!/usr/bin/env python3
"""Motif extraction and tiling helpers for HH continuation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import json
from typing import Any, Mapping, Sequence

from pipelines.hardcoded.hh_continuation_types import MotifLibrary, MotifMetadata, MotifRecord


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
        optimal_point = payload.get("adapt_vqe", {}).get("optimal_point", None) if isinstance(payload.get("adapt_vqe", {}), Mapping) else None
        settings = payload.get("settings", None)
        if isinstance(generator_metadata, Sequence) and isinstance(optimal_point, Sequence) and isinstance(settings, Mapping):
            return extract_motif_library(
                generator_metadata=[dict(x) for x in generator_metadata if isinstance(x, Mapping)],
                theta=[float(x) for x in optimal_point],
                source_num_sites=int(settings.get("L", 0)),
                source_tag=str(payload.get("generated_utc", "source_payload")),
                ordering=str(settings.get("ordering", "blocked")),
                boson_encoding=str(settings.get("boson_encoding", "binary")),
            )
    return None


def load_motif_library_from_json(path: str | Path) -> dict[str, Any] | None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_motif_library_from_payload(payload)


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
