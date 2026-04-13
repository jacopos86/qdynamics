"""HH preset pool composition helpers extracted from the static ADAPT pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm

from .primitive_pools import (
    _build_full_hamiltonian_pool,
    _build_hh_termwise_augmented_pool,
    _build_hh_uccsd_fermion_lifted_pool,
    _build_hva_pool,
    _build_paop_pool,
    _build_vlf_sq_pool,
    _deduplicate_pool_terms,
    _deduplicate_pool_terms_lightweight,
    _polynomial_signature,
)

_HH_FULL_META_CLASSIFIER_VERSION = "hh_full_meta_v1"
_HH_FULL_META_ALLOWED_CLASSES = (
    "hh_termwise_unit",
    "hh_termwise_quadrature",
    "uccsd_sing",
    "uccsd_dbl",
    "hva_layer",
    "paop_cloud_p",
    "paop_cloud_x",
    "paop_disp",
    "paop_dbl",
    "paop_hopdrag",
    "paop_dbl_p",
    "paop_dbl_x",
    "paop_curdrag",
    "paop_hop2",
)


@dataclass(frozen=True)
class HHFullMetaClassFilterSpec:
    keep_classes: tuple[str, ...]
    classifier_version: str = _HH_FULL_META_CLASSIFIER_VERSION
    source_pool: str = "full_meta"
    source_problem: str = "hh"
    source_num_sites: int | None = None
    source_n_ph_max: int | None = None
    source_json: str | None = None


@dataclass(frozen=True)
class HHFullMetaLabelFilterSpec:
    drop_labels: tuple[str, ...] = ()
    drop_prefixes: tuple[str, ...] = ()
    classifier_version: str = _HH_FULL_META_CLASSIFIER_VERSION
    source_pool: str = "full_meta"
    source_problem: str = "hh"
    source_num_sites: int | None = None
    source_n_ph_max: int | None = None
    source_json: str | None = None


def _classify_hh_full_meta_label(label: str) -> str | None:
    label_str = str(label)
    if any(
        label_str.startswith(prefix)
        for prefix in ("hop_layer", "onsite_layer", "phonon_layer", "eph_layer")
    ):
        return "hva_layer"
    if label_str.startswith("hh_termwise_ham_unit_term("):
        return "hh_termwise_unit"
    if label_str.startswith("hh_termwise_ham_quadrature_term("):
        return "hh_termwise_quadrature"
    if label_str.startswith("uccsd_ferm_lifted::uccsd_sing("):
        return "uccsd_sing"
    if label_str.startswith("uccsd_ferm_lifted::uccsd_dbl("):
        return "uccsd_dbl"
    if label_str.startswith("paop_full:paop_cloud_p("):
        return "paop_cloud_p"
    if label_str.startswith("paop_full:paop_cloud_x("):
        return "paop_cloud_x"
    if label_str.startswith("paop_full:paop_disp("):
        return "paop_disp"
    if label_str.startswith("paop_full:paop_dbl("):
        return "paop_dbl"
    if label_str.startswith("paop_full:paop_hopdrag("):
        return "paop_hopdrag"
    if label_str.startswith("paop_lf_full:paop_dbl_p("):
        return "paop_dbl_p"
    if label_str.startswith("paop_lf_full:paop_dbl_x("):
        return "paop_dbl_x"
    if label_str.startswith("paop_lf_full:paop_curdrag("):
        return "paop_curdrag"
    if label_str.startswith("paop_lf_full:paop_hop2("):
        return "paop_hop2"
    return None


def _normalize_hh_full_meta_keep_classes(classes: Sequence[Any]) -> tuple[str, ...]:
    keep_classes: list[str] = []
    seen: set[str] = set()
    for raw in classes:
        name = str(raw).strip()
        if name == "":
            continue
        if name not in _HH_FULL_META_ALLOWED_CLASSES:
            raise ValueError(
                "Unknown HH full_meta class "
                f"{name!r}; allowed classes are {list(_HH_FULL_META_ALLOWED_CLASSES)}."
            )
        if name in seen:
            continue
        seen.add(name)
        keep_classes.append(name)
    if not keep_classes:
        raise ValueError("HH full_meta class filter must keep at least one class.")
    return tuple(keep_classes)


def _normalize_nonempty_unique_strings(items: Sequence[Any], *, field_name: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        value = str(raw).strip()
        if value == "":
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    if not out:
        raise ValueError(f"HH full_meta label filter field {field_name!r} must contain at least one non-empty string.")
    return tuple(out)


def _load_hh_full_meta_class_filter_spec(path: Path) -> HHFullMetaClassFilterSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("HH full_meta class filter JSON must be an object with keep_classes.")
    keep_raw = raw.get("keep_classes")
    if not isinstance(keep_raw, list):
        raise ValueError("HH full_meta class filter JSON must contain list field 'keep_classes'.")
    classifier_version = str(raw.get("classifier_version", _HH_FULL_META_CLASSIFIER_VERSION)).strip()
    if classifier_version != _HH_FULL_META_CLASSIFIER_VERSION:
        raise ValueError(
            "HH full_meta class filter classifier_version mismatch: "
            f"got {classifier_version!r}, expected {_HH_FULL_META_CLASSIFIER_VERSION!r}."
        )
    source_pool = str(raw.get("source_pool", "")).strip().lower()
    if source_pool != "full_meta":
        raise ValueError(
            "HH full_meta class filter JSON must declare source_pool='full_meta'."
        )
    source_problem = str(raw.get("source_problem", "hh")).strip().lower()
    if source_problem != "hh":
        raise ValueError(
            "HH full_meta class filter JSON must declare source_problem='hh'."
        )
    source_num_sites_raw = raw.get("source_num_sites")
    source_n_ph_max_raw = raw.get("source_n_ph_max")
    return HHFullMetaClassFilterSpec(
        keep_classes=_normalize_hh_full_meta_keep_classes(keep_raw),
        classifier_version=str(classifier_version),
        source_pool=str(source_pool),
        source_problem=str(source_problem),
        source_num_sites=(
            None if source_num_sites_raw is None else int(source_num_sites_raw)
        ),
        source_n_ph_max=(
            None if source_n_ph_max_raw is None else int(source_n_ph_max_raw)
        ),
        source_json=str(path),
    )


def _load_hh_full_meta_label_filter_spec(path: Path) -> HHFullMetaLabelFilterSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("HH full_meta label filter JSON must be an object.")
    drop_labels_raw = raw.get("drop_labels", [])
    drop_prefixes_raw = raw.get("drop_prefixes", [])
    if not isinstance(drop_labels_raw, list):
        raise ValueError("HH full_meta label filter JSON field 'drop_labels' must be a list.")
    if not isinstance(drop_prefixes_raw, list):
        raise ValueError("HH full_meta label filter JSON field 'drop_prefixes' must be a list.")
    if len(drop_labels_raw) == 0 and len(drop_prefixes_raw) == 0:
        raise ValueError("HH full_meta label filter JSON must contain non-empty 'drop_labels' or 'drop_prefixes'.")
    classifier_version = str(raw.get("classifier_version", _HH_FULL_META_CLASSIFIER_VERSION)).strip()
    if classifier_version != _HH_FULL_META_CLASSIFIER_VERSION:
        raise ValueError(
            "HH full_meta label filter classifier_version mismatch: "
            f"got {classifier_version!r}, expected {_HH_FULL_META_CLASSIFIER_VERSION!r}."
        )
    source_pool = str(raw.get("source_pool", "")).strip().lower()
    if source_pool != "full_meta":
        raise ValueError(
            "HH full_meta label filter JSON must declare source_pool='full_meta'."
        )
    source_problem = str(raw.get("source_problem", "hh")).strip().lower()
    if source_problem != "hh":
        raise ValueError(
            "HH full_meta label filter JSON must declare source_problem='hh'."
        )
    source_num_sites_raw = raw.get("source_num_sites")
    source_n_ph_max_raw = raw.get("source_n_ph_max")
    return HHFullMetaLabelFilterSpec(
        drop_labels=(
            _normalize_nonempty_unique_strings(drop_labels_raw, field_name="drop_labels")
            if len(drop_labels_raw) > 0
            else tuple()
        ),
        drop_prefixes=(
            _normalize_nonempty_unique_strings(drop_prefixes_raw, field_name="drop_prefixes")
            if len(drop_prefixes_raw) > 0
            else tuple()
        ),
        classifier_version=str(classifier_version),
        source_pool=str(source_pool),
        source_problem=str(source_problem),
        source_num_sites=(
            None if source_num_sites_raw is None else int(source_num_sites_raw)
        ),
        source_n_ph_max=(
            None if source_n_ph_max_raw is None else int(source_n_ph_max_raw)
        ),
        source_json=str(path),
    )


def _summarize_hh_full_meta_pool_classes(pool: Sequence[AnsatzTerm]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for term in pool:
        family = _classify_hh_full_meta_label(str(term.label))
        if family is None:
            raise ValueError(f"Unable to classify HH full_meta operator label {term.label!r}.")
        counts[family] = int(counts.get(family, 0) + 1)
    return {
        family: int(counts[family])
        for family in _HH_FULL_META_ALLOWED_CLASSES
        if family in counts
    }


def _filter_hh_full_meta_pool_by_class(
    pool: Sequence[AnsatzTerm],
    spec: HHFullMetaClassFilterSpec,
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    counts_before = _summarize_hh_full_meta_pool_classes(pool)
    keep_set = set(spec.keep_classes)
    filtered_pool = [
        term
        for term in pool
        if _classify_hh_full_meta_label(str(term.label)) in keep_set
    ]
    if not filtered_pool:
        raise ValueError("HH full_meta class filter removed every operator from the pool.")
    counts_after = _summarize_hh_full_meta_pool_classes(filtered_pool)
    dropped_classes = [
        family for family in counts_before.keys()
        if family not in keep_set
    ]
    meta = {
        "classifier_version": str(spec.classifier_version),
        "source_pool": str(spec.source_pool),
        "source_problem": str(spec.source_problem),
        "source_num_sites": (
            int(spec.source_num_sites) if spec.source_num_sites is not None else None
        ),
        "source_n_ph_max": (
            int(spec.source_n_ph_max) if spec.source_n_ph_max is not None else None
        ),
        "source_json": str(spec.source_json) if spec.source_json is not None else None,
        "keep_classes": [str(x) for x in spec.keep_classes],
        "dropped_classes": [str(x) for x in dropped_classes],
        "class_counts_before": dict(counts_before),
        "class_counts_after": dict(counts_after),
        "dedup_total_before": int(len(pool)),
        "dedup_total_after": int(len(filtered_pool)),
    }
    return filtered_pool, meta


def _filter_hh_full_meta_pool_by_label(
    pool: Sequence[AnsatzTerm],
    spec: HHFullMetaLabelFilterSpec,
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    drop_labels = set(str(x) for x in spec.drop_labels)
    drop_prefixes = tuple(str(x) for x in spec.drop_prefixes)
    removed_labels: list[str] = []
    unmatched_labels = set(drop_labels)
    unmatched_prefixes = set(drop_prefixes)
    filtered_pool: list[AnsatzTerm] = []
    for term in pool:
        label = str(term.label)
        drop_exact = label in drop_labels
        drop_prefix = next((prefix for prefix in drop_prefixes if label.startswith(prefix)), None)
        if drop_exact or drop_prefix is not None:
            removed_labels.append(label)
            unmatched_labels.discard(label)
            if drop_prefix is not None:
                unmatched_prefixes.discard(str(drop_prefix))
            continue
        filtered_pool.append(term)
    if not filtered_pool:
        raise ValueError("HH full_meta label filter removed every operator from the pool.")
    counts_before = _summarize_hh_full_meta_pool_classes(pool)
    counts_after = _summarize_hh_full_meta_pool_classes(filtered_pool)
    meta = {
        "classifier_version": str(spec.classifier_version),
        "source_pool": str(spec.source_pool),
        "source_problem": str(spec.source_problem),
        "source_json": str(spec.source_json) if spec.source_json is not None else None,
        "drop_labels": list(spec.drop_labels),
        "drop_prefixes": list(spec.drop_prefixes),
        "removed_count": int(len(removed_labels)),
        "removed_labels_sample": [str(x) for x in removed_labels[:16]],
        "unmatched_drop_labels": sorted(str(x) for x in unmatched_labels),
        "unmatched_drop_prefixes": sorted(str(x) for x in unmatched_prefixes),
        "class_counts_before": counts_before,
        "class_counts_after": counts_after,
    }
    return filtered_pool, meta


_PARETO_LEAN_PAOP_FULL_KEEP = {"paop_cloud_p", "paop_disp", "paop_hopdrag"}
_PARETO_LEAN_PAOP_LF_KEEP = {"paop_dbl_p"}
_PARETO_LEAN_L2_PAOP_FULL_KEEP = {"paop_cloud_p", "paop_hopdrag"}
_PARETO_LEAN_L2_PAOP_LF_KEEP = {"paop_dbl_p"}
_PARETO_LEAN_L2_DPL_P_KEEP_SUFFIXES = {"site=0->phonon=1", "site=1->phonon=0"}


def _pareto_lean_paop_match(label: str, allowed: set[str]) -> bool:
    colon_idx = label.find(":")
    if colon_idx < 0:
        return False
    after_colon = label[colon_idx + 1:]
    for family in allowed:
        if after_colon.startswith(family + "("):
            return True
    return False


def _build_hh_full_meta_pool(
    *,
    h_poly: Any,
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, int]]:
    uccsd_lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        num_particles=num_particles,
    )
    hva_pool = _build_hva_pool(
        int(num_sites),
        float(t),
        float(u),
        float(omega0),
        float(g_ep),
        float(dv),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
    )
    termwise_aug: list[AnsatzTerm] = []
    if abs(float(g_ep)) > 1e-15:
        termwise_aug = [
            AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
            for term in _build_hh_termwise_augmented_pool(h_poly)
        ]
    paop_full_pool = _build_paop_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        "paop_full",
        int(paop_r),
        bool(paop_split_paulis),
        float(paop_prune_eps),
        str(paop_normalization),
        num_particles,
    )
    paop_lf_full_pool = _build_paop_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        "paop_lf_full",
        int(paop_r),
        bool(paop_split_paulis),
        float(paop_prune_eps),
        str(paop_normalization),
        num_particles,
    )
    merged = (
        list(uccsd_lifted_pool)
        + list(hva_pool)
        + list(termwise_aug)
        + list(paop_full_pool)
        + list(paop_lf_full_pool)
    )
    meta = {
        "raw_uccsd_lifted": int(len(uccsd_lifted_pool)),
        "raw_hva": int(len(hva_pool)),
        "raw_hh_termwise_augmented": int(len(termwise_aug)),
        "raw_paop_full": int(len(paop_full_pool)),
        "raw_paop_lf_full": int(len(paop_lf_full_pool)),
        "raw_total": int(len(merged)),
    }
    if int(n_ph_max) >= 2:
        dedup_pool = _deduplicate_pool_terms_lightweight(merged)
    else:
        dedup_pool = _deduplicate_pool_terms(merged)
    return dedup_pool, meta


def _build_hh_pareto_lean_pool(
    *,
    h_poly: Any,
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, int]]:
    uccsd_lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        num_particles=num_particles,
    )
    quadrature_pool: list[AnsatzTerm] = []
    if abs(float(g_ep)) > 1e-15:
        termwise_aug = _build_hh_termwise_augmented_pool(h_poly)
        quadrature_pool = [
            AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
            for term in termwise_aug
            if "quadrature" in term.label
        ]
    paop_full_raw = _build_paop_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        "paop_full",
        int(paop_r),
        bool(paop_split_paulis),
        float(paop_prune_eps),
        str(paop_normalization),
        num_particles,
    )
    paop_full_kept = [
        t for t in paop_full_raw
        if _pareto_lean_paop_match(t.label, _PARETO_LEAN_PAOP_FULL_KEEP)
    ]
    paop_lf_raw = _build_paop_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        "paop_lf_full",
        int(paop_r),
        bool(paop_split_paulis),
        float(paop_prune_eps),
        str(paop_normalization),
        num_particles,
    )
    paop_lf_kept = [
        t for t in paop_lf_raw
        if _pareto_lean_paop_match(t.label, _PARETO_LEAN_PAOP_LF_KEEP)
    ]
    merged = (
        list(uccsd_lifted_pool)
        + list(quadrature_pool)
        + list(paop_full_kept)
        + list(paop_lf_kept)
    )
    meta = {
        "raw_uccsd_lifted": int(len(uccsd_lifted_pool)),
        "raw_hh_termwise_quadrature": int(len(quadrature_pool)),
        "raw_paop_full_kept": int(len(paop_full_kept)),
        "raw_paop_full_dropped": int(len(paop_full_raw) - len(paop_full_kept)),
        "raw_paop_lf_kept": int(len(paop_lf_kept)),
        "raw_paop_lf_dropped": int(len(paop_lf_raw) - len(paop_lf_kept)),
        "raw_total": int(len(merged)),
    }
    if int(n_ph_max) >= 2:
        dedup_pool = _deduplicate_pool_terms_lightweight(merged)
    else:
        dedup_pool = _deduplicate_pool_terms(merged)
    return dedup_pool, meta


def _build_hh_pareto_lean_l3_pool(
    *,
    h_poly: Any,
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, int]]:
    if int(num_sites) != 3:
        raise ValueError("adapt_pool='pareto_lean_l3' is only valid for L=3.")
    if int(n_ph_max) != 1:
        raise ValueError("adapt_pool='pareto_lean_l3' is only valid for n_ph_max=1.")
    return _build_hh_pareto_lean_pool(
        h_poly=h_poly,
        num_sites=int(num_sites),
        t=float(t),
        u=float(u),
        omega0=float(omega0),
        g_ep=float(g_ep),
        dv=float(dv),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        paop_r=int(paop_r),
        paop_split_paulis=bool(paop_split_paulis),
        paop_prune_eps=float(paop_prune_eps),
        paop_normalization=str(paop_normalization),
        num_particles=num_particles,
    )


def _build_hh_pareto_lean_l2_pool(
    *,
    h_poly: Any,
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, int]]:
    if int(num_sites) != 2:
        raise ValueError("adapt_pool='pareto_lean_l2' is only valid for L=2.")
    if int(n_ph_max) != 1:
        raise ValueError("adapt_pool='pareto_lean_l2' is only valid for n_ph_max=1.")
    all_uccsd = _build_hh_uccsd_fermion_lifted_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        num_particles=num_particles,
    )
    uccsd_singles = [t for t in all_uccsd if "uccsd_sing" in t.label]
    quadrature_pool: list[AnsatzTerm] = []
    if abs(float(g_ep)) > 1e-15:
        termwise_aug = _build_hh_termwise_augmented_pool(h_poly)
        quadrature_pool = [
            AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
            for term in termwise_aug
            if "quadrature" in term.label
        ]
    paop_full_raw = _build_paop_pool(
        int(num_sites), int(n_ph_max), str(boson_encoding), str(ordering),
        str(boundary), "paop_full", int(paop_r), bool(paop_split_paulis),
        float(paop_prune_eps), str(paop_normalization), num_particles,
    )
    paop_full_kept = [
        t for t in paop_full_raw
        if _pareto_lean_paop_match(t.label, _PARETO_LEAN_L2_PAOP_FULL_KEEP)
    ]
    paop_lf_raw = _build_paop_pool(
        int(num_sites), int(n_ph_max), str(boson_encoding), str(ordering),
        str(boundary), "paop_lf_full", int(paop_r), bool(paop_split_paulis),
        float(paop_prune_eps), str(paop_normalization), num_particles,
    )
    paop_lf_kept = []
    for t in paop_lf_raw:
        if not _pareto_lean_paop_match(t.label, _PARETO_LEAN_L2_PAOP_LF_KEEP):
            continue
        if any(suffix in t.label for suffix in _PARETO_LEAN_L2_DPL_P_KEEP_SUFFIXES):
            paop_lf_kept.append(t)

    merged = (
        list(uccsd_singles)
        + list(quadrature_pool)
        + list(paop_full_kept)
        + list(paop_lf_kept)
    )
    meta = {
        "raw_uccsd_singles": int(len(uccsd_singles)),
        "raw_hh_termwise_quadrature": int(len(quadrature_pool)),
        "raw_paop_full_kept": int(len(paop_full_kept)),
        "raw_paop_full_dropped": int(len(paop_full_raw) - len(paop_full_kept)),
        "raw_paop_lf_kept": int(len(paop_lf_kept)),
        "raw_paop_lf_dropped": int(len(paop_lf_raw) - len(paop_lf_kept)),
        "raw_total": int(len(merged)),
    }
    if int(n_ph_max) >= 2:
        dedup_pool = _deduplicate_pool_terms_lightweight(merged)
    else:
        dedup_pool = _deduplicate_pool_terms(merged)
    return dedup_pool, meta


_GATE_PRUNE_TERM_KEEP: dict[str, list[str] | None] = {
    "uccsd_sing(alpha:0->1)": ["eeeexy"],
    "uccsd_sing(beta:2->3)": ["eeyxee"],
    "paop_hopdrag": ["yeyyee"],
}


def _gate_prune_polynomial(
    label: str,
    poly: Any,
) -> Any:
    for pattern, keep_paulis in _GATE_PRUNE_TERM_KEEP.items():
        if keep_paulis is None:
            continue
        if pattern in label:
            terms = poly.return_polynomial()
            if not terms:
                return poly
            nq = int(terms[0].nqubit())
            keep_set = set(keep_paulis)
            kept_terms = [t for t in terms if str(t.pw2strng()) in keep_set]
            if not kept_terms:
                return poly
            pruned = PauliPolynomial("JW", [
                PauliTerm(nq, ps=str(t.pw2strng()), pc=float(t.p_coeff))
                for t in kept_terms
            ])
            return pruned
    return poly


def _build_hh_pareto_lean_gate_pruned_pool(
    *,
    h_poly: Any,
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, int]]:
    base_pool, base_meta = _build_hh_pareto_lean_pool(
        h_poly=h_poly,
        num_sites=int(num_sites),
        t=float(t),
        u=float(u),
        omega0=float(omega0),
        g_ep=float(g_ep),
        dv=float(dv),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        paop_r=int(paop_r),
        paop_split_paulis=bool(paop_split_paulis),
        paop_prune_eps=float(paop_prune_eps),
        paop_normalization=str(paop_normalization),
        num_particles=num_particles,
    )
    pruned_pool: list[AnsatzTerm] = []
    n_pruned = 0
    for term in base_pool:
        pruned_poly = _gate_prune_polynomial(term.label, term.polynomial)
        if pruned_poly is not term.polynomial:
            n_pruned += 1
        pruned_pool.append(AnsatzTerm(label=term.label, polynomial=pruned_poly))

    meta = dict(base_meta)
    meta["gate_pruned_operators"] = int(n_pruned)
    meta["gate_prune_rules"] = {k: v for k, v in _GATE_PRUNE_TERM_KEEP.items() if v is not None}
    return pruned_pool, meta


def build_hh_pool_by_key(
    *,
    pool_key_hh: str,
    h_poly: Any,
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
    full_meta_class_filter_spec: HHFullMetaClassFilterSpec | None = None,
    full_meta_label_filter_spec: HHFullMetaLabelFilterSpec | None = None,
    ai_log: Callable[..., None] | None = None,
) -> tuple[list[AnsatzTerm], str, dict[str, Any] | None, dict[str, Any] | None]:
    key = str(pool_key_hh).strip().lower()
    full_meta_class_filter_meta: dict[str, Any] | None = None
    full_meta_label_filter_meta: dict[str, Any] | None = None

    if key == "hva":
        hva_pool = _build_hva_pool(
            int(num_sites),
            float(t),
            float(u),
            float(omega0),
            float(g_ep),
            float(dv),
            int(n_ph_max),
            str(boson_encoding),
            str(ordering),
            str(boundary),
        )
        if abs(float(g_ep)) <= 1e-15:
            return list(hva_pool), "hardcoded_adapt_vqe_hva_hh", None, None
        ham_term_pool = _build_hh_termwise_augmented_pool(h_poly)
        merged_pool = list(hva_pool) + [
            AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
            for term in ham_term_pool
        ]
        seen: set[tuple[tuple[str, float], ...]] = set()
        dedup_pool: list[AnsatzTerm] = []
        for term in merged_pool:
            sig = _polynomial_signature(term.polynomial)
            if sig in seen:
                continue
            seen.add(sig)
            dedup_pool.append(term)
        return dedup_pool, "hardcoded_adapt_vqe_hva_hh", None, None

    if key == "full_meta":
        pool_full, full_meta_sizes = _build_hh_full_meta_pool(
            h_poly=h_poly,
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            omega0=float(omega0),
            g_ep=float(g_ep),
            dv=float(dv),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            ordering=str(ordering),
            boundary=str(boundary),
            paop_r=int(paop_r),
            paop_split_paulis=bool(paop_split_paulis),
            paop_prune_eps=float(paop_prune_eps),
            paop_normalization=str(paop_normalization),
            num_particles=num_particles,
        )
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_full_meta_pool_built",
                **full_meta_sizes,
                dedup_total=int(len(pool_full)),
            )
        if full_meta_class_filter_spec is not None:
            pool_full, full_meta_class_filter_meta = _filter_hh_full_meta_pool_by_class(
                pool_full,
                full_meta_class_filter_spec,
            )
            if callable(ai_log):
                ai_log(
                    "hardcoded_adapt_full_meta_class_filter_applied",
                    **dict(full_meta_class_filter_meta),
                )
        if full_meta_label_filter_spec is not None:
            pool_full, full_meta_label_filter_meta = _filter_hh_full_meta_pool_by_label(
                pool_full,
                full_meta_label_filter_spec,
            )
            if callable(ai_log):
                ai_log(
                    "hardcoded_adapt_full_meta_label_filter_applied",
                    **dict(full_meta_label_filter_meta),
                )
        return list(pool_full), "hardcoded_adapt_vqe_full_meta", full_meta_class_filter_meta, full_meta_label_filter_meta

    if key == "pareto_lean":
        pool_lean, lean_sizes = _build_hh_pareto_lean_pool(
            h_poly=h_poly,
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            omega0=float(omega0),
            g_ep=float(g_ep),
            dv=float(dv),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            ordering=str(ordering),
            boundary=str(boundary),
            paop_r=int(paop_r),
            paop_split_paulis=bool(paop_split_paulis),
            paop_prune_eps=float(paop_prune_eps),
            paop_normalization=str(paop_normalization),
            num_particles=num_particles,
        )
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_pareto_lean_pool_built",
                **lean_sizes,
                dedup_total=int(len(pool_lean)),
            )
        return list(pool_lean), "hardcoded_adapt_vqe_pareto_lean", None, None

    if key == "pareto_lean_l3":
        pool_lean_l3, lean_l3_sizes = _build_hh_pareto_lean_l3_pool(
            h_poly=h_poly,
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            omega0=float(omega0),
            g_ep=float(g_ep),
            dv=float(dv),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            ordering=str(ordering),
            boundary=str(boundary),
            paop_r=int(paop_r),
            paop_split_paulis=bool(paop_split_paulis),
            paop_prune_eps=float(paop_prune_eps),
            paop_normalization=str(paop_normalization),
            num_particles=num_particles,
        )
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_pareto_lean_l3_pool_built",
                **lean_l3_sizes,
                dedup_total=int(len(pool_lean_l3)),
            )
        return list(pool_lean_l3), "hardcoded_adapt_vqe_pareto_lean_l3", None, None

    if key == "pareto_lean_l2":
        pool_lean_l2, lean_l2_sizes = _build_hh_pareto_lean_l2_pool(
            h_poly=h_poly,
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            omega0=float(omega0),
            g_ep=float(g_ep),
            dv=float(dv),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            ordering=str(ordering),
            boundary=str(boundary),
            paop_r=int(paop_r),
            paop_split_paulis=bool(paop_split_paulis),
            paop_prune_eps=float(paop_prune_eps),
            paop_normalization=str(paop_normalization),
            num_particles=num_particles,
        )
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_pareto_lean_l2_pool_built",
                **lean_l2_sizes,
                dedup_total=int(len(pool_lean_l2)),
            )
        return list(pool_lean_l2), "hardcoded_adapt_vqe_pareto_lean_l2", None, None

    if key == "pareto_lean_gate_pruned":
        pool_gp, gp_sizes = _build_hh_pareto_lean_gate_pruned_pool(
            h_poly=h_poly,
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            omega0=float(omega0),
            g_ep=float(g_ep),
            dv=float(dv),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            ordering=str(ordering),
            boundary=str(boundary),
            paop_r=int(paop_r),
            paop_split_paulis=bool(paop_split_paulis),
            paop_prune_eps=float(paop_prune_eps),
            paop_normalization=str(paop_normalization),
            num_particles=num_particles,
        )
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_pareto_lean_gate_pruned_pool_built",
                **gp_sizes,
                dedup_total=int(len(pool_gp)),
            )
        return list(pool_gp), "hardcoded_adapt_vqe_pareto_lean_gate_pruned", None, None

    if key == "uccsd_paop_lf_full":
        uccsd_lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
            int(num_sites),
            int(n_ph_max),
            str(boson_encoding),
            str(ordering),
            str(boundary),
            num_particles=num_particles,
        )
        paop_pool = _build_paop_pool(
            int(num_sites),
            int(n_ph_max),
            str(boson_encoding),
            str(ordering),
            str(boundary),
            "paop_lf_full",
            int(paop_r),
            bool(paop_split_paulis),
            float(paop_prune_eps),
            str(paop_normalization),
            num_particles,
        )
        return _deduplicate_pool_terms(list(uccsd_lifted_pool) + list(paop_pool)), "hardcoded_adapt_vqe_uccsd_paop_lf_full", None, None

    if key in {
        "paop",
        "paop_min",
        "paop_std",
        "paop_full",
        "paop_lf",
        "paop_lf_std",
        "paop_lf2_std",
        "paop_lf3_std",
        "paop_lf4_std",
        "paop_lf_full",
        "paop_sq_std",
        "paop_sq_full",
        "paop_bond_disp_std",
        "paop_hop_sq_std",
        "paop_pair_sq_std",
    }:
        paop_pool = _build_paop_pool(
            int(num_sites),
            int(n_ph_max),
            str(boson_encoding),
            str(ordering),
            str(boundary),
            key,
            int(paop_r),
            bool(paop_split_paulis),
            float(paop_prune_eps),
            str(paop_normalization),
            num_particles,
        )
        if abs(float(g_ep)) <= 1e-15:
            return list(paop_pool), f"hardcoded_adapt_vqe_{key}", None, None
        hva_pool = _build_hva_pool(
            int(num_sites),
            float(t),
            float(u),
            float(omega0),
            float(g_ep),
            float(dv),
            int(n_ph_max),
            str(boson_encoding),
            str(ordering),
            str(boundary),
        )
        ham_term_pool = _build_hh_termwise_augmented_pool(h_poly)
        merged_pool = list(hva_pool) + [
            AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
            for term in ham_term_pool
        ] + list(paop_pool)
        seen: set[tuple[tuple[str, float], ...]] = set()
        dedup_pool: list[AnsatzTerm] = []
        for term in merged_pool:
            sig = _polynomial_signature(term.polynomial)
            if sig in seen:
                continue
            seen.add(sig)
            dedup_pool.append(term)
        return dedup_pool, f"hardcoded_adapt_vqe_{key}", None, None

    if key in {"vlf_only", "sq_only", "vlf_sq", "sq_dens_only", "vlf_sq_dens"}:
        vlf_pool, _vlf_meta = _build_vlf_sq_pool(
            int(num_sites),
            int(n_ph_max),
            str(boson_encoding),
            str(ordering),
            str(boundary),
            key,
            int(paop_r),
            bool(paop_split_paulis),
            float(paop_prune_eps),
            str(paop_normalization),
            num_particles,
        )
        return list(vlf_pool), f"hardcoded_adapt_vqe_{key}", None, None

    if key == "full_hamiltonian":
        return _build_full_hamiltonian_pool(h_poly, normalize_coeff=True), "hardcoded_adapt_vqe_full_hamiltonian_hh", None, None

    raise ValueError(
        "For problem='hh', supported ADAPT pools are: "
        "hva, full_meta, pareto_lean, pareto_lean_l2, pareto_lean_gate_pruned, uccsd_paop_lf_full, paop, paop_min, paop_std, paop_full, "
        "paop_lf, paop_lf_std, paop_lf2_std, paop_lf_full, full_hamiltonian"
    )


__all__ = [
    "HHFullMetaClassFilterSpec",
    "HHFullMetaLabelFilterSpec",
    "_build_hh_full_meta_pool",
    "_build_hh_pareto_lean_gate_pruned_pool",
    "_build_hh_pareto_lean_l2_pool",
    "_build_hh_pareto_lean_l3_pool",
    "_build_hh_pareto_lean_pool",
    "_filter_hh_full_meta_pool_by_class",
    "_filter_hh_full_meta_pool_by_label",
    "_load_hh_full_meta_class_filter_spec",
    "_load_hh_full_meta_label_filter_spec",
    "build_hh_pool_by_key",
]
