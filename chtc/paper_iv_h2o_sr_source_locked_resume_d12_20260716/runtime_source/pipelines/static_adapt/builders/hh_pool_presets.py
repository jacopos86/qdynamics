"""HH preset pool composition helpers extracted from the static ADAPT pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm

from pipelines.contracts.static_provenance import (
    HHFullMetaClassFilterSpec,
    HHFullMetaLabelFilterSpec,
    HH_FULL_META_ALLOWED_CLASSES,
    HH_FULL_META_CLASSIFIER_VERSION,
    HH_MATH_MD_FULL_META_DISPLAY_NAME,
    HH_MATH_MD_FULL_META_POOL_ALIASES,
    HH_MATH_MD_FULL_META_POOL_KEY,
    classify_hh_full_meta_label,
    load_hh_full_meta_class_filter_spec,
    load_hh_full_meta_label_filter_spec,
    normalize_hh_full_meta_keep_classes,
    summarize_hh_full_meta_pool_classes,
)

from .legal_subspace_filter import sanitize_pool_for_binary_boson_legal_subspace
from .primitive_pools import (
    _HH_UCCSD_PAOP_PRODUCT_SPECS,
    _build_full_hamiltonian_pool,
    _build_hamiltonian_blocks_pool,
    _build_hh_fermionic_reusable_pool,
    _build_hh_pure_phonon_pool,
    _build_hh_termwise_augmented_pool,
    _build_hh_sq_lf_pool,
    _build_hh_uccsd_paop_product_pool,
    _build_hh_uccsd_fermion_lifted_pool,
    _build_hva_pool,
    _build_paop_pool,
    _build_vlf_sq_pool,
    _deduplicate_pool_terms,
    _deduplicate_pool_terms_lightweight,
    _polynomial_signature,
)

_HH_FULL_META_CLASSIFIER_VERSION = HH_FULL_META_CLASSIFIER_VERSION
_HH_MATH_MD_FULL_META_POOL_ALIASES = HH_MATH_MD_FULL_META_POOL_ALIASES
_HH_FULL_META_ALLOWED_CLASSES = HH_FULL_META_ALLOWED_CLASSES

_HH_POOL_CACHE_ENV = "STATIC_ADAPT_HH_POOL_CACHE"
_HH_POOL_CACHE_DIR_ENV = "STATIC_ADAPT_HH_POOL_CACHE_DIR"
_HH_POOL_CACHE_SCOPE_ENV = "STATIC_ADAPT_HH_POOL_CACHE_SCOPE"
_HH_POOL_CACHE_DISABLED_VALUES = {"", "0", "off", "false", "no", "disabled", "disable", "none"}
_HH_POOL_CACHE_MEMORY_ONLY_VALUES = {"memory", "mem", "ram", "process"}
_HH_POOL_CACHE_SCHEMA = "hh_pool_cache_v1"
_HH_POOL_CACHE_CODE_VERSION = "hh_pool_cache_code_20260625_full_meta_pure_phonon_v1"
_HH_POOL_CACHE_SCOPE_EXACT = "exact"
_HH_POOL_CACHE_SCOPE_PAPER_I_HOLSTEIN_SECTOR = "paper_i_holstein_sector"
_HH_POOL_CACHE_BYTES: dict[str, bytes] = {}

_HH_FULL_META_EXTRA_PAOP_KEYS = (
    "paop_min",
    "paop_std",
    "paop_lf_std",
    "paop_lf2_std",
)
_HH_FULL_META_OPTIONAL_PAOP_KEYS = (
    "paop_lf3_std",
    "paop_lf4_std",
    "paop_sq_std",
    "paop_sq_full",
    "paop_bond_disp_std",
    "paop_hop_sq_std",
    "paop_pair_sq_std",
)
_HH_FULL_META_OPTIONAL_VLF_SQ_KEYS = (
    "vlf_only",
    "sq_only",
    "vlf_sq",
    "sq_dens_only",
    "vlf_sq_dens",
)
_HH_FULL_META_PRODUCT_KEYS = tuple(_HH_UCCSD_PAOP_PRODUCT_SPECS.keys())
_HH_FULL_META_EXCLUDED_COMPONENT_KEYS = (
    "full_hamiltonian",
    "paop",
    "paop_lf",
    "uccsd_paop_lf_full",
    "pareto_lean",
    "pareto_lean_l2",
    "pareto_lean_l3",
    "pareto_lean_gate_pruned",
)
_classify_hh_full_meta_label = classify_hh_full_meta_label
_normalize_hh_full_meta_keep_classes = normalize_hh_full_meta_keep_classes
_load_hh_full_meta_class_filter_spec = load_hh_full_meta_class_filter_spec
_load_hh_full_meta_label_filter_spec = load_hh_full_meta_label_filter_spec
_summarize_hh_full_meta_pool_classes = summarize_hh_full_meta_pool_classes


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
    full_meta_class_filter_spec: HHFullMetaClassFilterSpec | None = None,
    ai_log: Callable[..., None] | None = None,
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    merged: list[AnsatzTerm] = []
    required_component_keys = (
        "uccsd_lifted",
        "hva",
        "hh_termwise_augmented",
        "paop_full",
        "paop_lf_full",
        *_HH_FULL_META_EXTRA_PAOP_KEYS,
        "hamiltonian_blocks",
        "hh_fermionic_reusable",
    )
    optional_component_keys = (
        *_HH_FULL_META_OPTIONAL_PAOP_KEYS,
        *_HH_FULL_META_OPTIONAL_VLF_SQ_KEYS,
        *_HH_FULL_META_PRODUCT_KEYS,
    )
    meta: dict[str, Any] = {
        "pool_surface_key": HH_MATH_MD_FULL_META_POOL_KEY,
        "pool_display_name": HH_MATH_MD_FULL_META_DISPLAY_NAME,
        "pool_surface_source": "MATH/Math.md HH operator families wired through full-meta builder",
        "required_component_keys": [str(x) for x in required_component_keys],
        "optional_component_keys": [str(x) for x in optional_component_keys],
        "excluded_component_keys": [str(x) for x in _HH_FULL_META_EXCLUDED_COMPONENT_KEYS],
    }
    built_component_keys: list[str] = []
    skipped_component_keys: list[str] = []
    skipped_component_errors: dict[str, str] = {}
    class_filter_skipped_component_keys: list[str] = []
    class_filter_skipped_classes: list[str] = []
    class_keep_set = (
        set(str(x) for x in full_meta_class_filter_spec.keep_classes)
        if full_meta_class_filter_spec is not None
        else None
    )
    skip_hva_component = bool(class_keep_set is not None and "hva_layer" not in class_keep_set)

    def _append_component(
        component_key: str,
        pool_terms: Sequence[AnsatzTerm],
        component_meta: Mapping[str, Any] | None = None,
    ) -> None:
        built_component_keys.append(str(component_key))
        meta[f"raw_{component_key}"] = int(len(pool_terms))
        if component_meta is not None:
            meta[f"component_meta_{component_key}"] = dict(component_meta)
        merged.extend(list(pool_terms))

    def _log_built(component_key: str, pool_terms: Sequence[AnsatzTerm], started_at: float) -> None:
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_full_meta_subpool_built",
                subpool=str(component_key),
                size=int(len(pool_terms)),
                elapsed_s=float(time.perf_counter() - started_at),
            )

    def _mark_optional_skip(component_key: str, exc: Exception, started_at: float) -> None:
        skipped_component_keys.append(str(component_key))
        skipped_component_errors[str(component_key)] = f"{type(exc).__name__}: {exc}"
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_full_meta_subpool_skipped",
                subpool=str(component_key),
                error_type=type(exc).__name__,
                error=str(exc),
                elapsed_s=float(time.perf_counter() - started_at),
            )

    def _is_optional_paop_unavailable(exc: Exception) -> bool:
        return isinstance(exc, ValueError) and "PAOP pool name must be one of" in str(exc)

    def _is_optional_vlf_sq_unavailable(exc: Exception) -> bool:
        return isinstance(exc, RuntimeError) or (
            isinstance(exc, ValueError) and "--paop-split-paulis" in str(exc)
        )

    def _is_optional_product_unavailable(exc: Exception) -> bool:
        return isinstance(exc, RuntimeError) or (
            isinstance(exc, ValueError) and "--paop-split-paulis" in str(exc)
        )

    uccsd_lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        num_particles=num_particles,
    )
    _append_component("uccsd_lifted", uccsd_lifted_pool)
    _log_built("uccsd_lifted", uccsd_lifted_pool, time.perf_counter())

    if skip_hva_component:
        class_filter_skipped_component_keys.append("hva")
        class_filter_skipped_classes.append("hva_layer")
        meta["raw_hva"] = 0
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_full_meta_subpool_class_filter_skipped",
                subpool="hva",
                skipped_class="hva_layer",
            )
    else:
        hva_t0 = time.perf_counter()
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
            include_lifted_uccsd=False,
        )
        _append_component("hva", hva_pool)
        _log_built("hva", hva_pool, hva_t0)
    termwise_aug = [
        AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
        for term in _build_hh_termwise_augmented_pool(h_poly)
    ]
    _append_component("hh_termwise_augmented", termwise_aug)
    _log_built("hh_termwise_augmented", termwise_aug, time.perf_counter())
    pure_phonon_t0 = time.perf_counter()
    pure_phonon_pool, pure_phonon_meta = _build_hh_pure_phonon_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        float(paop_prune_eps),
        str(paop_normalization),
    )
    _append_component("hh_pure_phonon", pure_phonon_pool, pure_phonon_meta)
    _log_built("hh_pure_phonon", pure_phonon_pool, pure_phonon_t0)
    paop_full_t0 = time.perf_counter()
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
    _append_component("paop_full", paop_full_pool)
    _log_built("paop_full", paop_full_pool, paop_full_t0)
    paop_lf_full_t0 = time.perf_counter()
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
    _append_component("paop_lf_full", paop_lf_full_pool)
    _log_built("paop_lf_full", paop_lf_full_pool, paop_lf_full_t0)

    for pool_key in _HH_FULL_META_EXTRA_PAOP_KEYS:
        t0 = time.perf_counter()
        pool_terms = _build_paop_pool(
            int(num_sites),
            int(n_ph_max),
            str(boson_encoding),
            str(ordering),
            str(boundary),
            str(pool_key),
            int(paop_r),
            bool(paop_split_paulis),
            float(paop_prune_eps),
            str(paop_normalization),
            num_particles,
        )
        _append_component(str(pool_key), pool_terms)
        _log_built(str(pool_key), pool_terms, t0)

    ham_blocks_t0 = time.perf_counter()
    ham_blocks_pool = _build_hamiltonian_blocks_pool(
        problem_key="hh",
        num_sites=int(num_sites),
        t=float(t),
        u=float(u),
        dv=float(dv),
        v_nn=0.0,
        t_prime=0.0,
        omega0=float(omega0),
        g_ep=float(g_ep),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
    )
    _append_component("hamiltonian_blocks", ham_blocks_pool)
    _log_built("hamiltonian_blocks", ham_blocks_pool, ham_blocks_t0)

    reusable_t0 = time.perf_counter()
    reusable_pool = _build_hh_fermionic_reusable_pool(
        num_sites=int(num_sites),
        t=float(t),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        prune_eps=0.0,
    )
    _append_component("hh_fermionic_reusable", reusable_pool)
    _log_built("hh_fermionic_reusable", reusable_pool, reusable_t0)

    for pool_key in _HH_FULL_META_OPTIONAL_PAOP_KEYS:
        t0 = time.perf_counter()
        try:
            pool_terms = _build_paop_pool(
                int(num_sites),
                int(n_ph_max),
                str(boson_encoding),
                str(ordering),
                str(boundary),
                str(pool_key),
                int(paop_r),
                bool(paop_split_paulis),
                float(paop_prune_eps),
                str(paop_normalization),
                num_particles,
            )
        except ValueError as exc:
            if not _is_optional_paop_unavailable(exc):
                raise
            _mark_optional_skip(str(pool_key), exc, t0)
            continue
        _append_component(str(pool_key), pool_terms)
        _log_built(str(pool_key), pool_terms, t0)

    for pool_key in _HH_FULL_META_OPTIONAL_VLF_SQ_KEYS:
        t0 = time.perf_counter()
        try:
            pool_terms, component_meta = _build_vlf_sq_pool(
                int(num_sites),
                int(n_ph_max),
                str(boson_encoding),
                str(ordering),
                str(boundary),
                str(pool_key),
                int(paop_r),
                bool(paop_split_paulis),
                float(paop_prune_eps),
                str(paop_normalization),
                num_particles,
            )
        except (RuntimeError, ValueError) as exc:
            if not _is_optional_vlf_sq_unavailable(exc):
                raise
            _mark_optional_skip(str(pool_key), exc, t0)
            continue
        wrapped_pool = [
            AnsatzTerm(
                label=f"hh_vlf_sq::{pool_key}::{term.label}",
                polynomial=term.polynomial,
            )
            for term in pool_terms
        ]
        _append_component(str(pool_key), wrapped_pool, component_meta)
        _log_built(str(pool_key), wrapped_pool, t0)

    for pool_key in _HH_FULL_META_PRODUCT_KEYS:
        t0 = time.perf_counter()
        try:
            pool_terms, component_meta = _build_hh_uccsd_paop_product_pool(
                int(num_sites),
                int(n_ph_max),
                str(boson_encoding),
                str(ordering),
                str(boundary),
                str(pool_key),
                int(paop_r),
                bool(paop_split_paulis),
                float(paop_prune_eps),
                str(paop_normalization),
                num_particles,
                allow_paop_pool_motif_adapter=True,
            )
        except (RuntimeError, ValueError) as exc:
            if not _is_optional_product_unavailable(exc):
                raise
            _mark_optional_skip(str(pool_key), exc, t0)
            continue
        _append_component(str(pool_key), pool_terms, component_meta)
        _log_built(str(pool_key), pool_terms, t0)

    meta["raw_total"] = int(len(merged))
    meta["built_component_keys"] = list(built_component_keys)
    meta["skipped_component_keys"] = list(skipped_component_keys)
    meta["skipped_component_errors"] = dict(skipped_component_errors)
    if class_filter_skipped_component_keys:
        meta["class_filter_skipped_component_keys"] = list(class_filter_skipped_component_keys)
        meta["class_filter_skipped_classes"] = list(class_filter_skipped_classes)
    dedup_t0 = time.perf_counter()
    if int(n_ph_max) >= 2:
        dedup_pool = _deduplicate_pool_terms_lightweight(merged)
    else:
        dedup_pool = _deduplicate_pool_terms(merged)
    if callable(ai_log):
        ai_log(
            "hardcoded_adapt_full_meta_dedup_done",
            raw_total=int(len(merged)),
            dedup_total=int(len(dedup_pool)),
            elapsed_s=float(time.perf_counter() - dedup_t0),
        )
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


def _hh_pool_cache_mode() -> str:
    raw = os.environ.get(_HH_POOL_CACHE_ENV, "disk")
    value = str(raw).strip().lower()
    if value in _HH_POOL_CACHE_DISABLED_VALUES:
        return "off"
    if value in _HH_POOL_CACHE_MEMORY_ONLY_VALUES:
        return "memory"
    return "disk"


def _hh_pool_cache_dir() -> Path:
    raw = os.environ.get(_HH_POOL_CACHE_DIR_ENV)
    if raw is not None and str(raw).strip() != "":
        return Path(str(raw)).expanduser()
    return Path("raw_outputs") / "cache" / _HH_POOL_CACHE_SCHEMA


def _hh_pool_cache_requested_scope() -> str:
    raw = os.environ.get(_HH_POOL_CACHE_SCOPE_ENV, _HH_POOL_CACHE_SCOPE_EXACT)
    value = str(raw).strip().lower().replace("-", "_")
    if value in {"", _HH_POOL_CACHE_SCOPE_EXACT, "regime", "exact_regime"}:
        return _HH_POOL_CACHE_SCOPE_EXACT
    if value in {
        "paper_i_holstein_sector",
        "paper_i_holstein_sector_v1",
        "holstein_sector",
        "structural_holstein_sector",
    }:
        return _HH_POOL_CACHE_SCOPE_PAPER_I_HOLSTEIN_SECTOR
    return _HH_POOL_CACHE_SCOPE_EXACT


def _hh_pool_cache_effective_scope(
    *,
    key: str,
    n_ph_max: int,
    full_meta_class_filter_spec: HHFullMetaClassFilterSpec | None,
) -> str:
    requested = _hh_pool_cache_requested_scope()
    if requested != _HH_POOL_CACHE_SCOPE_PAPER_I_HOLSTEIN_SECTOR:
        return _HH_POOL_CACHE_SCOPE_EXACT
    if str(key).strip().lower() not in set(_HH_MATH_MD_FULL_META_POOL_ALIASES):
        return _HH_POOL_CACHE_SCOPE_EXACT
    keep_classes = (
        set(str(x) for x in full_meta_class_filter_spec.keep_classes)
        if full_meta_class_filter_spec is not None
        else None
    )
    if keep_classes is None or "hva_layer" in keep_classes:
        return _HH_POOL_CACHE_SCOPE_EXACT
    if int(n_ph_max) not in {2, 4}:
        return _HH_POOL_CACHE_SCOPE_EXACT
    return _HH_POOL_CACHE_SCOPE_PAPER_I_HOLSTEIN_SECTOR


def _hh_paper_i_holstein_sector_label(n_ph_max: int) -> str:
    if int(n_ph_max) == 2:
        return "weak_holstein_nph2"
    if int(n_ph_max) == 4:
        return "strong_holstein_nph4"
    return f"nonpaper_i_nph{int(n_ph_max)}"


def _polynomial_support_signature(poly: Any, tol: float = 1e-12) -> tuple[str, ...]:
    labels: set[str] = set()
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        labels.add(label)
    return tuple(sorted(labels))


def _hh_pool_cache_normalize(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return _hh_pool_cache_normalize(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _hh_pool_cache_normalize(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_hh_pool_cache_normalize(item) for item in value]
    if isinstance(value, set):
        return [_hh_pool_cache_normalize(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return format(float(value), ".17g")
    return str(value)


def _hh_pool_cache_key_payload(
    *,
    key: str,
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
    full_meta_class_filter_spec: HHFullMetaClassFilterSpec | None,
    full_meta_label_filter_spec: HHFullMetaLabelFilterSpec | None,
) -> dict[str, Any]:
    cache_scope = _hh_pool_cache_effective_scope(
        key=str(key),
        n_ph_max=int(n_ph_max),
        full_meta_class_filter_spec=full_meta_class_filter_spec,
    )
    params: dict[str, Any] = {
        "num_sites": int(num_sites),
        "t": format(float(t), ".17g"),
        "u": format(float(u), ".17g"),
        "omega0": format(float(omega0), ".17g"),
        "g_ep": format(float(g_ep), ".17g"),
        "dv": format(float(dv), ".17g"),
        "n_ph_max": int(n_ph_max),
        "boson_encoding": str(boson_encoding),
        "ordering": str(ordering),
        "boundary": str(boundary),
        "paop_r": int(paop_r),
        "paop_split_paulis": bool(paop_split_paulis),
        "paop_prune_eps": format(float(paop_prune_eps), ".17g"),
        "paop_normalization": str(paop_normalization),
        "num_particles": [int(x) for x in tuple(num_particles)],
    }
    hamiltonian_signature: Any = _hh_pool_cache_normalize(_polynomial_signature(h_poly))
    cache_scope_payload: dict[str, Any] = {"cache_scope": _HH_POOL_CACHE_SCOPE_EXACT}
    if cache_scope == _HH_POOL_CACHE_SCOPE_PAPER_I_HOLSTEIN_SECTOR:
        # Paper-I HH speed path: reuse one legal-filtered adaptive pool per
        # Holstein sector across weak/intermediate/strong Hubbard U. Keep g_ep
        # and n_ph in the key; exclude only U and refresh Hamiltonian-block
        # coefficients on cache hit.
        params.pop("u", None)
        params["u_nonzero"] = bool(abs(float(u)) > 1e-15)
        hamiltonian_signature = _hh_pool_cache_normalize(_polynomial_support_signature(h_poly))
        cache_scope_payload = {
            "cache_scope": _HH_POOL_CACHE_SCOPE_PAPER_I_HOLSTEIN_SECTOR,
            "structural_cache_version": "paper_i_holstein_sector_20260614_v1",
            "paper_i_holstein_sector": _hh_paper_i_holstein_sector_label(int(n_ph_max)),
            "hubbard_u_policy": "excluded_from_key_reinstantiate_ham_blocks",
        }
    return {
        "schema": _HH_POOL_CACHE_SCHEMA,
        "code_version": _HH_POOL_CACHE_CODE_VERSION,
        "python_pickle_abi": f"{sys.version_info.major}.{sys.version_info.minor}",
        "classifier_version": _HH_FULL_META_CLASSIFIER_VERSION,
        "math_md_full_meta_pool_key": HH_MATH_MD_FULL_META_POOL_KEY,
        "pool_key_hh": str(key),
        "hamiltonian_signature": hamiltonian_signature,
        **cache_scope_payload,
        "params": params,
        "full_meta_class_filter_spec": _hh_pool_cache_normalize(full_meta_class_filter_spec),
        "full_meta_label_filter_spec": _hh_pool_cache_normalize(full_meta_label_filter_spec),
    }


def _hh_pool_cache_digest(key_payload: Mapping[str, Any]) -> str:
    key_json = json.dumps(key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(key_json.encode("utf-8")).hexdigest()


def _hh_pool_cache_path(digest: str) -> Path:
    return _hh_pool_cache_dir() / f"{digest}.pickle"


def _hh_pool_cache_load(
    *,
    key_payload: Mapping[str, Any],
    digest: str,
    ai_log: Callable[..., None] | None,
) -> dict[str, Any] | None:
    mode = _hh_pool_cache_mode()
    if mode == "off":
        return None
    cache_level = "memory"
    blob = _HH_POOL_CACHE_BYTES.get(str(digest))
    cache_path: Path | None = None
    if blob is None and mode == "disk":
        cache_path = _hh_pool_cache_path(str(digest))
        try:
            if cache_path.is_file():
                blob = cache_path.read_bytes()
                _HH_POOL_CACHE_BYTES[str(digest)] = blob
                cache_level = "disk"
        except Exception as exc:  # pragma: no cover - defensive cache fallback
            if callable(ai_log):
                ai_log(
                    "hardcoded_adapt_pool_cache_load_failed",
                    cache_key=str(digest),
                    cache_path=str(cache_path),
                    error=str(exc),
                )
            return None
    if blob is None:
        return None
    try:
        payload = pickle.loads(blob)
    except Exception as exc:  # pragma: no cover - defensive cache fallback
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_pool_cache_decode_failed",
                cache_key=str(digest),
                cache_path=(str(cache_path) if cache_path is not None else None),
                error=str(exc),
            )
        _HH_POOL_CACHE_BYTES.pop(str(digest), None)
        return None
    if not isinstance(payload, dict) or payload.get("schema") != _HH_POOL_CACHE_SCHEMA or payload.get("key") != dict(key_payload):
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_pool_cache_ignored",
                cache_key=str(digest),
                reason="schema_or_key_mismatch",
            )
        return None
    if callable(ai_log):
        ai_log(
            "hardcoded_adapt_pool_cache_hit",
            cache_key=str(digest),
            cache_level=str(cache_level),
            cache_path=(str(cache_path) if cache_path is not None else str(_hh_pool_cache_path(str(digest))) if mode == "disk" else None),
            pool_key=str(payload.get("pool_key_hh")),
            pool_size=int(len(payload.get("pool", []))),
            method_name=str(payload.get("method_name")),
            cache_scope=str(payload.get("cache_scope", dict(key_payload).get("cache_scope", _HH_POOL_CACHE_SCOPE_EXACT))),
            paper_i_holstein_sector=dict(key_payload).get("paper_i_holstein_sector"),
        )
    return payload


def _hh_pool_cache_store(
    *,
    key_payload: Mapping[str, Any],
    digest: str,
    pool_key_hh: str,
    pool: list[AnsatzTerm],
    method_name: str,
    class_meta: dict[str, Any] | None,
    label_meta: dict[str, Any] | None,
    legal_filter_meta: dict[str, Any] | None,
    ai_log: Callable[..., None] | None,
) -> None:
    mode = _hh_pool_cache_mode()
    if mode == "off":
        return
    payload = {
        "schema": _HH_POOL_CACHE_SCHEMA,
        "key": dict(key_payload),
        "cache_key": str(digest),
        "pool_key_hh": str(pool_key_hh),
        "cache_scope": str(dict(key_payload).get("cache_scope", _HH_POOL_CACHE_SCOPE_EXACT)),
        "paper_i_holstein_sector": dict(key_payload).get("paper_i_holstein_sector"),
        "pool": list(pool),
        "method_name": str(method_name),
        "class_meta": class_meta,
        "label_meta": label_meta,
        "legal_filter_meta": legal_filter_meta,
        "created_unix_s": time.time(),
    }
    try:
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        _HH_POOL_CACHE_BYTES[str(digest)] = blob
    except Exception as exc:  # pragma: no cover - defensive cache fallback
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_pool_cache_encode_failed",
                cache_key=str(digest),
                error=str(exc),
            )
        return
    if mode != "disk":
        return
    cache_dir = _hh_pool_cache_dir()
    cache_path = cache_dir / f"{digest}.pickle"
    tmp_path: Path | None = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=cache_dir, prefix=f".{digest}.", suffix=".tmp") as fh:
            tmp_path = Path(fh.name)
            fh.write(blob)
        os.replace(tmp_path, cache_path)
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_pool_cache_stored",
                cache_key=str(digest),
                cache_path=str(cache_path),
                pool_key=str(pool_key_hh),
                pool_size=int(len(pool)),
                bytes=int(len(blob)),
                cache_scope=str(dict(key_payload).get("cache_scope", _HH_POOL_CACHE_SCOPE_EXACT)),
                paper_i_holstein_sector=dict(key_payload).get("paper_i_holstein_sector"),
            )
    except Exception as exc:  # pragma: no cover - defensive cache fallback
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_pool_cache_store_failed",
                cache_key=str(digest),
                cache_path=str(cache_path),
                error=str(exc),
            )


def _hh_pool_cache_is_paper_i_holstein_sector(payload: Mapping[str, Any]) -> bool:
    key_payload = payload.get("key") if isinstance(payload, Mapping) else None
    return bool(
        isinstance(key_payload, Mapping)
        and str(key_payload.get("cache_scope")) == _HH_POOL_CACHE_SCOPE_PAPER_I_HOLSTEIN_SECTOR
    )


def _hh_pool_cache_reinstantiate_paper_i_holstein_sector(
    payload: Mapping[str, Any],
    *,
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
) -> dict[str, Any]:
    """Refresh U-dependent HH Hamiltonian block polynomials for a structural hit."""
    if not _hh_pool_cache_is_paper_i_holstein_sector(payload):
        return dict(payload)
    current_blocks = _build_hamiltonian_blocks_pool(
        problem_key="hh",
        num_sites=int(num_sites),
        t=float(t),
        u=float(u),
        dv=float(dv),
        v_nn=0.0,
        t_prime=0.0,
        omega0=float(omega0),
        g_ep=float(g_ep),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
    )
    current_by_label = {str(term.label): term for term in current_blocks}
    pool: list[AnsatzTerm] = []
    cached_ham_labels: list[str] = []
    refreshed = 0
    for term in list(payload.get("pool", [])):
        label = str(getattr(term, "label", ""))
        if label.startswith("ham_block::"):
            cached_ham_labels.append(label)
            current = current_by_label.get(label)
            if current is None:
                raise ValueError(
                    f"Paper-I Holstein-sector cache hit cannot refresh missing HH Hamiltonian block {label!r}."
                )
            pool.append(
                AnsatzTerm(
                    label=label,
                    polynomial=current.polynomial,
                    execution_mode=str(getattr(term, "execution_mode", "termwise_product")),
                )
            )
            refreshed += 1
        else:
            pool.append(term)
    extra_current = sorted(set(current_by_label) - set(cached_ham_labels))
    if extra_current:
        raise ValueError(
            "Paper-I Holstein-sector cache hit has stale HH Hamiltonian block support; "
            f"extra current labels: {extra_current[:8]}"
        )
    out = dict(payload)
    out["pool"] = pool
    legal_meta = out.get("legal_filter_meta")
    if isinstance(legal_meta, Mapping):
        legal_meta = dict(legal_meta)
        legal_meta["paper_i_holstein_sector_reinstantiation"] = {
            "enabled": True,
            "refreshed_hamiltonian_block_count": int(refreshed),
            "u": format(float(u), ".17g"),
            "policy": "refresh_ham_block_polynomials_after_structural_cache_hit",
        }
        out["legal_filter_meta"] = legal_meta
    out["paper_i_holstein_sector_reinstantiation"] = {
        "enabled": True,
        "refreshed_hamiltonian_block_count": int(refreshed),
        "u": format(float(u), ".17g"),
    }
    return out


def _hh_pool_cache_return(payload: Mapping[str, Any]):
    return (
        list(payload["pool"]),
        str(payload["method_name"]),
        payload.get("class_meta"),
        payload.get("label_meta"),
        payload.get("legal_filter_meta"),
    )


def clear_hh_pool_cache_memory() -> None:
    """Clear the process-local HH pool cache layer; disk cache files are preserved."""
    _HH_POOL_CACHE_BYTES.clear()



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
    include_legal_subspace_filter_meta: bool = False,
) -> tuple[list[AnsatzTerm], str, dict[str, Any] | None, dict[str, Any] | None] | tuple[list[AnsatzTerm], str, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    key = str(pool_key_hh).strip().lower()
    full_meta_class_filter_meta: dict[str, Any] | None = None
    full_meta_label_filter_meta: dict[str, Any] | None = None
    cache_key_payload: dict[str, Any] | None = None
    cache_digest: str | None = None
    if bool(include_legal_subspace_filter_meta):
        cache_key_payload = _hh_pool_cache_key_payload(
            key=key,
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
            num_particles=tuple(int(x) for x in tuple(num_particles)),
            full_meta_class_filter_spec=full_meta_class_filter_spec,
            full_meta_label_filter_spec=full_meta_label_filter_spec,
        )
        cache_digest = _hh_pool_cache_digest(cache_key_payload)
        cached_payload = _hh_pool_cache_load(
            key_payload=cache_key_payload,
            digest=cache_digest,
            ai_log=ai_log,
        )
        if cached_payload is not None:
            try:
                cached_payload = _hh_pool_cache_reinstantiate_paper_i_holstein_sector(
                    cached_payload,
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
                )
                if callable(ai_log) and _hh_pool_cache_is_paper_i_holstein_sector(cached_payload):
                    rein = cached_payload.get("paper_i_holstein_sector_reinstantiation", {})
                    ai_log(
                        "hardcoded_adapt_pool_cache_reinstantiated",
                        cache_key=str(cache_digest),
                        cache_scope=_HH_POOL_CACHE_SCOPE_PAPER_I_HOLSTEIN_SECTOR,
                        paper_i_holstein_sector=cache_key_payload.get("paper_i_holstein_sector"),
                        refreshed_hamiltonian_block_count=(
                            int(rein.get("refreshed_hamiltonian_block_count", 0))
                            if isinstance(rein, Mapping)
                            else 0
                        ),
                        u=format(float(u), ".17g"),
                    )
                return _hh_pool_cache_return(cached_payload)
            except Exception as exc:
                if callable(ai_log):
                    ai_log(
                        "hardcoded_adapt_pool_cache_reinstantiate_failed",
                        cache_key=str(cache_digest),
                        cache_scope=str(cache_key_payload.get("cache_scope", _HH_POOL_CACHE_SCOPE_EXACT)),
                        error=str(exc),
                    )

    def _return(
        pool: list[AnsatzTerm],
        method_name: str,
        class_meta: dict[str, Any] | None = None,
        label_meta: dict[str, Any] | None = None,
        legal_filter_meta: dict[str, Any] | None = None,
    ):
        if bool(include_legal_subspace_filter_meta):
            if cache_key_payload is not None and cache_digest is not None:
                _hh_pool_cache_store(
                    key_payload=cache_key_payload,
                    digest=cache_digest,
                    pool_key_hh=key,
                    pool=list(pool),
                    method_name=str(method_name),
                    class_meta=class_meta,
                    label_meta=label_meta,
                    legal_filter_meta=legal_filter_meta,
                    ai_log=ai_log,
                )
            return list(pool), str(method_name), class_meta, label_meta, legal_filter_meta
        return list(pool), str(method_name), class_meta, label_meta

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
            return _return(list(hva_pool), "hardcoded_adapt_vqe_hva_hh")
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
        return _return(dedup_pool, "hardcoded_adapt_vqe_hva_hh")

    if key in set(_HH_MATH_MD_FULL_META_POOL_ALIASES):
        requested_full_meta_key = str(key)
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
            full_meta_class_filter_spec=full_meta_class_filter_spec,
            ai_log=ai_log,
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
            full_meta_class_filter_meta["stage"] = "pre_legal_subspace_filter"
            prebuild_skipped_classes = [
                str(x) for x in full_meta_sizes.get("class_filter_skipped_classes", [])
            ]
            prebuild_skipped_components = [
                str(x) for x in full_meta_sizes.get("class_filter_skipped_component_keys", [])
            ]
            if prebuild_skipped_classes:
                existing_dropped = [str(x) for x in full_meta_class_filter_meta.get("dropped_classes", [])]
                full_meta_class_filter_meta["dropped_classes"] = sorted(
                    set(existing_dropped) | set(prebuild_skipped_classes)
                )
                full_meta_class_filter_meta["prebuild_skipped_classes"] = prebuild_skipped_classes
                full_meta_class_filter_meta["prebuild_skipped_component_keys"] = prebuild_skipped_components
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
            full_meta_label_filter_meta["stage"] = "pre_legal_subspace_filter"
            if callable(ai_log):
                ai_log(
                    "hardcoded_adapt_full_meta_label_filter_applied",
                    **dict(full_meta_label_filter_meta),
                )
        pool_full, legal_filter_meta = sanitize_pool_for_binary_boson_legal_subspace(
            pool_full,
            problem_key="hh",
            num_sites=int(num_sites),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            label_classifier=_classify_hh_full_meta_label,
        )
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_pool_legal_subspace_filter_applied",
                **dict(legal_filter_meta),
            )
        if bool(include_legal_subspace_filter_meta):
            method_name = (
                "hardcoded_adapt_vqe_full_meta"
                if requested_full_meta_key == "full_meta"
                else "hardcoded_adapt_vqe_math_md_full_meta_v1"
            )
            legal_filter_payload = dict(legal_filter_meta)
            legal_filter_payload["pool_surface_key"] = HH_MATH_MD_FULL_META_POOL_KEY
            legal_filter_payload["pool_display_name"] = HH_MATH_MD_FULL_META_DISPLAY_NAME
            legal_filter_payload["adapt_pool_requested"] = str(requested_full_meta_key)
            return _return(
                list(pool_full),
                method_name,
                full_meta_class_filter_meta,
                full_meta_label_filter_meta,
                legal_filter_payload,
            )
        method_name = (
            "hardcoded_adapt_vqe_full_meta"
            if requested_full_meta_key == "full_meta"
            else "hardcoded_adapt_vqe_math_md_full_meta_v1"
        )
        return _return(
            list(pool_full),
            method_name,
            full_meta_class_filter_meta,
            full_meta_label_filter_meta,
            dict(legal_filter_meta),
        )

    if key == "sq_lf_std":
        sq_pool, sq_meta = _build_hh_sq_lf_pool(
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
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_sq_lf_pool_built",
                **dict(sq_meta),
            )
        return _return(list(sq_pool), "hardcoded_adapt_vqe_sq_lf_std")

    if key in _HH_FULL_META_PRODUCT_KEYS:
        product_pool, product_meta = _build_hh_uccsd_paop_product_pool(
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
            allow_paop_pool_motif_adapter=True,
        )
        if callable(ai_log):
            ai_log(
                "hardcoded_adapt_uccsd_paop_product_pool_built",
                **dict(product_meta),
            )
        return _return(list(product_pool), f"hardcoded_adapt_vqe_{key}")

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
        return _return(list(pool_lean), "hardcoded_adapt_vqe_pareto_lean")

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
        return _return(list(pool_lean_l3), "hardcoded_adapt_vqe_pareto_lean_l3")

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
        return _return(list(pool_lean_l2), "hardcoded_adapt_vqe_pareto_lean_l2")

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
        return _return(list(pool_gp), "hardcoded_adapt_vqe_pareto_lean_gate_pruned")

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
        return _return(
            _deduplicate_pool_terms(list(uccsd_lifted_pool) + list(paop_pool)),
            "hardcoded_adapt_vqe_uccsd_paop_lf_full",
        )

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
            return _return(list(paop_pool), f"hardcoded_adapt_vqe_{key}")
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
        return _return(dedup_pool, f"hardcoded_adapt_vqe_{key}")

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
        return _return(list(vlf_pool), f"hardcoded_adapt_vqe_{key}")

    if key == "full_hamiltonian":
        return _return(
            _build_full_hamiltonian_pool(h_poly, normalize_coeff=True),
            "hardcoded_adapt_vqe_full_hamiltonian_hh",
        )

    if key == "hamiltonian_blocks":
        return _return(
            _build_hamiltonian_blocks_pool(
                problem_key="hh",
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                dv=float(dv),
                v_nn=0.0,
                t_prime=0.0,
                omega0=float(omega0),
                g_ep=float(g_ep),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
            ),
            "hardcoded_adapt_vqe_hamiltonian_blocks_hh",
        )

    raise ValueError(
        "For problem='hh', supported ADAPT pools are: "
        "hva, full_meta, pareto_lean, pareto_lean_l2, pareto_lean_gate_pruned, uccsd_paop_lf_full, sq_lf_std, paop, paop_min, paop_std, paop_full, "
        "paop_lf, paop_lf_std, paop_lf2_std, paop_lf_full, full_hamiltonian, hamiltonian_blocks"
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
    "clear_hh_pool_cache_memory",
]
