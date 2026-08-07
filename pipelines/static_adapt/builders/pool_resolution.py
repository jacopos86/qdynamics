"""Pool resolution helpers for the static ADAPT pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from src.quantum.hubbard_latex_python_pairs import boson_qubits_per_site
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, half_filled_num_particles
from src.quantum.chemistry.vibronic_h2o_linear_fd import (
    H2O_LINEAR_FD_DERIVATIVE_RESOLVED_POOL_KEY,
    build_h2o_linear_fd_derivative_resolved_pool_v2,
)

from pipelines.scaffold.hh_continuation_generators import build_pool_generator_registry
from pipelines.scaffold.hh_continuation_motifs import (
    load_selected_logical_library_from_json,
    match_selected_logical_generators,
    match_selected_logical_operator_families,
)
from pipelines.scaffold.hh_continuation_symmetry import build_symmetry_spec

from pipelines.contracts.static_provenance import (
    HHFullMetaClassFilterSpec,
    HHFullMetaLabelFilterSpec,
    HH_MATH_MD_FULL_META_POOL_KEY,
    load_hh_full_meta_class_filter_spec,
    load_hh_full_meta_label_filter_spec,
)

from .hh_pool_presets import build_hh_pool_by_key
from .primitive_pools import (
    _build_cse_pool,
    _build_full_meta_pool,
    _build_family_max_pool,
    _build_family_hva_pool,
    _build_full_hamiltonian_pool,
    _build_hamiltonian_blocks_pool,
    _build_hamiltonian_quadratures_pool,
    _build_hubbard_uccsd_qeb_pool,
    _build_hubbard_uccsd_qeb_hva_blocks_pool,
    _build_molecular_uccsd_pool,
    _build_uccsd_pool,
    _hubbard_uccsd_qeb_family_id_for_label,
    _hubbard_uccsd_qeb_hva_family_id_for_label,
    _polynomial_signature,
)
from pipelines.contracts.problem import ResolvedProblemContext

from .problem_registry import resolve_runtime_default_pool_key
from .problem_setup import _HH_STAGED_CONTINUATION_MODES

SELECTED_LOGICAL_MODE_CHOICES = frozenset(
    {
        "off",
        "filter_with_full_fallback",
        "family_closure_with_full_fallback",
        "filter_fail_closed",
        "family_closure_fail_closed",
    }
)
SELECTED_LOGICAL_FAMILY_CLOSURE_MODES = frozenset(
    {
        "family_closure_with_full_fallback",
        "family_closure_fail_closed",
    }
)
SELECTED_LOGICAL_FAIL_CLOSED_MODES = frozenset(
    {
        "filter_fail_closed",
        "family_closure_fail_closed",
    }
)
HH_FULL_META_FILTER_POOL_KEYS = frozenset({"full_meta", "math_md_full_meta", HH_MATH_MD_FULL_META_POOL_KEY})


@dataclass(frozen=True)
class PoolFilterResolution:
    pool_key_input: str | None
    full_meta_class_filter_spec: HHFullMetaClassFilterSpec | None
    full_meta_label_filter_spec: HHFullMetaLabelFilterSpec | None
    selected_logical_source_json: Path | None = None
    selected_logical_mode: str = "off"
    selected_logical_transfer_mode: str = "exact_match_v1"


@dataclass
class PoolResolution:
    pool: list[AnsatzTerm]
    pool_key: str
    method_name: str
    pool_stage_family: list[str]
    pool_family_ids: list[str]
    phase1_core_limit: int
    phase1_residual_indices: set[int]
    phase1_depth0_full_meta_override: bool
    full_meta_class_filter_spec: HHFullMetaClassFilterSpec | None
    full_meta_label_filter_spec: HHFullMetaLabelFilterSpec | None
    full_meta_class_filter_meta: dict[str, Any] | None
    full_meta_label_filter_meta: dict[str, Any] | None
    pool_legal_subspace_filter_meta: dict[str, Any] | None
    selected_logical_filter_meta: dict[str, Any] | None
    pool_symmetry_specs: list[dict[str, Any] | None]
    pool_generator_registry: dict[str, dict[str, Any]]
    qpb: int


def resolve_requested_pool_filters(
    *,
    problem_key: str,
    num_sites: int,
    n_ph_max: int,
    adapt_pool: str | None,
    adapt_pool_class_filter_json: Path | None,
    adapt_pool_label_filter_json: Path | None,
    adapt_selected_logical_source_json: Path | None = None,
    adapt_selected_logical_mode: str = "off",
    adapt_selected_logical_transfer_mode: str = "exact_match_v1",
) -> PoolFilterResolution:
    pool_key_input = None if adapt_pool is None else str(adapt_pool).strip().lower()
    full_meta_class_filter_spec: HHFullMetaClassFilterSpec | None = None
    full_meta_label_filter_spec: HHFullMetaLabelFilterSpec | None = None
    selected_logical_mode_key = str(adapt_selected_logical_mode or "off").strip().lower()
    if selected_logical_mode_key not in SELECTED_LOGICAL_MODE_CHOICES:
        raise ValueError(
            "adapt_selected_logical_mode must be one of "
            f"{sorted(SELECTED_LOGICAL_MODE_CHOICES)}."
        )
    selected_logical_transfer_mode_key = str(adapt_selected_logical_transfer_mode or "exact_match_v1").strip().lower()
    if selected_logical_transfer_mode_key not in {"exact_match_v1", "boundary_v1"}:
        raise ValueError("adapt_selected_logical_transfer_mode must be one of {'exact_match_v1','boundary_v1'}.")
    if adapt_pool_class_filter_json is not None:
        if str(problem_key) != "hh":
            raise ValueError("adapt_pool_class_filter_json is only valid for problem='hh'.")
        if pool_key_input not in HH_FULL_META_FILTER_POOL_KEYS:
            raise ValueError("adapt_pool_class_filter_json is only valid when adapt_pool='full_meta' or 'math_md_full_meta_v1'.")
        full_meta_class_filter_spec = load_hh_full_meta_class_filter_spec(Path(adapt_pool_class_filter_json))
        if (
            full_meta_class_filter_spec.source_num_sites is not None
            and int(full_meta_class_filter_spec.source_num_sites) != int(num_sites)
        ):
            raise ValueError(
                "HH full_meta class filter source_num_sites does not match this run: "
                f"got {full_meta_class_filter_spec.source_num_sites}, expected {num_sites}."
            )
        if (
            full_meta_class_filter_spec.source_n_ph_max is not None
            and int(full_meta_class_filter_spec.source_n_ph_max) != int(n_ph_max)
        ):
            raise ValueError(
                "HH full_meta class filter source_n_ph_max does not match this run: "
                f"got {full_meta_class_filter_spec.source_n_ph_max}, expected {n_ph_max}."
            )
    if adapt_pool_label_filter_json is not None:
        if str(problem_key) != "hh":
            raise ValueError("adapt_pool_label_filter_json is only valid for problem='hh'.")
        if pool_key_input not in HH_FULL_META_FILTER_POOL_KEYS:
            raise ValueError("adapt_pool_label_filter_json is only valid when adapt_pool='full_meta' or 'math_md_full_meta_v1'.")
        full_meta_label_filter_spec = load_hh_full_meta_label_filter_spec(Path(adapt_pool_label_filter_json))
        if (
            full_meta_label_filter_spec.source_num_sites is not None
            and int(full_meta_label_filter_spec.source_num_sites) != int(num_sites)
        ):
            raise ValueError(
                "HH full_meta label filter source_num_sites does not match this run: "
                f"got {full_meta_label_filter_spec.source_num_sites}, expected {num_sites}."
            )
        if (
            full_meta_label_filter_spec.source_n_ph_max is not None
            and int(full_meta_label_filter_spec.source_n_ph_max) != int(n_ph_max)
        ):
            raise ValueError(
                "HH full_meta label filter source_n_ph_max does not match this run: "
                f"got {full_meta_label_filter_spec.source_n_ph_max}, expected {n_ph_max}."
            )
    return PoolFilterResolution(
        pool_key_input=pool_key_input,
        full_meta_class_filter_spec=full_meta_class_filter_spec,
        full_meta_label_filter_spec=full_meta_label_filter_spec,
        selected_logical_source_json=(
            Path(adapt_selected_logical_source_json)
            if adapt_selected_logical_source_json is not None
            else None
        ),
        selected_logical_mode=str(selected_logical_mode_key),
        selected_logical_transfer_mode=str(selected_logical_transfer_mode_key),
    )


def _selected_logical_filter_off_meta(
    *,
    mode: str,
    source_json: Path | None,
    transfer_mode: str,
    pool_size: int,
) -> dict[str, Any]:
    return {
        "schema": "adapt_selected_logical_pool_filter_v1",
        "mode": str(mode),
        "source_json": (str(source_json) if source_json is not None else None),
        "transfer_mode": str(transfer_mode),
        "applied": False,
        "fallback_to_full_pool": False,
        "fallback_reason": None,
        "pool_size_before": int(pool_size),
        "pool_size_after": int(pool_size),
        "selected_record_count": 0,
        "matched_count": 0,
        "match_method_counts": {},
    }


def _fallback_selected_logical_filter_meta(
    *,
    mode: str,
    source_json: Path | None,
    transfer_mode: str,
    pool_size: int,
    reason: str,
    detail: str | None = None,
    selected_record_count: int = 0,
) -> dict[str, Any]:
    meta = _selected_logical_filter_off_meta(
        mode=str(mode),
        source_json=source_json,
        transfer_mode=str(transfer_mode),
        pool_size=int(pool_size),
    )
    meta.update(
        {
            "fallback_to_full_pool": True,
            "fallback_reason": str(reason),
            "fallback_detail": (None if detail in {None, ""} else str(detail)),
            "selected_record_count": int(selected_record_count),
        }
    )
    return meta


def _record_string_values(record: Mapping[str, Any], keys: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for key in keys:
        value = record.get(key)
        values: list[Any]
        if isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            values = [value]
        for item in values:
            if item is None or item == "":
                continue
            text = str(item)
            if text and text not in out:
                out.append(text)
    return out


def _selected_logical_record_samples(records_raw: Any, *, limit: int = 8) -> list[dict[str, Any]]:
    if not isinstance(records_raw, (list, tuple)):
        return []
    samples: list[dict[str, Any]] = []
    for raw in records_raw:
        if not isinstance(raw, Mapping):
            continue
        labels = _record_string_values(raw, ("candidate_label", "operator_label", "label", "selected_op"))
        families = _record_string_values(raw, ("operator_family_ids", "operator_family_id", "family_id"))
        generators = _record_string_values(raw, ("generator_ids", "generator_id"))
        sample: dict[str, Any] = {}
        if labels:
            sample["label"] = labels[0]
            if len(labels) > 1:
                sample["labels"] = labels[:3]
        if families:
            sample["family_ids"] = families[:4]
        if generators:
            sample["generator_id"] = generators[0]
        if raw.get("template_id") not in {None, ""}:
            sample["template_id"] = str(raw.get("template_id"))
        if not sample:
            sample["keys"] = sorted(str(key) for key in raw.keys())[:8]
        samples.append(sample)
        if len(samples) >= int(limit):
            break
    return samples


def _selected_logical_missing_record_samples(
    records_raw: Any,
    matches: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if not isinstance(records_raw, (list, tuple)):
        return []
    matched_labels: set[str] = set()
    matched_families: set[str] = set()
    matched_generators: set[str] = set()
    for match in matches:
        selected = match.get("selected_logical_record")
        if isinstance(selected, Mapping):
            matched_labels.update(_record_string_values(selected, ("candidate_label", "operator_label", "label", "selected_op")))
            matched_families.update(_record_string_values(selected, ("operator_family_ids", "operator_family_id", "family_id")))
            matched_generators.update(_record_string_values(selected, ("generator_ids", "generator_id")))
        matched_families.update(_record_string_values(match, ("selected_operator_family_ids", "operator_family_ids", "operator_family_id")))
    missing: list[Mapping[str, Any]] = []
    for raw in records_raw:
        if not isinstance(raw, Mapping):
            continue
        labels = set(_record_string_values(raw, ("candidate_label", "operator_label", "label", "selected_op")))
        families = set(_record_string_values(raw, ("operator_family_ids", "operator_family_id", "family_id")))
        generators = set(_record_string_values(raw, ("generator_ids", "generator_id")))
        covered = bool((labels & matched_labels) or (families & matched_families) or (generators & matched_generators))
        if not covered:
            missing.append(raw)
        if len(missing) >= int(limit):
            break
    return _selected_logical_record_samples(missing, limit=limit)


def _build_selected_logical_pool_match_report(
    *,
    pool: list[AnsatzTerm],
    pool_family_ids: list[str],
    pool_symmetry_specs: list[dict[str, Any] | None],
    pool_generator_registry: dict[str, dict[str, Any]],
    request: Any,
    qpb: int,
    mode: str,
    source_json: Path | None,
    transfer_mode: str,
    paop_split_paulis: bool,
    ai_log: Callable[..., None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    mode_key = str(mode or "off").strip().lower()
    transfer_mode_key = str(transfer_mode or "exact_match_v1").strip().lower()
    source_path = Path(source_json) if source_json is not None else None
    active_labels = [str(term.label) for term in pool]
    report: dict[str, Any] = {
        "schema": "adapt_selected_logical_pool_match_report_v1",
        "mode": str(mode_key),
        "source_json": (str(source_path) if source_path is not None else None),
        "source_exists": bool(source_path is not None and source_path.exists()),
        "transfer_mode": str(transfer_mode_key),
        "applied": False,
        "fallback_to_full_pool": False,
        "fallback_reason": None,
        "fallback_detail": None,
        "pool_size_before": int(len(pool)),
        "pool_size_after": 0,
        "active_pool_label_count": int(len(active_labels)),
        "active_pool_label_samples": active_labels[:24],
        "selected_record_count": 0,
        "matched_count": 0,
        "match_method_counts": {},
        "closure_mode": (
            "operator_family_v1"
            if mode_key in SELECTED_LOGICAL_FAMILY_CLOSURE_MODES
            else "exact_logical_v1"
        ),
        "operator_family_ids": [],
        "kept_labels_sample": [],
        "selected_label_family_samples": [],
        "missing_label_family_samples": [],
        "match_rows_sample": [],
        "registry_build_error": None,
        "status": "fail",
        "reason": None,
    }
    if mode_key == "off":
        report.update({"status": "pass", "reason": None, "pool_size_after": int(len(pool))})
        return report, [], dict(pool_generator_registry)
    if mode_key not in SELECTED_LOGICAL_MODE_CHOICES - {"off"}:
        report["reason"] = "unsupported_mode"
        report["fallback_detail"] = str(mode_key)
        return report, [], dict(pool_generator_registry)
    if source_path is None:
        report["reason"] = "missing_source_json"
        return report, [], dict(pool_generator_registry)
    try:
        selected_library = load_selected_logical_library_from_json(source_path)
    except Exception as exc:
        report["reason"] = "malformed_source_json"
        report["fallback_detail"] = f"{type(exc).__name__}: {exc}"
        return report, [], dict(pool_generator_registry)
    if not isinstance(selected_library, Mapping):
        report["reason"] = "empty_or_unrecognized_source"
        return report, [], dict(pool_generator_registry)
    records_raw = selected_library.get("records", [])
    selected_record_count = int(len(records_raw)) if isinstance(records_raw, list) else 0
    report.update(
        {
            "source_kind": str(selected_library.get("source_kind", "unknown")),
            "source_tag": str(selected_library.get("source_tag", "payload")),
            "selected_record_count": int(selected_record_count),
            "selected_label_family_samples": _selected_logical_record_samples(records_raw),
        }
    )
    source_ordering = selected_library.get("ordering")
    if source_ordering not in {None, ""} and str(source_ordering) != str(request.ordering):
        report["reason"] = "incompatible_ordering"
        report["fallback_detail"] = f"source={source_ordering}, target={request.ordering}"
        report["missing_label_family_samples"] = _selected_logical_record_samples(records_raw)
        return report, [], dict(pool_generator_registry)
    source_boson_encoding = selected_library.get("boson_encoding")
    if source_boson_encoding not in {None, ""} and str(source_boson_encoding) != str(request.boson_encoding):
        report["reason"] = "incompatible_boson_encoding"
        report["fallback_detail"] = f"source={source_boson_encoding}, target={request.boson_encoding}"
        report["missing_label_family_samples"] = _selected_logical_record_samples(records_raw)
        return report, [], dict(pool_generator_registry)

    registry = dict(pool_generator_registry)
    registry_build_error: str | None = None
    if not registry:
        try:
            registry = build_pool_generator_registry(
                terms=pool,
                family_ids=pool_family_ids,
                num_sites=int(request.num_sites),
                ordering=str(request.ordering),
                qpb=int(max(1, qpb)),
                symmetry_specs=pool_symmetry_specs if pool_symmetry_specs else [None] * len(pool),
                split_policy=("deliberate_split" if bool(paop_split_paulis) else "preserve"),
                ai_log=ai_log,
            )
        except Exception as exc:
            registry_build_error = f"{type(exc).__name__}: {exc}"
            registry = {str(term.label): {"candidate_label": str(term.label)} for term in pool}
    if not registry:
        registry = {str(term.label): {"candidate_label": str(term.label)} for term in pool}
    report["registry_build_error"] = registry_build_error
    try:
        matcher = (
            match_selected_logical_operator_families
            if mode_key in SELECTED_LOGICAL_FAMILY_CLOSURE_MODES
            else match_selected_logical_generators
        )
        matches = matcher(
            selected_logical_library=selected_library,
            registry_by_label=registry,
            target_num_sites=int(request.num_sites),
            transfer_mode=str(transfer_mode_key),
        )
    except Exception as exc:
        report["reason"] = "match_error"
        report["fallback_detail"] = f"{type(exc).__name__}: {exc}"
        report["missing_label_family_samples"] = _selected_logical_record_samples(records_raw)
        return report, [], registry

    match_by_label = {str(row.get("registry_label") or row.get("candidate_label")): dict(row) for row in matches}
    if not match_by_label:
        report["reason"] = "no_matches"
        report["semantic_reason"] = "selected_logical_no_pool_matches"
        report["missing_label_family_samples"] = _selected_logical_record_samples(records_raw)
        return report, [], registry

    kept_labels: list[str] = []
    method_counts: dict[str, int] = {}
    for term in pool:
        label = str(term.label)
        match = match_by_label.get(label)
        if match is None:
            continue
        method = str(match.get("match_method", "unknown"))
        method_counts[method] = int(method_counts.get(method, 0)) + 1
        kept_labels.append(label)
    if not kept_labels:
        report["reason"] = "all_matches_filtered_empty"
        report["missing_label_family_samples"] = _selected_logical_record_samples(records_raw)
        return report, matches, registry

    operator_family_ids = sorted(
        {
            str(family)
            for row in matches
            for family in (
                row.get("selected_operator_family_ids")
                if isinstance(row.get("selected_operator_family_ids"), list)
                else ([row.get("operator_family_id")] if row.get("operator_family_id") not in {None, ""} else [])
            )
            if family not in {None, ""}
        }
    )
    report.update(
        {
            "applied": True,
            "pool_size_after": int(len(kept_labels)),
            "matched_count": int(len(kept_labels)),
            "match_method_counts": dict(sorted(method_counts.items())),
            "operator_family_ids": operator_family_ids,
            "kept_labels_sample": [str(x) for x in kept_labels[:24]],
            "missing_label_family_samples": _selected_logical_missing_record_samples(records_raw, matches),
            "match_rows_sample": [
                {
                    "registry_label": str(row.get("registry_label") or row.get("candidate_label")),
                    "candidate_label": str(row.get("candidate_label") or row.get("registry_label")),
                    "match_method": str(row.get("match_method", "unknown")),
                }
                for row in matches[:24]
            ],
            "status": "pass",
            "reason": None,
        }
    )
    return report, matches, registry


def build_selected_logical_pool_match_report(
    *,
    pool: list[AnsatzTerm],
    pool_family_ids: list[str],
    pool_symmetry_specs: list[dict[str, Any] | None],
    pool_generator_registry: dict[str, dict[str, Any]],
    request: Any,
    qpb: int,
    mode: str,
    source_json: Path | None,
    transfer_mode: str,
    paop_split_paulis: bool,
) -> dict[str, Any]:
    """Return the runtime selected-logical pool-match diagnostics without filtering."""

    report, _matches, _registry = _build_selected_logical_pool_match_report(
        pool=pool,
        pool_family_ids=pool_family_ids,
        pool_symmetry_specs=pool_symmetry_specs,
        pool_generator_registry=pool_generator_registry,
        request=request,
        qpb=int(qpb),
        mode=str(mode),
        source_json=source_json,
        transfer_mode=str(transfer_mode),
        paop_split_paulis=bool(paop_split_paulis),
    )
    return dict(report)


def _apply_selected_logical_pool_filter(
    *,
    pool: list[AnsatzTerm],
    pool_stage_family: list[str],
    pool_family_ids: list[str],
    pool_symmetry_specs: list[dict[str, Any] | None],
    pool_generator_registry: dict[str, dict[str, Any]],
    request: Any,
    qpb: int,
    mode: str,
    source_json: Path | None,
    transfer_mode: str,
    paop_split_paulis: bool,
    ai_log: Callable[..., None] | None = None,
) -> tuple[
    list[AnsatzTerm],
    list[str],
    list[str],
    list[dict[str, Any] | None],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    mode_key = str(mode or "off").strip().lower()
    transfer_mode_key = str(transfer_mode or "exact_match_v1").strip().lower()
    if mode_key == "off":
        return (
            pool,
            pool_stage_family,
            pool_family_ids,
            pool_symmetry_specs,
            pool_generator_registry,
            _selected_logical_filter_off_meta(
                mode=mode_key,
                source_json=source_json,
                transfer_mode=transfer_mode_key,
                pool_size=len(pool),
            ),
        )
    if mode_key not in SELECTED_LOGICAL_MODE_CHOICES - {"off"}:
        raise ValueError("Unsupported selected-logical pool filter mode.")
    fail_closed = mode_key in SELECTED_LOGICAL_FAIL_CLOSED_MODES

    def _fallback_or_raise(
        *,
        reason: str,
        detail: str | None = None,
        selected_record_count: int = 0,
    ) -> tuple[
        list[AnsatzTerm],
        list[str],
        list[str],
        list[dict[str, Any] | None],
        dict[str, dict[str, Any]],
        dict[str, Any],
    ]:
        if fail_closed:
            message = f"selected-logical pool filter failed closed: {reason}"
            if detail:
                message = f"{message}: {detail}"
            raise ValueError(message)
        return (
            pool,
            pool_stage_family,
            pool_family_ids,
            pool_symmetry_specs,
            pool_generator_registry,
            _fallback_selected_logical_filter_meta(
                mode=mode_key,
                source_json=(Path(source_json) if source_json is not None else None),
                transfer_mode=transfer_mode_key,
                pool_size=len(pool),
                reason=reason,
                detail=detail,
                selected_record_count=selected_record_count,
            ),
        )

    report, matches, registry = _build_selected_logical_pool_match_report(
        pool=pool,
        pool_family_ids=pool_family_ids,
        pool_symmetry_specs=pool_symmetry_specs,
        pool_generator_registry=pool_generator_registry,
        request=request,
        qpb=int(qpb),
        mode=mode_key,
        source_json=source_json,
        transfer_mode=transfer_mode_key,
        paop_split_paulis=bool(paop_split_paulis),
        ai_log=ai_log,
    )
    if report.get("status") != "pass":
        return _fallback_or_raise(
            reason=str(report.get("reason") or "unknown"),
            detail=(None if report.get("fallback_detail") in {None, ""} else str(report.get("fallback_detail"))),
            selected_record_count=int(report.get("selected_record_count") or 0),
        )

    match_by_label = {str(row.get("registry_label") or row.get("candidate_label")): dict(row) for row in matches}
    filtered_pool: list[AnsatzTerm] = []
    filtered_stage_family: list[str] = []
    filtered_family_ids: list[str] = []
    filtered_specs: list[dict[str, Any] | None] = []
    filtered_registry: dict[str, dict[str, Any]] = {}
    for idx, term in enumerate(pool):
        label = str(term.label)
        match = match_by_label.get(label)
        if match is None:
            continue
        filtered_pool.append(term)
        filtered_stage_family.append(str(pool_stage_family[idx] if idx < len(pool_stage_family) else "selected_logical"))
        filtered_family_ids.append(str(pool_family_ids[idx] if idx < len(pool_family_ids) else "selected_logical"))
        spec = pool_symmetry_specs[idx] if idx < len(pool_symmetry_specs) else None
        filtered_specs.append(dict(spec) if isinstance(spec, Mapping) else None)
        meta = registry.get(label)
        if isinstance(meta, Mapping):
            filtered_registry[label] = dict(meta)

    if not filtered_pool:
        return _fallback_or_raise(
            reason="all_matches_filtered_empty",
            selected_record_count=int(report.get("selected_record_count") or 0),
        )

    meta = {
        "schema": "adapt_selected_logical_pool_filter_v1",
        "mode": str(mode_key),
        "source_json": str(Path(source_json)) if source_json is not None else None,
        "source_kind": str(report.get("source_kind", "unknown")),
        "source_tag": str(report.get("source_tag", "payload")),
        "transfer_mode": str(transfer_mode_key),
        "applied": True,
        "fallback_to_full_pool": False,
        "fallback_reason": None,
        "pool_size_before": int(len(pool)),
        "pool_size_after": int(len(filtered_pool)),
        "active_pool_label_count": int(report.get("active_pool_label_count") or len(pool)),
        "selected_record_count": int(report.get("selected_record_count") or 0),
        "matched_count": int(len(filtered_pool)),
        "match_method_counts": dict(report.get("match_method_counts") or {}),
        "closure_mode": str(report.get("closure_mode") or ("operator_family_v1" if mode_key in SELECTED_LOGICAL_FAMILY_CLOSURE_MODES else "exact_logical_v1")),
        "operator_family_ids": list(report.get("operator_family_ids") or []),
        "kept_labels_sample": [str(x) for x in list(report.get("kept_labels_sample") or [])[:24]],
        "selected_label_family_samples": list(report.get("selected_label_family_samples") or []),
        "missing_label_family_samples": list(report.get("missing_label_family_samples") or []),
        "registry_build_error": report.get("registry_build_error"),
    }
    if callable(ai_log):
        ai_log(
            "adapt_selected_logical_pool_filtered",
            source_json=str(Path(source_json)),
            pool_size_before=int(len(pool)),
            pool_size_after=int(len(filtered_pool)),
            match_method_counts=dict(report.get("match_method_counts") or {}),
        )
    return (
        filtered_pool,
        filtered_stage_family,
        filtered_family_ids,
        filtered_specs,
        filtered_registry,
        meta,
    )


def resolve_pool_plan(
    *,
    resolved_problem: ResolvedProblemContext,
    continuation_mode: str,
    adapt_pool: str | None,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    phase3_symmetry_mitigation_mode: str,
    filter_resolution: PoolFilterResolution | None = None,
    ai_log: Callable[..., None] | None = None,
) -> PoolResolution:
    request = resolved_problem.request
    problem_key = str(resolved_problem.family_key)
    pool_key_input = (
        filter_resolution.pool_key_input
        if filter_resolution is not None
        else (None if adapt_pool is None else str(adapt_pool).strip().lower())
    )
    full_meta_class_filter_spec = (
        filter_resolution.full_meta_class_filter_spec
        if filter_resolution is not None
        else None
    )
    full_meta_label_filter_spec = (
        filter_resolution.full_meta_label_filter_spec
        if filter_resolution is not None
        else None
    )
    selected_logical_source_json = (
        filter_resolution.selected_logical_source_json
        if filter_resolution is not None
        else None
    )
    selected_logical_mode = (
        filter_resolution.selected_logical_mode
        if filter_resolution is not None
        else "off"
    )
    selected_logical_transfer_mode = (
        filter_resolution.selected_logical_transfer_mode
        if filter_resolution is not None
        else "exact_match_v1"
    )
    num_particles = (
        tuple(resolved_problem.sector.num_particles)
        if resolved_problem.sector.num_particles is not None
        else tuple(half_filled_num_particles(int(request.num_sites)))
    )
    runtime_data = (
        dict(resolved_problem.runtime_data)
        if isinstance(resolved_problem.runtime_data, Mapping)
        else {}
    )
    molecular_problem = runtime_data.get("molecular_problem")

    full_meta_class_filter_meta: dict[str, Any] | None = None
    full_meta_label_filter_meta: dict[str, Any] | None = None
    pool_legal_subspace_filter_meta: dict[str, Any] | None = None
    selected_logical_filter_meta: dict[str, Any] | None = None
    pool_stage_family: list[str] = []
    pool_family_ids: list[str] = []
    phase1_core_limit = 0
    phase1_residual_indices: set[int] = set()
    phase1_depth0_full_meta_override = False
    runtime_default_pool_key = resolve_runtime_default_pool_key(
        resolved_problem,
        continuation_mode=str(continuation_mode),
    )

    if continuation_mode in _HH_STAGED_CONTINUATION_MODES and problem_key == "hh":
        if pool_key_input in (HH_FULL_META_FILTER_POOL_KEYS | {"pareto_lean", "pareto_lean_l2", "pareto_lean_gate_pruned"}):
            phase1_depth0_full_meta_override = True
            pool, _pool_method, full_meta_class_filter_meta, full_meta_label_filter_meta, pool_legal_subspace_filter_meta = build_hh_pool_by_key(
                pool_key_hh=str(pool_key_input),
                h_poly=resolved_problem.hamiltonian,
                num_sites=int(request.num_sites),
                t=float(request.t),
                u=float(request.u),
                omega0=float(request.omega0),
                g_ep=float(request.g_ep),
                dv=float(request.dv),
                n_ph_max=int(request.n_ph_max),
                boson_encoding=str(request.boson_encoding),
                ordering=str(request.ordering),
                boundary=str(request.boundary),
                paop_r=int(paop_r),
                paop_split_paulis=bool(paop_split_paulis),
                paop_prune_eps=float(paop_prune_eps),
                paop_normalization=str(paop_normalization),
                num_particles=num_particles,
                full_meta_class_filter_spec=full_meta_class_filter_spec,
                full_meta_label_filter_spec=full_meta_label_filter_spec,
                ai_log=ai_log,
                include_legal_subspace_filter_meta=True,
            )
            phase1_core_limit = int(len(pool))
            phase1_residual_indices = set()
            pool_stage_family = ["core"] * int(len(pool))
            pool_family_ids = [str(pool_key_input)] * int(len(pool))
            if callable(ai_log):
                ai_log(
                    "hardcoded_adapt_phase1_depth0_full_meta_override",
                    continuation_mode=str(continuation_mode),
                    pool_key=str(pool_key_input),
                    pool_size=int(len(pool)),
                )
        else:
            core_key = str(pool_key_input if pool_key_input is not None else runtime_default_pool_key)
            core_pool, _core_method, _core_class_filter_meta, _core_label_filter_meta, _core_legal_filter_meta = build_hh_pool_by_key(
                pool_key_hh=core_key,
                h_poly=resolved_problem.hamiltonian,
                num_sites=int(request.num_sites),
                t=float(request.t),
                u=float(request.u),
                omega0=float(request.omega0),
                g_ep=float(request.g_ep),
                dv=float(request.dv),
                n_ph_max=int(request.n_ph_max),
                boson_encoding=str(request.boson_encoding),
                ordering=str(request.ordering),
                boundary=str(request.boundary),
                paop_r=int(paop_r),
                paop_split_paulis=bool(paop_split_paulis),
                paop_prune_eps=float(paop_prune_eps),
                paop_normalization=str(paop_normalization),
                num_particles=num_particles,
                full_meta_class_filter_spec=full_meta_class_filter_spec,
                full_meta_label_filter_spec=full_meta_label_filter_spec,
                ai_log=ai_log,
                include_legal_subspace_filter_meta=True,
            )
            residual_pool, _residual_method, full_meta_class_filter_meta, full_meta_label_filter_meta, pool_legal_subspace_filter_meta = build_hh_pool_by_key(
                pool_key_hh="full_meta",
                h_poly=resolved_problem.hamiltonian,
                num_sites=int(request.num_sites),
                t=float(request.t),
                u=float(request.u),
                omega0=float(request.omega0),
                g_ep=float(request.g_ep),
                dv=float(request.dv),
                n_ph_max=int(request.n_ph_max),
                boson_encoding=str(request.boson_encoding),
                ordering=str(request.ordering),
                boundary=str(request.boundary),
                paop_r=int(paop_r),
                paop_split_paulis=bool(paop_split_paulis),
                paop_prune_eps=float(paop_prune_eps),
                paop_normalization=str(paop_normalization),
                num_particles=num_particles,
                full_meta_class_filter_spec=full_meta_class_filter_spec,
                full_meta_label_filter_spec=full_meta_label_filter_spec,
                ai_log=ai_log,
                include_legal_subspace_filter_meta=True,
            )
            seen_sig = {_polynomial_signature(op.polynomial) for op in core_pool}
            residual_unique: list[AnsatzTerm] = []
            for op in residual_pool:
                sig = _polynomial_signature(op.polynomial)
                if sig in seen_sig:
                    continue
                seen_sig.add(sig)
                residual_unique.append(op)
            pool = list(core_pool) + list(residual_unique)
            phase1_core_limit = int(len(core_pool))
            phase1_residual_indices = set(range(int(phase1_core_limit), int(len(pool))))
            pool_stage_family = (["core"] * int(phase1_core_limit)) + (["residual"] * int(len(residual_unique)))
            pool_family_ids = ([str(core_key)] * int(phase1_core_limit)) + (["full_meta"] * int(len(residual_unique)))
        method_name = f"hardcoded_adapt_vqe_{str(continuation_mode)}_hh"
        pool_key = str(continuation_mode)
    else:
        pool_key = str(pool_key_input if pool_key_input is not None else runtime_default_pool_key)
        if problem_key == "hh":
            pool, method_name, full_meta_class_filter_meta, full_meta_label_filter_meta, pool_legal_subspace_filter_meta = build_hh_pool_by_key(
                pool_key_hh=pool_key,
                h_poly=resolved_problem.hamiltonian,
                num_sites=int(request.num_sites),
                t=float(request.t),
                u=float(request.u),
                omega0=float(request.omega0),
                g_ep=float(request.g_ep),
                dv=float(request.dv),
                n_ph_max=int(request.n_ph_max),
                boson_encoding=str(request.boson_encoding),
                ordering=str(request.ordering),
                boundary=str(request.boundary),
                paop_r=int(paop_r),
                paop_split_paulis=bool(paop_split_paulis),
                paop_prune_eps=float(paop_prune_eps),
                paop_normalization=str(paop_normalization),
                num_particles=num_particles,
                full_meta_class_filter_spec=full_meta_class_filter_spec,
                full_meta_label_filter_spec=full_meta_label_filter_spec,
                ai_log=ai_log,
                include_legal_subspace_filter_meta=True,
            )
        else:
            if problem_key == "molecular_restricted_closed_shell":
                if pool_key == "uccsd":
                    pool = _build_molecular_uccsd_pool(
                        int(request.num_sites),
                        num_particles,
                        str(request.ordering),
                    )
                    method_name = "hardcoded_adapt_vqe_uccsd_molecular"
                elif pool_key == "full_hamiltonian":
                    pool = _build_full_hamiltonian_pool(resolved_problem.hamiltonian)
                    method_name = "hardcoded_adapt_vqe_full_hamiltonian"
                elif pool_key == "hamiltonian_blocks":
                    pool = _build_hamiltonian_blocks_pool(
                        problem_key=str(problem_key),
                        num_sites=int(request.num_sites),
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        v_nn=float(request.v_nn),
                        t_prime=float(request.t_prime),
                        omega0=float(request.omega0),
                        g_ep=float(request.g_ep),
                        n_ph_max=int(request.n_ph_max),
                        boson_encoding=str(request.boson_encoding),
                        ordering=str(request.ordering),
                        boundary=str(request.boundary),
                        include_zero_point=bool(request.include_zero_point),
                        molecular_problem=molecular_problem,
                    )
                    method_name = "hardcoded_adapt_vqe_hamiltonian_blocks_molecular"
                elif pool_key == "hva":
                    pool = _build_family_hva_pool(
                        problem_key=str(problem_key),
                        num_sites=int(request.num_sites),
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        v_nn=float(request.v_nn),
                        t_prime=float(request.t_prime),
                        omega0=float(request.omega0),
                        g_ep=float(request.g_ep),
                        n_ph_max=int(request.n_ph_max),
                        boson_encoding=str(request.boson_encoding),
                        ordering=str(request.ordering),
                        boundary=str(request.boundary),
                        include_zero_point=bool(request.include_zero_point),
                        molecular_problem=molecular_problem,
                    )
                    method_name = "hardcoded_adapt_vqe_hva_molecular"
                elif pool_key == "family_max":
                    pool = _build_family_max_pool(
                        problem_key=str(problem_key),
                        num_sites=int(request.num_sites),
                        num_particles=num_particles,
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        v_nn=float(request.v_nn),
                        t_prime=float(request.t_prime),
                        omega0=float(request.omega0),
                        g_ep=float(request.g_ep),
                        n_ph_max=int(request.n_ph_max),
                        boson_encoding=str(request.boson_encoding),
                        ordering=str(request.ordering),
                        boundary=str(request.boundary),
                        include_zero_point=bool(request.include_zero_point),
                        molecular_problem=molecular_problem,
                    )
                    method_name = "hardcoded_adapt_vqe_family_max_molecular"
                elif pool_key == "full_meta":
                    pool, pool_legal_subspace_filter_meta = _build_full_meta_pool(
                        problem_key=str(problem_key),
                        h_poly=resolved_problem.hamiltonian,
                        num_sites=int(request.num_sites),
                        num_particles=num_particles,
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        v_nn=float(request.v_nn),
                        t_prime=float(request.t_prime),
                        omega0=float(request.omega0),
                        g_ep=float(request.g_ep),
                        n_ph_max=int(request.n_ph_max),
                        boson_encoding=str(request.boson_encoding),
                        ordering=str(request.ordering),
                        boundary=str(request.boundary),
                        include_zero_point=bool(request.include_zero_point),
                        molecular_problem=molecular_problem,
                        return_legal_subspace_filter_meta=True,
                    )
                    method_name = "hardcoded_adapt_vqe_full_meta_molecular"
                else:
                    raise ValueError(
                        "For problem='molecular_restricted_closed_shell', valid pools are "
                        "uccsd, full_hamiltonian, hamiltonian_blocks, hva, family_max, and full_meta."
                    )
            elif problem_key == "molecular_vibronic_h2":
                vibronic_model = runtime_data.get("vibronic_h2_model")
                if pool_key == "full_meta":
                    if vibronic_model is None or not hasattr(vibronic_model, "pool"):
                        raise ValueError("molecular_vibronic_h2 full_meta pool requires runtime_data['vibronic_h2_model'].")
                    pool = list(getattr(vibronic_model, "pool"))
                    method_name = "hardcoded_adapt_vqe_full_meta_molecular_vibronic_h2"
                elif pool_key == "full_hamiltonian":
                    pool = _build_full_hamiltonian_pool(resolved_problem.hamiltonian)
                    method_name = "hardcoded_adapt_vqe_full_hamiltonian_molecular_vibronic_h2"
                else:
                    raise ValueError(
                        "For problem='molecular_vibronic_h2', valid pools are full_meta and full_hamiltonian."
                    )
            elif problem_key == "molecular_vibronic_h2o":
                vibronic_model = runtime_data.get("vibronic_h2o_model")
                if pool_key == "full_meta":
                    if vibronic_model is None or not hasattr(vibronic_model, "pool"):
                        raise ValueError("molecular_vibronic_h2o full_meta pool requires runtime_data['vibronic_h2o_model'].")
                    pool = list(getattr(vibronic_model, "pool"))
                    method_name = "hardcoded_adapt_vqe_full_meta_molecular_vibronic_h2o"
                elif pool_key == "full_hamiltonian":
                    pool = _build_full_hamiltonian_pool(resolved_problem.hamiltonian)
                    method_name = "hardcoded_adapt_vqe_full_hamiltonian_molecular_vibronic_h2o"
                else:
                    raise ValueError(
                        "For problem='molecular_vibronic_h2o', valid pools are full_meta and full_hamiltonian."
                    )
            elif problem_key == "molecular_vibronic_h2o_linear_fd":
                vibronic_model = runtime_data.get("vibronic_h2o_linear_fd_model")
                vibronic_fixture = runtime_data.get("vibronic_h2o_linear_fd_fixture")
                if pool_key == "full_meta":
                    if vibronic_model is None or not hasattr(vibronic_model, "pool"):
                        raise ValueError(
                            "molecular_vibronic_h2o_linear_fd full_meta pool requires "
                            "runtime_data['vibronic_h2o_linear_fd_model']."
                        )
                    pool = list(getattr(vibronic_model, "pool"))
                    method_name = "hardcoded_adapt_vqe_full_meta_molecular_vibronic_h2o_linear_fd"
                elif pool_key == H2O_LINEAR_FD_DERIVATIVE_RESOLVED_POOL_KEY:
                    if vibronic_fixture is None:
                        raise ValueError(
                            "molecular_vibronic_h2o_linear_fd derivative-resolved "
                            "pool requires runtime_data['vibronic_h2o_linear_fd_fixture']."
                        )
                    pool = list(
                        build_h2o_linear_fd_derivative_resolved_pool_v2(
                            vibronic_fixture
                        )
                    )
                    method_name = (
                        "hardcoded_adapt_vqe_"
                        "full_meta_derivative_resolved_v2_"
                        "molecular_vibronic_h2o_linear_fd"
                    )
                elif pool_key == "full_hamiltonian":
                    pool = _build_full_hamiltonian_pool(resolved_problem.hamiltonian)
                    method_name = "hardcoded_adapt_vqe_full_hamiltonian_molecular_vibronic_h2o_linear_fd"
                else:
                    raise ValueError(
                        "For problem='molecular_vibronic_h2o_linear_fd', valid "
                        "pools are full_meta, full_meta_derivative_resolved_v2, "
                        "and full_hamiltonian."
                    )
            elif problem_key == "spinless_tv":
                if pool_key == "full_hamiltonian":
                    pool = _build_full_hamiltonian_pool(resolved_problem.hamiltonian)
                    method_name = "hardcoded_adapt_vqe_full_hamiltonian"
                elif pool_key == "hamiltonian_quadratures":
                    pool = _build_hamiltonian_quadratures_pool(
                        problem_key=str(problem_key),
                        num_sites=int(request.num_sites),
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        v_nn=float(request.v_nn),
                        t_prime=float(request.t_prime),
                        omega0=float(request.omega0),
                        g_ep=float(request.g_ep),
                        n_ph_max=int(request.n_ph_max),
                        boson_encoding=str(request.boson_encoding),
                        ordering=str(request.ordering),
                        boundary=str(request.boundary),
                        include_zero_point=bool(request.include_zero_point),
                    )
                    method_name = "hardcoded_adapt_vqe_hamiltonian_quadratures_spinless"
                elif pool_key == "hamiltonian_blocks":
                    pool = _build_hamiltonian_blocks_pool(
                        problem_key=str(problem_key),
                        num_sites=int(request.num_sites),
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        v_nn=float(request.v_nn),
                        t_prime=float(request.t_prime),
                        omega0=float(request.omega0),
                        g_ep=float(request.g_ep),
                        n_ph_max=int(request.n_ph_max),
                        boson_encoding=str(request.boson_encoding),
                        ordering=str(request.ordering),
                        boundary=str(request.boundary),
                        include_zero_point=bool(request.include_zero_point),
                    )
                    method_name = "hardcoded_adapt_vqe_hamiltonian_blocks_spinless"
                elif pool_key == "hva":
                    pool = _build_family_hva_pool(
                        problem_key=str(problem_key),
                        num_sites=int(request.num_sites),
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        v_nn=float(request.v_nn),
                        t_prime=float(request.t_prime),
                        omega0=float(request.omega0),
                        g_ep=float(request.g_ep),
                        n_ph_max=int(request.n_ph_max),
                        boson_encoding=str(request.boson_encoding),
                        ordering=str(request.ordering),
                        boundary=str(request.boundary),
                        include_zero_point=bool(request.include_zero_point),
                    )
                    method_name = "hardcoded_adapt_vqe_hva_spinless"
                elif pool_key == "family_max":
                    pool = _build_family_max_pool(
                        problem_key=str(problem_key),
                        num_sites=int(request.num_sites),
                        num_particles=num_particles,
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        v_nn=float(request.v_nn),
                        t_prime=float(request.t_prime),
                        omega0=float(request.omega0),
                        g_ep=float(request.g_ep),
                        n_ph_max=int(request.n_ph_max),
                        boson_encoding=str(request.boson_encoding),
                        ordering=str(request.ordering),
                        boundary=str(request.boundary),
                        include_zero_point=bool(request.include_zero_point),
                    )
                    method_name = "hardcoded_adapt_vqe_family_max_spinless"
                elif pool_key == "full_meta":
                    pool, pool_legal_subspace_filter_meta = _build_full_meta_pool(
                        problem_key=str(problem_key),
                        h_poly=resolved_problem.hamiltonian,
                        num_sites=int(request.num_sites),
                        num_particles=num_particles,
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        v_nn=float(request.v_nn),
                        t_prime=float(request.t_prime),
                        omega0=float(request.omega0),
                        g_ep=float(request.g_ep),
                        n_ph_max=int(request.n_ph_max),
                        boson_encoding=str(request.boson_encoding),
                        ordering=str(request.ordering),
                        boundary=str(request.boundary),
                        include_zero_point=bool(request.include_zero_point),
                        return_legal_subspace_filter_meta=True,
                    )
                    method_name = "hardcoded_adapt_vqe_full_meta_spinless"
                else:
                    raise ValueError(
                        "For problem='spinless_tv', valid pools are full_hamiltonian, "
                        "hamiltonian_quadratures, hamiltonian_blocks, hva, family_max, and full_meta."
                    )
            else:
                if pool_key == "uccsd":
                    pool = _build_uccsd_pool(int(request.num_sites), num_particles, str(request.ordering))
                    method_name = "hardcoded_adapt_vqe_uccsd"
                elif pool_key == "uccsd_qeb":
                    if problem_key != "hubbard":
                        raise ValueError("Pool 'uccsd_qeb' is currently only valid for problem='hubbard'.")
                    pool = _build_hubbard_uccsd_qeb_pool(
                        int(request.num_sites),
                        num_particles,
                        str(request.ordering),
                    )
                    method_name = "hardcoded_adapt_vqe_uccsd_qeb_hubbard"
                elif pool_key == "uccsd_qeb_hva_blocks":
                    if problem_key != "hubbard":
                        raise ValueError("Pool 'uccsd_qeb_hva_blocks' is currently only valid for problem='hubbard'.")
                    pool = _build_hubbard_uccsd_qeb_hva_blocks_pool(
                        num_sites=int(request.num_sites),
                        num_particles=num_particles,
                        ordering=str(request.ordering),
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        boundary=str(request.boundary),
                    )
                    method_name = "hardcoded_adapt_vqe_uccsd_qeb_hva_blocks_hubbard"
                elif pool_key == "cse":
                    if problem_key != "hubbard":
                        raise ValueError("Pool 'cse' is currently only valid for problem='hubbard'.")
                    pool = _build_cse_pool(
                        int(request.num_sites),
                        str(request.ordering),
                        float(request.t),
                        float(request.u),
                        float(request.dv),
                        str(request.boundary),
                    )
                    method_name = "hardcoded_adapt_vqe_cse"
                elif pool_key == "full_hamiltonian":
                    pool = _build_full_hamiltonian_pool(resolved_problem.hamiltonian)
                    method_name = "hardcoded_adapt_vqe_full_hamiltonian"
                elif pool_key == "hamiltonian_quadratures":
                    pool = _build_hamiltonian_quadratures_pool(
                        problem_key=str(problem_key),
                        num_sites=int(request.num_sites),
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        v_nn=float(request.v_nn),
                        t_prime=float(request.t_prime),
                        omega0=float(request.omega0),
                        g_ep=float(request.g_ep),
                        n_ph_max=int(request.n_ph_max),
                        boson_encoding=str(request.boson_encoding),
                        ordering=str(request.ordering),
                        boundary=str(request.boundary),
                        include_zero_point=bool(request.include_zero_point),
                    )
                    method_name = "hardcoded_adapt_vqe_hamiltonian_quadratures"
                elif pool_key == "hamiltonian_blocks":
                    pool = _build_hamiltonian_blocks_pool(
                        problem_key=str(problem_key),
                        num_sites=int(request.num_sites),
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        v_nn=float(request.v_nn),
                        t_prime=float(request.t_prime),
                        omega0=float(request.omega0),
                        g_ep=float(request.g_ep),
                        n_ph_max=int(request.n_ph_max),
                        boson_encoding=str(request.boson_encoding),
                        ordering=str(request.ordering),
                        boundary=str(request.boundary),
                        include_zero_point=bool(request.include_zero_point),
                    )
                    method_name = "hardcoded_adapt_vqe_hamiltonian_blocks"
                elif pool_key == "hva":
                    if problem_key == "hubbard":
                        raise ValueError(
                            "For problem='hubbard', pool='hva' is not valid. "
                            "Use uccsd, cse, full_hamiltonian, or hamiltonian_blocks."
                        )
                    pool = _build_family_hva_pool(
                        problem_key=str(problem_key),
                        num_sites=int(request.num_sites),
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        v_nn=float(request.v_nn),
                        t_prime=float(request.t_prime),
                        omega0=float(request.omega0),
                        g_ep=float(request.g_ep),
                        n_ph_max=int(request.n_ph_max),
                        boson_encoding=str(request.boson_encoding),
                        ordering=str(request.ordering),
                        boundary=str(request.boundary),
                        include_zero_point=bool(request.include_zero_point),
                    )
                    method_name = "hardcoded_adapt_vqe_hva_family"
                elif pool_key == "family_max":
                    if problem_key == "hubbard":
                        raise ValueError(
                            "For problem='hubbard', pool='family_max' is not valid. "
                            "Use uccsd, cse, full_hamiltonian, or hamiltonian_blocks."
                        )
                    pool = _build_family_max_pool(
                        problem_key=str(problem_key),
                        num_sites=int(request.num_sites),
                        num_particles=num_particles,
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        v_nn=float(request.v_nn),
                        t_prime=float(request.t_prime),
                        omega0=float(request.omega0),
                        g_ep=float(request.g_ep),
                        n_ph_max=int(request.n_ph_max),
                        boson_encoding=str(request.boson_encoding),
                        ordering=str(request.ordering),
                        boundary=str(request.boundary),
                        include_zero_point=bool(request.include_zero_point),
                    )
                    method_name = "hardcoded_adapt_vqe_family_max"
                elif pool_key == "full_meta":
                    pool, pool_legal_subspace_filter_meta = _build_full_meta_pool(
                        problem_key=str(problem_key),
                        h_poly=resolved_problem.hamiltonian,
                        num_sites=int(request.num_sites),
                        num_particles=num_particles,
                        t=float(request.t),
                        u=float(request.u),
                        dv=float(request.dv),
                        v_nn=float(request.v_nn),
                        t_prime=float(request.t_prime),
                        omega0=float(request.omega0),
                        g_ep=float(request.g_ep),
                        n_ph_max=int(request.n_ph_max),
                        boson_encoding=str(request.boson_encoding),
                        ordering=str(request.ordering),
                        boundary=str(request.boundary),
                        include_zero_point=bool(request.include_zero_point),
                        molecular_problem=molecular_problem,
                        return_legal_subspace_filter_meta=True,
                    )
                    method_name = "hardcoded_adapt_vqe_full_meta"
                elif pool_key == "uccsd_paop_lf_full":
                    raise ValueError("Pool 'uccsd_paop_lf_full' is only valid for problem='hh'.")
                else:
                    raise ValueError(f"Unsupported adapt pool '{adapt_pool}'.")
        pool_stage_family = [str(pool_key)] * int(len(pool))
        if problem_key == "hubbard" and pool_key == "uccsd_qeb":
            pool_family_ids = [_hubbard_uccsd_qeb_family_id_for_label(str(term.label)) for term in pool]
        elif problem_key == "hubbard" and pool_key == "uccsd_qeb_hva_blocks":
            pool_family_ids = [_hubbard_uccsd_qeb_hva_family_id_for_label(str(term.label)) for term in pool]
        else:
            pool_family_ids = [str(pool_key)] * int(len(pool))

    qpb = (
        int(boson_qubits_per_site(int(request.n_ph_max), str(request.boson_encoding)))
        if int(getattr(resolved_problem.layout, "boson_qubits", 0)) > 0
        else 1
    )
    pool_symmetry_specs: list[dict[str, Any] | None] = [None] * int(len(pool))
    pool_generator_registry: dict[str, dict[str, Any]] = {}
    if problem_key == "hh" and len(pool) > 0:
        base_pool_symmetry_specs = [
            dict(
                build_symmetry_spec(
                    family_id=str(pool_family_ids[idx] if idx < len(pool_family_ids) else "unknown"),
                    mitigation_mode=str(phase3_symmetry_mitigation_mode),
                ).__dict__
            )
            for idx in range(len(pool))
        ]
        raw_pool_generator_registry = build_pool_generator_registry(
            terms=pool,
            family_ids=pool_family_ids,
            num_sites=int(request.num_sites),
            ordering=str(request.ordering),
            qpb=int(max(1, qpb)),
            symmetry_specs=base_pool_symmetry_specs,
            split_policy=("deliberate_split" if bool(paop_split_paulis) else "preserve"),
            ai_log=ai_log,
        )
        filtered_pool: list[AnsatzTerm] = []
        filtered_stage_family: list[str] = []
        filtered_family_ids: list[str] = []
        filtered_specs: list[dict[str, Any] | None] = []
        filtered_registry: dict[str, dict[str, Any]] = {}
        removed_labels: list[str] = []
        removed_family_ids: list[str] = []
        for idx, term in enumerate(pool):
            label = str(term.label)
            meta = raw_pool_generator_registry.get(label)
            spec = (
                meta.get("symmetry_spec")
                if isinstance(meta, Mapping)
                else base_pool_symmetry_specs[idx]
            )
            if isinstance(spec, Mapping) and bool(spec.get("hard_guard", False)):
                removed_labels.append(label)
                removed_family_ids.append(str(pool_family_ids[idx] if idx < len(pool_family_ids) else "unknown"))
                continue
            filtered_pool.append(term)
            filtered_stage_family.append(str(pool_stage_family[idx] if idx < len(pool_stage_family) else pool_key))
            filtered_family_ids.append(str(pool_family_ids[idx] if idx < len(pool_family_ids) else pool_key))
            filtered_specs.append(dict(spec) if isinstance(spec, Mapping) else None)
            if isinstance(meta, Mapping):
                filtered_registry[label] = dict(meta)
        if removed_labels and callable(ai_log):
            ai_log(
                "hardcoded_adapt_hh_pool_symmetry_filtered",
                removed_count=int(len(removed_labels)),
                kept_count=int(len(filtered_pool)),
                removed_labels_sample=[str(x) for x in removed_labels[:12]],
                removed_families_sample=[str(x) for x in removed_family_ids[:12]],
            )
        pool = filtered_pool
        pool_stage_family = filtered_stage_family
        pool_family_ids = filtered_family_ids
        pool_symmetry_specs = filtered_specs
        pool_generator_registry = filtered_registry
        if continuation_mode in _HH_STAGED_CONTINUATION_MODES:
            phase1_core_limit = int(sum(1 for stage in pool_stage_family if str(stage) == "core"))
            phase1_residual_indices = {
                int(idx) for idx, stage in enumerate(pool_stage_family) if str(stage) == "residual"
            }

    if continuation_mode in _HH_STAGED_CONTINUATION_MODES and problem_key != "hh":
        phase1_core_limit = int(len(pool))
        phase1_residual_indices = set()

    (
        pool,
        pool_stage_family,
        pool_family_ids,
        pool_symmetry_specs,
        pool_generator_registry,
        selected_logical_filter_meta,
    ) = _apply_selected_logical_pool_filter(
        pool=pool,
        pool_stage_family=pool_stage_family,
        pool_family_ids=pool_family_ids,
        pool_symmetry_specs=pool_symmetry_specs,
        pool_generator_registry=pool_generator_registry,
        request=request,
        qpb=int(qpb),
        mode=str(selected_logical_mode),
        source_json=selected_logical_source_json,
        transfer_mode=str(selected_logical_transfer_mode),
        paop_split_paulis=bool(paop_split_paulis),
        ai_log=ai_log,
    )
    if continuation_mode in _HH_STAGED_CONTINUATION_MODES:
        if problem_key == "hh":
            phase1_core_limit = int(sum(1 for stage in pool_stage_family if str(stage) == "core"))
            phase1_residual_indices = {
                int(idx) for idx, stage in enumerate(pool_stage_family) if str(stage) == "residual"
            }
        else:
            phase1_core_limit = int(len(pool))
            phase1_residual_indices = set()

    if len(pool) == 0:
        raise ValueError(f"ADAPT pool '{pool_key}' produced no operators for problem='{problem_key}'.")

    return PoolResolution(
        pool=pool,
        pool_key=str(pool_key),
        method_name=str(method_name),
        pool_stage_family=list(pool_stage_family),
        pool_family_ids=list(pool_family_ids),
        phase1_core_limit=int(phase1_core_limit),
        phase1_residual_indices=set(phase1_residual_indices),
        phase1_depth0_full_meta_override=bool(phase1_depth0_full_meta_override),
        full_meta_class_filter_spec=full_meta_class_filter_spec,
        full_meta_label_filter_spec=full_meta_label_filter_spec,
        full_meta_class_filter_meta=full_meta_class_filter_meta,
        full_meta_label_filter_meta=full_meta_label_filter_meta,
        pool_legal_subspace_filter_meta=(
            dict(pool_legal_subspace_filter_meta)
            if isinstance(pool_legal_subspace_filter_meta, Mapping)
            else None
        ),
        selected_logical_filter_meta=(
            dict(selected_logical_filter_meta)
            if isinstance(selected_logical_filter_meta, Mapping)
            else None
        ),
        pool_symmetry_specs=list(pool_symmetry_specs),
        pool_generator_registry=dict(pool_generator_registry),
        qpb=int(qpb),
    )


__all__ = [
    "PoolFilterResolution",
    "PoolResolution",
    "resolve_pool_plan",
    "resolve_requested_pool_filters",
]
