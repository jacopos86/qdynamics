from __future__ import annotations

from typing import Mapping

_METHOD_LABELS: dict[str, str] = {
    "suzuki2": "Suzuki-2",
    "magnus2": "Magnus-2",
    "piecewise_exact": "Piecewise exact",
    "exact": "Exact reference",
    "ideal": "Ideal estimator",
    "shots": "Shot-based estimator",
    "aer_noise": "Aer noise",
}

_STAGE_LABELS: dict[str, str] = {
    "warm": "Warm start",
    "warm_start": "Warm start",
    "adapt": "ADAPT",
    "adapt_pool_b": "ADAPT Pool-B",
    "final": "Final replay",
    "conventional_vqe": "Final conventional VQE",
}

_BRANCH_LABELS: dict[str, str] = {
    "selected": "Selected branch",
    "paop": "PAOP seed",
    "hva": "HVA replay",
    "exact_gs_filtered": "Exact filtered ground state",
    "exact_paop": "Exact PAOP seed",
    "trotter_paop": "Trotter PAOP seed",
    "exact_hva": "Exact HVA replay",
    "trotter_hva": "Trotter HVA replay",
}

_METRIC_LABELS: dict[str, str] = {
    "final_delta_abs": "Final |ΔE|",
    "max_abs_delta": "Max |Δ(noisy-ideal)|",
    "max_abs_delta_over_stderr": "Max |Δ| / stderr",
    "mean_abs_delta_over_stderr": "Mean |Δ| / stderr",
    "noisy_audit_max_abs_delta": "Audit max |Δ(noisy-ideal)|",
    "noisy_audit_max_abs_delta_over_stderr": "Audit max |Δ| / stderr",
    "noisy_audit_mean_abs_delta_over_stderr": "Audit mean |Δ| / stderr",
    "noisy_modes_completed": "Completed noisy modes",
    "noisy_method_modes_completed": "Completed noisy method-modes",
    "noisy_audit_modes_completed": "Completed noisy audit modes",
    "dynamics_benchmark_rows": "Benchmark rows",
}

_TOKEN_REWRITES: dict[str, str] = {
    "vqe": "VQE",
    "qpe": "QPE",
    "paop": "PAOP",
    "hh": "HH",
    "jw": "JW",
    "gs": "GS",
    "dn": "DN",
    "up": "UP",
}


def _normalized(raw: str) -> str:
    return str(raw).strip().lower()


"""display(raw) = shared_mapping(raw) or readable_fallback(raw)"""
def report_display_label(
    raw: str,
    *,
    kind: str | None = None,
    overrides: Mapping[str, str] | None = None,
) -> str:
    key = _normalized(raw)
    if overrides and key in overrides:
        return str(overrides[key])

    mapping = _mapping_for_kind(kind)
    if key in mapping:
        return mapping[key]

    for shared in (_METHOD_LABELS, _STAGE_LABELS, _BRANCH_LABELS, _METRIC_LABELS):
        if key in shared:
            return shared[key]

    return _fallback_label(raw)


"""method_label(method) = display(method | kind='method')"""
def report_method_label(raw: str) -> str:
    return report_display_label(raw, kind="method")


"""stage_label(stage) = display(stage | kind='stage')"""
def report_stage_label(raw: str) -> str:
    return report_display_label(raw, kind="stage")


"""branch_label(branch) = display(branch | kind='branch')"""
def report_branch_label(raw: str) -> str:
    return report_display_label(raw, kind="branch")


"""metric_label(metric) = display(metric | kind='metric')"""
def report_metric_label(raw: str) -> str:
    return report_display_label(raw, kind="metric")


def _mapping_for_kind(kind: str | None) -> dict[str, str]:
    normalized = _normalized(kind or "")
    if normalized == "method":
        return _METHOD_LABELS
    if normalized == "stage":
        return _STAGE_LABELS
    if normalized == "branch":
        return _BRANCH_LABELS
    if normalized == "metric":
        return _METRIC_LABELS
    return {}


def _fallback_label(raw: str) -> str:
    words = str(raw).strip().replace("-", " ").replace("_", " ").split()
    if not words:
        return ""
    rendered: list[str] = []
    for word in words:
        lower = word.lower()
        rendered.append(_TOKEN_REWRITES.get(lower, word.capitalize()))
    return " ".join(rendered)
