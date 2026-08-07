#!/usr/bin/env python3
"""Build shadow Paper-I HH S-accounting reconciliation artifacts.

This script is intentionally non-destructive:

- it does not edit ``MATH/paper_details/Paper_I.tex``;
- it does not overwrite existing Paper-I HH support artifacts;
- it writes a new shadow output directory; and
- it writes duplicated manuscript candidates with updated HH SNAKE ``S`` cells.

The script exists to compare candidate S-accounting conventions before the user
chooses one for the active manuscript/table update.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.snake_table_i_measurement_work import (  # noqa: E402
    snake_algorithmic_work_from_payload,
    snake_mechanism_resolved_work_from_payload,
)


DEFAULT_STAMP = "20260709"

SNAKE_RAW_BASE = REPO_ROOT / "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708"
COMPARISON_CSV = (
    REPO_ROOT
    / "output/pdf/paper_i_hh_physical_operator_lane_comparison_20260708"
    / "paper_i_hh_physical_operator_lane_comparison_20260708_provenance.csv"
)
PAPER_I_TEX = REPO_ROOT / "MATH/paper_details/Paper_I.tex"
COMPARATOR_SUPPORT_CSV = Path(
    "/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/"
    "output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630/"
    "paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_powell_pool_exposure_support.csv"
)
APPEND_K8_OVERRIDE_CSV = (
    REPO_ROOT
    / "output/pdf/paper_i_hh_append_k8_prefix_qiskit_20260708"
    / "paper_i_hh_append_plot_iteration8_qiskit_20260708.csv"
)

# Displayed HH SNAKE rows in the active Paper_I.tex table.  These exact strings
# keep the shadow copy conservative: if upstream table text drifts, this script
# fails closed instead of replacing the wrong cell.
HH_SNAKE_TEX_ROWS: dict[str, str] = {
    "weak-weak": r"SNAKE & 13 & 4.524e-04 & 48 & 34 & 183 & 7,144 \\",
    "intermediate-weak": r"SNAKE & 10 & 2.158e-04 & 38 & 30 & 133 & 5,799 \\",
    "strong-weak": r"SNAKE & 11 & 1.591e-06 & 44 & 37 & 200 & 5,933 \\",
    "weak-strong": r"SNAKE & 16 & 1.841e-02 & 70 & 61 & 206 & 20,108 \\",
    "intermediate-strong": r"SNAKE & 28 & 6.871e-04 & 150 & 121 & 540 & 33,487 \\",
    "strong-strong": r"SNAKE & 13 & 4.683e-05 & 48 & 39 & 188 & 7,434 \\",
}

REGIME_ORDER: tuple[tuple[str, str, int], ...] = (
    ("weak-weak", "weak_weak", 13),
    ("intermediate-weak", "intermediate_weak", 10),
    ("strong-weak", "strong_weak", 11),
    ("weak-strong", "weak_strong", 16),
    ("intermediate-strong", "intermediate_strong", 28),
    ("strong-strong", "strong_strong", 13),
)

HH_COMPARATOR_VISIBLE: dict[tuple[str, str], dict[str, int]] = {
    ("Geo", "weak-weak"): {"k_pl": 5, "visible_s": 24859},
    ("Append", "weak-weak"): {"k_pl": 23, "visible_s": 108633},
    ("Geo", "intermediate-weak"): {"k_pl": 4, "visible_s": 19792},
    ("Append", "intermediate-weak"): {"k_pl": 28, "visible_s": 231208},
    ("Geo", "strong-weak"): {"k_pl": 6, "visible_s": 30007},
    ("Append", "strong-weak"): {"k_pl": 6, "visible_s": 2137},
    ("Geo", "weak-strong"): {"k_pl": 8, "visible_s": 47155},
    ("Append", "weak-strong"): {"k_pl": 23, "visible_s": 136947},
    ("Geo", "intermediate-strong"): {"k_pl": 8, "visible_s": 47245},
    ("Append", "intermediate-strong"): {"k_pl": 25, "visible_s": 167416},
    ("Geo", "strong-strong"): {"k_pl": 6, "visible_s": 33962},
    ("Append", "strong-strong"): {"k_pl": 8, "visible_s": 6378},
}

APPENDIX_S_INVENTORY: tuple[dict[str, Any], ...] = (
    {
        "surface": "appendix_spin_boson_rabi_weak",
        "family": "spin-boson/Rabi",
        "regime": "weak",
        "k_pl": 2,
        "current_s": 472,
        "visible_figure_pdf": "MATH/paper_details/figures/paper_i_alt_hamiltonian_1p75_20260709/paper_i_alt_snake_spin_boson_weak_error_vs_iteration.pdf",
        "source_json": "raw_outputs/paper_i_alt_hamiltonian_physical_operator_lanes_1p75_fullreopt_agent_repair1_20260709/spin_boson/spin_boson_L2_nph1_g0p05_physical_lanes_1p75/json/result.json",
        "line_region": "Paper_I.tex around spin-boson/Rabi appendix panel",
        "status": "source_mapped",
    },
    {
        "surface": "appendix_spin_boson_rabi_strong",
        "family": "spin-boson/Rabi",
        "regime": "strong",
        "k_pl": 3,
        "current_s": 3380,
        "visible_figure_pdf": "MATH/paper_details/figures/paper_i_alt_hamiltonian_1p75_20260709/paper_i_alt_snake_spin_boson_strong_error_vs_iteration.pdf",
        "source_json": "raw_outputs/paper_i_alt_hamiltonian_physical_operator_lanes_1p75_fullreopt_agent_repair1_20260709/spin_boson/spin_boson_L2_nph2_g0p1_physical_lanes_1p75/json/result.json",
        "line_region": "Paper_I.tex around spin-boson/Rabi appendix panel",
        "status": "source_mapped",
    },
    {
        "surface": "appendix_bose_hubbard_weak",
        "family": "Bose--Hubbard",
        "regime": "weak",
        "k_pl": 1,
        "current_s": 1807,
        "visible_figure_pdf": "MATH/paper_details/figures/paper_i_alt_hamiltonian_1p75_20260709/paper_i_alt_snake_bose_hubbard_weak_error_vs_iteration.pdf",
        "source_json": "raw_outputs/paper_i_alt_hamiltonian_physical_operator_lanes_1p75_fullreopt_agent_repair1_20260709/bose_hubbard/bose_hubbard_L2_nph2_u2_physical_lanes_1p75/json/result.json",
        "line_region": "Paper_I.tex around Bose-Hubbard appendix panel",
        "status": "source_mapped",
    },
    {
        "surface": "appendix_bose_hubbard_strong",
        "family": "Bose--Hubbard",
        "regime": "strong",
        "k_pl": 1,
        "current_s": 1838,
        "visible_figure_pdf": "MATH/paper_details/figures/paper_i_alt_hamiltonian_1p75_20260709/paper_i_alt_snake_bose_hubbard_strong_error_vs_iteration.pdf",
        "source_json": "raw_outputs/paper_i_alt_hamiltonian_physical_operator_lanes_1p75_fullreopt_agent_repair1_20260709/bose_hubbard/bose_hubbard_L2_nph2_u6_physical_lanes_1p75/json/result.json",
        "line_region": "Paper_I.tex around Bose-Hubbard appendix panel",
        "status": "source_mapped",
    },
    {
        "surface": "appendix_hubbard_weak",
        "family": "Hubbard",
        "regime": "weak",
        "k_pl": 8,
        "current_s": 5936,
        "visible_figure_pdf": "MATH/paper_details/figures/paper_i_alt_hamiltonian_1p75_20260709/paper_i_alt_snake_hubbard_weak_error_vs_iteration.pdf",
        "source_json": "raw_outputs/paper_i_hubbard_uccsd_qeb_hva_blocks_child3_batch3_shortlist2_depth10_20260709/hubbard/hubbard_L2_open_u0p25_uccsd_qeb_hva_blocks_child3_batch3_shortlist2_depth10/json/result.json",
        "qiskit_sidecar_json": "output/pdf/paper_i_selected_prefix_qiskit_20260709/hubbard_weak_snake_k8_selected_prefix_qiskit_cost_20260709.json",
        "qiskit_costs": {"N_2q": 18, "D_2q": 10, "D_c": 120},
        "line_region": "Paper_I.tex around Hubbard appendix panel",
        "status": "source_mapped_selected_prefix_qiskit_sidecar", 
    },
    {
        "surface": "appendix_hubbard_strong",
        "family": "Hubbard",
        "regime": "strong",
        "k_pl": 2,
        "current_s": 773,
        "visible_figure_pdf": "MATH/paper_details/figures/paper_i_alt_hamiltonian_1p75_20260709/paper_i_alt_snake_hubbard_strong_error_vs_iteration.pdf",
        "source_json": "raw_outputs/paper_i_alt_hamiltonian_physical_operator_lanes_1p75_fullreopt_agent_repair1_20260709/hubbard/hubbard_L2_open_u8_physical_lanes_1p75/json/result.json",
        "line_region": "Paper_I.tex around Hubbard appendix panel",
        "status": "source_mapped",
    },
)

APPENDIX_TEX_ROWS: dict[str, str] = {
    "appendix_spin_boson_rabi_weak": "2 & \\(6{\\times}10^{-7}\\) & 19 & 17 & 123 & 472 " + r"\\",
    "appendix_spin_boson_rabi_strong": "3 & \\(10{\\times}10^{-6}\\) & 115 & 111 & 723 & 3{,}380 " + r"\\",
    "appendix_bose_hubbard_weak": "1 & \\(8{\\times}10^{-16}\\) & 160 & 160 & 870 & 1{,}807 " + r"\\",
    "appendix_bose_hubbard_strong": "1 & \\(1{\\times}10^{-15}\\) & 160 & 160 & 870 & 1{,}838 " + r"\\",
    "appendix_hubbard_weak": "8 & \\(2{\\times}10^{-11}\\) & 18 & 10 & 120 & 5{,}936 " + r"\\",
    "appendix_hubbard_strong": "2 & \\(8{\\times}10^{-16}\\) & 52 & 52 & 274 & 773 " + r"\\",
}

OLD_APPENDIX_SENTENCE = (
    "An estimator query denotes one logical expectation-value query for the prepared "
    "state at the current ansatz prefix. Compatible Pauli terms grouped into the same "
    "estimator setting count once. Thus \\(S\\) is logical estimator-query accounting "
    "for reaching the displayed plateau prefix, not a calibrated physical-shot count."
)

NEW_APPENDIX_SENTENCE = (
    "For the reported tables, an estimator query denotes one logical scalar "
    "expectation-value request used by the algorithm at the relevant prepared state, "
    "such as one candidate gradient, one Fubini--Study norm, one Gram/metric, "
    "curvature, Hessian, or coupling entry, or one Hamiltonian objective evaluation "
    "in an optimizer/refit. Algebraic symmetry and same-state reuse are applied, so "
    "a scalar primitive reused by multiple formulas at the same prepared state is "
    "counted once and symmetric entries are not double-counted. Raw Pauli-word "
    "decompositions, compatible-Pauli grouping, shot allocation, and physical shot "
    "counts are not resolved in \\(S\\); those enter through the hardware/measurement "
    "proxy \\(K_t\\) or separate grouped-shot diagnostics."
)

OLD_SNAKE_APPENDIX_BLOCK = r"""For SNAKE, let \(P_1(k)\), \(P_2(k)\), and \(P_3(k)\) denote the Phase-I, Phase-II, and Phase-III estimator-query sets requested before the \(k\)-th admission. Reused estimator queries are counted once across phases. If \(f_k\) is the number of Hamiltonian objective evaluations used by the optimizer and refit checks after the \(k\)-th admission, then
\begin{align}
S_{\rm SNAKE}
&=
\sum_{k=0}^{k_{\rm pl}}
\left[
|P_1(k)|+|P_2(k)\setminus P_1(k)|
\right.
\nonumber\\
&\qquad\left.
+|P_3(k)\setminus (P_1(k)\cup P_2(k))|+f_k
\right].
\label{eq:appendix_s_snake}
\end{align}"""

NEW_SNAKE_APPENDIX_BLOCK = r"""For SNAKE, let \(P_0(k)\), \(P_1(k)\), \(P_2(k)\), and \(P_3(k)\) denote the Phase-0, Phase-I, Phase-II, and Phase-III scalar estimator-query sets requested before the \(k\)-th admission. Phase 0 contains the first-pass gradient screen; later phases may reuse those same-state scalar estimates rather than recounting them. Reused scalar primitives are counted once across phases. If \(f_k\) is the number of Hamiltonian objective evaluations used by the optimizer and refit checks after the \(k\)-th admission, then
\begin{align}
S_{\rm SNAKE}
&=
\sum_{k=0}^{k_{\rm pl}}
\left[
|P_0(k)|+|P_1(k)\setminus P_0(k)|
\right.
\nonumber\\
&\qquad\left.
+|P_2(k)\setminus (P_0(k)\cup P_1(k))|
\right.
\nonumber\\
&\qquad\left.
+|P_3(k)\setminus (P_0(k)\cup P_1(k)\cup P_2(k))|+f_k
\right].
\label{eq:appendix_s_snake}
\end{align}"""


@dataclass(frozen=True)
class ComparatorShadowRow:
    method: str
    regime: str
    k_pl: int
    visible_s: int
    support_s: int | None
    nested_grad: int | None
    nested_metric: int | None
    nested_h_refit: int | None
    nested_h_outer: int | None
    nested_other_quantum: int | None
    nested_component_sum: int | None
    visible_matches_support_s: bool
    nested_sum_matches_support_s: bool | None
    metric_zero_for_append: bool | None
    source_kind: str
    source_path: str
    source_sha256: str | None
    status: str


@dataclass(frozen=True)
class ShadowSnakeRow:
    method: str
    regime: str
    k_pl: int
    source_json: str
    source_sha256: str
    visible_s: int
    visible_grad: int
    visible_metric: int
    visible_h_refit: int
    runtime_s: int
    runtime_grad: int
    runtime_metric: int
    runtime_h_refit: int
    mechanism_formula_s: int
    mechanism_formula_grad: int
    mechanism_formula_metric: int
    mechanism_formula_h_refit: int
    mechanism_phase0_gradient: int
    mechanism_phase1_first_stage: int
    mechanism_phase2_formula_metric: int
    mechanism_phase3_metric_preserved: int
    mechanism_refit_h: int
    mechanism_formula_component_sum: int
    mechanism_formula_components_sum_to_s: bool
    candidate_count_total: int
    visible_minus_runtime: int
    visible_minus_mechanism_formula: int
    visible_equals_runtime_plus_candidate_total: bool
    visible_grad_is_2x_runtime_grad: bool
    visible_metric_is_2x_runtime_metric: bool
    visible_h_matches_runtime_h: bool
    phase2_formula_metric_total: int
    phase2_replaced_coarse_metric: int
    phase2_non_phase2_metric_preserved: int
    runtime_status: str
    mechanism_status: str
    row_status: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _format_int(value: int | float | None) -> str:
    if value is None:
        return "--"
    return f"{int(value):,}"


def _format_latex_int(value: int | float | None) -> str:
    if value is None:
        return "--"
    return f"{int(value):,}".replace(",", "{,}")


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _path_for_report(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_visible_rows(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"missing current comparison CSV: {path}")
    rows: dict[str, dict[str, int]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("row_source") != "local_physical_operator_lanes_x3":
                continue
            regime = str(row["regime"]).replace("_", "-")
            rows[regime] = {
                "k_pl": _int(row.get("k_pl")),
                "s_alg": _int(row.get("s_alg")),
                "s_grad": _int(row.get("s_grad")),
                "s_metric": _int(row.get("s_metric")),
                "s_h": _int(row.get("s_h")),
            }
    missing = [label for label, _, _ in REGIME_ORDER if label not in rows]
    if missing:
        raise ValueError(f"missing SNAKE visible rows in {path}: {missing}")
    return rows


def _support_component_int(components: Mapping[str, Any], key: str) -> int:
    return _int(components.get(key), 0)


def compute_comparator_rows() -> list[ComparatorShadowRow]:
    rows: list[ComparatorShadowRow] = []
    if not COMPARATOR_SUPPORT_CSV.exists():
        for (method, regime), visible in HH_COMPARATOR_VISIBLE.items():
            rows.append(
                ComparatorShadowRow(
                    method=method,
                    regime=regime,
                    k_pl=int(visible["k_pl"]),
                    visible_s=int(visible["visible_s"]),
                    support_s=None,
                    nested_grad=None,
                    nested_metric=None,
                    nested_h_refit=None,
                    nested_h_outer=None,
                    nested_other_quantum=None,
                    nested_component_sum=None,
                    visible_matches_support_s=False,
                    nested_sum_matches_support_s=None,
                    metric_zero_for_append=None,
                    source_kind="comparator_support_csv_missing",
                    source_path=str(COMPARATOR_SUPPORT_CSV),
                    source_sha256=None,
                    status="blocked_missing_comparator_support_csv",
                )
            )
        return rows

    support_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    with COMPARATOR_SUPPORT_CSV.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            role_key = str(raw.get("role_key") or "")
            if role_key == "geo_macro_c":
                support_by_key[("Geo", str(raw.get("regime") or ""))] = raw
            elif role_key == "append_macro_c":
                support_by_key[("Append", str(raw.get("regime") or ""))] = raw

    override_by_regime: dict[str, dict[str, Any]] = {}
    if APPEND_K8_OVERRIDE_CSV.exists():
        with APPEND_K8_OVERRIDE_CSV.open(newline="") as fh:
            for raw in csv.DictReader(fh):
                override_by_regime[str(raw.get("regime") or "")] = raw

    for regime_label, slug, _snake_k in REGIME_ORDER:
        for method in ("Geo", "Append"):
            visible = HH_COMPARATOR_VISIBLE[(method, regime_label)]
            if method == "Append" and regime_label == "strong-strong":
                override = override_by_regime.get(regime_label)
                support_s = _int(override.get("S")) if override else None
                source_sha = _sha256(APPEND_K8_OVERRIDE_CSV) if APPEND_K8_OVERRIDE_CSV.exists() else None
                rows.append(
                    ComparatorShadowRow(
                        method=method,
                        regime=regime_label,
                        k_pl=int(visible["k_pl"]),
                        visible_s=int(visible["visible_s"]),
                        support_s=support_s,
                        nested_grad=None,
                        nested_metric=None,
                        nested_h_refit=None,
                        nested_h_outer=None,
                        nested_other_quantum=None,
                        nested_component_sum=None,
                        visible_matches_support_s=(support_s == int(visible["visible_s"])),
                        nested_sum_matches_support_s=None,
                        metric_zero_for_append=None,
                        source_kind="strong_strong_append_k8_override_csv",
                        source_path=_path_for_report(APPEND_K8_OVERRIDE_CSV),
                        source_sha256=source_sha,
                        status=(
                            "ok_visible_matches_override_s"
                            if support_s == int(visible["visible_s"])
                            else "blocked_visible_override_mismatch"
                        ),
                    )
                )
                continue

            support = support_by_key.get((method, regime_label))
            if not support:
                rows.append(
                    ComparatorShadowRow(
                        method=method,
                        regime=regime_label,
                        k_pl=int(visible["k_pl"]),
                        visible_s=int(visible["visible_s"]),
                        support_s=None,
                        nested_grad=None,
                        nested_metric=None,
                        nested_h_refit=None,
                        nested_h_outer=None,
                        nested_other_quantum=None,
                        nested_component_sum=None,
                        visible_matches_support_s=False,
                        nested_sum_matches_support_s=None,
                        metric_zero_for_append=None,
                        source_kind="prefix_support_row_missing",
                        source_path=_path_for_report(COMPARATOR_SUPPORT_CSV),
                        source_sha256=_sha256(COMPARATOR_SUPPORT_CSV),
                        status="blocked_missing_prefix_support_row",
                    )
                )
                continue

            detail_raw = str(support.get("s_work_status_detail") or "{}")
            try:
                detail = json.loads(detail_raw)
            except json.JSONDecodeError:
                detail = {}
            components = detail.get("components") if isinstance(detail, Mapping) else {}
            if not isinstance(components, Mapping):
                components = {}
            nested_grad = _support_component_int(components, "N_grad_probe")
            nested_metric = _support_component_int(components, "N_metric_probe")
            nested_h_refit = _support_component_int(components, "N_H_refit_eval")
            nested_h_outer = _support_component_int(components, "N_H_outer_eval")
            nested_other = _support_component_int(components, "N_other_quantum")
            nested_sum = nested_grad + nested_metric + nested_h_refit + nested_h_outer + nested_other
            support_s = _int(support.get("S_alg"))
            source_k = _int(support.get("selected_prefix_k"))
            k_matches = source_k == int(visible["k_pl"])
            visible_matches = int(visible["visible_s"]) == support_s
            nested_matches = nested_sum == support_s
            metric_zero = (nested_metric == 0) if method == "Append" else None
            status = "ok_prefix_nested_components_match_visible_s"
            if not k_matches:
                status = "blocked_prefix_k_mismatch"
            elif not visible_matches:
                status = "blocked_visible_support_s_mismatch"
            elif not nested_matches:
                status = "blocked_nested_component_sum_mismatch"
            elif method == "Append" and not metric_zero:
                status = "blocked_append_metric_nonzero"
            rows.append(
                ComparatorShadowRow(
                    method=method,
                    regime=regime_label,
                    k_pl=int(visible["k_pl"]),
                    visible_s=int(visible["visible_s"]),
                    support_s=support_s,
                    nested_grad=nested_grad,
                    nested_metric=nested_metric,
                    nested_h_refit=nested_h_refit,
                    nested_h_outer=nested_h_outer,
                    nested_other_quantum=nested_other,
                    nested_component_sum=nested_sum,
                    visible_matches_support_s=visible_matches,
                    nested_sum_matches_support_s=nested_matches,
                    metric_zero_for_append=metric_zero,
                    source_kind="comparator_support_csv_nested_s_work_status_detail",
                    source_path=_path_for_report(COMPARATOR_SUPPORT_CSV),
                    source_sha256=_sha256(COMPARATOR_SUPPORT_CSV),
                    status=status,
                )
            )
    return rows


def compute_appendix_inventory() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in APPENDIX_S_INVENTORY:
        row = dict(item)
        fig_path = REPO_ROOT / str(item.get("visible_figure_pdf"))
        source_path = REPO_ROOT / str(item.get("source_json"))
        row["visible_figure_exists"] = fig_path.exists()
        row["source_json_exists"] = source_path.exists()
        row["source_json_sha256"] = _sha256(source_path) if source_path.exists() else None
        row["source_json_status"] = "ok" if source_path.exists() else "missing"
        row["skip_tex_rewrite"] = bool(item.get("skip_tex_rewrite", False))
        row["skip_tex_rewrite_reason"] = item.get("skip_tex_rewrite_reason")
        row["runtime_s"] = None
        row["runtime_status"] = "not_computed"
        row["mechanism_formula_s"] = None
        row["mechanism_status"] = "not_computed"
        row["mechanism_grad"] = None
        row["mechanism_metric"] = None
        row["mechanism_h_refit"] = None
        row["mechanism_phase2_formula_metric"] = None
        row["mechanism_phase3_metric_preserved"] = None
        if source_path.exists():
            payload = json.loads(source_path.read_text())
            runtime, _runtime_audit = snake_algorithmic_work_from_payload(
                payload,
                scope="display_prefix",
                history_position=int(item["k_pl"]),
                source_label=str(item["surface"]),
            )
            mechanism, _mechanism_audit = snake_mechanism_resolved_work_from_payload(
                payload,
                scope="display_prefix",
                history_position=int(item["k_pl"]),
                source_label=str(item["surface"]),
            )
            mechanism_work = mechanism.get("mechanism_algorithmic_work") or {}
            phase2_components = (
                (mechanism_work.get("phase2_formula_reconstruction") or {}).get("components")
                or {}
            )
            row["runtime_s"] = _int(runtime.get("S_alg"))
            row["runtime_status"] = str(runtime.get("S_alg_status") or "unknown")
            row["mechanism_formula_s"] = _int(mechanism_work.get("S_alg"))
            row["mechanism_status"] = str(
                mechanism_work.get("status") or mechanism.get("status") or "unknown"
            )
            row["mechanism_grad"] = _int(mechanism_work.get("S_alg_N_grad_probe"))
            row["mechanism_metric"] = _int(mechanism_work.get("S_alg_N_metric_probe"))
            row["mechanism_h_refit"] = _int(mechanism_work.get("S_alg_N_H_refit_eval")) + _int(
                mechanism_work.get("S_alg_N_H_outer_eval")
            )
            row["mechanism_phase2_formula_metric"] = _int(
                phase2_components.get("phase2_formula_metric_total")
            )
            row["mechanism_phase3_metric_preserved"] = _int(
                phase2_components.get("non_phase2_metric_preserved")
            )
        row["current_matches_runtime"] = row["current_s"] == row["runtime_s"]
        row["current_matches_mechanism_formula"] = (
            row["current_s"] == row["mechanism_formula_s"]
        )
        if row["skip_tex_rewrite"]:
            row["shadow_update_policy"] = "computed_for_audit_only; tex_rewrite_skipped_until_qiskit_cost_sidecar_or_explicit_pending_cost_policy"
        else:
            row["shadow_update_policy"] = (
                "mechanism_shadow_rewrites_appendix_s; active_rewrite_requires_user_approval"
                if row["mechanism_formula_s"] is not None
                else "do_not_rewrite_until_source_artifact_or_explicit_manual_policy_is_available"
            )
        out.append(row)
    return out


def compute_shadow_rows() -> list[ShadowSnakeRow]:
    visible = _load_visible_rows(COMPARISON_CSV)
    out: list[ShadowSnakeRow] = []
    for label, slug, expected_k in REGIME_ORDER:
        source = SNAKE_RAW_BASE / slug / "json/result.json"
        if not source.exists():
            raise FileNotFoundError(f"missing SNAKE source JSON for {label}: {source}")
        payload = json.loads(source.read_text())
        runtime, _runtime_audit = snake_algorithmic_work_from_payload(
            payload,
            scope="display_prefix",
            history_position=expected_k,
            source_label=label,
        )
        mechanism, _mechanism_audit = snake_mechanism_resolved_work_from_payload(
            payload,
            scope="display_prefix",
            history_position=expected_k,
            source_label=label,
        )
        mechanism_work = mechanism.get("mechanism_algorithmic_work") or {}
        ledger = runtime.get("table_i_measurement_event_ledger") or {}
        runtime_reconstruction = ledger.get("runtime_reconstruction") or {}
        controller_phase_counts = (
            runtime_reconstruction.get("controller_phase_actual_operator_probe_counts") or {}
        )
        candidate_ledger = ledger.get("candidate_work_ledger") or {}
        phase2_components = (
            (mechanism_work.get("phase2_formula_reconstruction") or {}).get("components")
            or {}
        )
        vis = visible[label]
        if vis["k_pl"] != expected_k:
            raise ValueError(
                f"visible/current k_pl mismatch for {label}: {vis['k_pl']} != {expected_k}"
            )

        runtime_s = _int(runtime.get("S_alg"))
        runtime_grad = _int(runtime.get("S_alg_N_grad_probe"))
        runtime_metric = _int(runtime.get("S_alg_N_metric_probe"))
        runtime_h = _int(runtime.get("S_alg_N_H_refit_eval")) + _int(
            runtime.get("S_alg_N_H_outer_eval")
        )
        mechanism_s = _int(mechanism_work.get("S_alg"))
        mechanism_grad = _int(mechanism_work.get("S_alg_N_grad_probe"))
        mechanism_metric = _int(mechanism_work.get("S_alg_N_metric_probe"))
        mechanism_h = _int(mechanism_work.get("S_alg_N_H_refit_eval")) + _int(
            mechanism_work.get("S_alg_N_H_outer_eval")
        )
        mechanism_phase0_gradient = _int(controller_phase_counts.get("phase0"))
        mechanism_phase1_first_stage = _int(controller_phase_counts.get("phase1"))
        mechanism_phase2_formula_metric = _int(
            phase2_components.get("phase2_formula_metric_total")
        )
        mechanism_phase3_metric_preserved = _int(
            phase2_components.get("non_phase2_metric_preserved")
        )
        mechanism_refit_h = mechanism_h
        mechanism_formula_component_sum = (
            mechanism_phase0_gradient
            + mechanism_phase1_first_stage
            + mechanism_phase2_formula_metric
            + mechanism_phase3_metric_preserved
            + mechanism_refit_h
        )
        candidate_total = _int(candidate_ledger.get("candidate_count_total"))
        row_status = "blocked_visible_equals_runtime_plus_candidate_total"
        if vis["s_alg"] == runtime_s:
            row_status = "visible_matches_runtime"
        elif vis["s_alg"] == mechanism_s:
            row_status = "visible_matches_mechanism_formula"

        out.append(
            ShadowSnakeRow(
                method="SNAKE",
                regime=label,
                k_pl=expected_k,
                source_json=str(source.relative_to(REPO_ROOT)),
                source_sha256=_sha256(source),
                visible_s=vis["s_alg"],
                visible_grad=vis["s_grad"],
                visible_metric=vis["s_metric"],
                visible_h_refit=vis["s_h"],
                runtime_s=runtime_s,
                runtime_grad=runtime_grad,
                runtime_metric=runtime_metric,
                runtime_h_refit=runtime_h,
                mechanism_formula_s=mechanism_s,
                mechanism_formula_grad=mechanism_grad,
                mechanism_formula_metric=mechanism_metric,
                mechanism_formula_h_refit=mechanism_h,
                mechanism_phase0_gradient=mechanism_phase0_gradient,
                mechanism_phase1_first_stage=mechanism_phase1_first_stage,
                mechanism_phase2_formula_metric=mechanism_phase2_formula_metric,
                mechanism_phase3_metric_preserved=mechanism_phase3_metric_preserved,
                mechanism_refit_h=mechanism_refit_h,
                mechanism_formula_component_sum=mechanism_formula_component_sum,
                mechanism_formula_components_sum_to_s=(
                    mechanism_formula_component_sum == mechanism_s
                ),
                candidate_count_total=candidate_total,
                visible_minus_runtime=vis["s_alg"] - runtime_s,
                visible_minus_mechanism_formula=vis["s_alg"] - mechanism_s,
                visible_equals_runtime_plus_candidate_total=(
                    vis["s_alg"] == runtime_s + candidate_total
                ),
                visible_grad_is_2x_runtime_grad=vis["s_grad"] == 2 * runtime_grad,
                visible_metric_is_2x_runtime_metric=vis["s_metric"] == 2 * runtime_metric,
                visible_h_matches_runtime_h=vis["s_h"] == runtime_h,
                phase2_formula_metric_total=_int(
                    phase2_components.get("phase2_formula_metric_total")
                ),
                phase2_replaced_coarse_metric=_int(
                    phase2_components.get("phase2_replaced_coarse_metric")
                ),
                phase2_non_phase2_metric_preserved=_int(
                    phase2_components.get("non_phase2_metric_preserved")
                ),
                runtime_status=str(runtime.get("S_alg_status") or "unknown"),
                mechanism_status=str(
                    mechanism_work.get("status") or mechanism.get("status") or "unknown"
                ),
                row_status=row_status,
            )
        )
    return out


def _write_csv(path: Path, rows: Iterable[ShadowSnakeRow | ComparatorShadowRow]) -> None:
    dict_rows = [asdict(row) for row in rows]
    if not dict_rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(dict_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dict_rows)


def _write_json(
    path: Path,
    *,
    rows: list[ShadowSnakeRow],
    comparator_rows: list[ComparatorShadowRow],
    stamp: str,
) -> None:
    payload = {
        "schema": "paper_i_hh_s_accounting_shadow_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stamp": stamp,
        "non_destructive": True,
        "active_paper_i_modified": False,
        "raw_outputs_modified": False,
        "source_comparison_csv": str(COMPARISON_CSV.relative_to(REPO_ROOT)),
        "snake_raw_base": str(SNAKE_RAW_BASE.relative_to(REPO_ROOT)),
        "comparator_support_csv": _path_for_report(COMPARATOR_SUPPORT_CSV),
        "append_k8_override_csv": _path_for_report(APPEND_K8_OVERRIDE_CSV),
        "candidate_conventions": {
            "runtime": {
                "description": "Current display-prefix runtime aggregate S_alg from snake_algorithmic_work_from_payload.",
                "field": "runtime_s",
            },
            "mechanism_formula": {
                "description": "Nested mechanism-formula reconstruction from snake_mechanism_resolved_work_from_payload.",
                "field": "mechanism_formula_s",
            },
        },
        "summary": {
            "all_visible_equal_runtime_plus_candidate_total": all(
                row.visible_equals_runtime_plus_candidate_total for row in rows
            ),
            "all_visible_grad_double_runtime_grad": all(
                row.visible_grad_is_2x_runtime_grad for row in rows
            ),
            "all_visible_metric_double_runtime_metric": all(
                row.visible_metric_is_2x_runtime_metric for row in rows
            ),
            "all_visible_h_matches_runtime": all(row.visible_h_matches_runtime_h for row in rows),
            "all_mechanism_formula_components_sum_to_s": all(
                row.mechanism_formula_components_sum_to_s for row in rows
            ),
            "all_comparator_visible_matches_support_s": all(
                row.visible_matches_support_s for row in comparator_rows
            ),
            "all_comparator_nested_rows_sum_to_support_s": all(
                row.nested_sum_matches_support_s is not False for row in comparator_rows
            ),
        },
        "rows": [asdict(row) for row in rows],
        "comparator_rows": [asdict(row) for row in comparator_rows],
        "appendix_s_inventory": compute_appendix_inventory(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_md(
    path: Path,
    *,
    rows: list[ShadowSnakeRow],
    comparator_rows: list[ComparatorShadowRow],
    stamp: str,
    tex_paths: Mapping[str, str],
) -> None:
    lines: list[str] = []
    lines.append(f"# Paper-I HH S-accounting shadow artifacts ({stamp})")
    lines.append("")
    lines.append("This is a non-destructive shadow report. It does not modify `Paper_I.tex`, raw outputs, or existing support artifacts.")
    lines.append("")
    lines.append("## Shadow manuscript candidates")
    lines.append("")
    for convention, tex_path in tex_paths.items():
        lines.append(f"- `{convention}`: `{tex_path}`")
    lines.append("")
    lines.append("## SNAKE HH reconciliation")
    lines.append("")
    lines.append("| regime | k_pl | current visible S | runtime S | mechanism formula S | candidate total | visible-runtime | status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row.regime} | {row.k_pl} | {row.visible_s} | {row.runtime_s} | "
            f"{row.mechanism_formula_s} | {row.candidate_count_total} | "
            f"{row.visible_minus_runtime} | {row.row_status} |"
        )
    lines.append("")
    lines.append("All rows are expected to show the current visible value as runtime plus candidate total if the double-count diagnosis is present.")
    lines.append("")
    lines.append("## Mechanism-formula component check")
    lines.append("")
    lines.append("| regime | P0 gradient | P1 first-stage | P2 formula metric | P3 metric preserved | f/H refit | component sum | mechanism S | check |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row.regime} | {row.mechanism_phase0_gradient} | "
            f"{row.mechanism_phase1_first_stage} | {row.mechanism_phase2_formula_metric} | "
            f"{row.mechanism_phase3_metric_preserved} | {row.mechanism_refit_h} | "
            f"{row.mechanism_formula_component_sum} | {row.mechanism_formula_s} | "
            f"{row.mechanism_formula_components_sum_to_s} |"
        )
    lines.append("")
    lines.append("The mechanism candidate uses `P0 + P1 + P2 + P3 + f_k` scalar-entry accounting: Phase-0 gradients, Phase-I first-stage scalar probes, Phase-II formula-level window metric/curvature reconstruction, Phase-III preserved metric work, and Hamiltonian objective/refit evaluations.")
    lines.append("")
    lines.append("## Component pattern")
    lines.append("")
    lines.append("| regime | runtime grad | visible grad | runtime metric | visible metric | runtime H/refit | visible H/refit |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row.regime} | {row.runtime_grad} | {row.visible_grad} | "
            f"{row.runtime_metric} | {row.visible_metric} | "
            f"{row.runtime_h_refit} | {row.visible_h_refit} |"
        )
    lines.append("")
    lines.append("## Append/Geo comparator prefix reconciliation")
    lines.append("")
    lines.append("This validates the currently displayed Append/Geo HH S cells against prefix-aligned nested support components. Top-level support component columns are intentionally not used for this check.")
    lines.append("")
    lines.append("| method | regime | k_pl | visible S | support S | grad | metric | H/refit | component sum | status |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in comparator_rows:
        lines.append(
            f"| {row.method} | {row.regime} | {row.k_pl} | {row.visible_s} | "
            f"{_format_int(row.support_s)} | {_format_int(row.nested_grad)} | "
            f"{_format_int(row.nested_metric)} | {_format_int(row.nested_h_refit)} | "
            f"{_format_int(row.nested_component_sum)} | {row.status} |"
        )
    lines.append("")
    lines.append("Strong--strong Append uses the explicit k=8 override sidecar; that row validates visible S against the override S but does not expose nested component fields in the override CSV.")
    lines.append("")
    lines.append("## Appendix S inventory")
    lines.append("")
    lines.append("These S cells are now source-mapped in the shadow audit. The mechanism shadow rewrites them; active Paper-I remains unchanged until approval.")
    lines.append("")
    lines.append("| surface | k_pl | current S | runtime S | mechanism formula S | mechanism grad | mechanism metric | mechanism H/refit | rewrite policy | status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for item in compute_appendix_inventory():
        lines.append(
            f"| {item['surface']} | {item['k_pl']} | {item['current_s']} | {_format_int(item['runtime_s'])} | "
            f"{_format_int(item['mechanism_formula_s'])} | {_format_int(item['mechanism_grad'])} | "
            f"{_format_int(item['mechanism_metric'])} | {_format_int(item['mechanism_h_refit'])} | "
            f"{item['shadow_update_policy']} | {item['status']}; source_json_status={item['source_json_status']}; "
            f"current_matches_mechanism={item['current_matches_mechanism_formula']} |"
        )
    lines.append("")
    lines.append("## Safety policy")
    lines.append("")
    lines.append("- Active `MATH/paper_details/Paper_I.tex` is not edited.")
    lines.append("- Existing generated support artifacts are not overwritten.")
    lines.append("- Shadow TeX copies update only HH SNAKE visible S cells and Appendix A convention wording.")
    lines.append("- Appendix benchmark S values are inventoried only until their source artifacts are mapped.")
    lines.append("- Promotion to active Paper-I requires explicit user approval after reviewing the shadow outputs.")
    path.write_text("\n".join(lines) + "\n")


def _shadow_provenance_block(*, convention: str, rows: list[ShadowSnakeRow], stamp: str) -> str:
    key = "runtime_s" if convention == "runtime" else "mechanism_formula_s"
    values = {row.regime: getattr(row, key) for row in rows}
    appendix_key = "runtime_s" if convention == "runtime" else "mechanism_formula_s"
    appendix_inventory = compute_appendix_inventory()
    appendix_values = {
        str(row["surface"]): row.get(appendix_key)
        for row in appendix_inventory
        if not row.get("skip_tex_rewrite")
    }
    appendix_skipped = {
        str(row["surface"]): row.get("skip_tex_rewrite_reason")
        for row in appendix_inventory
        if row.get("skip_tex_rewrite")
    }
    payload = {
        "schema": "paper_i_hh_s_accounting_shadow_tex_update_v1",
        "stamp": stamp,
        "convention": convention,
        "active_paper_i_modified": False,
        "changed_snake_s_cells": values,
        "changed_appendix_s_cells": appendix_values,
        "appendix_s_cells_computed_but_not_tex_rewritten": appendix_skipped,
        "source": "shadow artifact generated from current raw SNAKE JSONs and existing Paper_I.tex duplicate",
    }
    raw = json.dumps(payload, indent=2)
    commented = "\n".join(f"% {line}" for line in raw.splitlines())
    marker = f"HH_S_ACCOUNTING_SHADOW_{convention.upper()}_{stamp}"
    return f"% BEGIN_MACHINE_READABLE_{marker}\n{commented}\n% END_MACHINE_READABLE_{marker}\n"


def _replace_hh_snake_rows(tex: str, *, rows: list[ShadowSnakeRow], convention: str) -> str:
    value_attr = "runtime_s" if convention == "runtime" else "mechanism_formula_s"
    out = tex
    for row in rows:
        template = HH_SNAKE_TEX_ROWS[row.regime]
        visible_value = _format_int(row.visible_s)
        target_value = _format_int(getattr(row, value_attr))
        candidates = [row.visible_s, row.runtime_s, row.mechanism_formula_s]
        replaced = False
        for candidate_value_raw in candidates:
            candidate_value = _format_int(candidate_value_raw)
            candidate = template.replace(f"& {visible_value} ", f"& {candidate_value} ", 1)
            new = template.replace(f"& {visible_value} ", f"& {target_value} ", 1)
            if candidate in out:
                out = out.replace(candidate, new, 1)
                replaced = True
                break
        if not replaced:
            raise ValueError(f"expected active HH SNAKE row not found for {row.regime}: {template}")
    return out


def _replace_appendix_s_rows(tex: str, *, convention: str) -> str:
    value_key = "runtime_s" if convention == "runtime" else "mechanism_formula_s"
    out = tex
    for row in compute_appendix_inventory():
        if row.get("skip_tex_rewrite"):
            continue
        surface = str(row["surface"])
        template = APPENDIX_TEX_ROWS[surface]
        current_value = _format_latex_int(row["current_s"])
        target_value = _format_latex_int(row.get(value_key))
        candidates = [row["current_s"], row.get("runtime_s"), row.get("mechanism_formula_s")]
        replaced = False
        for candidate_value_raw in candidates:
            candidate_value = _format_latex_int(candidate_value_raw)
            candidate = template.replace(f"& {current_value} ", f"& {candidate_value} ", 1)
            new = template.replace(f"& {current_value} ", f"& {target_value} ", 1)
            if candidate in out:
                out = out.replace(candidate, new, 1)
                replaced = True
                break
        if not replaced:
            raise ValueError(f"expected appendix S row not found for {surface}: {template}")
    return out


def _appendix_already_scalar_compliant(tex: str) -> bool:
    return all(
        marker in tex
        for marker in (
            "displayed-prefix logical scalar estimator-query count",
            "|P_0(k)|+|P_1(k)\\setminus P_0(k)|",
            "lower-level measurement diagnostics",
            "not a physical-shot count, raw Pauli-word count, or post-Pauli-grouped measurement-setting count",
        )
    )


def _replace_appendix_sentence(tex: str) -> str:
    if OLD_APPENDIX_SENTENCE in tex:
        return tex.replace(OLD_APPENDIX_SENTENCE, NEW_APPENDIX_SENTENCE, 1)
    if _appendix_already_scalar_compliant(tex):
        return tex
    raise ValueError("expected Appendix A estimator-query sentence not found and scalar-compliant replacement not detected")


def _replace_snake_appendix_block(tex: str) -> str:
    if OLD_SNAKE_APPENDIX_BLOCK in tex:
        return tex.replace(OLD_SNAKE_APPENDIX_BLOCK, NEW_SNAKE_APPENDIX_BLOCK, 1)
    if _appendix_already_scalar_compliant(tex):
        return tex
    raise ValueError("expected Appendix A SNAKE block not found and scalar-compliant replacement not detected")


def _insert_shadow_block(tex: str, block: str) -> str:
    marker = "% END_MACHINE_READABLE_HH_PHYSICAL_LANE_DUPLICATE_UPDATE_20260708\n"
    if marker not in tex:
        raise ValueError("could not locate HH machine-readable block insertion point")
    return tex.replace(marker, marker + "\n" + block, 1)


def write_shadow_tex(*, rows: list[ShadowSnakeRow], convention: str, stamp: str) -> Path:
    tex = PAPER_I_TEX.read_text()
    tex = _replace_hh_snake_rows(tex, rows=rows, convention=convention)
    tex = _replace_appendix_s_rows(tex, convention=convention)
    tex = _replace_snake_appendix_block(tex)
    tex = _replace_appendix_sentence(tex)
    tex = _insert_shadow_block(tex, _shadow_provenance_block(convention=convention, rows=rows, stamp=stamp))
    path = REPO_ROOT / "MATH/paper_details" / f"Paper_I_s_accounting_shadow_{convention}_{stamp}.tex"
    path.write_text(tex)
    return path


def build_shadow(*, stamp: str, output_dir: Path, write_tex: bool) -> dict[str, Any]:
    rows = compute_shadow_rows()
    comparator_rows = compute_comparator_rows()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"paper_i_hh_s_accounting_shadow_{stamp}.csv"
    comparator_csv_path = output_dir / f"paper_i_hh_s_accounting_comparator_shadow_{stamp}.csv"
    json_path = output_dir / f"paper_i_hh_s_accounting_shadow_{stamp}.json"
    md_path = output_dir / f"paper_i_hh_s_accounting_shadow_{stamp}.md"

    _write_csv(csv_path, rows)
    _write_csv(comparator_csv_path, comparator_rows)
    _write_json(json_path, rows=rows, comparator_rows=comparator_rows, stamp=stamp)

    tex_paths: dict[str, str] = {}
    if write_tex:
        for convention in ("runtime", "mechanism_formula"):
            tex_path = write_shadow_tex(rows=rows, convention=convention, stamp=stamp)
            tex_paths[convention] = str(tex_path.relative_to(REPO_ROOT))

    _write_md(
        md_path,
        rows=rows,
        comparator_rows=comparator_rows,
        stamp=stamp,
        tex_paths=tex_paths,
    )
    return {
        "csv": str(csv_path.relative_to(REPO_ROOT)),
        "comparator_csv": str(comparator_csv_path.relative_to(REPO_ROOT)),
        "json": str(json_path.relative_to(REPO_ROOT)),
        "md": str(md_path.relative_to(REPO_ROOT)),
        "tex": tex_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default=DEFAULT_STAMP, help="date/run stamp used in output names")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / f"output/pdf/paper_i_hh_s_accounting_shadow_{DEFAULT_STAMP}"),
        help="shadow output directory; created if missing",
    )
    parser.add_argument(
        "--no-write-tex",
        action="store_true",
        help="write CSV/JSON/MD only; do not create shadow Paper_I tex copies",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_shadow(
        stamp=str(args.stamp),
        output_dir=Path(args.output_dir),
        write_tex=not bool(args.no_write_tex),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
