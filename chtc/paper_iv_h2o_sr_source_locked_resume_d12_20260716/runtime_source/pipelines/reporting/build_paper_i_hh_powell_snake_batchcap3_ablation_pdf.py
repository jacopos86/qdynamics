#!/usr/bin/env python3
"""Build Paper-I HH SNAKE batch-cap-3 ablation support PDF.

This is a local support artifact for the current Paper-I HH SNAKE Powell
ablation.  It compares the no-batch anchor against greedy/combinatorial
batch-cap-3 rows at both the first plateau prefix and terminal ansatz depth.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.snake_table_i_measurement_work import (  # noqa: E402
    snake_algorithmic_work_from_payload,
)
from pipelines.exact_bench.table_i_qiskit_resource_compile import (  # noqa: E402
    TABLE_I_QISKIT_COMPILE_CONVENTION,
    TableICompileUnavailable,
    compile_table_i_pauli_label_groups,
)
from pipelines.reporting.build_paper_i_hh_child_fairness_pdf import (  # noqa: E402
    _num_qubits_from_terminal_groups,
    _statevector_from_state_payload,
    _terminal_snake_pauli_label_groups,
)


WEAK_ROOT = REPO_ROOT / "raw_outputs/paper_i_hh_powell_weak_weak_snake_batchcap3_ablation_20260707"
STRONG_ROOT = REPO_ROOT / "raw_outputs/paper_i_hh_powell_strong_strong_snake_batchcap3_ablation_20260707"
NO_BATCH_ROOT = REPO_ROOT / "raw_outputs/paper_i_hh_powell_visible_batchroute_nobatch_20260706"
OUT_DIR = REPO_ROOT / "output/pdf/paper_i_hh_powell_snake_batchcap3_ablation_weakweak_strongstrong_20260707"
STEM = "paper_i_hh_powell_snake_batchcap3_ablation_weakweak_strongstrong_20260707"
PLATEAU_REL_TOL = 0.10


@dataclass(frozen=True)
class RunSpec:
    label: str
    row_id: str
    result_json: Path
    color: str
    linestyle: str
    marker: str


@dataclass(frozen=True)
class RegimeSpec:
    key: str
    label: str
    runs: tuple[RunSpec, ...]


def _runs_for_regime(regime_key: str, run_root: Path) -> tuple[RunSpec, ...]:
    return (
        RunSpec(
            label="No batch",
            row_id=f"{regime_key}__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__nobatch_fullv2",
            result_json=NO_BATCH_ROOT
            / f"{regime_key}__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__nobatch_fullv2"
            / "json/result.json",
            color="#1f77b4",
            linestyle="-",
            marker="o",
        ),
        RunSpec(
            label="Greedy cap 3",
            row_id=f"{regime_key}__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__greedy_cap3",
            result_json=run_root
            / f"{regime_key}__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__greedy_cap3"
            / "json/result.json",
            color="#d62728",
            linestyle="-",
            marker="s",
        ),
        RunSpec(
            label="Combinatorial cap 3",
            row_id=f"{regime_key}__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__combinatorial_cap3",
            result_json=run_root
            / f"{regime_key}__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__combinatorial_cap3"
            / "json/result.json",
            color="#2ca02c",
            linestyle="-",
            marker="^",
        ),
    )


REGIMES = (
    RegimeSpec("weak_weak", "weak-weak", _runs_for_regime("weak_weak", WEAK_ROOT)),
    RegimeSpec("strong_strong", "strong-strong", _runs_for_regime("strong_strong", STRONG_ROOT)),
)


@dataclass
class TableRow:
    regime: str
    scope: str
    label: str
    row_id: str
    status: str
    k_iter: int | None
    d_ans: int | None
    abs_delta_e: float | None
    one_minus_f: float | None
    n2q: int | None
    d2q: int | None
    dc: int | None
    s_alg: float | None
    s_grad: float | None
    s_h: float | None
    s_metric: float | None
    s_beam_search_total: float | None
    max_observed_batch_size: int | None
    batch_sequence: str
    qiskit_cost_source: str
    s_alg_source: str
    s_alg_status: str
    source_json: str
    source_sha256: str | None
    note: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _fmt_err(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.3e}"


def _fmt_int(value: int | float | None) -> str:
    if value is None:
        return "--"
    try:
        return str(int(round(float(value))))
    except Exception:
        return "--"


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.6g}"


def _tex_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _adapt(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    adapt = payload.get("adapt_vqe")
    if isinstance(adapt, Mapping):
        return adapt
    return payload


def _history(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    adapt = _adapt(payload)
    history = adapt.get("history")
    if isinstance(history, list):
        return [row for row in history if isinstance(row, Mapping)]
    return []


def _history_error(row: Mapping[str, Any]) -> float | None:
    for key in (
        "delta_abs_current",
        "abs_delta_e",
        "benchmark_target_abs_delta_e_current",
        "exact_final_state_benchmark_target_abs_delta_e_current",
    ):
        value = _float_or_none(row.get(key))
        if value is not None and value > 0.0:
            return value
    return None


def _history_points(payload: Mapping[str, Any]) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for idx, row in enumerate(_history(payload), start=1):
        err = _history_error(row)
        if err is not None:
            points.append((idx, err))
    return points


def _plateau_k(points: Sequence[tuple[int, float]]) -> tuple[int | None, dict[str, Any]]:
    clean = [(int(k), float(err)) for k, err in points if err > 0.0]
    if not clean:
        return None, {"status": "missing_positive_trajectory"}
    best = min(err for _k, err in clean)
    threshold = best * (1.0 + PLATEAU_REL_TOL)
    for k, err in clean:
        if err <= threshold:
            return k, {
                "status": "ok",
                "rule": "first_prefix_with_error_within_10pct_of_best_trajectory_error",
                "best_error": best,
                "threshold": threshold,
                "selected_error": err,
            }
    k, err = clean[-1]
    return k, {"status": "fallback_terminal", "best_error": best, "threshold": threshold, "selected_error": err}


def _selected_labels(row: Mapping[str, Any]) -> list[str]:
    for key in ("selected_batch_labels", "selected_ops"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
    for key in ("selected_op", "selected_logical_op", "selected_label"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return [value]
    return []


def _selected_positions(row: Mapping[str, Any], count: int) -> list[int | None]:
    raw = row.get("selected_positions")
    if isinstance(raw, list) and len(raw) >= count:
        out: list[int | None] = []
        for value in raw[:count]:
            out.append(_int_or_none(value))
        return out
    value = row.get("selected_position")
    if value is not None and count == 1:
        return [_int_or_none(value)]
    return [None for _ in range(count)]


def _pauli_map(row: Mapping[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key in ("top_candidates", "admitted_records", "selected_feature_rows"):
        records = row.get(key)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            label = record.get("label") or record.get("candidate_label") or record.get("selected_label")
            paulis = record.get("pauli_labels_exyz") or record.get("pauli_labels") or record.get("pauli_strings")
            if isinstance(label, str) and isinstance(paulis, list):
                clean = [str(pauli).strip().lower() for pauli in paulis if str(pauli).strip()]
                if clean:
                    out[label] = clean
    return out


def _clean_pauli_group(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        label = item.get("pauli_exyz") if isinstance(item, Mapping) else item
        text = str(label).strip().lower()
        if text:
            out.append(text)
    return out


def _fallback_step_groups(
    payload: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    *,
    max_prefix: int | None = None,
) -> tuple[list[list[list[str]]] | None, str | None]:
    adapt = _adapt(payload)
    parameterization = adapt.get("parameterization")
    if not isinstance(parameterization, Mapping):
        return None, None
    blocks = parameterization.get("blocks")
    if not isinstance(blocks, list):
        return None, None
    by_label: dict[str, list[list[str]]] = {}
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        label = block.get("candidate_label")
        terms = block.get("runtime_terms_exyz")
        group = _clean_pauli_group(terms)
        if isinstance(label, str) and group:
            by_label.setdefault(label, []).append(group)
    if not by_label:
        return None, None
    used: dict[str, int] = {}
    out: list[list[list[str]]] = []
    scoped_history = list(history[:max_prefix]) if max_prefix is not None else list(history)
    for row in scoped_history:
        step: list[list[str]] = []
        for label in _selected_labels(row):
            groups = by_label.get(label)
            if not groups:
                return None, None
            idx = used.get(label, 0)
            step.append(groups[idx] if idx < len(groups) else groups[-1])
            used[label] = idx + 1
        if not step:
            return None, None
        out.append(step)
    return out, "adapt_vqe_parameterization_runtime_terms_exyz"


def _reference_state(payload: Mapping[str, Any]) -> tuple[Any | None, str]:
    return _statevector_from_state_payload(
        payload.get("ansatz_input_state") if isinstance(payload.get("ansatz_input_state"), Mapping) else None
    )


def _num_qubits_from_groups(groups: Sequence[Sequence[str]], reference_state: Any | None) -> int:
    for group in groups:
        for label in group:
            if str(label):
                return len(str(label))
    if reference_state is not None:
        import numpy as np

        return int(np.log2(reference_state.size))
    raise ValueError("cannot infer num_qubits")


def _compile_prefix(payload: Mapping[str, Any], result_path: Path, k: int) -> dict[str, Any]:
    history = _history(payload)
    if k < 1 or k > len(history):
        raise ValueError(f"prefix k={k} outside history length {len(history)}")
    fallback_groups, fallback_source = _fallback_step_groups(payload, history, max_prefix=k)
    reference_state, reference_status = _reference_state(payload)
    pauli_groups: list[list[str]] = []
    selected_error: float | None = None
    selected_labels: list[str] = []
    selected_pauli_source = ""
    for idx, row in enumerate(history, start=1):
        if idx > k:
            break
        labels = _selected_labels(row)
        lookup = _pauli_map(row)
        if labels and all(lookup.get(label) for label in labels):
            step_groups = [list(lookup[label]) for label in labels]
            step_source = "history_selected_candidate_pauli_labels"
        elif fallback_groups is not None and idx <= len(fallback_groups):
            step_groups = [list(group) for group in fallback_groups[idx - 1]]
            step_source = str(fallback_source)
        else:
            missing = [label for label in labels if label not in lookup]
            raise ValueError(f"missing Pauli group at prefix {idx}: {missing or labels}")
        for group, position in zip(step_groups, _selected_positions(row, len(step_groups))):
            if position is None or position < 0 or position > len(pauli_groups):
                pauli_groups.append(group)
            else:
                pauli_groups.insert(int(position), group)
        if idx == k:
            selected_error = _history_error(row)
            selected_labels = labels
            selected_pauli_source = step_source
    num_qubits = _num_qubits_from_groups(pauli_groups, reference_state)
    try:
        compiled = compile_table_i_pauli_label_groups(
            pauli_label_groups=tuple(tuple(group) for group in pauli_groups),
            num_qubits=int(num_qubits),
            reference_state=reference_state,
            source_kind="paper_i_hh_batchcap3_selected_snake_prefix",
        )
        compile_status = "ok"
        compile_error = None
    except TableICompileUnavailable as exc:
        compiled = {}
        compile_status = exc.status
        compile_error = exc.reason
    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="display_prefix",
        history_position=int(k),
        source_label=str(result_path),
    )
    return {
        "k_iter": int(k),
        "d_ans": int(len(pauli_groups)),
        "abs_delta_e": selected_error,
        "n2q": _int_or_none(compiled.get("compiled_count_2q_total")),
        "d2q": _int_or_none(compiled.get("compiled_depth_2q_total")),
        "dc": _int_or_none(compiled.get("compiled_depth_total")),
        "qiskit_cost_status": compile_status,
        "qiskit_cost_source": "qiskit_selected_snake_history_prefix_compile",
        "compile_error": compile_error,
        "compile_convention": TABLE_I_QISKIT_COMPILE_CONVENTION,
        "selected_labels": selected_labels,
        "selected_pauli_source": selected_pauli_source,
        "reference_state_status": reference_status,
        "s_alg": _float_or_none(work.get("S_alg")),
        "s_alg_status": str(work.get("S_alg_status") or audit.get("status") or "unknown"),
        "s_alg_source": "snake_algorithmic_work_from_payload(scope=display_prefix)",
        "s_work_audit_status": audit.get("status"),
        "s_grad": _float_or_none(work.get("S_alg_N_grad_probe")),
        "s_metric": _float_or_none(work.get("S_alg_N_metric_probe")),
        "s_h": (_float_or_none(work.get("S_alg_N_H_refit_eval")) or 0.0)
        + (_float_or_none(work.get("S_alg_N_H_outer_eval")) or 0.0),
        "s_beam_search_total": _float_or_none(work.get("S_beam_search_total")),
        "s_beam_search_total_status": work.get("S_beam_search_total_status"),
        "s_alg_work_scope": work.get("S_alg_work_scope"),
        "s_alg_row_policy": work.get("S_alg_row_policy"),
    }


def _compile_terminal(payload: Mapping[str, Any], result_path: Path) -> dict[str, Any]:
    adapt = _adapt(payload)
    history = _history(payload)
    groups, groups_meta = _terminal_snake_pauli_label_groups(payload)
    reference_state, reference_status = _reference_state(payload)
    nq = _num_qubits_from_terminal_groups(groups or (), payload)
    if groups is None or nq is None:
        compiled = {}
        compile_status = f"blocked:{groups_meta.get('status') if groups is None else 'num_qubits_missing'}"
        compile_error = None
    else:
        try:
            compiled = compile_table_i_pauli_label_groups(
                pauli_label_groups=tuple(tuple(group) for group in groups),
                num_qubits=int(nq),
                reference_state=reference_state,
                source_kind="snake_qiskit_compiled_terminal_ansatz_circuit",
            )
            compile_status = "ok"
            compile_error = None
        except TableICompileUnavailable as exc:
            compiled = {}
            compile_status = exc.status
            compile_error = exc.reason
    work, audit = snake_algorithmic_work_from_payload(payload, scope="terminal", source_label=str(result_path))
    return {
        "k_iter": int(len(history)),
        "d_ans": _int_or_none(adapt.get("ansatz_depth")),
        "abs_delta_e": _float_or_none(adapt.get("abs_delta_e")),
        "one_minus_f": None
        if _float_or_none(adapt.get("exact_state_fidelity")) is None
        else 1.0 - float(_float_or_none(adapt.get("exact_state_fidelity"))),
        "n2q": _int_or_none(compiled.get("compiled_count_2q_total")),
        "d2q": _int_or_none(compiled.get("compiled_depth_2q_total")),
        "dc": _int_or_none(compiled.get("compiled_depth_total")),
        "qiskit_cost_status": compile_status,
        "qiskit_cost_source": "snake_qiskit_compiled_terminal_ansatz_circuit",
        "compile_error": compile_error,
        "compile_convention": TABLE_I_QISKIT_COMPILE_CONVENTION,
        "reference_state_status": reference_status,
        "parameterization_reconstruction": groups_meta,
        "s_alg": _float_or_none(work.get("S_alg")),
        "s_alg_status": str(work.get("S_alg_status") or audit.get("status") or "unknown"),
        "s_alg_source": "snake_algorithmic_work_from_payload(scope=terminal)",
        "s_work_audit_status": audit.get("status"),
        "s_grad": _float_or_none(work.get("S_alg_N_grad_probe")),
        "s_metric": _float_or_none(work.get("S_alg_N_metric_probe")),
        "s_h": (_float_or_none(work.get("S_alg_N_H_refit_eval")) or 0.0)
        + (_float_or_none(work.get("S_alg_N_H_outer_eval")) or 0.0),
        "s_beam_search_total": _float_or_none(work.get("S_beam_search_total")),
        "s_beam_search_total_status": work.get("S_beam_search_total_status"),
        "s_beam_search_scope": work.get("S_beam_search_scope"),
        "s_alg_work_scope": work.get("S_alg_work_scope"),
        "s_alg_row_policy": work.get("S_alg_row_policy"),
    }


def _max_batch_size(payload: Mapping[str, Any]) -> tuple[int | None, str]:
    sizes: list[int] = []
    for row in _history(payload):
        size = _int_or_none(row.get("batch_size"))
        if size is None:
            labels = _selected_labels(row)
            size = len(labels) if labels else None
        if size is not None:
            sizes.append(int(size))
    if not sizes:
        return None, ""
    return max(sizes), ",".join(str(value) for value in sizes)


def _row(regime: RegimeSpec, spec: RunSpec, scope: str) -> TableRow:
    payload = _read_json(spec.result_json)
    points = _history_points(payload)
    plateau_k, plateau_meta = _plateau_k(points)
    max_batch, batch_sequence = _max_batch_size(payload)
    if scope == "plateau":
        if plateau_k is None:
            raise ValueError(f"{spec.row_id}: plateau unavailable")
        metrics = _compile_prefix(payload, spec.result_json, int(plateau_k))
        note = json.dumps({"plateau": plateau_meta}, sort_keys=True)
        one_minus_f = None
    else:
        metrics = _compile_terminal(payload, spec.result_json)
        note = "terminal final ansatz"
        one_minus_f = metrics.get("one_minus_f")
    s_status = str(metrics.get("s_alg_status") or "")
    q_status = str(metrics.get("qiskit_cost_status") or "")
    status = "done" if s_status == "ok" and q_status == "ok" else f"blocked:qiskit={q_status};s_alg={s_status}"
    return TableRow(
        regime=regime.label,
        scope=scope,
        label=spec.label,
        row_id=spec.row_id,
        status=status,
        k_iter=_int_or_none(metrics.get("k_iter")),
        d_ans=_int_or_none(metrics.get("d_ans")),
        abs_delta_e=_float_or_none(metrics.get("abs_delta_e")),
        one_minus_f=_float_or_none(one_minus_f),
        n2q=_int_or_none(metrics.get("n2q")),
        d2q=_int_or_none(metrics.get("d2q")),
        dc=_int_or_none(metrics.get("dc")),
        s_alg=_float_or_none(metrics.get("s_alg")),
        s_grad=_float_or_none(metrics.get("s_grad")),
        s_h=_float_or_none(metrics.get("s_h")),
        s_metric=_float_or_none(metrics.get("s_metric")),
        s_beam_search_total=_float_or_none(metrics.get("s_beam_search_total")),
        max_observed_batch_size=max_batch,
        batch_sequence=batch_sequence,
        qiskit_cost_source=str(metrics.get("qiskit_cost_source") or ""),
        s_alg_source=str(metrics.get("s_alg_source") or ""),
        s_alg_status=s_status,
        source_json=_rel(spec.result_json),
        source_sha256=_sha256(spec.result_json),
        note=note,
    )


def _write_csv(rows: Sequence[TableRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _plot_trajectory(
    regime: RegimeSpec,
    rows_by_label: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 3.7))
    for spec in regime.runs:
        payload = rows_by_label[spec.label]
        points = _history_points(payload)
        if not points:
            continue
        xs, ys = zip(*points)
        k_pl, _meta = _plateau_k(points)
        ax.plot(xs, ys, color=spec.color, linestyle=spec.linestyle, linewidth=1.7, label=spec.label)
        if k_pl is not None:
            y_pl = dict(points).get(int(k_pl))
            if y_pl is not None:
                ax.scatter([k_pl], [y_pl], color=spec.color, marker=spec.marker, s=55, zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("ADAPT iteration $k$")
    ax.set_ylabel(r"$|\Delta E|$")
    ax.set_title(regime.label)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _table_tex(rows: Sequence[TableRow], *, title: str) -> list[str]:
    lines = [
        rf"\section*{{{_tex_escape(title)}}}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"Route & $k$ & $d_{\rm ans}$ & $|\Delta E|$ & $1-F$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S_{\rm alg}$ & $S_{\rm beam}$ & max $B$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(row.label),
                    _fmt_int(row.k_iter),
                    _fmt_int(row.d_ans),
                    _fmt_err(row.abs_delta_e),
                    _fmt_err(row.one_minus_f),
                    _fmt_int(row.n2q),
                    _fmt_int(row.d2q),
                    _fmt_int(row.dc),
                    _fmt_int(row.s_alg),
                    _fmt_int(row.s_beam_search_total),
                    _fmt_int(row.max_observed_batch_size),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\normalsize",
        ]
    )
    return lines


def _rows_for(rows: Sequence[TableRow], regime: str, scope: str) -> list[TableRow]:
    return [row for row in rows if row.regime == regime and row.scope == scope]


def _tex_document(
    rows: Sequence[TableRow],
    fig_rels: Mapping[str, str],
    generated: str,
    pending_regimes: Sequence[str],
) -> str:
    completed_regimes = [regime.label for regime in REGIMES if any(row.regime == regime.label for row in rows)]
    manifest_rows = [
        ("Report", STEM),
        ("Generated UTC", generated),
        ("Completed regimes", "; ".join(completed_regimes) if completed_regimes else "none"),
        ("Pending regimes", "; ".join(pending_regimes) if pending_regimes else "none"),
        ("Optimizer", "POWELL"),
        ("Depth cap", "30"),
        ("Optimizer budget", "maxiter 200; final/refit maxiter 200"),
        ("Pool", "full_meta unfiltered; HVA included"),
        ("Child policy", "native Phase-III archival singleton split; subset cap 1"),
        ("Beam/prune", "new beam route; lambda 0.005; live branches 3; children per parent 2; metric prune"),
        ("Ablation variable", "batching off vs greedy/combinatorial reduced-plane target/cap 3"),
        ("Plateau rule", "first prefix within 10 percent of each route's best trajectory error"),
        ("S_alg rule", "winner-lineage row work; S_beam is aggregate beam-search provenance only"),
    ]
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\usepackage{amsmath}",
        r"\begin{document}",
        r"\title{Paper-I HH SNAKE Batch-Cap-3 Ablation Support}",
        r"\author{}",
        r"\date{}",
        r"\maketitle",
        r"\section*{Manifest}",
        r"\scriptsize",
        r"\begin{tabular}{p{0.27\linewidth}p{0.66\linewidth}}",
        r"\toprule",
        r"Field & Value \\",
        r"\midrule",
    ]
    for key, value in manifest_rows:
        lines.append(f"{_tex_escape(key)} & {_tex_escape(value)} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\normalsize",
        ]
    )
    for regime in REGIMES:
        regime_rows = [row for row in rows if row.regime == regime.label]
        if not regime_rows:
            continue
        lines.extend(
            [
                rf"\section*{{{_tex_escape(regime.label)} trajectory}}",
                rf"\includegraphics[width=\linewidth]{{{fig_rels[regime.label]}}}",
                r"\clearpage",
                *_table_tex(
                    _rows_for(rows, regime.label, "plateau"),
                    title=f"{regime.label}: plateau-prefix Qiskit and work costs",
                ),
                r"\clearpage",
                *_table_tex(
                    _rows_for(rows, regime.label, "terminal"),
                    title=f"{regime.label}: terminal-depth Qiskit and work costs",
                ),
                r"\clearpage",
            ]
        )
    lines.extend(
        [
            r"\section*{Provenance}",
            r"\scriptsize",
            r"Full result paths, SHA-256 hashes, plateau metadata, and work-source fields are stored in the JSON and CSV sidecars. "
            r"The PDF intentionally keeps provenance compact to avoid hiding table content behind long filesystem paths.",
            r"\begin{center}",
            r"\begin{tabular}{llll}",
            r"\toprule",
            r"Regime & Scope & Route & Source hash prefix \\",
            r"\midrule",
        ]
    )
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex_escape(row.regime),
                    _tex_escape(row.scope),
                    _tex_escape(row.label),
                    _tex_escape((row.source_sha256 or "")[:12]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def build() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()
    rows: list[TableRow] = []
    fig_rels: dict[str, str] = {}
    pending_regimes: list[str] = []
    row_status: dict[str, Any] = {}
    for regime in REGIMES:
        missing = [spec for spec in regime.runs if not spec.result_json.exists()]
        if missing:
            pending_regimes.append(regime.label)
            row_status[regime.label] = {
                "status": "pending",
                "missing_result_json": [_rel(spec.result_json) for spec in missing],
            }
            continue
        payloads = {spec.label: _read_json(spec.result_json) for spec in regime.runs}
        rows.extend([_row(regime, spec, "plateau") for spec in regime.runs])
        rows.extend([_row(regime, spec, "terminal") for spec in regime.runs])
        fig_path = OUT_DIR / f"figures/{regime.key}_trajectory.png"
        _plot_trajectory(regime, payloads, fig_path)
        fig_rels[regime.label] = f"figures/{regime.key}_trajectory.png"
        row_status[regime.label] = {"status": "done", "row_count": len(regime.runs)}
    csv_path = OUT_DIR / f"{STEM}.csv"
    json_path = OUT_DIR / f"{STEM}.json"
    tex_path = OUT_DIR / f"{STEM}.tex"
    pdf_path = OUT_DIR / f"{STEM}.pdf"
    if not rows:
        raise RuntimeError("no completed regimes available for report")
    _write_csv(rows, csv_path)
    payload = {
        "schema": "paper_i_hh_powell_snake_batchcap3_ablation_support_v2",
        "generated_utc": generated,
        "report_pdf": _rel(pdf_path),
        "report_tex": _rel(tex_path),
        "report_csv": _rel(csv_path),
        "trajectory_figures": {key: value for key, value in fig_rels.items()},
        "plateau_rel_tol": PLATEAU_REL_TOL,
        "regime_status": row_status,
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tex_path.write_text(_tex_document(rows, fig_rels, generated, pending_regimes), encoding="utf-8")
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=OUT_DIR,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return payload


def main() -> None:
    payload = build()
    print(payload["report_pdf"])


if __name__ == "__main__":
    main()
