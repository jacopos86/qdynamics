#!/usr/bin/env python3
"""Build a human-facing Paper-I replacement-candidate report.

The report is intentionally narrow:

* pure Hubbard strong sector, U/t = 8;
* Hubbard--Holstein true-strong Hubbard sectors, U/t = 8, with lambda=0.25
  and lambda=1.25.

It does not edit the manuscript.  It writes a TeX/PDF report plus a JSON sidecar
under output/pdf/.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "output/pdf"
STEM = "paper_i_true_strong_replacement_20260613"

PURE_COMP_ROOT = (
    REPO_ROOT
    / "raw_outputs/chtc_paper_i_clean_current_20260520/remote_raw_outputs/"
    / "paper_i_clean_fermionic_benchmarks_v1"
)
PURE_SNAKE_ROOT = (
    REPO_ROOT
    / "raw_outputs/chtc_snake_v6_fetch/unpacked/raw_outputs/"
    / "routeA_paper_i_snake_tau2e4_hubbard_L2_clean_strong_marrakesh_v6/"
    / "run/hubbard_L2_clean_strong"
)
HH_COMP_ROOTS = (
    REPO_ROOT
    / "raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/"
    / "raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records",
    REPO_ROOT
    / "raw_outputs/chtc_fetches/u8_retrieval_20260614T1203Z/extracted/"
    / "paper_i_hh_u8_comparator_spsa_v1_partial_20260614T120120Z/"
    / "raw_outputs/paper_i_hh_u8_comparator_spsa_v1/records",
)
HH_SNAKE = {
    "hh_u8_strong_weak": REPO_ROOT
    / "raw_outputs/chtc_fetches/paper_i_hh_20260612_quota_retrieval/"
    / "raw_outputs/paper_i_hh_snake_novelty_surface_optuna_20260611_v2_u8_strong_weak/"
    / "run/hh_L2_nph2_three_model_sym_u8_strong_weak/trial_0007/"
    / "hh_L2_nph2_three_model_sym_u8_strong_weak/json/result.json",
    "hh_u8_strong_strong": REPO_ROOT
    / "raw_outputs/chtc_retrievals/paper_i_u8_hh_strong_strong_snake_current_best/"
    / "paper_i_u8_hh_ss_v2_7702629_2_20260614T180758Z/trial_0001_current.json",
}
HH_SNAKE_SOURCE_NOTES = {
    "hh_u8_strong_weak": "CHTC SNAKE novelty-surface row",
    "hh_u8_strong_strong": "live CHTC current-json snapshot; costs unavailable until final cost sidecar",
}
HH_SNAKE_COST_SOURCE_NOTE = "costs from Qiskit compile-scout FakeMarrakesh selected-circuit JSON"
HH_SNAKE_COST_OVERRIDES = {
    "hh_u8_strong_weak": (455, 284, 1079),
}

METHOD_ORDER = (
    "HEA VQE",
    "family VQE",
    "Append-ADAPT",
    "TETRIS-ADAPT",
    "Geo-ADAPT",
    "Qubit/QEB",
    "SNAKE",
)

PURE_METHOD_PATHS = {
    "HEA VQE": "static_hea_qiskit_vqe",
    "family VQE": "static_family_informed_vqe",
    "Append-ADAPT": "static_full_meta_append_adapt_vqe",
    "TETRIS-ADAPT": "static_tetris_qubit_adapt_vqe",
    "Geo-ADAPT": "static_pos_geo_adapt_vqe",
    "Qubit/QEB": "static_qubit_qeb_adapt_vqe",
}

HH_RECORD_METHODS = {
    "HEA VQE": "static_hea_qiskit_vqe",
    "family VQE": "static_family_informed_vqe",
    "Append-ADAPT": "static_full_meta_append_adapt_vqe",
    "TETRIS-ADAPT": "static_tetris_qubit_adapt_vqe",
    "Geo-ADAPT": "static_geo_adapt_vqe",
    "Qubit/QEB": "static_qubit_qeb_adapt_vqe",
}

PENDING_HH = {
    ("hh_u8_strong_weak", "Qubit/QEB"): "running on CHTC",
    ("hh_u8_strong_strong", "Append-ADAPT"): "running on CHTC",
    ("hh_u8_strong_strong", "TETRIS-ADAPT"): "running on CHTC",
    ("hh_u8_strong_strong", "Qubit/QEB"): "running on CHTC",
}

STYLES = {
    "HEA VQE": ("#8C8C8C", "o"),
    "family VQE": ("#5B5B5B", "s"),
    "Append-ADAPT": ("#4C78A8", "o"),
    "TETRIS-ADAPT": ("#F58518", "s"),
    "Geo-ADAPT": ("#54A24B", "^"),
    "Qubit/QEB": ("#72B7B2", "v"),
    "SNAKE": ("#E45756", "*"),
}


@dataclass
class ResultRow:
    section: str
    method: str
    status: str
    k: int | None
    error: float | None
    n2q: float | None
    d2q: float | None
    dc: float | None
    s_work: float | None
    source_json: str | None
    source_note: str
    x: list[int]
    y: list[float]


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    first = text.find("{")
    if first > 0:
        text = text[first:]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> int | None:
    out = _num(value)
    if out is None:
        return None
    return int(round(out))


def _first_num(block: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in block:
            value = _num(block.get(key))
            if value is not None:
                return value
    return None


def _first_int(block: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    value = _first_num(block, keys)
    return None if value is None else int(round(value))


def _row_payload(data: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = data.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        return rows[0]
    result = data.get("result")
    if isinstance(result, Mapping):
        return result
    return data


def _history_from_payload(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    hist = payload.get("adapt_history") or payload.get("history") or payload.get("history_tail") or []
    if not isinstance(hist, list):
        return []
    return [row for row in hist if isinstance(row, Mapping)]


def _reference_energy(data: Mapping[str, Any], payload: Mapping[str, Any]) -> float | None:
    blocks: list[Mapping[str, Any]] = [payload, data]
    for key in ("cutoff_diagnostics", "ground_state", "physical_target_manifest"):
        block = data.get(key)
        if isinstance(block, Mapping):
            blocks.append(block)
    keys = (
        "same_cutoff_exact_gs_energy",
        "exact_gs_energy",
        "exact_energy",
        "energy_exact_Nph",
        "reference_energy_same_cutoff",
        "benchmark_stop_reference_energy",
        "benchmark_target_reference_energy",
    )
    for block in blocks:
        value = _first_num(block, keys)
        if value is not None:
            return value
    return None


def _energy_after(row: Mapping[str, Any]) -> float | None:
    return _first_num(
        row,
        (
            "energy_after_opt",
            "energy_after",
            "energy_current",
            "optimizer_reported_energy",
            "energy",
            "exact_energy_from_final_state",
        ),
    )


def _series_from_history(history: Sequence[Mapping[str, Any]], ref: float | None) -> tuple[list[int], list[float]]:
    if ref is None:
        return [], []
    xs: list[int] = []
    ys: list[float] = []
    for idx, row in enumerate(history, start=1):
        energy = _energy_after(row)
        if energy is None:
            continue
        depth = _first_int(row, ("depth", "adapt_depth", "ansatz_depth", "accepted_depth"))
        xs.append(depth if depth is not None else idx)
        ys.append(max(abs(energy - ref), 1e-14))
    return xs, ys


def _table_row(path: Path, section: str, method: str, note: str) -> ResultRow:
    data = _load_json(path)
    payload = _row_payload(data)
    ref = _reference_energy(data, payload)
    hist = _history_from_payload(payload)
    xs, ys = _series_from_history(hist, ref)
    error = _first_num(
        payload,
        (
            "abs_delta_e_same_cutoff",
            "same_cutoff_abs_delta_e",
            "same_cutoff_error",
            "abs_delta_e",
            "delta_E_abs",
        ),
    )
    if error is None:
        energy = _first_num(payload, ("energy", "optimizer_reported_energy"))
        if energy is not None and ref is not None:
            error = abs(energy - ref)
    k = _first_int(payload, ("adapt_depth_reached", "ansatz_depth", "adapt_depth", "depth"))
    if k is None and xs:
        k = xs[-1]
    return ResultRow(
        section=section,
        method=method,
        status="live snapshot" if path.name.endswith("_current.json") else "completed",
        k=k,
        error=error,
        n2q=_first_num(payload, ("compiled_count_2q_total", "compiled_two_qubit_count", "count_2q")),
        d2q=_first_num(payload, ("compiled_depth_2q_total", "compiled_two_qubit_depth", "depth_2q")),
        dc=_first_num(payload, ("compiled_depth_total", "compiled_circuit_depth", "circuit_depth")),
        s_work=_first_num(payload, ("S_norm", "normalized_estimator_work", "S")),
        source_json=_rel(path),
        source_note=note,
        x=xs,
        y=ys,
    )


def _route_snake_row(path: Path, section: str, method: str, note: str) -> ResultRow:
    data = _load_json(path)
    payload = data.get("adapt_vqe") if isinstance(data.get("adapt_vqe"), Mapping) else _row_payload(data)
    checkpoint = data.get("checkpoint") if isinstance(data.get("checkpoint"), Mapping) else {}
    cutoff = data.get("cutoff_diagnostics") if isinstance(data.get("cutoff_diagnostics"), Mapping) else {}
    first = data.get("paper_i_first_crossing") if isinstance(data.get("paper_i_first_crossing"), Mapping) else {}
    ref = _reference_energy(data, payload)
    hist = _history_from_payload(payload)
    xs, ys = _series_from_history(hist, ref)
    cost = first.get("qiskit_compiled_first_hit_cost")
    if not isinstance(cost, Mapping):
        cost = first.get("paper_i_first_crossing_compiled_cost")
    if not isinstance(cost, Mapping):
        cost = {}
    error = _first_num(
        first,
        ("same_cutoff_error_at_crossing", "terminal_same_cutoff_error", "primary_error_at_crossing"),
    )
    if error is None:
        error = _first_num(cutoff, ("abs_error_same_cutoff", "primary_error"))
    if error is None:
        error = _first_num(
            payload,
            (
                "benchmark_target_abs_delta_e_current",
                "abs_delta_e_same_cutoff",
                "abs_delta_e",
                "delta_abs_current",
                "delta_e",
            ),
        )
    k = _first_int(first, ("history_position_tau", "k_tau"))
    if k is None:
        k = _first_int(checkpoint, ("depth",))
    if k is None:
        k = _first_int(payload, ("history_count", "adapt_depth_reached", "depth", "ansatz_depth"))
    n2q = _first_num(cost, ("compiled_two_qubit_count", "compiled_count_2q_total", "count_2q"))
    d2q = _first_num(cost, ("compiled_two_qubit_depth", "compiled_depth_2q_total", "depth_2q"))
    dc = _first_num(cost, ("compiled_depth", "compiled_depth_total", "circuit_depth"))
    if n2q is None:
        n2q = _first_num(payload, ("compiled_count_2q_total", "compiled_two_qubit_count", "count_2q"))
    if d2q is None:
        d2q = _first_num(payload, ("compiled_depth_2q_total", "compiled_two_qubit_depth", "depth_2q"))
    if dc is None:
        dc = _first_num(payload, ("compiled_depth_total", "compiled_circuit_depth", "circuit_depth"))
    return ResultRow(
        section=section,
        method=method,
        status="completed",
        k=k,
        error=error,
        n2q=n2q,
        d2q=d2q,
        dc=dc,
        s_work=_first_num(payload, ("S_norm", "normalized_estimator_work", "S")),
        source_json=_rel(path),
        source_note=note,
        x=xs,
        y=ys,
    )


def _best_route_snake(root: Path, section: str) -> ResultRow | None:
    best: tuple[float, Path] | None = None
    for path in sorted(root.glob("trial_*/hubbard_L2_clean_strong/json/result.json")):
        try:
            row = _route_snake_row(path, section, "SNAKE", "best same-error SNAKE trial")
        except Exception:
            continue
        if row.error is None:
            continue
        if best is None or row.error < best[0]:
            best = (row.error, path)
    if best is None:
        return None
    return _route_snake_row(best[1], section, "SNAKE", "best same-error SNAKE trial")


def _best_hh_record(section: str, method: str) -> ResultRow | None:
    regime = "strong_weak" if section == "hh_u8_strong_weak" else "strong_strong"
    method_id = HH_RECORD_METHODS[method]
    record = f"paper_i_hh_u8_comp_spsa__full__{method_id}__hh_u8_{regime}"
    candidates: list[ResultRow] = []
    for root in HH_COMP_ROOTS:
        rec_root = root / record
        if not rec_root.exists():
            continue
        for path in rec_root.glob("trial_*/cases/*/result.json"):
            try:
                row = _table_row(path, section, method, "best same-error CHTC trial")
            except Exception:
                continue
            if row.error is not None:
                candidates.append(row)
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row.error if row.error is not None else float("inf")))


def _pending_row(section: str, method: str, status: str) -> ResultRow:
    return ResultRow(section, method, status, None, None, None, None, None, None, None, "", [], [])


def _apply_hh_snake_cost_override(row: ResultRow) -> ResultRow:
    override = HH_SNAKE_COST_OVERRIDES.get(row.section)
    if row.method != "SNAKE" or override is None:
        return row
    n2q, d2q, dc = override
    if row.n2q is None:
        row.n2q = n2q
    if row.d2q is None:
        row.d2q = d2q
    if row.dc is None:
        row.dc = dc
    if HH_SNAKE_COST_SOURCE_NOTE not in row.source_note:
        row.source_note = f"{row.source_note}; {HH_SNAKE_COST_SOURCE_NOTE}"
    return row


def _build_rows() -> dict[str, list[ResultRow]]:
    rows: dict[str, list[ResultRow]] = {
        "hubbard_u8_strong": [],
        "hh_u8_strong_weak": [],
        "hh_u8_strong_strong": [],
    }
    for method in METHOD_ORDER:
        if method == "SNAKE":
            snake = _best_route_snake(PURE_SNAKE_ROOT, "hubbard_u8_strong")
            rows["hubbard_u8_strong"].append(snake or _pending_row("hubbard_u8_strong", method, "missing locally"))
            continue
        method_id = PURE_METHOD_PATHS[method]
        path = PURE_COMP_ROOT / f"static_table__hubbard__hubbard_L2_clean_strong__{method_id}" / "result/result.json"
        if path.exists():
            rows["hubbard_u8_strong"].append(_table_row(path, "hubbard_u8_strong", method, "clean U/t=8 competitor row"))
        else:
            rows["hubbard_u8_strong"].append(_pending_row("hubbard_u8_strong", method, "missing locally"))

    for section in ("hh_u8_strong_weak", "hh_u8_strong_strong"):
        for method in METHOD_ORDER:
            if method == "SNAKE":
                path = HH_SNAKE[section]
                row = (
                    _route_snake_row(path, section, method, HH_SNAKE_SOURCE_NOTES.get(section, "CHTC SNAKE row"))
                    if path.exists()
                    else _pending_row(section, method, "missing locally")
                )
                rows[section].append(_apply_hh_snake_cost_override(row))
                continue
            row = _best_hh_record(section, method)
            if row is None:
                row = _pending_row(section, method, PENDING_HH.get((section, method), "missing locally"))
            rows[section].append(row)
    return rows


def _fmt_num(value: float | None, *, integer: bool = False) -> str:
    if value is None:
        return "--"
    if integer:
        return f"{int(round(value))}"
    if value == 0:
        return "0"
    if abs(value) < 1e-2 or abs(value) >= 1e3:
        return f"{value:.3e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{value:.5f}".rstrip("0").rstrip(".")


def _tex_escape(text: str) -> str:
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


def _plot(section: str, rows: Sequence[ResultRow]) -> Path | None:
    plot_rows = [row for row in rows if row.x and row.y]
    if not plot_rows:
        return None
    fig, ax = plt.subplots(figsize=(7.1, 3.75))
    handles: list[Line2D] = []
    for row in plot_rows:
        color, marker = STYLES.get(row.method, ("#333333", "o"))
        linewidth = 2.8 if row.method == "SNAKE" else 1.9
        pairs = list(zip(row.x, row.y, strict=False))
        marker_k = row.k if row.k is not None else row.x[-1]
        marker_error = row.error if row.error is not None else row.y[-1]
        display_pair = (marker_k, max(marker_error, 1e-14))
        for idx, (x_value, _) in enumerate(pairs):
            if x_value == marker_k:
                pairs[idx] = display_pair
                break
        else:
            pairs.append(display_pair)
        pairs = sorted(pairs, key=lambda item: item[0])
        plot_x = [x_value for x_value, _ in pairs]
        plot_y = [y_value for _, y_value in pairs]
        ax.plot(plot_x, plot_y, color=color, linestyle="-", linewidth=linewidth, alpha=0.95)
        ax.scatter([marker_k], [max(marker_error, 1e-14)], color=color, marker=marker, s=80, zorder=5)
        handles.append(Line2D([0], [0], color=color, linestyle="-", linewidth=linewidth, label=row.method))
    ax.set_yscale("log")
    ax.set_xlabel("ADAPT iteration")
    ax.set_ylabel(r"$|\Delta E|$")
    titles = {
        "hubbard_u8_strong": r"Hubbard model, strong sector: $U/t=8$",
        "hh_u8_strong_weak": r"Hubbard--Holstein: $U/t=8$, $\lambda=0.25$, $n_{\rm ph}=2$",
        "hh_u8_strong_strong": r"Hubbard--Holstein: $U/t=8$, $\lambda=1.25$, $n_{\rm ph}=4$",
    }
    ax.set_title(titles[section])
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(handles=handles, fontsize=7.5, ncol=2, title="solid lines; marker marks displayed row", title_fontsize=7.5)
    fig.tight_layout()
    path = OUT_DIR / f"{STEM}_{section}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _table_tex(rows: Sequence[ResultRow]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Method & Status & $k$ & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                _tex_escape(row.method),
                _tex_escape(row.status),
                "--" if row.k is None else str(row.k),
                _fmt_num(row.error),
                _fmt_num(row.n2q, integer=True),
                _fmt_num(row.d2q, integer=True),
                _fmt_num(row.dc, integer=True),
                _fmt_num(row.s_work, integer=True),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _section_tex(section: str, rows: Sequence[ResultRow], plot: Path | None) -> str:
    headings = {
        "hubbard_u8_strong": r"Hubbard Model: Strong Sector, \(U/t=8\)",
        "hh_u8_strong_weak": r"Hubbard--Holstein: \(U/t=8,\ \lambda=0.25,\ n_{\rm ph}=2\)",
        "hh_u8_strong_strong": r"Hubbard--Holstein: \(U/t=8,\ \lambda=1.25,\ n_{\rm ph}=4\)",
    }
    captions = {
        "hubbard_u8_strong": "Pure Hubbard strong-sector comparison. The error is the raw absolute energy error against exact diagonalization for U/t=8.",
        "hh_u8_strong_weak": r"Hubbard--Holstein strong-Hubbard, weak-Holstein comparison. The phonon cutoff is \(n_{\rm ph}=2\) for both the algorithm and exact diagonalization.",
        "hh_u8_strong_strong": r"Hubbard--Holstein strong-Hubbard, strong-Holstein comparison. The phonon cutoff is \(n_{\rm ph}=4\) for both the algorithm and exact diagonalization.",
    }
    out = [rf"\section*{{{headings[section]}}}"]
    if plot is not None:
        out.extend(
            [
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.95\linewidth]{{{plot.name}}}",
                rf"\caption{{{captions[section]} Solid lines show available ADAPT histories; each marker is the displayed table row.}}",
                r"\end{figure}",
            ]
        )
    out.append(_table_tex(rows))
    return "\n".join(out)


def _write_tex(rows_by_section: Mapping[str, Sequence[ResultRow]], plots: Mapping[str, Path | None]) -> Path:
    tex = OUT_DIR / f"{STEM}.tex"
    comments = []
    for rows in rows_by_section.values():
        for row in rows:
            if row.source_json:
                comments.append(f"% source: {row.section} | {row.method} | {row.source_json} | {row.source_note}")
    body = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{5pt}",
        *comments,
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large Paper-I Replacement Candidate: Strong Hubbard Evidence}\\[3pt]",
        r"{\normalsize Generated June 14, 2026}",
        r"\end{center}",
        r"All energy errors are raw absolute errors against exact diagonalization for the same Hamiltonian shown in the section title. For Hubbard--Holstein rows, \(n_{\rm ph}\) is the phonon cutoff used both by the algorithm and by exact diagonalization.",
        _section_tex("hubbard_u8_strong", rows_by_section["hubbard_u8_strong"], plots.get("hubbard_u8_strong")),
        r"\clearpage",
        _section_tex("hh_u8_strong_weak", rows_by_section["hh_u8_strong_weak"], plots.get("hh_u8_strong_weak")),
        r"\clearpage",
        _section_tex("hh_u8_strong_strong", rows_by_section["hh_u8_strong_strong"], plots.get("hh_u8_strong_strong")),
        r"\end{document}",
    ]
    tex.write_text("\n".join(body) + "\n", encoding="utf-8")
    return tex


def _compile(tex: Path) -> Path:
    cmd = ["tectonic", "--keep-logs", "--reruns", "2", tex.name]
    subprocess.run(cmd, cwd=tex.parent, check=True)
    return tex.with_suffix(".pdf")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_section = _build_rows()
    plots = {section: _plot(section, rows) for section, rows in rows_by_section.items()}
    tex = _write_tex(rows_by_section, plots)
    pdf = _compile(tex)
    sidecar = OUT_DIR / f"{STEM}.json"
    sidecar.write_text(
        json.dumps(
            {
                "pdf": _rel(pdf),
                "tex": _rel(tex),
                "sections": {section: [asdict(row) for row in rows] for section, rows in rows_by_section.items()},
                "plots": {section: _rel(path) if path else None for section, path in plots.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
