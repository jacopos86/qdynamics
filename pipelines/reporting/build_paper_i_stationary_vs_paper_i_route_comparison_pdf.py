#!/usr/bin/env python3
"""Build a direct Paper-I route-vs-stationary-source RA comparison PDF.

The report compares like-for-like no-insertion and plateau-insertion RA
trajectories at the same cutoff and 50-round horizon.  It consumes the
source-locked Paper-I insertion-overlay sidecars and the validated evolving
stationary-core sidecar.  RA-always and conventional Append-ADAPT are outside
this comparison.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
OLD_ROOT = REPO_ROOT / (
    "MATH/paper_details/figures/"
    "paper_i_hh_macro_common_accuracy_20260723"
)
OLD_OVERLAYS = {
    "macro": OLD_ROOT
    / "paper_i_hh_macro_common_accuracy_20260723_"
    "macro_insertion_policy_overlay_provenance.json",
    "singleton": OLD_ROOT
    / "paper_i_hh_macro_common_accuracy_20260723_"
    "singleton_insertion_policy_overlay_provenance.json",
}
CURRENT_PROVENANCE = REPO_ROOT / (
    "output/pdf/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_"
    "evolving_partial_progress_provenance.json"
)
OUTPUT_DIR = REPO_ROOT / (
    "output/pdf/"
    "paper_i_stationary_vs_paper_i_route_comparison_20260729"
)
STEM = "paper_i_stationary_vs_paper_i_route_comparison_20260729"

REGIMES = (
    ("weak_weak", "Weak--weak", "WW"),
    ("intermediate_weak", "Intermediate--weak", "IW"),
    ("strong_weak_u8", "Strong--weak", "SW"),
    ("weak_strong", "Weak--strong", "WS"),
    ("intermediate_strong", "Intermediate--strong", "IS"),
    ("strong_strong_u8", "Strong--strong", "SS"),
)
POLICIES = ("no_insertion", "plateau")
REPRESENTATIONS = ("macro", "singleton")
COST_FIELDS = ("N2q", "D2q", "Dc", "W1q", "S_alg")

CURVE_STYLES: Mapping[tuple[str, str], Mapping[str, Any]] = {
    ("old", "no_insertion"): {
        "label": "Paper I -- no insertion",
        "color": "#3A3A3A",
        "linestyle": "--",
        "linewidth": 1.45,
        "marker": "o",
    },
    ("new", "no_insertion"): {
        "label": "Stationary source -- no insertion",
        "color": "#2F6B9A",
        "linestyle": "-",
        "linewidth": 1.85,
        "marker": "s",
    },
    ("old", "plateau"): {
        "label": "Paper I -- plateau insertion",
        "color": "#8E8E8E",
        "linestyle": ":",
        "linewidth": 1.55,
        "marker": "D",
    },
    ("new", "plateau"): {
        "label": "Stationary source -- plateau insertion",
        "color": "#C94C4C",
        "linestyle": "-",
        "linewidth": 1.85,
        "marker": "D",
    },
}


class ComparisonInputError(ValueError):
    """Raised when source evidence cannot support the comparison."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonInputError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComparisonInputError(f"{label} must be a JSON object")
    return payload


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonInputError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ComparisonInputError(f"{label} must be an array")
    return value


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonInputError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ComparisonInputError(f"{label} must be finite and nonnegative")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ComparisonInputError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonInputError(f"{label} must be an integer") from exc
    if result < 0:
        raise ComparisonInputError(f"{label} must be nonnegative")
    return result


def _normalize_points(
    raw_points: Any,
    *,
    label: str,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for expected_k, raw in enumerate(
        _sequence(raw_points, label=f"{label} points"),
        start=1,
    ):
        row = _mapping(raw, label=f"{label} point {expected_k}")
        k = _integer(row.get("k"), label=f"{label} k")
        error = _finite(row.get("error"), label=f"{label} error")
        if k != expected_k:
            raise ComparisonInputError(
                f"{label} has noncontiguous controller rounds"
            )
        points.append({"k": k, "error": error})
    if len(points) != 50:
        raise ComparisonInputError(f"{label} does not contain 50 rounds")
    return points


def _normalize_costs(raw: Any, *, label: str) -> dict[str, int]:
    row = _mapping(raw, label=label)
    return {
        field: _integer(row.get(field), label=f"{label} {field}")
        for field in COST_FIELDS
    }


def _old_cells() -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    sources: dict[str, Any] = {}
    expected_regimes = {row[0] for row in REGIMES}
    for representation, path in OLD_OVERLAYS.items():
        payload = _load_object(path, label=f"Paper-I {representation} overlay")
        if (
            payload.get("schema") != "paper_i_hh_insertion_policy_overlay_v4"
            or payload.get("metric") != "same_cutoff_absolute_energy_error"
            or payload.get("display_horizon") != 50
        ):
            raise ComparisonInputError(
                f"Paper-I {representation} overlay identity drifted"
            )
        expected_representation = (
            "intact_macro"
            if representation == "macro"
            else "projected_singleton"
        )
        if payload.get("representation") != expected_representation:
            raise ComparisonInputError(
                f"Paper-I {representation} representation drifted"
            )
        rows = _sequence(payload.get("rows"), label="Paper-I overlay rows")
        if {str(_mapping(row, label="old row").get("regime")) for row in rows} != (
            expected_regimes
        ):
            raise ComparisonInputError(
                f"Paper-I {representation} regime matrix drifted"
            )
        for raw in rows:
            row = _mapping(raw, label=f"Paper-I {representation} row")
            regime = str(row.get("regime", ""))
            curves = _mapping(row.get("curves"), label=f"{regime} old curves")
            costs = _mapping(
                row.get("endpoint_costs"),
                label=f"{regime} old endpoint costs",
            )
            for policy in POLICIES:
                curve = _mapping(
                    curves.get(policy),
                    label=f"{regime} Paper-I {policy} curve",
                )
                points = _normalize_points(
                    curve.get("displayed_points"),
                    label=f"{regime} Paper-I {representation} {policy}",
                )
                terminal = _finite(
                    curve.get("terminal_error"),
                    label=f"{regime} Paper-I {policy} terminal error",
                )
                if not math.isclose(
                    terminal,
                    points[-1]["error"],
                    rel_tol=1.0e-11,
                    abs_tol=1.0e-12,
                ):
                    raise ComparisonInputError(
                        f"{regime} Paper-I {policy} terminal error drifted"
                    )
                cost_key = (
                    "no_insertion_ra_adapt"
                    if policy == "no_insertion"
                    else "plateau_insertion_ra_adapt"
                )
                cells[(representation, regime, policy)] = {
                    "source": "old",
                    "representation": representation,
                    "regime": regime,
                    "policy": policy,
                    "points": points,
                    "terminal_error": terminal,
                    "costs": _normalize_costs(
                        costs.get(cost_key),
                        label=f"{regime} Paper-I {policy} costs",
                    ),
                }
        compiler = _mapping(
            payload.get("uniform_qiskit_compiler"),
            label=f"Paper-I {representation} compiler",
        )
        sources[representation] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "schema": payload["schema"],
            "compiler_fingerprint_sha256": compiler.get(
                "fingerprint_sha256"
            ),
        }
    if len(cells) != 24:
        raise ComparisonInputError("Paper-I comparison matrix is incomplete")
    return cells, sources


def _current_policy(route_id: str) -> str | None:
    if route_id.endswith("_append_only"):
        return "no_insertion"
    if route_id.endswith("_plateau"):
        return "plateau"
    return None


def _current_representation(candidate_representation: str) -> str:
    if candidate_representation == "macro_generator_v1":
        return "macro"
    if candidate_representation == "single_pauli_word_v1":
        return "singleton"
    raise ComparisonInputError(
        f"unknown current representation: {candidate_representation}"
    )


def _read_current_cell(
    source: Mapping[str, Any],
    source_records: Mapping[int, Mapping[str, Any]],
) -> tuple[tuple[str, str, str], dict[str, Any]]:
    execution_id = str(source.get("execution_id", ""))
    policy = _current_policy(str(source.get("route_id", "")))
    if policy is None or source.get("method_family") != "ra":
        raise ComparisonInputError(
            f"{execution_id}: non-comparison source reached RA loader"
        )
    representation = _current_representation(
        str(source.get("candidate_representation", ""))
    )
    regime = str(source.get("regime_id", ""))
    source_index = _integer(
        source.get("source_receipt_index"),
        label=f"{execution_id} source receipt index",
    )
    record = source_records.get(source_index)
    if record is None:
        raise ComparisonInputError(
            f"{execution_id}: source receipt is unavailable"
        )
    attempt = Path(str(record.get("fetched_dir", ""))) / str(
        source.get("attempt_path", "")
    )
    if not attempt.is_file() or attempt.is_symlink():
        raise ComparisonInputError(
            f"{execution_id}: selected attempt archive is unavailable"
        )
    attempt_sha = _sha256_file(attempt)
    if attempt_sha != source.get("attempt_sha256"):
        raise ComparisonInputError(
            f"{execution_id}: selected attempt archive hash drifted"
        )
    try:
        with tarfile.open(attempt, "r:gz") as archive:
            member = archive.getmember("worker_outputs/summary.json")
            if not member.isfile():
                raise ComparisonInputError(
                    f"{execution_id}: summary is not a regular member"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise ComparisonInputError(
                    f"{execution_id}: summary bytes are unavailable"
                )
            raw_summary = stream.read()
    except (KeyError, tarfile.TarError) as exc:
        raise ComparisonInputError(
            f"{execution_id}: summary archive read failed: {exc}"
        ) from exc
    if _sha256_bytes(raw_summary) != source.get("summary_file_sha256"):
        raise ComparisonInputError(f"{execution_id}: summary hash drifted")
    try:
        summary = json.loads(raw_summary.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonInputError(
            f"{execution_id}: summary JSON is invalid"
        ) from exc
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema") != "paper_i_run_summary_v1"
    ):
        raise ComparisonInputError(f"{execution_id}: summary schema drifted")
    trace = _sequence(
        summary.get("accepted_error_trace"),
        label=f"{execution_id} error trace",
    )
    points: list[dict[str, Any]] = []
    for expected_k, raw in enumerate(trace, start=1):
        row = _mapping(raw, label=f"{execution_id} trace row")
        k = _integer(
            row.get("controller_round"),
            label=f"{execution_id} controller round",
        )
        error = _finite(
            row.get("absolute_energy_error"),
            label=f"{execution_id} absolute error",
        )
        if k != expected_k:
            raise ComparisonInputError(
                f"{execution_id}: trace rounds are noncontiguous"
            )
        points.append({"k": k, "error": error})
    if len(points) != 50:
        raise ComparisonInputError(f"{execution_id}: trace is not 50 rounds")
    terminal = _mapping(
        source.get("terminal"),
        label=f"{execution_id} terminal observation",
    )
    terminal_error = _finite(
        terminal.get("error"),
        label=f"{execution_id} terminal error",
    )
    if (
        terminal.get("status") != "complete"
        or terminal.get("k") != 50
        or source.get("plotted_point_count") != 51
        or not math.isclose(
            terminal_error,
            points[-1]["error"],
            rel_tol=1.0e-11,
            abs_tol=1.0e-12,
        )
    ):
        raise ComparisonInputError(
            f"{execution_id}: terminal observation drifted"
        )
    return (
        (representation, regime, policy),
        {
            "source": "new",
            "execution_id": execution_id,
            "representation": representation,
            "regime": regime,
            "policy": policy,
            "points": points,
            "terminal_error": terminal_error,
            "costs": _normalize_costs(
                terminal,
                label=f"{execution_id} terminal costs",
            ),
            "attempt": {
                "path": str(attempt),
                "sha256": attempt_sha,
                "summary_file_sha256": source["summary_file_sha256"],
                "worker_receipt_sha256": source[
                    "worker_receipt_sha256"
                ],
                "package_id": source["package_id"],
                "core_materialization_id": source[
                    "core_materialization_id"
                ],
            },
        },
    )


def _current_cells() -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, Any],
]:
    provenance = _load_object(
        CURRENT_PROVENANCE,
        label="stationary-source evolving provenance",
    )
    if (
        provenance.get("schema")
        != (
            "paper_i_ra_adapt_stationary_core_master_"
            "cross_revision_partial_progress_v1"
        )
        or provenance.get("metric") != "same_cutoff_absolute_energy_error"
        or provenance.get("partial_progress") is not True
        or provenance.get("paper_evidence_adopted") is not False
    ):
        raise ComparisonInputError(
            "stationary-source evolving provenance identity drifted"
        )
    raw_records = _sequence(
        provenance.get("source_records"),
        label="current source records",
    )
    source_records = {
        _integer(
            _mapping(row, label="current source record").get(
                "source_receipt_index"
            ),
            label="current source record index",
        ): _mapping(row, label="current source record")
        for row in raw_records
    }
    comparison_sources = [
        _mapping(raw, label="current included source")
        for raw in _sequence(
            provenance.get("included_sources"),
            label="current included sources",
        )
        if (
            _mapping(raw, label="current included source").get(
                "method_family"
            )
            == "ra"
            and _current_policy(
                str(
                    _mapping(raw, label="current included source").get(
                        "route_id", ""
                    )
                )
            )
            is not None
        )
    ]
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_read_current_cell, row, source_records)
            for row in comparison_sources
        ]
        for future in concurrent.futures.as_completed(futures):
            key, cell = future.result()
            if key in cells:
                raise ComparisonInputError(
                    f"duplicate current comparison cell: {key}"
                )
            cells[key] = cell
    expected_no_insertion = {
        (representation, regime, "no_insertion")
        for representation in REPRESENTATIONS
        for regime, _title, _abbr in REGIMES
    }
    observed_no_insertion = {
        key for key in cells if key[2] == "no_insertion"
    }
    if observed_no_insertion != expected_no_insertion:
        raise ComparisonInputError(
            "stationary-source no-insertion matrix is incomplete"
        )
    return cells, {
        "path": str(CURRENT_PROVENANCE),
        "sha256": _sha256_file(CURRENT_PROVENANCE),
        "schema": provenance["schema"],
        "included_count_in_source_report": provenance["included_count"],
        "pending_count_in_source_report": provenance["pending_count"],
        "comparison_cell_count": len(cells),
        "package_ids": provenance["package_ids"],
        "terminal_cost_policy": provenance["terminal_cost_policy"],
        "parameter_manifest": provenance["parameter_manifest"],
    }


def _format_s_alg(value: int) -> str:
    if value == 0:
        return "0.0e0"
    exponent = int(math.floor(math.log10(abs(value))))
    coefficient = value / (10**exponent)
    return f"{coefficient:.1f}e{exponent}"


def _cost_tuple_tex(costs: Mapping[str, Any] | None) -> str:
    if costs is None:
        return r"\textemdash"
    return (
        r"$("
        + ",".join(
            (
                str(_integer(costs["N2q"], label="N2q")),
                str(_integer(costs["D2q"], label="D2q")),
                str(_integer(costs["Dc"], label="Dc")),
                str(_integer(costs["W1q"], label="W1q")),
                _format_s_alg(_integer(costs["S_alg"], label="S_alg")),
            )
        )
        + r")$"
    )


def _error_tex(value: float | None) -> str:
    if value is None:
        return r"\textemdash"
    if value == 0:
        return r"$0$"
    exponent = int(math.floor(math.log10(abs(value))))
    coefficient = value / (10**exponent)
    return rf"${coefficient:.2f}\!\times\!10^{{{exponent}}}$"


def _delta_tex(value: float | None) -> str:
    if value is None:
        return r"\textemdash"
    return rf"${value:+.2f}$"


def _tex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _comparison_rows(
    old_cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
    current_cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for regime, title, abbreviation in REGIMES:
            for policy in POLICIES:
                key = (representation, regime, policy)
                old = old_cells[key]
                new = current_cells.get(key)
                old_error = float(old["terminal_error"])
                new_error = (
                    None if new is None else float(new["terminal_error"])
                )
                log10_ratio = (
                    None
                    if new_error is None
                    else math.log10(
                        max(new_error, 1.0e-300)
                        / max(old_error, 1.0e-300)
                    )
                )
                rows.append(
                    {
                        "representation": representation,
                        "regime": regime,
                        "regime_title": title,
                        "abbreviation": abbreviation,
                        "policy": policy,
                        "old_terminal_error": old_error,
                        "new_terminal_error": new_error,
                        "log10_new_over_old": log10_ratio,
                        "old_costs": dict(old["costs"]),
                        "new_costs": (
                            None if new is None else dict(new["costs"])
                        ),
                        "current_execution_id": (
                            None if new is None else new["execution_id"]
                        ),
                        "status": (
                            "compared" if new is not None else "unavailable"
                        ),
                    }
                )
    return rows


def _render_plot(
    *,
    representation: str,
    old_cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
    current_cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
    destination: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.75,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.70, 4.95), dpi=300)
    for index, (regime, title, _abbreviation) in enumerate(REGIMES):
        ax = axes.flat[index]
        values: list[float] = []
        missing_plateau = False
        for source_name, policy in (
            ("old", "no_insertion"),
            ("new", "no_insertion"),
            ("old", "plateau"),
            ("new", "plateau"),
        ):
            key = (representation, regime, policy)
            cell = (
                old_cells.get(key)
                if source_name == "old"
                else current_cells.get(key)
            )
            if cell is None:
                if source_name == "new" and policy == "plateau":
                    missing_plateau = True
                continue
            points = _sequence(
                cell.get("points"),
                label=f"{key} {source_name} points",
            )
            x = [int(_mapping(row, label="plot point")["k"]) for row in points]
            y = [
                max(
                    _finite(
                        _mapping(row, label="plot point")["error"],
                        label="plot error",
                    ),
                    1.0e-14,
                )
                for row in points
            ]
            values.extend(y)
            style = CURVE_STYLES[(source_name, policy)]
            ax.plot(
                x,
                y,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                zorder=2 if source_name == "new" else 1,
            )
            ax.scatter(
                [x[-1]],
                [y[-1]],
                s=20,
                marker=style["marker"],
                facecolor=(
                    style["color"] if source_name == "new" else "white"
                ),
                edgecolor=style["color"],
                linewidth=0.75,
                zorder=4,
            )
        ax.set_yscale("log")
        ax.set_xlim(1, 50)
        if values:
            lower = max(min(values) / 2.5, 1.0e-14)
            upper = max(values) * 2.5
            ax.set_ylim(lower, upper)
        ax.set_title(title.replace("--", "\N{EN DASH}"), fontsize=8.0, pad=2.0)
        ax.grid(which="major", color="#D9D9D9", linewidth=0.45, alpha=0.8)
        ax.grid(which="minor", color="#EEEEEE", linewidth=0.3, alpha=0.6)
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", labelsize=6.4, length=2.3)
        ax.set_xticks((1, 10, 20, 30, 40, 50))
        if index // 3 == 1:
            ax.set_xlabel("ADAPT iteration", fontsize=7.0)
        if index % 3 == 0:
            ax.set_ylabel(
                r"$|E_k-E_{\mathrm{ED}}^{(n_{\mathrm{ph}})}|$",
                fontsize=7.0,
            )
        if missing_plateau:
            ax.text(
                0.98,
                0.04,
                "stationary plateau unavailable",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.6,
                color="#9A3D3D",
            )
    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markerfacecolor=(
                style["color"] if source_name == "new" else "white"
            ),
            markeredgecolor=style["color"],
            markersize=4.2,
            label=style["label"],
        )
        for (source_name, _policy), style in CURVE_STYLES.items()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=7.0,
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.4,
        handlelength=2.6,
    )
    fig.tight_layout(rect=(0.01, 0.00, 0.99, 0.925), h_pad=0.95, w_pad=0.8)
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _table_tex(rows: Sequence[Mapping[str, Any]], *, representation: str) -> str:
    selected = [row for row in rows if row["representation"] == representation]
    lines = [
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        (
            r"Regime & Policy & "
            r"$\epsilon_{50}^{\rm Paper\,I}$ & "
            r"$\epsilon_{50}^{\rm stat}$ & "
            r"$\log_{10}(\epsilon_{\rm stat}/\epsilon_{\rm Paper\,I})$ & "
            r"$C_{50}^{\rm Paper\,I}$ & $C_{50}^{\rm stat}$ \\"
        ),
        r"\midrule",
    ]
    previous_regime: str | None = None
    for row in selected:
        regime = str(row["regime"])
        if previous_regime is not None and regime != previous_regime:
            lines.append(r"\addlinespace[0.3ex]")
        previous_regime = regime
        policy_label = (
            "none" if row["policy"] == "no_insertion" else "plateau"
        )
        lines.append(
            " & ".join(
                (
                    _tex_escape(row["abbreviation"]),
                    _tex_escape(policy_label),
                    _error_tex(float(row["old_terminal_error"])),
                    _error_tex(
                        None
                        if row["new_terminal_error"] is None
                        else float(row["new_terminal_error"])
                    ),
                    _delta_tex(
                        None
                        if row["log10_new_over_old"] is None
                        else float(row["log10_new_over_old"])
                    ),
                    _cost_tuple_tex(
                        _mapping(row["old_costs"], label="old costs")
                    ),
                    _cost_tuple_tex(
                        None
                        if row["new_costs"] is None
                        else _mapping(row["new_costs"], label="new costs")
                    ),
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _write_tex(
    *,
    rows: Sequence[Mapping[str, Any]],
    macro_plot: Path,
    singleton_plot: Path,
    current_source: Mapping[str, Any],
) -> Path:
    tex = OUTPUT_DIR / f"{STEM}.tex"
    manifest = _mapping(
        current_source.get("parameter_manifest"),
        label="current parameter manifest",
    )
    regimes = _sequence(
        manifest.get("regimes"),
        label="current parameter regimes",
    )
    regime_text = "; ".join(
        (
            f"{str(_mapping(row, label='manifest regime')['regime_id'])}: "
            f"U={_mapping(row, label='manifest regime')['u']}, "
            f"g={_mapping(row, label='manifest regime')['g_ep']}, "
            f"nph={_mapping(row, label='manifest regime')['n_ph_max']}"
        )
        for row in regimes
    )
    missing = [
        str(row["current_execution_id"] or (
            f"{row['regime']}:{row['representation']}:{row['policy']}"
        ))
        for row in rows
        if row["status"] == "unavailable"
    ]
    page_template = r"""
\begin{center}
{\large\bfseries %s}\\[-0.2ex]
{\fontsize{7.2}{8.2}\selectfont Same-cutoff absolute energy error; 50 controller rounds.}
\end{center}
\vspace{0.25ex}
%s
\vspace{0.35ex}
\begin{center}
\includegraphics[width=0.995\textwidth,height=4.52in,keepaspectratio]{%s}
\end{center}
\vspace{-1.0ex}
{\fontsize{6.0}{6.7}\selectfont
%s
}
\vfill
{\fontsize{5.65}{6.35}\selectfont
$C_{50}=(N_{2q},D_{2q},D_c,W_{1q},S_{\rm alg})$; $S_{\rm alg}$ uses
X.YeZ notation. Positive log ratios mean the stationary-source route has a
larger terminal error. End markers are terminal $k=50$ points, not promoted
plateau costs. RA-always and conventional ADAPT are excluded.
}
"""
    manifest_block = (
        r"\fcolorbox{black!35}{black!2}{\begin{minipage}{0.975\textwidth}"
        r"\raggedright\fontsize{6.15}{7.0}\selectfont "
        r"\textbf{Parameter and provenance manifest.} "
        + _tex_escape(
            "Hubbard--Holstein; L=2; open boundary; half-filled sector; "
            "binary bosons; same-cutoff exact diagonalization; "
            f"horizon={manifest['horizon']}; optimizer="
            f"{manifest['optimizer']}-{manifest['optimizer_maxiter']}; "
            f"seed={_mapping(manifest['seeds'], label='seeds')['adapt']}. "
            "Comparison: source-locked Paper-I route versus "
            f"{manifest['active_gradient_policy']}."
        )
        + r"\par "
        + _tex_escape(regime_text)
        + r"\par "
        + _tex_escape(
            "Current validated comparison cells="
            f"{current_source['comparison_cell_count']}/24; unavailable="
            f"{len(missing)}. This is a diagnostic comparison, not evidence "
            "promotion."
        )
        + r"\end{minipage}}"
    )
    second_manifest = (
        r"{\fontsize{6.15}{7.0}\selectfont "
        r"Same parameter matrix and source locks as page 1. "
        + (
            _tex_escape(
                f"{len(missing)} stationary plateau cells are unavailable; "
                "the exact matrix positions and source statuses are recorded "
                "in the provenance sidecar."
            )
            if missing
            else r"All stationary comparison cells are available."
        )
        + r"}"
    )
    body = (
        r"\documentclass[letterpaper]{article}" "\n"
        r"\usepackage[margin=0.24in]{geometry}" "\n"
        r"\usepackage{booktabs}" "\n"
        r"\usepackage{graphicx}" "\n"
        r"\usepackage{xcolor}" "\n"
        r"\pagestyle{empty}" "\n"
        r"\setlength{\parindent}{0pt}" "\n"
        r"\setlength{\tabcolsep}{1.45pt}" "\n"
        r"\begin{document}" "\n"
        + (
            page_template
            % (
                "Macro-generator: Paper-I route vs stationary source",
                manifest_block,
                _tex_escape(macro_plot.name),
                _table_tex(rows, representation="macro"),
            )
        )
        + r"\clearpage"
        + "\n"
        + (
            page_template
            % (
                "Single-Pauli-word: Paper-I route vs stationary source",
                second_manifest,
                _tex_escape(singleton_plot.name),
                _table_tex(rows, representation="singleton"),
            )
        )
        + r"\end{document}"
        + "\n"
    )
    tex.write_text(body, encoding="utf-8")
    return tex


def _compile_tex(tex: Path) -> tuple[Path, dict[str, Any]]:
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    build_dir = REPO_ROOT / "tmp" / "pdfs" / tex.stem
    build_dir.mkdir(parents=True, exist_ok=True)
    if latexmk:
        command = [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={build_dir}",
            tex.name,
        ]
    elif pdflatex:
        command = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={build_dir}",
            tex.name,
        ]
    else:
        raise RuntimeError("latexmk or pdflatex is required")
    completed = subprocess.run(
        command,
        cwd=tex.parent,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "FORCE_SOURCE_DATE": "1",
            "SOURCE_DATE_EPOCH": "1785283200",
            "TZ": "UTC",
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "LaTeX build failed:\n"
            + completed.stdout[-5000:]
            + completed.stderr[-5000:]
        )
    compiled = build_dir / f"{tex.stem}.pdf"
    if not compiled.is_file():
        raise RuntimeError("LaTeX completed without a PDF")
    destination = tex.with_suffix(".pdf")
    shutil.copy2(compiled, destination)
    log = build_dir / f"{tex.stem}.log"
    log_text = log.read_text(encoding="utf-8", errors="replace")
    return destination, {
        "engine": Path(command[0]).name,
        "returncode": completed.returncode,
        "overfull_hbox_count": log_text.count("Overfull \\hbox"),
        "underfull_hbox_count": log_text.count("Underfull \\hbox"),
        "fatal_error_present": "!  ==> Fatal error occurred" in log_text,
    }


def build() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    old_cells, old_sources = _old_cells()
    current_cells, current_source = _current_cells()
    rows = _comparison_rows(old_cells, current_cells)
    macro_plot = OUTPUT_DIR / f"{STEM}_macro_plot.png"
    singleton_plot = OUTPUT_DIR / f"{STEM}_singleton_plot.png"
    _render_plot(
        representation="macro",
        old_cells=old_cells,
        current_cells=current_cells,
        destination=macro_plot,
    )
    _render_plot(
        representation="singleton",
        old_cells=old_cells,
        current_cells=current_cells,
        destination=singleton_plot,
    )
    tex = _write_tex(
        rows=rows,
        macro_plot=macro_plot,
        singleton_plot=singleton_plot,
        current_source=current_source,
    )
    pdf, latex_validation = _compile_tex(tex)
    if pdf.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("generated file does not have a PDF header")
    try:
        from pypdf import PdfReader

        page_count = len(PdfReader(str(pdf)).pages)
    except Exception as exc:
        raise RuntimeError(
            f"generated PDF structural read failed: {exc}"
        ) from exc
    if page_count != 2:
        raise RuntimeError(
            f"generated comparison PDF has {page_count} pages, expected 2"
        )
    unavailable = [
        {
            "representation": row["representation"],
            "regime": row["regime"],
            "policy": row["policy"],
        }
        for row in rows
        if row["status"] == "unavailable"
    ]
    provenance_path = OUTPUT_DIR / f"{STEM}_provenance.json"
    provenance = {
        "schema": "paper_i_stationary_vs_source_locked_route_comparison_v1",
        "status": "diagnostic_comparison_not_paper_evidence",
        "paper_evidence_adopted": False,
        "metric": "same_cutoff_absolute_energy_error",
        "controller_round_domain": [1, 50],
        "terminal_comparison_round": 50,
        "representations": list(REPRESENTATIONS),
        "policies": list(POLICIES),
        "excluded_methods": {
            "ra_always": (
                "excluded because the corrected commutation-reduced "
                "replacement is pending"
            ),
            "conventional_append_adapt": (
                "excluded because this report compares RA route changes"
            ),
        },
        "old_route_sources": old_sources,
        "stationary_source": current_source,
        "comparison_row_count": len(rows),
        "compared_row_count": sum(
            row["status"] == "compared" for row in rows
        ),
        "unavailable_row_count": len(unavailable),
        "unavailable_rows": unavailable,
        "rows": rows,
        "cost_tuple": {
            "fields": list(COST_FIELDS),
            "controller_round": 50,
            "s_alg_display_notation": "X.YeZ_two_significant_digits",
        },
        "limitations": [
            (
                "Four stationary plateau attempts ended as retained failures "
                "and are shown as unavailable; they are not inferred from "
                "scheduler completion."
            ),
            (
                "The comparison is diagnostic and does not promote, replace, "
                "or demote Paper-I evidence."
            ),
            (
                "RA-always is intentionally absent until the corrected "
                "commutation-reduced replacement finishes."
            ),
        ],
        "validation": {
            "visual_inspection_performed": False,
            "page_count": page_count,
            "pdf_header_valid": True,
            "latex": latex_validation,
        },
        "outputs": {
            "pdf": {
                "path": str(pdf),
                "sha256": _sha256_file(pdf),
                "size_bytes": pdf.stat().st_size,
            },
            "tex": {
                "path": str(tex),
                "sha256": _sha256_file(tex),
            },
            "macro_plot_png": {
                "path": str(macro_plot),
                "sha256": _sha256_file(macro_plot),
            },
            "singleton_plot_png": {
                "path": str(singleton_plot),
                "sha256": _sha256_file(singleton_plot),
            },
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pdf, provenance_path


def main() -> int:
    try:
        pdf, provenance = build()
    except (ComparisonInputError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
    print(
        json.dumps(
            {
                "status": "passed",
                "pdf": str(pdf),
                "pdf_sha256": _sha256_file(pdf),
                "provenance": str(provenance),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
