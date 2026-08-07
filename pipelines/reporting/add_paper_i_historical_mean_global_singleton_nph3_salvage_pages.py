#!/usr/bin/env python3
"""Append the three recovered nph3 historical-mean RA trajectory pages.

The CHTC jobs completed 50 controller rounds, but their original workers lost
the complete result directories while publishing from /tmp to Condor scratch
across filesystems.  Scheduler stdout still contains one accepted-energy event
per round.  This reporter admits only that log-derived trajectory, marks the
missing result/checkpoint/ledger and cost data explicitly, and never promotes
the observations to Paper-I evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from pipelines.reporting import (
    add_paper_i_append_r70_singleton_progress_page as report_support,
)


PACKAGE_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau6_"
    "r50_20260801_v1_chtc"
)
CLUSTER_ID = 9_400_252
ADAPTER_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_nph3_"
    "stdout_salvage_adapter_v1"
)
PROVENANCE_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_nph3_"
    "stdout_salvage_pages_v1"
)
REGIMES = (
    ("weak_weak", "Weak--weak", 0),
    ("intermediate_weak", "Intermediate--weak", 1),
    ("strong_weak_u8", "Strong--weak", 2),
)
PAGE_IDS = {
    regime: f"historical_mean_global_singleton_{regime}_stdout_salvage_v1"
    for regime, _label, _proc in REGIMES
}
SAFE_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
LIMITATION = (
    "Pages 7--9 show scheduler-stdout-recovered 50-round nph3 trajectories "
    "for the historical-mean global-singleton RA plateau route. The complete "
    "result, checkpoint, estimator ledger, and compiled cost data were lost "
    "in the original EXDEV publication failure; these pages are diagnostic "
    "and are not adopted Paper-I evidence."
)


class SalvagePageError(ValueError):
    """Raised when the recovered sources cannot support the report pages."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "sha256" in result:
        raise SalvagePageError("self-digest input already contains sha256")
    result["sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    observed = value.get("sha256")
    unsigned = copy.deepcopy(dict(value))
    unsigned.pop("sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if observed != expected:
        raise SalvagePageError(f"{label} self-digest drifted")
    return str(observed)


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SalvagePageError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise SalvagePageError(f"{label} must be a JSON object")
    return value


def finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SalvagePageError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SalvagePageError(f"{label} must be finite")
    return result


def parse_scheduler_stdout(path: Path) -> list[dict[str, Any]]:
    """Return the exact 50 accepted-energy rows from one scheduler stdout."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise SalvagePageError(f"scheduler stdout is unreadable: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.startswith("AI_LOG "):
            continue
        try:
            event = json.loads(line[len("AI_LOG ") :])
        except json.JSONDecodeError as exc:
            raise SalvagePageError("scheduler stdout contains malformed AI_LOG") from exc
        if not isinstance(event, dict) or event.get("event") != "hardcoded_adapt_iter":
            continue
        depth = event.get("depth")
        position = event.get("selected_position")
        if (
            isinstance(depth, bool)
            or not isinstance(depth, int)
            or isinstance(position, bool)
            or not isinstance(position, int)
            or depth < 1
            or position < 0
            or position >= depth
        ):
            raise SalvagePageError("iteration depth/position telemetry is invalid")
        rows.append(
            {
                "round": depth,
                "energy": finite(event.get("energy"), label="iteration energy"),
                "selected_position": position,
                "best_op": str(event.get("best_op", "")),
                "max_grad": finite(event.get("max_grad"), label="max_grad"),
            }
        )
    if [row["round"] for row in rows] != list(range(1, 51)):
        raise SalvagePageError("scheduler stdout does not contain exactly rounds 1..50")
    return rows


def execution_id(regime: str) -> str:
    return (
        f"historical_mean_global_singleton_v1_r50__{regime}__nph3__"
        "ra_global_singleton_plateau"
    )


def stdout_name(regime: str, proc: int) -> str:
    return f"{CLUSTER_ID}.{proc}__{execution_id(regime)}.out"


def _salvage_log_root(salvage_root: Path) -> Path:
    candidates = list(salvage_root.rglob(f"{CLUSTER_ID}.0__*.out"))
    if len(candidates) != 1:
        raise SalvagePageError("salvage root does not resolve one proc-0 stdout")
    root = candidates[0].parent
    if any(not (root / stdout_name(regime, proc)).is_file() for regime, _label, proc in REGIMES):
        raise SalvagePageError("one or more recovered scheduler stdout files are missing")
    return root


def build_adapter(
    *, package_dir: Path, salvage_root: Path, receipt_path: Path, output: Path
) -> dict[str, Any]:
    manifest = load_json(package_dir / "package_manifest.json", label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("package_id") != PACKAGE_ID
        or manifest.get("row_count") != 6
        or manifest.get("status") != "passed_inert_six_rows"
    ):
        raise SalvagePageError("source package identity or closure drifted")

    receipt = load_json(receipt_path, label="retrieval receipt")
    retrieval = receipt.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise SalvagePageError("retrieval receipt payload is missing")
    if (
        receipt.get("original_cluster") != CLUSTER_ID
        or retrieval.get("completed_scientific_procs") != [0, 1, 2]
        or retrieval.get("complete_result_bundles_recovered") is not False
        or retrieval.get("gzip_integrity") != "passed"
        or retrieval.get("tar_readability") != "passed"
    ):
        raise SalvagePageError("retrieval receipt semantics drifted")
    trajectory_receipts = retrieval.get("scientific_trajectory_validation")
    if not isinstance(trajectory_receipts, list) or len(trajectory_receipts) != 3:
        raise SalvagePageError("retrieval receipt trajectory closure drifted")
    receipt_by_regime = {
        str(item.get("regime")): item
        for item in trajectory_receipts
        if isinstance(item, Mapping)
    }
    log_root = _salvage_log_root(salvage_root)

    cells: list[dict[str, Any]] = []
    for regime, display_name, proc in REGIMES:
        eid = execution_id(regime)
        job_path = package_dir / "jobs" / f"{eid}.json"
        protocol_path = package_dir / "protocols" / f"{eid}.json"
        job = load_json(job_path, label=f"{regime} job")
        protocol = load_json(protocol_path, label=f"{regime} protocol")
        verify_self_digest(job, label=f"{regime} job")
        verify_self_digest(protocol, label=f"{regime} protocol")
        if (
            job.get("execution_id") != eid
            or job.get("regime_id") != regime
            or job.get("nph") != 3
            or job.get("target_horizon") != 50
            or job.get("phase_i_shortlist_size") != 24
            or job.get("phase_ii_shortlist_size") != 12
            or job.get("phase_iii_admission_cardinality") != 1
            or job.get("plateau_prior_mean_decrease_ratio_threshold") != 0.0001
            or job.get("active_gradient_policy") != "stationary_source_response_v1"
            or job.get("resource_weighting_scope") != "all_phase_resource_weighting_v1"
            or report_support._sha256_file(protocol_path)
            != job.get("protocol_file_sha256")
        ):
            raise SalvagePageError(f"{regime} job/protocol contract drifted")
        exact = finite(job.get("exact_same_cutoff_energy"), label=f"{regime} exact energy")
        log_path = log_root / stdout_name(regime, proc)
        rows = parse_scheduler_stdout(log_path)
        for row in rows:
            row["delta_e"] = abs(float(row["energy"]) - exact)
            row["placement"] = (
                "append" if row["selected_position"] == row["round"] - 1 else "interior"
            )
        interior_rounds = [row["round"] for row in rows if row["placement"] == "interior"]
        terminal = rows[-1]
        receipt_row = receipt_by_regime.get(regime)
        if (
            not isinstance(receipt_row, Mapping)
            or receipt_row.get("proc") != proc
            or receipt_row.get("controller_rounds") != 50
            or not math.isclose(
                finite(receipt_row.get("final_energy"), label="receipt final energy"),
                float(terminal["energy"]),
                rel_tol=0.0,
                abs_tol=5e-16,
            )
        ):
            raise SalvagePageError(f"{regime} receipt/log terminal mismatch")
        cells.append(
            {
                "regime_id": regime,
                "display_name": display_name,
                "proc": proc,
                "execution_id": eid,
                "nph": 3,
                "exact_same_cutoff_energy": exact,
                "points": rows,
                "point_count": 50,
                "initial_round_zero_available": False,
                "marker_policy": "terminal_observed_point_no_serialized_plateau_prefix",
                "marker_round": 50,
                "terminal": copy.deepcopy(terminal),
                "interior_placement_count": len(interior_rounds),
                "append_placement_count": 50 - len(interior_rounds),
                "first_interior_round": interior_rounds[0] if interior_rounds else None,
                "source": {
                    "scheduler_stdout": report_support._file_binding(log_path),
                    "job": report_support._file_binding(job_path),
                    "job_canonical_sha256": job["sha256"],
                    "protocol": report_support._file_binding(protocol_path),
                    "protocol_canonical_sha256": protocol["sha256"],
                },
            }
        )

    adapter = digested(
        {
            "schema": ADAPTER_SCHEMA,
            "status": "passed_log_salvage_three_nph3_cells",
            "classification": "diagnostic_stdout_salvage_not_adopted_evidence",
            "package_id": PACKAGE_ID,
            "cluster_id": CLUSTER_ID,
            "route": {
                "candidate_supply": "global_guarded_singleton_pool_v1",
                "observed_available_candidate_count": 948,
                "shortlist_cardinality": {"phase_i": 24, "phase_ii": 12, "phase_iii": 1},
                "active_gradient_policy": "stationary_source_response_v1",
                "resource_weighting_scope": "all_phase_resource_weighting_v1",
                "insertion_policy": "plateau_commutation_v2_historical_mean",
                "threshold": 0.0001,
                "comparison": "marginal_to_prior_mean_strictly_below_v2",
            },
            "source_package_manifest": {
                **report_support._file_binding(package_dir / "package_manifest.json"),
                "canonical_sha256": manifest["sha256"],
            },
            "retrieval_receipt": report_support._file_binding(receipt_path),
            "limitations": [LIMITATION],
            "cells": cells,
        }
    )
    report_support._atomic_write_json(output, adapter)
    return adapter


def _format_scientific(value: float) -> str:
    coefficient, exponent = f"{value:.2e}".split("e")
    return rf"${coefficient}\mathord{{\times}}10^{{{int(exponent)}}}$"


def _plot_cell(cell: Mapping[str, Any], *, png: Path, pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    points = cell["points"]
    x = [int(point["round"]) for point in points]
    y = [max(float(point["delta_e"]), 1e-16) for point in points]
    color = "#54A24B"
    fig, ax = plt.subplots(figsize=(7.15, 4.25))
    ax.plot(x, y, color=color, linewidth=2.0)
    ax.scatter([x[-1]], [y[-1]], color=color, marker="D", s=56, zorder=4)
    ax.set_yscale("log")
    ax.set_xlim(0, 50.8)
    ax.set_xticks(range(0, 51, 5))
    ax.set_xlabel("RA-ADAPT controller round")
    ax.set_ylabel(r"Same-cutoff $|\Delta E|$")
    ax.grid(True, which="both", alpha=0.18, linewidth=0.55)
    ax.set_title(
        f"{cell['display_name']}, nph=3 - historical-mean global-singleton plateau",
        fontsize=10.2,
    )
    first = cell.get("first_interior_round")
    if isinstance(first, int):
        ax.axvline(first, color="#777777", linewidth=0.9, alpha=0.55)
        ax.text(
            first + 0.45,
            0.96,
            f"first interior placement: k={first}",
            transform=ax.get_xaxis_transform(),
            va="top",
            fontsize=7.8,
            color="#555555",
        )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.0,
                marker="D",
                markersize=6,
                label="RA plateau (diamond: terminal observed point)",
            )
        ],
        loc="upper right",
        fontsize=8,
        frameon=False,
    )
    terminal = cell["terminal"]
    ax.text(
        0.985,
        0.06,
        rf"$k=50$: $|\Delta E|={float(terminal['delta_e']):.3e}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#BBBBBB", "alpha": 0.9},
    )
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def _write_page_tex(cell: Mapping[str, Any], *, plot_pdf: Path, tex_path: Path) -> None:
    terminal = cell["terminal"]
    label = str(cell["display_name"])
    plot_argument = report_support._latex_escape(plot_pdf.resolve().as_posix())
    first = cell["first_interior_round"]
    direct = (
        f"The recovered trajectory reaches same-cutoff error "
        f"{float(terminal['delta_e']):.3e} at round 50. "
        f"Interior positions occur in {cell['interior_placement_count']}/50 rounds, "
        f"first at round {first}; the other {cell['append_placement_count']} rounds append."
    )
    tex = rf"""\documentclass[10pt,letterpaper]{{article}}
\usepackage[margin=0.42in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
{{\large\bfseries {label} historical-mean global-singleton plateau}}\\[-0.1ex]
{{\fontsize{{8}}{{9}}\selectfont
\textcolor{{red!65!black}}{{\textbf{{LOG-SALVAGED DIAGNOSTIC - NOT PAPER EVIDENCE}}}}}}
\end{{center}}
\vspace{{0.35ex}}
\fcolorbox{{black!35}}{{black!2}}{{%
\begin{{minipage}}{{0.965\textwidth}}
\fontsize{{7.1}}{{8.2}}\selectfont
\textbf{{Route.}} Hubbard--Holstein $L=2$, $n_{{\rm ph}}=3$; global guarded
singleton supply; stationary-source gradients; all-phase resource weighting
(including the Phase-I cost term); Powell-200; seed 7; 50 rounds. Candidate
funnel: 948 observed executable candidates $\to$ Phase I 24 $\to$ Phase II
12 $\to$ Phase III singleton admission 1.
\par
\textbf{{Insertion trigger.}} Historical marginal decrease divided by the
prior accepted-transition mean, strict threshold $10^{{-4}}$; when open,
logical positions are commutation reduced before deterministic selection.
\end{{minipage}}}}
\vspace{{0.25ex}}
\begin{{center}}
\includegraphics[width=0.965\textwidth,height=4.5in,keepaspectratio]{{{plot_argument}}}
\end{{center}}
\vspace{{-0.4ex}}
{{\fontsize{{7.2}}{{8.2}}\selectfont
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lrrrrrr}}
\toprule
Source & $k_T$ & $E_T$ & $E_{{\rm ED}}$ & $|\Delta E_T|$ & placements & first interior \\
\midrule
CHTC stdout 9400252.{cell['proc']} & 50 & {float(terminal['energy']):.15g} &
{float(cell['exact_same_cutoff_energy']):.15g} &
{_format_scientific(float(terminal['delta_e']))} &
{cell['interior_placement_count']} interior / {cell['append_placement_count']} append &
$k={first}$ \\
\bottomrule
\end{{tabular*}}
}}
\vspace{{0.7ex}}
{{\fontsize{{7.05}}{{8.2}}\selectfont
\textbf{{Observed outcome.}} {direct}
\par
\textbf{{Evidence boundary.}} The worker completed the scientific trajectory,
but its post-run cross-filesystem publication failed. Only scheduler energy,
selected-position, operator, and gradient telemetry survived. The result JSON,
checkpoint, estimator ledger, and compiled cost tuple are unavailable, so this
page neither supplies cost evidence nor enters the stationary-core matrix.
The plotted diamond is the terminal observed point because no serialized
effective-plateau prefix survived.
}}
\end{{document}}
"""
    tex_path.write_text(tex, encoding="utf-8")


def _assets(asset_dir: Path, stem: str, regime: str) -> dict[str, Path]:
    base = asset_dir / f"{stem}_{regime}"
    return {
        "plot_png": base.with_name(base.name + "_plot.png"),
        "plot_pdf": base.with_name(base.name + "_plot.pdf"),
        "page_tex": base.with_name(base.name + "_page.tex"),
        "page_pdf": base.with_name(base.name + "_page.pdf"),
    }


def append_pages(
    *,
    target_pdf: Path,
    target_provenance: Path,
    adapter_path: Path,
    asset_dir: Path,
    asset_stem: str,
) -> dict[str, Any]:
    if not SAFE_STEM.fullmatch(asset_stem) or asset_stem in {".", ".."}:
        raise SalvagePageError("asset_stem must be a safe filename component")
    adapter = load_json(adapter_path, label="salvage adapter")
    adapter_sha = verify_self_digest(adapter, label="salvage adapter")
    if adapter.get("schema") != ADAPTER_SCHEMA or len(adapter.get("cells", [])) != 3:
        raise SalvagePageError("salvage adapter schema or cell closure drifted")
    provenance = load_json(target_provenance, label="target provenance")
    outputs = provenance.get("outputs")
    layout = provenance.get("layout")
    if not isinstance(outputs, Mapping) or not isinstance(layout, Mapping):
        raise SalvagePageError("target provenance layout/output closure drifted")
    pdf_binding = outputs.get("partial_progress_pdf")
    if (
        not isinstance(pdf_binding, Mapping)
        or pdf_binding.get("sha256") != report_support._sha256_file(target_pdf)
    ):
        raise SalvagePageError("target PDF/provenance binding drifted")
    existing = provenance.get("historical_mean_global_singleton_nph3_salvage")
    if layout.get("page_count") == 9:
        if not isinstance(existing, Mapping):
            raise SalvagePageError("nine-page report lacks salvage provenance")
        binding = existing.get("adapter")
        if not isinstance(binding, Mapping) or binding.get("canonical_sha256") != adapter_sha:
            raise SalvagePageError("existing salvage pages bind a different adapter")
        return {
            "status": "already_current",
            "pages": 9,
            "output_pdf": str(target_pdf),
            "sha256": report_support._sha256_file(target_pdf),
        }
    if layout.get("page_count") != 6:
        raise SalvagePageError("target report must contain exactly six base pages")

    from pypdf import PdfReader, PdfWriter

    before_hashes = report_support._page_content_hashes(target_pdf)
    if len(before_hashes) != 6:
        raise SalvagePageError("target PDF is not physically six pages")
    asset_dir.mkdir(parents=True, exist_ok=True)
    built: dict[str, dict[str, Path]] = {}
    cells = adapter["cells"]
    for cell in cells:
        regime = str(cell["regime_id"])
        assets = _assets(asset_dir, asset_stem, regime)
        _plot_cell(cell, png=assets["plot_png"], pdf=assets["plot_pdf"])
        _write_page_tex(cell, plot_pdf=assets["plot_pdf"], tex_path=assets["page_tex"])
        report_support._compile_page(assets["page_tex"], assets["page_pdf"])
        if len(PdfReader(str(assets["page_pdf"]), strict=False).pages) != 1:
            raise SalvagePageError(f"{regime} page asset is not exactly one page")
        built[regime] = assets

    temporary_pdf = target_pdf.with_name(f".{target_pdf.name}.nph3-salvage.tmp")
    writer = PdfWriter()
    for page in PdfReader(str(target_pdf), strict=False).pages:
        writer.add_page(page)
    for cell in cells:
        regime = str(cell["regime_id"])
        writer.add_page(PdfReader(str(built[regime]["page_pdf"]), strict=False).pages[0])
    try:
        with temporary_pdf.open("wb") as stream:
            writer.write(stream)
        after_hashes = report_support._page_content_hashes(temporary_pdf)
        if len(after_hashes) != 9 or after_hashes[:6] != before_hashes:
            raise SalvagePageError("append changed existing pages or page closure")
        new_pdf_binding = report_support._file_binding(temporary_pdf)
        new_pdf_binding["path"] = str(target_pdf.resolve())
        updated = copy.deepcopy(provenance)
        updated["layout"]["page_count"] = 9
        for offset, cell in enumerate(cells, start=7):
            updated["layout"][f"page_{offset}"] = PAGE_IDS[str(cell["regime_id"])]
        updated["outputs"]["partial_progress_pdf"] = new_pdf_binding
        output_keys: list[str] = []
        for cell in cells:
            regime = str(cell["regime_id"])
            for asset_kind, path in built[regime].items():
                key = f"historical_mean_global_singleton_{regime}_{asset_kind}"
                updated["outputs"][key] = report_support._file_binding(path)
                output_keys.append(key)
        updated["historical_mean_global_singleton_nph3_salvage"] = {
            "schema": PROVENANCE_SCHEMA,
            "classification": "diagnostic_stdout_salvage_not_adopted_evidence",
            "page_ids": [PAGE_IDS[str(cell["regime_id"])] for cell in cells],
            "adapter": {
                **report_support._file_binding(adapter_path),
                "canonical_sha256": adapter_sha,
            },
            "cluster_id": CLUSTER_ID,
            "package_id": PACKAGE_ID,
            "cells": [
                {
                    "regime_id": cell["regime_id"],
                    "point_count": cell["point_count"],
                    "marker_round": cell["marker_round"],
                    "marker_policy": cell["marker_policy"],
                    "terminal": copy.deepcopy(cell["terminal"]),
                    "interior_placement_count": cell["interior_placement_count"],
                    "append_placement_count": cell["append_placement_count"],
                    "first_interior_round": cell["first_interior_round"],
                    "source": copy.deepcopy(cell["source"]),
                }
                for cell in cells
            ],
            "limitations": copy.deepcopy(adapter["limitations"]),
            "structural_validation": {
                "pages_before": 6,
                "pages_after": 9,
                "preserved_page_content_sha256": before_hashes,
                "new_page_content_sha256": after_hashes[6:],
            },
            "outputs": {
                key: copy.deepcopy(updated["outputs"][key]) for key in output_keys
            },
        }
        if LIMITATION not in updated.get("limitations", []):
            updated.setdefault("limitations", []).append(LIMITATION)
        os.replace(temporary_pdf, target_pdf)
        report_support._atomic_write_json(target_provenance, updated)
    finally:
        temporary_pdf.unlink(missing_ok=True)
    return {
        "status": "appended_three_salvage_pages",
        "pages": 9,
        "output_pdf": str(target_pdf),
        "output_provenance": str(target_provenance),
        "sha256": report_support._sha256_file(target_pdf),
        "preserved_pages_1_6": True,
    }


def remove_pages(*, target_pdf: Path, target_provenance: Path) -> dict[str, Any]:
    """Remove only this reporter's three pages and restore the six-page base."""

    from pypdf import PdfReader, PdfWriter

    provenance = load_json(target_provenance, label="target provenance")
    layout = provenance.get("layout")
    outputs = provenance.get("outputs")
    salvage = provenance.get("historical_mean_global_singleton_nph3_salvage")
    if (
        not isinstance(layout, Mapping)
        or not isinstance(outputs, Mapping)
        or not isinstance(salvage, Mapping)
        or layout.get("page_count") != 9
    ):
        raise SalvagePageError("target is not the nine-page salvage report")
    pdf_binding = outputs.get("partial_progress_pdf")
    if (
        not isinstance(pdf_binding, Mapping)
        or pdf_binding.get("sha256") != report_support._sha256_file(target_pdf)
    ):
        raise SalvagePageError("target PDF/provenance binding drifted")
    structural = salvage.get("structural_validation")
    if not isinstance(structural, Mapping):
        raise SalvagePageError("salvage structural validation is missing")
    expected_base_hashes = structural.get("preserved_page_content_sha256")
    before_hashes = report_support._page_content_hashes(target_pdf)
    if (
        not isinstance(expected_base_hashes, list)
        or len(expected_base_hashes) != 6
        or before_hashes[:5] != expected_base_hashes[:5]
        or len(before_hashes) != 9
    ):
        raise SalvagePageError("preserved six-page prefix drifted")
    # Page 6 is independently maintained by the R70 comparison updater.  It
    # may advance while these appended pages exist; preserve its current bytes
    # when its provenance identity is the supported replacement.
    if before_hashes[5] != expected_base_hashes[5] and (
        layout.get("page_6")
        != "ra_historical_average_vs_append_singleton_r70_progress_v1"
        or not isinstance(provenance.get("ra_append_singleton_r70_comparison"), Mapping)
    ):
        raise SalvagePageError("independently updated page 6 is unauthenticated")
    current_base_hashes = before_hashes[:6]

    temporary_pdf = target_pdf.with_name(f".{target_pdf.name}.remove-salvage.tmp")
    writer = PdfWriter()
    for page in PdfReader(str(target_pdf), strict=False).pages[:6]:
        writer.add_page(page)
    try:
        with temporary_pdf.open("wb") as stream:
            writer.write(stream)
        after_hashes = report_support._page_content_hashes(temporary_pdf)
        if after_hashes != current_base_hashes:
            raise SalvagePageError("six-page restoration changed preserved pages")
        new_pdf_binding = report_support._file_binding(temporary_pdf)
        new_pdf_binding["path"] = str(target_pdf.resolve())
        updated = copy.deepcopy(provenance)
        updated["layout"]["page_count"] = 6
        for page_number in (7, 8, 9):
            updated["layout"].pop(f"page_{page_number}", None)
        updated["outputs"]["partial_progress_pdf"] = new_pdf_binding
        for key in list(updated["outputs"]):
            if key.startswith("historical_mean_global_singleton_"):
                updated["outputs"].pop(key)
        updated.pop("historical_mean_global_singleton_nph3_salvage", None)
        updated["limitations"] = [
            item for item in updated.get("limitations", []) if item != LIMITATION
        ]
        os.replace(temporary_pdf, target_pdf)
        report_support._atomic_write_json(target_provenance, updated)
    finally:
        temporary_pdf.unlink(missing_ok=True)
    return {
        "status": "removed_three_noncompliant_salvage_pages",
        "pages": 6,
        "output_pdf": str(target_pdf),
        "output_provenance": str(target_provenance),
        "sha256": report_support._sha256_file(target_pdf),
        "preserved_pages_1_6": True,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--package-dir", type=Path, required=True)
    result.add_argument("--salvage-root", type=Path, required=True)
    result.add_argument("--retrieval-receipt", type=Path, required=True)
    result.add_argument("--adapter", type=Path, required=True)
    result.add_argument("--target-pdf", type=Path, required=True)
    result.add_argument("--target-provenance", type=Path, required=True)
    result.add_argument("--asset-dir", type=Path, required=True)
    result.add_argument("--asset-stem", required=True)
    result.add_argument("--remove-pages-only", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.remove_pages_only:
            result = remove_pages(
                target_pdf=args.target_pdf.resolve(),
                target_provenance=args.target_provenance.resolve(),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        build_adapter(
            package_dir=args.package_dir.resolve(),
            salvage_root=args.salvage_root.resolve(),
            receipt_path=args.retrieval_receipt.resolve(),
            output=args.adapter.resolve(),
        )
        result = append_pages(
            target_pdf=args.target_pdf.resolve(),
            target_provenance=args.target_provenance.resolve(),
            adapter_path=args.adapter.resolve(),
            asset_dir=args.asset_dir.resolve(),
            asset_stem=args.asset_stem,
        )
    except (OSError, RuntimeError, SalvagePageError, ValueError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
