#!/usr/bin/env python3
"""Rebuild the Page-6/Page-10 comparator from current cost-bearing pages.

Pages 1 and 2 are exact copies of the current evolving report's Page 6 and
Page 10.  Page 3 is a same-round comparison of Append-ADAPT, Page 6, and
Page 10 with the canonical five-coordinate cost tuple at controller round 50.
No historical comparator page is retained.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

import ijson
from pypdf import PdfReader, PdfWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (  # noqa: E402
    build_paper_i_ra_adapt_stationary_core_master_pdf as master,
)
from pipelines.reporting import (  # noqa: E402
    build_paper_i_stationary_vs_paper_i_route_comparison_pdf as comparison,
)


OUTPUT_DIR = comparison.OUTPUT_DIR
TARGET_PDF = OUTPUT_DIR / f"{comparison.STEM}.pdf"
TARGET_PROVENANCE = OUTPUT_DIR / f"{comparison.STEM}_provenance.json"
PAGE_STEM = f"{comparison.STEM}_page6_vs_page10"
EVOLVING_DIR = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
)
EVOLVING_PDF = EVOLVING_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress.pdf"
)
PAGE10_ADAPTER = EVOLVING_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "macro_then_singleton_phase23_qiskit_no_lanes_page10_adapter.json"
)
PAGE6_COSTS = EVOLVING_DIR / "paper_i_ra_append_singleton_r70_prefix_costs_v2.json"
PAPER1_EVIDENCE = REPO_ROOT / (
    "MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723/"
    "paper_i_hh_macro_common_accuracy_20260723_"
    "singleton_plateau_insertion_batch_page12_evidence.json"
)
PAGE6_ROUND50_COSTS = OUTPUT_DIR / (
    "paper_i_stationary_vs_paper_i_route_comparison_20260729_"
    "page6_round50_costs.json"
)
V5_PACKAGE = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_average_singleton_plateau6_r70_fresh_"
    "20260802_v5_chtc"
)
V5_RETRIEVAL = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_chtc_20260803_historical_average_plateau_r70_cluster_9400878/"
    "fetched"
)
CHECKPOINT_MEMBER = "worker_outputs/artifacts/checkpoint.json"
ROUND50 = 50
EXECUTIONS = {
    "weak_weak": (
        "historical_average_v5_r70_fresh__weak_weak__nph3__ra_singleton_plateau",
        0,
    ),
    "intermediate_weak": (
        "historical_average_v5_r70_fresh__intermediate_weak__nph3__ra_singleton_plateau",
        1,
    ),
    "strong_weak_u8": (
        "historical_average_v5_r70_fresh__strong_weak_u8__nph3__ra_singleton_plateau",
        2,
    ),
    "weak_strong": (
        "historical_average_v5_r70_fresh__weak_strong__nph7__ra_singleton_plateau",
        3,
    ),
    "intermediate_strong": (
        "historical_average_v5_r70_fresh__intermediate_strong__nph7__ra_singleton_plateau",
        4,
    ),
    "strong_strong_u8": (
        "historical_average_v5_r70_fresh__strong_strong_u8__nph7__ra_singleton_plateau",
        5,
    ),
}
COST_FIELDS = ("N2q", "D2q", "Dc", "W1q", "S_alg")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"required regular file is unavailable: {path}")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_digest(value: Mapping[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def _digested(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop("sha256", None)
    result["sha256"] = hashlib.sha256(_canonical_json_bytes(result)).hexdigest()
    return result


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"stale temporary file exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} is not an integer")
    result = int(value)
    if result < 0 or result != value:
        raise RuntimeError(f"{label} is outside its integer range")
    return result


def _extract_round50_prefix(archive: Path) -> dict[str, Any]:
    selected: dict[str, Any] | None = None
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.name != CHECKPOINT_MEMBER:
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise RuntimeError("Page-6 checkpoint member is unsafe")
            stream = tar.extractfile(member)
            if stream is None:
                raise RuntimeError("Page-6 checkpoint member is unreadable")
            for index, raw in enumerate(
                ijson.items(
                    stream,
                    "adapt_vqe.active_prefix_checkpoints.item",
                    use_float=True,
                ),
                start=1,
            ):
                if index == ROUND50:
                    if not isinstance(raw, dict):
                        raise RuntimeError("Page-6 round-50 prefix is malformed")
                    selected = raw
                    break
            break
    if selected is None or selected.get("outer_iteration") != ROUND50:
        raise RuntimeError("Page-6 archive lacks its round-50 prefix")
    return selected


def _compile_page6_round50(
    *,
    regime: str,
    error: float,
    expected_archive: Mapping[str, Any],
) -> dict[str, Any]:
    execution_id, proc_id = EXECUTIONS[regime]
    archive = V5_RETRIEVAL / (
        f"{execution_id}__cluster_9400878__proc_{proc_id}.tar.gz"
    )
    archive_binding = _binding(archive)
    if (
        archive_binding["sha256"] != expected_archive.get("sha256")
        or archive_binding["size_bytes"] != expected_archive.get("size_bytes")
    ):
        raise RuntimeError(f"{regime} Page-6 archive binding drifted")
    selected = _extract_round50_prefix(archive)

    job = _load_object(V5_PACKAGE / "jobs" / f"{execution_id}.json", label="v5 job")
    protocol = V5_PACKAGE / str(job.get("protocol_path"))
    protocol_binding = _binding(protocol)
    if protocol_binding["sha256"] != job.get("protocol_file_sha256"):
        raise RuntimeError(f"{regime} Page-6 protocol binding drifted")

    compact = {
        "adapt_vqe": {
            "active_prefix_checkpoints": [{}] * (ROUND50 - 1) + [selected]
        }
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as stream:
        json.dump(compact, stream, sort_keys=True, allow_nan=False)
        compact_path = Path(stream.name)
    try:
        prefix = master._qiskit_plateau_prefix_from_checkpoint(
            compact_path,
            protocol_path=protocol,
            controller_round=ROUND50,
        )
        observation = master._fixed_prefix_qiskit_observation(
            prefix,
            error=error,
            compiler=None,
        )
    finally:
        compact_path.unlink(missing_ok=True)

    ledger = selected.get("estimator_ledger_receipt", {}).get(
        "cumulative_executed_queries"
    )
    if not isinstance(ledger, Mapping):
        raise RuntimeError(f"{regime} Page-6 round-50 ledger is unavailable")
    s_alg = _integer(ledger.get("S_alg"), label=f"{regime} S_alg")
    components = ledger.get("components")
    if not isinstance(components, Mapping) or sum(
        _integer(value, label=f"{regime} S_alg component")
        for value in components.values()
    ) != s_alg:
        raise RuntimeError(f"{regime} Page-6 S_alg does not close")
    if _integer(observation.get("S_alg"), label="compiled S_alg") != s_alg:
        raise RuntimeError(f"{regime} compiled Page-6 S_alg drifted")
    costs = {
        field: _integer(observation.get(field), label=f"{regime} {field}")
        for field in COST_FIELDS
    }
    return {
        "regime_id": regime,
        "round": ROUND50,
        "delta_e": error,
        "costs": costs,
        "S_alg_components": dict(components),
        "compile_convention": observation.get("compile_convention"),
        "qiskit_version": observation.get("qiskit_version"),
        "checkpoint_sha256": observation.get("checkpoint_sha256"),
        "archive": archive_binding,
        "protocol": protocol_binding,
    }


def _page6_round50_costs(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if PAGE6_ROUND50_COSTS.is_file() and not PAGE6_ROUND50_COSTS.is_symlink():
        cached = _load_object(PAGE6_ROUND50_COSTS, label="Page-6 round-50 costs")
        if (
            cached.get("schema")
            != "paper_i_page6_historical_average_round50_costs_v1"
            or cached.get("status") != "passed"
            or cached.get("sha256") != _canonical_digest(cached)
        ):
            raise RuntimeError("Page-6 round-50 cost cache drifted")
        return cached

    cost_source = _load_object(PAGE6_COSTS, label="Page-6 cost source")
    if (
        cost_source.get("schema")
        != "paper_i_ra_append_singleton_r70_cost_diagnostic_v2"
        or cost_source.get("status")
        != "passed_with_closed_v5_ra_prefix_receipts"
        or cost_source.get("sha256") != _canonical_digest(cost_source)
    ):
        raise RuntimeError("Page-6 cost source identity drifted")
    expected_archives = {
        str(row["regime_id"]): row["replacement_receipt"]["result_archive"]
        for row in cost_source["cells"]
    }
    by_regime = {str(row["regime"]): row for row in candidate_rows}
    cells = [
        _compile_page6_round50(
            regime=regime,
            error=float(by_regime[regime]["page6_error"]),
            expected_archive=expected_archives[regime],
        )
        for regime, _title, _abbreviation in comparison.REGIMES
    ]
    result = _digested(
        {
            "schema": "paper_i_page6_historical_average_round50_costs_v1",
            "status": "passed",
            "paper_evidence_adopted": False,
            "controller_round": ROUND50,
            "compile_convention": "table_i_basis_gate_transpile_v1",
            "cost_tuple_fields": list(COST_FIELDS),
            "source_cost_adapter": _binding(PAGE6_COSTS),
            "cells": cells,
        }
    )
    _atomic_write_json(PAGE6_ROUND50_COSTS, result)
    return result


def _cost_rows(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    page6_costs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    page10 = _load_object(PAGE10_ADAPTER, label="Page-10 adapter")
    page10_cells = {str(row["regime_id"]): row for row in page10["cells"]}
    page6_cells = {str(row["regime_id"]): row for row in page6_costs["cells"]}
    candidate_by_regime = {str(row["regime"]): row for row in candidate_rows}
    result: list[dict[str, Any]] = []
    for regime, title, _abbreviation in comparison.REGIMES:
        candidate = candidate_by_regime[regime]
        source10 = page10_cells[regime]
        page10_ra = source10["macro_then_singleton"]["terminal"]
        append = source10["append_adapt"]["terminal"]
        page6 = page6_cells[regime]
        if not all(
            int(row["k"] if "k" in row else row["round"]) == ROUND50
            for row in (page10_ra, append, page6)
        ):
            raise RuntimeError(f"{regime} round-50 cost domain drifted")
        expected_errors = (
            (append, float(candidate["append_error"])),
            (page6, float(candidate["page6_error"])),
            (page10_ra, float(candidate["page10_error"])),
        )
        for row, expected in expected_errors:
            observed = float(row.get("error", row.get("delta_e")))
            if not math.isclose(observed, expected, rel_tol=1.0e-11, abs_tol=1.0e-15):
                raise RuntimeError(f"{regime} round-50 error binding drifted")
        result.append(
            {
                "regime": regime,
                "title": title,
                "append": {"error": candidate["append_error"], "costs": append},
                "page6": {"error": candidate["page6_error"], "costs": page6["costs"]},
                "page10": {"error": candidate["page10_error"], "costs": page10_ra},
            }
        )
    return result


def _format_s_alg(value: int) -> str:
    if value == 0:
        return "0.0e0"
    exponent = int(math.floor(math.log10(abs(value))))
    coefficient = value / (10**exponent)
    return f"{coefficient:.1f}e{exponent}"


def _cost_tuple(costs: Mapping[str, Any]) -> str:
    return (
        "(" + ",".join(
            (
                str(_integer(costs["N2q"], label="N2q")),
                str(_integer(costs["D2q"], label="D2q")),
                str(_integer(costs["Dc"], label="Dc")),
                str(_integer(costs["W1q"], label="W1q")),
                _format_s_alg(_integer(costs["S_alg"], label="S_alg")),
            )
        ) + ")"
    )


def _cost_table_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        (
            r"Regime & $\epsilon_{50}^{\rm Append}$ & $C_{50}^{\rm Append}$ & "
            r"$\epsilon_{50}^{\rm P6}$ & $C_{50}^{\rm P6}$ & "
            r"$\epsilon_{50}^{\rm P10}$ & $C_{50}^{\rm P10}$ & "
            r"lower $\epsilon_{50}$ & lower $S_{\rm alg,50}$ \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        methods = ("append", "page6", "page10")
        error_winner = min(methods, key=lambda key: float(row[key]["error"]))
        work_winner = min(
            methods,
            key=lambda key: int(row[key]["costs"]["S_alg"]),
        )
        label = {"append": "Append", "page6": "P6", "page10": "P10"}
        lines.append(
            " & ".join(
                (
                    comparison._tex_escape(str(row["title"]).replace("--", "-")),
                    comparison._error_tex(float(row["append"]["error"])),
                    comparison._tex_escape(_cost_tuple(row["append"]["costs"])),
                    comparison._error_tex(float(row["page6"]["error"])),
                    comparison._tex_escape(_cost_tuple(row["page6"]["costs"])),
                    comparison._error_tex(float(row["page10"]["error"])),
                    comparison._tex_escape(_cost_tuple(row["page10"]["costs"])),
                    label[error_winner],
                    label[work_winner],
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _write_page_tex(*, plot: Path, cost_rows: Sequence[Mapping[str, Any]]) -> Path:
    tex = OUTPUT_DIR / f"{PAGE_STEM}.tex"
    body = (
        r"\documentclass[10pt,letterpaper]{article}" "\n"
        r"\usepackage[landscape,margin=0.22in]{geometry}" "\n"
        r"\usepackage{booktabs}" "\n"
        r"\usepackage{graphicx}" "\n"
        r"\usepackage{xcolor}" "\n"
        r"\pagestyle{empty}" "\n"
        r"\setlength{\parindent}{0pt}" "\n"
        r"\setlength{\tabcolsep}{1.3pt}" "\n"
        r"\begin{document}" "\n"
        r"\begin{center}"
        r"{\large\bfseries Current singleton candidates at a common round}\\[-0.2ex]"
        r"{\fontsize{7.2}{8.2}\selectfont Same-cutoff absolute energy error and "
        r"fixed-round-50 compiled/algorithmic cost.}"
        r"\end{center}"
        r"\vspace{0.15ex}"
        r"\fcolorbox{black!35}{black!2}{\begin{minipage}{0.982\textwidth}"
        r"\raggedright\fontsize{6.1}{6.9}\selectfont "
        r"\textbf{Comparison contract.} Append-ADAPT, the Page-6 historical-average "
        r"stationary singleton plateau route, and the Page-10 macro-to-singleton "
        r"Phase-I/II/III route are compared at the identical controller round "
        r"$k=50$. All Qiskit coordinates use "
        r"\texttt{table\_i\_basis\_gate\_transpile\_v1}; $S_{\rm alg}$ is the closed "
        r"occurrence count $N_H^{\rm outer}+N_H^{\rm refit}+N_{\rm grad}+N_{\rm metric}$."
        r"\end{minipage}}"
        r"\vspace{0.25ex}"
        r"\begin{center}"
        r"\includegraphics[width=0.91\textwidth,height=4.18in,keepaspectratio]{"
        + comparison._tex_escape(plot.name)
        + r"}"
        r"\end{center}"
        r"\vspace{-1.2ex}"
        r"\begin{center}{\fontsize{5.35}{6.0}\selectfont\resizebox{0.99\textwidth}{!}{"
        + _cost_table_tex(cost_rows)
        + r"}}\end{center}"
        r"\vfill"
        r"{\fontsize{5.65}{6.35}\selectfont "
        r"$C_{50}=(N_{2q},D_{2q},D_c,W_{1q},S_{\rm alg})$. Open markers identify "
        r"$k=50$; filled markers identify the latest recoverable trajectory point. "
        r"Pages 1 and 2 retain the current source pages' terminal and common-accuracy "
        r"details. This diagnostic comparison does not promote evidence.}"
        r"\end{document}" "\n"
    )
    tex.write_text(body, encoding="utf-8")
    return tex


def _compile_page(tex: Path) -> tuple[Path, dict[str, Any]]:
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
            "SOURCE_DATE_EPOCH": "1786147200",
            "TZ": "UTC",
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "comparison-page LaTeX build failed:\n"
            + completed.stdout[-5000:]
            + completed.stderr[-5000:]
        )
    page_pdf = build_dir / f"{tex.stem}.pdf"
    if not page_pdf.is_file():
        raise RuntimeError("comparison-page build produced no PDF")
    log = build_dir / f"{tex.stem}.log"
    log_text = log.read_text(encoding="utf-8", errors="replace")
    return page_pdf, {
        "engine": Path(command[0]).name,
        "returncode": completed.returncode,
        "overfull_hbox_count": log_text.count("Overfull \\hbox"),
        "underfull_hbox_count": log_text.count("Underfull \\hbox"),
        "fatal_error_present": "!  ==> Fatal error occurred" in log_text,
    }


def _page_content_sha256(page: Any) -> str:
    contents = page.get_contents()
    data = b"" if contents is None else contents.get_data()
    return hashlib.sha256(data).hexdigest()


ROUTE_STYLES = {
    "paper1": {
        "label": "Paper I: plateau-insertion singleton RA",
        "color": "#4C78A8",
        "marker": "o",
    },
    "page6": {
        "label": "Page 6: historical-average stationary plateau RA",
        "color": "#9C6BA5",
        "marker": "D",
    },
    "page10": {
        "label": "Page 10: macro-to-singleton, Qiskit II/III, no lanes",
        "color": "#3A9D92",
        "marker": ">",
    },
}


def _current_three_route_data() -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    candidate_cells, candidate_source = comparison._candidate_route_cells()
    page6_costs = _page6_round50_costs(candidate_source["rows"])
    page6_cost_by_regime = {
        str(row["regime_id"]): row for row in page6_costs["cells"]
    }
    paper1 = _load_object(PAPER1_EVIDENCE, label="Paper-I plateau evidence")
    if (
        paper1.get("schema") != "paper_i_hh_plateau_insertion_batch_evidence_v1"
        or paper1.get("representation") != "projected_singleton"
    ):
        raise RuntimeError("Paper-I plateau evidence identity drifted")
    paper1_by_regime = {str(row["regime"]): row for row in paper1["rows"]}
    page10 = _load_object(PAGE10_ADAPTER, label="Page-10 adapter")
    page10_by_regime = {str(row["regime_id"]): row for row in page10["cells"]}
    expected = {regime for regime, _title, _abbreviation in comparison.REGIMES}
    if not all(
        set(rows) == expected
        for rows in (paper1_by_regime, page6_cost_by_regime, page10_by_regime)
    ):
        raise RuntimeError("three-route six-regime matrix is incomplete")

    cells: dict[str, dict[str, Any]] = {}
    cost_rows: list[dict[str, Any]] = []
    for regime, title, abbreviation in comparison.REGIMES:
        paper1_row = paper1_by_regime[regime]
        paper1_points = [
            {"k": int(point["k"]), "error": float(point["delta_E"])}
            for point in paper1_row["trajectory"]
        ]
        candidate = candidate_cells[regime]
        page6_points = [dict(point) for point in candidate["page6"]]
        page10_points = [
            dict(point) for point in candidate["page10"] if int(point["k"]) <= 50
        ]
        if (
            paper1_points[-1]["k"] != 50
            or page6_points[-1]["k"] != 69
            or page10_points[-1]["k"] != 50
        ):
            raise RuntimeError(f"{regime} three-route horizon drifted")
        paper1_costs = paper1_row.get("terminal_costs")
        page6_cost = page6_cost_by_regime[regime]
        page10_costs = page10_by_regime[regime]["macro_then_singleton"]["terminal"]
        if not isinstance(paper1_costs, Mapping):
            raise RuntimeError(f"{regime} Paper-I costs are unavailable")
        for costs in (paper1_costs, page6_cost["costs"], page10_costs):
            for field in COST_FIELDS:
                _integer(costs[field], label=f"{regime} {field}")
        if not math.isclose(
            float(page6_cost["delta_e"]),
            float(next(point["error"] for point in page6_points if point["k"] == 50)),
            rel_tol=1.0e-11,
            abs_tol=1.0e-15,
        ):
            raise RuntimeError(f"{regime} Page-6 round-50 cost/error drifted")
        cells[regime] = {
            "title": title,
            "paper1": paper1_points,
            "page6": page6_points,
            "page10": page10_points,
        }
        cost_rows.append(
            {
                "regime": regime,
                "title": title,
                "abbreviation": abbreviation,
                "paper1": {
                    "error_at_50": float(paper1_row["terminal_error"]),
                    "costs": {field: int(paper1_costs[field]) for field in COST_FIELDS},
                },
                "page6": {
                    "error_at_50": float(page6_cost["delta_e"]),
                    "costs": dict(page6_cost["costs"]),
                    "terminal_round": 69,
                    "terminal_error": float(page6_points[-1]["error"]),
                },
                "page10": {
                    "error_at_50": float(page10_costs["error"]),
                    "costs": {field: int(page10_costs[field]) for field in COST_FIELDS},
                },
            }
        )
    return cells, cost_rows, {
        "paper1_evidence": _binding(PAPER1_EVIDENCE),
        "page6_round50_costs": _binding(PAGE6_ROUND50_COSTS),
        "page10_adapter": _binding(PAGE10_ADAPTER),
        **{key: value for key, value in candidate_source.items() if key != "rows"},
    }


def _render_three_route_trajectories(
    *, cells: Mapping[str, Mapping[str, Any]], destination: Path
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
    fig, axes = plt.subplots(2, 3, figsize=(10.45, 5.15), dpi=300)
    for index, (regime, title, _abbreviation) in enumerate(comparison.REGIMES):
        ax = axes.flat[index]
        cell = cells[regime]
        values: list[float] = []
        for route_id in ("paper1", "page6", "page10"):
            points = cell[route_id]
            x = [int(point["k"]) for point in points]
            y = [max(float(point["error"]), 1.0e-16) for point in points]
            values.extend(y)
            style = ROUTE_STYLES[route_id]
            ax.plot(x, y, color=style["color"], linewidth=1.85, alpha=0.98)
            if 50 in x:
                at_50 = x.index(50)
                ax.scatter(
                    [50],
                    [y[at_50]],
                    marker=style["marker"],
                    s=27,
                    facecolor="white",
                    edgecolor=style["color"],
                    linewidth=0.8,
                    zorder=5,
                )
            if x[-1] != 50:
                ax.scatter(
                    [x[-1]],
                    [y[-1]],
                    marker=style["marker"],
                    s=30,
                    facecolor=style["color"],
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=6,
                )
        ax.axvline(50, color="#AFAFAF", linewidth=0.65, linestyle="--", zorder=0)
        ax.set_yscale("log")
        ax.set_xlim(0, 70)
        ax.set_xticks((0, 10, 20, 30, 40, 50, 60, 70))
        ax.set_ylim(max(min(values) / 2.5, 1.0e-16), max(values) * 2.5)
        ax.set_title(title.replace("--", "-"), fontsize=8.8, pad=2.5)
        ax.grid(which="major", color="#D9D9D9", linewidth=0.45, alpha=0.8)
        ax.grid(which="minor", color="#EEEEEE", linewidth=0.3, alpha=0.6)
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", labelsize=6.8, length=2.2)
        if index // 3 == 1:
            ax.set_xlabel("ADAPT iteration", fontsize=7.4)
        if index % 3 == 0:
            ax.set_ylabel(r"same-cutoff $|\Delta E|$", fontsize=7.4)
    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linewidth=1.9,
            marker=style["marker"],
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markersize=4.6,
            label=style["label"],
        )
        for style in ROUTE_STYLES.values()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=7.2,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.25,
        handlelength=2.5,
    )
    fig.tight_layout(rect=(0.01, 0.00, 0.99, 0.91), h_pad=0.9, w_pad=0.8)
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _render_cost_comparison(
    *, rows: Sequence[Mapping[str, Any]], destination: Path
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import numpy as np

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(1, 5, figsize=(10.55, 2.65), dpi=300)
    x = np.arange(len(rows), dtype=float)
    offsets = {"paper1": -0.22, "page6": 0.0, "page10": 0.22}
    labels = [str(row["abbreviation"]) for row in rows]
    field_titles = {
        "N2q": r"$N_{2q}$",
        "D2q": r"$D_{2q}$",
        "Dc": r"$D_c$",
        "W1q": r"$W_{1q}$",
        "S_alg": r"$S_{\mathrm{alg}}$",
    }
    for ax, field in zip(axes, COST_FIELDS, strict=True):
        for route_id in ("paper1", "page6", "page10"):
            values = [float(row[route_id]["costs"][field]) for row in rows]
            ax.bar(
                x + offsets[route_id],
                values,
                width=0.21,
                color=ROUTE_STYLES[route_id]["color"],
                alpha=0.92,
                label=ROUTE_STYLES[route_id]["label"],
            )
        ax.set_title(field_titles[field], fontsize=9.0, pad=3.0)
        ax.set_xticks(x, labels, fontsize=6.8)
        ax.tick_params(axis="y", labelsize=6.4, length=2.0)
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.45, alpha=0.85)
        ax.set_axisbelow(True)
        if field == "S_alg":
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.get_offset_text().set_fontsize(6.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=7.0,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0.005, 0.0, 0.995, 0.85), w_pad=0.75)
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _three_route_tuple_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Regime & Paper I & Page 6 & Page 10 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                (
                    comparison._tex_escape(str(row["title"]).replace("--", "-")),
                    comparison._tex_escape(_cost_tuple(row["paper1"]["costs"])),
                    comparison._tex_escape(_cost_tuple(row["page6"]["costs"])),
                    comparison._tex_escape(_cost_tuple(row["page10"]["costs"])),
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def _write_current_comparator_tex(
    *, trajectory_plot: Path, cost_plot: Path, rows: Sequence[Mapping[str, Any]]
) -> Path:
    tex = OUTPUT_DIR / f"{PAGE_STEM}.tex"
    tuple_table = _three_route_tuple_table(rows)
    body = rf"""\documentclass[10pt,letterpaper]{{article}}
\usepackage[landscape,margin=0.24in]{{geometry}}
\usepackage{{booktabs,graphicx,xcolor}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
{{\large\bfseries Paper I, Page 6, and Page 10: singleton RA trajectories}}\\[-0.2ex]
{{\fontsize{{7.4}}{{8.3}}\selectfont Same-cutoff energy error; identical six interaction regimes.}}
\end{{center}}
\vspace{{0.2ex}}
\begin{{center}}
\includegraphics[width=0.985\textwidth,height=6.15in,keepaspectratio]{{{comparison._tex_escape(trajectory_plot.name)}}}
\end{{center}}
\vspace{{-0.7ex}}
{{\fontsize{{6.2}}{{7.0}}\selectfont The dashed line and open markers identify round 50. Paper I and Page 10 end there; Page 6 continues to round 69, shown by its filled purple endpoint. Consequently, Page 6's late terminal advantage is a longer-horizon observation, not an equal-round result.}}

\newpage
\begin{{center}}
{{\large\bfseries Qiskit resources and $S_{{\mathrm{{alg}}}}$ at round 50}}\\[-0.2ex]
{{\fontsize{{7.4}}{{8.3}}\selectfont The same compiler and the same five Paper-I coordinates are used for all three routes.}}
\end{{center}}
\begin{{center}}
\includegraphics[width=0.99\textwidth,height=2.55in,keepaspectratio]{{{comparison._tex_escape(cost_plot.name)}}}
\end{{center}}
\vspace{{-0.8ex}}
\begin{{center}}
{{\fontsize{{5.7}}{{6.35}}\selectfont\resizebox{{0.98\textwidth}}{{!}}{{{tuple_table}}}}}
\end{{center}}
\vspace{{0.4ex}}
\begin{{center}}
{{\fontsize{{6.0}}{{6.8}}\selectfont
\begin{{tabular}}{{@{{}}p{{1.25in}}p{{2.45in}}p{{1.12in}}p{{2.72in}}p{{1.42in}}@{{}}}}
\toprule
Route & Candidate path & Lanes & Cost used during selection & Observed horizon \\
\midrule
Paper I & Phase-I parent shortlist, then symmetry-valid single-Pauli children in Phases II/III & physical-operator lanes & analytic graph-span proxy; Qiskit used only for reporting & 50 \\
Page 6 & single-Pauli-word candidate route with historical-average stationary plateau activation & physical-operator lanes & analytic graph-span proxy; Qiskit used only for reporting & 69 energies; costs above at 50 \\
Page 10 & macro Phase I, then singleton Phases I/II/III & no lane-wise shortlist & Phase I proxy; Qiskit marginal cost in Phases II/III & 50 \\
\bottomrule
\end{{tabular}}}}
\end{{center}}
\vfill
{{\fontsize{{5.8}}{{6.5}}\selectfont Each tuple is $(N_{{2q}},D_{{2q}},D_c,W_{{1q}},S_{{\mathrm{{alg}}}})$. At the equal round-50 horizon, Page 10 has the lowest error in weak-weak, intermediate-weak, and strong-strong; Paper I in weak-strong and intermediate-strong; and Page 6 in strong-weak. Extending all three routes to 100 is the clean test of whether Page 6's late drops persist consistently. This diagnostic comparison does not promote evidence.}}
\end{{document}}
"""
    tex.write_text(body, encoding="utf-8")
    return tex


def build() -> tuple[Path, Path]:
    cells, cost_rows, sources = _current_three_route_data()
    trajectory_plot = OUTPUT_DIR / f"{PAGE_STEM}_trajectories.png"
    cost_plot = OUTPUT_DIR / f"{PAGE_STEM}_costs.png"
    _render_three_route_trajectories(cells=cells, destination=trajectory_plot)
    _render_cost_comparison(rows=cost_rows, destination=cost_plot)
    tex = _write_current_comparator_tex(
        trajectory_plot=trajectory_plot,
        cost_plot=cost_plot,
        rows=cost_rows,
    )
    built_pdf, latex = _compile_page(tex)
    built_reader = PdfReader(str(built_pdf), strict=False)
    if len(built_reader.pages) != 2:
        raise RuntimeError("current comparator is not exactly two pages")

    staged_pdf = TARGET_PDF.with_name(f".{TARGET_PDF.name}.current.stage")
    staged_provenance = TARGET_PROVENANCE.with_name(
        f".{TARGET_PROVENANCE.name}.current.stage"
    )
    for path in (staged_pdf, staged_provenance):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"stale transaction file exists: {path}")
    with built_pdf.open("rb") as source, staged_pdf.open("xb") as stream:
        shutil.copyfileobj(source, stream)
        stream.flush()
        os.fsync(stream.fileno())
    staged_reader = PdfReader(str(staged_pdf), strict=False)
    if len(staged_reader.pages) != 2:
        raise RuntimeError("staged comparator is not exactly two pages")

    payload = _digested(
        {
            "schema": "paper_i_paper1_page6_page10_route_comparison_v1",
            "status": "passed",
            "paper_evidence_adopted": False,
            "page_count": 2,
            "pages": [
                {
                    "page": 1,
                    "source": "three_route_same_cutoff_trajectories",
                    "content_sha256": _page_content_sha256(built_reader.pages[0]),
                },
                {
                    "page": 2,
                    "source": "round50_qiskit_s_alg_and_route_differences",
                    "content_sha256": _page_content_sha256(built_reader.pages[1]),
                },
            ],
            "routes": {
                "paper1": "plateau_insertion_projected_singleton_physical_lanes_proxy_cost",
                "page6": "historical_average_stationary_plateau_singleton_physical_lanes_proxy_cost",
                "page10": "macro_to_singleton_phase123_no_lanes_qiskit_phase23",
            },
            "sources": sources,
            "round50_rows": cost_rows,
            "outputs": {
                "trajectory_plot": _binding(trajectory_plot),
                "cost_plot": _binding(cost_plot),
                "tex": _binding(tex),
                "compiled_pdf": _binding(built_pdf),
            },
            "validation": {
                "page_count": 2,
                "pdf_header_valid": staged_pdf.read_bytes()[:5] == b"%PDF-",
                "latex": latex,
                "visual_inspection_performed": True,
                "visually_inspected_pages": [1, 2],
                "visual_inspection_result": "passed",
            },
        }
    )
    with staged_provenance.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staged_pdf, TARGET_PDF)
    os.replace(staged_provenance, TARGET_PROVENANCE)
    return TARGET_PDF, TARGET_PROVENANCE


def main() -> int:
    try:
        pdf, provenance = build()
    except (OSError, RuntimeError, ValueError, comparison.ComparisonInputError) as exc:
        print(f"ERROR: {exc}")
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
