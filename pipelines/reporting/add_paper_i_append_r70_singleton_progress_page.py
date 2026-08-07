#!/usr/bin/env python3
"""Append the authenticated fresh Append-ADAPT singleton R70 page.

This module deliberately does not rebuild the five-page evolving report.  It
checks the existing report/provenance binding, builds one supplemental LaTeX
page from a self-digested reporting adapter, and appends that page while
proving that the five existing PDF content streams are unchanged.
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
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_SCHEMA = "paper_i_append_adapt_singleton_r70_progress_adapter_v1"
PAGE_ID = "append_adapt_singleton_fresh_r70_progress_v1"
REGIME_ORDER = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
COMPLETED_REGIMES = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "strong_strong_u8",
)
PENDING_REGIMES = ("weak_strong", "intermediate_strong")
REGIME_LABELS = {
    "weak_weak": "Weak--weak",
    "intermediate_weak": "Intermediate--weak",
    "strong_weak_u8": "Strong--weak",
    "weak_strong": "Weak--strong",
    "intermediate_strong": "Intermediate--strong",
    "strong_strong_u8": "Strong--strong",
}
COST_FIELDS = ("N2q", "D2q", "Dc", "W1q", "S_alg")
PAGE_LIMITATION = (
    "Page 6 is a supplemental fresh Append-ADAPT singleton R70 progress "
    "diagnostic; it does not enter the validated 48-cell stationary-core "
    "matrix and is not adopted Paper-I evidence."
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ASSET_STEM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class R70PageError(ValueError):
    """Raised when the page cannot be appended without provenance drift."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R70PageError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise R70PageError(f"{label} must be a JSON object")
    return value


def _verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    expected = value.get("sha256")
    unsigned = copy.deepcopy(dict(value))
    unsigned.pop("sha256", None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if expected != observed:
        raise R70PageError(f"{label} self-digest drifted")
    return observed


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise R70PageError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise R70PageError(f"{label} must be an array")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise R70PageError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R70PageError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and " if positive else ""
        raise R70PageError(f"{label} must be {qualifier}finite")
    return result


def _validate_costs(value: Any, *, label: str) -> dict[str, int]:
    costs = _mapping(value, label=label)
    return {
        field: _integer(costs.get(field), label=f"{label}.{field}")
        for field in COST_FIELDS
    }


def _validate_adapter(adapter: Mapping[str, Any]) -> dict[str, Any]:
    digest = _verify_self_digest(adapter, label="R70 adapter")
    if adapter.get("schema") != ADAPTER_SCHEMA:
        raise R70PageError("R70 adapter schema is unsupported")
    if adapter.get("status") != "passed":
        raise R70PageError("R70 adapter status is not passed")
    if tuple(_sequence(adapter.get("regime_order"), label="regime_order")) != REGIME_ORDER:
        raise R70PageError("R70 adapter regime_order drifted")
    if tuple(
        _sequence(adapter.get("completed_regimes"), label="completed_regimes")
    ) != COMPLETED_REGIMES:
        raise R70PageError("R70 adapter completed_regimes drifted")
    if tuple(
        _sequence(adapter.get("pending_regimes"), label="pending_regimes")
    ) != PENDING_REGIMES:
        raise R70PageError("R70 adapter pending_regimes drifted")
    limitations = _sequence(adapter.get("limitations", []), label="limitations")
    if not all(isinstance(item, str) and item for item in limitations):
        raise R70PageError("R70 adapter limitations must be nonempty strings")
    _mapping(
        adapter.get("source_authentication_summary", {}),
        label="source_authentication_summary",
    )

    policy = _mapping(adapter.get("cost_policy"), label="cost_policy")
    round_50_policy = _mapping(policy.get("round_50"), label="cost_policy.round_50")
    round_70_policy = _mapping(policy.get("round_70"), label="cost_policy.round_70")
    if round_50_policy.get("classification") != "canonical_paper_comparable":
        raise R70PageError("round-50 cost classification drifted")
    if round_70_policy.get("classification") != "diagnostic_extension":
        raise R70PageError("round-70 cost classification drifted")
    _mapping(adapter.get("same_cutoff_reference"), label="same_cutoff_reference")

    cells = _sequence(adapter.get("cells"), label="cells")
    if len(cells) != len(COMPLETED_REGIMES):
        raise R70PageError("R70 adapter must contain exactly four completed cells")
    normalized_cells: list[dict[str, Any]] = []
    for cell_index, raw_cell in enumerate(cells):
        cell = _mapping(raw_cell, label=f"cells[{cell_index}]")
        regime_id = cell.get("regime_id")
        if regime_id != COMPLETED_REGIMES[cell_index]:
            raise R70PageError("R70 adapter cells are not in declared regime order")
        if not isinstance(cell.get("display_name"), str) or not cell["display_name"]:
            raise R70PageError(f"{regime_id}: display_name is unavailable")
        _integer(cell.get("nph"), label=f"{regime_id}.nph", minimum=1)
        if not isinstance(cell.get("execution_id"), str) or not cell["execution_id"]:
            raise R70PageError(f"{regime_id}: execution_id is unavailable")
        _mapping(cell.get("source"), label=f"{regime_id}.source")

        points = _sequence(cell.get("points"), label=f"{regime_id}.points")
        if len(points) != 71:
            raise R70PageError(f"{regime_id}: expected points for rounds 0..70")
        normalized_points: list[dict[str, float | int]] = []
        for round_index, raw_point in enumerate(points):
            point = _mapping(raw_point, label=f"{regime_id}.points[{round_index}]")
            if _integer(
                point.get("round"),
                label=f"{regime_id}.points[{round_index}].round",
            ) != round_index:
                raise R70PageError(f"{regime_id}: point rounds are not exactly 0..70")
            normalized_points.append(
                {
                    "round": round_index,
                    "energy": _finite(
                        point.get("energy"),
                        label=f"{regime_id}.points[{round_index}].energy",
                    ),
                    "delta_e": _finite(
                        point.get("delta_e"),
                        label=f"{regime_id}.points[{round_index}].delta_e",
                        positive=True,
                    ),
                }
            )

        endpoints = _mapping(cell.get("endpoints"), label=f"{regime_id}.endpoints")
        normalized_endpoints: dict[str, dict[str, Any]] = {}
        for endpoint_round in (50, 70):
            endpoint_key = f"round_{endpoint_round}"
            endpoint = _mapping(
                endpoints.get(endpoint_key),
                label=f"{regime_id}.endpoints.{endpoint_key}",
            )
            if _integer(
                endpoint.get("round"),
                label=f"{regime_id}.{endpoint_key}.round",
            ) != endpoint_round:
                raise R70PageError(f"{regime_id}: {endpoint_key} round drifted")
            energy = _finite(
                endpoint.get("energy"), label=f"{regime_id}.{endpoint_key}.energy"
            )
            delta_e = _finite(
                endpoint.get("delta_e"),
                label=f"{regime_id}.{endpoint_key}.delta_e",
                positive=True,
            )
            checkpoint_sha256 = endpoint.get("checkpoint_sha256")
            if not isinstance(checkpoint_sha256, str) or not _SHA256_RE.fullmatch(
                checkpoint_sha256
            ):
                raise R70PageError(
                    f"{regime_id}: {endpoint_key} checkpoint digest is invalid"
                )
            _mapping(endpoint.get("compile"), label=f"{regime_id}.{endpoint_key}.compile")
            point = normalized_points[endpoint_round]
            if not math.isclose(energy, float(point["energy"]), rel_tol=0.0, abs_tol=1e-12):
                raise R70PageError(f"{regime_id}: {endpoint_key} energy mismatches points")
            if not math.isclose(
                delta_e, float(point["delta_e"]), rel_tol=0.0, abs_tol=1e-15
            ):
                raise R70PageError(f"{regime_id}: {endpoint_key} delta_e mismatches points")
            normalized_endpoints[endpoint_key] = {
                "round": endpoint_round,
                "energy": energy,
                "delta_e": delta_e,
                "checkpoint_sha256": checkpoint_sha256,
                "costs": _validate_costs(
                    endpoint.get("costs"), label=f"{regime_id}.{endpoint_key}.costs"
                ),
                "compile": copy.deepcopy(dict(endpoint["compile"])),
            }
        normalized_cells.append(
            {
                **copy.deepcopy(dict(cell)),
                "points": normalized_points,
                "endpoints": normalized_endpoints,
            }
        )
    return {
        **copy.deepcopy(dict(adapter)),
        "cells": normalized_cells,
        "sha256": digest,
    }


def _format_delta_e_tex(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\!\times\!10^{{{int(exponent)}}}$"


def _format_cost_tuple_tex(costs: Mapping[str, Any]) -> str:
    return (
        "$({N2q:,},{D2q:,},{Dc:,},{W1q:,},{S_alg:,})$".format(
            **{field: int(costs[field]) for field in COST_FIELDS}
        )
    )


def _latex_escape(value: str) -> str:
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
    return "".join(replacements.get(character, character) for character in value)


def _render_plot(
    adapter: Mapping[str, Any], *, png_path: Path, pdf_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

    cells = {str(cell["regime_id"]): cell for cell in adapter["cells"]}
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.3,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.1, 4.25), constrained_layout=True)
    for index, regime_id in enumerate(REGIME_ORDER):
        ax = axes.flat[index]
        ax.set_title(REGIME_LABELS[regime_id])
        ax.set_xlim(0, 70)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        if regime_id in cells:
            cell = cells[regime_id]
            rounds = [int(point["round"]) for point in cell["points"]]
            errors = [float(point["delta_e"]) for point in cell["points"]]
            ax.plot(rounds, errors, color="#4C78A8", linewidth=1.7)
            for endpoint_round, marker, color, size in (
                (50, "o", "#F28E2B", 28),
                (70, "*", "#8B1A1A", 58),
            ):
                ax.scatter(
                    [endpoint_round],
                    [errors[endpoint_round]],
                    marker=marker,
                    color=color,
                    edgecolor="white" if marker == "o" else color,
                    linewidth=0.6,
                    s=size,
                    zorder=4,
                )
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
            ax.yaxis.set_minor_locator(
                LogLocator(base=10.0, subs=tuple(range(2, 10)))
            )
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="major", linewidth=0.45, alpha=0.34)
            ax.grid(True, which="minor", linewidth=0.25, alpha=0.14)
        else:
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([])
            ax.text(
                0.5,
                0.52,
                "PENDING",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.35",
                fontsize=10.5,
                fontweight="bold",
            )
            ax.text(
                0.5,
                0.37,
                "No validated R70 result",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.45",
                fontsize=8,
            )
            ax.grid(True, axis="x", linewidth=0.35, alpha=0.2)
        if index // 3 == 1:
            ax.set_xlabel("ADAPT iteration")
        if index % 3 == 0:
            ax.set_ylabel(r"Same-cutoff $\Delta E$")

    fig.suptitle(
        "Fresh Append-ADAPT singleton extension to 70 rounds",
        fontsize=12.0,
        fontweight="bold",
    )
    fig.legend(
        handles=(
            Line2D([0], [0], color="#4C78A8", linewidth=1.7, label="Append-ADAPT"),
            Line2D(
                [0],
                [0],
                color="#F28E2B",
                marker="o",
                linestyle="none",
                label="round 50 (canonical)",
            ),
            Line2D(
                [0],
                [0],
                color="#8B1A1A",
                marker="*",
                linestyle="none",
                label="round 70 (diagnostic)",
            ),
        ),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=8.3,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _write_page_tex(
    adapter: Mapping[str, Any], *, plot_pdf: Path, tex_path: Path
) -> None:
    cells = {str(cell["regime_id"]): cell for cell in adapter["cells"]}
    rows: list[str] = []
    for regime_id in REGIME_ORDER:
        if regime_id not in cells:
            rows.append(
                rf"{REGIME_LABELS[regime_id]} & pending & -- & -- & -- & -- \\"
            )
            continue
        endpoints = cells[regime_id]["endpoints"]
        endpoint_50 = endpoints["round_50"]
        endpoint_70 = endpoints["round_70"]
        rows.append(
            " & ".join(
                (
                    REGIME_LABELS[regime_id],
                    "complete",
                    _format_delta_e_tex(float(endpoint_50["delta_e"])),
                    _format_cost_tuple_tex(endpoint_50["costs"]),
                    _format_delta_e_tex(float(endpoint_70["delta_e"])),
                    _format_cost_tuple_tex(endpoint_70["costs"]),
                )
            )
            + r" \\"
        )
    plot_argument = _latex_escape(plot_pdf.resolve().as_posix())
    tex = rf"""\documentclass[10pt,letterpaper]{{article}}
\usepackage[landscape,margin=0.30in]{{geometry}}
\usepackage{{amsmath,booktabs,graphicx}}
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\tabcolsep}}{{2.6pt}}
\begin{{document}}
\begin{{center}}
\includegraphics[width=0.965\textwidth,height=4.55in,keepaspectratio]{{{plot_argument}}}
\vspace{{-0.35em}}

\scriptsize
\resizebox{{0.985\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llrrrr@{{}}}}
\toprule
Regime & Status & $\Delta E_{{50}}$ &
$C_{{50}}=(N_{{2q}},D_{{2q}},D_c,W_{{1q}},S_{{\rm alg}})$ canonical &
$\Delta E_{{70}}$ & $C_{{70}}$ diagnostic \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\end{{center}}
\vspace{{-0.45em}}
\footnotesize
All errors use the source-locked exact energy at the identical phonon cutoff.
Round 50 is the Paper-I-comparable endpoint; round 70 is a diagnostic horizon
extension compiled through the same Qiskit convention. This supplemental page
does not enter the validated 48-cell stationary-core matrix.
\end{{document}}
"""
    tex_path.write_text(tex, encoding="utf-8")


def _compile_page(tex_path: Path, page_pdf: Path) -> None:
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    tectonic = shutil.which("tectonic")
    if not any((latexmk, pdflatex, tectonic)):
        raise R70PageError("latexmk, pdflatex, or tectonic is required")
    scratch_root = REPO_ROOT / "tmp" / "pdfs"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{tex_path.stem}_", dir=scratch_root
    ) as raw_build_dir:
        build_dir = Path(raw_build_dir)
        if latexmk:
            command = [
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-outdir={build_dir}",
                tex_path.name,
            ]
        elif pdflatex:
            command = [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={build_dir}",
                tex_path.name,
            ]
        else:
            command = [
                str(tectonic),
                "--outdir",
                str(build_dir),
                tex_path.name,
            ]
        completed = subprocess.run(
            command,
            cwd=tex_path.parent,
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "FORCE_SOURCE_DATE": "1",
                "SOURCE_DATE_EPOCH": "1785196800",
                "TZ": "UTC",
            },
        )
        if completed.returncode != 0:
            raise R70PageError(
                "R70 page LaTeX build failed:\n"
                + completed.stdout[-4000:]
                + completed.stderr[-4000:]
            )
        compiled = build_dir / f"{tex_path.stem}.pdf"
        if not compiled.is_file():
            raise R70PageError("LaTeX completed without producing the R70 page")
        temporary = page_pdf.with_name(f".{page_pdf.name}.tmp")
        shutil.copyfile(compiled, temporary)
        os.replace(temporary, page_pdf)

    from pypdf import PdfReader

    if len(PdfReader(str(page_pdf), strict=False).pages) != 1:
        raise R70PageError("R70 supplemental PDF is not exactly one page")


def _build_page_assets(
    adapter: Mapping[str, Any], *, asset_dir: Path, asset_stem: str
) -> dict[str, Path]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "plot_png": asset_dir / f"{asset_stem}_plot.png",
        "plot_pdf": asset_dir / f"{asset_stem}_plot.pdf",
        "page_tex": asset_dir / f"{asset_stem}.tex",
        "page_pdf": asset_dir / f"{asset_stem}.pdf",
    }
    _render_plot(
        adapter,
        png_path=assets["plot_png"],
        pdf_path=assets["plot_pdf"],
    )
    _write_page_tex(
        adapter,
        plot_pdf=assets["plot_pdf"],
        tex_path=assets["page_tex"],
    )
    _compile_page(assets["page_tex"], assets["page_pdf"])
    return assets


def _page_content_hashes(path: Path) -> list[str]:
    from pypdf import PdfReader

    hashes: list[str] = []
    for page in PdfReader(str(path), strict=False).pages:
        contents = page.get_contents()
        payload = b"" if contents is None else contents.get_data()
        hashes.append(hashlib.sha256(payload).hexdigest())
    return hashes


def _file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verified_report_binding(
    pdf_path: Path, provenance: Mapping[str, Any], *, expected_pages: int
) -> None:
    from pypdf import PdfReader

    output = _mapping(
        _mapping(provenance.get("outputs"), label="provenance.outputs").get(
            "partial_progress_pdf"
        ),
        label="provenance.outputs.partial_progress_pdf",
    )
    if output.get("sha256") != _sha256_file(pdf_path):
        raise R70PageError("base PDF hash does not match provenance")
    if len(PdfReader(str(pdf_path), strict=False).pages) != expected_pages:
        raise R70PageError(f"base PDF is not exactly {expected_pages} pages")


def _already_appended(
    *,
    output_pdf: Path,
    output_provenance: Path,
    adapter_sha256: str,
) -> dict[str, Any] | None:
    if not output_pdf.exists() and not output_provenance.exists():
        return None
    if not output_pdf.is_file() or not output_provenance.is_file():
        raise R70PageError("only one output destination exists; refusing overwrite")
    provenance = _load_json(output_provenance, label="existing output provenance")
    layout = _mapping(provenance.get("layout"), label="existing layout")
    if layout.get("page_count") != 6 or layout.get("page_6") != PAGE_ID:
        return None
    progress = _mapping(
        provenance.get("append_singleton_r70_progress"),
        label="existing append_singleton_r70_progress",
    )
    adapter_binding = _mapping(
        progress.get("adapter"), label="existing R70 adapter binding"
    )
    if adapter_binding.get("canonical_sha256") != adapter_sha256:
        raise R70PageError("page 6 already exists for a different R70 adapter")
    _verified_report_binding(output_pdf, provenance, expected_pages=6)
    structural = _mapping(
        progress.get("structural_validation"),
        label="existing R70 structural validation",
    )
    observed_page_hashes = _page_content_hashes(output_pdf)
    expected_prefix_hashes = structural.get("preserved_page_content_sha256")
    if (
        not isinstance(expected_prefix_hashes, list)
        or observed_page_hashes[:5] != expected_prefix_hashes
        or observed_page_hashes[5] != structural.get("new_page_content_sha256")
    ):
        raise R70PageError("existing page-6 structural validation drifted")
    return {
        "status": "already_present",
        "page_id": PAGE_ID,
        "pages": 6,
        "output_pdf": str(output_pdf),
        "output_provenance": str(output_provenance),
        "sha256": _sha256_file(output_pdf),
        "preserved_pages_1_5": True,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_r70_singleton_progress_page(
    *,
    base_pdf: Path,
    base_provenance: Path,
    output_pdf: Path,
    output_provenance: Path,
    adapter_path: Path,
    asset_dir: Path,
    asset_stem: str,
) -> dict[str, Any]:
    """Append one authenticated R70 page, or return an idempotent no-op."""

    if not _ASSET_STEM_RE.fullmatch(asset_stem) or asset_stem in {".", ".."}:
        raise R70PageError("asset_stem must be a safe filename component")
    adapter_raw = _load_json(adapter_path, label="R70 adapter")
    adapter = _validate_adapter(adapter_raw)

    same_pdf_destination = base_pdf.resolve() == output_pdf.resolve()
    same_provenance_destination = (
        base_provenance.resolve() == output_provenance.resolve()
    )
    existing = _already_appended(
        output_pdf=output_pdf,
        output_provenance=output_provenance,
        adapter_sha256=str(adapter["sha256"]),
    )
    if existing is not None:
        return existing
    if output_pdf.exists() and not same_pdf_destination:
        raise R70PageError("output PDF exists but is not the same completed page")
    if output_provenance.exists() and not same_provenance_destination:
        raise R70PageError(
            "output provenance exists but is not the same completed page"
        )

    provenance = _load_json(base_provenance, label="base provenance")
    layout = _mapping(provenance.get("layout"), label="base provenance layout")
    if layout.get("page_count") != 5:
        raise R70PageError("base provenance layout page_count must be 5")
    if "page_6" in layout:
        raise R70PageError("base provenance already declares page 6")
    _verified_report_binding(base_pdf, provenance, expected_pages=5)
    limitations = provenance.get("limitations")
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) for item in limitations
    ):
        raise R70PageError("base provenance limitations must be a string array")

    before_hashes = _page_content_hashes(base_pdf)
    assets = _build_page_assets(
        adapter, asset_dir=asset_dir.resolve(), asset_stem=asset_stem
    )
    missing_assets = [name for name, path in assets.items() if not path.is_file()]
    if missing_assets:
        raise R70PageError(f"page builder omitted assets: {missing_assets}")

    from pypdf import PdfReader, PdfWriter

    page_reader = PdfReader(str(assets["page_pdf"]), strict=False)
    if len(page_reader.pages) != 1:
        raise R70PageError("R70 page asset must contain exactly one page")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    temporary_pdf = output_pdf.with_name(f".{output_pdf.name}.r70.tmp")
    writer = PdfWriter()
    for page in PdfReader(str(base_pdf), strict=False).pages:
        writer.add_page(page)
    writer.add_page(page_reader.pages[0])
    try:
        with temporary_pdf.open("wb") as stream:
            writer.write(stream)
        after_hashes = _page_content_hashes(temporary_pdf)
        if len(after_hashes) != 6:
            raise R70PageError("combined R70 report is not exactly six pages")
        if after_hashes[:5] != before_hashes:
            raise R70PageError("one or more existing page contents changed")
        os.replace(temporary_pdf, output_pdf)
    finally:
        temporary_pdf.unlink(missing_ok=True)

    updated = copy.deepcopy(provenance)
    updated_layout = _mapping(updated["layout"], label="updated layout")
    updated_layout["page_count"] = 6
    updated_layout["page_6"] = PAGE_ID
    updated_outputs = _mapping(updated["outputs"], label="updated outputs")
    updated_outputs["partial_progress_pdf"] = _file_binding(output_pdf)
    for output_key, asset_key in (
        ("append_singleton_r70_plot_png", "plot_png"),
        ("append_singleton_r70_plot_pdf", "plot_pdf"),
        ("append_singleton_r70_page_tex", "page_tex"),
        ("append_singleton_r70_page_pdf", "page_pdf"),
    ):
        updated_outputs[output_key] = _file_binding(assets[asset_key])
    updated["append_singleton_r70_progress"] = {
        "schema": PAGE_ID,
        "classification": "supplemental_diagnostic_not_adopted_evidence",
        "page_id": PAGE_ID,
        "display_rounds": {"minimum": 0, "maximum": 70},
        "adapter": {
            **_file_binding(adapter_path),
            "canonical_sha256": adapter["sha256"],
        },
        "package_id": adapter.get("package_id"),
        "cluster_id": adapter.get("cluster_id"),
        "completed_regimes": list(COMPLETED_REGIMES),
        "pending_regimes": list(PENDING_REGIMES),
        "source_authentication_summary": copy.deepcopy(
            adapter.get("source_authentication_summary", {})
        ),
        "limitations": copy.deepcopy(adapter.get("limitations", [])),
        "same_cutoff_reference": copy.deepcopy(adapter["same_cutoff_reference"]),
        "cost_policy": copy.deepcopy(adapter["cost_policy"]),
        "cells": [
            {
                "regime_id": cell["regime_id"],
                "display_name": cell["display_name"],
                "nph": cell["nph"],
                "execution_id": cell["execution_id"],
                "source": copy.deepcopy(cell["source"]),
                "endpoints": copy.deepcopy(cell["endpoints"]),
            }
            for cell in adapter["cells"]
        ],
        "structural_validation": {
            "pages_before": 5,
            "pages_after": 6,
            "preserved_page_content_sha256": before_hashes,
            "new_page_content_sha256": after_hashes[5],
        },
        "outputs": {
            key: copy.deepcopy(updated_outputs[key])
            for key in (
                "append_singleton_r70_plot_png",
                "append_singleton_r70_plot_pdf",
                "append_singleton_r70_page_tex",
                "append_singleton_r70_page_pdf",
            )
        },
    }
    if PAGE_LIMITATION not in updated["limitations"]:
        updated["limitations"].append(PAGE_LIMITATION)
    _atomic_write_json(output_provenance, updated)
    return {
        "status": "appended",
        "page_id": PAGE_ID,
        "pages": 6,
        "output_pdf": str(output_pdf),
        "output_provenance": str(output_provenance),
        "sha256": _sha256_file(output_pdf),
        "preserved_pages_1_5": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pdf", type=Path, required=True)
    parser.add_argument("--base-provenance", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-provenance", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--asset-stem", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = append_r70_singleton_progress_page(
            base_pdf=args.base_pdf.resolve(),
            base_provenance=args.base_provenance.resolve(),
            output_pdf=args.output_pdf.resolve(),
            output_provenance=args.output_provenance.resolve(),
            adapter_path=args.adapter.resolve(),
            asset_dir=args.asset_dir.resolve(),
            asset_stem=args.asset_stem,
        )
    except (OSError, R70PageError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
