#!/usr/bin/env python3
"""Append weak- and strong-Holstein 3-D macro-generator bar pages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader, PdfWriter


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = (
    REPO_ROOT
    / "output/pdf/paper_i_ra_macro_append_only_generator_type_regime_heatmap.pdf"
)
DEFAULT_SOURCE = (
    REPO_ROOT
    / "output/pdf/paper_i_ra_macro_append_only_generator_type_regime_heatmap_provenance.json"
)
DEFAULT_WEAK_PAGE = (
    REPO_ROOT
    / "output/pdf/paper_i_ra_macro_append_only_generator_type_regime_bar3d_weak_holstein_height_ordered_linear_page3.pdf"
)
DEFAULT_STRONG_PAGE = (
    REPO_ROOT
    / "output/pdf/paper_i_ra_macro_append_only_generator_type_regime_bar3d_strong_holstein_height_ordered_linear_page4.pdf"
)
DEFAULT_PROVENANCE = (
    REPO_ROOT
    / "output/pdf/paper_i_ra_macro_append_only_generator_type_regime_sector_bar3d_height_ordered_linear_provenance.json"
)

BG = "#f7f6f2"
INK = "#172033"
MUTED = "#5e6878"
GRID = "#d9dee7"

CLASS_NAMES = {
    "hh_termwise_quadrature": "HVA quadrature",
    "hh_fermionic_reusable": "correlated fermionic",
    "hh_phonon_linear": "phonon linear",
    "hh_phonon_quadratic": "phonon quadratic",
    "uccsd_sing": "UCCSD single",
    "uccsd_dbl": "UCCSD double",
    "uccsd_paop_product_seq_motif": "UCCSD x PAOP (motif step)",
    "paop_cloud_p": "PAOP cloud-p",
    "paop_disp": "PAOP displacement",
    "paop_dbl_p": "PAOP doublon-p",
    "paop_other": "other PAOP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-pdf", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--source-provenance", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--weak-page-pdf", type=Path, default=DEFAULT_WEAK_PAGE)
    parser.add_argument("--strong-page-pdf", type=Path, default=DEFAULT_STRONG_PAGE)
    parser.add_argument("--provenance-json", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--base-pdf", type=Path)
    parser.add_argument("--prior-pdf-backup", type=Path)
    parser.add_argument("--repair-existing-pages", action="store_true")
    parser.add_argument("--finalize-existing-repair", action="store_true")
    parser.add_argument(
        "--write-sector-pages-only",
        action="store_true",
        help="Replace the target with the two newly built normalized sector pages.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def verify_self_digest(payload: dict[str, Any]) -> str:
    claimed = str(payload.get("sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != claimed:
        raise ValueError("source provenance self-digest drifted")
    return actual


def digested(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["sha256"] = hashlib.sha256(canonical_json_bytes(out)).hexdigest()
    return out


def page_content_sha256(page: Any) -> str:
    contents = page.get_contents()
    raw = b"" if contents is None else contents.get_data()
    return hashlib.sha256(raw).hexdigest()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "pdf.fonttype": 42,
        }
    )


def class_name(key: str) -> str:
    return CLASS_NAMES.get(key, key.replace("_", " "))


def sector_indices(regimes: list[dict[str, Any]]) -> list[tuple[str, float, list[int]]]:
    by_lambda: dict[float, list[int]] = {}
    for index, row in enumerate(regimes):
        value = round(float(row["problem"]["lambda"]), 8)
        by_lambda.setdefault(value, []).append(index)
    if len(by_lambda) != 2 or sorted(len(indices) for indices in by_lambda.values()) != [3, 3]:
        raise ValueError("expected two Holstein sectors with three Hubbard regimes each")
    sectors: list[tuple[str, float, list[int]]] = []
    for label, value in zip(("weak", "strong"), sorted(by_lambda), strict=True):
        indices = sorted(
            by_lambda[value],
            key=lambda idx: float(regimes[idx]["problem"]["u_over_t"]),
        )
        sectors.append((label, value, indices))
    return sectors


def build_sector_page(
    payload: dict[str, Any],
    *,
    sector_label: str,
    lambda_value: float,
    regime_indices: list[int],
    page_pdf: Path,
) -> dict[str, Any]:
    figure = payload.get("figure")
    regimes = payload.get("regimes")
    if not isinstance(figure, dict) or not isinstance(regimes, list):
        raise ValueError("source heat-map provenance is incomplete")
    source_classes = [str(value) for value in figure.get("classes", [])]
    full_matrix = np.asarray(figure.get("matrix"), dtype=float)
    full_counts = np.asarray(figure.get("counts"), dtype=int)
    if full_matrix.shape != (len(source_classes), 6) or full_counts.shape != full_matrix.shape:
        raise ValueError("source matrix must be generator-class by six regimes")
    unsorted_matrix = full_matrix[:, regime_indices]
    unsorted_counts = full_counts[:, regime_indices]
    regime_raw_totals = unsorted_matrix.sum(axis=0)
    if np.any(regime_raw_totals <= 0.0):
        raise ValueError("each regime must have positive realized path drop")
    normalized_unsorted_matrix = (
        100.0 * unsorted_matrix / regime_raw_totals[np.newaxis, :]
    )
    sector_totals = normalized_unsorted_matrix.sum(axis=1)
    sector_maxima = normalized_unsorted_matrix.max(axis=1)
    sector_admissions = unsorted_counts.sum(axis=1)
    order = sorted(
        range(len(source_classes)),
        key=lambda index: (
            -float(sector_totals[index]),
            -float(sector_maxima[index]),
            -int(sector_admissions[index]),
            index,
        ),
    )
    classes = [source_classes[index] for index in order]
    class_codes = [f"G{index + 1}" for index in order]
    matrix = normalized_unsorted_matrix[order, :]
    raw_matrix = unsorted_matrix[order, :]
    counts = unsorted_counts[order, :]
    ordered_totals = sector_totals[order]
    ordered_raw_totals = raw_matrix.sum(axis=1)
    ordered_admissions = sector_admissions[order]
    selected_regimes = [regimes[index] for index in regime_indices]

    configure_matplotlib()
    fig = plt.figure(figsize=(11, 8.5))
    title_sector = "Weak" if sector_label == "weak" else "Strong"
    fig.text(
        0.055,
        0.955,
        f"{title_sector} Holstein sector: normalized drop by macro-generator type",
        fontsize=16.2,
        weight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.055,
        0.92,
        rf"Fixed $\lambda={lambda_value:.2f}$; each Hubbard regime is normalized by its own total realized path drop.",
        fontsize=9.2,
        color=MUTED,
        va="top",
    )
    fig.text(
        0.945,
        0.985,
        f"PAPER I DIAGNOSTIC | {title_sector.upper()} HOLSTEIN",
        fontsize=7.3,
        color=MUTED,
        ha="right",
        va="top",
        weight="bold",
    )

    ax = fig.add_axes([0.02, 0.16, 0.75, 0.72], projection="3d")
    ax.set_proj_type("ortho")
    source_colors = mpl.colormaps["tab20"](
        np.linspace(0.02, 0.92, len(source_classes))
    )
    colors = source_colors[order]
    x_positions = np.arange(len(classes), dtype=float) * 1.20
    y_positions = np.arange(3, dtype=float) * 1.55
    dx = 0.50
    dy = 0.54
    plotted_count = 0
    visibility_values: dict[str, list[float | None]] = {
        "G2": [],
        "G3": [],
        "G4": [],
    }
    bar_x: list[float] = []
    bar_y: list[float] = []
    bar_height: list[float] = []
    bar_colors: list[Any] = []
    count_labels: list[tuple[float, float, float, int]] = []
    for class_index, code in enumerate(class_codes):
        for local_regime_index in range(3):
            value = float(matrix[class_index, local_regime_index])
            count = int(counts[class_index, local_regime_index])
            if code in visibility_values:
                visibility_values[code].append(value if count > 0 else None)
            if count <= 0:
                continue
            count_labels.append(
                (
                    x_positions[class_index],
                    y_positions[local_regime_index],
                    max(value, 0.0),
                    count,
                )
            )
            if value <= 0:
                continue
            plotted_count += 1
            bar_x.append(x_positions[class_index] - dx / 2)
            bar_y.append(y_positions[local_regime_index] - dy / 2)
            bar_height.append(value)
            bar_colors.append(colors[class_index])
    if bar_height:
        ax.bar3d(
            np.asarray(bar_x),
            np.asarray(bar_y),
            np.zeros(len(bar_height), dtype=float),
            np.full(len(bar_height), dx, dtype=float),
            np.full(len(bar_height), dy, dtype=float),
            np.asarray(bar_height),
            color=np.asarray(bar_colors),
            edgecolor="#263244",
            linewidth=0.34,
            alpha=0.90,
            shade=True,
            zsort="average",
        )

    largest_bar = max(bar_height, default=1.0)
    label_offset = largest_bar * 0.018
    for label_x, label_y, label_height, count in count_labels:
        ax.text(
            label_x,
            label_y,
            label_height + label_offset,
            str(count),
            fontsize=6.2,
            color=INK,
            ha="center",
            va="bottom",
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": BG,
                "edgecolor": "none",
                "alpha": 0.82,
            },
        )

    ax.set_xlim(-0.75, x_positions[-1] + 0.8)
    ax.set_ylim(-0.75, y_positions[-1] + 0.8)
    ax.set_zlim(0.0, largest_bar * 1.14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_codes, fontsize=7.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"U/t={float(row['problem']['u_over_t']):g}" for row in selected_regimes],
        fontsize=7.5,
    )
    ax.set_xlabel("macro-generator type", labelpad=10)
    ax.set_ylabel("Hubbard strength", labelpad=10)
    ax.set_zlabel(
        r"within-regime realized-drop share  (\%)",
        labelpad=7,
    )
    ax.view_init(elev=31, azim=-49)
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.72))
        axis.pane.set_edgecolor(GRID)

    key_x = 0.785
    fig.text(
        key_x,
        0.86,
        "GENERATOR KEY (SECTOR ADMISSIONS)",
        fontsize=7.2,
        color=MUTED,
        weight="bold",
    )
    for index, (class_key, code) in enumerate(zip(classes, class_codes, strict=True)):
        y = 0.825 - index * 0.043
        fig.patches.append(
            mpl.patches.Rectangle(
                (key_x, y - 0.008),
                0.012,
                0.012,
                transform=fig.transFigure,
                facecolor=colors[index],
                edgecolor="none",
            )
        )
        fig.text(
            key_x + 0.018,
            y,
            f"{code}  {class_name(class_key)}  n={int(ordered_admissions[index])}",
            fontsize=6.45,
            color=INK,
            va="center",
        )

    fig.text(key_x, 0.32, "HUBBARD REGIMES", fontsize=7.2, color=MUTED, weight="bold")
    regime_lines = [
        f"U/t={float(row['problem']['u_over_t']):g}  {row['id']}"
        for row in selected_regimes
    ]
    fig.text(
        key_x,
        0.292,
        "\n".join(regime_lines),
        fontsize=6.6,
        family="DejaVu Sans Mono",
        color=INK,
        va="top",
        linespacing=1.35,
    )

    fig.text(
        0.055,
        0.090,
        r"Bar height is the within-regime share  $p_{a,r}=100R_{a,r}/\sum_bR_{b,r}$, where "
        r"$R_{a,r}=\sum_{k\in(a,r)}\max(0,E_{\mathrm{before},k}-E_{\mathrm{after},k})$.  "
        r"For every regime, $\sum_a p_{a,r}=100\%$.",
        fontsize=7.5,
        color=INK,
        va="top",
    )
    fig.text(
        0.055,
        0.052,
        "Integer labels give per-regime admission counts; key values n give sector totals. "
        "Classes are ordered by their equally weighted sum of within-regime shares.",
        fontsize=7.0,
        color=MUTED,
        va="top",
    )
    fig.text(
        0.055,
        0.024,
        f"Source: CHTC cluster {payload['cluster_id']}; three 50-round stationary-core RA-ADAPT macro append-only trajectories at fixed lambda={lambda_value:.2f}.",
        fontsize=6.6,
        color=MUTED,
        va="bottom",
    )

    page_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(page_pdf)
    plt.close(fig)
    return {
        "sector": sector_label,
        "lambda": lambda_value,
        "generator_count": len(classes),
        "regime_count": 3,
        "regime_ids": [str(row["id"]) for row in selected_regimes],
        "u_over_t": [float(row["problem"]["u_over_t"]) for row in selected_regimes],
        "plotted_bar_count": plotted_count,
        "admitted_cell_count": len(count_labels),
        "bar_label_semantics": "integer_per_regime_admission_count",
        "height_scale": "linear_percent_of_each_regime_total_realized_drop",
        "normalization": "100*class_regime_raw_drop/sum_class(class_regime_raw_drop)",
        "ordering": "descending_sum_of_within_regime_percent_shares",
        "display_order": class_codes,
        "sector_sum_of_regime_percent_shares": {
            code: float(total)
            for code, total in zip(class_codes, ordered_totals, strict=True)
        },
        "sector_total_raw_drop_diagnostic": {
            code: float(total)
            for code, total in zip(class_codes, ordered_raw_totals, strict=True)
        },
        "regime_raw_drop_denominators": {
            str(row["id"]): float(total)
            for row, total in zip(
                selected_regimes, regime_raw_totals, strict=True
            )
        },
        "regime_normalized_share_sums": {
            str(row["id"]): float(total)
            for row, total in zip(
                selected_regimes, matrix.sum(axis=0), strict=True
            )
        },
        "sector_admission_count": {
            code: int(count)
            for code, count in zip(class_codes, ordered_admissions, strict=True)
        },
        "generator_codes": {
            f"G{index + 1}": class_key
            for index, class_key in enumerate(source_classes)
        },
        "visibility_check_normalized_percent_values": visibility_values,
    }


def write_sector_pages_only(
    *,
    target_pdf: Path,
    page_pdfs: list[Path],
    backup_pdf: Path,
) -> dict[str, Any]:
    if not target_pdf.is_file():
        raise FileNotFoundError(target_pdf)
    page_readers = [PdfReader(str(path)) for path in page_pdfs]
    if any(len(reader.pages) != 1 for reader in page_readers):
        raise ValueError("each normalized sector page must contain exactly one page")
    source_file_sha = sha256_file(target_pdf)
    if backup_pdf.exists():
        raise FileExistsError(f"refusing to overwrite backup {backup_pdf}")
    shutil.copy2(target_pdf, backup_pdf)
    if sha256_file(backup_pdf) != source_file_sha:
        raise ValueError("pre-normalization backup hash mismatch")

    expected_page_shas = [
        page_content_sha256(reader.pages[0]) for reader in page_readers
    ]
    writer = PdfWriter()
    for reader in page_readers:
        writer.add_page(reader.pages[0])
    temporary = target_pdf.with_name(
        f".{target_pdf.name}.normalized_sector_pages.tmp"
    )
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        writer.write(handle)
    final_reader = PdfReader(str(temporary))
    if len(final_reader.pages) != 2:
        raise ValueError("normalized macro diagnostic must contain two pages")
    actual_page_shas = [
        page_content_sha256(page) for page in final_reader.pages
    ]
    if actual_page_shas != expected_page_shas:
        raise ValueError("normalized sector pages changed during assembly")
    temporary.replace(target_pdf)
    return {
        "operation": "replace_with_two_normalized_sector_pages",
        "source_page_count": len(PdfReader(str(backup_pdf)).pages),
        "final_page_count": 2,
        "source_pdf_sha256": source_file_sha,
        "backup_pdf": str(backup_pdf),
        "backup_pdf_sha256": sha256_file(backup_pdf),
        "normalized_page_content_sha256": actual_page_shas,
        "final_pdf_sha256": sha256_file(target_pdf),
    }


def append_pages(
    *, target_pdf: Path, page_pdfs: list[Path], backup_pdf: Path
) -> dict[str, Any]:
    source_reader = PdfReader(str(target_pdf))
    if len(source_reader.pages) != 2:
        raise ValueError(
            f"refusing to append: expected two-page source PDF, found {len(source_reader.pages)}"
        )
    page_readers = [PdfReader(str(path)) for path in page_pdfs]
    if any(len(reader.pages) != 1 for reader in page_readers):
        raise ValueError("each sector page asset must contain exactly one page")
    before_content_shas = [page_content_sha256(page) for page in source_reader.pages]
    before_file_sha = sha256_file(target_pdf)
    if backup_pdf.exists():
        raise FileExistsError(f"refusing to overwrite backup {backup_pdf}")
    shutil.copy2(target_pdf, backup_pdf)
    if sha256_file(backup_pdf) != before_file_sha:
        raise ValueError("pre-append backup hash mismatch")

    writer = PdfWriter()
    for page in source_reader.pages:
        writer.add_page(page)
    for reader in page_readers:
        writer.add_page(reader.pages[0])
    temporary = target_pdf.with_name(f".{target_pdf.name}.sector_bar3d.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        writer.write(handle)
    merged_reader = PdfReader(str(temporary))
    if len(merged_reader.pages) != 4:
        raise ValueError("merged diagnostic did not contain four pages")
    after_content_shas = [page_content_sha256(merged_reader.pages[index]) for index in range(2)]
    if after_content_shas != before_content_shas:
        raise ValueError("an existing page changed during additive append")
    temporary.replace(target_pdf)
    return {
        "source_page_count": 2,
        "final_page_count": 4,
        "existing_page_content_sha256_before": before_content_shas,
        "existing_page_content_sha256_after": after_content_shas,
        "source_pdf_sha256": before_file_sha,
        "backup_pdf": str(backup_pdf),
        "backup_pdf_sha256": sha256_file(backup_pdf),
        "final_pdf_sha256": sha256_file(target_pdf),
    }


def replace_sector_pages(
    *,
    target_pdf: Path,
    base_pdf: Path,
    page_pdfs: list[Path],
    prior_pdf_backup: Path,
) -> dict[str, Any]:
    current_reader = PdfReader(str(target_pdf))
    if len(current_reader.pages) != 4:
        raise ValueError(
            f"refusing to repair: expected four-page target PDF, found {len(current_reader.pages)}"
        )
    base_reader = PdfReader(str(base_pdf))
    if len(base_reader.pages) != 2:
        raise ValueError(
            f"refusing to repair: expected two-page preserved base, found {len(base_reader.pages)}"
        )
    page_readers = [PdfReader(str(path)) for path in page_pdfs]
    if any(len(reader.pages) != 1 for reader in page_readers):
        raise ValueError("each corrected sector page asset must contain exactly one page")
    current_prefix_shas = [
        page_content_sha256(current_reader.pages[index]) for index in range(2)
    ]
    base_prefix_shas = [page_content_sha256(page) for page in base_reader.pages]
    if current_prefix_shas != base_prefix_shas:
        raise ValueError("preserved two-page base does not match current pages 1-2")
    current_file_sha = sha256_file(target_pdf)
    if prior_pdf_backup.exists():
        raise FileExistsError(f"refusing to overwrite repair backup {prior_pdf_backup}")
    shutil.copy2(target_pdf, prior_pdf_backup)
    if sha256_file(prior_pdf_backup) != current_file_sha:
        raise ValueError("pre-repair four-page backup hash mismatch")

    writer = PdfWriter()
    for page in base_reader.pages:
        writer.add_page(page)
    for reader in page_readers:
        writer.add_page(reader.pages[0])
    temporary = target_pdf.with_name(
        f".{target_pdf.name}.height_ordered_linear_sector_repair.tmp"
    )
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        writer.write(handle)
    merged_reader = PdfReader(str(temporary))
    if len(merged_reader.pages) != 4:
        raise ValueError("repaired diagnostic did not contain four pages")
    repaired_prefix_shas = [
        page_content_sha256(merged_reader.pages[index]) for index in range(2)
    ]
    if repaired_prefix_shas != current_prefix_shas:
        raise ValueError("pages 1-2 changed during sector-page repair")
    temporary.replace(target_pdf)
    return {
        "operation": "replace_pages_3_4_from_preserved_two_page_base",
        "source_page_count": 4,
        "final_page_count": 4,
        "existing_page_content_sha256_before": current_prefix_shas,
        "existing_page_content_sha256_after": repaired_prefix_shas,
        "preserved_base_pdf": str(base_pdf),
        "preserved_base_pdf_sha256": sha256_file(base_pdf),
        "prior_target_pdf_sha256": current_file_sha,
        "prior_target_backup": str(prior_pdf_backup),
        "prior_target_backup_sha256": sha256_file(prior_pdf_backup),
        "final_pdf_sha256": sha256_file(target_pdf),
    }


def validate_repaired_sector_pages(
    *,
    target_pdf: Path,
    base_pdf: Path,
    page_pdfs: list[Path],
    prior_pdf_backup: Path,
) -> dict[str, Any]:
    for path in (target_pdf, base_pdf, prior_pdf_backup, *page_pdfs):
        if not path.is_file():
            raise FileNotFoundError(path)
    target_reader = PdfReader(str(target_pdf))
    base_reader = PdfReader(str(base_pdf))
    page_readers = [PdfReader(str(path)) for path in page_pdfs]
    if len(target_reader.pages) != 4 or len(base_reader.pages) != 2:
        raise ValueError("repaired target/base page count mismatch")
    if any(len(reader.pages) != 1 for reader in page_readers):
        raise ValueError("corrected page asset must contain exactly one page")
    prefix_shas = [page_content_sha256(target_reader.pages[index]) for index in range(2)]
    base_shas = [page_content_sha256(page) for page in base_reader.pages]
    if prefix_shas != base_shas:
        raise ValueError("repaired pages 1-2 do not match preserved base")
    target_sector_shas = [
        page_content_sha256(target_reader.pages[index]) for index in range(2, 4)
    ]
    asset_shas = [page_content_sha256(reader.pages[0]) for reader in page_readers]
    if target_sector_shas != asset_shas:
        raise ValueError("repaired pages 3-4 do not match corrected page assets")
    return {
        "operation": "validate_completed_pages_3_4_repair",
        "source_page_count": 4,
        "final_page_count": 4,
        "existing_page_content_sha256_before": prefix_shas,
        "existing_page_content_sha256_after": prefix_shas,
        "preserved_base_pdf": str(base_pdf),
        "preserved_base_pdf_sha256": sha256_file(base_pdf),
        "prior_target_backup": str(prior_pdf_backup),
        "prior_target_backup_sha256": sha256_file(prior_pdf_backup),
        "corrected_sector_page_content_sha256": target_sector_shas,
        "final_pdf_sha256": sha256_file(target_pdf),
    }


def main() -> None:
    args = parse_args()
    for path in (args.target_pdf, args.source_provenance):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_payload = json.loads(args.source_provenance.read_text(encoding="utf-8"))
    if source_payload.get("schema") != "paper_i_ra_macro_generator_regime_heatmap_v1":
        raise ValueError("unexpected source provenance schema")
    source_provenance_sha = verify_self_digest(source_payload)
    regimes = source_payload.get("regimes")
    if not isinstance(regimes, list):
        raise ValueError("source provenance regimes are missing")
    sectors = sector_indices(regimes)
    page_paths = {"weak": args.weak_page_pdf, "strong": args.strong_page_pdf}
    page_receipts: dict[str, Any] = {}
    for sector_label, lambda_value, regime_indices in sectors:
        page_receipts[sector_label] = build_sector_page(
            source_payload,
            sector_label=sector_label,
            lambda_value=lambda_value,
            regime_indices=regime_indices,
            page_pdf=page_paths[sector_label],
        )
    backup_pdf = args.target_pdf.with_name(
        f"{args.target_pdf.stem}_pre_sector_bar3d_pages34_backup.pdf"
    )
    if args.write_sector_pages_only:
        backup_pdf = args.prior_pdf_backup or args.target_pdf.with_name(
            f"{args.target_pdf.stem}_pre_within_regime_normalization_backup.pdf"
        )
        append_receipt = write_sector_pages_only(
            target_pdf=args.target_pdf,
            page_pdfs=[args.weak_page_pdf, args.strong_page_pdf],
            backup_pdf=backup_pdf,
        )
        schema = (
            "paper_i_ra_macro_generator_regime_sector_bar3d_"
            "within_regime_normalized_v1"
        )
        status = "passed_within_regime_normalized_two_page_rebuild"
    elif args.repair_existing_pages:
        base_pdf = args.base_pdf or backup_pdf
        if not base_pdf.is_file():
            raise FileNotFoundError(base_pdf)
        prior_pdf_backup = args.prior_pdf_backup or args.target_pdf.with_name(
            f"{args.target_pdf.stem}_pre_height_ordered_linear_sector_repair_backup.pdf"
        )
        if args.finalize_existing_repair:
            append_receipt = validate_repaired_sector_pages(
                target_pdf=args.target_pdf,
                base_pdf=base_pdf,
                page_pdfs=[args.weak_page_pdf, args.strong_page_pdf],
                prior_pdf_backup=prior_pdf_backup,
            )
        else:
            append_receipt = replace_sector_pages(
                target_pdf=args.target_pdf,
                base_pdf=base_pdf,
                page_pdfs=[args.weak_page_pdf, args.strong_page_pdf],
                prior_pdf_backup=prior_pdf_backup,
            )
        schema = (
            "paper_i_ra_macro_generator_regime_sector_bar3d_"
            "height_ordered_linear_repair_v1"
        )
        status = "passed_sector_page_repair"
    else:
        append_receipt = append_pages(
            target_pdf=args.target_pdf,
            page_pdfs=[args.weak_page_pdf, args.strong_page_pdf],
            backup_pdf=backup_pdf,
        )
        schema = "paper_i_ra_macro_generator_regime_sector_bar3d_append_v2"
        status = "passed_additive_append"
    provenance = digested(
        {
            "schema": schema,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "source_provenance_path": repo_relative(args.source_provenance),
            "source_provenance_sha256": source_provenance_sha,
            "page_pdfs": {
                sector: {
                    "path": repo_relative(path),
                    "sha256": sha256_file(path),
                }
                for sector, path in page_paths.items()
            },
            "target_pdf": repo_relative(args.target_pdf),
            "pages": page_receipts,
            "append": append_receipt,
        }
    )
    args.provenance_json.write_bytes(canonical_json_bytes(provenance) + b"\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
