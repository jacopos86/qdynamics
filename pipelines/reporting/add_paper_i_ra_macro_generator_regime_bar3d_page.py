#!/usr/bin/env python3
"""Append a 3-D macro-generator raw-drop bar page to the diagnostic PDF."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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
DEFAULT_PAGE = (
    REPO_ROOT
    / "output/pdf/paper_i_ra_macro_append_only_generator_type_regime_bar3d_page2.pdf"
)
DEFAULT_PROVENANCE = (
    REPO_ROOT
    / "output/pdf/paper_i_ra_macro_append_only_generator_type_regime_bar3d_append_provenance.json"
)
DISPLAY_FLOOR = 1e-12

BG = "#f7f6f2"
INK = "#172033"
MUTED = "#5e6878"
GRID = "#d9dee7"

CLASS_NAMES = {
    "hva_layer": "HVA layer",
    "hh_termwise_unit": "HVA unit term",
    "hh_termwise_quadrature": "HVA quadrature",
    "hh_hamiltonian_block": "Hamiltonian block",
    "hh_fermionic_reusable": "correlated fermionic",
    "hh_phonon_linear": "phonon linear",
    "hh_phonon_quadratic": "phonon quadratic",
    "hh_vlf_sq": "VLF quadratic",
    "uccsd_sing": "UCCSD single",
    "uccsd_dbl": "UCCSD double",
    "uccsd_paop_product_seq_ferm": "UCCSD x PAOP (fermion step)",
    "uccsd_paop_product_seq_motif": "UCCSD x PAOP (motif step)",
    "uccsd_paop_product": "UCCSD x PAOP product",
    "paop_cloud_p": "PAOP cloud-p",
    "paop_cloud_x": "PAOP cloud-x",
    "paop_disp": "PAOP displacement",
    "paop_dbl": "PAOP doublon",
    "paop_hopdrag": "PAOP hopping drag",
    "paop_dbl_p": "PAOP doublon-p",
    "paop_dbl_x": "PAOP doublon-x",
    "paop_curdrag": "PAOP current drag",
    "paop_hop2": "PAOP second-order hop",
    "paop_other": "other PAOP",
}

REGIME_CODES = ("WW", "IW", "SW", "WS", "IS", "SS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-pdf", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--source-provenance", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--page-pdf", type=Path, default=DEFAULT_PAGE)
    parser.add_argument("--provenance-json", type=Path, default=DEFAULT_PROVENANCE)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_page(payload: dict[str, Any], page_pdf: Path) -> dict[str, Any]:
    figure = payload.get("figure")
    regimes = payload.get("regimes")
    if not isinstance(figure, dict) or not isinstance(regimes, list):
        raise ValueError("source heat-map provenance is incomplete")
    classes = [str(value) for value in figure.get("classes", [])]
    matrix = np.asarray(figure.get("matrix"), dtype=float)
    counts = np.asarray(figure.get("counts"), dtype=int)
    if matrix.shape != (len(classes), 6) or counts.shape != matrix.shape:
        raise ValueError("source matrix must be generator-class by six regimes")
    if len(regimes) != 6:
        raise ValueError("source provenance must contain six regimes")

    configure_matplotlib()
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(
        0.055,
        0.955,
        "Raw drop by macro-generator type and interaction regime",
        fontsize=16.5,
        weight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.055,
        0.92,
        "The original generator-type bar chart extended with interaction regime as the second horizontal axis.",
        fontsize=9.2,
        color=MUTED,
        va="top",
    )
    fig.text(
        0.945,
        0.955,
        "PAPER I DIAGNOSTIC | 2/2",
        fontsize=7.3,
        color=MUTED,
        ha="right",
        va="top",
        weight="bold",
    )

    ax = fig.add_axes([0.035, 0.17, 0.72, 0.70], projection="3d")
    colors = mpl.colormaps["tab20"](np.linspace(0.02, 0.92, len(classes)))
    floor_log = math.log10(DISPLAY_FLOOR)
    dx = 0.62
    dy = 0.62
    below_floor_count = 0
    plotted_count = 0
    for class_index, _class_key in enumerate(classes):
        for regime_index in range(6):
            value = float(matrix[class_index, regime_index])
            count = int(counts[class_index, regime_index])
            if count <= 0 or value <= 0:
                continue
            plotted_count += 1
            if value <= DISPLAY_FLOOR:
                height = 0.14
                color = "#a7afb9"
                below_floor_count += 1
            else:
                height = math.log10(value) - floor_log
                color = colors[class_index]
            ax.bar3d(
                class_index - dx / 2,
                regime_index - dy / 2,
                0.0,
                dx,
                dy,
                height,
                color=color,
                edgecolor="#263244",
                linewidth=0.28,
                alpha=0.92,
                shade=True,
            )

    ax.set_xlim(-0.7, len(classes) - 0.3)
    ax.set_ylim(-0.7, 5.7)
    ax.set_zlim(0.0, 13.35)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"G{index + 1}" for index in range(len(classes))], fontsize=7)
    ax.set_yticks(range(6))
    ax.set_yticklabels(REGIME_CODES, fontsize=7.5)
    z_powers = (-12, -9, -6, -3, 0, 1)
    ax.set_zticks([power - floor_log for power in z_powers])
    ax.set_zticklabels([rf"$10^{{{power}}}$" for power in z_powers], fontsize=7)
    ax.set_xlabel("macro-generator type", labelpad=9)
    ax.set_ylabel("interaction regime", labelpad=9)
    ax.set_zlabel("raw cumulative accepted drop", labelpad=7)
    ax.view_init(elev=25, azim=-56)
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.72))
        axis.pane.set_edgecolor(GRID)

    key_x = 0.77
    fig.text(key_x, 0.86, "GENERATOR KEY", fontsize=7.2, color=MUTED, weight="bold")
    for index, class_key in enumerate(classes):
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
            f"G{index + 1}  {class_name(class_key)}",
            fontsize=6.7,
            color=INK,
            va="center",
        )

    fig.text(key_x, 0.32, "REGIME KEY", fontsize=7.2, color=MUTED, weight="bold")
    regime_lines = []
    for code, row in zip(REGIME_CODES, regimes, strict=True):
        problem = row["problem"]
        regime_lines.append(
            f"{code}  U/t={float(problem['u_over_t']):g}, "
            f"lambda={float(problem['lambda']):.2f}"
        )
    fig.text(
        key_x,
        0.292,
        "\n".join(regime_lines),
        fontsize=6.6,
        family="DejaVu Sans Mono",
        color=INK,
        va="top",
        linespacing=1.28,
    )

    fig.text(
        0.055,
        0.095,
        r"Bar height is logarithmic:  $R_{t,r}=\sum_{k\in(t,r)} "
        r"\max(0,E_{\mathrm{before},k}-E_{\mathrm{after},k})$.  "
        "Flat gray caps denote admitted classes at or below the 1e-12 display floor.",
        fontsize=7.6,
        color=INK,
        va="top",
    )
    fig.text(
        0.055,
        0.055,
        "This remains path-dependent admission credit, not leave-one-out importance in the terminal ansatz. "
        "Exact raw values and admission counts are retained on page 1 and in the companion provenance JSON.",
        fontsize=7.0,
        color=MUTED,
        va="top",
    )
    fig.text(
        0.055,
        0.024,
        f"Source: CHTC cluster {payload['cluster_id']}; six 50-round stationary-core RA-ADAPT macro append-only trajectories.",
        fontsize=6.6,
        color=MUTED,
        va="bottom",
    )

    page_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(page_pdf)
    plt.close(fig)
    return {
        "generator_count": len(classes),
        "regime_count": 6,
        "plotted_bar_count": plotted_count,
        "below_display_floor_bar_count": below_floor_count,
        "display_floor": DISPLAY_FLOOR,
        "vertical_coordinate": "log10(raw_drop) - log10(display_floor)",
        "generator_codes": {
            f"G{index + 1}": class_key for index, class_key in enumerate(classes)
        },
        "regime_codes": {
            code: str(row["id"])
            for code, row in zip(REGIME_CODES, regimes, strict=True)
        },
    }


def append_page(
    *, target_pdf: Path, page_pdf: Path, backup_pdf: Path
) -> dict[str, Any]:
    source_reader = PdfReader(str(target_pdf))
    if len(source_reader.pages) != 1:
        raise ValueError(
            f"refusing to append: expected one-page source PDF, found {len(source_reader.pages)}"
        )
    page_reader = PdfReader(str(page_pdf))
    if len(page_reader.pages) != 1:
        raise ValueError("bar-plot page asset must contain exactly one page")
    before_content_sha = page_content_sha256(source_reader.pages[0])
    before_file_sha = sha256_file(target_pdf)
    if backup_pdf.exists():
        raise FileExistsError(f"refusing to overwrite backup {backup_pdf}")
    shutil.copy2(target_pdf, backup_pdf)
    if sha256_file(backup_pdf) != before_file_sha:
        raise ValueError("pre-append backup hash mismatch")

    writer = PdfWriter()
    writer.add_page(source_reader.pages[0])
    writer.add_page(page_reader.pages[0])
    temporary = target_pdf.with_name(f".{target_pdf.name}.bar3d.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        writer.write(handle)
    merged_reader = PdfReader(str(temporary))
    if len(merged_reader.pages) != 2:
        raise ValueError("merged diagnostic did not contain two pages")
    after_content_sha = page_content_sha256(merged_reader.pages[0])
    if after_content_sha != before_content_sha:
        raise ValueError("page-1 content changed during additive append")
    temporary.replace(target_pdf)
    return {
        "source_page_count": 1,
        "final_page_count": 2,
        "page1_content_sha256_before": before_content_sha,
        "page1_content_sha256_after": after_content_sha,
        "source_pdf_sha256": before_file_sha,
        "backup_pdf": str(backup_pdf),
        "backup_pdf_sha256": sha256_file(backup_pdf),
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
    page_receipt = build_page(source_payload, args.page_pdf)
    backup_pdf = args.target_pdf.with_name(
        f"{args.target_pdf.stem}_pre_bar3d_page2_backup.pdf"
    )
    append_receipt = append_page(
        target_pdf=args.target_pdf,
        page_pdf=args.page_pdf,
        backup_pdf=backup_pdf,
    )
    provenance = digested(
        {
            "schema": "paper_i_ra_macro_generator_regime_bar3d_append_v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed_additive_append",
            "source_provenance_path": str(args.source_provenance.relative_to(REPO_ROOT)),
            "source_provenance_sha256": source_provenance_sha,
            "page_pdf": str(args.page_pdf.relative_to(REPO_ROOT)),
            "page_pdf_sha256": sha256_file(args.page_pdf),
            "target_pdf": str(args.target_pdf.relative_to(REPO_ROOT)),
            "page": page_receipt,
            "append": append_receipt,
        }
    )
    args.provenance_json.write_bytes(canonical_json_bytes(provenance) + b"\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
