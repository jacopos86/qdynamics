#!/usr/bin/env python3
"""Build the Paper-I pure-Hubbard common-prefix noise figures.

The locked input adapter contains completed low-, high-, and extreme-noise
trajectories for U/t = 1.5 and U/t = 8.  The manuscript comparison uses the
common displayed prefix k <= 20 for every trajectory.  The later extreme-noise
points remain in the source adapter and are intentionally outside the display
window.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ADAPTER = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "pure_hubbard_page12_fullnoise_page15_adapter.json"
)
OUTPUT_DIR = REPO_ROOT / (
    "MATH/paper_details/figures/"
    "paper_i_hubbard_noise_common_prefix_k20_20260814"
)
PROVENANCE = OUTPUT_DIR / (
    "paper_i_hubbard_noise_common_prefix_k20_20260814_provenance.json"
)

DISPLAY_K_MAX = 20
X_LIMITS = (0.0, 20.0)
Y_LIMITS = (5.0e-7, 2.0)
U_ORDER = ("u1p5", "u8")
U_LABELS = {"u1p5": r"$U/t=1.5$", "u8": r"$U/t=8$"}
LEVEL_ORDER = ("low", "high", "extreme")
LEVEL_STYLES = {
    "low": {"color": "#009E73", "marker": "o", "label": "low"},
    "high": {"color": "#E69F00", "marker": "s", "label": "high"},
    "extreme": {"color": "#CC79A7", "marker": "D", "label": "extreme"},
}
EXPECTED_NOISE_TUPLES = {
    "low": (1.0e-6, 1.0e-8, 1.0e-7, 2.0e-4, 6.0e-4),
    "high": (7.071067811865475e-5, 1.0e-6, 1.0e-5, 2.0e-3, 6.0e-3),
    "extreme": (1.0e-2, 1.0e-3, 1.0e-2, 6.0e-2, 6.0e-2),
}


class FigureBuildError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FigureBuildError(f"unsafe or missing file: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def load_adapter() -> dict[str, Any]:
    value = json.loads(SOURCE_ADAPTER.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FigureBuildError("source adapter must be a JSON object")
    claimed = value.get("sha256")
    unsigned = {key: row for key, row in value.items() if key != "sha256"}
    if claimed != canonical_sha256(unsigned):
        raise FigureBuildError("source adapter self digest drifted")
    if value.get("status") != "completed_6_of_6_mixed_horizon":
        raise FigureBuildError("source adapter lacks six completed cells")
    cells = value.get("cells")
    if not isinstance(cells, list) or len(cells) != 6:
        raise FigureBuildError("source adapter must contain exactly six cells")
    return value


def validate_cells(adapter: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    cells_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for cell in adapter["cells"]:
        if not isinstance(cell, Mapping):
            raise FigureBuildError("noise cell must be a mapping")
        key = (str(cell.get("u_key")), str(cell.get("noise_level_id")))
        if key in cells_by_key:
            raise FigureBuildError(f"duplicate noise cell: {key}")
        if key[0] not in U_ORDER or key[1] not in LEVEL_ORDER:
            raise FigureBuildError(f"unexpected noise cell: {key}")
        observed_tuple = tuple(float(value) for value in cell.get("noise_tuple", ()))
        expected_tuple = EXPECTED_NOISE_TUPLES[key[1]]
        if len(observed_tuple) != len(expected_tuple) or any(
            not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-16)
            for observed, expected in zip(observed_tuple, expected_tuple)
        ):
            raise FigureBuildError(f"noise tuple drifted for {key}")
        result = cell.get("result")
        if not isinstance(result, Mapping):
            raise FigureBuildError(f"completed result missing for {key}")
        points = result.get("points")
        if not isinstance(points, list):
            raise FigureBuildError(f"history missing for {key}")
        displayed = [row for row in points if int(row["k"]) <= DISPLAY_K_MAX]
        if [int(row["k"]) for row in displayed] != list(
            range(1, DISPLAY_K_MAX + 1)
        ):
            raise FigureBuildError(f"common k=1:20 history is incomplete for {key}")
        if any(float(row["error"]) <= 0.0 for row in displayed):
            raise FigureBuildError(f"nonpositive log-scale error for {key}")
        cells_by_key[key] = cell
    expected_keys = {(u_key, level) for u_key in U_ORDER for level in LEVEL_ORDER}
    if set(cells_by_key) != expected_keys:
        raise FigureBuildError("six-cell noise coverage drifted")
    return cells_by_key


def render_one(
    *,
    u_key: str,
    cells_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(3.45, 2.62))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.19, top=0.80)
    curve_records: list[dict[str, Any]] = []
    legend_handles = []
    for level in LEVEL_ORDER:
        cell = cells_by_key[(u_key, level)]
        result = cell["result"]
        displayed = [
            row for row in result["points"] if int(row["k"]) <= DISPLAY_K_MAX
        ]
        style = LEVEL_STYLES[level]
        x_values = [int(row["k"]) for row in displayed]
        y_values = [float(row["error"]) for row in displayed]
        axis.plot(x_values, y_values, color=style["color"], lw=1.55)
        axis.scatter(
            [x_values[-1]],
            [y_values[-1]],
            color=style["color"],
            marker=style["marker"],
            s=23,
            zorder=4,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                lw=1.55,
                marker=style["marker"],
                markersize=4.0,
                label=style["label"],
            )
        )
        curve_records.append(
            {
                "noise_level": level,
                "source_cluster_id": int(cell["source_cluster_id"]),
                "source_proc_id": int(result["proc_id"]),
                "source_target_horizon": int(cell["target_horizon"]),
                "source_point_count": len(result["points"]),
                "display_point_count": len(displayed),
                "display_terminal_k": x_values[-1],
                "display_terminal_error": y_values[-1],
                "noise_tuple": [float(value) for value in cell["noise_tuple"]],
            }
        )
    axis.set_yscale("log")
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(*Y_LIMITS)
    axis.xaxis.set_major_locator(MultipleLocator(5))
    axis.set_xlabel("ADAPT iteration")
    axis.set_ylabel(r"same-cutoff $|\Delta E|$")
    axis.text(
        0.97,
        0.95,
        U_LABELS[u_key],
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80},
    )
    axis.grid(True, which="major", alpha=0.22, lw=0.45)
    axis.legend(
        handles=legend_handles,
        title=r"noise level (marker at $k=20$)",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        title_fontsize=6.8,
        handlelength=1.5,
        columnspacing=0.8,
    )

    stem = f"paper_i_hubbard_noise_{u_key}_common_prefix_k20_20260814"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return (
        {"pdf": binding(pdf_path), "png": binding(png_path)},
        {"u_key": u_key, "u_over_t": float(cells_by_key[(u_key, "low")]["u_over_t"]), "curves": curve_records},
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adapter = load_adapter()
    cells_by_key = validate_cells(adapter)
    outputs: dict[str, Any] = {}
    panels: list[dict[str, Any]] = []
    for u_key in U_ORDER:
        output_record, panel_record = render_one(
            u_key=u_key,
            cells_by_key=cells_by_key,
        )
        outputs[u_key] = output_record
        panels.append(panel_record)
    provenance = {
        "schema": "paper_i_hubbard_noise_common_prefix_k20_figure_v1",
        "source_adapter": {
            **binding(SOURCE_ADAPTER),
            "canonical_sha256": adapter["sha256"],
            "status": adapter["status"],
        },
        "generation_script": binding(Path(__file__)),
        "display_contract": {
            "quantity": "absolute same-cutoff energy error",
            "iteration_window": [0, DISPLAY_K_MAX],
            "source_history_window_used": [1, DISPLAY_K_MAX],
            "x_limits": list(X_LIMITS),
            "y_limits": list(Y_LIMITS),
            "y_scale": "log",
            "guide_lines": "none",
            "curve_markers": "one marker at displayed k=20 endpoint",
            "later_source_points": "retained in source adapter and not displayed",
        },
        "panels": panels,
        "outputs": outputs,
    }
    provenance["sha256"] = canonical_sha256(provenance)
    PROVENANCE.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"provenance": binding(PROVENANCE), "outputs": outputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
