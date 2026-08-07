#!/usr/bin/env python3
"""Build the six Paper-I novelty-off SR-SNAKE convergence figures.

The Geo-ADAPT and Append-ADAPT trajectories and their displayed plateau
markers are copied exactly from the July-8 support CSV.  Only the SNAKE curve
is replaced, using source-locked novelty-off result histories.  The three
round-50 continuation results are read directly from their preserved transfer
archives; the archives are never extracted or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCHEMA = "paper_i_no_ordinary_novelty_sr_snake_convergence_v1"
OUTPUT_STEM = "paper_i_no_ordinary_novelty_sr_snake_20260717"
PRIOR_PROVENANCE = Path(
    "MATH/paper_details/figures/paper_i_physical_lane_snake_duplicate_20260708/"
    "paper_i_physical_lane_snake_duplicate_20260708_append_parent_only_provenance.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "MATH/paper_details/figures/paper_i_no_ordinary_novelty_sr_snake_20260717"
)
DEFAULT_SUPPORT_CSV = DEFAULT_OUTPUT_DIR / (
    "paper_i_no_ordinary_novelty_sr_snake_20260717_comparator_support_source.csv"
)


@dataclass(frozen=True)
class Regime:
    key: str
    prior_key: str
    title: str
    terminal_k: int
    display_marker_k: int
    snake_source_kind: str
    snake_source: Path


REGIMES = (
    Regime(
        "weak_weak",
        "weak-weak",
        r"weak-weak: $U/t=0.25$, $\lambda=0.25$, $M=2$",
        30,
        30,
        "json",
        Path(
            "raw_outputs/paper_i_hh_sr_snake_weak_weak_undamped_no_prune_no_beam_no_ordinary_novelty_fallback_on_20260715/json/result.json"
        ),
    ),
    Regime(
        "intermediate_weak",
        "intermediate-weak",
        r"intermediate-weak: $U/t=1.25$, $\lambda=0.25$, $M=2$",
        30,
        23,
        "json",
        Path(
            "raw_outputs/paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_five_20260715_v1_chtc/intermediate_weak/json/result.json"
        ),
    ),
    Regime(
        "strong_weak",
        "strong-weak",
        r"strong-weak: $U/t=8$, $\lambda=0.25$, $M=2$",
        50,
        9,
        "json",
        Path(
            "raw_outputs/paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_strong_weak_u8_r50_repair_20260716_v2_chtc/strong_weak_u8/json/result.json"
        ),
    ),
    Regime(
        "weak_strong",
        "weak-strong",
        r"weak-strong: $U/t=0.25$, $\lambda=1.25$, $M=4$",
        50,
        50,
        "tar_json",
        Path(
            "raw_outputs/chtc_fetch_paper_i_hh_sr_20260716/paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_r50_continuations_20260715_v1_chtc/weak_strong_transfer.tar.gz"
        ),
    ),
    Regime(
        "intermediate_strong",
        "intermediate-strong",
        r"intermediate-strong: $U/t=1.25$, $\lambda=1.25$, $M=4$",
        50,
        50,
        "tar_json",
        Path(
            "raw_outputs/chtc_fetch_paper_i_hh_sr_20260716/paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_r50_continuations_20260715_v1_chtc/intermediate_strong_transfer.tar.gz"
        ),
    ),
    Regime(
        "strong_strong",
        "strong-strong",
        r"strong-strong: $U/t=8$, $\lambda=1.25$, $M=4$",
        50,
        33,
        "tar_json",
        Path(
            "raw_outputs/chtc_fetch_paper_i_hh_sr_20260716/paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_r50_continuations_20260715_v1_chtc/strong_strong_u8_transfer.tar.gz"
        ),
    ),
)

STYLE = {
    "SNAKE": {"color": "#E45756", "marker": "*", "linewidth": 2.8, "size": 120},
    "Geo-ADAPT": {"color": "#54A24B", "marker": "^", "linewidth": 1.8, "size": 65},
    "Append-ADAPT": {"color": "#4C78A8", "marker": "o", "linewidth": 1.8, "size": 65},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_result(root: Path, regime: Regime) -> tuple[dict[str, Any], dict[str, Any]]:
    source = root / regime.snake_source
    if not source.is_file():
        raise FileNotFoundError(source)
    if regime.snake_source_kind == "json":
        return json.loads(source.read_text()), {
            "path": rel(root, source),
            "sha256": sha256(source),
            "kind": "result_json",
        }
    if regime.snake_source_kind != "tar_json":
        raise ValueError(regime.snake_source_kind)
    with tarfile.open(source, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.name.endswith("/json/result.json")]
        if len(members) != 1:
            raise RuntimeError(f"{source}: expected one result.json, found {len(members)}")
        member = members[0]
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"{source}: cannot read {member.name}")
        payload = extracted.read()
    return json.loads(payload), {
        "path": rel(root, source),
        "sha256": sha256(source),
        "kind": "preserved_transfer_archive_member",
        "member": member.name,
        "member_sha256": hashlib.sha256(payload).hexdigest(),
    }


def snake_points(result: dict[str, Any], terminal_k: int) -> list[list[float]]:
    adapt = result.get("adapt_vqe")
    if not isinstance(adapt, dict):
        raise RuntimeError("missing adapt_vqe")
    history = adapt.get("history")
    if not isinstance(history, list) or len(history) != terminal_k:
        raise RuntimeError(f"expected {terminal_k} history records, found {len(history or [])}")
    points: list[list[float]] = []
    for index, row in enumerate(history):
        value = row.get("delta_abs_current")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"history[{index}].delta_abs_current is not finite positive")
        points.append([float(index), float(value)])
    terminal = adapt.get("abs_delta_e")
    if not isinstance(terminal, (int, float)) or not math.isfinite(terminal) or terminal <= 0:
        raise RuntimeError("adapt_vqe.abs_delta_e is not finite positive")
    # The result's terminal error includes the preserved terminal refit and is
    # therefore an additional point after the final admission history record.
    points.append([float(terminal_k), float(terminal)])
    return points


def load_comparator_rows(
    support_csv: Path, prior: dict[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    prior_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for plot in prior.get("plots", []):
        regime = plot.get("regime")
        for method in plot.get("methods", []):
            if method.get("method") in ("Geo-ADAPT", "Append-ADAPT"):
                prior_rows[(regime, method["method"])] = method
    support_by_role: dict[tuple[str, str], dict[str, str]] = {}
    with support_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            support_by_role[(row["regime"], row["role_key"])] = row

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for key, meta in prior_rows.items():
        support = support_by_role.get((key[0], meta["role_key"]))
        if support is None:
            raise RuntimeError(f"support row missing for {key} / {meta['role_key']}")
        points = json.loads(support["trajectory_points_json"])
        if len(points) != int(meta["point_count"]):
            raise RuntimeError(f"point-count mismatch for {key}")
        normalized = [[float(x), float(y)] for x, y in points]
        marker_k = int(meta["marker_k"])
        marker_y = next((y for x, y in normalized if int(x) == marker_k), None)
        if marker_y is None or not math.isclose(
            marker_y, float(meta["marker_error"]), rel_tol=1e-11, abs_tol=1e-15
        ):
            raise RuntimeError(f"marker mismatch for {key}")
        rows[key] = {
            "method": key[1],
            "role_key": meta["role_key"],
            "points": normalized,
            "marker_k": marker_k,
            "marker_error": float(marker_y),
            "marker_policy": "retained_existing_plateau_marker",
            "original_source_json": meta["source_json"],
            "original_source_sha256": meta["source_sha256"],
        }
    return rows


def plot_one(
    regime: Regime,
    curves: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(5.17, 3.58), dpi=200)
    handles: list[Line2D] = []
    for curve in curves:
        method = curve["method"]
        style = STYLE[method]
        xs = [p[0] for p in curve["points"]]
        ys = [p[1] for p in curve["points"]]
        ax.plot(xs, ys, color=style["color"], linewidth=style["linewidth"], solid_capstyle="round")
        ax.scatter(
            [curve["marker_k"]],
            [curve["marker_error"]],
            marker=style["marker"],
            s=style["size"],
            color=style["color"],
            edgecolor="black",
            linewidth=0.65,
            zorder=5,
        )
        handles.append(
            Line2D(
                [0], [0], color=style["color"], linewidth=style["linewidth"],
                marker=style["marker"], markersize=8 if method != "SNAKE" else 10,
                markerfacecolor=style["color"], markeredgecolor="black",
                label=f"{method}, k={curve['marker_k']}",
            )
        )
    ax.set_yscale("log")
    x_max = max(regime.terminal_k, 30)
    ax.set_xlim(0, x_max + 1)
    tick_step = 3 if x_max == 30 else 5
    ax.set_xticks(list(range(0, x_max + 1, tick_step)))
    ax.set_xlabel(r"ADAPT iteration $k$")
    ax.set_ylabel(r"$|\Delta E|$")
    ax.set_title(regime.title)
    ax.grid(which="major", alpha=0.23, linewidth=0.6)
    ax.grid(which="minor", alpha=0.10, linewidth=0.45)
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95, fontsize=9)
    fig.tight_layout(pad=0.55)
    png = output_dir / f"{OUTPUT_STEM}__{regime.key}.png"
    pdf = output_dir / f"{OUTPUT_STEM}__{regime.key}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "build_paper_i_no_ordinary_novelty_sr_snake_figures_20260717.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return png, pdf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--support-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    prior_path = root / PRIOR_PROVENANCE
    prior = json.loads(prior_path.read_text())
    if args.support_csv:
        support_csv = args.support_csv.resolve()
    else:
        local_support_csv = root / DEFAULT_SUPPORT_CSV
        support_csv = (
            local_support_csv.resolve()
            if local_support_csv.is_file()
            else Path(prior["support_csv"]).resolve()
        )
    if not support_csv.is_file():
        raise FileNotFoundError(
            f"July-2 comparator support CSV not available: {support_csv}; pass --support-csv"
        )
    if sha256(support_csv) != prior["support_csv_sha256"]:
        raise RuntimeError("comparator support CSV SHA-256 does not match July-8 provenance")
    output_dir = (args.output_dir or (root / DEFAULT_OUTPUT_DIR)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    comparators = load_comparator_rows(support_csv, prior)
    consumed = [
        {"role": "july8_plot_provenance", "path": rel(root, prior_path), "sha256": sha256(prior_path)},
        {"role": "july2_comparator_support_csv", "path": str(support_csv), "sha256": sha256(support_csv)},
    ]
    plots: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for regime in REGIMES:
        result, source = load_result(root, regime)
        consumed.append({"role": f"novelty_off_{regime.key}", **source})
        snake_curve = snake_points(result, regime.terminal_k)
        if int(regime.display_marker_k) == int(regime.terminal_k):
            marker_error = float(result["adapt_vqe"]["abs_delta_e"])
            marker_policy = "terminal_result_after_preserved_terminal_refit"
            marker_history_position = None
        else:
            # SNAKE history records are post-admission checkpoints.  The plot's
            # visible x index is zero based, so x=k corresponds to one-based
            # history position k+1 and, with pruning disabled, active depth k+1.
            marker_error = float(snake_curve[int(regime.display_marker_k)][1])
            marker_policy = "user_selected_last_plateau_history_checkpoint"
            marker_history_position = int(regime.display_marker_k) + 1
        snake = {
            "method": "SNAKE",
            "role_key": "snake_no_ordinary_novelty",
            "points": snake_curve,
            "marker_k": int(regime.display_marker_k),
            "marker_error": marker_error,
            "marker_policy": marker_policy,
            "marker_history_position": marker_history_position,
            "source": source,
        }
        curves = [
            snake,
            comparators[(regime.prior_key, "Geo-ADAPT")],
            comparators[(regime.prior_key, "Append-ADAPT")],
        ]
        png, pdf = plot_one(regime, curves, output_dir)
        method_records: list[dict[str, Any]] = []
        for curve in curves:
            record = {k: v for k, v in curve.items() if k != "points"}
            record["point_count"] = len(curve["points"])
            record["terminal_k"] = int(curve["points"][-1][0])
            record["terminal_error"] = float(curve["points"][-1][1])
            method_records.append(record)
            for x, y in curve["points"]:
                point_rows.append(
                    {
                        "regime": regime.key,
                        "method": curve["method"],
                        "iteration": int(x),
                        "absolute_same_cutoff_error": format(y, ".17g"),
                        "is_display_marker": int(int(x) == curve["marker_k"]),
                        "marker_policy": curve["marker_policy"],
                    }
                )
        plots.append(
            {
                "regime": regime.key,
                "title": regime.title,
                "error_metric": "absolute same-cutoff energy error |E_alg(n_ph_work)-E_ED(n_ph_work)|",
                "x_policy": "integer ADAPT iteration; SNAKE x=0..horizon-1 are post-admission history positions 1..horizon, and x=horizon is the terminal-refit result",
                "display_crop": {
                    "x_min": 0,
                    "last_data_x": max(regime.terminal_k, 30),
                    "axis_x_max": max(regime.terminal_k, 30) + 1,
                    "y_limits": None,
                },
                "methods": method_records,
                "png": rel(root, png),
                "png_sha256": sha256(png),
                "pdf": rel(root, pdf),
                "pdf_sha256": sha256(pdf),
            }
        )

    provenance_csv = output_dir / f"{OUTPUT_STEM}_provenance.csv"
    with provenance_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)
    provenance_json = output_dir / f"{OUTPUT_STEM}_provenance.json"
    payload = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "separate Paper-I novelty-off manuscript copy; no active-manuscript edit",
        "builder": {
            "path": rel(root, Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "plot_policy": "separate one-column files; solid curves; exactly one method marker; SNAKE user-selected plateau markers for intermediate-weak, strong-weak, and strong-strong and terminal markers otherwise; retained comparator plateau markers; log y; no target lines; integer x",
        "visible_methods": ["SNAKE", "Geo-ADAPT", "Append-ADAPT"],
        "consumed_artifacts": consumed,
        "plots": plots,
        "provenance_csv": rel(root, provenance_csv),
        "provenance_csv_sha256": sha256(provenance_csv),
        "blockers": [],
    }
    provenance_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "provenance_json": rel(root, provenance_json),
        "provenance_json_sha256": sha256(provenance_json),
        "provenance_csv": rel(root, provenance_csv),
        "plots": [{"regime": p["regime"], "png": p["png"], "pdf": p["pdf"]} for p in plots],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
