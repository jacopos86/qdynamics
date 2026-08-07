#!/usr/bin/env python3
"""Plot the displayed Paper-I Hubbard--Holstein endpoints versus cost.

The three methods use the endpoint conventions currently rendered in
``MATH/paper_details/Paper_I.tex``.  These figures are therefore descriptive
endpoint Pareto views, not prefix-matched convergence curves.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_TEX = REPO_ROOT / "MATH/paper_details/Paper_I.tex"
OUTPUT_DIR = REPO_ROOT / (
    "MATH/paper_details/figures/"
    "paper_i_hh_endpoint_error_vs_cost_20260718"
)
SCHEMA = "paper_i_hh_displayed_endpoint_error_vs_cost_v1"

METHOD_STYLE = {
    "SNAKE": {"color": "#E45756", "marker": "*", "size": 120},
    "Geo-ADAPT": {"color": "#54A24B", "marker": "^", "size": 65},
    "Append-ADAPT": {"color": "#4C78A8", "marker": "o", "size": 65},
}

REGIME_TITLES = {
    "weak_weak": "weak--weak\n" r"$U/t=0.25$, $\lambda=0.25$, $M=2$",
    "intermediate_weak": "intermediate--weak\n" r"$U/t=1.25$, $\lambda=0.25$, $M=2$",
    "strong_weak": "strong--weak\n" r"$U/t=8$, $\lambda=0.25$, $M=2$",
    "weak_strong": "weak--strong\n" r"$U/t=0.25$, $\lambda=1.25$, $M=4$",
    "intermediate_strong": "intermediate--strong\n" r"$U/t=1.25$, $\lambda=1.25$, $M=4$",
    "strong_strong": "strong--strong\n" r"$U/t=8$, $\lambda=1.25$, $M=4$",
}

# Exact values rendered in the six mini-tables under Fig. 1 on 2026-07-18.
ROWS = (
    ("weak_weak", "SNAKE", 30, 7.993605777301127e-15, 258, 1234, 39947),
    ("weak_weak", "Geo-ADAPT", 5, 0.0013117770378563431, 98, 480, 24859),
    ("weak_weak", "Append-ADAPT", 23, 0.0009192819178269751, 4216, 15936, 108633),
    ("intermediate_weak", "SNAKE", 23, 3.4416913763379853e-15, 196, 901, 33814),
    ("intermediate_weak", "Geo-ADAPT", 4, 0.09893293320249796, 70, 528, 19792),
    ("intermediate_weak", "Append-ADAPT", 28, 0.005223690514773949, 4276, 15924, 231208),
    ("strong_weak", "SNAKE", 9, 1.5711479179891796e-6, 54, 329, 7639),
    ("strong_weak", "Geo-ADAPT", 6, 2.354651946201436e-6, 146, 763, 30007),
    ("strong_weak", "Append-ADAPT", 6, 2.3546519453132575e-6, 146, 762, 2137),
    ("weak_strong", "SNAKE", 50, 3.051703169054676e-6, 822, 3187, 182236),
    ("weak_strong", "Geo-ADAPT", 8, 0.043646714159213396, 1784, 6160, 47155),
    ("weak_strong", "Append-ADAPT", 23, 0.023007750483881706, 32596, 94899, 136947),
    ("intermediate_strong", "SNAKE", 50, 4.7338917443573436e-5, 824, 3320, 196013),
    ("intermediate_strong", "Geo-ADAPT", 8, 0.02897791863102661, 1784, 6104, 47245),
    ("intermediate_strong", "Append-ADAPT", 25, 0.009295831578261082, 33340, 97248, 167416),
    ("strong_strong", "SNAKE", 33, 3.2083758372269955e-7, 404, 1687, 80909),
    ("strong_strong", "Geo-ADAPT", 6, 4.982206763848307e-5, 348, 1149, 33962),
    ("strong_strong", "Append-ADAPT", 8, 3.61579243989274e-5, 7976, 26219, 6378),
)

METRICS = {
    "n2q": ("compiled_count_2q", r"Two-qubit gate count $N_{2q}$"),
    "dc": ("compiled_depth", r"Total circuit depth $D_c$"),
    "s_alg": ("s_alg", r"Logical estimator queries $S_{\mathrm{alg}}$"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def records() -> list[dict[str, object]]:
    return [
        {
            "regime": regime,
            "method": method,
            "displayed_k": k,
            "absolute_same_cutoff_error": error,
            "compiled_count_2q": n2q,
            "compiled_depth": dc,
            "s_alg": s_alg,
        }
        for regime, method, k, error, n2q, dc, s_alg in ROWS
    ]


def plot_one(
    regime: str,
    rows: list[dict[str, object]],
    metric_key: str,
    field: str,
    xlabel: str,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    for row in rows:
        method = str(row["method"])
        style = METHOD_STYLE[method]
        ax.scatter(
            float(row[field]),
            float(row["absolute_same_cutoff_error"]),
            color=style["color"],
            marker=style["marker"],
            s=style["size"],
            edgecolor="black",
            linewidth=0.45,
            label=method,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Same-cutoff error $|\Delta E|$")
    ax.set_title(REGIME_TITLES[regime], fontsize=9)
    ax.grid(True, which="both", color="#D8D8D8", linewidth=0.55, alpha=0.75)
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.tick_params(axis="both", which="major", labelsize=7.5)
    ax.tick_params(axis="both", which="minor", labelsize=6.5)

    stem = f"paper_i_hh_endpoint_error_vs_{metric_key}__{regime}"
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> int:
    if not SOURCE_TEX.is_file():
        raise FileNotFoundError(SOURCE_TEX)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    endpoint_rows = records()

    csv_path = OUTPUT_DIR / "paper_i_hh_endpoint_error_vs_cost_20260718.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoint_rows[0]))
        writer.writeheader()
        writer.writerows(endpoint_rows)

    plots: list[dict[str, object]] = []
    for regime in REGIME_TITLES:
        regime_rows = [row for row in endpoint_rows if row["regime"] == regime]
        if len(regime_rows) != 3:
            raise ValueError(f"{regime}: expected three displayed methods")
        for metric_key, (field, xlabel) in METRICS.items():
            png, pdf = plot_one(regime, regime_rows, metric_key, field, xlabel)
            plots.append(
                {
                    "regime": regime,
                    "metric": metric_key,
                    "png": rel(png),
                    "png_sha256": sha256(png),
                    "pdf": rel(pdf),
                    "pdf_sha256": sha256(pdf),
                }
            )

    provenance_path = OUTPUT_DIR / "paper_i_hh_endpoint_error_vs_cost_20260718.json"
    payload = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "standalone candidate figures; no manuscript figure insertion",
        "comparison_policy": (
            "descriptive displayed endpoints; no interpolation, no connecting lines, "
            "and no prefix-matched or equal-budget inference"
        ),
        "error_metric": "absolute same-cutoff energy error",
        "cost_metrics": {
            "n2q": "Qiskit compiled two-qubit gate count N2q",
            "dc": "Qiskit total compiled circuit depth Dc",
            "s_alg": "logical estimator-query count S_alg; not physical shots",
        },
        "endpoint_policy": (
            "SNAKE uses the currently displayed terminal/selected-last-plateau "
            "endpoints; Geo-ADAPT and Append-ADAPT retain their displayed "
            "historical first-plateau prefixes"
        ),
        "source_tex": rel(SOURCE_TEX),
        "source_tex_sha256": sha256(SOURCE_TEX),
        "source_region": "Hubbard--Holstein Fig. 1 mini-tables",
        "builder": rel(Path(__file__)),
        "builder_sha256": sha256(Path(__file__)),
        "endpoint_csv": rel(csv_path),
        "endpoint_csv_sha256": sha256(csv_path),
        "plots": plots,
        "blockers": [],
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"provenance": rel(provenance_path), "plot_count": len(plots)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
