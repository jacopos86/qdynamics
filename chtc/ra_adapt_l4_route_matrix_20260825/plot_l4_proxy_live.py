"""Plot the recorded L=4 proxy-RA prefix against completed AVQE."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


EXACT_ENERGY = -3.3827681604170525


def _read_json_with_hash(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _avqe_curve(payload: dict[str, object]) -> tuple[list[int], list[float]]:
    history = payload["result_payload"]["history"]
    energies = [float(history[0]["energy_before"])]
    energies.extend(float(row["energy_after"]) for row in history)
    return list(range(len(energies))), [abs(value - EXACT_ENERGY) for value in energies]


def _ra_curve(payload: dict[str, object]) -> tuple[list[int], list[float]]:
    history = payload["adapt_vqe"]["history"]
    energies = [float(history[0]["energy_before_opt"])]
    energies.extend(float(row["energy_after_opt"]) for row in history)
    return list(range(len(energies))), [abs(value - EXACT_ENERGY) for value in energies]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("avqe_result", type=Path)
    parser.add_argument("ra_checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    avqe_path = args.avqe_result.resolve()
    ra_path = args.ra_checkpoint.resolve()
    avqe_payload, avqe_sha = _read_json_with_hash(avqe_path)
    ra_payload, ra_sha = _read_json_with_hash(ra_path)
    avqe_x, avqe_y = _avqe_curve(avqe_payload)
    ra_x, ra_y = _ra_curve(ra_payload)

    output_dir = args.output_directory.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"l4_proxy_ra_k{ra_x[-1]}_vs_avqe_k{avqe_x[-1]}"
    png = prefix.with_suffix(".png")
    pdf = prefix.with_suffix(".pdf")
    provenance_path = prefix.with_name(prefix.name + "_provenance.json")

    styles = {
        "AVQE": ("#4C78A8", "o", 1.8, 36),
        "RA, graph-span proxy": ("#E45756", "*", 2.2, 64),
    }
    curves = {
        "AVQE": (avqe_x, avqe_y),
        "RA, graph-span proxy": (ra_x, ra_y),
    }

    fig, ax = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    handles: list[Line2D] = []
    for label, (x, y) in curves.items():
        color, marker, linewidth, size = styles[label]
        ax.plot(x, y, color=color, linewidth=linewidth, solid_capstyle="round")
        ax.scatter([x[-1]], [y[-1]], color=color, marker=marker, s=size, zorder=4)
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=linewidth,
                marker=marker,
                markersize=5.5 if marker != "*" else 7.0,
                label=label,
            )
        )

    ax.set_yscale("log")
    ax.set_xlim(0, max(avqe_x[-1], ra_x[-1]))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_xlabel("ADAPT iteration")
    ax.set_ylabel("Absolute energy error")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.55)
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.4)
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    sources = [
        {
            "label": "AVQE",
            "history_source": str(avqe_path),
            "history_source_sha256": avqe_sha,
            "point_count": len(avqe_x),
            "marker_policy": "terminal_observed_point",
            "marker_k": avqe_x[-1],
            "marker_absolute_energy_error": avqe_y[-1],
            "source_status": "complete",
        },
        {
            "label": "RA, graph-span proxy",
            "history_source": str(ra_path),
            "history_source_sha256": ra_sha,
            "point_count": len(ra_x),
            "marker_policy": "terminal_observed_point",
            "marker_k": ra_x[-1],
            "marker_absolute_energy_error": ra_y[-1],
            "source_status": "recorded_live_prefix",
            "selector_cost": "marrakesh_graph_span_family_robust_arctan_v1",
            "refit_parameterization": "fixed_fs_whitened_affine_coordinate_chart_v1",
            "active_gradient_policy": "stationary_source_response_v1",
            "powell_tolerances": {"xtol": 1.0e-5, "ftol": 1.0e-12},
        },
    ]
    provenance = {
        "schema": "paper_i_exploratory_l4_live_proxy_comparison_plot_v1",
        "problem": {
            "model": "open_half_filled_pure_hubbard_chain",
            "num_sites": 4,
            "num_qubits": 8,
            "u_over_t": 1.25,
            "exact_energy": EXACT_ENERGY,
        },
        "displayed_error": "absolute_same_problem_exact_energy_error",
        "method_curves": "solid_with_one_terminal_marker",
        "no_extrapolation": True,
        "sources": sources,
        "generated_png": str(png),
        "generated_pdf": str(pdf),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
