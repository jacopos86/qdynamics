"""Plot the exploratory Paper-I L=4 route matrix through a common horizon."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


EXACT_ENERGY = -3.3827681604170525


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_avqe(path: Path) -> tuple[list[int], list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))["result_payload"]
    history = payload["history"]
    energies = [float(history[0]["energy_before"])]
    energies.extend(float(record["energy_after"]) for record in history)
    return list(range(len(energies))), [abs(value - EXACT_ENERGY) for value in energies]


def _load_ra(path: Path) -> tuple[list[int], list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))["run"]
    trajectory = payload["accepted_trajectory"]
    energies = [2.5]
    energies.extend(float(record["energy"]) for record in trajectory)
    return list(range(len(energies))), [abs(value - EXACT_ENERGY) for value in energies]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()

    root = args.matrix_root.resolve()
    output = args.output_prefix.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    avqe_path = root / "avqe_k20/result.json"
    families = {
        "RA, exact Qiskit marginals": sorted(root.glob("ra_qiskit_*_k20/result.json")),
        "RA, graph-span proxy": sorted(root.glob("ra_proxy_*_k20/result.json")),
    }
    if any(len(paths) != 4 for paths in families.values()):
        raise RuntimeError("Expected four chart/residual variants in each RA family.")

    avqe_x, avqe_y = _load_avqe(avqe_path)
    representative: dict[str, tuple[list[int], list[float]]] = {}
    overlap: dict[str, float] = {}
    for label, paths in families.items():
        curves = [_load_ra(path) for path in paths]
        x0, y0 = curves[0]
        if any(x != x0 for x, _ in curves[1:]):
            raise RuntimeError(f"Iteration grids differ within {label}.")
        overlap[label] = max(
            float(np.max(np.abs(np.asarray(y) - np.asarray(y0))))
            for _, y in curves[1:]
        )
        if overlap[label] > 1.0e-9:
            raise RuntimeError(f"{label} variants no longer overlap: {overlap[label]:.3e}")
        representative[label] = (x0, y0)

    styles = {
        "AVQE": {"color": "#4C78A8", "marker": "o", "linewidth": 1.8},
        "RA, exact Qiskit marginals": {
            "color": "#E45756",
            "marker": "*",
            "linewidth": 2.1,
        },
        "RA, graph-span proxy": {
            "color": "#F58518",
            "marker": "D",
            "linewidth": 2.0,
        },
    }

    fig, ax = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    curves = {"AVQE": (avqe_x, avqe_y), **representative}
    handles: list[Line2D] = []
    for label, (x, y) in curves.items():
        style = styles[label]
        ax.plot(x, y, color=style["color"], linewidth=style["linewidth"], solid_capstyle="round")
        ax.scatter(
            [x[-1]],
            [y[-1]],
            color=style["color"],
            marker=style["marker"],
            s=38 if style["marker"] != "*" else 62,
            zorder=4,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=5.5 if style["marker"] != "*" else 7.0,
                label=label,
            )
        )

    ax.set_yscale("log")
    ax.set_xlim(0, 20)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_xlabel("ADAPT iteration")
    ax.set_ylabel(r"Absolute energy error")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.55)
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7.1)

    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    provenance_path = output.with_name(output.name + "_provenance.json")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    sources = []
    for label, (x, y) in curves.items():
        paths = [avqe_path] if label == "AVQE" else families[label]
        sources.append(
            {
                "label": label,
                "history_sources": [str(path) for path in paths],
                "history_source_sha256": [_sha256(path) for path in paths],
                "point_count": len(x),
                "marker_policy": "terminal_observed_point",
                "marker_k": x[-1],
                "marker_absolute_energy_error": y[-1],
                "maximum_within_family_error_difference": overlap.get(label, 0.0),
            }
        )
    provenance = {
        "schema": "paper_i_exploratory_l4_route_matrix_plot_v1",
        "problem": {
            "model": "open_half_filled_pure_hubbard_chain",
            "num_sites": 4,
            "num_qubits": 8,
            "u_over_t": 1.25,
            "exact_energy": EXACT_ENERGY,
        },
        "displayed_error": "absolute_same_problem_exact_energy_error",
        "display_horizon": 20,
        "method_curves": "solid_with_one_terminal_marker",
        "ra_family_note": (
            "Each RA family contains measured/stationary active-residual variants "
            "in native coordinates with FS-whitened Powell directions and in the "
            "fixed FS-whitened affine chart; the four energy trajectories overlap "
            "to the recorded tolerance."
        ),
        "sources": sources,
        "generated_png": str(png),
        "generated_pdf": str(pdf),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__).resolve()),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
