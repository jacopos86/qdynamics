"""Driven frequency-domain response for the reported Paper-II cells.

The manuscript introduction lists frequency-domain response among the physical
outputs of a driven mixed fermion--boson simulation.  This module produces it
from artifacts that already exist: every reported trajectory carries the
doublon occupation and the staggered charge imbalance on the declared
501-point grid, and the independent exact trajectory for the same cell is
recovered from the content-addressed reference store and verified by SHA-256
before use.

No trajectory is rerun.  The transform is a Hann-windowed real DFT of the
mean-removed signal, so the reported magnitudes are response amplitudes about
the time-averaged value and the drive fundamental appears at its nominal
frequency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

OBSERVABLES = (("doublon", r"doublon $D(t)$"), ("staggered", r"charge imbalance $\Delta n(t)$"))
APM_COLOR = "#8B1A1A"
AVQDS_COLOR = "#1F4E9C"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(run_json: Path) -> dict[str, np.ndarray]:
    payload = json.loads(run_json.read_text())
    rows = payload["plot_rows"]
    out = {"time": np.array([row["time"] for row in rows], dtype=float)}
    for name, _ in OBSERVABLES:
        out[name] = np.array([row[name] for row in rows], dtype=float)
    return out


def _reference(store: Path, key: str, expected_sha: str) -> dict[str, np.ndarray]:
    data = store / key / "reference.npz"
    actual = _sha256(data)
    if actual != expected_sha:
        raise ValueError(f"exact reference {key} sha mismatch: {actual} != {expected_sha}")
    payload = np.load(data)
    return {name: np.asarray(payload[name], dtype=float) for name in ("time", *(n for n, _ in OBSERVABLES))}


def spectrum(times: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed response amplitude of a mean-removed signal."""

    count = signal.size
    step = float(times[1] - times[0])
    if not np.allclose(np.diff(times), step):
        raise ValueError("response spectra require a uniform reporting grid")
    window = np.hanning(count)
    transform = np.fft.rfft((signal - signal.mean()) * window)
    omega = 2.0 * np.pi * np.fft.rfftfreq(count, step)
    return omega, np.abs(transform) * (2.0 / window.sum())


def relative_error(reference: np.ndarray, other: np.ndarray, band: np.ndarray) -> float:
    return float(np.linalg.norm(other[band] - reference[band]) / np.linalg.norm(reference[band]))


def residual_floor(
    times: np.ndarray,
    exact: np.ndarray,
    routes: Sequence[np.ndarray],
    band: np.ndarray,
) -> float:
    """Broadband amplitude floor set by the trajectories' own time-domain error.

    Each route's residual against the exact trajectory is transformed with the
    same window, and the floor is the larger of the two median residual
    amplitudes over the displayed band.  Taking the maximum keeps the admitted
    bins identical for both routes, so the reported spectral errors are scored
    on the same frequencies.
    """

    medians = [float(np.median(spectrum(times, route - exact)[1][band])) for route in routes]
    return max(medians)


def _cells(comparison: Mapping[str, Any], controller: str) -> list[dict[str, Any]]:
    selected = []
    for cell in comparison.get("cells", []):
        if str(cell.get("controller")) != controller:
            continue
        routes = cell.get("routes", {})
        if any(routes.get(r) is None or routes[r].get("accuracy") is None for r in ("q_compiled", "avqds")):
            continue
        drive = routes["q_compiled"]["run_locks"][0]["physics"]["drive_profile"]
        selected.append(
            {
                "cell_id": str(cell["cell_id"]),
                "drive": str(cell["drive"]),
                "A": float(drive["A"]),
                "omega": float(drive["omega"]),
                "reference_key": Path(str(cell["reference"]["manifest_path"])).parent.name,
                "reference_sha256": str(cell["reference"]["data_sha256"]),
            }
        )
    selected.sort(key=lambda entry: entry["omega"])
    return selected


def build(
    comparison_json: Path,
    raw_root: Path,
    reference_store: Path,
    figure: Path,
    manifest: Path,
    controller: str = "delta_theta_5e-5",
    omega_max: float = 25.0,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparison = json.loads(comparison_json.read_text())
    cells = _cells(comparison, controller)
    if not cells:
        raise ValueError(f"no reportable cells for controller {controller}")

    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(OBSERVABLES), len(cells), figsize=(2.55 * len(cells), 3.6), sharex=True
    )
    axes = np.atleast_2d(axes)
    records: list[dict[str, Any]] = []

    for column, cell in enumerate(cells):
        apm = _rows(raw_root / f"{cell['cell_id']}__exchange_q_compiled" / "run.json")
        avqds = _rows(raw_root / f"{cell['cell_id']}__avqds" / "run.json")
        exact = _reference(reference_store, cell["reference_key"], cell["reference_sha256"])
        errors: dict[str, dict[str, float]] = {}
        for row, (name, label) in enumerate(OBSERVABLES):
            omega, ref = spectrum(exact["time"], exact[name])
            _, a = spectrum(apm["time"], apm[name])
            _, b = spectrum(avqds["time"], avqds[name])
            band = (omega > 0.0) & (omega <= omega_max)
            floor = residual_floor(exact["time"], exact[name], (apm[name], avqds[name]), band)
            admitted = band & (ref > floor)
            if not admitted.any():
                raise ValueError(f"{cell['cell_id']}/{name}: exact spectrum lies entirely under the floor")
            errors[name] = {
                "apm": relative_error(ref, a, admitted),
                "avqds": relative_error(ref, b, admitted),
                "residual_floor": floor,
                "admitted_bins": int(admitted.sum()),
                "displayed_bins": int(band.sum()),
                "omega_max_admitted": float(omega[admitted].max()),
            }
            ax = axes[row, column]
            ax.semilogy(omega[band], ref[band], color="black", ls=":", lw=1.1, label="exact")
            ax.semilogy(omega[band], a[band], color=APM_COLOR, lw=1.0, label="APM")
            ax.semilogy(omega[band], b[band], color=AVQDS_COLOR, ls="--", lw=1.0, label="AVQDS")
            for harmonic in range(1, int(omega_max // cell["omega"]) + 1):
                ax.axvline(harmonic * cell["omega"], color="0.85", lw=0.6, zorder=0)
            # Nothing below the floor is scored, so nothing below it is drawn:
            # the axis starts there instead of leaving a third of the panel
            # blank.
            ax.set_ylim(bottom=floor)
            ax.tick_params(labelsize=6)
            if column == 0:
                ax.set_ylabel(label, fontsize=7)
            if row == 0:
                ax.set_title(
                    f"{cell['drive']}  $(A,\\omega)=({cell['A']:g},{cell['omega']:g})$", fontsize=7
                )
            if row == len(OBSERVABLES) - 1:
                ax.set_xlabel(r"$\omega$", fontsize=7)
        records.append({**cell, "spectral_relative_error": errors})

    axes[0, 0].legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.savefig(figure, format="pdf")
    plt.close(fig)

    payload = {
        "schema": "paper_ii_response_spectra_v1",
        "controller": controller,
        "omega_band": [0.0, omega_max],
        "transform": "Hann-windowed rFFT of the mean-removed signal, amplitude normalized",
        "error_band_policy": (
            "relative L2 error is scored only on bins whose exact amplitude exceeds the "
            "broadband residual floor, so noise-dominated bins do not enter the metric"
        ),
        "comparison": {"path": str(comparison_json), "sha256": _sha256(comparison_json)},
        "cells": records,
        "figure": {"path": str(figure), "sha256": _sha256(figure)},
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-json", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--reference-store", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--controller", default="delta_theta_5e-5")
    args = parser.parse_args()
    payload = build(
        args.comparison_json,
        args.raw_root,
        args.reference_store,
        args.figure,
        args.manifest,
        args.controller,
    )
    print(
        json.dumps(
            {
                c["cell_id"]: c["spectral_relative_error"] for c in payload["cells"]
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
