"""Declared common observables for reduced electron--phonon trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .contracts import ReducedTrajectory, Spectrum, SpectrumConvention
from .validation import require_structurally_valid


def _basis_indices(
    basis: tuple[str, ...],
    labels: Sequence[str],
) -> tuple[int, ...]:
    if not labels:
        raise ValueError("site label collections must not be empty")
    missing = tuple(label for label in labels if label not in basis)
    if missing:
        raise ValueError(f"basis labels are missing: {missing}")
    indices = tuple(basis.index(label) for label in labels)
    if len(set(indices)) != len(indices):
        raise ValueError("site label collections must not contain duplicates")
    return indices


def dimer_polarization(
    trajectory: ReducedTrajectory,
    *,
    site_1_labels: Sequence[str],
    site_2_labels: Sequence[str],
    imaginary_atol: float = 1.0e-10,
) -> NDArray[np.float64]:
    """Return ``P(t) = (rho_22-rho_11)/2`` in a declared site basis."""

    if imaginary_atol < 0.0:
        raise ValueError("imaginary_atol must be nonnegative")
    require_structurally_valid(trajectory)
    site_1 = _basis_indices(trajectory.electronic_basis, site_1_labels)
    site_2 = _basis_indices(trajectory.electronic_basis, site_2_labels)
    if set(site_1).intersection(site_2):
        raise ValueError("site label collections must be disjoint")

    diagonal = np.diagonal(trajectory.electron_1rdm, axis1=-2, axis2=-1)
    imaginary = float(np.max(np.abs(diagonal.imag)))
    if imaginary > imaginary_atol:
        raise ValueError(
            "electronic populations have an imaginary component above tolerance"
        )
    population_1 = np.sum(diagonal[:, site_1].real, axis=1)
    population_2 = np.sum(diagonal[:, site_2].real, axis=1)
    return np.asarray(0.5 * (population_2 - population_1), dtype=float)


def hann_spectrum(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    *,
    convention: SpectrumConvention,
    uniform_atol: float = 1.0e-12,
    uniform_rtol: float = 1.0e-10,
) -> Spectrum:
    """Return a Hann-apodized DFT with every normalization choice explicit."""

    samples = np.asarray(time, dtype=float)
    values = np.asarray(signal, dtype=float)
    if samples.ndim != 1 or values.ndim != 1 or samples.shape != values.shape:
        raise ValueError("time and signal must be same-length one-dimensional arrays")
    if samples.size < 2:
        raise ValueError("at least two time samples are required")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(values)):
        raise ValueError("time and signal must be finite")
    steps = np.diff(samples)
    if np.any(steps <= 0.0):
        raise ValueError("time must be strictly increasing")
    step = float(steps[0])
    if not np.allclose(steps, step, atol=uniform_atol, rtol=uniform_rtol):
        raise ValueError("hann_spectrum requires a uniform time grid")

    prepared = values.copy()
    if convention.detrend == "mean":
        prepared -= float(np.mean(prepared))
    window = np.hanning(prepared.size)
    prepared *= window

    if convention.sided == "one-sided":
        transform = np.fft.rfft(prepared)
        frequency = np.fft.rfftfreq(prepared.size, d=step)
    else:
        transform = np.fft.fft(prepared)
        frequency = np.fft.fftfreq(prepared.size, d=step)

    if convention.normalization == "forward":
        transform = transform / prepared.size
    elif convention.normalization == "ortho":
        transform = transform / np.sqrt(prepared.size)

    if convention.angular_frequency:
        frequency = 2.0 * np.pi * frequency
    complex_transform = np.asarray(transform, dtype=np.complex128)
    return Spectrum(
        frequency=np.asarray(frequency, dtype=float),
        transform=complex_transform,
        magnitude=np.asarray(np.abs(complex_transform), dtype=float),
        convention=convention,
    )


__all__ = ["dimer_polarization", "hann_spectrum"]
