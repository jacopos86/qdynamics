from __future__ import annotations

import numpy as np

from pipelines.open_dynamics import (
    SpectrumConvention,
    dimer_polarization,
    hann_spectrum,
)


def test_polarization_and_hann_spectrum_match_direct_calculation(
    paper5_vertical_slice,
) -> None:
    trajectory = paper5_vertical_slice.exact
    polarization = dimer_polarization(
        trajectory,
        site_1_labels=("site_1_spin_up",),
        site_2_labels=("site_2_spin_up",),
    )
    direct = 0.5 * (
        trajectory.electron_1rdm[:, 1, 1].real
        - trajectory.electron_1rdm[:, 0, 0].real
    )
    np.testing.assert_allclose(polarization, direct, atol=1e-14)

    convention = SpectrumConvention(
        normalization="forward",
        detrend="none",
        sided="one-sided",
        angular_frequency=True,
    )
    spectrum = hann_spectrum(
        trajectory.time,
        polarization,
        convention=convention,
    )
    direct_transform = np.fft.rfft(
        polarization * np.hanning(polarization.size)
    ) / polarization.size
    direct_frequency = 2.0 * np.pi * np.fft.rfftfreq(
        polarization.size,
        d=trajectory.time[1] - trajectory.time[0],
    )
    np.testing.assert_allclose(spectrum.transform, direct_transform, atol=1e-15)
    np.testing.assert_allclose(spectrum.frequency, direct_frequency, atol=1e-15)
    np.testing.assert_allclose(spectrum.magnitude, np.abs(direct_transform))
