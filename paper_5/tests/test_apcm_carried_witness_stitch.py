from __future__ import annotations

import numpy as np
import pytest

from paper5.stability.apcm_carried_witness_stitch import _load_and_stitch


def _segment(path, times: np.ndarray) -> None:
    row_count = times.size
    np.savez_compressed(
        path,
        times=times,
        carried_states=np.zeros((row_count, 2)),
        approximate_archive_coordinates=np.zeros((row_count, 31)),
        exact_archive_coordinates=np.zeros((row_count, 31)),
        minimum_unshifted_eigenvalues=np.zeros(row_count),
        minimum_shifted_lower_bounds=np.ones(row_count),
        completion_correction_norms=np.zeros(row_count),
        critical_modes=np.zeros(row_count, dtype=int),
    )


def test_stitch_can_record_an_explicit_mixed_step_grid(tmp_path) -> None:
    coarse = tmp_path / "coarse.npz"
    fine = tmp_path / "fine.npz"
    _segment(coarse, np.asarray([0.0, 0.01, 0.02]))
    _segment(fine, np.asarray([0.02, 0.025, 0.03]))

    with pytest.raises(ValueError, match="trajectory gap"):
        _load_and_stitch([coarse, fine], time_step=0.01)

    stitched, _ = _load_and_stitch(
        [coarse, fine],
        time_step=0.01,
        allow_variable_time_step=True,
    )

    np.testing.assert_allclose(
        stitched["times"],
        np.asarray([0.0, 0.01, 0.02, 0.025, 0.03]),
    )
