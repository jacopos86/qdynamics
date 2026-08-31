"""One full SIAM exact artifact supplies a digest-bound continuation slice."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.time_dynamics.diagnostics.siam_exact_reference_store import (
    SIAM_EXACT_SLICE_VIEW_V2,
    SIAMExactReferenceStore,
)
from pipelines.time_dynamics.paper_ii_runs import (
    SIAM_NB4_CONTINUATION_TIMES,
    SIAM_NB4_FULL_TIMES,
    SIAM_NB4_SCREEN_TIMES,
)
from pipelines.time_dynamics.siam_screening import PILOT_NB4
from pipelines.time_dynamics.siam_seed import write_siam_seed_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_digest(core: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_nb4_full_reference_supplies_screen_and_continuation_without_new_npz(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "seed.json"
    write_siam_seed_manifest(seed, PILOT_NB4)
    root = tmp_path / "exact"
    store = SIAMExactReferenceStore(root)
    full = store.get_or_build(seed_path=seed, times=SIAM_NB4_FULL_TIMES)
    prefix_receipt = tmp_path / "screen_reference.json"
    slice_receipt = tmp_path / "continuation_reference.json"

    screen = store.prefix_view(
        full,
        times=SIAM_NB4_SCREEN_TIMES,
        receipt_path=prefix_receipt,
    )
    continuation = store.slice_view(
        full,
        times=SIAM_NB4_CONTINUATION_TIMES,
        receipt_path=slice_receipt,
    )

    assert screen.key == continuation.key == full.key
    assert screen.data_path == continuation.data_path == full.data_path
    np.testing.assert_array_equal(screen.sector_states, full.sector_states[:101])
    np.testing.assert_array_equal(
        continuation.sector_states,
        full.sector_states[100:],
    )
    np.testing.assert_array_equal(
        continuation.observable("Sigma_4"),
        full.observable("Sigma_4")[100:],
    )
    assert list(root.rglob("reference.npz")) == [full.data_path]
    assert list(root.rglob("manifest.json")) == [full.manifest_path]

    receipt = json.loads(slice_receipt.read_text(encoding="utf-8"))
    assert receipt["schema"] == SIAM_EXACT_SLICE_VIEW_V2
    assert receipt["materialized_data_file"] is None
    assert receipt["source"]["reference_key"] == full.key
    assert receipt["source"]["manifest_sha256"] == _sha256(full.manifest_path)
    assert receipt["source"]["data_sha256"] == _sha256(full.data_path)
    assert receipt["slice"]["start_index"] == 100
    assert receipt["slice"]["stop_index_exclusive"] == 501
    assert receipt["slice"]["point_count"] == 401
    assert receipt["slice"]["t_initial"] == 1.0
    assert receipt["slice"]["t_final"] == 5.0

    loaded = store.load_slice_view(slice_receipt)
    np.testing.assert_array_equal(loaded.sector_states, continuation.sector_states)
    assert loaded.receipt == continuation.receipt


def test_slice_view_rejects_shifted_noncontiguous_and_foreign_grids(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "seed.json"
    write_siam_seed_manifest(seed, PILOT_NB4)
    store = SIAMExactReferenceStore(tmp_path / "exact")
    full = store.get_or_build(seed_path=seed, times=SIAM_NB4_FULL_TIMES)

    shifted = list(SIAM_NB4_CONTINUATION_TIMES)
    shifted[1] += 1.0e-12
    with pytest.raises(ValueError, match="exact contiguous slice"):
        store.slice_view(full, times=shifted)

    noncontiguous = SIAM_NB4_FULL_TIMES[100:110:2]
    with pytest.raises(ValueError, match="exact contiguous slice"):
        store.slice_view(full, times=noncontiguous)

    signed_zero = (-0.0, *SIAM_NB4_FULL_TIMES[1:3])
    with pytest.raises(ValueError, match="exact contiguous slice"):
        store.slice_view(full, times=signed_zero)

    with pytest.raises(ValueError, match="unique exact start"):
        store.slice_view(full, times=(5.01,))

    foreign_store = SIAMExactReferenceStore(tmp_path / "foreign_exact")
    with pytest.raises(ValueError, match="originate from this exact-reference store"):
        foreign_store.slice_view(full, times=SIAM_NB4_CONTINUATION_TIMES)


def test_slice_view_receipt_binds_grid_arrays_and_full_source(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "seed.json"
    write_siam_seed_manifest(seed, PILOT_NB4)
    store = SIAMExactReferenceStore(tmp_path / "exact")
    full = store.get_or_build(seed_path=seed, times=SIAM_NB4_FULL_TIMES)
    receipt_path = tmp_path / "continuation_reference.json"
    store.slice_view(
        full,
        times=SIAM_NB4_CONTINUATION_TIMES,
        receipt_path=receipt_path,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["slice"]["start_index"] = 99
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="receipt digest"):
        store.load_slice_view(receipt_path)

    valid_path = tmp_path / "valid_continuation_reference.json"
    store.slice_view(
        full,
        times=SIAM_NB4_CONTINUATION_TIMES,
        receipt_path=valid_path,
    )
    source_tamper = json.loads(valid_path.read_text(encoding="utf-8"))
    source_tamper["source"]["data_sha256"] = "0" * 64
    source_core = dict(source_tamper)
    source_core.pop("receipt_sha256")
    source_tamper["receipt_sha256"] = _receipt_digest(source_core)
    valid_path.write_text(
        json.dumps(source_tamper, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="receipt/source mismatch"):
        store.load_slice_view(valid_path)

    valid_path = tmp_path / "valid_continuation_reference_2.json"
    store.slice_view(
        full,
        times=SIAM_NB4_CONTINUATION_TIMES,
        receipt_path=valid_path,
    )
    content = bytearray(full.data_path.read_bytes())
    content[-1] ^= 0x01
    full.data_path.write_bytes(bytes(content))
    with pytest.raises(ValueError, match="data digest"):
        store.load_slice_view(valid_path)
