"""One full SIAM exact artifact supplies every verified prefix view."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.time_dynamics.diagnostics.siam_exact_reference_store import (
    SIAM_EXACT_PREFIX_VIEW_V2,
    SIAMExactReferenceStore,
)
from pipelines.time_dynamics.siam_screening import PILOT_NB4
from pipelines.time_dynamics.siam_seed import write_siam_seed_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nb4_prefix_view_reuses_verified_full_reference_bytes(tmp_path: Path) -> None:
    seed = tmp_path / "seed.json"
    write_siam_seed_manifest(seed, PILOT_NB4)
    root = tmp_path / "exact"
    store = SIAMExactReferenceStore(root)
    full = store.get_or_build(seed_path=seed, times=PILOT_NB4.times)
    receipt_path = tmp_path / "prefix_t1.json"

    prefix = store.prefix_view(
        full,
        times=PILOT_NB4.times[:101],
        receipt_path=receipt_path,
    )

    assert prefix.key == full.key
    assert prefix.data_path == full.data_path
    assert prefix.manifest_path == full.manifest_path
    assert len(prefix.times) == 101
    assert prefix.times[-1] == 1.0
    np.testing.assert_array_equal(prefix.sector_states, full.sector_states[:101])
    np.testing.assert_array_equal(prefix.observable("C_4"), full.observable("C_4")[:101])
    assert list(root.rglob("reference.npz")) == [full.data_path]
    assert list(root.rglob("manifest.json")) == [full.manifest_path]

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == SIAM_EXACT_PREFIX_VIEW_V2
    assert receipt["materialized_data_file"] is None
    assert receipt["source"]["reference_key"] == full.key
    assert receipt["source"]["manifest_sha256"] == _sha256(full.manifest_path)
    assert receipt["source"]["data_sha256"] == _sha256(full.data_path)
    assert receipt["prefix"]["point_count"] == 101
    assert receipt["prefix"]["t_final"] == 1.0
    loaded = store.load_prefix_view(receipt_path)
    np.testing.assert_array_equal(loaded.sector_states, prefix.sector_states)

    # The same immutable view write is idempotent and still does not create a
    # second exact artifact.
    repeated = store.prefix_view(
        full,
        times=PILOT_NB4.times[:101],
        receipt_path=receipt_path,
    )
    assert repeated.receipt == prefix.receipt
    assert len(list(root.rglob("reference.npz"))) == 1


def test_prefix_view_rejects_nonprefix_grid_and_full_data_tampering(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "seed.json"
    write_siam_seed_manifest(seed, PILOT_NB4)
    store = SIAMExactReferenceStore(tmp_path / "exact")
    full = store.get_or_build(seed_path=seed, times=PILOT_NB4.times)

    shifted = list(PILOT_NB4.times[:101])
    shifted[-1] += 0.001
    with pytest.raises(ValueError, match="exact leading prefix"):
        store.prefix_view(full, times=shifted)

    valid_receipt = tmp_path / "valid_prefix.json"
    store.prefix_view(
        full,
        times=PILOT_NB4.times[:101],
        receipt_path=valid_receipt,
    )
    tampered_receipt = json.loads(valid_receipt.read_text(encoding="utf-8"))
    tampered_receipt["prefix"]["t_final"] = 0.99
    valid_receipt.write_text(
        json.dumps(tampered_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="receipt digest"):
        store.load_prefix_view(valid_receipt)

    content = bytearray(full.data_path.read_bytes())
    content[-1] ^= 0x01
    full.data_path.write_bytes(bytes(content))
    with pytest.raises(ValueError, match="data digest"):
        store.prefix_view(full, times=PILOT_NB4.times[:101])
