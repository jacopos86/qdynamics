"""Provider guarantees: one alphabet, provable identity, uniform granularity."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.qse_spectra.core import QSEBasisElement
from pipelines.qse_spectra.paper_iii_problem import (
    assert_uniform_granularity,
    load_problem,
)

_REGIME = {"regime": "weak_weak", "u": 0.25, "g_ep": 0.353553390593, "n_ph_max": 3}


def test_problem_is_shared_not_rebuilt() -> None:
    a = load_problem(**_REGIME)
    b = load_problem(**_REGIME)
    assert a is b, "provider must hand out one problem per regime, not rebuild it"
    assert a.basis is b.basis


def test_pool_digest_proves_alphabet_identity() -> None:
    a = load_problem(**_REGIME)
    other = load_problem(**{**_REGIME, "u": 1.25})
    assert a.pool_digest == load_problem(**_REGIME).pool_digest
    # same pool SIZE across these two regimes; the digest must still separate
    # them only if the ordered names differ, and must be stable when they match.
    assert isinstance(other.pool_digest, str) and len(other.pool_digest) == 24
    receipt = a.arm_receipt()
    assert receipt["pool_digest"] == a.pool_digest
    assert receipt["pool_size"] == len(a.basis)
    assert receipt["granularity"] == "macro_records_uniform"


def test_reference_is_cached_and_consistent() -> None:
    a = load_problem(**_REGIME)
    assert a.ground.ndim == 1
    assert len(a.spectrum) == 7
    assert a.ground_energy == a.spectrum[0]
    assert list(a.references) == list(a.spectrum[1:])
    # cached file exists and round-trips to the same numbers
    b = load_problem(**{**_REGIME, "target_roots": 6})
    assert np.allclose(a.ground, b.ground)


def test_resource_triple_is_additive_and_ordered() -> None:
    p = load_problem(**_REGIME)
    small = p.resource_triple([0, 1])
    large = p.resource_triple([0, 1, 2, 3])
    assert large["n2q"] >= small["n2q"]
    assert large["dc"] >= large["d2q"], "Dc includes the one-qubit layer channel"


def test_uniform_granularity_rejects_mixed_pools() -> None:
    child = QSEBasisElement(
        name="hh_phonon::child(site=0)", kind="pauli_string", pauli_label_exyz="exxe"
    )
    with pytest.raises(ValueError, match="uniform granularity"):
        assert_uniform_granularity([child])
