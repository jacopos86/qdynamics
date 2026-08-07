from __future__ import annotations

import numpy as np

from src.quantum.pauli_actions import (
    apply_compiled_pauli,
    compile_pauli_action_exyz,
)
from src.quantum.vqe_latex_python_pairs import apply_pauli_string


def test_compiled_pauli_action_is_compact_at_h2o_nph3_width() -> None:
    action = compile_pauli_action_exyz("xyz" * 6, 18)

    assert action.retained_bytes <= 256


def test_compact_compiled_pauli_action_matches_reference_application() -> None:
    rng = np.random.default_rng(271828)
    state = rng.normal(size=32) + 1.0j * rng.normal(size=32)
    state = np.asarray(state / np.linalg.norm(state), dtype=complex)

    for label in ("exyze", "yyyyy", "zxxyz", "eeeee"):
        action = compile_pauli_action_exyz(label, 5)
        observed = apply_compiled_pauli(state, action)
        expected = apply_pauli_string(state, label)
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1.0e-14)
