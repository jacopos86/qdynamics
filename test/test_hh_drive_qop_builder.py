from __future__ import annotations

import pytest

from pipelines.exact_bench.hh_seq_transition_utils import build_time_dependent_sparse_qop


def _coeff_map(qop) -> dict[str, complex]:
    out: dict[str, complex] = {}
    for label, coeff in qop.to_list():
        out[str(label)] = complex(coeff)
    return out


def test_drive_qop_builder_combines_static_and_drive_coefficients() -> None:
    qop = build_time_dependent_sparse_qop(
        ordered_labels_exyz=["ze", "ee"],
        static_coeff_map_exyz={"ze": 1.0 + 0.0j, "ee": 0.0 + 0.0j},
        drive_coeff_map_exyz={"ee": 0.5 + 0.0j, "ez": -0.2 + 0.0j},
    )

    coeffs = _coeff_map(qop)
    assert coeffs["ZI"] == pytest.approx(1.0 + 0.0j)
    assert coeffs["II"] == pytest.approx(0.5 + 0.0j)
    assert coeffs["IZ"] == pytest.approx(-0.2 + 0.0j)


def test_drive_qop_builder_identity_fallback_when_all_terms_cancel() -> None:
    qop = build_time_dependent_sparse_qop(
        ordered_labels_exyz=["ee"],
        static_coeff_map_exyz={"ee": 0.0 + 0.0j},
        drive_coeff_map_exyz={},
    )
    coeffs = _coeff_map(qop)
    assert "I" in coeffs or "II" in coeffs
