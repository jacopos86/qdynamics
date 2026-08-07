from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra import (
    HHResponseLayout,
    build_hh_current_observable_bundle,
    directed_hh_current_edges,
    resolve_hh_current_hopping_from_sources,
)
from pipelines.qse_spectra.__main__ import main as qse_main
from pipelines.qse_spectra.hh_current_observables import HHCurrentObservableError


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _term_map(observable) -> dict[str, complex]:
    assert observable.polynomial is not None
    return {str(term.pw2strng()): complex(term.p_coeff) for term in observable.polynomial.return_polynomial()}


def _minimal_hh_settings_payload(*, t: float = 1.0, boundary: str = "open") -> dict[str, object]:
    return {
        "settings": {
            "problem": "hh",
            "L": 2,
            "t": float(t),
            "u": 4.0,
            "omega0": 1.0,
            "g_ep": 0.25,
            "dv": 0.0,
            "n_ph_max": 1,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": str(boundary),
        },
        "adapt_vqe": {"num_particles": {"n_up": 1, "n_dn": 1}},
    }


def test_hh_current_builder_uses_full_register_and_explicit_peierls_contact_policy() -> None:
    layout = HHResponseLayout(
        num_sites=2,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        total_qubits=6,
        num_particles=(1, 1),
    )

    bundle = build_hh_current_observable_bundle(layout=layout, hopping_amplitude=1.0)

    assert [obs.name for obs in bundle.observables] == ["hh_J[positive_chain]", "hh_K[positive_chain]"]
    assert bundle.current_labels == ("hh_J[positive_chain]",)
    assert bundle.contact_label == "hh_K[positive_chain]"
    assert bundle.metadata["schema_version"] == "hh_current_observables_v1"
    assert bundle.metadata["edge_orientation"] == "positive_chain"
    assert bundle.metadata["directed_edges"] == [[0, 1]]
    assert bundle.metadata["peierls_policy"] == "standard_hh_1d_charge_peierls"
    assert bundle.metadata["contact_policy"] == "peierls_second_derivative_record_only"
    assert bundle.metadata["current_zero_operator"] is False
    assert bundle.metadata["contact_zero_operator"] is False

    current_terms = _term_map(bundle.observables[0])
    # Full HH register: q5,q4 are phonon qubits and remain identity in every current term.
    assert all(label.startswith("ee") and len(label) == 6 for label in current_terms)
    assert current_terms["eeeexy"] == pytest.approx(0.5)
    assert current_terms["eeeeyx"] == pytest.approx(-0.5)
    assert current_terms["eexyee"] == pytest.approx(0.5)
    assert current_terms["eeyxee"] == pytest.approx(-0.5)

    contact_terms = _term_map(bundle.observables[1])
    assert all(label.startswith("ee") and len(label) == 6 for label in contact_terms)
    assert contact_terms["eeeexx"] == pytest.approx(0.5)
    assert contact_terms["eeeeyy"] == pytest.approx(0.5)
    assert contact_terms["eexxee"] == pytest.approx(0.5)
    assert contact_terms["eeyyee"] == pytest.approx(0.5)
    assert bundle.observables[0].metadata["source"] == "hh_current_observables_v1"
    assert bundle.observables[1].metadata["zero_operator"] is False


def test_hh_current_zero_hopping_is_explicit_zero_operator() -> None:
    layout = HHResponseLayout(
        num_sites=2,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        total_qubits=6,
    )

    bundle = build_hh_current_observable_bundle(layout=layout, hopping_amplitude=0.0)

    assert bundle.metadata["current_zero_operator"] is True
    assert bundle.metadata["contact_zero_operator"] is True
    assert _term_map(bundle.observables[0]) == {"eeeeee": 0.0j}
    assert _term_map(bundle.observables[1]) == {"eeeeee": 0.0j}


def test_hh_current_fails_closed_on_ambiguous_periodic_l2_orientation() -> None:
    layout = HHResponseLayout(
        num_sites=2,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="periodic",
        total_qubits=6,
    )

    with pytest.raises(HHCurrentObservableError, match="ambiguous"):
        directed_hh_current_edges(layout, edge_orientation="positive_chain")


def test_hh_current_hopping_resolver_uses_settings_and_rejects_conflicts(tmp_path: Path) -> None:
    artifact_path = _write_json(tmp_path / "hh_artifact.json", _minimal_hh_settings_payload(t=1.25))

    resolved = resolve_hh_current_hopping_from_sources(sources={"hamiltonian_json": artifact_path})

    assert resolved.hopping_amplitude == pytest.approx(1.25)
    assert resolved.metadata["resolved_from"][0]["field"] == "t_or_J"

    conflict_path = _write_json(tmp_path / "hh_conflict.json", _minimal_hh_settings_payload(t=2.0))
    with pytest.raises(HHCurrentObservableError, match="conflict"):
        resolve_hh_current_hopping_from_sources(
            sources={"hamiltonian_json": artifact_path, "basis_artifact_json": conflict_path}
        )


def test_cli_hh_current_response_generates_conductivity_payload(tmp_path: Path) -> None:
    artifact_path = _write_json(tmp_path / "hh_settings_artifact.json", _minimal_hh_settings_payload(t=1.0))
    out_path = tmp_path / "qse_hh_current.json"

    assert qse_main(
        [
            "--hamiltonian-json",
            str(artifact_path),
            "--state-bitstring",
            "001001",
            "--hh-current-response",
            "--spectral-grid-min",
            "0",
            "--spectral-grid-max",
            "4",
            "--spectral-grid-num",
            "21",
            "--spectral-eta",
            "0.1",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "qse_spectra_v1"
    assert data["settings"]["conductivity_response_enabled"] is True
    assert data["settings"]["hh_current_response_enabled"] is True
    assert data["settings"]["conductivity_current_labels"] == ["hh_J[positive_chain]"]
    assert any(
        item["source_schema"] == "hh_current_observables_v1"
        for item in data["input"]["transition_observables"]
    )

    names = [row["name"] for row in data["transition_observables"]]
    assert "hh_J[positive_chain]" in names
    assert "hh_K[positive_chain]" in names
    conductivity = data["qse_conductivity_response_v1"]
    assert conductivity["schema_version"] == "qse_conductivity_response_v1"
    assert conductivity["peierls_policy"]["name"] == "standard_hh_1d_charge_peierls"
    channel = conductivity["channels"][0]
    assert channel["channel_kind"] == "hh_longitudinal_charge"
    assert channel["current_label"] == "hh_J[positive_chain]"
    assert channel["contact_label"] == "hh_K[positive_chain]"
    assert channel["contact_term"]["status"] == "evaluated"
    assert channel["current_source"]["status"] == "evaluated"
