from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra import (
    HHResponseLayout,
    build_hh_neutral_response_observable_bundle,
    computational_basis_state,
    density_baseline_from_state,
    resolve_hh_response_layout_from_sources,
)
from pipelines.qse_spectra.__main__ import main as qse_main
from pipelines.qse_spectra.hh_response_observables import HHResponseObservableError


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _pauli_label(nq: int, qubit: int, symbol: str) -> str:
    chars = ["e"] * int(nq)
    chars[int(nq) - 1 - int(qubit)] = str(symbol)
    return "".join(chars)


def _term_map(observable) -> dict[str, complex]:
    assert observable.polynomial is not None
    return {str(term.pw2strng()): complex(term.p_coeff) for term in observable.polynomial.return_polynomial()}


def _minimal_hh_settings_payload() -> dict[str, object]:
    return {
        "settings": {
            "problem": "hh",
            "L": 2,
            "t": 1.0,
            "u": 4.0,
            "omega0": 1.0,
            "g_ep": 0.25,
            "dv": 0.0,
            "n_ph_max": 1,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
        },
        "adapt_vqe": {"num_particles": {"n_up": 1, "n_dn": 1}},
    }


def test_hh_builder_constructs_density_phonon_and_mixed_observables() -> None:
    layout = HHResponseLayout(
        num_sites=2,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        total_qubits=6,
        num_particles=(1, 1),
    )

    bundle = build_hh_neutral_response_observable_bundle(
        layout=layout,
        channels=["nn", "XX", "PP", "nX", "C_nX"],
        form_factor=[1.0, -1.0],
        density_baseline=1.0,
        nx_separation=0,
    )

    assert [obs.name for obs in bundle.observables] == [
        "hh_n[custom]",
        "hh_X[custom]",
        "hh_P[custom]",
        "hh_C_nX[r=0]",
    ]
    assert [(ch.A_label, ch.B_label, ch.channel_kind) for ch in bundle.response_channels] == [
        ("hh_n[custom]", "hh_n[custom]", "nn"),
        ("hh_X[custom]", "hh_X[custom]", "XX"),
        ("hh_P[custom]", "hh_P[custom]", "PP"),
        ("hh_n[custom]", "hh_X[custom]", "nX"),
        ("hh_C_nX[r=0]", "hh_C_nX[r=0]", "C_nX"),
    ]

    n_terms = _term_map(bundle.observables[0])
    assert "eeeeee" not in n_terms
    assert n_terms[_pauli_label(6, 0, "z")] == pytest.approx(-0.5)
    assert n_terms[_pauli_label(6, 2, "z")] == pytest.approx(-0.5)
    assert n_terms[_pauli_label(6, 1, "z")] == pytest.approx(0.5)
    assert n_terms[_pauli_label(6, 3, "z")] == pytest.approx(0.5)

    x_terms = _term_map(bundle.observables[1])
    assert x_terms[_pauli_label(6, 4, "x")] == pytest.approx(1.0)
    assert x_terms[_pauli_label(6, 5, "x")] == pytest.approx(-1.0)

    p_terms = _term_map(bundle.observables[2])
    assert p_terms[_pauli_label(6, 4, "y")] == pytest.approx(1.0)
    assert p_terms[_pauli_label(6, 5, "y")] == pytest.approx(-1.0)

    cnx_terms = _term_map(bundle.observables[3])
    assert any(label.count("x") == 1 and label.count("z") == 1 for label in cnx_terms)
    assert bundle.metadata["schema_version"] == "hh_neutral_response_observables_v1"
    assert bundle.metadata["response_channel_count"] == 5
    assert bundle.observables[0].metadata["source"] == "hh_neutral_response_observables_v1"


def test_pure_phonon_channels_do_not_require_density_baseline() -> None:
    layout = HHResponseLayout(
        num_sites=2,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        total_qubits=6,
    )

    bundle = build_hh_neutral_response_observable_bundle(
        layout=layout,
        channels="XX,PP",
        form_factor="site:0",
    )

    assert [obs.name for obs in bundle.observables] == ["hh_X[site_0]", "hh_P[site_0]"]
    assert bundle.metadata["density_baseline"] is None
    assert all(obs.metadata["density_baseline"] is None for obs in bundle.observables)


@pytest.mark.parametrize("bad_spec", ["site:not_an_int", "obc_sine:1.5"])
def test_bad_hh_form_factor_indices_raise_builder_error(bad_spec: str) -> None:
    layout = HHResponseLayout(
        num_sites=2,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        total_qubits=6,
    )

    with pytest.raises(HHResponseObservableError, match="integer"):
        build_hh_neutral_response_observable_bundle(
            layout=layout,
            channels="XX",
            form_factor=bad_spec,
        )


def test_hh_density_baseline_can_be_inferred_from_prepared_state() -> None:
    layout = HHResponseLayout(
        num_sites=2,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        total_qubits=6,
    )
    # q_(5)..q_0 with occupied blocked spin-orbitals q0 and q3: total density 2 over L=2.
    psi = computational_basis_state(6, "001001")

    assert density_baseline_from_state(layout, psi) == pytest.approx(1.0)

    bundle = build_hh_neutral_response_observable_bundle(
        layout=layout,
        channels="nn,nX",
        form_factor="staggered",
        prepared_state=psi,
    )
    assert bundle.metadata["density_baseline"] == pytest.approx(1.0)
    assert [ch.channel_kind for ch in bundle.response_channels] == ["nn", "nX"]


def test_hh_layout_resolver_uses_existing_settings_and_rejects_missing_layout(tmp_path: Path) -> None:
    artifact_path = _write_json(tmp_path / "hh_artifact.json", _minimal_hh_settings_payload())

    layout = resolve_hh_response_layout_from_sources(
        expected_nq=6,
        sources={"basis_artifact_json": artifact_path},
    )

    assert layout.num_sites == 2
    assert layout.n_ph_max == 1
    assert layout.boson_encoding == "binary"
    assert layout.ordering == "blocked"
    assert layout.boundary == "open"
    assert layout.num_particles == (1, 1)

    plain_ham_path = _write_json(
        tmp_path / "plain_ham.json",
        {"terms": [{"pauli_exyz": "z", "coeff_re": 1.0, "coeff_im": 0.0}]},
    )
    with pytest.raises(HHResponseObservableError, match="explicit --transition-observable-json"):
        resolve_hh_response_layout_from_sources(
            expected_nq=1,
            sources={"hamiltonian_json": plain_ham_path},
        )


def test_cli_generates_hh_response_channels_from_code_facing_names(tmp_path: Path) -> None:
    artifact_path = _write_json(tmp_path / "hh_settings_artifact.json", _minimal_hh_settings_payload())
    out_path = tmp_path / "qse_hh_response.json"

    assert qse_main(
        [
            "--hamiltonian-json",
            str(artifact_path),
            "--state-bitstring",
            "001001",
            "--hh-neutral-response-channel",
            "nn,nX",
            "--hh-response-form-factor",
            "staggered",
            "--spectral-grid-min",
            "0",
            "--spectral-grid-max",
            "4",
            "--spectral-grid-num",
            "21",
            "--spectral-eta",
            "0.1",
            "--response-time-grid-min",
            "0",
            "--response-time-grid-max",
            str(math.pi / 8.0),
            "--response-time-grid-num",
            "2",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    provenance = data["input"]["transition_observables"][0]
    assert provenance["source_schema"] == "hh_neutral_response_observables_v1"
    assert provenance["channels_requested"] == ["nn", "nX"]
    assert provenance["layout"]["num_sites"] == 2
    assert data["settings"]["hh_neutral_response_channels"] == ["nn,nX"]

    transition_names = [row["name"] for row in data["transition_observables"]]
    assert "hh_n[staggered]" in transition_names
    assert "hh_X[staggered]" in transition_names
    response_channels = data["qse_response_functions_v1"]["channels"]
    assert [(row["A_label"], row["B_label"], row["channel_kind"]) for row in response_channels] == [
        ("hh_n[staggered]", "hh_n[staggered]", "nn"),
        ("hh_n[staggered]", "hh_X[staggered]", "nX"),
    ]


def test_cli_hh_response_fails_closed_without_layout_but_explicit_json_still_works(tmp_path: Path) -> None:
    ham_path = _write_json(
        tmp_path / "one_qubit_ham.json",
        {"terms": [{"pauli_exyz": "z", "coeff_re": -1.0, "coeff_im": 0.0}]},
    )

    with pytest.raises(SystemExit) as exc_info:
        qse_main(
            [
                "--hamiltonian-json",
                str(ham_path),
                "--state-bitstring",
                "0",
                "--operator-basis-label",
                "X",
                "--hh-neutral-response-channel",
                "nn",
                "--spectral-grid-min",
                "0",
                "--spectral-grid-max",
                "4",
                "--spectral-grid-num",
                "21",
                "--spectral-eta",
                "0.1",
                "--response-time-grid-min",
                "0",
                "--response-time-grid-max",
                "1",
                "--response-time-grid-num",
                "2",
            ]
        )
    assert exc_info.value.code == 2

    obs_path = _write_json(
        tmp_path / "explicit_observables.json",
        {"transition_observables": [{"kind": "pauli_string", "name": "probe", "pauli_exyz": "x"}]},
    )
    out_path = tmp_path / "explicit_response.json"
    assert qse_main(
        [
            "--hamiltonian-json",
            str(ham_path),
            "--state-bitstring",
            "0",
            "--operator-basis-label",
            "X",
            "--transition-observable-json",
            str(obs_path),
            "--response-functions",
            "--response-channel",
            "probe:probe:XX",
            "--spectral-grid-min",
            "0",
            "--spectral-grid-max",
            "4",
            "--spectral-grid-num",
            "21",
            "--spectral-eta",
            "0.1",
            "--response-time-grid-min",
            "0",
            "--response-time-grid-max",
            "1",
            "--response-time-grid-num",
            "2",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["input"]["transition_observables"][0]["source_schema"] == "transition_observables"
    assert data["qse_response_functions_v1"]["channels"][0]["A_label"] == "probe"

