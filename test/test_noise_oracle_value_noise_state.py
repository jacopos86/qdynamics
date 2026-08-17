from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.noise_oracle_runtime import (  # noqa: E402
    ExpectationOracle,
    OracleConfig,
)


def _iid_config(*, std: float = 0.25, seed: int = 702688422) -> OracleConfig:
    return OracleConfig(
        noise_mode="ideal",
        oracle_repeats=3,
        value_noise_model="gaussian_iid_v1",
        value_noise_std=std,
        value_noise_seed=seed,
    )


def _identity_problem() -> tuple[QuantumCircuit, SparsePauliOp]:
    return QuantumCircuit(1), SparsePauliOp.from_list([("I", 1.0)])


def test_value_noise_state_restore_continues_the_sequential_iid_stream() -> None:
    circuit, observable = _identity_problem()

    with ExpectationOracle(_iid_config()) as uninterrupted:
        first = uninterrupted.evaluate(circuit, observable)
        state = json.loads(
            json.dumps(uninterrupted.snapshot_value_noise_state(), sort_keys=True)
        )
        expected_next = uninterrupted.evaluate(circuit, observable)

    with ExpectationOracle(_iid_config()) as resumed:
        resumed.restore_value_noise_state(state)
        observed_next = resumed.evaluate(circuit, observable)

    assert first.metadata["value_noise"]["draw_index_start"] == 0
    assert first.metadata["value_noise"]["draw_index_stop"] == 3
    assert state["schema"] == "expectation_oracle_value_noise_rng_state_v1"
    assert state["effective_seed"] == 702688422
    assert state["seed_source"] == "value_noise_seed"
    assert state["bit_generator"]["class"] == "PCG64"
    assert state["draw_count"] == 3
    assert state["config"]["value_noise_model"] == "gaussian_iid_v1"
    assert state["config"]["value_noise_std"] == pytest.approx(0.25)
    assert state["policy"]["independence"] == "sequential_iid_draws_v1"
    assert (
        state["policy"]["draw_accounting"]
        == "one_draw_per_raw_expectation_sample_v1"
    )
    assert len(state["config_sha256"]) == 64
    assert len(state["policy_sha256"]) == 64
    assert len(state["payload_sha256"]) == 64
    assert observed_next.raw_values == expected_next.raw_values
    assert observed_next.metadata["value_noise"]["draw_index_start"] == 3
    assert observed_next.metadata["value_noise"]["draw_index_stop"] == 6


@pytest.mark.parametrize(
    "tamper",
    (
        lambda payload: payload.__setitem__("draw_count", 17),
        lambda payload: payload.__setitem__("effective_seed", 17),
        lambda payload: payload.__setitem__("seed_source", "oracle_config_seed"),
        lambda payload: payload["config"].__setitem__("value_noise_std", 0.5),
        lambda payload: payload["policy"].__setitem__(
            "draw_accounting", "frozen_keyed_draws"
        ),
        lambda payload: payload["bit_generator"]["state"]["state"].__setitem__(
            "state", 1
        ),
    ),
)
def test_value_noise_state_tamper_is_rejected_before_stream_mutation(tamper) -> None:
    circuit, observable = _identity_problem()
    with ExpectationOracle(_iid_config()) as source:
        source.evaluate(circuit, observable)
        state = source.snapshot_value_noise_state()

    tampered = copy.deepcopy(state)
    tamper(tampered)
    with ExpectationOracle(_iid_config()) as target:
        with pytest.raises(ValueError, match="value_noise_state_digest_mismatch"):
            target.restore_value_noise_state(tampered)
        after_rejection = target.evaluate(circuit, observable)
    with ExpectationOracle(_iid_config()) as fresh:
        expected_fresh = fresh.evaluate(circuit, observable)

    assert after_rejection.raw_values == expected_fresh.raw_values
    assert after_rejection.metadata["value_noise"]["draw_index_start"] == 0


def test_value_noise_state_rejects_an_authenticated_different_configuration() -> None:
    circuit, observable = _identity_problem()
    with ExpectationOracle(_iid_config(std=0.25)) as source:
        state = source.snapshot_value_noise_state()

    with ExpectationOracle(_iid_config(std=0.5)) as target:
        with pytest.raises(ValueError, match="value_noise_state_config_mismatch"):
            target.restore_value_noise_state(state)
        after_rejection = target.evaluate(circuit, observable)
    with ExpectationOracle(_iid_config(std=0.5)) as fresh:
        expected_fresh = fresh.evaluate(circuit, observable)

    assert after_rejection.raw_values == expected_fresh.raw_values
    assert after_rejection.metadata["value_noise"]["draw_index_start"] == 0


def test_value_noise_state_fails_closed_when_value_noise_is_disabled() -> None:
    with ExpectationOracle(OracleConfig(noise_mode="ideal")) as oracle:
        with pytest.raises(RuntimeError, match="value_noise_state_disabled"):
            oracle.snapshot_value_noise_state()
        with pytest.raises(RuntimeError, match="value_noise_state_disabled"):
            oracle.restore_value_noise_state({})
