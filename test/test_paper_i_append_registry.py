from __future__ import annotations

from dataclasses import replace
import hashlib

from pipelines.reporting import paper_i_append_registry as registry
from pipelines.reporting import paper_i_run_summary as summary
from pipelines.reporting.paper_i_run_summary import (
    PaperIAppendResolutionRequest,
    PaperIReferenceState,
)
from pipelines.static_adapt.estimator_call_ledger import (
    projective_state_fingerprint,
)


STRONG_WEAK_PROBLEM_SHA256 = (
    "5197b317fe67b5eedabd726e29b897260c18bda9eaf6bc9cc05cf3b0a468b65d"
)
EXPECTED_ROUND_50_REPORTING = {
    "weak_weak": (260, 224, 1045, 474, 313231, 9.416688540042628e-10),
    "intermediate_weak": (
        234,
        197,
        1132,
        448,
        372793,
        3.1970953107141042e-9,
    ),
    "strong_weak_u8": (250, 210, 1112, 462, 129405, 8.010331058461162e-7),
    "weak_strong": (274, 237, 909, 508, 1154148, 0.0006059547563721512),
    "intermediate_strong": (
        272,
        241,
        962,
        492,
        1284661,
        0.00010857173613953996,
    ),
    "strong_strong_u8": (
        242,
        209,
        927,
        448,
        1080929,
        1.161507312552601e-8,
    ),
}


def test_locked_registry_resolves_complete_typed_append_source() -> None:
    raw = registry.REGISTRY_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == registry.REGISTRY_SHA256
    resolver = registry.LockedPaperIAppendRegistry()
    records = resolver._load_records()
    assert len(records) == 6
    record = records[STRONG_WEAK_PROBLEM_SHA256]
    comparison = registry._comparison_contract(record)
    reference = registry._reference_state(record)

    source = resolver.resolve_canonical_append(
        PaperIAppendResolutionRequest(
            comparison_contract=comparison,
            exact_same_cutoff_energy=float(
                record["exact_same_cutoff_energy"]
            ),
            reference_state=reference,
        )
    )

    assert source is not None
    assert source.horizon_scope == "deliberately_stopped_prefix"
    assert len(source.accepted_error_trace) == 50
    assert len(source.accepted_prefixes) == 50
    assert source.accepted_prefixes[10].algorithmic_work.s_alg == 16393
    assert (
        source.accepted_error_trace[10].checkpoint_sha256
        == source.accepted_prefixes[10].checkpoint_sha256
    )


def test_locked_registry_pins_adopted_v6_round_50_reporting() -> None:
    assert registry.SOURCE_ADOPTION_SHA256 == (
        "3373d5d54d267a0f5f75af7efb63518463a11c308c69d595b40f3516983b8cfc"
    )
    assert registry.SOURCE_PARTIAL_REPORT_PROVENANCE_SHA256 == (
        "485cc623974a5e2000937b76f576f70fe81427f7ceaf3655bfb8c2f2af0c9691"
    )
    assert registry.SOURCE_VALIDATION_SHA256 == (
        "2a55c1509e9112e75c9a87201a8a7ea511529a0659d7fb58ba08b6f9875eb853"
    )
    assert registry.SOURCE_PACKAGE_MANIFEST_SHA256 == (
        "75063a0d8de86518d91a55283e025037229d20c185681db74b79175f9b9e6176"
    )

    records = registry.LockedPaperIAppendRegistry()._load_records()
    records_by_regime = {
        str(record["regime"]): record for record in records.values()
    }
    assert set(records_by_regime) == set(EXPECTED_ROUND_50_REPORTING)

    for regime, expected in EXPECTED_ROUND_50_REPORTING.items():
        record = records_by_regime[regime]
        reporting = registry._reporting_resources(record)
        source = registry._run_source(
            record,
            comparison=registry._comparison_contract(record),
            reference=registry._reference_state(record),
        )
        terminal = record["accepted_prefixes"][-1]
        observed = (
            reporting["compiled_two_qubit_count"],
            reporting["compiled_two_qubit_depth"],
            reporting["compiled_total_depth"],
            reporting["pauli_one_qubit_work"],
            reporting["s_alg"],
            reporting["absolute_energy_error"],
        )

        assert observed == expected
        assert reporting["policy"] == "fixed_controller_round_50_v1"
        assert reporting["controller_round"] == 50
        assert reporting["qiskit_validated"] is True
        assert len(source.accepted_prefixes) == 50
        assert source.accepted_prefixes[-1].controller_round == 50
        assert (
            source.accepted_prefixes[-1].algorithmic_work.s_alg
            == reporting["s_alg"]
        )
        assert terminal["controller_round"] == 50
        assert terminal["algorithmic_work"]["s_alg"] == reporting["s_alg"]
        assert (
            terminal["absolute_energy_error"]
            == reporting["absolute_energy_error"]
        )
        assert "stationary_core_v6_" in record["source"]["archive_path"]


def test_default_append_sentinel_resolves_the_locked_registry() -> None:
    resolver = registry.LockedPaperIAppendRegistry()
    record = resolver._load_records()[STRONG_WEAK_PROBLEM_SHA256]
    request = PaperIAppendResolutionRequest(
        comparison_contract=registry._comparison_contract(record),
        exact_same_cutoff_energy=float(record["exact_same_cutoff_energy"]),
        reference_state=registry._reference_state(record),
    )

    source, early_observation = summary._resolve_append_source(
        summary.CANONICAL_APPEND_REFERENCE,
        request,
    )

    assert early_observation is None
    assert source is not None
    assert len(source.accepted_prefixes) == 50


def test_locked_registry_requires_exact_reference_and_comparison() -> None:
    resolver = registry.LockedPaperIAppendRegistry()
    record = resolver._load_records()[STRONG_WEAK_PROBLEM_SHA256]
    comparison = registry._comparison_contract(record)
    reference = registry._reference_state(record)

    assert (
        resolver.resolve_canonical_append(
            PaperIAppendResolutionRequest(
                comparison_contract=replace(
                    comparison,
                    optimizer_maxiter=comparison.optimizer_maxiter + 1,
                ),
                exact_same_cutoff_energy=float(
                    record["exact_same_cutoff_energy"]
                ),
                reference_state=reference,
            )
        )
        is None
    )
    assert (
        resolver.resolve_canonical_append(
            PaperIAppendResolutionRequest(
                comparison_contract=comparison,
                exact_same_cutoff_energy=float(
                    record["exact_same_cutoff_energy"]
                ),
                reference_state=PaperIReferenceState(
                    amplitudes_real=(1.0,)
                    + (0.0,) * ((1 << reference.qubit_count) - 1),
                    amplitudes_imaginary=(0.0,)
                    * (1 << reference.qubit_count),
                    qubit_count=reference.qubit_count,
                    source_label="different-authenticated-reference",
                    state_fingerprint=projective_state_fingerprint(
                        (1.0,)
                        + (0.0,) * ((1 << reference.qubit_count) - 1)
                    ),
                ),
            )
        )
        is None
    )
