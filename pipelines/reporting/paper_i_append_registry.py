"""Source-locked canonical Append comparator registry.

The registry is a compact reporting input derived explicitly from the six
adopted v6 projected-singleton Append archives.  Loading is lazy and exact-hash
bound; this module never searches artifact trees or interprets historical
schemas.  Complete prefix data remain available for trajectory selectors, while
paper-facing resource tuples are source-locked to controller round 50.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.reporting.paper_i_run_summary import (
    PaperIAcceptedError,
    PaperIAlgorithmicWork,
    PaperIAppendResolutionRequest,
    PaperIAppendRunSource,
    PaperIComparisonContract,
    PaperIPrefixCompileInput,
    PaperIPrefixOperator,
    PaperIPrefixPauliTerm,
    PaperIReferenceState,
    PaperIWorkComponents,
)
from pipelines.static_adapt.estimator_call_ledger import (
    projective_state_fingerprint,
)


REGISTRY_SCHEMA = "paper_i_canonical_append_registry_v1"
REGISTRY_ROUTE_ID = "append_adapt_projected_singleton_nph3_7"
SOURCE_ADOPTION_SHA256 = (
    "3373d5d54d267a0f5f75af7efb63518463a11c308c69d595b40f3516983b8cfc"
)
SOURCE_PARTIAL_REPORT_PROVENANCE_SHA256 = (
    "485cc623974a5e2000937b76f576f70fe81427f7ceaf3655bfb8c2f2af0c9691"
)
SOURCE_VALIDATION_SHA256 = (
    "2a55c1509e9112e75c9a87201a8a7ea511529a0659d7fb58ba08b6f9875eb853"
)
SOURCE_PACKAGE_MANIFEST_SHA256 = (
    "75063a0d8de86518d91a55283e025037229d20c185681db74b79175f9b9e6176"
)
REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "agent_guidance/static-adapt/reporting/canonical-append-registry-v1.json"
)
REGISTRY_SHA256 = (
    "2d59ee3d92ccf79d7c8f5fa826516159576220011872e7d1142a6b5b612f722a"
)
REPORTING_RESOURCE_POLICY = "fixed_controller_round_50_v1"
REPORTING_RESOURCE_ROUND = 50


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object.")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise TypeError(f"{name} must be an array.")
    return value


def _finite(value: Any, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _sha256(value: Any, *, name: str) -> str:
    resolved = str(value)
    if len(resolved) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest.")
    try:
        int(resolved, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest.") from exc
    return resolved


def _file_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _comparison_contract(
    record: Mapping[str, Any],
) -> PaperIComparisonContract:
    problem_resolution = _mapping(
        record.get("problem_request_resolution"),
        name="append registry problem_request_resolution",
    )
    if (
        problem_resolution.get("include_zero_point") is not True
        or problem_resolution.get("include_zero_point_source")
        != "paper_i_stationary_core_v6_parameter_manifest_v1"
    ):
        raise ValueError("append registry problem-request resolution drift.")
    source = _mapping(
        record.get("comparison_contract"),
        name="append registry comparison_contract",
    )
    return PaperIComparisonContract(
        problem_request_sha256=_sha256(
            record.get("problem_request_sha256"),
            name="append registry problem_request_sha256",
        ),
        optimizer=str(source["optimizer"]),
        optimizer_maxiter=int(source["optimizer_maxiter"]),
        seed=int(source["seed"]),
        candidate_representation=str(source["candidate_representation"]),
        compile_convention=str(source["compile_convention"]),
    )


def _reference_state(record: Mapping[str, Any]) -> PaperIReferenceState:
    source = _mapping(
        record.get("reference_state"),
        name="append registry reference_state",
    )
    qubit_count = int(source.get("qubit_count", -1))
    if qubit_count < 1:
        raise ValueError("append registry reference qubit count is invalid.")
    real = [0.0] * (1 << qubit_count)
    imaginary = [0.0] * (1 << qubit_count)
    sparse = _mapping(
        source.get("sparse_amplitudes_qn_to_q0"),
        name="append registry sparse reference state",
    )
    for raw_bitstring, raw_amplitude in sparse.items():
        bitstring = str(raw_bitstring)
        if len(bitstring) != qubit_count or set(bitstring) - {"0", "1"}:
            raise ValueError("append registry reference bitstring is invalid.")
        amplitude = _mapping(
            raw_amplitude,
            name=f"append registry reference amplitude {bitstring}",
        )
        index = int(bitstring, 2)
        real[index] = _finite(
            amplitude.get("real"),
            name=f"append registry reference amplitude {bitstring} real",
        )
        imaginary[index] = _finite(
            amplitude.get("imaginary"),
            name=f"append registry reference amplitude {bitstring} imaginary",
        )
    norm_squared = math.fsum(
        re * re + im * im
        for re, im in zip(real, imaginary, strict=True)
    )
    if not math.isclose(
        norm_squared,
        1.0,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise ValueError("append registry reference state is not normalized.")
    fingerprint = str(source.get("state_fingerprint") or "")
    observed_fingerprint = projective_state_fingerprint(
        tuple(complex(re, im) for re, im in zip(real, imaginary, strict=True))
    )
    if fingerprint != observed_fingerprint:
        raise ValueError("append registry reference-state fingerprint drift.")
    return PaperIReferenceState(
        amplitudes_real=tuple(real),
        amplitudes_imaginary=tuple(imaginary),
        qubit_count=qubit_count,
        source_label=str(source.get("source_label") or ""),
        state_fingerprint=fingerprint,
    )


def _work(row: Mapping[str, Any]) -> PaperIAlgorithmicWork:
    source = _mapping(
        row.get("algorithmic_work"),
        name="append registry algorithmic_work",
    )
    components_source = _mapping(
        source.get("components"),
        name="append registry algorithmic_work.components",
    )
    components = PaperIWorkComponents(
        n_h_outer=int(components_source["n_h_outer"]),
        n_h_refit=int(components_source["n_h_refit"]),
        n_grad=int(components_source["n_grad"]),
        n_metric=int(components_source["n_metric"]),
    )
    s_alg = int(source["s_alg"])
    if s_alg != components.s_alg:
        raise ValueError("append registry algorithmic work does not close.")
    _sha256(
        source.get("clean_receipt_sha256"),
        name="append registry clean_receipt_sha256",
    )
    return PaperIAlgorithmicWork(components=components, s_alg=s_alg)


def _reporting_resources(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and return the adopted fixed-round-50 resource tuple."""

    source = _mapping(
        record.get("reporting_resources"),
        name="append registry reporting_resources",
    )
    if (
        source.get("policy") != REPORTING_RESOURCE_POLICY
        or int(source.get("controller_round", -1))
        != REPORTING_RESOURCE_ROUND
        or source.get("compile_convention")
        != "table_i_basis_gate_transpile_v1"
        or source.get("qiskit_validated") is not True
    ):
        raise ValueError("append registry reporting-resource policy drift.")
    normalized = {
        "policy": REPORTING_RESOURCE_POLICY,
        "controller_round": REPORTING_RESOURCE_ROUND,
        "compiled_two_qubit_count": int(
            source["compiled_two_qubit_count"]
        ),
        "compiled_two_qubit_depth": int(
            source["compiled_two_qubit_depth"]
        ),
        "compiled_total_depth": int(source["compiled_total_depth"]),
        "pauli_one_qubit_work": int(source["pauli_one_qubit_work"]),
        "s_alg": int(source["s_alg"]),
        "absolute_energy_error": _finite(
            source.get("absolute_energy_error"),
            name="append registry reporting absolute_energy_error",
        ),
        "compile_convention": "table_i_basis_gate_transpile_v1",
        "qiskit_validated": True,
    }
    if any(
        normalized[key] < 0
        for key in (
            "compiled_two_qubit_count",
            "compiled_two_qubit_depth",
            "compiled_total_depth",
            "pauli_one_qubit_work",
            "s_alg",
        )
    ):
        raise ValueError("append registry reporting resources must be nonnegative.")
    return normalized


def _operator(
    raw: Mapping[str, Any],
    *,
    qubit_count: int,
) -> PaperIPrefixOperator:
    terms = tuple(
        PaperIPrefixPauliTerm(
            pauli_exyz=str(term["pauli_exyz"]),
            coefficient_real=_finite(
                term.get("coefficient_real"),
                name="append registry Pauli coefficient real",
            ),
            coefficient_imaginary=_finite(
                term.get("coefficient_imaginary"),
                name="append registry Pauli coefficient imaginary",
            ),
            qubit_count=int(term["qubit_count"]),
        )
        for term in (
            _mapping(value, name="append registry Pauli term")
            for value in _sequence(
                raw.get("runtime_terms"),
                name="append registry runtime_terms",
            )
        )
    )
    if any(term.qubit_count != qubit_count for term in terms):
        raise ValueError("append registry Pauli term register drift.")
    return PaperIPrefixOperator(
        candidate_label=str(raw["candidate_label"]),
        logical_index=int(raw["logical_index"]),
        runtime_start=int(raw["runtime_start"]),
        runtime_count=int(raw["runtime_count"]),
        execution_mode=str(raw["execution_mode"]),
        runtime_terms=terms,
    )


def _run_source(
    record: Mapping[str, Any],
    *,
    comparison: PaperIComparisonContract,
    reference: PaperIReferenceState,
) -> PaperIAppendRunSource:
    exact_energy = _finite(
        record.get("exact_same_cutoff_energy"),
        name="append registry exact_same_cutoff_energy",
    )
    route_profile = str(record.get("route_profile") or "")
    if route_profile != REGISTRY_ROUTE_ID:
        raise ValueError("append registry route identity drift.")
    route_contract_sha256 = _sha256(
        record.get("route_contract_sha256"),
        name="append registry route_contract_sha256",
    )
    problem_request_sha256 = comparison.problem_request_sha256
    trace: list[PaperIAcceptedError] = []
    prefixes: list[PaperIPrefixCompileInput] = []
    for expected_round, raw_row in enumerate(
        _sequence(
            record.get("accepted_prefixes"),
            name="append registry accepted_prefixes",
        ),
        start=1,
    ):
        row = _mapping(
            raw_row,
            name=f"append registry accepted prefix {expected_round}",
        )
        controller_round = int(row["controller_round"])
        if controller_round != expected_round:
            raise ValueError("append registry accepted rounds are not contiguous.")
        operators = tuple(
            _operator(
                _mapping(
                    raw_operator,
                    name=f"append registry round {expected_round} operator",
                ),
                qubit_count=reference.qubit_count,
            )
            for raw_operator in _sequence(
                row.get("parameter_blocks"),
                name=f"append registry round {expected_round} parameter_blocks",
            )
        )
        work = _work(row)
        checkpoint_sha256 = _sha256(
            row.get("checkpoint_sha256"),
            name=f"append registry round {expected_round} checkpoint_sha256",
        )
        fingerprint = str(row.get("projective_state_fingerprint") or "")
        depth = int(row["active_ansatz_depth"])
        prefix = PaperIPrefixCompileInput(
            source_method="append_adapt",
            controller_round=controller_round,
            active_ansatz_depth=depth,
            ordered_operator_labels=tuple(
                str(value)
                for value in _sequence(
                    row.get("ordered_operator_labels"),
                    name=(
                        f"append registry round {expected_round} "
                        "ordered_operator_labels"
                    ),
                )
            ),
            operators=operators,
            logical_parameters=tuple(
                _finite(
                    value,
                    name=(
                        f"append registry round {expected_round} "
                        "logical_parameters"
                    ),
                )
                for value in _sequence(
                    row.get("logical_parameters"),
                    name=(
                        f"append registry round {expected_round} "
                        "logical_parameters"
                    ),
                )
            ),
            runtime_parameters=tuple(
                _finite(
                    value,
                    name=(
                        f"append registry round {expected_round} "
                        "runtime_parameters"
                    ),
                )
                for value in _sequence(
                    row.get("runtime_parameters"),
                    name=(
                        f"append registry round {expected_round} "
                        "runtime_parameters"
                    ),
                )
            ),
            reference_state=reference,
            checkpoint_sha256=checkpoint_sha256,
            projective_state_fingerprint=fingerprint,
            problem_request_sha256=problem_request_sha256,
            route_profile=route_profile,
            route_contract_sha256=route_contract_sha256,
            algorithmic_work=work,
        )
        accepted_energy = _finite(
            row.get("accepted_energy"),
            name=f"append registry round {expected_round} accepted_energy",
        )
        trace.append(
            PaperIAcceptedError(
                controller_round=controller_round,
                active_ansatz_depth=depth,
                accepted_energy=accepted_energy,
                exact_same_cutoff_energy=exact_energy,
                absolute_energy_error=_finite(
                    row.get("absolute_energy_error"),
                    name=(
                        f"append registry round {expected_round} "
                        "absolute_energy_error"
                    ),
                ),
                projective_state_fingerprint=fingerprint,
                checkpoint_sha256=checkpoint_sha256,
            )
        )
        prefixes.append(prefix)
    if len(trace) != 50:
        raise ValueError("canonical append registry must contain 50 prefixes.")
    reporting = _reporting_resources(record)
    if (
        int(reporting["s_alg"]) != prefixes[-1].algorithmic_work.s_alg
        or not math.isclose(
            float(reporting["absolute_energy_error"]),
            trace[-1].absolute_energy_error,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "append registry round-50 reporting tuple is not bound to "
            "the terminal prefix."
        )
    return PaperIAppendRunSource(
        comparison_contract=comparison,
        accepted_error_trace=tuple(trace),
        accepted_prefixes=tuple(prefixes),
        horizon_scope=str(record.get("horizon_scope") or ""),
    )


class LockedPaperIAppendRegistry:
    """Exact-hash resolver for the compact frozen append comparator registry."""

    def __init__(
        self,
        *,
        registry_path: Path = REGISTRY_PATH,
        expected_sha256: str = REGISTRY_SHA256,
    ) -> None:
        self._registry_path = Path(registry_path)
        self._expected_sha256 = str(expected_sha256)
        self._records: dict[str, Mapping[str, Any]] | None = None

    def _load_records(self) -> dict[str, Mapping[str, Any]]:
        if self._records is not None:
            return self._records
        expected = _sha256(
            self._expected_sha256,
            name="canonical append registry expected_sha256",
        )
        raw = self._registry_path.read_bytes()
        if _file_sha256(raw) != expected:
            raise ValueError("canonical append registry SHA drift.")
        payload = _mapping(
            json.loads(raw),
            name="canonical append registry",
        )
        if (
            payload.get("schema") != REGISTRY_SCHEMA
            or payload.get("route_id") != REGISTRY_ROUTE_ID
            or _mapping(
                payload.get("source_adoption"),
                name="canonical append registry source_adoption",
            ).get("sha256")
            != SOURCE_ADOPTION_SHA256
            or _mapping(
                payload.get("source_partial_report_provenance"),
                name=(
                    "canonical append registry "
                    "source_partial_report_provenance"
                ),
            ).get("sha256")
            != SOURCE_PARTIAL_REPORT_PROVENANCE_SHA256
            or _mapping(
                payload.get("source_partial_report_provenance"),
                name=(
                    "canonical append registry "
                    "source_partial_report_provenance"
                ),
            ).get("aggregate_report_adopted")
            is not False
            or _mapping(
                payload.get("source_validation"),
                name="canonical append registry source_validation",
            ).get("sha256")
            != SOURCE_VALIDATION_SHA256
            or _mapping(
                payload.get("source_package_manifest"),
                name="canonical append registry source_package_manifest",
            ).get("sha256")
            != SOURCE_PACKAGE_MANIFEST_SHA256
        ):
            raise ValueError("canonical append registry source identity drift.")
        resource_policy = _mapping(
            payload.get("reporting_resource_policy"),
            name="canonical append registry reporting_resource_policy",
        )
        if (
            resource_policy.get("policy") != REPORTING_RESOURCE_POLICY
            or int(resource_policy.get("controller_round", -1))
            != REPORTING_RESOURCE_ROUND
            or tuple(resource_policy.get("fields") or ())
            != ("N2q", "D2q", "Dc", "W1q", "S_alg")
        ):
            raise ValueError(
                "canonical append registry reporting-resource identity drift."
            )
        records: dict[str, Mapping[str, Any]] = {}
        for raw_record in _sequence(
            payload.get("records"),
            name="canonical append registry records",
        ):
            record = _mapping(
                raw_record,
                name="canonical append registry record",
            )
            problem_sha256 = _sha256(
                record.get("problem_request_sha256"),
                name="canonical append registry problem_request_sha256",
            )
            if problem_sha256 in records:
                raise ValueError(
                    "canonical append registry contains a duplicate problem."
                )
            records[problem_sha256] = record
            _reporting_resources(record)
        if len(records) != 6:
            raise ValueError("canonical append registry must contain six records.")
        self._records = records
        return records

    def resolve_canonical_append(
        self,
        request: PaperIAppendResolutionRequest,
    ) -> PaperIAppendRunSource | None:
        if not isinstance(request, PaperIAppendResolutionRequest):
            raise TypeError("request must be PaperIAppendResolutionRequest.")
        record = self._load_records().get(
            request.comparison_contract.problem_request_sha256
        )
        if record is None:
            return None
        comparison = _comparison_contract(record)
        if comparison != request.comparison_contract:
            return None
        exact_energy = _finite(
            record.get("exact_same_cutoff_energy"),
            name="append registry exact_same_cutoff_energy",
        )
        if not math.isclose(
            exact_energy,
            request.exact_same_cutoff_energy,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            return None
        reference = _reference_state(record)
        if (
            reference.qubit_count != request.reference_state.qubit_count
            or reference.state_fingerprint
            != request.reference_state.state_fingerprint
        ):
            return None
        return _run_source(
            record,
            comparison=comparison,
            reference=reference,
        )


_DEFAULT_RESOLVER: LockedPaperIAppendRegistry | None = None


def default_paper_i_append_reference_resolver() -> LockedPaperIAppendRegistry:
    """Return the process-local lazy resolver for the locked registry."""

    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        _DEFAULT_RESOLVER = LockedPaperIAppendRegistry()
    return _DEFAULT_RESOLVER


__all__ = [
    "LockedPaperIAppendRegistry",
    "REGISTRY_PATH",
    "REGISTRY_SCHEMA",
    "REGISTRY_SHA256",
    "default_paper_i_append_reference_resolver",
]
