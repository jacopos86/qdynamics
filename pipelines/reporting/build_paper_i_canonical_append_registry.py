#!/usr/bin/env python3
"""Build the compact, source-locked canonical Paper-I Append registry.

The registry is derived only by this explicit maintenance command.  It pins the
user-authorized Append-only adoption receipt and the six validated v6
projected-singleton worker archives.  Each archive contributes its complete
signed 50-round trajectory, compiler-ready prefix inputs, closed prefix
``S_alg`` receipts, and the fixed-round-50 reporting-resource tuple.  Ordinary
run completion consumes the compact registry and never scans artifact trees.
"""

from __future__ import annotations

from dataclasses import fields
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
from typing import Any, BinaryIO, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.contracts.problem import ProblemRequest  # noqa: E402
from pipelines.reporting.build_paper_i_hh_comparator_tracking_summary import (  # noqa: E402
    _iter_named_json_array,
)
from pipelines.static_adapt.builders.problem_registry import (  # noqa: E402
    resolve_problem_context,
)
from pipelines.static_adapt.estimator_call_ledger import (  # noqa: E402
    projective_state_fingerprint,
)
from pipelines.static_adapt.ra_adapt.adapters import (  # noqa: E402
    SinglePauliWordCandidateAdapter,
)
from pipelines.static_adapt.sr_snake.contracts import (  # noqa: E402
    CANONICAL_CANDIDATE_REPRESENTATION,
)


SCHEMA = "paper_i_canonical_append_registry_v1"
ADOPTION_PATH = REPO_ROOT / (
    "MATH/paper_facing/paper_I_static_scaffold/provenance/"
    "paper_i_append_stationary_core_v6_component_adoption_20260729.json"
)
ADOPTION_SHA256 = (
    "3373d5d54d267a0f5f75af7efb63518463a11c308c69d595b40f3516983b8cfc"
)
PARTIAL_REPORT_PROVENANCE_PATH = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress_provenance.json"
)
PARTIAL_REPORT_PROVENANCE_SHA256 = (
    "485cc623974a5e2000937b76f576f70fe81427f7ceaf3655bfb8c2f2af0c9691"
)
VALIDATION_PATH = REPO_ROOT / (
    "raw_outputs/paper_i_ra_adapt_stationary_core_v6_partial_report_"
    "12of48_20260729/fetched_validation.json"
)
VALIDATION_FILE_SHA256 = (
    "65119e8ca07c8a7f4e0360e131c26f0b5a558bf2de503b48afd121744ad86218"
)
VALIDATION_SHA256 = (
    "2a55c1509e9112e75c9a87201a8a7ea511529a0659d7fb58ba08b6f9875eb853"
)
PACKAGE_MANIFEST_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_core_full48_r50_20260728_v6_chtc/package_manifest.json"
)
PACKAGE_MANIFEST_FILE_SHA256 = (
    "48a32ad64773a794f748b4c4e6013cf52e7d637a0c8f8c7da83b71a07df280cc"
)
PACKAGE_MANIFEST_SHA256 = (
    "75063a0d8de86518d91a55283e025037229d20c185681db74b79175f9b9e6176"
)
OUTPUT_PATH = REPO_ROOT / (
    "agent_guidance/static-adapt/reporting/"
    "canonical-append-registry-v1.json"
)
PACKAGE_ID = "paper_i_ra_adapt_stationary_core_full48_r50_20260728_v6_chtc"
CORE_MATERIALIZATION_ID = "ra_adapt_stationary_late_core_v10"
SOURCE_ARCHIVE_SHA256 = (
    "1f949b0cc8b61dca63911832e8dc8bb32614174755ac476827956bb0812accee"
)
APPEND_ROUTE_ID = "append_adapt_projected_singleton_nph3_7"
SOURCE_ROUTE_ID = "append_singleton"
COMPILE_CONVENTION = "table_i_basis_gate_transpile_v1"
REPORTING_RESOURCE_POLICY = "fixed_controller_round_50_v1"
REPORTING_RESOURCE_ROUND = 50
EXPECTED_SOURCE_LOCKS = {
    "append_module_sha256": (
        "f04cec3fedfb0dbcc0716f88941ada668d9832107d303dc01073313eccf531fa"
    ),
    "pool_module_sha256": (
        "cf8964a8acd1b7b5851d9ff27f2f2aa05d8d2848a7c1bde0f02e18751767e842"
    ),
}
EXPECTED_PROBLEM_SHA256 = {
    "weak_weak": (
        "1ed335ecd47cccea28e1e9b4046e92a55eeffa429b714efc43a9ff4126192aeb"
    ),
    "intermediate_weak": (
        "93d46c40cc1cd32f61019923522f567a77d2b32379b618ad9907e82199696d84"
    ),
    "strong_weak_u8": (
        "5197b317fe67b5eedabd726e29b897260c18bda9eaf6bc9cc05cf3b0a468b65d"
    ),
    "weak_strong": (
        "00af7451ac5d551220ca7de167b096813d692d3c2cd3962f59891692f637bd91"
    ),
    "intermediate_strong": (
        "5ff9444d0a9fe8131fc50bba60ef814afe242167a5cabb2b2d9247aa842e9c0a"
    ),
    "strong_strong_u8": (
        "e9e9287c677cd2f2af5e9990b2a5742faa225b27fac38f54f7e054ed1fc29a2d"
    ),
}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object.")
    return value


def _required_sequence(value: Any, *, name: str) -> Sequence[Any]:
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


def _canonical_receipt(
    payload: Mapping[str, Any],
    *,
    name: str,
) -> str:
    expected = str(payload.get("sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    observed = _canonical_sha256(unsigned)
    if observed != expected:
        raise ValueError(f"{name} canonical SHA drift.")
    return expected


def _repo_path(value: Any, *, name: str) -> Path:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} is outside the active checkout.") from exc
    return resolved


def _display_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


class _DigestingReader:
    """Binary stream wrapper that records exact bytes and size."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()
        self.size_bytes = 0

    def read(self, size: int = -1) -> bytes:
        raw = self._handle.read(size)
        self._digest.update(raw)
        self.size_bytes += len(raw)
        return raw

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


def _problem_request_sha256(request: ProblemRequest) -> str:
    payload = {
        field.name: getattr(request, field.name)
        for field in fields(request)
    }
    return _canonical_sha256(payload)


def _problem_request(
    regime: Mapping[str, Any],
    parameter_manifest: Mapping[str, Any],
) -> ProblemRequest:
    return ProblemRequest(
        problem_key=str(parameter_manifest["problem_key"]),
        num_sites=int(parameter_manifest["num_sites"]),
        t=_finite(regime["t"], name="regime t"),
        u=_finite(regime["u"], name="regime u"),
        dv=_finite(regime["dv"], name="regime dv"),
        omega0=_finite(regime["omega0"], name="regime omega0"),
        g_ep=_finite(regime["g_ep"], name="regime g_ep"),
        n_ph_max=int(regime["n_ph_max"]),
        boson_encoding=str(parameter_manifest["boson_encoding"]),
        ordering=str(parameter_manifest["ordering"]),
        boundary=str(parameter_manifest["boundary"]),
        include_zero_point=bool(parameter_manifest["include_zero_point"]),
        v_nn=_finite(regime.get("v_nn", 0.0), name="regime v_nn"),
        t_prime=_finite(
            regime.get("t_prime", 0.0),
            name="regime t_prime",
        ),
    )


def _reference_state(problem: Any) -> dict[str, Any]:
    raw = tuple(
        complex(value)
        for value in problem.reference_state.build_state()
    )
    norm = math.sqrt(math.fsum(abs(value) ** 2 for value in raw))
    if not math.isclose(norm, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError("resolved Append reference state is not normalized.")
    normalized = tuple(value / norm for value in raw)
    qubit_count = int(problem.layout.total_qubits)
    if len(normalized) != 1 << qubit_count:
        raise ValueError("resolved Append reference-state register drift.")
    sparse = {
        format(index, f"0{qubit_count}b"): {
            "real": float(value.real),
            "imaginary": float(value.imag),
        }
        for index, value in enumerate(normalized)
        if abs(value) > 1.0e-15
    }
    return {
        "qubit_count": qubit_count,
        "source_label": str(problem.reference_state.source_label),
        "state_fingerprint": projective_state_fingerprint(normalized),
        "sparse_amplitudes_qn_to_q0": sparse,
    }


def _normalized_operator(
    candidate: Any,
    *,
    logical_index: int,
    qubit_count: int,
) -> dict[str, Any]:
    terms = []
    for raw_term in candidate.serialized_terms_exyz:
        term = _required_mapping(
            raw_term,
            name=f"operator {logical_index} serialized term",
        )
        word = str(term.get("pauli_exyz") or "")
        if (
            int(term.get("nq", -1)) != qubit_count
            or len(word) != qubit_count
            or set(word) - set("exyz")
        ):
            raise ValueError(
                f"operator {logical_index} Pauli register drift."
            )
        terms.append(
            {
                "pauli_exyz": word,
                "coefficient_real": _finite(
                    term.get("coeff_re"),
                    name=f"operator {logical_index} coefficient real",
                ),
                "coefficient_imaginary": _finite(
                    term.get("coeff_im"),
                    name=f"operator {logical_index} coefficient imaginary",
                ),
                "qubit_count": qubit_count,
            }
        )
    if len(terms) != 1:
        raise ValueError("projected-singleton operator is not a singleton.")
    return {
        "candidate_label": str(candidate.label),
        "logical_index": logical_index,
        "runtime_start": logical_index,
        "runtime_count": 1,
        "execution_mode": str(candidate.execution_mode),
        "runtime_terms": terms,
    }


def _normalize_history_row(
    row: Mapping[str, Any],
    *,
    controller_round: int,
    exact_energy: float,
    problem_sha256: str,
    protocol_sha256: str,
    qubit_count: int,
    candidates: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        int(row.get("controller_round", -1)) != controller_round
        or int(row.get("insertion_position", -1)) != controller_round - 1
        or row.get("ra_staged_funnel_invoked") is not False
    ):
        raise ValueError(
            f"Append round {controller_round} controller identity drift."
        )
    checkpoint = dict(
        _required_mapping(
            row.get("active_prefix_checkpoint"),
            name=f"Append round {controller_round} checkpoint",
        )
    )
    if (
        checkpoint.get("schema")
        != "paper_i_signed_append_active_prefix_checkpoint_v1"
    ):
        raise ValueError(
            f"Append round {controller_round} checkpoint schema drift."
        )
    checkpoint_sha256 = str(checkpoint.get("checkpoint_sha256") or "")
    unsigned = dict(checkpoint)
    unsigned.pop("checkpoint_sha256", None)
    if _canonical_sha256(unsigned) != checkpoint_sha256:
        raise ValueError(
            f"Append round {controller_round} checkpoint SHA drift."
        )
    if (
        int(checkpoint.get("controller_round", -1)) != controller_round
        or checkpoint.get("problem_request_sha256") != problem_sha256
        or checkpoint.get("protocol_sha256") != protocol_sha256
    ):
        raise ValueError(
            f"Append round {controller_round} checkpoint binding drift."
        )
    labels = tuple(
        str(value)
        for value in _required_sequence(
            checkpoint.get("accepted_operator_labels"),
            name=f"Append round {controller_round} labels",
        )
    )
    identities = tuple(
        str(value)
        for value in _required_sequence(
            checkpoint.get("accepted_generator_identities"),
            name=f"Append round {controller_round} identities",
        )
    )
    if (
        len(labels) != controller_round
        or len(identities) != controller_round
        or str(row.get("selected_label")) != labels[-1]
        or str(row.get("selected_generator_identity")) != identities[-1]
    ):
        raise ValueError(
            f"Append round {controller_round} accepted lineage drift."
        )
    selected = []
    for label, identity in zip(labels, identities, strict=True):
        candidate = candidates.get(label)
        if candidate is None or str(candidate.generator_identity) != identity:
            raise ValueError(
                f"Append round {controller_round} candidate source drift."
            )
        selected.append(candidate)
    logical = tuple(
        _finite(
            value,
            name=f"Append round {controller_round} logical parameter",
        )
        for value in _required_sequence(
            checkpoint.get("logical_parameters"),
            name=f"Append round {controller_round} logical parameters",
        )
    )
    runtime = tuple(
        _finite(
            value,
            name=f"Append round {controller_round} runtime parameter",
        )
        for value in _required_sequence(
            checkpoint.get("runtime_parameters"),
            name=f"Append round {controller_round} runtime parameters",
        )
    )
    if len(logical) != controller_round or runtime != logical:
        raise ValueError(
            f"Append round {controller_round} parameter layout drift."
        )
    energy = _finite(
        checkpoint.get("accepted_energy"),
        name=f"Append round {controller_round} accepted energy",
    )
    if not math.isclose(
        energy,
        _finite(
            row.get("energy_after"),
            name=f"Append round {controller_round} row energy",
        ),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"Append round {controller_round} energy drift.")
    estimator_prefix = _required_mapping(
        checkpoint.get("estimator_prefix"),
        name=f"Append round {controller_round} estimator prefix",
    )
    executed = _required_mapping(
        estimator_prefix.get("cumulative_executed_queries"),
        name=f"Append round {controller_round} executed queries",
    )
    components = _required_mapping(
        executed.get("components"),
        name=f"Append round {controller_round} work components",
    )
    normalized_components = {
        "n_h_outer": int(components["N_H_outer"]),
        "n_h_refit": int(components["N_H_refit"]),
        "n_grad": int(components["N_grad"]),
        "n_metric": int(components["N_metric"]),
    }
    s_alg = int(executed["S_alg"])
    if s_alg != sum(normalized_components.values()):
        raise ValueError(
            f"Append round {controller_round} work does not close."
        )
    return {
        "controller_round": controller_round,
        "active_ansatz_depth": controller_round,
        "accepted_energy": energy,
        "absolute_energy_error": abs(energy - exact_energy),
        "checkpoint_sha256": checkpoint_sha256,
        "projective_state_fingerprint": str(
            checkpoint["projective_state_fingerprint"]
        ),
        "ordered_operator_labels": list(labels),
        "parameter_blocks": [
            _normalized_operator(
                candidate,
                logical_index=index,
                qubit_count=qubit_count,
            )
            for index, candidate in enumerate(selected)
        ],
        "logical_parameters": list(logical),
        "runtime_parameters": list(runtime),
        "algorithmic_work": {
            "components": normalized_components,
            "s_alg": s_alg,
            "clean_receipt_sha256": _canonical_sha256(estimator_prefix),
        },
    }


def _read_json_member(
    handle: BinaryIO,
    *,
    name: str,
) -> tuple[dict[str, Any], str, int]:
    raw = handle.read()
    payload = dict(
        _required_mapping(json.loads(raw), name=name)
    )
    return payload, hashlib.sha256(raw).hexdigest(), len(raw)


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    protocol_sha256: str,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        summary.get("schema") != "paper_i_append_run_summary_v1"
        or summary.get("candidate_representation") != "single_pauli_word_v1"
        or summary.get("optimizer") != "powell"
        or int(summary.get("optimizer_maxiter", -1)) != 200
        or int(summary.get("controller_rounds_completed", -1)) != 50
        or summary.get("stop_reason") != "maximum_controller_rounds"
        or summary.get("protocol_sha256") != protocol_sha256
        or summary.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or summary.get("resource_weighting_scope")
        != "late_resource_weighting_v1"
    ):
        raise ValueError("v6 Append summary identity drift.")
    seeds = _required_mapping(summary.get("seeds"), name="Append seeds")
    compile_identity = _required_mapping(
        summary.get("compile_identity"),
        name="Append compile identity",
    )
    if (
        int(seeds.get("adapt", -1)) != 7
        or int(seeds.get("transpiler", -1)) != 7
        or compile_identity.get("policy") != COMPILE_CONVENTION
        or int(compile_identity.get("optimization_level", -1)) != 0
        or int(compile_identity.get("transpiler_seed", -1)) != 7
        or compile_identity.get("reference_preparation_included") is not True
    ):
        raise ValueError("v6 Append comparison contract drift.")
    summary_history = _required_sequence(
        summary.get("accepted_history"),
        name="Append summary accepted history",
    )
    if len(summary_history) != 50 or len(history) != 50:
        raise ValueError("v6 Append accepted history is not complete.")
    for expected_round, (raw_summary, registry_row) in enumerate(
        zip(summary_history, history, strict=True),
        start=1,
    ):
        summary_row = _required_mapping(
            raw_summary,
            name=f"Append summary round {expected_round}",
        )
        if (
            int(summary_row.get("controller_round", -1)) != expected_round
            or not math.isclose(
                _finite(
                    summary_row.get("energy_after"),
                    name=f"Append summary round {expected_round} energy",
                ),
                float(registry_row["accepted_energy"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("v6 Append summary trajectory drift.")
    accounting = _required_mapping(
        summary.get("estimator_accounting"),
        name="Append terminal accounting",
    )
    terminal_s_alg = int(accounting["S_alg"])
    if (
        accounting.get("closed_occurrence_reconciliation") is not True
        or terminal_s_alg
        != int(history[-1]["algorithmic_work"]["s_alg"])
    ):
        raise ValueError("v6 Append terminal accounting drift.")
    resources = _required_mapping(
        summary.get("resources"),
        name="Append resource summary",
    )
    compiled = _required_mapping(
        resources.get("terminal_compiled_resources"),
        name="Append terminal compiled resources",
    )
    if (
        compiled.get("compiled_circuit_stats_status") != "ok"
        or compiled.get("compiled_resource_qiskit_validated") is not True
        or compiled.get("compile_convention") != COMPILE_CONVENTION
        or int(compiled.get("logical_operator_count", -1)) != 50
    ):
        raise ValueError("v6 Append terminal Qiskit resource gate failed.")
    terminal = {
        "policy": REPORTING_RESOURCE_POLICY,
        "controller_round": REPORTING_RESOURCE_ROUND,
        "compiled_two_qubit_count": int(compiled["compiled_count_2q_total"]),
        "compiled_two_qubit_depth": int(compiled["compiled_depth_2q_total"]),
        "compiled_total_depth": int(compiled["compiled_depth_total"]),
        "pauli_one_qubit_work": int(
            compiled["qiskit_pretranspile_pauli_1q_work_total"]
        ),
        "s_alg": terminal_s_alg,
        "absolute_energy_error": float(history[-1]["absolute_energy_error"]),
        "compile_convention": COMPILE_CONVENTION,
        "qiskit_validated": True,
    }
    source_terminal = _required_mapping(
        source.get("terminal"),
        name="partial-report terminal Append row",
    )
    expected = {
        "k": terminal["controller_round"],
        "N2q": terminal["compiled_two_qubit_count"],
        "D2q": terminal["compiled_two_qubit_depth"],
        "Dc": terminal["compiled_total_depth"],
        "W1q": terminal["pauli_one_qubit_work"],
        "S_alg": terminal["s_alg"],
    }
    for key, value in expected.items():
        if int(source_terminal.get(key, -1)) != int(value):
            raise ValueError(f"partial-report terminal {key} drift.")
    if not math.isclose(
        _finite(source_terminal.get("error"), name="terminal source error"),
        terminal["absolute_energy_error"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("partial-report terminal error drift.")
    return terminal


def _validate_worker_receipt(
    receipt: Mapping[str, Any],
    *,
    execution_id: str,
    exact_energy: float,
    pool_sha256: str,
    terminal: Mapping[str, Any],
) -> None:
    if (
        receipt.get("schema")
        != "paper_i_ra_adapt_stationary_core_worker_receipt_v1"
        or receipt.get("status") != "passed"
        or receipt.get("execution_id") != execution_id
        or receipt.get("package_id") != PACKAGE_ID
        or receipt.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or receipt.get("package_manifest_sha256") != PACKAGE_MANIFEST_SHA256
    ):
        raise ValueError(f"{execution_id} worker receipt identity drift.")
    closure = _required_mapping(
        receipt.get("scientific_closure"),
        name=f"{execution_id} scientific closure",
    )
    gates = _required_mapping(
        closure.get("gates"),
        name=f"{execution_id} closure gates",
    )
    if (
        closure.get("status") != "passed"
        or int(closure.get("full_controller_rounds", -1)) != 50
        or set(gates) != {f"G{index}" for index in range(1, 14)}
        or any(
            _required_mapping(value, name=f"{execution_id} gate").get("status")
            != "passed"
            for value in gates.values()
        )
    ):
        raise ValueError(f"{execution_id} scientific gates are not closed.")
    g1 = _required_mapping(gates["G1"]["evidence"], name=f"{execution_id} G1")
    for key, expected in EXPECTED_SOURCE_LOCKS.items():
        if g1.get(key) not in {None, expected}:
            raise ValueError(f"{execution_id} {key} drift.")
    g2 = _required_mapping(gates["G2"]["evidence"], name=f"{execution_id} G2")
    verified_ed = _required_mapping(
        g2.get("verified_ed_reference"),
        name=f"{execution_id} verified ED",
    )
    if not math.isclose(
        _finite(verified_ed.get("E_ED"), name=f"{execution_id} ED energy"),
        exact_energy,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{execution_id} same-cutoff ED drift.")
    g3 = _required_mapping(gates["G3"]["evidence"], name=f"{execution_id} G3")
    if (
        g3.get("candidate_representation") != "single_pauli_word_v1"
        or g3.get("executable_pool_sha256") != pool_sha256
    ):
        raise ValueError(f"{execution_id} executable-pool drift.")
    g9 = _required_mapping(gates["G9"]["evidence"], name=f"{execution_id} G9")
    if (
        g9.get("sector_leak_flag") is not False
        or g9.get("boson_truncation_leak_flag") is not False
        or int(g9.get("accepted_transition_check_count", -1)) != 50
    ):
        raise ValueError(f"{execution_id} sector gate failed.")
    g10 = _required_mapping(
        gates["G10"]["evidence"],
        name=f"{execution_id} G10",
    )
    if int(g10.get("S_alg", -1)) != int(terminal["s_alg"]):
        raise ValueError(f"{execution_id} G10 accounting drift.")
    g12 = _required_mapping(
        gates["G12"]["evidence"],
        name=f"{execution_id} G12",
    )
    qiskit = _required_mapping(
        g12.get("resources"),
        name=f"{execution_id} G12 resources",
    )
    expected_qiskit = {
        "N2q": terminal["compiled_two_qubit_count"],
        "D2q": terminal["compiled_two_qubit_depth"],
        "Dc": terminal["compiled_total_depth"],
    }
    if any(
        int(qiskit.get(key, -1)) != int(value)
        for key, value in expected_qiskit.items()
    ):
        raise ValueError(f"{execution_id} G12 resource drift.")


def _load_archive_record(
    *,
    regime: Mapping[str, Any],
    parameter_manifest: Mapping[str, Any],
    source_record: Mapping[str, Any],
    source: Mapping[str, Any],
    validation_attempt: Mapping[str, Any],
) -> dict[str, Any]:
    regime_id = str(regime["regime_id"])
    execution_id = f"core__{regime_id}__nph{int(regime['n_ph_max'])}__append_singleton"
    if (
        source.get("execution_id") != execution_id
        or source.get("route_id") != SOURCE_ROUTE_ID
        or source.get("method_family") != "append"
        or source.get("candidate_representation") != "single_pauli_word_v1"
        or validation_attempt.get("execution_id") != execution_id
        or validation_attempt.get("status") != "passed"
    ):
        raise ValueError(f"{regime_id} adopted source identity drift.")
    request = _problem_request(regime, parameter_manifest)
    problem_sha256 = _problem_request_sha256(request)
    if (
        problem_sha256 != EXPECTED_PROBLEM_SHA256[regime_id]
        or problem_sha256 != str(regime["problem_request_sha256"])
    ):
        raise ValueError(f"{regime_id} problem-request SHA drift.")
    problem = resolve_problem_context(request)
    reference = _reference_state(problem)
    inventory = SinglePauliWordCandidateAdapter().global_executable_pool(
        problem
    )
    candidates = {
        str(candidate.label): candidate
        for candidate in inventory.candidates
    }
    fetched_dir = _repo_path(
        source_record["fetched_dir"],
        name="v6 fetched archive directory",
    )
    archive_path = (fetched_dir / str(source["attempt_path"])).resolve()
    if archive_path.parent != fetched_dir or not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    execution_manifest: dict[str, Any] | None = None
    execution_manifest_file_sha256: str | None = None
    checkpoint: dict[str, Any] | None = None
    checkpoint_file_sha256: str | None = None
    summary: dict[str, Any] | None = None
    summary_file_sha256: str | None = None
    worker_receipt: dict[str, Any] | None = None
    worker_receipt_file_sha256: str | None = None
    job_spec: dict[str, Any] | None = None
    job_file_sha256: str | None = None
    result_file_sha256: str | None = None
    result_size_bytes: int | None = None
    history: list[dict[str, Any]] = []

    with archive_path.open("rb") as archive_handle:
        archive_reader = _DigestingReader(archive_handle)
        with tarfile.open(fileobj=archive_reader, mode="r|gz") as archive:
            for info in archive:
                handle = archive.extractfile(info)
                if handle is None:
                    archive.members.clear()
                    continue
                if info.name == "worker_outputs/checkpoint.json":
                    (
                        checkpoint,
                        checkpoint_file_sha256,
                        _checkpoint_size,
                    ) = _read_json_member(handle, name=f"{execution_id} checkpoint")
                elif info.name == "worker_outputs/execution_manifest.json":
                    (
                        execution_manifest,
                        execution_manifest_file_sha256,
                        _manifest_size,
                    ) = _read_json_member(
                        handle,
                        name=f"{execution_id} execution manifest",
                    )
                elif info.name == "worker_outputs/result.json":
                    result_reader = _DigestingReader(handle)
                    for controller_round, raw_row in enumerate(
                        _iter_named_json_array(result_reader, "history"),
                        start=1,
                    ):
                        protocol_sha256 = (
                            ""
                            if execution_manifest is None
                            else str(execution_manifest.get("protocol_sha256") or "")
                        )
                        history.append(
                            _normalize_history_row(
                                _required_mapping(
                                    raw_row,
                                    name=f"{execution_id} history row",
                                ),
                                controller_round=controller_round,
                                exact_energy=_finite(
                                    source["exact_same_cutoff_energy"],
                                    name=f"{execution_id} exact energy",
                                ),
                                problem_sha256=problem_sha256,
                                protocol_sha256=protocol_sha256,
                                qubit_count=int(problem.layout.total_qubits),
                                candidates=candidates,
                            )
                        )
                    while result_reader.read(1024 * 1024):
                        pass
                    result_file_sha256 = result_reader.sha256
                    result_size_bytes = result_reader.size_bytes
                elif info.name == "worker_outputs/summary.json":
                    summary, summary_file_sha256, _summary_size = (
                        _read_json_member(
                            handle,
                            name=f"{execution_id} summary",
                        )
                    )
                elif info.name == "worker_outputs/worker_receipt.json":
                    (
                        worker_receipt,
                        worker_receipt_file_sha256,
                        _worker_size,
                    ) = _read_json_member(
                        handle,
                        name=f"{execution_id} worker receipt",
                    )
                elif info.name.endswith(f"/jobs/{execution_id}.json"):
                    job_spec, job_file_sha256, _job_size = _read_json_member(
                        handle,
                        name=f"{execution_id} job spec",
                    )
                archive.members.clear()
        while archive_reader.read(1024 * 1024):
            pass
        archive_sha256 = archive_reader.sha256

    required = {
        "checkpoint": checkpoint,
        "execution manifest": execution_manifest,
        "summary": summary,
        "worker receipt": worker_receipt,
        "job spec": job_spec,
        "result digest": result_file_sha256,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"{execution_id} archive lacks {missing}.")
    assert checkpoint is not None
    assert execution_manifest is not None
    assert summary is not None
    assert worker_receipt is not None
    assert job_spec is not None
    protocol_sha256 = str(execution_manifest["protocol_sha256"])
    if (
        archive_sha256 != str(source["attempt_sha256"])
        or execution_manifest_file_sha256
        != str(source["execution_manifest_file_sha256"])
        or result_file_sha256 != str(source["result_file_sha256"])
        or summary_file_sha256 != str(source["summary_file_sha256"])
        or worker_receipt_file_sha256
        != str(source["worker_receipt_file_sha256"])
        or job_file_sha256 != str(source["job_file_sha256"])
    ):
        raise ValueError(f"{execution_id} archive/member SHA drift.")
    _canonical_receipt(
        execution_manifest,
        name=f"{execution_id} execution manifest",
    )
    worker_receipt_sha256 = _canonical_receipt(
        worker_receipt,
        name=f"{execution_id} worker receipt",
    )
    job_spec_sha256 = _canonical_receipt(
        job_spec,
        name=f"{execution_id} job spec",
    )
    checkpoint_sha256 = _canonical_receipt(
        checkpoint,
        name=f"{execution_id} terminal checkpoint",
    )
    if (
        worker_receipt_sha256 != str(source["worker_receipt_sha256"])
        or worker_receipt_sha256
        != str(validation_attempt["worker_receipt_sha256"])
        or execution_manifest.get("status") != "passed"
        or execution_manifest.get("paper_facing_result_allowed") is not True
        or execution_manifest.get("execution_id") != execution_id
        or execution_manifest.get("package_id") != PACKAGE_ID
        or job_spec.get("execution_id") != execution_id
        or job_spec.get("route_id") != SOURCE_ROUTE_ID
        or job_spec.get("candidate_representation")
        != "single_pauli_word_v1"
        or job_spec.get("package_id") != PACKAGE_ID
        or int(job_spec.get("nph", -1)) != int(regime["n_ph_max"])
        or checkpoint.get("controller_rounds_completed") != 50
        or checkpoint.get("protocol_sha256") != protocol_sha256
    ):
        raise ValueError(f"{execution_id} signed identity drift.")
    terminal = _validate_summary(
        summary,
        source=source,
        protocol_sha256=protocol_sha256,
        history=history,
    )
    _validate_worker_receipt(
        worker_receipt,
        execution_id=execution_id,
        exact_energy=_finite(
            source["exact_same_cutoff_energy"],
            name=f"{execution_id} exact energy",
        ),
        pool_sha256=str(inventory.receipt.sha256),
        terminal=terminal,
    )
    route_contract = {
        "schema": "paper_i_frozen_append_route_contract_v1",
        "route_id": APPEND_ROUTE_ID,
        "source_route_id": SOURCE_ROUTE_ID,
        "comparison_contract": {
            "optimizer": "POWELL",
            "optimizer_maxiter": 200,
            "seed": 7,
            "candidate_representation": CANONICAL_CANDIDATE_REPRESENTATION,
            "compile_convention": COMPILE_CONVENTION,
        },
        "source_archive_sha256": archive_sha256,
        "source_result_member_sha256": result_file_sha256,
        "source_protocol_sha256": protocol_sha256,
    }
    return {
        "regime": regime_id,
        "problem_request_sha256": problem_sha256,
        "problem_request_resolution": {
            "include_zero_point": True,
            "include_zero_point_source": (
                "paper_i_stationary_core_v6_parameter_manifest_v1"
            ),
        },
        "exact_same_cutoff_energy": _finite(
            source["exact_same_cutoff_energy"],
            name=f"{execution_id} exact energy",
        ),
        "horizon_scope": "deliberately_stopped_prefix",
        "route_profile": APPEND_ROUTE_ID,
        "route_contract_sha256": _canonical_sha256(route_contract),
        "comparison_contract": route_contract["comparison_contract"],
        "reference_state": reference,
        "reporting_resources": terminal,
        "source": {
            "execution_id": execution_id,
            "archive_path": _display_path(archive_path),
            "archive_sha256": archive_sha256,
            "result_member": "worker_outputs/result.json",
            "result_member_sha256": result_file_sha256,
            "result_member_size_bytes": result_size_bytes,
            "summary_member": "worker_outputs/summary.json",
            "summary_member_sha256": summary_file_sha256,
            "execution_manifest_member": (
                "worker_outputs/execution_manifest.json"
            ),
            "execution_manifest_member_sha256": (
                execution_manifest_file_sha256
            ),
            "worker_receipt_member": "worker_outputs/worker_receipt.json",
            "worker_receipt_member_sha256": worker_receipt_file_sha256,
            "worker_receipt_sha256": worker_receipt_sha256,
            "job_spec_sha256": job_spec_sha256,
            "terminal_checkpoint_sha256": checkpoint_sha256,
            "protocol_sha256": protocol_sha256,
            "pool_receipt_sha256": str(inventory.receipt.sha256),
        },
        "accepted_prefixes": history,
    }


def _load_locked_json(
    path: Path,
    *,
    file_sha256: str,
    name: str,
    canonical_sha256: str | None = None,
) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != file_sha256:
        raise ValueError(f"{name} file SHA drift.")
    payload = dict(_required_mapping(json.loads(raw), name=name))
    if canonical_sha256 is not None:
        if _canonical_receipt(payload, name=name) != canonical_sha256:
            raise ValueError(f"{name} canonical identity drift.")
    return payload


def build_registry() -> dict[str, Any]:
    locked_modules = {
        "append_module_sha256": (
            REPO_ROOT / "pipelines/static_adapt/ra_adapt/append.py"
        ),
        "pool_module_sha256": (
            REPO_ROOT / "pipelines/static_adapt/ra_adapt/pools.py"
        ),
    }
    for key, path in locked_modules.items():
        if _sha256_path(path) != EXPECTED_SOURCE_LOCKS[key]:
            raise ValueError(f"v6 source lock {key} drift.")
    adoption = _load_locked_json(
        ADOPTION_PATH,
        file_sha256=ADOPTION_SHA256,
        name="Paper-I Append component-adoption receipt",
    )
    adoption_scope = _required_mapping(
        adoption.get("adoption"),
        name="Paper-I Append component-adoption scope",
    )
    adoption_horizon = _required_mapping(
        adoption_scope.get("fixed_horizon_contract"),
        name="Paper-I Append fixed-horizon adoption",
    )
    if (
        adoption.get("schema")
        != "paper_i_append_stationary_core_component_adoption_v1"
        or adoption.get("package_id") != PACKAGE_ID
        or adoption.get("core_materialization_id")
        != CORE_MATERIALIZATION_ID
        or adoption.get("source_archive_sha256")
        != SOURCE_ARCHIVE_SHA256
        or adoption_scope.get("paper_evidence_adopted") is not True
        or adoption_scope.get("aggregate_partial_report_adopted")
        is not False
        or adoption_scope.get("ra_cells_adopted") is not False
        or int(adoption_scope.get("component_count", -1)) != 12
        or int(adoption_horizon.get("resource_iteration", -1)) != 50
        or tuple(adoption_horizon.get("resource_fields") or ())
        != ("N2q", "D2q", "Dc", "W1q", "S_alg")
    ):
        raise ValueError("Paper-I Append component-adoption scope drift.")
    partial = _load_locked_json(
        PARTIAL_REPORT_PROVENANCE_PATH,
        file_sha256=PARTIAL_REPORT_PROVENANCE_SHA256,
        name="stationary-core partial-report provenance",
    )
    if (
        partial.get("partial_progress") is not True
        or partial.get("not_paper_evidence") is not True
        or partial.get("paper_evidence_adopted") is not False
    ):
        raise ValueError("aggregate partial-report status drift.")
    validation = _load_locked_json(
        VALIDATION_PATH,
        file_sha256=VALIDATION_FILE_SHA256,
        name="v6 fetched validation",
        canonical_sha256=VALIDATION_SHA256,
    )
    package = _load_locked_json(
        PACKAGE_MANIFEST_PATH,
        file_sha256=PACKAGE_MANIFEST_FILE_SHA256,
        name="v6 package manifest",
        canonical_sha256=PACKAGE_MANIFEST_SHA256,
    )
    if (
        validation.get("schema")
        != "paper_i_ra_adapt_stationary_core_fetched_validation_v1"
        or validation.get("status") != "validated_no_selection"
        or validation.get("package_id") != PACKAGE_ID
        or package.get("package_id") != PACKAGE_ID
    ):
        raise ValueError("v6 package/validation identity drift.")
    source_record = next(
        (
            _required_mapping(row, name="Append source record")
            for row in _required_sequence(
                partial.get("source_records"),
                name="partial-report source records",
            )
            if isinstance(row, Mapping)
            and row.get("method_family") == "append"
        ),
        None,
    )
    if source_record is None:
        raise ValueError("partial report lacks the adopted Append source.")
    if (
        source_record.get("package_id") != PACKAGE_ID
        or source_record.get("core_materialization_id")
        != CORE_MATERIALIZATION_ID
        or int(source_record.get("included_count", -1)) != 12
        or source_record.get("automatic_attempt_selection_performed")
        is not False
    ):
        raise ValueError("adopted Append source-record drift.")
    if set(
        str(value)
        for value in _required_sequence(
            adoption_scope.get("execution_ids"),
            name="adopted Append execution IDs",
        )
    ) != set(
        str(value)
        for value in _required_sequence(
            source_record.get("included_execution_ids"),
            name="source-record Append execution IDs",
        )
    ):
        raise ValueError("Append adoption/source execution-ID drift.")
    package_sources = _required_mapping(
        source_record.get("package_sources"),
        name="Append package sources",
    )
    if (
        package_sources.get("source_archive_sha256")
        != SOURCE_ARCHIVE_SHA256
        or _required_mapping(
            package_sources.get("package_manifest"),
            name="Append package-manifest source",
        ).get("sha256")
        != PACKAGE_MANIFEST_SHA256
        or _required_mapping(
            source_record.get("validation"),
            name="Append validation source",
        ).get("sha256")
        != VALIDATION_SHA256
    ):
        raise ValueError("adopted Append package-source drift.")
    parameter_manifest = _required_mapping(
        partial.get("parameter_manifest"),
        name="stationary-core parameter manifest",
    )
    if (
        int(parameter_manifest.get("horizon", -1)) != 50
        or str(parameter_manifest.get("optimizer")).lower() != "powell"
        or int(parameter_manifest.get("optimizer_maxiter", -1)) != 200
        or parameter_manifest.get("include_zero_point") is not True
    ):
        raise ValueError("stationary-core comparison contract drift.")
    regimes = {
        str(row["regime_id"]): _required_mapping(
            row,
            name="stationary-core regime",
        )
        for row in _required_sequence(
            parameter_manifest.get("regimes"),
            name="stationary-core regimes",
        )
        if isinstance(row, Mapping)
    }
    sources = {
        str(row["regime_id"]): _required_mapping(
            row,
            name="projected-singleton Append source",
        )
        for row in _required_sequence(
            partial.get("included_sources"),
            name="partial-report included sources",
        )
        if isinstance(row, Mapping)
        and row.get("method_family") == "append"
        and row.get("route_id") == SOURCE_ROUTE_ID
    }
    attempts = {
        str(row["execution_id"]): _required_mapping(
            row,
            name="v6 validation attempt",
        )
        for row in _required_sequence(
            validation.get("attempts"),
            name="v6 validation attempts",
        )
        if isinstance(row, Mapping)
    }
    if set(regimes) != set(EXPECTED_PROBLEM_SHA256) or set(sources) != set(
        EXPECTED_PROBLEM_SHA256
    ):
        raise ValueError("adopted singleton six-regime coverage drift.")
    records = []
    for regime_id in EXPECTED_PROBLEM_SHA256:
        execution_id = (
            f"core__{regime_id}__nph"
            f"{int(regimes[regime_id]['n_ph_max'])}__append_singleton"
        )
        print(
            f"building canonical v6 Append registry: {regime_id}",
            flush=True,
        )
        records.append(
            _load_archive_record(
                regime=regimes[regime_id],
                parameter_manifest=parameter_manifest,
                source_record=source_record,
                source=sources[regime_id],
                validation_attempt=attempts[execution_id],
            )
        )
    return {
        "schema": SCHEMA,
        "source_adoption": {
            "path": _display_path(ADOPTION_PATH),
            "sha256": ADOPTION_SHA256,
        },
        "source_partial_report_provenance": {
            "path": _display_path(PARTIAL_REPORT_PROVENANCE_PATH),
            "sha256": PARTIAL_REPORT_PROVENANCE_SHA256,
            "aggregate_report_adopted": False,
        },
        "source_validation": {
            "path": _display_path(VALIDATION_PATH),
            "file_sha256": VALIDATION_FILE_SHA256,
            "sha256": VALIDATION_SHA256,
        },
        "source_package_manifest": {
            "path": _display_path(PACKAGE_MANIFEST_PATH),
            "file_sha256": PACKAGE_MANIFEST_FILE_SHA256,
            "sha256": PACKAGE_MANIFEST_SHA256,
        },
        "route_id": APPEND_ROUTE_ID,
        "reporting_resource_policy": {
            "policy": REPORTING_RESOURCE_POLICY,
            "controller_round": REPORTING_RESOURCE_ROUND,
            "fields": ["N2q", "D2q", "Dc", "W1q", "S_alg"],
        },
        "records": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    payload = build_registry()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256_path(output),
                "records": len(payload["records"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
