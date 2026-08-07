from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from pipelines.reporting import (
    build_paper_i_ra_adapt_stationary_core_master_pdf as report,
)
from pipelines.reporting.paper_i_run_summary import (
    PaperIAlgorithmicWork,
    PaperIWorkComponents,
)
from pipelines.static_adapt.estimator_call_ledger import (
    projective_state_fingerprint,
)


def _sha(character: str) -> str:
    return character * 64


def _closure(*, s_alg: int = 10) -> dict:
    gates = {
        f"G{index}": {"gate_id": f"G{index}", "status": "passed", "evidence": {}}
        for index in range(1, 14)
    }
    gates["G2"]["evidence"] = {
        "verified_ed_reference": {"status": "passed", "E_ED": 0.0}
    }
    gates["G10"]["evidence"] = {"S_alg": s_alg}
    return {
        "status": "passed",
        "full_controller_rounds": 50,
        "gate_ids": list(gates),
        "gates": gates,
    }


def _json_bytes(payload: dict) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _generic_digested(payload: dict) -> dict:
    value = copy.deepcopy(payload)
    value["sha256"] = report._canonical_sha256(value)
    return value


def _write_recovery_adapter_fixture(
    tmp_path: Path,
    *,
    g5_qualification_overrides: dict | None = None,
) -> tuple[Path, dict[str, dict], tuple[str, str]]:
    always_id = "core__weak_weak__nph3__ra_macro_always"
    plateau_id = "core__weak_weak__nph3__ra_macro_plateau"
    expected_jobs = {
        always_id: {
            "regime_id": "weak_weak",
            "candidate_representation": "macro_generator_v1",
            "route_id": "ra_macro_always",
        },
        plateau_id: {
            "regime_id": "weak_weak",
            "candidate_representation": "macro_generator_v1",
            "route_id": "ra_macro_plateau",
        },
    }

    def cell(*, execution_id: str, method: str, status: str) -> dict:
        points = [
            {"k": index, "error": 1.0 / (index + 2)}
            for index in range(51)
        ]
        return {
            "execution_id": execution_id,
            "regime": "weak_weak",
            "representation": "macro_generator_v1",
            "method": method,
            "points": points,
            "marker": {
                "k": 50,
                "error": points[-1]["error"],
                "policy": "terminal_observed_point",
            },
            "terminal": {
                "k": 50,
                "energy": -1.0 + points[-1]["error"],
                "error": points[-1]["error"],
                "N2q": 70,
                "D2q": 60,
                "Dc": 180,
                "W1q": 140,
                "B1q": 100,
                "qiskit_basis_work_status": "ok",
                "qiskit_basis_work_schema": (
                    "qiskit_pretranspile_pauli_basis_work_v1"
                ),
                "S_alg": 1234,
                "status": status,
            },
            "exact_same_cutoff_energy": -1.0,
        }

    def source(
        *,
        package_id: str,
        attempt_status: str,
        worker_exit_status: int,
        digest_character: str,
    ) -> dict:
        return {
            "package_id": package_id,
            "attempt_status": attempt_status,
            "worker_exit_status": worker_exit_status,
            "archive": {
                "path": f"{package_id}.tar.gz",
                "sha256": _sha(digest_character),
                "size_bytes": 4096,
            },
            "result": {
                "path": f"{package_id}/result.json",
                "sha256": _sha({"a": "c", "b": "d"}[digest_character]),
                "size_bytes": 2048,
            },
        }

    g5_qualification = {
        "route_domain_status": "unexercised",
        "interior_scored_count": 0,
        "full_controller_rounds": 50,
        "execution_manifest_status": "passed",
    }
    g5_qualification.update(g5_qualification_overrides or {})
    rows = [
        _generic_digested(
            {
                "target_execution_id": always_id,
                "source_execution_id": (
                    "factorial__weak_weak__ra_macro_always"
                ),
                "recovery_class": (
                    report.RECOVERY_CROSS_CAMPAIGN_CLASS
                ),
                "paper_evidence_eligible": False,
                "source": source(
                    package_id="factorial-source",
                    attempt_status="passed",
                    worker_exit_status=0,
                    digest_character="a",
                ),
                "qualification": {
                    "science_equivalence_status": "passed",
                },
                "cell": cell(
                    execution_id=always_id,
                    method="always",
                    status="complete-Xrev",
                ),
            }
        ),
        _generic_digested(
            {
                "target_execution_id": plateau_id,
                "source_execution_id": plateau_id,
                "recovery_class": report.RECOVERY_G5_UNEXERCISED_CLASS,
                "paper_evidence_eligible": False,
                "source": source(
                    package_id="plateau-source",
                    attempt_status="failed_attempt_retained",
                    worker_exit_status=2,
                    digest_character="b",
                ),
                "qualification": g5_qualification,
                "cell": cell(
                    execution_id=plateau_id,
                    method="plateau",
                    status="complete-G5*",
                ),
            }
        ),
    ]
    adapter = _generic_digested(
        {
            "schema": report.RECOVERY_ADAPTER_SCHEMA,
            "status": "passed",
            "not_paper_evidence": True,
            "cells": rows,
            "recovery_counts": {
                report.RECOVERY_CROSS_CAMPAIGN_CLASS: 1,
                report.RECOVERY_G5_UNEXERCISED_CLASS: 1,
            },
        }
    )
    adapter_path = tmp_path / "recovery_adapter.json"
    adapter_path.write_bytes(_json_bytes(adapter))
    return adapter_path, expected_jobs, (always_id, plateau_id)


def _write_global_singleton_weak_weak_fixture(
    tmp_path: Path,
) -> tuple[Path, dict]:
    exact_energy = -0.9183809199948214
    append_error = 1.2656542480726785e-14
    plateau_error = 9.769962616701378e-15

    def points(*, terminal_error: float) -> list[dict]:
        initial_error = 0.4
        ratio = (terminal_error / initial_error) ** (1.0 / 50.0)
        return [
            {
                "k": index,
                "error": (
                    terminal_error
                    if index == 50
                    else initial_error * ratio**index
                ),
            }
            for index in range(51)
        ]

    append_points = points(terminal_error=append_error)
    plateau_points = points(terminal_error=plateau_error)
    plateau_points[49]["error"] = 9.9e-15

    def observation(
        *,
        k: int,
        error: float,
        s_alg: int,
        n2q: int,
        d2q: int,
        dc: int,
    ) -> dict:
        return {
            "k": k,
            "energy": exact_energy + error,
            "error": error,
            "S_alg": s_alg,
            "N2q": n2q,
            "D2q": d2q,
            "Dc": dc,
            "W1q": 520,
            "B1q": 310,
            "compile_convention": "table_i_basis_gate_transpile_v1",
        }

    append_arm = _generic_digested(
        {
            "schema": (
                "paper_i_ra_adapt_global_singleton_weak_weak_"
                "comparison_arm_v1"
            ),
            "execution_id": (
                "global_singleton__weak_weak__nph3"
                "__ra_global_singleton_append_commutation_reduced"
            ),
            "route_id": (
                "ra_global_singleton_append_commutation_reduced"
            ),
            "insertion_policy": "append_commutation_reduced",
            "points": append_points,
            "terminal": observation(
                k=50,
                error=append_points[50]["error"],
                s_alg=179_375,
                n2q=234,
                d2q=196,
                dc=874,
            ),
            "effective_plateau": observation(
                k=50,
                error=append_points[50]["error"],
                s_alg=179_375,
                n2q=234,
                d2q=196,
                dc=874,
            ),
            "insertion_counts": {
                "round_count": 50,
                "append_count": 50,
                "interior_count": 0,
                "first_interior_round": None,
            },
            "source": {
                "archive": {
                    "path": "append-arm.tar.gz",
                    "sha256": _sha("a"),
                    "size_bytes": 3_340_648_793,
                }
            },
            "qualification": {
                "status": "passed",
                "result_schema": "paper_i_ra_adapt_result_v1",
                "full_controller_rounds": 50,
                "same_cutoff_trace_math": "passed",
                "canonical_work_closure": "passed",
                "authenticated_prefix_reconstruction": "passed",
                "serialized_plateau_qiskit_cross_check": "passed",
                "exact_same_cutoff_energy": exact_energy,
                "route_profile": (
                    "fixture__stationary_source_response_v1__"
                    "all_phase_resource_weighting_v1"
                ),
            },
        }
    )
    plateau_arm = _generic_digested(
        {
            "schema": (
                "paper_i_ra_adapt_global_singleton_weak_weak_"
                "comparison_arm_v1"
            ),
            "execution_id": (
                "global_singleton__weak_weak__nph3"
                "__ra_global_singleton_plateau_commutation"
            ),
            "route_id": "ra_global_singleton_plateau_commutation",
            "insertion_policy": "plateau_commutation",
            "points": plateau_points,
            "terminal": observation(
                k=50,
                error=plateau_points[50]["error"],
                s_alg=903_285,
                n2q=250,
                d2q=211,
                dc=915,
            ),
            "effective_plateau": observation(
                k=49,
                error=plateau_points[49]["error"],
                s_alg=848_329,
                n2q=246,
                d2q=208,
                dc=906,
            ),
            "insertion_counts": {
                "round_count": 50,
                "append_count": 32,
                "interior_count": 18,
                "first_interior_round": 28,
            },
            "source": {
                "archive": {
                    "path": "plateau-arm.tar.gz",
                    "sha256": _sha("b"),
                    "size_bytes": 1_450_570_988,
                }
            },
            "qualification": {
                "status": "passed",
                "result_schema": "paper_i_ra_adapt_result_v1",
                "full_controller_rounds": 50,
                "same_cutoff_trace_math": "passed",
                "canonical_work_closure": "passed",
                "authenticated_prefix_reconstruction": "passed",
                "serialized_plateau_qiskit_cross_check": "passed",
                "exact_same_cutoff_energy": exact_energy,
                "route_profile": (
                    "fixture__stationary_source_response_v1__"
                    "all_phase_resource_weighting_v1"
                ),
            },
        }
    )
    adapter = _generic_digested(
        {
            "schema": report.GLOBAL_SINGLETON_WW_DIAGNOSTIC_SCHEMA,
            "status": "passed",
            "diagnostic_only": True,
            "paper_evidence_adopted": False,
            "campaign_id": report.GLOBAL_SINGLETON_WW_CAMPAIGN_ID,
            "regime_id": "weak_weak",
            "nph": 3,
            "horizon": 50,
            "same_cutoff_exact_energy": exact_energy,
            "cross_arm_audit": {
                "status": "passed",
                "allowed_axis": "insertion_policy",
                "canonical_sha256": (
                    report.GLOBAL_SINGLETON_WW_CROSS_ARM_SHA256
                ),
            },
            "arms": [append_arm, plateau_arm],
            "comparison": {
                "comparison_order": list(
                    report.GLOBAL_SINGLETON_WW_POLICIES
                ),
                "same_cutoff_exact_energy": exact_energy,
                "terminal_energy_interpretation": (
                    "indistinguishable_at_double_precision_floor"
                ),
                "plateau_insertion_domain_exercised": True,
            },
        }
    )
    adapter_path = tmp_path / "global_singleton_weak_weak_adapter.json"
    adapter_path.write_bytes(_json_bytes(adapter))
    return adapter_path, adapter


def _write_partial_append_fixture(
    tmp_path: Path,
    *,
    tamper_job_source_lock: bool = False,
    exact_reference_status: str = "passed",
) -> tuple[Path, Path, str]:
    contract = report._package_contract()
    execution_id = "core__weak_weak__nph3__append_macro"
    expected_job = report._expected_jobs()[execution_id]
    job = copy.deepcopy(expected_job)
    if tamper_job_source_lock:
        job["source_archive_sha256"] = _sha("f")
        job = contract.digested(job)
    closure = _closure(s_alg=1234)
    closure["gates"]["G2"]["evidence"]["verified_ed_reference"][
        "status"
    ] = exact_reference_status
    manifest = {
        "execution_id": execution_id,
        "status": "passed",
        "paper_facing_result_allowed": True,
        "maximum_controller_rounds_override": None,
    }
    history = [
        {
            "controller_round": index,
            "energy_before": 1.0 if index == 1 else 1.0 / index,
            "energy_after": 1.0 / (index + 1),
        }
        for index in range(1, 51)
    ]
    summary = {
        "schema": "paper_i_append_run_summary_v1",
        "controller_rounds_completed": 50,
        "protocol_horizon": 50,
        "stop_reason": "maximum_controller_rounds",
        "final_energy": 1.0 / 51,
        "accepted_history": history,
        "estimator_accounting": {"S_alg": 1234},
        "resources": {
            "terminal_observation_status": "ok",
            "terminal_compiled_resources": {
                "compiled_circuit_stats_status": "ok",
                "compiled_count_2q_total": 70,
                "compiled_depth_2q_total": 60,
                "compiled_depth_total": 180,
                "qiskit_basis_work_status": "ok",
                "qiskit_basis_work_schema": (
                    "qiskit_pretranspile_pauli_basis_work_v1"
                ),
                "qiskit_pretranspile_basis_change_1q_total": 100,
                "qiskit_pretranspile_pauli_1q_work_total": 140,
            },
        },
    }
    result = {
        "schema": "paper_i_append_adapt_result_v1",
        "paper_i_summary": summary,
    }
    artifact_payloads = {
        "execution_manifest": manifest,
        "result": result,
        "summary": summary,
    }
    artifact_bytes = {
        role: _json_bytes(payload)
        for role, payload in artifact_payloads.items()
    }
    worker = contract.digested(
        {
            "schema": contract.WORKER_RECEIPT_SCHEMA,
            "package_id": report.PACKAGE_ID,
            "execution_id": execution_id,
            "status": "passed",
            "scientific_closure": closure,
            "artifact_bindings": [
                {
                    "role": role,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw),
                }
                for role, raw in artifact_bytes.items()
            ],
        }
    )
    payloads = {
        "job": job,
        "worker": worker,
        "manifest": manifest,
        "result": result,
        "summary": summary,
    }
    fetched_dir = tmp_path / "fetched"
    fetched_dir.mkdir()
    attempt = fetched_dir / "attempt.tar.gz"
    names = report._attempt_member_names(execution_id)
    with tarfile.open(attempt, "w:gz") as archive:
        for role, payload in payloads.items():
            raw = _json_bytes(payload)
            member = tarfile.TarInfo(names[role])
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    attempt_sha = report._sha256_file(attempt)
    validation = contract.digested(
        {
            "schema": report.VALIDATION_SCHEMA,
            "package_id": report.PACKAGE_ID,
            "attempt_count": 1,
            "attempts": [
                {
                    "execution_id": execution_id,
                    "path": attempt.name,
                    "sha256": attempt_sha,
                    "worker_receipt_sha256": worker["sha256"],
                    "status": "passed",
                }
            ],
            "execution_ids_with_passed_attempts": [execution_id],
            "automatic_attempt_selection_performed": False,
            "paper_evidence_adopted": False,
            "status": "validated_no_selection",
        }
    )
    validation_path = tmp_path / "validation.json"
    validation_path.write_bytes(contract.canonical_json_bytes(validation) + b"\n")
    return validation_path, fetched_dir, execution_id


def _ra_payloads() -> tuple[dict, dict, dict, dict]:
    labels = tuple(f"operator-{index}" for index in range(50))
    reference = (1.0 + 0.0j, 0.0 + 0.0j)
    reference_fingerprint = projective_state_fingerprint(reference)
    state = {
        "controller_round": 50,
        "energy": 0.01,
        "operators": list(labels),
        "logical_parameters": [0.1] * 50,
        "runtime_parameters": [0.1] * 50,
        "projective_state_fingerprint": "terminal-state",
    }
    checkpoint = {
        "outer_iteration": 50,
        "active_ansatz_depth": 50,
        "ordered_operator_labels": list(labels),
        "logical_parameters": [0.1] * 50,
        "runtime_parameters": [0.1] * 50,
        "parameter_blocks": [
            {
                "candidate_label": label,
                "logical_index": index,
                "runtime_start": index,
                "runtime_count": 1,
                "execution_mode": "termwise_product",
                "runtime_terms": [
                    {
                        "pauli_exyz": "x",
                        "coefficient_real": 1.0,
                        "coefficient_imaginary": 0.0,
                        "qubit_count": 1,
                    }
                ],
            }
            for index, label in enumerate(labels)
        ],
        "checkpoint_sha256": _sha("a"),
        "projective_state_fingerprint": "terminal-state",
        "parameterization_mode": "per_pauli_term_v1",
        "parameterization_term_order": "sorted",
        "estimator_ledger_status": "complete",
        "estimator_ledger_s_alg": 10,
        "strict_replay_passed": True,
        "strict_replay_fidelity": 1.0,
        "route_profile": "ra-profile",
        "route_contract_sha256": _sha("b"),
    }
    transitions = [
        {
            "controller_round": index,
            "energy_before": 1.0 if index == 1 else 1.0 / index,
            "energy_after": 1.0 / (index + 1),
            "cumulative_s_alg": 10 if index == 50 else min(index, 10),
        }
        for index in range(1, 51)
    ]
    result = {
        "schema": "paper_i_ra_adapt_result_v1",
        "run": {
            "accepted_trajectory": [{}, *({} for _ in range(48)), state],
            "accepted_transitions": transitions,
            "scientific_replay": [
                {},
                *({} for _ in range(48)),
                {"accepted_state": state, "checkpoint": checkpoint},
            ],
            "canonical_reporting": {
                "accepted_prefix_work": [
                    {
                        "components": {
                            "n_h_outer": 1,
                            "n_h_refit": 2,
                            "n_grad": 3,
                            "n_metric": 4,
                        },
                        "s_alg": 10,
                    }
                    for _ in range(50)
                ],
                "reference_state": {
                    "amplitudes_real": [1.0, 0.0],
                    "amplitudes_imaginary": [0.0, 0.0],
                    "qubit_count": 1,
                    "source_label": "hf",
                    "state_fingerprint": reference_fingerprint,
                },
            },
            "route": {
                "profile": "ra-profile",
                "contract_sha256": _sha("b"),
            },
            "problem": {"problem_request_sha256": _sha("c")},
        },
    }
    summary = {
        "schema": "paper_i_run_summary_v1",
        "accepted_error_trace": [
            {
                "controller_round": index,
                "accepted_energy": 1.0 / (index + 1),
                "absolute_energy_error": 1.0 / (index + 1),
                "active_ansatz_depth": index,
            }
            for index in range(1, 51)
        ],
        "canonical_all_work": {
            "components": {
                "n_h_outer": 1,
                "n_h_refit": 2,
                "n_grad": 3,
                "n_metric": 4,
            },
            "s_alg": 10,
        },
        "effective_plateau": {
            "controller_round": 12,
            "resources": {
                "compiled_two_qubit_count": 999_999,
                "compiled_two_qubit_depth": 999_999,
                "compiled_total_depth": 999_999,
            },
        },
        "provenance": {"exact_same_cutoff_energy": 0.0},
    }
    job = {
        "execution_id": "ra-test",
        "regime_id": "weak_weak",
        "candidate_representation": "macro_generator_v1",
        "route_id": "ra_macro_plateau",
    }
    return job, result, summary, _closure()


def _append_prefix_fixture(*, s_alg: int = 1234):
    _, result, _, _ = _ra_payloads()
    prefix = report._ra_terminal_prefix(result, s_alg=10)
    return replace(
        prefix,
        source_method="append_adapt",
        algorithmic_work=PaperIAlgorithmicWork(
            components=PaperIWorkComponents(
                n_h_outer=s_alg,
                n_h_refit=0,
                n_grad=0,
                n_metric=0,
            ),
            s_alg=s_alg,
        ),
    )


def test_package_jobs_are_exact_48_cell_matrix() -> None:
    jobs = report._expected_jobs()
    assert len(jobs) == 48
    assert len(report._pending_cells()) == 48
    assert {
        (
            row["regime_id"],
            row["candidate_representation"],
            report._method_key(row["route_id"]),
        )
        for row in jobs.values()
    } == {
        (regime, representation, method)
        for regime in report.REGIME_ORDER
        for representation in report.REPRESENTATIONS.values()
        for method in report.METHOD_ORDER
    }


def test_ra_terminal_cost_uses_authenticated_terminal_compile_not_plateau() -> None:
    job, result, summary, closure = _ra_payloads()
    observed_prefix = None

    def compiler(prefix):
        nonlocal observed_prefix
        observed_prefix = prefix
        return {
            "compile_convention": "table_i_basis_gate_transpile_v1",
            "compiled_two_qubit_count": 101,
            "compiled_two_qubit_depth": 88,
            "compiled_total_depth": 250,
            "qiskit_basis_work_status": "ok",
            "qiskit_basis_work_schema": (
                "qiskit_pretranspile_pauli_basis_work_v1"
            ),
            "qiskit_pretranspile_basis_change_1q_total": 150,
            "qiskit_pretranspile_pauli_1q_work_total": 200,
        }

    cell = report._extract_ra_cell(
        execution_id="ra-test",
        job=job,
        result=result,
        summary=summary,
        closure=closure,
        exact_energy=0.0,
        compiler=compiler,
    )
    assert observed_prefix is not None
    assert observed_prefix.source_method == "ra_adapt"
    assert observed_prefix.controller_round == 50
    assert observed_prefix.checkpoint_sha256 == _sha("a")
    assert len(cell["points"]) == 51
    assert [row["k"] for row in cell["points"]] == list(range(51))
    assert cell["terminal"] == {
        "k": 50,
        "error": pytest.approx(1.0 / 51),
        "N2q": 101,
        "D2q": 88,
        "Dc": 250,
        "W1q": 200,
        "B1q": 150,
        "qiskit_basis_work_status": "ok",
        "qiskit_basis_work_schema": (
            "qiskit_pretranspile_pauli_basis_work_v1"
        ),
        "S_alg": 10,
        "status": "complete",
    }
    assert cell["marker"] == {
        "k": 12,
        "error": pytest.approx(1.0 / 13),
        "policy": "first_effective_plateau_prefix",
    }
    assert cell["terminal"]["N2q"] != 999_999

    def unavailable_basis_work(prefix):
        return {
            "compile_convention": "table_i_basis_gate_transpile_v1",
            "compiled_two_qubit_count": 101,
            "compiled_two_qubit_depth": 88,
            "compiled_total_depth": 250,
            "qiskit_basis_work_status": (
                "unavailable_noncommuting_grouped_exact_synthesis"
            ),
            "qiskit_pretranspile_pauli_1q_work_total": None,
        }

    with pytest.raises(
        report.ReportInputError,
        match="five-coordinate Qiskit cost is unavailable",
    ):
        report._extract_ra_cell(
            execution_id="ra-test",
            job=job,
            result=result,
            summary=summary,
            closure=closure,
            exact_energy=0.0,
            compiler=unavailable_basis_work,
        )

    with pytest.raises(
        report.ReportInputError,
        match="compiler convention drifted",
    ):
        report._extract_ra_cell(
            execution_id="ra-test",
            job=job,
            result=result,
            summary=summary,
            closure=closure,
            exact_energy=0.0,
            compiler=lambda prefix: {
                "compiled_two_qubit_count": 101,
                "compiled_two_qubit_depth": 88,
                "compiled_total_depth": 250,
                "qiskit_basis_work_status": "ok",
                "qiskit_basis_work_schema": (
                    "qiskit_pretranspile_pauli_basis_work_v1"
                ),
                "qiskit_pretranspile_basis_change_1q_total": 150,
                "qiskit_pretranspile_pauli_1q_work_total": 200,
            },
        )


def test_append_terminal_recompiles_through_common_seam_and_crosschecks_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = [
        {
            "controller_round": index,
            "energy_before": 1.0 if index == 1 else 1.0 / index,
            "energy_after": 1.0 / (index + 1),
        }
        for index in range(1, 51)
    ]
    summary = {
        "schema": "paper_i_append_run_summary_v1",
        "controller_rounds_completed": 50,
        "protocol_horizon": 50,
        "stop_reason": "maximum_controller_rounds",
        "final_energy": 1.0 / 51,
        "accepted_history": history,
        "estimator_accounting": {"S_alg": 1234},
        "resources": {
            "terminal_observation_status": "ok",
            "terminal_compiled_resources": {
                "compiled_circuit_stats_status": "ok",
                "compiled_count_2q_total": 70,
                "compiled_depth_2q_total": 60,
                "compiled_depth_total": 180,
                "qiskit_basis_work_status": "ok",
                "qiskit_basis_work_schema": (
                    "qiskit_pretranspile_pauli_basis_work_v1"
                ),
                "qiskit_pretranspile_basis_change_1q_total": 100,
                "qiskit_pretranspile_pauli_1q_work_total": 140,
            },
        },
    }
    closure = _closure(s_alg=1234)
    prefix = _append_prefix_fixture()
    observed = []

    monkeypatch.setattr(
        report,
        "_append_terminal_prefix",
        lambda result, *, job, s_alg: prefix,
    )

    def compiler(received):
        observed.append((received.source_method, received.controller_round))
        return {
            "compile_convention": "table_i_basis_gate_transpile_v1",
            "compiled_two_qubit_count": 70,
            "compiled_two_qubit_depth": 60,
            "compiled_total_depth": 180,
            "qiskit_basis_work_status": "ok",
            "qiskit_basis_work_schema": (
                "qiskit_pretranspile_pauli_basis_work_v1"
            ),
            "qiskit_pretranspile_basis_change_1q_total": 100,
            "qiskit_pretranspile_pauli_1q_work_total": 140,
        }

    cell = report._extract_append_cell(
        execution_id="append-test",
        job={
            "regime_id": "weak_weak",
            "candidate_representation": "macro_generator_v1",
            "route_id": "append_macro",
        },
        result={
            "schema": "paper_i_append_adapt_result_v1",
            "paper_i_summary": summary,
        },
        summary=summary,
        closure=closure,
        exact_energy=0.0,
        compiler=compiler,
    )
    assert observed == [("append_adapt", 50)]
    assert len(cell["points"]) == 51
    assert cell["terminal"] == {
        "k": 50,
        "energy": pytest.approx(1.0 / 51),
        "error": pytest.approx(1.0 / 51),
        "N2q": 70,
        "D2q": 60,
        "Dc": 180,
        "W1q": 140,
        "B1q": 100,
        "qiskit_basis_work_status": "ok",
        "qiskit_basis_work_schema": (
            "qiskit_pretranspile_pauli_basis_work_v1"
        ),
        "S_alg": 1234,
        "status": "complete",
    }
    assert cell["marker"] == {
        "k": 50,
        "error": pytest.approx(1.0 / 51),
        "policy": "terminal_observed_point",
    }
    assert cell["terminal_checkpoint_sha256"] == prefix.checkpoint_sha256
    assert cell["terminal_compile_source"] == (
        "common_typed_terminal_prefix_recompile_v1"
    )
    assert cell["serialized_terminal_cross_check"] == "passed"


@pytest.mark.parametrize(
    ("field", "label"),
    (
        ("compiled_count_2q_total", "N2q"),
        ("compiled_depth_2q_total", "D2q"),
        ("compiled_depth_total", "Dc"),
        ("qiskit_pretranspile_pauli_1q_work_total", "W1q"),
        ("qiskit_pretranspile_basis_change_1q_total", "B1q"),
    ),
)
def test_append_terminal_recompile_rejects_serialized_cost_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    label: str,
) -> None:
    prefix = _append_prefix_fixture()
    monkeypatch.setattr(
        report,
        "_append_terminal_prefix",
        lambda result, *, job, s_alg: prefix,
    )
    summary = {
        "schema": "paper_i_append_run_summary_v1",
        "controller_rounds_completed": 50,
        "protocol_horizon": 50,
        "stop_reason": "maximum_controller_rounds",
        "final_energy": 1.0 / 51,
        "accepted_history": [
            {
                "controller_round": index,
                "energy_before": 1.0 if index == 1 else 1.0 / index,
                "energy_after": 1.0 / (index + 1),
            }
            for index in range(1, 51)
        ],
        "estimator_accounting": {"S_alg": 1234},
        "resources": {
            "terminal_observation_status": "ok",
            "terminal_compiled_resources": {
                "compiled_circuit_stats_status": "ok",
                "compiled_count_2q_total": 70,
                "compiled_depth_2q_total": 60,
                "compiled_depth_total": 180,
                "qiskit_basis_work_status": "ok",
                "qiskit_basis_work_schema": (
                    "qiskit_pretranspile_pauli_basis_work_v1"
                ),
                "qiskit_pretranspile_basis_change_1q_total": 100,
                "qiskit_pretranspile_pauli_1q_work_total": 140,
            },
        },
    }
    summary["resources"]["terminal_compiled_resources"][field] += 1

    with pytest.raises(
        report.ReportInputError,
        match=rf"serialized terminal Qiskit {label} mismatch",
    ):
        report._extract_append_cell(
            execution_id="append-test",
            job={
                "regime_id": "weak_weak",
                "candidate_representation": "macro_generator_v1",
                "route_id": "append_macro",
            },
            result={
                "schema": "paper_i_append_adapt_result_v1",
                "paper_i_summary": summary,
            },
            summary=summary,
            closure=_closure(s_alg=1234),
            exact_energy=0.0,
            compiler=lambda received: {
                "compile_convention": "table_i_basis_gate_transpile_v1",
                "compiled_two_qubit_count": 70,
                "compiled_two_qubit_depth": 60,
                "compiled_total_depth": 180,
                "qiskit_basis_work_status": "ok",
                "qiskit_basis_work_schema": (
                    "qiskit_pretranspile_pauli_basis_work_v1"
                ),
                "qiskit_pretranspile_basis_change_1q_total": 100,
                "qiskit_pretranspile_pauli_1q_work_total": 140,
            },
        )


def test_terminal_qiskit_compile_identity_is_independent_of_s_alg() -> None:
    first = _append_prefix_fixture(s_alg=1234)
    second = _append_prefix_fixture(s_alg=5678)
    assert first.compile_cache_key == second.compile_cache_key

    def compiler(prefix):
        return {
            "compile_convention": "table_i_basis_gate_transpile_v1",
            "compiled_two_qubit_count": 70,
            "compiled_two_qubit_depth": 60,
            "compiled_total_depth": 180,
            "qiskit_basis_work_status": "ok",
            "qiskit_basis_work_schema": (
                "qiskit_pretranspile_pauli_basis_work_v1"
            ),
            "qiskit_pretranspile_basis_change_1q_total": 100,
            "qiskit_pretranspile_pauli_1q_work_total": 140,
        }

    first_cost, _, _ = report._compile_terminal_qiskit(
        first, compiler=compiler
    )
    second_cost, _, _ = report._compile_terminal_qiskit(
        second, compiler=compiler
    )
    assert first_cost == second_cost
    assert first.algorithmic_work.s_alg == 1234
    assert second.algorithmic_work.s_alg == 5678


def test_append_terminal_prefix_rejects_job_bound_protocol_drift() -> None:
    execution_id = "core__weak_weak__nph3__append_macro"
    job = report._expected_jobs()[execution_id]
    with pytest.raises(
        report.ReportInputError,
        match="differs from the job-bound protocol",
    ):
        report._append_terminal_prefix(
            {
                "schema": "paper_i_append_adapt_result_v1",
                "protocol": {"schema": "tampered"},
            },
            job=job,
            s_alg=1234,
        )


def test_append_reporting_protocol_loader_is_cell_scoped() -> None:
    package_root = report.REPO_ROOT / (
        "chtc/paper_i_ra_adapt_repair_20260727"
    )
    v6_package = (
        package_root / "stationary_core_full48_r50_20260728_v6_chtc"
    )
    report._configure_package_dir(v6_package)
    try:
        execution_id = "core__weak_weak__nph3__append_macro"
        job = report._expected_jobs()[execution_id]
        expected_protocol = report._protocol_for_job(job)
        protocol = report._append_protocol_for_reporting(
            job=job,
            expected_protocol=expected_protocol,
        )
        assert protocol.to_dict() == expected_protocol
        assert protocol._materialization_authority is None
    finally:
        report._configure_package_dir(report.DEFAULT_PACKAGE_DIR)


def test_terminal_cost_plot_overlay_formats_only_completed_round_50_rows() -> None:
    completed = {
        "k": 50,
        "error": 1.0e-3,
        "N2q": 70,
        "D2q": 60,
        "Dc": 180,
        "W1q": 140,
        "S_alg": 1234,
        "status": "complete",
    }
    assert report._terminal_cost_plot_overlay(
        terminal=completed,
        method="append",
    ) == r"$\bullet\;(70,60,180,140,1.2\mathrm{e}3)$"
    assert report._terminal_cost_plot_overlay(
        terminal=completed,
        method="no_insertion",
    ) == r"$\boxminus\;(70,60,180,140,1.2\mathrm{e}3)$"
    assert report._terminal_cost_plot_overlay(
        terminal={**completed, "k": 49},
        method="append",
    ) is None
    assert report._terminal_cost_plot_overlay(
        terminal={
            "k": None,
            "error": None,
            "N2q": None,
            "D2q": None,
            "Dc": None,
            "W1q": None,
            "S_alg": None,
            "status": "pending",
        },
        method="append",
    ) is None


def test_terminal_cost_plot_overlay_mathtext_is_pdf_renderable() -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.mathtext import MathTextParser

    parser = MathTextParser("path")
    terminal = {
        "k": 50,
        "error": 1.0e-3,
        "N2q": 70,
        "D2q": 60,
        "Dc": 180,
        "W1q": 140,
        "S_alg": 1234,
        "status": "complete",
    }
    for method in report.METHOD_ORDER:
        overlay = report._terminal_cost_plot_overlay(
            terminal=terminal,
            method=method,
        )
        assert overlay is not None
        parser.parse(overlay, dpi=72)


def test_attempt_loader_uses_real_package_prefixed_worker_archive_layout(
    tmp_path: Path,
) -> None:
    execution_id = "core__weak_weak__nph3__append_macro"
    member_names = report._attempt_member_names(execution_id)
    assert member_names["job"] == (
        f"{report.PACKAGE_RELATIVE_ROOT}/jobs/{execution_id}.json"
    )
    assert member_names["worker"] == "worker_outputs/worker_receipt.json"

    archive_path = tmp_path / "attempt.tar.gz"
    expected = {
        role: {"role": role, "execution_id": execution_id}
        for role in member_names
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        for role, name in member_names.items():
            raw = (
                json.dumps(expected[role], sort_keys=True) + "\n"
            ).encode("utf-8")
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))

    loaded = report._load_attempt_payloads(
        archive_path,
        execution_id=execution_id,
    )
    assert {role: row[0] for role, row in loaded.items()} == expected


def test_partial_loader_consumes_unique_validated_v6_subset_and_fills_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation, fetched_dir, execution_id = _write_partial_append_fixture(
        tmp_path
    )
    prefix = _append_prefix_fixture()
    monkeypatch.setattr(
        report,
        "_append_terminal_prefix",
        lambda result, *, job, s_alg: prefix,
    )
    included, sources = report.load_partial_cells(
        validation_path=validation,
        fetched_dir=fetched_dir,
        terminal_qiskit_compiler=lambda received: {
            "compile_convention": "table_i_basis_gate_transpile_v1",
            "compiled_two_qubit_count": 70,
            "compiled_two_qubit_depth": 60,
            "compiled_total_depth": 180,
            "qiskit_basis_work_status": "ok",
            "qiskit_basis_work_schema": (
                "qiskit_pretranspile_pauli_basis_work_v1"
            ),
            "qiskit_pretranspile_basis_change_1q_total": 100,
            "qiskit_pretranspile_pauli_1q_work_total": 140,
        },
    )
    assert [row["execution_id"] for row in included] == [execution_id]
    assert sources["automatic_attempt_selection_performed"] is False
    assert sources["inclusion_policy"] == (
        "all_execution_ids_with_exactly_one_passed_validated_attempt_v1"
    )
    assert sources["included_sources"][0][
        "exact_same_cutoff_energy"
    ] == 0.0
    merged = report._merge_partial_with_pending(included)
    assert len(merged) == 48
    assert sum(bool(row["points"]) for row in merged) == 1
    assert sum(
        row["terminal"]["status"] == "pending" for row in merged
    ) == 47
    assert next(
        row for row in merged if row["execution_id"] == execution_id
    )["terminal"]["status"] == "complete"


@pytest.mark.parametrize(
    ("tamper_job_source_lock", "exact_reference_status", "error_match"),
    (
        (True, "passed", "archived job spec drifted"),
        (False, "failed", "same-cutoff reference failed"),
    ),
)
def test_partial_loader_rechecks_source_lock_and_same_cutoff_closure(
    tmp_path: Path,
    tamper_job_source_lock: bool,
    exact_reference_status: str,
    error_match: str,
) -> None:
    validation, fetched_dir, _ = _write_partial_append_fixture(
        tmp_path,
        tamper_job_source_lock=tamper_job_source_lock,
        exact_reference_status=exact_reference_status,
    )
    with pytest.raises(report.ReportInputError, match=error_match):
        report.load_partial_cells(
            validation_path=validation,
            fetched_dir=fetched_dir,
        )


def test_partial_loader_never_chooses_between_passed_retries(
    tmp_path: Path,
) -> None:
    validation_path, fetched_dir, execution_id = (
        _write_partial_append_fixture(tmp_path)
    )
    contract = report._package_contract()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation.pop("sha256")
    duplicate = dict(validation["attempts"][0])
    duplicate["path"] = "second-attempt.tar.gz"
    validation["attempts"].append(duplicate)
    validation["attempt_count"] = 2
    validation = contract.digested(validation)
    validation_path.write_bytes(
        contract.canonical_json_bytes(validation) + b"\n"
    )
    with pytest.raises(
        report.ReportInputError,
        match="will not choose among passed retries",
    ):
        report.load_partial_cells(
            validation_path=validation_path,
            fetched_dir=fetched_dir,
        )
    assert validation["execution_ids_with_passed_attempts"] == [execution_id]


def test_partial_loader_refuses_complete_matrix_without_final_selection(
    tmp_path: Path,
) -> None:
    contract = report._package_contract()
    execution_ids = sorted(report._expected_jobs())
    validation = contract.digested(
        {
            "schema": report.VALIDATION_SCHEMA,
            "package_id": report.PACKAGE_ID,
            "attempt_count": 48,
            "attempts": [
                {
                    "execution_id": execution_id,
                    "path": f"{execution_id}.tar.gz",
                    "sha256": _sha("a"),
                    "worker_receipt_sha256": _sha("b"),
                    "status": "passed",
                }
                for execution_id in execution_ids
            ],
            "execution_ids_with_passed_attempts": execution_ids,
            "automatic_attempt_selection_performed": False,
            "paper_evidence_adopted": False,
            "status": "validated_no_selection",
        }
    )
    validation_path = tmp_path / "validation.json"
    validation_path.write_bytes(
        contract.canonical_json_bytes(validation) + b"\n"
    )
    with pytest.raises(
        report.ReportInputError,
        match="final mode requires explicit selection",
    ):
        report.load_partial_cells(
            validation_path=validation_path,
            fetched_dir=tmp_path / "fetched",
        )


def test_cross_revision_loader_keeps_disjoint_explicit_method_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = report.REPO_ROOT / (
        "chtc/paper_i_ra_adapt_repair_20260727"
    )
    v6_package = (
        package_root / "stationary_core_full48_r50_20260728_v6_chtc"
    )
    v7_package = (
        package_root / "stationary_core_full48_r50_20260728_v7_chtc"
    )
    report._configure_package_dir(v6_package)
    append_cell = next(
        dict(row)
        for row in report._pending_cells()
        if row["execution_id"]
        == "core__weak_weak__nph3__append_macro"
    )
    report._configure_package_dir(v7_package)
    ra_cells = [
        next(
            dict(row)
            for row in report._pending_cells()
            if row["execution_id"] == execution_id
        )
        for execution_id in (
            "core__weak_weak__nph3__ra_macro_plateau",
            "core__strong_weak_u8__nph3__ra_singleton_always",
        )
    ]

    def fake_manifest() -> dict:
        return {
            "schema": "fixture_science_manifest_v1",
            "model_family": "Hubbard-Holstein",
            "package_provenance": {"package_id": report.PACKAGE_ID},
        }

    def fake_package_sources() -> dict:
        return {
            "package_manifest": {"sha256": _sha("a")},
            "source_archive_sha256": _sha("b"),
            "core_materialization_id": report.CORE_MATERIALIZATION_ID,
        }

    def fake_load_partial_cells(**kwargs):
        family = kwargs["method_family"]
        validation_name = kwargs["validation_path"].name
        if family == "append":
            cells = [append_cell]
        elif validation_name == "ra-1.json":
            cells = [ra_cells[0]]
        else:
            cells = [ra_cells[1]]
        return cells, {
            "validation": {
                "path": str(kwargs["validation_path"]),
                "sha256": _sha(str(len(validation_name) % 10)),
                "file_sha256": _sha("c"),
            },
            "inclusion_policy": (
                "all_execution_ids_with_exactly_one_passed_"
                "validated_attempt_v1"
            ),
            "automatic_attempt_selection_performed": False,
            "excluded_nonpassed_execution_ids": (
                ["core__weak_weak__nph3__ra_macro_append_only"]
                if validation_name == "ra-1.json"
                else []
            ),
            "included_sources": [
                {
                    "execution_id": cell["execution_id"],
                    "attempt_path": f"{cell['execution_id']}.tar.gz",
                    "attempt_sha256": _sha("d"),
                }
                for cell in cells
            ],
        }

    monkeypatch.setattr(report, "_parameter_manifest", fake_manifest)
    monkeypatch.setattr(report, "_package_sources", fake_package_sources)
    monkeypatch.setattr(
        report, "load_partial_cells", fake_load_partial_cells
    )
    try:
        cells, provenance, manifest = (
            report.load_cross_revision_partial_cells(
                source_specs=[
                    {
                        "method_family": "append",
                        "package_dir": v6_package,
                        "validation_path": tmp_path / "append.json",
                        "fetched_dir": tmp_path / "v6",
                    },
                    {
                        "method_family": "ra",
                        "package_dir": v7_package,
                        "validation_path": tmp_path / "ra-1.json",
                        "fetched_dir": tmp_path / "v7",
                    },
                    {
                        "method_family": "ra",
                        "package_dir": v7_package,
                        "validation_path": tmp_path / "ra-2.json",
                        "fetched_dir": tmp_path / "v7",
                    },
                ]
            )
        )
    finally:
        report._configure_package_dir(report.DEFAULT_PACKAGE_DIR)

    assert {row["execution_id"] for row in cells} == {
        append_cell["execution_id"],
        *(row["execution_id"] for row in ra_cells),
    }
    assert manifest["package_provenance"]["cross_revision"] is True
    assert manifest["package_provenance"]["source_package_count"] == 2
    assert len(provenance["source_records"]) == 3
    assert {
        row["package_id"] for row in provenance["included_sources"]
    } == {
        "paper_i_ra_adapt_stationary_core_full48_r50_20260728_v6_chtc",
        "paper_i_ra_adapt_stationary_core_full48_r50_20260728_v7_chtc",
    }
    assert provenance["source_records"][1][
        "excluded_nonpassed_execution_ids"
    ] == ["core__weak_weak__nph3__ra_macro_append_only"]


def test_cross_revision_loader_rejects_overlapping_passed_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = report.REPO_ROOT / (
        "chtc/paper_i_ra_adapt_repair_20260727"
    )
    v6_package = (
        package_root / "stationary_core_full48_r50_20260728_v6_chtc"
    )
    v7_package = (
        package_root / "stationary_core_full48_r50_20260728_v7_chtc"
    )
    report._configure_package_dir(v6_package)
    append_cell = next(
        dict(row)
        for row in report._pending_cells()
        if row["execution_id"]
        == "core__weak_weak__nph3__append_macro"
    )
    report._configure_package_dir(v7_package)
    ra_cell = next(
        dict(row)
        for row in report._pending_cells()
        if row["execution_id"]
        == "core__weak_weak__nph3__ra_macro_plateau"
    )

    monkeypatch.setattr(
        report,
        "_parameter_manifest",
        lambda: {
            "schema": "fixture_science_manifest_v1",
            "package_provenance": {"package_id": report.PACKAGE_ID},
        },
    )
    monkeypatch.setattr(
        report,
        "_package_sources",
        lambda: {
            "package_manifest": {"sha256": _sha("a")},
            "source_archive_sha256": _sha("b"),
            "core_materialization_id": report.CORE_MATERIALIZATION_ID,
        },
    )

    call_count = 0

    def duplicate_ra_loader(**kwargs):
        nonlocal call_count
        call_count += 1
        cells = [append_cell] if call_count == 1 else [ra_cell]
        return cells, {
            "validation": {
                "path": str(kwargs["validation_path"]),
                "sha256": _sha("c"),
                "file_sha256": _sha("d"),
            },
            "inclusion_policy": "fixture",
            "automatic_attempt_selection_performed": False,
            "excluded_nonpassed_execution_ids": [],
            "included_sources": [
                {"execution_id": cell["execution_id"]}
                for cell in cells
            ],
        }

    monkeypatch.setattr(
        report, "load_partial_cells", duplicate_ra_loader
    )
    try:
        with pytest.raises(
            report.ReportInputError,
            match="sources overlap successful cells",
        ):
            report.load_cross_revision_partial_cells(
                source_specs=[
                    {
                        "method_family": "append",
                        "package_dir": v6_package,
                        "validation_path": tmp_path / "append.json",
                        "fetched_dir": tmp_path / "v6",
                    },
                    {
                        "method_family": "ra",
                        "package_dir": v7_package,
                        "validation_path": tmp_path / "ra-1.json",
                        "fetched_dir": tmp_path / "v7",
                    },
                    {
                        "method_family": "ra",
                        "package_dir": v7_package,
                        "validation_path": tmp_path / "ra-2.json",
                        "fetched_dir": tmp_path / "v7",
                    },
                ]
            )
    finally:
        report._configure_package_dir(report.DEFAULT_PACKAGE_DIR)


def test_global_singleton_weak_weak_loader_rejects_tampered_arm_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_path, adapter = _write_global_singleton_weak_weak_fixture(
        tmp_path
    )
    tampered = copy.deepcopy(adapter)
    tampered["arms"][0]["terminal"]["S_alg"] += 1
    tampered["sha256"] = report._canonical_sha256(
        {
            key: value
            for key, value in tampered.items()
            if key != "sha256"
        }
    )
    adapter_path.write_bytes(_json_bytes(tampered))
    monkeypatch.setattr(report, "REPO_ROOT", tmp_path)

    with pytest.raises(
        report.ReportInputError,
        match="global-singleton weak-weak arm 1 self-digest failed",
    ):
        report._load_global_singleton_weak_weak_diagnostic(adapter_path)


def test_recovery_adapter_loads_cross_campaign_and_g5_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_path, expected_jobs, execution_ids = (
        _write_recovery_adapter_fixture(tmp_path)
    )
    monkeypatch.setattr(report, "REPO_ROOT", tmp_path)

    cells, provenance = report._load_recovery_adapter(
        adapter_path=adapter_path,
        expected_jobs=expected_jobs,
    )

    assert [cell["execution_id"] for cell in cells] == list(execution_ids)
    assert [cell["terminal"]["status"] for cell in cells] == [
        "complete-Xrev",
        "complete-G5*",
    ]
    assert all(len(cell["points"]) == 51 for cell in cells)
    assert [cell["terminal"]["W1q"] for cell in cells] == [140, 140]
    assert [cell["terminal"]["S_alg"] for cell in cells] == [1234, 1234]
    assert provenance["included_count"] == 2
    assert provenance["included_execution_ids"] == sorted(execution_ids)
    assert provenance["recovery_counts"] == {
        report.RECOVERY_CROSS_CAMPAIGN_CLASS: 1,
        report.RECOVERY_G5_UNEXERCISED_CLASS: 1,
    }
    assert provenance["source_package_ids"] == [
        "factorial-source",
        "plateau-source",
    ]
    assert {
        (
            source["attempt_size_bytes"],
            source["result_size_bytes"],
            source["paper_evidence_eligible"],
        )
        for source in provenance["included_sources"]
    } == {(4096, 2048, False)}
    assert provenance["sha256"] == json.loads(
        adapter_path.read_text(encoding="utf-8")
    )["sha256"]
    assert provenance["file_sha256"] == report._sha256_file(adapter_path)


@pytest.mark.parametrize(
    "qualification_override",
    [
        {"interior_scored_count": 1},
        {"route_domain_status": "exercised"},
    ],
)
def test_recovery_adapter_rejects_g5_qualification_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qualification_override: dict,
) -> None:
    adapter_path, expected_jobs, _ = _write_recovery_adapter_fixture(
        tmp_path,
        g5_qualification_overrides=qualification_override,
    )
    monkeypatch.setattr(report, "REPO_ROOT", tmp_path)

    with pytest.raises(
        report.ReportInputError,
        match="G5 recovery qualification drifted",
    ):
        report._load_recovery_adapter(
            adapter_path=adapter_path,
            expected_jobs=expected_jobs,
        )


def test_parameter_manifest_is_v10_same_cutoff_full48_authority() -> None:
    package_manifest = json.loads(
        (report.PACKAGE_DIR / "package_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = report._parameter_manifest()
    assert manifest["model_family"] == "Hubbard-Holstein"
    assert manifest["num_sites"] == 2
    assert manifest["drive_enabled"] is False
    assert manifest["optimizer"] == "powell"
    assert manifest["optimizer_maxiter"] == 200
    assert manifest["horizon"] == 50
    assert manifest["reference_definition"] == (
        "exact_ground_state_energy_at_identical_n_ph_max"
    )
    assert manifest["error_metric"] == "same_cutoff_absolute_energy_error"
    assert [row["regime_id"] for row in manifest["regimes"]] == list(
        report.REGIME_ORDER
    )
    assert {row["n_ph_max"] for row in manifest["regimes"]} == {3, 7}
    provenance = manifest["package_provenance"]
    assert provenance["core_materialization_id"] == (
        "ra_adapt_stationary_late_core_v10"
    )
    assert package_manifest["package_id"] == report.PACKAGE_ID
    assert provenance["execution_count"] == 48


def test_pending_preview_is_two_pages_and_never_uses_canonical_name(
    tmp_path: Path,
) -> None:
    pypdf = pytest.importorskip("pypdf")
    pdf, pending = report.build_pending_preview(output_dir=tmp_path)
    payload = json.loads(pending.read_text(encoding="utf-8"))
    assert pdf.name == f"{report.STEM}_pending_preview.pdf"
    assert not (tmp_path / f"{report.STEM}.pdf").exists()
    assert len(pypdf.PdfReader(str(pdf)).pages) == 2
    assert payload["not_paper_evidence"] is True
    assert payload["canonical_results_pdf_emitted"] is False
    assert len(payload["missing_execution_ids"]) == 48
    assert payload["layout"]["terminal_rows_per_page"] == 24
    manifest = payload["parameter_manifest"]
    assert manifest["package_provenance"]["core_materialization_id"] == (
        "ra_adapt_stationary_late_core_v10"
    )
    assert len(manifest["regimes"]) == 6
    assert (tmp_path / f"{report.STEM}_pending_macro_master.png").is_file()
    assert (tmp_path / f"{report.STEM}_pending_singleton_master.png").is_file()
    macro_plot_pdf = (
        tmp_path / f"{report.STEM}_pending_macro_plots.pdf"
    )
    singleton_plot_pdf = (
        tmp_path / f"{report.STEM}_pending_singleton_plots.pdf"
    )
    assert macro_plot_pdf.is_file()
    assert singleton_plot_pdf.is_file()
    tex = (
        tmp_path / f"{report.STEM}_pending_preview.tex"
    ).read_text(encoding="utf-8")
    assert tex.count(r"\begin{tabular*}") == 2
    assert report._tex_escape(macro_plot_pdf.name) in tex
    assert report._tex_escape(singleton_plot_pdf.name) in tex
    assert "_master.png" not in tex
    reader = pypdf.PdfReader(str(pdf))
    page_text = [page.extract_text() for page in reader.pages]
    assert "Parameter and provenance manifest" in page_text[0]
    assert "Macro-generator stationary-source core" in page_text[0]
    assert "Single-Pauli-word stationary-source core" in page_text[1]
    assert all("Round-50" in text for text in page_text)


def test_partial_progress_is_two_pages_with_available_and_pending_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pypdf = pytest.importorskip("pypdf")
    cell = next(
        dict(row)
        for row in report._pending_cells()
        if row["execution_id"] == "core__weak_weak__nph3__append_macro"
    )
    cell["points"] = [
        {"k": index, "error": 1.0 / (index + 1)}
        for index in range(51)
    ]
    cell["marker"] = {
        "k": 50,
        "error": 1.0 / 51,
        "policy": "terminal_observed_point",
    }
    cell["terminal"] = {
        "k": 50,
        "error": 1.0 / 51,
        "N2q": 70,
        "D2q": 60,
        "Dc": 180,
        "W1q": 140,
        "B1q": 100,
        "qiskit_basis_work_status": "ok",
        "qiskit_basis_work_schema": (
            "qiskit_pretranspile_pauli_basis_work_v1"
        ),
        "S_alg": 1234,
        "status": "complete",
    }
    monkeypatch.setattr(
        report,
        "load_partial_cells",
        lambda **_: (
            [cell],
            {
                "validation": {
                    "path": "fixture-validation.json",
                    "sha256": _sha("a"),
                    "file_sha256": _sha("b"),
                },
                "inclusion_policy": (
                    "all_execution_ids_with_exactly_one_passed_"
                    "validated_attempt_v1"
                ),
                "automatic_attempt_selection_performed": False,
                "included_sources": [
                    {
                        "execution_id": cell["execution_id"],
                        "exact_same_cutoff_energy": 0.0,
                    }
                ],
            },
        ),
    )
    pdf, provenance = report.build_partial_progress(
        validation_path=tmp_path / "fixture-validation.json",
        fetched_dir=tmp_path / "fetched",
        output_dir=tmp_path,
    )
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    assert pdf.name == f"{report.STEM}_partial_progress.pdf"
    assert not (tmp_path / f"{report.STEM}.pdf").exists()
    assert len(pypdf.PdfReader(str(pdf)).pages) == 2
    assert payload["partial_progress"] is True
    assert payload["not_paper_evidence"] is True
    assert payload["paper_evidence_adopted"] is False
    assert payload["canonical_results_pdf_emitted"] is False
    assert payload["final_selection_consumed"] is False
    assert payload["included_count"] == 1
    assert payload["pending_count"] == 47
    assert payload["included_execution_ids"] == [cell["execution_id"]]
    assert len(payload["missing_execution_ids"]) == 47
    assert payload["layout"]["page_count"] == 2
    assert (
        tmp_path / f"{report.STEM}_partial_macro_master.png"
    ).is_file()
    assert (
        tmp_path / f"{report.STEM}_partial_singleton_master.png"
    ).is_file()
    tex = (
        tmp_path / f"{report.STEM}_partial_progress.tex"
    ).read_text(encoding="utf-8")
    assert tex.count(r"\begin{tabular*}") == 2
    assert "PARTIAL PROGRESS" in tex
    assert "1/48 validated" in tex
    assert report.PAPER_I_QISKIT_COST_TUPLE_LATEX in tex
    assert r"(70,60,180,140,1.2\mathrm{e}3)" in tex
    assert payload["terminal_cost_policy"]["tuple_fields"] == [
        "N2q",
        "D2q",
        "Dc",
        "W1q",
        "S_alg",
    ]
    assert payload["terminal_cost_policy"]["controller_round"] == 50
    assert payload["terminal_cost_policy"]["fifth_coordinate"][
        "display_notation"
    ] == "X.YeZ_two_significant_digits"
    reader = pypdf.PdfReader(str(pdf))
    page_text = [page.extract_text() for page in reader.pages]
    assert all("PARTIAL PROGRESS" in text for text in page_text)
    assert "complete" in page_text[0]
    assert "pending" in page_text[0]
    assert "pending" in page_text[1]


def test_cross_revision_progress_is_two_pages_and_names_exact_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pypdf = pytest.importorskip("pypdf")
    report._configure_package_dir(report.DEFAULT_PACKAGE_DIR)
    execution_ids = (
        "core__weak_weak__nph3__append_macro",
        "core__weak_weak__nph3__ra_macro_plateau",
    )
    cells = []
    for execution_id in execution_ids:
        cell = next(
            dict(row)
            for row in report._pending_cells()
            if row["execution_id"] == execution_id
        )
        cell["points"] = [
            {"k": index, "error": 1.0 / (index + 1)}
            for index in range(51)
        ]
        cell["marker"] = {
            "k": 50,
            "error": 1.0 / 51,
            "policy": "terminal_observed_point",
        }
        cell["terminal"] = {
            "k": 50,
            "error": 1.0 / 51,
            "N2q": 70,
            "D2q": 60,
            "Dc": 180,
            "W1q": 140,
            "B1q": 100,
            "qiskit_basis_work_status": "ok",
            "qiskit_basis_work_schema": (
                "qiskit_pretranspile_pauli_basis_work_v1"
            ),
            "S_alg": 1234,
            "status": "complete",
        }
        cells.append(cell)

    manifest = report._parameter_manifest()
    manifest["package_provenance"] = {
        "cross_revision": True,
        "source_package_count": 2,
        "source_receipt_count": 8,
        "package_ids": ["v6-package", "v7-package"],
        "sources": [
            {
                "source_receipt_index": 1,
                "method_family": "append",
                "package_id": "v6-package",
                "core_materialization_id": "core-v10",
                "package_manifest_sha256": _sha("a"),
                "source_archive_sha256": _sha("b"),
                "validation_sha256": _sha("c"),
            },
            {
                "source_receipt_index": 2,
                "method_family": "ra",
                "package_id": "v7-package",
                "core_materialization_id": "core-v11",
                "package_manifest_sha256": _sha("d"),
                "source_archive_sha256": _sha("e"),
                "validation_sha256": _sha("f"),
            },
            *[
                {
                    "source_receipt_index": source_index,
                    "method_family": "ra",
                    "package_id": "v7-package",
                    "core_materialization_id": "core-v11",
                    "package_manifest_sha256": _sha("d"),
                    "source_archive_sha256": _sha("e"),
                    "validation_sha256": _sha(
                        f"validation-{source_index}"
                    ),
                }
                for source_index in range(3, 9)
            ],
        ],
    }
    monkeypatch.setattr(
        report,
        "load_cross_revision_partial_cells",
        lambda **_: (
            cells,
            {
                "source_policy": "fixture",
                "automatic_attempt_selection_performed": False,
                "source_records": [],
                "included_sources": [],
            },
            manifest,
        ),
    )
    pdf, provenance = report.build_cross_revision_partial_progress(
        source_specs=[{"fixture": True}],
        output_dir=tmp_path,
    )
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    assert pdf.name == f"{report.CROSS_REVISION_STEM}.pdf"
    assert len(pypdf.PdfReader(str(pdf)).pages) == 2
    assert payload["cross_revision"] is True
    assert payload["not_paper_evidence"] is True
    assert payload["included_count"] == 2
    assert payload["pending_count"] == 46
    assert payload["package_ids"] == ["v6-package", "v7-package"]
    tex = (tmp_path / f"{report.CROSS_REVISION_STEM}.tex").read_text(
        encoding="utf-8"
    )
    assert "PARTIAL CROSS-REVISION PROGRESS" in tex
    assert "source[1] append" in tex
    assert "source[2] ra" in tex
    assert "source[8] ra" in tex
    assert "package/core/archive as source[2]" in tex
    page_text = [
        page.extract_text() for page in pypdf.PdfReader(str(pdf)).pages
    ]
    assert all(
        "PARTIAL CROSS-REVISION PROGRESS" in text for text in page_text
    )
    assert "v6-package" in page_text[0]
    assert "v7-package" in page_text[0]


def test_cross_revision_progress_appends_global_singleton_weak_weak_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pypdf = pytest.importorskip("pypdf")
    report._configure_package_dir(report.DEFAULT_PACKAGE_DIR)
    cell = next(
        dict(row)
        for row in report._pending_cells()
        if row["execution_id"] == "core__weak_weak__nph3__append_macro"
    )
    cell["points"] = [
        {"k": index, "error": 1.0 / (index + 2)}
        for index in range(51)
    ]
    cell["marker"] = {
        "k": 50,
        "error": cell["points"][-1]["error"],
        "policy": "terminal_observed_point",
    }
    cell["exact_same_cutoff_energy"] = -1.0
    cell["terminal"] = {
        "k": 50,
        "energy": -1.0 + cell["points"][-1]["error"],
        "error": cell["points"][-1]["error"],
        "N2q": 70,
        "D2q": 60,
        "Dc": 180,
        "W1q": 140,
        "B1q": 100,
        "qiskit_basis_work_status": "ok",
        "qiskit_basis_work_schema": (
            "qiskit_pretranspile_pauli_basis_work_v1"
        ),
        "S_alg": 1234,
        "status": "complete",
    }
    merged_cells = report._merge_partial_with_pending([cell])
    manifest = report._parameter_manifest()
    manifest["package_provenance"] = {
        "cross_revision": True,
        "source_package_count": 1,
        "source_receipt_count": 1,
        "package_ids": ["fixture-package"],
        "sources": [
            {
                "source_receipt_index": 1,
                "method_family": "append",
                "package_id": "fixture-package",
                "core_materialization_id": "fixture-core",
                "package_manifest_sha256": _sha("a"),
                "source_archive_sha256": _sha("b"),
                "validation_sha256": _sha("c"),
            }
        ],
    }
    monkeypatch.setattr(
        report,
        "load_cross_revision_partial_cells",
        lambda **_: (
            [cell],
            {
                "source_policy": "fixture",
                "automatic_attempt_selection_performed": False,
                "source_records": [],
                "included_sources": [],
            },
            manifest,
        ),
    )
    monkeypatch.setattr(
        report,
        "_merge_partial_with_pending",
        lambda _: copy.deepcopy(merged_cells),
    )
    adapter_path, adapter = _write_global_singleton_weak_weak_fixture(
        tmp_path
    )
    monkeypatch.setattr(report, "REPO_ROOT", tmp_path)

    pdf, provenance = report.build_cross_revision_partial_progress(
        source_specs=[{"fixture": True}],
        output_dir=tmp_path,
        global_singleton_weak_weak_adapter_path=adapter_path,
    )

    payload = json.loads(provenance.read_text(encoding="utf-8"))
    reader = pypdf.PdfReader(str(pdf))
    assert len(reader.pages) == 3
    assert payload["layout"]["page_count"] == 3
    assert payload["layout"]["page_3"] == (
        "weak_weak_global_singleton_append_vs_plateau_diagnostic_v1"
    )
    diagnostic = payload["global_singleton_weak_weak_comparison"]
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["paper_evidence_adopted"] is False
    assert diagnostic["adapter_source"]["sha256"] == adapter["sha256"]
    assert diagnostic["derived"][
        "terminal_s_alg_ratio_plateau_over_append"
    ] == pytest.approx(903_285 / 179_375)
    assert diagnostic["arms_by_policy"][
        "plateau_commutation"
    ]["insertion_counts"]["interior_count"] == 18
    assert any(
        "not the conventional Append-ADAPT comparator" in limitation
        for limitation in payload["limitations"]
    )
    plot_pdf = payload["outputs"][
        "global_singleton_weak_weak_plot_pdf"
    ]
    plot_png = payload["outputs"][
        "global_singleton_weak_weak_plot_png"
    ]
    assert Path(plot_pdf["path"]).is_file()
    assert Path(plot_png["path"]).is_file()
    assert len(plot_pdf["sha256"]) == 64
    assert len(plot_png["sha256"]) == 64

    tex = (tmp_path / f"{report.CROSS_REVISION_STEM}.tex").read_text(
        encoding="utf-8"
    )
    assert (
        "Separate weak--weak global-singleton insertion diagnostic" in tex
    )
    assert "insertion policy is the sole changed axis" in tex
    assert "not conventional Append-ADAPT" in tex
    assert "5.036$\\times$" in tex
    assert "18 interior insertions, first at" in tex
    page_text = reader.pages[2].extract_text()
    assert "global-singleton insertion diagnostic" in page_text
    assert "insertion policy is the sole changed axis" in page_text
    assert "not conventional Append-ADAPT" in page_text
    assert "18 interior insertions" in page_text
    assert "not adopted Paper-I evidence" in page_text


def test_qiskit_plateau_checkpoint_projection_streams_fifty_rounds(
    tmp_path: Path,
) -> None:
    pytest.importorskip("ijson")
    exact_energy = 0.5
    history = []
    before = 9.0
    for depth in range(1, 51):
        after = exact_energy + 1.0 / (depth + 10)
        history.append(
            {
                "depth": depth,
                "energy_before_opt": before,
                "energy_after_opt": after,
                "max_grad": 1.0 / depth,
                "selected_op": f"generator-{depth}",
                "selected_position": depth - 1 if depth < 13 else 2,
            }
        )
        before = after
    checkpoint = {
        "adapt_vqe": {
            "S_alg": 100,
            "S_unique": 90,
            "S_alg_components": {
                "N_H_outer": 10,
                "N_H_refit": 20,
                "N_grad": 30,
                "N_metric": 40,
            },
            "ansatz_depth": 50,
            "final_full_refit": {"executed": False},
            "history_checkpoint_complete": True,
            "history_count": 50,
            "history_tail_count": 50,
            "logical_num_parameters": 50,
            "nfev_total": 99,
            "num_parameters": 265,
            "partial_checkpoint": True,
            "stop_reason": None,
            "success": False,
            "terminal_active_prefix_checkpoint": {
                "active_ansatz_depth": 50
            },
            "history": history,
        }
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(checkpoint),
        encoding="utf-8",
    )
    projection = report._qiskit_plateau_checkpoint_projection(
        checkpoint_path,
        exact_energy=exact_energy,
    )
    assert len(projection["points"]) == 51
    assert projection["points"][0]["k"] == 0
    assert projection["points"][-1]["k"] == 50
    assert projection["terminal"]["energy"] == history[-1]["energy_after_opt"]
    assert projection["terminal"]["S_alg"] == 100
    assert projection["accounting"]["components"] == {
        "N_H_outer": 10,
        "N_H_refit": 20,
        "N_grad": 30,
        "N_metric": 40,
    }
    assert projection["insertion"] == {
        "first_interior_round": 13,
        "interior_count": 38,
        "append_position_count": 12,
    }


def test_cross_revision_progress_can_append_qiskit_plateau_diagnostic_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pypdf = pytest.importorskip("pypdf")
    report._configure_package_dir(report.DEFAULT_PACKAGE_DIR)
    append_cell = next(
        dict(row)
        for row in report._pending_cells()
        if row["execution_id"]
        == "core__strong_weak_u8__nph3__append_macro"
    )
    append_cell["points"] = [
        {"k": index, "error": 1.0 / (index + 2)}
        for index in range(51)
    ]
    append_cell["marker"] = {
        "k": 50,
        "error": append_cell["points"][-1]["error"],
        "policy": "terminal_observed_point",
    }
    append_cell["exact_same_cutoff_energy"] = 0.5
    append_cell["terminal"] = {
        "k": 50,
        "energy": 0.5 + append_cell["points"][-1]["error"],
        "error": append_cell["points"][-1]["error"],
        "N2q": 70,
        "D2q": 60,
        "Dc": 180,
        "W1q": 140,
        "B1q": 100,
        "qiskit_basis_work_status": "ok",
        "qiskit_basis_work_schema": (
            "qiskit_pretranspile_pauli_basis_work_v1"
        ),
        "S_alg": 100,
        "status": "complete",
    }
    append_cell["fixed_iteration_qiskit"] = {
        "k": 10,
        "error": append_cell["points"][10]["error"],
        "N2q": 50,
        "D2q": 40,
        "Dc": 120,
        "W1q": 90,
        "B1q": 70,
        "S_alg": 80,
        "status": "complete",
    }
    proxy_cells = []
    for execution_id, method, depth in (
        (
            "core__strong_weak_u8__nph3__ra_macro_plateau",
            "plateau",
            50,
        ),
        (
            "core__strong_weak_u8__nph3__ra_macro_append_only",
            "no_insertion",
            55,
        ),
    ):
        cell = next(
            dict(row)
            for row in report._pending_cells()
            if row["execution_id"] == execution_id
        )
        cell["method"] = method
        cell["points"] = copy.deepcopy(append_cell["points"])
        cell["marker"] = copy.deepcopy(append_cell["marker"])
        cell["exact_same_cutoff_energy"] = 0.5
        cell["terminal"] = copy.deepcopy(append_cell["terminal"])
        cell["fixed_iteration_qiskit"] = {
            **copy.deepcopy(append_cell["fixed_iteration_qiskit"]),
            "D2q": depth,
            "Dc": depth * 3,
        }
        proxy_cells.append(cell)
    manifest = report._parameter_manifest()
    manifest["package_provenance"] = {
        "cross_revision": True,
        "source_package_count": 1,
        "source_receipt_count": 1,
        "package_ids": ["fixture-package"],
        "sources": [
            {
                "source_receipt_index": 1,
                "method_family": "append",
                "package_id": "fixture-package",
                "core_materialization_id": "fixture-core",
                "package_manifest_sha256": _sha("a"),
                "source_archive_sha256": _sha("b"),
                "validation_sha256": _sha("c"),
            }
        ],
    }
    source_row = {
        "execution_id": append_cell["execution_id"],
        "method_family": "append",
        "attempt_sha256": _sha("d"),
    }
    monkeypatch.setattr(
        report,
        "load_cross_revision_partial_cells",
        lambda **_: (
            [append_cell, *proxy_cells],
            {
                "source_policy": "fixture",
                "automatic_attempt_selection_performed": False,
                "source_records": [],
                "included_sources": [source_row],
            },
            manifest,
        ),
    )
    diagnostic_points = [
        {
            "k": index,
            "energy": 0.5 + 1.01 / (index + 2),
            "error": 1.01 / (index + 2),
        }
        for index in range(51)
    ]
    diagnostic = {
        "schema": "paper_i_ra_qiskit_plateau_vs_append_diagnostic_v1",
        "status": "50_scientific_rounds_post_run_summary_failed",
        "not_paper_evidence": True,
        "execution_id": report.QISKIT_PLATEAU_MACRO_EXECUTION_ID,
        "algorithm_id": report.QISKIT_PLATEAU_MACRO_ALGORITHM_ID,
        "regime_id": "strong_weak_u8",
        "candidate_representation": "macro_generator_v1",
        "same_cutoff_exact_energy": 0.5,
        "points": diagnostic_points,
        "marker": {
            "k": 10,
            "error": diagnostic_points[10]["error"],
            "policy": (
                "earliest_prefix_within_10_percent_of_best_available_error"
            ),
        },
        "terminal": {
            "k": 50,
            "energy": diagnostic_points[-1]["energy"],
            "error": diagnostic_points[-1]["error"],
            "S_alg": 2000,
        },
        "accounting": {
            "S_alg": 2000,
            "S_unique": 1900,
            "components": {
                "N_H_outer": 50,
                "N_H_refit": 150,
                "N_grad": 300,
                "N_metric": 1500,
            },
            "nfev_total": 151,
        },
        "parameterization": {
            "logical_num_parameters": 50,
            "runtime_num_parameters": 265,
        },
        "insertion": {
            "first_interior_round": 13,
            "interior_count": 37,
            "append_position_count": 13,
        },
        "checkpoint_state": {
            "history_checkpoint_complete": True,
            "partial_checkpoint": True,
            "success": False,
            "stop_reason": None,
            "final_full_refit_executed": False,
        },
        "comparison": {
            "append_execution_id": append_cell["execution_id"],
            "append_terminal": append_cell["terminal"],
            "terminal_error_difference_qiskit_ra_minus_append": (
                diagnostic_points[-1]["error"]
                - append_cell["terminal"]["error"]
            ),
            "terminal_error_ratio_qiskit_ra_over_append": (
                diagnostic_points[-1]["error"]
                / append_cell["terminal"]["error"]
            ),
            "s_alg_ratio_qiskit_ra_over_append": 20.0,
        },
        "online_qiskit_selector": {
            "backend": "FakeMarrakesh",
            "optimization_level": 1,
            "transpile_seed": 7,
            "cost_mode": "transpile_single_v1",
            "application_scope": "phase1_phase2_phase3_and_fallback_v1",
        },
        "terminal_qiskit_tuple": {
            "status": "unavailable",
            "reason": "canonical_post_run_summary_generation_failed",
            "selector_time_candidate_deltas_substituted": False,
        },
        "fixed_iteration_qiskit": {
            "k": 10,
            "error": diagnostic_points[10]["error"],
            "N2q": 50,
            "D2q": 30,
            "Dc": 100,
            "W1q": 90,
            "B1q": 70,
            "S_alg": 120,
            "status": "complete",
        },
        "source_bindings": {},
    }
    always_points = [
        {
            "k": index,
            "energy": 0.5 + 1.02 / (index + 2),
            "error": 1.02 / (index + 2),
        }
        for index in range(14)
    ]
    always_diagnostic = {
        "schema": "paper_i_ra_qiskit_always13_diagnostic_v1",
        "status": "13_scientific_rounds_complete",
        "not_paper_evidence": True,
        "execution_id": report.QISKIT_ALWAYS_MACRO_EXECUTION_ID,
        "algorithm_id": report.QISKIT_ALWAYS_MACRO_ALGORITHM_ID,
        "regime_id": "strong_weak_u8",
        "candidate_representation": "macro_generator_v1",
        "same_cutoff_exact_energy": 0.5,
        "points": always_points,
        "marker": {
            "k": 9,
            "error": always_points[9]["error"],
            "policy": "paper_i_effective_plateau_v1",
        },
        "terminal": {
            "k": 13,
            "energy": always_points[-1]["energy"],
            "error": always_points[-1]["error"],
            "S_alg": 2500,
        },
        "fixed_iteration_qiskit": {
            "k": 10,
            "error": always_points[10]["error"],
            "N2q": 48,
            "D2q": 35,
            "Dc": 110,
            "W1q": 86,
            "B1q": 68,
            "S_alg": 200,
            "status": "complete",
        },
        "insertion": {"positions": list(range(13)), "interior_count": 0},
        "online_qiskit_selector": diagnostic["online_qiskit_selector"],
        "source_bindings": {},
    }
    monkeypatch.setattr(
        report,
        "_load_qiskit_plateau_macro_diagnostic",
        lambda **_: diagnostic,
    )
    monkeypatch.setattr(
        report,
        "_load_qiskit_always_macro_diagnostic",
        lambda **_: always_diagnostic,
    )
    pdf, provenance = report.build_cross_revision_partial_progress(
        source_specs=[{"fixture": True}],
        output_dir=tmp_path,
        diagnostic_qiskit_plateau_run_dir=tmp_path / "run",
        diagnostic_qiskit_plateau_log=tmp_path / "run.log",
        diagnostic_qiskit_always_run_dir=tmp_path / "always-run",
    )
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    reader = pypdf.PdfReader(str(pdf))
    assert len(reader.pages) == 3
    assert payload["layout"]["page_count"] == 3
    assert payload["layout"]["page_3"] == (
        "strong_weak_macro_qiskit_ranked_insertion_vs_proxy_diagnostic_v2"
    )
    assert payload["included_count"] == 3
    assert payload["diagnostic_comparison"]["execution_id"] == (
        report.QISKIT_PLATEAU_MACRO_EXECUTION_ID
    )
    assert (
        payload["diagnostic_comparison"]["matched_append_curve"]["source"]
        == source_row
    )
    page_text = reader.pages[2].extract_text()
    assert "Plateau RA completed 50/50 rounds" in page_text
    assert "All-phase Qiskit: plateau RA" in page_text
    assert "All-phase Qiskit: always RA" in page_text
    assert "Page-1 proxy: plateau RA" in page_text
    assert "outside the validated 48-cell evidence matrix" in page_text


def test_final_mode_requires_explicit_selection_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["report"])
    assert report.main() == 2


def test_final_loader_preserves_exact_48_selection_gate(
    tmp_path: Path,
) -> None:
    contract = report._package_contract()
    validation = contract.digested(
        {
            "schema": report.VALIDATION_SCHEMA,
            "package_id": report.PACKAGE_ID,
            "attempt_count": 0,
            "attempts": [],
            "execution_ids_with_passed_attempts": [],
            "automatic_attempt_selection_performed": False,
            "paper_evidence_adopted": False,
            "status": "validated_no_selection",
        }
    )
    validation_path = tmp_path / "validation.json"
    validation_path.write_bytes(
        contract.canonical_json_bytes(validation) + b"\n"
    )
    selection = contract.digested(
        {
            "schema": report.SELECTION_SCHEMA,
            "package_id": report.PACKAGE_ID,
            "selected_count": 1,
            "selected_attempts": [],
            "fetched_validation_sha256": validation["sha256"],
            "automatic_attempt_selection_performed": False,
            "paper_evidence_adopted": False,
        }
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_bytes(
        contract.canonical_json_bytes(selection) + b"\n"
    )
    with pytest.raises(
        report.ReportInputError,
        match="selection/validation authority drifted",
    ):
        report.load_selected_cells(
            selection_path=selection_path,
            validation_path=validation_path,
            fetched_dir=tmp_path / "fetched",
        )


def test_local_paused_always_prefix_loads_authenticated_partial_history(
    tmp_path: Path,
) -> None:
    target_id = "core__weak_strong__nph7__ra_macro_always"
    source_id = target_id + "__gradient_stationary__phase1_cost_off"
    job = _generic_digested(
        {
            "package_id": report.LOCAL_PAUSED_ALWAYS_PACKAGE_ID,
            "base_cell_id": target_id,
            "execution_id": source_id,
            "cell_id": source_id,
            "horizon": 50,
            "regime_id": "weak_strong",
            "candidate_representation": "macro_generator_v1",
            "route_id": "ra_macro_always",
            "active_gradient_policy": "stationary_source_response_v1",
            "resource_weighting_scope": "late_resource_weighting_v1",
            "phase1_cost_term": "disabled_for_phase1_only",
        }
    )
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    checkpoint = {
        "adapt_vqe": {
            "history_tail": [
                {
                    "depth": 1,
                    "energy_before_opt": 0.0,
                    "energy_after_opt": -0.5,
                    "selected_position": 0,
                    "active_prefix_checkpoint": {
                        "outer_iteration": 1,
                        "active_ansatz_depth": 1,
                        "checkpoint_sha256": _sha("a"),
                        "estimator_ledger_receipt": {
                            "status": "complete",
                            "outer_iteration": 1,
                            "cumulative_executed_queries": {"S_alg": 10},
                        },
                    },
                },
                {
                    "depth": 2,
                    "energy_before_opt": -0.5,
                    "energy_after_opt": -0.75,
                    "selected_position": 0,
                    "active_prefix_checkpoint": {
                        "outer_iteration": 2,
                        "active_ansatz_depth": 2,
                        "checkpoint_sha256": _sha("b"),
                        "estimator_ledger_receipt": {
                            "status": "complete",
                            "outer_iteration": 2,
                            "cumulative_executed_queries": {"S_alg": 20},
                        },
                    },
                },
            ]
        }
    }
    checkpoint_path = tmp_path / "current.json"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            (
                'AI_LOG {"event":"hardcoded_adapt_iter","depth":1,'
                '"selected_position":0,"energy":0.0}',
                'AI_LOG {"event":"hardcoded_adapt_iter","depth":2,'
                '"selected_position":0,"energy":-0.5}',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    expected_jobs = {
        target_id: {
            "regime_id": "weak_strong",
            "candidate_representation": "macro_generator_v1",
            "route_id": "ra_macro_always",
        }
    }

    cell, source = report._load_local_paused_always_prefix(
        job_path=job_path,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        expected_jobs=expected_jobs,
        exact_same_cutoff_energy=-1.0,
    )

    assert [row["k"] for row in cell["points"]] == [0, 1, 2]
    assert cell["terminal"] == {
        "k": 2,
        "error": 0.25,
        "S_alg": 20,
        "status": "paused-local",
    }
    assert cell["marker"]["k"] == 2
    assert source["paused_controller_round"] == 2
    assert source["paper_evidence_eligible"] is False


def test_local_paused_always_prefix_rejects_log_position_tamper(
    tmp_path: Path,
) -> None:
    target_id = "core__weak_strong__nph7__ra_macro_always"
    source_id = target_id + "__gradient_stationary__phase1_cost_off"
    job_path = tmp_path / "job.json"
    job_path.write_text(
        json.dumps(
            _generic_digested(
                {
                    "package_id": report.LOCAL_PAUSED_ALWAYS_PACKAGE_ID,
                    "base_cell_id": target_id,
                    "execution_id": source_id,
                    "cell_id": source_id,
                    "horizon": 50,
                    "regime_id": "weak_strong",
                    "candidate_representation": "macro_generator_v1",
                    "route_id": "ra_macro_always",
                    "active_gradient_policy": (
                        "stationary_source_response_v1"
                    ),
                    "resource_weighting_scope": "late_resource_weighting_v1",
                    "phase1_cost_term": "disabled_for_phase1_only",
                }
            )
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "current.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "adapt_vqe": {
                    "history_tail": [
                        {
                            "depth": 1,
                            "energy_before_opt": 0.0,
                            "energy_after_opt": -0.5,
                            "selected_position": 0,
                            "active_prefix_checkpoint": {
                                "outer_iteration": 1,
                                "active_ansatz_depth": 1,
                                "checkpoint_sha256": _sha("c"),
                                "estimator_ledger_receipt": {
                                    "status": "complete",
                                    "outer_iteration": 1,
                                    "cumulative_executed_queries": {
                                        "S_alg": 10
                                    },
                                },
                            },
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "run.log"
    log_path.write_text(
        'AI_LOG {"event":"hardcoded_adapt_iter","depth":1,'
        '"selected_position":1,"energy":0.0}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        report.ReportInputError,
        match="log/checkpoint round identity drifted",
    ):
        report._load_local_paused_always_prefix(
            job_path=job_path,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            expected_jobs={
                target_id: {
                    "regime_id": "weak_strong",
                    "candidate_representation": "macro_generator_v1",
                    "route_id": "ra_macro_always",
                }
            },
            exact_same_cutoff_energy=-1.0,
        )
