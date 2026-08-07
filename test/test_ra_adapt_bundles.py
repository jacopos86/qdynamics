from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any

import pytest

from pipelines.static_adapt.ra_adapt import bundles as bundle_module
from pipelines.static_adapt.ra_adapt.bundles import (
    BLOCKED_PROTOCOL_SCHEMA,
    BUNDLE_SCHEMA,
    EXECUTION_TARGET,
    EXECUTION_TEMPLATE_SCHEMA,
    EXPECTED_ARTIFACT_ROLES,
    CLAIM_FACING_REGIME_CUTOFF_PAIRS,
    CORE_BUNDLE_ID,
    CORE_CAMPAIGN_ID,
    CORE_RUN_CLASS,
    CORE_VISIBLE_TARGET_ID,
    FACTORIAL_BUNDLE_POLICIES,
    FACTORIAL_CAMPAIGN_ID,
    FACTORIAL_RUN_CLASS,
    FACTORIAL_VISIBLE_TARGET_ID,
    FULL_HORIZON,
    FULL_VISIBLE_REGIME_CUTOFF_PAIRS,
    GLOBAL_SINGLETON_BUNDLE_ID,
    GLOBAL_SINGLETON_CAMPAIGN_ID,
    GLOBAL_SINGLETON_INSERTION_ROUTE_IDS,
    GLOBAL_SINGLETON_RUN_CLASS,
    GLOBAL_SINGLETON_VISIBLE_TARGET_ID,
    MACRO_ROUTE_IDS,
    MEASURED_BUNDLE_ID,
    PHASE3_QISKIT_BUNDLE_ID,
    PHASE3_QISKIT_CAMPAIGN_ID,
    PHASE3_QISKIT_EXECUTION_TARGET,
    PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256,
    PHASE3_QISKIT_PAGE7_PARENT_ROUTE_PROFILE,
    PHASE3_QISKIT_ROUTE_IDS,
    PHASE3_QISKIT_RUN_CLASS,
    PHASE3_QISKIT_STRONG_HOLSTEIN_HORIZON,
    PHASE3_QISKIT_VISIBLE_TARGET_ID,
    PHASE3_QISKIT_WEAK_HOLSTEIN_HORIZON,
    QISKIT_COST_PILOT_BUNDLE_ID,
    QISKIT_COST_PILOT_CAMPAIGN_ID,
    QISKIT_COST_PILOT_EXECUTION_TARGET,
    QISKIT_COST_PILOT_GLOBAL_SINGLETON_ALGORITHM_ID,
    QISKIT_COST_PILOT_MACRO_ALGORITHM_ID,
    QISKIT_COST_PILOT_ROUTE_IDS,
    QISKIT_COST_PILOT_RUN_CLASS,
    QISKIT_COST_PILOT_VISIBLE_TARGET_ID,
    QISKIT_COST_ALWAYS13_ALGORITHM_ID,
    QISKIT_COST_ALWAYS13_BUNDLE_ID,
    QISKIT_COST_ALWAYS13_CAMPAIGN_ID,
    QISKIT_COST_ALWAYS13_EXECUTION_TARGET,
    QISKIT_COST_ALWAYS13_HORIZON,
    QISKIT_COST_ALWAYS13_ROUTE_IDS,
    QISKIT_COST_ALWAYS13_RUN_CLASS,
    QISKIT_COST_ALWAYS13_SOURCE_PROTOCOL_SHA256,
    QISKIT_COST_ALWAYS13_VISIBLE_TARGET_ID,
    RUN_CLASS,
    SOURCE_LOCK_SCHEMA,
    STATIONARY_BUNDLE_ID,
    STUDY_ID,
    SUBMISSION_STATE,
    VALIDATION_REGIMES,
    VALIDATION_ROUTE_IDS,
    VISIBLE_TARGET_ID,
    BundleMaterializationError,
    ProtocolResolutionContext,
    ROUTE_APPEND_SINGLETON,
    ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED,
    ROUTE_RA_GLOBAL_SINGLETON_PLATEAU,
    ROUTE_RA_MACRO_ALWAYS,
    ROUTE_RA_SINGLETON_ALWAYS,
    ROUTE_RA_SINGLETON_APPEND_ONLY,
    ROUTE_RA_SINGLETON_PLATEAU,
    SINGLETON_CORE_ROUTE_IDS,
    STUDY1_EXECUTION_DEDUPE_SCHEMA,
    build_core_cell_specs,
    build_factorial_always_cell_specs,
    build_global_singleton_insertion_cell_specs,
    build_phase3_qiskit_mixed_horizon_cell_specs,
    build_qiskit_cost_plateau_pilot_cell_specs,
    build_qiskit_cost_always13_cell_specs,
    build_study1_cell_specs,
    load_validated_bundle_protocol,
    materialize_core_bundle,
    materialize_factorial_always_bundles,
    materialize_global_singleton_insertion_bundle,
    materialize_phase3_qiskit_mixed_horizon_bundle,
    materialize_qiskit_cost_plateau_pilot_bundle,
    materialize_qiskit_cost_always13_bundle,
    materialize_study1_bundles,
    study1_shared_execution_dedupe_contract,
    validate_full_matrix_progression,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    ACTIVE_GRADIENT_STATIONARY,
    APPEND_ADAPT_PROTOCOL_SCHEMA,
    APPEND_CONVENTIONAL_SELECTOR_ID,
    APPEND_CONVENTIONAL_SELECTOR_SCOPE,
    AppendAdaptRequest,
    CANDIDATE_REPRESENTATION_MACRO,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    CANDIDATE_INVENTORY_LINEAGE_SCHEMA,
    CandidateInventoryLineageReceipt,
    CandidateInventoryLineageRow,
    EXPANDED_RUNTIME_PROJECTED_LOGICAL_BASE_CHART,
    LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART,
    NATIVE_REFIT_CHART,
    PhaseIIIMultiplierContract,
    RA_ADAPT_PROTOCOL_SCHEMA,
    RA_STAGED_SELECTOR_ID,
    RAAdaptRequest,
    RESOURCE_WEIGHTING_ALL_PHASE,
    RESOURCE_WEIGHTING_LATE,
    canonical_json_bytes,
    canonical_sha256,
    append_adapt_request_from_mapping,
    load_resolved_ra_adapt_protocol,
    ra_adapt_request_from_mapping,
    resolved_ra_adapt_protocol_from_mapping,
)
from pipelines.static_adapt.ra_adapt.adapters import (
    GLOBAL_SINGLE_PAULI_ADAPTER_ID,
    PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON,
    PHASE_I_VISIBILITY_ALL_EXECUTABLE,
    PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY,
    GlobalSinglePauliWordCandidateAdapter,
    MacroCandidateAdapter,
    SinglePauliWordCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.engine import (
    RA_ADAPT_COMPILE_IDENTITY,
    RA_ADAPT_ESTIMATOR_ACCOUNTING,
    RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID,
    RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_SOURCE_ALGORITHM_ID,
    RA_ADAPT_PHASE3_QISKIT_COST_PHASE_REUSE,
    RA_ADAPT_PHASE3_QISKIT_COST_POLICY,
    RA_ADAPT_PHASE3_QISKIT_COST_ROUTE_SUFFIX,
    _repaired_route_contract,
)
from pipelines.static_adapt.hh_backend_compile_oracle import (
    BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1,
    MARRAKESH_GRAPH_SPAN_MODE,
)
from pipelines.static_adapt.ra_adapt.pools import (
    GUARDED_SINGLETON_POOL_SCHEMA,
    PARENT_TEMPLATE_INVENTORY_SCHEMA,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AlwaysCommutationReducedInsertion,
    AppendCommutationReducedInsertion,
    AppendOnlyInsertion,
    BeamOff,
    CheckpointObservation,
    EstimatorLedgerObservation,
    FreshStart,
    GreedyBatchAdmission,
    MetricPruning,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRObservationPolicy,
    SRStopPolicy,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_locked_archive_member_hashing_is_streamed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"source-locked-member" * 100_000

    class SizedReadOnly(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            assert size > 0
            return super().read(size)

    class Member:
        @staticmethod
        def isfile() -> bool:
            return True

    class Bundle:
        def __enter__(self) -> "Bundle":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def getmember(_name: str) -> Member:
            return Member()

        @staticmethod
        def extractfile(_member: Member) -> SizedReadOnly:
            return SizedReadOnly(payload)

    monkeypatch.setattr(bundle_module.tarfile, "is_tarfile", lambda _path: True)
    monkeypatch.setattr(bundle_module.tarfile, "open", lambda *_args, **_kwargs: Bundle())
    assert bundle_module._hash_archive_member(
        Path("locked.tar.gz"),
        "resolved/result.json",
    ) == _sha(payload)


def test_repeated_source_locks_hash_each_archive_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_locks = _source_locks(tmp_path)
    archive_path = Path(
        next(iter(source_locks["cell_locks"].values()))["archive"]["path"]
    ).resolve()
    original_hash_file = bundle_module._hash_file
    archive_hash_calls = 0

    def _counted_hash_file(path: Path) -> str:
        nonlocal archive_hash_calls
        if path.resolve() == archive_path:
            archive_hash_calls += 1
        return original_hash_file(path)

    monkeypatch.setattr(bundle_module, "_hash_file", _counted_hash_file)
    bundle_module.normalize_and_verify_source_locks(
        source_locks,
        cells=build_study1_cell_specs(validation_horizon=6),
        repo_root=REPO_ROOT,
        verify_files=True,
    )
    assert archive_hash_calls == 1


def _source_locks(
    tmp_path: Path,
    *,
    cells: tuple[bundle_module.BundleCellSpec, ...] | None = None,
) -> dict[str, Any]:
    member_path = "resolved/result.json"
    member_payload = b'{"settings":{"optimizer":"powell","maxiter":200}}\n'
    archive = tmp_path / "visible_sources.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo(member_path)
        info.size = len(member_payload)
        info.mtime = 0
        bundle.addfile(info, io.BytesIO(member_payload))
    member_sha = _sha(member_payload)
    archive_sha = _sha(archive.read_bytes())

    if cells is None:
        cells = build_study1_cell_specs(validation_horizon=6)
    locks: dict[str, Any] = {}
    for cell in cells:
        trace = {
            "source_map": (
                "MATH/paper_details/figures/"
                "paper_i_hh_macro_common_accuracy_20260723/"
                "paper_i_hh_macro_common_accuracy_20260723_provenance.json"
            ),
            "target_axis": "regimes",
            "regime_or_case": cell.regime_id,
            "method": cell.route_id,
            "source_json": member_path,
            "source_json_exists_locally": False,
            "source_sha256_expected": member_sha,
            "source_sha256_actual": None,
            "source_sha256_match": None,
            "settings_reused": {
                "settings": {
                    "optimizer": "powell",
                    "maxiter": 200,
                    "seed": 7,
                }
            },
            "settings_changed": [],
            "same_cutoff_ed_reference": {
                "path": bundle_module.GLOBAL_SOURCE_LOCKS[
                    "ed_cutoff_reference"
                ]["path"],
                "sha256": bundle_module.GLOBAL_SOURCE_LOCKS[
                    "ed_cutoff_reference"
                ]["sha256"],
                "nph": cell.nph,
                "required": True,
                "reference_role": (
                    "same_cutoff_reporting_reference"
                ),
            },
            "status": "ok",
            "problems": ["source JSON is missing locally"],
        }
        if (
            cell.candidate_representation
            == CANDIDATE_REPRESENTATION_SINGLE_PAULI
            and cell.stage in {"core", "factorial"}
        ):
            is_append = cell.route_id == ROUTE_APPEND_SINGLETON
            route_delta_id = (
                "core_conventional_append_baseline"
                if is_append
                else "core_insertion_policy_variant"
            )
            delta_ids = [
                "core_stationary_gradient_policy",
                "core_candidate_representation_axis",
                "core_fixed_horizon",
                route_delta_id,
            ]
            trace["settings_changed"] = [
                {"id": delta_id} for delta_id in delta_ids
            ]
            target_insertion = (
                "conventional_unwhitened_append_v1"
                if is_append
                else {
                    ROUTE_RA_SINGLETON_APPEND_ONLY: (
                        AppendOnlyInsertion.kind
                    ),
                    ROUTE_RA_SINGLETON_PLATEAU: (
                        bundle_module.PlateauCommutationInsertion.kind
                    ),
                    ROUTE_RA_SINGLETON_ALWAYS: (
                        AlwaysCommutationReducedInsertion.kind
                    ),
                }[cell.route_id]
            )
            trace["core_source_anchor"] = {
                "schema": (
                    "paper_i_ra_adapt_core_singleton_source_anchor_v1"
                ),
                "anchor_family": (
                    "canonical_append_registry_v1"
                    if is_append
                    else "chtc_9381198_singleton_plateau_v1"
                ),
                "regime_id": cell.regime_id,
                "nph": cell.nph,
                "scientific_result_anchor_claimed": False,
                "route_derivation": {
                    "source_route_id": (
                        ROUTE_APPEND_SINGLETON
                        if is_append
                        else ROUTE_RA_SINGLETON_PLATEAU
                    ),
                    "target_route_id": cell.route_id,
                    "target_insertion_policy": target_insertion,
                    "declared_delta_ids": delta_ids,
                },
            }
        if cell.stage == "factorial":
            resource_scope = (
                RESOURCE_WEIGHTING_ALL_PHASE
                if cell.cell_id.endswith("phase1_cost_on")
                else RESOURCE_WEIGHTING_LATE
            )
            gradient_policy = (
                ACTIVE_GRADIENT_STATIONARY
                if "__gradient_stationary__" in cell.cell_id
                else ACTIVE_GRADIENT_MEASURED
            )
            trace["settings_changed"].extend(
                [
                    {
                        "id": "D5",
                        "field": "resource_weighting_scope",
                        "to": resource_scope,
                    },
                    {
                        "id": "study1_axis",
                        "field": "active_gradient_policy",
                        "to": gradient_policy,
                    },
                    {
                        "id": "study1_insertion_policy_variant",
                        "field": "insertion_policy",
                        "to": AlwaysCommutationReducedInsertion.kind,
                    },
                ]
            )
        if cell.stage == "global_singleton_insertion":
            common_delta_ids = [
                "D5",
                "global_singleton_candidate_adapter",
                "global_singleton_phase_i_candidate_supply",
                "global_singleton_phase_i_candidate_visibility",
                "global_singleton_phase_ii_candidate_exposure",
                "global_singleton_route_identity",
            ]
            trace["settings_changed"] = [
                {
                    "id": "D5",
                    "field": "resource_weighting_scope",
                    "to": RESOURCE_WEIGHTING_ALL_PHASE,
                },
                {
                    "id": "global_singleton_candidate_adapter",
                    "field": "candidate_adapter_id",
                    "to": GLOBAL_SINGLE_PAULI_ADAPTER_ID,
                },
                {
                    "id": "global_singleton_phase_i_candidate_supply",
                    "field": "phase_i_candidate_supply",
                    "to": PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON,
                },
                {
                    "id": "global_singleton_phase_i_candidate_visibility",
                    "field": "phase_i_candidate_visibility",
                    "to": PHASE_I_VISIBILITY_ALL_EXECUTABLE,
                },
                {
                    "id": "global_singleton_phase_ii_candidate_exposure",
                    "field": "phase_ii_candidate_exposure",
                    "to": (
                        PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY
                    ),
                },
                {
                    "id": "global_singleton_route_identity",
                    "field": "route_id",
                    "from": ROUTE_RA_SINGLETON_PLATEAU,
                    "to": cell.route_id,
                },
            ]
            target_insertion = (
                AppendCommutationReducedInsertion.kind
                if cell.route_id
                == ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED
                else bundle_module.PlateauCommutationInsertion.kind
            )
            declared_delta_ids = list(common_delta_ids)
            if (
                cell.route_id
                == ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED
            ):
                trace["settings_changed"].append(
                    {
                        "id": (
                            "global_singleton_insertion_policy_variant"
                        ),
                        "field": "insertion_policy",
                        "from": (
                            bundle_module.PlateauCommutationInsertion.kind
                        ),
                        "to": target_insertion,
                    }
                )
                declared_delta_ids.append(
                    "global_singleton_insertion_policy_variant"
                )
            trace["global_singleton_source_anchor"] = {
                "schema": (
                    "paper_i_ra_adapt_global_singleton_source_anchor_v1"
                ),
                "anchor_family": (
                    "sealed_stationary_core_v13_singleton_plateau_v1"
                ),
                "regime_id": cell.regime_id,
                "nph": cell.nph,
                "scientific_result_anchor_claimed": False,
                "predecessor": {
                    "materialization_id": (
                        "ra_adapt_stationary_late_core_v13"
                    ),
                    "source_route_id": ROUTE_RA_SINGLETON_PLATEAU,
                    "source_insertion_policy": (
                        bundle_module.PlateauCommutationInsertion.kind
                    ),
                },
                "route_derivation": {
                    "target_route_id": cell.route_id,
                    "target_insertion_policy": target_insertion,
                    "declared_delta_ids": declared_delta_ids,
                },
            }
        if cell.stage == "qiskit_cost_plateau_pilot":
            is_macro = (
                cell.candidate_representation
                == CANDIDATE_REPRESENTATION_MACRO
            )
            declared_delta_ids = [
                "qiskit_selector_cost_oracle",
                "qiskit_cost_pilot_exact_cell_selection",
            ]
            trace["settings_changed"] = [
                {
                    "id": "qiskit_selector_cost_oracle",
                    "field": "selector_cost_policy",
                    "from": "marrakesh_graph_span_v1",
                    "to": bundle_module.RA_ADAPT_QISKIT_COST_POLICY,
                },
                {
                    "id": "qiskit_cost_pilot_exact_cell_selection",
                    "field": "campaign_cell_selection",
                    "from": "source_campaign_matrix_v1",
                    "to": cell.cell_id,
                },
            ]
            if is_macro:
                declared_delta_ids.append(
                    "qiskit_cost_all_phase_scope"
                )
                trace["settings_changed"].append(
                    {
                        "id": "qiskit_cost_all_phase_scope",
                        "field": "resource_weighting_scope",
                        "from": RESOURCE_WEIGHTING_LATE,
                        "to": RESOURCE_WEIGHTING_ALL_PHASE,
                    }
                )
            trace["qiskit_cost_pilot_source_anchor"] = {
                "schema": (
                    "paper_i_ra_adapt_qiskit_cost_plateau_pilot_"
                    "source_anchor_v1"
                ),
                "source_campaign_id": (
                    CORE_CAMPAIGN_ID
                    if is_macro
                    else GLOBAL_SINGLETON_CAMPAIGN_ID
                ),
                "source_bundle_id": (
                    CORE_BUNDLE_ID
                    if is_macro
                    else GLOBAL_SINGLETON_BUNDLE_ID
                ),
                "source_route_id": cell.route_id,
                "source_algorithm_id": (
                    "paper_i_ra_adapt_macro_plateau_insertion_repair_v1"
                    if is_macro
                    else (
                        "paper_i_ra_adapt_global_singleton_plateau_"
                        "commutation_v1"
                    )
                ),
                "target_campaign_id": QISKIT_COST_PILOT_CAMPAIGN_ID,
                "target_bundle_id": QISKIT_COST_PILOT_BUNDLE_ID,
                "target_algorithm_id": cell.algorithm_id,
                "regime_id": cell.regime_id,
                "nph": cell.nph,
                "scientific_result_anchor_claimed": False,
                "declared_delta_ids": declared_delta_ids,
            }
        if cell.stage == "qiskit_cost_always13_diagnostic":
            declared_delta_ids = [
                "qiskit_cost_always13_insertion_policy",
                "qiskit_cost_always13_horizon",
                "qiskit_cost_always13_exact_cell_selection",
            ]
            trace["settings_changed"] = [
                {
                    "id": "qiskit_cost_always13_insertion_policy",
                    "field": "insertion_policy",
                    "from": (
                        bundle_module.PlateauCommutationInsertion.kind
                    ),
                    "to": AlwaysCommutationReducedInsertion.kind,
                },
                {
                    "id": "qiskit_cost_always13_horizon",
                    "field": "maximum_controller_rounds",
                    "from": FULL_HORIZON,
                    "to": QISKIT_COST_ALWAYS13_HORIZON,
                },
                {
                    "id": (
                        "qiskit_cost_always13_exact_cell_selection"
                    ),
                    "field": "campaign_cell_selection",
                    "from": (
                        "qiskit_cost_pilot__strong_weak_u8__nph3__"
                        f"{bundle_module.ROUTE_RA_MACRO_PLATEAU}"
                    ),
                    "to": cell.cell_id,
                },
            ]
            trace["qiskit_cost_always13_source_anchor"] = {
                "schema": (
                    "paper_i_ra_adapt_qiskit_cost_always13_"
                    "source_anchor_v1"
                ),
                "source_campaign_id": QISKIT_COST_PILOT_CAMPAIGN_ID,
                "source_bundle_id": QISKIT_COST_PILOT_BUNDLE_ID,
                "source_route_id": (
                    bundle_module.ROUTE_RA_MACRO_PLATEAU
                ),
                "source_algorithm_id": (
                    QISKIT_COST_PILOT_MACRO_ALGORITHM_ID
                ),
                "source_protocol_sha256": (
                    QISKIT_COST_ALWAYS13_SOURCE_PROTOCOL_SHA256
                ),
                "target_campaign_id": QISKIT_COST_ALWAYS13_CAMPAIGN_ID,
                "target_bundle_id": QISKIT_COST_ALWAYS13_BUNDLE_ID,
                "target_route_id": cell.route_id,
                "target_algorithm_id": cell.algorithm_id,
                "regime_id": cell.regime_id,
                "nph": cell.nph,
                "scientific_result_anchor_claimed": False,
                "changed_scientific_fields": [
                    "request.method.insertion",
                    (
                        "request.execution.stop."
                        "maximum_controller_rounds"
                    ),
                ],
                "declared_delta_ids": declared_delta_ids,
            }
        if cell.stage == "phase3_qiskit_candidate":
            declared_delta_ids = [
                "phase3_qiskit_selector_cost_scope",
                "phase3_qiskit_exact_cell_selection",
            ]
            trace["settings_changed"] = [
                {
                    "id": "phase3_qiskit_selector_cost_scope",
                    "field": "selector_compile_cost_scope",
                    "from": "marrakesh_graph_span_all_phases_v1",
                    "to": BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1,
                },
                {
                    "id": "phase3_qiskit_exact_cell_selection",
                    "field": "campaign_cell_selection",
                    "from": "page7_global_singleton_plateau_matrix_v1",
                    "to": cell.cell_id,
                },
            ]
            trace["phase3_qiskit_source_anchor"] = {
                "schema": (
                    "paper_i_ra_adapt_phase3_qiskit_source_anchor_v1"
                ),
                "source_algorithm_id": (
                    RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_SOURCE_ALGORITHM_ID
                ),
                "source_route_id": ROUTE_RA_GLOBAL_SINGLETON_PLATEAU,
                "source_route_profile": (
                    PHASE3_QISKIT_PAGE7_PARENT_ROUTE_PROFILE
                ),
                "source_route_contract_sha256": (
                    PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256
                ),
                "target_campaign_id": PHASE3_QISKIT_CAMPAIGN_ID,
                "target_bundle_id": PHASE3_QISKIT_BUNDLE_ID,
                "target_algorithm_id": cell.algorithm_id,
                "regime_id": cell.regime_id,
                "nph": cell.nph,
                "source_horizon": cell.horizon,
                "target_horizon": cell.horizon,
                "scientific_result_anchor_claimed": False,
                "declared_delta_ids": declared_delta_ids,
            }
        locks[cell.source_lock_id] = {
            "regime_id": cell.regime_id,
            "nph": cell.nph,
            "route_id": cell.route_id,
            "archive": {
                "path": str(archive),
                "sha256": archive_sha,
            },
            "member": {
                "path": member_path,
                "sha256": member_sha,
            },
            "resolver_trace": trace,
        }
    return {
        "schema": SOURCE_LOCK_SCHEMA,
        "cell_locks": locks,
    }


def _factorial_source_locks_by_bundle(
    tmp_path: Path,
) -> dict[str, dict[str, Any]]:
    first_bundle_id, first_gradient, first_resource = (
        FACTORIAL_BUNDLE_POLICIES[0]
    )
    base = _source_locks(
        tmp_path,
        cells=build_factorial_always_cell_specs(
            active_gradient_policy=first_gradient,
            resource_weighting_scope=first_resource,
        ),
    )
    result: dict[str, dict[str, Any]] = {}
    for bundle_id, gradient_policy, resource_scope in (
        FACTORIAL_BUNDLE_POLICIES
    ):
        arm_locks = json.loads(canonical_json_bytes(base))
        for lock in arm_locks["cell_locks"].values():
            changes = lock["resolver_trace"]["settings_changed"]
            d5 = next(row for row in changes if row["id"] == "D5")
            d5["to"] = resource_scope
            gradient = next(
                row
                for row in changes
                if row.get("field") == "active_gradient_policy"
            )
            gradient["to"] = gradient_policy
        result[bundle_id] = arm_locks
    assert first_bundle_id in result
    return result


def _problem_resolver(regime_id: str, nph: int) -> dict[str, Any]:
    return {"regime_id": regime_id, "nph": nph}


def _pool(
    *,
    count: int,
    prefix: str,
    schema: str = "test_pool_v1",
    source_parent_ordered_labels_sha256: str | None = None,
    ordered_labels: list[str] | None = None,
) -> dict[str, Any]:
    labels = (
        list(ordered_labels)
        if ordered_labels is not None
        else [f"{prefix}-{index:03d}" for index in range(count)]
    )
    assert len(labels) == count
    labels_sha = canonical_sha256(labels)
    pool_sha = canonical_sha256(
        {"prefix": prefix, "count": count, "kind": "pool"}
    )
    payload = {
        "schema": schema,
        "candidate_representation": (
            CANDIDATE_REPRESENTATION_MACRO
            if prefix.startswith("macro")
            else "single_pauli_word_v1"
        ),
        "ordered_labels": labels,
        "ordered_labels_sha256": labels_sha,
        "ordered_pool_sha256": pool_sha,
        "count": count,
        "removed_labels": [],
    }
    if source_parent_ordered_labels_sha256 is not None:
        payload["source_parent_ordered_labels_sha256"] = (
            source_parent_ordered_labels_sha256
        )
    return payload


def _fake_protocol(context: ProtocolResolutionContext) -> dict[str, Any]:
    cell = context.cell
    is_append = cell.selector_family == "append_adapt"
    locked_inventory = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / f"ra_adapt_pool_inventory_{cell.nph}.json"
        ).read_text(encoding="utf-8")
    )
    parent_count = 123 if cell.nph == 3 else 171
    is_global_singleton = (
        context.request.adapter.adapter_id
        == GLOBAL_SINGLE_PAULI_ADAPTER_ID
    )
    executable_count = (
        (102 if cell.nph == 3 else 148)
        if cell.candidate_representation == CANDIDATE_REPRESENTATION_MACRO
        else (
            (948 if cell.nph == 3 else 6508)
            if is_global_singleton
            else parent_count
        )
    )
    is_singleton = (
        cell.candidate_representation
        != CANDIDATE_REPRESENTATION_MACRO
    )
    parent = _pool(
        count=parent_count,
        prefix=f"parent-nph{cell.nph}",
        schema=(
            PARENT_TEMPLATE_INVENTORY_SCHEMA
            if is_singleton
            else "test_pool_v1"
        ),
        ordered_labels=locked_inventory["parent_inventory"][
            "ordered_labels"
        ],
    )
    if is_global_singleton:
        executable = _pool(
            count=executable_count,
            prefix=f"singleton-global-children-nph{cell.nph}",
            schema=GUARDED_SINGLETON_POOL_SCHEMA,
            source_parent_ordered_labels_sha256=(
                parent["ordered_labels_sha256"]
            ),
        )
        executable["ordered_pool_sha256"] = canonical_sha256(
            {
                "kind": "global-singleton-test-pool",
                "regime_id": cell.regime_id,
                "nph": cell.nph,
            }
        )
    elif is_singleton and is_append:
        executable = _pool(
            count=executable_count,
            prefix=f"singleton-global-children-nph{cell.nph}",
            schema=GUARDED_SINGLETON_POOL_SCHEMA,
            source_parent_ordered_labels_sha256=(
                parent["ordered_labels_sha256"]
            ),
        )
    elif is_singleton:
        executable = dict(parent)
    else:
        executable = _pool(
            count=executable_count,
            prefix=f"macro-executable-nph{cell.nph}",
            ordered_labels=locked_inventory["executable_macro_pool"][
                "ordered_labels"
            ],
        )
    lineage_rows = tuple(
        CandidateInventoryLineageRow(
            label=label,
            representation_id=cell.candidate_representation,
            generator_identity=f"test-generator::{label}",
        )
        for label in executable["ordered_labels"]
    )
    inventory_lineage = CandidateInventoryLineageReceipt(
        schema=CANDIDATE_INVENTORY_LINEAGE_SCHEMA,
        candidate_representation=cell.candidate_representation,
        pool_inventory_sha256=canonical_sha256(executable),
        ordered_rows=lineage_rows,
        ordered_rows_sha256=canonical_sha256(
            [row.to_dict() for row in lineage_rows]
        ),
        count=len(lineage_rows),
    ).authority_binding()
    if is_append:
        lineage_authority = {
            "source_lock_schema": "test_append_source_locks_v1",
            "selector_source_id": "test_selector_v1",
            "ra_staged_funnel_invoked": False,
            "candidate_inventory_lineage": inventory_lineage,
        }
    else:
        _requested, _resolved, route_contract, _digest = (
            _repaired_route_contract(
                context.request,
                active_gradient_policy=context.active_gradient_policy,
                resource_weighting_scope=context.resource_weighting_scope,
                algorithm_id=cell.algorithm_id,
            )
        )
        lineage_authority = {
            "parent_route_profile": route_contract[
                "lineage_authority"
            ]["parent_route_profile"],
            "parent_contract_sha256": route_contract[
                "lineage_authority"
            ]["parent_contract_sha256"],
            "candidate_inventory_lineage": inventory_lineage,
        }
    if is_global_singleton:
        lineage_authority["candidate_supply"] = {
            "candidate_adapter_id": GLOBAL_SINGLE_PAULI_ADAPTER_ID,
            "phase_i_candidate_supply": (
                PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON
            ),
            "phase_i_candidate_visibility": (
                PHASE_I_VISIBILITY_ALL_EXECUTABLE
            ),
            "phase_ii_candidate_exposure": (
                PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY
            ),
        }
    payload = {
        "schema": (
            APPEND_ADAPT_PROTOCOL_SCHEMA
            if is_append
            else RA_ADAPT_PROTOCOL_SCHEMA
        ),
        "algorithm_id": cell.algorithm_id,
        "candidate_representation": cell.candidate_representation,
        "adapter_id": context.request.adapter.adapter_id,
        "selector_identity": (
            APPEND_CONVENTIONAL_SELECTOR_ID
            if is_append
            else RA_STAGED_SELECTOR_ID
        ),
        **(
            {"selector_scope": APPEND_CONVENTIONAL_SELECTOR_SCOPE}
            if is_append
            else {}
        ),
        "active_gradient_policy": context.active_gradient_policy,
        "resource_weighting_scope": context.resource_weighting_scope,
        "derivative_chart_id": (
            "exact_ordered_insertion_zero_angle_v1"
        ),
        "trust_policy_id": (
            "supported_source_gram_no_endpoint_overlap_trust_v1"
        ),
        "phase3_solver_id": (
            "supported_metric_projected_generalized_trust_v1"
        ),
        "phase3_multiplier_contract": (
            PhaseIIIMultiplierContract().to_dict()
        ),
        "accepted_refit_scope": "full_ansatz_v1",
        "accepted_refit_coordinate_chart": (
            NATIVE_REFIT_CHART
            if is_append
            else "supported_fs_whitened_fixed_v1"
        ),
        "accepted_refit_base_chart_policy": (
            LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART
            if is_append
            else EXPANDED_RUNTIME_PROJECTED_LOGICAL_BASE_CHART
        ),
        "problem": {
            "family_key": "hh",
            "problem_request_sha256": "b" * 64,
            "num_sites": 2,
            "t": 1.0,
            "u": 8.0,
            "dv": 0.0,
            "v_nn": 0.0,
            "t_prime": 0.0,
            "omega0": 1.0,
            "g_ep": 0.5,
            "n_ph_max": cell.nph,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
            "include_zero_point": True,
            "problem_key": (
                f"test__{cell.regime_id}__nph{cell.nph}"
            ),
            "sector_label": "half_filled_sz0",
            "comparison_space_label": "same_cutoff",
            "reference_label": "hubbard_holstein_reference_state",
            "exact_target_label": "same_cutoff_exact_ed",
            "total_qubits": 8,
        },
        "parent_inventory": parent,
        "executable_pool": executable,
        "optimizer": "powell",
        "optimizer_maxiter": 200,
        "stopping_rule": {
            "maximum_controller_rounds": cell.horizon
        },
        "horizon": cell.horizon,
        "seeds": {"adapt": 7, "transpiler": 7},
        "estimator_accounting_convention": (
            RA_ADAPT_ESTIMATOR_ACCOUNTING
        ),
        "compile_identity": dict(RA_ADAPT_COMPILE_IDENTITY),
        "lineage_authority": lineage_authority,
        "source_locks": dict(context.source_lock_refs),
        "bundle_id": context.bundle_id,
        "bundle_manifest_sha256": context.bundle_manifest_sha256,
        "execution_authorized": False,
        "request": context.request.to_dict(),
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _state() -> dict[str, Any]:
    return {
        "git_commit": "a" * 40,
        "dirty_working_tree": True,
        "cwd": str(REPO_ROOT),
    }


def _patch_fake_global_singleton_pool_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    membership = {}
    for nph, count in ((3, 948), (7, 6508)):
        labels = [
            f"singleton-global-children-nph{nph}-{index:03d}"
            for index in range(count)
        ]
        membership[nph] = {
            "count": count,
            "ordered_labels_sha256": canonical_sha256(labels),
        }
    pool_hashes = {
        regime_id: canonical_sha256(
            {
                "kind": "global-singleton-test-pool",
                "regime_id": regime_id,
                "nph": nph,
            }
        )
        for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS
    }
    monkeypatch.setattr(
        bundle_module,
        "GLOBAL_SINGLETON_POOL_MEMBERSHIP_BY_NPH",
        membership,
    )
    monkeypatch.setattr(
        bundle_module,
        "GLOBAL_SINGLETON_ORDERED_POOL_SHA256_BY_REGIME",
        pool_hashes,
    )


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_canonical_digested(path: Path) -> None:
    payload = _load(path)
    assert path.read_bytes() == canonical_json_bytes(payload) + b"\n"
    digest_payload = dict(payload)
    observed = digest_payload.pop("sha256")
    assert observed == canonical_sha256(digest_payload)


def test_request_policy_discriminators_round_trip_without_guessing() -> None:
    execution = SRExecutionPolicy(
        stop=SRStopPolicy(maximum_controller_rounds=9),
        resume=FreshStart(),
    )
    observation = SRObservationPolicy(
        checkpoint=CheckpointObservation(
            path=Path("runs/example/checkpoints/current.json")
        ),
        estimator_ledger=EstimatorLedgerObservation(
            path=Path("runs/example/result/estimator_ledger.json")
        ),
    )
    request = RAAdaptRequest(
        adapter=MacroCandidateAdapter(),
        method=SRMethodPolicy(
            admission=GreedyBatchAdmission(
                maximum_size=3,
                search_window_size=None,
            ),
            insertion=AppendOnlyInsertion(),
            pruning=MetricPruning(),
            beam=BeamOff(),
        ),
        execution=execution,
        observation=observation,
    )
    payload = request.to_dict()
    assert payload["kind"] == "ra_adapt_request"
    assert payload["method"]["admission"]["kind"] == "greedy_batch"
    assert payload["method"]["insertion"]["kind"] == "append_only"
    assert payload["method"]["pruning"]["kind"] == "metric"
    assert payload["method"]["beam"]["kind"] == "off"
    assert payload["execution"]["resume"]["kind"] == "fresh_start"
    assert ra_adapt_request_from_mapping(payload).to_dict() == payload

    append = AppendAdaptRequest(
        adapter=MacroCandidateAdapter(),
        execution=execution,
        observation=observation,
    )
    append_payload = append.to_dict()
    assert append_payload["kind"] == "append_adapt_request"
    assert (
        append_adapt_request_from_mapping(append_payload).to_dict()
        == append_payload
    )
    broken = json.loads(canonical_json_bytes(payload))
    broken["method"]["insertion"].pop("kind")
    with pytest.raises(ValueError, match="insertion policy kind"):
        ra_adapt_request_from_mapping(broken)


def test_full_matrix_progression_requires_nonvacuous_validation_events() -> None:
    report = bundle_module._digested(
        {
            "schema": bundle_module.VALIDATION_REPORT_SCHEMA,
            "bundle_id": STATIONARY_BUNDLE_ID,
            "materialization_status": "passed",
            "execution_progression_status": "not_run",
            "objective_execution_gates": (
                bundle_module._objective_execution_gates()
            ),
        }
    )
    with pytest.raises(
        BundleMaterializationError,
        match="executed, passed validation",
    ):
        validate_full_matrix_progression(report)

    executed = dict(report)
    executed.pop("sha256")
    executed["execution_progression_status"] = "validation_passed"
    gates = json.loads(
        canonical_json_bytes(executed["objective_execution_gates"])
    )
    for gate in gates:
        gate["status"] = "passed"
        if "observed_count" in gate:
            gate["observed_count"] = 0
            gate["observation_status"] = "observed"
    executed["objective_execution_gates"] = gates
    zero_occurrences = bundle_module._digested(executed)
    with pytest.raises(
        BundleMaterializationError,
        match="non-vacuous g5_insertion_position_correctness_v2",
    ):
        validate_full_matrix_progression(zero_occurrences)

    for gate in gates:
        if "observed_count" in gate:
            gate["observed_count"] = 1
    executed["objective_execution_gates"] = gates
    ready = validate_full_matrix_progression(
        bundle_module._digested(executed)
    )
    assert ready["full_matrix_progression_ready"] is True
    assert ready["interior_insertion_count"] == 1
    assert ready["trust_contraction_count"] == 1
    assert ready["execution_authorized"] is False


def test_study1_cell_matrix_is_exact_and_never_guesses_validation_horizon() -> None:
    blocked = build_study1_cell_specs(validation_horizon=None)
    assert len(blocked) == 58
    validation = [cell for cell in blocked if cell.stage == "validation"]
    full = [cell for cell in blocked if cell.stage == "full"]
    assert len(validation) == 10
    assert len(full) == 48
    assert {cell.regime_id for cell in validation} == set(
        VALIDATION_REGIMES
    )
    assert {cell.route_id for cell in validation} == set(
        VALIDATION_ROUTE_IDS
    )
    assert {cell.nph for cell in validation} == {3}
    assert {cell.horizon for cell in validation} == {None}
    assert {
        (cell.regime_id, cell.nph) for cell in full
    } == set(FULL_VISIBLE_REGIME_CUTOFF_PAIRS)
    assert all(
        {
            cell.nph
            for cell in full
            if cell.regime_id == regime_id
        }
        == {3, 7}
        for regime_id in {
            cell.regime_id for cell in full
        }
    )
    assert {cell.route_id for cell in full} == set(MACRO_ROUTE_IDS)
    assert {cell.horizon for cell in full} == {50}
    assert len({cell.source_lock_id for cell in blocked}) == 50

    with pytest.raises(BundleMaterializationError, match="fixed at 50"):
        build_study1_cell_specs(validation_horizon=6, full_horizon=49)


def test_core_cell_matrix_is_exactly_the_claim_facing_48() -> None:
    cells = build_core_cell_specs()
    assert len(cells) == 48
    assert {cell.stage for cell in cells} == {"core"}
    assert {cell.horizon for cell in cells} == {50}
    assert {
        (cell.regime_id, cell.nph) for cell in cells
    } == set(CLAIM_FACING_REGIME_CUTOFF_PAIRS)
    for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS:
        regime_cells = [
            cell
            for cell in cells
            if cell.regime_id == regime_id and cell.nph == nph
        ]
        assert len(regime_cells) == 8
        macro = [
            cell
            for cell in regime_cells
            if cell.candidate_representation
            == CANDIDATE_REPRESENTATION_MACRO
        ]
        singleton = [
            cell
            for cell in regime_cells
            if cell.candidate_representation
            == CANDIDATE_REPRESENTATION_SINGLE_PAULI
        ]
        assert {cell.route_id for cell in macro} == set(MACRO_ROUTE_IDS)
        assert {cell.route_id for cell in singleton} == set(
            SINGLETON_CORE_ROUTE_IDS
        )
        assert sum(
            cell.selector_family == "append_adapt"
            for cell in regime_cells
        ) == 2
        assert sum(
            cell.selector_family == "ra_adapt"
            for cell in regime_cells
        ) == 6
    assert len({cell.cell_id for cell in cells}) == 48
    assert len({cell.source_lock_id for cell in cells}) == 48

    with pytest.raises(
        BundleMaterializationError,
        match="core horizon is fixed at 50",
    ):
        build_core_cell_specs(horizon=49)


def test_factorial_always_matrix_is_exactly_four_homogeneous_arms() -> None:
    expected_policies = [
        (
            "ra_repair_always_factorial_stationary_late_v1",
            ACTIVE_GRADIENT_STATIONARY,
            RESOURCE_WEIGHTING_LATE,
            "gradient_stationary__phase1_cost_off",
        ),
        (
            "ra_repair_always_factorial_measured_late_v1",
            ACTIVE_GRADIENT_MEASURED,
            RESOURCE_WEIGHTING_LATE,
            "gradient_measured__phase1_cost_off",
        ),
        (
            "ra_repair_always_factorial_stationary_all_phase_v1",
            ACTIVE_GRADIENT_STATIONARY,
            RESOURCE_WEIGHTING_ALL_PHASE,
            "gradient_stationary__phase1_cost_on",
        ),
        (
            "ra_repair_always_factorial_measured_all_phase_v1",
            ACTIVE_GRADIENT_MEASURED,
            RESOURCE_WEIGHTING_ALL_PHASE,
            "gradient_measured__phase1_cost_on",
        ),
    ]
    assert [
        (bundle_id, gradient, resource)
        for bundle_id, gradient, resource, _suffix in expected_policies
    ] == list(FACTORIAL_BUNDLE_POLICIES)

    all_cells = []
    for (
        _bundle_id,
        gradient_policy,
        resource_scope,
        suffix,
    ) in expected_policies:
        cells = build_factorial_always_cell_specs(
            active_gradient_policy=gradient_policy,
            resource_weighting_scope=resource_scope,
        )
        all_cells.extend(cells)
        assert len(cells) == 12
        assert {cell.stage for cell in cells} == {"factorial"}
        assert {cell.horizon for cell in cells} == {FULL_HORIZON}
        assert {cell.selector_family for cell in cells} == {"ra_adapt"}
        assert {
            (cell.regime_id, cell.nph) for cell in cells
        } == set(CLAIM_FACING_REGIME_CUTOFF_PAIRS)
        assert {
            cell.route_id for cell in cells
        } == {
            ROUTE_RA_MACRO_ALWAYS,
            ROUTE_RA_SINGLETON_ALWAYS,
        }
        for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS:
            regime_cells = [
                cell
                for cell in cells
                if cell.regime_id == regime_id and cell.nph == nph
            ]
            assert [
                cell.candidate_representation
                for cell in regime_cells
            ] == [
                CANDIDATE_REPRESENTATION_MACRO,
                CANDIDATE_REPRESENTATION_SINGLE_PAULI,
            ]
            assert [
                cell.route_id for cell in regime_cells
            ] == [
                ROUTE_RA_MACRO_ALWAYS,
                ROUTE_RA_SINGLETON_ALWAYS,
            ]
        assert all(
            cell.cell_id.endswith(f"__{suffix}") for cell in cells
        )
        for cell in cells:
            request = bundle_module._build_request(
                cell,
                bundle_dir=REPO_ROOT,
            )
            assert isinstance(request, RAAdaptRequest)
            assert isinstance(
                request.method.insertion,
                AlwaysCommutationReducedInsertion,
            )

    assert len(all_cells) == 48
    assert len({cell.cell_id for cell in all_cells}) == 48
    assert len({cell.source_lock_id for cell in all_cells}) == 12
    with pytest.raises(
        BundleMaterializationError,
        match="factorial horizon is fixed at 50",
    ):
        build_factorial_always_cell_specs(
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
            horizon=49,
        )
    with pytest.raises(
        BundleMaterializationError,
        match="accepts only the declared",
    ):
        build_factorial_always_cell_specs(
            active_gradient_policy="not_a_gradient_policy",
            resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
        )


def test_global_singleton_insertion_matrix_and_typed_requests_are_exact() -> None:
    cells = build_global_singleton_insertion_cell_specs()
    assert len(cells) == 12
    assert {cell.stage for cell in cells} == {
        "global_singleton_insertion"
    }
    assert {cell.horizon for cell in cells} == {FULL_HORIZON}
    assert {cell.selector_family for cell in cells} == {"ra_adapt"}
    assert {
        (cell.regime_id, cell.nph) for cell in cells
    } == set(CLAIM_FACING_REGIME_CUTOFF_PAIRS)
    assert bundle_module.GLOBAL_SINGLETON_POOL_MEMBERSHIP_BY_NPH == {
        3: {
            "count": 948,
            "ordered_labels_sha256": (
                "02995a2c570d4322e46e55e3a532381ff7eff85dc3c2de8cb2b30ed888b76906"
            ),
        },
        7: {
            "count": 6508,
            "ordered_labels_sha256": (
                "079478057eea213139dc2f3c7486097496454421a44677c290b5dc55860accb7"
            ),
        },
    }
    for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS:
        pair = [
            cell
            for cell in cells
            if cell.regime_id == regime_id and cell.nph == nph
        ]
        assert [cell.route_id for cell in pair] == list(
            GLOBAL_SINGLETON_INSERTION_ROUTE_IDS
        )
        assert [cell.cell_id for cell in pair] == [
            (
                f"global_singleton__{regime_id}__nph{nph}__"
                f"{route_id}"
            )
            for route_id in GLOBAL_SINGLETON_INSERTION_ROUTE_IDS
        ]
        assert [cell.algorithm_id for cell in pair] == [
            (
                "paper_i_ra_adapt_global_singleton_"
                "append_commutation_reduced_v1"
            ),
            (
                "paper_i_ra_adapt_global_singleton_"
                "plateau_commutation_v1"
            ),
        ]
        requests = [
            bundle_module._build_request(cell, bundle_dir=REPO_ROOT)
            for cell in pair
        ]
        assert all(isinstance(request, RAAdaptRequest) for request in requests)
        assert all(
            isinstance(
                request.adapter,
                GlobalSinglePauliWordCandidateAdapter,
            )
            for request in requests
        )
        assert isinstance(
            requests[0].method.insertion,
            AppendCommutationReducedInsertion,
        )
        assert isinstance(
            requests[1].method.insertion,
            bundle_module.PlateauCommutationInsertion,
        )
    with pytest.raises(
        BundleMaterializationError,
        match="comparison horizon is fixed at 50",
    ):
        build_global_singleton_insertion_cell_specs(horizon=49)


def test_qiskit_cost_plateau_pilot_matrix_and_requests_are_exact() -> None:
    cells = build_qiskit_cost_plateau_pilot_cell_specs()
    assert [cell.cell_id for cell in cells] == [
        (
            "qiskit_cost_pilot__strong_weak_u8__nph3__"
            "ra_macro_plateau"
        ),
        (
            "qiskit_cost_pilot__strong_strong_u8__nph7__"
            "ra_global_singleton_plateau_commutation"
        ),
    ]
    assert [cell.route_id for cell in cells] == list(
        QISKIT_COST_PILOT_ROUTE_IDS
    )
    assert [cell.algorithm_id for cell in cells] == [
        QISKIT_COST_PILOT_MACRO_ALGORITHM_ID,
        QISKIT_COST_PILOT_GLOBAL_SINGLETON_ALGORITHM_ID,
    ]
    assert [
        (cell.regime_id, cell.nph) for cell in cells
    ] == [
        ("strong_weak_u8", 3),
        ("strong_strong_u8", 7),
    ]
    assert {cell.stage for cell in cells} == {
        "qiskit_cost_plateau_pilot"
    }
    assert {cell.horizon for cell in cells} == {FULL_HORIZON}
    assert {cell.selector_family for cell in cells} == {"ra_adapt"}
    assert [
        cell.candidate_representation for cell in cells
    ] == [
        CANDIDATE_REPRESENTATION_MACRO,
        CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    ]
    requests = [
        bundle_module._build_request(cell, bundle_dir=REPO_ROOT)
        for cell in cells
    ]
    assert isinstance(requests[0].adapter, MacroCandidateAdapter)
    assert isinstance(
        requests[1].adapter,
        GlobalSinglePauliWordCandidateAdapter,
    )
    assert all(
        isinstance(
            request.method.insertion,
            bundle_module.PlateauCommutationInsertion,
        )
        for request in requests
    )
    with pytest.raises(
        BundleMaterializationError,
        match="pilot horizon is fixed at 50",
    ):
        build_qiskit_cost_plateau_pilot_cell_specs(horizon=49)


def test_qiskit_cost_always13_matrix_and_request_are_exact() -> None:
    cells = build_qiskit_cost_always13_cell_specs()
    assert len(cells) == 1
    cell = cells[0]
    assert cell.cell_id == (
        "qiskit_cost_always13__strong_weak_u8__nph3__"
        "ra_macro_always"
    )
    assert cell.stage == "qiskit_cost_always13_diagnostic"
    assert cell.regime_id == "strong_weak_u8"
    assert cell.nph == 3
    assert cell.route_id == bundle_module.ROUTE_RA_MACRO_ALWAYS
    assert cell.algorithm_id == QISKIT_COST_ALWAYS13_ALGORITHM_ID
    assert cell.candidate_representation == (
        CANDIDATE_REPRESENTATION_MACRO
    )
    assert cell.horizon == QISKIT_COST_ALWAYS13_HORIZON
    assert [cell.route_id] == list(QISKIT_COST_ALWAYS13_ROUTE_IDS)
    request = bundle_module._build_request(cell, bundle_dir=REPO_ROOT)
    assert isinstance(request.adapter, MacroCandidateAdapter)
    assert isinstance(
        request.method.insertion,
        AlwaysCommutationReducedInsertion,
    )
    assert (
        request.execution.stop.maximum_controller_rounds
        == QISKIT_COST_ALWAYS13_HORIZON
    )
    with pytest.raises(
        BundleMaterializationError,
        match="always13 diagnostic horizon is fixed at 13",
    ):
        build_qiskit_cost_always13_cell_specs(horizon=12)


def test_phase3_qiskit_mixed_horizon_matrix_and_requests_are_exact() -> None:
    cells = build_phase3_qiskit_mixed_horizon_cell_specs()
    assert len(cells) == 6
    assert [
        (cell.regime_id, cell.nph) for cell in cells
    ] == list(CLAIM_FACING_REGIME_CUTOFF_PAIRS)
    assert [cell.route_id for cell in cells] == [
        ROUTE_RA_GLOBAL_SINGLETON_PLATEAU
    ] * 6
    assert set(cell.algorithm_id for cell in cells) == {
        RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
    }
    assert {cell.stage for cell in cells} == {
        "phase3_qiskit_candidate"
    }
    assert {
        cell.horizon for cell in cells if cell.nph == 3
    } == {PHASE3_QISKIT_WEAK_HOLSTEIN_HORIZON}
    assert {
        cell.horizon for cell in cells if cell.nph == 7
    } == {PHASE3_QISKIT_STRONG_HOLSTEIN_HORIZON}
    assert list(PHASE3_QISKIT_ROUTE_IDS) == [
        ROUTE_RA_GLOBAL_SINGLETON_PLATEAU
    ]
    for cell in cells:
        request = bundle_module._build_request(cell, bundle_dir=REPO_ROOT)
        assert isinstance(
            request.adapter,
            GlobalSinglePauliWordCandidateAdapter,
        )
        assert isinstance(
            request.method.insertion,
            bundle_module.PlateauCommutationInsertion,
        )
        assert (
            request.execution.stop.maximum_controller_rounds
            == cell.horizon
        )
        _requested, route_profile, route_contract, _digest = (
            _repaired_route_contract(
                request,
                active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
                resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
                algorithm_id=cell.algorithm_id,
            )
        )
        assert route_profile.endswith(
            "__" + RA_ADAPT_PHASE3_QISKIT_COST_ROUTE_SUFFIX
        )
        assert route_contract["lineage_authority"][
            "parent_route_profile"
        ] == PHASE3_QISKIT_PAGE7_PARENT_ROUTE_PROFILE
        assert route_contract["lineage_authority"][
            "parent_contract_sha256"
        ] == PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256

    with pytest.raises(
        BundleMaterializationError,
        match="horizons are fixed at 50.*70",
    ):
        build_phase3_qiskit_mixed_horizon_cell_specs(
            strong_holstein_horizon=69
        )


def test_implementation_inventory_closes_public_sr_snake_imports() -> None:
    inventory = bundle_module._implementation_source_inventory(REPO_ROOT)
    assert inventory["schema"] == (
        "ra_adapt_implementation_source_inventory_v2"
    )
    assert inventory["resolution"] == (
        "static_repo_local_import_closure_with_package_initializers_v2"
    )
    paths = {row["path"] for row in inventory["files"]}
    required_paths = {
        "pipelines/static_adapt/__init__.py",
        "pipelines/static_adapt/builders/__init__.py",
        "pipelines/static_adapt/ra_adapt/__init__.py",
        "pipelines/static_adapt/sr_snake/__init__.py",
        "pipelines/static_adapt/sr_snake/runner.py",
        "src/__init__.py",
        "src/quantum/__init__.py",
    }
    assert required_paths <= paths
    initializer_paths = set(inventory["package_initializer_paths"])
    assert required_paths - {
        "pipelines/static_adapt/sr_snake/runner.py"
    } <= initializer_paths
    assert inventory["package_initializer_count"] == len(
        initializer_paths
    )
    assert initializer_paths <= paths


def test_core_requests_pin_typed_insertion_and_singleton_exposure() -> None:
    cells = build_core_cell_specs()
    protocols: dict[str, dict[str, Any]] = {}
    for cell in cells:
        request = bundle_module._build_request(
            cell,
            bundle_dir=REPO_ROOT,
        )
        if cell.selector_family == "append_adapt":
            assert isinstance(request, AppendAdaptRequest)
        else:
            assert isinstance(request, RAAdaptRequest)
            expected_type = (
                AlwaysCommutationReducedInsertion
                if cell.route_id
                in {
                    ROUTE_RA_MACRO_ALWAYS,
                    ROUTE_RA_SINGLETON_ALWAYS,
                }
                else (
                    AppendOnlyInsertion
                    if cell.route_id == ROUTE_RA_SINGLETON_APPEND_ONLY
                    or cell.route_id
                    == bundle_module.ROUTE_RA_MACRO_APPEND_ONLY
                    else bundle_module.PlateauCommutationInsertion
                )
            )
            assert isinstance(request.method.insertion, expected_type)
        context = ProtocolResolutionContext(
            cell=cell,
            problem=_problem_resolver(cell.regime_id, cell.nph),
            request=request,
            active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
            resource_weighting_scope=RESOURCE_WEIGHTING_LATE,
            bundle_id="test_core_bundle",
            bundle_manifest_sha256="1" * 64,
            source_lock_refs={},
            materialization_authority=None,  # type: ignore[arg-type]
        )
        protocols[cell.cell_id] = _fake_protocol(context)

    bundle_module._validate_singleton_pool_contracts(protocols, cells)
    singleton_always = next(
        cell
        for cell in cells
        if cell.route_id == ROUTE_RA_SINGLETON_ALWAYS
    )
    singleton_parent_route = protocols[singleton_always.cell_id][
        "lineage_authority"
    ]["parent_route_profile"]
    assert "macro_only" not in singleton_parent_route
    assert "commutation_reduced_insertion" in singleton_parent_route
    for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS:
        singleton_cells = [
            cell
            for cell in cells
            if (
                cell.regime_id == regime_id
                and cell.nph == nph
                and cell.candidate_representation
                == CANDIDATE_REPRESENTATION_SINGLE_PAULI
            )
        ]
        parents = {
            (
                protocols[cell.cell_id]["parent_inventory"][
                    "ordered_labels_sha256"
                ],
                protocols[cell.cell_id]["parent_inventory"][
                    "ordered_pool_sha256"
                ],
            )
            for cell in singleton_cells
        }
        assert len(parents) == 1
        append_cell = next(
            cell
            for cell in singleton_cells
            if cell.route_id == ROUTE_APPEND_SINGLETON
        )
        assert (
            protocols[append_cell.cell_id]["executable_pool"]["schema"]
            == GUARDED_SINGLETON_POOL_SCHEMA
        )
        for cell in singleton_cells:
            if cell.selector_family == "ra_adapt":
                assert (
                    protocols[cell.cell_id]["executable_pool"]["schema"]
                    == PARENT_TEMPLATE_INVENTORY_SCHEMA
                )

    append_cell = next(
        cell
        for cell in cells
        if cell.route_id == ROUTE_APPEND_SINGLETON
    )
    drifted = json.loads(canonical_json_bytes(protocols))
    drifted[append_cell.cell_id]["executable_pool"][
        "source_parent_ordered_labels_sha256"
    ] = "0" * 64
    with pytest.raises(
        BundleMaterializationError,
        match="global guarded child pool",
    ):
        bundle_module._validate_singleton_pool_contracts(drifted, cells)


def test_materializes_selected_core_without_study1_progression_fields(
    tmp_path: Path,
) -> None:
    cells = build_core_cell_specs()
    destination = tmp_path / "core-bundles"
    receipt = materialize_core_bundle(
        destination,
        problem_resolver=_problem_resolver,
        protocol_resolver=_fake_protocol,
        source_locks=_source_locks(tmp_path, cells=cells),
        repository_state=_state(),
        repo_root=REPO_ROOT,
        environment_fingerprint={
            "python_implementation": "CPython",
            "python_version": "test",
            "platform_system": "test",
            "platform_release": "test",
            "platform_machine": "test",
        },
        materialization_timestamp="2026-07-28T12:00:00Z",
    )
    assert receipt.bundle_id == CORE_BUNDLE_ID
    assert receipt.cell_count == 48
    assert receipt.materialization_status == "passed"
    bundle_dir = destination / CORE_BUNDLE_ID
    manifest = _load(bundle_dir / "bundle_manifest.json")
    source_locks = _load(bundle_dir / "source_locks.json")
    expected = _load(bundle_dir / "expected_artifacts.json")
    validation = _load(bundle_dir / "validation_report.json")
    assert manifest["campaign_id"] == CORE_CAMPAIGN_ID
    assert manifest["study_id"] == CORE_CAMPAIGN_ID
    assert manifest["run_class"] == CORE_RUN_CLASS
    assert manifest["visible_target"]["target_id"] == CORE_VISIBLE_TARGET_ID
    assert manifest["stationarity_winner_selected"] is True
    assert manifest["active_gradient_policy"] == ACTIVE_GRADIENT_STATIONARY
    assert manifest["core_cell_count"] == 48
    assert source_locks["campaign_authorities"][
        "stationarity_selection"
    ]["verified"] is True
    forbidden_manifest_fields = {
        "study1_shared_execution_dedupe",
        "execution_progression_contract",
        "post_study_1_user_decision_required",
        "validation_cell_count",
        "full_cell_count",
    }
    assert not forbidden_manifest_fields.intersection(manifest)
    assert not {
        "execution_progression_status",
        "objective_execution_gates",
        "user_decision_required_after_study_1",
    }.intersection(validation)
    assert validation["stationarity_winner_selected"] is True
    assert validation["scientific_execution_status"] == "not_run"
    assert {
        check["id"] for check in validation["checks"]
    } == {
        "bundle_schema_and_digest",
        "exact_core_cell_matrix",
        "source_locks_exact_bytes",
        "resolved_protocol_contracts",
        "macro_pool_hash_equality",
        "singleton_pool_exposure_contracts",
        "all_cells_direct_execution",
        "protocol_execution_separation",
        "paper_i_run_materialization_gate",
    }
    binding = validation["core_validation_binding"]
    assert binding["cell_count"] == 48
    assert binding["direct_execution_cell_count"] == 48
    assert len(binding["semantic_route_ids"]) == 8
    for cell in cells:
        fulfillment = expected["cells"][cell.cell_id][
            "execution_fulfillment"
        ]
        assert fulfillment == {
            "fulfillment_kind": "direct_execution_v1",
            "canonical_execution": {
                "bundle_id": CORE_BUNDLE_ID,
                "cell_id": cell.cell_id,
            },
        }
        template = _load(
            bundle_dir
            / "execution_templates"
            / f"{cell.cell_id}.json"
        )
        assert template["execution_fulfillment"] == fulfillment
        assert template["campaign_id"] == CORE_CAMPAIGN_ID
        assert template["run_class"] == CORE_RUN_CLASS
        assert "dedupe_contract_sha256" not in fulfillment
    validated = load_validated_bundle_protocol(
        bundle_dir / "protocols" / f"{cells[0].cell_id}.json"
    )
    assert validated.bundle_id == CORE_BUNDLE_ID
    assert validated.active_gradient_policy == ACTIVE_GRADIENT_STATIONARY


def test_materializes_and_loads_all_four_factorial_policy_arms(
    tmp_path: Path,
) -> None:
    source_locks_by_bundle = _factorial_source_locks_by_bundle(
        tmp_path
    )
    destination = tmp_path / "factorial-bundles"
    receipts = materialize_factorial_always_bundles(
        destination,
        problem_resolver=_problem_resolver,
        protocol_resolver=_fake_protocol,
        source_locks_by_bundle=source_locks_by_bundle,
        repository_state=_state(),
        repo_root=REPO_ROOT,
        environment_fingerprint={
            "python_implementation": "CPython",
            "python_version": "test",
            "platform_system": "test",
            "platform_release": "test",
            "platform_machine": "test",
        },
        materialization_timestamp="2026-07-29T12:00:00Z",
    )
    assert [receipt.bundle_id for receipt in receipts] == [
        bundle_id
        for bundle_id, _gradient, _resource
        in FACTORIAL_BUNDLE_POLICIES
    ]
    assert {receipt.cell_count for receipt in receipts} == {12}
    assert {
        receipt.materialization_status for receipt in receipts
    } == {"passed"}

    protocols_by_base: dict[str, list[dict[str, Any]]] = {}
    loaded_scopes = set()
    all_output_paths = set()
    for bundle_id, gradient_policy, resource_scope in (
        FACTORIAL_BUNDLE_POLICIES
    ):
        bundle_dir = destination / bundle_id
        manifest = _load(bundle_dir / "bundle_manifest.json")
        source_locks = _load(bundle_dir / "source_locks.json")
        expected = _load(bundle_dir / "expected_artifacts.json")
        validation = _load(bundle_dir / "validation_report.json")
        cells = build_factorial_always_cell_specs(
            active_gradient_policy=gradient_policy,
            resource_weighting_scope=resource_scope,
        )
        assert manifest["campaign_id"] == FACTORIAL_CAMPAIGN_ID
        assert manifest["study_id"] == FACTORIAL_CAMPAIGN_ID
        assert manifest["run_class"] == FACTORIAL_RUN_CLASS
        assert manifest["visible_target"]["target_id"] == (
            FACTORIAL_VISIBLE_TARGET_ID
        )
        assert manifest["active_gradient_policy"] == gradient_policy
        assert manifest["resource_weighting_scope"] == resource_scope
        assert manifest["factorial_arm_cell_count"] == 12
        assert manifest["stationarity_winner_selected"] is False
        assert [row["cell_id"] for row in manifest["cells"]] == [
            cell.cell_id for cell in cells
        ]
        arm_contract = manifest["factorial_arm_contract"]
        assert arm_contract["factorial_shape"] == {
            "active_gradient_policy_count": 2,
            "resource_weighting_scope_count": 2,
            "regime_cutoff_pair_count": 6,
            "candidate_representation_count": 2,
            "total_cell_count": 48,
            "bundle_count": 4,
            "cells_per_bundle": 12,
        }
        assert arm_contract["typed_insertion_policy"] == (
            AlwaysCommutationReducedInsertion.kind
        )
        assert arm_contract["runtime_insertion_mode"] == (
            bundle_module.ALWAYS_REDUCED_INSERTION_MODE
        )
        assert arm_contract["phase1_cost_term"] == (
            "enabled_v1"
            if resource_scope == RESOURCE_WEIGHTING_ALL_PHASE
            else "disabled_for_phase1_only_v1"
        )
        assert {
            check["id"] for check in validation["checks"]
        } == {
            "bundle_schema_and_digest",
            "exact_factorial_arm_matrix",
            "source_locks_exact_bytes",
            "resolved_protocol_contracts",
            "macro_pool_hash_equality",
            "singleton_pool_exposure_contracts",
            "all_cells_direct_execution",
            "protocol_execution_separation",
            "paper_i_run_materialization_gate",
        }
        binding = validation["factorial_validation_binding"]
        assert binding["active_gradient_policy"] == gradient_policy
        assert binding["resource_weighting_scope"] == resource_scope
        assert binding["direct_execution_cell_count"] == 12

        for lock in source_locks["cell_locks"].values():
            changes = lock["resolver_trace"]["settings_changed"]
            d5 = [row for row in changes if row["id"] == "D5"]
            assert len(d5) == 1
            assert d5[0]["field"] == "resource_weighting_scope"
            assert d5[0]["to"] == resource_scope
        for cell in cells:
            protocol_path = (
                bundle_dir / "protocols" / f"{cell.cell_id}.json"
            )
            protocol = _load(protocol_path)
            assert protocol["active_gradient_policy"] == gradient_policy
            assert protocol["resource_weighting_scope"] == resource_scope
            materialization = protocol["bundle_materialization"]
            assert materialization["active_gradient_policy"] == (
                gradient_policy
            )
            assert materialization["resource_weighting_scope"] == (
                resource_scope
            )
            route = protocol["route_contract"]
            assert route["execution_settings"][
                "ra_active_gradient_policy"
            ] == gradient_policy
            assert route["execution_settings"][
                "ra_resource_weighting_scope"
            ] == resource_scope
            assert route["execution_settings"][
                "adapt_insertion_mode"
            ] == bundle_module.ALWAYS_REDUCED_INSERTION_MODE
            assert route["semantic_invariants"][
                "active_gradient_policy"
            ] == gradient_policy
            assert route["semantic_invariants"][
                "resource_weighting_scope"
            ] == resource_scope
            assert route["semantic_invariants"][
                "insertion_position_scope"
            ] == bundle_module.ALWAYS_REDUCED_INSERTION_SCOPE
            assert route["semantic_invariants"][
                "insertion_equivalence_policy"
            ] == bundle_module.ALWAYS_REDUCED_INSERTION_EQUIVALENCE
            assert protocol["request"]["method"]["insertion"]["kind"] == (
                AlwaysCommutationReducedInsertion.kind
            )
            template = _load(
                bundle_dir
                / "execution_templates"
                / f"{cell.cell_id}.json"
            )
            assert template["campaign_id"] == FACTORIAL_CAMPAIGN_ID
            assert template["run_class"] == FACTORIAL_RUN_CLASS
            assert template["execution_fulfillment"] == {
                "fulfillment_kind": "direct_execution_v1",
                "canonical_execution": {
                    "bundle_id": bundle_id,
                    "cell_id": cell.cell_id,
                },
            }
            output_paths = {
                artifact["path"]
                for artifact in expected["cells"][cell.cell_id][
                    "expected_run_artifacts"
                ].values()
            }
            qualified_paths = {
                f"{bundle_id}/{path}" for path in output_paths
            }
            assert not all_output_paths.intersection(qualified_paths)
            all_output_paths.update(qualified_paths)
            base_id = cell.cell_id.split("__gradient_", 1)[0]
            protocols_by_base.setdefault(base_id, []).append(protocol)

        loaded = load_validated_bundle_protocol(
            bundle_dir / "protocols" / f"{cells[0].cell_id}.json"
        )
        assert loaded.bundle_id == bundle_id
        assert loaded.active_gradient_policy == gradient_policy
        assert loaded.resource_weighting_scope == resource_scope
        loaded_scopes.add(loaded.resource_weighting_scope)

    assert loaded_scopes == {
        RESOURCE_WEIGHTING_LATE,
        RESOURCE_WEIGHTING_ALL_PHASE,
    }
    assert len(all_output_paths) == 48 * len(EXPECTED_ARTIFACT_ROLES)
    assert set(protocols_by_base) == {
        f"core__{regime_id}__nph{nph}__{route_id}"
        for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS
        for route_id in (
            ROUTE_RA_MACRO_ALWAYS,
            ROUTE_RA_SINGLETON_ALWAYS,
        )
    }
    for arm_protocols in protocols_by_base.values():
        assert len(arm_protocols) == 4
        projections = []
        for protocol in arm_protocols:
            route_execution = dict(
                protocol["route_contract"]["execution_settings"]
            )
            route_execution.pop("ra_active_gradient_policy")
            route_execution.pop("ra_resource_weighting_scope")
            route_invariants = dict(
                protocol["route_contract"]["semantic_invariants"]
            )
            route_invariants.pop("active_gradient_policy")
            route_invariants.pop("resource_weighting_scope")
            projections.append(
                {
                    "algorithm_id": protocol["algorithm_id"],
                    "candidate_representation": protocol[
                        "candidate_representation"
                    ],
                    "selector_identity": protocol["selector_identity"],
                    "derivative_chart_id": protocol[
                        "derivative_chart_id"
                    ],
                    "trust_policy_id": protocol["trust_policy_id"],
                    "phase3_solver_id": protocol["phase3_solver_id"],
                    "phase3_multiplier_contract": protocol[
                        "phase3_multiplier_contract"
                    ],
                    "accepted_refit_scope": protocol[
                        "accepted_refit_scope"
                    ],
                    "accepted_refit_coordinate_chart": protocol[
                        "accepted_refit_coordinate_chart"
                    ],
                    "accepted_refit_base_chart_policy": protocol[
                        "accepted_refit_base_chart_policy"
                    ],
                    "problem": protocol["problem"],
                    "parent_inventory": protocol["parent_inventory"],
                    "executable_pool": protocol["executable_pool"],
                    "optimizer": protocol["optimizer"],
                    "optimizer_maxiter": protocol["optimizer_maxiter"],
                    "stopping_rule": protocol["stopping_rule"],
                    "seeds": protocol["seeds"],
                    "estimator_accounting_convention": protocol[
                        "estimator_accounting_convention"
                    ],
                    "compile_identity": protocol["compile_identity"],
                    "lineage_authority": protocol["lineage_authority"],
                    "request_method": protocol["request"]["method"],
                    "route_execution": route_execution,
                    "route_invariants": route_invariants,
                }
            )
        assert len({canonical_sha256(row) for row in projections}) == 1


def test_materializes_and_loads_global_singleton_insertion_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_global_singleton_pool_authority(monkeypatch)
    cells = build_global_singleton_insertion_cell_specs()
    destination = tmp_path / "global-singleton-bundles"
    receipt = materialize_global_singleton_insertion_bundle(
        destination,
        problem_resolver=_problem_resolver,
        protocol_resolver=_fake_protocol,
        source_locks=_source_locks(tmp_path, cells=cells),
        repository_state=_state(),
        repo_root=REPO_ROOT,
        environment_fingerprint={
            "python_implementation": "CPython",
            "python_version": "test",
            "platform_system": "test",
            "platform_release": "test",
            "platform_machine": "test",
        },
        materialization_timestamp="2026-07-30T12:00:00Z",
    )
    assert receipt.bundle_id == GLOBAL_SINGLETON_BUNDLE_ID
    assert receipt.cell_count == 12
    assert receipt.materialization_status == "passed"
    bundle_dir = destination / GLOBAL_SINGLETON_BUNDLE_ID
    manifest = _load(bundle_dir / "bundle_manifest.json")
    expected = _load(bundle_dir / "expected_artifacts.json")
    validation = _load(bundle_dir / "validation_report.json")
    assert manifest["campaign_id"] == GLOBAL_SINGLETON_CAMPAIGN_ID
    assert manifest["study_id"] == GLOBAL_SINGLETON_CAMPAIGN_ID
    assert manifest["run_class"] == GLOBAL_SINGLETON_RUN_CLASS
    assert manifest["visible_target"]["target_id"] == (
        GLOBAL_SINGLETON_VISIBLE_TARGET_ID
    )
    assert manifest["active_gradient_policy"] == (
        ACTIVE_GRADIENT_STATIONARY
    )
    assert manifest["resource_weighting_scope"] == (
        RESOURCE_WEIGHTING_ALL_PHASE
    )
    assert manifest["stationarity_condition"] == "always_applied_v1"
    assert manifest["phase1_cost_term"] == "always_applied_v1"
    contract = manifest["global_singleton_insertion_contract"]
    assert contract["candidate_adapter_id"] == (
        GLOBAL_SINGLE_PAULI_ADAPTER_ID
    )
    assert contract["phase_i_candidate_supply"] == (
        PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON
    )
    assert contract["phase_i_candidate_visibility"] == (
        PHASE_I_VISIBILITY_ALL_EXECUTABLE
    )
    assert contract["phase_i_shortlist_size"] == 24
    assert contract["phase_ii_candidate_exposure"] == (
        PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY
    )
    assert contract["phase_ii_shortlist_size"] == 12
    assert contract["phase_iii_admission_cardinality"] == 1
    assert {
        check["id"] for check in validation["checks"]
    } == {
        "bundle_schema_and_digest",
        "exact_global_singleton_insertion_matrix",
        "source_locks_exact_bytes",
        "resolved_protocol_contracts",
        "macro_pool_hash_equality",
        "singleton_pool_exposure_contracts",
        "global_singleton_source_lock_pair_equality",
        "global_singleton_cross_arm_scientific_equality",
        "all_cells_direct_execution",
        "protocol_execution_separation",
        "paper_i_run_materialization_gate",
    }
    binding = validation[
        "global_singleton_insertion_validation_binding"
    ]
    assert binding["direct_execution_cell_count"] == 12
    assert binding["candidate_adapter_id"] == (
        GLOBAL_SINGLE_PAULI_ADAPTER_ID
    )

    protocols: dict[str, dict[str, Any]] = {}
    for cell in cells:
        protocol_path = (
            bundle_dir / "protocols" / f"{cell.cell_id}.json"
        )
        protocol = _load(protocol_path)
        protocols[cell.cell_id] = protocol
        assert protocol["adapter_id"] == GLOBAL_SINGLE_PAULI_ADAPTER_ID
        assert protocol["parent_inventory"]["count"] == (
            123 if cell.nph == 3 else 171
        )
        assert protocol["executable_pool"]["count"] == (
            948 if cell.nph == 3 else 6508
        )
        assert protocol["active_gradient_policy"] == (
            ACTIVE_GRADIENT_STATIONARY
        )
        assert protocol["resource_weighting_scope"] == (
            RESOURCE_WEIGHTING_ALL_PHASE
        )
        route = protocol["route_contract"]
        assert route["semantic_invariants"][
            "phase_i_candidate_supply"
        ] == PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON
        assert route["semantic_invariants"][
            "phase_i_candidate_visibility"
        ] == PHASE_I_VISIBILITY_ALL_EXECUTABLE
        assert route["semantic_invariants"][
            "phase_ii_candidate_exposure"
        ] == PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY
        assert route["execution_settings"]["phase1_shortlist_size"] == 24
        assert route["execution_settings"]["phase2_shortlist_size"] == 12
        assert route["semantic_invariants"]["admission_cardinality"] == 1
        expected_profile = (
            "paper_i_ra_adapt__single_pauli_word_v1__"
            + (
                "append_commutation_reduced"
                if cell.route_id
                == ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED
                    else "insertion_commutation_plateau_v2"
            )
            + "__global_guarded_singleton_phase_i__identity_phase_ii__"
            "stationary_source_response_v1__"
            "all_phase_resource_weighting_v1"
        )
        assert route["route_profile"] == expected_profile
        template = _load(
            bundle_dir
            / "execution_templates"
            / f"{cell.cell_id}.json"
        )
        assert template["campaign_id"] == GLOBAL_SINGLETON_CAMPAIGN_ID
        assert template["run_class"] == GLOBAL_SINGLETON_RUN_CLASS
        assert template["execution_fulfillment"] == {
            "fulfillment_kind": "direct_execution_v1",
            "canonical_execution": {
                "bundle_id": GLOBAL_SINGLETON_BUNDLE_ID,
                "cell_id": cell.cell_id,
            },
        }
        assert expected["cells"][cell.cell_id][
            "execution_fulfillment"
        ] == template["execution_fulfillment"]

    bundle_module._validate_global_singleton_cross_arm_equality(
        protocols, cells
    )
    drifted = json.loads(canonical_json_bytes(protocols))
    first_pair_second = cells[1].cell_id
    drifted[first_pair_second]["optimizer_maxiter"] = 201
    with pytest.raises(
        BundleMaterializationError,
        match="differ outside the insertion policy",
    ):
        bundle_module._validate_global_singleton_cross_arm_equality(
            drifted, cells
        )

    loaded = load_validated_bundle_protocol(
        bundle_dir / "protocols" / f"{cells[0].cell_id}.json"
    )
    assert loaded.bundle_id == GLOBAL_SINGLETON_BUNDLE_ID
    assert loaded.adapter_id == GLOBAL_SINGLE_PAULI_ADAPTER_ID
    assert isinstance(
        loaded.request.adapter,
        GlobalSinglePauliWordCandidateAdapter,
    )
    assert isinstance(
        loaded.request.method.insertion,
        AppendCommutationReducedInsertion,
    )


def test_materializes_and_loads_qiskit_cost_plateau_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_global_singleton_pool_authority(monkeypatch)
    cells = build_qiskit_cost_plateau_pilot_cell_specs()
    destination = tmp_path / "qiskit-cost-pilot-bundles"
    receipt = materialize_qiskit_cost_plateau_pilot_bundle(
        destination,
        problem_resolver=_problem_resolver,
        protocol_resolver=_fake_protocol,
        source_locks=_source_locks(tmp_path, cells=cells),
        repository_state=_state(),
        repo_root=REPO_ROOT,
        environment_fingerprint={
            "python_implementation": "CPython",
            "python_version": "test",
            "platform_system": "test",
            "platform_release": "test",
            "platform_machine": "test",
        },
        materialization_timestamp="2026-07-30T12:00:00Z",
    )
    assert receipt.bundle_id == QISKIT_COST_PILOT_BUNDLE_ID
    assert receipt.cell_count == 2
    assert receipt.materialization_status == "passed"

    bundle_dir = destination / QISKIT_COST_PILOT_BUNDLE_ID
    manifest = _load(bundle_dir / "bundle_manifest.json")
    validation = _load(bundle_dir / "validation_report.json")
    source_locks = _load(bundle_dir / "source_locks.json")
    assert manifest["campaign_id"] == QISKIT_COST_PILOT_CAMPAIGN_ID
    assert manifest["study_id"] == QISKIT_COST_PILOT_CAMPAIGN_ID
    assert manifest["run_class"] == QISKIT_COST_PILOT_RUN_CLASS
    assert manifest["execution_target"] == (
        QISKIT_COST_PILOT_EXECUTION_TARGET
    )
    assert manifest["visible_target"]["target_id"] == (
        QISKIT_COST_PILOT_VISIBLE_TARGET_ID
    )
    assert manifest["active_gradient_policy"] == (
        ACTIVE_GRADIENT_STATIONARY
    )
    assert manifest["resource_weighting_scope"] == (
        RESOURCE_WEIGHTING_ALL_PHASE
    )
    contract = manifest["qiskit_cost_plateau_pilot_contract"]
    assert contract["route_ids"] == list(QISKIT_COST_PILOT_ROUTE_IDS)
    assert contract["algorithm_ids"] == [
        QISKIT_COST_PILOT_MACRO_ALGORITHM_ID,
        QISKIT_COST_PILOT_GLOBAL_SINGLETON_ALGORITHM_ID,
    ]
    assert contract["selector_compile_cost_policy"] == (
        bundle_module.RA_ADAPT_QISKIT_COST_POLICY
    )
    assert contract["backend_cost_mode"] == "transpile_single_v1"
    assert contract["backend_name"] == "FakeMarrakesh"
    assert contract["backend_optimization_level"] == 1
    assert contract["backend_transpile_seed"] == 7
    assert contract["parallel_gradient_workers"] == 4
    assert contract["horizon"] == 50
    assert {
        check["id"] for check in validation["checks"]
    } == {
        "bundle_schema_and_digest",
        "exact_qiskit_cost_plateau_pilot_matrix",
        "source_locks_exact_bytes",
        "resolved_protocol_contracts",
        "macro_pool_hash_equality",
        "singleton_pool_exposure_contracts",
        "qiskit_cost_pilot_source_derivations",
        "qiskit_cost_route_contracts",
        "all_cells_direct_execution",
        "protocol_execution_separation",
        "paper_i_run_materialization_gate",
    }
    binding = validation["qiskit_cost_pilot_validation_binding"]
    assert binding["direct_execution_cell_count"] == 2
    assert binding["execution_target"] == (
        QISKIT_COST_PILOT_EXECUTION_TARGET
    )

    for cell in cells:
        lock = source_locks["cell_locks"][cell.source_lock_id]
        anchor = lock["resolver_trace"][
            "qiskit_cost_pilot_source_anchor"
        ]
        assert anchor["target_algorithm_id"] == cell.algorithm_id
        protocol_path = (
            bundle_dir / "protocols" / f"{cell.cell_id}.json"
        )
        protocol = _load(protocol_path)
        route = protocol["route_contract"]
        assert route["route_profile"].endswith(
            "__qiskit_full_ansatz_transpile_cost_all_phases_v1"
        )
        assert route["execution_settings"][
            "phase3_backend_cost_mode"
        ] == "transpile_single_v1"
        assert route["semantic_invariants"][
            "selector_compile_cost_policy"
        ] == bundle_module.RA_ADAPT_QISKIT_COST_POLICY
        template = _load(
            bundle_dir
            / "execution_templates"
            / f"{cell.cell_id}.json"
        )
        assert template["execution_target"] == (
            QISKIT_COST_PILOT_EXECUTION_TARGET
        )
        loaded = load_validated_bundle_protocol(protocol_path)
        assert loaded.algorithm_id == cell.algorithm_id
        assert loaded.active_gradient_policy == (
            ACTIVE_GRADIENT_STATIONARY
        )
        assert loaded.resource_weighting_scope == (
            RESOURCE_WEIGHTING_ALL_PHASE
        )

    tamper_root = tmp_path / "tamper"
    tamper_root.mkdir()
    tampered = _source_locks(tamper_root, cells=cells)
    tampered_lock = tampered["cell_locks"][cells[0].source_lock_id]
    tampered_lock["resolver_trace"]["settings_changed"][0]["to"] = (
        "not_qiskit_cost"
    )
    with pytest.raises(
        BundleMaterializationError,
        match="source derivation drifted",
    ):
        materialize_qiskit_cost_plateau_pilot_bundle(
            tmp_path / "tampered-destination",
            problem_resolver=_problem_resolver,
            protocol_resolver=_fake_protocol,
            source_locks=tampered,
            repository_state=_state(),
            repo_root=REPO_ROOT,
        )


def test_materializes_and_loads_qiskit_cost_always13(
    tmp_path: Path,
) -> None:
    cells = build_qiskit_cost_always13_cell_specs()
    destination = tmp_path / "qiskit-cost-always13-bundles"
    receipt = materialize_qiskit_cost_always13_bundle(
        destination,
        problem_resolver=_problem_resolver,
        protocol_resolver=_fake_protocol,
        source_locks=_source_locks(tmp_path, cells=cells),
        repository_state=_state(),
        repo_root=REPO_ROOT,
        environment_fingerprint={
            "python_implementation": "CPython",
            "python_version": "test",
            "platform_system": "test",
            "platform_release": "test",
            "platform_machine": "test",
        },
        materialization_timestamp="2026-07-30T18:00:00Z",
    )
    assert receipt.bundle_id == QISKIT_COST_ALWAYS13_BUNDLE_ID
    assert receipt.cell_count == 1
    assert receipt.materialization_status == "passed"

    bundle_dir = destination / QISKIT_COST_ALWAYS13_BUNDLE_ID
    manifest = _load(bundle_dir / "bundle_manifest.json")
    validation = _load(bundle_dir / "validation_report.json")
    assert manifest["campaign_id"] == QISKIT_COST_ALWAYS13_CAMPAIGN_ID
    assert manifest["study_id"] == QISKIT_COST_ALWAYS13_CAMPAIGN_ID
    assert manifest["run_class"] == QISKIT_COST_ALWAYS13_RUN_CLASS
    assert manifest["execution_target"] == (
        QISKIT_COST_ALWAYS13_EXECUTION_TARGET
    )
    assert manifest["visible_target"]["target_id"] == (
        QISKIT_COST_ALWAYS13_VISIBLE_TARGET_ID
    )
    contract = manifest["qiskit_cost_always13_contract"]
    assert contract["route_ids"] == list(
        QISKIT_COST_ALWAYS13_ROUTE_IDS
    )
    assert contract["algorithm_ids"] == [
        QISKIT_COST_ALWAYS13_ALGORITHM_ID
    ]
    assert contract["horizon"] == QISKIT_COST_ALWAYS13_HORIZON
    assert contract["typed_insertion_policy"] == (
        AlwaysCommutationReducedInsertion.kind
    )
    assert contract["selector_compile_cost_policy"] == (
        bundle_module.RA_ADAPT_QISKIT_COST_POLICY
    )
    assert contract["changed_scientific_fields"] == [
        "request.method.insertion",
        "request.execution.stop.maximum_controller_rounds",
    ]
    assert {
        check["id"] for check in validation["checks"]
    } == {
        "bundle_schema_and_digest",
        "exact_qiskit_cost_always13_matrix",
        "source_locks_exact_bytes",
        "resolved_protocol_contracts",
        "macro_pool_hash_equality",
        "singleton_pool_exposure_contracts",
        "qiskit_cost_always13_source_derivation",
        "qiskit_cost_always13_route_contract",
        "all_cells_direct_execution",
        "protocol_execution_separation",
        "paper_i_run_materialization_gate",
    }

    cell = cells[0]
    protocol_path = (
        bundle_dir / "protocols" / f"{cell.cell_id}.json"
    )
    protocol = load_validated_bundle_protocol(protocol_path)
    assert protocol.algorithm_id == QISKIT_COST_ALWAYS13_ALGORITHM_ID
    assert protocol.active_gradient_policy == (
        ACTIVE_GRADIENT_STATIONARY
    )
    assert protocol.resource_weighting_scope == (
        RESOURCE_WEIGHTING_ALL_PHASE
    )
    assert protocol.horizon == QISKIT_COST_ALWAYS13_HORIZON
    assert isinstance(
        protocol.request.method.insertion,
        AlwaysCommutationReducedInsertion,
    )
    assert protocol.route_contract["execution_settings"][
        "adapt_insertion_mode"
    ] == "full_commutation_reduced"
    assert protocol.route_contract["semantic_invariants"][
        "selector_compile_cost_policy"
    ] == bundle_module.RA_ADAPT_QISKIT_COST_POLICY

    tampered_root = tmp_path / "tampered-locks"
    tampered_root.mkdir()
    tampered = _source_locks(tampered_root, cells=cells)
    target_lock = tampered["cell_locks"][cell.source_lock_id]
    target_lock["resolver_trace"]["settings_changed"][0]["to"] = (
        bundle_module.PlateauCommutationInsertion.kind
    )
    with pytest.raises(
        BundleMaterializationError,
        match="always13 source derivation drifted",
    ):
        materialize_qiskit_cost_always13_bundle(
            tmp_path / "tampered-always13-destination",
            problem_resolver=_problem_resolver,
            protocol_resolver=_fake_protocol,
            source_locks=tampered,
            repository_state=_state(),
            repo_root=REPO_ROOT,
        )


def test_materializes_and_loads_phase3_qiskit_mixed_horizon_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_global_singleton_pool_authority(monkeypatch)
    cells = build_phase3_qiskit_mixed_horizon_cell_specs()
    destination = tmp_path / "phase3-qiskit-candidate-bundles"
    receipt = materialize_phase3_qiskit_mixed_horizon_bundle(
        destination,
        problem_resolver=_problem_resolver,
        protocol_resolver=_fake_protocol,
        source_locks=_source_locks(tmp_path, cells=cells),
        repository_state=_state(),
        repo_root=REPO_ROOT,
        environment_fingerprint={
            "python_implementation": "CPython",
            "python_version": "test",
            "platform_system": "test",
            "platform_release": "test",
            "platform_machine": "test",
        },
        materialization_timestamp="2026-08-06T12:00:00Z",
    )
    assert receipt.bundle_id == PHASE3_QISKIT_BUNDLE_ID
    assert receipt.cell_count == 6
    assert receipt.materialization_status == "passed"

    bundle_dir = destination / PHASE3_QISKIT_BUNDLE_ID
    manifest = _load(bundle_dir / "bundle_manifest.json")
    validation = _load(bundle_dir / "validation_report.json")
    source_locks = _load(bundle_dir / "source_locks.json")
    assert manifest["campaign_id"] == PHASE3_QISKIT_CAMPAIGN_ID
    assert manifest["study_id"] == PHASE3_QISKIT_CAMPAIGN_ID
    assert manifest["run_class"] == PHASE3_QISKIT_RUN_CLASS
    assert manifest["execution_target"] == PHASE3_QISKIT_EXECUTION_TARGET
    assert manifest["execution_authorized"] is False
    assert manifest["submitted"] is False
    assert manifest["submission_state"] == SUBMISSION_STATE
    assert manifest["visible_target"]["target_id"] == (
        PHASE3_QISKIT_VISIBLE_TARGET_ID
    )
    contract = manifest["phase3_qiskit_mixed_horizon_contract"]
    assert contract["route_ids"] == list(PHASE3_QISKIT_ROUTE_IDS)
    assert contract["algorithm_ids"] == [
        RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
    ]
    assert contract["selector_compile_cost_policy"] == (
        RA_ADAPT_PHASE3_QISKIT_COST_POLICY
    )
    assert contract["selector_compile_cost_phase_reuse"] == (
        RA_ADAPT_PHASE3_QISKIT_COST_PHASE_REUSE
    )
    assert contract["selector_compile_cost_scope"] == (
        BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
    )
    assert contract["phase_i_phase_ii_compile_cost_source"] == (
        MARRAKESH_GRAPH_SPAN_MODE
    )
    assert contract["weak_holstein"]["horizon"] == 50
    assert contract["strong_holstein"]["horizon"] == 70
    assert contract["source_route_lineage"] == {
        "algorithm_id": (
            RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_SOURCE_ALGORITHM_ID
        ),
        "route_profile": PHASE3_QISKIT_PAGE7_PARENT_ROUTE_PROFILE,
        "route_contract_sha256": (
            PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256
        ),
    }
    assert {
        check["id"] for check in validation["checks"]
    } == {
        "bundle_schema_and_digest",
        "exact_phase3_qiskit_mixed_horizon_matrix",
        "source_locks_exact_bytes",
        "resolved_protocol_contracts",
        "macro_pool_hash_equality",
        "singleton_pool_exposure_contracts",
        "phase3_qiskit_source_derivations",
        "phase3_qiskit_route_contracts",
        "all_cells_direct_execution",
        "protocol_execution_separation",
        "paper_i_run_materialization_gate",
    }

    for cell in cells:
        lock = source_locks["cell_locks"][cell.source_lock_id]
        anchor = lock["resolver_trace"]["phase3_qiskit_source_anchor"]
        assert anchor["source_route_contract_sha256"] == (
            PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256
        )
        assert anchor["source_horizon"] == cell.horizon
        protocol_path = (
            bundle_dir / "protocols" / f"{cell.cell_id}.json"
        )
        protocol_payload = _load(protocol_path)
        route = protocol_payload["route_contract"]
        assert route["lineage_authority"]["parent_route_profile"] == (
            PHASE3_QISKIT_PAGE7_PARENT_ROUTE_PROFILE
        )
        assert route["lineage_authority"][
            "parent_contract_sha256"
        ] == PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256
        assert route["execution_settings"][
            "phase3_backend_cost_mode"
        ] == MARRAKESH_GRAPH_SPAN_MODE
        assert route["execution_settings"][
            "phase3_backend_cost_scope"
        ] == BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
        assert route["semantic_invariants"][
            "phase_iii_qiskit_backend_fallback_allowed"
        ] is False
        assert route["semantic_invariants"][
            "phase_iii_qiskit_negative_delta_reward_enabled"
        ] is False
        template = _load(
            bundle_dir
            / "execution_templates"
            / f"{cell.cell_id}.json"
        )
        assert template["execution_target"] == PHASE3_QISKIT_EXECUTION_TARGET
        assert template["execution_authorized"] is False

    loaded_weak = load_validated_bundle_protocol(
        bundle_dir / "protocols" / f"{cells[0].cell_id}.json"
    )
    loaded_strong = load_validated_bundle_protocol(
        bundle_dir / "protocols" / f"{cells[-1].cell_id}.json"
    )
    assert loaded_weak.horizon == PHASE3_QISKIT_WEAK_HOLSTEIN_HORIZON
    assert loaded_strong.horizon == PHASE3_QISKIT_STRONG_HOLSTEIN_HORIZON
    assert loaded_weak.algorithm_id == (
        RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
    )

    tamper_root = tmp_path / "phase3-qiskit-tampered-locks"
    tamper_root.mkdir()
    tampered = _source_locks(tamper_root, cells=cells)
    target_lock = tampered["cell_locks"][cells[0].source_lock_id]
    target_lock["resolver_trace"]["phase3_qiskit_source_anchor"][
        "source_route_contract_sha256"
    ] = "0" * 64
    with pytest.raises(
        BundleMaterializationError,
        match="Phase-III-Qiskit source derivation drifted",
    ):
        materialize_phase3_qiskit_mixed_horizon_bundle(
            tmp_path / "tampered-phase3-qiskit-destination",
            problem_resolver=_problem_resolver,
            protocol_resolver=_fake_protocol,
            source_locks=tampered,
            repository_state=_state(),
            repo_root=REPO_ROOT,
        )


def test_factorial_rejects_one_shared_late_source_lock_surface(
    tmp_path: Path,
) -> None:
    first_bundle_id, gradient_policy, resource_scope = (
        FACTORIAL_BUNDLE_POLICIES[0]
    )
    shared_locks = _source_locks(
        tmp_path,
        cells=build_factorial_always_cell_specs(
            active_gradient_policy=gradient_policy,
            resource_weighting_scope=resource_scope,
        ),
    )
    destination = tmp_path / "shared-lock-rejected"
    with pytest.raises(
        BundleMaterializationError,
        match="Factorial source-lock policy axes drifted",
    ):
        materialize_factorial_always_bundles(
            destination,
            problem_resolver=_problem_resolver,
            protocol_resolver=_fake_protocol,
            source_locks=shared_locks,
            repository_state=_state(),
            repo_root=REPO_ROOT,
        )
    assert first_bundle_id
    assert not destination.exists()


def test_study1_append_dedupe_keeps_20_logical_and_18_executed_cells() -> None:
    contract = study1_shared_execution_dedupe_contract()
    assert contract["schema"] == STUDY1_EXECUTION_DEDUPE_SCHEMA
    assert contract["materialized_validation_cell_count"] == 20
    assert contract["unique_validation_execution_count"] == 18
    assert contract["shared_execution_savings"] == 2
    assert len(contract["groups"]) == 2
    assert {
        group["canonical_execution"]["bundle_id"]
        for group in contract["groups"]
    } == {STATIONARY_BUNDLE_ID}
    assert {
        group["shared_result_reference"]["bundle_id"]
        for group in contract["groups"]
    } == {MEASURED_BUNDLE_ID}
    assert contract["bundle_protocol_authority"] == (
        "retained_independently_per_bundle_v1"
    )
    digest_payload = dict(contract)
    observed = digest_payload.pop("sha256")
    assert observed == canonical_sha256(digest_payload)


def test_materializes_two_matched_canonical_nonexecuting_bundles(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "bundles"
    receipts = materialize_study1_bundles(
        destination,
        problem_resolver=_problem_resolver,
        protocol_resolver=_fake_protocol,
        source_locks=_source_locks(tmp_path),
        validation_horizon=6,
        repository_state=_state(),
        repo_root=REPO_ROOT,
        environment_fingerprint={
            "python_implementation": "CPython",
            "python_version": "test",
            "platform_system": "test",
            "platform_release": "test",
            "platform_machine": "test",
        },
        materialization_timestamp="2026-07-27T12:00:00Z",
    )
    assert [receipt.bundle_id for receipt in receipts] == [
        STATIONARY_BUNDLE_ID,
        MEASURED_BUNDLE_ID,
    ]
    assert {receipt.cell_count for receipt in receipts} == {58}
    assert {receipt.materialization_status for receipt in receipts} == {
        "passed"
    }

    stationary_dir = destination / STATIONARY_BUNDLE_ID
    measured_dir = destination / MEASURED_BUNDLE_ID
    for bundle_dir, policy in (
        (stationary_dir, ACTIVE_GRADIENT_STATIONARY),
        (measured_dir, ACTIVE_GRADIENT_MEASURED),
    ):
        manifest = _load(bundle_dir / "bundle_manifest.json")
        report = _load(bundle_dir / "validation_report.json")
        expected = _load(bundle_dir / "expected_artifacts.json")
        source_locks = _load(bundle_dir / "source_locks.json")
        assert manifest["schema"] == BUNDLE_SCHEMA
        assert manifest["cell_count"] == 58
        assert manifest["validation_cell_count"] == 10
        assert manifest["full_cell_count"] == 48
        assert manifest["run_class"] == RUN_CLASS
        assert manifest["study_id"] == STUDY_ID
        assert manifest["campaign_id"] == STUDY_ID
        assert manifest["visible_target"]["target_id"] == (
            VISIBLE_TARGET_ID
        )
        assert manifest["visible_target"]["source_lock_role"] == (
            "macro_visible_provenance"
        )
        assert manifest["execution_target"] == EXECUTION_TARGET
        assert manifest["active_gradient_policy"] == policy
        assert manifest["resource_weighting_scope"] == (
            RESOURCE_WEIGHTING_LATE
        )
        assert manifest["study_2_included"] is False
        assert manifest["stationarity_winner_selected"] is False
        assert manifest["execution_authorized"] is False
        assert manifest["submission_state"] == SUBMISSION_STATE
        assert manifest["submitted"] is False
        assert manifest["ordered_cells_contract"]["pointer"] == "#/cells"
        dedupe = manifest["study1_shared_execution_dedupe"]
        assert dedupe == study1_shared_execution_dedupe_contract()
        assert dedupe["unique_validation_execution_count"] == 18
        assert [row["cell_id"] for row in manifest["cells"]] == [
            cell.cell_id
            for cell in build_study1_cell_specs(validation_horizon=6)
        ]
        assert manifest["compile_identity"] == RA_ADAPT_COMPILE_IDENTITY
        assert report["materialization_status"] == "passed"
        assert report["execution_progression_status"] == "not_run"
        assert report["execution_authorized"] is False
        assert report["submission_state"] == SUBMISSION_STATE
        assert report["submitted"] is False
        assert {
            row["id"] for row in report["objective_execution_gates"]
        } == set(bundle_module.OBJECTIVE_EXECUTION_GATE_IDS)
        for occurrence_id in (
            "g5_insertion_position_correctness_v2",
            "g6_phase3_integrity_v2",
        ):
            occurrence = next(
                row
                for row in report["objective_execution_gates"]
                if row["id"] == occurrence_id
            )
            assert occurrence["status"] == "not_run"
            assert occurrence["observed_count"] is None
            assert occurrence["required_minimum_count"] == 1
            assert occurrence["blocks_full_matrix"] is True
        gate = next(
            check
            for check in report["checks"]
            if check["id"] == "paper_i_run_materialization_gate"
        )
        assert gate["status"] == "passed"
        assert gate["observed"]["cell_count"] == 58
        assert gate["observed"]["typed_protocol_count"] == 58
        assert gate["observed"]["source_lock_count"] == 50
        assert gate["observed"]["execution_template_count"] == 58
        assert gate["observed"]["execution_target"] == EXECUTION_TARGET
        assert gate["observed"]["execution_authorized"] is False
        assert gate["observed"]["submission_state"] == SUBMISSION_STATE
        assert gate["observed"]["submitted"] is False
        assert len(expected["cells"]) == 58
        assert len(list((bundle_dir / "protocols").glob("*.json"))) == 58
        assert (
            len(list((bundle_dir / "execution_templates").glob("*.json")))
            == 58
        )
        implementation = source_locks["implementation_sources"]
        implementation_payload = dict(implementation)
        implementation_sha = implementation_payload.pop("sha256")
        assert implementation_sha == canonical_sha256(
            implementation_payload
        )
        expected_ra_roots = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in (
                REPO_ROOT / "pipelines/static_adapt/ra_adapt"
            ).glob("*.py")
        }
        assert set(implementation["root_paths"]) == expected_ra_roots
        assert implementation["root_count"] == len(expected_ra_roots)
        assert implementation["file_count"] >= implementation["root_count"]
        for row in manifest["cells"]:
            protocol_path = bundle_dir / row["protocol_path"]
            protocol = _load(protocol_path)
            template = _load(
                bundle_dir / row["execution_template_path"]
            )
            assert protocol["active_gradient_policy"] == policy
            assert protocol["resource_weighting_scope"] == (
                RESOURCE_WEIGHTING_LATE
            )
            assert protocol["execution_authorized"] is False
            assert protocol["optimizer"] == "powell"
            assert protocol["optimizer_maxiter"] == 200
            assert protocol["seeds"] == {"adapt": 7, "transpiler": 7}
            assert protocol["problem"]["reference_label"] == (
                "hubbard_holstein_reference_state"
            )
            assert protocol["request"]["kind"] == (
                "append_adapt_request"
                if row["selector_family"] == "append_adapt"
                else "ra_adapt_request"
            )
            if row["selector_family"] == "ra_adapt":
                assert all(
                    protocol["request"]["method"][name]["kind"]
                    for name in ("admission", "insertion", "pruning", "beam")
                )
            for receipt_name in (
                "parent_inventory",
                "executable_pool",
                "route_contract",
                "baseline_consumption",
            ):
                receipt = dict(protocol[receipt_name])
                observed = receipt.pop("sha256")
                assert observed == canonical_sha256(receipt)
            baseline = protocol["baseline_consumption"]
            assert baseline["status"] == "passed"
            assert baseline["unconsumed_declared_field_paths"] == []
            assert baseline["unapproved_change_ids"] == []
            assert baseline["consumed_source_field_paths"]
            assert (
                load_resolved_ra_adapt_protocol(protocol_path).to_dict()
                == protocol
            )
            assert protocol["bundle_materialization"]["bundle_id"] == (
                manifest["bundle_id"]
            )
            materialization = dict(
                protocol["bundle_materialization"]
            )
            observed_materialization_sha = materialization.pop("sha256")
            assert observed_materialization_sha == canonical_sha256(
                materialization
            )
            lock = source_locks["cell_locks"][row["source_lock_id"]]
            assert lock["resolver_trace"]["settings_reused"]
            assert lock["resolver_trace"][
                "normalized_protocol_used_field_paths"
            ]
            assert lock["resolver_trace"]["protocol_used_field_audit"] == (
                "declared_pending_protocol_resolution"
            )
            same_cutoff_ed = lock["resolver_trace"][
                "same_cutoff_ed_reference"
            ]
            assert same_cutoff_ed["nph"] == row["nph"]
            assert same_cutoff_ed["required"] is True
            assert same_cutoff_ed["sha256"] == (
                source_locks["global_sources"]["ed_cutoff_reference"][
                    "sha256"
                ]
            )
            assert template["schema"] == EXECUTION_TEMPLATE_SCHEMA
            assert template["run_class"] == RUN_CLASS
            assert template["study_id"] == STUDY_ID
            assert template["campaign_id"] == STUDY_ID
            assert template["execution_state"] == "not_started"
            assert template["execution_target"] == EXECUTION_TARGET
            assert template["execution_entrypoint"] == (
                "pipelines.static_adapt.ra_adapt.run_append_adapt"
                if row["selector_family"] == "append_adapt"
                else "pipelines.static_adapt.ra_adapt.run_ra_adapt"
            )
            assert template["execution_authorized"] is False
            assert template["submission_state"] == SUBMISSION_STATE
            assert template["submitted"] is False
            assert tuple(
                template["expected_artifact_contract"][
                    "required_roles"
                ]
            ) == EXPECTED_ARTIFACT_ROLES
            assert set(
                expected["cells"][row["cell_id"]][
                    "expected_run_artifacts"
                ]
            ) == set(EXPECTED_ARTIFACT_ROLES)
            assert template["command_argv"] is None
            assert template["cwd"] is None
            assert template["git_commit"] is None
            assert template["dirty_working_tree"] is None
            assert template["environment_fingerprint"] is None
            assert template["working_directory_policy"] == "bundle_root_v1"
            fulfillment = template["execution_fulfillment"]
            assert fulfillment == expected["cells"][row["cell_id"]][
                "execution_fulfillment"
            ]
            if (
                row["stage"] == "validation"
                and row["route_id"] == bundle_module.ROUTE_APPEND_MACRO
            ):
                assert fulfillment["fulfillment_kind"] == (
                    "canonical_shared_execution_v1"
                    if policy == ACTIVE_GRADIENT_STATIONARY
                    else "shared_result_reference_v1"
                )
            else:
                assert fulfillment["fulfillment_kind"] == (
                    "direct_execution_v1"
                )
            reference_fulfilled = (
                fulfillment["fulfillment_kind"]
                == "shared_result_reference_v1"
            )
            for artifact in expected["cells"][row["cell_id"]][
                "expected_run_artifacts"
            ].values():
                assert artifact["required"] is True
                assert artifact["fulfillment_kind"] == (
                    fulfillment["fulfillment_kind"]
                )
                assert artifact["direct_file_required"] is (
                    not reference_fulfilled
                )
                assert artifact["reference_receipt_required"] is (
                    reference_fulfilled
                )
            assert template["timestamps"] == {
                "started_at": None,
                "finished_at": None,
            }
            assert template["exit_status"] is None
            assert template["dependency_lock_status"] == "verified"
            observation = protocol["request"]["observation"]
            for receipt_name in ("checkpoint", "estimator_ledger"):
                observed_path = Path(observation[receipt_name]["path"])
                assert not observed_path.is_absolute()
                assert observed_path.as_posix().startswith(
                    f"runs/{row['cell_id']}/"
                )
            if row.get("preservation_contract_id") is not None:
                preservation_gate = row["preservation_execution_gate"]
                assert preservation_gate == expected["cells"][
                    row["cell_id"]
                ]["preservation_execution_gate"]
                assert preservation_gate["active_gradient_policy"] == policy
                requirements = preservation_gate["requirements"]
                assert requirements[
                    "same_problem_deterministic_replay_required"
                ] is True
                assert requirements[
                    "paired_policy_comparison_required"
                ] is True
                assert requirements[
                    "trajectory_deviation_is_pass_condition"
                ] is False
                assert requirements[
                    "zero_active_gradient_indices_required"
                ] is (policy == ACTIVE_GRADIENT_STATIONARY)
                assert preservation_gate[
                    "generic_route_characterization"
                ]["study1_numerical_baseline"] is False

        first_protocol_path = next(
            (bundle_dir / "protocols").glob("*.json")
        )
        raw_protocol = load_resolved_ra_adapt_protocol(
            first_protocol_path
        )
        assert raw_protocol._materialization_authority is None
        validated_protocol = load_validated_bundle_protocol(
            first_protocol_path
        )
        assert validated_protocol.to_dict() == raw_protocol.to_dict()
        assert (
            validated_protocol._materialization_authority.protocol_sha256
            == validated_protocol.sha256
        )

        original_protocol_bytes = first_protocol_path.read_bytes()
        drifted_protocol = _load(first_protocol_path)
        drifted_protocol["horizon"] += 1
        drifted_protocol.pop("sha256")
        drifted_protocol["sha256"] = canonical_sha256(drifted_protocol)
        first_protocol_path.write_bytes(
            canonical_json_bytes(drifted_protocol) + b"\n"
        )
        with pytest.raises(
            BundleMaterializationError,
            match="Expected-artifact index",
        ):
            load_validated_bundle_protocol(first_protocol_path)
        first_protocol_path.write_bytes(original_protocol_bytes)

        manifest_path = bundle_dir / "bundle_manifest.json"
        original_manifest_bytes = manifest_path.read_bytes()
        drifted_manifest = _load(manifest_path)
        drifted_manifest["materialization_timestamp"] = (
            "2026-07-27T12:00:01Z"
        )
        drifted_manifest.pop("sha256")
        drifted_manifest["sha256"] = canonical_sha256(drifted_manifest)
        manifest_path.write_bytes(
            canonical_json_bytes(drifted_manifest) + b"\n"
        )
        with pytest.raises(
            BundleMaterializationError,
            match="Protocol field drift.*bundle_manifest_sha256",
        ):
            load_validated_bundle_protocol(first_protocol_path)
        manifest_path.write_bytes(original_manifest_bytes)

        source_locks_path = bundle_dir / "source_locks.json"
        original_source_locks_bytes = source_locks_path.read_bytes()
        drifted_source_locks = _load(source_locks_path)
        drifted_source_locks["tamper_probe"] = True
        drifted_source_locks.pop("sha256")
        drifted_source_locks["sha256"] = canonical_sha256(
            drifted_source_locks
        )
        source_locks_path.write_bytes(
            canonical_json_bytes(drifted_source_locks) + b"\n"
        )
        with pytest.raises(
            BundleMaterializationError,
            match="not bound to the source-lock manifest",
        ):
            load_validated_bundle_protocol(first_protocol_path)
        source_locks_path.write_bytes(original_source_locks_bytes)

        expected_path = bundle_dir / "expected_artifacts.json"
        original_expected_bytes = expected_path.read_bytes()
        drifted_expected = _load(expected_path)
        drifted_expected["stale_report_probe"] = True
        drifted_expected.pop("sha256")
        drifted_expected["sha256"] = canonical_sha256(drifted_expected)
        expected_path.write_bytes(
            canonical_json_bytes(drifted_expected) + b"\n"
        )
        with pytest.raises(
            BundleMaterializationError,
            match="Stale bundle validation report.*cross-file binding",
        ):
            load_validated_bundle_protocol(first_protocol_path)
        expected_path.write_bytes(original_expected_bytes)

        tampered = _load(
            next((bundle_dir / "protocols").glob("*.json"))
        )
        tampered["route_contract"]["semantic_invariants"][
            "candidate_geometry_chart"
        ] = "drifted"
        tampered.pop("sha256")
        tampered["sha256"] = canonical_sha256(tampered)
        with pytest.raises(ValueError, match="route_contract digest"):
            resolved_ra_adapt_protocol_from_mapping(tampered)

        for name in (
            "bundle_manifest.json",
            "source_locks.json",
            "expected_artifacts.json",
            "validation_report.json",
        ):
            _assert_canonical_digested(bundle_dir / name)
        _assert_canonical_digested(
            next((bundle_dir / "protocols").glob("*.json"))
        )
        _assert_canonical_digested(
            next((bundle_dir / "execution_templates").glob("*.json"))
        )

    stationary_manifest = _load(
        stationary_dir / "bundle_manifest.json"
    )
    measured_manifest = _load(measured_dir / "bundle_manifest.json")
    for payload in (stationary_manifest, measured_manifest):
        payload.pop("sha256")
        payload.pop("bundle_id")
        payload.pop("active_gradient_policy")
        for row in payload["cells"]:
            row.pop("preservation_execution_gate", None)
    assert stationary_manifest == measured_manifest


def test_missing_validation_horizon_is_serialized_as_blocked(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "blocked"
    receipts = materialize_study1_bundles(
        destination,
        problem_resolver=_problem_resolver,
        protocol_resolver=_fake_protocol,
        source_locks=_source_locks(tmp_path),
        validation_horizon=None,
        repository_state=_state(),
        repo_root=REPO_ROOT,
        environment_fingerprint={"python_version": "test"},
    )
    assert {receipt.materialization_status for receipt in receipts} == {
        "blocked"
    }
    for receipt in receipts:
        report = _load(receipt.bundle_path / "validation_report.json")
        assert report["materialization_status"] == "blocked"
        assert report["submission_state"] == SUBMISSION_STATE
        assert report["submitted"] is False
        gate = next(
            check
            for check in report["checks"]
            if check["id"] == "paper_i_run_materialization_gate"
        )
        assert gate["status"] == "blocked"
        protocols = [
            _load(path)
            for path in (receipt.bundle_path / "protocols").glob("*.json")
        ]
        blocked = [
            protocol
            for protocol in protocols
            if protocol["schema"] == BLOCKED_PROTOCOL_SCHEMA
        ]
        assert len(blocked) == 10
        assert {
            protocol["blocking_reason"] for protocol in blocked
        } == {"validation_horizon_not_supplied_or_validated"}
        assert all(
            protocol["execution_authorized"] is False
            for protocol in blocked
        )
        assert all(
            protocol["submission_state"] == SUBMISSION_STATE
            and protocol["submitted"] is False
            for protocol in blocked
        )


def test_source_archive_or_member_drift_fails_closed_before_writing(
    tmp_path: Path,
) -> None:
    source_locks = _source_locks(tmp_path)
    lock = next(iter(source_locks["cell_locks"].values()))
    lock["member"]["sha256"] = "0" * 64
    destination = tmp_path / "rejected"
    with pytest.raises(
        BundleMaterializationError,
        match="archive member SHA-256 drift",
    ):
        materialize_study1_bundles(
            destination,
            problem_resolver=_problem_resolver,
            protocol_resolver=_fake_protocol,
            source_locks=source_locks,
            validation_horizon=6,
            repository_state=_state(),
            repo_root=REPO_ROOT,
        )
    assert not destination.exists()


def test_unconsumed_or_drifted_baseline_field_fails_closed(
    tmp_path: Path,
) -> None:
    source_locks = _source_locks(tmp_path)
    for lock in source_locks["cell_locks"].values():
        lock["resolver_trace"]["settings_reused"]["settings"][
            "optimizer"
        ] = "SPSA"
    destination = tmp_path / "baseline-drift"
    with pytest.raises(
        BundleMaterializationError,
        match="Source baseline field drifted",
    ):
        materialize_study1_bundles(
            destination,
            problem_resolver=_problem_resolver,
            protocol_resolver=_fake_protocol,
            source_locks=source_locks,
            validation_horizon=6,
            repository_state=_state(),
            repo_root=REPO_ROOT,
        )
    assert not destination.exists()


def test_materializer_refuses_to_replace_historical_bundle_directory(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "bundles"
    (destination / STATIONARY_BUNDLE_ID).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        materialize_study1_bundles(
            destination,
            problem_resolver=_problem_resolver,
            protocol_resolver=_fake_protocol,
            source_locks=_source_locks(tmp_path),
            validation_horizon=6,
            repository_state=_state(),
            repo_root=REPO_ROOT,
        )
