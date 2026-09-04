"""Accepted-lineage projection for staged H2O sector-complete Pauli blocks.

The Paper-IV ``sector_complete_pauli_block_v1`` arm stages its accepted
candidates through ``expose_children``: the executable pool is the retained
derivative-resolved macro parent supply, so an accepted block is *absent*
from that pool by construction.  Before the fix the lineage projector treated
absence as tampering unless the protocol-level representation was
``single_pauli_word_v1``, and the arm exited 1 after its first accepted round
with "Accepted RA candidate is absent from the authenticated executable
inventory".

These tests pin the replacement contract:

* a staged sector block is recognised by what the row *is* -- the adapter's
  declared sector-block Phase-II exposure capability plus the row's own
  identity/pool schemas -- and is projected through its manifest;
* a macro row that is genuinely absent still raises;
* the recognition is two-keyed, so a row cannot claim sector-block lineage on
  a route whose adapter does not stage sector blocks;
* the Paper-I ``single_pauli_word_v1`` predicate is unchanged.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

import pytest

from pipelines.static_adapt.ra_adapt.adapters import (
    H2O_LINEAR_FD_SECTOR_COMPLETE_PAULI_BLOCK_ADAPTER_ID,
    H2O_LINEAR_FD_SYMMETRY_COMPLETE_ADAPTER_ID,
    PHASE_II_EXPOSURE_RETAINED_MACRO_IDENTITY,
    PHASE_II_EXPOSURE_RETAINED_PARENT_CHILDREN,
    PHASE_II_EXPOSURE_RETAINED_PARENT_SECTOR_BLOCKS,
    H2OLinearFDSectorCompletePauliBlockCandidateAdapter,
    H2OLinearFDSymmetryCompleteCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    CANDIDATE_REPRESENTATION_MACRO,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    canonical_sha256,
)
from pipelines.static_adapt.ra_adapt.engine import (
    _accepted_candidate_lineage_receipts,
    _is_staged_sector_complete_pauli_block_row,
)
from pipelines.static_adapt.ra_adapt.pools import (
    H2O_SECTOR_COMPLETE_PAULI_BLOCK_CONSTRUCTION,
    H2O_SECTOR_COMPLETE_PAULI_BLOCK_GROUPING_POLICY,
    H2O_SECTOR_COMPLETE_PAULI_BLOCK_IDENTITY_SCHEMA,
    H2O_SECTOR_COMPLETE_PAULI_BLOCK_POOL_SCHEMA,
    CandidateInventory,
    CandidateRecord,
    _receipt,
    sector_complete_pauli_block_identity,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


# The eight serialized terms of the block accepted in round 1 of
# ``h2o_application --candidate-representation sector_complete_pauli_block_v1
# --insertion append_only`` on the July fixture
# (chtc/paper_iv_h2o_sr_paper_i_noprune_nobeam_derivative_resolved_d50
# _20260724).
# They pin the production identity digest, not just a synthetic one.
_PRODUCTION_BLOCK_TERMS: tuple[dict[str, Any], ...] = (
    {"coeff_im": 0.0, "coeff_re": -0.125, "nq": 15,
     "pauli_exyz": "eeeyzzzyexzzzye"},
    {"coeff_im": 0.0, "coeff_re": -0.125, "nq": 15,
     "pauli_exyz": "eeexzzzyeyzzzye"},
    {"coeff_im": 0.0, "coeff_re": 0.125, "nq": 15,
     "pauli_exyz": "eeeyzzzxeyzzzye"},
    {"coeff_im": 0.0, "coeff_re": -0.125, "nq": 15,
     "pauli_exyz": "eeexzzzxexzzzye"},
    {"coeff_im": 0.0, "coeff_re": 0.125, "nq": 15,
     "pauli_exyz": "eeeyzzzyeyzzzxe"},
    {"coeff_im": 0.0, "coeff_re": -0.125, "nq": 15,
     "pauli_exyz": "eeexzzzyexzzzxe"},
    {"coeff_im": 0.0, "coeff_re": 0.125, "nq": 15,
     "pauli_exyz": "eeeyzzzxexzzzxe"},
    {"coeff_im": 0.0, "coeff_re": 0.125, "nq": 15,
     "pauli_exyz": "eeexzzzxeyzzzxe"},
)
_PRODUCTION_BLOCK_LABEL = "h2o_sector_block::5866a76984e45e4c"
_PRODUCTION_BLOCK_IDENTITY = "h2o-sector-block:5866a76984e45e4c"
_PARENT_LABEL = "el::uccsd_dbl(ab:1,7->5,11)"
_PARENT_IDENTITY = "parent:64a783c65a2462b5"
_FAMILY = "full_meta_derivative_resolved_v2"


def _term(label: str, rows: tuple[Mapping[str, Any], ...]) -> AnsatzTerm:
    return AnsatzTerm(
        label=label,
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(
                    int(row["nq"]),
                    pc=complex(
                        float(row["coeff_re"]), float(row["coeff_im"])
                    ),
                    ps=str(row["pauli_exyz"]),
                )
                for row in rows
            ],
        ),
        execution_mode="grouped_exact",
    )


def _macro_parent(label: str, identity: str) -> CandidateRecord:
    rows = ({"coeff_im": 0.0, "coeff_re": 1.0, "nq": 15,
             "pauli_exyz": "eeeyzzzyexzzzye"},)
    return CandidateRecord(
        label=label,
        term=_term(label, rows),
        representation_id=CANDIDATE_REPRESENTATION_MACRO,
        generator_identity=identity,
        parent_identities=(),
        family_id=_FAMILY,
        stage_family=_FAMILY,
        construction="h2o_derivative_resolved_symmetry_complete_v1",
        execution_mode="grouped_exact",
        serialized_terms_exyz=rows,
        symmetry_receipt={"grouped_preserves_fixed_counts": True},
        generator_metadata={"generator_id": identity},
    )


@pytest.fixture()
def executable_inventory() -> CandidateInventory:
    """Stand in for the H2O derivative-resolved macro parent supply."""

    parents = (
        _macro_parent(_PARENT_LABEL, _PARENT_IDENTITY),
        _macro_parent("el::uccsd_sing(alpha:0->4)", "parent:c99b9e1eb2ac7ca1"),
    )
    return CandidateInventory(
        candidates=parents,
        receipt=_receipt(
            schema="paper_iv_h2o_ra_adapt_symmetry_complete_generator_pool_v1",
            representation_id=CANDIDATE_REPRESENTATION_MACRO,
            candidates=parents,
        ),
        metadata={},
    )


def _sector_block_record() -> CandidateRecord:
    """Rebuild the accepted block exactly as the sector pool emits it."""

    label, identity = sector_complete_pauli_block_identity(
        serialized_terms_exyz=_PRODUCTION_BLOCK_TERMS,
    )
    return CandidateRecord(
        label=label,
        term=_term(label, _PRODUCTION_BLOCK_TERMS),
        representation_id=CANDIDATE_REPRESENTATION_MACRO,
        generator_identity=identity,
        parent_identities=(_PARENT_IDENTITY,),
        family_id=_FAMILY,
        stage_family=_FAMILY,
        construction=H2O_SECTOR_COMPLETE_PAULI_BLOCK_CONSTRUCTION,
        execution_mode="grouped_exact",
        serialized_terms_exyz=_PRODUCTION_BLOCK_TERMS,
        symmetry_receipt={
            "all_nonzero_components_mutually_commute": True,
            "component_count": len(_PRODUCTION_BLOCK_TERMS),
            "components_individually_preserve_fixed_counts": False,
            "execution_mode": "grouped_exact",
            "execution_preserves_fixed_counts": True,
            "grouped_commutator_l1": {"n_dn": 0.0, "n_up": 0.0},
            "grouped_preserves_fixed_counts": True,
            "label": label,
            "nonzero_component_count": len(_PRODUCTION_BLOCK_TERMS),
            "requires_logical_shared_parameterization": True,
        },
        generator_metadata={},
    )


def _accepted_row() -> dict[str, Any]:
    """One accepted round carrying a staged sector-block feature row."""

    record = _sector_block_record()
    manifest = record.manifest_row()
    metadata = {
        "generator_id": str(record.generator_identity),
        "generator_identity_schema": (
            H2O_SECTOR_COMPLETE_PAULI_BLOCK_IDENTITY_SCHEMA
        ),
        "parent_generator_ids": list(record.parent_identities),
        "parent_labels": [_PARENT_LABEL],
        "sector_complete_pauli_block": {
            "schema": H2O_SECTOR_COMPLETE_PAULI_BLOCK_POOL_SCHEMA,
            "grouping_policy": (
                H2O_SECTOR_COMPLETE_PAULI_BLOCK_GROUPING_POLICY
            ),
            "component_count": len(_PRODUCTION_BLOCK_TERMS),
            "individually_sector_preserving": False,
            "raw_single_pauli_child": False,
        },
        "ra_adapter_id": (
            H2O_LINEAR_FD_SECTOR_COMPLETE_PAULI_BLOCK_ADAPTER_ID
        ),
        "ra_candidate_representation": CANDIDATE_REPRESENTATION_MACRO,
        "ra_parent_generator_ids": list(record.parent_identities),
        "ra_candidate_manifest": manifest,
        "ra_candidate_manifest_sha256": canonical_sha256(dict(manifest)),
    }
    return {
        "selected_logical_size": 1,
        "selected_positions": [0],
        "selected_feature_rows": [
            {
                "candidate_label": str(record.label),
                "candidate_family": _FAMILY,
                "stage_name": "core",
                "position_id": 0,
                "generator_metadata": metadata,
            }
        ],
    }


def _project(row: Mapping[str, Any], inventory: CandidateInventory, **kw):
    defaults = {
        "candidate_representation": CANDIDATE_REPRESENTATION_MACRO,
        "executable_inventory": inventory,
        "adapter_id": H2O_LINEAR_FD_SECTOR_COMPLETE_PAULI_BLOCK_ADAPTER_ID,
        "staged_child_exposure_id": (
            PHASE_II_EXPOSURE_RETAINED_PARENT_SECTOR_BLOCKS
        ),
    }
    defaults.update(kw)
    return _accepted_candidate_lineage_receipts(row, **defaults)


def test_sector_block_identity_reproduces_the_production_block() -> None:
    """The block label and generator id are a digest of its own terms."""

    assert sector_complete_pauli_block_identity(
        serialized_terms_exyz=_PRODUCTION_BLOCK_TERMS,
    ) == (_PRODUCTION_BLOCK_LABEL, _PRODUCTION_BLOCK_IDENTITY)


def test_adapter_exposure_capability_identifies_the_sector_arm() -> None:
    """Only the sector adapter declares the sector-block child exposure."""

    sector = H2OLinearFDSectorCompletePauliBlockCandidateAdapter()
    macro = H2OLinearFDSymmetryCompleteCandidateAdapter()
    assert sector.phase_ii_candidate_exposure_id == (
        PHASE_II_EXPOSURE_RETAINED_PARENT_SECTOR_BLOCKS
    )
    assert macro.phase_ii_candidate_exposure_id == (
        PHASE_II_EXPOSURE_RETAINED_MACRO_IDENTITY
    )
    assert sector.adapter_id == (
        H2O_LINEAR_FD_SECTOR_COMPLETE_PAULI_BLOCK_ADAPTER_ID
    )


def test_staged_sector_block_projects_through_its_manifest(
    executable_inventory: CandidateInventory,
) -> None:
    """The regression: an accepted sector block no longer fails closed."""

    receipts = _project(_accepted_row(), executable_inventory)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.candidate_label == _PRODUCTION_BLOCK_LABEL
    assert receipt.generator_identity == _PRODUCTION_BLOCK_IDENTITY
    assert receipt.representation_id == CANDIDATE_REPRESENTATION_MACRO
    assert receipt.parent_identities == (_PARENT_IDENTITY,)
    assert receipt.insertion_position == 0
    assert len(str(receipt.candidate_manifest_sha256)) == 64


def test_absent_macro_row_without_sector_receipts_still_raises(
    executable_inventory: CandidateInventory,
) -> None:
    """A macro row genuinely missing from the pool is still tampering."""

    row = _accepted_row()
    metadata = row["selected_feature_rows"][0]["generator_metadata"]
    for key in (
        "generator_identity_schema",
        "sector_complete_pauli_block",
    ):
        stripped = copy.deepcopy(row)
        stripped["selected_feature_rows"][0]["generator_metadata"].pop(key)
        with pytest.raises(
            RuntimeError,
            match="absent from the authenticated executable inventory",
        ):
            _project(stripped, executable_inventory)

    foreign = copy.deepcopy(row)
    foreign["selected_feature_rows"][0]["generator_metadata"][
        "ra_adapter_id"
    ] = H2O_LINEAR_FD_SYMMETRY_COMPLETE_ADAPTER_ID
    with pytest.raises(
        RuntimeError,
        match="absent from the authenticated executable inventory",
    ):
        _project(foreign, executable_inventory)
    assert metadata["ra_adapter_id"] == (
        H2O_LINEAR_FD_SECTOR_COMPLETE_PAULI_BLOCK_ADAPTER_ID
    )


def test_sector_lineage_requires_the_declared_adapter_capability(
    executable_inventory: CandidateInventory,
) -> None:
    """Recognition is two-keyed: the executing adapter must stage blocks."""

    row = _accepted_row()
    for override in (
        {"staged_child_exposure_id": None},
        {
            "staged_child_exposure_id": (
                PHASE_II_EXPOSURE_RETAINED_MACRO_IDENTITY
            )
        },
        {"adapter_id": H2O_LINEAR_FD_SYMMETRY_COMPLETE_ADAPTER_ID},
        {"adapter_id": None},
    ):
        with pytest.raises(
            RuntimeError,
            match="absent from the authenticated executable inventory",
        ):
            _project(row, executable_inventory, **override)


def test_sector_block_terms_are_bound_to_the_accepted_identity(
    executable_inventory: CandidateInventory,
) -> None:
    """Tampering with the block's terms breaks its intrinsic digest."""

    tampered = json.loads(json.dumps(_accepted_row()))
    tampered["selected_feature_rows"][0]["generator_metadata"][
        "ra_candidate_manifest"
    ]["serialized_terms_exyz"][0]["coeff_re"] = 9.0
    with pytest.raises(
        RuntimeError,
        match="intrinsic identity or construction binding",
    ):
        _project(tampered, executable_inventory)


def test_sector_block_ancestry_is_bound_to_the_parent_inventory(
    executable_inventory: CandidateInventory,
) -> None:
    """An unauthenticated retained parent is rejected."""

    unknown_label = json.loads(json.dumps(_accepted_row()))
    unknown_label["selected_feature_rows"][0]["generator_metadata"][
        "parent_labels"
    ] = ["el::not_in_the_pool"]
    with pytest.raises(
        RuntimeError,
        match="unauthenticated parent label",
    ):
        _project(unknown_label, executable_inventory)

    mismatched = json.loads(json.dumps(_accepted_row()))
    metadata = mismatched["selected_feature_rows"][0]["generator_metadata"]
    metadata["parent_labels"] = ["el::uccsd_sing(alpha:0->4)"]
    with pytest.raises(
        RuntimeError,
        match="unauthenticated parent label",
    ):
        _project(mismatched, executable_inventory)

    drifted_pool_parents = json.loads(json.dumps(_accepted_row()))
    drifted_pool_parents["selected_feature_rows"][0]["generator_metadata"][
        "parent_generator_ids"
    ] = ["parent:c99b9e1eb2ac7ca1"]
    with pytest.raises(
        RuntimeError,
        match="not bound to the authenticated parent inventory",
    ):
        _project(drifted_pool_parents, executable_inventory)


def test_sector_block_hard_guard_receipt_must_pass(
    executable_inventory: CandidateInventory,
) -> None:
    """A block whose fixed-sector audit did not pass is rejected."""

    for field in (
        "grouped_preserves_fixed_counts",
        "execution_preserves_fixed_counts",
    ):
        row = json.loads(json.dumps(_accepted_row()))
        metadata = row["selected_feature_rows"][0]["generator_metadata"]
        metadata["ra_candidate_manifest"]["symmetry_receipt"][field] = False
        metadata["ra_candidate_manifest_sha256"] = canonical_sha256(
            dict(metadata["ra_candidate_manifest"])
        )
        with pytest.raises(
            RuntimeError,
            match="fixed-sector hard-guard binding",
        ):
            _project(row, executable_inventory)


def test_sector_block_grouping_receipt_must_match_the_block(
    executable_inventory: CandidateInventory,
) -> None:
    """The pool's grouping receipt is bound to the executed component set."""

    row = json.loads(json.dumps(_accepted_row()))
    row["selected_feature_rows"][0]["generator_metadata"][
        "sector_complete_pauli_block"
    ]["component_count"] = 1
    with pytest.raises(
        RuntimeError,
        match="grouping receipt is inconsistent",
    ):
        _project(row, executable_inventory)


def test_single_pauli_recognition_is_unchanged(
    executable_inventory: CandidateInventory,
) -> None:
    """Paper-I rows never enter the sector-block arm, whatever is passed."""

    single_pauli_metadata = {
        "generator_id": "gen:517e8dfb60b9efce",
        "ra_adapter_id": (
            "paper_i_ra_adapt_single_pauli_word_candidate_adapter_v1"
        ),
        "ra_candidate_representation": CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        "shared_pauli_pool_contract": {
            "mode": "guarded_singleton_children_only_v1",
        },
    }
    for adapter_id, exposure in (
        (None, None),
        (
            "paper_i_ra_adapt_single_pauli_word_candidate_adapter_v1",
            PHASE_II_EXPOSURE_RETAINED_PARENT_CHILDREN,
        ),
        (
            H2O_LINEAR_FD_SECTOR_COMPLETE_PAULI_BLOCK_ADAPTER_ID,
            PHASE_II_EXPOSURE_RETAINED_PARENT_SECTOR_BLOCKS,
        ),
    ):
        assert not _is_staged_sector_complete_pauli_block_row(
            single_pauli_metadata,
            adapter_id=adapter_id,
            staged_child_exposure_id=exposure,
        )
