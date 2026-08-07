#!/usr/bin/env python3
"""Provenance catalog for external ADAPT competitor references.

The catalog is benchmark-local metadata.  It records which public/reference
implementations can anchor future competitor rows, but importing this module does
not import, clone, or execute any third-party package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Availability = Literal["public_git", "library_package", "request_only", "conceptual"]
ReferenceTier = Literal["author_code", "library_reference", "third_party_reference", "request_only"]
AdapterStatus = Literal[
    "cataloged_no_adapter",
    "adapter_scaffold_only",
    "ceo_and_tetris_hubbard_L2_adapters",
    "ceo_first_slice_tetris_hubbard_L2_parameterized_cases",
    "request_required",
    "library_adapter_exists_elsewhere",
]

CEO_ADAPT_VQE_PINNED_COMMIT = "712f6dd3bc56e9e3f5a10b5f46ad6194c9f6ac63"


@dataclass(frozen=True)
class ExternalReferenceSpec:
    """Reference implementation/provenance metadata for a competitor method."""

    reference_id: str
    display_name: str
    availability: Availability
    reference_tier: ReferenceTier
    url: str
    intended_algorithm_ids: tuple[str, ...]
    adapter_status: AdapterStatus
    clone_url: str | None = None
    default_ref: str | None = None
    pinned_commit: str | None = None
    package_imports: tuple[str, ...] = ()
    license_note: str = "verify from pinned checkout before using benchmark data in a manuscript"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Existing registry IDs for external ADAPT rows.  These rows are intentionally
# kept separate from the project Phase3 controller and remain skipped until a
# conformance-tested adapter is wired.
EXTERNAL_ADAPT_ALGORITHM_IDS = frozenset(
    {
        "static_ceo_adapt_phase3",
        "static_tetris_adapt_phase3",
        "static_overlap_adapt_phase3",
    }
)

_EXTERNAL_ADAPT_ALGORITHM_ADAPTER_STATUS = {
    "static_ceo_adapt_phase3": "implemented_hubbard_L2_public_code_first_slice",
    "static_tetris_adapt_phase3": "implemented_hubbard_L2_tetris_public_code_parameterized_cases",
    "static_overlap_adapt_phase3": "request_required",
}

_REFERENCE_CATALOG: tuple[ExternalReferenceSpec, ...] = (
    ExternalReferenceSpec(
        reference_id="ceo_adapt_vqe",
        display_name="CEO-ADAPT-VQE / resource-reduced ADAPT reference implementation",
        availability="public_git",
        reference_tier="author_code",
        url="https://github.com/mafaldaramoa/ceo-adapt-vqe",
        clone_url="https://github.com/mafaldaramoa/ceo-adapt-vqe.git",
        default_ref=CEO_ADAPT_VQE_PINNED_COMMIT,
        pinned_commit=CEO_ADAPT_VQE_PINNED_COMMIT,
        package_imports=("adaptvqe", "openfermion", "openfermionpyscf", "pyscf", "qiskit", "quimb", "scipy"),
        intended_algorithm_ids=("static_ceo_adapt_phase3", "static_tetris_adapt_phase3"),
        adapter_status="ceo_first_slice_tetris_hubbard_L2_parameterized_cases",
        notes=(
            "Public reference surface for CEO-style operators and related ADAPT "
            "resource-reduction subroutines.  Treat TETRIS support here as a "
            "reference implementation of TETRIS-style batching, not necessarily "
            "the original TETRIS authors' code."
        ),
    ),
    ExternalReferenceSpec(
        reference_id="hrgrimsl_adapt",
        display_name="Original fermionic ADAPT-VQE reference implementation",
        availability="public_git",
        reference_tier="author_code",
        url="https://github.com/hrgrimsl/adapt",
        clone_url="https://github.com/hrgrimsl/adapt.git",
        default_ref="master",
        intended_algorithm_ids=(),
        adapter_status="cataloged_no_adapter",
        notes=(
            "Useful as a conformance oracle for gradient selection, parameter "
            "recycling, and convergence criteria before implementing original "
            "ADAPT rows in the common HH benchmark harness."
        ),
    ),
    ExternalReferenceSpec(
        reference_id="jordanovsj_vqe",
        display_name="Qubit-excitation ADAPT reference code surface",
        availability="public_git",
        reference_tier="author_code",
        url="https://github.com/JordanovSJ/VQE",
        clone_url="https://github.com/JordanovSJ/VQE.git",
        default_ref="master",
        intended_algorithm_ids=("static_qeb_sq_lf_adapt",),
        adapter_status="cataloged_no_adapter",
        notes=(
            "Reference source for QEB/qubit-excitation mechanics.  The current "
            "HH SQ-LF row is repo-native; this catalog entry is for future "
            "conformance checks, not an automatic replacement."
        ),
    ),
    ExternalReferenceSpec(
        reference_id="openvqe",
        display_name="OpenVQE ADAPT and Qubit-ADAPT library reference",
        availability="public_git",
        reference_tier="third_party_reference",
        url="https://github.com/OpenVQE/OpenVQE",
        clone_url="https://github.com/OpenVQE/OpenVQE.git",
        default_ref="master",
        intended_algorithm_ids=(),
        adapter_status="cataloged_no_adapter",
        notes="Useful independent implementation surface for ADAPT and Qubit-ADAPT cross-checks.",
    ),
    ExternalReferenceSpec(
        reference_id="qiskit_adapt_vqe",
        display_name="Qiskit AdaptVQE library implementation",
        availability="library_package",
        reference_tier="library_reference",
        url="https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.AdaptVQE.html",
        clone_url=None,
        default_ref=None,
        package_imports=("qiskit_algorithms",),
        intended_algorithm_ids=(),
        adapter_status="library_adapter_exists_elsewhere",
        notes=(
            "Benchmark-local Qiskit usage is permitted for validation/reference "
            "rows only.  Keep it out of production/core VQE paths."
        ),
    ),
    ExternalReferenceSpec(
        reference_id="overlap_adapt_vqe_request",
        display_name="Overlap-ADAPT-VQE code available on request",
        availability="request_only",
        reference_tier="request_only",
        url="https://www.nature.com/articles/s42005-023-01312-y",
        clone_url=None,
        default_ref=None,
        intended_algorithm_ids=("static_overlap_adapt_phase3",),
        adapter_status="request_required",
        notes=(
            "Do not benchmark this as an executable reference unless the authors' "
            "code is obtained or a faithful reimplementation is explicitly labeled."
        ),
    ),
    ExternalReferenceSpec(
        reference_id="tencirchem_vbse",
        display_name="TenCirChem variational basis-state encoder examples",
        availability="public_git",
        reference_tier="third_party_reference",
        url="https://github.com/tencent-quantum-lab/TenCirChem",
        clone_url="https://github.com/tencent-quantum-lab/TenCirChem.git",
        default_ref="main",
        intended_algorithm_ids=(),
        adapter_status="cataloged_no_adapter",
        notes=(
            "Encoding comparator for electron-phonon simulations, not an ADAPT "
            "selector competitor.  Check license terms before manuscript use."
        ),
    ),
)


def reference_catalog() -> tuple[ExternalReferenceSpec, ...]:
    """Return the immutable external reference catalog."""

    return _REFERENCE_CATALOG


def get_external_reference_spec(reference_id: str) -> ExternalReferenceSpec:
    """Return one catalog entry by reference ID."""

    key = str(reference_id).strip()
    for spec in _REFERENCE_CATALOG:
        if spec.reference_id == key:
            return spec
    known = ", ".join(spec.reference_id for spec in _REFERENCE_CATALOG)
    raise ValueError(f"Unknown external reference {reference_id!r}. Known references: {known}")


def reference_specs_for_algorithm(algorithm_id: str) -> tuple[ExternalReferenceSpec, ...]:
    """Return catalog entries intended for a benchmark algorithm ID."""

    alg = str(algorithm_id).strip()
    return tuple(spec for spec in _REFERENCE_CATALOG if alg in spec.intended_algorithm_ids)


def external_algorithm_manifest_metadata(
    algorithm_id: str,
    *,
    status: str | None = None,
    dispatch: str | None = None,
) -> dict[str, Any]:
    """Return manifest metadata for benchmark-local external competitor rows."""

    refs = reference_specs_for_algorithm(algorithm_id)
    if not refs:
        return {}
    return {
        "external_algorithm": True,
        "external_adapt_reference_ids": [ref.reference_id for ref in refs],
        "external_adapt_reference_urls": {ref.reference_id: ref.url for ref in refs},
        "external_adapt_reference_tiers": {ref.reference_id: ref.reference_tier for ref in refs},
        "external_adapt_availability": {ref.reference_id: ref.availability for ref in refs},
        "external_adapt_adapter_status": {ref.reference_id: ref.adapter_status for ref in refs},
        "external_adapt_algorithm_adapter_status": _EXTERNAL_ADAPT_ALGORITHM_ADAPTER_STATUS.get(
            str(algorithm_id).strip(), "cataloged_no_adapter"
        ),
        "external_adapt_default_refs": {
            ref.reference_id: ref.default_ref for ref in refs if ref.default_ref is not None
        },
        "external_adapt_pinned_commits": {
            ref.reference_id: ref.pinned_commit for ref in refs if ref.pinned_commit is not None
        },
        "external_adapt_package_imports": {
            ref.reference_id: list(ref.package_imports) for ref in refs if ref.package_imports
        },
        "external_adapt_manifest_status": str(status) if status is not None else None,
        "external_adapt_dispatch": dispatch,
        "phase3_controller_called": False,
        "external_adapt_policy": "do_not_emulate_through_phase3_controller",
    }


__all__ = [
    "CEO_ADAPT_VQE_PINNED_COMMIT",
    "EXTERNAL_ADAPT_ALGORITHM_IDS",
    "ExternalReferenceSpec",
    "external_algorithm_manifest_metadata",
    "get_external_reference_spec",
    "reference_catalog",
    "reference_specs_for_algorithm",
]
