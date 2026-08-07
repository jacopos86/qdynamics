"""Shared normalized Pauli-pool contracts for Paper-II comparisons."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Sequence

from pipelines.scaffold.runtime_contract import CandidatePoolSource, ScaffoldRuntimeInput
from src.quantum.ansatz_parameterization import iter_runtime_rotation_terms
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


NORMALIZED_POOL_HAMILTONIAN_DRIVE = "hamiltonian_drive_pauli"
NORMALIZED_POOL_FULL_META_CHILDREN = "full_meta_pauli_children"
NORMALIZED_POOL_PROFILES = frozenset(
    {
        NORMALIZED_POOL_HAMILTONIAN_DRIVE,
        NORMALIZED_POOL_FULL_META_CHILDREN,
    }
)
NORMALIZED_POOL_SCHEMA_V1 = "paper_ii_normalized_pauli_pool_v1"


@dataclass(frozen=True)
class NormalizedPauliPoolAtom:
    """One unique unit-coefficient Pauli generator in a normalized pool."""

    pauli_exyz: str
    nq: int
    repr_mode: str
    source_labels: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "pauli_exyz": str(self.pauli_exyz),
            "nq": int(self.nq),
            "repr_mode": str(self.repr_mode),
            "source_labels": [str(label) for label in self.source_labels],
        }


@dataclass(frozen=True)
class NormalizedPauliPoolContract:
    """Deterministic ordered-set receipt shared by APM and comparators."""

    profile: str
    atoms: tuple[NormalizedPauliPoolAtom, ...]
    source_occurrence_count: int
    source_parent_count: int
    untruncated_atom_count: int
    candidate_limit: int | None = None

    @property
    def ordered_paulis(self) -> tuple[str, ...]:
        return tuple(str(atom.pauli_exyz) for atom in self.atoms)

    @property
    def ordered_unique_pauli_sha256(self) -> str:
        return _sha256_json(list(self.ordered_paulis))

    @property
    def ordered_atom_contract_sha256(self) -> str:
        return _sha256_json(
            [
                {
                    "pauli_exyz": str(atom.pauli_exyz),
                    "nq": int(atom.nq),
                    "repr_mode": str(atom.repr_mode),
                }
                for atom in self.atoms
            ]
        )

    @property
    def truncated(self) -> bool:
        return int(len(self.atoms)) != int(self.untruncated_atom_count)

    def limited(self, candidate_limit: int | None) -> "NormalizedPauliPoolContract":
        if candidate_limit is None:
            return self
        limit = int(candidate_limit)
        if limit <= 0:
            raise ValueError("candidate_limit must be positive when provided")
        return replace(
            self,
            atoms=tuple(self.atoms[:limit]),
            candidate_limit=limit,
        )

    def to_json_dict(self, *, include_atoms: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": NORMALIZED_POOL_SCHEMA_V1,
            "profile": str(self.profile),
            "normalization": "unique_sorted_unit_coefficient_pauli_generators",
            "atom_count": int(len(self.atoms)),
            "untruncated_atom_count": int(self.untruncated_atom_count),
            "source_occurrence_count": int(self.source_occurrence_count),
            "source_parent_count": int(self.source_parent_count),
            "candidate_limit": (
                None if self.candidate_limit is None else int(self.candidate_limit)
            ),
            "truncated": bool(self.truncated),
            "ordered_unique_pauli_sha256": str(
                self.ordered_unique_pauli_sha256
            ),
            "ordered_atom_contract_sha256": str(
                self.ordered_atom_contract_sha256
            ),
            "first_paulis": [str(value) for value in self.ordered_paulis[:10]],
            "last_paulis": [str(value) for value in self.ordered_paulis[-10:]],
        }
        if include_atoms:
            payload["atoms"] = [atom.to_json_dict() for atom in self.atoms]
        return payload


def build_normalized_pauli_pool(
    *,
    profile: str,
    static_poly: Any,
    drive_poly: Any | None = None,
    candidate_pool_terms: Sequence[Any] = (),
) -> NormalizedPauliPoolContract:
    """Build one of the two shared Paper-II normalized pool profiles."""

    normalized_profile = str(profile).strip().lower()
    if normalized_profile not in NORMALIZED_POOL_PROFILES:
        raise ValueError(
            "normalized pool profile must be one of "
            f"{sorted(NORMALIZED_POOL_PROFILES)}, got {profile!r}"
        )

    specs: list[tuple[str, int, str, str]] = []
    source_parent_count = 0
    if normalized_profile == NORMALIZED_POOL_HAMILTONIAN_DRIVE:
        specs.extend(
            _pool_specs_from_polynomial(
                static_poly,
                source_label="static_hamiltonian",
            )
        )
        source_parent_count += 1
        if drive_poly is not None:
            specs.extend(
                _pool_specs_from_polynomial(
                    drive_poly,
                    source_label="drive_hamiltonian",
                )
            )
            source_parent_count += 1
    else:
        for index, term in enumerate(tuple(candidate_pool_terms or ())):
            execution_mode = str(
                getattr(term, "execution_mode", "termwise_product")
                or "termwise_product"
            ).strip().lower()
            if execution_mode == "grouped_exact":
                continue
            source_parent_count += 1
            specs.extend(
                _pool_specs_from_polynomial(
                    getattr(term, "polynomial"),
                    source_label=str(
                        getattr(term, "label", f"candidate_{index}")
                    ),
                )
            )

    atoms = _deduplicate_pool_specs(specs)
    if not atoms:
        raise ValueError(
            f"normalized Pauli pool {normalized_profile!r} is empty"
        )
    return NormalizedPauliPoolContract(
        profile=normalized_profile,
        atoms=atoms,
        source_occurrence_count=int(len(specs)),
        source_parent_count=int(source_parent_count),
        untruncated_atom_count=int(len(atoms)),
    )


def normalized_pool_candidate_terms(
    contract: NormalizedPauliPoolContract,
) -> tuple[AnsatzTerm, ...]:
    """Convert a normalized pool receipt into APM candidate terms."""

    return tuple(
        AnsatzTerm(
            label=(
                f"normalized_pool::{contract.profile}::"
                f"p{index:04d}::{atom.pauli_exyz}"
            ),
            polynomial=PauliPolynomial(
                str(atom.repr_mode),
                [
                    PauliTerm(
                        int(atom.nq),
                        ps=str(atom.pauli_exyz),
                        pc=1.0,
                    )
                ],
            ),
            execution_mode="termwise_product",
        )
        for index, atom in enumerate(contract.atoms)
    )


def runtime_input_with_normalized_candidate_pool(
    runtime_input: ScaffoldRuntimeInput,
    contract: NormalizedPauliPoolContract,
) -> ScaffoldRuntimeInput:
    """Return the same seed/runtime input with only its future pool replaced."""

    manifest = contract.to_json_dict(include_atoms=False)
    source = CandidatePoolSource(
        source_kind="resolved_pool",
        pool_key=f"normalized::{contract.profile}",
        completeness="complete",
        pool_build_kwargs={},
        filter_payload={"normalized_pauli_pool": manifest},
    )
    extensions = dict(runtime_input.extensions or {})
    extensions["normalized_pauli_pool"] = manifest
    provenance = dict(runtime_input.provenance or {})
    provenance["normalized_candidate_pool"] = manifest
    return replace(
        runtime_input,
        candidate_pool_terms=normalized_pool_candidate_terms(contract),
        candidate_pool_source=source,
        provenance=provenance,
        extensions=extensions,
    )


def _pool_specs_from_polynomial(
    poly: Any,
    *,
    source_label: str,
) -> list[tuple[str, int, str, str]]:
    repr_mode = str(getattr(poly, "_repr_mode", "JW") or "JW")
    return [
        (
            str(spec.pauli_exyz),
            int(spec.nq),
            repr_mode,
            str(source_label),
        )
        for spec in iter_runtime_rotation_terms(
            poly,
            ignore_identity=True,
            coefficient_tolerance=1.0e-12,
            sort_terms=True,
        )
    ]


def _deduplicate_pool_specs(
    specs: Sequence[tuple[str, int, str, str]],
) -> tuple[NormalizedPauliPoolAtom, ...]:
    by_pauli: dict[str, dict[str, Any]] = {}
    for pauli_exyz, nq, repr_mode, source_label in specs:
        key = str(pauli_exyz)
        row = by_pauli.setdefault(
            key,
            {
                "nq": int(nq),
                "repr_mode": str(repr_mode),
                "source_labels": [],
            },
        )
        if int(row["nq"]) != int(nq) or str(row["repr_mode"]) != str(repr_mode):
            raise ValueError(
                "Conflicting normalized Pauli metadata for "
                f"{key!r}: {(row['nq'], row['repr_mode'])!r} versus "
                f"{(nq, repr_mode)!r}"
            )
        if str(source_label) not in row["source_labels"]:
            row["source_labels"].append(str(source_label))
    return tuple(
        NormalizedPauliPoolAtom(
            pauli_exyz=str(pauli_exyz),
            nq=int(row["nq"]),
            repr_mode=str(row["repr_mode"]),
            source_labels=tuple(str(label) for label in row["source_labels"]),
        )
        for pauli_exyz, row in sorted(by_pauli.items())
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "NORMALIZED_POOL_FULL_META_CHILDREN",
    "NORMALIZED_POOL_HAMILTONIAN_DRIVE",
    "NORMALIZED_POOL_PROFILES",
    "NORMALIZED_POOL_SCHEMA_V1",
    "NormalizedPauliPoolAtom",
    "NormalizedPauliPoolContract",
    "build_normalized_pauli_pool",
    "normalized_pool_candidate_terms",
    "runtime_input_with_normalized_candidate_pool",
]
