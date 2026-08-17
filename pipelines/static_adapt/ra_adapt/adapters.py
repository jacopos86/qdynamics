"""Typed candidate-representation adapters for the shared RA engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol, Sequence, runtime_checkable

from pipelines.contracts.problem import ResolvedProblemContext
from pipelines.static_adapt.ra_adapt.contracts import (
    CANDIDATE_REPRESENTATION_MACRO,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
)
from pipelines.static_adapt.ra_adapt.phase0 import (
    GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY,
)
from pipelines.static_adapt.ra_adapt.pools import (
    CandidateInventory,
    CandidateRecord,
    build_executable_macro_pool,
    build_guarded_single_pauli_pool,
    build_h2o_sector_complete_pauli_block_pool,
    build_h2o_symmetry_complete_generator_pool,
    build_parent_template_inventory,
    build_staged_single_pauli_pool,
)


MACRO_ADAPTER_ID = "paper_i_ra_adapt_macro_candidate_adapter_v1"
SINGLE_PAULI_ADAPTER_ID = (
    "paper_i_ra_adapt_single_pauli_word_candidate_adapter_v1"
)
MACRO_THEN_SINGLETON_PHASE_I_ADAPTER_ID = (
    "paper_i_ra_adapt_macro_then_singleton_phase_i_candidate_adapter_v1"
)
MACRO_GRADIENT_PHASE0_THEN_SINGLETON_ADAPTER_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_then_singleton_"
    "candidate_adapter_v1"
)
MACRO_GRADIENT_PHASE0_ADAPTER_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_candidate_adapter_v1"
)
GLOBAL_SINGLE_PAULI_ADAPTER_ID = (
    "paper_i_ra_adapt_global_single_pauli_word_candidate_adapter_v1"
)
GLOBAL_SINGLETON_GRADIENT_PHASE0_ADAPTER_ID = (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_candidate_adapter_v1"
)
H2O_LINEAR_FD_SINGLE_PAULI_ADAPTER_ID = (
    "paper_iv_h2o_linear_fd_ra_adapt_single_pauli_word_candidate_adapter_v1"
)
H2O_LINEAR_FD_SYMMETRY_COMPLETE_ADAPTER_ID = (
    "paper_iv_h2o_linear_fd_ra_adapt_symmetry_complete_candidate_adapter_v1"
)
H2O_LINEAR_FD_SECTOR_COMPLETE_PAULI_BLOCK_ADAPTER_ID = (
    "paper_iv_h2o_linear_fd_ra_adapt_sector_complete_pauli_block_adapter_v1"
)
H2O_LINEAR_FD_APPLICATION_FAMILY = "molecular_vibronic_h2o_linear_fd"

PHASE_I_SUPPLY_EXECUTABLE_MACRO = "executable_macro_pool_v1"
PHASE_I_SUPPLY_PARENT_TEMPLATE_FACTORY = "parent_template_factory_v1"
PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON = (
    "global_guarded_singleton_pool_v1"
)
PHASE_I_SUPPLY_H2O_SYMMETRY_COMPLETE = (
    "h2o_derivative_resolved_symmetry_complete_pool_v1"
)
PHASE_II_EXPOSURE_RETAINED_MACRO_IDENTITY = (
    "identity_on_retained_macro_generators_v1"
)
PHASE_II_EXPOSURE_RETAINED_PARENT_CHILDREN = (
    "guarded_children_from_retained_parent_shortlist_v1"
)
PHASE_II_EXPOSURE_RETAINED_PARENT_SECTOR_BLOCKS = (
    "sector_complete_pauli_blocks_from_retained_parent_shortlist_v1"
)
PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY = (
    "identity_on_retained_singletons_v1"
)
PHASE_I_VISIBILITY_ALL_EXECUTABLE = "all_executable_candidates_v1"
POST_EXPOSURE_PHASE_I_RETAINED_PARENT_SINGLETONS = (
    "phase_i_on_guarded_singletons_from_retained_macro_shortlist_v1"
)
MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY = (
    "standard_adapt_abs_gradient_macro_phase0_v1"
)


@runtime_checkable
class CandidateRepresentationAdapter(Protocol):
    candidate_representation_id: str
    adapter_id: str
    phase_i_candidate_supply_id: str
    phase_ii_candidate_exposure_id: str

    def parent_inventory(
        self, problem: ResolvedProblemContext
    ) -> CandidateInventory: ...

    def executable_pool(
        self, problem: ResolvedProblemContext
    ) -> CandidateInventory: ...

    def expose_children(
        self,
        retained_parents: Sequence[CandidateRecord],
        *,
        problem: ResolvedProblemContext | None = None,
    ) -> CandidateInventory: ...

    def candidate_geometry(
        self, record: CandidateRecord, position: int
    ) -> object: ...


@dataclass(frozen=True)
class MacroCandidateAdapter:
    """Keep each retained safe parent as one macro generator."""

    phase_i_candidate_supply_id: ClassVar[str] = (
        PHASE_I_SUPPLY_EXECUTABLE_MACRO
    )
    phase_ii_candidate_exposure_id: ClassVar[str] = (
        PHASE_II_EXPOSURE_RETAINED_MACRO_IDENTITY
    )
    candidate_representation_id: str = CANDIDATE_REPRESENTATION_MACRO
    adapter_id: str = MACRO_ADAPTER_ID

    def parent_inventory(
        self, problem: ResolvedProblemContext
    ) -> CandidateInventory:
        return build_parent_template_inventory(
            problem, representation_id=self.candidate_representation_id
        )

    def executable_pool(
        self, problem: ResolvedProblemContext
    ) -> CandidateInventory:
        return build_executable_macro_pool(problem)

    def expose_children(
        self,
        retained_parents: Sequence[CandidateRecord],
        *,
        problem: ResolvedProblemContext | None = None,
    ) -> CandidateInventory:
        """Macro exposure is the exact identity on retained parents."""

        retained = tuple(retained_parents)
        if not retained:
            raise ValueError("Macro exposure requires retained parents.")
        if any(
            record.representation_id != CANDIDATE_REPRESENTATION_MACRO
            for record in retained
        ):
            raise ValueError("Macro exposure received a non-macro candidate.")
        from pipelines.static_adapt.ra_adapt.pools import _receipt

        return CandidateInventory(
            candidates=retained,
            receipt=_receipt(
                schema="ra_adapt_retained_macro_exposure_v1",
                representation_id=CANDIDATE_REPRESENTATION_MACRO,
                candidates=retained,
            ),
            metadata={
                "exposure_policy": "identity_on_retained_parents_v1",
                "source_parent_count": int(len(retained)),
            },
        )

    def candidate_geometry(
        self, record: CandidateRecord, position: int
    ) -> object:
        from pipelines.static_adapt.ra_adapt.insertion_geometry import (
            exact_ordered_insertion_request,
        )

        return exact_ordered_insertion_request(
            record=record,
            insertion_position=int(position),
            representation_id=self.candidate_representation_id,
        )


@dataclass(frozen=True)
class MacroGradientPhase0CandidateAdapter(MacroCandidateAdapter):
    """Screen intact macros by ``|g|`` before the ordinary macro funnel.

    The retained generator identities remain intact through Phase I, Phase II,
    and Phase III.  This adapter never exposes singleton children.
    """

    macro_phase0_policy_id: ClassVar[str] = (
        MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY
    )
    adapter_id: str = MACRO_GRADIENT_PHASE0_ADAPTER_ID

    def __post_init__(self) -> None:
        if (
            self.candidate_representation_id
            != CANDIDATE_REPRESENTATION_MACRO
            or self.adapter_id != MACRO_GRADIENT_PHASE0_ADAPTER_ID
        ):
            raise ValueError(
                "Macro-gradient Phase-0 adapter identity fields are fixed."
            )


@dataclass(frozen=True)
class SinglePauliWordCandidateAdapter:
    """Expose hard-guarded unit-Pauli children with explicit ancestry."""

    phase_i_candidate_supply_id: ClassVar[str] = (
        PHASE_I_SUPPLY_PARENT_TEMPLATE_FACTORY
    )
    phase_ii_candidate_exposure_id: ClassVar[str] = (
        PHASE_II_EXPOSURE_RETAINED_PARENT_CHILDREN
    )
    candidate_representation_id: str = CANDIDATE_REPRESENTATION_SINGLE_PAULI
    adapter_id: str = SINGLE_PAULI_ADAPTER_ID

    def parent_inventory(
        self, problem: ResolvedProblemContext
    ) -> CandidateInventory:
        return build_parent_template_inventory(
            problem, representation_id=self.candidate_representation_id
        )

    def executable_pool(
        self, problem: ResolvedProblemContext
    ) -> CandidateInventory:
        """Return the parent factory inventory used by staged RA exposure."""

        inventory = self.parent_inventory(problem)
        return CandidateInventory(
            candidates=inventory.candidates,
            receipt=inventory.receipt,
            metadata={
                **dict(inventory.metadata),
                "executable_pool_kind": (
                    "guarded_singleton_children_factory_v1"
                ),
                "children_constructed": False,
            },
        )

    def expose_children(
        self,
        retained_parents: Sequence[CandidateRecord],
        *,
        problem: ResolvedProblemContext | None = None,
    ) -> CandidateInventory:
        if problem is None:
            raise TypeError(
                "Single-Pauli child exposure requires the resolved problem "
                "for symmetry and padding guards."
            )
        return build_staged_single_pauli_pool(
            problem,
            retained_parents=tuple(retained_parents),
        )

    def global_executable_pool(
        self, problem: ResolvedProblemContext
    ) -> CandidateInventory:
        """Build the global guarded child pool used only by Append-ADAPT."""

        return build_guarded_single_pauli_pool(problem)

    def candidate_geometry(
        self, record: CandidateRecord, position: int
    ) -> object:
        from pipelines.static_adapt.ra_adapt.insertion_geometry import (
            exact_ordered_insertion_request,
        )

        return exact_ordered_insertion_request(
            record=record,
            insertion_position=int(position),
            representation_id=self.candidate_representation_id,
        )


@dataclass(frozen=True)
class MacroThenSingletonPhaseICandidateAdapter(
    SinglePauliWordCandidateAdapter
):
    """Screen retained macro children through a fresh singleton Phase I.

    The initial Phase-I population remains the authenticated parent-template
    factory.  After its macro shortlist, guarded singleton children are
    exposed and receive their own cheap Phase-I shortlist before any
    Phase-II/III evaluation.
    """

    post_exposure_phase_i_shortlist_id: ClassVar[str] = (
        POST_EXPOSURE_PHASE_I_RETAINED_PARENT_SINGLETONS
    )
    adapter_id: str = MACRO_THEN_SINGLETON_PHASE_I_ADAPTER_ID

    def __post_init__(self) -> None:
        if (
            self.candidate_representation_id
            != CANDIDATE_REPRESENTATION_SINGLE_PAULI
            or self.adapter_id
            != MACRO_THEN_SINGLETON_PHASE_I_ADAPTER_ID
        ):
            raise ValueError(
                "Macro-then-singleton Phase-I adapter identity fields are "
                "fixed."
            )


@dataclass(frozen=True)
class MacroGradientPhase0ThenSingletonCandidateAdapter(
    MacroThenSingletonPhaseICandidateAdapter
):
    """Use an exact gradient-only macro screen before singleton Phase I.

    The macro stage ranks authenticated parent identities only by the
    standard ADAPT endpoint gradient magnitude.  Guarded singleton children
    of the retained parents then form the official Phase-I population.
    """

    macro_phase0_policy_id: ClassVar[str] = (
        MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY
    )
    adapter_id: str = MACRO_GRADIENT_PHASE0_THEN_SINGLETON_ADAPTER_ID

    def __post_init__(self) -> None:
        if (
            self.candidate_representation_id
            != CANDIDATE_REPRESENTATION_SINGLE_PAULI
            or self.adapter_id
            != MACRO_GRADIENT_PHASE0_THEN_SINGLETON_ADAPTER_ID
        ):
            raise ValueError(
                "Macro-gradient Phase-0 then singleton adapter identity "
                "fields are fixed."
            )


@dataclass(frozen=True)
class H2OLinearFDSinglePauliWordCandidateAdapter(
    SinglePauliWordCandidateAdapter
):
    """Named staged-singleton adapter for the production H2O application."""

    application_family_key: ClassVar[str] = H2O_LINEAR_FD_APPLICATION_FAMILY
    application_pool_key: ClassVar[str] = "full_meta_derivative_resolved_v2"
    adapter_id: str = H2O_LINEAR_FD_SINGLE_PAULI_ADAPTER_ID

    def __post_init__(self) -> None:
        if (
            self.candidate_representation_id
            != CANDIDATE_REPRESENTATION_SINGLE_PAULI
            or self.adapter_id != H2O_LINEAR_FD_SINGLE_PAULI_ADAPTER_ID
        ):
            raise ValueError("H2O RA adapter identity fields are fixed.")


@dataclass(frozen=True)
class H2OLinearFDSymmetryCompleteCandidateAdapter(MacroCandidateAdapter):
    """Admit one physical derivative-resolved H2O generator per round."""

    application_family_key: ClassVar[str] = H2O_LINEAR_FD_APPLICATION_FAMILY
    application_pool_key: ClassVar[str] = "full_meta_derivative_resolved_v2"
    phase_i_candidate_supply_id: ClassVar[str] = (
        PHASE_I_SUPPLY_H2O_SYMMETRY_COMPLETE
    )
    candidate_representation_id: str = CANDIDATE_REPRESENTATION_MACRO
    adapter_id: str = H2O_LINEAR_FD_SYMMETRY_COMPLETE_ADAPTER_ID

    def __post_init__(self) -> None:
        if (
            self.candidate_representation_id
            != CANDIDATE_REPRESENTATION_MACRO
            or self.adapter_id
            != H2O_LINEAR_FD_SYMMETRY_COMPLETE_ADAPTER_ID
        ):
            raise ValueError("H2O RA adapter identity fields are fixed.")

    def executable_pool(
        self, problem: ResolvedProblemContext
    ) -> CandidateInventory:
        return build_h2o_symmetry_complete_generator_pool(problem)


@dataclass(frozen=True)
class H2OLinearFDSectorCompletePauliBlockCandidateAdapter(
    MacroCandidateAdapter
):
    """Stage the finest Pauli blocks that preserve the H2O electron sector."""

    application_family_key: ClassVar[str] = H2O_LINEAR_FD_APPLICATION_FAMILY
    application_pool_key: ClassVar[str] = "full_meta_derivative_resolved_v2"
    phase_i_candidate_supply_id: ClassVar[str] = (
        PHASE_I_SUPPLY_H2O_SYMMETRY_COMPLETE
    )
    phase_ii_candidate_exposure_id: ClassVar[str] = (
        PHASE_II_EXPOSURE_RETAINED_PARENT_SECTOR_BLOCKS
    )
    candidate_representation_id: str = CANDIDATE_REPRESENTATION_MACRO
    adapter_id: str = H2O_LINEAR_FD_SECTOR_COMPLETE_PAULI_BLOCK_ADAPTER_ID

    def __post_init__(self) -> None:
        if (
            self.candidate_representation_id
            != CANDIDATE_REPRESENTATION_MACRO
            or self.adapter_id
            != H2O_LINEAR_FD_SECTOR_COMPLETE_PAULI_BLOCK_ADAPTER_ID
        ):
            raise ValueError(
                "H2O sector-complete block adapter identity fields are fixed."
            )

    def executable_pool(
        self, problem: ResolvedProblemContext
    ) -> CandidateInventory:
        return build_h2o_symmetry_complete_generator_pool(problem)

    def expose_children(
        self,
        retained_parents: Sequence[CandidateRecord],
        *,
        problem: ResolvedProblemContext | None = None,
    ) -> CandidateInventory:
        if problem is None:
            raise TypeError(
                "H2O sector-complete block exposure requires the resolved "
                "problem."
            )
        return build_h2o_sector_complete_pauli_block_pool(
            problem,
            retained_parents=tuple(retained_parents),
        )


@dataclass(frozen=True)
class GlobalSinglePauliWordCandidateAdapter:
    """Run every globally guarded Append singleton through the RA funnel.

    Phase I receives exactly the conventional Append guarded-singleton
    inventory.  Phase II reranks the retained Phase-I singleton identities;
    it does not split retained macro parents into a new child population.
    """

    phase_i_candidate_supply_id: ClassVar[str] = (
        PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON
    )
    phase_ii_candidate_exposure_id: ClassVar[str] = (
        PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY
    )
    phase_i_candidate_visibility_id: ClassVar[str] = (
        PHASE_I_VISIBILITY_ALL_EXECUTABLE
    )
    candidate_representation_id: str = CANDIDATE_REPRESENTATION_SINGLE_PAULI
    adapter_id: str = GLOBAL_SINGLE_PAULI_ADAPTER_ID

    def __post_init__(self) -> None:
        if (
            self.candidate_representation_id
            != CANDIDATE_REPRESENTATION_SINGLE_PAULI
            or self.adapter_id != GLOBAL_SINGLE_PAULI_ADAPTER_ID
        ):
            raise ValueError(
                "Global singleton adapter identity fields are fixed."
            )

    def parent_inventory(
        self, problem: ResolvedProblemContext
    ) -> CandidateInventory:
        """Retain the complete parent-template ancestry as provenance."""

        return build_parent_template_inventory(
            problem, representation_id=self.candidate_representation_id
        )

    def executable_pool(
        self, problem: ResolvedProblemContext
    ) -> CandidateInventory:
        """Return the exact global pool used by conventional Append-ADAPT."""

        return build_guarded_single_pauli_pool(problem)

    def expose_children(
        self,
        retained_parents: Sequence[CandidateRecord],
        *,
        problem: ResolvedProblemContext | None = None,
    ) -> CandidateInventory:
        """Expose the retained Phase-I singletons by exact identity."""

        if problem is None:
            raise TypeError(
                "Global singleton identity exposure requires the resolved "
                "problem for executable-pool authentication."
            )
        retained = tuple(retained_parents)
        if not retained:
            raise ValueError(
                "Global singleton identity exposure requires retained "
                "Phase-I candidates."
            )
        if any(
            candidate.representation_id
            != CANDIDATE_REPRESENTATION_SINGLE_PAULI
            for candidate in retained
        ):
            raise ValueError(
                "Global singleton identity exposure received a "
                "non-singleton candidate."
            )
        global_pool = self.executable_pool(problem)
        authenticated = {
            (
                str(candidate.label),
                str(candidate.generator_identity),
            ): candidate
            for candidate in global_pool.candidates
        }
        if len(authenticated) != len(global_pool.candidates):
            raise RuntimeError(
                "Global singleton executable inventory contains duplicate "
                "identities."
            )
        seen: set[tuple[str, str]] = set()
        for candidate in retained:
            identity = (
                str(candidate.label),
                str(candidate.generator_identity),
            )
            expected = authenticated.get(identity)
            if (
                identity in seen
                or expected is None
                or expected.manifest_row() != candidate.manifest_row()
            ):
                raise ValueError(
                    "Retained singleton identity is duplicated or absent "
                    "from the authenticated global executable pool."
                )
            seen.add(identity)

        from pipelines.static_adapt.ra_adapt.pools import _receipt

        return CandidateInventory(
            candidates=retained,
            receipt=_receipt(
                schema="ra_adapt_retained_singleton_identity_exposure_v1",
                representation_id=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
                candidates=retained,
            ),
            metadata={
                "exposure_scope": "phase_i_retained_singletons_v1",
                "exposure_policy": (
                    self.phase_ii_candidate_exposure_id
                ),
                "phase_i_candidate_supply": (
                    self.phase_i_candidate_supply_id
                ),
                "phase_i_candidate_visibility": (
                    self.phase_i_candidate_visibility_id
                ),
                "source_singleton_count": int(len(retained)),
                "source_singleton_labels": [
                    str(candidate.label) for candidate in retained
                ],
                "source_singleton_generator_identities": [
                    str(candidate.generator_identity)
                    for candidate in retained
                ],
                "global_executable_pool_sha256": str(
                    global_pool.receipt.sha256
                ),
            },
        )

    def global_executable_pool(
        self, problem: ResolvedProblemContext
    ) -> CandidateInventory:
        """Compatibility spelling for exact Append-pool comparisons."""

        return self.executable_pool(problem)

    def candidate_geometry(
        self, record: CandidateRecord, position: int
    ) -> object:
        from pipelines.static_adapt.ra_adapt.insertion_geometry import (
            exact_ordered_insertion_request,
        )

        return exact_ordered_insertion_request(
            record=record,
            insertion_position=int(position),
            representation_id=self.candidate_representation_id,
        )


@dataclass(frozen=True)
class GlobalSingletonGradientPhase0CandidateAdapter(
    GlobalSinglePauliWordCandidateAdapter
):
    """Screen the complete guarded-singleton supply by standard ADAPT ``|g|``."""

    phase0_shortlist_policy_id: ClassVar[str] = (
        GLOBAL_SINGLETON_ABSOLUTE_GRADIENT_PHASE0_POLICY
    )
    adapter_id: str = GLOBAL_SINGLETON_GRADIENT_PHASE0_ADAPTER_ID

    def __post_init__(self) -> None:
        if (
            self.candidate_representation_id
            != CANDIDATE_REPRESENTATION_SINGLE_PAULI
            or self.adapter_id
            != GLOBAL_SINGLETON_GRADIENT_PHASE0_ADAPTER_ID
        ):
            raise ValueError(
                "Global-singleton gradient Phase-0 adapter identity fields "
                "are fixed."
            )


__all__ = [
    "CandidateRepresentationAdapter",
    "GLOBAL_SINGLE_PAULI_ADAPTER_ID",
    "GLOBAL_SINGLETON_GRADIENT_PHASE0_ADAPTER_ID",
    "GlobalSinglePauliWordCandidateAdapter",
    "GlobalSingletonGradientPhase0CandidateAdapter",
    "H2O_LINEAR_FD_APPLICATION_FAMILY",
    "H2O_LINEAR_FD_SINGLE_PAULI_ADAPTER_ID",
    "H2O_LINEAR_FD_SECTOR_COMPLETE_PAULI_BLOCK_ADAPTER_ID",
    "H2OLinearFDSinglePauliWordCandidateAdapter",
    "H2OLinearFDSectorCompletePauliBlockCandidateAdapter",
    "H2OLinearFDSymmetryCompleteCandidateAdapter",
    "MACRO_ADAPTER_ID",
    "MACRO_GRADIENT_PHASE0_ADAPTER_ID",
    "MACRO_GRADIENT_PHASE0_THEN_SINGLETON_ADAPTER_ID",
    "MACRO_PHASE0_STANDARD_ADAPT_ABS_GRADIENT_POLICY",
    "MACRO_THEN_SINGLETON_PHASE_I_ADAPTER_ID",
    "MacroCandidateAdapter",
    "MacroGradientPhase0CandidateAdapter",
    "MacroGradientPhase0ThenSingletonCandidateAdapter",
    "MacroThenSingletonPhaseICandidateAdapter",
    "PHASE_I_SUPPLY_EXECUTABLE_MACRO",
    "PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON",
    "PHASE_I_SUPPLY_PARENT_TEMPLATE_FACTORY",
    "PHASE_I_VISIBILITY_ALL_EXECUTABLE",
    "POST_EXPOSURE_PHASE_I_RETAINED_PARENT_SINGLETONS",
    "PHASE_II_EXPOSURE_RETAINED_MACRO_IDENTITY",
    "PHASE_II_EXPOSURE_RETAINED_PARENT_CHILDREN",
    "PHASE_II_EXPOSURE_RETAINED_PARENT_SECTOR_BLOCKS",
    "PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY",
    "SINGLE_PAULI_ADAPTER_ID",
    "SinglePauliWordCandidateAdapter",
]
