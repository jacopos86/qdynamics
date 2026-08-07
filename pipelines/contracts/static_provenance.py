"""Static ADAPT / HH full-meta provenance contracts.

This module owns data-only contract symbols shared by static ADAPT and QSE
consumers. It must remain implementation-free: no static builder, QSE, scaffold,
reporting, exact-bench, hardcoded, or Qiskit imports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


HH_FULL_META_CLASSIFIER_VERSION = "hh_full_meta_v4"
HH_MATH_MD_FULL_META_POOL_KEY = "math_md_full_meta_v1"
HH_MATH_MD_FULL_META_DISPLAY_NAME = "Math.md Full Meta"
HH_MATH_MD_FULL_META_POOL_ALIASES = (
    "full_meta",
    HH_MATH_MD_FULL_META_POOL_KEY,
    "math_md_full_meta",
)
HH_FULL_META_ALLOWED_CLASSES = (
    "hh_termwise_unit",
    "hh_termwise_quadrature",
    "uccsd_sing",
    "uccsd_dbl",
    "hva_layer",
    "hh_hamiltonian_block",
    "hh_fermionic_reusable",
    "hh_phonon_linear",
    "hh_phonon_quadratic",
    "hh_vlf_sq",
    "paop_cloud_p",
    "paop_cloud_x",
    "paop_disp",
    "paop_dbl",
    "paop_hopdrag",
    "paop_dbl_p",
    "paop_dbl_x",
    "paop_curdrag",
    "paop_hop2",
    "paop_other",
    "uccsd_paop_product",
    "uccsd_paop_product_seq_ferm",
    "uccsd_paop_product_seq_motif",
)
HH_FULL_META_OPERATOR_BASIS_SOURCES = ("full_meta", "full_meta_filtered")
HH_FULL_META_PROVENANCE_SCHEMA_VERSION = "paper_iii_hh_full_meta_provenance_v1"
HH_FULL_META_BUILDER_MODULE = "pipelines.static_adapt.builders.hh_pool_presets"
HH_FULL_META_BUILDER_FUNCTION = "_build_hh_full_meta_pool"
HH_FULL_META_REQUIRED_LAYOUT_FIELDS = (
    "num_sites",
    "n_ph_max",
    "t",
    "u",
    "omega0",
    "g_ep",
    "dv",
    "boson_encoding",
    "ordering",
    "boundary",
    "paop_r",
    "paop_split_paulis",
    "paop_prune_eps",
    "paop_normalization",
)

HH_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION = "hh_physical_operator_lanes_v2_uccsd_split"
HH_PHYSICAL_OPERATOR_LANE_UCCSD_SINGLE = "uccsd_single"
HH_PHYSICAL_OPERATOR_LANE_UCCSD_DOUBLE = "uccsd_double"
HH_PHYSICAL_OPERATOR_LANE_UCCSD_CORRELATION = "uccsd_correlation"
HH_PHYSICAL_OPERATOR_LANE_ELECTRONIC_CURRENT = "electronic_current"
HH_PHYSICAL_OPERATOR_LANE_PHONON_DISPLACEMENT = "phonon_displacement"
HH_PHYSICAL_OPERATOR_LANE_PHONON_SQUEEZE_RELAXATION = "phonon_squeeze_relaxation"
HH_PHYSICAL_OPERATOR_LANE_DRESSED_PHONON_CORRELATION = "dressed_phonon_correlation"
HH_PHYSICAL_OPERATOR_LANE_HVA_HAMILTONIAN_BLOCKS = "hva_hamiltonian_blocks"
HH_PHYSICAL_OPERATOR_LANE_OTHER = "other"
HH_PHYSICAL_OPERATOR_LANES = (
    HH_PHYSICAL_OPERATOR_LANE_UCCSD_SINGLE,
    HH_PHYSICAL_OPERATOR_LANE_UCCSD_DOUBLE,
    HH_PHYSICAL_OPERATOR_LANE_UCCSD_CORRELATION,
    HH_PHYSICAL_OPERATOR_LANE_ELECTRONIC_CURRENT,
    HH_PHYSICAL_OPERATOR_LANE_PHONON_DISPLACEMENT,
    HH_PHYSICAL_OPERATOR_LANE_PHONON_SQUEEZE_RELAXATION,
    HH_PHYSICAL_OPERATOR_LANE_DRESSED_PHONON_CORRELATION,
    HH_PHYSICAL_OPERATOR_LANE_HVA_HAMILTONIAN_BLOCKS,
    HH_PHYSICAL_OPERATOR_LANE_OTHER,
)

HUBBARD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION = "hubbard_physical_operator_lanes_v3_uccsd_qeb_hva_blocks"
HUBBARD_PHYSICAL_OPERATOR_LANE_QEB_EXCITATION = "qeb_excitation"
HUBBARD_PHYSICAL_OPERATOR_LANE_HVA_HAMILTONIAN_BLOCKS = "hva_hamiltonian_blocks"
HUBBARD_PHYSICAL_OPERATOR_LANES = (
    HH_PHYSICAL_OPERATOR_LANE_UCCSD_SINGLE,
    HH_PHYSICAL_OPERATOR_LANE_UCCSD_DOUBLE,
    HUBBARD_PHYSICAL_OPERATOR_LANE_QEB_EXCITATION,
    HUBBARD_PHYSICAL_OPERATOR_LANE_HVA_HAMILTONIAN_BLOCKS,
    HH_PHYSICAL_OPERATOR_LANE_OTHER,
)

_HUBBARD_QEB_SINGLE_LABEL_RE = re.compile(r"^qeb_pair\(\d+,\d+\)$")
_HUBBARD_QEB_DOUBLE_LABEL_RE = re.compile(r"^qeb_double\(\d+,\d+->\d+,\d+\)$")
_HUBBARD_HVA_BLOCK_LABEL_RE = re.compile(r"^hva_block::(hop_layer|onsite_layer|potential_layer)$")
_HUBBARD_HAM_BLOCK_LABEL_RE = re.compile(r"^ham_block::(hop|onsite|pot)(\(|$)")
_HUBBARD_CHILD_SET_SUFFIX_RE = re.compile(r"::child_set\[[^\]]+\]$")

SPIN_BOSON_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION = "spin_boson_physical_operator_lanes_v2_full_meta_hamiltonian_blocks"
SPIN_BOSON_PHYSICAL_OPERATOR_LANE_EMITTER_MATTER = "emitter_matter"
SPIN_BOSON_PHYSICAL_OPERATOR_LANE_BOSON_LINEAR = "boson_linear"
SPIN_BOSON_PHYSICAL_OPERATOR_LANE_BOSON_NONLINEAR = "boson_nonlinear"
SPIN_BOSON_PHYSICAL_OPERATOR_LANE_TRANSVERSE_COUPLING = "transverse_coupling"
SPIN_BOSON_PHYSICAL_OPERATOR_LANE_LONGITUDINAL_COUPLING = "longitudinal_coupling"
SPIN_BOSON_PHYSICAL_OPERATOR_LANE_EMITTER_Y_CORRELATION = "emitter_y_correlation"
SPIN_BOSON_PHYSICAL_OPERATOR_LANE_HAMILTONIAN_BLOCKS = "hamiltonian_blocks"
SPIN_BOSON_PHYSICAL_OPERATOR_LANES = (
    SPIN_BOSON_PHYSICAL_OPERATOR_LANE_EMITTER_MATTER,
    SPIN_BOSON_PHYSICAL_OPERATOR_LANE_BOSON_LINEAR,
    SPIN_BOSON_PHYSICAL_OPERATOR_LANE_BOSON_NONLINEAR,
    SPIN_BOSON_PHYSICAL_OPERATOR_LANE_TRANSVERSE_COUPLING,
    SPIN_BOSON_PHYSICAL_OPERATOR_LANE_LONGITUDINAL_COUPLING,
    SPIN_BOSON_PHYSICAL_OPERATOR_LANE_EMITTER_Y_CORRELATION,
    SPIN_BOSON_PHYSICAL_OPERATOR_LANE_HAMILTONIAN_BLOCKS,
    HH_PHYSICAL_OPERATOR_LANE_OTHER,
)

BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION = "bose_hubbard_physical_operator_lanes_v2_full_meta_hamiltonian_blocks"
BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_NUMBER_DENSITY_INTERACTION = "number_density_interaction"
BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_ONSITE_QUADRATURE = "onsite_quadrature"
BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_SINGLE_PARTICLE_TRANSPORT = "single_particle_transport"
BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_INTERSITE_QUADRATURE = "intersite_quadrature"
BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_DENSITY_ASSISTED_TRANSPORT = "density_assisted_transport"
BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_PAIR_TRANSPORT = "pair_transport"
BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_HAMILTONIAN_BLOCKS = "hamiltonian_blocks"
BOSE_HUBBARD_PHYSICAL_OPERATOR_LANES = (
    BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_NUMBER_DENSITY_INTERACTION,
    BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_ONSITE_QUADRATURE,
    BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_SINGLE_PARTICLE_TRANSPORT,
    BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_INTERSITE_QUADRATURE,
    BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_DENSITY_ASSISTED_TRANSPORT,
    BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_PAIR_TRANSPORT,
    BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_HAMILTONIAN_BLOCKS,
    HH_PHYSICAL_OPERATOR_LANE_OTHER,
)

H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION = (
    "h2o_linear_fd_physical_operator_lanes_v2_derivative_resolved"
)
H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_ELECTRONIC_SINGLE = "electronic_single"
H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_ELECTRONIC_DOUBLE = "electronic_double"
H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRATIONAL_MOMENTUM = (
    "vibrational_momentum"
)
H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_DERIVATIVE_MOMENTUM = (
    "vibronic_derivative_momentum"
)
H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_ONE_BODY_RESPONSE = (
    "vibronic_one_body_response"
)
H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_TWO_BODY_RESPONSE = (
    "vibronic_two_body_response"
)
H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_CONDITIONAL_SINGLE = (
    "vibronic_conditional_single"
)
H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_CONDITIONAL_DOUBLE = (
    "vibronic_conditional_double"
)
H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANES = (
    H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_ELECTRONIC_SINGLE,
    H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_ELECTRONIC_DOUBLE,
    H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRATIONAL_MOMENTUM,
    H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_DERIVATIVE_MOMENTUM,
    H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_ONE_BODY_RESPONSE,
    H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_TWO_BODY_RESPONSE,
    H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_CONDITIONAL_SINGLE,
    H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_CONDITIONAL_DOUBLE,
    HH_PHYSICAL_OPERATOR_LANE_OTHER,
)

MOLECULAR_RESTRICTED_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION = (
    "molecular_restricted_physical_operator_lanes_v1"
)
MOLECULAR_RESTRICTED_PHYSICAL_OPERATOR_LANES = (
    H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_ELECTRONIC_SINGLE,
    H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_ELECTRONIC_DOUBLE,
    HH_PHYSICAL_OPERATOR_LANE_OTHER,
)

_HH_FULL_META_CLASS_TO_PHYSICAL_OPERATOR_LANE = {
    "uccsd_sing": HH_PHYSICAL_OPERATOR_LANE_UCCSD_SINGLE,
    "uccsd_dbl": HH_PHYSICAL_OPERATOR_LANE_UCCSD_DOUBLE,
    "hh_termwise_unit": HH_PHYSICAL_OPERATOR_LANE_HVA_HAMILTONIAN_BLOCKS,
    "hh_termwise_quadrature": HH_PHYSICAL_OPERATOR_LANE_HVA_HAMILTONIAN_BLOCKS,
    "hva_layer": HH_PHYSICAL_OPERATOR_LANE_HVA_HAMILTONIAN_BLOCKS,
    "hh_hamiltonian_block": HH_PHYSICAL_OPERATOR_LANE_HVA_HAMILTONIAN_BLOCKS,
    "hh_fermionic_reusable": HH_PHYSICAL_OPERATOR_LANE_UCCSD_CORRELATION,
    "hh_phonon_linear": HH_PHYSICAL_OPERATOR_LANE_PHONON_DISPLACEMENT,
    "hh_phonon_quadratic": HH_PHYSICAL_OPERATOR_LANE_PHONON_SQUEEZE_RELAXATION,
    "hh_vlf_sq": HH_PHYSICAL_OPERATOR_LANE_PHONON_SQUEEZE_RELAXATION,
    "paop_cloud_p": HH_PHYSICAL_OPERATOR_LANE_PHONON_DISPLACEMENT,
    "paop_cloud_x": HH_PHYSICAL_OPERATOR_LANE_PHONON_DISPLACEMENT,
    "paop_disp": HH_PHYSICAL_OPERATOR_LANE_PHONON_DISPLACEMENT,
    "paop_dbl": HH_PHYSICAL_OPERATOR_LANE_DRESSED_PHONON_CORRELATION,
    "paop_hopdrag": HH_PHYSICAL_OPERATOR_LANE_DRESSED_PHONON_CORRELATION,
    "paop_dbl_p": HH_PHYSICAL_OPERATOR_LANE_DRESSED_PHONON_CORRELATION,
    "paop_dbl_x": HH_PHYSICAL_OPERATOR_LANE_DRESSED_PHONON_CORRELATION,
    "paop_curdrag": HH_PHYSICAL_OPERATOR_LANE_ELECTRONIC_CURRENT,
    "paop_hop2": HH_PHYSICAL_OPERATOR_LANE_DRESSED_PHONON_CORRELATION,
    "paop_other": HH_PHYSICAL_OPERATOR_LANE_DRESSED_PHONON_CORRELATION,
    "uccsd_paop_product": HH_PHYSICAL_OPERATOR_LANE_DRESSED_PHONON_CORRELATION,
    "uccsd_paop_product_seq_ferm": HH_PHYSICAL_OPERATOR_LANE_DRESSED_PHONON_CORRELATION,
    "uccsd_paop_product_seq_motif": HH_PHYSICAL_OPERATOR_LANE_DRESSED_PHONON_CORRELATION,
}


@dataclass(frozen=True)
class HHFullMetaClassFilterSpec:
    keep_classes: tuple[str, ...]
    classifier_version: str = HH_FULL_META_CLASSIFIER_VERSION
    source_pool: str = "full_meta"
    source_problem: str = "hh"
    source_num_sites: int | None = None
    source_n_ph_max: int | None = None
    source_json: str | None = None


@dataclass(frozen=True)
class HHFullMetaLabelFilterSpec:
    drop_labels: tuple[str, ...] = ()
    drop_prefixes: tuple[str, ...] = ()
    classifier_version: str = HH_FULL_META_CLASSIFIER_VERSION
    source_pool: str = "full_meta"
    source_problem: str = "hh"
    source_num_sites: int | None = None
    source_n_ph_max: int | None = None
    source_json: str | None = None


def classify_hh_full_meta_label(label: str) -> str | None:
    label_str = str(label)
    if any(
        label_str.startswith(prefix)
        for prefix in ("hop_layer", "onsite_layer", "potential_layer", "phonon_layer", "eph_layer", "drive_layer")
    ):
        return "hva_layer"
    if label_str.startswith("hh_termwise_ham_unit_term("):
        return "hh_termwise_unit"
    if label_str.startswith("hh_termwise_ham_quadrature_term("):
        return "hh_termwise_quadrature"
    if label_str.startswith("ham_block::"):
        return "hh_hamiltonian_block"
    if label_str.startswith("hh_fermionic_reusable::"):
        return "hh_fermionic_reusable"
    if label_str.startswith("hh_phonon::"):
        if any(label_str.startswith(f"hh_phonon::{name}(") for name in ("x", "p", "n")):
            return "hh_phonon_linear"
        if any(label_str.startswith(f"hh_phonon::{name}(") for name in ("s", "x_sq", "p_sq", "n_sq", "xp_sym")):
            return "hh_phonon_quadratic"
    if label_str.startswith("hh_vlf_sq::"):
        return "hh_vlf_sq"
    if label_str.startswith("uccsd_ferm_lifted::uccsd_sing("):
        return "uccsd_sing"
    if label_str.startswith("uccsd_ferm_lifted::uccsd_dbl("):
        return "uccsd_dbl"
    if label_str.startswith("uccsd_otimes_paop_seq2p::"):
        if label_str.endswith("::step=ferm"):
            return "uccsd_paop_product_seq_ferm"
        if label_str.endswith("::step=motif"):
            return "uccsd_paop_product_seq_motif"
    if label_str.startswith("uccsd_otimes_paop::"):
        return "uccsd_paop_product"
    if label_str.startswith("paop_"):
        if ":paop_cloud_p(" in label_str:
            return "paop_cloud_p"
        if ":paop_cloud_x(" in label_str:
            return "paop_cloud_x"
        if ":paop_disp(" in label_str:
            return "paop_disp"
        if ":paop_dbl(" in label_str:
            return "paop_dbl"
        if ":paop_hopdrag(" in label_str:
            return "paop_hopdrag"
        if ":paop_dbl_p(" in label_str:
            return "paop_dbl_p"
        if ":paop_dbl_x(" in label_str:
            return "paop_dbl_x"
        if ":paop_curdrag(" in label_str:
            return "paop_curdrag"
        if ":paop_hop2(" in label_str:
            return "paop_hop2"
        return "paop_other"
    return None


def classify_hh_physical_operator_lane(
    label: str,
    *,
    hh_full_meta_class: str | None = None,
) -> dict[str, str | None]:
    """Classify an HH full_meta operator label into a physical shortlist lane."""

    label_str = str(label)
    class_name = (
        str(hh_full_meta_class).strip()
        if hh_full_meta_class not in {None, ""}
        else classify_hh_full_meta_label(label_str)
    )
    if class_name == "":
        class_name = None
    lane = (
        _HH_FULL_META_CLASS_TO_PHYSICAL_OPERATOR_LANE.get(str(class_name))
        if class_name is not None
        else None
    )
    label_lower = label_str.lower()
    if lane in {
        HH_PHYSICAL_OPERATOR_LANE_UCCSD_SINGLE,
        HH_PHYSICAL_OPERATOR_LANE_UCCSD_DOUBLE,
        HH_PHYSICAL_OPERATOR_LANE_UCCSD_CORRELATION,
    } and (
        "current" in label_lower or "curdrag" in label_lower
    ):
        lane = HH_PHYSICAL_OPERATOR_LANE_ELECTRONIC_CURRENT
    if lane is None:
        lane = HH_PHYSICAL_OPERATOR_LANE_OTHER
    return {
        "schema": "hh_physical_operator_lane_classification_v1",
        "classifier_version": HH_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION,
        "label": label_str,
        "hh_full_meta_class": class_name,
        "physical_operator_lane": str(lane),
    }


def normalize_static_physical_operator_problem(problem: Any) -> str:
    raw = str(problem).strip().lower().replace("-", "_")
    aliases = {
        "hubbard_holstein": "hh",
        "spinboson": "spin_boson",
        "rabi": "spin_boson",
        "spin_boson_rabi": "spin_boson",
        "bosehubbard": "bose_hubbard",
        "h2o_linear_fd": "molecular_vibronic_h2o_linear_fd",
    }
    problem_key = aliases.get(raw, raw)
    if problem_key not in {
        "hh",
        "hubbard",
        "spin_boson",
        "bose_hubbard",
        "molecular_restricted_closed_shell",
        "molecular_vibronic_h2o_linear_fd",
    }:
        raise ValueError(
            "physical_operator_type lanes are defined for "
            "{'hh', 'hubbard', 'spin_boson', 'bose_hubbard', "
            "'molecular_restricted_closed_shell', "
            "'molecular_vibronic_h2o_linear_fd'}; "
            f"got {problem!r}."
        )
    return str(problem_key)


def physical_operator_lanes_for_problem(problem: Any) -> tuple[str, ...]:
    problem_key = normalize_static_physical_operator_problem(problem)
    if problem_key == "hh":
        return tuple(str(x) for x in HH_PHYSICAL_OPERATOR_LANES)
    if problem_key == "hubbard":
        return tuple(str(x) for x in HUBBARD_PHYSICAL_OPERATOR_LANES)
    if problem_key == "spin_boson":
        return tuple(str(x) for x in SPIN_BOSON_PHYSICAL_OPERATOR_LANES)
    if problem_key == "bose_hubbard":
        return tuple(str(x) for x in BOSE_HUBBARD_PHYSICAL_OPERATOR_LANES)
    if problem_key == "molecular_restricted_closed_shell":
        return tuple(str(x) for x in MOLECULAR_RESTRICTED_PHYSICAL_OPERATOR_LANES)
    if problem_key == "molecular_vibronic_h2o_linear_fd":
        return tuple(str(x) for x in H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANES)
    raise AssertionError(f"Unhandled physical-operator problem key {problem_key!r}")


def physical_operator_classifier_version_for_problem(problem: Any) -> str:
    problem_key = normalize_static_physical_operator_problem(problem)
    if problem_key == "hh":
        return str(HH_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION)
    if problem_key == "hubbard":
        return str(HUBBARD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION)
    if problem_key == "spin_boson":
        return str(SPIN_BOSON_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION)
    if problem_key == "bose_hubbard":
        return str(BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION)
    if problem_key == "molecular_restricted_closed_shell":
        return str(MOLECULAR_RESTRICTED_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION)
    if problem_key == "molecular_vibronic_h2o_linear_fd":
        return str(H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION)
    raise AssertionError(f"Unhandled physical-operator problem key {problem_key!r}")


def _full_meta_core_label(label: str) -> str:
    label_str = str(label)
    return label_str[len("full_meta::") :] if label_str.startswith("full_meta::") else label_str


def _static_physical_lane_payload(
    *,
    schema: str,
    classifier_version: str,
    problem: str,
    label: str,
    lane: str,
    family_class: str | None = None,
) -> dict[str, str | None]:
    return {
        "schema": schema,
        "classifier_version": str(classifier_version),
        "problem": str(problem),
        "label": str(label),
        "hh_full_meta_class": family_class,
        "physical_operator_lane": str(lane),
    }


def _classify_hubbard_physical_operator_lane(label: str) -> dict[str, str | None]:
    label_str = str(label)
    label_core = _HUBBARD_CHILD_SET_SUFFIX_RE.sub("", label_str)
    lane = HH_PHYSICAL_OPERATOR_LANE_OTHER
    if label_core.startswith(("uccsd_sing(", "uccsd_ferm_lifted::uccsd_sing(")):
        lane = HH_PHYSICAL_OPERATOR_LANE_UCCSD_SINGLE
    elif label_core.startswith(("uccsd_dbl(", "uccsd_ferm_lifted::uccsd_dbl(")):
        lane = HH_PHYSICAL_OPERATOR_LANE_UCCSD_DOUBLE
    elif _HUBBARD_QEB_SINGLE_LABEL_RE.match(label_core) or _HUBBARD_QEB_DOUBLE_LABEL_RE.match(label_core):
        lane = HUBBARD_PHYSICAL_OPERATOR_LANE_QEB_EXCITATION
    elif (
        _HUBBARD_HVA_BLOCK_LABEL_RE.match(label_core)
        or _HUBBARD_HAM_BLOCK_LABEL_RE.match(label_core)
        or label_core in {"hop_layer", "onsite_layer", "potential_layer"}
    ):
        lane = HUBBARD_PHYSICAL_OPERATOR_LANE_HVA_HAMILTONIAN_BLOCKS
    return _static_physical_lane_payload(
        schema="hubbard_physical_operator_lane_classification_v3",
        classifier_version=HUBBARD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION,
        problem="hubbard",
        label=label_str,
        lane=lane,
    )


def _classify_spin_boson_physical_operator_lane(label: str) -> dict[str, str | None]:
    label_str = str(label)
    core = _full_meta_core_label(label_str)
    lane = HH_PHYSICAL_OPERATOR_LANE_OTHER
    if core.startswith(("ham_full", "ham_term(", "ham_unit_term(")):
        lane = SPIN_BOSON_PHYSICAL_OPERATOR_LANE_HAMILTONIAN_BLOCKS
    elif core.startswith(("x_sq_emitter_y", "p_sq_emitter_y")):
        lane = SPIN_BOSON_PHYSICAL_OPERATOR_LANE_EMITTER_Y_CORRELATION
    elif core.startswith(
        (
            "longitudinal_coupling",
            "longitudinal_x",
            "longitudinal_p",
            "number_weighted_imbalance",
            "x_sq_imbalance",
            "p_sq_imbalance",
            "n_sq_imbalance",
            "n_x_imbalance",
            "n_p_imbalance",
        )
    ):
        lane = SPIN_BOSON_PHYSICAL_OPERATOR_LANE_LONGITUDINAL_COUPLING
    elif core.startswith(
        (
            "transverse_coupling",
            "transverse_x",
            "transverse_p",
            "number_weighted_flip",
            "x_sq_flip",
            "p_sq_flip",
            "n_sq_flip",
            "n_x_flip",
            "n_p_flip",
        )
    ):
        lane = SPIN_BOSON_PHYSICAL_OPERATOR_LANE_TRANSVERSE_COUPLING
    elif core.startswith(
        (
            "boson_x_sq",
            "boson_p_sq",
            "boson_n_sq",
            "boson_squeeze_x",
            "boson_xp_sym",
            "n_x",
            "n_p",
        )
    ):
        lane = SPIN_BOSON_PHYSICAL_OPERATOR_LANE_BOSON_NONLINEAR
    elif core.startswith(("boson_number", "boson_displacement", "boson_momentum")):
        lane = SPIN_BOSON_PHYSICAL_OPERATOR_LANE_BOSON_LINEAR
    elif core.startswith(("emitter_flip", "emitter_imbalance", "emitter_y")):
        lane = SPIN_BOSON_PHYSICAL_OPERATOR_LANE_EMITTER_MATTER
    return _static_physical_lane_payload(
        schema="spin_boson_physical_operator_lane_classification_v1",
        classifier_version=SPIN_BOSON_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION,
        problem="spin_boson",
        label=label_str,
        lane=lane,
    )


def _classify_bose_hubbard_physical_operator_lane(label: str) -> dict[str, str | None]:
    label_str = str(label)
    core = _full_meta_core_label(label_str)
    lane = HH_PHYSICAL_OPERATOR_LANE_OTHER
    if core.startswith(("ham_full", "ham_term(", "ham_unit_term(")):
        lane = BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_HAMILTONIAN_BLOCKS
    elif core.startswith(("density_hop_", "density_current_")):
        lane = BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_DENSITY_ASSISTED_TRANSPORT
    elif core.startswith(("pair_hop_", "pair_current_")):
        lane = BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_PAIR_TRANSPORT
    elif core.startswith(("hop_", "current_")):
        lane = BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_SINGLE_PARTICLE_TRANSPORT
    elif core.startswith(("xx_", "pp_")):
        lane = BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_INTERSITE_QUADRATURE
    elif core.startswith(
        (
            "x_",
            "p_",
            "x_sq_",
            "p_sq_",
            "squeeze_x_",
            "squeeze_p_",
            "n_x_",
            "n_p_",
            "n_x_sq_",
            "n_p_sq_",
        )
    ):
        lane = BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_ONSITE_QUADRATURE
    elif core.startswith(("n_", "n_sq_", "number_", "interaction_", "staggered_number_", "nn_")):
        lane = BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_NUMBER_DENSITY_INTERACTION
    return _static_physical_lane_payload(
        schema="bose_hubbard_physical_operator_lane_classification_v1",
        classifier_version=BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION,
        problem="bose_hubbard",
        label=label_str,
        lane=lane,
    )


def _classify_h2o_linear_fd_physical_operator_lane(
    label: str,
) -> dict[str, str | None]:
    label_str = str(label)
    core = _full_meta_core_label(label_str)
    lane = HH_PHYSICAL_OPERATOR_LANE_OTHER
    if core.startswith("el::uccsd_sing("):
        lane = H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_ELECTRONIC_SINGLE
    elif core.startswith("el::uccsd_dbl("):
        lane = H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_ELECTRONIC_DOUBLE
    elif core.startswith("boson::") and "::p" in core:
        lane = H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRATIONAL_MOMENTUM
    elif core.startswith("coupled::") and "::dH_dQ_one_body_factor[" in core:
        lane = H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_ONE_BODY_RESPONSE
    elif core.startswith("coupled::") and "::dH_dQ_two_body_factor[" in core:
        lane = H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_TWO_BODY_RESPONSE
    elif core.startswith("coupled::") and "::dH_dQ_times_p" in core:
        lane = (
            H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_DERIVATIVE_MOMENTUM
        )
    elif core.startswith("conditional::") and "::q_times_uccsd_sing(" in core:
        lane = (
            H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_CONDITIONAL_SINGLE
        )
    elif core.startswith("conditional::") and "::q_times_uccsd_dbl(" in core:
        lane = (
            H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_CONDITIONAL_DOUBLE
        )
    return _static_physical_lane_payload(
        schema="h2o_linear_fd_physical_operator_lane_classification_v2",
        classifier_version=H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION,
        problem="molecular_vibronic_h2o_linear_fd",
        label=label_str,
        lane=lane,
    )


def _classify_molecular_restricted_physical_operator_lane(
    label: str,
) -> dict[str, str | None]:
    label_str = str(label)
    core = _full_meta_core_label(label_str)
    lane = HH_PHYSICAL_OPERATOR_LANE_OTHER
    if core.startswith("uccsd_sing("):
        lane = H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_ELECTRONIC_SINGLE
    elif core.startswith("uccsd_dbl("):
        lane = H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_ELECTRONIC_DOUBLE
    return _static_physical_lane_payload(
        schema="molecular_restricted_physical_operator_lane_classification_v1",
        classifier_version=(
            MOLECULAR_RESTRICTED_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION
        ),
        problem="molecular_restricted_closed_shell",
        label=label_str,
        lane=lane,
    )


def classify_static_physical_operator_lane(
    label: str,
    *,
    problem: Any,
    hh_full_meta_class: str | None = None,
) -> dict[str, str | None]:
    problem_key = normalize_static_physical_operator_problem(problem)
    if problem_key == "hh":
        return classify_hh_physical_operator_lane(
            label,
            hh_full_meta_class=hh_full_meta_class,
        )
    if problem_key == "hubbard":
        return _classify_hubbard_physical_operator_lane(label)
    if problem_key == "spin_boson":
        return _classify_spin_boson_physical_operator_lane(label)
    if problem_key == "bose_hubbard":
        return _classify_bose_hubbard_physical_operator_lane(label)
    if problem_key == "molecular_restricted_closed_shell":
        return _classify_molecular_restricted_physical_operator_lane(label)
    if problem_key == "molecular_vibronic_h2o_linear_fd":
        return _classify_h2o_linear_fd_physical_operator_lane(label)
    raise AssertionError(f"Unhandled physical-operator problem key {problem_key!r}")


def summarize_static_physical_operator_pool_labels(
    labels: Sequence[Any],
    *,
    problem: Any,
) -> dict[str, Any]:
    """Summarize physical-operator lane classification over final emitted labels."""
    problem_key = normalize_static_physical_operator_problem(problem)
    lanes = tuple(str(x) for x in physical_operator_lanes_for_problem(problem_key))
    lane_counts = {str(lane): 0 for lane in lanes}
    exact_other_labels: list[str] = []
    classified_count = 0
    for raw in labels:
        label = str(raw)
        payload = classify_static_physical_operator_lane(label, problem=problem_key)
        lane = str(payload.get("physical_operator_lane", HH_PHYSICAL_OPERATOR_LANE_OTHER))
        if lane not in lane_counts:
            lane_counts[lane] = 0
        lane_counts[lane] = int(lane_counts.get(lane, 0)) + 1
        classified_count += 1
        if lane == HH_PHYSICAL_OPERATOR_LANE_OTHER:
            exact_other_labels.append(label)
    other_count = int(lane_counts.get(HH_PHYSICAL_OPERATOR_LANE_OTHER, 0))
    return {
        "schema": "static_physical_operator_pool_lane_audit_v1",
        "problem": str(problem_key),
        "classifier_version": physical_operator_classifier_version_for_problem(problem_key),
        "classified_count": int(classified_count),
        "lane_counts": lane_counts,
        "other_count": int(other_count),
        "exact_other_labels": exact_other_labels,
        "require_no_other_pass": bool(other_count == 0 and not exact_other_labels),
    }


def normalize_hh_full_meta_keep_classes(classes: Sequence[Any]) -> tuple[str, ...]:
    keep_classes: list[str] = []
    seen: set[str] = set()
    for raw in classes:
        name = str(raw).strip()
        if name == "":
            continue
        if name not in HH_FULL_META_ALLOWED_CLASSES:
            raise ValueError(
                "Unknown HH full_meta class "
                f"{name!r}; allowed classes are {list(HH_FULL_META_ALLOWED_CLASSES)}."
            )
        if name in seen:
            continue
        seen.add(name)
        keep_classes.append(name)
    if not keep_classes:
        raise ValueError("HH full_meta class filter must keep at least one class.")
    return tuple(keep_classes)


def _normalize_nonempty_unique_strings(items: Sequence[Any], *, field_name: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        value = str(raw).strip()
        if value == "":
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    if not out:
        raise ValueError(f"HH full_meta label filter field {field_name!r} must contain at least one non-empty string.")
    return tuple(out)


def load_hh_full_meta_class_filter_spec(path: Path) -> HHFullMetaClassFilterSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("HH full_meta class filter JSON must be an object with keep_classes.")
    keep_raw = raw.get("keep_classes")
    if not isinstance(keep_raw, list):
        raise ValueError("HH full_meta class filter JSON must contain list field 'keep_classes'.")
    classifier_version_raw = raw.get("classifier_version")
    if classifier_version_raw is None:
        raise ValueError("HH full_meta class filter JSON must contain string field 'classifier_version'.")
    classifier_version = str(classifier_version_raw).strip()
    if classifier_version != HH_FULL_META_CLASSIFIER_VERSION:
        raise ValueError(
            "HH full_meta class filter classifier_version mismatch: "
            f"got {classifier_version!r}, expected {HH_FULL_META_CLASSIFIER_VERSION!r}."
        )
    source_pool = str(raw.get("source_pool", "")).strip().lower()
    if source_pool != "full_meta":
        raise ValueError(
            "HH full_meta class filter JSON must declare source_pool='full_meta'."
        )
    source_problem = str(raw.get("source_problem", "hh")).strip().lower()
    if source_problem != "hh":
        raise ValueError(
            "HH full_meta class filter JSON must declare source_problem='hh'."
        )
    source_num_sites_raw = raw.get("source_num_sites")
    source_n_ph_max_raw = raw.get("source_n_ph_max")
    return HHFullMetaClassFilterSpec(
        keep_classes=normalize_hh_full_meta_keep_classes(keep_raw),
        classifier_version=str(classifier_version),
        source_pool=str(source_pool),
        source_problem=str(source_problem),
        source_num_sites=(
            None if source_num_sites_raw is None else int(source_num_sites_raw)
        ),
        source_n_ph_max=(
            None if source_n_ph_max_raw is None else int(source_n_ph_max_raw)
        ),
        source_json=str(path),
    )


def load_hh_full_meta_label_filter_spec(path: Path) -> HHFullMetaLabelFilterSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("HH full_meta label filter JSON must be an object.")
    drop_labels_raw = raw.get("drop_labels", [])
    drop_prefixes_raw = raw.get("drop_prefixes", [])
    if not isinstance(drop_labels_raw, list):
        raise ValueError("HH full_meta label filter JSON field 'drop_labels' must be a list.")
    if not isinstance(drop_prefixes_raw, list):
        raise ValueError("HH full_meta label filter JSON field 'drop_prefixes' must be a list.")
    if len(drop_labels_raw) == 0 and len(drop_prefixes_raw) == 0:
        raise ValueError("HH full_meta label filter JSON must contain non-empty 'drop_labels' or 'drop_prefixes'.")
    classifier_version_raw = raw.get("classifier_version")
    if classifier_version_raw is None:
        raise ValueError("HH full_meta label filter JSON must contain string field 'classifier_version'.")
    classifier_version = str(classifier_version_raw).strip()
    if classifier_version != HH_FULL_META_CLASSIFIER_VERSION:
        raise ValueError(
            "HH full_meta label filter classifier_version mismatch: "
            f"got {classifier_version!r}, expected {HH_FULL_META_CLASSIFIER_VERSION!r}."
        )
    source_pool = str(raw.get("source_pool", "")).strip().lower()
    if source_pool != "full_meta":
        raise ValueError(
            "HH full_meta label filter JSON must declare source_pool='full_meta'."
        )
    source_problem = str(raw.get("source_problem", "hh")).strip().lower()
    if source_problem != "hh":
        raise ValueError(
            "HH full_meta label filter JSON must declare source_problem='hh'."
        )
    source_num_sites_raw = raw.get("source_num_sites")
    source_n_ph_max_raw = raw.get("source_n_ph_max")
    return HHFullMetaLabelFilterSpec(
        drop_labels=(
            _normalize_nonempty_unique_strings(drop_labels_raw, field_name="drop_labels")
            if len(drop_labels_raw) > 0
            else tuple()
        ),
        drop_prefixes=(
            _normalize_nonempty_unique_strings(drop_prefixes_raw, field_name="drop_prefixes")
            if len(drop_prefixes_raw) > 0
            else tuple()
        ),
        classifier_version=str(classifier_version),
        source_pool=str(source_pool),
        source_problem=str(source_problem),
        source_num_sites=(
            None if source_num_sites_raw is None else int(source_num_sites_raw)
        ),
        source_n_ph_max=(
            None if source_n_ph_max_raw is None else int(source_n_ph_max_raw)
        ),
        source_json=str(path),
    )


def summarize_hh_full_meta_pool_classes(pool: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for term in pool:
        family = classify_hh_full_meta_label(str(term.label))
        if family is None:
            raise ValueError(f"Unable to classify HH full_meta operator label {term.label!r}.")
        counts[family] = int(counts.get(family, 0) + 1)
    return {
        family: int(counts[family])
        for family in HH_FULL_META_ALLOWED_CLASSES
        if family in counts
    }


__all__ = [
    "BOSE_HUBBARD_PHYSICAL_OPERATOR_LANES",
    "BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION",
    "BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_DENSITY_ASSISTED_TRANSPORT",
    "BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_HAMILTONIAN_BLOCKS",
    "BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_INTERSITE_QUADRATURE",
    "BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_NUMBER_DENSITY_INTERACTION",
    "BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_ONSITE_QUADRATURE",
    "BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_PAIR_TRANSPORT",
    "BOSE_HUBBARD_PHYSICAL_OPERATOR_LANE_SINGLE_PARTICLE_TRANSPORT",
    "HUBBARD_PHYSICAL_OPERATOR_LANES",
    "HUBBARD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION",
    "HUBBARD_PHYSICAL_OPERATOR_LANE_HVA_HAMILTONIAN_BLOCKS",
    "HUBBARD_PHYSICAL_OPERATOR_LANE_QEB_EXCITATION",
    "H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANES",
    "H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION",
    "H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_ELECTRONIC_DOUBLE",
    "H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_ELECTRONIC_SINGLE",
    "H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRATIONAL_MOMENTUM",
    "H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_CONDITIONAL_DOUBLE",
    "H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_CONDITIONAL_SINGLE",
    "H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_DERIVATIVE_MOMENTUM",
    "H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_ONE_BODY_RESPONSE",
    "H2O_LINEAR_FD_PHYSICAL_OPERATOR_LANE_VIBRONIC_TWO_BODY_RESPONSE",
    "HHFullMetaClassFilterSpec",
    "HHFullMetaLabelFilterSpec",
    "HH_FULL_META_ALLOWED_CLASSES",
    "HH_FULL_META_BUILDER_FUNCTION",
    "HH_FULL_META_BUILDER_MODULE",
    "HH_FULL_META_CLASSIFIER_VERSION",
    "HH_FULL_META_OPERATOR_BASIS_SOURCES",
    "HH_FULL_META_PROVENANCE_SCHEMA_VERSION",
    "HH_FULL_META_REQUIRED_LAYOUT_FIELDS",
    "HH_PHYSICAL_OPERATOR_LANES",
    "HH_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION",
    "HH_PHYSICAL_OPERATOR_LANE_DRESSED_PHONON_CORRELATION",
    "HH_PHYSICAL_OPERATOR_LANE_ELECTRONIC_CURRENT",
    "HH_PHYSICAL_OPERATOR_LANE_HVA_HAMILTONIAN_BLOCKS",
    "HH_PHYSICAL_OPERATOR_LANE_OTHER",
    "HH_PHYSICAL_OPERATOR_LANE_PHONON_DISPLACEMENT",
    "HH_PHYSICAL_OPERATOR_LANE_PHONON_SQUEEZE_RELAXATION",
    "HH_PHYSICAL_OPERATOR_LANE_UCCSD_CORRELATION",
    "HH_PHYSICAL_OPERATOR_LANE_UCCSD_DOUBLE",
    "HH_PHYSICAL_OPERATOR_LANE_UCCSD_SINGLE",
    "HH_MATH_MD_FULL_META_DISPLAY_NAME",
    "HH_MATH_MD_FULL_META_POOL_ALIASES",
    "HH_MATH_MD_FULL_META_POOL_KEY",
    "SPIN_BOSON_PHYSICAL_OPERATOR_LANES",
    "SPIN_BOSON_PHYSICAL_OPERATOR_LANE_BOSON_LINEAR",
    "SPIN_BOSON_PHYSICAL_OPERATOR_LANE_BOSON_NONLINEAR",
    "SPIN_BOSON_PHYSICAL_OPERATOR_LANE_CLASSIFIER_VERSION",
    "SPIN_BOSON_PHYSICAL_OPERATOR_LANE_EMITTER_MATTER",
    "SPIN_BOSON_PHYSICAL_OPERATOR_LANE_EMITTER_Y_CORRELATION",
    "SPIN_BOSON_PHYSICAL_OPERATOR_LANE_HAMILTONIAN_BLOCKS",
    "SPIN_BOSON_PHYSICAL_OPERATOR_LANE_LONGITUDINAL_COUPLING",
    "SPIN_BOSON_PHYSICAL_OPERATOR_LANE_TRANSVERSE_COUPLING",
    "classify_hh_full_meta_label",
    "classify_hh_physical_operator_lane",
    "classify_static_physical_operator_lane",
    "load_hh_full_meta_class_filter_spec",
    "load_hh_full_meta_label_filter_spec",
    "normalize_hh_full_meta_keep_classes",
    "normalize_static_physical_operator_problem",
    "physical_operator_classifier_version_for_problem",
    "physical_operator_lanes_for_problem",
    "summarize_hh_full_meta_pool_classes",
    "summarize_static_physical_operator_pool_labels",
]
