"""CLI entrypoint for the isolated QSE spectra sidecar."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

HH_CURRENT_EDGE_ORIENTATIONS = ("positive_chain",)
HH_CURRENT_PEIERLS_POLICY = "standard_hh_1d_charge_peierls"
HH_CURRENT_CONTACT_POLICY = "peierls_second_derivative_record_only"

from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    QSEPruningConfig,
    computational_basis_state,
    compute_qse_spectra,
)
from pipelines.qse_spectra.io import (
    basis_elements_from_artifact_source,
    basis_elements_from_labels,
    load_operator_basis_json,
    load_polynomial_json,
    load_state_json,
    load_transition_observables_json,
    qse_result_to_manifest,
    transition_observables_from_labels,
    write_manifest_json,
)
from pipelines.qse_spectra.production_contracts import (
    HH_FULL_META_OPERATOR_BASIS_SOURCES,
    PAPER_III_RUN_CLASSES,
    PaperIIIProductionContractError,
    build_paper_iii_contract,
    resolve_hh_full_meta_provenance,
)
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    finalize_static_record_selection_payload,
    select_static_qse_records,
)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute an ideal/statevector Quantum Subspace Expansion spectrum."
    )
    parser.add_argument("--hamiltonian-json", type=Path, required=True)
    state_group = parser.add_mutually_exclusive_group(required=True)
    state_group.add_argument("--state-json", type=Path, default=None)
    state_group.add_argument("--state-bitstring", type=str, default=None)
    parser.add_argument(
        "--state-json-key",
        choices=["auto", "initial_state", "ansatz_input_state"],
        default="auto",
        help="State block to read when --state-json is a full artifact.",
    )

    basis_group = parser.add_mutually_exclusive_group()
    basis_group.add_argument(
        "--operator-basis-label",
        action="append",
        default=None,
        help="Repeatable Pauli operator-basis label; accepts exyz or IXYZ. If supplied, identity is not auto-prepended.",
    )
    basis_group.add_argument("--operator-basis-json", type=Path, default=None)
    basis_group.add_argument(
        "--operator-basis-source",
        choices=["selected_adapt_blocks", "full_meta", "full_meta_filtered", "hamiltonian_terms"],
        default=None,
        help=(
            "Build the QSE basis from an ADAPT/HH artifact. "
            "selected_adapt_blocks uses adapt_vqe.parameterization.blocks or adapt_vqe.operators; "
            "full_meta builds the HH full_meta pool; full_meta_filtered uses a compact default class subset; "
            "hamiltonian_terms uses non-identity Hamiltonian Pauli terms."
        ),
    )
    parser.add_argument(
        "--basis-artifact-json",
        type=Path,
        default=None,
        help="Artifact to use for --operator-basis-source. Defaults to --state-json when available, else --hamiltonian-json.",
    )
    parser.add_argument(
        "--seed-manifest-json",
        type=Path,
        default=None,
        help="Explicit static seed/artifact manifest for Paper III full_meta provenance.",
    )
    parser.add_argument(
        "--full-meta-keep-classes",
        type=str,
        default=None,
        help="Comma-separated full_meta classes to retain for --operator-basis-source full_meta/full_meta_filtered.",
    )
    parser.add_argument(
        "--include-hamiltonian-term-basis",
        action="store_true",
        help="Append non-identity Hamiltonian Pauli terms to artifact-derived bases.",
    )

    parser.add_argument("--static-record-selection-mode", choices=["input_order", "cost_proxy", "geometry_selected", "compiled_cost"], default=None)
    parser.add_argument(
        "--static-record-selection-compiled-cost-oracle",
        choices=["marrakesh_graph_span_v1", "backend_transpile_single_v1"],
        default=None,
        help=(
            "Annotate candidates with Paper I compiled hardware costs from this oracle. "
            "Required by mode compiled_cost (defaults there to marrakesh_graph_span_v1); "
            "with geometry_selected it replaces the structural cost proxy in the cost term."
        ),
    )
    parser.add_argument(
        "--static-record-selection-geometry-cost-discount-alpha",
        type=float,
        default=None,
        help="Optional multiplicative geometry cost discount: score = utility / cost^alpha (default off).",
    )
    parser.add_argument(
        "--static-record-selection-cost-weights",
        choices=["canonical_paper_i_v1", "two_qubit_only_v1"],
        default="canonical_paper_i_v1",
        help=(
            "Scalarization preset for compiled costs: the canonical Paper I lambda blend, "
            "or two_qubit_only_v1 (cost = compiled 2Q gate coordinate exactly)."
        ),
    )
    parser.add_argument(
        "--static-record-selection-cost-frontier",
        action="store_true",
        help="Emit the accuracy-versus-compiled-cost frontier over admitted prefixes of the selected basis.",
    )
    parser.add_argument("--static-record-selection-max-records", type=int, default=None)
    parser.add_argument("--static-record-selection-max-term-count", type=int, default=None)
    parser.add_argument("--static-record-selection-max-pauli-weight", type=int, default=None)
    parser.add_argument("--static-record-selection-min-retained-rank", type=int, default=None)
    parser.add_argument("--static-record-selection-max-overlap-condition", type=float, default=None)
    parser.add_argument("--static-record-selection-geometry-target-roots", type=int, default=None)
    parser.add_argument("--static-record-selection-geometry-metric-novelty-weight", type=float, default=None)
    parser.add_argument("--static-record-selection-geometry-residual-weight", type=float, default=None)
    parser.add_argument("--static-record-selection-geometry-ritz-weight", type=float, default=None)
    parser.add_argument("--static-record-selection-geometry-transition-weight", type=float, default=None)
    parser.add_argument("--static-record-selection-geometry-cost-weight", type=float, default=None)
    parser.add_argument("--static-record-selection-geometry-condition-penalty-weight", type=float, default=None)
    parser.add_argument("--static-record-selection-geometry-min-metric-novelty", type=float, default=None)

    parser.add_argument(
        "--paper-iii-static-qse-mode",
        action="store_true",
        help="Opt in to Paper III static QSE defaults: Q0 projection and raw projected basis vectors.",
    )
    parser.add_argument(
        "--paper-iii-run-class",
        choices=list(PAPER_III_RUN_CLASSES),
        default="candidate",
        help="Paper III production contract run class. Emitted only with --paper-iii-static-qse-mode.",
    )
    parser.add_argument(
        "--paper-iii-visible-target",
        type=str,
        default="tab:qse_static_claims",
        help="Visible Paper III table/figure/claim target for the production contract.",
    )
    parser.add_argument(
        "--paper-iii-compatibility-tier",
        type=str,
        default="not_evaluated",
        help="Paper III compatibility tier recorded in the production contract.",
    )
    parser.add_argument("--reference-projection", choices=["none", "q0"], default=None)
    parser.add_argument("--basis-vector-normalization", choices=["normalized", "raw_projected"], default=None)
    parser.add_argument(
        "--sector-label",
        type=str,
        default=None,
        help="Diagnostic sector label only; sector projection is identity in this slice.",
    )
    parser.add_argument(
        "--transition-observable-label",
        action="append",
        default=None,
        help="Repeatable Pauli transition observable label, e.g. X or dipole=X.",
    )
    parser.add_argument(
        "--transition-observable-json",
        action="append",
        type=Path,
        default=None,
        help="Optional JSON list of Pauli-string/polynomial transition observables.",
    )
    parser.add_argument(
        "--hh-neutral-response-channel",
        action="append",
        default=None,
        help=(
            "Repeatable/comma-separated HH neutral response channel name. "
            "Supported code-facing names: nn, XX, PP, nX, C_nX, all. "
            "Requires resolvable HH layout settings; otherwise use explicit --transition-observable-json."
        ),
    )
    parser.add_argument(
        "--hh-response-form-factor",
        type=str,
        default="staggered",
        help=(
            "HH response form factor for n[f], X[f], and P[f]. "
            "Use staggered (default), staggered_normalized, uniform, uniform_normalized, "
            "site:<i>, obc_sine:<m>, or csv:w0,w1,... ."
        ),
    )
    parser.add_argument(
        "--hh-response-density-baseline",
        type=float,
        default=None,
        help="Optional explicit bar_n for HH density fluctuation; default is inferred from the prepared state.",
    )
    parser.add_argument(
        "--hh-response-nx-separation",
        type=int,
        default=0,
        help="Relative displacement r for HH C_nX(r) composite channels.",
    )
    parser.add_argument(
        "--conductivity-response",
        action="store_true",
        help="Emit additive qse_conductivity_response_v1 current/conductivity postprocessing.",
    )
    parser.add_argument(
        "--conductivity-current-label",
        action="append",
        default=None,
        help="Repeatable current transition-observable name to use for qse_conductivity_response_v1.",
    )
    parser.add_argument(
        "--conductivity-contact-label",
        type=str,
        default=None,
        help="Optional contact/diamagnetic transition-observable name; recorded only, not combined into Drude delta.",
    )
    parser.add_argument("--conductivity-omega-floor", type=float, default=1.0e-12)
    parser.add_argument(
        "--conductivity-contact-policy",
        choices=["contact_record_only_no_drude_delta"],
        default="contact_record_only_no_drude_delta",
    )
    parser.add_argument(
        "--conductivity-peierls-policy",
        choices=[HH_CURRENT_PEIERLS_POLICY],
        default=HH_CURRENT_PEIERLS_POLICY,
    )
    parser.add_argument(
        "--hh-current-response",
        action="store_true",
        help=(
            "Generate HH charge current/contact observables and emit qse_conductivity_response_v1. "
            "Requires resolvable HH layout settings; otherwise provide explicit transition observables."
        ),
    )
    parser.add_argument(
        "--hh-current-hopping-amplitude",
        type=float,
        default=None,
        help="Optional explicit HH hopping t for current/contact observables; default resolves from HH settings.",
    )
    parser.add_argument(
        "--hh-current-edge-orientation",
        choices=list(HH_CURRENT_EDGE_ORIENTATIONS),
        default="positive_chain",
        help="Directed edge orientation convention for HH current observables.",
    )
    parser.add_argument(
        "--hh-current-contact-policy",
        choices=[HH_CURRENT_CONTACT_POLICY],
        default=HH_CURRENT_CONTACT_POLICY,
        help="Peierls contact-observable convention; fail-closed to the supported explicit policy.",
    )
    parser.add_argument(
        "--hh-current-peierls-policy",
        choices=[HH_CURRENT_PEIERLS_POLICY],
        default=HH_CURRENT_PEIERLS_POLICY,
        help="Peierls phase convention for HH current/contact observables.",
    )
    parser.add_argument(
        "--hh-current-disable-contact",
        action="store_true",
        help="Build HH current only; conductivity payload records contact as not supplied.",
    )
    parser.add_argument(
        "--green-functions",
        "--green-function",
        dest="green_functions",
        action="store_true",
        help="Emit additive qse_green_function_v1 single-particle retarded Green-function postprocessing.",
    )
    parser.add_argument(
        "--green-function-mode",
        action="append",
        default=None,
        help="Repeatable fermionic mode spec: mode, label=mode, or label:mode. Modes use repo JW qubit indices.",
    )
    parser.add_argument(
        "--green-function-fermion-qubits",
        type=int,
        default=None,
        help=(
            "Required with --green-functions: number of fermionic JW modes in the total register. "
            "Valid Green-function mode indices are 0..N_f-1; use this to exclude boson encoding qubits."
        ),
    )

    parser.add_argument(
        "--response-functions",
        "--response-function",
        dest="response_functions",
        action="store_true",
        help="Emit additive qse_response_functions_v1 neutral response postprocessing.",
    )
    parser.add_argument(
        "--response-channel",
        action="append",
        default=None,
        help="Repeatable response channel as A:B or A:B:channel_kind using transition observable names.",
    )
    parser.add_argument("--response-time-grid-min", type=float, default=None)
    parser.add_argument("--response-time-grid-max", type=float, default=None)
    parser.add_argument("--response-time-grid-num", type=int, default=None)
    parser.add_argument("--response-moment-max-order", type=int, default=1)
    parser.add_argument(
        "--response-disable-sum-rules",
        action="store_true",
        help="Do not evaluate direct m0/m1 sum-rule targets even when state and Hamiltonian are available.",
    )

    parser.add_argument("--spectral-grid-min", type=float, default=None)
    parser.add_argument("--spectral-grid-max", type=float, default=None)
    parser.add_argument("--spectral-grid-num", type=int, default=None)
    parser.add_argument("--spectral-eta", type=float, default=None)
    parser.add_argument("--spectral-kernel", choices=["lorentzian", "gaussian"], default=None)
    parser.add_argument(
        "--spectral-window",
        action="append",
        default=None,
        help="Repeatable diagnostic spectral window as min:max or name:min:max.",
    )
    parser.add_argument(
        "--spectral-reference-json",
        type=Path,
        default=None,
        help="Optional diagnostic reference spectrum JSON for spectral-window comparison.",
    )

    parser.add_argument("--cutoff-boundary-diagnostics", action="store_true")
    parser.add_argument("--cutoff-num-sites", type=int, default=None)
    parser.add_argument("--cutoff-n-ph-max", type=int, default=None)
    parser.add_argument("--cutoff-boson-encoding", choices=["binary", "unary"], default=None)
    parser.add_argument("--cutoff-fermion-qubits", type=int, default=0)

    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--overlap-relative-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--overlap-absolute-cutoff", type=float, default=1.0e-12)
    parser.add_argument("--overlap-negative-absolute-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--overlap-negative-relative-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--hermitian-absolute-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--hermitian-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--hamiltonian-coeff-imag-absolute-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--polynomial-drop-abs-tol", type=float, default=1.0e-15)
    parser.add_argument("--omit-matrices", action="store_true")
    return parser


def _default_output_path(hamiltonian_json: Path) -> Path:
    stem = Path(hamiltonian_json).with_suffix("")
    return stem.with_name(f"{stem.name}_qse_spectra.json")


def _hamiltonian_nq(poly) -> int:
    terms = poly.return_polynomial()
    if not terms:
        raise ValueError("Hamiltonian polynomial is empty.")
    return int(terms[0].nqubit())


def _spectral_mode_requested(args: argparse.Namespace) -> bool:
    return any(
        item is not None
        for item in (
            args.spectral_grid_min,
            args.spectral_grid_max,
            args.spectral_grid_num,
            args.spectral_eta,
            args.spectral_kernel,
            args.spectral_window,
            args.spectral_reference_json,
        )
    )


def _response_mode_requested(args: argparse.Namespace) -> bool:
    return bool(args.response_functions) or bool(args.response_channel) or bool(args.hh_neutral_response_channel) or any(
        item is not None
        for item in (
            args.response_time_grid_min,
            args.response_time_grid_max,
            args.response_time_grid_num,
        )
    )


def _conductivity_mode_requested(args: argparse.Namespace) -> bool:
    return (
        bool(args.conductivity_response)
        or bool(args.conductivity_current_label)
        or args.conductivity_contact_label is not None
        or bool(args.hh_current_response)
    )


def _green_function_mode_requested(args: argparse.Namespace) -> bool:
    return bool(args.green_functions) or bool(args.green_function_mode)


def _validate_postprocessing_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[bool, bool, bool, bool]:
    spectral_requested = _spectral_mode_requested(args)
    response_requested = _response_mode_requested(args)
    conductivity_requested = _conductivity_mode_requested(args)
    green_function_requested = _green_function_mode_requested(args)
    if spectral_requested:
        missing = [
            flag
            for flag, value in (
                ("--spectral-grid-min", args.spectral_grid_min),
                ("--spectral-grid-max", args.spectral_grid_max),
                ("--spectral-grid-num", args.spectral_grid_num),
                ("--spectral-eta", args.spectral_eta),
            )
            if value is None
        ]
        if missing:
            parser.error("spectral mode requires " + ", ".join(missing))
        if args.spectral_reference_json is not None and not args.spectral_window:
            parser.error("--spectral-reference-json requires at least one --spectral-window.")
    if response_requested:
        missing = [
            flag
            for flag, value in (
                ("--spectral-grid-min", args.spectral_grid_min),
                ("--spectral-grid-max", args.spectral_grid_max),
                ("--spectral-grid-num", args.spectral_grid_num),
                ("--spectral-eta", args.spectral_eta),
                ("--response-time-grid-min", args.response_time_grid_min),
                ("--response-time-grid-max", args.response_time_grid_max),
                ("--response-time-grid-num", args.response_time_grid_num),
            )
            if value is None
        ]
        if missing:
            parser.error("response function mode requires " + ", ".join(missing))
        if args.response_moment_max_order < 0:
            parser.error("--response-moment-max-order must be non-negative.")
    if conductivity_requested:
        missing = [
            flag
            for flag, value in (
                ("--spectral-grid-min", args.spectral_grid_min),
                ("--spectral-grid-max", args.spectral_grid_max),
                ("--spectral-grid-num", args.spectral_grid_num),
                ("--spectral-eta", args.spectral_eta),
            )
            if value is None
        ]
        if missing:
            parser.error("conductivity response mode requires " + ", ".join(missing))
        if float(args.conductivity_omega_floor) <= 0.0:
            parser.error("--conductivity-omega-floor must be positive.")
        if args.conductivity_contact_label is not None and not args.conductivity_current_label and not args.hh_current_response:
            parser.error("--conductivity-contact-label requires --conductivity-current-label or --hh-current-response.")
    if green_function_requested:
        missing = [
            flag
            for flag, value in (
                ("--spectral-grid-min", args.spectral_grid_min),
                ("--spectral-grid-max", args.spectral_grid_max),
                ("--spectral-grid-num", args.spectral_grid_num),
                ("--spectral-eta", args.spectral_eta),
            )
            if value is None
        ]
        if missing:
            parser.error("green function mode requires " + ", ".join(missing))
        if not args.green_function_mode:
            parser.error("green function mode requires at least one --green-function-mode.")
        if args.green_function_fermion_qubits is None:
            parser.error("green function mode requires --green-function-fermion-qubits to define the fermionic mode domain.")
        if int(args.green_function_fermion_qubits) <= 0:
            parser.error("--green-function-fermion-qubits must be positive.")
        if args.spectral_kernel is not None and str(args.spectral_kernel) != "lorentzian":
            parser.error("green function mode requires --spectral-kernel lorentzian because eta is used as +i*eta.")
    if bool(args.cutoff_boundary_diagnostics):
        missing = [
            flag
            for flag, value in (
                ("--cutoff-num-sites", args.cutoff_num_sites),
                ("--cutoff-n-ph-max", args.cutoff_n_ph_max),
                ("--cutoff-boson-encoding", args.cutoff_boson_encoding),
            )
            if value is None
        ]
        if missing:
            parser.error("--cutoff-boundary-diagnostics requires " + ", ".join(missing))
    return spectral_requested, response_requested, conductivity_requested, green_function_requested


def _build_static_record_selection_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> StaticRecordSelectionConfig | None:
    dependent_flags = (
        ("--static-record-selection-max-records", args.static_record_selection_max_records),
        ("--static-record-selection-max-term-count", args.static_record_selection_max_term_count),
        ("--static-record-selection-max-pauli-weight", args.static_record_selection_max_pauli_weight),
        ("--static-record-selection-min-retained-rank", args.static_record_selection_min_retained_rank),
        ("--static-record-selection-max-overlap-condition", args.static_record_selection_max_overlap_condition),
        ("--static-record-selection-geometry-target-roots", args.static_record_selection_geometry_target_roots),
        (
            "--static-record-selection-geometry-metric-novelty-weight",
            args.static_record_selection_geometry_metric_novelty_weight,
        ),
        ("--static-record-selection-geometry-residual-weight", args.static_record_selection_geometry_residual_weight),
        ("--static-record-selection-geometry-ritz-weight", args.static_record_selection_geometry_ritz_weight),
        (
            "--static-record-selection-geometry-transition-weight",
            args.static_record_selection_geometry_transition_weight,
        ),
        ("--static-record-selection-geometry-cost-weight", args.static_record_selection_geometry_cost_weight),
        (
            "--static-record-selection-geometry-condition-penalty-weight",
            args.static_record_selection_geometry_condition_penalty_weight,
        ),
        (
            "--static-record-selection-geometry-min-metric-novelty",
            args.static_record_selection_geometry_min_metric_novelty,
        ),
        (
            "--static-record-selection-compiled-cost-oracle",
            args.static_record_selection_compiled_cost_oracle,
        ),
        (
            "--static-record-selection-geometry-cost-discount-alpha",
            args.static_record_selection_geometry_cost_discount_alpha,
        ),
        (
            "--static-record-selection-cost-frontier",
            True if args.static_record_selection_cost_frontier else None,
        ),
    )
    supplied_dependents = [flag for flag, value in dependent_flags if value is not None]
    if args.static_record_selection_mode is None:
        if supplied_dependents:
            parser.error(
                "static record selection dependent flags require --static-record-selection-mode: "
                + ", ".join(supplied_dependents)
            )
        return None
    if args.static_record_selection_max_records is None:
        parser.error("--static-record-selection-mode requires --static-record-selection-max-records.")
    try:
        return StaticRecordSelectionConfig(
            mode=str(args.static_record_selection_mode),
            max_records=int(args.static_record_selection_max_records),
            max_term_count=args.static_record_selection_max_term_count,
            max_pauli_weight=args.static_record_selection_max_pauli_weight,
            min_retained_rank=args.static_record_selection_min_retained_rank,
            max_overlap_condition=args.static_record_selection_max_overlap_condition,
            geometry_target_roots=(
                int(args.static_record_selection_geometry_target_roots)
                if args.static_record_selection_geometry_target_roots is not None
                else 6
            ),
            geometry_metric_novelty_weight=(
                float(args.static_record_selection_geometry_metric_novelty_weight)
                if args.static_record_selection_geometry_metric_novelty_weight is not None
                else 0.25
            ),
            geometry_residual_weight=(
                float(args.static_record_selection_geometry_residual_weight)
                if args.static_record_selection_geometry_residual_weight is not None
                else 1.0
            ),
            geometry_ritz_weight=(
                float(args.static_record_selection_geometry_ritz_weight)
                if args.static_record_selection_geometry_ritz_weight is not None
                else 0.25
            ),
            geometry_transition_weight=(
                float(args.static_record_selection_geometry_transition_weight)
                if args.static_record_selection_geometry_transition_weight is not None
                else 0.5
            ),
            geometry_cost_weight=(
                float(args.static_record_selection_geometry_cost_weight)
                if args.static_record_selection_geometry_cost_weight is not None
                else 1.0
            ),
            geometry_condition_penalty_weight=(
                float(args.static_record_selection_geometry_condition_penalty_weight)
                if args.static_record_selection_geometry_condition_penalty_weight is not None
                else 0.05
            ),
            geometry_min_metric_novelty=(
                float(args.static_record_selection_geometry_min_metric_novelty)
                if args.static_record_selection_geometry_min_metric_novelty is not None
                else 1.0e-12
            ),
            geometry_cost_discount_alpha=(
                float(args.static_record_selection_geometry_cost_discount_alpha)
                if args.static_record_selection_geometry_cost_discount_alpha is not None
                else None
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))
    raise AssertionError("parser.error should have exited")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    spectral_requested, response_requested, conductivity_requested, green_function_requested = _validate_postprocessing_args(parser, args)
    static_record_selection_config = _build_static_record_selection_config(parser, args)

    config = QSEPruningConfig(
        overlap_relative_cutoff=float(args.overlap_relative_cutoff),
        overlap_absolute_cutoff=float(args.overlap_absolute_cutoff),
        overlap_negative_absolute_tolerance=float(args.overlap_negative_absolute_tolerance),
        overlap_negative_relative_tolerance=float(args.overlap_negative_relative_tolerance),
        hermitian_absolute_tolerance=float(args.hermitian_absolute_tolerance),
        hermitian_relative_tolerance=float(args.hermitian_relative_tolerance),
        hamiltonian_coeff_imag_absolute_tolerance=float(args.hamiltonian_coeff_imag_absolute_tolerance),
        polynomial_drop_abs_tol=float(args.polynomial_drop_abs_tol),
    )

    has_explicit_basis = any(
        item is not None
        for item in (args.operator_basis_label, args.operator_basis_json, args.operator_basis_source)
    )

    reference_projection = args.reference_projection
    if reference_projection is None:
        reference_projection = "q0" if bool(args.paper_iii_static_qse_mode) else "none"
    basis_vector_normalization = args.basis_vector_normalization
    if basis_vector_normalization is None:
        basis_vector_normalization = "raw_projected" if bool(args.paper_iii_static_qse_mode) else "normalized"
    if bool(args.paper_iii_static_qse_mode) and (
        reference_projection != "q0" or basis_vector_normalization != "raw_projected"
    ):
        parser.error("--paper-iii-static-qse-mode requires reference_projection=q0 and basis_vector_normalization=raw_projected.")
    if reference_projection == "q0" and not has_explicit_basis:
        parser.error("--reference-projection q0 requires an explicit operator basis label/json/source.")
    if bool(args.paper_iii_static_qse_mode) and args.operator_basis_source in HH_FULL_META_OPERATOR_BASIS_SOURCES:
        if args.basis_artifact_json is None:
            parser.error(
                "--paper-iii-static-qse-mode with HH full_meta/full_meta_filtered requires explicit --basis-artifact-json."
            )
        if args.seed_manifest_json is None:
            parser.error(
                "--paper-iii-static-qse-mode with HH full_meta/full_meta_filtered requires explicit --seed-manifest-json."
            )
    basis_vector_policy = QSEBasisVectorPolicy(
        reference_projection=str(reference_projection),
        basis_vector_normalization=str(basis_vector_normalization),
        sector_projection="identity",
        sector_label=args.sector_label,
    )

    hamiltonian, hamiltonian_provenance = load_polynomial_json(
        args.hamiltonian_json,
        drop_abs_tol=float(config.polynomial_drop_abs_tol),
        require_real_coefficients=True,
        coeff_imag_abs_tol=float(config.hamiltonian_coeff_imag_absolute_tolerance),
    )
    nq = _hamiltonian_nq(hamiltonian)

    if args.state_json is not None:
        prepared_state, state_provenance = load_state_json(
            args.state_json,
            expected_nq=int(nq),
            state_key=str(args.state_json_key),
        )
    else:
        prepared_state = computational_basis_state(int(nq), str(args.state_bitstring))
        state_provenance = {
            "source_schema": "computational_basis_state",
            "state_bitstring": str(args.state_bitstring),
            "nq_total": int(nq),
        }

    resolved_basis_artifact_json: Path | None = None
    if args.operator_basis_json is not None:
        basis, basis_provenance = load_operator_basis_json(args.operator_basis_json, nq=int(nq))
    elif args.operator_basis_label is not None:
        basis = basis_elements_from_labels(args.operator_basis_label, nq=int(nq))
        basis_provenance = {
            "source_schema": "cli_operator_basis_label",
            "basis_size": int(len(basis)),
            "labels_input": list(args.operator_basis_label),
        }
    elif args.operator_basis_source is not None:
        basis_artifact_json = args.basis_artifact_json
        if basis_artifact_json is None:
            basis_artifact_json = args.state_json if args.state_json is not None else args.hamiltonian_json
        resolved_basis_artifact_json = basis_artifact_json
        basis, basis_provenance = basis_elements_from_artifact_source(
            basis_artifact_json,
            nq=int(nq),
            hamiltonian=hamiltonian,
            source=str(args.operator_basis_source),
            full_meta_keep_classes=args.full_meta_keep_classes,
            include_hamiltonian_terms=bool(args.include_hamiltonian_term_basis),
            canonical_hh_full_meta=bool(args.paper_iii_static_qse_mode),
        )
    else:
        identity = "e" * int(nq)
        basis = basis_elements_from_labels([identity], nq=int(nq))
        basis_provenance = {
            "source_schema": "default_identity_only",
            "basis_size": 1,
            "labels_input": [identity],
        }

    transition_observables = []
    transition_observable_provenance: list[dict[str, object]] = []
    conductivity_channels = []
    if bool(args.conductivity_current_label) or bool(args.hh_current_response):
        from pipelines.qse_spectra.conductivity import ConductivityChannel

    green_function_modes = []
    if args.transition_observable_label is not None:
        labeled = transition_observables_from_labels(args.transition_observable_label, nq=int(nq))
        transition_observables.extend(labeled)
        transition_observable_provenance.append(
            {
                "source_schema": "cli_transition_observable_label",
                "observable_count": int(len(labeled)),
                "labels_input": list(args.transition_observable_label),
            }
        )
    for observable_json in args.transition_observable_json or []:
        loaded_observables, provenance = load_transition_observables_json(observable_json, nq=int(nq))
        transition_observables.extend(loaded_observables)
        transition_observable_provenance.append(provenance)
    for current_label in args.conductivity_current_label or []:
        conductivity_channels.append(
            ConductivityChannel(
                current_label=str(current_label),
                contact_label=args.conductivity_contact_label,
                channel_kind="custom_current",
                metadata={"source": "cli_conductivity_current_label"},
            )
        )

    hh_response_channels = []
    if args.hh_neutral_response_channel:
        from pipelines.qse_spectra.hh_response_observables import (
            HHResponseObservableError,
            build_hh_neutral_response_observable_bundle,
            resolve_hh_response_layout_from_sources,
        )

        try:
            hh_layout = resolve_hh_response_layout_from_sources(
                expected_nq=int(nq),
                sources={
                    "hamiltonian_json": args.hamiltonian_json,
                    "basis_artifact_json": args.basis_artifact_json,
                    "seed_manifest_json": args.seed_manifest_json,
                    "state_json": args.state_json,
                },
            )
            hh_bundle = build_hh_neutral_response_observable_bundle(
                layout=hh_layout,
                channels=args.hh_neutral_response_channel,
                form_factor=str(args.hh_response_form_factor),
                prepared_state=prepared_state,
                density_baseline=args.hh_response_density_baseline,
                nx_separation=int(args.hh_response_nx_separation),
                config=config,
            )
        except HHResponseObservableError as exc:
            parser.error(str(exc))
        transition_observables.extend(hh_bundle.observables)
        hh_response_channels.extend(hh_bundle.response_channels)
        transition_observable_provenance.append(
            {
                "source_schema": "hh_neutral_response_observables_v1",
                **dict(hh_bundle.metadata),
            }
        )
    if args.hh_current_response:
        from pipelines.qse_spectra.hh_current_observables import (
            HHCurrentObservableError,
            build_hh_current_observable_bundle,
            resolve_hh_current_hopping_from_sources,
        )
        from pipelines.qse_spectra.hh_response_observables import (
            HHResponseObservableError,
            resolve_hh_response_layout_from_sources,
        )

        try:
            hh_current_layout = resolve_hh_response_layout_from_sources(
                expected_nq=int(nq),
                sources={
                    "hamiltonian_json": args.hamiltonian_json,
                    "basis_artifact_json": args.basis_artifact_json,
                    "seed_manifest_json": args.seed_manifest_json,
                    "state_json": args.state_json,
                },
            )
            if args.hh_current_hopping_amplitude is None:
                hopping_resolution = resolve_hh_current_hopping_from_sources(
                    sources={
                        "hamiltonian_json": args.hamiltonian_json,
                        "basis_artifact_json": args.basis_artifact_json,
                        "seed_manifest_json": args.seed_manifest_json,
                        "state_json": args.state_json,
                    }
                )
                hh_current_hopping = float(hopping_resolution.hopping_amplitude)
                hh_current_hopping_metadata = dict(hopping_resolution.metadata)
            else:
                hh_current_hopping = float(args.hh_current_hopping_amplitude)
                hh_current_hopping_metadata = {"source_schema": "cli_hh_current_hopping_amplitude"}
            hh_current_bundle = build_hh_current_observable_bundle(
                layout=hh_current_layout,
                hopping_amplitude=float(hh_current_hopping),
                edge_orientation=str(args.hh_current_edge_orientation),
                include_contact=not bool(args.hh_current_disable_contact),
                contact_policy=str(args.hh_current_contact_policy),
                peierls_policy=str(args.hh_current_peierls_policy),
                config=config,
                hopping_source_metadata=hh_current_hopping_metadata,
            )
        except (HHResponseObservableError, HHCurrentObservableError) as exc:
            parser.error(str(exc))
        transition_observables.extend(hh_current_bundle.observables)
        transition_observable_provenance.append(
            {
                "source_schema": "hh_current_observables_v1",
                **dict(hh_current_bundle.metadata),
            }
        )
        for current_label in hh_current_bundle.current_labels:
            conductivity_channels.append(
                ConductivityChannel(
                    current_label=str(current_label),
                    contact_label=hh_current_bundle.contact_label,
                    channel_kind="hh_longitudinal_charge",
                    metadata={"source": "hh_current_observables_v1"},
                )
            )
    if green_function_requested:
        from pipelines.qse_spectra.green_functions import (
            build_green_function_payload,
            parse_green_function_mode_specs,
        )

        try:
            green_function_modes = list(parse_green_function_mode_specs(args.green_function_mode or ()))
        except ValueError as exc:
            parser.error(str(exc))
    if spectral_requested and not transition_observables and not green_function_requested:
        parser.error(
            "spectral mode requires at least one --transition-observable-label, "
            "--transition-observable-json, or --hh-neutral-response-channel."
        )
    if response_requested and not transition_observables:
        parser.error(
            "response function mode requires at least one --transition-observable-label, "
            "--transition-observable-json, or --hh-neutral-response-channel."
        )
    if conductivity_requested and not conductivity_channels:
        parser.error(
            "conductivity response mode requires at least one --conductivity-current-label "
            "or --hh-current-response."
        )

    static_record_selection_result = None
    compiled_cost_rows = None
    compiled_cost_oracle_kind = None
    if static_record_selection_config is not None:
        candidate_basis = tuple(basis)
        compiled_cost_oracle_kind = args.static_record_selection_compiled_cost_oracle
        if compiled_cost_oracle_kind is None and str(static_record_selection_config.mode) == "compiled_cost":
            compiled_cost_oracle_kind = "marrakesh_graph_span_v1"
        candidate_compiled_costs = None
        if compiled_cost_oracle_kind is not None:
            from pipelines.qse_spectra.compiled_costs import (
                annotate_basis_with_compiled_costs,
                resolve_cost_weights_preset,
            )

            try:
                compiled_cost_weights = resolve_cost_weights_preset(
                    str(args.static_record_selection_cost_weights)
                )
                compiled_cost_rows = annotate_basis_with_compiled_costs(
                    candidate_basis,
                    num_qubits=int(nq),
                    oracle_kind=str(compiled_cost_oracle_kind),
                    cost_weights=compiled_cost_weights,
                )
            except ValueError as exc:
                parser.error(str(exc))
            candidate_compiled_costs = tuple(
                float(row.scalarized_canonical_cost) for row in compiled_cost_rows
            )
        try:
            static_record_selection_result = select_static_qse_records(
                candidate_basis,
                config=static_record_selection_config,
                hamiltonian=hamiltonian,
                prepared_state=prepared_state,
                qse_config=config,
                basis_vector_policy=basis_vector_policy,
                transition_observables=transition_observables,
                compiled_costs=candidate_compiled_costs,
            )
        except ValueError as exc:
            parser.error(str(exc))
        basis = static_record_selection_result.selected_basis_elements
        basis_provenance = {
            **basis_provenance,
            "static_record_selection_enabled": True,
            "static_record_selection_mode": str(static_record_selection_config.mode),
            "candidate_basis_size": int(len(candidate_basis)),
            "selected_basis_size": int(len(basis)),
            "selected_original_basis_indices": [
                int(index) for index in static_record_selection_result.selected_original_indices
            ],
        }

    result = compute_qse_spectra(
        hamiltonian,
        prepared_state,
        basis,
        config=config,
        basis_vector_policy=basis_vector_policy,
        transition_observables=transition_observables,
    )
    static_record_selection_payload = None
    compiled_costs_payload = None
    if static_record_selection_result is not None:
        static_record_selection_payload = finalize_static_record_selection_payload(
            static_record_selection_result,
            result,
        )
        if compiled_cost_rows is not None:
            from pipelines.qse_spectra.compiled_costs import (
                build_accuracy_cost_frontier,
                compiled_costs_manifest_payload,
            )

            compiled_costs_payload = compiled_costs_manifest_payload(
                compiled_cost_rows,
                oracle_kind=str(compiled_cost_oracle_kind),
                num_qubits=int(nq),
                cost_weights=resolve_cost_weights_preset(
                    str(args.static_record_selection_cost_weights)
                ),
                cost_weights_preset=str(args.static_record_selection_cost_weights),
            )
            if bool(args.static_record_selection_cost_frontier):
                selected_rows = tuple(
                    compiled_cost_rows[int(index)]
                    for index in static_record_selection_result.selected_original_indices
                )
                try:
                    static_record_selection_payload["accuracy_cost_frontier"] = (
                        build_accuracy_cost_frontier(
                            static_record_selection_result.selected_basis_elements,
                            selected_rows,
                            hamiltonian=hamiltonian,
                            prepared_state=prepared_state,
                            qse_config=config,
                            basis_vector_policy=basis_vector_policy,
                            transition_observables=tuple(transition_observables),
                        )
                    )
                except ValueError as exc:
                    parser.error(str(exc))
    spectral_functions_payload = None
    spectral_window_metrics_payload = None
    if spectral_requested and transition_observables:
        from pipelines.qse_spectra.spectral_functions import (
            BroadeningKernelConfig,
            SpectralGrid,
            build_spectral_postprocessing_payloads,
            load_spectral_references_json,
            parse_spectral_window_spec,
        )

        spectral_grid = SpectralGrid(
            omega_min=float(args.spectral_grid_min),
            omega_max=float(args.spectral_grid_max),
            num_points=int(args.spectral_grid_num),
        )
        kernel_config = BroadeningKernelConfig(
            kernel=str(args.spectral_kernel or "lorentzian"),
            eta=float(args.spectral_eta),
        )
        windows = tuple(
            parse_spectral_window_spec(spec, index=idx)
            for idx, spec in enumerate(args.spectral_window or [])
        )
        references = ()
        if args.spectral_reference_json is not None:
            references = load_spectral_references_json(args.spectral_reference_json)
        spectral_functions_payload, spectral_window_metrics_payload = build_spectral_postprocessing_payloads(
            result,
            grid=spectral_grid,
            kernel_config=kernel_config,
            windows=windows,
            references=references,
        )

    response_functions_payload = None
    if response_requested:
        from pipelines.qse_spectra.response_functions import (
            ResponseTimeGrid,
            build_response_functions_payload,
            parse_response_channel_specs,
        )
        from pipelines.qse_spectra.spectral_functions import BroadeningKernelConfig, SpectralGrid

        try:
            response_grid = SpectralGrid(
                omega_min=float(args.spectral_grid_min),
                omega_max=float(args.spectral_grid_max),
                num_points=int(args.spectral_grid_num),
            )
            response_kernel_config = BroadeningKernelConfig(
                kernel=str(args.spectral_kernel or "lorentzian"),
                eta=float(args.spectral_eta),
            )
            response_time_grid = ResponseTimeGrid(
                t_min=float(args.response_time_grid_min),
                t_max=float(args.response_time_grid_max),
                num_points=int(args.response_time_grid_num),
            )
            response_channels = list(parse_response_channel_specs(args.response_channel or ()))
            response_channels.extend(hh_response_channels)
            response_functions_payload = build_response_functions_payload(
                result,
                grid=response_grid,
                kernel_config=response_kernel_config,
                time_grid=response_time_grid,
                channels=tuple(response_channels) or None,
                max_moment_order=int(args.response_moment_max_order),
                hamiltonian=None if bool(args.response_disable_sum_rules) else hamiltonian,
                prepared_state=None if bool(args.response_disable_sum_rules) else prepared_state,
                config=config,
                evaluate_sum_rules=not bool(args.response_disable_sum_rules),
            )
        except ValueError as exc:
            parser.error(str(exc))

    conductivity_response_payload = None
    if conductivity_requested:
        from pipelines.qse_spectra.conductivity import build_conductivity_response_payload
        from pipelines.qse_spectra.spectral_functions import BroadeningKernelConfig, SpectralGrid

        try:
            conductivity_grid = SpectralGrid(
                omega_min=float(args.spectral_grid_min),
                omega_max=float(args.spectral_grid_max),
                num_points=int(args.spectral_grid_num),
            )
            conductivity_kernel_config = BroadeningKernelConfig(
                kernel=str(args.spectral_kernel or "lorentzian"),
                eta=float(args.spectral_eta),
            )
            conductivity_response_payload = build_conductivity_response_payload(
                result,
                grid=conductivity_grid,
                kernel_config=conductivity_kernel_config,
                channels=tuple(conductivity_channels),
                prepared_state=prepared_state,
                config=config,
                omega_floor=float(args.conductivity_omega_floor),
                peierls_policy=str(args.conductivity_peierls_policy),
                contact_policy=str(args.conductivity_contact_policy),
            )
        except ValueError as exc:
            parser.error(str(exc))

    green_function_payload = None
    if green_function_requested:
        from pipelines.qse_spectra.spectral_functions import BroadeningKernelConfig, SpectralGrid

        try:
            green_grid = SpectralGrid(
                omega_min=float(args.spectral_grid_min),
                omega_max=float(args.spectral_grid_max),
                num_points=int(args.spectral_grid_num),
            )
            green_kernel_config = BroadeningKernelConfig(
                kernel=str(args.spectral_kernel or "lorentzian"),
                eta=float(args.spectral_eta),
            )
            green_function_payload = build_green_function_payload(
                result,
                hamiltonian=hamiltonian,
                prepared_state=prepared_state,
                modes=tuple(green_function_modes),
                grid=green_grid,
                kernel_config=green_kernel_config,
                fermion_mode_count=int(args.green_function_fermion_qubits),
                config=config,
            )
        except ValueError as exc:
            parser.error(str(exc))

    cutoff_boundary_diagnostics_payload = None
    if bool(args.cutoff_boundary_diagnostics):
        from pipelines.qse_spectra.spectral_functions import (
            CutoffBoundaryLayout,
            build_cutoff_boundary_diagnostics,
        )

        cutoff_layout = CutoffBoundaryLayout(
            num_sites=int(args.cutoff_num_sites),
            n_ph_max=int(args.cutoff_n_ph_max),
            boson_encoding=str(args.cutoff_boson_encoding),
            fermion_qubits=int(args.cutoff_fermion_qubits),
        )
        cutoff_boundary_diagnostics_payload = build_cutoff_boundary_diagnostics(
            result,
            layout=cutoff_layout,
        )

    output_json = args.output_json if args.output_json is not None else _default_output_path(args.hamiltonian_json)
    input_provenance = {
        "hamiltonian": hamiltonian_provenance,
        "state": state_provenance,
        "operator_basis": basis_provenance,
    }
    if transition_observable_provenance:
        input_provenance["transition_observables"] = transition_observable_provenance
    settings_provenance = {
        **asdict(config),
        "paper_iii_static_qse_mode": bool(args.paper_iii_static_qse_mode),
        "response_functions_enabled": bool(response_requested),
        "conductivity_response_enabled": bool(conductivity_requested),
        "green_functions_enabled": bool(green_function_requested),
    }
    if args.hh_neutral_response_channel:
        settings_provenance.update(
            {
                "hh_neutral_response_channels": list(args.hh_neutral_response_channel or []),
                "hh_response_form_factor": str(args.hh_response_form_factor),
                "hh_response_nx_separation": int(args.hh_response_nx_separation),
            }
        )
    if conductivity_requested:
        settings_provenance.update(
            {
                "conductivity_current_labels": [str(channel.current_label) for channel in conductivity_channels],
                "conductivity_contact_labels": [channel.contact_label for channel in conductivity_channels],
                "conductivity_omega_floor": float(args.conductivity_omega_floor),
                "conductivity_contact_policy": str(args.conductivity_contact_policy),
                "conductivity_peierls_policy": str(args.conductivity_peierls_policy),
            }
        )
    if args.hh_current_response:
        settings_provenance.update(
            {
                "hh_current_response_enabled": True,
                "hh_current_edge_orientation": str(args.hh_current_edge_orientation),
                "hh_current_contact_policy": str(args.hh_current_contact_policy),
                "hh_current_peierls_policy": str(args.hh_current_peierls_policy),
                "hh_current_contact_enabled": not bool(args.hh_current_disable_contact),
            }
        )
    if green_function_requested:
        settings_provenance.update(
            {
                "green_function_modes": [
                    {"label": str(mode.label), "mode_index": int(mode.mode_index)}
                    for mode in green_function_modes
                ],
                "green_function_fermion_qubits": int(args.green_function_fermion_qubits),
            }
        )
    paper_iii_contract_payload = None
    if bool(args.paper_iii_static_qse_mode):
        hh_full_meta_provenance = None
        if args.operator_basis_source in HH_FULL_META_OPERATOR_BASIS_SOURCES:
            try:
                hh_full_meta_provenance = resolve_hh_full_meta_provenance(
                    operator_basis_source=str(args.operator_basis_source),
                    basis_artifact_path=resolved_basis_artifact_json or args.basis_artifact_json,
                    seed_manifest_path=args.seed_manifest_json,
                    cli_pool_key=str(args.operator_basis_source),
                    basis_provenance=basis_provenance,
                    hamiltonian_source=hamiltonian_provenance,
                    state_source=state_provenance,
                )
            except PaperIIIProductionContractError as exc:
                parser.error(str(exc))
        try:
            paper_iii_contract_payload = build_paper_iii_contract(
                run_class=str(args.paper_iii_run_class),
                visible_target=str(args.paper_iii_visible_target),
                compatibility_tier=str(args.paper_iii_compatibility_tier),
                hh_full_meta_provenance=hh_full_meta_provenance,
            )
        except PaperIIIProductionContractError as exc:
            parser.error(str(exc))
    manifest = qse_result_to_manifest(
        result,
        input_provenance=input_provenance,
        settings_provenance=settings_provenance,
        include_matrices=not bool(args.omit_matrices),
        static_record_selection_payload=static_record_selection_payload,
        spectral_functions_payload=spectral_functions_payload,
        spectral_window_metrics_payload=spectral_window_metrics_payload,
        cutoff_boundary_diagnostics_payload=cutoff_boundary_diagnostics_payload,
        qse_response_functions_payload=response_functions_payload,
        qse_conductivity_response_payload=conductivity_response_payload,
        qse_green_function_payload=green_function_payload,
        paper_iii_contract_payload=paper_iii_contract_payload,
        compiled_costs_payload=compiled_costs_payload,
    )
    write_manifest_json(output_json, manifest)

    print(f"output_json: {output_json}")
    print(f"num_qubits: {result.matrices.nq}")
    print(f"basis_size: {len(result.matrices.basis_elements)}")
    print(f"retained_rank: {result.retained_rank}")
    print(f"lowest_energy: {float(result.eigenvalues[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
