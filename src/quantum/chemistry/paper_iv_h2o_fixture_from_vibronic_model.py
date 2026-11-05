"""Paper IV H2O lane fixture from a vibronic qubit model whose linear coupling is given analytically.

The Paper IV lane runs RA-ADAPT/SNAKE from one ``molecular_vibronic_h2o_linear_fd_fixture_v1``
file, which defines the Hamiltonian, the generator pool, the reference state and the exact stop
target (see ``paper_iv_h2o_fixture_from_builder``).  That producer resolves the fixture's per-mode
derivatives dH/dQ_mu by central differences of displaced SCFs, i.e. the builder's full linear
coupling.  This producer takes the derivatives as the model's own coupling tensors instead:
our builder set to the PySCF mean-field electron-phonon vertex (Doron Wu's model, with or
without the reference-force counter-term), or the model read from his driver's HDF5 files
(``qdynamics_vendored.vibronic_model_from_qdynamics_outputs``).  The modes are labelled bend /
symmetric_stretch / antisymmetric_stretch by internal-coordinate character and oriented by the
lane's policy, and the coupling tensors carry that orientation sign.  Everything from the encoded
operators on is the lane's own generator core (``build_h2o_linear_fd_fixture_from_components``):
register layout, encoded operators, reference state, exact sector reference, cutoff diagnostics,
production pool, manifest and evidence gate.  The fixture declares its derivative source as
analytic and, when its active space is not CAS(8e,6o), its active-space variant; it records the
size of the derivative-resolved pool the lane locks the parent inventory against.

The summary written next to the fixture cross-checks the fixture's exact sector ground energy
against the model's own exact solution, compares the fixture's Hamiltonian term by term with the
model's operator (in his JSON schema, modes matched by frequency and orientation), and, when a
qdynamics JSON is given, with that file too.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.chemistry.generate_h2o_linear_fd_fixture import (
    _assign_h2o_mode_labels_by_character,
    _h2o_mode_character,
    _parse_int_tuple,
    _translation_rotation_overlap,
    _validate_h2o_symbols,
    _write_json,
    build_h2o_linear_fd_fixture_from_components,
    sha256_jsonable,
)
from src.quantum.chemistry.molecular_vibronic import (
    MODEL_SCHEMA,
    MolecularGeometry,
    VibronicQubitModel,
    assemble_vibronic_model,
    build_molecular_vibronic_model,
    exact_sector_energies,
    isotope_masses_me,
    parse_geometry,
    reference_force,
)
from src.quantum.chemistry.paper_iv_h2o_fixture_from_builder import (
    LANE_MODE_LABELS,
    PAPER_IV_H2O_GEOMETRY,
    _geometry_source,
    _sha256_file,
    lane_active_space_kind,
)
from src.quantum.chemistry.qdynamics_interop import (
    OUR_EQUIVALENT_COUPLING_MODEL,
    compare_pauli_polynomials,
    hartree_fock_vacuum_energy,
    match_modes_to_reference,
    qdynamics_payload_from_model,
    qdynamics_qubit_hamiltonian_from_payload,
    save_qdynamics_qubit_hamiltonian,
)
from src.quantum.chemistry.vibronic_h2o_linear_fd import (
    H2O_LINEAR_FD_ANALYTIC_DERIVATIVE_SOURCE,
    ActiveSpaceRecord,
    DerivativeNorms,
    FirstDerivativeRecord,
    GeometryRecord,
    NormalModeRecord,
    ProductionVibronicH2OFixture,
    build_h2o_linear_fd_derivative_resolved_pool_v2,
    load_cached_production_vibronic_h2o_linear_fd_fixture,
    load_production_vibronic_h2o_fixture,
    pauli_polynomial_from_jsonable,
    production_vibronic_h2o_fixture_to_jsonable,
    validate_paper_iv_h2o_linear_fd_evidence_fixture,
)

PRODUCER = "src.quantum.chemistry.paper_iv_h2o_fixture_from_vibronic_model:paper_iv_h2o_fixture_from_vibronic_model_v1"
SUMMARY_SCHEMA = "paper_iv_h2o_fixture_from_vibronic_model_summary_v1"
DERIVATIVE_METHOD = "analytic"
SCALAR_DERIVATIVE_CONVENTION = "model_coupling_scalar_per_unit_q_counter_term_included_when_the_model_applies_it"
MODEL_DEFINITION_KEYS = (
    "schema", "source", "harmonic_curvature", "alignment", "mode_basis", "reference_force_counter_term",
    "rhf_hessian_frequencies_cm1", "hf_energy_hartree", "backend", "backend_version", "isotope_masses_amu",
    "qdynamics_eph_mode_order", "qdynamics_vendored_commit", "files",
)


def _jsonable(value: Any) -> Any:
    """Plain JSON values from numpy arrays and scalars, tuples, paths and mappings."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in (sorted(value) if isinstance(value, (set, frozenset)) else value)]
    if isinstance(value, Path):
        return str(value)
    return value


def model_masses_me(model: VibronicQubitModel, masses_me: Sequence[float] | None = None) -> np.ndarray:
    """The atomic masses (electron-mass units) the model's modes were mass-weighted with: the
    caller's, else the ones the model's provenance records, else our most-abundant-isotope masses
    (the builder's default)."""
    if masses_me is not None:
        masses = np.asarray(masses_me, dtype=float).reshape(-1)
    elif model.provenance.get("masses_me") is not None:
        masses = np.asarray(model.provenance["masses_me"], dtype=float).reshape(-1)
    else:
        masses = isotope_masses_me(model.geometry.symbols)
    if masses.shape != (model.geometry.n_atoms,) or not np.all(masses > 0.0):
        raise ValueError(f"masses must be {model.geometry.n_atoms} positive numbers; got {masses}.")
    return masses


def lane_mode_rows(model: VibronicQubitModel, masses_me: np.ndarray) -> tuple[dict[str, Any], ...]:
    """The model's modes labelled bend / symmetric_stretch / antisymmetric_stretch by internal-coordinate
    character, oriented by the lane's policy (each mode's own indicator positive) and ordered by label.
    Each row keeps ``builder_mode_index`` and ``orientation_sign_vs_builder_mode``."""
    geometry = model.geometry
    _validate_h2o_symbols(geometry.symbols)
    if len(model.modes) != 3:
        raise ValueError(f"the Paper IV H2O lane needs three modes; the model has {len(model.modes)}.")
    rows: list[dict[str, Any]] = []
    for a, mode in enumerate(model.modes):
        vector = np.asarray(mode.mass_weighted_vector, dtype=float)
        rows.append({
            "mode_index": int(a), "label": str(mode.label),
            "frequency_hartree": float(mode.frequency_hartree), "frequency_cm1": float(mode.frequency_cm1),
            "mass_weighted_eigenvector": vector,
            "mode_character": _h2o_mode_character(symbols=geometry.symbols, coordinates_bohr=geometry.coordinates_bohr,
                                                  mode_vector=vector, masses_me=masses_me),
            "mode_character_for_negative_vector": _h2o_mode_character(
                symbols=geometry.symbols, coordinates_bohr=geometry.coordinates_bohr, mode_vector=-vector, masses_me=masses_me),
            "builder_mode_index": int(a),
            "model_mode_label": str(mode.label),
        })
    labeled = [dict(r) for r in _assign_h2o_mode_labels_by_character(rows)]
    if tuple(str(r["label"]) for r in labeled) != LANE_MODE_LABELS:
        raise ValueError(f"mode labelling by character did not yield {LANE_MODE_LABELS}: {[r['label'] for r in labeled]}")
    for row in labeled:
        dominant = str(dict(row["mode_character"]).get("dominant_label"))
        if dominant != str(row["label"]):
            raise ValueError(f"mode labelled {row['label']!r} has dominant internal-coordinate character {dominant!r}; "
                             "the model modes do not map one-to-one onto bend / symmetric_stretch / antisymmetric_stretch.")
        a = int(row["builder_mode_index"])
        vector = np.asarray(row["mass_weighted_eigenvector"], dtype=float)
        orientation = float(np.sign(np.sum(vector * np.asarray(model.modes[a].mass_weighted_vector, dtype=float))))
        if orientation == 0.0:
            raise ValueError(f"oriented vector of mode {a} is orthogonal to the model's; labelling is inconsistent.")
        row["orientation_sign_vs_builder_mode"] = orientation
        row["trans_rot_overlap"] = float(_translation_rotation_overlap(
            vector, coordinates_bohr=geometry.coordinates_bohr, masses_me=masses_me))
        row["label_source"] = str(row.get("label_source", "internal_coordinate_character_v1"))
    return tuple(labeled)


def analytic_first_derivatives(model: VibronicQubitModel, rows: Sequence[Mapping[str, Any]]) -> tuple[FirstDerivativeRecord, ...]:
    """dH/dQ_mu of each lane mode as the model's coupling tensors (per unit mass-weighted Q), carrying
    the lane orientation sign; ``active_equilibrium_force`` is <HF| dH/dQ_mu |HF> in that orientation."""
    n = int(np.asarray(model.active_tensors.one_body).shape[0])
    n_occ = int(model.active_space.n_alpha)
    out: list[FirstDerivativeRecord] = []
    for row in rows:
        a = int(row["builder_mode_index"])
        sign = float(row["orientation_sign_vs_builder_mode"])
        coupling = model.couplings[a]
        one = sign * np.asarray(coupling.one_body, dtype=float)
        two = sign * (np.zeros((n,) * 4, dtype=float) if coupling.two_body is None else np.asarray(coupling.two_body, dtype=float))
        scalar = sign * float(coupling.scalar)
        out.append(FirstDerivativeRecord(
            derivative_id=f"d_{row['label']}",
            mode_index=int(row["mode_index"]),
            mode_label=str(row["label"]),
            q_step_au=1.0,
            plus_aligned_tensor_id=None,
            minus_aligned_tensor_id=None,
            scalar_derivative_hartree_per_q=float(scalar),
            one_body_derivative=one,
            two_body_derivative=two,
            scalar_derivative_included=True,
            scalar_derivative_convention=SCALAR_DERIVATIVE_CONVENTION,
            derivative_source=H2O_LINEAR_FD_ANALYTIC_DERIVATIVE_SOURCE,
            norms=DerivativeNorms(scalar_abs=float(abs(scalar)), one_body_fro=float(np.linalg.norm(one)),
                                  two_body_fro=float(np.linalg.norm(two))),
            finite_difference_drift=None,
            finite_difference_diagnostics={
                "derivative_method": DERIVATIVE_METHOD,
                "coupling_model": str(model.coupling_model),
                "builder_mode_index": a,
                "orientation_sign_vs_builder_mode": sign,
                "two_body_included": coupling.two_body is not None,
                "q_step_au_meaning": "unit of the mass-weighted normal coordinate; no displaced tensors exist for an analytic derivative",
                "coupling_diagnostics": _jsonable(coupling.diagnostics),
            },
            active_equilibrium_force=float(sign * reference_force(coupling, n_occ)),
            passed=True,
        ))
    return tuple(out)


def builder_model_in_his_conventions(
    geometry: MolecularGeometry,
    *,
    basis: str = "sto-3g",
    active_space: str | Mapping[str, Any] = "full",
    reference_force_counter_term: bool = False,
) -> tuple[VibronicQubitModel, Any]:
    """Our builder set to his model: RHF-Hessian frequencies and modes, the PySCF mean-field vertex,
    optionally the reference-force counter-term; the full MO space unless an active space is named.
    With ``active_space='full'`` and no counter-term this is ``qdynamics_interop.builder_model_in_qdynamics_conventions``."""
    return build_molecular_vibronic_model(
        geometry, basis=basis, active_space=active_space, cutoffs=1, coupling_model=OUR_EQUIVALENT_COUPLING_MODEL,
        reference_force_counter_term=bool(reference_force_counter_term), harmonic_curvature="rhf_hessian",
        mode_basis="rhf_normal_modes")


def work_model_at_lane_cutoffs(model: VibronicQubitModel, rows: Sequence[Mapping[str, Any]], mode_cutoffs: Sequence[int]) -> tuple[VibronicQubitModel, list[int]]:
    """The same model assembled at the fixture's work cutoffs, which the lane lists in bend /
    symmetric_stretch / antisymmetric_stretch order, permuted into the builder's mode order."""
    cutoffs = [int(v) for v in mode_cutoffs]
    builder_cutoffs = [0] * len(cutoffs)
    for row, cutoff in zip(rows, cutoffs):
        builder_cutoffs[int(row["builder_mode_index"])] = int(cutoff)
    work = assemble_vibronic_model(
        geometry=model.geometry, basis=str(model.basis), active_space=model.active_space,
        active_tensors=model.active_tensors, modes=model.modes, couplings=model.couplings,
        cutoffs=builder_cutoffs, encoding=str(model.encoding), coupling_model=str(model.coupling_model),
        provenance=dict(model.provenance))
    return work, builder_cutoffs


def compare_fixture_with_qdynamics_payload(fixture: ProductionVibronicH2OFixture, payload: Mapping[str, Any], *, atol: float = 1.0e-9) -> dict[str, Any]:
    """The fixture's Hamiltonian against an operator in his JSON schema, term by term, after
    reordering his registers to the fixture's modes (nearest frequency) and choosing each
    register's orientation.  Registers can be permuted only when every mode has the same cutoff."""
    cutoffs = tuple(int(block.n_ph_max) for block in fixture.layout.boson_modes)
    if len(set(cutoffs)) != 1 or len(set(int(v) for v in payload.get("mode_cutoffs") or ())) != 1:
        return {"compared": False, "reason": "registers of unequal width cannot be permuted; use uniform cutoffs for the term comparison"}
    fixture_h = pauli_polynomial_from_jsonable(fixture.encoded_operators.h_vibronic, expected_nq=int(fixture.layout.n_total_qubits))
    frequencies = [float(mode.frequency_hartree) for mode in fixture.normal_modes]
    matched, order, signs = match_modes_to_reference(payload, frequencies, fixture_h, atol=atol)
    other = qdynamics_qubit_hamiltonian_from_payload(matched).operator
    report = compare_pauli_polynomials(fixture_h, other, atol=atol)
    return {"compared": True, "atol": float(atol), "payload_mode_order_for_lane_modes": [int(v) for v in order],
            "register_signs": [int(v) for v in signs], **report}


def fixture_from_vibronic_model(
    model: VibronicQubitModel,
    *,
    mode_cutoffs: Sequence[int],
    reference_cutoffs: Sequence[int] | None = None,
    masses_me: Sequence[float] | None = None,
    geometry_source: str | None = None,
    source: Mapping[str, Any] | None = None,
    dense_full_dim_cap: int = 8192,
    embed_exact_state: bool = True,
    exact_reference_policy: str = "auto_sector",
    operator_cleanup_tol: float = 1.0e-12,
) -> tuple[ProductionVibronicH2OFixture, tuple[dict[str, Any], ...], VibronicQubitModel]:
    """The lane fixture of ``model`` at ``mode_cutoffs`` (lane order), with the work model it was
    cross-checked against.  The fixture records the derivative-resolved pool size in
    ``report_summary.derivative_resolved_pool_v2_parent_count`` and passes the evidence gate."""
    masses = model_masses_me(model, masses_me)
    rows = lane_mode_rows(model, masses)
    cutoffs = tuple(int(v) for v in mode_cutoffs)
    if len(cutoffs) != 3 or any(c < 1 for c in cutoffs):
        raise ValueError(f"three positive mode cutoffs in lane order are required; got {cutoffs}.")
    reference = None if reference_cutoffs is None else tuple(int(v) for v in reference_cutoffs)
    if reference is not None and (len(reference) != 3 or any(r <= w for r, w in zip(reference, cutoffs))):
        raise ValueError(f"each reference cutoff must exceed its work cutoff; got reference {reference} for work {cutoffs}.")
    derivatives = analytic_first_derivatives(model, rows)
    normal_modes = tuple(
        NormalModeRecord(
            mode_index=int(row["mode_index"]), label=str(row["label"]),
            frequency_hartree=float(row["frequency_hartree"]), frequency_cm1=float(row["frequency_cm1"]),
            mass_weighted_eigenvector=np.asarray(row["mass_weighted_eigenvector"], dtype=float),
            q_step_au=1.0, q_step_alt_au=None, trans_rot_residual=float(row["trans_rot_overlap"]),
        )
        for row in rows
    )
    space = model.active_space
    tensors = model.active_tensors
    n = int(np.asarray(tensors.one_body).shape[0])
    active = ActiveSpaceRecord(
        active_space_kind=lane_active_space_kind(space),
        frozen_core_indices=tuple(int(v) for v in space.frozen_core),
        active_indices_center=tuple(int(v) for v in space.active),
        external_indices=tuple(int(v) for v in space.external),
        n_spatial_orbitals=n,
        num_particles=(int(space.n_alpha), int(space.n_beta)),
        scalar_energy_hartree=float(tensors.scalar),
        one_body_integrals=np.asarray(tensors.one_body, dtype=float),
        two_body_integrals=np.asarray(tensors.two_body, dtype=float),
    )
    geometry_source = _geometry_source(model.geometry, geometry_source)
    geometry = GeometryRecord(
        geometry_id="h2o_center", molecule="H2O", symbols=tuple(str(v) for v in model.geometry.symbols),
        coordinates_bohr=np.asarray(model.geometry.coordinates_bohr, dtype=float), masses_me=masses,
        charge=int(model.geometry.charge), multiplicity=int(model.geometry.multiplicity), method="scf",
        basis=str(model.basis), reference="RHF", optimized=False,
        provenance={"coordinate_units": "bohr", "mass_units": "electron_mass", "geometry_source": geometry_source},
    )
    prov = _jsonable(dict(model.provenance))
    model_definition = {k: prov.get(k) for k in MODEL_DEFINITION_KEYS}
    model_definition.update({
        "builder_model_schema": MODEL_SCHEMA,
        "coupling_model": str(model.coupling_model),
        "active_space": space.to_jsonable(),
        "derivative_method": DERIVATIVE_METHOD,
        "geometry_source": geometry_source,
        "lane_label_by_builder_mode": {str(r["builder_mode_index"]): str(r["label"]) for r in rows},
        "lane_orientation_by_builder_mode": {str(r["builder_mode_index"]): float(r["orientation_sign_vs_builder_mode"]) for r in rows},
        "lane_frequencies_cm1": {str(r["label"]): float(r["frequency_cm1"]) for r in rows},
        "model_source": _jsonable(dict(source or {})),
    })
    report_summary: dict[str, Any] = {
        "backend_record_producer": PRODUCER,
        "normal_mode_label_policy": "internal_coordinate_character_v1",
        "active_space_policy": str(space.kind),
        "active_space_kind": active.active_space_kind,
        "derivative_method": DERIVATIVE_METHOD,
        "normal_mode_diagnostics": [
            {k: _jsonable(row[k]) for k in ("mode_index", "label", "frequency_hartree", "frequency_cm1", "label_source",
                                             "mode_character", "builder_mode_index", "model_mode_label",
                                             "orientation_sign_vs_builder_mode", "trans_rot_overlap")}
            for row in rows],
        "model_definition": model_definition,
    }
    if active.active_space_kind != "cas_8e_6o_valence":
        report_summary["paper_iv_active_space_variant"] = active.active_space_kind
    work_model, builder_cutoffs = work_model_at_lane_cutoffs(model, rows, cutoffs)
    payload = qdynamics_payload_from_model(work_model)
    provenance = {
        "producer": PRODUCER,
        "model_schema": MODEL_SCHEMA,
        "model_provenance": prov,
        "coupling_model": str(model.coupling_model),
        "model_source": _jsonable(dict(source or {})),
        "generator_core": "src.quantum.chemistry.generate_h2o_linear_fd_fixture:build_h2o_linear_fd_fixture_from_components",
        "builder_order_cutoffs": [int(v) for v in builder_cutoffs],
    }
    fixture = build_h2o_linear_fd_fixture_from_components(
        geometry=geometry, active_space=active, normal_modes=normal_modes, first_derivatives=derivatives,
        mode_cutoffs=cutoffs, reference_cutoffs=reference, derivative_source=H2O_LINEAR_FD_ANALYTIC_DERIVATIVE_SOURCE,
        report_summary=report_summary, provenance=provenance,
        provenance_hashes={"model_qdynamics_payload_sha256": sha256_jsonable(payload)},
        electronic_reference="rhf", dense_full_dim_cap=int(dense_full_dim_cap), embed_exact_state=bool(embed_exact_state),
        require_reference_cutoff=True, exact_reference_policy=str(exact_reference_policy),
        operator_cleanup_tol=float(operator_cleanup_tol))
    parent_count = len(build_h2o_linear_fd_derivative_resolved_pool_v2(fixture))
    fixture = replace(fixture, report_summary={**dict(fixture.report_summary), "derivative_resolved_pool_v2_parent_count": int(parent_count)})
    validate_paper_iv_h2o_linear_fd_evidence_fixture(fixture, require_exact_state=bool(embed_exact_state),
                                                     require_reference_cutoff=True, require_cutoff_converged=False)
    return fixture, rows, work_model


def write_paper_iv_h2o_fixture_from_vibronic_model(
    output_dir: str | Path,
    *,
    tag: str,
    model: VibronicQubitModel,
    mode_cutoffs: Sequence[int] = (1, 1, 1),
    reference_cutoffs: Sequence[int] | None = None,
    masses_me: Sequence[float] | None = None,
    geometry_source: str | None = None,
    source: Mapping[str, Any] | None = None,
    compare_qdynamics_json: str | Path | None = None,
    force: bool = False,
    **fixture_kwargs: Any,
) -> dict[str, Any]:
    """Write ``<tag>_fixture.json``, ``<tag>_qdynamics.json`` (the work model in his schema, builder
    mode order) and ``<tag>_summary.json`` into ``output_dir``; return the summary."""
    started = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {kind: out / f"{tag}_{kind}.json" for kind in ("fixture", "qdynamics", "summary")}
    if not force:
        existing = [str(p) for p in paths.values() if p.exists()]
        if existing:
            raise FileExistsError(f"output exists (pass force=True to overwrite): {existing}")
    fixture, rows, work_model = fixture_from_vibronic_model(
        model, mode_cutoffs=mode_cutoffs, reference_cutoffs=reference_cutoffs, masses_me=masses_me,
        geometry_source=geometry_source, source=source, **fixture_kwargs)
    _write_json(paths["fixture"], production_vibronic_h2o_fixture_to_jsonable(fixture), force=bool(force))
    if force and paths["qdynamics"].exists():
        paths["qdynamics"].unlink()
    save_qdynamics_qubit_hamiltonian(
        work_model, paths["qdynamics"], source_files=[str(v) for v in _jsonable(dict(source or {})).get("files", [])] if isinstance(_jsonable(dict(source or {})).get("files"), list) else [],
        metadata={"paper_iv_fixture": paths["fixture"].name, "producer": PRODUCER,
                  "mode_order": "builder order (ascending frequency); the fixture's registers are in lane order "
                                "bend / symmetric_stretch / antisymmetric_stretch with the lane orientation"})
    reloaded = load_production_vibronic_h2o_fixture(paths["fixture"])
    validate_paper_iv_h2o_linear_fd_evidence_fixture(reloaded)
    cached = load_cached_production_vibronic_h2o_linear_fd_fixture(paths["fixture"])
    fixture_sha256, fixture_bytes = _sha256_file(paths["fixture"])
    model_energy = float(exact_sector_energies(work_model, n_low=1)[0][0])
    exact = reloaded.exact_reference
    fixture_energy = None if exact.ground_energy_hartree is None else float(exact.ground_energy_hartree)
    own_payload = json.loads(paths["qdynamics"].read_text(encoding="utf-8"))
    own_comparison = compare_fixture_with_qdynamics_payload(reloaded, own_payload)
    external_comparison = None
    if compare_qdynamics_json is not None:
        external_path = Path(compare_qdynamics_json)
        external_comparison = {"path": str(external_path), "sha256": _sha256_file(external_path)[0],
                               **compare_fixture_with_qdynamics_payload(reloaded, json.loads(external_path.read_text(encoding="utf-8")))}
    cutoff = reloaded.cutoff_diagnostics
    summary = {
        "schema": SUMMARY_SCHEMA, "producer": PRODUCER, "tag": str(tag),
        "fixture_file": paths["fixture"].name, "fixture_sha256": fixture_sha256, "fixture_byte_count": int(fixture_bytes),
        "qdynamics_file": paths["qdynamics"].name, "qdynamics_file_sha256": _sha256_file(paths["qdynamics"])[0],
        "fixture_schema": str(reloaded.manifest.schema), "production_status": str(reloaded.manifest.production_status),
        "derivative_source": str(reloaded.manifest.derivative_source),
        "coupling_model": str(model.coupling_model),
        "reference_force_counter_term": _jsonable(model.provenance.get("reference_force_counter_term")),
        "active_space_kind": str(reloaded.active_space.active_space_kind),
        "paper_iv_active_space_variant": reloaded.report_summary.get("paper_iv_active_space_variant"),
        "n_spatial_orbitals": int(reloaded.active_space.n_spatial_orbitals),
        "num_particles": [int(v) for v in reloaded.active_space.num_particles],
        "work_cutoffs": [int(v) for v in mode_cutoffs],
        "builder_order_cutoffs": [int(v) for v in reloaded.provenance["builder_order_cutoffs"]],
        "reference_cutoffs": None if cutoff is None or cutoff.reference_cutoffs is None else [int(v) for v in cutoff.reference_cutoffs],
        "n_total_qubits": int(reloaded.layout.n_total_qubits), "sector_dimension": int(exact.sector_dimension),
        "mode_labels": [str(m.label) for m in reloaded.normal_modes],
        "mode_frequencies_cm1": [float(m.frequency_cm1) for m in reloaded.normal_modes],
        "lane_orientation_by_builder_mode": {str(r["builder_mode_index"]): float(r["orientation_sign_vs_builder_mode"]) for r in rows},
        "active_equilibrium_force_by_mode": {str(d.mode_label): float(d.active_equilibrium_force) for d in reloaded.first_derivatives},
        "exact_reference_method": str(exact.method),
        "fixture_ground_energy_hartree": fixture_energy,
        "model_ground_energy_hartree": model_energy,
        "model_minus_fixture_hartree": None if fixture_energy is None else model_energy - fixture_energy,
        "hartree_fock_vacuum_energy_hartree": float(hartree_fock_vacuum_energy(work_model)),
        "reference_cutoff_ground_energy_hartree": None if cutoff is None else cutoff.reference_ground_energy_hartree,
        "cutoff_energy_drift_hartree": None if cutoff is None else cutoff.delta_energy_hartree,
        "cutoff_diagnostics_passed": None if cutoff is None else bool(cutoff.passed),
        "work_boundary_weight": None if cutoff is None or cutoff.work_boundary_weight is None else float(cutoff.work_boundary_weight.total_boundary_weight),
        "derivative_resolved_pool_v2_parent_count": int(reloaded.report_summary["derivative_resolved_pool_v2_parent_count"]),
        "fixture_pool_rows": int(len(reloaded.pool)),
        "cached_model_pool_terms": int(len(cached.model.pool)),
        "own_term_comparison": own_comparison,
        "external_qdynamics_comparison": external_comparison,
        "model_definition": reloaded.report_summary["model_definition"],
        "evidence_validation": "passed",
        "elapsed_seconds": time.time() - started,
    }
    _write_json(paths["summary"], summary, force=bool(force))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper IV H2O lane fixture from a vibronic model with an analytic linear coupling "
                                                 "(our builder in Doron Wu's conventions, or his driver's HDF5 files).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", required=True, help="file stem: <tag>_fixture.json, <tag>_qdynamics.json, <tag>_summary.json")
    parser.add_argument("--source", choices=("builder", "qdynamics-h5"), default="builder")
    parser.add_argument("--geometry", default=None, help="builder source: 'Sym x y z; ...' (default: the Paper IV H2O centre geometry)")
    parser.add_argument("--units", default="angstrom", choices=["angstrom", "bohr"])
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--active-space", default="full", help="builder source: 'full' (his model) or 'frozen_core'")
    parser.add_argument("--counter-term", action="store_true", help="apply C_I -> C_I - <HF|C_I|HF> (his model does not)")
    parser.add_argument("--ele-h5", default=None, help="qdynamics-h5 source: his <prefix>_ele.h5")
    parser.add_argument("--vib-h5", default=None, help="qdynamics-h5 source: his <prefix>_vib.h5")
    parser.add_argument("--eph-h5", default=None, help="qdynamics-h5 source: his <prefix>_eph.h5")
    parser.add_argument("--mode-cutoffs", default="1,1,1", help="lane order bend,symmetric_stretch,antisymmetric_stretch")
    parser.add_argument("--reference-mode-cutoffs", default=None, help="default: one more than each work cutoff")
    parser.add_argument("--compare-qdynamics-json", default=None, help="a qdynamics_qubit_hamiltonian_v1 file to compare the fixture with")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    mode_cutoffs = _parse_int_tuple(str(args.mode_cutoffs), label="mode-cutoffs")
    reference = None if args.reference_mode_cutoffs is None else _parse_int_tuple(str(args.reference_mode_cutoffs), label="reference-mode-cutoffs")
    masses_me = None
    if args.source == "builder":
        geometry = PAPER_IV_H2O_GEOMETRY if args.geometry is None else parse_geometry(str(args.geometry), units=str(args.units))
        model, computation = builder_model_in_his_conventions(
            geometry, basis=str(args.basis), active_space=str(args.active_space), reference_force_counter_term=bool(args.counter_term))
        masses_me = computation.masses_me
        source = {"kind": "builder_in_qdynamics_conventions", "active_space": str(args.active_space), "basis": str(args.basis),
                  "reference_force_counter_term": bool(args.counter_term)}
    else:
        if not (args.ele_h5 and args.vib_h5 and args.eph_h5):
            raise SystemExit("--source qdynamics-h5 needs --ele-h5, --vib-h5 and --eph-h5")
        from src.quantum.chemistry.qdynamics_vendored import vibronic_model_from_qdynamics_outputs

        model = vibronic_model_from_qdynamics_outputs(args.ele_h5, args.vib_h5, args.eph_h5, cutoffs=1,
                                                      reference_force_counter_term=bool(args.counter_term))
        source = {"kind": "qdynamics_driver_outputs", "files": [str(args.ele_h5), str(args.vib_h5), str(args.eph_h5)],
                  "reference_force_counter_term": bool(args.counter_term)}
    summary = write_paper_iv_h2o_fixture_from_vibronic_model(
        args.output_dir, tag=str(args.tag), model=model, mode_cutoffs=mode_cutoffs, reference_cutoffs=reference,
        masses_me=masses_me, source=source, compare_qdynamics_json=args.compare_qdynamics_json, force=bool(args.force))
    keys = ("fixture_file", "fixture_sha256", "fixture_byte_count", "n_total_qubits", "sector_dimension", "active_space_kind",
            "mode_labels", "mode_frequencies_cm1", "reference_cutoffs", "fixture_ground_energy_hartree",
            "model_ground_energy_hartree", "model_minus_fixture_hartree", "hartree_fock_vacuum_energy_hartree",
            "cutoff_energy_drift_hartree", "cutoff_diagnostics_passed", "derivative_resolved_pool_v2_parent_count",
            "own_term_comparison", "external_qdynamics_comparison", "elapsed_seconds")
    print(json.dumps({k: summary[k] for k in keys}, indent=1))


if __name__ == "__main__":
    main()
