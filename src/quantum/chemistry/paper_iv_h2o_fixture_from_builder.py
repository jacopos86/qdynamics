"""Paper IV H2O production fixture from the molecule-agnostic vibronic builder.

The Paper IV lane consumes ``molecular_vibronic_h2o_linear_fd_fixture_v1`` files written by
``generate_h2o_linear_fd_fixture.build_h2o_linear_fd_fixture_from_record`` from a backend
record (``h2o_linear_fd_backend_record_v1``): the centre active-space tensors, three normal
modes with their frequencies, and the displaced active-space tensors at +/- q (a primary and
an alternative step) along each mode, aligned to the R0 orbital gauge. The generator takes the
central differences, validates their drift, encodes the operators, solves the exact sector
reference and writes the fixture; the lane attests the fixture bytes.

This module produces that record from the builder's model, the one decided on 2026-11-02:
oscillators omega_a^2 from K_PES + chi (the frozen-core FCI surface curvature plus the model's
own correlated response), displaced orbitals aligned block by block so the frozen core stays
canonical, and model-consistent modes (the eigenvectors of K_PES + chi, so the exactly solved
model carries the full FCI Hessian). The record's modes are those model modes, labelled bend /
symmetric_stretch / antisymmetric_stretch by their internal-coordinate character (the lane's
own policy) and kept in that order; their frequencies are the builder's; the displaced tensors
are real SCFs along the model modes, so the generator's central difference along a model mode
reproduces the builder's rotated coupling to finite-difference accuracy and the lane's
validation applies unchanged. The builder's model definition (K_PES, chi, the mode rotation,
all frequency sets) travels in the record's ``backend`` block, which the generator copies into
the fixture provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.chemistry.generate_h2o_linear_fd_fixture import (
    BACKEND_RECORD_SCHEMA,
    _assign_h2o_mode_labels_by_character,
    _h2o_mode_character,
    _parse_int_tuple,
    _translation_rotation_overlap,
    _write_json,
    build_h2o_linear_fd_fixture_from_record,
    sha256_jsonable,
)
from src.quantum.chemistry.molecular_vibronic import (
    MODEL_SCHEMA,
    CouplingComputation,
    MolecularGeometry,
    NormalMode,
    VibronicQubitModel,
    _check_block_overlap,
    align_to_center,
    assemble_vibronic_model,
    build_molecular_vibronic_model,
    displace_along_mode,
    exact_sector_energies,
    frozen_core_contract,
    parse_geometry,
    pyscf_ao_cross_overlap,
    pyscf_rhf_snapshot,
)
from src.quantum.chemistry.vibronic_h2o_linear_fd import (
    load_production_vibronic_h2o_fixture,
    production_vibronic_h2o_fixture_to_jsonable,
    validate_paper_iv_h2o_linear_fd_evidence_fixture,
)

PRODUCER = "src.quantum.chemistry.paper_iv_h2o_fixture_from_builder:paper_iv_h2o_fixture_from_builder_v1"
SUMMARY_SCHEMA = "paper_iv_h2o_fixture_from_builder_summary_v1"
LANE_MODE_LABELS = ("bend", "symmetric_stretch", "antisymmetric_stretch")
MODEL_DECIDED_ON = "2026-11-02"
# The Paper IV H2O centre geometry (Bohr), from the production fixture
# h2o_linear_fd_nph3_reference_nph4_fixture_20260731; its masses are the default isotope masses.
PAPER_IV_H2O_GEOMETRY = MolecularGeometry(
    ("O", "H", "H"),
    np.array([[0.0, 7.452305433104561e-16, -0.06340060329635555],
              [0.0, 1.432278617817422, 1.1388573462159217],
              [0.0, -1.4322786178174232, 1.1388573462159226]]),
)
MODEL_DEFINITION_KEYS = (
    "harmonic_curvature", "alignment", "mode_basis", "reference_force_counter_term",
    "rhf_hessian_frequencies_cm1", "frozen_reference_frequencies_cm1", "pes_consistent_frequencies_cm1",
    "model_consistent_frequencies_cm1", "fci_surface_curvature_hartree_per_q2", "active_response_hartree_per_q2",
    "active_response_drift", "fci_surface_hessian_hartree_per_q2", "active_response_matrix_hartree_per_q2",
    "model_consistent_rotation", "center_fci_energy_hartree", "hf_energy_hartree",
    "center_gradient_max_abs_hartree_per_bohr", "q_step_au", "q_step_alt_au", "backend", "backend_version",
)


def _tensor_payload(scalar: float, one_body: np.ndarray, two_body: np.ndarray) -> dict[str, Any]:
    return {"scalar_energy_hartree": float(scalar),
            "one_body_integrals": np.asarray(one_body, dtype=float).tolist(),
            "two_body_integrals": np.asarray(two_body, dtype=float).tolist()}


def lane_active_space_kind(space: Any) -> str:
    """The lane's name for the space: ``full_mo_space`` with nothing frozen or discarded, the
    valence name (``cas_8e_6o_valence`` for H2O/STO-3G) for a frozen-core valence space, else the builder's."""
    if not space.frozen_core and not space.external:
        return "full_mo_space"
    if tuple(space.frozen_core) == tuple(range(len(space.frozen_core))) and not space.external:
        return f"cas_{2 * int(space.n_alpha)}e_{len(space.active)}o_valence"
    return str(space.kind)


DECIDED_MODEL = ("pes_consistent", "block_diagonal", "model_consistent")


def _geometry_source(geometry: MolecularGeometry, given: str | None) -> str:
    if given is not None:
        return str(given)
    same = tuple(geometry.symbols) == tuple(PAPER_IV_H2O_GEOMETRY.symbols) and np.allclose(
        geometry.coordinates_bohr, PAPER_IV_H2O_GEOMETRY.coordinates_bohr, atol=1e-12)
    return "paper_iv_h2o_fixture_center_20260731" if same else "caller-supplied geometry"


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            count += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), count


def builder_backend_record(
    geometry: MolecularGeometry,
    *,
    basis: str = "sto-3g",
    active_space: str | Mapping[str, Any] = "frozen_core",
    q_step_au: float = 0.1,
    q_step_alt_au: float = 0.05,
    harmonic_curvature: str = "pes_consistent",
    mode_basis: str | None = None,
    masses_amu: Mapping[str, float] | None = None,
    block_overlap_tolerance: float = 0.5,
    geometry_source: str | None = None,
) -> tuple[dict[str, Any], VibronicQubitModel, CouplingComputation]:
    """The ``h2o_linear_fd_backend_record_v1`` of the builder's model at ``geometry``.

    The builder decides the modes and their frequencies; this function labels the modes by
    their H2O internal-coordinate character, orders them bend / symmetric_stretch /
    antisymmetric_stretch, reconverges the SCF at +/- q_step and +/- q_step_alt along each
    (oriented) model mode, aligns each displaced SCF to the centre block by block and
    contracts the frozen core. Returns the record with the builder's model and coupling
    computation (cutoff 1, used only for the modes, couplings and provenance).
    """
    alignment = "block_diagonal"
    model, computation = build_molecular_vibronic_model(
        geometry, basis=basis, active_space=active_space, cutoffs=1, q_step_au=q_step_au, q_step_alt_au=q_step_alt_au,
        masses_amu=masses_amu, harmonic_curvature=harmonic_curvature, alignment=alignment, mode_basis=mode_basis)
    if str(model.coupling_model) != "full_linear":
        raise ValueError("the Paper IV linear-FD fixture carries the full linear coupling; the builder model must too.")
    center, space, masses = computation.center, computation.active_space, computation.masses_me
    if len(model.modes) != 3:
        raise ValueError(f"the Paper IV H2O record needs three modes; the builder found {len(model.modes)}.")
    prov = dict(model.provenance)
    n_modes = len(model.modes)
    rotation = np.asarray(prov.get("model_consistent_rotation") or np.eye(n_modes), dtype=float)
    rhf_cm1 = [float(v) for v in prov["rhf_hessian_frequencies_cm1"]]
    decided = (str(prov["harmonic_curvature"]), str(prov["alignment"]), str(prov["mode_basis"])) == DECIDED_MODEL
    geometry_source = _geometry_source(geometry, geometry_source)

    rows: list[dict[str, Any]] = []
    for a, mode in enumerate(model.modes):
        vector = np.asarray(mode.mass_weighted_vector, dtype=float)
        dominant = int(np.argmax(np.abs(rotation[:, a])))
        rows.append({
            "mode_index": int(a), "label": str(mode.label),
            "frequency_hartree": float(mode.frequency_hartree), "frequency_cm1": float(mode.frequency_cm1),
            "mass_weighted_eigenvector": vector,
            "mode_character": _h2o_mode_character(symbols=geometry.symbols, coordinates_bohr=geometry.coordinates_bohr,
                                                  mode_vector=vector, masses_me=masses),
            "mode_character_for_negative_vector": _h2o_mode_character(
                symbols=geometry.symbols, coordinates_bohr=geometry.coordinates_bohr, mode_vector=-vector, masses_me=masses),
            "builder_mode_index": int(a),
            "rhf_mode_weights": [float(v) for v in rotation[:, a]],
            "dominant_rhf_mode_index": dominant,
            "rhf_hessian_frequency_cm1_of_dominant_mode": rhf_cm1[dominant],
        })
    labeled = [dict(r) for r in _assign_h2o_mode_labels_by_character(rows)]
    if tuple(str(r["label"]) for r in labeled) != LANE_MODE_LABELS:
        raise ValueError(f"mode labelling by character did not yield {LANE_MODE_LABELS}: {[r['label'] for r in labeled]}")
    for row in labeled:
        # The assignment maximizes the summed character scores; demand that each mode's own dominant
        # character agrees, so two modes of one character cannot be split between labels silently.
        dominant = str(dict(row["mode_character"]).get("dominant_label"))
        if dominant != str(row["label"]):
            raise ValueError(f"mode labelled {row['label']!r} has dominant internal-coordinate character {dominant!r}; "
                             "the model modes do not map one-to-one onto bend / symmetric_stretch / antisymmetric_stretch.")
        # The lane orients each vector so its own indicator is positive; carry that sign into the
        # weights so sum_nu w_nu e_nu^RHF reproduces the stored vector, not its negative.
        a = int(row["builder_mode_index"])
        orientation = float(np.sign(np.sum(np.asarray(row["mass_weighted_eigenvector"], dtype=float)
                                           * np.asarray(model.modes[a].mass_weighted_vector, dtype=float))))
        row["orientation_sign_vs_builder_mode"] = orientation
        row["rhf_mode_weights"] = [float(orientation * v) for v in rotation[:, a]]
        row["trans_rot_overlap"] = float(_translation_rotation_overlap(
            np.asarray(row["mass_weighted_eigenvector"], dtype=float), coordinates_bohr=geometry.coordinates_bohr, masses_me=masses))

    mf = center.backend_handle["mf"]
    center_dm = mf.make_rdm1()
    gradient = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)
    gradient_diagnostics = {
        "available": True, "units": "hartree_per_bohr", "backend": "pyscf",
        "gradient_hartree_per_bohr": gradient.tolist(),
        "fro_norm_hartree_per_bohr": float(np.linalg.norm(gradient)),
        "max_abs_hartree_per_bohr": float(np.max(np.abs(gradient))),
    }
    partition = (space.frozen_core, space.active, space.external)
    active = [int(i) for i in space.active]
    other = [i for i in range(center.n_spatial_orbitals) if i not in set(active)]
    aligned_tensors: list[dict[str, Any]] = []
    for row in labeled:
        label = str(row["label"])
        vector = np.asarray(row["mass_weighted_eigenvector"], dtype=float)
        probe = NormalMode(int(row["mode_index"]), float(row["frequency_hartree"]), vector, label)
        for step_kind, q in (("primary", float(q_step_au)), ("alt", float(q_step_alt_au))):
            for sign in (1, -1):
                displacement_id = f"{step_kind}_{label}_{'plus' if sign > 0 else 'minus'}"
                displaced = displace_along_mode(geometry, probe, masses, sign * q)
                snap = pyscf_rhf_snapshot(displaced, basis=basis, initial_density=center_dm)
                s_cross = pyscf_ao_cross_overlap(center, snap)
                h, g, _, diag = align_to_center(snap, center, cross_overlap_ao=s_cross, blocks=partition, partition=partition)
                _check_block_overlap(diag["block_min_singular_values"], block_overlap_tolerance, where=displacement_id)
                tensors = frozen_core_contract(h, g, snap.nuclear_repulsion, space)
                overlap_mo = center.mo_coeff.T @ np.asarray(s_cross, dtype=float) @ snap.mo_coeff
                o_aa = overlap_mo[np.ix_(active, active)]
                w_act, s_act, vh_act = np.linalg.svd(o_aa, full_matrices=False)
                r_aa = vh_act.T @ w_act.T   # the active block of the block-diagonal rotation, as align_to_center applied it
                delta = np.asarray(displaced.coordinates_bohr, dtype=float) - np.asarray(geometry.coordinates_bohr, dtype=float)
                aligned_tensors.append({
                    "aligned_tensor_id": f"aligned_{displacement_id}",
                    "source_snapshot_id": f"snap_{displacement_id}",
                    "displacement_id": displacement_id,
                    "mode_index": int(row["mode_index"]), "mode_label": label,
                    "step_kind": step_kind, "sign": int(sign),
                    "q_displacement_au": float(q), "signed_q_displacement_au": float(sign * q),
                    "mass_weighted_displacement_norm_au": float(np.sqrt(np.sum(masses[:, None] * delta ** 2))),
                    "max_cartesian_displacement_bohr": float(np.max(np.abs(delta))),
                    "geometry_id": f"h2o_{displacement_id}",
                    "coordinates_bohr": np.asarray(displaced.coordinates_bohr, dtype=float).tolist(),
                    "scf": {"hf_energy_hartree": float(snap.hf_energy), "nuclear_repulsion_hartree": float(snap.nuclear_repulsion),
                            "backend": snap.backend, "backend_version": snap.backend_version},
                    "alignment": {
                        "alignment_id": f"align_{displacement_id}", "displacement_id": displacement_id,
                        "policy": alignment,
                        "blocks": {"frozen_core": list(space.frozen_core), "active": active, "external": list(space.external)},
                        # Active-block diagnostics: the rotation the lane validates is the active one; the
                        # block-diagonal alignment leaves the frozen core canonical by construction.
                        "singular_values": [float(v) for v in s_act],
                        "min_singular_value": float(np.min(s_act)),
                        "alignment_residual_fro": float(np.linalg.norm(o_aa @ r_aa - np.eye(len(active)))),
                        "active_rotation": r_aa.tolist(),
                        # "external" in these two keys is the generator's meaning: every non-active
                        # orbital of the displaced SCF (the frozen core here), not the builder's
                        # discarded virtuals listed under blocks.external.
                        "leakage_partner_orbitals": [int(i) for i in other],
                        "active_to_external_leakage_fro": float(np.linalg.norm(overlap_mo[np.ix_(active, other)])) if other else None,
                        "external_to_active_leakage_fro": float(np.linalg.norm(overlap_mo[np.ix_(other, active)])) if other else None,
                        "full_singular_values": [float(v) for v in np.linalg.svd(overlap_mo, compute_uv=False)],
                        "full_set_residual_fro": float(diag["alignment_residual_fro"]),
                        "block_min_singular_values": [float(v) for v in diag["block_min_singular_values"]],
                        "occupied_block_min_singular_value": float(diag["occupied_block_min_singular_value"]),
                        "passed": True,
                    },
                    **_tensor_payload(tensors.scalar, tensors.one_body, tensors.two_body),
                })

    normal_modes = []
    for row in labeled:
        vector = np.asarray(row["mass_weighted_eigenvector"], dtype=float)
        normal_modes.append({
            "mode_index": int(row["mode_index"]), "label": str(row["label"]),
            "frequency_hartree": float(row["frequency_hartree"]), "frequency_cm1": float(row["frequency_cm1"]),
            "frequency_source": "builder " + str(prov["harmonic_curvature"]) + " / " + str(prov["mode_basis"]),
            "mass_weighted_eigenvector": vector.tolist(),
            "q_step_au": float(q_step_au), "q_step_alt_au": float(q_step_alt_au),
            "label_source": str(row.get("label_source", "internal_coordinate_character_v1")),
            "mode_character": dict(row.get("mode_character", {})),
            "builder_mode_index": int(row["builder_mode_index"]),
            "orientation_sign_vs_builder_mode": float(row["orientation_sign_vs_builder_mode"]),
            "rhf_mode_weights": list(row["rhf_mode_weights"]),
            "dominant_rhf_mode_index": int(row["dominant_rhf_mode_index"]),
            "rhf_hessian_frequency_cm1_of_dominant_mode": float(row["rhf_hessian_frequency_cm1_of_dominant_mode"]),
            "trans_rot_overlap": float(row["trans_rot_overlap"]),
            "primary_max_cartesian_displacement_bohr": float(np.max(np.abs(float(q_step_au) * vector / np.sqrt(masses)[:, None]))),
            "alt_max_cartesian_displacement_bohr": float(np.max(np.abs(float(q_step_alt_au) * vector / np.sqrt(masses)[:, None]))),
        })
    active_kind = lane_active_space_kind(space)
    t0 = computation.active_tensors
    model_definition = {k: prov.get(k) for k in MODEL_DEFINITION_KEYS}
    model_definition.update({
        "builder": "src.quantum.chemistry.molecular_vibronic", "builder_model_schema": MODEL_SCHEMA,
        "coupling_model": str(model.coupling_model),
        "decided_model": bool(decided), "decided_on": MODEL_DECIDED_ON if decided else None,
        "geometry_source": geometry_source,
        # The RHF normal modes the rotation refers to (mass-weighted, builder order), so the model
        # modes can be reconstructed from the fixture alone: e_lane = sign * sum_nu U[nu, a] e_nu^RHF.
        "rhf_normal_mode_vectors": [np.asarray(m.mass_weighted_vector, dtype=float).tolist() for m in computation.modes],
        "lane_label_by_builder_mode": {str(r["builder_mode_index"]): str(r["label"]) for r in labeled},
        "lane_orientation_by_builder_mode": {str(r["builder_mode_index"]): float(r["orientation_sign_vs_builder_mode"]) for r in labeled},
        "lane_frequencies_cm1": {str(r["label"]): float(r["frequency_cm1"]) for r in labeled},
        "displaced_scf_along": "model modes (the record's normal_modes), aligned block by block, canonical frozen core",
    })
    report_summary = {
        "psi4_backend_record_ready": False,
        "pyscf_backend_record_ready": True,
        "backend_record_producer": PRODUCER,
        "normal_mode_label_policy": "internal_coordinate_character_v1",
        "active_space_policy": str(active_space),
        "active_space_kind": active_kind,
        "center_gradient_diagnostics": gradient_diagnostics,
        "normal_mode_diagnostics": [
            {k: m[k] for k in ("mode_index", "label", "frequency_hartree", "frequency_cm1", "frequency_source", "label_source",
                               "mode_character", "builder_mode_index", "orientation_sign_vs_builder_mode", "rhf_mode_weights",
                               "dominant_rhf_mode_index", "rhf_hessian_frequency_cm1_of_dominant_mode", "trans_rot_overlap",
                               "primary_max_cartesian_displacement_bohr", "alt_max_cartesian_displacement_bohr")}
            for m in normal_modes],
        "model_definition": model_definition,
    }
    if active_kind != "cas_8e_6o_valence":
        report_summary["paper_iv_active_space_variant"] = active_kind
    record = {
        "schema": BACKEND_RECORD_SCHEMA,
        "backend": {
            "name": "pyscf", "method": "scf", "basis": str(basis), "reference": "rhf", "scf_type": "incore",
            "version": str(center.backend_version), "conv_tol": 1.0e-12, "conv_tol_grad": 1.0e-9,
            "model_definition": model_definition,
        },
        "system": {"molecule": "H2O", "charge": int(geometry.charge), "multiplicity": int(geometry.multiplicity),
                   "optimized": False, "geometry_source": str(geometry_source)},
        "center_snapshot_id": "snap_center",
        "geometry": {
            "geometry_id": "h2o_center", "symbols": [str(v) for v in geometry.symbols],
            "coordinates_bohr": np.asarray(geometry.coordinates_bohr, dtype=float).tolist(),
            "masses_me": np.asarray(masses, dtype=float).tolist(),
            "gradient_diagnostics": gradient_diagnostics,
            "provenance": {"coordinate_units": "bohr", "mass_units": "electron_mass", "gradient_units": "hartree_per_bohr",
                           "geometry_source": str(geometry_source)},
        },
        "active_space": {
            "active_space_kind": active_kind,
            "frozen_core_indices": [int(v) for v in space.frozen_core],
            "active_indices_center": active,
            "external_indices": [int(v) for v in space.external],
            "n_spatial_orbitals": int(len(active)),
            "num_particles": [int(space.n_alpha), int(space.n_beta)],
            **_tensor_payload(t0.scalar, t0.one_body, t0.two_body),
        },
        "normal_modes": normal_modes,
        "aligned_tensors": aligned_tensors,
        "report_summary": report_summary,
        "provenance": {
            "generator": PRODUCER,
            "q_step_au": float(q_step_au), "q_step_alt_au": float(q_step_alt_au),
            "model_definition": model_definition,
        },
    }
    return record, model, computation


def write_paper_iv_h2o_fixture_from_builder(
    output_dir: str | Path,
    *,
    tag: str,
    geometry: MolecularGeometry = PAPER_IV_H2O_GEOMETRY,
    mode_cutoffs: Sequence[int] = (3, 3, 3),
    reference_cutoffs: Sequence[int] | None = None,
    derivative_validation_tier: str = "production",
    max_derivative_drift: float = 1.0e-5,
    force: bool = False,
    **record_kwargs: Any,
) -> dict[str, Any]:
    """Write ``<tag>_record.json``, ``<tag>_fixture.json`` and ``<tag>_summary.json`` into ``output_dir``.

    ``reference_cutoffs`` default to one more than each work cutoff (the generator's own rule);
    given explicitly, each must exceed its work cutoff. The generator validates the
    finite-difference drift between the two steps with the production tier at relative tolerance
    ``max_derivative_drift``; 1e-5 is what the production fixture of 2026-08-26 was validated with
    (its records carry ``rel_tol: 1e-05``). The fixture passes the lane's evidence gate
    (``validate_paper_iv_h2o_linear_fd_evidence_fixture``) before it is written and again after
    reloading; the summary cross-checks the fixture's exact sector ground energy against the
    builder's own exact solution of the same model at the work cutoffs (permuted from lane order
    into the builder's mode order).
    """
    started = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {kind: out / f"{tag}_{kind}.json" for kind in ("record", "fixture", "summary")}
    if not force:
        existing = [str(p) for p in paths.values() if p.exists()]
        if existing:
            raise FileExistsError(f"output exists (pass force=True to overwrite): {existing}")
    cutoffs = tuple(int(v) for v in mode_cutoffs)
    reference = None if reference_cutoffs is None else tuple(int(v) for v in reference_cutoffs)
    if reference is not None and (len(reference) != len(cutoffs) or any(r <= w for r, w in zip(reference, cutoffs))):
        raise ValueError(f"each reference cutoff must exceed its work cutoff; got reference {reference} for work {cutoffs}.")
    record, model, computation = builder_backend_record(geometry, **record_kwargs)
    fixture = build_h2o_linear_fd_fixture_from_record(
        record, mode_cutoffs=cutoffs, reference_cutoffs=reference, require_reference_cutoff=True,
        derivative_validation_tier=str(derivative_validation_tier), max_derivative_drift=float(max_derivative_drift),
        exact_reference_policy="auto_sector")
    validate_paper_iv_h2o_linear_fd_evidence_fixture(fixture)
    payload = production_vibronic_h2o_fixture_to_jsonable(fixture)
    _write_json(paths["record"], record, force=bool(force))
    _write_json(paths["fixture"], payload, force=bool(force))
    record_path, fixture_path = paths["record"], paths["fixture"]
    reloaded = load_production_vibronic_h2o_fixture(fixture_path)
    validate_paper_iv_h2o_linear_fd_evidence_fixture(reloaded)
    fixture_sha256, fixture_bytes = _sha256_file(fixture_path)
    record_file_sha256, _ = _sha256_file(record_path)

    # The same model (the builder's modes and couplings) solved by the builder at the work cutoffs,
    # which the lane lists in bend / symmetric_stretch / antisymmetric_stretch order.
    lane_to_builder = {int(m["mode_index"]): int(m["builder_mode_index"]) for m in record["normal_modes"]}
    builder_cutoffs = [0] * len(cutoffs)
    for lane_index, cutoff in enumerate(cutoffs):
        builder_cutoffs[lane_to_builder[lane_index]] = int(cutoff)
    work_model = assemble_vibronic_model(
        geometry=geometry, basis=str(model.basis), active_space=computation.active_space,
        active_tensors=computation.active_tensors, modes=model.modes, couplings=model.couplings,
        cutoffs=builder_cutoffs, coupling_model=str(model.coupling_model))
    builder_energy = float(exact_sector_energies(work_model, n_low=1)[0][0])
    exact = fixture.exact_reference
    fixture_energy = None if exact.ground_energy_hartree is None else float(exact.ground_energy_hartree)
    applied = {(str(d.finite_difference_diagnostics.get("tier")), float(d.finite_difference_diagnostics.get("rel_tol")))
               for d in fixture.first_derivatives}
    if len(applied) != 1:
        raise ValueError(f"the generator applied different drift settings per mode: {sorted(applied)}")
    applied_tier, applied_rel_tol = next(iter(applied))
    definition = record["backend"]["model_definition"]
    summary = {
        "schema": SUMMARY_SCHEMA, "producer": PRODUCER, "tag": str(tag),
        "decided_model": bool(definition["decided_model"]), "model_decided_on": definition["decided_on"],
        "fixture_file": fixture_path.name, "fixture_sha256": fixture_sha256, "fixture_byte_count": int(fixture_bytes),
        "fixture_payload_sha256": sha256_jsonable(payload),
        "record_file": record_path.name, "record_file_sha256": record_file_sha256,
        "record_payload_sha256": sha256_jsonable(record),
        "fixture_schema": str(fixture.manifest.schema), "production_status": str(fixture.manifest.production_status),
        "work_cutoffs": list(cutoffs), "builder_order_cutoffs": builder_cutoffs,
        "reference_cutoffs": None if fixture.cutoff_diagnostics is None or fixture.cutoff_diagnostics.reference_cutoffs is None
        else [int(v) for v in fixture.cutoff_diagnostics.reference_cutoffs],
        "n_total_qubits": int(fixture.layout.n_total_qubits), "sector_dimension": exact.sector_dimension,
        "mode_labels": [str(m.label) for m in fixture.normal_modes],
        "mode_frequencies_cm1": [float(m.frequency_cm1) for m in fixture.normal_modes],
        "exact_reference_method": str(exact.method),
        "fixture_ground_energy_hartree": fixture_energy,
        "reference_cutoff_ground_energy_hartree": fixture.cutoff_diagnostics.reference_ground_energy_hartree
        if fixture.cutoff_diagnostics is not None else None,
        "cutoff_diagnostics_passed": None if fixture.cutoff_diagnostics is None else bool(fixture.cutoff_diagnostics.passed),
        "builder_ground_energy_hartree": builder_energy,
        "builder_minus_fixture_hartree": None if builder_energy is None or fixture_energy is None else builder_energy - fixture_energy,
        # The settings the generator applied, read back from its per-mode diagnostics (a non-positive
        # requested tolerance falls back to the tier default inside the generator).
        "derivative_validation": {"tier": applied_tier, "rel_tol": applied_rel_tol},
        "finite_difference_drift_by_mode": {str(d.mode_label): float(d.finite_difference_drift) for d in fixture.first_derivatives},
        "model_definition": definition,
        "evidence_validation": "passed",
        "elapsed_seconds": time.time() - started,
    }
    _write_json(paths["summary"], summary, force=bool(force))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper IV H2O production fixture from the vibronic builder's decided model.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", required=True, help="file stem: <tag>_record.json, <tag>_fixture.json, <tag>_summary.json")
    parser.add_argument("--geometry", default=None, help="'Sym x y z; ...' (default: the Paper IV H2O centre geometry)")
    parser.add_argument("--units", default="angstrom", choices=["angstrom", "bohr"])
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--active-space", default="frozen_core")
    parser.add_argument("--mode-cutoffs", default="3,3,3")
    parser.add_argument("--reference-mode-cutoffs", default=None, help="default: one more than each work cutoff")
    parser.add_argument("--q-step", type=float, default=0.1)
    parser.add_argument("--q-step-alt", type=float, default=0.05)
    parser.add_argument("--harmonic-curvature", default="pes_consistent")
    parser.add_argument("--mode-basis", default=None)
    parser.add_argument("--derivative-validation-tier", default="production")
    parser.add_argument("--max-derivative-drift", type=float, default=1.0e-5,
                        help="relative drift tolerance between the two finite-difference steps (the production fixture used 1e-5)")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    geometry = PAPER_IV_H2O_GEOMETRY if args.geometry is None else parse_geometry(str(args.geometry), units=str(args.units))
    mode_cutoffs = _parse_int_tuple(str(args.mode_cutoffs), label="mode-cutoffs")
    reference = None if args.reference_mode_cutoffs is None else _parse_int_tuple(str(args.reference_mode_cutoffs), label="reference-mode-cutoffs")
    summary = write_paper_iv_h2o_fixture_from_builder(
        args.output_dir, tag=str(args.tag), geometry=geometry, mode_cutoffs=mode_cutoffs, reference_cutoffs=reference,
        derivative_validation_tier=str(args.derivative_validation_tier), max_derivative_drift=float(args.max_derivative_drift),
        force=bool(args.force), basis=str(args.basis), active_space=str(args.active_space), q_step_au=float(args.q_step),
        q_step_alt_au=float(args.q_step_alt), harmonic_curvature=str(args.harmonic_curvature), mode_basis=args.mode_basis)
    print(json.dumps({k: summary[k] for k in ("fixture_file", "fixture_sha256", "fixture_byte_count", "n_total_qubits",
                                              "mode_labels", "mode_frequencies_cm1", "reference_cutoffs",
                                              "fixture_ground_energy_hartree", "builder_ground_energy_hartree",
                                              "builder_minus_fixture_hartree", "cutoff_diagnostics_passed", "decided_model",
                                              "elapsed_seconds")}, indent=1))


if __name__ == "__main__":
    main()
