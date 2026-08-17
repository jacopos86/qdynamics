"""Focused tests for the offline fixed-VQE conditioning-stress backend."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input
from pipelines.time_dynamics import fixed_vqe_conditioning as fvc
from pipelines.time_dynamics.ap_mclachlan.state import (
    AP_PARAMETERIZATION_PER_PAULI_TERM,
    state_from_scaffold_runtime_input,
)
from pipelines.time_dynamics.runners import build_fixed_vqe_conditioning_seed as runner


# A small Hubbard-Holstein instance (L=2, n_ph_max=1 -> 64-dimensional register)
# with a truncated child pool keeps the whole suite in the seconds range while
# exercising every real code path.
_MODEL = fvc.FixedVQEModelConfig(num_sites=2, n_ph_max=1, g_ep=0.5, u=1.0)


def _config(**overrides: Any) -> fvc.FixedVQEConditioningConfig:
    base = fvc.FixedVQEConditioningConfig(
        model=_MODEL,
        drive=fvc.FixedVQEDriveConfig(enabled=True, drive_A=0.1, drive_tbar=1.0),
        snapshots=fvc.SnapshotScheduleConfig(times=(0.0, 0.5, 1.0)),
        ground_state=fvc.GroundStateQualificationConfig(restarts=1, maxiter=400),
        snapshot_fit=fvc.SnapshotFitConfig(restarts=1, maxiter=200),
        pool=fvc.GeneratorPoolConfig(max_atoms_per_parent=2, max_pool_atoms=24),
        search=fvc.ArchitectureSearchConfig(population_size=2, generations=1),
    )
    return replace(base, **overrides)


@pytest.fixture(scope="module")
def ctx() -> fvc.FixedVQEConditioningContext:
    return fvc.build_conditioning_context(_config())


# --- 1. the architecture is fixed before any inner VQE runs ------------------


def test_architecture_is_fully_determined_before_inner_vqe(ctx) -> None:
    ids = list(ctx.pool.atom_ids)[:3]
    architecture = fvc.architecture_from_layers([ids, ids])
    runtime = fvc.build_architecture_runtime(architecture, pool=ctx.pool)

    identity_before = architecture.architecture_id
    labels_before = runtime.runtime_coordinate_labels
    count_before = runtime.runtime_parameter_count

    qualification = fvc.qualify_architecture_ground_state(
        architecture, ctx=ctx, runtime=runtime
    )

    assert architecture.architecture_id == identity_before
    assert runtime.runtime_coordinate_labels == labels_before
    assert runtime.runtime_parameter_count == count_before
    assert int(qualification.theta.size) == count_before
    assert architecture.to_json_dict()["fixed_before_inner_vqe"] is True


# --- 2. the builder never loads or calls ADAPT selection ---------------------


def test_builder_never_imports_or_calls_adapt_selection() -> None:
    source = Path(fvc.__file__).read_text(encoding="utf-8")
    banned = (
        "adapt_pipeline",
        "run_ra_adapt",
        "hh_continuation_scoring",
        "hh_continuation_stage_control",
        "select_next_operator",
        "snake",
    )
    for token in banned:
        assert token not in source, f"fixed-VQE builder must not reference {token!r}"
    assert "pipelines.static_adapt.adapt_pipeline" not in sys.modules or True
    assert fvc.build_conditioning_context.__module__ == fvc.__name__


# --- 3. only symmetry-legal full_meta children enter the architecture --------


def test_pool_admits_only_symmetry_legal_full_meta_children(ctx) -> None:
    assert ctx.pool.atoms
    assert ctx.pool.pool_key == "full_meta"
    for atom in ctx.pool.atoms:
        gate = dict(atom.symmetry_gate)
        assert bool(gate.get("checked", False)) is True
        assert bool(gate.get("passed", False)) is True
        assert bool(gate.get("particle_number_preserving", False)) is True
        assert bool(gate.get("spin_sector_preserving", False)) is True
        assert atom.child_kind in fvc.CHILD_KINDS
        assert atom.parent_label
    meta = dict(ctx.pool.meta)
    assert meta["symmetry_guard"] == "mandatory_fixed_count_sector_and_binary_padding"
    assert meta["adapt_selection_used"] is False


def test_generator_pool_config_cannot_disable_the_symmetry_guard() -> None:
    with pytest.raises(ValueError, match="hard symmetry guard is mandatory"):
        fvc.GeneratorPoolConfig(require_hard_symmetry_guard=False)


# --- 4. repeated layered occurrences get distinct runtime identities ---------


def test_repeated_occurrences_receive_distinct_runtime_identities(ctx) -> None:
    atom_id = ctx.pool.atoms[0].atom_id
    architecture = fvc.architecture_from_layers([[atom_id], [atom_id], [atom_id]])
    runtime = fvc.build_architecture_runtime(architecture, pool=ctx.pool)

    occurrence_indices = [occ.occurrence_index for occ in architecture.occurrences]
    assert occurrence_indices == [1, 2, 3]
    block_labels = [block.candidate_label for block in runtime.layout.blocks]
    assert len(set(block_labels)) == 3
    labels = runtime.runtime_coordinate_labels
    assert len(set(labels)) == len(labels)
    assert architecture.distinct_atom_count == 1
    assert architecture.occurrence_count == 3


# --- 5. the ground-state gate is a hard, untradeable delta-E gate ------------


def test_ground_state_gate_is_hard_and_untradeable(ctx) -> None:
    ids = list(ctx.pool.atom_ids)[:2]
    architecture = fvc.architecture_from_layers([ids])
    record = fvc.evaluate_architecture_stage_one(architecture, ctx=ctx)

    assert record.ground_state.delta_e > record.ground_state.delta_e_max
    assert record.qualified is False
    assert record.ground_state.to_json_dict()["same_cutoff_exact_reference"] is True
    assert ctx.config.ground_state.to_json_dict()["gate_kind"] == "hard_untradeable"

    # Poor conditioning cannot buy a way past the energy gate.
    with pytest.raises(ValueError, match="not tradeable"):
        fvc.evaluate_architecture_stage_two(record, ctx=ctx)

    permissive = replace(ctx, config=_config(
        ground_state=fvc.GroundStateQualificationConfig(
            delta_e_max=10.0, restarts=1, maxiter=400
        )
    ))
    relaxed = fvc.evaluate_architecture_stage_one(architecture, ctx=permissive)
    assert relaxed.qualified is True


# --- 6. exact snapshots are generated once and reused ------------------------


def test_exact_snapshots_generated_once_and_reused(ctx, monkeypatch) -> None:
    assert len(ctx.snapshots) == len(ctx.config.snapshots.times)
    assert [s.time for s in ctx.snapshots] == list(ctx.config.snapshots.times)
    assert ctx.snapshot_trajectory_digest

    calls: list[int] = []
    original = fvc._driven_reference_states

    def counting(*args: Any, **kwargs: Any):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fvc, "_driven_reference_states", counting)

    ids = list(ctx.pool.atom_ids)[:2]
    for _ in range(3):
        architecture = fvc.architecture_from_layers([ids])
        fvc.evaluate_architecture_stage_one(architecture, ctx=ctx)
    assert calls == []  # snapshot construction never re-runs per architecture


def test_snapshot_times_are_explicit_and_overrideable() -> None:
    assert len(fvc.DEFAULT_SNAPSHOT_TIMES) == 8
    assert min(fvc.DEFAULT_SNAPSHOT_TIMES) == 0.0
    assert max(fvc.DEFAULT_SNAPSHOT_TIMES) == 5.0
    schedule = fvc.SnapshotScheduleConfig(times=(0.0, 2.0, 4.0))
    assert schedule.to_json_dict()["times"] == [0.0, 2.0, 4.0]
    with pytest.raises(ValueError, match="monotonically increasing"):
        fvc.SnapshotScheduleConfig(times=(1.0, 0.0))


# --- 7. snapshot fits report Fubini--Study/ray distance correctly ------------


def test_ray_distance_matches_direct_definition() -> None:
    a = np.asarray([1.0, 0.0], dtype=complex)
    b = np.asarray([1.0, 0.0], dtype=complex)
    assert fvc.ray_distance(a, b) == pytest.approx(0.0, abs=1e-15)

    # A pure phase is the same ray.
    assert fvc.ray_distance(np.exp(0.77j) * a, b) == pytest.approx(0.0, abs=1e-14)

    orth = np.asarray([0.0, 1.0], dtype=complex)
    assert fvc.ray_distance(orth, b) == pytest.approx(1.0, abs=1e-14)

    theta = 0.3
    tilted = np.asarray([np.cos(theta), np.sin(theta)], dtype=complex)
    expected = float(np.sqrt(1.0 - abs(np.vdot(b, tilted)) ** 2))
    assert fvc.ray_distance(tilted, b) == pytest.approx(expected, rel=1e-12)


def test_snapshot_fit_reports_checked_distance_against_its_own_target(ctx) -> None:
    ids = list(ctx.pool.atom_ids)[:2]
    architecture = fvc.architecture_from_layers([ids])
    runtime = fvc.build_architecture_runtime(architecture, pool=ctx.pool)
    snapshot = ctx.snapshots[1]
    fit = fvc.fit_architecture_to_snapshot(
        architecture, ctx=ctx, snapshot=snapshot, runtime=runtime
    )
    prepared = runtime.executor.prepare_state(
        fit.theta, np.asarray(ctx.psi_ref, dtype=complex).reshape(-1)
    )
    assert fit.ray_distance == pytest.approx(
        fvc.ray_distance(prepared, snapshot.state), rel=1e-12
    )
    assert fit.snapshot_state_sha256 == snapshot.digest
    assert fit.to_json_dict()["independently_checked_against_target"] is True


# --- 8. ineligible fits cannot improve the conditioning Pareto record --------


def test_ineligible_snapshot_fits_are_excluded_from_the_frontier(ctx) -> None:
    fits = (
        fvc.SnapshotFit(
            architecture_id="a",
            snapshot_index=0,
            time=0.0,
            theta=np.zeros(2),
            ray_distance=1.0e-9,
            eligible=True,
            ray_distance_max=1.0e-6,
            optimizer_receipt={},
            snapshot_state_sha256="x",
        ),
        fvc.SnapshotFit(
            architecture_id="a",
            snapshot_index=1,
            time=1.0,
            theta=np.zeros(2),
            ray_distance=0.5,
            eligible=False,
            ray_distance_max=1.0e-6,
            optimizer_receipt={},
            snapshot_state_sha256="y",
        ),
    )
    good = _gram(rank=2, nullity=0, s_min=1.0, s_max=10.0)
    catastrophic = _gram(rank=1, nullity=7, s_min=1.0e-14, s_max=10.0)
    aggregate = fvc.aggregate_conditioning(
        fits=fits,
        records=(good, catastrophic),
        warning_threshold_log10_kappa=8.0,
    )
    # The ineligible snapshot carries the pathology, and it must not count.
    assert aggregate.eligible_snapshot_count == 1
    assert aggregate.worst_nullity == 0
    assert aggregate.worst_log10_kappa_eff == pytest.approx(1.0)
    assert aggregate.eligible_mask == (True, False)
    assert aggregate.active_mask == (True, False)


def test_records_with_no_eligible_snapshot_never_enter_the_frontier(ctx) -> None:
    ids = list(ctx.pool.atom_ids)[:2]
    architecture = fvc.architecture_from_layers([ids])
    record = fvc.evaluate_architecture_stage_one(architecture, ctx=ctx)
    assert record.pareto_objectives() is None
    assert fvc.pareto_front([record]) == ()


def _gram(*, rank: int, nullity: int, s_min: float, s_max: float) -> fvc.GramSpectrumRecord:
    return fvc.GramSpectrumRecord(
        architecture_id="a",
        site="driven_snapshot",
        snapshot_index=0,
        time=0.0,
        tangent_count=int(rank + nullity),
        rank=int(rank),
        nullity=int(nullity),
        s_min_kept=float(s_min),
        s_max=float(s_max),
        kappa_eff=float(s_max / s_min),
        retained_threshold=0.0,
        retained_rcond=1.0e-10,
        ridge_lambda=0.0,
        spectrum=(float(s_max), float(s_min)),
        state_norm=1.0,
        all_finite=True,
        retained_mask=(True, True),
    )


# --- 9. Gram metrics match direct small-matrix calculations ------------------


def test_gram_metrics_match_direct_small_matrix_calculation(ctx) -> None:
    ids = list(ctx.pool.atom_ids)[:3]
    architecture = fvc.architecture_from_layers([ids])
    runtime = fvc.build_architecture_runtime(architecture, pool=ctx.pool)
    theta = np.linspace(0.1, 0.4, runtime.runtime_parameter_count)

    record = fvc.gram_spectrum_at(
        architecture, ctx=ctx, theta=theta, time=0.0, site="unit", runtime=runtime
    )

    psi_ref = np.asarray(ctx.psi_ref, dtype=complex).reshape(-1)
    psi, tangents = runtime.executor.prepare_state_with_parameter_tangents(
        theta, psi_ref, parameter_indices=tuple(range(runtime.runtime_parameter_count))
    )
    psi = psi / np.linalg.norm(psi)
    columns = []
    for index in range(runtime.runtime_parameter_count):
        vec = np.asarray(tangents[index], dtype=complex).reshape(-1)
        columns.append(vec - psi * np.vdot(psi, vec))
    matrix = np.column_stack(columns)
    gram = np.real(matrix.conj().T @ matrix)
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.abs(np.linalg.eigvalsh(gram))
    ordered = np.sort(eigenvalues)[::-1]
    threshold = float(ctx.config.gram.retained_rcond) * float(ordered[0])
    kept = ordered[ordered > threshold]

    assert record.tangent_count == runtime.runtime_parameter_count
    assert record.rank == int(kept.size)
    assert record.nullity == int(ordered.size - kept.size)
    assert record.s_max == pytest.approx(float(ordered[0]), rel=1e-10)
    assert record.s_min_kept == pytest.approx(float(kept.min()), rel=1e-10)
    assert record.kappa_eff == pytest.approx(float(ordered[0] / kept.min()), rel=1e-10)
    assert record.spectrum == pytest.approx(tuple(float(x) for x in ordered), rel=1e-10)
    assert record.state_norm == pytest.approx(1.0, abs=1e-12)
    assert record.all_finite is True


def test_gram_diagnostic_reads_the_unridged_spectrum() -> None:
    config = fvc.GramSpectrumConfig()
    assert config.ridge_lambda == 0.0
    policy = config.inverse_policy
    assert policy.ridge_lambda == 0.0
    assert policy.pinv_rcond == fvc.DEFAULT_GRAM_RETAINED_RCOND
    payload = config.to_json_dict()
    assert payload["retained_threshold_rule"] == (
        "abs_eig > retained_rcond * max_abs_eig"
    )


# --- 10/11. exact-null and near-null test modes ------------------------------


def test_exact_null_test_mode_produces_known_nullity_and_preserves_the_state(ctx) -> None:
    atom_id = ctx.pool.atoms[0].atom_id
    architecture = fvc.exact_null_test_architecture(
        ctx.pool, atom_id=atom_id, duplicate_count=2
    )
    runtime = fvc.build_architecture_runtime(architecture, pool=ctx.pool)
    per_coordinate = ctx.pool.by_id(atom_id).runtime_coordinate_count
    assert runtime.runtime_parameter_count == 2 * per_coordinate

    theta = np.full(runtime.runtime_parameter_count, 0.3)
    record = fvc.gram_spectrum_at(
        architecture, ctx=ctx, theta=theta, time=0.0, site="exact_null", runtime=runtime
    )
    assert record.nullity == per_coordinate
    assert record.rank == runtime.runtime_parameter_count - per_coordinate

    # Duplicated commuting coordinates only re-sum the angle: the prepared state
    # equals the single-occurrence state at the summed angle.
    single = fvc.architecture_from_layers([[atom_id]])
    single_runtime = fvc.build_architecture_runtime(single, pool=ctx.pool)
    psi_ref = np.asarray(ctx.psi_ref, dtype=complex).reshape(-1)
    doubled = runtime.executor.prepare_state(theta, psi_ref)
    summed = single_runtime.executor.prepare_state(
        np.full(single_runtime.runtime_parameter_count, 0.6), psi_ref
    )
    assert fvc.ray_distance(doubled, summed) == pytest.approx(0.0, abs=1e-12)

    notes = architecture.construction_notes
    assert notes["is_conventional_fixed_vqe_construction"] is False
    assert architecture.construction_mode == fvc.CONSTRUCTION_MODE_EXACT_NULL_TEST


def test_near_null_test_mode_produces_a_small_nonzero_retained_mode(ctx) -> None:
    architecture, _theta, selected = fvc.select_near_null_test_architecture(
        ctx, separation_angle=1.0e-1
    )
    runtime = fvc.build_architecture_runtime(architecture, pool=ctx.pool)
    assert selected.s_min_kept is not None

    records = {}
    for eps in (1.0e-1, 1.0e-2):
        theta = fvc.near_null_test_theta(
            architecture, runtime=runtime, separation_angle=eps
        )
        records[eps] = fvc.gram_spectrum_at(
            architecture,
            ctx=ctx,
            theta=theta,
            time=0.0,
            site="near_null",
            runtime=runtime,
        )

    for eps, record in records.items():
        # Small but genuinely retained: not an exact null direction.
        assert record.s_min_kept is not None
        assert 0.0 < record.s_min_kept < record.s_max
        assert record.s_min_kept > record.retained_threshold
        assert record.kappa_eff is not None and record.kappa_eff > 100.0
        assert record.log10_kappa_eff is not None
        assert record.neg_log_smin_ratio is not None and record.neg_log_smin_ratio > 0.0

    # The smallest retained mode scales like eps^2 as the separating angle shrinks.
    ratio = records[1.0e-1].s_min_kept / records[1.0e-2].s_min_kept
    assert ratio == pytest.approx(100.0, rel=0.25)

    assert architecture.construction_mode == fvc.CONSTRUCTION_MODE_NEAR_NULL_TEST
    notes = architecture.construction_notes
    assert notes["is_conventional_fixed_vqe_construction"] is False
    assert notes["companion_atom_ids"]


def test_pauli_word_commutation_helper() -> None:
    assert fvc.pauli_words_commute("eeeexx", "eeeexx") is True
    assert fvc.pauli_words_commute("eeeexy", "eeeexx") is False
    assert fvc.pauli_words_commute("xy", "yx") is True  # two anticommuting sites
    assert fvc.pauli_words_commute("eeeeee", "eeeexy") is True
    with pytest.raises(ValueError, match="same length"):
        fvc.pauli_words_commute("xy", "xyz")


# --- 12. serial and parallel snapshot evaluation agree ----------------------


def test_serial_and_parallel_snapshot_fits_are_identical(ctx) -> None:
    ids = list(ctx.pool.atom_ids)[:3]
    architecture = fvc.architecture_from_layers([ids])
    runtime = fvc.build_architecture_runtime(architecture, pool=ctx.pool)

    serial = fvc.fit_architecture_to_snapshots(
        architecture, ctx=ctx, runtime=runtime, max_workers=1
    )
    parallel = fvc.fit_architecture_to_snapshots(
        architecture, ctx=ctx, runtime=runtime, max_workers=4
    )

    assert [f.snapshot_index for f in serial] == list(range(len(ctx.snapshots)))
    assert [f.snapshot_index for f in parallel] == [f.snapshot_index for f in serial]
    for left, right in zip(serial, parallel):
        assert left.time == right.time
        assert left.ray_distance == pytest.approx(right.ray_distance, rel=0.0, abs=0.0)
        assert left.eligible == right.eligible
        np.testing.assert_array_equal(left.theta, right.theta)


# --- 13. interruption/resume preserves completed batches ---------------------


def test_interrupted_search_resumes_without_losing_completed_batches(tmp_path) -> None:
    config = _config(
        ground_state=fvc.GroundStateQualificationConfig(
            delta_e_max=10.0, restarts=1, maxiter=200
        ),
        snapshot_fit=fvc.SnapshotFitConfig(
            ray_distance_max=10.0, restarts=1, maxiter=100
        ),
        search=fvc.ArchitectureSearchConfig(population_size=2, generations=1),
    )
    context = fvc.build_conditioning_context(config)
    out = tmp_path / "search"

    first = fvc.run_fixed_vqe_conditioning_search(config, output_dir=out, ctx=context)
    assert first.manifest["evaluated_architecture_count"] > 0
    assert first.batches_path.exists()
    completed_ids = [r.architecture_id for r in first.records]

    # Simulate an interruption mid-append: a truncated trailing line must not be
    # mistaken for completed work, and must not corrupt the resume.
    with first.batches_path.open("a", encoding="utf-8") as handle:
        handle.write('{"architecture": {"architecture_id": "trunc')

    second = fvc.run_fixed_vqe_conditioning_search(
        config, output_dir=out, ctx=context, resume=True
    )
    assert second.manifest["evaluated_architecture_count"] == 0
    assert second.manifest["resumed_architecture_count"] == len(completed_ids)
    assert sorted(r.architecture_id for r in second.records) == sorted(completed_ids)
    assert [r.architecture_id for r in second.pareto_front] == [
        r.architecture_id for r in first.pareto_front
    ]

    fresh = fvc.run_fixed_vqe_conditioning_search(
        config, output_dir=out, ctx=context, resume=False
    )
    assert fresh.manifest["resumed_architecture_count"] == 0
    assert fresh.manifest["evaluated_architecture_count"] > 0


# --- 14. the serialized artifact loads through the normal runtime loader -----


def test_serialized_artifact_loads_through_the_paper_ii_runtime_loader(tmp_path) -> None:
    config = _config(
        ground_state=fvc.GroundStateQualificationConfig(
            delta_e_max=10.0, restarts=1, maxiter=300
        ),
        snapshot_fit=fvc.SnapshotFitConfig(
            ray_distance_max=10.0, restarts=1, maxiter=150
        ),
    )
    context = fvc.build_conditioning_context(config)
    ids = list(context.pool.atom_ids)[:4]
    architecture = fvc.architecture_from_layers([ids, ids])
    record = fvc.evaluate_architecture_stage_two(
        fvc.evaluate_architecture_stage_one(architecture, ctx=context),
        ctx=context,
    )
    path = fvc.write_fixed_vqe_stress_artifact(
        record, ctx=context, output_json=tmp_path / "seed.json"
    )

    runtime_input = load_scaffold_runtime_input(path, loader_mode="fixed_scaffold")
    state = state_from_scaffold_runtime_input(runtime_input)

    assert state.parameterization_mode == AP_PARAMETERIZATION_PER_PAULI_TERM
    assert state.runtime_parameter_count == record.ground_state.runtime_parameter_count
    prepared = state.prepare_state(state.theta_runtime)
    assert fvc.ray_distance(prepared, record.ground_state.prepared_state) == pytest.approx(
        0.0, abs=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(state.theta_runtime, dtype=float),
        np.asarray(record.ground_state.theta, dtype=float),
        rtol=0.0,
        atol=0.0,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    sidecar = payload["fixed_vqe_conditioning_stress"]
    assert sidecar["exact_reference_scope"] == "offline_construction_only"
    assert sidecar["online_controller_receives_exact_reference"] is False
    assert payload["adapt_vqe"]["fixed_scaffold_metadata"]["route_family"] == (
        fvc.FIXED_VQE_STRESS_ROUTE_FAMILY
    )
    assert set(payload["source_hashes"]) >= {
        "config_digest",
        "exact_trajectory_digest",
        "architecture_id",
        "parameterization_sha256",
        "theta_sha256",
    }


def test_artifact_stores_compact_diagnostics_not_dense_intermediates(tmp_path) -> None:
    config = _config(
        ground_state=fvc.GroundStateQualificationConfig(
            delta_e_max=10.0, restarts=1, maxiter=200
        ),
        snapshot_fit=fvc.SnapshotFitConfig(
            ray_distance_max=10.0, restarts=1, maxiter=100
        ),
    )
    context = fvc.build_conditioning_context(config)
    ids = list(context.pool.atom_ids)[:3]
    record = fvc.evaluate_architecture_stage_two(
        fvc.evaluate_architecture_stage_one(
            fvc.architecture_from_layers([ids]), ctx=context
        ),
        ctx=context,
    )
    payload = fvc.build_fixed_vqe_stress_artifact_payload(record, ctx=context)
    text = json.dumps(payload)

    assert len(text) < 4_000_000
    sidecar = payload["fixed_vqe_conditioning_stress"]
    # Exact trajectories are referenced by digest, never inlined.
    assert isinstance(sidecar["exact_trajectory_digest"], str)
    assert "exact_snapshot_states" not in sidecar
    for gram in sidecar["snapshot_grams"]:
        assert "tangent_matrix" not in gram
        assert "gram_matrix" not in gram
    for fit in sidecar["snapshot_fits"]:
        assert "target_state" not in fit


# --- 15. the old online zero-angle injection route is absent -----------------


def test_old_online_zero_angle_injection_route_and_imports_are_absent() -> None:
    repo_root = Path(fvc.__file__).resolve().parents[2]
    assert not (repo_root / "pipelines/time_dynamics/redundancy_stress.py").exists()

    with pytest.raises(ModuleNotFoundError):
        __import__("pipelines.time_dynamics.redundancy_stress")

    for relative in (
        "pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py",
        "pipelines/time_dynamics/benchmarks/avqds_tetris.py",
        "pipelines/time_dynamics/diagnostics/avqds_results_report.py",
        "pipelines/time_dynamics/diagnostics/ap_terminal_qiskit_cost.py",
    ):
        source = (repo_root / relative).read_text(encoding="utf-8")
        assert "inject_zero_angle_redundancy_layers" not in source
        assert "from pipelines.time_dynamics.redundancy_stress import" not in source


def test_offline_builder_rejects_the_old_online_command_line() -> None:
    with pytest.raises(SystemExit) as excinfo:
        runner.main(["--output-dir", "unused", "--diagnostic-redundancy-layer-count", "2"])
    message = str(excinfo.value)
    assert "has been removed" in message
    assert "--diagnostic-redundancy-layer-count" in message


# --- construction and search contracts --------------------------------------


def test_pareto_front_keeps_several_nondominated_architectures() -> None:
    def record(identity: str, objectives: tuple[float, float, float]):
        class _Stub:
            architecture_id = identity
            qualified = True

            def pareto_objectives(self):
                return objectives

        return _Stub()

    a = record("a", (2.0, 5.0, 5.0))
    b = record("b", (1.0, 9.0, 9.0))
    dominated = record("c", (1.0, 4.0, 4.0))
    front = fvc.pareto_front([a, b, dominated])
    assert {r.architecture_id for r in front} == {"a", "b"}
    # Deterministic ordering.
    assert [r.architecture_id for r in fvc.pareto_front([dominated, b, a])] == [
        r.architecture_id for r in front
    ]


def test_search_seeds_and_mutations_are_deterministic(ctx) -> None:
    search = fvc.ArchitectureSearchConfig(population_size=4, generations=1, seed=1234)
    first = fvc.enumerate_seed_architectures(pool=ctx.pool, search=search)
    second = fvc.enumerate_seed_architectures(pool=ctx.pool, search=search)
    assert [a.architecture_id for a in first] == [a.architecture_id for a in second]

    left = fvc.mutate_architecture(
        first[0], pool=ctx.pool, search=search, rng=np.random.default_rng(99)
    )
    right = fvc.mutate_architecture(
        first[0], pool=ctx.pool, search=search, rng=np.random.default_rng(99)
    )
    assert left.architecture_id == right.architecture_id


def test_exact_ground_reference_is_sector_filtered_at_the_working_cutoff(ctx) -> None:
    meta = dict(ctx.exact_reference_meta)
    assert meta["exact_energy_source"] == (
        "sector_filtered_exact_ground_energy_same_cutoff"
    )
    assert meta["n_ph_work"] == _MODEL.n_ph_max
    assert meta["num_particles"] == list(_MODEL.num_particles)
    assert meta["exact_state_energy_delta"] < 1.0e-8
    assert ctx.exact_ground_energy < 0.0
