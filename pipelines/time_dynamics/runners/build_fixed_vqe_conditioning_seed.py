"""CLI for the offline fixed-VQE conditioning-stress seed builder.

This runner only resolves configuration and I/O.  Every construction decision
lives in :mod:`pipelines.time_dynamics.fixed_vqe_conditioning`.

The builder is offline: it may use exact ground states and exact driven
trajectories to construct and qualify diagnostic inputs, and it writes ordinary
runtime-loadable ansatz artifacts.  Nothing here injects exact-reference data
into AP-McLachlan propagation or online control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.fixed_vqe_conditioning import (
    CONSTRUCTION_MODES,
    CONSTRUCTION_MODE_CONVENTIONAL,
    CONSTRUCTION_MODE_EXACT_NULL_TEST,
    CONSTRUCTION_MODE_NEAR_NULL_TEST,
    CONVENTIONAL_HH_LAYER_PARENTS,
    DEFAULT_DELTA_E_MAX,
    DEFAULT_GRAM_RETAINED_RCOND,
    DEFAULT_NEAR_NULL_SEPARATION_ANGLE,
    DEFAULT_SNAPSHOT_RAY_DISTANCE_MAX,
    DEFAULT_SNAPSHOT_TIMES,
    ArchitectureSearchConfig,
    FixedVQEConditioningConfig,
    FixedVQEDriveConfig,
    FixedVQEModelConfig,
    GeneratorPoolConfig,
    GramSpectrumConfig,
    GroundStateQualificationConfig,
    SnapshotFitConfig,
    SnapshotScheduleConfig,
    build_conditioning_context,
    evaluate_architecture_stage_one,
    evaluate_architecture_stage_two,
    exact_null_test_architecture,
    near_null_test_architecture,
    run_fixed_vqe_conditioning_search,
    select_near_null_test_architecture,
    write_fixed_vqe_stress_artifact,
)


RUNNER_SCHEMA_V1 = "build_fixed_vqe_conditioning_seed_v1"

_REMOVED_ONLINE_INJECTION_FLAGS = (
    "--diagnostic-redundancy-layer-count",
    "--diagnostic-redundancy-pool-profile",
    "--diagnostic-redundancy-layout-mode",
    "--diagnostic-redundancy-state-parity-atol",
)


def _parse_float_list(raw: str | None) -> tuple[float, ...] | None:
    if raw in {None, ""}:
        return None
    return tuple(float(token.strip()) for token in str(raw).split(",") if token.strip())


def _parse_int_list(raw: str | None) -> tuple[int, ...] | None:
    if raw in {None, ""}:
        return None
    return tuple(int(token.strip()) for token in str(raw).split(",") if token.strip())


def _parse_str_list(raw: str | None) -> tuple[str, ...]:
    if raw in {None, ""}:
        return tuple()
    return tuple(token.strip() for token in str(raw).split(",") if token.strip())


def config_from_args(args: argparse.Namespace) -> FixedVQEConditioningConfig:
    """Resolve the typed construction configuration from parsed CLI arguments."""

    model = FixedVQEModelConfig(
        num_sites=int(args.num_sites),
        t=float(args.t),
        u=float(args.u),
        dv=float(args.dv),
        omega0=float(args.omega0),
        g_ep=float(args.g_ep),
        n_ph_max=int(args.n_ph_max),
        boson_encoding=str(args.boson_encoding),
        ordering=str(args.ordering),
        boundary=str(args.boundary),
        sector_n_up=(None if args.sector_n_up is None else int(args.sector_n_up)),
        sector_n_dn=(None if args.sector_n_dn is None else int(args.sector_n_dn)),
    )
    drive = FixedVQEDriveConfig(
        enabled=bool(args.enable_drive),
        drive_A=float(args.drive_A),
        drive_omega=float(args.drive_omega),
        drive_tbar=float(args.drive_tbar),
        drive_phi=float(args.drive_phi),
        drive_pattern=str(args.drive_pattern),
        drive_custom_weights=_parse_float_list(args.drive_custom_weights),
        drive_include_identity=bool(args.drive_include_identity),
        drive_time_sampling=str(args.drive_time_sampling),
        drive_t0=float(args.drive_t0),
    )
    times = _parse_float_list(args.snapshot_times) or DEFAULT_SNAPSHOT_TIMES
    snapshots = SnapshotScheduleConfig(
        times=tuple(times),
        rtol=float(args.snapshot_rtol),
        atol=float(args.snapshot_atol),
        max_internal_step=(
            None if args.snapshot_max_internal_step is None else float(args.snapshot_max_internal_step)
        ),
        norm_drift_tolerance=float(args.snapshot_norm_drift_tolerance),
    )
    ground_state = GroundStateQualificationConfig(
        delta_e_max=float(args.delta_e_max),
        method=str(args.vqe_method),
        maxiter=int(args.vqe_maxiter),
        restarts=int(args.vqe_restarts),
        seed=int(args.vqe_seed),
    )
    snapshot_fit = SnapshotFitConfig(
        ray_distance_max=float(args.snapshot_ray_distance_max),
        method=str(args.fit_method),
        maxiter=int(args.fit_maxiter),
        restarts=int(args.fit_restarts),
        seed=int(args.fit_seed),
    )
    gram = GramSpectrumConfig(
        retained_rcond=float(args.gram_retained_rcond),
        ridge_lambda=float(args.gram_ridge_lambda),
        conditioning_warning_log10_kappa=float(args.conditioning_warning_log10_kappa),
        store_full_spectrum=bool(args.store_full_spectrum),
    )
    pool = GeneratorPoolConfig(
        pool_key=str(args.pool_key),
        include_pauli_children=bool(args.include_pauli_children),
        include_polyterm_children=bool(args.include_polyterm_children),
        polyterm_subset_sizes=tuple(_parse_int_list(args.polyterm_subset_sizes) or (2,)),
        max_atoms_per_parent=(
            None if args.max_atoms_per_parent is None else int(args.max_atoms_per_parent)
        ),
        max_pool_atoms=(None if args.max_pool_atoms is None else int(args.max_pool_atoms)),
        priority_parent_labels=(
            _parse_str_list(args.priority_parent_labels)
            or CONVENTIONAL_HH_LAYER_PARENTS
        ),
    )
    search = ArchitectureSearchConfig(
        construction_mode=str(args.construction_mode),
        layer_counts=tuple(_parse_int_list(args.layer_counts) or (1, 2, 3)),
        atoms_per_layer=tuple(_parse_int_list(args.atoms_per_layer) or (4, 6, 8)),
        population_size=int(args.population_size),
        generations=int(args.generations),
        mutation_count=int(args.mutation_count),
        seed=int(args.search_seed),
        max_architecture_workers=int(args.max_architecture_workers),
        max_snapshot_workers=int(args.max_snapshot_workers),
        retain_beyond_pareto=int(args.retain_beyond_pareto),
        seed_parent_complete_layers=bool(args.seed_parent_complete_layers),
        seed_parent_labels=_parse_str_list(args.seed_parent_labels),
        seed_parent_complete_repeats=tuple(
            _parse_int_list(args.seed_parent_complete_repeats) or (1, 2)
        ),
    )
    return FixedVQEConditioningConfig(
        model=model,
        drive=drive,
        snapshots=snapshots,
        ground_state=ground_state,
        snapshot_fit=snapshot_fit,
        gram=gram,
        pool=pool,
        search=search,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construct conventional fixed-structure Hubbard-Holstein VQE ansaetze with "
            "controlled McLachlan tangent-geometry pathology, and serialize them as "
            "ordinary runtime-loadable ansatz artifacts."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--construction-mode",
        choices=CONSTRUCTION_MODES,
        default=CONSTRUCTION_MODE_CONVENTIONAL,
        help=(
            "conventional_fixed_layered_v1 is the ordinary construction; the "
            "exact/near-null modes are validation fixtures for pseudoinverse and "
            "null-space behavior and are not conventional fixed-VQE constructions."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--write-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Serialize every nondominated/retained architecture as a seed artifact.",
    )

    model = parser.add_argument_group("Hubbard-Holstein model")
    model.add_argument("--num-sites", type=int, default=2)
    model.add_argument("--t", type=float, default=1.0)
    model.add_argument("--u", type=float, default=1.0)
    model.add_argument("--dv", type=float, default=0.0)
    model.add_argument("--omega0", type=float, default=1.0)
    model.add_argument("--g-ep", type=float, default=0.5)
    model.add_argument("--n-ph-max", type=int, default=1)
    model.add_argument("--boson-encoding", default="binary")
    model.add_argument("--ordering", default="blocked")
    model.add_argument("--boundary", default="open")
    model.add_argument("--sector-n-up", type=int, default=None)
    model.add_argument("--sector-n-dn", type=int, default=None)

    drive = parser.add_argument_group("driven snapshot Hamiltonian")
    drive.add_argument("--enable-drive", action=argparse.BooleanOptionalAction, default=True)
    drive.add_argument("--drive-A", type=float, default=0.1)
    drive.add_argument("--drive-omega", type=float, default=1.0)
    drive.add_argument("--drive-tbar", type=float, default=2.0)
    drive.add_argument("--drive-phi", type=float, default=0.0)
    drive.add_argument("--drive-pattern", default="staggered")
    drive.add_argument("--drive-custom-weights", default=None)
    drive.add_argument("--drive-include-identity", action="store_true")
    drive.add_argument("--drive-time-sampling", default="midpoint")
    drive.add_argument("--drive-t0", type=float, default=0.0)

    snaps = parser.add_argument_group("exact snapshot schedule")
    snaps.add_argument(
        "--snapshot-times",
        default=",".join(str(x) for x in DEFAULT_SNAPSHOT_TIMES),
        help="Explicit comma-separated exact-snapshot times; recorded in the manifest.",
    )
    snaps.add_argument("--snapshot-rtol", type=float, default=1.0e-10)
    snaps.add_argument("--snapshot-atol", type=float, default=1.0e-12)
    snaps.add_argument("--snapshot-max-internal-step", type=float, default=None)
    snaps.add_argument("--snapshot-norm-drift-tolerance", type=float, default=1.0e-8)

    gates = parser.add_argument_group("qualification gates")
    gates.add_argument("--delta-e-max", type=float, default=DEFAULT_DELTA_E_MAX)
    gates.add_argument(
        "--snapshot-ray-distance-max",
        type=float,
        default=DEFAULT_SNAPSHOT_RAY_DISTANCE_MAX,
    )

    vqe = parser.add_argument_group("inner fixed-structure VQE")
    vqe.add_argument("--vqe-method", default="L-BFGS-B")
    vqe.add_argument("--vqe-maxiter", type=int, default=20000)
    vqe.add_argument("--vqe-restarts", type=int, default=4)
    vqe.add_argument("--vqe-seed", type=int, default=7)

    fit = parser.add_argument_group("snapshot fitting")
    fit.add_argument("--fit-method", default="L-BFGS-B")
    fit.add_argument("--fit-maxiter", type=int, default=6000)
    fit.add_argument("--fit-restarts", type=int, default=3)
    fit.add_argument("--fit-seed", type=int, default=11)

    gram = parser.add_argument_group("tangent Gram diagnostics")
    gram.add_argument("--gram-retained-rcond", type=float, default=DEFAULT_GRAM_RETAINED_RCOND)
    gram.add_argument("--gram-ridge-lambda", type=float, default=0.0)
    gram.add_argument("--conditioning-warning-log10-kappa", type=float, default=8.0)
    gram.add_argument(
        "--store-full-spectrum", action=argparse.BooleanOptionalAction, default=True
    )

    pool = parser.add_argument_group("generator pool")
    pool.add_argument("--pool-key", default="full_meta")
    pool.add_argument(
        "--include-pauli-children", action=argparse.BooleanOptionalAction, default=True
    )
    pool.add_argument(
        "--include-polyterm-children", action=argparse.BooleanOptionalAction, default=True
    )
    pool.add_argument("--polyterm-subset-sizes", default="2")
    pool.add_argument(
        "--max-atoms-per-parent",
        type=int,
        default=None,
        help=(
            "Cap children kept per full_meta parent. Capping below a parent's "
            "child count means a parent-complete seed layer no longer reproduces "
            "that parent's termwise product."
        ),
    )
    pool.add_argument("--max-pool-atoms", type=int, default=None)
    pool.add_argument(
        "--priority-parent-labels",
        default=",".join(CONVENTIONAL_HH_LAYER_PARENTS),
        help=(
            "Comma-separated full_meta parent labels emitted first, so an "
            "atom-count cap never drops the conventional layer parents."
        ),
    )

    search = parser.add_argument_group("architecture search")
    search.add_argument("--layer-counts", default="1,2,3")
    search.add_argument("--atoms-per-layer", default="4,6,8")
    search.add_argument("--population-size", type=int, default=8)
    search.add_argument("--generations", type=int, default=3)
    search.add_argument("--mutation-count", type=int, default=2)
    search.add_argument("--search-seed", type=int, default=20260814)
    search.add_argument("--max-architecture-workers", type=int, default=1)
    search.add_argument("--max-snapshot-workers", type=int, default=1)
    search.add_argument("--retain-beyond-pareto", type=int, default=0)
    search.add_argument(
        "--seed-parent-complete-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    search.add_argument("--seed-parent-labels", default=None)
    search.add_argument("--seed-parent-complete-repeats", default="1,2")

    null_mode = parser.add_argument_group("null-space test modes")
    null_mode.add_argument("--test-mode-atom-id", default=None)
    null_mode.add_argument("--test-mode-duplicate-count", type=int, default=2)
    null_mode.add_argument(
        "--test-mode-separation-angle",
        type=float,
        default=DEFAULT_NEAR_NULL_SEPARATION_ANGLE,
        help=(
            "Near-null separating angle. The smallest retained Gram mode scales "
            "like the square of this angle."
        ),
    )

    return parser


def _reject_removed_online_injection(argv: Sequence[str]) -> None:
    """Fail loudly on the deleted online zero-angle redundancy command line."""

    used = [
        token
        for token in argv
        for flag in _REMOVED_ONLINE_INJECTION_FLAGS
        if str(token) == flag or str(token).startswith(f"{flag}=")
    ]
    if not used:
        return
    raise SystemExit(
        "error: the online zero-angle redundancy injection path has been removed. "
        f"Rejected flags: {sorted(set(used))}. Build a fixed-VQE conditioning-stress "
        "artifact offline with this runner instead, then point the dynamics runner or "
        "benchmark at the serialized artifact."
    )


def run_test_mode(
    args: argparse.Namespace,
    *,
    config: FixedVQEConditioningConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Construct and evaluate one explicit null-space validation fixture."""

    ctx = build_conditioning_context(config)
    mode = str(args.construction_mode)
    selection_gram = None
    if mode == CONSTRUCTION_MODE_EXACT_NULL_TEST:
        architecture = exact_null_test_architecture(
            ctx.pool,
            atom_id=args.test_mode_atom_id,
            duplicate_count=int(args.test_mode_duplicate_count),
        )
    elif args.test_mode_atom_id not in {None, ""}:
        architecture = near_null_test_architecture(
            ctx.pool,
            atom_id=str(args.test_mode_atom_id),
        )
    else:
        architecture, _theta, selection_gram = select_near_null_test_architecture(
            ctx,
            separation_angle=float(args.test_mode_separation_angle),
        )
    record = evaluate_architecture_stage_one(architecture, ctx=ctx)
    if record.qualified:
        record = evaluate_architecture_stage_two(
            record,
            ctx=ctx,
            max_snapshot_workers=int(config.search.max_snapshot_workers),
        )
    payload = {
        "schema": RUNNER_SCHEMA_V1,
        "construction": ctx.to_json_dict(),
        "test_mode": mode,
        "near_null_selection_gram": (
            None
            if selection_gram is None
            else selection_gram.to_json_dict(
                store_full_spectrum=bool(config.gram.store_full_spectrum)
            )
        ),
        "record": record.to_json_dict(
            store_full_spectrum=bool(config.gram.store_full_spectrum)
        ),
        "is_conventional_fixed_vqe_construction": False,
    }
    manifest_path = output_dir / "fixed_vqe_conditioning_test_mode_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    artifact_paths: list[str] = []
    if bool(args.write_artifacts):
        artifact_paths.append(
            str(
                write_fixed_vqe_stress_artifact(
                    record,
                    ctx=ctx,
                    output_json=output_dir / "artifacts" / f"{mode}_{record.architecture_id}.json",
                    subject_kind=mode,
                )
            )
        )
    payload["manifest_json"] = str(manifest_path)
    payload["artifact_json_paths"] = artifact_paths
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    _reject_removed_online_injection(raw_argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config_from_args(args)

    if str(args.construction_mode) != CONSTRUCTION_MODE_CONVENTIONAL:
        payload = run_test_mode(args, config=config, output_dir=output_dir)
        print(f"manifest_json={payload['manifest_json']}")
        for path in payload.get("artifact_json_paths", []):
            print(f"artifact_json={path}")
        return 0

    def progress(event: Mapping[str, Any]) -> None:
        print(json.dumps(dict(event), sort_keys=True), flush=True)

    result = run_fixed_vqe_conditioning_search(
        config,
        output_dir=output_dir,
        resume=bool(args.resume),
        progress=progress,
    )
    print(f"manifest_json={result.manifest_path}")
    print(f"search_batches_jsonl={result.batches_path}")
    print(f"architecture_record_count={result.manifest['architecture_record_count']}")
    print(f"evaluated_architecture_count={result.manifest['evaluated_architecture_count']}")
    print(f"resumed_architecture_count={result.manifest['resumed_architecture_count']}")
    print(f"ground_state_qualified_count={result.manifest['ground_state_qualified_count']}")
    print(f"pareto_front_size={len(result.pareto_front)}")

    if bool(args.write_artifacts):
        artifact_dir = output_dir / "artifacts"
        for record in tuple(result.pareto_front) + tuple(result.retained):
            path = write_fixed_vqe_stress_artifact(
                record,
                ctx=result.context,
                output_json=artifact_dir / f"fixed_vqe_stress_{record.architecture_id}.json",
            )
            print(f"artifact_json={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RUNNER_SCHEMA_V1",
    "build_parser",
    "config_from_args",
    "main",
    "run_test_mode",
]
