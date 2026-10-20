"""Command line for the Doron chemistry-quantum bridge.

    python -m pipelines.chemistry_bridge probe-backends
    python -m pipelines.chemistry_bridge export-qdynamics --h5 <prefix>_ele.h5 --out problem.json [--frozen-core 1]
    python -m pipelines.chemistry_bridge export-synthetic --kind contract4 --out problem.json [--frozen-core 0]
    python -m pipelines.chemistry_bridge run --manifest problem.manifest.json --out-dir out/ [--backend torch:mps:float32] [--methods mapping,exact,qse,adapt,ra,apm]

Exit codes of ``run`` (receipts are written before the code is decided):
    0  every requested method ended cpu_verified or gpu_verified
    1  at least one requested method ended failed, OR a method/output failure
       raised after the input was accepted (traceback printed; "run failed after
       intake"); such a failure is operational, not a usage error
    2  usage error: unknown or empty --methods, a --backend spec that does not parse,
       an --apm-times grid that is not finite and strictly increasing, an inadmissible
       --apm-pool, an --apm-support-config that is not a JSON object of controller
       fields, an --ra-rounds below 1 or an --ra-pool that is not a repository pool
       key, a missing or unreadable manifest/document, or the strict intake
       refused the input (no receipt written; "run refused")
    3  nothing failed, but a requested method was skipped_unsupported or blocked
       (e.g. an unavailable accelerator); this is never GPU evidence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Sequence

from pipelines.chemistry_bridge.array_backend import parse_backend_spec, probe_backends
from pipelines.chemistry_bridge.methods import (
    DEFAULT_METHODS,
    EXIT_CHECK_FAILED,
    EXIT_USAGE,
    exit_code_for_statuses,
    normalize_methods,
    QSE_BASIS_KINDS,
    RA_DEFAULT_POOL_KEY,
    RA_DEFAULT_ROUNDS,
    RA_INSERTION_POLICIES,
    admissible_pool_keys,
    parse_apm_support_config,
    resolve_bridge_context,
    run_bridge_resolved,
    validate_apm_times,
    validate_ra_construction_request,
)
from pipelines.chemistry_bridge.qdynamics_export import (
    BridgeDependencyUnavailable,
    FrozenCoreReduction,
    QdynamicsConventionError,
    bridge_problem_from_qdynamics,
    load_qdynamics_matrix_elements_h5,
    qdynamics_matrix_elements_from_mapping,
    write_bridge_problem_json,
)
from pipelines.chemistry_bridge.synthetic_witnesses import SYNTHETIC_KINDS, synthetic_qdynamics_mapping


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipelines.chemistry_bridge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe-backends", help="report which array backends/devices/dtypes can be exercised here")
    probe.add_argument("--json", action="store_true")
    export = sub.add_parser("export-qdynamics", help="convert a qdynamics <prefix>_ele.h5 export into a strict-intake problem document")
    export.add_argument("--h5", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--frozen-core", type=int, default=0, help="freeze the lowest N occupied orbitals (declared nuclear_plus_frozen_core)")
    export.add_argument("--declared-origin", choices=("unknown", "synthetic", "simulated"), default="unknown", help="origin of the file's data as declared by you; 'unknown' unless you made it synthetic")
    export.add_argument("--claimed-producer", default=None, help="who is claimed to have produced the file (recorded as a claim, never as established origin)")
    synthetic = sub.add_parser("export-synthetic", help="write a labelled synthetic witness in the producer's schema (not a molecule)")
    synthetic.add_argument("--kind", choices=SYNTHETIC_KINDS, default="contract4")
    synthetic.add_argument("--out", required=True)
    synthetic.add_argument("--frozen-core", type=int, default=0)
    synthetic.add_argument("--seed", type=int, default=20261019)
    run = sub.add_parser("run", help="strict intake + method runners with receipts")
    run.add_argument("--manifest", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--backend", default="numpy:float64")
    run.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    run.add_argument("--apm-times", default="0,0.02,0.04,0.06,0.08,0.10")
    run.add_argument("--apm-pool", default="hermitian_excitations", help="repository pool key (uccsd, full_meta, ...) or the bridge-local Hermitian number-conserving excitation pool (default)")
    run.add_argument("--apm-parameterization", default="logical_shared", choices=("logical_shared", "per_pauli_term"))
    run.add_argument("--apm-candidate-filter", default="executed_fixed_counts", choices=("executed_fixed_counts", "none"), help="'none' offers the unaudited pool and exists only to reproduce particle-number leakage")
    run.add_argument("--qse-basis", default="uccsd_singles_doubles", choices=QSE_BASIS_KINDS, help="QSE basis on the HF reference: UCCSD singles, singles+doubles (CISD), or products of commuting generators up to triples / quadruples")
    run.add_argument("--ra-rounds", default=RA_DEFAULT_ROUNDS, type=int, help="RA-ADAPT controller rounds for the named molecular closed-shell application (method 'ra')")
    run.add_argument("--ra-insertion", default="plateau_commutation", choices=RA_INSERTION_POLICIES)
    run.add_argument("--ra-pool", default=RA_DEFAULT_POOL_KEY, help="repository pool key for the RA-ADAPT macro generators (uccsd, full_meta, ...)")
    run.add_argument("--apm-support-config", default="{}", help="JSON object of SupportPatchControllerConfig field overrides for the AP-McLachlan exchange controller (validated before intake; recorded in the receipt)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "probe-backends":
        reports = probe_backends()
        if args.json:
            print(json.dumps([r.to_json_dict() for r in reports], indent=2))
        else:
            for r in reports:
                print(f"{r.spec:24s} {'available' if r.available else 'UNAVAILABLE'}  exercised={r.exercised_device}  gpu={r.is_gpu}  {r.reason or ''}")
        return 0
    if args.command in {"export-qdynamics", "export-synthetic"}:
        try:
            if args.command == "export-qdynamics":
                elements = load_qdynamics_matrix_elements_h5(args.h5, declared_origin=str(args.declared_origin), declared_producer=args.claimed_producer)
            else:
                elements = qdynamics_matrix_elements_from_mapping(
                    synthetic_qdynamics_mapping(args.kind, seed=int(args.seed)),
                    source={"kind": str(args.kind), "seed": int(args.seed), "simulation": True, "producer_generated": False},
                    declared_origin="synthetic",
                )
            reduction = FrozenCoreReduction(int(args.frozen_core)) if int(args.frozen_core) > 0 else None
            bridge_problem = bridge_problem_from_qdynamics(elements, frozen_core=reduction)
        except (BridgeDependencyUnavailable, QdynamicsConventionError, FileNotFoundError, OSError, ValueError) as exc:
            print(f"export refused: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        manifest = write_bridge_problem_json(bridge_problem, args.out)
        print(json.dumps({**{k: manifest[k] for k in ("problem_json_path", "manifest_path", "raw_sha256", "n_qubits", "num_particles", "reduction")}, "origin": bridge_problem.provenance["origin"], "container_schema": bridge_problem.provenance["container_schema"]}, indent=2))
        return 0
    if args.command == "run":
        # Phase 1 — input resolution and request validation only.  Failures here
        # mean the request could not be turned into a calculation (unknown or
        # empty methods, a backend spec that does not parse, an APM grid that is
        # not finite and strictly increasing, an inadmissible pool key, a missing
        # or unreadable manifest/document, strict-intake refusal): usage error,
        # no receipt.  Backend *availability* is not decided here: a well-formed
        # but unavailable backend yields skipped_unsupported receipts and exit 3.
        try:
            methods = normalize_methods(str(args.methods))
            parse_backend_spec(str(args.backend))
            times = validate_apm_times([v for v in str(args.apm_times).split(",") if v.strip()])
            apm_support = parse_apm_support_config(str(args.apm_support_config))
            bridge = resolve_bridge_context(args.manifest)
            if "apm" in methods and str(args.apm_pool) not in set(admissible_pool_keys(bridge)):
                raise ValueError(f"apm pool {args.apm_pool!r} is not admissible for this problem; choose from {list(admissible_pool_keys(bridge))}")
            if "ra" in methods:
                validate_ra_construction_request(args.ra_rounds, str(args.ra_insertion), str(args.ra_pool), bridge)
        except (ValueError, FileNotFoundError, OSError, KeyError) as exc:
            print(f"run refused: {type(exc).__name__}: {exc}", file=sys.stderr)
            return EXIT_USAGE
        # Phase 2 — method execution and output.  Anything raised here is an
        # operational failure after the input was accepted; it keeps its type and
        # traceback and exits 1, never 2.
        try:
            summary = run_bridge_resolved(
                bridge,
                backend_spec=str(args.backend),
                methods=methods,
                out_dir=Path(args.out_dir),
                apm_times=times,
                apm_pool_key=str(args.apm_pool),
                apm_parameterization_mode=str(args.apm_parameterization),
                apm_candidate_filter=str(args.apm_candidate_filter),
                apm_support_config_overrides=apm_support,
                qse_basis_kind=str(args.qse_basis),
                ra_rounds=int(args.ra_rounds),
                ra_insertion=str(args.ra_insertion),
                ra_pool_key=str(args.ra_pool),
            )
        except Exception as exc:
            print(f"run failed after intake: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return EXIT_CHECK_FAILED
        code = exit_code_for_statuses(summary["status_by_method"])
        print(json.dumps({"status_by_method": summary["status_by_method"], "out_path": summary.get("out_path"), "backend": summary["backend_spec"], "exit_code": code, "exit_policy": summary["exit_policy"].get(str(code))}, indent=2))
        return code
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
