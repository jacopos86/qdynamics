#!/usr/bin/env python3
"""Build the exact-response-scale sidecar for one registered Paper-II regime.

Stage 1.5 of the regime survey.  For each campaign drive it builds the
content-addressed exact reference of the regime's installed seed on the locked
0.04 grid to ``t=20`` (``ExactReferenceStore``, propagator
``exponential_midpoint_v1``, 40 substeps), evaluates the response scales with
the loader's own formula (``response_scales._calculated_scales``), and writes a
sidecar in the ``paper_ii_exact_response_scales_v1`` schema plus a tracked
registry row.  ``paper_ii_staged_accuracy_v1 --regime R`` then qualifies survey
cells under the same response-normalized matched-pair policy as the locked
calibration campaign.

The drive profile is produced by the runner's own drive-config and Hamiltonian
builders, so the store key computed here is the key every survey run computes
again at record time; the reference is built once and verified on every read.
Exact data stays offline: nothing here is visible to a propagating controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input
from pipelines.time_dynamics import paper_ii_runs as runs
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    time_dependent_hamiltonian_from_runtime_input,
)
from pipelines.time_dynamics.diagnostics.exact_reference_store import (
    ExactReferenceStore,
)
from pipelines.time_dynamics.response_scales import (
    _calculated_scales,
    load_verified_response_scale_index,
)
from pipelines.time_dynamics.run_lock import resolve_code_revision
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    _build_parser as _runner_parser,
)
from pipelines.time_dynamics.runners.ap_fixed_from_adapt_artifact import (
    _drive_config_from_args,
)

SIDECAR_SCHEMA_V1 = "paper_ii_exact_response_scales_v1"
REGISTRY_SCHEMA_V1 = "paper_ii_exact_response_scales_by_regime_v1"
DEFINITION = (
    "RMS_t[O_exact(t)-O_exact(0)] on the declared 0.04 output grid; "
    "site scale is RMS_t[max_i |n_i(t)-n_i(0)|]"
)
HORIZON_FINAL_TIME = {"t5": 5.0, "t20": 20.0}
GRID_STEP = 0.04
GRID_POINTS = 501
DEFAULT_SUBSTEPS = 40
DEFAULT_DRIVES = (
    "fastweak", "slowstrong", "weakslow", "strongfast", "midres", "fastfast",
)
DEFAULT_CONTROLLERS = (
    "state_motion_5e-5",
    "delta_theta_5e-5",
    "state_motion_1e-3_plus_parameter_5e-5",
)
FAMILY_KEY = "hubbard_holstein"
CAMPAIGNS_DIR = Path("pipelines/time_dynamics/campaigns")
REGISTRY_PATH = CAMPAIGNS_DIR / "paper_ii_exact_response_scales_by_regime.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_grid() -> list[float]:
    """The 501-point 0.04 grid every Paper-II response scale is defined on."""

    # The staged campaign reaches t=20 as a 126-point t5 screen followed by a
    # 376-point continuation whose grid is np.linspace(5, 20, 376); the record
    # step keys the exact reference on the combined floats.  The producer uses
    # the same construction so its keys equal the locked store's and the keys
    # every survey run will compute again at record time (a plain
    # linspace(0, 20, 501) differs in the last bit at 72 of the 501 points).
    screen = np.linspace(0.0, 5.0, 126).tolist()
    continuation = np.linspace(5.0, 20.0, 376).tolist()[1:]
    return [float(value) for value in screen + continuation]


def drive_profile_for(seed_path: str | Path, drive_id: str) -> dict[str, Any]:
    """Return the drive profile the runner records for ``drive_id`` on this seed."""

    drive = runs.DRIVES[drive_id]
    args = _runner_parser().parse_args(
        ["--artifact-json", str(seed_path), "--output-json", "unused.json", *drive.flags]
    )
    runtime_input = load_scaffold_runtime_input(str(seed_path))
    drive_config = _drive_config_from_args(args, runtime_input)
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(
        runtime_input, drive_config=drive_config
    )
    profile = hamiltonian.to_json_dict().get("drive_profile")
    if not isinstance(profile, Mapping):
        raise ValueError(f"runner Hamiltonian recorded no drive profile for {drive_id}")
    return dict(profile)


def exact_reference_root_for(regime: str) -> Path:
    """Three levels below the repository root, as the loader requires."""

    return Path("output") / f"paper_ii_regime_survey_{regime}" / "exact_references"


def build_sidecar(
    *,
    regime: str,
    seed_path: Path,
    exact_reference_root: Path,
    substeps: int = DEFAULT_SUBSTEPS,
    drives: Sequence[str] = DEFAULT_DRIVES,
    repository_root: Path = Path("."),
) -> dict[str, Any]:
    """Build every reference and return the sidecar payload (not yet written)."""

    if int(substeps) < 1:
        raise ValueError("substeps must be at least one")
    grid = locked_grid()
    store = ExactReferenceStore(exact_reference_root)
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory() as scratch:
        for drive_id in drives:
            profile = drive_profile_for(seed_path, drive_id)
            carrier = Path(scratch) / f"{drive_id}.json"
            carrier.write_text(
                json.dumps({"hamiltonian": {"drive_profile": profile}}), encoding="utf-8"
            )
            artifact = store.get_or_build(
                seed_path=seed_path, run_path=carrier, times=grid, substeps=int(substeps)
            )
            manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
            with np.load(artifact.data_path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            rows[drive_id] = {
                horizon: {
                    **_calculated_scales(arrays, final_time=final_time),
                    "exact_reference_key": artifact.key,
                    "exact_reference_data_sha256": str(manifest["data_sha256"]),
                }
                for horizon, final_time in HORIZON_FINAL_TIME.items()
            }
    return {
        "schema": SIDECAR_SCHEMA_V1,
        "definition": DEFINITION,
        "regime": str(regime),
        "seed": {
            "path": str(seed_path),
            "sha256": _sha256(Path(seed_path)),
        },
        "exact_reference_root": str(exact_reference_root),
        "propagator_schema": "exponential_midpoint_v1",
        "substeps": int(substeps),
        "grid": {"start": 0.0, "step": GRID_STEP, "points": GRID_POINTS},
        "produced_by": {
            "module": "pipelines.time_dynamics.diagnostics.build_regime_response_scales",
            "code_revision": resolve_code_revision(repository_root),
        },
        "drives": rows,
    }


def write_sidecar(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return _sha256(path)


def update_registry(
    registry_path: Path,
    *,
    regime: str,
    sidecar_path: str,
    sidecar_sha256: str,
    seed_path: str,
    seed_sha256: str,
    exact_reference_root: str,
) -> dict[str, Any]:
    registry: dict[str, Any] = {"schema": REGISTRY_SCHEMA_V1, "regimes": {}}
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("schema") != REGISTRY_SCHEMA_V1:
            raise ValueError(f"unsupported response-scale registry: {registry_path}")
    registry.setdefault("regimes", {})[str(regime)] = {
        "sidecar": sidecar_path,
        "sha256": sidecar_sha256,
        "seed_path": seed_path,
        "seed_sha256": seed_sha256,
        "exact_reference_root": exact_reference_root,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return registry


def verify_sidecar(
    *,
    sidecar_path: Path,
    sidecar_sha256: str,
    exact_reference_root: Path,
    seed_path: Path,
    regime: str,
    controllers: Sequence[str] = DEFAULT_CONTROLLERS,
    drives: Sequence[str] = DEFAULT_DRIVES,
) -> dict[str, Any]:
    """Round-trip the sidecar through the staged campaign's own loader."""

    return load_verified_response_scale_index(
        sidecar_path,
        expected_sidecar_sha256=sidecar_sha256,
        exact_reference_root=exact_reference_root,
        seed_path=seed_path,
        regime_id=regime,
        family_key=FAMILY_KEY,
        controller_ids=tuple(controllers),
        expected_drives={
            drive_id: {
                "amplitude": runs.DRIVES[drive_id].amplitude,
                "omega": runs.DRIVES[drive_id].omega,
            }
            for drive_id in drives
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regime", required=True, choices=tuple(runs.REGIMES))
    parser.add_argument("--seed-json", default=None,
                        help="defaults to the regime's registered seed path")
    parser.add_argument("--exact-reference-root", default=None,
                        help="defaults to output/paper_ii_regime_survey_<regime>/exact_references")
    parser.add_argument("--output", default=None,
                        help="defaults to pipelines/time_dynamics/campaigns/paper_ii_exact_response_scales_<regime>.json")
    parser.add_argument("--registry", default=str(REGISTRY_PATH))
    parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    regime = str(args.regime)
    seed_rel = args.seed_json or runs.REGIMES[regime].seed_path
    seed_path = (root / seed_rel).resolve() if not Path(seed_rel).is_absolute() else Path(seed_rel)
    if not seed_path.is_file():
        raise SystemExit(f"seed is not installed: {seed_path} (run install_seeds.py first)")
    ref_rel = args.exact_reference_root or str(exact_reference_root_for(regime))
    ref_root = (root / ref_rel).resolve() if not Path(ref_rel).is_absolute() else Path(ref_rel)
    out_rel = args.output or str(CAMPAIGNS_DIR / f"paper_ii_exact_response_scales_{regime}.json")
    out_path = (root / out_rel).resolve() if not Path(out_rel).is_absolute() else Path(out_rel)
    payload = build_sidecar(
        regime=regime,
        seed_path=seed_path,
        exact_reference_root=ref_root,
        substeps=int(args.substeps),
        repository_root=root,
    )
    # Record repository-relative paths so the sidecar is portable across checkouts.
    payload["seed"]["path"] = seed_rel
    payload["exact_reference_root"] = ref_rel
    sidecar_sha = write_sidecar(payload, out_path)
    index = verify_sidecar(
        sidecar_path=out_path,
        sidecar_sha256=sidecar_sha,
        exact_reference_root=ref_root,
        seed_path=seed_path,
        regime=regime,
    )
    registry_path = (root / args.registry).resolve() if not Path(args.registry).is_absolute() else Path(args.registry)
    update_registry(
        registry_path,
        regime=regime,
        sidecar_path=out_rel,
        sidecar_sha256=sidecar_sha,
        seed_path=seed_rel,
        seed_sha256=payload["seed"]["sha256"],
        exact_reference_root=ref_rel,
    )
    summary = {
        "regime": regime,
        "sidecar": out_rel,
        "sidecar_sha256": sidecar_sha,
        "registry": str(args.registry),
        "exact_reference_root": ref_rel,
        "verified_index_schema": index["schema"],
        "scales": {
            drive: {
                horizon: {
                    key: payload["drives"][drive][horizon][key]
                    for key in (
                        "energy_rms_excursion",
                        "doublon_rms_excursion",
                        "max_site_occupation_rms_excursion",
                    )
                }
                for horizon in HORIZON_FINAL_TIME
            }
            for drive in payload["drives"]
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DEFINITION",
    "REGISTRY_PATH",
    "REGISTRY_SCHEMA_V1",
    "SIDECAR_SCHEMA_V1",
    "build_sidecar",
    "drive_profile_for",
    "exact_reference_root_for",
    "locked_grid",
    "main",
    "update_registry",
    "verify_sidecar",
    "write_sidecar",
]
