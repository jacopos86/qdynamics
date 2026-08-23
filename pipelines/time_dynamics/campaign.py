"""Typed dynamics-campaign specification for the Paper-II lane.

Thin orchestration around the existing neutral seam: a campaign is a product
of seeds (each an accepted-ansatz export or seed artifact), physics conditions
(drive), horizons, and policy arms.  It resolves to concrete runner invocations
with locked provenance, so a regime/Hamiltonian matrix is declared once instead
of being rebuilt as ad-hoc shell each time.

This module deliberately owns no scientific defaults.  Numerics come from the
canonical profile, structural settings from the policy arm, and physics from
the seed and drive; the campaign only takes their product and records what it
ran.  It never searches artifact trees: every seed is named explicitly.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from pipelines.time_dynamics import paper_ii_runs as _runs

CAMPAIGN_SCHEMA_V1 = "paper_ii_dynamics_campaign_v1"

# Numerics, structure, and arms come from the run registry -- this module owns
# the *matrix* (seed x drive x horizon x arm), never the run configuration.
# They were duplicated here until 2026-08-23, and the copies drifted: this
# module's guards still carried `max_structural_pool_size: 8` weeks after that
# cap was identified as the largest single accuracy defect in the lane (it
# discarded ~117 of the 125 deduplicated pool words). Any campaign built from
# the stale copy would have silently reproduced it.
SHARED_NUMERICS = _runs.SHARED_NUMERICS
STATE_MOTION_CONTROL = _runs.STATE_MOTION_CONTROL
PRODUCTION_STRUCTURE: tuple[str, ...] = _runs.PRODUCTION_STRUCTURE


@dataclass(frozen=True)
class SeedSpec:
    """One physical starting point, named explicitly."""

    seed_id: str
    artifact_json: str
    family_key: str
    n_ph_max: int
    regime: str
    note: str = ""

    def __post_init__(self) -> None:
        if int(self.n_ph_max) not in (1, 3, 7):
            raise ValueError(
                "n_ph_max must fill the binary phonon register (1, 3, or 7); "
                f"got {self.n_ph_max}."
            )

    def sha256(self) -> str:
        path = Path(self.artifact_json)
        if not path.exists():
            return "missing"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()


@dataclass(frozen=True)
class DriveSpec:
    """Drive condition; ``enabled=False`` is the quench-free static case."""

    drive_id: str = "undriven"
    enabled: bool = False
    amplitude: float = 0.0
    omega: float = 1.0

    def flags(self) -> list[str]:
        if not self.enabled:
            return []
        return [
            "--enable-drive",
            "--drive-A", repr(float(self.amplitude)),
            "--drive-omega", repr(float(self.omega)),
        ]


@dataclass(frozen=True)
class PolicyArm:
    """One structural decision rule, named for reporting."""

    arm_id: str
    flags: tuple[str, ...]
    is_comparator: bool = False


def exchange_arm(ray_tol: float = 2.0e-3) -> PolicyArm:
    """The paper's route, from the run registry.

    ``ray_tol`` stays a parameter because tightening it is a real ablation;
    everything else comes from :mod:`pipelines.time_dynamics.paper_ii_runs`.
    """

    flags = list(_runs.EXCHANGE.flags)
    idx = flags.index("--prune-ray-distance-tol")
    flags[idx + 1] = repr(float(ray_tol))
    return PolicyArm(arm_id=f"exchange_tau{ray_tol:g}", flags=tuple(flags))


def append_only_arm() -> PolicyArm:
    return PolicyArm(arm_id="append_only", flags=tuple(_runs.APPEND_ONLY.flags))


def avqds_arm(l2_cut: float, max_appends: int | None = None) -> PolicyArm:
    """The comparator, from the run registry.

    Unbounded by default: Yao et al. (PRX Quantum 2, 030307) append "repeated
    until L^2 < L^2_cut" with no per-checkpoint cap, so a cap here would
    handicap the comparator rather than reproduce it.
    """

    flags = list(_runs.AVQDS.flags)
    idx = flags.index("--avqds-l2-cut")
    flags[idx + 1] = repr(float(l2_cut))
    if max_appends is not None:
        flags.extend(["--avqds-max-appends-per-checkpoint", str(int(max_appends))])
    return PolicyArm(
        arm_id=f"avqds_cut{l2_cut:g}", flags=tuple(flags), is_comparator=True
    )


@dataclass(frozen=True)
class HorizonSpec:
    horizon_id: str
    t_final: float
    num_times: int


@dataclass(frozen=True)
class CampaignSpec:
    """Full product of seeds x drives x horizons x arms."""

    campaign_id: str
    seeds: tuple[SeedSpec, ...]
    drives: tuple[DriveSpec, ...]
    horizons: tuple[HorizonSpec, ...]
    arms: tuple[PolicyArm, ...]
    # The insertion gate is a registry part, not a bare float: the two gates
    # (normalized residual ratio, absolute McLachlan distance) take different
    # flags and are not interchangeable.
    gate_id: str = _runs.MCLACHLAN_L2_GATE.gate_id
    numerics_id: str = _runs.SHARED_NUMERICS.numerics_id
    step_control_id: str = _runs.STATE_MOTION_CONTROL.control_id
    structure: tuple[str, ...] = PRODUCTION_STRUCTURE
    output_root: str = "output"

    def cells(self) -> Iterator["CampaignCell"]:
        for seed, drive, horizon, arm in itertools.product(
            self.seeds, self.drives, self.horizons, self.arms
        ):
            yield CampaignCell(
                campaign=self, seed=seed, drive=drive, horizon=horizon, arm=arm
            )

    def cell_count(self) -> int:
        return (
            len(self.seeds) * len(self.drives) * len(self.horizons) * len(self.arms)
        )


@dataclass(frozen=True)
class CampaignCell:
    """One runnable point of the matrix."""

    campaign: CampaignSpec
    seed: SeedSpec
    drive: DriveSpec
    horizon: HorizonSpec
    arm: PolicyArm

    @property
    def cell_id(self) -> str:
        return "__".join(
            (self.seed.seed_id, self.drive.drive_id, self.horizon.horizon_id,
             self.arm.arm_id)
        )

    @property
    def output_dir(self) -> str:
        return str(Path(self.campaign.output_root) / self.campaign.campaign_id
                   / self.cell_id)

    def runner_argv(self) -> list[str]:
        """Compose this cell's argv from the run registry.

        Numerics and structure are shared verbatim across every arm, so a
        comparison can only differ in the arm's own flags -- which is the
        property `assert_comparable` checks after the fact and this method
        guarantees before it.
        """

        gate = _runs.GATES[self.campaign.gate_id]
        # A comparator carries its own append condition; layering this route's
        # insertion gate on top would misrepresent it.
        gate_flags = () if self.arm.is_comparator else gate.flags
        return [
            "--artifact-json", self.seed.artifact_json,
            "--output-json", str(Path(self.output_dir) / "run.json"),
            "--t-final", repr(float(self.horizon.t_final)),
            "--num-times", str(int(self.horizon.num_times)),
            *_runs.NUMERICS[self.campaign.numerics_id].flags,
            *_runs.STEP_CONTROLS[self.campaign.step_control_id].flags,
            *self.campaign.structure,
            *gate_flags,
            *self.drive.flags(),
            *self.arm.flags,
            "--progress-log-every", "5",
        ]

    def provenance(self) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_SCHEMA_V1,
            "campaign_id": self.campaign.campaign_id,
            "cell_id": self.cell_id,
            "seed": {
                "seed_id": self.seed.seed_id,
                "artifact_json": self.seed.artifact_json,
                "family_key": self.seed.family_key,
                "n_ph_max": int(self.seed.n_ph_max),
                "regime": self.seed.regime,
                "sha256": self.seed.sha256(),
            },
            "drive": {
                "drive_id": self.drive.drive_id,
                "enabled": bool(self.drive.enabled),
                "amplitude": float(self.drive.amplitude),
                "omega": float(self.drive.omega),
            },
            "horizon": {
                "horizon_id": self.horizon.horizon_id,
                "t_final": float(self.horizon.t_final),
                "num_times": int(self.horizon.num_times),
            },
            "arm": {
                "arm_id": self.arm.arm_id,
                "is_comparator": bool(self.arm.is_comparator),
                "flags": list(self.arm.flags),
            },
            "runner_argv": self.runner_argv(),
        }


def write_campaign_manifest(spec: CampaignSpec, path: str | Path) -> Path:
    """Serialize every cell's provenance; the record of what a matrix means."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CAMPAIGN_SCHEMA_V1,
        "campaign_id": spec.campaign_id,
        "cell_count": spec.cell_count(),
        "cells": [cell.provenance() for cell in spec.cells()],
    }
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return out


__all__ = [
    "CAMPAIGN_SCHEMA_V1",
    "SHARED_NUMERICS",
    "PRODUCTION_STRUCTURE",
    "CampaignCell",
    "CampaignSpec",
    "DriveSpec",
    "HorizonSpec",
    "PolicyArm",
    "SeedSpec",
    "append_only_arm",
    "avqds_arm",
    "exchange_arm",
    "write_campaign_manifest",
    "write_chtc_package",
]


def write_chtc_package(
    spec: CampaignSpec,
    package_dir: str | Path,
    *,
    image_path: str = "chtc/time_dynamics_optuna/image.sif",
    request_cpus: int = 2,
    request_memory_gb: int = 16,
    max_runtime_seconds: int = 86400,
) -> dict[str, Path]:
    """Emit a runnable CHTC package for the whole matrix.

    One declaration produces the cell list, the per-cell runner invocation, the
    apptainer wrapper, the submit file, and the manifest that records what each
    cell means.  Cells are addressed by id, so a cluster's job N maps back to a
    seed, drive, horizon, and policy arm without consulting anything else.
    """

    out = Path(package_dir)
    (out / "input").mkdir(parents=True, exist_ok=True)
    rel = out.as_posix()

    cells = list(spec.cells())
    (out / "input" / "cell_ids.txt").write_text(
        "\n".join(c.cell_id for c in cells) + "\n", encoding="utf-8"
    )

    # Seeds travel inside the package. An execute node receives only what
    # transfer_input_files names, and seeds otherwise live in sibling chtc
    # directories that a submit file has no reason to know about; copying them
    # here also means the campaign carries the exact bytes it ran on.
    seed_dir = out / "input" / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    packaged: dict[str, str] = {}
    for seed in spec.seeds:
        src = Path(seed.artifact_json)
        dst = seed_dir / f"{seed.seed_id}.json"
        if src.exists():
            dst.write_bytes(src.read_bytes())
        packaged[seed.artifact_json] = dst.as_posix()

    def _repoint(argv: list[str]) -> list[str]:
        return [packaged.get(token, token) for token in argv]

    # Each cell's argv, one line per cell, so the shell need not re-derive it.
    argv_lines = [
        c.cell_id + "\t" + " ".join(_repoint(c.runner_argv())) for c in cells
    ]
    (out / "input" / "cell_argv.tsv").write_text(
        "\n".join(argv_lines) + "\n", encoding="utf-8"
    )
    manifest = write_campaign_manifest(spec, out / "input" / "manifest.json")

    run_sh = out / "run_cell.sh"
    run_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'CELL_ID="${1:?cell_id required}"\n'
        f'BASE={rel}\n'
        'ARGV=$(awk -F"\\t" -v id="$CELL_ID" \'$1==id {print $2}\' '
        '"$BASE/input/cell_argv.tsv")\n'
        'if [[ -z "$ARGV" ]]; then echo "unknown cell_id $CELL_ID" >&2; exit 3; fi\n'
        'mkdir -p logs raw_outputs\n'
        'export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1\n'
        "# shellcheck disable=SC2086\n"
        "python3 pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py $ARGV\n",
        encoding="utf-8",
    )
    run_sh.chmod(0o755)

    wrapper = out / "run_cell_apptainer.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'IMAGE="${{PROJECT_IMAGE:-{image_path}}}"\n'
        'if [[ ! -f "$IMAGE" ]]; then echo "Missing Apptainer image: $IMAGE" >&2; exit 2; fi\n'
        'if command -v apptainer >/dev/null 2>&1; then APPTAINER_BIN="$(command -v apptainer)";\n'
        'elif command -v singularity >/dev/null 2>&1; then APPTAINER_BIN="$(command -v singularity)";\n'
        'else echo "No apptainer/singularity on this execute node." >&2; exit 127; fi\n'
        'ROOT="$PWD"\n'
        f'"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \\\n'
        f"  bash -lc 'cd /work && bash {rel}/run_cell.sh \"$@\"' -- \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    tag = spec.campaign_id.replace("_", "-")
    submit = out / f"submit_{spec.campaign_id}.sub"
    submit.write_text(
        "universe = vanilla\n"
        f"executable = {rel}/run_cell_apptainer.sh\n"
        "arguments = $(cell_id)\n"
        "should_transfer_files = YES\n"
        "when_to_transfer_output = ON_EXIT\n"
        "transfer_executable = True\n"
        "preserve_relative_paths = True\n"
        f"transfer_input_files = pipelines, src, {rel}, {image_path}\n"
        "transfer_output_files = raw_outputs, logs\n"
        f"log = logs/{tag}.$(Cluster).$(Process).log\n"
        f"output = logs/{tag}.$(Cluster).$(Process).out\n"
        f"error = logs/{tag}.$(Cluster).$(Process).err\n"
        "requirements = TARGET.HasSIF\n"
        f"request_cpus = {int(request_cpus)}\n"
        f"request_memory = {int(request_memory_gb)}GB\n"
        "request_disk = 40GB\n"
        f"+MaxRuntime = {int(max_runtime_seconds)}\n"
        f'+JobBatchName = "{tag}"\n'
        f"queue cell_id from {rel}/input/cell_ids.txt\n",
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "cell_ids": out / "input" / "cell_ids.txt",
        "cell_argv": out / "input" / "cell_argv.tsv",
        "run_cell": run_sh,
        "wrapper": wrapper,
        "submit": submit,
    }
