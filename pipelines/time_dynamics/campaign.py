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

CAMPAIGN_SCHEMA_V1 = "paper_ii_dynamics_campaign_v1"

# Canonical numerics (2026-08-18 stabilization sweep): rk4 with the state-motion
# guard is the accuracy configuration; the runner's Euler default is diagnostic.
CANONICAL_NUMERICS: dict[str, Any] = {
    "integrator": "rk4",
    "solve_repair": True,
    "solve_repair_profile": "minimal",
    "solve_repair_state_motion_l2_step_max": 1.0e-2,
    "solve_repair_kink_eta_max": 5.0e-3,
}

# Computational guards required at scale (see lane AGENTS.md).
DEFAULT_GUARDS: dict[str, Any] = {
    "max_joint_patch_evaluations": 50000,
    "max_certification_attempts_per_level": 12,
    "max_certification_attempts_per_deletion_branch": 2,
    "max_insertion_batch_size": 1,
    "max_structural_pool_size": 8,
}


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
    return PolicyArm(
        arm_id=f"exchange_tau{ray_tol:g}",
        flags=(
            "--prune-target-policy", "all_active",
            "--prune-ray-distance-tol", repr(float(ray_tol)),
            "--certification-refit",
            "--certification-refit-trust-radius", "0.6",
            "--certification-refit-max-iterations", "15",
            "--prune-history-lambda", "0.0",
        ),
    )


def append_only_arm() -> PolicyArm:
    return PolicyArm(
        arm_id="append_only",
        flags=(
            "--prune-target-policy", "appended_only",
            "--prune-cooldown-steps", "1000000",
            "--certification-refit",
            "--certification-refit-trust-radius", "0.6",
            "--certification-refit-max-iterations", "15",
            "--prune-history-lambda", "0.0",
        ),
    )


def avqds_arm(l2_cut: float, max_appends: int = 2) -> PolicyArm:
    return PolicyArm(
        arm_id=f"avqds_cut{l2_cut:g}",
        flags=(
            "--dynamics-policy", "avqds",
            "--avqds-l2-cut", repr(float(l2_cut)),
            "--avqds-max-appends-per-checkpoint", str(int(max_appends)),
        ),
        is_comparator=True,
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
    residual_ratio_threshold: float = 0.02
    guards: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_GUARDS))
    numerics: Mapping[str, Any] = field(default_factory=lambda: dict(CANONICAL_NUMERICS))
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
        argv: list[str] = [
            "--artifact-json", self.seed.artifact_json,
            "--output-json", str(Path(self.output_dir) / "run.json"),
            "--t-final", repr(float(self.horizon.t_final)),
            "--num-times", str(int(self.horizon.num_times)),
            "--residual-ratio-threshold",
            repr(float(self.campaign.residual_ratio_threshold)),
            "--progress-log-every", "5",
        ]
        numerics = dict(self.campaign.numerics)
        if numerics.pop("solve_repair", False):
            argv.append("--solve-repair")
        for key, value in numerics.items():
            flag = "--" + key.replace("_", "-")
            argv.extend([flag, repr(value) if isinstance(value, float) else str(value)])
        for key, value in dict(self.campaign.guards).items():
            argv.extend(["--" + key.replace("_", "-"),
                         repr(value) if isinstance(value, float) else str(value)])
        argv.extend(self.drive.flags())
        argv.extend(self.arm.flags)
        return argv

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
    "CANONICAL_NUMERICS",
    "DEFAULT_GUARDS",
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
]
