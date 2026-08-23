"""Single source of truth for Paper-II AP-McLachlan run commands.

Every Paper-II trajectory -- local smoke, ablation, campaign cell, CHTC job --
is built here, so that a configuration exists in exactly one place and drifts
in exactly one place.  Before this module the same run was spelled out in ad
hoc shell scripts, in three campaign modules, and in a handoff document, and
they disagreed: an "append-only" arm once silently ran the exchange
configuration because a shell array substitution failed, and an ablation arm
that was supposed to test the full candidate pool fell back to the config
default of 8 because its flag was omitted.  Both were invisible in the
artifacts.

Layout
------

* :data:`CANONICAL_NUMERICS` -- integrator and solve-repair settings that every
  accuracy-bearing run shares.  Never varied by an arm.
* :data:`PRODUCTION_STRUCTURE` -- candidate pool and guard settings.
* :class:`Arm` -- the structural policy under test (this is what a comparison
  is allowed to vary).
* :class:`InsertionGate` -- the condition under which insertions are considered.
* :class:`RunCommand` -- seed x drive x horizon x arm x gate -> argv.
* :data:`PAPER_II_RUNS` -- the named runs the paper actually reports.

The accompanying test asserts that every registered command parses against the
runner's live argument parser, so a renamed or deleted flag fails a test rather
than a six-hour cluster job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

RUNNER_MODULE = "pipelines.time_dynamics.runners.ap_append_from_adapt_artifact"
RUNNER_PATH = "pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py"

# Numerics that every accuracy-bearing run shares.  Measured 2026-08-18 on the
# stress seed: Euler + loose repair caps gave 1.6e-2 energy error, rk4 alone
# 1.1e-2, rk4 + these caps 1.3e-3.  The subdivision budget was raised from 4 to
# 10 on 2026-08-22: at 4 a step could exhaust its budget, fail to cure a cap
# violation, and then advance unsubdivided anyway, which took a measured HH
# error from 3.8e-3 to 2.1e-1.
CANONICAL_NUMERICS: tuple[str, ...] = (
    "--integrator", "rk4",
    "--solve-repair",
    "--solve-repair-profile", "minimal",
    "--solve-repair-state-motion-l2-step-max", "1.0e-2",
    "--solve-repair-kink-eta-max", "5.0e-3",
    "--solve-repair-max-local-subdivisions", "10",
    "--certification-refit",
    "--certification-refit-trust-radius", "0.6",
    "--certification-refit-max-iterations", "15",
)

# Structural settings.  The pool cap is deliberately above the deduplicated
# pool size (125 words on the HH nph=1 seed) so it does not bind: a cap of 8
# discarded ~117 usable words and was the single largest accuracy defect found
# in the 2026-08-22 audit.  The conditioning gate is off because no setting of
# it was useful -- 5e7 never binds against observed kappa ~1e8, 3e7 costs
# accuracy, and 1e7 rejects every candidate and starves the ansatz.
PRODUCTION_STRUCTURE: tuple[str, ...] = (
    "--max-structural-pool-size", "128",
    "--no-append-schur-condition-gate",
    "--max-joint-patch-evaluations", "50000",
    "--max-certification-attempts-per-level", "12",
    "--max-certification-attempts-per-deletion-branch", "2",
    "--max-insertion-batch-size", "1",
)

PROGRESS: tuple[str, ...] = ("--progress-log-every", "1")


@dataclass(frozen=True)
class Arm:
    """A structural policy.  This is the only axis a comparison may vary."""

    arm_id: str
    flags: tuple[str, ...]
    is_comparator: bool = False
    note: str = ""


EXCHANGE = Arm(
    arm_id="exchange",
    flags=("--prune-target-policy", "all_active", "--prune-ray-distance-tol", "2.0e-3"),
    note=(
        "The paper's route: deletions and positioned insertions compete as one "
        "atomic patch. Ray tolerance 2e-3 rather than the 5e-2 default, which "
        "admits cumulative structural damage over a long horizon."
    ),
)

APPEND_ONLY = Arm(
    arm_id="append_only",
    flags=("--prune-target-policy", "appended_only", "--prune-cooldown-steps", "1000000"),
    note="Growth-only ablation of the same route; isolates what pruning buys.",
)

AVQDS = Arm(
    arm_id="avqds",
    flags=("--dynamics-policy", "avqds", "--avqds-l2-cut", "1.0e-3"),
    is_comparator=True,
    note=(
        "Yao et al., PRX Quantum 2, 030307, on shared geometry/integrator/pool. "
        "Deliberately uncapped: the rule appends until L^2 < cut, and capping "
        "it is a different algorithm."
    ),
)

ARMS: Mapping[str, Arm] = {a.arm_id: a for a in (EXCHANGE, APPEND_ONLY, AVQDS)}


@dataclass(frozen=True)
class InsertionGate:
    """The condition under which insertions are considered."""

    gate_id: str
    flags: tuple[str, ...]
    note: str = ""


RESIDUAL_GATE = InsertionGate(
    gate_id="residual_1e-4",
    flags=("--insertion-gate-mode", "residual_ratio", "--residual-ratio-threshold", "0.0001"),
    note=(
        "This route's historical normalized gate. 1e-4 measured best at t=2 "
        "(7.8e-4 vs 3.8e-3 at 2e-3); 1e-5 buys nothing and costs parameters."
    ),
)

MCLACHLAN_L2_GATE = InsertionGate(
    gate_id="mclachlan_l2_1e-3",
    flags=("--insertion-gate-mode", "mclachlan_l2", "--insertion-l2-cut", "1.0e-3"),
    note=(
        "The published AVQDS append condition: absolute McLachlan distance with "
        "greedy repeat inside one checkpoint. Adopted because the normalized "
        "ratio is small precisely while the state is still accurate and so "
        "defers growth past the cheap early window."
    ),
)

GATES: Mapping[str, InsertionGate] = {
    g.gate_id: g for g in (RESIDUAL_GATE, MCLACHLAN_L2_GATE)
}


@dataclass(frozen=True)
class Drive:
    drive_id: str
    amplitude: float
    omega: float

    @property
    def flags(self) -> tuple[str, ...]:
        return (
            "--enable-drive",
            "--drive-A", repr(float(self.amplitude)),
            "--drive-omega", repr(float(self.omega)),
        )


DRIVES: Mapping[str, Drive] = {
    d.drive_id: d
    for d in (
        Drive("fastweak", 0.6, 3.0),
        Drive("slowstrong", 1.2, 1.0),
        Drive("weakslow", 0.3, 1.0),
        Drive("strongfast", 2.4, 3.0),
        Drive("midres", 1.2, 2.0),
        Drive("fastfast", 0.6, 6.0),
    )
}


@dataclass(frozen=True)
class Horizon:
    horizon_id: str
    t_final: float
    num_times: int

    @property
    def flags(self) -> tuple[str, ...]:
        return ("--t-final", repr(float(self.t_final)),
                "--num-times", str(int(self.num_times)))


HORIZONS: Mapping[str, Horizon] = {
    h.horizon_id: h
    for h in (
        Horizon("smoke", 0.5, 13),
        Horizon("t2", 2.0, 51),
        Horizon("t10", 10.0, 251),
        Horizon("t20", 20.0, 501),
    )
}


@dataclass(frozen=True)
class RunCommand:
    """One fully-specified Paper-II run."""

    seed_path: str
    arm: Arm
    gate: InsertionGate
    drive: Drive
    horizon: Horizon
    output_json: str
    extra_flags: tuple[str, ...] = ()

    @property
    def run_id(self) -> str:
        return f"{self.drive.drive_id}_{self.horizon.horizon_id}_{self.arm.arm_id}_{self.gate.gate_id}"

    def argv(self) -> tuple[str, ...]:
        # AVQDS carries its own append condition; layering this route's
        # insertion gate on top would misrepresent the comparator.
        gate_flags = () if self.arm.is_comparator else self.gate.flags
        return (
            "--artifact-json", str(self.seed_path),
            *self.horizon.flags,
            *self.drive.flags,
            *CANONICAL_NUMERICS,
            *PRODUCTION_STRUCTURE,
            *gate_flags,
            *self.arm.flags,
            *PROGRESS,
            *self.extra_flags,
            "--output-json", str(self.output_json),
        )

    def shell(self) -> str:
        parts = ["PYTHONPATH=.", "PYTHONUNBUFFERED=1", "python3", RUNNER_PATH]
        return " \\\n  ".join(parts + list(self.argv()))


def build_run(
    *,
    seed_path: str,
    arm: str,
    drive: str,
    horizon: str,
    output_json: str,
    gate: str = MCLACHLAN_L2_GATE.gate_id,
    extra_flags: Sequence[str] = (),
) -> RunCommand:
    """Compose one run from registered parts, rejecting unknown names."""

    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; known: {sorted(ARMS)}")
    if gate not in GATES:
        raise KeyError(f"unknown gate {gate!r}; known: {sorted(GATES)}")
    if drive not in DRIVES:
        raise KeyError(f"unknown drive {drive!r}; known: {sorted(DRIVES)}")
    if horizon not in HORIZONS:
        raise KeyError(f"unknown horizon {horizon!r}; known: {sorted(HORIZONS)}")
    return RunCommand(
        seed_path=str(seed_path),
        arm=ARMS[arm],
        gate=GATES[gate],
        drive=DRIVES[drive],
        horizon=HORIZONS[horizon],
        output_json=str(output_json),
        extra_flags=tuple(str(f) for f in extra_flags),
    )


HH_SNAKE_NPH1 = "chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json"


def paper_ii_runs(
    *,
    seed_path: str = HH_SNAKE_NPH1,
    horizon: str = "t10",
    gate: str = MCLACHLAN_L2_GATE.gate_id,
    output_root: str = "output/paper_ii",
) -> tuple[RunCommand, ...]:
    """The full drive x arm matrix the paper reports."""

    runs: list[RunCommand] = []
    for drive_id in DRIVES:
        for arm_id in ARMS:
            run = build_run(
                seed_path=seed_path,
                arm=arm_id,
                drive=drive_id,
                horizon=horizon,
                gate=gate,
                output_json=f"{output_root}/{drive_id}_{arm_id}/run.json",
            )
            runs.append(run)
    return tuple(runs)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list registered arms, gates, drives, horizons")
    show = sub.add_parser("show", help="print the shell command for one run")
    show.add_argument("--seed", default=HH_SNAKE_NPH1)
    show.add_argument("--arm", default="exchange")
    show.add_argument("--gate", default=MCLACHLAN_L2_GATE.gate_id)
    show.add_argument("--drive", default="fastweak")
    show.add_argument("--horizon", default="t10")
    show.add_argument("--output-json", default="output/paper_ii/run.json")
    matrix = sub.add_parser("matrix", help="print the whole drive x arm matrix")
    matrix.add_argument("--seed", default=HH_SNAKE_NPH1)
    matrix.add_argument("--gate", default=MCLACHLAN_L2_GATE.gate_id)
    matrix.add_argument("--horizon", default="t10")
    matrix.add_argument("--output-root", default="output/paper_ii")
    args = parser.parse_args(argv)

    if args.command == "list":
        for title, registry in (
            ("arms", ARMS), ("gates", GATES), ("drives", DRIVES), ("horizons", HORIZONS)
        ):
            print(f"{title}:")
            for key in registry:
                note = getattr(registry[key], "note", "")
                print(f"  {key}" + (f" -- {note.splitlines()[0]}" if note else ""))
        return 0
    if args.command == "show":
        print(build_run(
            seed_path=args.seed, arm=args.arm, gate=args.gate, drive=args.drive,
            horizon=args.horizon, output_json=args.output_json,
        ).shell())
        return 0
    for run in paper_ii_runs(
        seed_path=args.seed, horizon=args.horizon, gate=args.gate,
        output_root=args.output_root,
    ):
        print(f"# {run.run_id}")
        print(run.shell())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS", "AVQDS", "APPEND_ONLY", "CANONICAL_NUMERICS", "DRIVES", "EXCHANGE",
    "GATES", "HH_SNAKE_NPH1", "HORIZONS", "MCLACHLAN_L2_GATE",
    "PRODUCTION_STRUCTURE", "RESIDUAL_GATE", "Arm", "Drive", "Horizon",
    "InsertionGate", "RunCommand", "build_run", "main", "paper_ii_runs",
]
