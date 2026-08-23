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

    regime_id: str = "unregistered"

    @property
    def run_id(self) -> str:
        return (
            f"{self.regime_id}_{self.drive.drive_id}_{self.horizon.horizon_id}"
            f"_{self.arm.arm_id}_{self.gate.gate_id}"
        )

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
    arm: str,
    drive: str,
    horizon: str,
    output_json: str,
    seed_path: str | None = None,
    regime: str | None = None,
    gate: str = MCLACHLAN_L2_GATE.gate_id,
    extra_flags: Sequence[str] = (),
    require_seed: bool = False,
) -> RunCommand:
    """Compose one run from registered parts, rejecting unknown names.

    Give either ``regime`` (preferred -- names the physics) or ``seed_path``
    (an unregistered one-off).  Sweeping regimes must change nothing else, so
    the regime resolves only to a seed path.
    """

    if (seed_path is None) == (regime is None):
        raise ValueError("pass exactly one of regime= or seed_path=")
    if regime is not None:
        seed_path = resolve_regime(regime, require_available=require_seed).seed_path
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
        regime_id=str(regime) if regime is not None else "unregistered",
        arm=ARMS[arm],
        gate=GATES[gate],
        drive=DRIVES[drive],
        horizon=HORIZONS[horizon],
        output_json=str(output_json),
        extra_flags=tuple(str(f) for f in extra_flags),
    )


HH_SNAKE_NPH1 = "chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json"
SEED_ROOT = "chtc/paper_ii_production_v2/input/seeds"
REGIME_SEED_ROOT = "chtc/paper_ii_regime_seeds_v1/input/seeds"


@dataclass(frozen=True)
class Regime:
    """One Hubbard-Holstein physical regime.

    A regime is a property of the *seed*, not of the trajectory policy, so
    sweeping regimes must never change anything else about a run.  Naming them
    here is what makes "the same algorithm across regimes" a one-word change
    instead of a hand-edited seed path.
    """

    regime_id: str
    seed_path: str
    u: float | None = None
    g_ep: float | None = None
    n_ph_max: int = 1
    note: str = ""

    def __post_init__(self) -> None:
        # Binary-aligned phonon cutoffs only: 1, 3, 7 fill the register exactly,
        # and a same-cutoff comparison is mandatory for any regime claim.
        if int(self.n_ph_max) not in (1, 3, 7):
            raise ValueError(
                "n_ph_max must fill the binary phonon register (1, 3, or 7); "
                f"got {self.n_ph_max} for regime {self.regime_id!r}."
            )

    @property
    def available(self) -> bool:
        from pathlib import Path as _Path

        return _Path(self.seed_path).exists()


# The six-cell HH regime matrix (chtc/paper_ii_regime_seeds_v1/input/regimes.tsv).
# The nph=7 cells are the expensive ones: 10 qubits against 8, and the three
# strong-phonon seed builds died on a 24 GB memory limit on 2026-08-22.
_HH_REGIME_ROWS = (
    ("weak_weak", 0.25, 0.353553390593, 3),
    ("intermediate_weak", 1.25, 0.353553390593, 3),
    ("strong_weak_u8", 8.00, 0.353553390593, 3),
    ("weak_strong", 0.25, 0.790569415042, 7),
    ("intermediate_strong", 1.25, 0.790569415042, 7),
    ("strong_strong_u8", 8.00, 0.790569415042, 7),
)

REGIMES: Mapping[str, Regime] = {
    r.regime_id: r
    for r in (
        Regime(
            regime_id="hh_snake_nph1",
            seed_path=HH_SNAKE_NPH1,
            n_ph_max=1,
            note="The calibration seed every measurement in this lane was taken on.",
        ),
        Regime(
            regime_id="hh_fixedvqe_nph3",
            seed_path=f"{SEED_ROOT}/hh_fixedvqe_nph3.json",
            n_ph_max=3,
            note="Fixed-VQE conditioning stress seed.",
        ),
        *(
            Regime(
                regime_id=regime_id,
                seed_path=f"{REGIME_SEED_ROOT}/{regime_id}.json",
                u=u,
                g_ep=g_ep,
                n_ph_max=nph,
                note="HH regime matrix cell; seed build pending.",
            )
            for regime_id, u, g_ep, nph in _HH_REGIME_ROWS
        ),
    )
}


class SeedNotBuiltError(FileNotFoundError):
    """A named regime whose seed artifact does not exist yet."""


def resolve_regime(regime_id: str, *, require_available: bool = True) -> Regime:
    """Look up a regime, failing loudly when its seed has not been built.

    Failing here beats failing inside a runner: a missing seed otherwise
    surfaces as a loader error hundreds of lines into a job log, or worse, as a
    held cluster job whose input file was never transferred.
    """

    if regime_id not in REGIMES:
        raise KeyError(f"unknown regime {regime_id!r}; known: {sorted(REGIMES)}")
    regime = REGIMES[regime_id]
    if require_available and not regime.available:
        raise SeedNotBuiltError(
            f"regime {regime_id!r} has no seed at {regime.seed_path}. "
            "Build it before running; see chtc/paper_ii_regime_seeds_v1/."
        )
    return regime


def available_regimes() -> tuple[str, ...]:
    return tuple(k for k, r in REGIMES.items() if r.available)


def paper_ii_runs(
    *,
    regimes: Sequence[str] = ("hh_snake_nph1",),
    drives: Sequence[str] | None = None,
    arms: Sequence[str] | None = None,
    horizon: str = "t10",
    gate: str = MCLACHLAN_L2_GATE.gate_id,
    output_root: str = "output/paper_ii",
    require_seed: bool = False,
) -> tuple[RunCommand, ...]:
    """The regime x drive x arm matrix, one algorithm across all of it.

    Every axis is a list of registered names, so widening the study is a longer
    list rather than an edited command.
    """

    drive_ids = tuple(drives) if drives is not None else tuple(DRIVES)
    arm_ids = tuple(arms) if arms is not None else tuple(ARMS)
    runs: list[RunCommand] = []
    for regime_id in regimes:
        for drive_id in drive_ids:
            for arm_id in arm_ids:
                runs.append(
                    build_run(
                        regime=regime_id,
                        arm=arm_id,
                        drive=drive_id,
                        horizon=horizon,
                        gate=gate,
                        require_seed=require_seed,
                        output_json=(
                            f"{output_root}/{regime_id}/{drive_id}_{arm_id}/run.json"
                        ),
                    )
                )
    return tuple(runs)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list registered arms, gates, drives, horizons")
    sub.add_parser("regimes", help="list regimes and whether their seed is built")
    show = sub.add_parser("show", help="print the shell command for one run")
    show.add_argument("--regime", default="hh_snake_nph1")
    show.add_argument("--arm", default="exchange")
    show.add_argument("--gate", default=MCLACHLAN_L2_GATE.gate_id)
    show.add_argument("--drive", default="fastweak")
    show.add_argument("--horizon", default="t10")
    show.add_argument("--output-json", default="output/paper_ii/run.json")
    matrix = sub.add_parser("matrix", help="print the whole drive x arm matrix")
    matrix.add_argument("--regimes", default="hh_snake_nph1",
                        help="comma-separated regime ids")
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
    if args.command == "regimes":
        for regime_id, regime in REGIMES.items():
            mark = "built  " if regime.available else "MISSING"
            extra = "" if regime.u is None else f" u={regime.u} g={regime.g_ep:.4f}"
            print(f"  [{mark}] {regime_id:20s} nph={regime.n_ph_max}{extra}")
        missing = [k for k, r in REGIMES.items() if not r.available]
        if missing:
            print(f"\n{len(missing)} regime seed(s) not built: {', '.join(missing)}")
        return 0
    if args.command == "show":
        print(build_run(
            regime=args.regime, arm=args.arm, gate=args.gate, drive=args.drive,
            horizon=args.horizon, output_json=args.output_json,
        ).shell())
        return 0
    for run in paper_ii_runs(
        regimes=tuple(r.strip() for r in args.regimes.split(",") if r.strip()),
        horizon=args.horizon, gate=args.gate, output_root=args.output_root,
    ):
        print(f"# {run.run_id}")
        print(run.shell())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS", "AVQDS", "APPEND_ONLY", "CANONICAL_NUMERICS", "DRIVES", "EXCHANGE",
    "GATES", "HH_SNAKE_NPH1", "HORIZONS", "REGIMES", "Regime",
    "SeedNotBuiltError", "available_regimes", "resolve_regime", "MCLACHLAN_L2_GATE",
    "PRODUCTION_STRUCTURE", "RESIDUAL_GATE", "Arm", "Drive", "Horizon",
    "InsertionGate", "RunCommand", "build_run", "main", "paper_ii_runs",
]
