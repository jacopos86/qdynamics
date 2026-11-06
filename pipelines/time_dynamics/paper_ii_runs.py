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

* :class:`Numerics` -- the inner numerical method (integrator, regularization,
  damping) shared by every arm in a comparison.
* :class:`StepControl` -- how the step size is chosen. Both this route and
  AVQDS control their step; they differ in the quantity they bound, so this is
  a separate axis from the structural rule and can be isolated from it.
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
import math
from typing import Mapping, Sequence

RUNNER_MODULE = "pipelines.time_dynamics.runners.ap_append_from_adapt_artifact"
RUNNER_PATH = "pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py"

@dataclass(frozen=True)
class Numerics:
    """Inner numerical method, shared by every arm in a comparison.

    A comparison is only about structural rules if everything under them is
    identical: the ODE integrator, the regularization and damping of the
    McLachlan solve, and the reporting grid.  These were previously spread
    across a shared flag tuple, a campaign module, and each comparator's own
    flags, and the copies drifted -- an arm labelled as a published method
    ended up running this route's solve repair with local subdivision applied
    at 249 of 251 checkpoints.
    """

    numerics_id: str
    integrator: str = "euler"
    ridge_lambda: float = 1.0e-7
    solve_damping: float = 0.0
    pinv_rcond: float = 1.0e-10
    note: str = ""

    @property
    def flags(self) -> tuple[str, ...]:
        return (
            "--integrator", str(self.integrator),
            "--ridge-lambda", repr(float(self.ridge_lambda)),
            "--solve-damping", repr(float(self.solve_damping)),
            "--pinv-rcond", repr(float(self.pinv_rcond)),
        )


# Euler remains available as an integrator ablation, but it is not the
# Paper-II default: the six-drive ablation found a material accuracy difference
# even when the step guard was satisfied.
EULER_NUMERICS = Numerics(
    numerics_id="euler_ridge1e-7",
    note="Euler ablation: ridge 1e-7, no extra damping.",
)

EULER_RIDGE1E6_NUMERICS = Numerics(
    numerics_id="euler_ridge1e-6",
    ridge_lambda=1.0e-6,
    note=(
        "Paper-II factorial campaign numerics: Euler and Tikhonov ridge "
        "xi=1e-6, shared by both structural methods and all three step "
        "controllers."
    ),
)

RK4_NUMERICS = Numerics(
    numerics_id="rk4_ridge1e-7",
    integrator="rk4",
    note=(
        "Canonical Paper-II inner method: RK4, ridge 1e-7, no extra damping. "
        "The integrator ablation shows that order remains accuracy-relevant."
    ),
)

# Published AVQDS owns this numerical convention.  It is intentionally not the
# shared Paper-II default: a method-level reproduction and a structural-rule
# ablation answer different questions and must remain separately named.
AVQDS_PUBLISHED_NUMERICS = EULER_RIDGE1E6_NUMERICS

# Backwards-compatible name used by campaign builders.  "Shared" now means the
# canonical shared-stack comparison numerics, which are RK4.
SHARED_NUMERICS = RK4_NUMERICS

NUMERICS: Mapping[str, Numerics] = {
    n.numerics_id: n
    for n in (EULER_NUMERICS, EULER_RIDGE1E6_NUMERICS, RK4_NUMERICS)
}


@dataclass(frozen=True)
class StepControl:
    """How the step size is chosen -- the axis to isolate.

    Both methods control their step; they differ in the quantity they bound.
    AVQDS bounds parameter motion (max|d theta| <= d theta_max); this route
    bounds state motion and subdivides. Pairing each rule with each control is
    what separates "their rule is better" from "their step control is better".
    """

    control_id: str
    flags: tuple[str, ...]
    required_integrator: str | None = None
    solve_repair_enabled: bool = False
    state_motion_l2_step_max: float | None = None
    parameter_step_max: float | None = None
    max_local_subdivisions: int | None = None
    note: str = ""


DELTA_THETA_CONTROL = StepControl(
    control_id="delta_theta_5e-3",
    flags=("--avqds-delta-theta-max", "5.0e-3", "--no-solve-repair"),
    required_integrator="euler",
    parameter_step_max=5.0e-3,
    note="Published AVQDS Euler control law, at the source's own value.",
)

STATE_MOTION_CONTROL = StepControl(
    control_id="state_motion_1e-2",
    flags=(
        "--solve-repair",
        "--solve-repair-profile", "minimal",
        "--solve-repair-state-motion-l2-step-max", "1.0e-2",
        "--solve-repair-kink-eta-max", "5.0e-3",
        "--solve-repair-max-local-subdivisions", "10",
    ),
    solve_repair_enabled=True,
    state_motion_l2_step_max=1.0e-2,
    max_local_subdivisions=10,
    note=(
        "This route's control law. The subdivision budget is 10 because at 4 a "
        "step could exhaust it, fail to cure a violation, and advance anyway, "
        "taking a measured error from 3.8e-3 to 2.1e-1."
    ),
)

COMPOSED_STATE_PARAMETER_CONTROL = StepControl(
    control_id="state_motion_1e-2_plus_parameter_5e-3",
    flags=(
        *STATE_MOTION_CONTROL.flags,
        "--solve-repair-parameter-step-max", "5.0e-3",
    ),
    solve_repair_enabled=True,
    state_motion_l2_step_max=1.0e-2,
    parameter_step_max=5.0e-3,
    max_local_subdivisions=10,
    note=(
        "Composed controller with both the tangent-state and maximum "
        "single-parameter step bounds active."
    ),
)

DELTA_THETA_CONTROL_TIGHT_10X = StepControl(
    control_id="delta_theta_5e-4_diagnostic",
    flags=("--avqds-delta-theta-max", "5.0e-4", "--no-solve-repair"),
    required_integrator="euler",
    parameter_step_max=5.0e-4,
    note="Diagnostic tenfold tightening of the parameter-motion controller.",
)

DELTA_THETA_CONTROL_TIGHT_100X = StepControl(
    control_id="delta_theta_5e-5_diagnostic",
    flags=("--avqds-delta-theta-max", "5.0e-5", "--no-solve-repair"),
    required_integrator="euler",
    parameter_step_max=5.0e-5,
    note="Diagnostic hundredfold tightening of the parameter-motion controller.",
)

STATE_MOTION_CONTROL_TIGHT_10X = StepControl(
    control_id="state_motion_1e-3_diagnostic",
    flags=(
        "--solve-repair",
        "--solve-repair-profile", "minimal",
        "--solve-repair-state-motion-l2-step-max", "1.0e-3",
        "--solve-repair-kink-eta-max", "5.0e-3",
        "--solve-repair-max-local-subdivisions", "10",
    ),
    solve_repair_enabled=True,
    state_motion_l2_step_max=1.0e-3,
    max_local_subdivisions=10,
    note="Diagnostic tenfold tightening of the state-motion controller.",
)

STATE_MOTION_CONTROL_TIGHT_100X = StepControl(
    control_id="state_motion_1e-4_diagnostic",
    flags=(
        "--solve-repair",
        "--solve-repair-profile", "minimal",
        "--solve-repair-state-motion-l2-step-max", "1.0e-4",
        "--solve-repair-kink-eta-max", "5.0e-3",
        "--solve-repair-max-local-subdivisions", "10",
    ),
    solve_repair_enabled=True,
    state_motion_l2_step_max=1.0e-4,
    max_local_subdivisions=10,
    note="Diagnostic hundredfold tightening of the state-motion controller.",
)

STATE_MOTION_CONTROL_5E5_DIAGNOSTIC = StepControl(
    control_id="state_motion_5e-5_diagnostic",
    flags=(
        "--solve-repair",
        "--solve-repair-profile", "minimal",
        "--solve-repair-state-motion-l2-step-max", "5.0e-5",
        "--solve-repair-kink-eta-max", "5.0e-3",
        "--solve-repair-max-local-subdivisions", "11",
    ),
    solve_repair_enabled=True,
    state_motion_l2_step_max=5.0e-5,
    max_local_subdivisions=11,
    note="Diagnostic state-motion calibration immediately below 1.0e-4.",
)

COMPOSED_STATE_PARAMETER_CONTROL_TIGHT_10X = StepControl(
    control_id="state_motion_1e-3_plus_parameter_5e-4_diagnostic",
    flags=(
        *STATE_MOTION_CONTROL_TIGHT_10X.flags,
        "--solve-repair-parameter-step-max", "5.0e-4",
    ),
    solve_repair_enabled=True,
    state_motion_l2_step_max=1.0e-3,
    parameter_step_max=5.0e-4,
    max_local_subdivisions=10,
    note="Diagnostic tenfold tightening of both controller bounds.",
)

COMPOSED_CONTROL_PARAMETER_TIGHT_100X = StepControl(
    control_id="state_motion_1e-3_plus_parameter_5e-5_diagnostic",
    flags=(
        "--solve-repair",
        "--solve-repair-profile", "minimal",
        "--solve-repair-state-motion-l2-step-max", "1.0e-3",
        "--solve-repair-kink-eta-max", "5.0e-3",
        "--solve-repair-max-local-subdivisions", "13",
        "--solve-repair-parameter-step-max", "5.0e-5",
    ),
    solve_repair_enabled=True,
    state_motion_l2_step_max=1.0e-3,
    parameter_step_max=5.0e-5,
    max_local_subdivisions=13,
    note=(
        "Diagnostic composition of the tenfold-tighter state-motion bound "
        "and hundredfold-tighter parameter-motion bound."
    ),
)

DELTA_THETA_CONTROL_CALIBRATED = StepControl(
    control_id="delta_theta_5e-5",
    flags=(
        *DELTA_THETA_CONTROL_TIGHT_100X.flags,
        "--fail-on-unsupported-steps",
    ),
    required_integrator="euler",
    parameter_step_max=5.0e-5,
    note="Paper-facing parameter-motion controller calibrated at fixed accuracy.",
)

STATE_MOTION_CONTROL_CALIBRATED = StepControl(
    control_id="state_motion_5e-5",
    flags=(
        *STATE_MOTION_CONTROL_5E5_DIAGNOSTIC.flags,
        "--fail-on-unsupported-steps",
    ),
    solve_repair_enabled=True,
    state_motion_l2_step_max=5.0e-5,
    max_local_subdivisions=11,
    note="Paper-facing state-motion controller calibrated at fixed accuracy.",
)

STATE_MOTION_CONTROL_CALIBRATED_SUB15 = StepControl(
    control_id="state_motion_5e-5_sub15",
    flags=(
        "--solve-repair",
        "--solve-repair-profile", "minimal",
        "--solve-repair-state-motion-l2-step-max", "5.0e-5",
        "--solve-repair-kink-eta-max", "5.0e-3",
        "--solve-repair-max-local-subdivisions", "15",
        "--fail-on-unsupported-steps",
    ),
    solve_repair_enabled=True,
    state_motion_l2_step_max=5.0e-5,
    max_local_subdivisions=15,
    note=(
        "Named experiment: the paper-facing state-motion controller with the local "
        "subdivision budget raised from 11 to 15 for the strong--fast (2.4,3) drive, "
        "where 11 leaves about half of the steps uncured in both arms."
    ),
)

COMPOSED_CONTROL_CALIBRATED = StepControl(
    control_id="state_motion_1e-3_plus_parameter_5e-5",
    flags=(
        *COMPOSED_CONTROL_PARAMETER_TIGHT_100X.flags,
        "--fail-on-unsupported-steps",
    ),
    solve_repair_enabled=True,
    state_motion_l2_step_max=1.0e-3,
    parameter_step_max=5.0e-5,
    max_local_subdivisions=13,
    note="Paper-facing composed controller calibrated at fixed accuracy.",
)

STEP_CONTROLS: Mapping[str, StepControl] = {
    c.control_id: c
    for c in (
        DELTA_THETA_CONTROL,
        STATE_MOTION_CONTROL,
        COMPOSED_STATE_PARAMETER_CONTROL,
        DELTA_THETA_CONTROL_TIGHT_10X,
        DELTA_THETA_CONTROL_TIGHT_100X,
        STATE_MOTION_CONTROL_TIGHT_10X,
        STATE_MOTION_CONTROL_TIGHT_100X,
        STATE_MOTION_CONTROL_5E5_DIAGNOSTIC,
        COMPOSED_STATE_PARAMETER_CONTROL_TIGHT_10X,
        COMPOSED_CONTROL_PARAMETER_TIGHT_100X,
        DELTA_THETA_CONTROL_CALIBRATED,
        STATE_MOTION_CONTROL_CALIBRATED,
        STATE_MOTION_CONTROL_CALIBRATED_SUB15,
        COMPOSED_CONTROL_CALIBRATED,
    )
}

@dataclass(frozen=True)
class SearchProfile:
    """One explicit approximation budget for generalized exchange.

    These settings bound computational work; they do not change the score or
    certification mathematics.  A binding cap can nevertheless change which
    patch is selected, so the profile id is part of every run id and campaign
    manifest.  ``paper_facing_allowed`` describes intended use only; it never
    judges a completed result.
    """

    profile_id: str
    pool_size: int
    joint_patch_evaluations: int
    certification_attempts_per_level: int
    certification_attempts_per_deletion_branch: int
    rounds_per_checkpoint: int
    insertion_cardinality: int = 1
    exchange_insertion_frontier: int | None = None
    paper_facing_allowed: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        for name in (
            "pool_size",
            "joint_patch_evaluations",
            "certification_attempts_per_level",
            "certification_attempts_per_deletion_branch",
            "rounds_per_checkpoint",
            "insertion_cardinality",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if (
            self.exchange_insertion_frontier is not None
            and int(self.exchange_insertion_frontier) < 1
        ):
            raise ValueError("exchange_insertion_frontier must be at least 1")

    @property
    def flags(self) -> tuple[str, ...]:
        emitted: tuple[str, ...] = (
            "--max-structural-pool-size", str(int(self.pool_size)),
            "--max-joint-patch-evaluations",
            str(int(self.joint_patch_evaluations)),
            "--max-certification-attempts-per-level",
            str(int(self.certification_attempts_per_level)),
            "--max-certification-attempts-per-deletion-branch",
            str(int(self.certification_attempts_per_deletion_branch)),
            "--max-insertion-rounds-per-checkpoint",
            str(int(self.rounds_per_checkpoint)),
            "--max-insertion-batch-size", str(int(self.insertion_cardinality)),
        )
        if self.exchange_insertion_frontier is None:
            # Byte-identical to every profile that predates the exchange-face
            # bound, so their run ids and argv stay unchanged.
            return emitted
        return emitted + (
            "--exchange-insertion-frontier",
            str(int(self.exchange_insertion_frontier)),
        )

DIAGNOSTIC_FAST_SEARCH = SearchProfile(
    profile_id="diagnostic_fast_v1",
    pool_size=32,
    joint_patch_evaluations=2_000,
    certification_attempts_per_level=3,
    certification_attempts_per_deletion_branch=1,
    rounds_per_checkpoint=3,
    note=(
        "Cheap execution-path diagnostic. Its capped candidate pool and work "
        "budgets make it unsuitable for Paper-II resource comparisons."
    ),
)

PAPER_DECLARED_SEARCH = SearchProfile(
    profile_id="paper_declared_v1",
    pool_size=128,
    joint_patch_evaluations=50_000,
    certification_attempts_per_level=12,
    certification_attempts_per_deletion_branch=2,
    rounds_per_checkpoint=12,
    paper_facing_allowed=True,
    note="Declared Paper-II finite-work search budget.",
)

SIAM_NB2_PROOF_SEARCH = SearchProfile(
    profile_id="siam_nb2_proof_v1",
    pool_size=64,
    joint_patch_evaluations=50_000,
    certification_attempts_per_level=12,
    certification_attempts_per_deletion_branch=2,
    rounds_per_checkpoint=12,
    paper_facing_allowed=True,
    note=(
        "SIAM Nb=2 proof budget. The cap is nonbinding for its complete "
        "14-block candidate pool."
    ),
)

SIAM_NB4_PILOT_SEARCH = SearchProfile(
    profile_id="siam_nb4_pilot_v1",
    pool_size=64,
    joint_patch_evaluations=50_000,
    certification_attempts_per_level=12,
    certification_attempts_per_deletion_branch=2,
    rounds_per_checkpoint=12,
    paper_facing_allowed=True,
    note=(
        "SIAM Nb=4 pilot budget. The cap is nonbinding for its complete "
        "24-block K/J/Qd/A/X/P logical pool."
    ),
)

# The SIAM Nb=4 pilot budget with the exchange face reachable.  The work
# limits are identical to ``siam_nb4_pilot_v1``; the only difference is that a
# d >= 1 deletion set is paired with the top 8 positioned insertions sealed
# from the d = 0 pass rather than the whole insertion face.  Before this
# profile no SIAM production run ever enumerated a true-exchange patch: every
# d >= 1 rung was refused whole at the joint-work cap, so the arm reduced to
# append-only in practice.  Its own ``profile_id``, so every packaged action,
# receipt, and campaign cell separates from the runs whose search could not
# reach exchange at all.
SIAM_NB4_PILOT_EXCHANGE_SEARCH = SearchProfile(
    profile_id="siam_nb4_pilot_exchange_v1",
    pool_size=64,
    joint_patch_evaluations=50_000,
    certification_attempts_per_level=12,
    certification_attempts_per_deletion_branch=2,
    rounds_per_checkpoint=12,
    exchange_insertion_frontier=8,
    paper_facing_allowed=True,
    note=(
        "SIAM Nb=4 pilot budget with the exchange face reachable. Identical "
        "work limits to siam_nb4_pilot_v1; the insertion face offered to a "
        "d >= 1 deletion set is a declared width-8 frontier sealed from the "
        "d = 0 pass, so an exchange-bearing rung fits under the same "
        "nonbinding 50,000 joint-evaluation cap."
    ),
)

# Every search budget a registered SIAM Nb=4 command may run under, keyed by
# the profile id that its action, manifest, and receipt record.
SIAM_NB4_SEARCH_PROFILES: Mapping[str, SearchProfile] = {
    profile.profile_id: profile
    for profile in (SIAM_NB4_PILOT_SEARCH, SIAM_NB4_PILOT_EXCHANGE_SEARCH)
}

CAP_SENSITIVITY_2X_SEARCH = SearchProfile(
    profile_id="cap_sensitivity_2x_v1",
    pool_size=128,
    joint_patch_evaluations=100_000,
    certification_attempts_per_level=24,
    certification_attempts_per_deletion_branch=4,
    rounds_per_checkpoint=24,
    paper_facing_allowed=True,
    note=(
        "Selected-cell sensitivity check at twice the declared work limits; "
        "not an exhaustive combinatorial search."
    ),
)

# The declared budget with the exchange face reachable.  Identical work limits
# to paper_declared_v1: the only difference is that a d >= 1 deletion set is
# paired with the top 8 positioned insertions of the d = 0 pass rather than all
# of them, so the first exchange-bearing rung declares |J_del| x 9 (~378 at the
# measured median |J_del| = 42) instead of |J_del| x 2030 (48,720-165,865
# measured) and fits under the same 50,000 cap.  Its own profile_id, so every
# run id and campaign cell separates from the runs whose search could not
# reach exchange at all.
PAPER_DECLARED_EXCHANGE_SEARCH = SearchProfile(
    profile_id="paper_declared_exchange_v1",
    pool_size=128,
    joint_patch_evaluations=50_000,
    certification_attempts_per_level=12,
    certification_attempts_per_deletion_branch=2,
    rounds_per_checkpoint=12,
    exchange_insertion_frontier=8,
    paper_facing_allowed=True,
    note=(
        "Declared Paper-II finite-work search budget with the exchange face "
        "reachable. Deletion rungs stay complete; the insertion face offered "
        "to a deletion set is a declared width-8 frontier sealed from the "
        "d = 0 pass, so the exchange face is an approximation of the "
        "specification's full product and the other two faces are not."
    ),
)

SEARCH_PROFILES: Mapping[str, SearchProfile] = {
    profile.profile_id: profile
    for profile in (
        DIAGNOSTIC_FAST_SEARCH,
        PAPER_DECLARED_SEARCH,
        PAPER_DECLARED_EXCHANGE_SEARCH,
        CAP_SENSITIVITY_2X_SEARCH,
    )
}

# Structural settings shared by every arm. The pool cap sits above the 125-word
# deduplicated pool so it does not bind: a cap of 8 discarded ~117 usable words
# and was the largest single accuracy defect found in the 2026-08-22 audit. The
# conditioning gate is off because no setting of it was useful.
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
    required_numerics_id: str | None = None
    required_step_control_id: str | None = None
    note: str = ""

    @property
    def dynamics_policy(self) -> str:
        """The runner's ``--dynamics-policy`` value for this arm.

        The runner admits exactly ``exchange`` and ``avqds``.  The arm id is a
        finer label (route bundles share the exchange policy), so it is never a
        policy value.
        """

        return "avqds" if self.is_comparator else "exchange"


EXCHANGE = Arm(
    arm_id="exchange",
    flags=(
        "--debt-policy", "drift_ranked",
        "--prune-history-lambda", "0.0",
        "--prune-target-policy", "all_active",
        "--prune-ray-distance-tol", "2.0e-3",
    ),
    note=(
        "The paper's route: deletions and positioned insertions compete as one "
        "atomic patch. Ray tolerance 2e-3 rather than the 5e-2 default, which "
        "admits cumulative structural damage over a long horizon."
    ),
)

APPEND_ONLY = Arm(
    arm_id="append_only",
    flags=("--no-exchange-deletions", "--debt-policy", "insertion_only"),
    note=(
        "Insert-face restriction of generalized exchange; isolates what the "
        "deletion component buys without invoking another selector."
    ),
)

AVQDS = Arm(
    arm_id="avqds",
    flags=("--dynamics-policy", "avqds", "--avqds-l2-cut", "1.0e-3"),
    is_comparator=True,
    note=(
        "AVQDS decision RULE on this route's numerical stack (shared geometry, "
        "inverse, solve repair, integrator, pool). Isolates structure from "
        "numerics; it is NOT the published method -- use avqds_published for "
        "that. Uncapped: the rule appends until L^2 < cut."
    ),
)

# Published AVQDS, as specified in Yao et al. (PRX Quantum 2, 030307), rather
# than the AVQDS decision rule running on this route's numerical stack.  The
# distinction is not cosmetic: measured on the strong fast drive, the
# shared-numerics arm had this route's local subdivision applied at 249 of 251
# checkpoints, plus solve repair and certification refit enabled, so it was
# stabilized by machinery the published method does not specify.  The paper
# states its own numerics explicitly -- "we adopt the Euler method", Tikhonov
# regularization with xi = 1e-6 added to the diagonal of M before inversion,
# a fixed step, and no repair or subdivision mechanism -- and this arm runs
# exactly that.
AVQDS_PUBLISHED = Arm(
    arm_id="avqds_published",
    flags=(
        "--dynamics-policy", "avqds",
        "--avqds-l2-cut", "1.0e-3",
        "--no-certification-refit",
    ),
    is_comparator=True,
    required_numerics_id=AVQDS_PUBLISHED_NUMERICS.numerics_id,
    required_step_control_id=DELTA_THETA_CONTROL.control_id,
    note=(
        "Yao et al. with the paper's own numerics: Euler, Tikhonov xi=1e-6, "
        "and the published parameter-controlled step (delta_theta_max=5e-3). "
        "AVQDS is NOT a fixed-step method -- it adapts dt so no parameter "
        "moves more than that budget, which the source calls its "
        "stabilization mechanism."
    ),
)

# Three closed selector bundles on the exchange route (2026-09-02).  Each arm
# is EXCHANGE plus one --exchange-route-bundle; objective, cost model, and
# kernel are fixed by the bundle registry in exchange_config.py, never here.
EXCHANGE_LEGACY_PROXY_FIXED = Arm(
    arm_id="exchange_legacy_proxy_fixed",
    flags=EXCHANGE.flags + (
        "--exchange-route-bundle", "paper_ii_exchange_legacy_proxy_fixed_v1",
    ),
    note=(
        "Frozen historical control with only the cost normalization repaired: "
        "the same asymmetric score and debt ordering, but the Paper-I proxy "
        "is normalized over the actual competing population instead of a "
        "singleton (which made K identically 1)."
    ),
)

EXCHANGE_Q_COMPILED = Arm(
    arm_id="exchange_q_compiled",
    flags=EXCHANGE.flags + (
        "--exchange-route-bundle", "paper_ii_exchange_q_compiled_v1",
    ),
    note=(
        "Realized route: complete-patch captured-drift change m_Q ranked with "
        "the shifted Paper-I kernel under population-relative compiled "
        "FakeMarrakesh cost K."
    ),
)

EXCHANGE_GAMMA_COMPILED = Arm(
    arm_id="exchange_gamma_compiled",
    flags=EXCHANGE.flags + (
        "--exchange-route-bundle", "paper_ii_exchange_gamma_compiled_v1",
    ),
    note=(
        "Ideal ablation: complete-patch ideal-projection change m_Gamma with "
        "the identical population and K map as the Q route; realized Q/L2 "
        "still certifies every finalist."
    ),
)

ARMS: Mapping[str, Arm] = {
    a.arm_id: a
    for a in (
        EXCHANGE,
        APPEND_ONLY,
        AVQDS,
        AVQDS_PUBLISHED,
        EXCHANGE_LEGACY_PROXY_FIXED,
        EXCHANGE_Q_COMPILED,
        EXCHANGE_GAMMA_COMPILED,
    )
}


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
        # Resonance discriminator. Both drives where this route loses to the
        # comparator sit at omega = 3, across a 4x amplitude range, while it
        # wins at omega = 1, 2, and 6. The seed has phonon frequency
        # omega_0 = 1, so omega = 3 is a third harmonic, and an n_ph = 1
        # truncation represents multi-phonon transitions badly. These drives
        # separate "sharp in omega" (resonance) from "smeared" (not).
        Drive("res_w2p5", 0.6, 2.5),
        Drive("res_w2p75", 0.6, 2.75),
        Drive("res_w3p25", 0.6, 3.25),
        Drive("res_w3p5", 0.6, 3.5),
        Drive("res_w4", 0.6, 4.0),
        # Amplitude at fixed omega = 3, to confirm the failure is frequency
        # selective rather than amplitude driven.
        Drive("res_a0p3", 0.3, 3.0),
        Drive("res_a1p2", 1.2, 3.0),
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
        Horizon("t5", 5.0, 126),
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
    numerics: "Numerics" = SHARED_NUMERICS
    step_control: "StepControl" = STATE_MOTION_CONTROL
    search_profile: "SearchProfile" = PAPER_DECLARED_SEARCH
    activation_cut: float | None = None

    def __post_init__(self) -> None:
        required_numerics = self.arm.required_numerics_id
        if required_numerics is not None and self.numerics.numerics_id != required_numerics:
            raise ValueError(
                f"arm {self.arm.arm_id!r} requires numerics {required_numerics!r}; "
                f"got {self.numerics.numerics_id!r}"
            )
        required_control = self.arm.required_step_control_id
        if required_control is not None and self.step_control.control_id != required_control:
            raise ValueError(
                f"arm {self.arm.arm_id!r} requires step control {required_control!r}; "
                f"got {self.step_control.control_id!r}"
            )
        required_integrator = self.step_control.required_integrator
        if (
            required_integrator is not None
            and self.numerics.integrator != required_integrator
        ):
            raise ValueError(
                f"step control {self.step_control.control_id!r} requires integrator "
                f"{required_integrator!r}; got {self.numerics.integrator!r}"
            )
        if self.activation_cut is not None and (
            not math.isfinite(float(self.activation_cut))
            or float(self.activation_cut) <= 0.0
        ):
            raise ValueError("activation_cut must be finite and positive")

    @property
    def run_id(self) -> str:
        base = (
            f"{self.regime_id}_{self.drive.drive_id}_{self.horizon.horizon_id}"
            f"_{self.arm.arm_id}_{self.gate.gate_id}"
            f"_{self.numerics.numerics_id}_{self.step_control.control_id}"
            f"_{self.search_profile.profile_id}"
        )
        if self.activation_cut is None:
            return base
        return f"{base}_cut{float(self.activation_cut):.12g}"

    def _replace_activation_cut(self, flags: tuple[str, ...]) -> tuple[str, ...]:
        """Materialize the one per-cell accuracy-search variable once."""

        if self.activation_cut is None:
            return flags
        flag = "--avqds-l2-cut" if self.arm.is_comparator else "--insertion-l2-cut"
        values = list(flags)
        if flag not in values:
            raise ValueError(
                f"{self.arm.arm_id!r} / {self.gate.gate_id!r} has no {flag} to vary"
            )
        values[values.index(flag) + 1] = repr(float(self.activation_cut))
        return tuple(values)

    def argv(self) -> tuple[str, ...]:
        # AVQDS carries its own append condition; layering this route's
        # insertion gate on top would misrepresent the comparator.
        gate_flags = () if self.arm.is_comparator else self.gate.flags
        arm_flags = self.arm.flags
        if self.activation_cut is not None:
            if self.arm.is_comparator:
                arm_flags = self._replace_activation_cut(arm_flags)
            else:
                gate_flags = self._replace_activation_cut(gate_flags)
        return (
            "--artifact-json", str(self.seed_path),
            *self.horizon.flags,
            *self.drive.flags,
            *self.numerics.flags,
            *self.step_control.flags,
            *self.search_profile.flags,
            *LOCKED_STRUCTURAL_POLICY,
            *gate_flags,
            *arm_flags,
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
    numerics: str | None = None,
    step_control: str | None = None,
    search_profile: str = PAPER_DECLARED_SEARCH.profile_id,
    extra_flags: Sequence[str] = (),
    activation_cut: float | None = None,
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
    if search_profile not in SEARCH_PROFILES:
        raise KeyError(
            f"unknown search_profile {search_profile!r}; "
            f"known: {sorted(SEARCH_PROFILES)}"
        )
    selected_arm = ARMS[arm]
    if selected_arm.required_step_control_id is not None:
        if (
            step_control is not None
            and step_control != selected_arm.required_step_control_id
        ):
            raise ValueError(
                f"arm {arm!r} fixes step_control to "
                f"{selected_arm.required_step_control_id!r}"
            )
        step_control = selected_arm.required_step_control_id
    elif step_control is None:
        step_control = STATE_MOTION_CONTROL.control_id
    if step_control not in STEP_CONTROLS:
        raise KeyError(
            f"unknown step_control {step_control!r}; known: {sorted(STEP_CONTROLS)}"
        )
    selected_control = STEP_CONTROLS[step_control]

    if selected_arm.required_numerics_id is not None:
        if numerics is not None and numerics != selected_arm.required_numerics_id:
            raise ValueError(
                f"arm {arm!r} fixes numerics to "
                f"{selected_arm.required_numerics_id!r}"
            )
        numerics = selected_arm.required_numerics_id
    elif numerics is None:
        numerics = (
            EULER_NUMERICS.numerics_id
            if selected_control.required_integrator == "euler"
            else SHARED_NUMERICS.numerics_id
        )
    if numerics not in NUMERICS:
        raise KeyError(f"unknown numerics {numerics!r}; known: {sorted(NUMERICS)}")
    return RunCommand(
        seed_path=str(seed_path),
        regime_id=str(regime) if regime is not None else "unregistered",
        arm=selected_arm,
        gate=GATES[gate],
        drive=DRIVES[drive],
        horizon=HORIZONS[horizon],
        numerics=NUMERICS[numerics],
        step_control=selected_control,
        search_profile=SEARCH_PROFILES[search_profile],
        output_json=str(output_json),
        extra_flags=tuple(str(f) for f in extra_flags),
        activation_cut=(
            None if activation_cut is None else float(activation_cut)
        ),
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


# The HH regime matrix (chtc/paper_ii_regime_seeds_v1/input/regimes.tsv): the
# 2026-08-22 six-cell grid plus the 2026-09-04 anchor row at the published
# calibration point (U=4, g_ep=0.5, n_ph=1), the only row that nests the locked
# hh_snake_nph1 evidence. Seeds come from the frozen three-complete-layer
# fixed-structure VQE (chtc/paper_ii_regime_seeds_v1/run_seed.sh, the
# prospective HH seed policy). The earlier conditioning-stress search never
# wrote an artifact at its 1e-4 gate (clusters 9663698 and 9674800: every
# Pareto front was empty because no driven-snapshot fit reached 1e-6), and its
# nph=7 rows were held at the 24 GB memory limit on 2026-08-22. The nph=7 cells
# are 10 qubits against 8, 16x the state space of the locked evidence. Every
# regime seed is built at dv = 0; the calibration seed's dv = 0.25 is a uniform
# site potential (a constant -dv*N_e at fixed filling), so the anchor is the
# same dynamics with energies 0.5 above the locked convention.
_HH_REGIME_ROWS = (
    ("weak_weak", 0.25, 0.353553390593, 3),
    ("intermediate_weak", 1.25, 0.353553390593, 3),
    ("strong_weak_u8", 8.00, 0.353553390593, 3),
    ("weak_strong", 0.25, 0.790569415042, 7),
    ("intermediate_strong", 1.25, 0.790569415042, 7),
    ("strong_strong_u8", 8.00, 0.790569415042, 7),
    ("anchor_u4_g0p5_nph1", 4.00, 0.5, 1),
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
    show.add_argument("--numerics", default=None)
    show.add_argument(
        "--step-control", default=None
    )
    show.add_argument("--drive", default="fastweak")
    show.add_argument("--horizon", default="t10")
    show.add_argument("--output-json", default="output/paper_ii/run.json")
    params = sub.add_parser(
        "params",
        help="print every effective parameter for one run (registry + defaults)",
    )
    params.add_argument("--regime", default="hh_snake_nph1")
    params.add_argument("--arm", default="exchange")
    params.add_argument("--gate", default=MCLACHLAN_L2_GATE.gate_id)
    params.add_argument("--numerics", default=None)
    params.add_argument("--step-control", default=None)
    params.add_argument("--drive", default="fastweak")
    params.add_argument("--horizon", default="t10")
    params.add_argument("--changed-only", action="store_true",
                        help="show only values the registry overrides")
    matrix = sub.add_parser("matrix", help="print the whole drive x arm matrix")
    matrix.add_argument("--regimes", default="hh_snake_nph1",
                        help="comma-separated regime ids")
    matrix.add_argument("--gate", default=MCLACHLAN_L2_GATE.gate_id)
    matrix.add_argument("--horizon", default="t10")
    matrix.add_argument("--output-root", default="output/paper_ii")
    args = parser.parse_args(argv)

    if args.command == "list":
        for title, registry in (
            ("arms", ARMS), ("gates", GATES), ("numerics", NUMERICS),
            ("step controls", STEP_CONTROLS), ("drives", DRIVES),
            ("horizons", HORIZONS),
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
    if args.command == "params":
        from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
            _build_parser as _runner_parser,
        )

        run = build_run(
            regime=args.regime, arm=args.arm, gate=args.gate, drive=args.drive,
            horizon=args.horizon, numerics=args.numerics,
            step_control=args.step_control, output_json="<output>",
        )
        parser_ = _runner_parser()
        effective = vars(parser_.parse_args(list(run.argv())))
        defaults = vars(parser_.parse_args(["--artifact-json", "x", "--output-json", "y"]))
        print(f"# run_id: {run.run_id}")
        print(f"# seed:   {run.seed_path}")
        print(f"# {'parameter':46s} {'value':>22s}   source")
        for key in sorted(effective):
            value, base = effective[key], defaults.get(key)
            changed = value != base
            if args.changed_only and not changed:
                continue
            print(f"  {key:46s} {str(value):>22s}   "
                  f"{'registry' if changed else 'runner default'}")
        return 0
    if args.command == "show":
        print(build_run(
            regime=args.regime, arm=args.arm, gate=args.gate, drive=args.drive,
            horizon=args.horizon, numerics=args.numerics,
            step_control=args.step_control, output_json=args.output_json,
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



__all__ = [
    "ARMS", "AVQDS", "AVQDS_PUBLISHED", "APPEND_ONLY", "DELTA_THETA_CONTROL",
    "EXCHANGE_GAMMA_COMPILED", "EXCHANGE_LEGACY_PROXY_FIXED", "EXCHANGE_Q_COMPILED",
    "AVQDS_PUBLISHED_NUMERICS", "COMPOSED_STATE_PARAMETER_CONTROL",
    "DRIVES", "EULER_NUMERICS", "EULER_RIDGE1E6_NUMERICS", "EXCHANGE",
    "NUMERICS", "Numerics",
    "RK4_NUMERICS", "SHARED_NUMERICS", "STATE_MOTION_CONTROL",
    "STEP_CONTROLS", "StepControl",
    "GATES", "HH_SNAKE_NPH1", "HORIZONS", "REGIMES", "Regime",
    "SeedNotBuiltError", "available_regimes", "resolve_regime", "MCLACHLAN_L2_GATE",
    "PRODUCTION_STRUCTURE", "RESIDUAL_GATE", "Arm", "Drive", "Horizon",
    "InsertionGate", "RunCommand", "build_run", "main", "paper_ii_runs",
]


# --- Ported 2026-08-31 from codex/paper-ii-search-profiles-20260824: SIAM
# nb2/nb4 arm registry and search-profile machinery (additive symbols only;
# the branch's wider rewrite of this file remains unmerged - RECOVERY-AUDIT.md
# sec 11). ---

LOCKED_STRUCTURAL_POLICY: tuple[str, ...] = (
    "--no-append-schur-condition-gate",
)

SIAM_NB2_PROOF_HORIZON = Horizon("t1_26", 1.0, 26)

SIAM_NB2_PROOF_ARMS: Mapping[str, Arm] = {
    arm.arm_id: arm for arm in (EXCHANGE, AVQDS, EXCHANGE_Q_COMPILED)
}

SIAM_NB4_TIME_STEP = 1.0e-2

SIAM_NB4_FULL_TIMES: tuple[float, ...] = tuple(
    float(index) * SIAM_NB4_TIME_STEP for index in range(501)
)

SIAM_NB4_SCREEN_TIMES: tuple[float, ...] = SIAM_NB4_FULL_TIMES[:101]

SIAM_NB4_CONTINUATION_TIMES: tuple[float, ...] = SIAM_NB4_FULL_TIMES[100:]

SIAM_NB4_INITIAL_ACTIVATION_CUT = 1.0e-3

SIAM_NB4_ARMS: Mapping[str, Arm] = {
    arm.arm_id: arm for arm in (EXCHANGE, AVQDS, EXCHANGE_Q_COMPILED)
}

@dataclass(frozen=True)
class SIAMNB4Segment:
    """One exact ordered-grid slice in the resumable SIAM Nb=4 pilot."""

    segment_id: str
    times: tuple[float, ...]
    requires_resume: bool

    def __post_init__(self) -> None:
        if not self.times:
            raise ValueError("SIAM Nb=4 segment requires a nonempty time grid")
        if any(
            not math.isfinite(float(value))
            for value in self.times
        ) or any(
            right <= left for left, right in zip(self.times, self.times[1:])
        ):
            raise ValueError("SIAM Nb=4 segment times must be finite and increasing")

    @property
    def flags(self) -> tuple[str, ...]:
        return (
            "--times",
            ",".join(repr(float(value)) for value in self.times),
        )

SIAM_NB4_SCREEN_SEGMENT = SIAMNB4Segment(
    segment_id="screen_t0_t1_101",
    times=SIAM_NB4_SCREEN_TIMES,
    requires_resume=False,
)

SIAM_NB4_CONTINUATION_SEGMENT = SIAMNB4Segment(
    segment_id="continue_t1_t5_401",
    times=SIAM_NB4_CONTINUATION_TIMES,
    requires_resume=True,
)

@dataclass(frozen=True)
class SIAMProofRunCommand:
    """One arm of the locked, reference-blind SIAM Nb=2 proof pair."""

    seed_path: str
    arm: Arm
    output_json: str
    code_revision: str
    extra_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.code_revision).strip():
            raise ValueError("code_revision must name the tested commit")

    @property
    def horizon(self) -> Horizon:
        return SIAM_NB2_PROOF_HORIZON

    @property
    def run_id(self) -> str:
        return f"siam_nb2_{self.horizon.horizon_id}_{self.arm.arm_id}"

    @property
    def search_profile(self) -> SearchProfile:
        return SIAM_NB2_PROOF_SEARCH

    def argv(self) -> tuple[str, ...]:
        gate_flags = () if self.arm.is_comparator else MCLACHLAN_L2_GATE.flags
        policy_flags = (
            ()
            if "--dynamics-policy" in self.arm.flags
            else ("--dynamics-policy", self.arm.dynamics_policy)
        )
        return (
            "--siam-seed-json", str(self.seed_path),
            *self.horizon.flags,
            *EULER_RIDGE1E6_NUMERICS.flags,
            *STATE_MOTION_CONTROL_CALIBRATED.flags,
            *self.search_profile.flags,
            *LOCKED_STRUCTURAL_POLICY,
            *gate_flags,
            *policy_flags,
            *self.arm.flags,
            "--parameterization-mode", "logical_shared",
            "--record-statevector",
            "--require-complete-candidate-pool",
            *PROGRESS,
            *self.extra_flags,
            "--code-revision", str(self.code_revision),
            "--output-json", str(self.output_json),
        )

    def shell(self) -> str:
        parts = ["PYTHONPATH=.", "PYTHONUNBUFFERED=1", "python3", RUNNER_PATH]
        return " \\\n  ".join(parts + list(self.argv()))

def build_siam_nb2_proof_run(
    *,
    seed_path: str,
    arm: str,
    output_json: str,
    code_revision: str,
    extra_flags: Sequence[str] = (),
) -> SIAMProofRunCommand:
    """Compose one arm of the single registered SIAM Nb=2 proof pair."""

    if arm not in SIAM_NB2_PROOF_ARMS:
        raise KeyError(
            f"unknown SIAM Nb=2 proof arm {arm!r}; "
            f"known: {sorted(SIAM_NB2_PROOF_ARMS)}"
        )
    return SIAMProofRunCommand(
        seed_path=str(seed_path),
        arm=SIAM_NB2_PROOF_ARMS[arm],
        output_json=str(output_json),
        code_revision=str(code_revision),
        extra_flags=tuple(str(flag) for flag in extra_flags),
    )

@dataclass(frozen=True)
class SIAMNB4RunCommand:
    """One source-locked SIAM Nb=4 screen or continuation command.

    The raw trajectory remains exact-reference blind.  Screen and continuation
    commands bind exact slices of one canonical ``T=5``/501-point reporting
    grid, so their later digest-bound stitch cannot accept a nearly-equal grid.
    """

    seed_path: str
    arm: Arm
    segment: SIAMNB4Segment
    output_json: str
    code_revision: str
    activation_cut: float
    resume_from_run_json: str | None = None
    # ``None`` is the historical pilot budget, so every pre-existing command
    # keeps its exact argv and run id.  A registered id selects another
    # registered SIAM Nb=4 budget (currently the exchange-frontier one).
    search_profile_id: str | None = None

    def __post_init__(self) -> None:
        registered_arm = SIAM_NB4_ARMS.get(self.arm.arm_id)
        if registered_arm is None or self.arm != registered_arm:
            raise ValueError(
                "SIAM Nb=4 supports only registered arms: "
                f"{sorted(SIAM_NB4_ARMS)}"
            )
        if self.segment not in (
            SIAM_NB4_SCREEN_SEGMENT,
            SIAM_NB4_CONTINUATION_SEGMENT,
        ):
            raise ValueError("SIAM Nb=4 command requires a registered grid segment")
        if not str(self.code_revision).strip():
            raise ValueError("code_revision must name the tested commit")
        cut = float(self.activation_cut)
        if not math.isfinite(cut) or cut <= 0.0:
            raise ValueError("activation_cut must be finite and positive")
        resume_path = str(self.resume_from_run_json or "").strip()
        if self.segment.requires_resume and not resume_path:
            raise ValueError("SIAM Nb=4 continuation requires a prior run JSON")
        if not self.segment.requires_resume and resume_path:
            raise ValueError("SIAM Nb=4 screen must start from the immutable seed")
        if (
            self.search_profile_id is not None
            and str(self.search_profile_id) not in SIAM_NB4_SEARCH_PROFILES
        ):
            raise ValueError(
                "SIAM Nb=4 supports only registered search profiles: "
                f"{sorted(SIAM_NB4_SEARCH_PROFILES)}"
            )

    @property
    def run_id(self) -> str:
        cut_token = format(float(self.activation_cut), ".12g").replace("-", "m")
        return (
            f"siam_nb4_{self.segment.segment_id}_{self.arm.arm_id}_"
            f"cut{cut_token}"
        )

    @property
    def search_profile(self) -> SearchProfile:
        if self.search_profile_id is None:
            return SIAM_NB4_PILOT_SEARCH
        return SIAM_NB4_SEARCH_PROFILES[str(self.search_profile_id)]

    def argv(self) -> tuple[str, ...]:
        cut = repr(float(self.activation_cut))
        resume_flags = (
            ()
            if not self.segment.requires_resume
            else ("--resume-from-run-json", str(self.resume_from_run_json))
        )
        # Gate on the structural policy, not the arm id: every non-comparator
        # arm is an exchange route and must keep its own flags (including
        # --exchange-route-bundle) and the mclachlan_l2 insertion gate.
        exchange_flags = () if self.arm.is_comparator else self.arm.flags
        exchange_gate_flags = (
            ()
            if self.arm.is_comparator
            else ("--insertion-gate-mode", "mclachlan_l2")
        )
        return (
            *resume_flags,
            "--siam-seed-json", str(self.seed_path),
            *self.segment.flags,
            *EULER_RIDGE1E6_NUMERICS.flags,
            *STATE_MOTION_CONTROL_CALIBRATED.flags,
            *self.search_profile.flags,
            *LOCKED_STRUCTURAL_POLICY,
            *exchange_gate_flags,
            "--insertion-l2-cut", cut,
            "--avqds-l2-cut", cut,
            "--dynamics-policy", self.arm.dynamics_policy,
            *exchange_flags,
            "--parameterization-mode", "logical_shared",
            "--append-occurrence-policy", "layer_reuse",
            "--record-statevector",
            "--require-complete-candidate-pool",
            *PROGRESS,
            "--code-revision", str(self.code_revision),
            "--output-json", str(self.output_json),
        )

    def shell(self) -> str:
        parts = ["PYTHONPATH=.", "PYTHONUNBUFFERED=1", "python3", RUNNER_PATH]
        return " \\\n  ".join(parts + list(self.argv()))

def build_siam_nb4_screen_run(
    *,
    seed_path: str,
    arm: str,
    output_json: str,
    code_revision: str,
    activation_cut: float = SIAM_NB4_INITIAL_ACTIVATION_CUT,
    search_profile_id: str | None = None,
) -> SIAMNB4RunCommand:
    """Build one ``0<=t<=1`` screen rung; the common initial cut is 1e-3."""

    if arm not in SIAM_NB4_ARMS:
        raise KeyError(
            f"unknown SIAM Nb=4 arm {arm!r}; known: {sorted(SIAM_NB4_ARMS)}"
        )
    return SIAMNB4RunCommand(
        seed_path=str(seed_path),
        arm=SIAM_NB4_ARMS[arm],
        segment=SIAM_NB4_SCREEN_SEGMENT,
        output_json=str(output_json),
        code_revision=str(code_revision),
        activation_cut=float(activation_cut),
        search_profile_id=(
            None if search_profile_id is None else str(search_profile_id)
        ),
    )

def build_siam_nb4_continuation_run(
    *,
    seed_path: str,
    arm: str,
    resume_from_run_json: str,
    activation_cut: float,
    output_json: str,
    code_revision: str,
    search_profile_id: str | None = None,
) -> SIAMNB4RunCommand:
    """Build one exact ``1<=t<=5`` continuation at a method-owned cut."""

    if arm not in SIAM_NB4_ARMS:
        raise KeyError(
            f"unknown SIAM Nb=4 arm {arm!r}; known: {sorted(SIAM_NB4_ARMS)}"
        )
    return SIAMNB4RunCommand(
        seed_path=str(seed_path),
        arm=SIAM_NB4_ARMS[arm],
        segment=SIAM_NB4_CONTINUATION_SEGMENT,
        output_json=str(output_json),
        code_revision=str(code_revision),
        activation_cut=float(activation_cut),
        search_profile_id=(
            None if search_profile_id is None else str(search_profile_id)
        ),
        resume_from_run_json=str(resume_from_run_json),
    )

__all__ = [
    "ARMS", "AVQDS", "AVQDS_PUBLISHED", "APPEND_ONLY", "DELTA_THETA_CONTROL",
    "AVQDS_PUBLISHED_NUMERICS", "CAP_SENSITIVITY_2X_SEARCH",
    "COMPOSED_STATE_PARAMETER_CONTROL", "DIAGNOSTIC_FAST_SEARCH",
    "DRIVES", "EULER_NUMERICS", "EULER_RIDGE1E6_NUMERICS", "EXCHANGE",
    "NUMERICS", "Numerics",
    "RK4_NUMERICS", "SHARED_NUMERICS", "STATE_MOTION_CONTROL",
    "STEP_CONTROLS", "StepControl",
    "GATES", "HH_SNAKE_NPH1", "HORIZONS", "REGIMES", "Regime",
    "SeedNotBuiltError", "available_regimes", "resolve_regime", "MCLACHLAN_L2_GATE",
    "LOCKED_STRUCTURAL_POLICY", "PAPER_DECLARED_SEARCH", "RESIDUAL_GATE",
    "SEARCH_PROFILES", "SearchProfile", "Arm", "Drive", "Horizon",
    "InsertionGate", "RunCommand", "build_run", "main", "paper_ii_runs",
    "SIAM_NB2_PROOF_ARMS", "SIAM_NB2_PROOF_HORIZON",
    "SIAM_NB2_PROOF_SEARCH",
    "SIAMProofRunCommand", "build_siam_nb2_proof_run",
    "SIAM_NB4_ARMS", "SIAM_NB4_CONTINUATION_SEGMENT",
    "SIAM_NB4_CONTINUATION_TIMES", "SIAM_NB4_FULL_TIMES",
    "SIAM_NB4_INITIAL_ACTIVATION_CUT", "SIAM_NB4_PILOT_SEARCH",
    "SIAM_NB4_PILOT_EXCHANGE_SEARCH", "SIAM_NB4_SEARCH_PROFILES",
    "SIAM_NB4_SCREEN_SEGMENT", "SIAM_NB4_SCREEN_TIMES",
    "SIAM_NB4_TIME_STEP", "SIAMNB4RunCommand", "SIAMNB4Segment",
    "build_siam_nb4_continuation_run", "build_siam_nb4_screen_run",
]


if __name__ == "__main__":
    raise SystemExit(main())
