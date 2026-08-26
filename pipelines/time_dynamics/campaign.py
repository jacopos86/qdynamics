"""Fail-closed Paper-II campaign preparation.

The public interface is deliberately small: describe a factorial campaign in
scientific terms, then call :func:`prepare_chtc_campaign`. Every cell is
resolved through :mod:`pipelines.time_dynamics.paper_ii_runs`; this module
never restates runner flags or scientific defaults.

The campaign factors are

``algorithmic method x time-step controller x drive x activation threshold``.

Exact reference data may inform the offline threshold worklist and the
post-fetch matched-accuracy report. It is never passed to a trajectory
controller.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from pipelines.time_dynamics import paper_ii_runs as _runs

CAMPAIGN_SCHEMA_V2 = "paper_ii_factorial_campaign_v2"
CAMPAIGN_SCHEMA_V1 = CAMPAIGN_SCHEMA_V2  # compatibility import; v1 is retired.
PREPARE_AUDIT_SCHEMA_V1 = "paper_ii_campaign_prepare_audit_v1"
THRESHOLD_PLAN_SCHEMA_V1 = "paper_ii_prior_informed_threshold_plan_v1"
RESULT_AUDIT_SCHEMA_V1 = "paper_ii_campaign_result_audit_v1"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cut_label(value: float) -> str:
    return f"{float(value):.1e}".replace("+", "").replace(".", "p")


def _canonical_cut(value: float) -> float:
    """Collapse arithmetic roundoff onto one scientific-ladder value."""

    return float(f"{float(value):.12e}")


def _configuration_id(method_id: str, controller_id: str, drive_id: str) -> str:
    return f"{method_id}__{controller_id}__{drive_id}"


@dataclass(frozen=True)
class SeedSpec:
    """One explicit static-ADAPT starting artifact."""

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

    @property
    def path(self) -> Path:
        return Path(self.artifact_json)

    def sha256(self) -> str:
        if not self.path.is_file():
            raise FileNotFoundError(f"missing campaign seed: {self.path}")
        return _sha256_file(self.path)


@dataclass(frozen=True)
class CompileProfile:
    """Locked terminal-circuit transpilation convention."""

    profile_id: str = "paper_i_fake_marrakesh_v1"
    backend_name: str = "FakeMarrakesh"
    optimization_level: int = 1
    seed_transpiler: int = 7
    native_basis: tuple[str, ...] = ("sx", "rz", "x", "cz", "id")
    source: str = "MATH/paper_details/Paper_I.tex"
    note: str = (
        "Paper-I terminal-resource profile: the full ansatz including "
        "reference-state preparation is transpiled to FakeMarrakesh."
    )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "backend_name": self.backend_name,
            "optimization_level": int(self.optimization_level),
            "seed_transpiler": int(self.seed_transpiler),
            "native_basis": list(self.native_basis),
            "source": self.source,
            "note": self.note,
            "reported_metrics": ["N2q", "D2q", "Dc"],
        }


PAPER_I_FAKE_MARRAKESH = CompileProfile()


@dataclass(frozen=True)
class ThresholdPlan:
    """Prior-informed activation cuts for each run configuration."""

    thresholds_by_configuration: Mapping[str, tuple[float, ...]]
    target_mean_abs_energy_error: float
    prior_root: str | None
    prior_sources: tuple[Mapping[str, str], ...] = ()
    rationale_by_drive: Mapping[str, str] = field(default_factory=dict)
    planner: str = "prior_target_anchors_plus_one_neighbor_each_side_v1"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": THRESHOLD_PLAN_SCHEMA_V1,
            "planner": self.planner,
            "target_mean_abs_energy_error": float(
                self.target_mean_abs_energy_error
            ),
            "target_semantics": (
                "offline matched-accuracy benchmark target; not an online "
                "controller input"
            ),
            "prior_root": self.prior_root,
            "prior_sources": [dict(row) for row in self.prior_sources],
            "rationale_by_drive": dict(self.rationale_by_drive),
            "thresholds_by_configuration": {
                key: [float(value) for value in values]
                for key, values in sorted(self.thresholds_by_configuration.items())
            },
        }


def _next_tighter_cut(cut: float) -> float:
    if not math.isfinite(float(cut)) or float(cut) <= 0.0:
        raise ValueError("cut must be finite and positive")
    exponent = int(math.floor(math.log10(float(cut))))
    mantissa = float(cut) / (10.0**exponent)
    if mantissa >= 2.0:
        return 1.0 * (10.0**exponent)
    return 3.0 * (10.0 ** (exponent - 1))


def _next_looser_cut(cut: float) -> float:
    if not math.isfinite(float(cut)) or float(cut) <= 0.0:
        raise ValueError("cut must be finite and positive")
    exponent = int(math.floor(math.log10(float(cut))))
    mantissa = float(cut) / (10.0**exponent)
    if mantissa < 2.0:
        return 3.0 * (10.0**exponent)
    return 1.0 * (10.0 ** (exponent + 1))


DEFAULT_THRESHOLD_LADDER = (
    1.0e-2,
    3.0e-3,
    1.0e-3,
    3.0e-4,
    1.0e-4,
    3.0e-5,
    1.0e-5,
    3.0e-6,
)


def _prior_source_rows(root: Path) -> tuple[Mapping[str, str], ...]:
    rows: list[Mapping[str, str]] = []
    for path in sorted(root.glob("*/run.json")):
        rows.append({"path": str(path), "sha256": _sha256_file(path)})
    return tuple(rows)


def plan_thresholds_from_prior(
    *,
    prior_root: str | Path,
    method_ids: Sequence[str],
    controller_ids: Sequence[str],
    drive_ids: Sequence[str],
    target_mean_abs_energy_error: float = 1.0e-4,
    fallback_ladder: Sequence[float] = DEFAULT_THRESHOLD_LADDER,
) -> ThresholdPlan:
    """Use old trajectories as a planning prior without promoting them.

    For every old arm, the resource-cheapest target-hitting cut is an anchor.
    If an arm has not hit the target, the next tighter cut is an anchor. Each
    anchor is padded by one adjacent 1--3 logarithmic rung in both directions.
    Drives with no prior evidence receive the complete fallback ladder.

    The resulting cuts are copied across all new method/controller pairs: old
    data narrows the search region but never substitutes for a factorial cell.
    """

    root = Path(prior_root)
    if not root.is_dir():
        raise FileNotFoundError(f"prior frontier root does not exist: {root}")
    if float(target_mean_abs_energy_error) <= 0.0:
        raise ValueError("target_mean_abs_energy_error must be positive")

    from pipelines.time_dynamics.accuracy_target_report import collect

    data = collect(root)
    drive_cuts: dict[str, tuple[float, ...]] = {}
    rationale: dict[str, str] = {}
    for drive_id in drive_ids:
        anchors: set[float] = set()
        prior_arms: list[str] = []
        for (prior_drive, prior_arm), series in sorted(data.items()):
            if prior_drive != drive_id or not series:
                continue
            prior_arms.append(prior_arm)
            meeting = [row for row in series if row[1] <= target_mean_abs_energy_error]
            if meeting:
                picked = min(meeting, key=lambda row: (row[2], row[1]))
                anchors.add(float(picked[0]))
            else:
                anchors.add(_next_tighter_cut(min(float(row[0]) for row in series)))
        if not anchors:
            cuts = tuple(float(value) for value in fallback_ladder)
            rationale[drive_id] = "no prior drive data; use complete fallback ladder"
        else:
            padded: set[float] = set()
            for anchor in anchors:
                padded.update(
                    _canonical_cut(value)
                    for value in (
                        anchor,
                        _next_looser_cut(anchor),
                        _next_tighter_cut(anchor),
                    )
                )
            cuts = tuple(sorted(padded, reverse=True))
            rationale[drive_id] = (
                f"prior arms {','.join(sorted(prior_arms))}; anchors "
                + ",".join(f"{value:.1e}" for value in sorted(anchors, reverse=True))
                + "; one neighboring rung on each side"
            )

        drive_cuts[drive_id] = cuts

    by_configuration = {
        _configuration_id(method_id, controller_id, drive_id): drive_cuts[drive_id]
        for method_id, controller_id, drive_id in itertools.product(
            method_ids, controller_ids, drive_ids
        )
    }
    return ThresholdPlan(
        thresholds_by_configuration=by_configuration,
        target_mean_abs_energy_error=float(target_mean_abs_energy_error),
        prior_root=str(root),
        prior_sources=_prior_source_rows(root),
        rationale_by_drive=rationale,
    )


def uniform_threshold_plan(
    *,
    method_ids: Sequence[str],
    controller_ids: Sequence[str],
    drive_ids: Sequence[str],
    thresholds: Sequence[float],
    target_mean_abs_energy_error: float = 1.0e-4,
    rationale: str = "explicit uniform ladder",
) -> ThresholdPlan:
    values = tuple(_canonical_cut(value) for value in thresholds)
    return ThresholdPlan(
        thresholds_by_configuration={
            _configuration_id(method_id, controller_id, drive_id): values
            for method_id, controller_id, drive_id in itertools.product(
                method_ids, controller_ids, drive_ids
            )
        },
        target_mean_abs_energy_error=float(target_mean_abs_energy_error),
        prior_root=None,
        rationale_by_drive={drive_id: rationale for drive_id in drive_ids},
        planner="explicit_uniform_ladder_v1",
    )


@dataclass(frozen=True)
class CampaignSpec:
    """The complete scientific declaration of a Paper-II run matrix."""

    campaign_id: str
    seeds: tuple[SeedSpec, ...]
    method_ids: tuple[str, ...]
    controller_ids: tuple[str, ...]
    drive_ids: tuple[str, ...]
    horizon_ids: tuple[str, ...]
    threshold_plan: ThresholdPlan
    numerics_id: str = _runs.EULER_RIDGE1E6_NUMERICS.numerics_id
    gate_id: str = _runs.MCLACHLAN_L2_GATE.gate_id
    output_root: str = "raw_outputs"
    compile_profile: CompileProfile = PAPER_I_FAKE_MARRAKESH

    def __post_init__(self) -> None:
        if not self.campaign_id.strip():
            raise ValueError("campaign_id must be non-empty")
        if not self.seeds:
            raise ValueError("campaign requires at least one seed")
        for name, values, registry in (
            ("method", self.method_ids, _runs.ARMS),
            ("controller", self.controller_ids, _runs.STEP_CONTROLS),
            ("drive", self.drive_ids, _runs.DRIVES),
            ("horizon", self.horizon_ids, _runs.HORIZONS),
        ):
            if not values:
                raise ValueError(f"campaign requires at least one {name}")
            unknown = sorted(set(values) - set(registry))
            if unknown:
                raise KeyError(f"unknown {name} ids: {unknown}")
        if self.numerics_id not in _runs.NUMERICS:
            raise KeyError(f"unknown numerics id: {self.numerics_id}")
        if self.gate_id not in _runs.GATES:
            raise KeyError(f"unknown gate id: {self.gate_id}")
        expected = {
            _configuration_id(method_id, controller_id, drive_id)
            for method_id, controller_id, drive_id in itertools.product(
                self.method_ids, self.controller_ids, self.drive_ids
            )
        }
        actual = set(self.threshold_plan.thresholds_by_configuration)
        if actual != expected:
            raise ValueError(
                "threshold plan configuration mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        for config_id, cuts in self.threshold_plan.thresholds_by_configuration.items():
            if not cuts:
                raise ValueError(f"configuration {config_id!r} has no thresholds")
            if any(not math.isfinite(float(cut)) or float(cut) <= 0.0 for cut in cuts):
                raise ValueError(f"configuration {config_id!r} has invalid thresholds")
            if len(set(float(cut) for cut in cuts)) != len(cuts):
                raise ValueError(f"configuration {config_id!r} repeats a threshold")

    def cells(self) -> Iterator["CampaignCell"]:
        for seed, horizon_id, method_id, controller_id, drive_id in itertools.product(
            self.seeds,
            self.horizon_ids,
            self.method_ids,
            self.controller_ids,
            self.drive_ids,
        ):
            config_id = _configuration_id(method_id, controller_id, drive_id)
            for threshold in self.threshold_plan.thresholds_by_configuration[config_id]:
                yield CampaignCell(
                    campaign=self,
                    seed=seed,
                    method_id=method_id,
                    controller_id=controller_id,
                    drive_id=drive_id,
                    horizon_id=horizon_id,
                    activation_cut=float(threshold),
                )

    def cell_count(self) -> int:
        return sum(1 for _ in self.cells())


@dataclass(frozen=True)
class CampaignCell:
    campaign: CampaignSpec
    seed: SeedSpec
    method_id: str
    controller_id: str
    drive_id: str
    horizon_id: str
    activation_cut: float

    @property
    def configuration_id(self) -> str:
        return _configuration_id(self.method_id, self.controller_id, self.drive_id)

    @property
    def cell_id(self) -> str:
        return "__".join(
            (
                self.seed.seed_id,
                self.method_id,
                self.controller_id,
                self.drive_id,
                self.horizon_id,
                f"cut{_cut_label(self.activation_cut)}",
            )
        )

    @property
    def output_dir(self) -> str:
        return str(
            Path(self.campaign.output_root)
            / self.campaign.campaign_id
            / self.cell_id
        )

    def run_command(self, *, seed_path: str | None = None) -> _runs.RunCommand:
        return _runs.build_run(
            seed_path=self.seed.artifact_json if seed_path is None else seed_path,
            arm=self.method_id,
            gate=self.campaign.gate_id,
            drive=self.drive_id,
            horizon=self.horizon_id,
            numerics=self.campaign.numerics_id,
            step_control=self.controller_id,
            activation_cut=self.activation_cut,
            output_json=str(Path(self.output_dir) / "run.json"),
        )

    def runner_argv(self, *, seed_path: str | None = None) -> list[str]:
        return list(self.run_command(seed_path=seed_path).argv())

    def provenance(self, *, seed_path: str | None = None) -> dict[str, Any]:
        run = self.run_command(seed_path=seed_path)
        return {
            "schema": CAMPAIGN_SCHEMA_V2,
            "campaign_id": self.campaign.campaign_id,
            "cell_id": self.cell_id,
            "configuration_id": self.configuration_id,
            "seed": {
                "seed_id": self.seed.seed_id,
                "artifact_json": self.seed.artifact_json,
                "packaged_artifact_json": seed_path,
                "family_key": self.seed.family_key,
                "n_ph_max": int(self.seed.n_ph_max),
                "regime": self.seed.regime,
                "sha256": self.seed.sha256(),
            },
            "factors": {
                "algorithmic_method": self.method_id,
                "time_step_controller": self.controller_id,
                "drive": self.drive_id,
                "horizon": self.horizon_id,
                "activation_cut": float(self.activation_cut),
                "numerics": self.campaign.numerics_id,
            },
            "run_id": run.run_id,
            "runner_module": _runs.RUNNER_MODULE,
            "runner_argv": list(run.argv()),
            "expected_output_json": str(Path(self.output_dir) / "run.json"),
        }


@dataclass(frozen=True)
class PreparedCampaign:
    package_dir: Path
    manifest: Path
    audit: Path
    cell_ids: Path
    cells_jsonl: Path
    run_cell: Path
    wrapper: Path
    submit: Path

    def to_json_dict(self) -> dict[str, str]:
        return {
            key: str(value)
            for key, value in (
                ("package_dir", self.package_dir),
                ("manifest", self.manifest),
                ("audit", self.audit),
                ("cell_ids", self.cell_ids),
                ("cells_jsonl", self.cells_jsonl),
                ("run_cell", self.run_cell),
                ("wrapper", self.wrapper),
                ("submit", self.submit),
            )
        }


def _validate_resolved_cells(cells: Sequence[CampaignCell]) -> dict[str, Any]:
    from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
        _build_parser,
    )

    parser = _build_parser()
    seen: set[str] = set()
    config_counts: dict[str, int] = {}
    errors: list[str] = []
    for cell in cells:
        if cell.cell_id in seen:
            errors.append(f"duplicate cell id: {cell.cell_id}")
        seen.add(cell.cell_id)
        config_counts[cell.configuration_id] = config_counts.get(cell.configuration_id, 0) + 1
        try:
            parsed = parser.parse_args(cell.runner_argv())
        except SystemExit as exc:
            errors.append(f"{cell.cell_id}: runner parser exited {exc.code}")
            continue
        expected_cut = float(cell.activation_cut)
        actual_cut = (
            float(parsed.avqds_l2_cut)
            if cell.method_id == "avqds"
            else float(parsed.insertion_l2_cut)
        )
        if not math.isclose(actual_cut, expected_cut, rel_tol=1.0e-12, abs_tol=0.0):
            errors.append(
                f"{cell.cell_id}: activation cut {actual_cut} != {expected_cut}"
            )
    if errors:
        raise ValueError("campaign preparation failed:\n" + "\n".join(errors))
    return {
        "schema": PREPARE_AUDIT_SCHEMA_V1,
        "status": "PASS",
        "cell_count": len(cells),
        "unique_cell_count": len(seen),
        "configuration_counts": dict(sorted(config_counts.items())),
        "checks": [
            "every seed exists and is sha256-bound",
            "every cell resolves through paper_ii_runs.build_run",
            "every resolved argv parses with the live runner parser",
            "every method receives its own activation-cut flag",
            "all cell ids are unique",
        ],
    }


def prepare_chtc_campaign(
    spec: CampaignSpec,
    package_dir: str | Path,
    *,
    image_path: str = "chtc/time_dynamics_optuna/image.sif",
    request_cpus: int = 2,
    request_memory_gb: int = 16,
    max_runtime_seconds: int = 86400,
) -> PreparedCampaign:
    """Prepare and validate a CHTC package without submitting it."""

    out = Path(package_dir)
    input_dir = out / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    cells = list(spec.cells())
    audit_payload = _validate_resolved_cells(cells)

    seed_dir = input_dir / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    packaged: dict[str, str] = {}
    for seed in spec.seeds:
        digest = seed.sha256()  # fail closed before writing a runnable package
        dst = seed_dir / f"{seed.seed_id}.json"
        if dst.exists() and _sha256_file(dst) != digest:
            raise ValueError(f"packaged seed collision at {dst}")
        if not dst.exists():
            dst.write_bytes(seed.path.read_bytes())
        packaged[seed.seed_id] = dst.as_posix()

    manifest_cells = [
        cell.provenance(seed_path=packaged[cell.seed.seed_id]) for cell in cells
    ]
    manifest_payload = {
        "schema": CAMPAIGN_SCHEMA_V2,
        "campaign_id": spec.campaign_id,
        "status": "PREPARED_NOT_SUBMITTED",
        "cell_count": len(cells),
        "scientific_design": {
            "algorithmic_methods": list(spec.method_ids),
            "time_step_controllers": list(spec.controller_ids),
            "drives": list(spec.drive_ids),
            "horizons": list(spec.horizon_ids),
            "numerics": spec.numerics_id,
            "gate": spec.gate_id,
            "factorial_relation": (
                "two algorithmic methods x three time-step controllers; "
                "threshold is an offline per-cell accuracy-search variable"
            ),
        },
        "threshold_plan": spec.threshold_plan.to_json_dict(),
        "compile_profile": spec.compile_profile.to_json_dict(),
        "cells": manifest_cells,
    }
    manifest = input_dir / "manifest.json"
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    cell_ids = input_dir / "cell_ids.txt"
    cell_ids.write_text(
        "\n".join(cell.cell_id for cell in cells) + "\n", encoding="utf-8"
    )
    cells_jsonl = input_dir / "cells.jsonl"
    cells_jsonl.write_text(
        "".join(
            json.dumps(
                {
                    "cell_id": cell.cell_id,
                    "runner_module": _runs.RUNNER_MODULE,
                    "runner_argv": cell.runner_argv(
                        seed_path=packaged[cell.seed.seed_id]
                    ),
                    "trajectory_json": str(Path(cell.output_dir) / "run.json"),
                    "compile_profile": spec.compile_profile.to_json_dict(),
                },
                sort_keys=True,
            )
            + "\n"
            for cell in cells
        ),
        encoding="utf-8",
    )

    run_cell = out / "run_cell.py"
    run_cell.write_text(
        "#!/usr/bin/env python3\n"
        "import json, subprocess, sys\n"
        "from pathlib import Path\n"
        "cell_id = sys.argv[1] if len(sys.argv) == 2 else None\n"
        "if not cell_id: raise SystemExit('usage: run_cell.py CELL_ID')\n"
        f"rows = Path({str(cells_jsonl)!r}).read_text(encoding='utf-8').splitlines()\n"
        "records = [json.loads(row) for row in rows]\n"
        "record = next((row for row in records if row['cell_id'] == cell_id), None)\n"
        "if record is None: raise SystemExit(f'unknown cell_id {cell_id}')\n"
        "command = [sys.executable, '-m', record['runner_module'], *record['runner_argv']]\n"
        "subprocess.run(command, check=True)\n"
        "trajectory = Path(record['trajectory_json'])\n"
        "profile = record['compile_profile']\n"
        "cost_command = [\n"
        "    sys.executable, '-m',\n"
        "    'pipelines.time_dynamics.diagnostics.ap_terminal_qiskit_cost',\n"
        "    '--trajectory-json', str(trajectory),\n"
        "    '--output-cost-table-json', str(trajectory.with_name('terminal_qiskit_cost.json')),\n"
        "    '--label', cell_id,\n"
        "    '--backend-name', str(profile['backend_name']),\n"
        "    '--seed-transpiler', str(profile['seed_transpiler']),\n"
        "    '--optimization-level', str(profile['optimization_level']),\n"
        "]\n"
        "subprocess.run(cost_command, check=True)\n",
        encoding="utf-8",
    )
    run_cell.chmod(0o755)

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
        f"  bash -lc 'cd /work && python3 {shlex.quote(str(run_cell))} \"$@\"' -- \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    tag = spec.campaign_id.replace("_", "-")
    submit = out / f"submit_{spec.campaign_id}.sub"
    submit.write_text(
        "universe = vanilla\n"
        f"executable = {wrapper}\n"
        "arguments = $(cell_id)\n"
        "should_transfer_files = YES\n"
        "when_to_transfer_output = ON_EXIT\n"
        "transfer_executable = True\n"
        "preserve_relative_paths = True\n"
        f"transfer_input_files = pipelines, src, {out}, {image_path}\n"
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
        f"queue cell_id from {cell_ids}\n",
        encoding="utf-8",
    )

    audit_payload.update(
        {
            "campaign_id": spec.campaign_id,
            "manifest": str(manifest),
            "manifest_sha256": _sha256_file(manifest),
            "submission_status": "NOT_SUBMITTED",
            "submit_command": f"condor_submit {submit}",
        }
    )
    audit = input_dir / "prepare_audit.json"
    audit.write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PreparedCampaign(
        package_dir=out,
        manifest=manifest,
        audit=audit,
        cell_ids=cell_ids,
        cells_jsonl=cells_jsonl,
        run_cell=run_cell,
        wrapper=wrapper,
        submit=submit,
    )


def audit_completed_campaign(
    manifest_path: str | Path,
    *,
    audit_path: str | Path | None = None,
    require_qiskit_cost: bool = True,
    attach_exact_energy_error: bool = True,
) -> Path:
    """Fail closed unless every declared result satisfies the campaign lock.

    Exact propagation is attached only here, after the trajectories exist. It
    is never passed to the online controller. Terminal Qiskit costs must use
    the compile profile embedded in the manifest.
    """

    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if str(manifest.get("schema")) != CAMPAIGN_SCHEMA_V2:
        raise ValueError(f"unsupported campaign manifest schema: {manifest.get('schema')!r}")
    cells = list(manifest.get("cells") or ())
    if len(cells) != int(manifest.get("cell_count", -1)):
        raise ValueError("manifest cell_count does not match cells")
    compile_profile = dict(manifest.get("compile_profile") or {})
    target = float(
        dict(manifest.get("threshold_plan") or {}).get(
            "target_mean_abs_energy_error", 1.0e-4
        )
    )

    from pipelines.time_dynamics.run_lock import assert_comparable

    exact_rows = None
    if attach_exact_energy_error:
        sys.path.insert(0, str(_REPO_ROOT / "output"))
        from exact_driven_reference import exact_rows as _exact_rows

        exact_rows = _exact_rows

    errors: list[str] = []
    locks_by_physics_cell: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    reference_cache: dict[str, list[dict[str, Any]]] = {}
    result_rows: list[dict[str, Any]] = []
    for declared in cells:
        cell_id = str(declared.get("cell_id", ""))
        factors = dict(declared.get("factors") or {})
        run_path = Path(str(declared.get("expected_output_json", "")))
        if not run_path.is_absolute():
            run_path = _REPO_ROOT / run_path
        if not run_path.is_file():
            errors.append(f"{cell_id}: missing output {run_path}")
            continue
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{cell_id}: unreadable output: {exc}")
            continue
        lock = run.get("run_lock")
        if not isinstance(lock, Mapping):
            errors.append(f"{cell_id}: missing run_lock")
            continue
        physics = dict(lock.get("physics") or {})
        numerics = dict(lock.get("numerics") or {})
        inverse = dict(numerics.get("inverse_policy") or {})
        repair = dict(numerics.get("solve_repair") or {})
        summary = dict(run.get("summary") or {})
        support = dict(summary.get("support_patch_config") or {})
        method_id = str(factors.get("algorithmic_method"))
        controller_id = str(factors.get("time_step_controller"))
        declared_cut = float(factors.get("activation_cut"))
        actual_cut = float(
            support.get("avqds_l2_cut")
            if method_id == "avqds"
            else support.get("insertion_l2_cut")
        )
        checks = {
            "method": str((lock.get("policy") or {}).get("structural_policy")) == method_id,
            "integrator": str(numerics.get("integrator_method")) == "euler",
            "ridge": math.isclose(float(inverse.get("ridge_lambda")), 1.0e-6),
            "activation_cut": math.isclose(
                actual_cut, declared_cut, rel_tol=1.0e-12, abs_tol=0.0
            ),
            "reference_not_used_for_decision": not any(
                bool(dict(run.get("decision_data_flow") or {}).get(key))
                for key in (
                    "uses_reference_for_decision",
                    "uses_exact_reference_for_decision",
                    "uses_future_exact_forecast_for_decision",
                )
            ),
            "accepted_step_telemetry": int(
                summary.get("accepted_internal_substep_count") or 0
            )
            > 0,
            "rhs_telemetry": int(summary.get("rhs_evaluation_count") or 0) > 0,
            "step_attempt_telemetry": bool(
                int(summary.get("attempted_internal_step_count") or 0)
                >= int(summary.get("accepted_internal_substep_count") or 0)
                and int(summary.get("rejected_internal_step_count") or 0)
                == int(summary.get("attempted_internal_step_count") or 0)
                - int(summary.get("accepted_internal_substep_count") or 0)
            ),
        }
        if controller_id == "delta_theta_5e-3":
            checks["controller"] = bool(
                support.get("avqds_delta_theta_max") == 5.0e-3
                and not bool(repair.get("enabled"))
                and int(summary.get("parameter_controlled_interval_count") or 0) > 0
            )
        elif controller_id == "state_motion_1e-2":
            checks["controller"] = bool(
                support.get("avqds_delta_theta_max") is None
                and bool(repair.get("enabled"))
                and repair.get("parameter_step_max") is None
                and repair.get("state_motion_l2_step_max") == 1.0e-2
            )
        elif controller_id == "state_motion_1e-2_plus_parameter_5e-3":
            checks["controller"] = bool(
                support.get("avqds_delta_theta_max") is None
                and bool(repair.get("enabled"))
                and repair.get("parameter_step_max") == 5.0e-3
                and repair.get("state_motion_l2_step_max") == 1.0e-2
            )
        else:
            checks["controller"] = False

        cost_path = run_path.with_name("terminal_qiskit_cost.json")
        cost_metrics: dict[str, Any] | None = None
        if require_qiskit_cost:
            if not cost_path.is_file():
                checks["qiskit_compile"] = False
            else:
                cost = json.loads(cost_path.read_text(encoding="utf-8"))
                defaults = dict(cost.get("compile_defaults") or {})
                rows = list(cost.get("rows") or ())
                cost_row = dict(rows[0]) if len(rows) == 1 else {}
                parity = dict(cost_row.get("terminal_reconstruction_parity") or {})
                checks["qiskit_compile"] = bool(
                    defaults.get("backend_name") == compile_profile.get("backend_name")
                    and int(defaults.get("optimization_level", -1))
                    == int(compile_profile.get("optimization_level", -2))
                    and int(defaults.get("seed_transpiler", -1))
                    == int(compile_profile.get("seed_transpiler", -2))
                    and parity.get("passed") is True
                    and cost_row.get("trajectory_json_sha256")
                    == _sha256_file(run_path)
                )
                cost_metrics = {
                    key: int(cost_row[key]) for key in ("N2q", "D2q", "Dc")
                }

        mean_error = max_error = final_error = None
        if exact_rows is not None:
            fingerprint = str(lock.get("physics_fingerprint", ""))
            if fingerprint not in reference_cache:
                reference_cache[fingerprint] = exact_rows(
                    str(run_path if not run.get("source_artifact_json") else run["source_artifact_json"]),
                    str(run_path),
                )
            reference = reference_cache[fingerprint]
            plot_rows = list(run.get("plot_rows") or ())
            if len(reference) != len(plot_rows):
                checks["exact_reporting_overlay"] = False
            else:
                deviations = [
                    abs(float(row["energy_expectation"]) - float(ref["energy"]))
                    for row, ref in zip(plot_rows, reference)
                ]
                mean_error = float(sum(deviations) / len(deviations))
                max_error = float(max(deviations))
                final_error = float(deviations[-1])
                checks["exact_reporting_overlay"] = True

        for name, passed in checks.items():
            if not passed:
                errors.append(f"{cell_id}: failed {name}")
        physics_key = (
            str(declared.get("seed", {}).get("seed_id")),
            str(factors.get("drive")),
            str(factors.get("horizon")),
        )
        locks_by_physics_cell.setdefault(physics_key, []).append(lock)
        result_rows.append(
            {
                "cell_id": cell_id,
                "factors": factors,
                "checks": checks,
                "mean_abs_energy_error": mean_error,
                "max_abs_energy_error": max_error,
                "final_abs_energy_error": final_error,
                "target_mean_abs_energy_error": target,
                "target_reached": None if mean_error is None else mean_error <= target,
                "runtime_parameter_count_final": int(
                    summary.get("runtime_parameter_count_final") or 0
                ),
                "accepted_internal_substep_count": int(
                    summary.get("accepted_internal_substep_count") or 0
                ),
                "rhs_evaluation_count": int(summary.get("rhs_evaluation_count") or 0),
                "attempted_internal_step_count": int(
                    summary.get("attempted_internal_step_count") or 0
                ),
                "rejected_internal_step_count": int(
                    summary.get("rejected_internal_step_count") or 0
                ),
                "qiskit_terminal_cost": cost_metrics,
            }
        )

    for key, locks in locks_by_physics_cell.items():
        try:
            assert_comparable(locks)
        except ValueError as exc:
            errors.append(f"{key}: incomparable run locks: {exc}")

    payload = {
        "schema": RESULT_AUDIT_SCHEMA_V1,
        "campaign_id": manifest.get("campaign_id"),
        "manifest": str(manifest_file),
        "manifest_sha256": _sha256_file(manifest_file),
        "status": "PASS" if not errors else "FAIL",
        "submission_recommendation": "READY_FOR_PRODUCTION_PREPARATION" if not errors else "STOP",
        "cell_count_declared": len(cells),
        "cell_count_audited": len(result_rows),
        "errors": errors,
        "results": result_rows,
    }
    output = (
        Path(audit_path)
        if audit_path is not None
        else manifest_file.with_name("result_audit.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise ValueError("campaign result audit failed:\n" + "\n".join(errors))
    return output


def write_campaign_manifest(spec: CampaignSpec, path: str | Path) -> Path:
    """Write a non-runnable manifest; package preparation is preferred."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CAMPAIGN_SCHEMA_V2,
        "campaign_id": spec.campaign_id,
        "status": "DECLARED_NOT_PREPARED",
        "cell_count": spec.cell_count(),
        "threshold_plan": spec.threshold_plan.to_json_dict(),
        "compile_profile": spec.compile_profile.to_json_dict(),
        "cells": [cell.provenance() for cell in spec.cells()],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


# Old name retained as a compatibility adapter for existing campaign scripts.
write_chtc_package = prepare_chtc_campaign


__all__ = [
    "CAMPAIGN_SCHEMA_V1",
    "CAMPAIGN_SCHEMA_V2",
    "DEFAULT_THRESHOLD_LADDER",
    "PAPER_I_FAKE_MARRAKESH",
    "CampaignCell",
    "CampaignSpec",
    "CompileProfile",
    "PreparedCampaign",
    "SeedSpec",
    "ThresholdPlan",
    "audit_completed_campaign",
    "plan_thresholds_from_prior",
    "prepare_chtc_campaign",
    "uniform_threshold_plan",
    "write_campaign_manifest",
    "write_chtc_package",
]
