"""CHTC package adapter for the current staged-campaign action frontier."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
import shlex
from typing import Any, Iterable

from pipelines.time_dynamics import paper_ii_runs as _runs
from pipelines.time_dynamics.campaign import PAPER_II_FAKE_MARRAKESH_FIXED_PATCH
from pipelines.time_dynamics.run_lock import resolve_code_revision
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import _build_parser
from pipelines.time_dynamics.staged_campaign import (
    ACTION_RUN,
    CampaignLedger,
    CellAction,
    StagedCampaign,
)


PACKAGE_SCHEMA_V1 = "paper_ii_staged_chtc_package_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_revision(repo_root: Path, declared: str | None) -> str:
    revision = (
        resolve_code_revision(repo_root)
        if declared is None
        else str(declared).strip()
    )
    if (
        revision == "unknown"
        or revision.endswith("-dirty")
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError(
            "staged CHTC package requires one committed 40-character revision; "
            f"resolved {revision!r}"
        )
    return revision


@dataclass(frozen=True)
class PreparedStagedCampaign:
    package_dir: Path
    manifest: Path
    audit: Path
    action_ids: Path
    actions_jsonl: Path
    run_action: Path
    wrapper: Path
    submit: Path

    def to_json_dict(self) -> dict[str, str]:
        return {
            name: str(value)
            for name, value in (
                ("package_dir", self.package_dir),
                ("manifest", self.manifest),
                ("audit", self.audit),
                ("action_ids", self.action_ids),
                ("actions_jsonl", self.actions_jsonl),
                ("run_action", self.run_action),
                ("wrapper", self.wrapper),
                ("submit", self.submit),
            )
        }


def prepare_staged_chtc_campaign(
    *,
    campaign_id: str,
    campaign: StagedCampaign,
    ledger: CampaignLedger,
    package_dir: str | Path,
    seed_path: str | Path,
    numerics: str,
    gate: str,
    search_profile: str,
    repo_root: str | Path,
    image_path: str = "chtc/time_dynamics_optuna/image.sif",
    request_cpus: int = 2,
    request_memory_gb: int = 10,
    max_runtime_seconds: int = 43200,
    code_revision: str | None = None,
    in_flight_action_ids: Iterable[str] = (),
) -> PreparedStagedCampaign:
    """Prepare, validate, and provenance-bind the current action frontier."""

    current_actions = tuple(campaign.next_actions(ledger))
    if not current_actions:
        raise ValueError("staged campaign has no current actions to package")
    current_by_id = {action.action_id: action for action in current_actions}
    in_flight = tuple(sorted({str(value) for value in in_flight_action_ids}))
    unknown = tuple(value for value in in_flight if value not in current_by_id)
    if unknown:
        raise ValueError(
            "in-flight receipt names actions outside the current frontier: "
            + ", ".join(unknown)
        )
    actions = tuple(
        action for action in current_actions if action.action_id not in in_flight
    )
    if not actions:
        raise ValueError("every current staged action is already in flight")
    if any(action.action != ACTION_RUN for action in current_actions):
        raise ValueError("record safeguard actions before preparing a run package")
    frontier_stage = campaign.stage(ledger)

    root = Path(repo_root).resolve()
    out = Path(package_dir)
    input_dir = out / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    revision = _validated_revision(root, code_revision)

    source_seed = Path(seed_path)
    if not source_seed.is_file() or source_seed.stat().st_size <= 0:
        raise ValueError(f"staged campaign seed is missing or empty: {source_seed}")
    packaged_seed = input_dir / "seed.json"
    packaged_seed.write_bytes(source_seed.read_bytes())
    seed_sha = _sha256(packaged_seed)

    resume_dir = input_dir / "resume"
    action_rows: list[dict[str, Any]] = []
    parser = _build_parser()
    for action in actions:
        packaged_action = action
        resume_receipt = None
        if action.resume_from_run_path is not None:
            source_resume = Path(action.resume_from_run_path)
            if not source_resume.is_file() or source_resume.stat().st_size <= 0:
                raise ValueError(
                    f"resume source is missing or empty for {action.action_id}: "
                    f"{source_resume}"
                )
            resume_dir.mkdir(parents=True, exist_ok=True)
            packaged_resume = resume_dir / f"{action.action_id}__screen.json"
            packaged_resume.write_bytes(source_resume.read_bytes())
            packaged_action = replace(
                action,
                resume_from_run_path=str(packaged_resume),
            )
            resume_receipt = {
                "source_path": str(source_resume),
                "packaged_path": str(packaged_resume),
                "sha256": _sha256(packaged_resume),
            }

        trajectory = Path("raw_outputs") / campaign_id / action.action_id / "run.json"
        command = campaign.build_run_command(
            packaged_action,
            seed_path=str(packaged_seed),
            output_json=str(trajectory),
            numerics=numerics,
            gate=gate,
            search_profile=search_profile,
        )
        argv = [*command.argv(), "--code-revision", revision]
        parsed = parser.parse_args(argv)
        if not bool(parsed.record_statevector):
            raise ValueError(f"{action.action_id} does not record statevectors")
        if str(parsed.output_json).strip() == "":
            raise ValueError(f"{action.action_id} resolved an empty output path")
        action_rows.append(
            {
                "action_id": action.action_id,
                "stage": action.stage,
                "action": action.to_json_dict(),
                "packaged_action": packaged_action.to_json_dict(),
                "runner_module": _runs.RUNNER_MODULE,
                "runner_argv": argv,
                "trajectory_json": str(trajectory),
                "compile_terminal_cost": action.horizon == "t20",
                "resume_receipt": resume_receipt,
                "run_id": command.run_id,
            }
        )

    ledger_snapshot = input_dir / "ledger_snapshot.json"
    ledger_snapshot.write_text(
        json.dumps(ledger.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    revision_receipt = input_dir / "code_revision.txt"
    revision_receipt.write_text(revision + "\n", encoding="utf-8")
    manifest_payload = {
        "schema": PACKAGE_SCHEMA_V1,
        "status": "PREPARED_NOT_SUBMITTED",
        "campaign_id": campaign_id,
        "stage": frontier_stage,
        "current_action_count": len(current_actions),
        "in_flight_action_ids": list(in_flight),
        "action_stages": sorted({action.stage for action in actions}),
        "action_count": len(action_rows),
        "code_revision": revision,
        "seed": {"path": str(packaged_seed), "sha256": seed_sha},
        "ledger_snapshot": {
            "path": str(ledger_snapshot),
            "sha256": _sha256(ledger_snapshot),
        },
        "numerics": numerics,
        "gate": gate,
        "search_profile": search_profile,
        "compile_profile": PAPER_II_FAKE_MARRAKESH_FIXED_PATCH.to_json_dict(),
        "actions": action_rows,
    }
    manifest = input_dir / "manifest.json"
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    action_ids = input_dir / "action_ids.txt"
    action_ids.write_text(
        "\n".join(row["action_id"] for row in action_rows) + "\n",
        encoding="utf-8",
    )
    actions_jsonl = input_dir / "actions.jsonl"
    actions_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in action_rows),
        encoding="utf-8",
    )

    run_action = out / "run_action.py"
    run_action.write_text(
        "#!/usr/bin/env python3\n"
        "import json, subprocess, sys\n"
        "from pathlib import Path\n"
        "action_id = sys.argv[1] if len(sys.argv) == 2 else None\n"
        "if not action_id: raise SystemExit('usage: run_action.py ACTION_ID')\n"
        f"rows = Path({str(actions_jsonl)!r}).read_text(encoding='utf-8').splitlines()\n"
        "records = [json.loads(row) for row in rows]\n"
        "record = next((row for row in records if row['action_id'] == action_id), None)\n"
        "if record is None: raise SystemExit(f'unknown action_id {action_id}')\n"
        "command = [sys.executable, '-m', record['runner_module'], *record['runner_argv']]\n"
        "if not record['runner_argv']: raise SystemExit('empty runner argv')\n"
        "subprocess.run(command, check=True)\n"
        "trajectory = Path(record['trajectory_json'])\n"
        "if not trajectory.is_file() or trajectory.stat().st_size == 0:\n"
        "    raise SystemExit(f'runner returned without artifact: {trajectory}')\n"
        "if record['compile_terminal_cost']:\n"
        f"    profile = {PAPER_II_FAKE_MARRAKESH_FIXED_PATCH.to_json_dict()!r}\n"
        "    cost_command = [\n"
        "        sys.executable, '-m',\n"
        "        'pipelines.time_dynamics.diagnostics.ap_terminal_qiskit_cost',\n"
        "        '--trajectory-json', str(trajectory),\n"
        "        '--output-cost-table-json', str(trajectory.with_name('terminal_qiskit_cost.json')),\n"
        "        '--label', action_id,\n"
        "        '--backend-name', str(profile['backend_name']),\n"
        "        '--seed-transpiler', str(profile['seed_transpiler']),\n"
        "        '--optimization-level', str(profile['optimization_level']),\n"
        "    ]\n"
        "    subprocess.run(cost_command, check=True)\n",
        encoding="utf-8",
    )
    run_action.chmod(0o755)

    wrapper = out / "run_action_apptainer.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p raw_outputs\n"
        f'IMAGE="${{PROJECT_IMAGE:-{image_path}}}"\n'
        'if [[ ! -f "$IMAGE" ]]; then echo "Missing Apptainer image: $IMAGE" >&2; exit 2; fi\n'
        'if command -v apptainer >/dev/null 2>&1; then APPTAINER_BIN="$(command -v apptainer)";\n'
        'elif command -v singularity >/dev/null 2>&1; then APPTAINER_BIN="$(command -v singularity)";\n'
        'else echo "No apptainer/singularity on this execute node." >&2; exit 127; fi\n'
        'ROOT="$PWD"\n'
        f'"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" "$IMAGE" \\\n'
        f"  bash -lc 'cd /work && python3 {shlex.quote(str(run_action))} \"$@\"' -- \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    tag = f"{campaign_id}-{frontier_stage}".replace("_", "-")
    submit = out / f"submit_{frontier_stage}.sub"
    submit.write_text(
        "universe = vanilla\n"
        f"executable = {wrapper}\n"
        "arguments = $(action_id)\n"
        "should_transfer_files = YES\n"
        "when_to_transfer_output = ON_EXIT\n"
        "transfer_executable = True\n"
        "preserve_relative_paths = True\n"
        f"transfer_input_files = pipelines, src, {out}, {image_path}\n"
        "transfer_output_files = raw_outputs\n"
        f"log = {out}/logs/{tag}.$(Cluster).$(Process).log\n"
        f"output = {out}/logs/{tag}.$(Cluster).$(Process).out\n"
        f"error = {out}/logs/{tag}.$(Cluster).$(Process).err\n"
        "requirements = TARGET.HasSIF\n"
        f"request_cpus = {int(request_cpus)}\n"
        f"request_memory = {int(request_memory_gb)}GB\n"
        "request_disk = 40GB\n"
        f"+MaxRuntime = {int(max_runtime_seconds)}\n"
        f'+JobBatchName = "{tag}"\n'
        f"queue action_id from {action_ids}\n",
        encoding="utf-8",
    )

    audit_payload = {
        "schema": PACKAGE_SCHEMA_V1,
        "status": "PASS_NOT_SUBMITTED",
        "campaign_id": campaign_id,
        "stage": frontier_stage,
        "action_stages": sorted({action.stage for action in actions}),
        "action_count": len(action_rows),
        "code_revision": revision,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "checks": [
            "ledger replay emitted a nonempty current frontier",
            "every action resolves through paper_ii_runs.build_run",
            "every argv parses with the live runner parser",
            "every action records checkpoint statevectors",
            "every resume source is copied and sha256-bound",
            "screening actions skip terminal compilation",
            "promotion actions request terminal compilation",
        ],
        "submit_command": f"condor_submit {submit}",
    }
    audit = input_dir / "prepare_audit.json"
    audit.write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PreparedStagedCampaign(
        package_dir=out,
        manifest=manifest,
        audit=audit,
        action_ids=action_ids,
        actions_jsonl=actions_jsonl,
        run_action=run_action,
        wrapper=wrapper,
        submit=submit,
    )


__all__ = [
    "PACKAGE_SCHEMA_V1",
    "PreparedStagedCampaign",
    "prepare_staged_chtc_campaign",
]
