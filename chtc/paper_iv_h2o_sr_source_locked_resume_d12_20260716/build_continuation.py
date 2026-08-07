from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_COMMAND_SHA256 = "9cea5e5a026ce2a8fe32cc44cd0722676e8b79c1c38a3d875235fedda2f11e07"
SOURCE_ARCHIVE_SHA256 = "94c2df6df22c6d277aefdd6559273d943e3724d476ecab6648c6dd11e1fd78c6"
FIXTURE_SHA256 = "570690bd126787305b340bd2f7493499c0f3101e3e2820c2d355c55c16afa594"
CHECKPOINT_SHA256 = "bfddcb9bf5620a690b3af1a73adb412bb2fc48647290a529a05e726c183628f2"
START_DEPTH = 12
TARGET_DEPTH = 30
REMAINING_ADMISSIONS = TARGET_DEPTH - START_DEPTH
RECORD_ID = (
    "paper_iv_h2o_sr_source_locked_no_novelty_no_prune_no_beam_full_refit_"
    "resume_d12_to_d30_20260716_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing source-lock input: {path}")
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def find_option(argv: list[str], option: str) -> int:
    positions = [index for index, token in enumerate(argv) if token == option]
    if len(positions) != 1:
        raise SystemExit(f"Expected one {option}; found {len(positions)}")
    return positions[0]


def set_option(argv: list[str], option: str, value: str) -> None:
    index = find_option(argv, option)
    argv[index + 1] = str(value)


def append_option(argv: list[str], option: str, value: str) -> None:
    if option in argv:
        raise SystemExit(f"Continuation source unexpectedly contains {option}")
    argv.extend([option, str(value)])


def parse_options(argv: list[str]) -> dict[str, Any]:
    first = next(index for index, token in enumerate(argv) if token.startswith("--"))
    options: dict[str, Any] = {}
    index = first
    while index < len(argv):
        option = argv[index]
        if not option.startswith("--") or option in options:
            raise SystemExit(f"Malformed or duplicate command option: {option}")
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            options[option] = argv[index + 1]
            index += 2
        else:
            options[option] = True
            index += 1
    return options


def command_diff(source: list[str], target: list[str]) -> list[dict[str, Any]]:
    source_options = parse_options(source)
    target_options = parse_options(target)
    differences = []
    for option in sorted(set(source_options) | set(target_options)):
        if source_options.get(option) != target_options.get(option):
            differences.append(
                {
                    "option": option,
                    "source": source_options.get(option),
                    "target": target_options.get(option),
                }
            )
    expected = {
        "--adapt-current-json",
        "--adapt-estimator-call-ledger-json",
        "--adapt-max-depth",
        "--adapt-resume-scaffold-json",
        "--adapt-segment-id",
        "--adapt-segment-max-new-admissions",
        "--molecular-vibronic-h2o-linear-fd-fixture-json",
        "--output-json",
    }
    actual = {entry["option"] for entry in differences}
    if actual != expected:
        raise SystemExit(
            "Unexpected continuation command drift: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return differences


def get(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def build_payload(
    *, mode: str, argv: list[str], source_argv: list[str], record_id: str
) -> dict[str, Any]:
    return {
        "schema": "paper_iv_h2o_source_locked_continuation_command_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "record_id": record_id,
        "run_class": "diagnostic_source_locked_application_continuation",
        "runner_mode": "direct_frozen_source_resume",
        "wrapper_used": True,
        "wrapper_kind": "dedicated_chtc_apptainer_resume_runner_v1",
        "cwd": "runtime_source",
        "argv": argv,
        "source": {
            "command_sha256": SOURCE_COMMAND_SHA256,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "fixture_sha256": FIXTURE_SHA256,
        },
        "resume": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "source_kind": "live_current_json_at_iteration_done",
            "starting_depth": START_DEPTH,
            "target_total_depth": TARGET_DEPTH,
            "max_new_admissions": REMAINING_ADMISSIONS,
            "checkpoint_energy": -74.98939132227477,
            "checkpoint_abs_delta_e": 0.014772263496467986,
        },
        "settings_changed_vs_stopped_run": [
            "execution.resume_scaffold",
            "execution.output_paths",
            "execution.segment_id",
            "execution.remaining_depth_cap",
        ],
        "scientific_settings_changed_vs_stopped_run": [],
        "command_diff_vs_stopped_run": command_diff(source_argv, argv),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    stage = Path(__file__).resolve().parent
    input_dir = stage / "input"
    runtime_inputs = stage / "runtime_inputs"
    runtime_source = stage / "runtime_source"
    source_run = repo / "raw_outputs" / (
        "paper_iv_h2o_sr_source_locked_no_novelty_no_prune_no_beam_full_refit_"
        "from_zero_r30_20260716_v1"
    )
    source_command = source_run / "command.json"
    checkpoint = source_run / "current.json"
    fixture = (
        repo
        / "tmp"
        / "h2o_linear_fd_valence_psi4_optimized"
        / "h2o_linear_fd_sparse_fixture_nph1_ref2_reencoded_v2.json"
    )
    source_archive = (
        repo
        / "raw_outputs"
        / "paper_i_hh_sr_snake_weak_weak_undamped_no_prune_no_beam_no_ordinary_"
        "novelty_fallback_on_20260715"
        / "source_lock"
        / "source_tree_no_beam_ablation_v1.tar.gz"
    )
    for path, expected in (
        (source_command, SOURCE_COMMAND_SHA256),
        (checkpoint, CHECKPOINT_SHA256),
        (fixture, FIXTURE_SHA256),
        (source_archive, SOURCE_ARCHIVE_SHA256),
    ):
        require_hash(path, expected)

    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    checks = {
        "checkpoint.reason": "iteration_done",
        "adapt_vqe.ansatz_depth": START_DEPTH,
        "settings.problem": "molecular_vibronic_h2o_linear_fd",
        "settings.route_family": "singleton_response_snake",
        "settings.phase2_gram_novelty_policy": "fallback_only_v1",
        "settings.phase3_gram_novelty_policy": "fallback_only_v1",
        "adapt_vqe.accepted_refit.scope": "full_ansatz_v1",
        "adapt_vqe.accepted_refit.coordinate_chart": "supported_fs_whitened_fixed_v1",
    }
    failures = {
        dotted: {"expected": expected, "actual": get(checkpoint_payload, dotted)}
        for dotted, expected in checks.items()
        if get(checkpoint_payload, dotted) != expected
    }
    if failures:
        raise SystemExit(f"Checkpoint contract mismatch: {json.dumps(failures)}")

    input_dir.mkdir(parents=True, exist_ok=True)
    runtime_inputs.mkdir(parents=True, exist_ok=True)
    copies = {
        "source_archive": (source_archive, input_dir / "source_tree_no_beam_ablation_v1.tar.gz"),
        "fixture": (fixture, input_dir / "h2o_fixture.json"),
        "checkpoint": (checkpoint, input_dir / "h2o_depth12_current.json"),
    }
    for source, destination in copies.values():
        shutil.copy2(source, destination)
    shutil.copy2(input_dir / "h2o_fixture.json", runtime_inputs / "h2o_fixture.json")
    shutil.copy2(
        input_dir / "h2o_depth12_current.json",
        runtime_inputs / "h2o_depth12_current.json",
    )

    source_payload = json.loads(source_command.read_text(encoding="utf-8"))
    source_argv = list(source_payload["argv"])

    def make_argv(*, mode: str, record_id: str) -> list[str]:
        argv = list(source_argv)
        max_depth = 0 if mode == "preflight" else REMAINING_ADMISSIONS
        set_option(argv, "--adapt-max-depth", str(max_depth))
        set_option(
            argv,
            "--molecular-vibronic-h2o-linear-fd-fixture-json",
            "../runtime_inputs/h2o_fixture.json",
        )
        set_option(argv, "--adapt-current-json", f"../raw_outputs/{record_id}/current.json")
        set_option(
            argv,
            "--adapt-estimator-call-ledger-json",
            f"../raw_outputs/{record_id}/estimator_call_ledger.json",
        )
        set_option(argv, "--output-json", f"../raw_outputs/{record_id}/result.json")
        append_option(
            argv,
            "--adapt-resume-scaffold-json",
            "../runtime_inputs/h2o_depth12_current.json",
        )
        append_option(argv, "--adapt-segment-id", record_id)
        append_option(
            argv,
            "--adapt-segment-max-new-admissions",
            str(REMAINING_ADMISSIONS),
        )
        return argv

    preflight_id = f"{RECORD_ID}_preflight"
    full_payload = build_payload(
        mode="full",
        argv=make_argv(mode="full", record_id=RECORD_ID),
        source_argv=source_argv,
        record_id=RECORD_ID,
    )
    preflight_payload = build_payload(
        mode="preflight",
        argv=make_argv(mode="preflight", record_id=preflight_id),
        source_argv=source_argv,
        record_id=preflight_id,
    )
    command_full = input_dir / "command.full.json"
    command_preflight = input_dir / "command.preflight.json"
    command_full.write_text(json.dumps(full_payload, indent=2, sort_keys=True) + "\n")
    command_preflight.write_text(
        json.dumps(preflight_payload, indent=2, sort_keys=True) + "\n"
    )
    shutil.copy2(command_full, runtime_inputs / "command.full.json")
    shutil.copy2(command_preflight, runtime_inputs / "command.preflight.json")

    if runtime_source.exists():
        shutil.rmtree(runtime_source)
    runtime_source.mkdir(parents=True)
    with tarfile.open(input_dir / "source_tree_no_beam_ablation_v1.tar.gz", "r:gz") as archive:
        archive.extractall(runtime_source, filter="data")

    manifest = {
        "schema": "paper_iv_h2o_source_locked_chtc_input_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_id": RECORD_ID,
        "files": {
            "source_archive": {
                "name": "source_tree_no_beam_ablation_v1.tar.gz",
                "sha256": sha256(input_dir / "source_tree_no_beam_ablation_v1.tar.gz"),
            },
            "fixture": {"name": "h2o_fixture.json", "sha256": sha256(input_dir / "h2o_fixture.json")},
            "checkpoint": {
                "name": "h2o_depth12_current.json",
                "sha256": sha256(input_dir / "h2o_depth12_current.json"),
            },
            "command_full": {"name": "command.full.json", "sha256": sha256(command_full)},
            "command_preflight": {
                "name": "command.preflight.json",
                "sha256": sha256(command_preflight),
            },
        },
        "start_depth": START_DEPTH,
        "target_depth": TARGET_DEPTH,
        "remaining_admissions": REMAINING_ADMISSIONS,
        "scientific_settings_changed_vs_stopped_run": [],
    }
    (input_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
