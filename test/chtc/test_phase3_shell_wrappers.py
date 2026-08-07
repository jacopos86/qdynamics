from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE3 = REPO_ROOT / "chtc" / "phase3_optuna"


def _copy_phase3_script(tmp_path: Path, name: str) -> Path:
    target = tmp_path / "chtc" / "phase3_optuna" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text((PHASE3 / name).read_text(encoding="utf-8"), encoding="utf-8")
    target.chmod(target.stat().st_mode | stat.S_IXUSR)
    return target


def _prepend_fake_bin(env: dict[str, str], bin_dir: Path) -> dict[str, str]:
    updated = dict(env)
    updated["PATH"] = f"{bin_dir}{os.pathsep}{updated.get('PATH', '')}"
    return updated


def test_run_task_discovers_nested_records_without_phase3_records_env(tmp_path: Path) -> None:
    _copy_phase3_script(tmp_path, "run_task.sh")
    nested_records = tmp_path / "chtc" / "phase3_optuna" / "input" / "nested_bundle" / "phase0_oracle_smoke_records.tsv"
    nested_records.parent.mkdir(parents=True, exist_ok=True)
    nested_records.write_text(
        "record_id\tmode\n"
        "smoke_a\toracle-grid\n"
        "smoke_b\toracle-grid\n"
        "smoke_c\toracle-grid\n",
        encoding="utf-8",
    )
    immediate_records = tmp_path / "chtc" / "phase3_optuna" / "input" / "other_records.tsv"
    immediate_records.write_text("record_id\tmode\nnot_the_record\toracle-grid\n", encoding="utf-8")

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    python_log = tmp_path / "fake_python_args.log"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "{ printf 'argv:'; for arg in \"$@\"; do printf ' [%s]' \"$arg\"; done; printf '\\n'; } >> \"$FAKE_PYTHON_LOG\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

    env = _prepend_fake_bin(os.environ.copy(), fake_bin)
    env.pop("PHASE3_RECORDS_PATH", None)
    env["FAKE_PYTHON_LOG"] = str(python_log)
    env["PHASE3_SHELL_HEARTBEAT_SEC"] = "1"
    expected_rel = "chtc/phase3_optuna/input/nested_bundle/phase0_oracle_smoke_records.tsv"

    for record_id in ("smoke_a", "smoke_b", "smoke_c"):
        result = subprocess.run(
            ["bash", "chtc/phase3_optuna/run_task.sh", record_id],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert f"records_path={expected_rel}" in result.stdout

    fake_invocations = python_log.read_text(encoding="utf-8")
    assert fake_invocations.count(f"[--records] [{expected_rel}]") == 3


def test_apptainer_wrapper_forwards_phase3_records_path_under_cleanenv(tmp_path: Path) -> None:
    _copy_phase3_script(tmp_path, "run_task_apptainer.sh")
    image = tmp_path / "chtc" / "phase3_optuna" / "image.sif"
    image.touch()

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    fake_log = tmp_path / "fake_apptainer_args.log"
    fake_runtime_script = (
        "#!/usr/bin/env bash\n"
        "{\n"
        "  printf 'APPTAINERENV_PHASE3_RECORDS_PATH=%s\\n' \"${APPTAINERENV_PHASE3_RECORDS_PATH-__unset__}\"\n"
        "  printf 'SINGULARITYENV_PHASE3_RECORDS_PATH=%s\\n' \"${SINGULARITYENV_PHASE3_RECORDS_PATH-__unset__}\"\n"
        "  for arg in \"$@\"; do printf '%s\\n' \"$arg\"; done\n"
        "} > \"$FAKE_APPTAINER_LOG\"\n"
        "exit 0\n"
    )
    fake_apptainer = fake_bin / "apptainer"
    fake_apptainer.write_text(fake_runtime_script, encoding="utf-8")
    fake_apptainer.chmod(fake_apptainer.stat().st_mode | stat.S_IXUSR)

    env = _prepend_fake_bin(os.environ.copy(), fake_bin)
    env["PATH"] = f"{fake_bin}{os.pathsep}/usr/bin{os.pathsep}/bin"
    env.pop("PROJECT_IMAGE", None)
    env.pop("APPTAINERENV_PHASE3_RECORDS_PATH", None)
    env.pop("SINGULARITYENV_PHASE3_RECORDS_PATH", None)
    env["FAKE_APPTAINER_LOG"] = str(fake_log)
    env["PHASE3_RECORDS_PATH"] = "chtc/phase3_optuna/input/nested_bundle/smoke_records.tsv"

    result = subprocess.run(
        ["bash", "chtc/phase3_optuna/run_task_apptainer.sh", "smoke_a"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    lines = fake_log.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == [
        "APPTAINERENV_PHASE3_RECORDS_PATH=chtc/phase3_optuna/input/nested_bundle/smoke_records.tsv",
        "SINGULARITYENV_PHASE3_RECORDS_PATH=__unset__",
    ]
    args = lines[2:]
    assert args[:2] == ["exec", "--cleanenv"]
    assert "--env" not in args

    fake_apptainer.unlink()
    fake_singularity = fake_bin / "singularity"
    fake_singularity.write_text(fake_runtime_script, encoding="utf-8")
    fake_singularity.chmod(fake_singularity.stat().st_mode | stat.S_IXUSR)
    fake_log.write_text("", encoding="utf-8")
    result = subprocess.run(
        ["bash", "chtc/phase3_optuna/run_task_apptainer.sh", "smoke_a"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    lines = fake_log.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == [
        "APPTAINERENV_PHASE3_RECORDS_PATH=__unset__",
        "SINGULARITYENV_PHASE3_RECORDS_PATH=chtc/phase3_optuna/input/nested_bundle/smoke_records.tsv",
    ]
    args = lines[2:]
    assert args[:2] == ["exec", "--cleanenv"]
    assert "--env" not in args

    fake_singularity.unlink()
    fake_apptainer.write_text(fake_runtime_script, encoding="utf-8")
    fake_apptainer.chmod(fake_apptainer.stat().st_mode | stat.S_IXUSR)
    fake_log.write_text("", encoding="utf-8")
    env.pop("PHASE3_RECORDS_PATH", None)
    result = subprocess.run(
        ["bash", "chtc/phase3_optuna/run_task_apptainer.sh", "smoke_a"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    lines = fake_log.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == [
        "APPTAINERENV_PHASE3_RECORDS_PATH=__unset__",
        "SINGULARITYENV_PHASE3_RECORDS_PATH=__unset__",
    ]
    args = lines[2:]
    assert args[:2] == ["exec", "--cleanenv"]
    assert "--env" not in args
    assert not any(arg.startswith("PHASE3_RECORDS_PATH=") for arg in args)

    fake_log.write_text("", encoding="utf-8")
    env["PHASE3_RECORDS_PATH"] = ""
    result = subprocess.run(
        ["bash", "chtc/phase3_optuna/run_task_apptainer.sh", "smoke_a"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    lines = fake_log.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == [
        "APPTAINERENV_PHASE3_RECORDS_PATH=__unset__",
        "SINGULARITYENV_PHASE3_RECORDS_PATH=__unset__",
    ]
    args = lines[2:]
    assert args[:2] == ["exec", "--cleanenv"]
    assert "--env" not in args
    assert not any(arg.startswith("PHASE3_RECORDS_PATH=") for arg in args)
