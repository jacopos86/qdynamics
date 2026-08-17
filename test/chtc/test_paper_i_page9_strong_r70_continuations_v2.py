from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
V1_PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_page9_strong3_r50_to_r70_20260809_v1_chtc"
)
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_page9_strong3_r50_to_r70_20260809_v2_chtc"
)
BASE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_denominator_no_lanes_"
    "tau1em6_r50_20260807_v3_chtc"
)
ROUTE_SHA256 = (
    "e649eaa50428f6f396c4ab6cf25542a21add58115beb61d42df32408ad1399b6"
)
SOURCE_IDS = [
    (
        "phase3_qiskit_denominator_no_lanes__weak_strong__nph7__"
        "ra_global_singleton_plateau_commutation"
    ),
    (
        "phase3_qiskit_denominator_no_lanes__intermediate_strong__nph7__"
        "ra_global_singleton_plateau_commutation"
    ),
    (
        "phase3_qiskit_denominator_no_lanes__strong_strong_u8__nph7__"
        "ra_global_singleton_plateau_commutation"
    ),
]
EXECUTION_IDS = [f"{identifier}__resume_r50_to_r70" for identifier in SOURCE_IDS]


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_page9_package_is_three_rows_with_only_strong_strong_blocked() -> None:
    manifest = _json(PACKAGE_DIR / "package_manifest.json")
    plan = _json(PACKAGE_DIR / "execution_plan.json")
    assert manifest["row_count"] == 3
    assert manifest["execution_ids"] == EXECUTION_IDS
    assert manifest["blocked_execution_ids"] == [EXECUTION_IDS[2]]
    assert plan["materializable_execution_ids"] == EXECUTION_IDS[:2]
    assert plan["blocked_execution_ids"] == [EXECUTION_IDS[2]]
    assert manifest["source_horizon"] == 50
    assert manifest["target_horizon"] == 70
    assert manifest["route_contract_sha256"] == ROUTE_SHA256
    assert manifest["execution_authorized"] is False
    assert manifest["submission_authorized"] is False
    assert manifest["submission_ready"] is False
    assert manifest["submitted"] is False
    assert manifest["remote_stage"] is False
    assert manifest["condor_submit"] is False
    assert not (PACKAGE_DIR / "submit.sub").exists()


def test_page9_v1_seal_is_preserved() -> None:
    manifest_path = V1_PACKAGE_DIR / "package_manifest.json"
    manifest = _json(manifest_path)
    import hashlib

    assert manifest["sha256"] == (
        "ab317fcad492c59fae4e0e11426442241a481c06a42dbb26d5427969e32fd245"
    )
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        "d7c2314ebfe48f47624534d4c56b617189af67a4312a4fc0ef6eba03e5871a50"
    )


def test_page9_v2_accepts_symlinked_staging_mount_but_not_symlink_file(
    tmp_path: Path,
) -> None:
    code = f"""
import hashlib, pathlib, sys
sys.path.insert(0, {str(PACKAGE_DIR)!r})
import materialize_resume_input as materializer
root = pathlib.Path({str(tmp_path)!r})
real_mount = root / 'real-staging'
real_mount.mkdir()
archive = real_mount / 'archive.tar.gz'
archive.write_bytes(b'authenticated-archive')
mount_alias = root / 'staging'
mount_alias.symlink_to(real_mount, target_is_directory=True)
lexical_archive = mount_alias / 'archive.tar.gz'
remote = {{
    'path': lexical_archive.as_posix(),
    'size_bytes': archive.stat().st_size,
    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
}}
observed = materializer._validated_source_archive(lexical_archive, remote)
assert observed == lexical_archive
assert observed.resolve() == archive.resolve()
assert observed.as_posix() != observed.resolve().as_posix()
try:
    materializer._validated_source_archive(archive, remote)
except materializer.PackageContractError:
    pass
else:
    raise AssertionError('resolved physical path must not replace declared lexical path')
file_alias = mount_alias / 'archive-link.tar.gz'
file_alias.symlink_to(archive)
link_remote = dict(remote, path=file_alias.as_posix())
try:
    materializer._validated_source_archive(file_alias, link_remote)
except materializer.PackageContractError:
    pass
else:
    raise AssertionError('direct archive-file symlink must remain rejected')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code],
        cwd="/",
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_page9_ws_is_preserve_exact_route_request_and_source_identity() -> None:
    for source_id, execution_id in zip(SOURCE_IDS, EXECUTION_IDS, strict=True):
        source_job = _json(BASE_DIR / "jobs" / f"{source_id}.json")
        source_protocol = _json(BASE_DIR / source_job["protocol_path"])
        job = _json(PACKAGE_DIR / "jobs" / f"{execution_id}.json")
        derived = _json(PACKAGE_DIR / job["derived_protocol"]["path"])
        assert job["source_job"]["canonical_sha256"] == source_job["sha256"]
        assert job["source_protocol_sha256"] == source_protocol["sha256"]
        assert source_protocol["route_contract"] == derived["route_contract"]
        assert derived["route_contract"]["sha256"] == ROUTE_SHA256
        assert source_protocol["source_locks"] == derived["source_locks"]
        assert source_protocol["algorithm_id"] == derived["algorithm_id"]
        assert source_protocol["adapter_id"] == derived["adapter_id"]
        source_request = copy.deepcopy(source_protocol["request"])
        derived_request = copy.deepcopy(derived["request"])
        assert source_request["execution"]["stop"].pop(
            "maximum_controller_rounds"
        ) == 50
        assert derived_request["execution"]["stop"].pop(
            "maximum_controller_rounds"
        ) == 70
        assert source_request == derived_request


def test_page9_strong_strong_cannot_materialize_activate_or_submit() -> None:
    strong_job = PACKAGE_DIR / "jobs" / f"{EXECUTION_IDS[2]}.json"
    materialize = subprocess.run(
        [
            sys.executable,
            "-B",
            str(PACKAGE_DIR / "materialize_resume_input.py"),
            "--job",
            str(strong_job),
            "--source-archive",
            "/staging/jsstrobel/not-present.tar.gz",
            "--output-dir",
            "/staging/jsstrobel/not-created-page9-resume",
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert materialize.returncode == 2
    assert "Blocked strong--strong requires" in materialize.stderr

    readiness = subprocess.run(
        [sys.executable, "-B", str(PACKAGE_DIR / "validate_package.py"), "--require-ready"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert readiness.returncode == 2
    assert "--require-ready requires --resume-root" in readiness.stderr
    assert not (PACKAGE_DIR / "submit.sub").exists()


def test_page9_submit_template_is_failure_safe_and_posix_remapped() -> None:
    template = (PACKAGE_DIR / "submit.sub.in").read_text(encoding="utf-8")
    output = "transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"
    assert f"transfer_output_files = {output}" in template
    assert (
        f'transfer_output_remaps = "{output}=@@REMOTE_OUTPUT_ROOT@@/'
        '$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"'
        in template
    )
    assert "output_destination" not in template
    assert "preserve_relative_paths = False" in template
    assert "periodic_release = False" in template
    wrapper = (PACKAGE_DIR / "execute_authorized_job.sh").read_text(encoding="utf-8")
    assert "trap finalize_attempt EXIT" in wrapper
    assert "attempt_status.json" in wrapper
    assert 'tar -C . -czf "$output_archive" "$work_root"' in wrapper


def test_page9_wrapper_captures_pre_execution_failure(tmp_path: Path) -> None:
    execution_id = EXECUTION_IDS[0]
    archive = tmp_path / "transfer" / f"{execution_id}__1__0.tar.gz"
    completed = subprocess.run(
        [
            "bash",
            str(PACKAGE_DIR / "execute_authorized_job.sh"),
            "missing-package",
            "missing-base",
            f"missing/{execution_id}.json",
            f"missing/{execution_id}.json",
            "missing/resume_materialization.json",
            "missing/resume_input.tar.gz",
            "0" * 64,
            "missing/image.sif",
            "0" * 64,
            execution_id,
            f"transfer/{execution_id}__1__0.tar.gz",
        ],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 66
    assert archive.is_file() and archive.stat().st_size > 0
    listing = subprocess.run(
        ["tar", "-tzf", str(archive)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    assert "worker_outputs/attempt_status.json" in listing
    assert "worker_outputs/attempt.stderr" in listing


def test_page9_streaming_parser_is_vendored_and_site_package_independent() -> None:
    composition = _json(PACKAGE_DIR / "source_composition.json")
    runtime = composition["streaming_json_runtime"]
    assert runtime["distribution"] == "ijson"
    assert runtime["upstream_version"] == "3.5.1"
    assert runtime["backend"] == "python"
    assert runtime["ambient_install_required"] is False
    code = f"""
import io, sys
sys.path.insert(0, {str(PACKAGE_DIR)!r})
import package_contract
from vendored_ijson_python import BACKEND, VENDORED_IJSON_VERSION, parse
events = list(parse(io.BytesIO(b'{{\"x\":[1,true,null]}}'), buf_size=2))
assert BACKEND == 'python'
assert VENDORED_IJSON_VERSION == '3.5.1'
assert ('x.item', 'number', 1) in events
assert ('x.item', 'boolean', True) in events
assert ('x.item', 'null', None) in events
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code],
        cwd="/",
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    forbidden = [
        path
        for path in PACKAGE_DIR.rglob("*")
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}
    ]
    assert forbidden == []


def test_page9_worker_imports_from_isolated_condor_layout(tmp_path: Path) -> None:
    isolated = tmp_path / "page9_package"
    isolated.mkdir()
    for name in ("run_cell.py", "package_contract.py", "vendored_ijson_python.py"):
        shutil.copyfile(PACKAGE_DIR / name, isolated / name)
    code = f"""
import sys
sys.path.insert(0, {str(isolated)!r})
import run_cell
assert not hasattr(run_cell, 'REPO_ROOT')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr

    activation = (PACKAGE_DIR / "activate_package.py").read_text(encoding="utf-8")
    assert "def _pinned_image_runtime_preflight()" in activation
    assert '"-I",' in activation and '"-S",' in activation and '"-B",' in activation
    assert "Vendored streaming parser failed in the pinned image" in activation


def test_page9_result_publication_survives_exdev(tmp_path: Path) -> None:
    code = f"""
import errno, pathlib, sys
sys.path.insert(0, {str(PACKAGE_DIR)!r})
import run_cell
root = pathlib.Path({str(tmp_path)!r})
source = root / 'source'
destination = root / 'destination'
source.mkdir()
(source / 'payload.txt').write_text('closed', encoding='utf-8')
real_rename = run_cell.os.rename
calls = [0]
def exdev_once(left, right):
    calls[0] += 1
    if calls[0] == 1:
        raise OSError(errno.EXDEV, 'simulated cross-device publication')
    return real_rename(left, right)
run_cell.os.rename = exdev_once
run_cell._atomic_publish_tree(source, destination)
assert calls[0] == 2
assert not source.exists()
assert (destination / 'payload.txt').read_text(encoding='utf-8') == 'closed'
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code],
        cwd="/",
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_page9_package_validator_passes_inert_state() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", str(PACKAGE_DIR / "validate_package.py")],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "passed_inert_one_blocked"
    assert receipt["row_count"] == 3
    assert receipt["blocked_execution_ids"] == [EXECUTION_IDS[2]]
