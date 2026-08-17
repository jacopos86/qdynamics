from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_append_adapt_stationary_singleton6_r100_fresh_"
    "20260808_v1_chtc"
)
ACTIVATION_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_append_adapt_stationary_singleton6_r100_fresh_"
    "20260808_v1_chtc_activation_ordinary_v3"
)


def _package_contract():
    spec = importlib.util.spec_from_file_location(
        "paper_i_append_r100_package_contract",
        PACKAGE_DIR / "package_contract.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_activation_authorization_is_accepted_by_worker_package() -> None:
    subprocess.run(
        [sys.executable, str(ACTIVATION_DIR / "validate_activation.py")],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    package_contract = _package_contract()
    authorization_path = ACTIVATION_DIR / "execution_authorization.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    assert (
        authorization["schema"]
        == package_contract.EXECUTION_AUTHORIZATION_SCHEMA
    )
    for execution_id in package_contract.EXPECTED_EXECUTION_IDS:
        accepted = package_contract.validate_execution_authorization(
            authorization_path,
            execution_id=execution_id,
        )
        assert accepted["sha256"] == authorization["sha256"]


def test_r100_archive_is_explicitly_selected_and_posix_remapped() -> None:
    submit = (ACTIVATION_DIR / "submit.sub").read_text(encoding="utf-8")
    archive = "transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"
    assert f"transfer_output_files = {archive}" in submit
    assert "output_destination" not in submit
    assert (
        f'transfer_output_remaps = "{archive}=/staging/jsstrobel/'
        "paper_i_append_adapt_stationary_singleton6_r100_20260809_v3/"
        'outputs/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"'
        in submit
    )
