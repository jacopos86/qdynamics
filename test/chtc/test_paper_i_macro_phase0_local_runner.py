from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "macro_gradient_phase0_macro_phase123_proxy_no_lanes_local_20260810_v1/"
    "local_runner.py"
)


def test_local_activation_is_exact_v3_bound_and_never_submits(tmp_path: Path) -> None:
    activation = tmp_path / "activation"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(RUNNER),
            "prepare",
            "--activation-dir",
            str(activation),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((activation / "activation_manifest.json").read_text())
    request = json.loads((activation / "activation_request.json").read_text())
    preflight = json.loads((activation / "host_preflight.json").read_text())
    assert manifest["authorization_count"] == 6
    assert manifest["execution_target"] == "local_mac_serial"
    assert manifest["execution_authorized"] is True
    assert manifest["submission_authorized"] is False
    assert manifest["submitted"] is False
    assert request["execution_target_change_only"] is True
    assert request["scientific_settings_changed"] is False
    assert preflight["sealed_source_preflight_count"] == 6
    assert preflight["scientific_execution_performed"] is False
    authorities = sorted((activation / "authorizations").glob("*.json"))
    assert len(authorities) == 6
    for path in authorities:
        authority = json.loads(path.read_text())
        assert authority["scope"] == "single_cell_local_execution_only"
        assert authority["execution_authorized"] is True
        assert authority["submission_authorized"] is False
        assert authority["paper_evidence_adoption_authorized"] is False
