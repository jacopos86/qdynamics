"""The non-submitting CHTC package for the model decided on 2026-11-02 (18 qubits, (3,3,3))."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "chtc" / "paper_iv_h2o_two_arm_model_consistent_20261103"
FIXTURE = REPO_ROOT / "test_support" / "paper_iv_h2o" / "h2o_builder_model_consistent_work333_ref444_20261103_fixture.json"
OLD_FIXTURE = REPO_ROOT / "test_support" / "paper_iv_h2o" / "h2o_linear_fd_work373_ref484_fixture_20260826.json"


def _load(name: str):
    if str(PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR))
    spec = importlib.util.spec_from_file_location(name, PACKAGE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bp():
    return _load("build_package")


def test_package_constants_name_the_decided_fixture(bp):
    assert bp.CAMPAIGN == "paper_iv_h2o_two_arm_model_consistent_20261103"
    assert bp.FIXTURE_NAME == FIXTURE.name and bp.FIXTURE_SEARCH_PATHS == (FIXTURE,)
    assert bp.FIXTURE_EXPECTED_BYTES == 15_673_312
    assert bp.FIXTURE_EXPECTED_SHA256 == "de32cd4d1e29b8dcc07e94ad132ee43893fe03fd36cf4538eb6dadf6d5e194df"
    assert bp.FIXTURE_EXPECTED_IDENTITY["n_total_qubits"] == 18 and bp.FIXTURE_EXPECTED_IDENTITY["n_ph_max_by_mode"] == [3, 3, 3]
    assert bp.REMOTE_HOME_ROOT.endswith("/chtc/paper_iv_h2o_two_arm_model_consistent_20261103")
    runner = (PACKAGE_DIR / "run_segment.sh").read_text(encoding="utf-8")
    assert f'FIXTURE="{FIXTURE.name}"' in runner and "STATIC_ADAPT_PAULI_SIGN_CACHE_MAX_NQ=18" in runner
    assert "paper_iv_h2o_two_arm_20260903" not in runner
    monitor = (PACKAGE_DIR / "monitor.sh").read_text(encoding="utf-8")
    assert 'CAMPAIGN="paper_iv_h2o_two_arm_model_consistent_20261103"' in monitor


@pytest.mark.skipif(not FIXTURE.is_file(), reason="builder-produced fixture not checked in")
def test_fixture_identity_guard_accepts_the_decided_fixture_only(bp, tmp_path):
    identity = bp.verify_fixture_identity(FIXTURE)
    assert identity["sha256"] == bp.FIXTURE_EXPECTED_SHA256 and identity["bytes"] == bp.FIXTURE_EXPECTED_BYTES
    assert identity["n_total_qubits"] == 18 and identity["n_ph_max_by_mode"] == [3, 3, 3]
    assert identity["mode_labels"] == ["bend", "symmetric_stretch", "antisymmetric_stretch"]
    assert identity["model_definition"]["decided_model"] is True and identity["model_decided_on"] == "2026-11-02"
    assert identity["exact_ground_energy_hartree"] == pytest.approx(-74.9999118, abs=1e-6) and identity["sector_dimension"] == 14400
    assert identity["mode_frequencies_cm1"] == pytest.approx([2130.8, 5314.4, 5061.8], abs=0.5)
    if OLD_FIXTURE.is_file():
        with pytest.raises(ValueError, match="n_total_qubits"):
            bp.verify_fixture_identity(OLD_FIXTURE)
    # A byte-identical copy under another name passes; a one-character edit of the same length does not.
    copy = tmp_path / "copy.json"
    copy.write_bytes(FIXTURE.read_bytes())
    assert bp.verify_fixture_identity(copy)["sha256"] == identity["sha256"]
    data = bytearray(FIXTURE.read_bytes())
    marker = data.find(b'"n_total_qubits": 18')
    assert marker > 0
    data[marker + len(b'"n_total_qubits": 1')] = ord("9")
    (tmp_path / "edited.json").write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        bp.verify_fixture_identity(tmp_path / "edited.json")


def test_render_submit_carries_the_campaign_and_memory(bp):
    text = bp.render_submit(arm="macro", segment=0, cost_mode="proxy", max_wall_seconds=129_600, results_mode="home")
    assert not bp.check_submit_text(text, label="macro")
    assert "request_memory = 16GB" in text and f"package/{FIXTURE.name}" in text
    assert '+JobBatchName = "paper_iv_h2o_two_arm_model_consistent_20261103_macro"' in text
    assert 'arguments = "macro 00 macro_generator_v1 proxy 129600 50 home -"' in text
    custom = bp.render_submit(arm="singleton", segment=0, cost_mode="proxy", max_wall_seconds=129_600, results_mode="home",
                              request_memory="22528MB")
    assert "request_memory = 22528MB" in custom and "sector_complete_pauli_block_v1" in custom
    with pytest.raises(ValueError):
        bp.render_submit(arm="macro", segment=1, cost_mode="proxy", max_wall_seconds=129_600, results_mode="home")


@pytest.mark.skipif(not (PACKAGE_DIR / "package" / "input_manifest.json").is_file(), reason="package not built")
def test_built_package_manifest_records_the_fixture_identity(bp):
    manifest = json.loads((PACKAGE_DIR / "package" / "input_manifest.json").read_text(encoding="utf-8"))
    assert manifest["campaign"] == bp.CAMPAIGN
    assert manifest["files"][bp.FIXTURE_NAME]["sha256"] == bp.FIXTURE_EXPECTED_SHA256
    assert manifest["fixture_identity"]["n_total_qubits"] == 18 and manifest["requests"]["memory"].endswith(("GB", "MB"))
    for arm in ("macro", "singleton"):
        submit = (PACKAGE_DIR / f"submit_{arm}.sub").read_text(encoding="utf-8")
        assert f"request_memory = {manifest['requests']['memory']}" in submit and bp.FIXTURE_NAME in submit
