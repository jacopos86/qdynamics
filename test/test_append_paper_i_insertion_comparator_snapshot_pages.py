from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pipelines.reporting import (
    append_paper_i_insertion_comparator_snapshot_pages as snapshot_pages,
)


@pytest.fixture
def isolated_sealed_source_import_state():
    """Restore process globals changed by the source-locked worker fixture."""

    def is_sealed_namespace(name: str) -> bool:
        return (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
            or name == "package_contract"
            or name.startswith("paper_i_page16_")
        )

    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if is_sealed_namespace(name)
    }
    saved_path = list(sys.path)
    saved_importer_cache = dict(sys.path_importer_cache)
    saved_environ = dict(os.environ)
    saved_dont_write_bytecode = sys.dont_write_bytecode
    saved_cwd = Path.cwd()
    try:
        yield
    finally:
        os.chdir(saved_cwd)
        for name in tuple(sys.modules):
            if is_sealed_namespace(name):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        sys.path_importer_cache.clear()
        sys.path_importer_cache.update(saved_importer_cache)
        os.environ.clear()
        os.environ.update(saved_environ)
        sys.dont_write_bytecode = saved_dont_write_bytecode
        importlib.invalidate_caches()


def _write_pdf(path: Path, payloads: list[bytes]) -> None:
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, NameObject

    writer = pypdf.PdfWriter()
    for index, payload in enumerate(payloads, 1):
        page = writer.add_blank_page(width=600 + index, height=800)
        stream = DecodedStreamObject()
        stream.set_data(payload)
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


def _content_hashes(path: Path) -> list[str]:
    pypdf = pytest.importorskip("pypdf")
    result = []
    for page in pypdf.PdfReader(str(path), strict=False).pages:
        contents = page.get_contents()
        payload = b"" if contents is None else contents.get_data()
        result.append(hashlib.sha256(payload).hexdigest())
    return result


def _write_digested(path: Path, unsigned: dict[str, object]) -> dict[str, object]:
    value = {
        **unsigned,
        "sha256": snapshot_pages._canonical_sha256(unsigned),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return value


def _adapter(path: Path, *, revision: str) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema": "paper_i_ra_adapt_page16_insertion_comparator_progress_adapter_v4",
        "status": f"fixture_{revision}",
        "campaign_counts": {
            "authenticated_curves_plotted": 2,
            "local_cells_closed_authenticated": 0,
            "local_cells_completed_at_k30": 0,
            "local_cells_right_censored_at_k30": 0,
            "completed_validated": 2,
        },
        "campaign_execution_state": {
            "page16_remaining_chtc_factory": "frozen_no_further_materialization",
            "local_campaign_state": "fixture",
        },
        "matrix": [],
        "completed_comparators": {
            "weak_weak": {
                "always_commutation_reduced": {
                    "terminal": {"k": 50, "energy": -0.918}
                }
            },
            "intermediate_weak": {
                "always_commutation_reduced": {
                    "terminal": {"k": 50, "energy": -0.827}
                }
            },
        },
        "sources": {},
        "limitations": ["fixture"],
    }
    value = {**unsigned, "sha256": snapshot_pages._canonical_sha256(unsigned)}
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def _patch_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path, Path, Path]:
    paths = (
        tmp_path / "report.pdf",
        tmp_path / "report-provenance.json",
        tmp_path / "page17.pdf",
        tmp_path / "page17.png",
        tmp_path / "adapter.json",
    )
    for name, value in zip(
        (
            "TARGET_PDF",
            "TARGET_PROVENANCE",
            "PAGE17_PDF",
            "PAGE17_PNG",
            "ADAPTER_PATH",
        ),
        paths,
        strict=True,
    ):
        monkeypatch.setattr(snapshot_pages, name, value)
    return paths


def _continuation_adapter():
    path = snapshot_pages.CONTINUATION_ADAPTER_PATH
    name = "paper_i_page16_reporting_continuation_fixture"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_continuation_adapter_loader_rejects_preloaded_module_poisoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "paper_i_page16_reporting_continuation_adapter"
    monkeypatch.setattr(snapshot_pages, "_CONTINUATION_ADAPTER", None)
    monkeypatch.setitem(sys.modules, name, object())

    with pytest.raises(snapshot_pages.UpdateError, match="untrusted.*preloaded"):
        snapshot_pages._load_continuation_adapter()


def test_continuation_adapter_loader_rejects_preloaded_k30_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer_name = "paper_i_page16_reporting_continuation_adapter"
    k30_name = "paper_i_page16_pinned_k30_runner_for_k50_continuation"
    monkeypatch.setattr(snapshot_pages, "_CONTINUATION_ADAPTER", None)
    monkeypatch.delitem(sys.modules, outer_name, raising=False)
    monkeypatch.setitem(sys.modules, k30_name, object())

    with pytest.raises(snapshot_pages.UpdateError, match="k30 authority.*preloaded"):
        snapshot_pages._load_continuation_adapter()


def test_reporting_module_import_does_not_load_continuation_authority() -> None:
    code = (
        "import json, sys\n"
        "from pipelines.reporting import "
        "append_paper_i_insertion_comparator_snapshot_pages\n"
        "print(json.dumps('paper_i_page16_reporting_continuation_adapter' "
        "in sys.modules))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=snapshot_pages.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    assert json.loads(completed.stdout) is False


def test_continuation_boundary_normalizes_producer_runtime_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class ProducerFailure(RuntimeError):
        pass

    class BrokenAdapter:
        ContinuationError = ProducerFailure

        @staticmethod
        def decision_snapshot(*, cached: object) -> dict[str, object]:
            raise ProducerFailure("fixture producer drift")

    monkeypatch.setattr(
        snapshot_pages, "_load_continuation_adapter", lambda: BrokenAdapter()
    )

    with pytest.raises(snapshot_pages.UpdateError, match="fixture producer drift"):
        snapshot_pages.authenticated_continuation_inventory(
            k30_inventory={"completed": {}},
            activation_dir=tmp_path / "activation",
            runtime_dir=tmp_path / "runtime",
            macro_terminal_receipt=tmp_path / "terminal.json",
        )


def test_no_eligible_decisions_remain_closed_but_require_macro_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    conditional = tuple(f"local-{index}" for index in range(9))
    terminals = tuple(f"remote-{index}" for index in range(3))
    decisions = [
        {
            "execution_id": execution_id,
            "extension_decision": "stop_at_k30",
            "k30_plateau_gate_sha256": f"{index + 1:064x}",
        }
        for index, execution_id in enumerate(conditional)
    ]
    unsigned = {
        "schema": "fixture_decision_status_v1",
        "status": "passed_all_k30_decisions_closed",
        "all_decisions_closed": True,
        "conditional_execution_ids": list(conditional),
        "terminal_chtc_k50_execution_ids": list(terminals),
        "closed_decision_count": 9,
        "pending_execution_ids": [],
        "eligible_execution_ids": [],
        "stop_at_k30_execution_ids": list(conditional),
        "decisions": decisions,
        "scientific_execution_performed": False,
    }
    decision_snapshot = {
        **unsigned,
        "sha256": snapshot_pages._canonical_sha256(unsigned),
    }

    class Adapter:
        DECISION_STATUS_SCHEMA = "fixture_decision_status_v1"
        CONDITIONAL_EXECUTION_IDS = conditional
        TERMINAL_CHTC_EXECUTION_IDS = terminals

        @staticmethod
        def decision_snapshot(*, cached: object) -> dict[str, object]:
            return decision_snapshot

    k30_completed = {
        row["execution_id"]: {
            "effective_plateau_gate": {
                "extension_decision": "stop_at_k30",
                "canonical_sha256": row["k30_plateau_gate_sha256"],
            }
        }
        for row in decisions
    }
    monkeypatch.setattr(
        snapshot_pages, "_load_continuation_adapter", lambda: Adapter()
    )

    inventory = snapshot_pages.authenticated_continuation_inventory(
        k30_inventory={"completed": k30_completed},
        activation_dir=tmp_path / "activation",
        runtime_dir=tmp_path / "runtime",
        macro_terminal_receipt=tmp_path / "terminal.json",
    )

    assert inventory["campaign_state"] == "not_activated"
    assert inventory["eligible_execution_ids"] == []
    assert inventory["stop_at_k30_execution_ids"] == list(conditional)
    assert inventory["all_required_continuations_closed"] is True
    assert inventory["macro_terminal_authenticated"] is False


def test_current_adapter_authenticates_available_page16_comparators() -> None:
    available = tuple(
        regime
        for regime, spec in snapshot_pages.COMPLETED_ARCHIVES.items()
        if (snapshot_pages.RETRIEVED_DIR / str(spec["filename"])).is_file()
    )
    assert available == tuple(snapshot_pages.COMPLETED_ARCHIVES)
    adapter = snapshot_pages.build_adapter(
        completed_regimes=available,
        include_local=False,
    )

    assert adapter["status"] == (
        f"provisional_page16_{len(available)}_authenticated_"
        "0_local_complete_0_local_right_censored"
    )
    assert adapter["campaign_counts"] == {
        "page16_comparator_jobs_planned": 12,
        "chtc_completed_authenticated": len(available),
        "local_cells_closed_authenticated": 0,
        "local_cells_completed_at_k30": 0,
        "local_cells_right_censored_at_k30": 0,
        "local_cells_completed_at_k50": 0,
        "authenticated_curves_plotted": len(available),
        "completed_validated": len(available),
        "always_open_authenticated": len(available),
        "append_only_authenticated": 0,
        "pending_or_unclosed": 12 - len(available),
        "published_partial_unclosed_not_plotted": 0,
        "plateau_reference_jobs_rerun": 0,
    }
    assert len(adapter["matrix"]) == 6
    assert len(adapter["reference_cells"]) == 6
    assert [row["target_horizon"] for row in adapter["reference_cells"]] == [
        50,
        50,
        50,
        30,
        30,
        30,
    ]
    completed = adapter["completed_comparators"]
    assert list(completed) == list(available)
    always = [completed[regime][snapshot_pages.EXPECTED_POLICIES[0]] for regime in available]
    assert [row["cluster_id"] for row in always] == [
        snapshot_pages.COMPLETED_ARCHIVES[regime]["cluster_id"] for regime in available
    ]
    assert [row["proc_id"] for row in always] == [0] * len(available)
    assert [row["terminal"]["k"] for row in always] == [50] * len(available)
    assert all(
        row["status"] == "completed_authenticated_chtc_archive"
        and set(row["costs"]) == {"N2q", "D2q", "Dc", "W1q", "S_alg"}
        and row["sources"]["archive_closure"]["all_declared_payload_hashes_verified"]
        is True
        for row in always
    )
    assert [row["always_open"] for row in adapter["matrix"][:2]] == [
        "complete / authenticated k=50",
        "complete / authenticated k=50",
    ]
    assert all(
        row["append_only"] == "pending / local k=30 campaign"
        for row in adapter["matrix"]
    )
    assert all(
        row["always_open"] == "pending / local k=30 campaign"
        for row in adapter["matrix"][2:]
    )
    assert [
        (
            row["resources"]["request_cpus"],
            row["resources"]["request_memory_mb"],
            row["resources"]["request_disk_mb"],
            row["resources"]["max_runtime_seconds"],
        )
        for row in adapter["matrix"]
    ] == [(4, 32768, 61440, 259200)] * 3 + [(4, 49152, 81920, 259200)] * 3
    assert all(
        "/outputs/transfer/" in row["sources"]["retrieval_identity"]["remote_path"]
        for row in always
    )
    assert adapter["campaign_execution_state"]["page16_remaining_chtc_factory"] == (
        "factory_retained_paused_at_completed_prefix_after_acknowledged_removal"
    )
    assert adapter["campaign_execution_state"]["local_campaign_state"] == (
        "excluded_by_caller"
    )
    assert len(
        adapter["campaign_execution_state"]["continuation_evidence_revision"]
    ) == 64
    assert adapter["paper_evidence_adopted"] is False
    assert adapter["plateau_reference_reused_not_rerun"] is True


def test_optional_sw_chtc_archive_requires_closed_remote_exclusion_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_relative = Path(
        "chtc/paper_i_ra_adapt_repair_20260727/"
        "retrieved_page16_insertion_comparators_20260812/"
        "strong_weak_u8_always__9647386__1.tar.gz"
    )
    archive = tmp_path / archive_relative
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"sealed archive fixture")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    unsigned = {
        "schema": (
            "paper_i_ra_adapt_page16_sw_always_"
            "remote_materialization_exclusion_receipt_v2"
        ),
        "status": (
            "passed_sw_always_k50_closed_remote_materialization_excluded"
        ),
        "scientific_execution_performed_by_action": False,
        "completed_remote_cell": {
            "regime_id": "strong_weak_u8",
            "comparator_policy": "always_commutation_reduced",
            "execution_id": snapshot_pages.SW_ALWAYS_EXECUTION_ID,
            "cluster_id": 9647386,
            "proc_id": 1,
            "controller_rounds_completed": 50,
            "authenticated_full_sealed_closure": True,
            "archive": {
                "path": archive_relative.as_posix(),
                "remote_path": "osdf:///fixed/outputs/transfer/archive.tar.gz",
                "size_bytes": archive.stat().st_size,
                "sha256": archive_sha,
            },
            "history": {"exit_code": 0},
            "worker_receipt": {"canonical_sha256": "b" * 64},
        },
        "remote_materialization_exclusion": {
            "outcome": (
                "factory_retained_paused_at_completed_prefix_"
                "after_acknowledged_removal"
            ),
            "removal_command": "condor_rm 9647386",
            "removal_attempts_authenticated": True,
            "before_snapshot": {
                "job_materialize_paused": 1,
                "job_materialize_next_proc_id": 2,
                "materialized_proc_ids": [],
                "history_completed_proc_ids": [0, 1],
            },
            "after_snapshot": {
                "cluster_present_in_queue": False,
                "factory_present": True,
                "factory_materialization_paused": True,
                "job_materialize_limit": 2,
                "job_materialize_max_idle": 0,
                "job_materialize_next_proc_id": 2,
                "history_completed_proc_ids": [0, 1],
            },
            "latent_proc_ids_never_materialized": list(range(2, 11)),
            "queue_cluster_absent": True,
            "remote_materialization_excluded": True,
        },
    }
    receipt = {
        **unsigned,
        "sha256": snapshot_pages._canonical_sha256(unsigned),
    }
    receipt_path = tmp_path / "closure.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    monkeypatch.setattr(snapshot_pages, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(snapshot_pages, "SW_ALWAYS_CLOSURE_RECEIPT", receipt_path)

    worker = object()
    job = {"execution_id": snapshot_pages.SW_ALWAYS_EXECUTION_ID}
    strict_calls: list[tuple[object, object]] = []

    class StrictContinuationAuthority:
        SW_ALWAYS_CHTC_EXECUTION_ID = snapshot_pages.SW_ALWAYS_EXECUTION_ID
        ContinuationError = RuntimeError

        class k30:
            @staticmethod
            def _load_worker() -> object:
                return worker

        @staticmethod
        def _job_by_id(observed_worker: object) -> dict[str, object]:
            assert observed_worker is worker
            return {snapshot_pages.SW_ALWAYS_EXECUTION_ID: job}

        @staticmethod
        def _authenticate_sw_always_closure(
            observed_worker: object, *, job: object
        ) -> dict[str, object]:
            strict_calls.append((observed_worker, job))
            return {
                "execution_id": snapshot_pages.SW_ALWAYS_EXECUTION_ID,
                "cluster_id": 9647386,
                "proc_id": 1,
                "controller_rounds_completed": 50,
                "source_closure_receipt_sha256": receipt["sha256"],
                "authenticated_full_sealed_closure": True,
                "remote_materialization_exclusion_outcome": (
                    "factory_retained_paused_at_completed_prefix_"
                    "after_acknowledged_removal"
                ),
                "remote_materialization_exclusion_authenticated": True,
                "archive": {
                    "path": archive_relative.as_posix(),
                    "remote_path": "osdf:///fixed/outputs/transfer/archive.tar.gz",
                    "size_bytes": archive.stat().st_size,
                    "sha256": archive_sha,
                },
            }

    monkeypatch.setattr(
        snapshot_pages,
        "_load_continuation_adapter",
        lambda: StrictContinuationAuthority,
    )

    spec, observed = snapshot_pages._optional_sw_always_archive()

    assert spec == {
        "cluster_id": 9647386,
        "proc_id": 1,
        "filename": "strong_weak_u8_always__9647386__1.tar.gz",
        "remote_path": "osdf:///fixed/outputs/transfer/archive.tar.gz",
        "size_bytes": archive.stat().st_size,
        "sha256": archive_sha,
    }
    assert observed == receipt
    assert strict_calls == [(worker, job)]

    def reject_strict_closure(
        observed_worker: object, *, job: object
    ) -> dict[str, object]:
        raise StrictContinuationAuthority.ContinuationError(
            "typed insertion identity drifted"
        )

    monkeypatch.setattr(
        StrictContinuationAuthority,
        "_authenticate_sw_always_closure",
        reject_strict_closure,
    )
    with pytest.raises(snapshot_pages.UpdateError, match="strict authentication"):
        snapshot_pages._optional_sw_always_archive()


def test_v2_local_activation_is_authenticated_before_runtime_materializes(
    tmp_path: Path,
) -> None:
    assert snapshot_pages.LOCAL_ACTIVATION_DIR.is_dir()

    inventory = snapshot_pages.authenticated_local_comparator_inventory(
        runtime_dir=tmp_path / "not-materialized-runtime"
    )

    assert inventory["campaign_state"] == "activated_runtime_pending"
    assert len(inventory["execution_ids"]) == 10
    assert inventory["completed"] == {}
    assert set(inventory["cell_states"].values()) == {
        "pending_runtime_not_materialized"
    }
    assert inventory["sources"]["expected_local_adapter_sha256"] == (
        snapshot_pages.EXPECTED_LOCAL_ADAPTER_SHA256
    )


def test_unclosed_local_publication_is_never_returned_as_a_curve(
    tmp_path: Path,
) -> None:
    execution_id = "fixture-execution"
    run_root = tmp_path / "runs" / execution_id
    run_root.mkdir(parents=True)
    state, result = snapshot_pages._local_summary_result(
        runtime_dir=tmp_path,
        execution_id=execution_id,
        job_path=tmp_path / "job.json",
        job={
            "regime_id": "weak_weak",
            "comparator_policy": "append_only",
        },
        authority={},
        compile_costs=False,
    )

    assert state == "published_partial_unclosed"
    assert result is None


def test_real_v2_weak_weak_append_closure_authenticates_gate_bytes() -> None:
    inventory = snapshot_pages.authenticated_local_comparator_inventory(
        compile_costs=False
    )
    execution_id = inventory["execution_ids"][0]
    result = inventory["completed"][execution_id]

    assert inventory["cell_states"][execution_id] == (
        "closed_authenticated_local_receipt"
    )
    assert result["regime_id"] == "weak_weak"
    assert result["comparator_policy"] == "append_only"
    assert result["status"] == "completed_authenticated_local_k30"
    assert result["terminal"]["k"] == 30
    assert result["effective_plateau_gate"]["selected_controller_round"] == 10
    assert result["effective_plateau_gate"]["classification"] == (
        "effective_plateau_observed_within_k30"
    )
    assert result["effective_plateau_gate"]["extension_decision"] == (
        "stop_at_k30"
    )
    assert result["sources"]["all_receipt_artifact_hashes_verified"] is True
    assert result["sources"]["unbound_run_file_count"] == 0


def test_authenticated_k50_continuation_supersedes_right_censored_k30_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_id = "eligible-id"
    k30_result = {
        "status": "authenticated_local_k30_right_censored_partial",
        "execution_id": execution_id,
        "regime_id": "weak_strong",
        "comparator_policy": "append_only",
        "target_horizon": 30,
        "terminal": {"k": 30, "error": 0.3},
        "effective_plateau_gate": {
            "extension_decision": "eligible_for_authenticated_resume_to_k50"
        },
    }
    k50_result = {
        "status": "completed_authenticated_local_k50_continuation",
        "execution_id": execution_id,
        "regime_id": "weak_strong",
        "comparator_policy": "append_only",
        "target_horizon": 50,
        "terminal": {"k": 50, "error": 0.1},
    }
    continuation = {
        "campaign_state": "runtime_materialized",
        "eligible_execution_ids": [execution_id],
        "stop_at_k30_execution_ids": [],
        "closed_execution_ids": [execution_id],
        "cell_states": {execution_id: "closed_authenticated_k50_continuation"},
        "completed": {execution_id: k50_result},
        "all_required_continuations_closed": True,
        "macro_terminal_authenticated": True,
        "sources": {},
    }

    merged = snapshot_pages._merge_authenticated_continuations(
        {execution_id: k30_result}, continuation
    )

    assert merged[execution_id] == k50_result
    assert merged[execution_id]["terminal"]["k"] == 50
    assert merged[execution_id]["target_horizon"] == 50


def test_pending_k50_continuation_preserves_authenticated_k30_partial() -> None:
    execution_id = "eligible-id"
    k30_result = {
        "status": "authenticated_local_k30_right_censored_partial",
        "execution_id": execution_id,
        "regime_id": "strong_strong_u8",
        "comparator_policy": "append_only",
        "target_horizon": 30,
        "terminal": {"k": 30, "error": 0.3},
        "effective_plateau_gate": {
            "extension_decision": "eligible_for_authenticated_resume_to_k50"
        },
    }
    continuation = {
        "eligible_execution_ids": [execution_id],
        "stop_at_k30_execution_ids": [],
        "closed_execution_ids": [],
        "completed": {},
    }

    merged = snapshot_pages._merge_authenticated_continuations(
        {execution_id: k30_result}, continuation
    )

    assert merged[execution_id] == k30_result
    assert merged[execution_id]["terminal"]["k"] == 30


def test_strong_reference_display_horizon_expands_for_closed_k50_trace() -> None:
    assert snapshot_pages._display_horizon(
        30,
        {
            "append_only": {
                "terminal": {"k": 50},
                "status": "completed_authenticated_local_k50_continuation",
            }
        },
    ) == 50


def test_macro_terminal_receipt_requires_exact_reauthenticated_closure(
    tmp_path: Path,
) -> None:
    conditional = ("local-a", "local-b")
    terminals = ("remote-a", "remote-b", "remote-c")
    terminal_status = {
        "status": "passed_all_three_authenticated_chtc_k50_terminals",
        "all_terminal_cells_authenticated": True,
        "sha256": "7" * 64,
    }

    class Adapter:
        CONDITIONAL_EXECUTION_IDS = conditional
        TERMINAL_CHTC_EXECUTION_IDS = terminals

        @staticmethod
        def terminal_chtc_status(*, cached: object) -> dict[str, object]:
            return terminal_status

    activation = {"sha256": "1" * 64}
    runtime = {
        "sha256": "2" * 64,
        "activation_manifest_sha256": activation["sha256"],
        "k30_runtime_manifest_sha256": "3" * 64,
    }
    decision = {
        "sha256": "4" * 64,
        "eligible_execution_ids": [conditional[0]],
        "stop_at_k30_execution_ids": [conditional[1]],
    }
    unsigned = {
        "schema": snapshot_pages.MACRO_TERMINAL_SCHEMA,
        "status": snapshot_pages.MACRO_TERMINAL_STATUS,
        "adapter_sha256": snapshot_pages.EXPECTED_CONTINUATION_ADAPTER_SHA256,
        "activation_manifest_sha256": activation["sha256"],
        "runtime_manifest_sha256": runtime["sha256"],
        "k30_runtime_manifest_sha256": runtime[
            "k30_runtime_manifest_sha256"
        ],
        "decision_status_sha256": decision["sha256"],
        "terminal_chtc_status_sha256": terminal_status["sha256"],
        "conditional_execution_ids": list(conditional),
        "terminal_chtc_k50_execution_ids": list(terminals),
        "eligible_k50_continuation_execution_ids": [conditional[0]],
        "stop_at_k30_execution_ids": [conditional[1]],
        "closed_k50_continuation_execution_ids": [conditional[0]],
        "all_k30_cells_closed": True,
        "all_extension_required_cells_closed_at_k50": True,
        "remaining_macro_execution_ids": [],
        "active_macro_execution_ids": [],
        "scientific_execution_performed_by_receipt": False,
    }
    receipt_path = tmp_path / "macro-terminal.json"
    receipt = _write_digested(receipt_path, unsigned)

    observed = snapshot_pages._authenticate_macro_terminal_receipt(
        continuation_adapter=Adapter(),
        activation=activation,
        runtime=runtime,
        decision_snapshot=decision,
        path=receipt_path,
    )
    assert observed == receipt

    drifted = dict(unsigned)
    drifted["terminal_chtc_status_sha256"] = "8" * 64
    _write_digested(receipt_path, drifted)
    with pytest.raises(snapshot_pages.UpdateError, match="terminal receipt drifted"):
        snapshot_pages._authenticate_macro_terminal_receipt(
            continuation_adapter=Adapter(),
            activation=activation,
            runtime=runtime,
            decision_snapshot=decision,
            path=receipt_path,
        )


def test_continuation_summary_closes_full_trace_and_authenticated_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    isolated_sealed_source_import_state: None,
) -> None:
    adapter = _continuation_adapter()
    worker = adapter.k30._load_worker()
    execution_id = adapter.CONDITIONAL_EXECUTION_IDS[0]
    job = adapter._job_by_id(worker)[execution_id]
    runtime_dir = tmp_path / "runtime"
    run_root = runtime_dir / "runs" / execution_id
    receipt_path = runtime_dir / "worker_receipts" / f"{execution_id}.json"
    exact = -2.0
    expected_derivation_kind = (
        "source_authorized_k50_protocol_reused_exactly"
        if int(job["target_horizon"]) >= 50
        else "source_locked_sole_horizon_delta_30_to_50"
    )
    trace = [
        {
            "controller_round": round_index,
            "accepted_energy": exact + (0.5 if round_index == 50 else 1.0),
            "absolute_energy_error": 0.5 if round_index == 50 else 1.0,
            "exact_same_cutoff_energy": exact,
            "active_ansatz_depth": round_index,
            "projective_state_fingerprint": (
                f"projective_state_v1:{round_index:064x}"
            ),
        }
        for round_index in range(1, 51)
    ]
    summary = {
        "schema": "paper_i_run_summary_v1",
        "horizon_scope": "deliberately_stopped_prefix",
        "available_controller_rounds": 50,
        "accepted_error_trace": trace,
        "effective_plateau": {
            "policy": "paper_i_effective_plateau_v1",
            "controller_round": 50,
            "absolute_energy_error": 0.5,
            "best_observed_error": 0.5,
            "selection_threshold": 0.55,
            "available_horizon_controller_rounds": 50,
            "horizon_scope": "deliberately_stopped_prefix",
        },
        "provenance": {"exact_same_cutoff_energy": exact},
    }
    summary_path = run_root / "summary/summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    resolved_protocol = _write_digested(
        run_root / "continuation/resolved_protocol.json",
        {"schema": "fixture_protocol", "horizon": 50},
    )
    authority = _write_digested(
        run_root / "continuation/resume_authorization.json",
        {
            "schema": adapter.RESUME_AUTHORIZATION_SCHEMA,
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "source_protocol_sha256": job["protocol_sha256"],
            "target_protocol": {
                "target_protocol_sha256": resolved_protocol["sha256"]
            },
            "route_contract_sha256": job["route_contract_sha256"],
            "comparator_policy": job["comparator_policy"],
            "regime_id": job["regime_id"],
            "resume_checkpoint": {"sha256": "4" * 64},
            "resume_checkpoint_siblings": [],
            "k30_plateau_gate_sha256": "9" * 64,
        },
    )
    prefix = {
        "status": "passed",
        "source_round": 30,
        "all_non_energy_fields_exact": True,
        "energy_comparison": "128_ulp_roundoff_only",
        "terminal_energy": trace[29]["accepted_energy"],
        "terminal_state_fingerprint": trace[29][
            "projective_state_fingerprint"
        ],
    }
    source_audit = _write_digested(
        run_root / "continuation/source_lock_audit.json",
        {
            "schema": "paper_i_page16_k30_to_k50_source_lock_audit_v2",
            "status": "passed",
            "execution_id": execution_id,
            "source_protocol_sha256": job["protocol_sha256"],
            "target_protocol_sha256": resolved_protocol["sha256"],
            "common_route_contract_sha256": job["route_contract_sha256"],
            "comparator_policy": job["comparator_policy"],
            "source_horizon": int(job["target_horizon"]),
            "resume_round": 30,
            "target_horizon": 50,
            "protocol_derivation_kind": expected_derivation_kind,
            "non_horizon_protocol_diff": [],
            "source_locks_exact": True,
            "resume_checkpoint_sha256": "4" * 64,
            "resume_checkpoint_siblings": [],
            "accepted_prefix_preservation": prefix,
        },
    )
    for relative, value in (
        ("result/result.json", {"result": "fixture"}),
        ("checkpoints/current.json", {"round": 50}),
        ("result/estimator_ledger.json", {"round": 50}),
    ):
        path = run_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
    payloads = {
        path.relative_to(run_root).as_posix(): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(run_root.rglob("*"))
        if path.is_file()
    }
    activation_sha = "6" * 64
    manifest = _write_digested(
        run_root / "execution_manifest.json",
        {
            "schema": adapter.EXECUTION_SCHEMA,
            "status": "passed",
            "execution_target": adapter.LOCAL_EXECUTION_TARGET,
            "source_package_manifest_sha256": adapter.k30.PACKAGE_MANIFEST_CANONICAL_SHA256,
            "adapter_sha256": snapshot_pages.EXPECTED_CONTINUATION_ADAPTER_SHA256,
            "activation_manifest_sha256": activation_sha,
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "resume_authorization_sha256": authority["sha256"],
            "source_protocol_sha256": job["protocol_sha256"],
            "protocol_sha256": resolved_protocol["sha256"],
            "protocol_derivation_kind": expected_derivation_kind,
            "route_contract_sha256": job["route_contract_sha256"],
            "comparator_policy": job["comparator_policy"],
            "resume_round": 30,
            "target_horizon": 50,
            "controller_rounds_completed": 50,
            "source_checkpoint_sha256": "4" * 64,
            "source_plateau_gate_sha256": "9" * 64,
            "accepted_state_resume": True,
            "fresh_start": False,
            "accepted_prefix_preservation": prefix,
            "source_lock_audit_sha256": source_audit["sha256"],
            "output_payloads": payloads,
            "paper_evidence_adoption_authorized": False,
        },
    )
    artifacts = [
        {
            "path": path.relative_to(runtime_dir).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(run_root.rglob("*"))
        if path.is_file()
    ]
    _write_digested(
        receipt_path,
        {
            "schema": adapter.WORKER_RECEIPT_SCHEMA,
            "status": "passed",
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "resume_authorization_sha256": authority["sha256"],
            "execution_manifest_sha256": manifest["sha256"],
            "resume_round": 30,
            "controller_rounds_completed": 50,
            "accepted_state_resume": True,
            "fresh_start": False,
            "artifacts": artifacts,
        },
    )
    monkeypatch.setattr(
        snapshot_pages.completed_pages,
        "_compile_cost_tuple",
        lambda _summary, *, round_index: (
            {"N2q": 1, "D2q": 2, "Dc": 3, "W1q": 4, "S_alg": 5},
            {"round": round_index},
        ),
    )
    source_points = [
        {
            "k": row["controller_round"],
            "energy": row["accepted_energy"],
            "error": row["absolute_energy_error"],
            "active_ansatz_depth": row["active_ansatz_depth"],
            "projective_state_fingerprint": row[
                "projective_state_fingerprint"
            ],
        }
        for row in trace[:30]
    ]

    result = snapshot_pages._continuation_summary_result(
        continuation_adapter=adapter,
        runtime_dir=runtime_dir,
        runtime={"activation_manifest_sha256": activation_sha},
        execution_id=execution_id,
        authority=authority,
        source_result={
            "points": source_points,
            "effective_plateau_gate": {
                "canonical_sha256": "9" * 64,
                "extension_decision": "eligible_for_authenticated_resume_to_k50",
            },
        },
        compile_costs=True,
    )

    assert result["status"] == "completed_authenticated_local_k50_continuation"
    assert result["target_horizon"] == 50
    assert result["resume_round"] == 30
    assert result["source_authorized_horizon"] == int(job["target_horizon"])
    assert "source_horizon" not in result
    assert result["terminal"]["k"] == 50
    assert result["effective_plateau"]["selected_controller_round"] == 50
    assert result["sources"]["accepted_prefix_preservation_authenticated"] is True

    source_audit_path = run_root / "continuation/source_lock_audit.json"
    drifted_audit = dict(source_audit)
    drifted_audit.pop("sha256")
    drifted_audit["source_horizon"] = 999
    drifted_audit = _write_digested(
        run_root / "continuation/source_lock_audit.json", drifted_audit
    )
    drifted_manifest = dict(manifest)
    drifted_manifest.pop("sha256")
    drifted_manifest["source_lock_audit_sha256"] = drifted_audit["sha256"]
    audit_binding = drifted_manifest["output_payloads"][
        "continuation/source_lock_audit.json"
    ]
    audit_binding.update(
        {
            "sha256": hashlib.sha256(source_audit_path.read_bytes()).hexdigest(),
            "size_bytes": source_audit_path.stat().st_size,
        }
    )
    drifted_manifest = _write_digested(
        run_root / "execution_manifest.json", drifted_manifest
    )
    drifted_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    drifted_receipt.pop("sha256")
    drifted_receipt["execution_manifest_sha256"] = drifted_manifest["sha256"]
    for row in drifted_receipt["artifacts"]:
        path = runtime_dir / row["path"]
        row.update(
            {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    _write_digested(receipt_path, drifted_receipt)

    with pytest.raises(snapshot_pages.UpdateError, match="source-lock audit drifted"):
        snapshot_pages._continuation_summary_result(
            continuation_adapter=adapter,
            runtime_dir=runtime_dir,
            runtime={"activation_manifest_sha256": activation_sha},
            execution_id=execution_id,
            authority=authority,
            source_result={
                "points": source_points,
                "effective_plateau_gate": {
                    "canonical_sha256": "9" * 64,
                    "extension_decision": (
                        "eligible_for_authenticated_resume_to_k50"
                    ),
                },
            },
            compile_costs=False,
        )


def test_append_and_replace_snapshot_pages_preserve_first_sixteen_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pypdf = pytest.importorskip("pypdf")
    (
        target_pdf,
        target_provenance,
        page17_pdf,
        page17_png,
        adapter_path,
    ) = _patch_paths(monkeypatch, tmp_path)
    _write_pdf(
        target_pdf,
        [f"q {index} 0 1 1 re f Q\n".encode() for index in range(1, 19)],
    )
    _write_pdf(page17_pdf, [b"q 17 0 1 1 re f Q\n"])
    page17_png.write_bytes(b"page17 fixture")
    adapter = _adapter(adapter_path, revision="a")
    provenance: dict[str, object] = {
        "schema": "fixture",
        "layout": {
            "page_count": 18,
            "page_16": snapshot_pages.PAGE16_ID,
            "page_17": snapshot_pages.LEGACY_PAGE17_ID,
            "page_18": snapshot_pages.LEGACY_PAGE18_ID,
        },
        "outputs": {
            "partial_progress_pdf": snapshot_pages.binding(target_pdf),
            "insertion_comparator_snapshot_page18_pdf": {"stale": True},
            "insertion_comparator_snapshot_page18_png": {"stale": True},
        },
        "phase0_insertion_comparator_live_snapshot": {"stale": True},
    }
    target_provenance.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    preserved = _content_hashes(target_pdf)[:16]

    result = snapshot_pages.append_or_replace_pages(adapter, provenance)

    updated = json.loads(target_provenance.read_text())
    assert result["page_count"] == 17
    assert len(pypdf.PdfReader(str(target_pdf), strict=False).pages) == 17
    assert _content_hashes(target_pdf)[:16] == preserved
    assert updated["layout"]["page_17"] == snapshot_pages.PAGE17_ID
    assert "page_18" not in updated["layout"]
    assert updated["layout"]["page_count"] == 17
    assert result["completed_comparator_count"] == 2
    assert "phase0_insertion_comparator_snapshot" in updated
    assert "phase0_insertion_comparator_live_snapshot" not in updated
    assert "insertion_comparator_snapshot_page18_pdf" not in updated["outputs"]
    assert "insertion_comparator_snapshot_page18_png" not in updated["outputs"]
    old_snapshot_hash = _content_hashes(target_pdf)[16]

    _write_pdf(page17_pdf, [b"q 117 0 2 2 re f Q\n"])
    replacement = _adapter(adapter_path, revision="b")
    current = json.loads(target_provenance.read_text())
    result = snapshot_pages.append_or_replace_pages(replacement, current)

    hashes = _content_hashes(target_pdf)
    assert result["page_count"] == 17
    assert hashes[:16] == preserved
    assert len(hashes) == 17
    assert hashes[16] != old_snapshot_hash
