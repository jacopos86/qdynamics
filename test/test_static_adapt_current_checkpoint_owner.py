from __future__ import annotations

import ast
import hashlib
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import pytest

import pipelines.static_adapt.current_checkpoint as current_checkpoint


def _checkpoint_payload(
    *,
    depth: int,
    route_family: str = "singleton_response_snake",
) -> dict[str, Any]:
    history = [
        {
            "depth": index,
            "pool_index": index - 1,
            "selected_batch_labels": [f"X{index}"],
            "selected_pool_indices": [index - 1],
            "selected_logical_size": 1,
            "selected_feature_rows": [
                {
                    "controller_snapshot": {
                        "controller_round": index,
                    }
                }
            ],
        }
        for index in range(1, depth + 1)
    ]
    if route_family in {
        "greedy_batch_response_snake",
        "combinatorial_batch_response_snake",
    }:
        admission_field = (
            "greedy_batch_admission"
            if route_family == "greedy_batch_response_snake"
            else "combinatorial_batch_admission"
        )
        for index, row in enumerate(history, start=1):
            row["active_prefix_checkpoint"] = {
                "schema": "test_active_prefix_checkpoint_v1",
                "depth": index,
            }
            row[admission_field] = {
                "composition_identity": f"batch-{index}",
                "selected_record_ids": [f"record-{index}"],
                "selected_generator_ids": [f"generator-{index}"],
                "selected_original_positions": [index - 1],
                "selected_effective_positions": [index - 1],
            }
    return {
        "adapt_vqe": {
            "route_family": route_family,
            "history": history,
            "history_count": depth,
            "pool_size": 8,
            "operators": [f"X{index}" for index in range(1, depth + 1)],
            "estimator_call_accounting": {
                "S_alg": depth,
                "S_unique": depth,
            },
            "terminal_active_prefix_checkpoint": {
                "schema": "test_active_prefix_checkpoint_v1",
                "depth": depth,
            },
        }
    }


def _ledger_payload(*, depth: int) -> dict[str, Any]:
    return {
        "schema": "estimator_call_ledger_v1",
        "ledger_fingerprint": f"ledger-at-depth-{depth}",
        "summary": {
            "unique_primitive_count": depth,
            "S_unique": depth,
        },
        "occurrence_summary": {
            "total_call_occurrences": depth,
        },
    }


def _published_sidecar_paths(
    current_path: Path,
    *,
    route_pointer_field: str = "verified_singleton_resume_sidecar",
) -> set[Path]:
    payload = json.loads(current_path.read_text(encoding="utf-8"))
    adapt = payload["adapt_vqe"]
    return {
        current_path.with_name(adapt[pointer_name]["path"])
        for pointer_name in (
            "estimator_call_ledger_checkpoint",
            route_pointer_field,
        )
    }


def test_neutral_checkpoint_helpers_are_owned_by_current_checkpoint() -> None:
    assert (
        current_checkpoint._publish_active_cli_current_checkpoint.__module__
        == "pipelines.static_adapt.current_checkpoint"
    )
    assert (
        current_checkpoint._stable_json_digest.__module__
        == "pipelines.static_adapt.current_checkpoint"
    )


def test_singleton_checkpoint_honors_bounded_history_tail(
    tmp_path: Path,
) -> None:
    current_path = tmp_path / "current.json"
    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=50),
        ledger_payload=_ledger_payload(depth=50),
        path=current_path,
        keep_history_tail=2,
    )

    adapt = json.loads(current_path.read_text(encoding="utf-8"))["adapt_vqe"]
    assert adapt["history_count"] == 50
    assert len(adapt["history"]) == 50
    assert adapt["history_tail_count"] == 2
    assert adapt["history_tail"] == adapt["history"][-2:]
    assert adapt["history_tail_retention"] == {
        "schema": "static_adapt_verified_resume_history_retention_v2",
        "requested_limit": 2,
        "requested_window_count": 2,
        "serialized_complete_history_count": 50,
        "serialized_tail_count": 2,
        "normalized_for_verified_singleton_resume": True,
        "no_credentials_serialized": True,
    }


def test_checkpoint_publication_streams_whole_json_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_dumps = current_checkpoint.json.dumps

    def _reject_whole_document_dumps(value: Any, *args: Any, **kwargs: Any) -> str:
        schema = value.get("schema") if isinstance(value, dict) else None
        if isinstance(value, dict) and (
            "adapt_vqe" in value
            or schema
            in {
                "paper_i_estimator_call_ledger_checkpoint_sidecar_v2",
                "static_adapt_signed_active_prefix_resume_sidecar_v2",
            }
        ):
            raise AssertionError("whole checkpoint documents must stream")
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(current_checkpoint.json, "dumps", _reject_whole_document_dumps)
    current_path = tmp_path / "current.json"
    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=3),
        ledger_payload=_ledger_payload(depth=3),
        path=current_path,
        keep_history_tail=1,
    )

    assert current_path.is_file()
    assert json.loads(current_path.read_text(encoding="utf-8"))[
        "adapt_vqe"
    ]["history_count"] == 3


def test_checkpoint_publication_does_not_deepcopy_history_or_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_deepcopy = current_checkpoint.copy.deepcopy

    def _reject_large_deepcopies(value: Any, memo: Any = None) -> Any:
        if isinstance(value, dict) and (
            "history" in value
            or value.get("schema") == "estimator_call_ledger_v1"
        ):
            raise AssertionError("history and ledger must not be deep-copied")
        return original_deepcopy(value, memo)

    monkeypatch.setattr(
        current_checkpoint.copy,
        "deepcopy",
        _reject_large_deepcopies,
    )
    current_path = tmp_path / "current.json"
    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=3),
        ledger_payload=_ledger_payload(depth=3),
        path=current_path,
        keep_history_tail=1,
    )

    current = json.loads(current_path.read_text(encoding="utf-8"))
    sidecar_path = current_path.with_name(
        current["adapt_vqe"]["estimator_call_ledger_checkpoint"]["path"]
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["consumer_complete_projection"] == {
        "schema": "static_adapt_consumer_projection_reference_v1",
        "source_projection_sha256": sidecar[
            "consumer_complete_projection"
        ]["source_projection_sha256"],
        "source_projection_digest_scope": "static_adapt_full_projection_v1",
        "materialized_in": "current_checkpoint.adapt_vqe",
        "embedded_full_ledgers_omitted": True,
    }


def test_current_checkpoint_defines_its_neutral_helpers() -> None:
    owner_path = Path(current_checkpoint.__file__)
    tree = ast.parse(owner_path.read_text(encoding="utf-8"))
    defined_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "_publish_active_cli_current_checkpoint",
        "_stable_json_digest",
    }.issubset(defined_functions)


def test_publishing_successor_retires_unreferenced_predecessor_sidecars(
    tmp_path: Path,
) -> None:
    current_path = tmp_path / "current.json"
    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=1),
        ledger_payload=_ledger_payload(depth=1),
        path=current_path,
        keep_history_tail=1,
    )
    predecessor_sidecars = _published_sidecar_paths(current_path)
    assert all(path.is_file() for path in predecessor_sidecars)

    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=2),
        ledger_payload=_ledger_payload(depth=2),
        path=current_path,
        keep_history_tail=1,
    )

    successor_sidecars = _published_sidecar_paths(current_path)
    assert predecessor_sidecars.isdisjoint(successor_sidecars)
    assert all(path.is_file() for path in successor_sidecars)
    assert all(not path.exists() for path in predecessor_sidecars)


@pytest.mark.parametrize(
    ("route_family", "route_pointer_field"),
    (
        (
            "greedy_batch_response_snake",
            "greedy_batch_checkpoint_sidecar",
        ),
        (
            "combinatorial_batch_response_snake",
            "combinatorial_batch_checkpoint_sidecar",
        ),
    ),
)
def test_publishing_successor_retires_predecessor_batch_route_sidecar(
    tmp_path: Path,
    route_family: str,
    route_pointer_field: str,
) -> None:
    current_path = tmp_path / "current.json"
    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=1, route_family=route_family),
        ledger_payload=_ledger_payload(depth=1),
        path=current_path,
        keep_history_tail=1,
    )
    predecessor_sidecars = _published_sidecar_paths(
        current_path,
        route_pointer_field=route_pointer_field,
    )

    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=2, route_family=route_family),
        ledger_payload=_ledger_payload(depth=2),
        path=current_path,
        keep_history_tail=1,
    )

    successor_sidecars = _published_sidecar_paths(
        current_path,
        route_pointer_field=route_pointer_field,
    )
    assert predecessor_sidecars.isdisjoint(successor_sidecars)
    assert all(path.is_file() for path in successor_sidecars)
    assert all(not path.exists() for path in predecessor_sidecars)


def test_sidecar_retention_preserves_unreferenced_and_tampered_evidence(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    current_path = tmp_path / "current.json"
    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=1),
        ledger_payload=_ledger_payload(depth=1),
        path=current_path,
        keep_history_tail=1,
    )
    predecessor_sidecars = _published_sidecar_paths(current_path)
    tampered_sidecar = next(
        path
        for path in predecessor_sidecars
        if ".estimator_call_ledger_checkpoint." in path.name
    )
    tampered_sidecar.write_text("tampered failure evidence\n", encoding="utf-8")
    unrelated_sidecar = (
        tmp_path / "current.verified_singleton_resume.0000000000000000.json"
    )
    unrelated_sidecar.write_text("unreferenced evidence\n", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with caplog.at_level(
            logging.WARNING,
            logger=current_checkpoint.__name__,
        ):
            current_checkpoint._publish_active_cli_current_checkpoint(
                _checkpoint_payload(depth=2),
                ledger_payload=_ledger_payload(depth=2),
                path=current_path,
                keep_history_tail=1,
            )

    assert tampered_sidecar.read_text(encoding="utf-8") == (
        "tampered failure evidence\n"
    )
    assert unrelated_sidecar.read_text(encoding="utf-8") == (
        "unreferenced evidence\n"
    )
    assert all(
        not path.exists()
        for path in predecessor_sidecars
        if path != tampered_sidecar
    )
    assert any(
        "digest mismatch" in message for message in caplog.messages
    )


def test_disagreeing_ledger_pointer_owners_preserve_both_sidecars(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    current_path = tmp_path / "current.json"
    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=1),
        ledger_payload=_ledger_payload(depth=1),
        path=current_path,
        keep_history_tail=1,
    )
    predecessor = json.loads(current_path.read_text(encoding="utf-8"))
    adapt_ledger_path = current_path.with_name(
        predecessor["adapt_vqe"]["estimator_call_ledger_checkpoint"]["path"]
    )
    disagreement_bytes = b"disagreeing pointer failure evidence\n"
    disagreement_sha256 = hashlib.sha256(disagreement_bytes).hexdigest()
    disagreement_path = current_path.with_name(
        "current.estimator_call_ledger_checkpoint."
        f"{disagreement_sha256[:16]}.json"
    )
    disagreement_path.write_bytes(disagreement_bytes)
    predecessor["checkpoint"]["estimator_call_ledger_checkpoint"].update(
        {
            "path": disagreement_path.name,
            "sha256": disagreement_sha256,
        }
    )
    current_path.write_text(
        json.dumps(predecessor, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with caplog.at_level(
        logging.WARNING,
        logger=current_checkpoint.__name__,
    ):
        current_checkpoint._publish_active_cli_current_checkpoint(
            _checkpoint_payload(depth=2),
            ledger_payload=_ledger_payload(depth=2),
            path=current_path,
            keep_history_tail=1,
        )

    assert adapt_ledger_path.is_file()
    assert disagreement_path.read_bytes() == disagreement_bytes
    assert any(
        "checkpoint owners disagree" in message
        for message in caplog.messages
    )


def test_failed_successor_publication_preserves_predecessor_and_new_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_path = tmp_path / "current.json"
    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=1),
        ledger_payload=_ledger_payload(depth=1),
        path=current_path,
        keep_history_tail=1,
    )
    predecessor_bytes = current_path.read_bytes()
    predecessor_sidecars = _published_sidecar_paths(current_path)
    original_replace = current_checkpoint.os.replace

    def _fail_current_publication(source: str, destination: str) -> None:
        if Path(destination) == current_path:
            raise OSError("test current-checkpoint publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(
        current_checkpoint.os,
        "replace",
        _fail_current_publication,
    )

    with pytest.raises(
        OSError,
        match="test current-checkpoint publication failure",
    ):
        current_checkpoint._publish_active_cli_current_checkpoint(
            _checkpoint_payload(depth=2),
            ledger_payload=_ledger_payload(depth=2),
            path=current_path,
            keep_history_tail=1,
        )

    assert current_path.read_bytes() == predecessor_bytes
    assert all(path.is_file() for path in predecessor_sidecars)
    assert len(
        tuple(
            tmp_path.glob(
                "current.estimator_call_ledger_checkpoint.*.json"
            )
        )
    ) == 2
    assert len(
        tuple(tmp_path.glob("current.verified_singleton_resume.*.json"))
    ) == 2
