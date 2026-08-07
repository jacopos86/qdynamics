from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.pareto_offline import proposal_cycle_autopilot


def test_path_b_autopilot_once_records_summary(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "proposal_cycle_manifest.json"
    manifest_path.write_text(json.dumps({"cases": []}), encoding="utf-8")
    state_path = tmp_path / "state.json"

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        proposal_cycle_autopilot,
        "run_proposal_cycle",
        lambda **kwargs: calls.append(dict(kwargs)) or {"run_tag": kwargs["tag"], "proposal_packet_count": 3, "summary_json": "x", "include_path_a_discovery": kwargs.get("include_path_a_discovery")},
    )

    payload = proposal_cycle_autopilot.run_autopilot_once(
        manifest_path=manifest_path,
        tag_prefix="hh_path_b_proposal_cycle",
        state_path=state_path,
        path_a_run_root=tmp_path / "artifacts" / "agent_runs",
        path_a_max_cases=4,
    )

    assert payload["status"] == "cycled"
    assert payload["summary"]["proposal_packet_count"] == 3
    assert payload["path_a_max_cases"] == 4
    assert calls[0]["include_path_a_discovery"] is True
    assert calls[0]["path_a_max_cases"] == 4
    assert state_path.exists()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["summary"]["proposal_packet_count"] == 3
