from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"
CONTROLLER_STATUS = ROOT / "controller.status.json"
SUMMARY_JSON = ROOT / "overnight_summary.json"
SUMMARY_MD = ROOT / "overnight_summary.md"
STATUS_JSON = ROOT / "summary_watcher.status.json"
ARTIFACT_JSON_DIR = ROOT.parents[1] / "json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def done(status: dict[str, Any], total: int) -> bool:
    return (
        len(status.get("completed", [])) + len(status.get("failed", [])) == total
        and not status.get("running")
        and not status.get("queue_remaining")
    )


def stage_metrics(doc: dict[str, Any], stage_key: str) -> dict[str, Any]:
    stage = ((doc.get("stage_pipeline") or {}).get(stage_key) or {})
    cm = (((doc.get("circuit_metrics") or {}).get("stages") or {}).get(stage_key) or {})
    transpiled = ((((cm.get("metadata") or {}).get("transpile_metrics") or {}).get("transpiled")) or {})
    return {
        "delta_abs": stage.get("delta_abs"),
        "depth": transpiled.get("depth"),
        "cx_count": transpiled.get("cx_count"),
    }


def build_summary() -> dict[str, Any]:
    manifest = load_json(MANIFEST)
    rows: list[dict[str, Any]] = []
    for run in manifest["runs"]:
        tag = run["tag"]
        status_path = ROOT / f"{tag}.status.json"
        status = load_json(status_path) if status_path.exists() else {"status": "missing"}
        final_json = ARTIFACT_JSON_DIR / f"{tag}.json"
        row: dict[str, Any] = {
            "tag": tag,
            "kind": run["kind"],
            "adapt_max_depth": run["adapt_max_depth"],
            "beam_live_branches": run["beam_live_branches"],
            "split_mode": run["split_mode"],
            "status": status.get("status"),
            "final_json": str(final_json) if final_json.exists() else None,
        }
        if final_json.exists():
            doc = load_json(final_json)
            warm = stage_metrics(doc, "warm_start")
            adapt = stage_metrics(doc, "adapt_vqe")
            replay = stage_metrics(doc, "conventional_replay")
            row.update({
                "warm": warm,
                "adapt": adapt,
                "replay": replay,
            })
            row["passes_energy"] = (replay.get("delta_abs") is not None and replay["delta_abs"] <= 1e-3)
            row["passes_depth_budget"] = all(
                item.get("depth") is not None and item["depth"] <= 400
                for item in (warm, adapt, replay)
            )
            row["passes_all"] = bool(row["passes_energy"] and row["passes_depth_budget"])
        else:
            row.update({
                "warm": None,
                "adapt": None,
                "replay": None,
                "passes_energy": False,
                "passes_depth_budget": False,
                "passes_all": False,
            })
        rows.append(row)

    ranked = sorted(
        rows,
        key=lambda r: (
            0 if r["passes_all"] else 1,
            (r.get("replay") or {}).get("depth") if (r.get("replay") or {}).get("depth") is not None else 10**9,
            (r.get("replay") or {}).get("cx_count") if (r.get("replay") or {}).get("cx_count") is not None else 10**9,
            (r.get("replay") or {}).get("delta_abs") if (r.get("replay") or {}).get("delta_abs") is not None else 10**9,
            r["tag"],
        ),
    )
    winners = [r for r in ranked if r["passes_all"]]
    return {
        "generated_utc": utc_now(),
        "budget": {"max_depth_per_stage": 400, "replay_delta_abs_max": 1e-3},
        "total_runs": len(rows),
        "passing_runs": len(winners),
        "best_tag": winners[0]["tag"] if winners else None,
        "runs": ranked,
    }


def write_markdown(summary: dict[str, Any]) -> None:
    lines = []
    lines.append(f"# Overnight shallow HH summary\n")
    lines.append(f"Generated: {summary['generated_utc']}\n")
    lines.append(f"Budget: replay ΔE <= 1e-3; warm/adapt/replay depth <= 400\n")
    lines.append(f"Passing runs: {summary['passing_runs']} / {summary['total_runs']}\n")
    if summary["best_tag"]:
        lines.append(f"Best tag: `{summary['best_tag']}`\n")
    else:
        lines.append("Best tag: none\n")
    lines.append("\n## Ranked runs\n")
    for row in summary["runs"]:
        lines.append(f"- `{row['tag']}` status={row['status']} passes_all={row['passes_all']} kind={row['kind']} depth_cap={row['adapt_max_depth']} beam={row['beam_live_branches']} split={row['split_mode']}")
        if row["replay"]:
            lines.append(
                f"  - replay: delta={row['replay']['delta_abs']} depth={row['replay']['depth']} cx={row['replay']['cx_count']}"
            )
            lines.append(
                f"  - adapt: delta={row['adapt']['delta_abs']} depth={row['adapt']['depth']} cx={row['adapt']['cx_count']}"
            )
            lines.append(
                f"  - warm: delta={row['warm']['delta_abs']} depth={row['warm']['depth']} cx={row['warm']['cx_count']}"
            )
        else:
            lines.append("  - final json missing")
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    manifest = load_json(MANIFEST)
    total = len(manifest["runs"])
    write_json(STATUS_JSON, {"status": "waiting", "started_utc": utc_now(), "total_runs": total})
    while True:
        if CONTROLLER_STATUS.exists():
            status = load_json(CONTROLLER_STATUS)
            write_json(STATUS_JSON, {
                "status": "watching",
                "updated_utc": utc_now(),
                "controller_status": status,
            })
            if done(status, total):
                break
        time.sleep(30)
    summary = build_summary()
    write_json(SUMMARY_JSON, summary)
    write_markdown(summary)
    write_json(STATUS_JSON, {
        "status": "completed",
        "updated_utc": utc_now(),
        "summary_json": str(SUMMARY_JSON),
        "summary_md": str(SUMMARY_MD),
        "passing_runs": summary["passing_runs"],
        "best_tag": summary["best_tag"],
    })


if __name__ == "__main__":
    main()
