"""Validate and execute one source-locked L=4 campaign cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_signed_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = str(payload.pop("sha256"))
    observed = _canonical_sha256(payload)
    if observed != expected:
        raise RuntimeError(f"digest mismatch for {path}: {observed} != {expected}")
    payload["sha256"] = expected
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--selector-cost", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = _verify_signed_payload(args.plan)
    authorization = _verify_signed_payload(args.authorization)
    if not authorization.get("execution_authorized") or not authorization.get(
        "submission_authorized"
    ):
        raise RuntimeError("execution is not authorized")
    if authorization.get("plan_sha256") != plan.get("sha256"):
        raise RuntimeError("authorization does not bind this plan")
    if args.execution_id not in authorization.get("execution_ids", []):
        raise RuntimeError("execution id is outside the authorization")
    cells = {
        str(cell["execution_id"]): cell
        for cell in plan["matrix"]["cells"]
    }
    cell = cells.get(args.execution_id)
    if cell is None:
        raise RuntimeError("execution id is absent from the plan")
    expected_selector = cell.get("selector_cost") or "none"
    if str(cell["arm"]) != args.arm or str(expected_selector) != args.selector_cost:
        raise RuntimeError("queue row drifted from the source-locked cell")

    args.output.mkdir(parents=True, exist_ok=False)
    runner = (
        args.source_root
        / "chtc/ra_adapt_l4_route_matrix_20260825/run_l4_arm.py"
    )
    command = [
        sys.executable,
        str(runner),
        args.arm,
        str(args.output),
        "--horizon",
        "50",
        "--powell-xtol",
        "1e-5",
        "--powell-ftol",
        "1e-12",
        "--variant-label",
        args.execution_id,
    ]
    if args.arm != "avqe":
        command.extend(
            [
                "--selector-cost",
                args.selector_cost,
                "--refit-chart",
                "whitened",
                "--active-residual",
                "stationary",
            ]
        )
    if args.arm == "metric_ablation_gated":
        command.extend(
            [
                "--prune-theta",
                "1e-3",
                "--prune-protect",
                "0",
                "--prune-max-regression",
                "1e-8",
                "--prune-max-predicted-loss-per-cost",
                "1e-8",
            ]
        )
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    result_dirs = [path for path in args.output.iterdir() if path.is_dir()]
    if len(result_dirs) != 1:
        raise RuntimeError(f"expected one result directory, found {result_dirs}")
    result_dir = result_dirs[0]
    summary_path = result_dir / "summary.json"
    result_path = result_dir / "result.json"
    if not summary_path.is_file() or not result_path.is_file():
        raise RuntimeError("completed cell lacks result or summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    receipt = {
        "schema": "paper_i_l4_qiskit_proxy_execution_receipt_v1",
        "campaign_id": plan["campaign_id"],
        "execution_id": args.execution_id,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan_sha256": plan["sha256"],
        "authorization_sha256": authorization["sha256"],
        "arm": args.arm,
        "selector_cost": None if args.selector_cost == "none" else args.selector_cost,
        "horizon": 50,
        "accepted_rounds": summary.get("accepted_rounds", summary.get("horizon")),
        "summary_sha256": _sha256_file(summary_path),
        "result_sha256": _sha256_file(result_path),
        "completion_status": "completed",
    }
    (args.output.parent / "execution_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
