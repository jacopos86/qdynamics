"""Run the exploratory L=4 AVQE/RA route matrix with bounded concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


RUNNER = Path(__file__).with_name("run_l4_arm.py")


def _command(output: Path, horizon: int, config: tuple[str, ...]) -> list[str]:
    if config == ("avqe",):
        return [
            sys.executable,
            str(RUNNER),
            "avqe",
            str(output),
            "--horizon",
            str(horizon),
        ]
    selector, chart, residual = config
    return [
        sys.executable,
        str(RUNNER),
        "baseline",
        str(output),
        "--horizon",
        str(horizon),
        "--selector-cost",
        selector,
        "--refit-chart",
        chart,
        "--active-residual",
        residual,
    ]


def _run(output: Path, horizon: int, config: tuple[str, ...]) -> dict[str, object]:
    name = "avqe" if config == ("avqe",) else "_".join(config)
    log_path = output / f"{name}.log"
    completed = subprocess.run(
        _command(output, horizon, config),
        cwd=RUNNER.parents[2],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "name": name,
        "returncode": completed.returncode,
        "log": str(log_path),
        "last_line": completed.stdout.strip().splitlines()[-1]
        if completed.stdout.strip()
        else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configs = [("avqe",)] + [
        (selector, chart, residual)
        for selector in ("qiskit", "proxy")
        for chart in ("native", "whitened")
        for residual in ("measured", "stationary")
    ]
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run, args.output, args.horizon, config): config
            for config in configs
        }
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    results.sort(key=lambda row: str(row["name"]))
    manifest = {
        "schema": "paper_i_exploratory_l4_route_matrix_execution_v1",
        "horizon": args.horizon,
        "maximum_concurrent_processes": args.workers,
        "runs": results,
    }
    (args.output / "matrix_execution.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    failed = [row for row in results if int(row["returncode"]) != 0]
    if failed:
        raise SystemExit(f"{len(failed)} L=4 matrix runs failed")


if __name__ == "__main__":
    main()
