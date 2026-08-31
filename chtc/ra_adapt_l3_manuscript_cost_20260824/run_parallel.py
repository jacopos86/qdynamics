"""Run the five noiseless Paper-I L=3 arms with two local workers."""

from __future__ import annotations

import concurrent.futures
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results"
ARMS = ("avqe", "baseline", "batching", "beam_escalated", "metric_ablation_gated")


def run(arm: str) -> tuple[str, int]:
    log = OUTPUT / f"{arm}.log"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        completed = subprocess.run(
            [
                sys.executable,
                str(HERE / "run_arm.py"),
                arm,
                str(OUTPUT),
                "--horizon",
                "12",
            ],
            cwd=HERE.parents[1],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return arm, completed.returncode


def main() -> None:
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(run, arm): arm for arm in ARMS}
        for future in concurrent.futures.as_completed(futures):
            arm, returncode = future.result()
            print(f"{arm}: returncode={returncode}", flush=True)
            if returncode:
                failures.append(arm)
    if failures:
        raise SystemExit("failed arms: " + ", ".join(failures))


if __name__ == "__main__":
    main()
