#!/usr/bin/env python3
"""Run the five source-matched Hubbard L=4 arms sequentially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ARM = Path(__file__).resolve().with_name("run_arm.py")
RAM_GUARD = REPO_ROOT / "pipelines" / "shell" / "ram_guard.py"
ARMS = ("append", "baseline", "prune", "batch", "beam")


def _require_empty_arm_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Refusing to splice or overwrite a nonempty arm directory: {path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        _require_empty_arm_directory(output_root / arm)

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "CUDA_VISIBLE_DEVICES": "",
            "STATIC_ADAPT_ALLOCATED_CPUS": "1",
            "QISKIT_NUM_PROCS": "1",
            "QISKIT_PARALLEL": "FALSE",
            "RAYON_NUM_THREADS": "1",
        }
    )
    for arm in ARMS:
        print(
            json.dumps(
                {
                    "event": "paper_i_hubbard_l4_arm_start",
                    "arm": arm,
                    "rounds": int(args.rounds),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(RAM_GUARD),
                "--limit-mb",
                "8000",
                "--",
                sys.executable,
                "-u",
                str(RUN_ARM),
                "--arm",
                arm,
                "--rounds",
                str(args.rounds),
                "--output-root",
                str(output_root),
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
        print(
            json.dumps(
                {"event": "paper_i_hubbard_l4_arm_complete", "arm": arm},
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
