from __future__ import annotations

import argparse
from typing import Sequence

from pipelines.excited_dynamics.io import write_excited_state_seed_from_qse_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build diagnostic excited-state seed manifests from QSE results.")
    parser.add_argument("--qse-result-json", required=True, help="Input qse_spectra_v1 manifest JSON.")
    parser.add_argument("--state-index", required=True, type=int, help="QSE Ritz state index to select.")
    parser.add_argument("--output-json", required=True, help="Output excited_state_seed_v1 JSON path.")
    parser.add_argument(
        "--allow-ground-state",
        action="store_true",
        help="Allow state-index 0. By default, seed generation rejects the QSE ground Ritz state.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    write_excited_state_seed_from_qse_json(
        qse_json_path=args.qse_result_json,
        output_json_path=args.output_json,
        state_index=args.state_index,
        allow_ground_state=args.allow_ground_state,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
