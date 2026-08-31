"""CLI for adopting a verified bootstrap AVQDS prefix after exact completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipelines.time_dynamics.prospective_avqds_scout_adoption import (
    ScoutAdoptionReceipt,
    adopt_completed_scout,
)
from pipelines.time_dynamics.prospective_avqds_scout_worker import (
    load_frozen_submitted_work_order,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ExactReferenceReceipt,
    build_scientific_bindings,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Adopt or validate a byte-verified bootstrap AVQDS scout in the "
            "canonical scientific lifecycle."
        )
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--work-order", type=Path, required=True)
    parser.add_argument("--reference-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--validate-receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.repository_root.resolve()
    order, source, seed = load_frozen_submitted_work_order(
        arguments.work_order, repository_root=root
    )
    reference = ExactReferenceReceipt.load(
        arguments.reference_receipt,
        repository_root=root,
        expected_reference=order.reference,
        expected_seed_identity=order.seed,
        expected_seed_receipt=seed,
        expected_source_epoch=source.source_epoch,
    )
    bindings = build_scientific_bindings(
        reference=order.reference,
        seed=order.seed,
        seed_receipt=seed,
        reference_receipt=reference,
        source_epoch=source.source_epoch,
        source_epoch_receipt=source,
    )
    if arguments.validate_receipt is None:
        receipt = adopt_completed_scout(
            bootstrap_order=order,
            source_receipt=source,
            seed_receipt=seed,
            reference_receipt=reference,
            bindings=bindings,
            repository_root=root,
            output_directory=arguments.output_directory,
        )
    else:
        receipt = ScoutAdoptionReceipt.load(
            arguments.validate_receipt,
            repository_root=root,
            bootstrap_order=order,
            source_receipt=source,
            seed_receipt=seed,
            reference_receipt=reference,
            bindings=bindings,
        )
    print(
        json.dumps(
            {
                "status": "validated",
                "scout_adoption_receipt": receipt.receipt_path,
                "scout_adoption_receipt_sha256": receipt.receipt_sha256,
                "scientific_action_id": receipt.scientific_action_id,
                "completed_lifecycle": receipt.completed_lifecycle.to_json_dict(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
