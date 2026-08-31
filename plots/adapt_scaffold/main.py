"""CLI entrypoint for standalone ADAPT scaffold plots."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence

from plots.adapt_scaffold.artifact_reader import load_adapt_plot_artifact
from plots.adapt_scaffold.renderers import (
    build_generator_usefulness_rows,
    write_energy_vs_iteration_outputs,
    write_generator_usefulness_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone ADAPT scaffold plotting tools.")
    parser.add_argument("--input-json", required=True, help="Path to raw/exported ADAPT artifact JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for plot outputs.")
    parser.add_argument("--ranking-top-k", type=int, default=30, help="Top-K bars to show in usefulness plot.")
    parser.add_argument(
        "--include-landscape",
        action="store_true",
        help="Write the projected final-scaffold trajectory view.",
    )
    parser.add_argument(
        "--compute-surrogate-energy",
        action="store_true",
        help="Also compute prefix energies for the final-scaffold trajectory if prerequisites exist.",
    )
    parser.add_argument(
        "--orqviz-overlay",
        action="store_true",
        help="Attempt an optional orqviz scan overlay for the final-scaffold trajectory.",
    )
    return parser.parse_args(argv)


def _command_string(argv: Sequence[str] | None) -> str:
    if argv is None:
        return " ".join(shlex.quote(x) for x in sys.argv)
    return " ".join(
        shlex.quote(x)
        for x in [sys.executable, "-m", "plots.adapt_scaffold.main", *list(argv)]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        artifact = load_adapt_plot_artifact(args.input_json)
        manifest: dict[str, Any] = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "command": _command_string(argv),
            "input_json": str(Path(args.input_json).expanduser().resolve()),
            "input_pipeline": artifact.payload.get("pipeline"),
            "history_step_count": int(len(artifact.history_steps)),
            "admission_record_count": int(len(artifact.admission_records)),
            "final_scaffold_count": int(len(artifact.final_scaffold_records)),
            "ranking_metric": "delta_abs_drop_from_prev -> exact_abs_error_improvement -> raw_energy_drop",
            "warnings": list(artifact.warnings),
            "views": {},
        }

        manifest["views"]["energy_vs_iteration"] = write_energy_vs_iteration_outputs(
            artifact,
            output_png=output_dir / "adapt_energy_vs_iteration.png",
            output_csv=output_dir / "adapt_energy_vs_iteration.csv",
        )
        manifest["views"]["generator_usefulness"] = write_generator_usefulness_outputs(
            artifact,
            output_png=output_dir / "adapt_generator_usefulness.png",
            output_csv=output_dir / "adapt_generator_usefulness.csv",
            top_k=int(args.ranking_top_k),
        )

        if args.include_landscape:
            try:
                from plots.adapt_scaffold.landscape import (
                    build_scaffold_trajectory_dataset,
                    maybe_build_final_scaffold_energy_model,
                    trajectory_prerequisite_failure_reason,
                    write_scaffold_trajectory_outputs,
                )
                from plots.adapt_scaffold.orqviz_adapter import try_write_orqviz_overlay

                reason = trajectory_prerequisite_failure_reason(artifact)
                if reason is not None:
                    manifest["views"]["final_scaffold_trajectory"] = {
                        "status": "skipped_prerequisite",
                        "reason": str(reason),
                    }
                else:
                    energy_model = (
                        maybe_build_final_scaffold_energy_model(artifact)
                        if bool(args.compute_surrogate_energy) or bool(args.orqviz_overlay)
                        else None
                    )
                    manifest["views"]["final_scaffold_trajectory"] = write_scaffold_trajectory_outputs(
                        artifact,
                        output_png=output_dir / "adapt_final_scaffold_trajectory.png",
                        output_json=output_dir / "adapt_final_scaffold_trajectory.json",
                        surrogate_energy_model=(
                            energy_model if bool(args.compute_surrogate_energy) else None
                        ),
                    )
                    if args.compute_surrogate_energy and energy_model is None:
                        manifest["warnings"].append("Surrogate-energy model unavailable; trajectory JSON omitted surrogate energies.")
                    if args.orqviz_overlay:
                        if energy_model is None:
                            manifest["views"]["final_scaffold_orqviz_overlay"] = {
                                "status": "skipped_prerequisite",
                                "reason": "missing_surrogate_energy_model",
                            }
                        else:
                            dataset = build_scaffold_trajectory_dataset(artifact)
                            manifest["views"]["final_scaffold_orqviz_overlay"] = try_write_orqviz_overlay(
                                energy_model=energy_model,
                                dataset=dataset,
                                output_png=output_dir / "adapt_final_scaffold_trajectory_orqviz.png",
                                output_json=output_dir / "adapt_final_scaffold_trajectory_orqviz.json",
                            )
            except Exception as exc:
                manifest["views"]["final_scaffold_trajectory"] = {
                    "status": "error_optional_view",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                manifest["warnings"].append(
                    f"Optional landscape path failed after required outputs were written: {type(exc).__name__}: {exc}"
                )
        elif args.orqviz_overlay:
            manifest["views"]["final_scaffold_orqviz_overlay"] = {
                "status": "skipped_prerequisite",
                "reason": "include_landscape_required",
            }

        manifest_path = output_dir / "adapt_scaffold_plot_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote ADAPT scaffold plots to {output_dir}")
        print(f"Energy rows: {len(artifact.history_steps)}")
        print(f"Ranking rows: {len(build_generator_usefulness_rows(artifact))}")
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
