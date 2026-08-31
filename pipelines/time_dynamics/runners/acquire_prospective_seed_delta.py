#!/usr/bin/env python3
"""Plan, validate, ingest, or replay prospective seed-delta acquisition."""

from pipelines.time_dynamics.prospective_seed_delta_acquisition import main


if __name__ == "__main__":  # pragma: no cover - exercised through coordinator tests
    raise SystemExit(main())
