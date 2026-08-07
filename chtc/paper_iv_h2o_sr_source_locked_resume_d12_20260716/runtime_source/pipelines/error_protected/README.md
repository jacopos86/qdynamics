# Error-Protected ADAPT Sidecar

This package is a **sidecar replay/postprocess lane** for exported HH ADAPT
artifacts.  It is intentionally separate from the canonical ADAPT pipeline.

## Scope

Milestone one does:

- load a canonical ADAPT/scaffold artifact;
- rebuild the exported parameterized ansatz;
- acquire raw grouped measurements through `RawMeasurementOracle`;
- report raw energy and a full-register HH sector audit;
- emit a distinct `hh_detect_only_replay_v1` sidecar JSON artifact.

Milestone one does **not**:

- mutate the canonical ADAPT JSON;
- alter `pipelines/static_adapt/adapt_pipeline.py`;
- steer phase-3 selection;
- replace the built-in final audit;
- apply postselection to energy;
- apply correction or claim QEC/fault tolerance.

`sector_filter` is deliberately unavailable in this milestone unless a later
implementation proves detector observability for every required observable
group.  Correction is reported as `off` and `applied=false`.

## Known working smoke command

From the repository root:

```bash
python -m pipelines.error_protected.adapt_detect_only_replay \
  --artifact-json artifacts/json/campaign_A6_L2_backend_proxy_baseline.json \
  --output-json /tmp/hh_detect_only_sidecar_smoke.json \
  --noise-mode backend_scheduled \
  --execution-surface raw_measurement_v1 \
  --raw-grouping-mode qwc_basis_cover_reuse \
  --backend-name FakeGuadalupeV2 \
  --use-fake-backend \
  --shots 32 \
  --oracle-repeats 1 \
  --oracle-aggregate mean \
  --detection-mode sector_detect \
  --min-accepted-shots 1 \
  --strict
```

Expected high-level result:

- `schema_version = "hh_detect_only_replay_v1"`
- `route_kind = "adapt_detect_only_replay"`
- `estimates.energy_raw.status = "ok"`
- `estimates.sector_audit.status = "ok"`
- `failure = null`

The exact energy and acceptance rate are shot-sampled diagnostics, not fixed
regression values.

## Convenience smoke module

The same smoke can be run with:

```bash
python -m pipelines.error_protected.smoke_detect_only_replay
```

By default it writes:

```text
/tmp/hh_detect_only_sidecar_smoke.json
```

Use `--help` to override the artifact, output path, backend, shots, and repeat
count.

## Sidecar artifact shape

Important top-level fields:

- `schema_version`
- `route_kind`
- `base_artifact_json`
- `physics_point`
- `ansatz_identity`
- `oracle_request`
- `detection`
- `raw_summary`
- `estimates`
- `comparisons`
- `diagnostics`
- `failure`

Current estimates:

- `energy_raw`: raw grouped energy estimate from `RawMeasurementOracle`;
- `sector_audit`: full-register HH sector leakage audit where observable;
- `sector_filter`: emitted only when requested, and unavailable in milestone one.

## Failure semantics

The sidecar must not silently reinterpret a failed detect-only estimate as a raw
estimate.  Missing detector observability, zero accepted shots, and too few
accepted shots are explicit statuses.

The canonical ADAPT artifact is an input only.  It must remain byte-for-byte
unchanged by sidecar runs.

## Next intended milestone

After this replay/audit lane is stable, the next detector family should be
measurement-consistency diagnostics.  Ancilla parity transforms, postprocessed
correction, controller integration, and canonical ADAPT hooks remain deferred.
