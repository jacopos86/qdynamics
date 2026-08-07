# Strong--strong cumulative-relative RA plateau r70 activation

This directory is the one-row ordinary-held CHTC activation overlay for the
authenticated strong--strong RA singleton-plateau continuation from controller
round 50 through round 70.  The source protocol remains stationary-gradient,
late-weighted, POWELL-200/seed-7, singleton, and cumulative-relative plateau;
the horizon is the only scientific change.

The overlay deliberately contains no strong--weak row.  `submit.sub` starts its
single ordinary job held, disables automatic release, and is intended for an
exact `ClusterId.ProcId` release only after remote package and image byte
validation.  The activation does not authorize Paper-I evidence adoption.

Build and local validation are fail-closed and non-overwriting:

```text
python3 -B build_activation.py --authorized-utc YYYY-MM-DDTHH:MM:SSZ
python3 -B validate_activation.py
python3 -B ../../validate_condor_submit_lifecycle.py submit.sub
```

The remote image is intentionally not copied into this local overlay.  Its
expected SHA-256 and size are sealed in the activation manifest and the worker
wrapper revalidates its bytes before execution.
