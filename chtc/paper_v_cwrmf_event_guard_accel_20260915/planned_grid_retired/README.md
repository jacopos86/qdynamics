# Retired planned grid, 2026-09-20

`submit_cwrmf_accel_s02.sub` through `s08.sub` were written on 2026-09-15 as
the planned eight-segment grid for `t = 0` to `4`.  Each names the *previous
planned* segment's run id for its checkpoint transfer, and the planned first
segment `accel_s01_t0000_t0500` never produced one: cluster 10554184 failed at
import after 14 s (see `../submission_receipt_10554184.json`), so the campaign
actually runs under the retry run id `accel_s01_t0000_t0500_r01` (cluster
10737278, `final_time`, `t = 0.5`).  Every file here therefore transfers a
checkpoint that does not exist, and each one's run id also blocked
`make_resume_submit.py`, whose reuse check scans the lane's `.sub` arguments.

They are kept, not deleted, because they record the segment sizing the
campaign was planned at.  Continuations are generated from the returned
segment's `receipt.json`:

```bash
NEXT=$(python3 chtc/paper_v_cwrmf_event_guard_accel_20260915/make_resume_submit.py \
         archives_by_receipt/<sha>/receipt.json | tail -1)
python3 chtc/validate_condor_submit_lifecycle.py "$NEXT"
condor_submit -dry-run /tmp/next.ad "$NEXT"
condor_submit "$NEXT"
```

The same seven files were moved to `planned_grid_retired/` on the submit node
before segment 2 was generated.
