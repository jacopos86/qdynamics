# Strong-strong singleton RA-plateau r50→r70 activation

This is a one-row ordinary-held activation for
`core__strong_strong_u8__nph7__ra_singleton_plateau__r70`.

It resumes the authenticated strong-strong round-50 checkpoint, uses the
sealed retention-v2 runtime unchanged, and excludes every strong-weak row.
The only scientific change is the controller horizon from 50 to 70. The row
must be submitted held and released only by exact `ClusterId.ProcId`.

Validation:

```text
python3 validate_activation.py
python3 ../../validate_condor_submit_lifecycle.py submit.sub
```
