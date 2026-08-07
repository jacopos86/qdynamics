# Geo projected-singleton IS/SS packaging repair (v6)

This immutable operational successor replaces only held parent jobs
`8887546.4` (intermediate-strong) and `8887546.5` (strong-strong).

It preserves the parent v5 source archive, worker, job manifests, normalized
manifests, physics points, exact references, candidate pools, controller,
optimizer, accounting, and validation contracts byte-for-byte. The only
execution repair is an EXIT/TERM/INT packaging trap in the bundle wrapper so a
Python or container failure still returns the job-owned diagnostic directory.

The queue contains exactly the two affected fresh round-zero jobs. This bundle
must not be used to relaunch the already running weak-strong parent job or the
three previously validated weak-Holstein jobs.

