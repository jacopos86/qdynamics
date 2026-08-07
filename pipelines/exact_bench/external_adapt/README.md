# External ADAPT benchmark references

This package is the benchmark-local home for public or request-only competitor
implementations used to validate Paper I static-ADAPT comparisons.

## Contract

- Keep third-party checkouts outside this repository by default:
  `~/.cache/holstein_external_competitors/`.
- Store only catalog, provenance, adapters, and conformance tests in this repo.
- Do not import third-party ADAPT packages from production/core paths.
- Do not emulate CEO, TETRIS, or Overlap-ADAPT by toggling the project Phase3
  controller. Promote a row to runnable only after a benchmark-local adapter is
  conformance-tested against the reference implementation.
- Record URL, requested ref, resolved commit, license files, and adapter status
  in row manifests before using competitor data in a manuscript.

## Current reference catalog

Run:

```bash
python -m pipelines.exact_bench.external_adapt.fetch_references --list
```

Fetch public git references into the external cache:

```bash
python -m pipelines.exact_bench.external_adapt.fetch_references --reference-id ceo_adapt_vqe
```

The generated lockfile is written beside the external cache, not inside this
repo, unless an explicit `--cache-root` points elsewhere.

## First runnable slice

`static_ceo_adapt_phase3` is wired only for `hubbard/hubbard_L2` through the
public `ceo_adapt_vqe` checkout pinned to
`712f6dd3bc56e9e3f5a10b5f46ad6194c9f6ac63`. `static_tetris_adapt_phase3` uses
the same public-code surface with `LinAlgAdapt(tetris=True)` and is additionally
parameterized for the Paper-I diagnostic Hubbard L2 weak/strong cases.  The
adapter runs the public `HubbardHamiltonian` + `OVP_CEO` + `LinAlgAdapt` path,
emits distinct TETRIS telemetry, and is not a Phase3 emulation.

`static_overlap_adapt_phase3` remains skipped/request-only until author code is
obtained or a separately labeled faithful implementation is explicitly approved.
