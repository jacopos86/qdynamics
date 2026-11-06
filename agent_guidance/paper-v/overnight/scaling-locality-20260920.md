# A locality-restricted entrance predicate: rows, frontier, and closure

**Round of 2026-09-20, key `scaling-locality`.** Branch
`claude/paper-v-routes-20260914`, worktree
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`,
`HEAD 614bc9e2`.

**What this is.** A counting study, the one the companion round
(`scaling-dimension-growth-20260920.md`) named as the next cheap step.
Nothing here propagates, prepares a state, solves a cone, edits a module or
changes a default. Every number is a count; none is adopted evidence.

**The negative result, first.** *Extension rows can be held linear in `L`,
and it does not help.* Under the same-bond restriction the 27-key
dictionary's analogue has `27 (L - 1)` active keys and `35 L - 34` extension
rows --- `36`, `71`, `106`, `141` at `L = 2` to `5`, against the declared
predicate's `36`, `281`, `1,192`, `3,621` --- so the verifier's claim as
stated, that rows cannot be held linear in this construction, is refuted by
the enumeration. The verifier's *reason* is upheld and is decisive on its
own: **the restricted dictionary is not invariant under the Holstein
commutator.** One commutator step takes it outside itself at `84`, `173` and
`267` moments of degree at most three at `L = 3`, `4`, `5`, and every one of
those moments is supported on more than one bond, so none of them can be
added back without abandoning the restriction. Iterating the commutator to a
fixed point gives `27`, `228`, `975`, `2,988` keys over `L = 2` to `5` ---
`L^5.0` by successive ratios, the same quintic law as the declared
dictionary, and a stable `81 %` of the whole declared enlargement at
`L = 3`, `4`, `5`. The restricted dictionary's Gram frontier behaves the
same way: `70`, `1,011`, `2,418`, `4,334`, of which `0 %`, `85.4 %`,
`91.0 %` and `93.2 %` are supported on more than one bond. Memory closes the
question independently: at `74.8 (n (2 n + 1))^2` bytes the `18 GB`
aggregate contract admits `87` extension rows, so the linear law is runnable
at three sites (`71` rows, `7.18 GiB`) and at no more (`106` rows,
`35.52 GiB` at four sites), while the modelled crossing under the same
linear law sits at `L = 6` (single-power exponent `3.274`) or `L = 7`
(two-term law). The companion round's `L = 5` or `6` crossing does not
survive: the counted linear law has slope `35`, not `18`, and even the `18 L`
law it was stated for needs `90` rows at `L = 5`, above the `87`-row
ceiling.

---

## (a) What the entrance predicate admits at two sites, and what the
## restriction would mean

`MomentKey(spin_up, spin_down, x_power, p_power)`
(`moment_hierarchy.py`) carries one Pauli label per spin species and one
Weyl monomial of the single interacting relative mode, canonicalized by
`_canonical_key` so that the ordered pair of species labels is sorted. Its
`degree` is the number of non-identity species labels plus `x_power` plus
`p_power`.

`ENTRANCE_RELATIVE_MOMENT_KEYS` (`adaptive_positive_moment.py`) selects, from
the keys of degree at most three that `_is_archive_resolved_relative_moment`
does not resolve, exactly two blocks:

* `degree == 2`, spin degree `2`, boson degree `0` --- both spin species
  carry a non-identity Pauli label and no phonon coordinate is present. There
  are `3 * 4 / 2 = 6` such canonical pairs.
* `degree == 3`, spin degree `1`, boson degree `2` --- one species carries a
  non-identity Pauli label, the other the identity, and the key carries a
  degree-two Weyl monomial (`x^2`, `x p`, `p^2`). There are `3 * 3 = 9`.

Fifteen keys, and `INITIAL_PROMOTION_CANDIDATE_KEYS` holds the remaining
sixteen hidden keys, of which the twelve two-spin/one-boson keys are the ones
the recorded 27-key dictionary promotes. Neither predicate mentions a site,
a bond, or a mode index: at `L` sites the first block is every canonical pair
of electronic one-body generators across the two species and the second is
every generator against every degree-two Weyl monomial over all `L - 1`
interacting modes, which is where the quartic and quintic come from.

**"Supported on a single site and its chain neighbour", as a restriction of
that predicate.** Write the electronic one-body Hermitian basis in the form
adapted to the chain: the identity; for every pair of sites `i < j`,
`S_ij = E_ij + E_ji` and `A_ij = -i E_ij + i E_ji`; and for every bond
`q = 0 .. L - 2`, `D_q = E_qq - E_{q+1,q+1}`. That is `L^2` elements and at
`L = 2` it is exactly `(I, X, Y, Z)`. Bond `q` carries the three generators
`S_{q,q+1}`, `A_{q,q+1}` and `D_q`; the remaining `(L - 1) (L - 2)`
generators carry a pair of sites that are not neighbours. Interacting mode
`q` is identified with bond `q`, the identification the companion round's row
budget declared.

The restricted set is then: *the keys the declared entrance predicate admits
for which one bond carries both spin-species generators and the Weyl monomial
uses no coordinate of any other mode*, an identity factor counting as carried
by every bond. At `L = 2` there is one bond and one interacting mode, so the
restriction removes nothing and every count below must equal the committed
two-site number.

The same restriction applied to `_base_half_words`
(`apcm_positive_extension.py`) leaves the identity, the `3 (L - 1)` bond
generators on each spin species and the `2 (L - 1)` Weyl coordinates:
`8 L - 7` words, `9` at two sites. Two extension variants are therefore
counted, because the committed construction separates the dictionary from
the half-words that expose the chart:

* **fully local** --- half-words restricted as well, `8 L - 7` base rows;
* **dictionary only** --- the declared half-words widened to all `L^2 - 1`
  generators, `2 L^2 + 2 L - 3` base rows, with only the active keys
  restricted. This is the variant the companion round's row arithmetic
  costed.

They agree at `L = 2` and diverge above it. Restricting the half-words is not
free: the pairwise Gram entries of the base half-words are the raw moment
chart, so under the fully local variant the non-neighbour entries of the
electronic one-body density matrix and the cross-mode electron--phonon
correlation entries carry no positivity row at all.

---

## (b) The counts, `L = 2` to `5`

Products carry the full structure constants: `G_a G_b` is expanded by an
exact linear solve against the whole `L^2`-element basis, so the identity
part, the symmetric `d` part and the antisymmetric `f` part are all kept.
The maximum number of basis elements in a product of two generators is `1`,
`3`, `4`, `5` at `L = 2`, `3`, `4`, `5`: the one-term law of `_pauli_product`
is the `su(2)` identity `d_abc = 0` and it holds only at two sites.

Restricted keys against the declared predicate:

| quantity | 2 | 3 | 4 | 5 | restricted form |
|---|---:|---:|---:|---:|---|
| entrance keys, restricted | 15 | 30 | 45 | 60 | `15 (L - 1)` |
| entrance keys, declared | 15 | 116 | 435 | 1164 | `c2 + c1 M(2)` |
| two-spin/one-boson, restricted | 12 | 24 | 36 | 48 | `12 (L - 1)` |
| two-spin/one-boson, declared | 12 | 144 | 720 | 2400 | `c2 M(1)` |
| 27-key analogue, restricted | 27 | 54 | 81 | 108 | `27 (L - 1)` |
| 27-key analogue, declared | 27 | 260 | 1155 | 3564 | --- |
| hidden keys, restricted | 31 | 62 | 93 | 124 | `31 (L - 1)` |
| hidden keys, declared | 31 | 280 | 1211 | 3684 | --- |
| base half-words, fully local | 9 | 17 | 25 | 33 | `8 L - 7` |
| base half-words, declared | 9 | 21 | 37 | 57 | `2 L^2 + 2 L - 3` |

The 27-key analogue under the restriction, fully local extension:

| quantity | 2 | 3 | 4 | 5 | form |
|---|---:|---:|---:|---:|---|
| active keys | 27 | 54 | 81 | 108 | `27 (L - 1)` |
| extension rows | 36 | 71 | 106 | 141 | `35 L - 34` |
| propagated state coordinates | 56 | 137 | 260 | 437 | `2L^3 + 3L^2 + L - 1 + 27 (L - 1)` |
| frontier keys | 70 | 1011 | 2418 | 4334 | enumerated |
| realified positive-extension cone side | 72 | 142 | 212 | 282 | `70 L - 68` |
| extension rows, dictionary-only variant | 36 | 75 | 118 | 165 | `2L^2 + 29L - 30` |

The entrance dictionary under the restriction, fully local extension:

| quantity | 2 | 3 | 4 | 5 | form |
|---|---:|---:|---:|---:|---|
| active keys | 15 | 30 | 45 | 60 | `15 (L - 1)` |
| extension rows | 24 | 47 | 70 | 93 | `23 L - 22` |
| propagated state coordinates | 44 | 113 | 224 | 389 | `2L^3 + 3L^2 + L - 1 + 15 (L - 1)` |
| frontier keys | 62 | 443 | 1018 | 1790 | enumerated |
| realified positive-extension cone side | 48 | 94 | 140 | 186 | `46 L - 44` |

**The `L = 2` row against the committed 27-key quantities: no departure.**
Every restricted two-site count equals the committed number --- archive-resolved
keys `14`, entrance keys `15`, two-spin/one-boson promotion keys `12`, the
27-key dictionary `27`, hidden keys `31`, base half-words `9`, extension rows
`24 / 36 / 40` at 15 / 27 / 31 active keys, lower keys `29 / 41 / 45`,
frontier keys `62 / 70 / 129`, propagated state coordinates `44 / 56 / 60`,
realified positive-extension cone side `48 / 72 / 80`. The restriction is the
identity at two sites, which is what makes the `L >= 3` rows a restriction of
the same construction rather than a different one.

The propagated state coordinates stay cubic in `L` under the restriction,
because the raw moment chart is the state, not a dictionary choice: only the
active-key term shrinks, from `27 (L^5/2 + ...)` to `27 (L - 1)`.

---

## (c) Does the frontier close? No, and the flow leaves the dictionary first

Three separate readings, all enumerated at the 27-key analogue with the fully
local extension.

**1. The Gram frontier is not local.** The frontier is by construction the
set of moments the extension rows' own products generate that are neither
archive-resolved nor active, so in the trivial reading it always "closes".
The question that decides the route is whether it stays inside the restricted
vocabulary, and it does not:

| `L` | frontier keys | supported on more than one bond | share |
|---:|---:|---:|---:|
| 2 | 70 | 0 | 0.000 |
| 3 | 1011 | 863 | 0.854 |
| 4 | 2418 | 2200 | 0.910 |
| 5 | 4334 | 4038 | 0.932 |

The frontier itself is super-linear where the dictionary is linear: `L^4.50`
fitted over `L = 2` to `5`, with successive exponents `6.59`, `3.03`, `2.62`,
so read it as the order and not a converged power; at the top end it is still
growing about as `L^2.6` against the dictionary's `L^1`.

**The closure defect, by identity.** The non-local frontier moments at the
27-key analogue fall into four kinds, counted at `L = 3 / 4 / 5`:

| kind of moment | 3 | 4 | 5 |
|---|---:|---:|---:|
| Weyl monomial carrying coordinates of two different modes | 541 | 1336 | 2397 |
| electronic factor spanning three or more sites | 207 | 588 | 1158 |
| generator and monomial on different bonds | 100 | 246 | 438 |
| the two species' generators on different bonds | 15 | 30 | 45 |
| **total** | **863** | **2200** | **4038** |

The maximum electronic site span of a generated moment is `3` at `L = 3` and
`4` at `L = 4` and `5`. Representative members, in the bond-adapted labels:
the pure-boson cross-mode moments `(I,I) x0 p0 x1 p1`, `(I,I) p0^2 x1^2`;
the next-nearest-neighbour electronic moments `(I,S_02) p0^2`,
`(I,S_02) x1 p1`; and the same-species/other-bond moments
`(I,A_01) x1^2`. The first kind is the Moyal product of two same-bond
degree-two monomials on adjacent bonds; the second is the matrix product
`S_{01} S_{12} ~ E_02`, which exists only because `d_abc` does not vanish
above `su(2)`.

**2. The equation of motion leaves the dictionary on the first step.** This
is the decisive count. Commuting the `8 (L - 1)` Holstein-chain words with
the restricted active keys generates, at degree at most three and excluding
archive-resolved moments:

| `L` | generated hidden moments | outside the restricted dictionary | of those, non-local |
|---:|---:|---:|---:|
| 2 | 27 | 0 | 0 |
| 3 | 138 | 84 | 84 |
| 4 | 254 | 173 | 173 |
| 5 | 375 | 267 | 267 |

At two sites the 27-key dictionary is exactly invariant --- and the flow
closure of the 15-key *entrance* dictionary adds exactly the twelve
two-spin/one-boson keys and stops at `27`, which is the committed promotion
block, reproduced here by iteration rather than by declaration. At three
sites and above every moment the flow requires and the restricted dictionary
lacks is supported on more than one bond. There is no subset of them that can
be promoted while keeping the restriction.

**3. The invariant hull is the quintic dictionary again.** Iterating the
Holstein commutator on the restricted dictionary to a fixed point, keeping
only generated moments of degree at most three that the chart does not
resolve:

| `L` | fixed point | rounds | declared hidden keys | fixed point / declared |
|---:|---:|---:|---:|---:|
| 2 | 27 | 1 | 31 | 0.871 |
| 3 | 228 | 4 | 280 | 0.814 |
| 4 | 975 | 6 | 1211 | 0.805 |
| 5 | 2988 | 8 | 3684 | 0.811 |

Successive exponents `5.26`, `5.05`, `5.02`: the hull grows as `L^5.0`,
which is the declared dictionary's own quintic law, and it is a stable four
fifths of the whole declared enlargement at every site count enumerated.
Holding the rows linear therefore means refusing to carry the moments the
Holstein commutator produces from the rows that are carried --- a truncation
of the equation of motion, which is a different scientific object from the
corrected closure, not a cheaper version of it.

---

## (d) Memory, and whether the crossing survives

At `74.8 (n (2 n + 1))^2` bytes for `n` extension rows, the `18 GB`
aggregate contract admits `87` rows (`16.15 GiB`); `88` rows is `16.91 GiB`
and is over.

| `L` | rows, fully local | GiB | fits | rows, dictionary-only | GiB | fits | rows, `18 L` | GiB | fits |
|---:|---:|---:|:--|---:|---:|:--|---:|---:|:--|
| 2 | 36 | 0.48 | yes | 36 | 0.48 | yes | 36 | 0.48 | yes |
| 3 | 71 | 7.18 | yes | 75 | 8.94 | yes | 54 | 2.41 | yes |
| 4 | 106 | 35.52 | no | 118 | 54.50 | no | 72 | 7.60 | yes |
| 5 | 141 | 110.96 | no | 165 | 207.86 | no | 90 | 18.49 | no |

**Runnable at three sites; not at four or five.** A locality-restricted
closure at `L = 3` needs `71` extension rows and a `7.18 GiB` conic block,
inside the contract with room. At `L = 4` it needs `106` rows and
`35.52 GiB`, over the contract by a factor of `2.1`, and at `L = 5`
`141` rows and `110.96 GiB`, over by `6.6`. The dictionary-only variant ---
the one that keeps the full chart exposed in the Gram --- is worse at every
site count and crosses the ceiling at the same place.

The modelled `t = 0` to `20` wall under the counted linear law
`35 L - 34`, against the exact cutoff-sixteen driven reference, with the two
cost statements already on record (single-power exponent `3.274` anchored at
`8.325 s` per completion solve at cone side `72`; the two-term law
`a m^1.5 + c m^3` in `m = n (n + 1) / 2` at `22` iterations, stated valid
over cone side `32` to `104`), the trajectory held at `500` completion solves
carrying `0.803` of the wall:

| `L` | cone side | rollout s, single power | rollout s, two-term | exact s | ratio, single power | ratio, two-term |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 72 | `5.18e3` | `4.45e3` | `3.50e0` | `1.48e3` | `1.27e3` |
| 3 | 142 | `4.79e4` | `1.64e5` | `2.01e2` | `2.39e2` | `8.18e2` |
| 4 | 212 | `1.78e5` | `1.70e6` | `8.09e3` | `2.20e1` | `2.10e2` |
| 5 | 282 | `4.53e5` | `9.24e6` | `2.69e5` | `1.69e0` | `3.44e1` |
| 6 | 352 | `9.36e5` | `3.47e7` | `7.89e6` | `1.19e-1` | `4.39e0` |
| 7 | 422 | `1.69e6` | `1.03e8` | `2.13e8` | `7.95e-3` | `4.81e-1` |

**The crossing does not survive as `L = 5` or `6`.** Under the counted linear
law it is at `L = 6` (single power) or `L = 7` (two-term), one site later
than the `18 L` law gave, because the counted slope is `35` rather than `18`.
And it is three or four sites past the largest site count the memory contract
admits. Even the `18 L` law's own `L = 5` crossing needs `90` rows and
`18.49 GiB`, above the `87`-row ceiling --- the companion round's
"inside or at the edge of the memory contract at exactly the site count where
the crossing happens" resolves to *outside* when the ceiling is applied.

At the one site count above two that is runnable, three sites, the modelled
closure wall is `4.79e4 s` (single power) to `1.64e5 s` (two-term) against
the *measured* exact three-site driven reference of `137 s` to `t = 20`, so
`3.5e2` to `1.2e3` times behind. A three-site locality-restricted closure
could be built and run inside the memory contract, and it would lose to exact
propagation by three orders of magnitude.

---

## Verdict

The route is dead at this construction, and locality does not revive it. The
enumeration refutes the verifier's claim as literally stated --- extension
rows *can* be held linear in the site count, `35 L - 34` under the same-bond
restriction, reproducing the committed `36` at two sites --- and then refutes
the route on the verifier's own reason, more sharply than the reason was
given. The restricted dictionary is not invariant under the Holstein
commutator at three sites or above: one commutator step demands `84`, `173`
and `267` moments it does not carry, every one of them supported on more than
one bond, and the transitive hull of that demand is `228`, `975` and `2,988`
keys, growing as `L^5.0` and recovering four fifths of the whole declared
quintic enlargement. A dictionary with linear rows is therefore not a closure
at all but a truncated equation of motion, and the Gram frontier says the same
thing from the other side, being `85 %` to `93 %` non-local wherever the
dictionary is local. Even granting the linear rows for free and ignoring the
closure defect, the `18 GB` aggregate contract's `87`-row ceiling admits the
construction only at three sites, where the modelled wall is `3.5e2` to
`1.2e3` times the measured exact three-site reference, while the modelled
crossing has moved out to `L = 6` or `7`, at row counts of `176` and `211`
that exceed the contract by factors of `16` and `33`. There is no site count
at which the corrected closure is both runnable and cheaper than exact
propagation, under any cost law on record and under either dictionary.

---

## Caveats

1. Nothing was propagated, prepared, solved or promoted; no module under
   `paper_5/src` was edited and no default changed. Every number is a count.
2. The restriction is a counting model, not an implemented `L`-site closure.
   It replays the committed predicates in a bond-adapted basis and says how
   large the restricted construction would be and what it generates, not that
   it would be correct.
3. Frontier and hull counts are basis-dependent in the same way the companion
   round's are; the closure defect stated as *supported on more than one
   bond* is read from the sites a generated operator touches by nonzero row or
   column and is basis-free.
4. Interacting mode `q` is identified with bond `q`. A construction that
   instead kept the `L` site-local modes and removed the uniform one would
   carry one more mode and a non-local mode basis; the identification used
   here is the one that favours locality.
5. The flow closure keeps only generated moments of total degree at most
   three; generated degree-four moments are the terminal cumulant
   reconstruction the hierarchy already declares.
6. The cost table is the companion rounds' model with their two named
   exponents, and the `500` solves and the step count are held at the
   two-site trajectory's values. The memory law and its `87`-row ceiling are
   measured and are not a model.

## Files

* `output/local_runs/paper_v_scaling_20260920/count_locality_restricted.py`
  --- the restricted enumeration; `--sites 2` is its validation.
* `output/local_runs/paper_v_scaling_20260920/locality_counts_L2_L5.json`
  --- the counts, defect classes and flow closures at `L = 2` to `5`.
* `output/local_runs/paper_v_scaling_20260920/summarize_locality.py` and
  `locality_reading.json` --- the tables, the memory reading and the cost
  model.
