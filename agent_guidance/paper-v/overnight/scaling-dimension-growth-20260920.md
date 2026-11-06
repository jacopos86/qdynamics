# Dimension growth of the archive moment closure with the number of sites

**Round of 2026-09-20, key `dimension-growth`.** Branch
`claude/paper-v-routes-20260914`, worktree
`/Users/lordskyler/Holstein_test.worktrees/claude-paper-v-routes-20260914`.

**What this is.** A counting study. It derives, from the committed two-site
construction's own definitions, how many coordinates, dictionary keys,
positive-extension rows, frontier keys and positive-semidefinite cone rows
the same construction has at `L` sites. Nothing here propagates, prepares a
state, solves a cone, or changes a default; no module was edited; no
`L`-site closure was implemented. Every count is exploratory and is not
adopted evidence.

**The negative result, first.** The count that decides whether the corrected
closure can pay for itself --- the side dimension of the cone the declared
completion solves --- is **quartic to quintic in `L`, not bounded**. The
dimer's `36` extension rows at 27 entrance keys become `281` at three sites,
`1,192` at four and `3,621` at five, because the entrance dictionary takes
**every** pair of electronic one-body generators across the two spin species
and **every** degree-two Weyl monomial across all `L - 1` interacting modes.
The realified cone the solve carries therefore goes `72 -> 562 -> 2,384 ->
7,242` over `L = 2` to `5`. Under the record's own measured cost exponent in
that side dimension (`3.274`, fitted on the 36 recorded conic instances,
`0.029 s` at side `12` to `14.462 s` at side `80`), one completion solve rises
from the measured `8.325 s` at two sites to `6.96e3 s` at three and
`7.90e5 s` at four, while the exact cutoff-16 reference's Hilbert dimension
rises `1,156 -> 44,217 -> 1,336,336`. The closure is behind the exact
reference by `1.48e3` at two sites, `2.16e4` at three, `6.08e4` at four and
`6.96e4` at five: **the gap widens over every site count that can be run**,
and in the model it closes only near `L = 14` --- `L = 31` under the companion round's measured conic cost law --- where the closure's own
`t = 0` to `20` rollout would cost `3.18e17 s`.

The one count that is genuinely cheap is the **scored** chart: the 31 closed
scalar coordinates become `2L^3 + 3L^2 + 3L - 1`, cubic. The closure's
*output* stays small; its *machinery* does not.

---

## 0. What was read, and what the counting model reproduces

Read: `adaptive_positive_moment.py` (`ARCHIVE_RELATIVE_MOMENT_KEYS`,
`ENTRANCE_RELATIVE_MOMENT_KEYS`, `HIDDEN_RELATIVE_MOMENT_KEYS`,
`INITIAL_PROMOTION_CANDIDATE_KEYS`, `RAW_MOMENT_COORDINATE_NAMES`),
`moment_hierarchy.py` (`MomentKey`, `build_moment_keys`, `_pauli_product`,
`_operator_product`, `_weyl_product`, `_canonical_key`),
`apcm_positive_extension.py` (`_base_half_words`, `apcm_extension_words`,
`apcm_extension_coefficients`, `SymmetryReducedPositiveExtension`),
`apcm_moment_projection.py` and `apcm_projective_guard.py`
(`unified_core_moment_matrix`, `unified_glued_moment_matrix`,
`unified_guard_dimension`, `projective_guard_outer_extension`,
`_HAMILTONIAN_OPERATOR_BASIS`), `matrix_reference.py`
(`CLOSED_SCALAR_STATE_NAMES`, `closed_scalar_to_matrix_state`,
`matrix_dimer_rhs`, `_correlation_source_terms_by_mode`,
`closed_scalar_embedding_normal_residual`), and `exact_reference.py`
(`_build_exact_dimer_model`).

The counting model widens two things and nothing else: the electronic
one-body operator basis of each spin species from the four Pauli labels to
the `L^2` generalized Gell-Mann elements (identity plus `L^2 - 1`
generators, which **are** the Pauli matrices at `L = 2`), and the interacting
phonon algebra from one relative mode to `L - 1` modes, i.e. `n = 2(L - 1)`
Weyl coordinates. It then replays the committed definitions verbatim:
`_canonical_key`'s spin-exchange ordering, the archive/entrance/hidden
predicates, `_base_half_words`, `apcm_extension_words`, the frontier as the
union of `_operator_product` supports over all word pairs, and
`projective_guard_outer_extension`'s halfword enlargement.

At `L = 2` every count it produces equals the number the committed code
returns:

| quantity | committed code | counting model |
|---|---:|---:|
| canonical keys through degree 3 (`build_moment_keys(3)`) | 45 | 45 |
| archive-resolved keys | 14 | 14 |
| hidden keys | 31 | 31 |
| entrance keys | 15 | 15 |
| initial promotion candidates | 16 | 16 |
| two-spin/one-boson promotion keys | 12 | 12 |
| base half-words | 9 | 9 |
| raw moment coordinates | 29 | 29 |
| closed scalar coordinates | 31 | 31 |
| extension rows at 15 / 27 / 31 keys | 24 / 36 / 40 | 24 / 36 / 40 |
| lower keys at 15 / 27 / 31 keys | 29 / 41 / 45 | 29 / 41 / 45 |
| frontier keys at 15 / 27 / 31 keys | 62 / 70 / 129 | 62 / 70 / 129 |
| propagated state coordinates at 15 / 27 / 31 keys | 44 / 56 / 60 | 44 / 56 / 60 |
| right-hand-side frontier keys at 15 / 27 / 31 keys | 15 / 16 / 16 | 15 / 16 / 16 |
| outer extension rows at 15 / 27 / 31 keys | 60 / 82 / 86 | 60 / 82 / 86 |
| unified Gram rows at 15 / 27 / 31 keys | 62 / 84 / 88 | 62 / 84 / 88 |
| realified extension cone side at 15 / 27 / 31 keys | 48 / 72 / 80 | 48 / 72 / 80 |

The record's stated two-site figures --- 31 closed scalar coordinates, 44 and
56 state coordinates, 24 and 36 extension rows, 62 and 70 frontier keys, the
62-row unified Gram, and the cone side dimension 80 --- are each in that
column. The carried-witness flow's 384 coordinates are
`(29 + 15) + 238 + 2 * 51`: the retained chart, the outer extension's 238
frontier keys and its 51 center-mode cross rows counted twice.

Notation used below: `n = 2(L - 1)` Weyl coordinates,
`M(b) = C(b + n - 1, n - 1)` the number of Weyl monomials of total boson
degree `b`, `c1 = L^2 - 1` the canonical keys carrying one non-identity
generator, `c2 = (L^2 - 1) L^2 / 2` those carrying two.

---

## (a) The closed scalar coordinates at `L` sites

Counted block by block from `closed_scalar_to_matrix_state`, for `L`
electronic sites and `L` local phonon modes, one spin channel (the other is
fixed by spin exchange):

| block | two sites | `L` sites | why |
|---|---:|---|---|
| electronic one-body density matrix | 3 | `L^2 - 1` | Hermitian with the trace fixed by the one-electron-per-spin sector |
| coherent phonon amplitude | 4 | `2 L` | one complex amplitude per local mode |
| phonon covariance | 4 | `L^2` | Hermitian |
| anomalous covariance | 6 | `L (L + 1)` | complex symmetric |
| electron--phonon correlation | 14 | `2 L^3` at `L >= 3`; `2 L^3 - 2 (L - 1)` at `L = 2` | one complex `L x L` electronic matrix per mode, less the shared-trace identification |
| **total** | **31** | **`2 L^3 + 3 L^2 + 3 L - 1`** at `L >= 3`; `31` at `L = 2` | |

`2 L^3 + 3 L^2 + 3 L - 1` gives `33` at two sites; the chart carries `31`
because the two modes' correlation matrices share one complex trace
(`correlation_shared_trace_real`, `correlation_shared_trace_imag`).

**That sharing is a two-site identity and does not survive.** It is the
invariance `correlation_trace_difference_rhs_abs` of
`closed_scalar_embedding_normal_residual`, measured at `3.43e-16` over 64
random states at `t = 0.37` with the canonical parameters. Term by term of
Eq. (14d): the transport term's trace difference is `2.29e-16` (a commutator
plus `-i omega` times the trace, so it cannot split the modes); the normal
particle and normal hole sources each have a trace difference of `0.399` and
the two anomalous sources each `0.449`, and each pair cancels exactly,
because the two members carry the same scalar prefactor times
`tr[(1 - rho) g rho]` and `tr[rho g (1 - rho)]`, equal by cyclicity at any
`L`. The only term left is the bare Pauli source, whose mode-`q` trace was
measured to equal `-i g (rho - rho @ rho)[q, q]` to `6.77e-17`. That diagonal
is constant exactly when `rho - rho @ rho` is a multiple of the identity,
which Cayley--Hamilton gives for a two-by-two matrix of fixed trace and for
no larger one: the measured spread of that diagonal over 256 random Hermitian
trace-one matrices is `1.67e-16` at `L = 2` against `0.198`, `0.188` and
`0.187` at `L = 3`, `4` and `5`. At three sites and above the `L` mode traces
are independent coordinates and the block is the full `2 L^3`.

**Assumptions.** One electron per spin species, so the electronic trace is
fixed; `L` local phonon modes, one per site; spin symmetry, so the down
channel's density and correlation are not separate coordinates; the same
Hermitian/symmetric structure `closed_scalar_embedding_normal_residual`
certifies at two sites.

---

## (b) Dictionary keys, extension rows and frontier keys

### The key algebra at `L` sites

`MomentKey` is `(spin_up, spin_down, x_power, p_power)`: one operator label
per spin species and one relative-mode Weyl monomial. At `L` sites the label
alphabet per species is the `L^2`-element Hermitian basis and the monomial is
taken over `n = 2 (L - 1)` Weyl coordinates. The number of canonical spin
pairs by spin degree is `1`, `c1`, `c2` --- `1`, `3`, `6` at two sites.

Closed forms, each checked against the enumeration at `L = 2`, `3`, `4` and
`5` with no mismatch:

| count | formula | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Weyl coordinates | `2 (L - 1)` | 2 | 4 | 6 | 8 | 10 | 12 | 14 |
| closed scalar coordinates | `2L^3 + 3L^2 + 3L - 1` (`31` at `L = 2`) | 31 | 89 | 187 | 339 | 557 | 853 | 1239 |
| raw moment coordinates | `2L^3 + 3L^2 + L - 1` | 29 | 83 | 179 | 329 | 545 | 839 | 1223 |
| archive-resolved keys | `M(1) + M(2) + c1 (1 + M(1))` | 14 | 54 | 132 | 260 | 450 | 714 | 1064 |
| entrance keys | `c2 + c1 M(2)` | 15 | 116 | 435 | 1164 | 2555 | 4920 | 8631 |
| two-spin/one-boson promotion keys | `c2 M(1)` | 12 | 144 | 720 | 2400 | 6300 | 14112 | 28224 |
| the 27-key dictionary's analogue | `c2 (1 + M(1)) + c1 M(2)` | 27 | 260 | 1155 | 3564 | 8855 | 19032 | 36855 |
| hidden keys (the whole declared enlargement) | `M(3) + c1 M(2) + c2 (1 + M(1))` | 31 | 280 | 1211 | 3684 | 9075 | 19396 | 37415 |
| base half-words | `2L^2 + 2L - 3` | 9 | 21 | 37 | 57 | 81 | 109 | 141 |
| extension rows, entrance dictionary | `2L^2 + 2L - 3 + c2 + c1 M(2)` | 24 | 137 | 472 | 1221 | 2636 | 5029 | 8772 |
| extension rows, 27-key analogue | `2L^2 + 2L - 3 + c2 (1 + M(1)) + c1 M(2)` | 36 | 281 | 1192 | 3621 | 8936 | 19141 | 36996 |

`M(1) = 2 (L - 1)`, `M(2) = (2L - 1)(L - 1)` and `M(3) = C(2L, 3)` are
the multiset counts of monomials in `n` Weyl coordinates; at two sites `M(1) = 2`, `M(2) = 3`, `M(3) = 4`.

Extension rows are `_base_half_words` plus one row per active key, exactly as
`apcm_extension_words` builds them: `9 + 15 = 24` and `9 + 27 = 36` at two
sites, `21 + 116 = 137` and `21 + 260 = 281` at three.

### Which symmetry the dimer construction uses, and which survives

Four reductions are in play at two sites. They do not have the same fate.

1. **Spin exchange (up against down), in `_canonical_key`.** The ordered pair
   of species labels is canonicalized, taking `L^4` ordered pairs to
   `L^2 (L^2 + 1) / 2`. This is the exchange of the two identical spin
   species in the fixed one-up/one-down sector; it has nothing to do with the
   site count and **survives at every `L`**, with its saving rising from
   `16 / 10 = 1.60` at two sites toward `2`.

2. **The center-mode decoupling** --- the module docstring's "exact
   spin-symmetric relative-mode reduction of the two-mode construction". The
   uniform phonon mode couples to the conserved total electron number and is
   exactly decoupled, so the online algebra keeps `L - 1` modes of `L`. The
   reduction **survives in form and collapses in value**: at two sites it
   halves the Weyl coordinates, `4 -> 2`, and takes the degree-two monomials
   from `C(5,2) = 10` to `C(3,2) = 3`, a factor `3.33`; at five sites
   `10 -> 8` and `55 -> 36`, a factor `1.53`; at ten sites `210 -> 171`, a
   factor `1.23`. The saving decays as `(1 - 1/L)^2`. This is the largest
   structural saving in the two-site construction and it is the one that goes
   away.

3. **The one-term product law of `_pauli_product`.** `sigma_a sigma_b` is a
   *single* basis element --- the identity when `a = b`, one `sigma_c`
   otherwise --- because the symmetric structure constants `d_abc` vanish for
   `su(2)`. This is an algebraic identity of two sites, not a symmetry, and
   it **fails at `L >= 3`**, where a product of two generalized Gell-Mann
   generators carries the identity trace part, the symmetric `d` terms and
   the antisymmetric `f` terms together. A word pair's generated spin support
   therefore goes from one canonical pair to as many as `L^2 (L^2 + 1) / 2`.
   This is what makes the frontier grow about two powers of `L` faster than
   the dictionary that generates it.

4. **The shared correlation trace** of the 31-coordinate chart: a two-site
   Cayley--Hamilton identity, measured above, gone at `L >= 3`.

### The frontier

The frontier keys are the moments generated by `_operator_product` over every
pair of extension rows that are not already archive-resolved or active. There
is no closed form here; the counts below are enumerated over all word pairs
(`576`, `18,769`, `222,784` and `1,490,841` pairs at the entrance dictionary
for `L = 2`, `3`, `4`, `5`), with the fitted log--log exponent over those
four site counts in the last column.

| quantity, entrance dictionary | 2 | 3 | 4 | 5 | fit |
|---|---:|---:|---:|---:|---|
| active keys | 15 | 116 | 435 | 1164 | `L^4.75` |
| extension rows | 24 | 137 | 472 | 1221 | `L^4.29` |
| propagated state coordinates | 44 | 199 | 614 | 1493 | `L^3.84` |
| frontier keys | 62 | 979 | 6096 | 24330 | `L^6.52` |
| right-hand-side frontier keys | 15 | 183 | 901 | 2924 | `L^5.76` |
| outer extension rows | 60 | 667 | 3419 | 11770 | `L^5.77` |
| unified Gram rows | 62 | 669 | 3421 | 11772 | `L^5.73` |

| quantity, 27-key analogue | 2 | 3 | 4 | 5 | fit |
|---|---:|---:|---:|---:|---|
| active keys | 27 | 260 | 1155 | 3564 | `L^5.33` |
| extension rows | 36 | 281 | 1192 | 3621 | `L^5.03` |
| propagated state coordinates | 56 | 343 | 1334 | 3893 | `L^4.63` |
| frontier keys | 70 | 1555 | 12096 | 57930 | `L^7.33` |
| outer extension rows | 82 | 1181 | 7575 | 31760 | `L^6.50` |
| unified Gram rows | 84 | 1183 | 7577 | 31762 | `L^6.48` |

At the whole hidden dictionary the frontier is `129`, `3347`, `31458`,
`177654` (`L^7.89`).

The frontier is bounded above by the keys the word degrees allow. For a
dictionary whose largest boson word degree is two --- the entrance dictionary
and the 27-key analogue --- the generated boson degree is at most four and
the ceiling is `(1 + c1 + c2) * (M(0) + ... + M(4)) - 1` minus the lower
keys: `120`, `2979`, `27992` and `159450` at the entrance dictionary. The
enumerated frontier is `0.52`, `0.33`, `0.22` and `0.15` of it, so the bound
is loose and loosening; the frontier is a genuine union of generated
products, not a shell count. (The ceiling does not apply to the hidden
dictionary, which contains pure-boson degree-three keys and therefore reaches
boson degree six.)

**Assumptions for the outer extension.** `projective_guard_outer_extension`
enlarges the halfwords by the right-hand-side frontier keys and their
Hamiltonian descendants. The counting model uses a Hamiltonian word basis of
`8 (L - 1)` entries --- per spin species, `L - 1` nearest-neighbour hopping
generators, `L - 1` diagonal drive generators and `L - 1` diagonal coupling
generators each carrying one relative position coordinate, plus `L - 1`
squared position and `L - 1` squared momentum words --- which equals the
committed `_HAMILTONIAN_OPERATOR_BASIS` exactly at `L = 2` (`8` words) and
assumes an open nearest-neighbour Holstein chain with uniform `omega`,
uniform coupling and a diagonal drive.

---

## (c) The positive-semidefinite cone side dimension

The record defines it as twice the spectral bundle rank, "because
`_realify_hermitian` doubles the 62-row complex Hermitian block"
(`cone-solve-instances-20260915.md`). Two cones carry that definition and
both reach `80` at two sites:

* the **declared completion's** cone, the realified extension Gram, side
  `2 * (extension rows)`: `48`, `72`, `80` at 15, 27 and 31 entrance keys.
  The `80` is the whole hidden dictionary's `40` rows realified.
* the **carried-witness spectral bundle's** cone, side `2 * rank` with the
  rank bounded by the 62-row unified Gram
  (`restore_bundle_rank_hint` admits `0` to `61`). Its recorded walk is
  `12 -> 52 -> 62 -> 70 -> 72 -> 76 -> 78 -> 80` over eight atoms to
  `t = 0.01`, i.e. rank `40` of `62`.

Growth, checked against the committed values at `L = 2`:

| cone | formula | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| completion cone, entrance dictionary | `2 (2L^2 + 2L - 3 + c2 + c1 M(2))` | 48 | 274 | 944 | 2442 | 5272 | 10058 | 17544 |
| completion cone, 27-key analogue | `2 (2L^2 + 2L - 3 + c2 (1 + M(1)) + c1 M(2))` | 72 | 562 | 2384 | 7242 | 17872 | 38282 | 73992 |
| completion cone, hidden dictionary | `2 (2L^2 + 2L - 3 + M(3) + c1 M(2) + c2 (1 + M(1)))` | 80 | 602 | 2496 | 7482 | 18312 | 39010 | 75112 |
| bundle ceiling, entrance dictionary | `2 (outer extension rows + 2)` | 124 | 1338 | 6842 | 23544 | --- | --- | --- |
| bundle ceiling, 27-key analogue | the same | 168 | 2366 | 15154 | 63524 | --- | --- | --- |

The unified Gram is the extension with the one decoupled mode's two rows
restored --- `unified_guard_dimension` is `11 + (dimension - 9)` at two
sites, which is `dimension + 2` at every `L`, the literal core being
`2L^2 + 2L - 1` rows against the relative base's `2L^2 + 2L - 3`.

---

## Polynomial or combinatorial

**Polynomial in `L` at the construction's fixed moment degree three:**

* the closed scalar coordinates, `2L^3 + 3L^2 + 3L - 1`, cubic;
* the raw moment coordinates, `2L^3 + 3L^2 + L - 1`, cubic;
* the entrance dictionary, `c2 + c1 M(2)`, leading term `(5/2) L^4`, quartic;
* the two-spin/one-boson promotion block and therefore the 27-key analogue
  and the whole hidden dictionary, leading term `L^5`, quintic;
* the extension rows and the completion cone side, which are the dictionary
  plus `2L^2 + 2L - 3`, so quartic or quintic with it;
* the propagated state coordinates, raw chart plus dictionary, quartic or
  quintic.

These are polynomial only because the moment degree is held at three. The
degree enters through `M(b) = C(b + 2L - 3, 2L - 3)`, a binomial: raising the
retained degree by one multiplies the boson factor by roughly `2L/b`, so the
construction is **polynomial in the sites at fixed degree and binomial in the
degree at fixed sites**.

**Combinatorial, with no closed form derived here:**

* the frontier keys, the union of `_operator_product` supports over all word
  pairs --- `L^6.52` (entrance), `L^7.33` (27-key analogue), `L^7.89`
  (hidden) fitted over `L = 2` to `5`, about two powers above the dictionary
  that generates them, and at `0.52` down to `0.15` of the degree ceiling;
* the right-hand-side frontier keys, `L^5.76`;
* the outer extension rows and hence the unified Gram and the bundle ceiling,
  `L^5.77` (entrance) and `L^6.50` (27-key analogue), because the halfword
  enlargement is the transitive closure of the Hamiltonian commutator on the
  right-hand-side frontier.

---

## What the growth costs, as a model

Two exponents, both taken from the record and labelled where they enter, and
one trajectory held fixed (`t = 0` to `20`, `h = 0.0025`, re-selection
cadence 16, so 500 completion solves whatever `L` is).

* completion solve against cone side: exponent `3.274`, fitted on the 36
  recorded conic instances (`0.029 s` at side `12`, `14.462 s` at side `80`);
  anchored at the measured `8.325 s` per solve at side `72`
  (`5,183.6 s * 0.803 / 500`).
* exact cutoff-16 driven reference: wall taken linear in the Hilbert
  dimension `L^2 * 17^L` times `L` (sparse Hamiltonian), anchored at `0.7 s`
  for `t = 0` to `4` at dimension `1,156`, scaled by five to `t = 20`.

| `L` | cone side, 27-key | s per completion solve | s per `t = 0` to `20` rollout | exact Hilbert dimension | exact s, `t = 0` to `20` | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 72 | `8.33e0` | `5.18e3` | 1,156 | `3.50e0` | `1.48e3` |
| 3 | 562 | `6.96e3` | `4.33e6` | 44,217 | `2.01e2` | `2.16e4` |
| 4 | 2,384 | `7.90e5` | `4.92e8` | 1,336,336 | `8.09e3` | `6.08e4` |
| 5 | 7,242 | `3.00e7` | `1.87e10` | 35,496,425 | `2.69e5` | `6.96e4` |
| 6 | 17,872 | `5.78e8` | `3.60e11` | 868,952,484 | `7.89e6` | `4.56e4` |
| 8 | 73,992 | `6.06e10` | `3.77e13` | 446,448,476,224 | `5.41e9` | `6.98e3` |

The ratio rises to `L = 5`, turns over, and reaches one near `L = 14`, where
the modelled rollout is `3.18e17 s`. At a phonon cutoff of four rather than
sixteen the turnover moves to about `L = 31`.

**That table is a floor, and the companion round key says by how much.** The
`conic-cost-law` study of the same round
(`agent_guidance/paper-v/overnight/scaling-conic-cost-law-20260920.md`,
its result not mine) finds the exponent is not a constant of the problem ---
`3.24 +/- 0.11` fitted over cone side `12` to `80`, `4.17 +/- 0.04` over
`40` to `80`, `4.87 +/- 0.12` over `84` to `104` --- and fits
`per_iteration_seconds = a m^1.5 + c m^3` in the constraint rows
`m = n (n + 1) / 2`, valid over side `32` to `104`. Two readings follow for
the sites question. That law evaluated at side `72` gives `0.325 s` per
iteration, which at the record's most frequent bundle iteration count (22)
is `7.1 s` against the `8.325 s` anchor above, so the two cost statements
agree where they overlap. And its dominant term is cubic in the rows, i.e.
**sixth power in the cone side**: extrapolated (far outside its stated
validity) it puts one completion solve at `1.88e6 s` at three sites,
`1.10e10 s` at four and `8.62e12 s` at five, moving the crossover from
`L = 14` to `L = 31`, at a modelled rollout of `1.70e39 s`. The truth for
this construction is at least the `3.274` table and at most that; neither
end changes the sign of the answer.

Two hard facts bound the model. The exact reference's state vector alone is
`0.568 GB` at five sites and `13.90 GB` at six at cutoff sixteen, so beyond
`L = 5` the exact side cannot be run under the 18 GB budget either: past
five sites the table compares two computations neither of which exists. And
the model understates the closure on a second axis as well, because it
scales only the cone side and not the frontier variable count, which grows
about two powers of `L` faster.

**The one structural option the counting supports.** The quartic comes from a
single choice in the key algebra: the entrance dictionary takes `c2` two-spin
keys, which is *every* pair of electronic generators across the two species,
and `c1 M(2)` one-spin/two-boson keys, which is every generator against every
degree-two monomial over all `L - 1` modes. Nothing in `MomentKey` or in the
entrance predicate refers to locality. A dictionary restricted to generator
pairs and monomials supported on one bond or one neighbouring pair would have
`O(L)` keys, `O(L)` extension rows and a cone side linear in `L`, at which
point the solve grows as a low power of `L` (between `L^3.3` and `L^6` on the two fitted windows) against `17^L` and the route's arithmetic
changes sign. That restriction is not measured here and is not in the
committed construction; it is the thing to size before any `L`-site closure
is written.

---

## Caveats

1. Nothing was propagated, prepared, solved or promoted. Every number is a
   count or a model; none is adopted evidence.
2. The counting model is not an `L`-site closure. It widens the electronic
   basis to `su(L)` and the boson algebra to `L - 1` modes and replays the
   committed definitions; it says how large the same construction would be,
   not that it would be correct or that its flow would close.
3. It is validated only at `L = 2`, where it reproduces every committed
   count. At `L >= 3` its electronic product support is taken from the
   generalized Gell-Mann structure constants; a different Hermitian basis
   would not change the counts, but a different *Hamiltonian* --- longer
   range hopping, non-uniform `omega`, a non-diagonal drive --- would change
   the right-hand-side frontier and the outer extension, which are the two
   counts that depend on `_HAMILTONIAN_OPERATOR_BASIS`.
4. The `L >= 3` correlation-trace result is analytic (trace cyclicity on the
   paired sources) plus two measurements at `L = 2` (the bare Pauli source's
   trace identity to `6.77e-17`, the chart invariance to `3.43e-16`) and the
   Cayley--Hamilton spread at `L = 2` to `5`. The `L >= 3` block itself was
   not implemented.
5. The fitted exponents are over four site counts, `L = 2` to `5`, and are
   still drifting downward at `L = 5` (the entrance frontier's successive
   ratios give `6.81`, `6.36`, `6.20`); read them as the order, not as
   converged powers.
6. The cost table is a model with the two exponents named above. The 500
   solves and the step count are held at the two-site trajectory's values;
   whether an `L`-site trajectory needs the same step or the same
   re-selection cadence is unmeasured.

## Files

* `output/local_runs/paper_v_scaling_20260920/probe_two_site_counts.py` ---
  reads the two-site counts from the committed modules.
* `output/local_runs/paper_v_scaling_20260920/count_l_site_construction.py`
  --- the `L`-site counting model; `--sites 2` is its validation.
* `output/local_runs/paper_v_scaling_20260920/check_correlation_trace_source.py`
  --- the Eq. (14d) term-by-term trace measurement.
* `output/local_runs/paper_v_scaling_20260920/summarize_scaling.py` --- the
  tables and the cost model.
* `output/local_runs/paper_v_scaling_20260920/counts_L2_L5.json` --- the
  enumerated counts at `L = 2` to `5`.
