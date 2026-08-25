# Static ADAPT runtime profile — where the time actually goes (2026-08-25)

Answer to: *"how to speed up our basic static ADAPT — should we use PennyLane,
PyTorch, CUDA, or is it already maximally quick?"*

**Short answer: none of those three.** The statevector numerics are **5.5% of
wall clock**. The other 94.5% is pure-Python Pauli-string bookkeeping, defensive
`deepcopy`, and Qiskit transpilation for cost metrics. PennyLane/PyTorch/CUDA all
attack the 5.5%; Amdahl caps that at **1.06×** even if the linear algebra became
free. The available headroom without any new dependency is **~2×**, and it is all
in code this repo already owns.

## 1. What was measured

`cProfile` over a canonical Paper-I Hubbard--Holstein ADAPT run:

| Setting | Value |
|---|---|
| Problem | `hh`, `L=2`, `t=1.0`, `U=8.0`, `omega0=1.0`, `g_ep=2.0` |
| Boson truncation | `n_ph_max=7`, binary encoding, `blocked`, PBC |
| Register | 4 fermion + 2 sites x 3 boson = **10 qubits**, `dim = 2^10 = 1024` |
| Pool | `hva` (87 available candidates per iteration) |
| Inner optimizer | SPSA, `maxiter=400`, `adapt_reopt_policy="full"` |
| Reached | `depth=30` (continuation `phase3_v1` drove past `max_depth=10`) |
| Entry point | `pipelines.static_adapt.adapt_pipeline._run_hardcoded_adapt_vqe` |

Two runs: **172.7 s** cold cache, **108.6 s** warm (`raw_outputs/cache` populated
by the phase-0 screen). Buckets below are the warm run, which is the steady state.

Profile artifact: `<scratchpad>/adapt.prof`; driver `<scratchpad>/prof_adapt.py`.

## 2. Cost buckets (warm run, 107.8 s profiled, `tottime`)

|  Time |  Share | Bucket |
|------:|-------:|---|
| 31.7 s | 29.4% | builtins/C (`sum`, `sorted`, `len`, `isinstance`, `str.lower/strip/replace`, `dict.get`) |
| 23.3 s | 21.6% | `pipelines/static_adapt/sector_invariants.py` — Pauli-word string algebra |
| 16.0 s | 14.9% | other repo Python |
| 11.5 s | 10.7% | `copy.deepcopy` |
| 11.0 s | 10.2% | Qiskit circuit build + transpile |
| **5.9 s** | **5.5%** | **statevector numerics** (`pauli_actions`, `compiled_ansatz`, `compiled_polynomial`) |
| 2.8 s | 2.6% | numpy misc |
| 2.4 s | 2.3% | JSON encode/decode |
| 2.0 s | 1.8% | hashing / state fingerprints |

The 29.4% "builtins" bucket is not independent: `str.lower` (1.41 s), `str.strip`
(1.26 s), `str.replace` (1.22 s) and `sorted` (2.36 s) are all called *from*
`_canonical_pauli_word`, and `sum` (2.75 s) is the `_pauli_words_commute`
generator. Charged to their callers, the **sector-invariant string algebra is
~32 s, or 30% of the run, in one subsystem.**

### Top functions, warm run

```
ncalls      tottime  cumtime  function
11,225,716   10.477   16.330  sector_invariants.py:146  _canonical_pauli_word
12,202,468    6.959   15.610  copy.py:118               deepcopy
 5,876,641    5.688    5.688  qubitization_module.py:48 PauliTerm.pw2strng
 5,369,198    5.375   27.619  sector_invariants.py:209  _pauli_words_commute
     1,742    4.631    4.631  {vf2_layout_pass_average}          (Qiskit)
20,467,882    3.347    3.347  sector_invariants.py:216  <genexpr>
       687    3.225   38.973  sector_invariants.py:226  audit_generator_sector_contract
   788,740    3.035    3.863  pauli_actions.py:231      apply_compiled_pauli   <-- the physics
     1,762    1.782    2.854  adapt_circuit_execution.py:63 _reference_state_digest
       871    1.692    1.692  {sabre_layout_and_routing}         (Qiskit)
     1,742    1.678    4.549  quantumcircuit.py:4082    QuantumCircuit.depth
```

`audit_generator_sector_contract` is **38.97 s cumulative — 36% of the run — from
687 calls.** It is a *verification* pass, not physics.

The cold run additionally showed `projective_state_fingerprint` at 15.2 s / 26.0 s
cum and `json.encoder.iterencode` at 12.5 s; both collapse once the phase-0 disk
cache is warm, so they are a **cold-start** tax, not a steady-state one.

## 3. Why PennyLane / PyTorch / CUDA are the wrong lever here

The register is `dim = 1024`, i.e. **16 KB of `complex128`** — it lives in L1.

Measured kernel cost (`src/quantum/pauli_actions.py`):

| `nq` | `dim` | `apply_compiled_pauli` | `apply_exp_term` | bare `np.copy` floor |
|---:|---:|---:|---:|---:|
| 10 | 1,024 | 3.86 us | 6.69 us | 0.50 us |
| 14 | 16,384 | 50.16 us | 68.37 us | 4.82 us |
| 20 | 1,048,576 | 7,316.70 us | 11,727.81 us | 1,613.12 us |

- **CUDA**: a kernel launch is ~5--10 us. The *entire* `apply_exp_term` at `nq=10`
  is 6.69 us. A GPU round trip is slower than the whole operation, before any
  host/device transfer. There is also no CUDA on this machine (`darwin`, 8 CPUs,
  16 GB) — MPS at best. GPU only becomes defensible at `nq >~ 20`, and only if the
  entire ADAPT loop lives on-device.
- **PyTorch**: higher per-op dispatch overhead than NumPy on 16 KB arrays. It buys
  autograd — but the inner optimizers in use are SPSA/POWELL/ROTOSOLVE, which are
  gradient-free by construction, and the ADAPT selection gradient is already an
  exact commutator (`gradient_source: "exact_commutator"`), not backprop. Nothing
  to differentiate.
- **PennyLane**: a framework layer *above* the kernel this repo already has. It
  would add dispatch overhead to the 5.5% and would not touch the 94.5%. It also
  imports its own Pauli/operator abstractions, which is precisely the layer whose
  string handling is the current bottleneck.

None are installed (`torch`, `pennylane`, `jax`, `numba`, `cupy`: absent).

## 4. Ranked, evidence-backed speedups

Measured micro-benchmarks, not estimates:

### (1) Memoize `audit_generator_sector_contract` — up to ~36%

`sector_invariants.py:226`. It is a pure function of
`(serialized polynomial, groups, total_qubits, tolerance)`. The pool is fixed
across ADAPT iterations, so the same generators are re-audited every depth. Key it
on `canonical_sha256(serialize_polynomial_terms_exyz(polynomial))` plus the group
signature. **This alone subsumes most of items (2)--(4).**

### (2) `lru_cache` on `_canonical_pauli_word` — measured 7.5x

`sector_invariants.py:146`. Pure `str -> str`, called 11.2M times on a vocabulary
of ~87 ten-character words.

```
current   0.387 us/call
lru_cache 0.052 us/call     -> 7.5x
```

Saves ~9 s (8%) directly, plus most of the `str.lower/strip/replace/sorted`
builtins charged above. One decorator.

### (3) Bitmask `_pauli_words_commute` — measured 12.3x

`sector_invariants.py:209`. Two Pauli words commute iff
`popcount(x1 & z2) + popcount(x2 & z1)` is even. The `(x, z)` masks are already
computed elsewhere in this repo by
`src/quantum/pauli_actions.compile_pauli_action_exyz` (`flip_mask`, `phase_mask`);
`sector_invariants` does not use them and re-derives everything from strings.

```
current (string zip + genexpr)  1.498 us/call
bitmask popcount parity         0.121 us/call   -> 12.3x
agreement over 1,600 word pairs: 0 disagreements
```

Saves ~8 s (7%), including the 20.5M-call `<genexpr>` at line 216.

### (4) Cache `PauliTerm.pw2strng()` — ~5%

`src/quantum/qubitization_module.py:48` rebuilds the word by per-qubit string
concatenation on every call: 5.9M calls, 5.69 s. Cache lazily on the term and
invalidate on mutation (or hoist it out of the `_commutator_l1_norm` double loop,
where the right-hand word is recomputed once per left-hand term).

### (5) Stop deep-copying candidate-cache reads — ~14%

All 12.2M `deepcopy` calls trace to
`pipelines/static_adapt/adapt_candidate_record_cache.py:208
_candidate_record_cache_get` (**15.19 s attributed**). It defensively deep-copies
every cached record on retrieval. Return frozen/immutable records — or copy only
on the write path — and this bucket goes to ~0.

### (6) Cache or elide the Qiskit transpile — ~10%

`vf2_layout_pass_average` 4.63 s + `sabre_layout_and_routing` 1.69 s +
`QuantumCircuit.depth` 1.68 s + `_append_standard_gate` 1.15 s. This is
`transpile(..., optimization_level=1)` computing two-qubit count / depth cost
metrics. Circuits repeat across candidates and depths — memoize by circuit
digest, drop to `optimization_level=0` for cost proxies, or compute the two-qubit
count analytically from the rotation layout.

### (7) Turn on the parallelism that already exists

The profiled run logged `gradient_parallel_backend: "serial"`,
`gradient_parallel_effective_workers: 1`, and
`hardcoded_adapt_phase2_full_feature_parallel ... disabled_reason:
"workers_leq_one"` (that phase 2 stage alone was 10.1 s at one worker). The knob
is `--adapt-parallel-gradient-workers` (`cli_config.py:2736`), with CHTC-aware
CPU detection already in `adapt_worker_limits.py`. 8 cores are available locally.

### (8) Only after all of the above: fuse the statevector kernel

`apply_exp_term` at `nq=10` is 6.69 us against a 0.50 us single-copy floor — ~13x
off, because it makes 5--6 separate passes over the array (int64 gather, `*=
signs`, `*= prefactor`, then `cos*psi - i sin*P psi`), each allocating a
temporary. A fused kernel would recover most of that. **But the whole bucket is
5.5%**, so the best case is ~5% — strictly less than any single item above.

## 5. Expected combined effect

Items (1)--(6) address roughly 45--50% of warm wall clock, i.e. **about 2x**, with
no new dependency and no change to any numerical result. Items (2) and (3) are
small, local, and independently verifiable; (1) and (5) are the large ones.

Item (7) multiplies on top of that for the pool-screening phases.

## 6. When the GPU question becomes real

The lever only inverts if the register grows. At `nq=20` the kernel is 7.3 ms per
Pauli application and the numerics would dominate. That means the honest trigger
is a change to `(L, n_ph_max)` — e.g. `L=4` at `n_ph_max=7` gives
`8 + 4x3 = 20` qubits — not a change of framework at `L=2`, `n_ph_max=7`. At
`dim = 1024` the machine is not the constraint; the bookkeeping is.

## 7. Reproduce

```
python3 <scratchpad>/prof_adapt.py       # cProfile -> adapt.prof, prints top 35
```

Caveat: the profiled configuration is one point in the matrix (HVA pool, SPSA,
`n_ph_max=7`). The bucket *ordering* is expected to be stable across regimes
because `audit_generator_sector_contract` and the candidate-record cache are
pool-size driven, not regime driven — but a POWELL/ROTOSOLVE overlay with full
re-optimization will shift more weight onto the optimizer's repeated state
preparations, i.e. onto item (8). Worth re-profiling once (1)--(5) land.

---

# Appendix A — what the audit actually commutes, and what the masks are

Added 2026-08-25 in response to two questions on the notation used above. Every
claim here is read off the code, not asserted.

## A.1 The commutator is against the JW fermion-number operators, per spin, on the fermion block only

Yes — it is a number operator, but it is neither the total particle number nor
anything phonon-facing. `_count_operator` (`sector_invariants.py:119`) builds

$$\hat N_G \;=\; \sum_{q \in G} \hat n_q, \qquad \hat n_q = \frac{I - Z_q}{2}$$

and drops the identity part, because $I$ commutes with everything. What the code
literally emits is therefore

$$\hat N_G \;\mapsto\; \sum_{q \in G} \left(-\tfrac{1}{2}\right) Z_q$$

— one `PauliTerm` per qubit in $G$, coefficient `pc=-0.5`, word `e...eze...e`.

**Which $G$?** Not a fixed choice; it is resolved from the problem registry's
declared sector by `resolve_fixed_count_qubit_groups` (`sector_invariants.py:52`),
which keeps only `FixedCountConstraint` entries and maps them to physical qubits:

| declared quantity | group $G$ |
|---|---|
| `n_f` / `n_fermion` / `fermion_number` / `particle_number` | every qubit in the fermion block |
| `n_up` / `n_alpha` | the spin-up modes only |
| `n_dn` / `n_down` / `n_beta` | the spin-down modes only |

For `family_key == "hh"` the registry declares
(`builders/problem_registry.py:571`):

```
FixedCountConstraint(quantity="n_up", value=n_up, scope="fermion_register")
FixedCountConstraint(quantity="n_dn", value=n_dn, scope="fermion_register")
TruncationConstraint(quantity="phonon_occupancy", max_local_occupancy=n_ph_max,
                     scope="boson_register")
```

So the Hubbard--Holstein audit runs **two** groups, and takes **two** commutators
per candidate generator:

$$[\hat N_\uparrow, \hat A_k] \quad\text{and}\quad [\hat N_\downarrow, \hat A_k]$$

At $L=2$, `blocked` ordering, the resolved groups are
$G_\uparrow = \{q_0, q_1\}$ and $G_\downarrow = \{q_2, q_3\}$, each with target
occupancy $1$ (half filling).

**The phonon register is deliberately not audited.** `phonon_occupancy` is a
`TruncationConstraint`, and `resolve_fixed_count_qubit_groups` only consumes
`FixedCountConstraint`. The module docstring states the reason: truncation
"describe[s] the finite computational representation, not a conserved phonon-number
law". That is physically correct — the Holstein coupling
$g\,\hat n_i (\hat b_i + \hat b_i^\dagger)$ means $[\hat H, \hat N_{\rm ph}] \neq 0$,
so there is no phonon-number sector to preserve. Only $\hat N_\uparrow$ and
$\hat N_\downarrow$ are conserved, and only those are checked.

## A.2 The norm is a Pauli-basis coefficient $\ell_1$ norm, not an operator norm

`_commutator_l1_norm` (`sector_invariants.py:176`) expands
$[\hat N_G, \hat A_k]$ symbolically in the Pauli basis, accumulates every
contribution to each resulting word, sums each word's coefficient with
`math.fsum` (real and imaginary parts separately, so exact cancellation survives
floating point), and returns

$$\big\| [\hat N_G, \hat A_k] \big\|_{1}^{\rm Pauli} \;=\; \sum_{P} \Big| \sum_{\text{contributions to } P} c \Big|$$

A generator passes when this is $\le$ `tolerance` (default $10^{-10}$). This is
the "cancellation-stable" wording in its docstring: the point is that a grouped
generator's components can produce large individual terms that cancel exactly.

## A.3 Grouped vs componentwise — why the audit is run twice per generator

This is the distinction the module exists for. For each group $G$,
`audit_generator_sector_contract` computes both:

- **grouped**: $\big\|[\hat N_G, \hat A_k]\big\|_1$ for the whole generator
  $\hat A_k = \sum_j c_j P_j$;
- **componentwise**: $\max_j \big\|[\hat N_G, c_j P_j]\big\|_1$, each Pauli
  component alone.

A generator can pass grouped and fail componentwise. That is exactly the case the
docstring flags: such a generator is sector-safe only as
`execution_mode="grouped_exact"` and **must not** be given independent per-component
angles — hence `requires_logical_shared_parameterization`. If it were executed
`termwise_product` with free angles, the optimizer could leave the
$(N_\uparrow, N_\downarrow) = (1,1)$ sector.

That is also why this is $O(|\text{components}|)$ commutator expansions per
generator per group, and why it costs 36% of wall clock at 687 calls.

## A.4 The $(x, z)$ masks are the binary-symplectic form — and this repo already stores them

An $n$-qubit Pauli word $P$, up to phase, is fully determined by two $n$-bit
integers:

$$x = \sum_q 2^q\,[\,P_q \in \{X, Y\}\,], \qquad z = \sum_q 2^q\,[\,P_q \in \{Z, Y\}\,]$$

| symbol | $x$ bit | $z$ bit |
|---|---:|---:|
| `e` ($I$) | 0 | 0 |
| `x` | 1 | 0 |
| `y` | 1 | 1 |
| `z` | 0 | 1 |

**These are not new symbols to introduce — they are already
`CompiledPauliAction.flip_mask` and `.phase_mask`**, produced by
`compile_pauli_action_exyz` in `src/quantum/pauli_actions.py`. Verified directly:

```
e: flip_mask(x)=0 phase_mask(z)=0
x: flip_mask(x)=1 phase_mask(z)=0
y: flip_mask(x)=1 phase_mask(z)=1
z: flip_mask(x)=0 phase_mask(z)=1
```

**What they are used for today.** They are the statevector kernel's entire
representation of a Pauli word — the module docstring of `pauli_actions.py` states
the action as

$$P\,|i\rangle \;=\; i^{\,n_y}\,(-1)^{\,\mathrm{popcount}(i \wedge z)}\;|\,i \oplus x\,\rangle$$

so $x$ is the basis-index XOR (the permutation, `flip_mask`) and $z$ is the sign
parity mask (`phase_mask`). This is why `apply_compiled_pauli` never builds a
matrix. The masks are constant-size — two integers per word regardless of $n$ —
which is precisely why that file stores them instead of $2^n$-long tables.

**What they are not used for today.** `sector_invariants` does not touch them. It
re-derives commutation from the character string on every call:
`_canonical_pauli_word` does `strip().lower().replace("i","e")` plus a
`sorted(set(...))` validation, 11.2M times, and `_pauli_words_commute` then does a
Python `zip` + generator `sum` over the characters, 5.4M times. Two representations
of the same object, one fast and already present, one slow and used in the hot path.

**The commutation predicate in that representation.** Two Pauli words commute iff

$$\langle P_i, P_j \rangle \;=\; \mathrm{popcount}(x_i \wedge z_j) \;\oplus\; \mathrm{popcount}(x_j \wedge z_i) \;=\; 0$$

where $\wedge$ is bitwise AND and $\oplus$ is XOR of the two parities (equivalently,
the sum taken mod 2). Per-qubit truth table, which reproduces the current
implementation's rule "both non-identity and different" exactly:

| $P_i$ | $P_j$ | $x_i z_j + x_j z_i \bmod 2$ | current rule |
|---|---|---:|---|
| `e` | any | 0 | commute |
| `x` | `x` | 0 | commute |
| `x` | `y` | 1 | anticommute |
| `x` | `z` | 1 | anticommute |
| `y` | `y` | 0 | commute |
| `y` | `z` | 1 | anticommute |
| `z` | `z` | 0 | commute |

Empirically checked over 1,600 word pairs at $n=10$: **0 disagreements**, at
**12.3×** the speed (1.498 us $\to$ 0.121 us per call).

## A.5 What this means for the two proposed changes

- Item (1) memoizes the audit of A.1--A.3. It changes no norm and no verdict; it
  stops recomputing $[\hat N_\uparrow, \hat A_k]$ and $[\hat N_\downarrow, \hat A_k]$
  for a pool $\{\hat A_k\}$ that is invariant across ADAPT depths.
- Item (3) replaces the string form of A.4 with the mask form the kernel already
  holds. It is a representation change inside one predicate, not a change to the
  sector contract.

Neither alters the fixed-count contract, the tolerance, or which generators are
admitted.

---

# Appendix B — measured outcome of items (1)--(3) (2026-08-25)

Implemented on branch `perf/sector-audit-memoization`, commit `a1ab4085`, in
`pipelines/static_adapt/sector_invariants.py`:

1. `audit_generator_sector_contract` memoized on the generator's exact Pauli
   components + fixed-count groups + register width + tolerance;
2. `_pauli_words_commute` computed from the binary-symplectic masks
   (`CompiledPauliAction.flip_mask` / `.phase_mask`) instead of walking the word;
3. `_canonical_pauli_word` memoized on its `str` path.

Rows are copied out of the cache so `audit_candidate_pool_sector_contract` can
still write `pool_index` into them. Non-`str` inputs to `_canonical_pauli_word`
keep the original body, so the raised message is unchanged for every input type.

## B.1 Wall clock

Same canonical run as Section 1 (HH `L=2`, `n_ph_max=7`, HVA pool, SPSA, depth
30), both states measured back-to-back in the same worktree:

| State | Cold cache | Warm cache |
|---|---:|---:|
| Baseline (`603d30c0`) | 160.44 s | 94.60 s |
| With items (1)--(3) (`a1ab4085`) | **79.19 s** | **79.23 s** |
| Speedup | **2.03x** | **1.19x** |

The change makes the phase-0 disk cache nearly irrelevant: cold and warm now
land within 0.04 s of each other, because the work that cache was hiding is the
work that is no longer repeated.

## B.2 The audit itself

| Metric | Baseline | After |
|---|---:|---:|
| `audit_generator_sector_contract` calls | 687 | 687 |
| audits actually computed | 687 | **166** (76% served from cache) |
| audit cumulative time | 30.11 s | **14.78 s** |
| `sector_invariants` bucket (`tottime`) | 18.0 s | **8.8 s** |

166 distinct audits remain because the pool is not perfectly invariant across
depths; those are genuinely different generators, and the symplectic predicate
is what halves their cost.

## B.3 Correctness evidence

- **Verdict parity**: 132 audit rows over six problem/pool combinations
  (`L=2/4`, `n_ph_max=1/7`, fermionic and HVA pools), captured before the edit
  and recomputed after — **bit-identical**, including every
  `grouped_commutator_l1` / `max_component_commutator_l1` value, both branches
  of `requires_logical_shared_parameterization`, and both branches of
  `execution_preserves_fixed_counts`. Plus a 40x40 commute-table digest, equal.
- **Energy**: all four timing runs returned `E = 5.876936059036121`, identical
  to the last digit.
- **Test suites**: 134 files (`test_ra_adapt_*`, `test_adapt_*`,
  `test_static_adapt_*`, `test_h2o_linear_fd_correctness_audit`) run at both
  states. Baseline `140 failed, 2064 passed, 10 skipped, 4 errors`; after
  `140 failed, 2070 passed, 10 skipped, 4 errors`. The FAILED/ERROR sets are
  **byte-identical (144 lines each, zero new, zero fixed)**; the +6 passes are
  exactly the six regression tests added with the change. All 140 failures are
  pre-existing and spread across 40+ unrelated files (`test_adapt_vqe_integration`
  29, `test_ra_adapt_bundles` 14, `test_static_adapt_sr_snake_*` 22, ...).
- **Impact analysis**: GitNexus rated all three edited symbols HIGH upstream
  risk (`audit_generator_sector_contract` 18 impacted / 4 direct;
  `_pauli_words_commute` 12 / 1; `_canonical_pauli_word` 8 / 3; zero affected
  processes in every case). The verdict-parity capture above is the mitigation
  that HIGH rating calls for.

## B.4 Where the time goes now (warm, 78.5 s)

|  Time |  Share | Bucket | vs baseline |
|------:|-------:|---|---|
| 23.5 s | 29.9% | builtins/C | 29.4 s |
| 18.7 s | 23.8% | other repo python | 19.1 s |
| 10.6 s | 13.4% | `deepcopy` | 10.4 s |
| 10.4 s | 13.2% | Qiskit | 10.3 s |
| 8.8 s | 11.2% | `sector_invariants` | **18.0 s** |
| 5.7 s | 7.2% | statevector numerics | 5.7 s |

The next two levers are unchanged and are items (5) and (6) of Section 4 —
the candidate-record-cache `deepcopy` (13.4%) and the Qiskit cost-metric
transpile (13.2%), ~26% combined. The statevector numerics are still 7.2%, so
the Section 3 conclusion stands: PennyLane, PyTorch, and CUDA remain the wrong
lever at `dim = 1024`.
