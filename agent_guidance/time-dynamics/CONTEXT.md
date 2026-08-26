# Paper II: AP-McLachlan Time Dynamics

This glossary defines the domain language used when planning, implementing,
testing, and reporting the Paper-II time-dynamics method.

## Mathematical method

**Active support**:
The ordered coordinate set (J_k) used by the variational state at checkpoint
(k).
_Avoid_: Ansatz size when order or coordinate identity matters

**Generalized exchange patch**:
One pair (mathcal B_k=(D_k,A_k)), where (D_k\subseteq J_k) is the set of
active coordinates proposed for deletion and (A_k) is the ordered set of
positioned occurrences proposed for insertion. The patched support is
(J_k^{\mathcal B}=(J_k\setminus D_k)\cup A_k).
_Avoid_: Append/prune pipeline, paired selectors

**Insert face**:
The boundary of generalized exchange with (D_k=\varnothing). A selected
point on this face is reported as an insertion outcome.
_Avoid_: Append method, append route

**Delete face**:
The boundary of generalized exchange with (A_k=\varnothing). A selected
point on this face is reported as a deletion outcome.
_Avoid_: Prune method, prune route

**True exchange**:
The interior case (D_k\ne\varnothing) and (A_k\ne\varnothing).
_Avoid_: Exchange as a third selector beside append and prune

**Stay point**:
The null patch ((\varnothing,\varnothing)), selected when no admissible
non-null patch passes the commit rules.
_Avoid_: Failed append, no-op prune

**Accuracy debt**:
The checkpoint condition (L_k^2>\tau_{L^2}), where (L_k^2) is the realized
McLachlan distance and (\tau_{L^2}) is its declared cut.
_Avoid_: Nonconvergence, structural miss

**Signed-drift ordering**:
The canonical ordering during accuracy debt: candidates are ranked first by
their signed realized captured-drift change, with bounded resource utility
used only inside the declared numerical tie tolerance.
_Avoid_: Cost-blind method, drift heuristic

**Cost-aware maintenance**:
The ordering used after accuracy debt is paid, when insertion measurements are
not opened and the delete face may relieve resource cost subject to
certification.
_Avoid_: Pruning phase, cleanup algorithm

**Insert-face ablation**:
A controlled restriction that closes the deletion component during the named
comparison. It is not the Paper-II operating method.
_Avoid_: Insertion-only default, append algorithm

## Realization and control

**AP realization adapter**:
The translation from a physical ansatz and generator pool into deletion
coordinates, positioned insertion occurrences, checkpoint geometry, resource
costs, and materialized finalist evaluations consumed by generalized exchange.
_Avoid_: The exchange mathematics

**Preference setting**:
A quantity that changes the mathematical ordering of admissible patches, such
as the debt rule, cost exponents, or history weight.
_Avoid_: Parameter when the class is ambiguous

**Eligibility setting**:
A rule that decides which patch components may be nominated, such as the
minimum surviving support, deletion target, cooldown, occurrence policy, or
drive protection.
_Avoid_: Score threshold

**Measurement-free deletion permission**:
The set-level eligibility rule evaluated before structural scoring. For a
proposed deletion set (D_k), it combines the rotation-angle upper bound on ray
motion with normalized reverse-Schur captured-drift loss, using only the
checkpoint's existing angles, generator coefficients, Gram matrix, and drive
vector. A refused set is never materialized or sent to overlap/refit
certification.
_Avoid_: Ray certification, Schur authorization

**Certification setting**:
A hard tolerance a materialized finalist must satisfy, including ray
continuity, velocity continuity, refit bounds, and any enabled finite-condition
assertion.
_Avoid_: Ranking weight

**Search budget**:
A finite-work bound on the combinatorial approximation, including pool size,
insertion cardinality, scored-patch work, materialized finalist attempts, and
within-checkpoint rounds.
_Avoid_: Method parameter without reporting whether it bound

**Time-step controller**:
The rule that subdivides a reporting interval using state displacement,
parameter displacement, or their composition. It is crossed with the two
algorithmic methods in the Paper-II comparison matrix.
_Avoid_: Integrator, method

**Algorithmic method**:
Either APM generalized exchange or AVQDS adaptive append. The three time-step
controllers form run configurations, not four additional methods.
_Avoid_: Six methods

## Evidence and reporting

**Matched-accuracy comparison**:
A comparison at a declared observable-error target in which each run
configuration may use its own empirically informed (L^2) cut.
_Avoid_: Matched threshold, convergence test

**Accuracy-target escalation**:
Extension of a run configuration's threshold ladder when the evaluated rungs
do not yet attain the declared reporting target.
_Avoid_: Nonconverged cell

**Compiled terminal resource observation**:
Two-qubit gate count, total circuit depth, and two-qubit depth of the identified
terminal ansatz compiled to the declared fake-backend profile.
_Avoid_: Parameter count as hardware cost
