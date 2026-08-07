# Qiskit Transpilation and the Marginal Cost of an ADAPT Generator

<div class="qiskit-two-column-note-marker"></div>

## Page 1 of 2 — From an operator record to a hardware circuit

RA-ADAPT already supplies the phase score

$$
S^{(t)}(r)=\frac{\Delta E_t(r)}{K_t(r)},
\qquad
r=(A,p),
$$

where $r$ pairs a candidate generator $A$ with an admissible insertion
position $p$, $\Delta E_t(r)$ is its phase-$t$ predicted energy decrease,
and $K_t(r)$ is the population-relative Paper-I cost factor. Qiskit changes
the construction of the circuit-cost coordinates entering $K_t(r)$. The
energy model, robust population normalization, parameter cost, and
measurement cost retain their established roles.

Let the accepted ansatz at round $k$ contain an ordered generator list
$\mathcal O_k=(G_1,\ldots,G_{m_k})$:

$$
\begin{aligned}
\lvert\psi_k(\boldsymbol\theta_k)\rangle
&=
U_k(\boldsymbol\theta_k)\lvert\phi_0\rangle,\\
U_k(\boldsymbol\theta_k)
&=
\prod_{j\in\mathcal O_k}U_j(\theta_{k,j}),\\
G_j
&=
\sum_{\mu}c_{j\mu}P_{j\mu},
\qquad
P_{j\mu}\in\{I,X,Y,Z\}^{\otimes Q}.
\end{aligned}
$$

Here $\lvert\phi_0\rangle$ is the encoded reference state, $Q$ is the
logical-qubit count, $P_{j\mu}$ is a Pauli word, and $c_{j\mu}$ is its real
coefficient. At the Qiskit boundary, each coefficient-bearing Pauli term
becomes a Pauli rotation

$$
R_{P_{j\mu}}(\varphi_{j\mu})
=
\exp\!\left(-\frac{i}{2}\varphi_{j\mu}P_{j\mu}\right),
\qquad
\varphi_{j\mu}=2c_{j\mu}\theta_{k,j}.
$$

An undecomposed macro generator therefore enters the circuit as an ordered
block of coefficient-bearing Pauli rotations. A singleton generator enters
as one such rotation. For selection-time cost evaluation, every runtime
parameter is assigned the fixed structural value $1$; the coefficients
$c_{j\mu}$, Pauli supports, generator order, insertion position, and
reference preparation remain present. The compiler consequently evaluates a
reproducible structural circuit whose parameter assignment precedes the
accepted refit.

For a Pauli word with active-qubit set $S_\mu$, the logical synthesis first
rotates $X$ and $Y$ axes into the $Z$ basis, accumulates parity through a
CNOT ladder, performs one $R_z$ rotation, uncomputes the ladder, and restores
the original axes:

$$
\begin{aligned}
X&\mapsto HZH,\\
Y&\mapsto (H S^\dagger)^\dagger Z(H S^\dagger),\\
R_{P_\mu}(\varphi_\mu)
&\mapsto
V_\mu^\dagger
\left[
\prod_{a=1}^{s_\mu-1}\mathrm{CX}_{a,a+1}
\right]
R_z(\varphi_\mu)
\left[
\prod_{a=s_\mu-1}^{1}\mathrm{CX}_{a,a+1}
\right]
V_\mu,
\end{aligned}
$$

where $s_\mu=\lvert S_\mu\rvert$ and $V_\mu$ denotes the required basis
changes. This construction yields the familiar Paper-I local estimate
$2\max(s_\mu-1,0)$ for the entangling-gate count. The graph-span proxy stops
at this structural level:

$$
\widehat C_{2q}^{\,P}(r)
=
\sum_\mu 2\max(s_\mu-1,0),
\qquad
\widehat C_d^{\,P}(r)
=
\sum_\mu 2\,\operatorname{span}_{\Gamma}(S_\mu).
$$

The superscript $P$ denotes the proxy, and $\Gamma$ is the FakeMarrakesh
coupling graph. These quantities depend upon the candidate’s own Pauli
supports. The accepted generator history is absent from both expressions.

Qiskit receives the entire logical gate list together with the physical
coupling graph $\Gamma=(V,E)$. Let $J_2(C)$ index the two-qubit gates in a
logical circuit $C$, and let $q_1(j),q_2(j)$ be the two logical qubits acted
upon by gate $j\in J_2(C)$. The circuit interaction weights are

$$
w_{ab}(C)
=
\sum_{j\in J_2(C)}
\mathbf 1\!\left[
\{q_1(j),q_2(j)\}=\{a,b\}
\right].
$$

Thus $w_{ab}(C)$ counts how often the complete circuit couples logical qubits
$a$ and $b$. A layout is an injective assignment

$$
\pi:\{0,\ldots,Q-1\}\hookrightarrow V
$$

from logical qubits to physical vertices. A useful representation of the
layout pressure is

$$
J_C(\pi)
=
\sum_{0\le a<b<Q}
w_{ab}(C)\,
d_\Gamma\!\left(\pi(a),\pi(b)\right),
$$

where $d_\Gamma(u,v)$ is the shortest-path distance between physical qubits
$u$ and $v$. Qiskit uses a seeded heuristic to seek a low-cost layout under a
richer internal objective; $J_C(\pi)$ exposes why the complete circuit,
rather than one generator, controls the placement problem.

Routing then maintains a time-dependent logical-to-physical assignment
$\pi_\tau$. A two-qubit gate on logical qubits $a,b$ is directly executable at
step $\tau$ when

$$
\bigl(\pi_\tau(a),\pi_\tau(b)\bigr)\in E.
$$

When the pair is separated, routing inserts SWAPs along paths in $\Gamma$. A
SWAP on adjacent physical qubits $u,v$ exchanges their logical occupants,

$$
\pi_{\tau+1}=(u\,v)\circ\pi_\tau,
\qquad
\operatorname{SWAP}_{uv}
=
\operatorname{CX}_{uv}
\operatorname{CX}_{vu}
\operatorname{CX}_{uv}.
$$

After every logical interaction is made executable, native-basis synthesis
and local circuit identities can change the instruction count and depth:

$$
H^2=I,\qquad
SS^\dagger=I,\qquad
\operatorname{CX}^2=I,\qquad
R_z(\alpha)R_z(\beta)=R_z(\alpha+\beta).
$$

These identities display the concrete cancellation and rotation-merging
mechanisms. Parallel gates acting on disjoint qubits can also occupy the same
circuit layer.

The returned circuit is then counted directly. The selector records all
post-transpilation one- and two-qubit operations, two-qubit depth, total depth,
total circuit size, and the final logical-to-physical layout. Barriers, delays,
identity operations, measurement, and reset are excluded from the one- and
two-qubit operation counts. The cost is structural: backend error probabilities
and execution noise are absent from the scalar Paper-I resource coordinates.

<div style="page-break-after: always;"></div>

## Page 2 of 2 — From a transpiled circuit to a routing-conditioned marginal cost

At round $k$, inserting $A$ at position $p$ changes the accepted generator
sequence from

$$
\mathcal O_k
=(G_1,\ldots,G_p,G_{p+1},\ldots,G_{m_k})
$$

to

$$
\mathcal O_{k,r}
=(G_1,\ldots,G_p,A,G_{p+1},\ldots,G_{m_k}).
$$

The corresponding ordered unitaries are

$$
\begin{aligned}
U_k(\boldsymbol\theta)
&=
U_{m_k}(\theta_{m_k})\cdots U_1(\theta_1),\\
U_{k,r}(\alpha,\boldsymbol\theta)
&=
U_{m_k}(\theta_{m_k})\cdots
U_{p+1}(\theta_{p+1})
e^{-i\alpha A}
U_p(\theta_p)\cdots U_1(\theta_1).
\end{aligned}
$$

Selection-time compilation assigns $\alpha=1$ and
$\boldsymbol\theta=\mathbf 1$, expands both ordered unitaries into the
coefficient-bearing Pauli-rotation circuits derived on Page 1, and includes
the same reference preparation $\lvert\phi_0\rangle$. Denote the resulting
logical circuits by $C_k$ and $C_{k,r}$, and their independently laid-out,
routed, native-basis circuits by $C_k^{\rm phys}$ and
$C_{k,r}^{\rm phys}$. Their logical actions are

$$
\begin{aligned}
C_k\lvert 0\rangle^{\otimes Q}
&=
U_k(\mathbf 1)\lvert\phi_0\rangle,
C_{k,r}\lvert 0\rangle^{\otimes Q}
&=
U_{k,r}(1,\mathbf 1)\lvert\phi_0\rangle.
\end{aligned}
$$

The superscript “phys” denotes the corresponding result after physical-qubit
assignment, SWAP routing, and native-basis synthesis.

Thus each candidate-position record is judged through two complete compiled
objects: the accepted base and the candidate-inserted trial. If

$$
\rho(C)
=
\bigl(
N_{2q}(C),D_{2q}(C),D_c(C),N_{1q}(C),L(C)
\bigr)
$$

denotes the compiled resource vector, with $L(C)$ the total instruction
count, then the raw marginal vector is

$$
\delta\rho_k(r)
=
\rho(C_{k,r}^{\rm phys})-\rho(C_k^{\rm phys}).
$$

The Qiskit-derived coordinates passed toward the Paper-I cost normalization
are nonnegative marginal burdens:

$$
\begin{aligned}
\widehat C_{2q,k}^{\,Q}(r)
&=
\max\!\left(
N_{2q}(C_{k,r}^{\rm phys})-N_{2q}(C_k^{\rm phys}),0
\right),\\
\widehat C_{d,k}^{\,Q}(r)
&=
\max\!\left(
D_{2q}(C_{k,r}^{\rm phys})-D_{2q}(C_k^{\rm phys}),0
\right),\\
\widehat C_{1q,k}^{\,Q}(r)
&=
\max\!\left(
N_{1q}(C_{k,r}^{\rm phys})-N_{1q}(C_k^{\rm phys}),0
\right).
\end{aligned}
$$

The superscript $Q$ denotes full Qiskit transpilation. Signed raw deltas are
also preserved in telemetry. A compiler-induced resource reduction therefore
enters the corresponding Paper-I cost coordinate as zero added burden. The
parameter and measurement coordinates retain the Paper-I constructions
$\widehat C_\theta(r)$ and $\widehat C_{\rm shot}(r)$. All five coordinates
then undergo the established phase-population median, scale, bounded signed
normalization, and $K_t(r)$ aggregation. Writing the usually suppressed
round index makes the unchanged Paper-I construction explicit:

$$
\begin{aligned}
m_{x,k}^{(t)}
&=
\operatorname*{median}_{r'\in\mathcal R_k^{(t)}}
\widehat C_{x,k}^{\,Q}(r'),\\
s_{x,k}^{(t)}
&=
\operatorname*{median}_{r'\in\mathcal R_k^{(t)}}
\left|
\widehat C_{x,k}^{\,Q}(r')-m_{x,k}^{(t)}
\right|,\\
\overline C_{x,k}^{(t)}(r)
&=
\frac{2}{\pi}
\arctan\!\left(
\frac{
\widehat C_{x,k}^{\,Q}(r)-m_{x,k}^{(t)}
}{
s_{x,k}^{(t)}
}
\right),\\
K_{t,k}(r)
&=
\left[
1-\frac{1}{2}
\frac{
\sum_{x\in\mathcal X}\lambda_x
\overline C_{x,k}^{(t)}(r)
}{
\sum_{x\in\mathcal X}\lambda_x
}
\right]^{-1}.
\end{aligned}
$$

Here $\mathcal R_k^{(t)}$ is the phase-$t$ candidate-position population at
round $k$, $\mathcal X=\{2q,d,1q,\theta,\mathrm{shot}\}$, and the scale
$s_{x,k}^{(t)}$ receives the prescribed positive floor when necessary.
Accordingly, the Paper-I score and robust relative normalization remain
intact. Full transpilation replaces the three circuit-cost estimators supplied
to that normalization.

The decisive algebraic property is the nonadditivity of transpilation:

$$
\rho\!\left((C_1C_2)^{\rm phys}\right)
\ne
\rho\!\left(C_1^{\rm phys}\right)
+
\rho\!\left(C_2^{\rm phys}\right).
$$

This inequality holds in general. The compiler sees boundaries between
rotations, the global interaction graph,
and the scheduling freedom of the complete circuit. Neighboring basis changes
may cancel; inverse entangling gates may disappear; compatible rotations may
merge; commuting instructions may acquire a more parallel schedule; an
existing physical layout may make a candidate cheap; and a newly inserted
interaction may induce additional routing. These mechanisms can occur inside
a macro block, between a macro and an adjacent generator, or across the
position at which a singleton is inserted.

Consequently, one abstract generator has no unique Qiskit marginal cost. The
notation

$$
\widehat{\mathbf C}_{k}^{\,Q}
(A,p;\Gamma,\mathcal B,\ell,s)
$$

records its dependence on the current accepted ansatz at round $k$, insertion
position $p$, target graph $\Gamma$, native basis $\mathcal B$, optimization
level $\ell$, and seed $s$. The graph-span proxy retains only the candidate
support and target graph.

This loss of history is precisely what makes the proxy inexpensive. Full
transpilation restores compiler context at the price of compiling every
candidate-position trial. Independent trials can be evaluated in parallel
without changing the mathematical score.

The current accepted ansatz is represented by

$$
U_k(\boldsymbol\theta_k)
=
\prod_{j\in\mathcal O_k}U_j(\theta_{k,j}).
$$

Here $U_k(\boldsymbol\theta_k)$ denotes the fixed ordered circuit
parameterization established by the accepted generator sequence. This
qualification matters because two distinct circuit decompositions can realize
the same abstract unitary and still transpile to different resource vectors.
Given this parameterized circuit and $r$, no earlier round enters the cost once
the target graph, native basis, optimization level, and transpiler seed are
fixed.

The dominant contextual dependence comes from layout and routing. Define

$$
\begin{aligned}
w_{ab}^{(k)}
&=w_{ab}(C_k),&
w_{ab}^{(k,r)}
&=w_{ab}(C_{k,r}),\\
J_k(\pi)
&=
\sum_{a<b}w_{ab}^{(k)}
d_\Gamma\!\left(\pi(a),\pi(b)\right),&
J_{k,r}(\pi)
&=
\sum_{a<b}w_{ab}^{(k,r)}
d_\Gamma\!\left(\pi(a),\pi(b)\right).
\end{aligned}
$$

The two seeded heuristic layout searches approximately pursue

$$
\pi_k\approx\operatorname*{arg\,min}_{\pi}J_k(\pi),
\qquad
\pi_{k,r}\approx\operatorname*{arg\,min}_{\pi}J_{k,r}(\pi).
$$

Inserting $A$ at $p$ changes the complete interaction weights, so Qiskit may
choose $\pi_{k,r}\ne\pi_k$. Every pre-existing logical interaction is then
embedded under a potentially different physical assignment, and the SWAP
sequence required to make those interactions adjacent can change globally.
The measured $\delta\rho_k(r)$ consequently includes the new generator, the
changed placement of prior generators, and the changed routing network. The
current full-transpilation route carries no physical layout from
$\pi_{k-1}$ into either search. Thus the scientifically relevant term is
**routing-conditioned marginal compiled cost**.

Two limitations govern interpretation. Base and trial circuits are transpiled
independently. The added candidate can alter Qiskit’s chosen layout or routing
solution, so $\delta\rho_k(r)$ measures the compiler’s global response to the
insertion as well as the gates locally emitted by $A$. The selection oracle
also uses the structural parameter value $1$.
Cancellations requiring the eventual refitted angles remain invisible during
selection. The resulting quantity is therefore a deterministic,
backend-conditioned marginal structural burden under the fixed compiler
contract. It represents the realized output of one reproducible compiler
policy; its heuristic search may exceed the mathematically minimal
implementation cost of the candidate unitary.

Finally, selection-time and reporting-time transpilation answer different
questions. The Qiskit-cost selector uses full trial ansätze on FakeMarrakesh at
optimization level $1$ and seed $7$ to influence $K_t(r)$. The
paper-facing tuple

$$
(N_{2q},D_{2q},D_c,W_{1q},S_{\rm alg})
$$

recompiles the accepted prefix through the common
`table_i_basis_gate_transpile_v1` convention: optimization level $0$, seed
$7$, a fixed basis, and no coupling map, initial layout, or routing
constraint. Every plotted method shares that reporting path. Hence the
selector asks, “Which record is cheap in the present compiled context on the
chosen target?” The endpoint tuple asks, “What resources does the accepted
ansatz exhibit under the common Paper-I reporting convention?”

For a Paper-I rewrite, the minimal mathematical change is therefore

$$
\begin{pmatrix}
\widehat C_{2q,k}^{\,Q}(r)\\
\widehat C_{d,k}^{\,Q}(r)\\
\widehat C_{1q,k}^{\,Q}(r)
\end{pmatrix}
=
\left[
\begin{pmatrix}
N_{2q}(C_{k,r}^{\rm phys})\\
D_{2q}(C_{k,r}^{\rm phys})\\
N_{1q}(C_{k,r}^{\rm phys})
\end{pmatrix}
-
\begin{pmatrix}
N_{2q}(C_k^{\rm phys})\\
D_{2q}(C_k^{\rm phys})\\
N_{1q}(C_k^{\rm phys})
\end{pmatrix}
\right]_+.
$$

The manuscript would retain $\Delta E_t(r)$, the robust population
normalization, $K_{t,k}(r)$, $\widehat C_\theta(r)$, and
$\widehat C_{\rm shot}(r)$. It would additionally specify the target graph,
native basis, optimization level, transpiler seed, structural parameter
assignment, independent base/trial layouts, positive-part clipping, and the
fact that the base and trial layout problems are solved independently.
Because the selected operator sequence can change,
proxy-selected benchmark trajectories remain evidence for the proxy
realization and transpilation-selected trajectories supply evidence for the
compiler-conditioned realization.

### Unsolved work

1. Suppose two records contain the same singleton $A$ at positions $p_1$
   and $p_2$. Identify the compiler mechanisms capable of making
   $\widehat C_{2q}^{\,Q}(A,p_1)\ne\widehat C_{2q}^{\,Q}(A,p_2)$.
2. Construct a circuit pair for which
   $\widehat C_{2q}^{\,P}(r)>0$ and
   $\widehat C_{2q}^{\,Q}(r)=0$, then determine whether the equality arose
   from local cancellation, changed layout, or changed routing.
3. Identify the missing implication between a lower selection-time
   FakeMarrakesh marginal cost and a lower topology-free round-50 reporting
   tuple.

### Implementation anchors

- Structural Pauli rotations:
  [`pipelines/hardcoded/adapt_circuit_execution.py`](../../pipelines/hardcoded/adapt_circuit_execution.py#L88-L175)
- Candidate-local graph-span proxy:
  [`pipelines/static_adapt/hh_backend_compile_oracle.py`](../../pipelines/static_adapt/hh_backend_compile_oracle.py#L397-L468)
- Full base/trial transpilation and marginal deltas:
  [`pipelines/static_adapt/hh_backend_compile_oracle.py`](../../pipelines/static_adapt/hh_backend_compile_oracle.py#L738-L894)
- Qiskit target invocation and compiled-resource extraction:
  [`pipelines/qiskit_backend_tools.py`](../../pipelines/qiskit_backend_tools.py#L247-L320)
- Qiskit-cost route contract:
  [`pipelines/static_adapt/ra_adapt/engine.py`](../../pipelines/static_adapt/ra_adapt/engine.py#L490-L503)
