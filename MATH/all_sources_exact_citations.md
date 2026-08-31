---
title: "Repository literature sources (exact `\\bibitem` entries)"
author: "auto-generated citation compendium"
date: "2026-05-11"
lang: en-US
geometry: "margin=1in"
fontsize: 10pt
documentclass: article
header-includes:
  - "\\usepackage{amsmath,amssymb}"
  - "\\usepackage{fontspec}"
  - "\\setmainfont{Times New Roman}"
  - "\\setsansfont{Times New Roman}"
  - "\\setmonofont{Courier New}"
  - "\\usepackage{unicode-math}"
  - "\\setmathfont{STIX Two Math}"
  - "\\usepackage{fancyvrb}"
  - "\\usepackage{underscore}"
  - "\\usepackage{listings}"
  - "\\lstset{basicstyle=\\ttfamily\\small,breaklines=true,columns=fullflexible,keepspaces=true}"
  - "\\setlength{\\parindent}{0pt}"
  - "\\setlength{\\parskip}{0.6em}"
---

# Repository literature sources (exact `\\bibitem` entries)

Generated: 2026-05-11 09:47:46 (local)
Total unique citation keys (deduplicated): 94


## Part 1 (Static)
These entries define the static-adapt part of the program: model and classical-baseline context, ADAPT-VQE lineages, and measurement-aware cost modeling for ground-state ansatz construction.
Field-level baseline for compact variational circuit design, plus support for static-paper context and downstream dynamics seed construction.

### Dao2025SU2MatrixVQE
**[Dao2025SU2MatrixVQE]**  
H.~L. Dao,  
``Exploring new variational quantum circuit ansatzes for solving SU(2) matrix models,''  
\emph{European Physical Journal C} \textbf{85}, 705 (2025).  
DOI: \href{https://doi.org/10.1140/epjc/s10052-025-14322-7}{10.1140/epjc/s10052-025-14322-7}.  

- **contribution:** Demonstrates fixed-structure and tailored VQE ansatzes for bosonic and supersymmetric SU(2) matrix-model Hamiltonians, including EfficientSU2, TwoLocal, and EvolvedOperatorAnsatz circuits.
- **relation:** Boundary source showing that fixed-structure VQE has already been applied to bosonic and mixed/supersymmetric boson-fermion models; Paper I novelty claims should therefore remain ADAPT-VQE-specific.
- **abstract:**
  - > "new forms of parameterized quantum circuits to be used as variational ansatzes"
  - > "solving the bosonic and supersymmetric SU(2) matrix models"

### Dutta2026QumodeVQE
**[Dutta2026QumodeVQE]**  
R. Dutta, C. Cianci, A.~V. Soudackov, Y. Wang, C. Xu, D.~A. Mazziotti, L.~F. Santos, and V.~S. Batista,  
``Qumode-Based Variational Quantum Eigensolver for Molecular Excited States,''  
\emph{Journal of Chemical Theory and Computation} \textbf{22}, 993--1003 (2026).  
DOI: \href{https://doi.org/10.1021/acs.jctc.5c01494}{10.1021/acs.jctc.5c01494}; arXiv:2509.04727.  

- **contribution:** Introduces qumode-subspace VQE using bosonic qumodes and demonstrates variational calculations involving bosonic degrees of freedom.
- **relation:** Boundary source for bosonic-mode VQE literature; useful for avoiding overly broad claims that fixed-structure VQE has not been applied beyond fermionic Hamiltonians.
- **abstract:**
  - > "using the Fock basis of bosonic qumodes"
  - > "electronic structure Hamiltonian is first mapped to a qubit representation"

### LiBenjamin2017VQS
**[LiBenjamin2017VQS]**  
Y. Li and S.~C. Benjamin,  
``Efficient variational quantum simulator incorporating active error minimization,''  
\emph{Physical Review X} \textbf{7}, 021050 (2017).  
DOI: \href{https://doi.org/10.1103/PhysRevX.7.021050}{10.1103/PhysRevX.7.021050}; arXiv:1611.09301.  

- **contribution:** Introduces a variational simulator for quantum-system dynamics on near-term processors with explicit attention to error accumulation.
- **relation:** Supports the Paper I introduction claim that ground-state ansatz preparation can seed downstream variational real-time dynamics.
- **abstract:**
  - > "One of the key applications for quantum computers will be the simulation of other quantum systems"
  - > "we propose a variational method involving closely integrated classical and quantum coprocessors"

### Yuan2019VQS
**[Yuan2019VQS]**  
X. Yuan, S. Endo, Q. Zhao, Y. Li, and S.~C. Benjamin,  
``Theory of variational quantum simulation,''  
\emph{Quantum} \textbf{3}, 191 (2019).  
DOI: \href{https://doi.org/10.22331/q-2019-10-07-191}{10.22331/q-2019-10-07-191}; arXiv:1812.08767.  

- **contribution:** Formalizes variational principles for static, real-time, and imaginary-time quantum simulation on near-term hardware.
- **relation:** Supports the Paper I introduction claim that ground-state preparation interfaces with variational real-time dynamics.
- **abstract:**
  - > "efficiently solving static many-body problems and simulating real and imaginary time dynamics"
  - > "applicable to near-term quantum hardware"

### McClean2017QSE
**[McClean2017QSE]**  
J.~R. McClean, M.~E. Kimchi-Schwartz, J. Carter, and W.~A. de Jong,  
``Hybrid quantum-classical hierarchy for mitigation of decoherence and determination of excited states,''  
\emph{Physical Review A} \textbf{95}, 042308 (2017).  
DOI: \href{https://doi.org/10.1103/PhysRevA.95.042308}{10.1103/PhysRevA.95.042308}; arXiv:1603.05681.  

- **contribution:** Introduces the quantum subspace expansion hierarchy for excited-state determination and decoherence mitigation.
- **relation:** Supports the Paper I introduction claim that ground-state ansatz preparation can seed subspace-based excited-state spectra.
- **abstract:**
  - > "allows for the accurate determination of excited electronic states"
  - > "without using any additional quantum coherence time or formal error correction codes"

### Hubbard1963
**[Hubbard1963]**  
J. Hubbard,  
``Electron correlations in narrow energy bands,''  
\emph{Proceedings of the Royal Society A} \textbf{276}, 238--257 (1963).  

- **contribution:** Hubbard1963 establishes the Hubbard model as the canonical lattice Hamiltonian for strong electron correlation, grounding the model setup
- **relation:** supports the static-adapt Hamiltonian construction and connects directly to Holstein benchmarks used across both papers
- **abstract:**
  - > "It is pointed out that one of the main effects of correlation phenomena in d- and f-bands is to give rise to behaviour characteristic of the atomic or Heitler-London model."
  - > "To investigate this situation a simple, approximate model for the interaction of electrons in narrow energy bands is introduced."
  - > "The results of applying the Hartree-Fock approximation to this model are examined."
  - > "To clarify the physical meaning of the solution a two-electron example is examined."

### Holstein1959a
**[Holstein1959a]**  
T. Holstein,  
``Studies of polaron motion: Part I. The molecular-crystal model,''  
\emph{Annals of Physics} \textbf{8}, 325 (1959).  

- **contribution:** Studies of polaron motion Part I introduces the Holstein electron-phonon model and small-coupling transport regime, motivating static-adapt Hamiltonian terms and polaronic observables
- **relation:** supports the physical model assumptions in the static paper’s Holstein baseline
- **abstract:**
  - > "In this paper is described a model for polaron motion which incorporates, in simplified form, the principal physical features of the problem."
  - > "The (crystalline) medium, within which a single excess electron (or hole) is contained, is taken to be a one-dimensional molecular crystal, consisting of diatomic molecular sites; each site possesses a single vibrational degree of freedom, represented by the deviation x_n of its internuclear separation from equilibrium."
  - > "The motion of the electron in this medium is treated by a tight-binding approach, in which the wave function is represented as a superposition of local molecular functions."
  - > "In line with the x_n dependence of the delta s, it is also assumed that the local electronic energy depends linearly on x_n."

### Holstein1959b
**[Holstein1959b]**  
T. Holstein,  
``Studies of polaron motion: Part II. The small polaron,''  
\emph{Annals of Physics} \textbf{8}, 343 (1959).  

- **contribution:** Holstein Part II adds small-polaron transport physics for weak electronic overlap, which motivates how the static adaptive basis handles localized transport regimes
- **relation:** reinforces why the workflow emphasizes physically faithful polaron Hamiltonians
- **abstract:**
  - > "The one-dimensional molecular-crystal model of polaron motion, described in the preceding paper, is here analyzed for the case in which the electronic-overlap term of the total Hamiltonian is a small perturbation."
  - > "In zeroth order, in the absence of this term, the electron is localized at a given site."
  - > "The vibrational state is specified by a set of quantum numbers N_k giving the degree of excitation of each vibration mode."
  - > "At low temperatures diagonal transitions dominate, while at higher temperatures non-diagonal transitions and diffusion dominate."

### Weisse2008
**[Weisse2008]**  
A. Wei{\ss}e and H. Fehske,  
``Exact diagonalization techniques,''  
in \emph{Computational Many-Particle Physics},  
edited by H. Fehske, R. Schneider, and A. Wei{\ss}e,  
Lecture Notes in Physics Vol. 739 (Springer, Berlin, Heidelberg, 2008), pp. 529--544.  

- **contribution:** gives practical exact-diagonalization and Krylov/Lanczos machinery for interacting quantum systems, useful as a classical benchmark for small-system reference
- **relation:** underpins static-adapt validation and Hamiltonian sanity checks before adaptive evolution
- **abstract:**
  - > "In this chapter we show how to calculate a few eigenstates of the full Hamiltonian matrix of an interacting quantum system."
  - > "Naturally this implies that the Hilbert space of the problem has to be truncated, either by considering finite systems or by imposing suitable cut-offs."
  - > "All presented methods are iterative and repeatedly apply the Hamiltonian matrix to vectors in the Hilbert space."
  - > "A key outcome is that Lanczos-based sparse linear algebra enables practical access to low-energy spectra and correlators for interacting models."

### Troyer2005
**[Troyer2005]**  
M. Troyer and U.-J. Wiese,  
``Computational complexity and fundamental limitations to fermionic quantum Monte Carlo simulations,''  
\emph{Physical Review Letters} \textbf{94}, 170201 (2005).  

- **contribution:** motivates why quantum methods are needed by detailing sign-problem hardness for fermionic QMC
- **relation:** justifies adaptive variational strategies as practical alternatives in the static paper
- **abstract:**
  - > "Quantum Monte Carlo simulations, while being efficient for bosons, suffer from the "negative sign problem" when applied to fermions--causing an exponential increase of the computing time with the number of particles."
  - > "A polynomial time solution to the sign problem is highly desired since it would provide an unbiased and numerically exact method to simulate correlated quantum systems."
  - > "Here we show that such a solution is almost certainly unattainable by proving that the sign problem is nondeterministic polynomial (NP) hard, implying that a generic solution of the sign problem would also solve all problems in the complexity class NP in polynomial time."

### Schollwock2011
**[Schollwock2011]**  
U. Schollw\"ock,  
``The density-matrix renormalization group in the age of matrix product states,''  
\emph{Annals of Physics} \textbf{326}, 96 (2011).  

- **contribution:** The review explains DMRG/MPS and benchmark accuracy for 1D strongly correlated systems
- **relation:** supplies classical reference quality and basis intuition that supports the static ansatz compactness goals
- **abstract:**
  - > "The density-matrix renormalization group method (DMRG) has established itself over the last decade as the leading method for the simulation of the statics and dynamics of one-dimensional strongly correlated quantum lattice systems."
  - > "In the further development of the method, the realization that DMRG operates on a highly interesting class of quantum states, so-called matrix product states (MPS), has allowed a much deeper understanding of the inner structure of the DMRG method, its further potential and its limitations."
  - > "In this paper, a detailed exposition of current DMRG thinking in the MPS language is given to make implementation of MPS variants transparent, and several directions for future algorithmic developments are discussed."

### Peruzzo2014
**[Peruzzo2014]**  
A. Peruzzo, J. McClean, P. Shadbolt, M.-H. Yung, X.-Q. Zhou, P.~J. Love, A. Aspuru-Guzik, and J.~L. O'Brien,  
``A variational eigenvalue solver on a photonic quantum processor,''  
\emph{Nature Communications} \textbf{5}, 4213 (2014).  

- **contribution:** Peruzzo2014 demonstrates VQE as practical hybrid quantum optimization on near-term hardware
- **relation:** provides historical context for the static-adapt ansatz and measurement workflow
- **abstract:**
  - > "Quantum computers promise to efficiently solve important problems that are intractable on a conventional computer."
  - > "For quantum systems, where the physical dimension grows exponentially, finding the eigenvalues of certain operators is one such intractable problem and remains a fundamental challenge."
  - > "The quantum phase estimation algorithm efficiently finds the eigenvalue of a given eigenvector but requires fully coherent evolution."
  - > "The proposed approach drastically reduces the coherence time requirements, enhancing the potential of quantum resources available today and in the near future."

### McClean2016
**[McClean2016]**  
J.~R. McClean, J. Romero, R. Babbush, and A. Aspuru-Guzik,  
``The theory of variational hybrid quantum-classical algorithms,''  
\emph{New Journal of Physics} \textbf{18}, 023023 (2016).  

- **contribution:** McClean2016 gives the formal basis for hybrid variational methods and error and sample complexity tradeoffs
- **relation:** helps justify algorithmic framing for the static-adapt optimization stack
- **abstract:**
  - > "Many quantum algorithms have daunting resource requirements when compared to what is available today."
  - > "To address this discrepancy, a quantum-classical hybrid optimization scheme known as 'the quantum variational eigensolver'was developed (Peruzzo et al 2014 Nat."
  - > "Commun."
  - > "Finally, we show how the use of modern derivative free optimization techniques can offer dramatic computational savings of up to three orders of magnitude over previously used optimization techniques."

### Kandala2017
**[Kandala2017]**  
A. Kandala, A. Mezzacapo, K. Temme, M. Takita, M. Brink, J.~M. Chow, and J.~M. Gambetta,  
``Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets,''  
\emph{Nature} \textbf{549}, 242 (2017).  

- **contribution:** hardware-efficient VQE study demonstrates realistic resource constraints and circuit-depth limitations on NISQ devices
- **relation:** supports the static paper’s feasibility framing under hardware-aware ansatz design
- **abstract:**
  - > "Quantum computers can be used to address electronic-structure problems and problems in materials science and condensed matter physics that can be formulated as interacting fermionic problems, problems which stretch the limits of existing high-performance computers."
  - > "Finding exact solutions to such problems numerically has a computational cost that scales exponentially with the size of the system, and Monte Carlo methods are unsuitable owing to the fermionic sign problem."
  - > "These limitations of classical computational methods have made solving even few-atom electronic-structure problems interesting for implementation using medium-sized quantum computers."
  - > "Our results help to elucidate the requirements for scaling the method to larger systems and for bridging the gap between key problems in high-performance computing and their implementation on quantum hardware."

### McClean2018BarrenPlateaus
**[McClean2018BarrenPlateaus]**  
J.~R. McClean, S. Boixo, V.~N. Smelyanskiy, R. Babbush, and H. Neven,  
``Barren plateaus in quantum neural network training landscapes,''  
\emph{Nature Communications} \textbf{9}, 4812 (2018).  

- **contribution:** identifies barren-plateau optimization pathology and explains why compact, structured ansätze are needed
- **relation:** supports the paper’s emphasis on stable parameterized circuits
- **abstract:**
  - > "Many experimental proposals for noisy intermediate scale quantum devices involve training a parameterized quantum circuit with a classical optimization loop."
  - > "Such hybrid quantum-classical algorithms are popular for applications in quantum simulation, optimization, and machine learning."
  - > "Due to its simplicity and hardware efficiency, random circuits are often proposed as initial guesses for exploring the space of quantum states."
  - > "We argue that this is related to the 2-design characteristic of random circuits, and that solutions to this problem must be studied."

### Grimsley2019
**[Grimsley2019]**  
H.~R. Grimsley, S.~E. Economou, E. Barnes, and N.~J. Mayhall,  
``An adaptive variational algorithm for exact molecular simulations on a quantum computer,''  
\emph{Nature Communications} \textbf{10}, 3007 (2019).  

- **contribution:** Grimsley introduces ADAPT-VQE’s operator-growth strategy guided by gradients, directly matching the adaptive ansatz expansion ideas
- **relation:** supports both the static construction and dynamic reuse of adaptive move sets
- **abstract:**
  - > "Quantum simulation of chemical systems is one of the most promising near-term applications of quantum computers."
  - > "The variational quantum eigensolver, a leading algorithm for molecular simulations on quantum hardware, has a serious limitation in that it typically relies on a pre-selected wavefunction ansatz that results in approximate wavefunctions and energies."
  - > "Here we present an arbitrarily accurate variational algorithm that, instead of fixing an ansatz upfront, grows it systematically one operator at a time in a way dictated by the molecule being simulated."
  - > "Our results highlight the potential of our adaptive algorithm for exact simulations with present-day and near-term quantum hardware."

### Tang2021
**[Tang2021]**  
H.~L. Tang, V.~O. Shkolnikov, G.~S. Barron, H.~R. Grimsley, N.~J. Mayhall, E. Barnes, and S.~E. Economou,  
``Qubit-ADAPT-VQE: An adaptive algorithm for constructing hardware-efficient ans\"atze on a quantum processor,''  
\emph{PRX Quantum} \textbf{2}, 020310 (2021).  

- **contribution:** Tang extends ADAPT with qubit-operator pools and hardware-efficient choices, directly informing the static paper’s ansatz generator design
- **relation:** supports operator-sequential growth under measurement-bounded hardware budgets
- **abstract:**
  - > "The resources required to run a high-accuracy variational quantum eigensolver algorithm with a dynamically"
  - > "created ansatz are quantified and reduced significantly, easing the quantum simulation of many-body systems."

### Yordanov2021
**[Yordanov2021]**  
Y.~S. Yordanov, V. Armaos, C.~H.~W. Barnes, and D.~R.~M. Arvidsson-Shukur,  
``Qubit-excitation-based adaptive variational quantum eigensolver,''  
\emph{Communications Physics} \textbf{4}, 228 (2021).  

- **contribution:** Yordanov introduces qubit-excitation-based ADAPT variants for more compact physically interpretable generators
- **relation:** contributes a practical route for the static generator list and low-depth ansatz structure
- **abstract:**
  - > "Molecular simulations with the variational quantum eigensolver (VQE) are a promising application for emerging noisy intermediate-scale quantum computers."
  - > "Constructing accurate molecular ansätze that are easy to optimize and implemented by shallow quantum circuits is crucial for the successful implementation of such simulations."
  - > "Ansätze are, generally, constructed as series of fermionic-excitation evolutions."
  - > "In terms of circuit efficiency and convergence speed, we demonstrate that the QEB-ADAPT-VQE outperforms the qubit-ADAPT-VQE, which to our knowledge was the previous most circuit-efficient scalable VQE protocol for molecular simulations."

### Feniou2023
**[Feniou2023]**  
C. Feniou, M. Hassan, D. Traor\'e, E. Giner, Y. Maday, and J.-P. Piquemal,  
``Overlap-ADAPT-VQE: Practical quantum chemistry on quantum computers via overlap-guided compact ans\"atze,''  
\emph{Communications Physics} \textbf{6}, 192 (2023).  

- **contribution:** Feniou’s overlap-guided growth reduces ansatz bloat by using overlap objectives
- **relation:** contributes a mechanism for keeping static-adapt circuits compact while preserving expressivity
- **abstract:**
  - > "ADAPT-VQE is a robust algorithm for hybrid quantum-classical simulations of quantum chemical systems on near-term quantum computers."
  - > "While its iterative process systematically reaches the ground state energy, practical implementations of ADAPT-VQE are sensitive to local energy minima, leading to over-parameterized ansätze."
  - > "We introduce the Overlap-ADAPT-VQE to grow wave-functions by maximizing their overlap with any intermediate target wave-function that already captures some electronic correlation."
  - > "Since this compression strategy can also be initialized with accurate Selected-Configuration Interaction (SCI) classical target wave-functions, it paves the way for chemically accurate simulations of larger systems, and strengthens the promise of decisively surpassing classical quantum chemistry through the power of quantum computing."

### Anastasiou2024
**[Anastasiou2024]**  
P.~G. Anastasiou, Y. Chen, N.~J. Mayhall, E. Barnes, and S.~E. Economou,  
``TETRIS-ADAPT-VQE: An adaptive algorithm that yields shallower, denser circuit ans\"atze,''  
\emph{Physical Review Research} \textbf{6}, 013254 (2024).  

- **contribution:** Anastasiou presents TETRIS-ADAPT ideas that append disjoint operators in blocks to reduce depth
- **relation:** contributes directly to the static objective of shallower, denser but shorter circuits
- **abstract:**
  - > "Adaptive quantum variational algorithms are particularly promising for simulating strongly correlated systems on near-term quantum hardware, but they are not yet viable due, in large part, to the severe coherence time limitations on current devices."
  - > "In this paper, we introduce an algorithm called TETRIS-ADAPT-VQE (tiling efficient trial circuits with rotations implemented simultaneously adaptive derivative-assembled problem-tailored variational quantum eigensolver), which iteratively builds up variational a few operators at a time in a way dictated by the problem being simulated."
  - > "This algorithm is a modified version of the ADAPT-VQE algorithm, in which the one-operator-at-a-time rule is lifted to allow for the addition of multiple operators with disjoint supports in each iteration."
  - > "Published by the American Physical Society 2024"

### Ramoa2025
**[Ramoa2025]**  
M. Ram\^oa, P.~G. Anastasiou, L.~P. Santos, N.~J. Mayhall, E. Barnes, and S.~E. Economou,  
``Reducing the resources required by ADAPT-VQE using coupled exchange operators and improved subroutines,''  
\emph{npj Quantum Information} \textbf{11}, 86 (2025).  

- **contribution:** Ramoa proposes coupled-exchange and improved subroutines for ADAPT operator selection
- **relation:** supports the static and McLachlan efficiency arguments in both gate-count and measurement usage
- **abstract:**
  - > "Adaptive variational quantum algorithms arguably offer the best prospects for quantum advantage in the Noisy Intermediate-Scale Quantum era."
  - > "Since the inception of the first such algorithm, the Adaptive Derivative-Assembled Problem-Tailored Variational Quantum Eigensolver (ADAPT-VQE), many improvements have appeared in the literature."
  - > "We combine the key improvements along with a novel operator pool—which we term Coupled Exchange Operator (CEO) pool—to assess the cost of running state-of-the-art ADAPT-VQE on hardware in terms of measurement counts and circuit depth."
  - > "We also find that our state-of-the-art CEO-ADAPT-VQE outperforms the Unitary Coupled Cluster Singles and Doubles ansatz, the most widely used static VQE ansatz, in all relevant metrics, and offers a five order of magnitude decrease in measurement costs as compared to other static ansätze with competitive CNOT counts."

### Ramoa2025HessianRecycling
**[Ramoa2025HessianRecycling]**  
M. Ramoa et al.,  
``Reducing measurement costs by recycling the Hessian in adaptive variational quantum algorithms,''  
\emph{Quantum Science and Technology} \textbf{10}, 015031 (2025).  

- **contribution:** introduces a quasi-Newton update that reuses approximate Hessian information across ADAPT-like iterations, reducing optimizer warm-up and measurement overhead in adaptive ansatz construction
- **relation:** supports the static-adapt workflow by motivating curvature reuse as a principled way to lower per-iteration cost while keeping adaptive growth efficient
- **abstract:**
  - > "Adaptive protocols enable the construction of more efficient state preparation circuits in variational quantum algorithms (VQAs) by utilizing data obtained from the quantum processor during the execution of the algorithm."
  - > "This idea originated with Adaptive Derivative-Assembled Problem-Tailored variational quantum eigensolver (ADAPT-VQE), an algorithm that iteratively grows the state preparation circuit operator by operator, with each new operator accompanied by a new variational parameter, and where all parameters acquired thus far are optimized in each iteration."
  - > "In ADAPT-VQE and other adaptive VQAs that followed it, it has been shown that initializing parameters to their optimal values from the previous iteration speeds up convergence and avoids shallow local traps in the parameter landscape."
  - > "Our protocol decreases the measurement costs in implementing adaptive VQAs on quantum hardware as well as the runtime of their classical simulation."

### Sohail2026GeoADAPT
**[Sohail2026GeoADAPT]**  
M. A. Sohail and T. Koike-Akino,  
``Geo-ADAPT-VQE: Quantum information metric-aware circuit optimization for quantum chemistry,''  
arXiv:2603.10325 (2026).  

- **contribution:** Sohail introduces geometry-aware ADAPT with natural-gradient-style metric information
- **relation:** contributes a conditioning and convergence stabilization principle for adaptive ansatz selection in the static design
- **abstract:**
  - > "Adaptive ansatz construction has emerged as a powerful technique for reducing circuit depth and improving optimization efficiency in variational quantum eigensolvers."
  - > "Existing adaptive methods, including ADAPT-VQE, rely solely on first-order gradients and ignore the underlying geometry of the quantum state space, limiting both convergence behavior and operator-selection efficiency."
  - > "Geo-ADAPT-VQE selects operators from a pool using the natural gradient rule, guiding the ansatz to grow along directions aligned with the underlying quantum-state geometry."
  - > "This improves convergence, reduces susceptibility to shallow local minima and saddle points, and can yield significantly shorter ans"atze."

### Stadelmann2025GradientTroughs
**[Stadelmann2025GradientTroughs]**  
T. Stadelmann and others,  
``Strategies for overcoming gradient troughs in the ADAPT-VQE algorithm,''  
\emph{arXiv:2512.25004} (2025).  

- **contribution:** addresses gradient-trough failures in ADAPT and optimization stabilization
- **relation:** informs robustness heuristics in the stopping/continuation logic for static-adapt growth
- **abstract:**
  - > "Quantum computing on near-term noisy intermediate-scale quantum devices holds significant promise for simulating complex chemical systems."
  - > "Among various variational quantum algorithms, the adaptive derivative-assembled pseudo-Trotter ansatz variational quantum eigensolver (ADAPT-VQE) is widely used for generating molecule-specific adaptive ansätze for different molecules, yet its measurement requirement is extensive, calling for suitable optimizers."
  - > "In this study, we utilize the ADAPT-VQE algorithm enhanced by a powerful optimizer termed sequential optimization with an approximate parabola (SOAP) to calculate molecular energies."
  - > "This work presents a strategy to mitigate the substantial measurement requirements associated with ADAPT-VQE."

### ProvostVallee1980
**[ProvostVallee1980]**  
J. P. Provost and G. Vall\'ee,  
``Riemannian structure on manifolds of quantum states,''  
\emph{Communications in Mathematical Physics} \textbf{76}, 289--301 (1980).  

- **contribution:** classic differential-geometry paper defines a quantum-state metric from Hilbert-space structure, supporting the use of geometric optimization information
- **relation:** provides the theoretical basis for metric-aware updates in the adaptive pipelines
- **abstract:**
  - > "A metric tensor is defined from the underlying Hilbert space structure for any submanifold of quantum states."
  - > "The case where the manifold is generated by the action of a Lie group on a fixed state vector (generalized coherent states manifold hereafter noted G.C.S.M.) is studied in details; the geometrical properties of some well-known G.C.S.M."
  - > "are reviewed and an explicit expression for the scalar Riemannian curvature is given in the general case."
  - > "It is shown on examples that the distance between nearby states is related to quantum fluctuations; in the particular case of the harmonic oscillator group the condition of zero curvature is identical to the condition of non-dispersion of wave packets."

### Stokes2020QNG
**[Stokes2020QNG]**  
J. Stokes, J. Izaac, N. Killoran, and G. Carleo,  
``Quantum natural gradient,''  
\emph{Quantum} \textbf{4}, 269 (2020).  

- **contribution:** Stokes develops quantum natural gradient methods and efficient metric approximations for VQE
- **relation:** supports the static paper’s use of curvature-aware parameter updates and conditioning
- **abstract:**
  - > "A quantum generalization of Natural Gradient Descent is presented as part of a general-purpose optimization framework for variational quantum circuits."
  - > "The optimization dynamics is interpreted as moving in the steepest descent direction with respect to the Quantum Information Geometry, corresponding to the real part of the Quantum Geometric Tensor (QGT), also known as the Fubini-Study metric tensor."
  - > "An efficient algorithm is presented for computing a block-diagonal approximation to the Fubini-Study metric tensor for parametrized quantum circuits, which may be of independent interest."

### Nocedal2006TrustRegion
**[Nocedal2006TrustRegion]**  
J. Nocedal and S. J. Wright,  
``Trust-Region Methods,'' in \emph{Numerical Optimization}, Springer (New York, NY, 1994).  

- **contribution:** Trust-region methods define robust step-control and acceptance criteria for nonconvex optimization, including trust radii and ratio-based updates to stabilize progress when local models are unreliable.
- **relation:** supports the trust-region rationale for Eq. (5), where step control bounds proposed updates and evaluates candidate progress before accepting costly rescales.
- **abstract:**
  - > "No abstract available."

### Ikhtiarudin2025ShotEfficient
**[Ikhtiarudin2025ShotEfficient]**  
A. Ikhtiarudin, G. Sunnardianto, F. Fathurrahman, M. K. Agusta, and H. Dipojono,  
``Shot-Efficient ADAPT-VQE via Reused Pauli Measurements and Variance-Based Shot Allocation,''  
\emph{arXiv:2507.16879} (2025).  

- **contribution:** introduces reusable-measurement and variance-driven budget-allocation strategies for ADAPT-VQE and shows direct shot reductions in optimization and operator-selection phases.
- **relation:** supports equation-6-style confidence and resource-allocation logic by tying shot allocation to measurement variance and gradient-estimate reuse.
- **abstract:**
  - > "The Adaptive Variational Quantum Eigensolver (ADAPT-VQE) is a promising approach for quantum algorithms in the Noisy Intermediate-Scale Quantum (NISQ) era, offering advantages over traditional VQE methods by reducing circuit depth and mitigating challenges in classical optimization."
  - > "we propose two integrated strategies to reduce the shot requirements in ADAPT-VQE."
  - > "First, we reuse Pauli measurement outcomes obtained during VQE parameter optimization in the subsequent operator selection step of the next ADAPT-VQE iteration, which involves operator gradient measurements."
  - > "Second, we apply variance-based shot allocation to both Hamiltonian and operator gradient measurements."
  - > "Our numerical results demonstrate that each method, individually and in combination, significantly reduces the number of shots needed to achieve chemical accuracy while maintaining result fidelity across the studied molecular systems."

### Vaquero2025PrunedADAPT
**[Vaquero2025PrunedADAPT]**  
N. Vaquero-Sabater, A. Carreras, and D. Casanova,  
``Pruned-ADAPT-VQE: Compacting Molecular Ansätze by Removing Irrelevant Operators,''  
\emph{Journal of Chemical Theory and Computation} \textbf{21}, 8720--8728 (2025).  
https://pubs.acs.org/doi/10.1021/acs.jctc.5c00535  

- **contribution:** proposes post-optimization operator pruning by jointly scoring parameter magnitude and ansatz position to remove ineffective generators and reduce redundant ansatz growth.
- **relation:** supports candidate-operator ablation in our solution strategy by offering a concrete mechanism for safely dropping low-impact generators without harming convergence.
- **abstract:**
  - > "The adaptive derivative-assembled problem-tailored variational quantum eigensolver (ADAPT-VQE) is one of the most widely used algorithms for electronic structure calculations in quantum computers."
  - > "However, this flexibility in reoptimization can lead to the inclusion of redundant or inefficient operators that have almost zero parameter value, barely contributing to the ansatz."
  - > "We identify three phenomena responsible for the appearance of these operators: poor operator selection, operator reordering, and fading operators."
  - > "Our approach evaluates each operator after ADAPT-VQE optimization by using a function that considers both its parameter value and position in the ansatz."

### Verteletskyi2020CliqueCover
**[Verteletskyi2020CliqueCover]**  
V. Verteletskyi, T.-C. Yen, and A.~F. Izmaylov,  
``Measurement optimization in the variational quantum eigensolver using a minimum clique cover,''  
\emph{Journal of Chemical Physics} \textbf{152}, 124114 (2020).  

- **contribution:** introduces clique-cover grouping for commuting Pauli measurements to cut measurement overhead
- **relation:** supports the static workflow’s resource budgeting and measurement grouping choices
- **abstract:**
  - > "Solving the electronic structure problem using the Variational Quantum Eigensolver (VQE) technique involves the measurement of the Hamiltonian expectation value."
  - > "The current hardware can perform only projective single-qubit measurements, and thus, the Hamiltonian expectation value is obtained by measuring parts of the Hamiltonian rather than the full Hamiltonian."
  - > "This restriction makes the measurement process inefficient because the number of terms in the Hamiltonian grows as O(N4) with the size of the system, N."
  - > "On average, grouping qubit-wise commuting terms reduced the number of operators to measure three times less compared to the total number of terms in the considered Hamiltonians."

### Yen2020CompatibleMeasurements
**[Yen2020CompatibleMeasurements]**  
T.-C. Yen, V. Verteletskyi, and A.~F. Izmaylov,  
``Measuring all compatible operators in one series of single-qubit measurements using unitary transformations,''  
\emph{Journal of Chemical Theory and Computation} \textbf{16}, 2400 (2020).  

- **contribution:** Yen formalizes measuring compatible operators in a transformed basis via unitary rotations
- **relation:** contributes a concrete reduction strategy for static-adapt measurement cost and shot allocation
- **abstract:**
  - > "The Variational Quantum Eigensolver approach to the electronic structure problem on a quantum computer involves measurement of the Hamiltonian expectation value."
  - > "Formally, quantum mechanics allows one to measure all mutually commuting or compatible operators simultaneously."
  - > "Unfortunately, the current hardware permits measuring only a much more limited subset of operators that share a common tensor product eigen-basis."
  - > "Based on the test set results, the obtained gain scales at least linearly with the number of qubits."

### Macridin2018
**[Macridin2018]**  
A. Macridin, P. Spentzouris, J. Amundson, and R. Harnik,  
``Electron-phonon systems on a universal quantum computer,''  
\emph{Physical Review Letters} \textbf{121}, 110504 (2018).  

- **contribution:** Macridin demonstrates electron-phonon simulation with explicit qubit encodings
- **relation:** validates the static-adapt target models and informs mapping choices
- **abstract:**
  - > "We present an algorithm that extends existing quantum algorithms for simulating fermion systems in quantum chemistry and condensed matter physics to include bosons in general and phonons in particular."
  - > "We introduce a qubit representation for the low-energy subspace of phonons which allows an efficient simulation of the evolution operator of the electron-phonon systems."
  - > "As a consequence of the Nyquist-Shannon sampling theorem, the phonons are represented with exponential accuracy on a discretized Hilbert space with a size that increases linearly with the cutoff of the maximum phonon number."
  - > "The energy and the phonon number distribution of the polaron state agree with exact diagonalization results for weak, intermediate, and strong electron-phonon coupling regimes."

### Sawaya2020
**[Sawaya2020]**  
N.~P.~D. Sawaya, T. Menke, T.~H. Kyaw, S. Johri, A. Aspuru-Guzik, and G.~G. Guerreschi,  
``Resource-efficient digital quantum simulation of $d$-level systems for photonic, vibrational, and spin-$s$ Hamiltonians,''  
\emph{npj Quantum Information} \textbf{6}, 49 (2020).  

- **contribution:** Sawaya provides d-level encoding efficiency ideas for bosonic-like systems and truncation/encoding tradeoffs
- **relation:** supports the static treatment of phonon encodings and circuit-depth planning
- **abstract:**
  - > "Simulation of quantum systems is expected to be one of the most important applications of quantum computing, with much of the theoretical work so far having focused on fermionic and spin- $$\frac{1}{2}$$ 1 2 systems."
  - > "Here, we instead consider encodings of d -level (i.e., qudit) quantum operators into multi-qubit operators, studying resource requirements for approximating operator exponentials by Trotterization."
  - > "We primarily focus on spin- s and truncated bosonic operators in second quantization, observing desirable properties for approaches based on the Gray code, which to our knowledge has not been used in this context previously."
  - > "Results and procedures outlined in this work may be useful for simulating a broad class of Hamiltonians on qubit-based digital quantum computers."

### Li2023
**[Li2023]**  
W. Li, J. Ren, S. Huai, T. Cai, Z. Shuai, and S. Zhang,  
``Efficient quantum simulation of electron-phonon systems by variational basis state encoder,''  
\emph{Physical Review Research} \textbf{5}, 023046 (2023).  

- **contribution:** Li designs variational basis-state encodings for electron-phonon systems with lower qubit and gate scaling in structured regimes
- **relation:** contributes to the static hardware-efficient representation strategy
- **abstract:**
  - > "Digital quantum simulation of electron-phonon systems requires truncating infinite phonon levels into $N$ basis states and then encoding them with qubit computational basis."
  - > "Unary encoding and binary encoding are the two most representative encoding schemes, which demand $O(N)$ and $O(logN)$ qubits as well as $O(N)$ and $O(NlogN)$ quantum gates, respectively."
  - > "In this paper, we propose a variational basis state encoding algorithm that reduces the scaling of the number of qubits and quantum gates to both $O(1)$ for systems obeying the area law of entanglement entropy."
  - > "Our approach paves the way for practical quantum simulation of electron-phonon systems on both near-term hardware and error-corrected quantum computers."

### Denner2023
**[Denner2023]**  
M.~M. Denner, H. Yan, N. A. Sinitsyn, and others,  
``A hybrid quantum-classical method for electron-phonon systems,''  
\emph{Communications Physics} \textbf{6}, 317 (2023).  

- **contribution:** hybrid quantum-classical method provides practical workflow patterns for electron-phonon problems
- **relation:** supports benchmarking and hybrid design cues used in the static-adapt validation path
- **abstract:**
  - > "Interactions between electrons and phonons play a crucial role in quantum materials."
  - > "Yet, there is no universal method that would simultaneously accurately account for strong electron-phonon interactions and electronic correlations."
  - > "By combining methods of the variational quantum eigensolver and the variational non-Gaussian solver, we develop a hybrid quantum-classical algorithm suitable for this type of correlated systems."
  - > "We benchmark our method by applying it to the paradigmatic Hubbard-Holstein model at half filling, and show that it correctly captures the competition between charge density wave and antiferromagnetic phases, quantitatively consistent with exact diagonalization."

## Part 2 (McLachlan)
These entries define the McLachlan and spectral dynamics axis: projected-time variational evolution, adaptive trajectory maintenance, product-formula comparators, and QSE/spectral-subspace conditioning.
Dynamic-evolution methodology and finite-shot/stability lessons support field-level trajectory design and McLachlan checkpoint controller/spectral manifold progression.

### Weisse2008
**[Weisse2008]**  
A. Wei{\ss}e and H. Fehske,  
``Exact diagonalization techniques,''  
in \emph{Computational Many-Particle Physics},  
edited by H. Fehske, R. Schneider, and A. Wei{\ss}e,  
Lecture Notes in Physics Vol. 739 (Springer, Berlin, Heidelberg, 2008), pp. 529--544.  

- **contribution:** Weisse2008 provides classical exact-diagonalization reference protocols for interacting models and helps frame fidelity/benchmarking baselines in trajectory-aware workflows
- **relation:** supports the McLachlan runs by supplying trusted static and dynamic reference checks where ED is tractable
- **abstract:**
  - > "In this chapter we show how to calculate a few eigenstates of the full Hamiltonian matrix of an interacting quantum system."
  - > "Naturally this implies that the Hilbert space of the problem has to be truncated, either by considering finite systems or by imposing suitable cut-offs."
  - > "All presented methods are iterative and repeatedly apply the Hamiltonian matrix to vectors in the Hilbert space."
  - > "A key outcome is that Lanczos-based sparse linear algebra enables practical access to low-energy spectra and correlators for interacting models."

### Troyer2005
**[Troyer2005]**  
M. Troyer and U.-J. Wiese,  
``Computational complexity and fundamental limitations to fermionic quantum Monte Carlo simulations,''  
\emph{Physical Review Letters} \textbf{94}, 170201 (2005).  

- **contribution:** TroyerWiese highlights fermionic QMC complexity barriers, motivating variational dynamics routes
- **relation:** supports the argument that measurement-based McLachlan evolution is practical where sign-problem methods are hard
- **abstract:**
  - > "Quantum Monte Carlo simulations, while being efficient for bosons, suffer from the "negative sign problem" when applied to fermions--causing an exponential increase of the computing time with the number of particles."
  - > "A polynomial time solution to the sign problem is highly desired since it would provide an unbiased and numerically exact method to simulate correlated quantum systems."
  - > "Here we show that such a solution is almost certainly unattainable by proving that the sign problem is nondeterministic polynomial (NP) hard, implying that a generic solution of the sign problem would also solve all problems in the complexity class NP in polynomial time."

### Schollwock2011
**[Schollwock2011]**  
U. Schollw\"ock,  
``The density-matrix renormalization group in the age of matrix product states,''  
\emph{Annals of Physics} \textbf{326}, 96 (2011).  

- **contribution:** review contextualizes high-accuracy tensor-network baselines for dynamics and spectral tasks
- **relation:** provides a quality reference when validating McLachlan-evolved states and excited-state manifolds
- **abstract:**
  - > "The density-matrix renormalization group method (DMRG) has established itself over the last decade as the leading method for the simulation of the statics and dynamics of one-dimensional strongly correlated quantum lattice systems."
  - > "In the further development of the method, the realization that DMRG operates on a highly interesting class of quantum states, so-called matrix product states (MPS), has allowed a much deeper understanding of the inner structure of the DMRG method, its further potential and its limitations."
  - > "In this paper, a detailed exposition of current DMRG thinking in the MPS language is given to make implementation of MPS variants transparent, and several directions for future algorithmic developments are discussed."

### Grimsley2019
**[Grimsley2019]**  
H.~R. Grimsley, S.~E. Economou, E. Barnes, and N.~J. Mayhall,  
``An adaptive variational algorithm for exact molecular simulations on a quantum computer,''  
\emph{Nature Communications} \textbf{10}, 3007 (2019).  

- **contribution:** The ADAPT mechanism for adaptive operator growth supports dynamic enrichment/maintenance of ansätze in McLachlan trajectories
- **relation:** complements the dynamic checkpoint controller by offering gradient-informed growth logic
- **abstract:**
  - > "Quantum simulation of chemical systems is one of the most promising near-term applications of quantum computers."
  - > "The variational quantum eigensolver, a leading algorithm for molecular simulations on quantum hardware, has a serious limitation in that it typically relies on a pre-selected wavefunction ansatz that results in approximate wavefunctions and energies."
  - > "Here we present an arbitrarily accurate variational algorithm that, instead of fixing an ansatz upfront, grows it systematically one operator at a time in a way dictated by the molecule being simulated."
  - > "Our results highlight the potential of our adaptive algorithm for exact simulations with present-day and near-term quantum hardware."

### Tang2021
**[Tang2021]**  
H.~L. Tang, V.~O. Shkolnikov, G.~S. Barron, H.~R. Grimsley, N.~J. Mayhall, E. Barnes, and S.~E. Economou,  
``Qubit-ADAPT-VQE: An adaptive algorithm for constructing hardware-efficient ans\"atze on a quantum processor,''  
\emph{PRX Quantum} \textbf{2}, 020310 (2021).  

- **contribution:** Qubit-ADAPT supports low-overhead adaptive generator selection usable in real-time trajectory control
- **relation:** informs the checkpoint-controller move-set design
- **abstract:**
  - > "The resources required to run a high-accuracy variational quantum eigensolver algorithm with a dynamically"
  - > "created ansatz are quantified and reduced significantly, easing the quantum simulation of many-body systems."

### Yordanov2021
**[Yordanov2021]**  
Y.~S. Yordanov, V. Armaos, C.~H.~W. Barnes, and D.~R.~M. Arvidsson-Shukur,  
``Qubit-excitation-based adaptive variational quantum eigensolver,''  
\emph{Communications Physics} \textbf{4}, 228 (2021).  

- **contribution:** Qubit-excitation ADAPT provides structured generators with favorable implementation properties
- **relation:** helps constrain McLachlan dynamics to physically interpretable, hardware-convenient updates
- **abstract:**
  - > "Molecular simulations with the variational quantum eigensolver (VQE) are a promising application for emerging noisy intermediate-scale quantum computers."
  - > "Constructing accurate molecular ansätze that are easy to optimize and implemented by shallow quantum circuits is crucial for the successful implementation of such simulations."
  - > "Ansätze are, generally, constructed as series of fermionic-excitation evolutions."
  - > "In terms of circuit efficiency and convergence speed, we demonstrate that the QEB-ADAPT-VQE outperforms the qubit-ADAPT-VQE, which to our knowledge was the previous most circuit-efficient scalable VQE protocol for molecular simulations."

### Feniou2023
**[Feniou2023]**  
C. Feniou, M. Hassan, D. Traor\'e, E. Giner, Y. Maday, and J.-P. Piquemal,  
``Overlap-ADAPT-VQE: Practical quantum chemistry on quantum computers via overlap-guided compact ans\"atze,''  
\emph{Communications Physics} \textbf{6}, 192 (2023).  

- **contribution:** Feniou’s overlap-guided criterion supports compact adaptive updates over long trajectories
- **relation:** motivates compactness controls in McLachlan state-maintenance loops
- **abstract:**
  - > "ADAPT-VQE is a robust algorithm for hybrid quantum-classical simulations of quantum chemical systems on near-term quantum computers."
  - > "While its iterative process systematically reaches the ground state energy, practical implementations of ADAPT-VQE are sensitive to local energy minima, leading to over-parameterized ansätze."
  - > "We introduce the Overlap-ADAPT-VQE to grow wave-functions by maximizing their overlap with any intermediate target wave-function that already captures some electronic correlation."
  - > "Since this compression strategy can also be initialized with accurate Selected-Configuration Interaction (SCI) classical target wave-functions, it paves the way for chemically accurate simulations of larger systems, and strengthens the promise of decisively surpassing classical quantum chemistry through the power of quantum computing."

### Anastasiou2024
**[Anastasiou2024]**  
P.~G. Anastasiou, Y. Chen, N.~J. Mayhall, E. Barnes, and S.~E. Economou,  
``TETRIS-ADAPT-VQE: An adaptive algorithm that yields shallower, denser circuit ans\"atze,''  
\emph{Physical Review Research} \textbf{6}, 013254 (2024).  

- **contribution:** TETRIS-ADAPT ideas improve expressivity-per-depth, aligning with the compressed-measurement McLachlan evolution goals
- **relation:** supports long-horizon dynamics with reduced circuit cost
- **abstract:**
  - > "Adaptive quantum variational algorithms are particularly promising for simulating strongly correlated systems on near-term quantum hardware, but they are not yet viable due, in large part, to the severe coherence time limitations on current devices."
  - > "In this paper, we introduce an algorithm called TETRIS-ADAPT-VQE (tiling efficient trial circuits with rotations implemented simultaneously adaptive derivative-assembled problem-tailored variational quantum eigensolver), which iteratively builds up variational a few operators at a time in a way dictated by the problem being simulated."
  - > "This algorithm is a modified version of the ADAPT-VQE algorithm, in which the one-operator-at-a-time rule is lifted to allow for the addition of multiple operators with disjoint supports in each iteration."
  - > "Published by the American Physical Society 2024"

### Ramoa2025
**[Ramoa2025]**  
M. Ram\^oa, P.~G. Anastasiou, L.~P. Santos, N.~J. Mayhall, E. Barnes, and S.~E. Economou,  
``Reducing the resources required by ADAPT-VQE using coupled exchange operators and improved subroutines,''  
\emph{npj Quantum Information} \textbf{11}, 86 (2025).  

- **contribution:** Ramoa’s improved subroutines and coupled operator pools support efficient repeated adaptive steps in controller loops
- **relation:** strengthens the McLachlan runtime and resource arguments
- **abstract:**
  - > "Adaptive variational quantum algorithms arguably offer the best prospects for quantum advantage in the Noisy Intermediate-Scale Quantum era."
  - > "Since the inception of the first such algorithm, the Adaptive Derivative-Assembled Problem-Tailored Variational Quantum Eigensolver (ADAPT-VQE), many improvements have appeared in the literature."
  - > "We combine the key improvements along with a novel operator pool—which we term Coupled Exchange Operator (CEO) pool—to assess the cost of running state-of-the-art ADAPT-VQE on hardware in terms of measurement counts and circuit depth."
  - > "We also find that our state-of-the-art CEO-ADAPT-VQE outperforms the Unitary Coupled Cluster Singles and Doubles ansatz, the most widely used static VQE ansatz, in all relevant metrics, and offers a five order of magnitude decrease in measurement costs as compared to other static ansätze with competitive CNOT counts."

### McLachlan1964
**[McLachlan1964]**  
A. D. McLachlan,  
``A variational solution of the time-dependent Schr\"odinger equation,''  
\emph{Molecular Physics} \textbf{8}, 39--44 (1964).  

- **contribution:** the foundational variational principle underlying the McLachlan equations used in the dynamic controller
- **relation:** directly defines the governing projector and residual-minimization logic for trajectory updates
- **abstract:**
  - > "A variational method is described for finding approximate solutions of the time-dependent Schrödinger equation."
  - > "It is closely related to Frenkel's variational approach but better in some respects because it always leads to a true minimum of the error."
  - > "For several interacting systems it leads to the time-dependent Hartree equations; for a system in an oscillating electric or magnetic field it gives a variational principle for calculating susceptibility at any frequency."

### LiBenjamin2017VQS
**[LiBenjamin2017VQS]**  
Y. Li and S.~C. Benjamin,  
``Efficient variational quantum simulator incorporating active error minimization,''  
\emph{Physical Review X} \textbf{7}, 021050 (2017).  

- **contribution:** introduces active error minimization within variational simulation, supporting robustness in the finite-shot evolution control stack
- **relation:** informs tolerance and mitigation behavior in McLachlan experiments
- **abstract:**
  - > "Quantum computers will need to be tolerant to errors introduced by noise, but current proposals estimate that the number of qubits required for error correction will be many orders of magnitude larger than the number needed for useful computation."
  - > "A new proposal uses a classical-quantum hybrid scheme to implement simple error-tolerant quantum processors with relatively few resources."

### Yuan2019VQS
**[Yuan2019VQS]**  
X. Yuan, S. Endo, Q. Zhao, Y. Li, and S. C. Benjamin,  
``Theory of variational quantum simulation,''  
\emph{Quantum} \textbf{3}, 191 (2019).  

- **contribution:** Yuan gives the general theory of variational quantum simulation, undergirding the state evolution derivation
- **relation:** supports correctness framing for McLachlan-style equations
- **abstract:**
  - > "The variational method is a versatile tool for classical simulation of a variety of quantum systems."
  - > "Great efforts have recently been devoted to its extension to quantum computing for efficiently solving static many-body problems and simulating real and imaginary time dynamics."
  - > "In this work, we first review the conventional variational principles, including the Rayleigh-Ritz method for solving static problems, and the Dirac and Frenkel variational principle, the McLachlan's variational principle, and the time-dependent variational principle, for simulating real time dynamics."
  - > "Our work completes the theory of variational quantum simulation of general real and imaginary time evolution and it is applicable to near-term quantum hardware."

### Endo2020GeneralProcesses
**[Endo2020GeneralProcesses]**  
S. Endo, J. Sun, Y. Li, S.~C. Benjamin, and X. Yuan,  
``Variational quantum simulation of general processes,''  
\emph{Physical Review Letters} \textbf{125}, 010501 (2020).  

- **contribution:** Endo extends VQS to general-process and open-system/generalized evolution cases, useful for robust controller interpretation
- **relation:** supports the generality of the decision-state dynamics logic
- **abstract:**
  - > "Variational quantum algorithms have been proposed to solve static and dynamic problems of closed many-body quantum systems."
  - > "Here we investigate variational quantum simulation of three general types of tasks-generalized time evolution with a non-Hermitian Hamiltonian, linear algebra problems, and open quantum system dynamics."
  - > "The algorithm for generalized time evolution provides a unified framework for variational quantum simulation."
  - > "We numerically test the algorithm with a 6-qubit 2D transverse field Ising model under dissipation."

### Benedetti2021TimeEvolution
**[Benedetti2021TimeEvolution]**  
M. Benedetti, M. Fiorentini, and M. Lubasch,  
``Hardware-efficient variational quantum algorithms for time evolution,''  
\emph{Physical Review Research} \textbf{3}, 033083 (2021).  

- **contribution:** Benedetti proposes hardware-efficient variational time-evolution methods, directly comparable to the McLachlan-projected strategies
- **relation:** provides alternative baselines and benchmarking structure
- **abstract:**
  - > "Parameterized quantum circuits are a promising technology for achieving a quantum advantage."
  - > "An important application is the variational simulation of time evolution of quantum systems."
  - > "To make the most of quantum hardware, variational algorithms need to be as hardware-efficient as possible."
  - > "We numerically analyze the performance of our algorithms using quantum Hamiltonians with local interactions."

### Barison2021
**[Barison2021]**  
S. Barison, F. Vicentini, and G. Carleo,  
``An efficient quantum algorithm for the time evolution of parameterized circuits,''  
\emph{Quantum} \textbf{5}, 512 (2021).  

- **contribution:** Barison gives a parameterized-circuit time-evolution formulation suitable for practical fast updates
- **relation:** supports McLachlan controller structure for short-horizon propagation routines
- **abstract:**
  - > "We introduce a novel hybrid algorithm to simulate the real-time evolution of quantum systems using parameterized quantum circuits."
  - > "The method, named "projected – Variational Quantum Dynamics" (p-VQD) realizes an iterative, global projection of the exact time evolution onto the parameterized manifold."
  - > "In the small time-step limit, this is equivalent to the McLachlan's variational principle."
  - > "Through numerical experiments, we also show that our approach is particularly advantageous over existing global optimization algorithms based on the time-dependent variational principle that, due to a demanding quadratic scaling with parameter numbers, are unsuitable for large parameterized quantum circuits."

### Yao2021
**[Yao2021]**  
Y.-X. Yao, N. Gomes, F. Zhang, C.-Z. Wang, K.-M. Ho, T. Iadecola, and P. P. Orth,  
``Adaptive variational quantum dynamics simulations,''  
\emph{PRX Quantum} \textbf{2}, 030307 (2021).  

- **contribution:** Yao introduces adaptive variational dynamics with McLachlan-distance control, aligning closely with the checkpointing logic
- **relation:** supports adaptive expansion and error-threshold checks in the flow
- **abstract:**
  - > "We propose a general-purpose, self-adaptive approach to construct a variational wave-function ansatz for highly accurate quantum dynamics simulations based on McLachlan's variational principle."
  - > "The key idea is to dynamically expand the variational ansatz along the time-evolution path such that the "McLachlan distance", which is a measure of the simulation accuracy, remains below a set threshold."
  - > "We apply this adaptive variational quantum dynamics simulation (AVQDS) approach to the integrable Lieb-Schultz-Mattis spin chain and the nonintegrable mixed-field Ising model, where it captures both finite-rate and sudden post-quench dynamics with high fidelity."
  - > "We envision that a wide range of dynamical simulations of quantum many-body systems on near-term quantum-computing devices will be made possible through the AVQDS framework."

### Linteau2024
**[Linteau2024]**  
D. Linteau, S. Barison, N.~H. Lindner, and G. Carleo,  
``Adaptive projected variational quantum dynamics,''  
\emph{Physical Review Research} \textbf{6}, 023130 (2024).  

- **contribution:** Linteau develops adaptive projected variational dynamics and provides a direct methodological cousin to the McLachlan projected updates
- **relation:** supports design of projection and truncation policies
- **abstract:**
  - > "We propose an adaptive quantum algorithm to prepare accurate variational time evolved wave functions."
  - > "The method is based on the projected variational quantum dynamics (pVQD) algorithm, that performs a global optimization with linear scaling in the number of variational parameters."
  - > "Instead of fixing a variational ansatz at the beginning of the simulation, the circuit is grown systematically during the time evolution."
  - > "Published by the American Physical Society 2024"

### Zhang2025
**[Zhang2025]**  
F. Zhang, N. Gomes, Y.-X. Yao, P. P. Orth, C.-Z. Wang, K.-M. Ho, and T. Iadecola,  
``Adaptive variational quantum dynamics simulations with compressed circuits,''  
\emph{Physical Review B} \textbf{111}, 094310 (2025).  

- **contribution:** Zhang extends adaptive dynamics with compressed circuits and fewer measurements, directly useful for large-scale McLachlan trajectory budgets
- **relation:** Zhang extends adaptive dynamics with compressed circuits and fewer measurements, directly useful for large-scale McLachlan trajectory budgets
- **abstract:**
  - > "The adaptive variational quantum dynamics simulation (AVQDS) method performs real-time evolution of quantum states using automatically generated parametrized quantum circuits that often contain substantially fewer gates than Trotter circuits."
  - > "Here we report an improved version of the method, which we call AVQDS(T), by porting the tiling efficient trial circuits with rotations implemented simultaneously technique."
  - > "The algorithm adaptively adds layers of disjoint unitary gates to the ansatz circuit so as to keep the McLachlan distance, a measure of the accuracy of the variational dynamics, below a fixed threshold."
  - > "We showcase that this approach enables AVQDS(T) to deliver more accurate results than simulations using a fixed ansatz of comparable final depth for a significant time duration with fewer quantum resources."

### Suzuki1985
**[Suzuki1985]**  
M. Suzuki,  
``Decomposition formulas of exponential operators and Lie exponentials with some applications to quantum mechanics and statistical physics,''  
\emph{Journal of Mathematical Physics} \textbf{26}, 601 (1985).  

- **contribution:** Suzuki1985 formalizes product-formula decomposition principles that benchmark splitting-based simulation baselines
- **relation:** supports the comparison of projector dynamics with reference Trotter approaches
- **abstract:**
  - > "Decomposition formulas of general exponential operators in a Banach algebra and in a Lie algebra are presented that yield a basis of Monte Carlo simulation of quantum systems."
  - > "They are applied to study the relaxation and fluctuation from the initially unstable point and to confirm algebraically the scaling theory of transient phenomena."
  - > "A global approximation method of transient phenomena is also formulated on the basis of decomposition formulas."
  - > "It is applied to the laser model as a simple example."

### Suzuki1990
**[Suzuki1990]**  
M. Suzuki,  
``Fractal decomposition of exponential operators with applications to many-body theories and Monte Carlo simulations,''  
\emph{Physics Letters A} \textbf{146}, 319 (1990).  

- **contribution:** Suzuki1990 provides systematic fractal operator splitting, useful for accuracy/cost comparisons against adaptive variational dynamics
- **relation:** supports the benchmark layer for integrator quality
- **abstract:**
  - > "A new systematic scheme of decomposition of exponential operators is presented, namely exp[x(A+B)] = S_{m}(x)+O(x^{m+1}) for any positive integer m, where S_{m}(x)=e^{t1A}e^{t2B}e^{t3A}e^{t4B}…e^{tMA}."
  - > "A general scheme of construction of {t_j} is given explicitly."
  - > "The decomposition exp[x(A+B)] = [S_m(x/n)]^n + O(x^{m+1}/n^m) yields a new efficient approach to quantum Monte Carlo simulations."

### ChildsSu2019ProductFormula
**[ChildsSu2019ProductFormula]**  
A.~M. Childs and Y. Su,  
``Nearly optimal lattice simulation by product formulas,''  
\emph{Physical Review Letters} \textbf{123}, 050503 (2019).  

- **contribution:** Childs and Su provide near-optimal product-formula scaling bounds for lattice simulations, giving a strong comparator for the method’s step-size and error behavior
- **relation:** Childs and Su provide near-optimal product-formula scaling bounds for lattice simulations, giving a strong comparator for the method’s step-size and error behavior
- **abstract:**
  - > "We consider simulating an n-qubit Hamiltonian with nearest-neighbor interactions evolving for time t on a quantum computer."
  - > "We show that this simulation has gate complexity (nt)^{1+o(1)} using product formulas, a straightforward approach that has been demonstrated by several experimental groups."
  - > "While it is reasonable to expect this complexity-in particular, this was claimed without rigorous justification by Jordan, Lee, and Preskill-we are not aware of a straightforward proof."
  - > "Combined with a previous lower bound, our result implies that product formulas can simulate lattice Hamiltonians with nearly optimal gate complexity."

### ChildsOstranderSu2019RandomizedPF
**[ChildsOstranderSu2019RandomizedPF]**  
A.~M. Childs, A. Ostrander, and Y. Su,  
``Faster quantum simulation by randomization,''  
\emph{Quantum} \textbf{3}, 182 (2019).  

- **contribution:** Randomized product formulas provide probabilistic error/complexity tradeoffs, informing McLachlan alternatives and hybrid benchmark narratives
- **relation:** Randomized product formulas provide probabilistic error/complexity tradeoffs, informing McLachlan alternatives and hybrid benchmark narratives
- **abstract:**
  - > "Product formulas can be used to simulate Hamiltonian dynamics on a quantum computer by approximating the exponential of a sum of operators by a product of exponentials of the individual summands."
  - > "This approach is both straightforward and surprisingly efficient."
  - > "We show that by simply randomizing how the summands are ordered, one can prove stronger bounds on the quality of approximation for product formulas of any given order, and thereby give more efficient simulations."
  - > "Numerical evidence suggests that the randomized approach has better empirical performance as well."

### Campbell2019QDRIFT
**[Campbell2019QDRIFT]**  
E. Campbell,  
``Random compiler for fast Hamiltonian simulation,''  
\emph{Physical Review Letters} \textbf{123}, 070503 (2019).  

- **contribution:** QDRIFT gives stochastic Hamiltonian-simulation baselines with randomized gate sequences
- **relation:** supports the runtime comparisons and robustness arguments against variational approaches
- **abstract:**
  - > "The dynamics of a quantum system can be simulated using a quantum computer by breaking down the unitary into a quantum circuit of one and two qubit gates."
  - > "The most established methods are the Trotter-Suzuki decompositions, for which rigorous bounds on the circuit size depend on the number of terms L in the system Hamiltonian and the size of the largest term in the Hamiltonian Λ."
  - > "Consequently, the Trotter-Suzuki method is only practical for sparse Hamiltonians."
  - > "Performing phase estimation at chemical accuracy, we report that the savings are similar."

### Macridin2018
**[Macridin2018]**  
A. Macridin, P. Spentzouris, J. Amundson, and R. Harnik,  
``Electron-phonon systems on a universal quantum computer,''  
\emph{Physical Review Letters} \textbf{121}, 110504 (2018).  

- **contribution:** Macridin’s electron-phonon quantum algorithm informs physically relevant Hamiltonians and oracle/access patterns for the dynamical tests
- **relation:** Macridin’s electron-phonon quantum algorithm informs physically relevant Hamiltonians and oracle/access patterns for the dynamical tests
- **abstract:**
  - > "We present an algorithm that extends existing quantum algorithms for simulating fermion systems in quantum chemistry and condensed matter physics to include bosons in general and phonons in particular."
  - > "We introduce a qubit representation for the low-energy subspace of phonons which allows an efficient simulation of the evolution operator of the electron-phonon systems."
  - > "As a consequence of the Nyquist-Shannon sampling theorem, the phonons are represented with exponential accuracy on a discretized Hilbert space with a size that increases linearly with the cutoff of the maximum phonon number."
  - > "The energy and the phonon number distribution of the polaron state agree with exact diagonalization results for weak, intermediate, and strong electron-phonon coupling regimes."

### Macridin2018PRA
**[Macridin2018PRA]**  
A. Macridin, P. Spentzouris, J. Amundson, and R. Harnik,  
``Digital quantum computation of fermion-boson interacting systems,''  
\emph{Physical Review A} \textbf{98}, 042312 (2018).  

- **contribution:** The PRA companion details a fermion-boson computation framework and encoding techniques that inform McLachlan model implementation constraints
- **relation:** The PRA companion details a fermion-boson computation framework and encoding techniques that inform McLachlan model implementation constraints
- **abstract:**
  - > "We introduce a method for representing the low-energy subspace of a bosonic field theory on the qubit space of digital quantum computers."
  - > "This discretization leads to an exponentially precise description of the subspace of the continuous theory thanks to the Nyquist-Shannon sampling theorem."
  - > "The method makes the implementation of quantum algorithms for purely bosonic systems as well as fermion-boson interacting systems feasible."
  - > "The polaron quantum simulations are in excellent agreement with the results obtained by exact diagonalization."

### Sawaya2020
**[Sawaya2020]**  
N.~P.~D. Sawaya, T. Menke, T.~H. Kyaw, S. Johri, A. Aspuru-Guzik, and G.~G. Guerreschi,  
``Resource-efficient digital quantum simulation of $d$-level systems for photonic, vibrational, and spin-$s$ Hamiltonians,''  
\emph{npj Quantum Information} \textbf{6}, 49 (2020).  

- **contribution:** Sawaya provides efficient d-level and truncation encoding ideas relevant to bosonic dimensions in the dynamical simulations
- **relation:** Sawaya provides efficient d-level and truncation encoding ideas relevant to bosonic dimensions in the dynamical simulations
- **abstract:**
  - > "Simulation of quantum systems is expected to be one of the most important applications of quantum computing, with much of the theoretical work so far having focused on fermionic and spin- $$\frac{1}{2}$$ 1 2 systems."
  - > "Here, we instead consider encodings of d -level (i.e., qudit) quantum operators into multi-qubit operators, studying resource requirements for approximating operator exponentials by Trotterization."
  - > "We primarily focus on spin- s and truncated bosonic operators in second quantization, observing desirable properties for approaches based on the Gray code, which to our knowledge has not been used in this context previously."
  - > "Results and procedures outlined in this work may be useful for simulating a broad class of Hamiltonians on qubit-based digital quantum computers."

### Li2023
**[Li2023]**  
W. Li, J. Ren, S. Huai, T. Cai, Z. Shuai, and S. Zhang,  
``Efficient quantum simulation of electron-phonon systems by variational basis state encoder,''  
\emph{Physical Review Research} \textbf{5}, 023046 (2023).  

- **contribution:** Li introduces variational basis-state encoding for electron-phonon systems, directly relevant to compact McLachlan trajectory state representation
- **relation:** Li introduces variational basis-state encoding for electron-phonon systems, directly relevant to compact McLachlan trajectory state representation
- **abstract:**
  - > "Digital quantum simulation of electron-phonon systems requires truncating infinite phonon levels into $N$ basis states and then encoding them with qubit computational basis."
  - > "Unary encoding and binary encoding are the two most representative encoding schemes, which demand $O(N)$ and $O(logN)$ qubits as well as $O(N)$ and $O(NlogN)$ quantum gates, respectively."
  - > "In this paper, we propose a variational basis state encoding algorithm that reduces the scaling of the number of qubits and quantum gates to both $O(1)$ for systems obeying the area law of entanglement entropy."
  - > "Our approach paves the way for practical quantum simulation of electron-phonon systems on both near-term hardware and error-corrected quantum computers."

### Denner2023
**[Denner2023]**  
M.~M. Denner, H. Yan, N. A. Sinitsyn, and others,  
``A hybrid quantum-classical method for electron-phonon systems,''  
\emph{Communications Physics} \textbf{6}, 317 (2023).  

- **contribution:** Denner’s hybrid electron-phonon workflow offers practical cross-checks and error-aware structure for dynamic simulation experiments
- **relation:** Denner’s hybrid electron-phonon workflow offers practical cross-checks and error-aware structure for dynamic simulation experiments
- **abstract:**
  - > "Interactions between electrons and phonons play a crucial role in quantum materials."
  - > "Yet, there is no universal method that would simultaneously accurately account for strong electron-phonon interactions and electronic correlations."
  - > "By combining methods of the variational quantum eigensolver and the variational non-Gaussian solver, we develop a hybrid quantum-classical algorithm suitable for this type of correlated systems."
  - > "We benchmark our method by applying it to the paradigmatic Hubbard-Holstein model at half filling, and show that it correctly captures the competition between charge density wave and antiferromagnetic phases, quantitatively consistent with exact diagonalization."

### Hubbard1963
**[Hubbard1963]**  
J. Hubbard,  
``Electron correlations in narrow energy bands,''  
\emph{Proceedings of the Royal Society A} \textbf{276}, 238--257 (1963).  

- **contribution:** Hubbard supplies the canonical strongly correlated lattice Hamiltonian for dynamic case studies
- **relation:** Hubbard supplies the canonical strongly correlated lattice Hamiltonian for dynamic case studies
- **abstract:**
  - > "It is pointed out that one of the main effects of correlation phenomena in d- and f-bands is to give rise to behaviour characteristic of the atomic or Heitler-London model."
  - > "To investigate this situation a simple, approximate model for the interaction of electrons in narrow energy bands is introduced."
  - > "The results of applying the Hartree-Fock approximation to this model are examined."
  - > "To clarify the physical meaning of the solution a two-electron example is examined."

### Holstein1959
**[Holstein1959]**  
T. Holstein,  
``Studies of polaron motion: Part I. The molecular-crystal model,''  
\emph{Annals of Physics} \textbf{8}, 325--342 (1959);  
``Studies of polaron motion: Part II. The `small' polaron,''  
\emph{Annals of Physics} \textbf{8}, 343--389 (1959).  

- **contribution:** Holstein1959 combines Part I/II physical polaron dynamics assumptions and transport regimes relevant to the Holstein checkpoint trajectories
- **relation:** Holstein1959 combines Part I/II physical polaron dynamics assumptions and transport regimes relevant to the Holstein checkpoint trajectories
- **abstract:**
  - > "The one-dimensional molecular-crystal model of polaron motion described in Part I incorporates the principal physical features of the problem and yields localized electron states in zeroth order."
  - > "Part II analyzes the same model when the electronic-overlap term is a small perturbation: zeroth order gives site-localized electrons, while non-vanishing overlap produces neighboring-site transitions."
  - > "It distinguishes diagonal and non-diagonal transitions, explains low-temperature Bloch-band behavior and high-temperature diffusion, and identifies the crossover temperature where perturbative broad-band transport breaks down."

### WeisseFehske2008
**[WeisseFehske2008]**  
A. Weisse and H. Fehske,  
``Exact diagonalization techniques,''  
in \emph{Computational Many-Particle Physics}, Lecture Notes in Physics Vol. 739 (Springer, 2008), pp. 529--544.  

- **contribution:** chapter’s exact-diagonalization methods give compact classical ED references and iterative eigensolvers for validation in dynamic benchmarks
- **relation:** chapter’s exact-diagonalization methods give compact classical ED references and iterative eigensolvers for validation in dynamic benchmarks
- **abstract:**
  - > "In this chapter we show how to calculate a few eigenstates of the full Hamiltonian matrix of an interacting quantum system."
  - > "Naturally this implies that the Hilbert space of the problem has to be truncated, either by considering finite systems or by imposing suitable cut-offs."
  - > "All presented methods are iterative and repeatedly apply the Hamiltonian matrix to vectors in the Hilbert space."
  - > "A key outcome is that Lanczos-based sparse linear algebra enables practical access to low-energy spectra and correlators for interacting models."

### TroyerWiese2005
**[TroyerWiese2005]**  
M. Troyer and U.-J. Wiese,  
``Computational complexity and fundamental limitations to fermionic quantum Monte Carlo simulations,''  
\emph{Physical Review Letters} \textbf{94}, 170201 (2005).  

- **contribution:** Troyer’s variant again anchors the QMC complexity rationale for pursuing variational dynamic routes in difficult fermionic settings
- **relation:** Troyer’s variant again anchors the QMC complexity rationale for pursuing variational dynamic routes in difficult fermionic settings
- **abstract:**
  - > "Quantum Monte Carlo simulations, while being efficient for bosons, suffer from the "negative sign problem" when applied to fermions--causing an exponential increase of the computing time with the number of particles."
  - > "A polynomial time solution to the sign problem is highly desired since it would provide an unbiased and numerically exact method to simulate correlated quantum systems."
  - > "Here we show that such a solution is almost certainly unattainable by proving that the sign problem is nondeterministic polynomial (NP) hard, implying that a generic solution of the sign problem would also solve all problems in the complexity class NP in polynomial time."

### Schollwoeck2011
**[Schollwoeck2011]**  
U. Schollw\"ock,  
``The density-matrix renormalization group in the age of matrix product states,''  
\emph{Annals of Physics} \textbf{326}, 96--192 (2011).  

- **contribution:** The spelling variant maps to the same DMRG/MPS benchmark body used as a reference for trajectory and observable comparisons
- **relation:** The spelling variant maps to the same DMRG/MPS benchmark body used as a reference for trajectory and observable comparisons
- **abstract:**
  - > "The density-matrix renormalization group method (DMRG) has established itself over the last decade as the leading method for the simulation of the statics and dynamics of one-dimensional strongly correlated quantum lattice systems."
  - > "In the further development of the method, the realization that DMRG operates on a highly interesting class of quantum states, so-called matrix product states (MPS), has allowed a much deeper understanding of the inner structure of the DMRG method, its further potential and its limitations."
  - > "In this paper, a detailed exposition of current DMRG thinking in the MPS language is given to make implementation of MPS variants transparent, and several directions for future algorithmic developments are discussed."

### Macridin2018ElectronPhonon
**[Macridin2018ElectronPhonon]**  
A. Macridin, P. Spentzouris, J. Amundson, and R. Harnik,  
``Electron-phonon systems on a universal quantum computer,''  
\emph{Physical Review Letters} \textbf{121}, 110504 (2018).  

- **contribution:** Equivalent electron-phonon implementation reference for the McLachlan paper’s dynamic modeling and validation
- **relation:** Equivalent electron-phonon implementation reference for the McLachlan paper’s dynamic modeling and validation
- **abstract:**
  - > "We present an algorithm that extends existing quantum algorithms for simulating fermion systems in quantum chemistry and condensed matter physics to include bosons in general and phonons in particular."
  - > "We introduce a qubit representation for the low-energy subspace of phonons which allows an efficient simulation of the evolution operator of the electron-phonon systems."
  - > "As a consequence of the Nyquist-Shannon sampling theorem, the phonons are represented with exponential accuracy on a discretized Hilbert space with a size that increases linearly with the cutoff of the maximum phonon number."
  - > "The energy and the phonon number distribution of the polaron state agree with exact diagonalization results for weak, intermediate, and strong electron-phonon coupling regimes."

### Sawaya2020DLevel
**[Sawaya2020DLevel]**  
N. P. D. Sawaya et al.,  
``Resource-efficient digital quantum simulation of $d$-level systems for photonic, vibrational, and spin-$s$ Hamiltonians,''  
\emph{npj Quantum Information} \textbf{6}, 49 (2020).  

- **contribution:** Equivalent d-level encoding reference for dynamic trajectories and cost-sensitive bosonic truncation
- **relation:** Equivalent d-level encoding reference for dynamic trajectories and cost-sensitive bosonic truncation
- **abstract:**
  - > "Simulation of quantum systems is expected to be one of the most important applications of quantum computing, with much of the theoretical work so far having focused on fermionic and spin- $$\frac{1}{2}$$ 1 2 systems."
  - > "Here, we instead consider encodings of d -level (i.e., qudit) quantum operators into multi-qubit operators, studying resource requirements for approximating operator exponentials by Trotterization."
  - > "We primarily focus on spin- s and truncated bosonic operators in second quantization, observing desirable properties for approaches based on the Gray code, which to our knowledge has not been used in this context previously."
  - > "Results and procedures outlined in this work may be useful for simulating a broad class of Hamiltonians on qubit-based digital quantum computers."

### Li2023VBSElectronPhonon
**[Li2023VBSElectronPhonon]**  
W. Li et al.,  
``Efficient quantum simulation of electron-phonon systems by variational basis state encoder,''  
\emph{Physical Review Research} \textbf{5}, 023046 (2023).  

- **contribution:** Equivalent variational basis encoding reference for dynamic bosonic state representations
- **relation:** Equivalent variational basis encoding reference for dynamic bosonic state representations
- **abstract:**
  - > "Digital quantum simulation of electron-phonon systems requires truncating infinite phonon levels into $N$ basis states and then encoding them with qubit computational basis."
  - > "Unary encoding and binary encoding are the two most representative encoding schemes, which demand $O(N)$ and $O(logN)$ qubits as well as $O(N)$ and $O(NlogN)$ quantum gates, respectively."
  - > "In this paper, we propose a variational basis state encoding algorithm that reduces the scaling of the number of qubits and quantum gates to both $O(1)$ for systems obeying the area law of entanglement entropy."
  - > "Our approach paves the way for practical quantum simulation of electron-phonon systems on both near-term hardware and error-corrected quantum computers."

### McClean2017QSE
**[McClean2017QSE]**  
J. R. McClean, M. E. Kimchi-Schwartz, J. Carter, and W. A. de Jong,  
``Hybrid quantum-classical hierarchy for mitigation of decoherence and determination of excited states,''  
\emph{Physical Review A} \textbf{95}, 042308 (2017).  

- **contribution:** hybrid hierarchy introduces QSE-style excited-state correction and decoherence mitigation, matching the spectral branch
- **relation:** hybrid hierarchy introduces QSE-style excited-state correction and decoherence mitigation, matching the spectral branch
- **abstract:**
  - > "Using quantum devices supported by classical computational resources is a promising approach to quantum-enabled computation."
  - > "One powerful example of such a hybrid quantum-classical approach optimized for classically intractable eigenvalue problems is the variational quantum eigensolver, built to utilize quantum resources for the solution of eigenvalue problems and optimizations with minimal coherence time requirements by leveraging classical computational resources."
  - > "These algorithms have been placed as leaders among the candidates for the first to achieve supremacy over classical computation."
  - > "We demonstrate numerically on a sample electronic system that this method both allows for the accurate determination of excited electronic states as well as reduces the impact of decoherence, without using any additional quantum coherence time or formal error-correction codes."

### Colless2018QSE
**[Colless2018QSE]**  
J. I. Colless, V. V. Ramasesh, D. Dahlen, M. S. Blok, M. E. Kimchi-Schwartz, J. R. McClean, J. Carter, W. A. de Jong, and I. Siddiqi,  
``Computation of molecular spectra on a quantum processor with an error-resilient algorithm,''  
\emph{Physical Review X} \textbf{8}, 011021 (2018).  

- **contribution:** Colless demonstrates error-resilient spectral estimation workflows on near-term hardware for excited-state extraction
- **relation:** Colless demonstrates error-resilient spectral estimation workflows on near-term hardware for excited-state extraction
- **abstract:**
  - > "Harnessing the full power of nascent quantum processors requires the efficient management of a limited number of quantum bits with finite coherent lifetimes."
  - > "Hybrid algorithms, such as the variational quantum eigensolver (VQE), leverage classical resources to reduce the required number of quantum gates."
  - > "Experimental demonstrations of VQE have resulted in calculation of Hamiltonian ground states, and a new theoretical approach based on a quantum subspace expansion (QSE) has outlined a procedure for determining excited states that are central to dynamical processes."
  - > "Further, we show that this extended protocol can mitigate the effects of incoherent errors, potentially enabling larger-scale quantum simulations without the need for complex error-correction techniques."

### Higgott2019VQD
**[Higgott2019VQD]**  
O. Higgott, D. Wang, and S. Brierley,  
``Variational quantum computation of excited states,''  
\emph{Quantum} \textbf{3}, 156 (2019).  

- **contribution:** VQD provides excited-state variational structure, supporting the McLachlan spectral extraction alternatives
- **relation:** VQD provides excited-state variational structure, supporting the McLachlan spectral extraction alternatives
- **abstract:**
  - > "The calculation of excited state energies of electronic structure Hamiltonians has many important applications, such as the calculation of optical spectra and reaction rates."
  - > "While low-depth quantum algorithms, such as the variational quantum eigenvalue solver (VQE), have been used to determine ground state energies, methods for calculating excited states currently involve the implementation of high-depth controlled-unitaries or a large number of additional samples."
  - > "Here we show how overlap estimation can be used to deflate eigenstates once they are found, enabling the calculation of excited state energies and their degeneracies."
  - > "Our method is robust to control errors, is compatible with error-mitigation strategies and can be implemented on near-term quantum computers."

### Parrish2019MCVQE
**[Parrish2019MCVQE]**  
R. M. Parrish et al.,  
``Quantum computation of electronic transitions using a variational quantum eigensolver,''  
\emph{Physical Review Letters} \textbf{122}, 230401 (2019).  

- **contribution:** Parrish gives transition-targeted excited-state VQE workflows supporting Hamiltonian spectroscopy paths
- **relation:** Parrish gives transition-targeted excited-state VQE workflows supporting Hamiltonian spectroscopy paths
- **abstract:**
  - > "We develop an extension of the variational quantum eigensolver (VQE) algorithm-multistate contracted VQE (MC-VQE)-that allows for the efficient computation of the transition energies between the ground state and several low-lying excited states of a molecule, as well as the oscillator strengths associated with these transitions."
  - > "We numerically simulate MC-VQE by computing the absorption spectrum of an ab initio exciton model of an 18-chromophore light-harvesting complex from purple photosynthetic bacteria."

### Jones2019Spectra
**[Jones2019Spectra]**  
T. Jones, S. Endo, S. McArdle, X. Yuan, and S. C. Benjamin,  
``Variational quantum algorithms for discovering Hamiltonian spectra,''  
\emph{Physical Review A} \textbf{99}, 062304 (2019).  

- **contribution:** Jones contributes a variational route to Hamiltonian spectra, useful for comparison to the McLachlan-informed spectral stack
- **relation:** Jones contributes a variational route to Hamiltonian spectra, useful for comparison to the McLachlan-informed spectral stack
- **abstract:**
  - > "Calculating the energy spectrum of a quantum system is an important task, for example to analyze reaction rates in drug discovery and catalysis."
  - > "There has been significant progress in developing algorithms to calculate the ground state energy of molecules on near-term quantum computers."
  - > "However, calculating excited state energies has attracted comparatively less attention, and it is currently unclear what the optimal method is."
  - > "As our algorithm uses only low depth circuits and variational algorithms, it is suitable for use on near-term quantum hardware."

### Nakanishi2019SSVQE
**[Nakanishi2019SSVQE]**  
K. M. Nakanishi, K. Mitarai, and K. Fujii,  
``Subspace-search variational quantum eigensolver for excited states,''  
\emph{Physical Review Research} \textbf{1}, 033062 (2019).  

- **contribution:** Subspace-search excited-state strategy informs the multi-state management and state-targeting in spectral workflows
- **relation:** Subspace-search excited-state strategy informs the multi-state management and state-targeting in spectral workflows
- **abstract:**
  - > "The authors propose an algorithm, the subspace-search variational quantum eigensolver (SSVQE) that searches a low energy subspace by supplying orthogonal input states to the variational ansatz and relies on the unitarity of transformations to ensure the orthogonality of output states."
  - > "This work extends the applicable domain of the Variational Quantum Eigensolver to excited states and their related properties."

### Ollitrault2020qEOM
**[Ollitrault2020qEOM]**  
P. J. Ollitrault et al.,  
``Quantum equation of motion for computing molecular excitation energies on a noisy quantum processor,''  
\emph{Physical Review Research} \textbf{2}, 043140 (2020).  

- **contribution:** qEOM methods support practical noisy-device excited-state estimation and transition-energy pipelines in the third axis
- **relation:** qEOM methods support practical noisy-device excited-state estimation and transition-energy pipelines in the third axis
- **abstract:**
  - > "The computation of molecular excitation energies is essential for predicting photo-induced reactions of chemical and technological interest."
  - > "While the classical computing resources needed for this task scale poorly, quantum algorithms emerge as promising alternatives."
  - > "In particular, the extension of the variational quantum eigensolver algorithm to the computation of the excitation energies is an attractive option."
  - > "In particular, we demonstrate the efficiency of this approach in the calculation of the excitation energies of the LiH molecule on an IBM Quantum computer."

### AsthanaKumar2023QScEOM
**[AsthanaKumar2023QScEOM]**  
A. Asthana et al.,  
``Quantum self-consistent equation-of-motion method for computing molecular excitation energies, ionization potentials, and electron affinities on a quantum computer,''  
\emph{Chemical Science} \textbf{14}, 2405--2418 (2023).  

- **contribution:** QScEOM provides a unified framework for excitations, ionization, and affinities on noisy hardware, useful for the excited-state section
- **relation:** QScEOM provides a unified framework for excitations, ionization, and affinities on noisy hardware, useful for the excited-state section
- **abstract:**
  - > "We present a new hybrid quantum algorithm to estimate molecular excited and charged s"
  - > "tates on near-term quantum computers following any VQE-based ground state estimation."

### Grimsley2019ADAPT
**[Grimsley2019ADAPT]**  
H. R. Grimsley, S. E. Economou, E. Barnes, and N. J. Mayhall,  
``An adaptive variational algorithm for exact molecular simulations on a quantum computer,''  
\emph{Nature Communications} \textbf{10}, 3007 (2019).  

- **contribution:** ADAPT variant reused for adaptive generator reuse in dynamic spectral workflows, maintaining continuity with the static roots
- **relation:** ADAPT variant reused for adaptive generator reuse in dynamic spectral workflows, maintaining continuity with the static roots
- **abstract:**
  - > "Quantum simulation of chemical systems is one of the most promising near-term applications of quantum computers."
  - > "The variational quantum eigensolver, a leading algorithm for molecular simulations on quantum hardware, has a serious limitation in that it typically relies on a pre-selected wavefunction ansatz that results in approximate wavefunctions and energies."
  - > "Here we present an arbitrarily accurate variational algorithm that, instead of fixing an ansatz upfront, grows it systematically one operator at a time in a way dictated by the molecule being simulated."
  - > "Our results highlight the potential of our adaptive algorithm for exact simulations with present-day and near-term quantum hardware."

### Tang2021QubitADAPT
**[Tang2021QubitADAPT]**  
H. L. Tang et al.,  
``Qubit-ADAPT-VQE: An adaptive algorithm for constructing hardware-efficient ans\"atze on a quantum processor,''  
\emph{PRX Quantum} \textbf{2}, 020310 (2021).  

- **contribution:** Qubit-ADAPT variant for dynamic contexts preserves a stable, hardware-friendly generator style in McLachlan checkpoints
- **relation:** Qubit-ADAPT variant for dynamic contexts preserves a stable, hardware-friendly generator style in McLachlan checkpoints
- **abstract:**
  - > "The resources required to run a high-accuracy variational quantum eigensolver algorithm with a dynamically"
  - > "created ansatz are quantified and reduced significantly, easing the quantum simulation of many-body systems."

### Yordanov2021QEB
**[Yordanov2021QEB]**  
Y. S. Yordanov, V. Armaos, C. H. W. Barnes, and D. R. M. Arvidsson-Shukur,  
``Qubit-excitation-based adaptive variational quantum eigensolver,''  
\emph{Communications Physics} \textbf{4}, 228 (2021).  

- **contribution:** Qubit-excitation ADAPT variant for dynamic updates and low-overhead move sets in time-evolution control
- **relation:** Qubit-excitation ADAPT variant for dynamic updates and low-overhead move sets in time-evolution control
- **abstract:**
  - > "Molecular simulations with the variational quantum eigensolver (VQE) are a promising application for emerging noisy intermediate-scale quantum computers."
  - > "Constructing accurate molecular ansätze that are easy to optimize and implemented by shallow quantum circuits is crucial for the successful implementation of such simulations."
  - > "Ansätze are, generally, constructed as series of fermionic-excitation evolutions."
  - > "In terms of circuit efficiency and convergence speed, we demonstrate that the QEB-ADAPT-VQE outperforms the qubit-ADAPT-VQE, which to our knowledge was the previous most circuit-efficient scalable VQE protocol for molecular simulations."

### Anastasiou2024TETRIS
**[Anastasiou2024TETRIS]**  
P. G. Anastasiou, Y. Chen, N. J. Mayhall, E. Barnes, and S. E. Economou,  
``TETRIS-ADAPT-VQE: An adaptive algorithm that yields shallower, denser circuit ans\"atze,''  
\emph{Physical Review Research} \textbf{6}, 013254 (2024).  

- **contribution:** TETRIS-style grouped operator growth supports dense/fast trajectory candidates in adaptive evolution
- **relation:** TETRIS-style grouped operator growth supports dense/fast trajectory candidates in adaptive evolution
- **abstract:**
  - > "Adaptive quantum variational algorithms are particularly promising for simulating strongly correlated systems on near-term quantum hardware, but they are not yet viable due, in large part, to the severe coherence time limitations on current devices."
  - > "In this paper, we introduce an algorithm called TETRIS-ADAPT-VQE (tiling efficient trial circuits with rotations implemented simultaneously adaptive derivative-assembled problem-tailored variational quantum eigensolver), which iteratively builds up variational a few operators at a time in a way dictated by the problem being simulated."
  - > "This algorithm is a modified version of the ADAPT-VQE algorithm, in which the one-operator-at-a-time rule is lifted to allow for the addition of multiple operators with disjoint supports in each iteration."
  - > "Published by the American Physical Society 2024"

### Ramoa2025CEO
**[Ramoa2025CEO]**  
M. Ram\^oa et al.,  
``Reducing the resources required by ADAPT-VQE using coupled exchange operators and improved subroutines,''  
\emph{npj Quantum Information} \textbf{11}, 86 (2025).  

- **contribution:** Coupled-exchange operator pool concepts support efficient repeated operator selection in dynamic controller steps
- **relation:** Coupled-exchange operator pool concepts support efficient repeated operator selection in dynamic controller steps
- **abstract:**
  - > "Adaptive variational quantum algorithms arguably offer the best prospects for quantum advantage in the Noisy Intermediate-Scale Quantum era."
  - > "Since the inception of the first such algorithm, the Adaptive Derivative-Assembled Problem-Tailored Variational Quantum Eigensolver (ADAPT-VQE), many improvements have appeared in the literature."
  - > "We combine the key improvements along with a novel operator pool—which we term Coupled Exchange Operator (CEO) pool—to assess the cost of running state-of-the-art ADAPT-VQE on hardware in terms of measurement counts and circuit depth."
  - > "We also find that our state-of-the-art CEO-ADAPT-VQE outperforms the Unitary Coupled Cluster Singles and Doubles ansatz, the most widely used static VQE ansatz, in all relevant metrics, and offers a five order of magnitude decrease in measurement costs as compared to other static ansätze with competitive CNOT counts."

### Sohail2026GeoADAPT
**[Sohail2026GeoADAPT]**  
M. A. Sohail and T. Koike-Akino,  
``Geo-ADAPT-VQE: Quantum information metric-aware circuit optimization for quantum chemistry,''  
arXiv:2603.10325 (2026).  

- **contribution:** Geo-ADAPT adds geometry-aware selection and conditioning ideas suitable for McLachlan checkpoint robustness
- **relation:** Geo-ADAPT adds geometry-aware selection and conditioning ideas suitable for McLachlan checkpoint robustness
- **abstract:**
  - > "Adaptive ansatz construction has emerged as a powerful technique for reducing circuit depth and improving optimization efficiency in variational quantum eigensolvers."
  - > "Existing adaptive methods, including ADAPT-VQE, rely solely on first-order gradients and ignore the underlying geometry of the quantum state space, limiting both convergence behavior and operator-selection efficiency."
  - > "Geo-ADAPT-VQE selects operators from a pool using the natural gradient rule, guiding the ansatz to grow along directions aligned with the underlying quantum-state geometry."
  - > "This improves convergence, reduces susceptibility to shallow local minima and saddle points, and can yield significantly shorter ans"atze."

### Barison2021PVQD
**[Barison2021PVQD]**  
S. Barison, F. Vicentini, and G. Carleo,  
``An efficient quantum algorithm for the time evolution of parameterized circuits,''  
\emph{Quantum} \textbf{5}, 512 (2021).  

- **contribution:** PVQD algorithmic framing supports practical parameterized-circuit time evolution in the dynamic section
- **relation:** PVQD algorithmic framing supports practical parameterized-circuit time evolution in the dynamic section
- **abstract:**
  - > "We introduce a novel hybrid algorithm to simulate the real-time evolution of quantum systems using parameterized quantum circuits."
  - > "The method, named "projected – Variational Quantum Dynamics" (p-VQD) realizes an iterative, global projection of the exact time evolution onto the parameterized manifold."
  - > "In the small time-step limit, this is equivalent to the McLachlan's variational principle."
  - > "Through numerical experiments, we also show that our approach is particularly advantageous over existing global optimization algorithms based on the time-dependent variational principle that, due to a demanding quadratic scaling with parameter numbers, are unsuitable for large parameterized quantum circuits."

### Yao2021AVQDS
**[Yao2021AVQDS]**  
Y.-X. Yao, N. Gomes, F. Zhang, C.-Z. Wang, K.-M. Ho, T. Iadecola, and P. P. Orth,  
``Adaptive variational quantum dynamics simulations,''  
\emph{PRX Quantum} \textbf{2}, 030307 (2021).  

- **contribution:** AVQDS contributes adaptive control of McLachlan distance and ansatz growth
- **relation:** AVQDS contributes adaptive control of McLachlan distance and ansatz growth
- **abstract:**
  - > "We propose a general-purpose, self-adaptive approach to construct a variational wave-function ansatz for highly accurate quantum dynamics simulations based on McLachlan's variational principle."
  - > "The key idea is to dynamically expand the variational ansatz along the time-evolution path such that the "McLachlan distance", which is a measure of the simulation accuracy, remains below a set threshold."
  - > "We apply this adaptive variational quantum dynamics simulation (AVQDS) approach to the integrable Lieb-Schultz-Mattis spin chain and the nonintegrable mixed-field Ising model, where it captures both finite-rate and sudden post-quench dynamics with high fidelity."
  - > "We envision that a wide range of dynamical simulations of quantum many-body systems on near-term quantum-computing devices will be made possible through the AVQDS framework."

### Linteau2024APVQD
**[Linteau2024APVQD]**  
D. Linteau, S. Barison, N. H. Lindner, and G. Carleo,  
``Adaptive projected variational quantum dynamics,''  
\emph{Physical Review Research} \textbf{6}, 023130 (2024).  

- **contribution:** Adaptive projected dynamics aligns with the projected-evolution controller and trajectory maintenance logic
- **relation:** Adaptive projected dynamics aligns with the projected-evolution controller and trajectory maintenance logic
- **abstract:**
  - > "We propose an adaptive quantum algorithm to prepare accurate variational time evolved wave functions."
  - > "The method is based on the projected variational quantum dynamics (pVQD) algorithm, that performs a global optimization with linear scaling in the number of variational parameters."
  - > "Instead of fixing a variational ansatz at the beginning of the simulation, the circuit is grown systematically during the time evolution."
  - > "Published by the American Physical Society 2024"

### Zhang2025CompressedAVQD
**[Zhang2025CompressedAVQD]**  
F. Zhang, C.-Z. Wang, T. Iadecola, P. P. Orth, and Y.-X. Yao,  
``Adaptive variational quantum dynamics simulations with compressed circuits and fewer measurements,''  
\emph{Physical Review B} \textbf{111}, 094310 (2025).  

- **contribution:** Compressed AVQD informs the reduced-measurement policy and low-cost evolution variants
- **relation:** Compressed AVQD informs the reduced-measurement policy and low-cost evolution variants
- **abstract:**
  - > "The adaptive variational quantum dynamics simulation (AVQDS) method performs real-time evolution of quantum states using automatically generated parametrized quantum circuits that often contain substantially fewer gates than Trotter circuits."
  - > "Here we report an improved version of the method, which we call AVQDS(T), by porting the tiling efficient trial circuits with rotations implemented simultaneously technique."
  - > "The algorithm adaptively adds layers of disjoint unitary gates to the ansatz circuit so as to keep the McLachlan distance, a measure of the accuracy of the variational dynamics, below a fixed threshold."
  - > "We showcase that this approach enables AVQDS(T) to deliver more accurate results than simulations using a fixed ansatz of comparable final depth for a significant time duration with fewer quantum resources."

### ProvostVallee1980
**[ProvostVallee1980]**  
J. P. Provost and G. Vall\'ee,  
``Riemannian structure on manifolds of quantum states,''  
\emph{Communications in Mathematical Physics} \textbf{76}, 289--301 (1980).  

- **contribution:** Provost-Vallée provides the geometric foundation for metric-aware projections and distance measures used throughout McLachlan updates
- **relation:** Provost-Vallée provides the geometric foundation for metric-aware projections and distance measures used throughout McLachlan updates
- **abstract:**
  - > "A metric tensor is defined from the underlying Hilbert space structure for any submanifold of quantum states."
  - > "The case where the manifold is generated by the action of a Lie group on a fixed state vector (generalized coherent states manifold hereafter noted G.C.S.M.) is studied in details; the geometrical properties of some well-known G.C.S.M."
  - > "are reviewed and an explicit expression for the scalar Riemannian curvature is given in the general case."
  - > "It is shown on examples that the distance between nearby states is related to quantum fluctuations; in the particular case of the harmonic oscillator group the condition of zero curvature is identical to the condition of non-dispersion of wave packets."

### Verteletskyi2020MeasurementGrouping
**[Verteletskyi2020MeasurementGrouping]**  
V. Verteletskyi, T.-C. Yen, and A. F. Izmaylov,  
``Measurement optimization in the variational quantum eigensolver using a minimum clique cover,''  
\emph{Journal of Chemical Physics} \textbf{152}, 124114 (2020).  

- **contribution:** Verteletskyi’s minimum-clique-grouping logic reduces measurement burden in both evolving trajectories and subspace post-processing
- **relation:** Verteletskyi’s minimum-clique-grouping logic reduces measurement burden in both evolving trajectories and subspace post-processing
- **abstract:**
  - > "Solving the electronic structure problem using the Variational Quantum Eigensolver (VQE) technique involves the measurement of the Hamiltonian expectation value."
  - > "The current hardware can perform only projective single-qubit measurements, and thus, the Hamiltonian expectation value is obtained by measuring parts of the Hamiltonian rather than the full Hamiltonian."
  - > "This restriction makes the measurement process inefficient because the number of terms in the Hamiltonian grows as O(N4) with the size of the system, N."
  - > "On average, grouping qubit-wise commuting terms reduced the number of operators to measure three times less compared to the total number of terms in the considered Hamiltonians."

### Urbanek2020VQSE
**[Urbanek2020VQSE]**  
M. Urbanek, D. Camps, R. Van Beeumen, and W. A. de Jong,  
``Chemistry on quantum computers with virtual quantum subspace expansion,''  
\emph{Journal of Chemical Theory and Computation} \textbf{16}, 5425--5431 (2020).  

- **contribution:** Virtual QSE contributes subspace-expansion infrastructure for excited-state corrections that can be layered after dynamic propagation
- **relation:** Virtual QSE contributes subspace-expansion infrastructure for excited-state corrections that can be layered after dynamic propagation
- **abstract:**
  - > "Simulating chemical systems on quantum computers has been limited to a few electrons in a minimal basis."
  - > "We demonstrate experimentally that the virtual quantum subspace expansion (Takeshita, T.; <i>Phys."
  - > "Rev."
  - > "In addition, we were able to obtain an accurate potential energy curve for the nitrogen dimer in a quantum simulation on a classical computer."

### Yoshioka2022GQSE
**[Yoshioka2022GQSE]**  
N. Yoshioka, H. Hakoshima, Y. Matsuzaki, Y. Tokunaga, Y. Suzuki, and S. Endo,  
``Generalized quantum subspace expansion,''  
\emph{Physical Review Letters} \textbf{129}, 020502 (2022).  

- **contribution:** Generalized QSE extends subspace construction to richer Hamiltonians, supporting robust excited-state estimation in dynamic contexts
- **relation:** Generalized QSE extends subspace construction to richer Hamiltonians, supporting robust excited-state estimation in dynamic contexts
- **abstract:**
  - > "One of the major challenges for erroneous quantum computers is undoubtedly the control over the effect of noise."
  - > "Considering the rapid growth of available quantum resources that are not fully fault tolerant, it is crucial to develop practical hardware-friendly quantum error mitigation (QEM) techniques to suppress unwanted errors."
  - > "Here, we propose a novel generalized quantum subspace expansion method which can handle stochastic, coherent, and algorithmic errors in quantum computers."
  - > "We numerically demonstrate in both situations that we can suppress errors by orders of magnitude, and show that our protocol inherits the advantages of previous error-agnostic QEM techniques as well as overcoming their drawbacks."

### CortesGray2022QKSD
**[CortesGray2022QKSD]**  
C. L. Cortes and S. K. Gray,  
``Quantum Krylov subspace algorithms for ground- and excited-state energy estimation,''  
\emph{Physical Review A} \textbf{105}, 022417 (2022).  

- **contribution:** QKSD framework proposes Krylov-space excited-state estimation strategies, relevant as an optional spectral branch for the time-dependent results
- **relation:** QKSD framework proposes Krylov-space excited-state estimation strategies, relevant as an optional spectral branch for the time-dependent results
- **abstract:**
  - > "Quantum Krylov subspace diagonalization (QKSD) algorithms provide a low-cost alternative to the conventional quantum phase estimation algorithm for estimating ground- and excited-state energies of a quantum many-body system."
  - > "While QKSD algorithms typically rely on the Hadamard test for estimating Krylov subspace matrix elements of the form (phi(i)|e^{-i H \tau}|phi(j)), these circuits can require expensive controlled multiqubit gates on near-term hardware."
  - > "The paper shows that, for a wide class of Hamiltonians with relevant symmetries, alternative ancilla-economical approaches can be used."
  - > "It proposes a multifidelity estimation protocol and develops a unified framework with three quantum-classical algorithms balancing circuit calls, depth, classical complexity, and generalized-eigenvalue stability."

### Motta2024SubspaceReview
**[Motta2024SubspaceReview]**  
M. Motta et al.,  
``Subspace methods for electronic structure simulations on quantum computers,''  
\emph{Electronic Structure} \textbf{6}, 013001 (2024).  

- **contribution:** Motta provides a broader subspace-method synthesis useful for positioning and benchmarking the spectral post-processing choices
- **relation:** Motta provides a broader subspace-method synthesis useful for positioning and benchmarking the spectral post-processing choices
- **abstract:**
  - > "Quantum subspace methods (QSMs) are a class of quantum computing algorithms where the time-independent Schrödinger equation for a quantum system is projected onto a subspace of the underlying Hilbert space."
  - > "This projection transforms the Schrödinger equation into an eigenvalue problem determined by measurements carried out on a quantum device."
  - > "The eigenvalue problem is then solved on a classical computer, yielding approximations to ground- and excited-state energies and wavefunctions."
  - > "We present the theoretical foundations and applications of QSMs, and we discuss their implementation on quantum hardware, illustrating the impact of noise on their performance."

### Epperly2022QSD
**[Epperly2022QSD]**  
E. N. Epperly, L. Lin, and Y. Nakatsukasa,  
``A theory of quantum subspace diagonalization,''  
\emph{SIAM Journal on Matrix Analysis and Applications} \textbf{43}, 1263--1290 (2022).  

- **contribution:** QSD formalism gives the linear-algebraic basis for stable subspace diagonalization used in the excited pipeline
- **relation:** QSD formalism gives the linear-algebraic basis for stable subspace diagonalization used in the excited pipeline
- **abstract:**
  - > "Quantum subspace diagonalization methods are an exciting new class of algorithms for solving large-scale eigenvalue problems using quantum computers."
  - > "Unfortunately, these methods require the solution of an ill-conditioned generalized eigenvalue problem, with a matrix pencil corrupted by a nonnegligible amount of noise that is far above the machine precision."
  - > "Despite pessimistic predictions from classical worst-case perturbation theories, these methods can perform reliably well if the generalized eigenvalue problem is solved using a standard truncation strategy."
  - > "Our new results can also be of independent interest to solving eigenvalue problems outside the context of quantum computation."

### Lee2024SamplingQKSD
**[Lee2024SamplingQKSD]**  
G. Lee, D. Lee, and J. Huh,  
``Sampling error analysis in quantum Krylov subspace diagonalization,''  
\emph{Quantum} \textbf{8}, 1477 (2024).  

- **contribution:** Sampling-error analysis in QKSD guides finite-shot confidence and estimator design in the spectral reconstructions
- **relation:** Sampling-error analysis in QKSD guides finite-shot confidence and estimator design in the spectral reconstructions
- **abstract:**
  - > "Quantum Krylov subspace diagonalization (QKSD) is an emerging method used in place of quantum phase estimation in the early fault-tolerant era, where limited quantum circuit depth is available."
  - > "In contrast to the classical Krylov subspace diagonalization (KSD) or the Lanczos method, QKSD exploits the quantum computer to efficiently estimate the eigenvalues of large-size Hamiltonians through a faster Krylov projection."
  - > "However, unlike classical KSD, which is solely concerned with machine precision, QKSD is inherently accompanied by errors originating from a finite number of samples."
  - > "Numerical simulations of the one-dimensional Hubbard model demonstrate that the error bound of finite samplings accurately predicts the experimental errors in well-conditioned regions."

### Getelina2024HardwareNoiseQSE
**[Getelina2024HardwareNoiseQSE]**  
J. C. Getelina, P. Sharma, T. Iadecola, P. P. Orth, and Y.-X. Yao,  
``Quantum subspace expansion in the presence of hardware noise,''  
\emph{APL Quantum} \textbf{1}, 036127 (2024).  

- **contribution:** Getelina studies hardware-noise impacts and mitigation in QSE, relevant for the realism-aware excited-state checks
- **relation:** Getelina studies hardware-noise impacts and mitigation in QSE, relevant for the realism-aware excited-state checks
- **abstract:**
  - > "Finding ground state energies on current quantum processing units (QPUs) using algorithms such as the variational quantum eigensolver (VQE) continues to pose challenges."
  - > "Hardware noise severely affects both the expressivity and trainability of parameterized quantum circuits, limiting them to shallow depths in practice."
  - > "Here, we demonstrate that both issues can be addressed by synergistically integrating VQE with a quantum subspace expansion, allowing for an optimal balance between quantum and classical computing capabilities and costs."
  - > "Finally, we show how to systematically approach the exact solution by performing controlled quantum error mitigation based on probabilistic error reduction on the noisy backend fake_guadalupe."

### OLeary2025PartitionedQSE
**[OLeary2025PartitionedQSE]**  
T. O'Leary, L. W. Anderson, D. Jaksch, and M. Kiffner,  
``Partitioned quantum subspace expansion,''  
\emph{Quantum} \textbf{9}, 1726 (2025).  

- **contribution:** Partitioned QSE adds scalable decomposition ideas for large Hilbert subspaces in spectral workflows
- **relation:** Partitioned QSE adds scalable decomposition ideas for large Hilbert subspaces in spectral workflows
- **abstract:**
  - > "We present an iterative generalisation of the quantum subspace expansion algorithm used with a Krylov basis."
  - > "The iterative construction connects a sequence of subspaces via their lowest energy states."
  - > "Diagonalising a Hamiltonian in a given Krylov subspace requires the same quantum resources in both the single step and sequential cases."
  - > "Our work suggests that the numerical instability limiting the accuracy of this approach can be substantially alleviated in a parameter-free way."

### Boyd2025ShadowSubspace
**[Boyd2025ShadowSubspace]**  
G. Boyd, B. Koczor, and Z. Cai,  
``High-dimensional subspace expansion using classical shadows,''  
\emph{Physical Review A} \textbf{111}, 022423 (2025).  

- **contribution:** Boyd demonstrates classical-shadow QSE scaling for high-dimensional settings, informing shot-efficient subspace methods
- **relation:** Boyd demonstrates classical-shadow QSE scaling for high-dimensional settings, informing shot-efficient subspace methods
- **abstract:**
  - > "We introduce a postprocessing technique for classical shadow measurement data that enhances the precision of ground state estimation through high-dimensional subspace expansion; the dimensionality is only limited by the amount of classical postprocessing resources rather than by quantum resources."
  - > "Crucial steps of our approach are the efficient identification of useful observables from shadow data, followed by our regularized subspace expansion that is designed to be numerically stable even when using noisy data."
  - > "We analytically investigate noise propagation within our method, and upper bound the statistical fluctuations due to the limited number of snapshots in classical shadows."
  - > "Furthermore, performance is guaranteed to be at least as good—and in many cases better—than direct energy estimation without using additional quantum resources, and the approach is thus a very natural alternative for estimating ground state energies directly from classical shadow data."

### Kwao2026GEPQSE
**[Kwao2026GEPQSE]**  
P. F. Kwao, S. P. Sundar, B. Gupt, and A. Asthana,  
``Generalized eigenvalue problem in subspace-based excited-state methods for quantum computers,''  
\emph{Journal of Chemical Theory and Computation} \textbf{22}, 2892--2903 (2026).  

- **contribution:** Generalized eigenvalue problem framing strengthens stability and interpretability of the excited-state postprocessing
- **relation:** Generalized eigenvalue problem framing strengthens stability and interpretability of the excited-state postprocessing
- **abstract:**
  - > "Solving challenging problems in quantum chemistry is one of the most promising applications of quantum computers."
  - > "Within the quantum algorithms proposed for problems in excited-state quantum chemistry, subspace-based quantum algorithms, including quantum subspace expansion (QSE), quantum equation of motion (qEOM), and quantum self-consistent equation-of-motion (q-sc-EOM), are promising for pre-fault-tolerant quantum devices."
  - > "The working equation of QSE and qEOM requires solving a generalized eigenvalue equation with associated matrix elements measured on a quantum computer."
  - > "We also show that excited-state methods that have an eigenvalue equation as the working equation, such as q-sc-EOM, do not have the problems associated with the condition number and could be generally more stable to errors and therefore more suitable candidates for excited-state quantum chemistry calculations using quantum computers."

### ChoiIzmaylov2023MeasurementQSE
**[ChoiIzmaylov2023MeasurementQSE]**  
S. Choi and A. F. Izmaylov,  
``Measurement optimization techniques for excited electronic states in near-term quantum computing algorithms,''  
\emph{Journal of Chemical Theory and Computation} \textbf{19}, 3184--3193 (2023).  

- **contribution:** optimizes measurement for excited-state QSE, directly supporting the measurable objective design
- **relation:** optimizes measurement for excited-state QSE, directly supporting the measurable objective design
- **abstract:**
  - > "The variational quantum eigensolver (VQE) remains one of the most popular near-term quantum algorithms for solving the electronic structure problem."
  - > "Yet, for its practicality, the main challenge to overcome is improving the quantum measurement efficiency."
  - > "Numerous quantum measurement techniques have been developed recently, but it is unclear how these state-of-the-art measurement techniques will perform in extensions of VQE for obtaining excited electronic states."
  - > "Nevertheless, when the best possible measurement technique for each excited state VQE algorithm is considered, significantly fewer measurements are required in multistate contraction than in quantum subspace expansion."

### Nakamura2024AdaptiveMeasurementQSE
**[Nakamura2024AdaptiveMeasurementQSE]**  
Y. Nakamura, Y. Yano, and N. Yoshioka,  
``Adaptive measurement strategy for quantum subspace methods,''  
\emph{New Journal of Physics} \textbf{26}, 033028 (2024).  

- **contribution:** Nakamura introduces adaptive measurement scheduling for subspace tasks, aligned with finite-shot McLachlan/controller constraints
- **relation:** Nakamura introduces adaptive measurement scheduling for subspace tasks, aligned with finite-shot McLachlan/controller constraints
- **abstract:**
  - > "Estimation of physical observables for unknown quantum states is an important problem that underlies a wide range of fields, including quantum information processing, quantum physics, and quantum chemistry."
  - > "In the context of quantum computation, in particular, existing studies have mainly focused on holistic state tomography or estimation on specific observables with known classical descriptions, while this lacks the important class of problems where the estimation target itself relies on the measurement outcome."
  - > "In this work, we propose an adaptive measurement optimization method that is useful for the quantum subspace methods, namely the variational simulation methods that utilize classical postprocessing on measurement outcomes."
  - > "Our work reveals that the potential of the QSE method can be empowered by elaborated measurement protocols, and opens a path to further pursue efficient quantum measurement techniques in practical computations."

### Jamet2022GreenQSE
**[Jamet2022GreenQSE]**  
F. Jamet, A. Agarwal, and I. Rungger,  
``Quantum subspace expansion algorithm for Green's functions,''  
arXiv:2205.00094 (2022).  

- **contribution:** Jamet extends QSE to Green’s functions, supporting response-function observables from dynamic trajectories
- **relation:** Jamet extends QSE to Green’s functions, supporting response-function observables from dynamic trajectories
- **abstract:**
  - > "One of the major challenges for erroneous quantum computers is undoubtedly the control over the effect of noise."
  - > "Considering the rapid growth of available quantum resources that are not fully fault tolerant, it is crucial to develop practical hardware-friendly quantum error mitigation (QEM) techniques to suppress unwanted errors."
  - > "Here, we propose a novel generalized quantum subspace expansion method which can handle stochastic, coherent, and algorithmic errors in quantum computers."
  - > "We numerically demonstrate in both situations that we can suppress errors by orders of magnitude, and show that our protocol inherits the advantages of previous error-agnostic QEM techniques as well as overcoming their drawbacks."

### Dhawan2024ImagTimeGreen
**[Dhawan2024ImagTimeGreen]**  
D. Dhawan, D. Zgid, and M. Motta,  
``Quantum algorithm for imaginary-time Green's functions,''  
\emph{Journal of Chemical Theory and Computation} \textbf{20}, 4629--4638 (2024).  

- **contribution:** Dhawan develops imaginary-time Green-function algorithms, relevant for response/correlation validation beyond raw energies
- **relation:** Dhawan develops imaginary-time Green-function algorithms, relevant for response/correlation validation beyond raw energies
- **abstract:**
  - > "Green's function methods lead to ab initio, systematically improvable simulations of molecules and materials while providing access to multiple experimentally observable properties such as the density of states and the spectral function."
  - > "The calculation of the exact one-particle Green's function remains a significant challenge for classical computers and was attempted only on very small systems."
  - > "Here, we present a hybrid quantum-classical algorithm to calculate the imaginary-time one-particle Green's function."
  - > "We demonstrate the validity of this algorithm by simulating H<sub>2</sub> and H<sub>4</sub> on quantum simulators and on IBM's quantum devices."

### Gauthier2024OccupationQSE
**[Gauthier2024OccupationQSE]**  
B. Gauthier, P. Rosenberg, A. Foley, and M. Charlebois,  
``Occupation-number quantum-subspace-expansion approach to computing the single-particle Green function: An opportunity for noise filtering,''  
\emph{Physical Review A} \textbf{110}, 032624 (2024).  

- **contribution:** Gauthier brings occupation-number expansion and noise filtering for QSE observables, supporting reliability in noisy trajectory outputs
- **relation:** Gauthier brings occupation-number expansion and noise filtering for QSE observables, supporting reliability in noisy trajectory outputs
- **abstract:**
  - > "We introduce a hybrid quantum-classical algorithm to compute the Green function for strongly correlated electrons on noisy intermediate-scale quantum (NISQ) devices."
  - > "The technique consists in the construction of a nonorthogonal excitation basis composed of a set of single-particle excitations generated by occupation-number operators."
  - > "The excited sectors of the Hamiltonian in this basis can then be measured on the quantum device and a classical postprocessing procedure yields the Green function in the Lehmann representation."
  - > "A simulation of a four-site system carried out on classical hardware suggests that the approach can achieve similar accuracy for larger systems."

### Kokcu2024LinearResponse
**[Kokcu2024LinearResponse]**  
E. K\"okc\"u, H. A. Labib, J. K. Freericks, and A. F. Kemper,  
``A linear response framework for simulating bosonic and fermionic correlation functions illustrated on quantum computers,''  
\emph{Nature Communications} \textbf{15}, 3881 (2024).  

- **contribution:** Kokcu establishes linear-response QSE frameworks for bosonic and fermionic correlations, informing the response-type checks
- **relation:** Kokcu establishes linear-response QSE frameworks for bosonic and fermionic correlations, informing the response-type checks
- **abstract:**
  - > "Response functions are a fundamental aspect of physics; they represent the link between experimental observations and the underlying quantum many-body state."
  - > "However, this link is often under-appreciated, as the Lehmann formalism for obtaining response functions in linear response has no direct link to experiment."
  - > "Within the context of quantum computing, and via a linear response framework, we restore this link by making the experiment an inextricable part of the quantum simulation."
  - > "The linear response method provides a robust framework for using quantum computers to study systems in physics and chemistry."

### Umeano2025KitaevQSE
**[Umeano2025KitaevQSE]**  
C. Umeano, F. Jamet, L. P. Lindoy, I. Rungger, and O. Kyriienko,  
``Quantum subspace expansion approach for simulating dynamical response functions of Kitaev spin liquids,''  
\emph{Physical Review Materials} \textbf{9}, 034401 (2025).  

- **contribution:** Umeano demonstrates QSE-based dynamical-response workflows on complex spin systems, supporting generality of the spectral branch
- **relation:** Umeano demonstrates QSE-based dynamical-response workflows on complex spin systems, supporting generality of the spectral branch
- **abstract:**
  - > "We develop a quantum simulation-based approach for studying properties of strongly correlated magnetic materials at increasing scale."
  - > "We consider a paradigmatic example of a quantum spin liquid (QSL) state hosted by the honeycomb Kitaev model and use a trainable, symmetry-guided ansatz for preparing its ground state."
  - > "Applying the tools of quantum subspace expansion (QSE), Hamiltonian operator approximation, and overlap measurements, we simulate the QSL at zero temperature and finite magnetic field, thus moving outside of the symmetric subspace."
  - > "Our results show that quantum simulators offer an insight into quasiparticle properties of strongly correlated magnets and can become a valuable tool for studying material science."

### Backes2023HHDMFT
**[Backes2023HHDMFT]**  
S. Backes, Y. Murakami, S. Sakai, and R. Arita,  
``Dynamical mean-field theory for the Hubbard-Holstein model on a quantum device,''  
\emph{Physical Review B} \textbf{107}, 165155 (2023).  

- **contribution:** Backes applies DMFT to Hubbard-Holstein on quantum hardware, giving advanced dynamic-context precedent and benchmarking context
- **relation:** Backes applies DMFT to Hubbard-Holstein on quantum hardware, giving advanced dynamic-context precedent and benchmarking context
- **abstract:**
  - > "Utilizing quantum computers for real materials calculations is highly promising, but hinges on qubit fidelity and fault-tolerant quantum algorithms."
  - > "Here, the authors show that a representative model for real materials, the Hubbard-Holstein model, which describes interacting electrons on a lattice coupled to bosonic degrees of freedom, can be reliably investigated using today's Quantum processors."
  - > "Adapting state-of-the-art mathematical tools is shown to be critical to overcome limitations imposed by the noise level of current quantum computing devices."

### Denner2023HybridEP
**[Denner2023HybridEP]**  
M. M. Denner et al.,  
``A hybrid quantum-classical method for electron-phonon systems,''  
\emph{Communications Physics} \textbf{6}, 233 (2023).  

- **contribution:** Denner2023HybridEP gives a hybrid protocol for electron-phonon dynamics, useful as an external benchmark line
- **relation:** Denner2023HybridEP gives a hybrid protocol for electron-phonon dynamics, useful as an external benchmark line
- **abstract:**
  - > "Interactions between electrons and phonons play a crucial role in quantum materials."
  - > "Yet, there is no universal method that would simultaneously accurately account for strong electron-phonon interactions and electronic correlations."
  - > "By combining methods of the variational quantum eigensolver and the variational non-Gaussian solver, we develop a hybrid quantum-classical algorithm suitable for this type of correlated systems."
  - > "We benchmark our method by applying it to the paradigmatic Hubbard-Holstein model at half filling, and show that it correctly captures the competition between charge density wave and antiferromagnetic phases, quantitatively consistent with exact diagonalization."

### Dutta2024BosonicChemistry
**[Dutta2024BosonicChemistry]**  
R. Dutta et al.,  
``Simulating chemistry on bosonic quantum devices,''  
\emph{Journal of Chemical Theory and Computation} \textbf{20}, 6426--6441 (2024).  

- **contribution:** bosonic-device chemistry survey informs architecture-level assumptions and method choices for phonon-heavy dynamics
- **relation:** bosonic-device chemistry survey informs architecture-level assumptions and method choices for phonon-heavy dynamics
- **abstract:**
  - > "Bosonic quantum devices offer a novel approach to realize quantum computations, where the quantum two-level system (<i>qubit</i>) is replaced with the quantum (an)harmonic oscillator (<i>qumode</i>) as the fundamental building block of the quantum simulator."
  - > "The simulation of chemical structure and dynamics can then be achieved by representing or mapping the system Hamiltonians in terms of bosonic operators."
  - > "In this Perspective, we review recent progress and future potential of using bosonic quantum devices for addressing a wide range of challenging chemical problems, including the calculation of molecular vibronic spectra, the simulation of gas-phase and solution-phase adiabatic and nonadiabatic chemical dynamics, the efficient solution of molecular graph theory problems, and the calculations of electronic structure."

### Peng2025BosonHamiltonians
**[Peng2025BosonHamiltonians]**  
B. Peng, Y. Su, D. Claudino, K. Kowalski, G. H. Low, and M. Roetteler,  
``Quantum simulation of boson-related Hamiltonians: Techniques, effective Hamiltonian construction, and error analysis,''  
\emph{Quantum Science and Technology} \textbf{10}, 023002 (2025).  

- **contribution:** Peng provides bosonic Hamiltonian encoding and error-analysis guidance for algorithmic choices in dynamic McLachlan tests
- **relation:** Peng provides bosonic Hamiltonian encoding and error-analysis guidance for algorithmic choices in dynamic McLachlan tests
- **abstract:**
  - > "Elementary quantum mechanics proposes that a closed physical system consistently evolves in a reversible manner."
  - > "However, control and readout necessitate the coupling of the quantum system to the external environment, subjecting it to relaxation and decoherence."
  - > "Consequently, system-environment interactions are indispensable for simulating physically significant theories."
  - > "Specifically, we emphasize two aspects: (1) the discussion of recently developed quantum algorithms for these interacting models and the construction of effective Hamiltonians, and (2) a detailed analysis regarding a tightened error bound for truncating the bosonic modes for a class of fermion-boson interacting Hamiltonians."

### Fischer2025ShadowQSE
**[Fischer2025ShadowQSE]**  
L. E. Fischer, D. Bultrini, I. Tavernelli, and F. Tacchino,  
``Large-scale implementation of quantum subspace expansion with classical shadows,''  
arXiv:2510.25640 (2025).  

- **contribution:** Fischer scales classical-shadow QSE to large spaces, informing the noise-aware spectral-stack design
- **relation:** Fischer scales classical-shadow QSE to large spaces, informing the noise-aware spectral-stack design
- **abstract:**
  - > "We introduce a postprocessing technique for classical shadow measurement data that enhances the precision of ground state estimation through high-dimensional subspace expansion; the dimensionality is only limited by the amount of classical postprocessing resources rather than by quantum resources."
  - > "Crucial steps of our approach are the efficient identification of useful observables from shadow data, followed by our regularized subspace expansion that is designed to be numerically stable even when using noisy data."
  - > "We analytically investigate noise propagation within our method, and upper bound the statistical fluctuations due to the limited number of snapshots in classical shadows."
  - > "Furthermore, performance is guaranteed to be at least as good—and in many cases better—than direct energy estimation without using additional quantum resources, and the approach is thus a very natural alternative for estimating ground state energies directly from classical shadow data."
## Part 3 (Paper III: Quantum Subspace Expansion)
These entries track the QSE-oriented paper, including nonorthogonal subspace methods, excited-state response theory, and live/frozen excited dynamics baselines.

### Hubbard1963
**[Hubbard1963]**  
J. Hubbard,  
``Electron correlations in narrow energy bands,''  
\emph{Proceedings of the Royal Society A} \textbf{276}, 238--257 (1963).  

- **contribution:** Hubbard1963 establishes the Hubbard model as the canonical lattice Hamiltonian for strong electron correlation, grounding the model setup
- **relation:** supports the static-adapt Hamiltonian construction and connects directly to Holstein benchmarks used across both papers
- **abstract:**
  - > "It is pointed out that one of the main effects of correlation phenomena in d- and f-bands is to give rise to behaviour characteristic of the atomic or Heitler-London model."
  - > "To investigate this situation a simple, approximate model for the interaction of electrons in narrow energy bands is introduced."
  - > "The results of applying the Hartree-Fock approximation to this model are examined."
  - > "To clarify the physical meaning of the solution a two-electron example is examined."

### Sohail2026GeoADAPT
**[Sohail2026GeoADAPT]**  
M. A. Sohail and T. Koike-Akino,  
``Geo-ADAPT-VQE: Quantum information metric-aware circuit optimization for quantum chemistry,''  
arXiv:2603.10325 (2026).  

- **contribution:** Sohail introduces geometry-aware ADAPT with natural-gradient-style metric information
- **relation:** contributes a conditioning and convergence stabilization principle for adaptive ansatz selection in the static design
- **abstract:**
  - > "Adaptive ansatz construction has emerged as a powerful technique for reducing circuit depth and improving optimization efficiency in variational quantum eigensolvers."
  - > "Existing adaptive methods, including ADAPT-VQE, rely solely on first-order gradients and ignore the underlying geometry of the quantum state space, limiting both convergence behavior and operator-selection efficiency."
  - > "Geo-ADAPT-VQE selects operators from a pool using the natural gradient rule, guiding the ansatz to grow along directions aligned with the underlying quantum-state geometry."
  - > "This improves convergence, reduces susceptibility to shallow local minima and saddle points, and can yield significantly shorter ans"atze."

### ProvostVallee1980
**[ProvostVallee1980]**  
J. P. Provost and G. Vall\'ee,  
``Riemannian structure on manifolds of quantum states,''  
\emph{Communications in Mathematical Physics} \textbf{76}, 289--301 (1980).  

- **contribution:** classic differential-geometry paper defines a quantum-state metric from Hilbert-space structure, supporting the use of geometric optimization information
- **relation:** provides the theoretical basis for metric-aware updates in the adaptive pipelines
- **abstract:**
  - > "A metric tensor is defined from the underlying Hilbert space structure for any submanifold of quantum states."
  - > "The case where the manifold is generated by the action of a Lie group on a fixed state vector (generalized coherent states manifold hereafter noted G.C.S.M.) is studied in details; the geometrical properties of some well-known G.C.S.M."
  - > "are reviewed and an explicit expression for the scalar Riemannian curvature is given in the general case."
  - > "It is shown on examples that the distance between nearby states is related to quantum fluctuations; in the particular case of the harmonic oscillator group the condition of zero curvature is identical to the condition of non-dispersion of wave packets."

### McLachlan1964
**[McLachlan1964]**  
A. D. McLachlan,  
``A variational solution of the time-dependent Schr\"odinger equation,''  
\emph{Molecular Physics} \textbf{8}, 39--44 (1964).  

- **contribution:** the foundational variational principle underlying the McLachlan equations used in the dynamic controller
- **relation:** directly defines the governing projector and residual-minimization logic for trajectory updates
- **abstract:**
  - > "A variational method is described for finding approximate solutions of the time-dependent Schrödinger equation."
  - > "It is closely related to Frenkel's variational approach but better in some respects because it always leads to a true minimum of the error."
  - > "For several interacting systems it leads to the time-dependent Hartree equations; for a system in an oscillating electric or magnetic field it gives a variational principle for calculating susceptibility at any frequency."

### Yuan2019VQS
**[Yuan2019VQS]**  
X. Yuan, S. Endo, Q. Zhao, Y. Li, and S. C. Benjamin,  
``Theory of variational quantum simulation,''  
\emph{Quantum} \textbf{3}, 191 (2019).  

- **contribution:** Yuan gives the general theory of variational quantum simulation, undergirding the state evolution derivation
- **relation:** supports correctness framing for McLachlan-style equations
- **abstract:**
  - > "The variational method is a versatile tool for classical simulation of a variety of quantum systems."
  - > "Great efforts have recently been devoted to its extension to quantum computing for efficiently solving static many-body problems and simulating real and imaginary time dynamics."
  - > "In this work, we first review the conventional variational principles, including the Rayleigh-Ritz method for solving static problems, and the Dirac and Frenkel variational principle, the McLachlan's variational principle, and the time-dependent variational principle, for simulating real time dynamics."
  - > "Our work completes the theory of variational quantum simulation of general real and imaginary time evolution and it is applicable to near-term quantum hardware."

### Hubbard1963
**[Hubbard1963]**  
J. Hubbard,  
``Electron correlations in narrow energy bands,''  
\emph{Proceedings of the Royal Society A} \textbf{276}, 238--257 (1963).  

- **contribution:** Hubbard supplies the canonical strongly correlated lattice Hamiltonian for dynamic case studies
- **relation:** Hubbard supplies the canonical strongly correlated lattice Hamiltonian for dynamic case studies
- **abstract:**
  - > "It is pointed out that one of the main effects of correlation phenomena in d- and f-bands is to give rise to behaviour characteristic of the atomic or Heitler-London model."
  - > "To investigate this situation a simple, approximate model for the interaction of electrons in narrow energy bands is introduced."
  - > "The results of applying the Hartree-Fock approximation to this model are examined."
  - > "To clarify the physical meaning of the solution a two-electron example is examined."

### Holstein1959
**[Holstein1959]**  
T. Holstein,  
``Studies of polaron motion: Part I. The molecular-crystal model,''  
\emph{Annals of Physics} \textbf{8}, 325--342 (1959);  
``Studies of polaron motion: Part II. The `small' polaron,''  
\emph{Annals of Physics} \textbf{8}, 343--389 (1959).  

- **contribution:** Holstein1959 combines Part I/II physical polaron dynamics assumptions and transport regimes relevant to the Holstein checkpoint trajectories
- **relation:** Holstein1959 combines Part I/II physical polaron dynamics assumptions and transport regimes relevant to the Holstein checkpoint trajectories
- **abstract:**
  - > "The one-dimensional molecular-crystal model of polaron motion described in Part I incorporates the principal physical features of the problem and yields localized electron states in zeroth order."
  - > "Part II analyzes the same model when the electronic-overlap term is a small perturbation: zeroth order gives site-localized electrons, while non-vanishing overlap produces neighboring-site transitions."
  - > "It distinguishes diagonal and non-diagonal transitions, explains low-temperature Bloch-band behavior and high-temperature diffusion, and identifies the crossover temperature where perturbative broad-band transport breaks down."

### WeisseFehske2008
**[WeisseFehske2008]**  
A. Weisse and H. Fehske,  
``Exact diagonalization techniques,''  
in \emph{Computational Many-Particle Physics}, Lecture Notes in Physics Vol. 739 (Springer, 2008), pp. 529--544.  

- **contribution:** chapter’s exact-diagonalization methods give compact classical ED references and iterative eigensolvers for validation in dynamic benchmarks
- **relation:** chapter’s exact-diagonalization methods give compact classical ED references and iterative eigensolvers for validation in dynamic benchmarks
- **abstract:**
  - > "In this chapter we show how to calculate a few eigenstates of the full Hamiltonian matrix of an interacting quantum system."
  - > "Naturally this implies that the Hilbert space of the problem has to be truncated, either by considering finite systems or by imposing suitable cut-offs."
  - > "All presented methods are iterative and repeatedly apply the Hamiltonian matrix to vectors in the Hilbert space."
  - > "A key outcome is that Lanczos-based sparse linear algebra enables practical access to low-energy spectra and correlators for interacting models."

### TroyerWiese2005
**[TroyerWiese2005]**  
M. Troyer and U.-J. Wiese,  
``Computational complexity and fundamental limitations to fermionic quantum Monte Carlo simulations,''  
\emph{Physical Review Letters} \textbf{94}, 170201 (2005).  

- **contribution:** Troyer’s variant again anchors the QMC complexity rationale for pursuing variational dynamic routes in difficult fermionic settings
- **relation:** Troyer’s variant again anchors the QMC complexity rationale for pursuing variational dynamic routes in difficult fermionic settings
- **abstract:**
  - > "Quantum Monte Carlo simulations, while being efficient for bosons, suffer from the "negative sign problem" when applied to fermions--causing an exponential increase of the computing time with the number of particles."
  - > "A polynomial time solution to the sign problem is highly desired since it would provide an unbiased and numerically exact method to simulate correlated quantum systems."
  - > "Here we show that such a solution is almost certainly unattainable by proving that the sign problem is nondeterministic polynomial (NP) hard, implying that a generic solution of the sign problem would also solve all problems in the complexity class NP in polynomial time."

### Schollwoeck2011
**[Schollwoeck2011]**  
U. Schollw\"ock,  
``The density-matrix renormalization group in the age of matrix product states,''  
\emph{Annals of Physics} \textbf{326}, 96--192 (2011).  

- **contribution:** The spelling variant maps to the same DMRG/MPS benchmark body used as a reference for trajectory and observable comparisons
- **relation:** The spelling variant maps to the same DMRG/MPS benchmark body used as a reference for trajectory and observable comparisons
- **abstract:**
  - > "The density-matrix renormalization group method (DMRG) has established itself over the last decade as the leading method for the simulation of the statics and dynamics of one-dimensional strongly correlated quantum lattice systems."
  - > "In the further development of the method, the realization that DMRG operates on a highly interesting class of quantum states, so-called matrix product states (MPS), has allowed a much deeper understanding of the inner structure of the DMRG method, its further potential and its limitations."
  - > "In this paper, a detailed exposition of current DMRG thinking in the MPS language is given to make implementation of MPS variants transparent, and several directions for future algorithmic developments are discussed."

### Macridin2018ElectronPhonon
**[Macridin2018ElectronPhonon]**  
A. Macridin, P. Spentzouris, J. Amundson, and R. Harnik,  
``Electron-phonon systems on a universal quantum computer,''  
\emph{Physical Review Letters} \textbf{121}, 110504 (2018).  

- **contribution:** Equivalent electron-phonon implementation reference for the McLachlan paper’s dynamic modeling and validation
- **relation:** Equivalent electron-phonon implementation reference for the McLachlan paper’s dynamic modeling and validation
- **abstract:**
  - > "We present an algorithm that extends existing quantum algorithms for simulating fermion systems in quantum chemistry and condensed matter physics to include bosons in general and phonons in particular."
  - > "We introduce a qubit representation for the low-energy subspace of phonons which allows an efficient simulation of the evolution operator of the electron-phonon systems."
  - > "As a consequence of the Nyquist-Shannon sampling theorem, the phonons are represented with exponential accuracy on a discretized Hilbert space with a size that increases linearly with the cutoff of the maximum phonon number."
  - > "The energy and the phonon number distribution of the polaron state agree with exact diagonalization results for weak, intermediate, and strong electron-phonon coupling regimes."

### Sawaya2020DLevel
**[Sawaya2020DLevel]**  
N. P. D. Sawaya et al.,  
``Resource-efficient digital quantum simulation of $d$-level systems for photonic, vibrational, and spin-$s$ Hamiltonians,''  
\emph{npj Quantum Information} \textbf{6}, 49 (2020).  

- **contribution:** Equivalent d-level encoding reference for dynamic trajectories and cost-sensitive bosonic truncation
- **relation:** Equivalent d-level encoding reference for dynamic trajectories and cost-sensitive bosonic truncation
- **abstract:**
  - > "Simulation of quantum systems is expected to be one of the most important applications of quantum computing, with much of the theoretical work so far having focused on fermionic and spin- $$\frac{1}{2}$$ 1 2 systems."
  - > "Here, we instead consider encodings of d -level (i.e., qudit) quantum operators into multi-qubit operators, studying resource requirements for approximating operator exponentials by Trotterization."
  - > "We primarily focus on spin- s and truncated bosonic operators in second quantization, observing desirable properties for approaches based on the Gray code, which to our knowledge has not been used in this context previously."
  - > "Results and procedures outlined in this work may be useful for simulating a broad class of Hamiltonians on qubit-based digital quantum computers."

### Li2023VBSElectronPhonon
**[Li2023VBSElectronPhonon]**  
W. Li et al.,  
``Efficient quantum simulation of electron-phonon systems by variational basis state encoder,''  
\emph{Physical Review Research} \textbf{5}, 023046 (2023).  

- **contribution:** Equivalent variational basis encoding reference for dynamic bosonic state representations
- **relation:** Equivalent variational basis encoding reference for dynamic bosonic state representations
- **abstract:**
  - > "Digital quantum simulation of electron-phonon systems requires truncating infinite phonon levels into $N$ basis states and then encoding them with qubit computational basis."
  - > "Unary encoding and binary encoding are the two most representative encoding schemes, which demand $O(N)$ and $O(logN)$ qubits as well as $O(N)$ and $O(NlogN)$ quantum gates, respectively."
  - > "In this paper, we propose a variational basis state encoding algorithm that reduces the scaling of the number of qubits and quantum gates to both $O(1)$ for systems obeying the area law of entanglement entropy."
  - > "Our approach paves the way for practical quantum simulation of electron-phonon systems on both near-term hardware and error-corrected quantum computers."

### McClean2017QSE
**[McClean2017QSE]**  
J. R. McClean, M. E. Kimchi-Schwartz, J. Carter, and W. A. de Jong,  
``Hybrid quantum-classical hierarchy for mitigation of decoherence and determination of excited states,''  
\emph{Physical Review A} \textbf{95}, 042308 (2017).  

- **contribution:** hybrid hierarchy introduces QSE-style excited-state correction and decoherence mitigation, matching the spectral branch
- **relation:** hybrid hierarchy introduces QSE-style excited-state correction and decoherence mitigation, matching the spectral branch
- **abstract:**
  - > "Using quantum devices supported by classical computational resources is a promising approach to quantum-enabled computation."
  - > "One powerful example of such a hybrid quantum-classical approach optimized for classically intractable eigenvalue problems is the variational quantum eigensolver, built to utilize quantum resources for the solution of eigenvalue problems and optimizations with minimal coherence time requirements by leveraging classical computational resources."
  - > "These algorithms have been placed as leaders among the candidates for the first to achieve supremacy over classical computation."
  - > "We demonstrate numerically on a sample electronic system that this method both allows for the accurate determination of excited electronic states as well as reduces the impact of decoherence, without using any additional quantum coherence time or formal error-correction codes."

### Colless2018QSE
**[Colless2018QSE]**  
J. I. Colless, V. V. Ramasesh, D. Dahlen, M. S. Blok, M. E. Kimchi-Schwartz, J. R. McClean, J. Carter, W. A. de Jong, and I. Siddiqi,  
``Computation of molecular spectra on a quantum processor with an error-resilient algorithm,''  
\emph{Physical Review X} \textbf{8}, 011021 (2018).  

- **contribution:** Colless demonstrates error-resilient spectral estimation workflows on near-term hardware for excited-state extraction
- **relation:** Colless demonstrates error-resilient spectral estimation workflows on near-term hardware for excited-state extraction
- **abstract:**
  - > "Harnessing the full power of nascent quantum processors requires the efficient management of a limited number of quantum bits with finite coherent lifetimes."
  - > "Hybrid algorithms, such as the variational quantum eigensolver (VQE), leverage classical resources to reduce the required number of quantum gates."
  - > "Experimental demonstrations of VQE have resulted in calculation of Hamiltonian ground states, and a new theoretical approach based on a quantum subspace expansion (QSE) has outlined a procedure for determining excited states that are central to dynamical processes."
  - > "Further, we show that this extended protocol can mitigate the effects of incoherent errors, potentially enabling larger-scale quantum simulations without the need for complex error-correction techniques."

### Higgott2019VQD
**[Higgott2019VQD]**  
O. Higgott, D. Wang, and S. Brierley,  
``Variational quantum computation of excited states,''  
\emph{Quantum} \textbf{3}, 156 (2019).  

- **contribution:** VQD provides excited-state variational structure, supporting the McLachlan spectral extraction alternatives
- **relation:** VQD provides excited-state variational structure, supporting the McLachlan spectral extraction alternatives
- **abstract:**
  - > "The calculation of excited state energies of electronic structure Hamiltonians has many important applications, such as the calculation of optical spectra and reaction rates."
  - > "While low-depth quantum algorithms, such as the variational quantum eigenvalue solver (VQE), have been used to determine ground state energies, methods for calculating excited states currently involve the implementation of high-depth controlled-unitaries or a large number of additional samples."
  - > "Here we show how overlap estimation can be used to deflate eigenstates once they are found, enabling the calculation of excited state energies and their degeneracies."
  - > "Our method is robust to control errors, is compatible with error-mitigation strategies and can be implemented on near-term quantum computers."

### Parrish2019MCVQE
**[Parrish2019MCVQE]**  
R. M. Parrish et al.,  
``Quantum computation of electronic transitions using a variational quantum eigensolver,''  
\emph{Physical Review Letters} \textbf{122}, 230401 (2019).  

- **contribution:** Parrish gives transition-targeted excited-state VQE workflows supporting Hamiltonian spectroscopy paths
- **relation:** Parrish gives transition-targeted excited-state VQE workflows supporting Hamiltonian spectroscopy paths
- **abstract:**
  - > "We develop an extension of the variational quantum eigensolver (VQE) algorithm-multistate contracted VQE (MC-VQE)-that allows for the efficient computation of the transition energies between the ground state and several low-lying excited states of a molecule, as well as the oscillator strengths associated with these transitions."
  - > "We numerically simulate MC-VQE by computing the absorption spectrum of an ab initio exciton model of an 18-chromophore light-harvesting complex from purple photosynthetic bacteria."

### Jones2019Spectra
**[Jones2019Spectra]**  
T. Jones, S. Endo, S. McArdle, X. Yuan, and S. C. Benjamin,  
``Variational quantum algorithms for discovering Hamiltonian spectra,''  
\emph{Physical Review A} \textbf{99}, 062304 (2019).  

- **contribution:** Jones contributes a variational route to Hamiltonian spectra, useful for comparison to the McLachlan-informed spectral stack
- **relation:** Jones contributes a variational route to Hamiltonian spectra, useful for comparison to the McLachlan-informed spectral stack
- **abstract:**
  - > "Calculating the energy spectrum of a quantum system is an important task, for example to analyze reaction rates in drug discovery and catalysis."
  - > "There has been significant progress in developing algorithms to calculate the ground state energy of molecules on near-term quantum computers."
  - > "However, calculating excited state energies has attracted comparatively less attention, and it is currently unclear what the optimal method is."
  - > "As our algorithm uses only low depth circuits and variational algorithms, it is suitable for use on near-term quantum hardware."

### Nakanishi2019SSVQE
**[Nakanishi2019SSVQE]**  
K. M. Nakanishi, K. Mitarai, and K. Fujii,  
``Subspace-search variational quantum eigensolver for excited states,''  
\emph{Physical Review Research} \textbf{1}, 033062 (2019).  

- **contribution:** Subspace-search excited-state strategy informs the multi-state management and state-targeting in spectral workflows
- **relation:** Subspace-search excited-state strategy informs the multi-state management and state-targeting in spectral workflows
- **abstract:**
  - > "The authors propose an algorithm, the subspace-search variational quantum eigensolver (SSVQE) that searches a low energy subspace by supplying orthogonal input states to the variational ansatz and relies on the unitarity of transformations to ensure the orthogonality of output states."
  - > "This work extends the applicable domain of the Variational Quantum Eigensolver to excited states and their related properties."

### Ollitrault2020qEOM
**[Ollitrault2020qEOM]**  
P. J. Ollitrault et al.,  
``Quantum equation of motion for computing molecular excitation energies on a noisy quantum processor,''  
\emph{Physical Review Research} \textbf{2}, 043140 (2020).  

- **contribution:** qEOM methods support practical noisy-device excited-state estimation and transition-energy pipelines in the third axis
- **relation:** qEOM methods support practical noisy-device excited-state estimation and transition-energy pipelines in the third axis
- **abstract:**
  - > "The computation of molecular excitation energies is essential for predicting photo-induced reactions of chemical and technological interest."
  - > "While the classical computing resources needed for this task scale poorly, quantum algorithms emerge as promising alternatives."
  - > "In particular, the extension of the variational quantum eigensolver algorithm to the computation of the excitation energies is an attractive option."
  - > "In particular, we demonstrate the efficiency of this approach in the calculation of the excitation energies of the LiH molecule on an IBM Quantum computer."

### AsthanaKumar2023QScEOM
**[AsthanaKumar2023QScEOM]**  
A. Asthana et al.,  
``Quantum self-consistent equation-of-motion method for computing molecular excitation energies, ionization potentials, and electron affinities on a quantum computer,''  
\emph{Chemical Science} \textbf{14}, 2405--2418 (2023).  

- **contribution:** QScEOM provides a unified framework for excitations, ionization, and affinities on noisy hardware, useful for the excited-state section
- **relation:** QScEOM provides a unified framework for excitations, ionization, and affinities on noisy hardware, useful for the excited-state section
- **abstract:**
  - > "We present a new hybrid quantum algorithm to estimate molecular excited and charged s"
  - > "tates on near-term quantum computers following any VQE-based ground state estimation."

### Grimsley2019ADAPT
**[Grimsley2019ADAPT]**  
H. R. Grimsley, S. E. Economou, E. Barnes, and N. J. Mayhall,  
``An adaptive variational algorithm for exact molecular simulations on a quantum computer,''  
\emph{Nature Communications} \textbf{10}, 3007 (2019).  

- **contribution:** ADAPT variant reused for adaptive generator reuse in dynamic spectral workflows, maintaining continuity with the static roots
- **relation:** ADAPT variant reused for adaptive generator reuse in dynamic spectral workflows, maintaining continuity with the static roots
- **abstract:**
  - > "Quantum simulation of chemical systems is one of the most promising near-term applications of quantum computers."
  - > "The variational quantum eigensolver, a leading algorithm for molecular simulations on quantum hardware, has a serious limitation in that it typically relies on a pre-selected wavefunction ansatz that results in approximate wavefunctions and energies."
  - > "Here we present an arbitrarily accurate variational algorithm that, instead of fixing an ansatz upfront, grows it systematically one operator at a time in a way dictated by the molecule being simulated."
  - > "Our results highlight the potential of our adaptive algorithm for exact simulations with present-day and near-term quantum hardware."

### Tang2021QubitADAPT
**[Tang2021QubitADAPT]**  
H. L. Tang et al.,  
``Qubit-ADAPT-VQE: An adaptive algorithm for constructing hardware-efficient ans\"atze on a quantum processor,''  
\emph{PRX Quantum} \textbf{2}, 020310 (2021).  

- **contribution:** Qubit-ADAPT variant for dynamic contexts preserves a stable, hardware-friendly generator style in McLachlan checkpoints
- **relation:** Qubit-ADAPT variant for dynamic contexts preserves a stable, hardware-friendly generator style in McLachlan checkpoints
- **abstract:**
  - > "The resources required to run a high-accuracy variational quantum eigensolver algorithm with a dynamically"
  - > "created ansatz are quantified and reduced significantly, easing the quantum simulation of many-body systems."

### Yordanov2021QEB
**[Yordanov2021QEB]**  
Y. S. Yordanov, V. Armaos, C. H. W. Barnes, and D. R. M. Arvidsson-Shukur,  
``Qubit-excitation-based adaptive variational quantum eigensolver,''  
\emph{Communications Physics} \textbf{4}, 228 (2021).  

- **contribution:** Qubit-excitation ADAPT variant for dynamic updates and low-overhead move sets in time-evolution control
- **relation:** Qubit-excitation ADAPT variant for dynamic updates and low-overhead move sets in time-evolution control
- **abstract:**
  - > "Molecular simulations with the variational quantum eigensolver (VQE) are a promising application for emerging noisy intermediate-scale quantum computers."
  - > "Constructing accurate molecular ansätze that are easy to optimize and implemented by shallow quantum circuits is crucial for the successful implementation of such simulations."
  - > "Ansätze are, generally, constructed as series of fermionic-excitation evolutions."
  - > "In terms of circuit efficiency and convergence speed, we demonstrate that the QEB-ADAPT-VQE outperforms the qubit-ADAPT-VQE, which to our knowledge was the previous most circuit-efficient scalable VQE protocol for molecular simulations."

### Anastasiou2024TETRIS
**[Anastasiou2024TETRIS]**  
P. G. Anastasiou, Y. Chen, N. J. Mayhall, E. Barnes, and S. E. Economou,  
``TETRIS-ADAPT-VQE: An adaptive algorithm that yields shallower, denser circuit ans\"atze,''  
\emph{Physical Review Research} \textbf{6}, 013254 (2024).  

- **contribution:** TETRIS-style grouped operator growth supports dense/fast trajectory candidates in adaptive evolution
- **relation:** TETRIS-style grouped operator growth supports dense/fast trajectory candidates in adaptive evolution
- **abstract:**
  - > "Adaptive quantum variational algorithms are particularly promising for simulating strongly correlated systems on near-term quantum hardware, but they are not yet viable due, in large part, to the severe coherence time limitations on current devices."
  - > "In this paper, we introduce an algorithm called TETRIS-ADAPT-VQE (tiling efficient trial circuits with rotations implemented simultaneously adaptive derivative-assembled problem-tailored variational quantum eigensolver), which iteratively builds up variational a few operators at a time in a way dictated by the problem being simulated."
  - > "This algorithm is a modified version of the ADAPT-VQE algorithm, in which the one-operator-at-a-time rule is lifted to allow for the addition of multiple operators with disjoint supports in each iteration."
  - > "Published by the American Physical Society 2024"

### Ramoa2025CEO
**[Ramoa2025CEO]**  
M. Ram\^oa et al.,  
``Reducing the resources required by ADAPT-VQE using coupled exchange operators and improved subroutines,''  
\emph{npj Quantum Information} \textbf{11}, 86 (2025).  

- **contribution:** Coupled-exchange operator pool concepts support efficient repeated operator selection in dynamic controller steps
- **relation:** Coupled-exchange operator pool concepts support efficient repeated operator selection in dynamic controller steps
- **abstract:**
  - > "Adaptive variational quantum algorithms arguably offer the best prospects for quantum advantage in the Noisy Intermediate-Scale Quantum era."
  - > "Since the inception of the first such algorithm, the Adaptive Derivative-Assembled Problem-Tailored Variational Quantum Eigensolver (ADAPT-VQE), many improvements have appeared in the literature."
  - > "We combine the key improvements along with a novel operator pool—which we term Coupled Exchange Operator (CEO) pool—to assess the cost of running state-of-the-art ADAPT-VQE on hardware in terms of measurement counts and circuit depth."
  - > "We also find that our state-of-the-art CEO-ADAPT-VQE outperforms the Unitary Coupled Cluster Singles and Doubles ansatz, the most widely used static VQE ansatz, in all relevant metrics, and offers a five order of magnitude decrease in measurement costs as compared to other static ansätze with competitive CNOT counts."

### Sohail2026GeoADAPT
**[Sohail2026GeoADAPT]**  
M. A. Sohail and T. Koike-Akino,  
``Geo-ADAPT-VQE: Quantum information metric-aware circuit optimization for quantum chemistry,''  
arXiv:2603.10325 (2026).  

- **contribution:** Geo-ADAPT adds geometry-aware selection and conditioning ideas suitable for McLachlan checkpoint robustness
- **relation:** Geo-ADAPT adds geometry-aware selection and conditioning ideas suitable for McLachlan checkpoint robustness
- **abstract:**
  - > "Adaptive ansatz construction has emerged as a powerful technique for reducing circuit depth and improving optimization efficiency in variational quantum eigensolvers."
  - > "Existing adaptive methods, including ADAPT-VQE, rely solely on first-order gradients and ignore the underlying geometry of the quantum state space, limiting both convergence behavior and operator-selection efficiency."
  - > "Geo-ADAPT-VQE selects operators from a pool using the natural gradient rule, guiding the ansatz to grow along directions aligned with the underlying quantum-state geometry."
  - > "This improves convergence, reduces susceptibility to shallow local minima and saddle points, and can yield significantly shorter ans"atze."

### Barison2021PVQD
**[Barison2021PVQD]**  
S. Barison, F. Vicentini, and G. Carleo,  
``An efficient quantum algorithm for the time evolution of parameterized circuits,''  
\emph{Quantum} \textbf{5}, 512 (2021).  

- **contribution:** PVQD algorithmic framing supports practical parameterized-circuit time evolution in the dynamic section
- **relation:** PVQD algorithmic framing supports practical parameterized-circuit time evolution in the dynamic section
- **abstract:**
  - > "We introduce a novel hybrid algorithm to simulate the real-time evolution of quantum systems using parameterized quantum circuits."
  - > "The method, named "projected – Variational Quantum Dynamics" (p-VQD) realizes an iterative, global projection of the exact time evolution onto the parameterized manifold."
  - > "In the small time-step limit, this is equivalent to the McLachlan's variational principle."
  - > "Through numerical experiments, we also show that our approach is particularly advantageous over existing global optimization algorithms based on the time-dependent variational principle that, due to a demanding quadratic scaling with parameter numbers, are unsuitable for large parameterized quantum circuits."

### Yao2021AVQDS
**[Yao2021AVQDS]**  
Y.-X. Yao, N. Gomes, F. Zhang, C.-Z. Wang, K.-M. Ho, T. Iadecola, and P. P. Orth,  
``Adaptive variational quantum dynamics simulations,''  
\emph{PRX Quantum} \textbf{2}, 030307 (2021).  

- **contribution:** AVQDS contributes adaptive control of McLachlan distance and ansatz growth
- **relation:** AVQDS contributes adaptive control of McLachlan distance and ansatz growth
- **abstract:**
  - > "We propose a general-purpose, self-adaptive approach to construct a variational wave-function ansatz for highly accurate quantum dynamics simulations based on McLachlan's variational principle."
  - > "The key idea is to dynamically expand the variational ansatz along the time-evolution path such that the "McLachlan distance", which is a measure of the simulation accuracy, remains below a set threshold."
  - > "We apply this adaptive variational quantum dynamics simulation (AVQDS) approach to the integrable Lieb-Schultz-Mattis spin chain and the nonintegrable mixed-field Ising model, where it captures both finite-rate and sudden post-quench dynamics with high fidelity."
  - > "We envision that a wide range of dynamical simulations of quantum many-body systems on near-term quantum-computing devices will be made possible through the AVQDS framework."

### Linteau2024APVQD
**[Linteau2024APVQD]**  
D. Linteau, S. Barison, N. H. Lindner, and G. Carleo,  
``Adaptive projected variational quantum dynamics,''  
\emph{Physical Review Research} \textbf{6}, 023130 (2024).  

- **contribution:** Adaptive projected dynamics aligns with the projected-evolution controller and trajectory maintenance logic
- **relation:** Adaptive projected dynamics aligns with the projected-evolution controller and trajectory maintenance logic
- **abstract:**
  - > "We propose an adaptive quantum algorithm to prepare accurate variational time evolved wave functions."
  - > "The method is based on the projected variational quantum dynamics (pVQD) algorithm, that performs a global optimization with linear scaling in the number of variational parameters."
  - > "Instead of fixing a variational ansatz at the beginning of the simulation, the circuit is grown systematically during the time evolution."
  - > "Published by the American Physical Society 2024"

### Zhang2025CompressedAVQD
**[Zhang2025CompressedAVQD]**  
F. Zhang, C.-Z. Wang, T. Iadecola, P. P. Orth, and Y.-X. Yao,  
``Adaptive variational quantum dynamics simulations with compressed circuits and fewer measurements,''  
\emph{Physical Review B} \textbf{111}, 094310 (2025).  

- **contribution:** Compressed AVQD informs the reduced-measurement policy and low-cost evolution variants
- **relation:** Compressed AVQD informs the reduced-measurement policy and low-cost evolution variants
- **abstract:**
  - > "The adaptive variational quantum dynamics simulation (AVQDS) method performs real-time evolution of quantum states using automatically generated parametrized quantum circuits that often contain substantially fewer gates than Trotter circuits."
  - > "Here we report an improved version of the method, which we call AVQDS(T), by porting the tiling efficient trial circuits with rotations implemented simultaneously technique."
  - > "The algorithm adaptively adds layers of disjoint unitary gates to the ansatz circuit so as to keep the McLachlan distance, a measure of the accuracy of the variational dynamics, below a fixed threshold."
  - > "We showcase that this approach enables AVQDS(T) to deliver more accurate results than simulations using a fixed ansatz of comparable final depth for a significant time duration with fewer quantum resources."

### ProvostVallee1980
**[ProvostVallee1980]**  
J. P. Provost and G. Vall\'ee,  
``Riemannian structure on manifolds of quantum states,''  
\emph{Communications in Mathematical Physics} \textbf{76}, 289--301 (1980).  

- **contribution:** Provost-Vallée provides the geometric foundation for metric-aware projections and distance measures used throughout McLachlan updates
- **relation:** Provost-Vallée provides the geometric foundation for metric-aware projections and distance measures used throughout McLachlan updates
- **abstract:**
  - > "A metric tensor is defined from the underlying Hilbert space structure for any submanifold of quantum states."
  - > "The case where the manifold is generated by the action of a Lie group on a fixed state vector (generalized coherent states manifold hereafter noted G.C.S.M.) is studied in details; the geometrical properties of some well-known G.C.S.M."
  - > "are reviewed and an explicit expression for the scalar Riemannian curvature is given in the general case."
  - > "It is shown on examples that the distance between nearby states is related to quantum fluctuations; in the particular case of the harmonic oscillator group the condition of zero curvature is identical to the condition of non-dispersion of wave packets."

### Verteletskyi2020MeasurementGrouping
**[Verteletskyi2020MeasurementGrouping]**  
V. Verteletskyi, T.-C. Yen, and A. F. Izmaylov,  
``Measurement optimization in the variational quantum eigensolver using a minimum clique cover,''  
\emph{Journal of Chemical Physics} \textbf{152}, 124114 (2020).  

- **contribution:** Verteletskyi’s minimum-clique-grouping logic reduces measurement burden in both evolving trajectories and subspace post-processing
- **relation:** Verteletskyi’s minimum-clique-grouping logic reduces measurement burden in both evolving trajectories and subspace post-processing
- **abstract:**
  - > "Solving the electronic structure problem using the Variational Quantum Eigensolver (VQE) technique involves the measurement of the Hamiltonian expectation value."
  - > "The current hardware can perform only projective single-qubit measurements, and thus, the Hamiltonian expectation value is obtained by measuring parts of the Hamiltonian rather than the full Hamiltonian."
  - > "This restriction makes the measurement process inefficient because the number of terms in the Hamiltonian grows as O(N4) with the size of the system, N."
  - > "On average, grouping qubit-wise commuting terms reduced the number of operators to measure three times less compared to the total number of terms in the considered Hamiltonians."

### Urbanek2020VQSE
**[Urbanek2020VQSE]**  
M. Urbanek, D. Camps, R. Van Beeumen, and W. A. de Jong,  
``Chemistry on quantum computers with virtual quantum subspace expansion,''  
\emph{Journal of Chemical Theory and Computation} \textbf{16}, 5425--5431 (2020).  

- **contribution:** Virtual QSE contributes subspace-expansion infrastructure for excited-state corrections that can be layered after dynamic propagation
- **relation:** Virtual QSE contributes subspace-expansion infrastructure for excited-state corrections that can be layered after dynamic propagation
- **abstract:**
  - > "Simulating chemical systems on quantum computers has been limited to a few electrons in a minimal basis."
  - > "We demonstrate experimentally that the virtual quantum subspace expansion (Takeshita, T.; <i>Phys."
  - > "Rev."
  - > "In addition, we were able to obtain an accurate potential energy curve for the nitrogen dimer in a quantum simulation on a classical computer."

### Yoshioka2022GQSE
**[Yoshioka2022GQSE]**  
N. Yoshioka, H. Hakoshima, Y. Matsuzaki, Y. Tokunaga, Y. Suzuki, and S. Endo,  
``Generalized quantum subspace expansion,''  
\emph{Physical Review Letters} \textbf{129}, 020502 (2022).  

- **contribution:** Generalized QSE extends subspace construction to richer Hamiltonians, supporting robust excited-state estimation in dynamic contexts
- **relation:** Generalized QSE extends subspace construction to richer Hamiltonians, supporting robust excited-state estimation in dynamic contexts
- **abstract:**
  - > "One of the major challenges for erroneous quantum computers is undoubtedly the control over the effect of noise."
  - > "Considering the rapid growth of available quantum resources that are not fully fault tolerant, it is crucial to develop practical hardware-friendly quantum error mitigation (QEM) techniques to suppress unwanted errors."
  - > "Here, we propose a novel generalized quantum subspace expansion method which can handle stochastic, coherent, and algorithmic errors in quantum computers."
  - > "We numerically demonstrate in both situations that we can suppress errors by orders of magnitude, and show that our protocol inherits the advantages of previous error-agnostic QEM techniques as well as overcoming their drawbacks."

### CortesGray2022QKSD
**[CortesGray2022QKSD]**  
C. L. Cortes and S. K. Gray,  
``Quantum Krylov subspace algorithms for ground- and excited-state energy estimation,''  
\emph{Physical Review A} \textbf{105}, 022417 (2022).  

- **contribution:** QKSD framework proposes Krylov-space excited-state estimation strategies, relevant as an optional spectral branch for the time-dependent results
- **relation:** QKSD framework proposes Krylov-space excited-state estimation strategies, relevant as an optional spectral branch for the time-dependent results
- **abstract:**
  - > "Quantum Krylov subspace diagonalization (QKSD) algorithms provide a low-cost alternative to the conventional quantum phase estimation algorithm for estimating ground- and excited-state energies of a quantum many-body system."
  - > "While QKSD algorithms typically rely on the Hadamard test for estimating Krylov subspace matrix elements of the form (phi(i)|e^{-i H \tau}|phi(j)), these circuits can require expensive controlled multiqubit gates on near-term hardware."
  - > "The paper shows that, for a wide class of Hamiltonians with relevant symmetries, alternative ancilla-economical approaches can be used."
  - > "It proposes a multifidelity estimation protocol and develops a unified framework with three quantum-classical algorithms balancing circuit calls, depth, classical complexity, and generalized-eigenvalue stability."

### Motta2024SubspaceReview
**[Motta2024SubspaceReview]**  
M. Motta et al.,  
``Subspace methods for electronic structure simulations on quantum computers,''  
\emph{Electronic Structure} \textbf{6}, 013001 (2024).  

- **contribution:** Motta provides a broader subspace-method synthesis useful for positioning and benchmarking the spectral post-processing choices
- **relation:** Motta provides a broader subspace-method synthesis useful for positioning and benchmarking the spectral post-processing choices
- **abstract:**
  - > "Quantum subspace methods (QSMs) are a class of quantum computing algorithms where the time-independent Schrödinger equation for a quantum system is projected onto a subspace of the underlying Hilbert space."
  - > "This projection transforms the Schrödinger equation into an eigenvalue problem determined by measurements carried out on a quantum device."
  - > "The eigenvalue problem is then solved on a classical computer, yielding approximations to ground- and excited-state energies and wavefunctions."
  - > "We present the theoretical foundations and applications of QSMs, and we discuss their implementation on quantum hardware, illustrating the impact of noise on their performance."

### Epperly2022QSD
**[Epperly2022QSD]**  
E. N. Epperly, L. Lin, and Y. Nakatsukasa,  
``A theory of quantum subspace diagonalization,''  
\emph{SIAM Journal on Matrix Analysis and Applications} \textbf{43}, 1263--1290 (2022).  

- **contribution:** QSD formalism gives the linear-algebraic basis for stable subspace diagonalization used in the excited pipeline
- **relation:** QSD formalism gives the linear-algebraic basis for stable subspace diagonalization used in the excited pipeline
- **abstract:**
  - > "Quantum subspace diagonalization methods are an exciting new class of algorithms for solving large-scale eigenvalue problems using quantum computers."
  - > "Unfortunately, these methods require the solution of an ill-conditioned generalized eigenvalue problem, with a matrix pencil corrupted by a nonnegligible amount of noise that is far above the machine precision."
  - > "Despite pessimistic predictions from classical worst-case perturbation theories, these methods can perform reliably well if the generalized eigenvalue problem is solved using a standard truncation strategy."
  - > "Our new results can also be of independent interest to solving eigenvalue problems outside the context of quantum computation."

### Lee2024SamplingQKSD
**[Lee2024SamplingQKSD]**  
G. Lee, D. Lee, and J. Huh,  
``Sampling error analysis in quantum Krylov subspace diagonalization,''  
\emph{Quantum} \textbf{8}, 1477 (2024).  

- **contribution:** Sampling-error analysis in QKSD guides finite-shot confidence and estimator design in the spectral reconstructions
- **relation:** Sampling-error analysis in QKSD guides finite-shot confidence and estimator design in the spectral reconstructions
- **abstract:**
  - > "Quantum Krylov subspace diagonalization (QKSD) is an emerging method used in place of quantum phase estimation in the early fault-tolerant era, where limited quantum circuit depth is available."
  - > "In contrast to the classical Krylov subspace diagonalization (KSD) or the Lanczos method, QKSD exploits the quantum computer to efficiently estimate the eigenvalues of large-size Hamiltonians through a faster Krylov projection."
  - > "However, unlike classical KSD, which is solely concerned with machine precision, QKSD is inherently accompanied by errors originating from a finite number of samples."
  - > "Numerical simulations of the one-dimensional Hubbard model demonstrate that the error bound of finite samplings accurately predicts the experimental errors in well-conditioned regions."

### Getelina2024HardwareNoiseQSE
**[Getelina2024HardwareNoiseQSE]**  
J. C. Getelina, P. Sharma, T. Iadecola, P. P. Orth, and Y.-X. Yao,  
``Quantum subspace expansion in the presence of hardware noise,''  
\emph{APL Quantum} \textbf{1}, 036127 (2024).  

- **contribution:** Getelina studies hardware-noise impacts and mitigation in QSE, relevant for the realism-aware excited-state checks
- **relation:** Getelina studies hardware-noise impacts and mitigation in QSE, relevant for the realism-aware excited-state checks
- **abstract:**
  - > "Finding ground state energies on current quantum processing units (QPUs) using algorithms such as the variational quantum eigensolver (VQE) continues to pose challenges."
  - > "Hardware noise severely affects both the expressivity and trainability of parameterized quantum circuits, limiting them to shallow depths in practice."
  - > "Here, we demonstrate that both issues can be addressed by synergistically integrating VQE with a quantum subspace expansion, allowing for an optimal balance between quantum and classical computing capabilities and costs."
  - > "Finally, we show how to systematically approach the exact solution by performing controlled quantum error mitigation based on probabilistic error reduction on the noisy backend fake_guadalupe."

### OLeary2025PartitionedQSE
**[OLeary2025PartitionedQSE]**  
T. O'Leary, L. W. Anderson, D. Jaksch, and M. Kiffner,  
``Partitioned quantum subspace expansion,''  
\emph{Quantum} \textbf{9}, 1726 (2025).  

- **contribution:** Partitioned QSE adds scalable decomposition ideas for large Hilbert subspaces in spectral workflows
- **relation:** Partitioned QSE adds scalable decomposition ideas for large Hilbert subspaces in spectral workflows
- **abstract:**
  - > "We present an iterative generalisation of the quantum subspace expansion algorithm used with a Krylov basis."
  - > "The iterative construction connects a sequence of subspaces via their lowest energy states."
  - > "Diagonalising a Hamiltonian in a given Krylov subspace requires the same quantum resources in both the single step and sequential cases."
  - > "Our work suggests that the numerical instability limiting the accuracy of this approach can be substantially alleviated in a parameter-free way."

### Boyd2025ShadowSubspace
**[Boyd2025ShadowSubspace]**  
G. Boyd, B. Koczor, and Z. Cai,  
``High-dimensional subspace expansion using classical shadows,''  
\emph{Physical Review A} \textbf{111}, 022423 (2025).  

- **contribution:** Boyd demonstrates classical-shadow QSE scaling for high-dimensional settings, informing shot-efficient subspace methods
- **relation:** Boyd demonstrates classical-shadow QSE scaling for high-dimensional settings, informing shot-efficient subspace methods
- **abstract:**
  - > "We introduce a postprocessing technique for classical shadow measurement data that enhances the precision of ground state estimation through high-dimensional subspace expansion; the dimensionality is only limited by the amount of classical postprocessing resources rather than by quantum resources."
  - > "Crucial steps of our approach are the efficient identification of useful observables from shadow data, followed by our regularized subspace expansion that is designed to be numerically stable even when using noisy data."
  - > "We analytically investigate noise propagation within our method, and upper bound the statistical fluctuations due to the limited number of snapshots in classical shadows."
  - > "Furthermore, performance is guaranteed to be at least as good—and in many cases better—than direct energy estimation without using additional quantum resources, and the approach is thus a very natural alternative for estimating ground state energies directly from classical shadow data."

### Kwao2026GEPQSE
**[Kwao2026GEPQSE]**  
P. F. Kwao, S. P. Sundar, B. Gupt, and A. Asthana,  
``Generalized eigenvalue problem in subspace-based excited-state methods for quantum computers,''  
\emph{Journal of Chemical Theory and Computation} \textbf{22}, 2892--2903 (2026).  

- **contribution:** Generalized eigenvalue problem framing strengthens stability and interpretability of the excited-state postprocessing
- **relation:** Generalized eigenvalue problem framing strengthens stability and interpretability of the excited-state postprocessing
- **abstract:**
  - > "Solving challenging problems in quantum chemistry is one of the most promising applications of quantum computers."
  - > "Within the quantum algorithms proposed for problems in excited-state quantum chemistry, subspace-based quantum algorithms, including quantum subspace expansion (QSE), quantum equation of motion (qEOM), and quantum self-consistent equation-of-motion (q-sc-EOM), are promising for pre-fault-tolerant quantum devices."
  - > "The working equation of QSE and qEOM requires solving a generalized eigenvalue equation with associated matrix elements measured on a quantum computer."
  - > "We also show that excited-state methods that have an eigenvalue equation as the working equation, such as q-sc-EOM, do not have the problems associated with the condition number and could be generally more stable to errors and therefore more suitable candidates for excited-state quantum chemistry calculations using quantum computers."

### ChoiIzmaylov2023MeasurementQSE
**[ChoiIzmaylov2023MeasurementQSE]**  
S. Choi and A. F. Izmaylov,  
``Measurement optimization techniques for excited electronic states in near-term quantum computing algorithms,''  
\emph{Journal of Chemical Theory and Computation} \textbf{19}, 3184--3193 (2023).  

- **contribution:** optimizes measurement for excited-state QSE, directly supporting the measurable objective design
- **relation:** optimizes measurement for excited-state QSE, directly supporting the measurable objective design
- **abstract:**
  - > "The variational quantum eigensolver (VQE) remains one of the most popular near-term quantum algorithms for solving the electronic structure problem."
  - > "Yet, for its practicality, the main challenge to overcome is improving the quantum measurement efficiency."
  - > "Numerous quantum measurement techniques have been developed recently, but it is unclear how these state-of-the-art measurement techniques will perform in extensions of VQE for obtaining excited electronic states."
  - > "Nevertheless, when the best possible measurement technique for each excited state VQE algorithm is considered, significantly fewer measurements are required in multistate contraction than in quantum subspace expansion."

### Nakamura2024AdaptiveMeasurementQSE
**[Nakamura2024AdaptiveMeasurementQSE]**  
Y. Nakamura, Y. Yano, and N. Yoshioka,  
``Adaptive measurement strategy for quantum subspace methods,''  
\emph{New Journal of Physics} \textbf{26}, 033028 (2024).  

- **contribution:** Nakamura introduces adaptive measurement scheduling for subspace tasks, aligned with finite-shot McLachlan/controller constraints
- **relation:** Nakamura introduces adaptive measurement scheduling for subspace tasks, aligned with finite-shot McLachlan/controller constraints
- **abstract:**
  - > "Estimation of physical observables for unknown quantum states is an important problem that underlies a wide range of fields, including quantum information processing, quantum physics, and quantum chemistry."
  - > "In the context of quantum computation, in particular, existing studies have mainly focused on holistic state tomography or estimation on specific observables with known classical descriptions, while this lacks the important class of problems where the estimation target itself relies on the measurement outcome."
  - > "In this work, we propose an adaptive measurement optimization method that is useful for the quantum subspace methods, namely the variational simulation methods that utilize classical postprocessing on measurement outcomes."
  - > "Our work reveals that the potential of the QSE method can be empowered by elaborated measurement protocols, and opens a path to further pursue efficient quantum measurement techniques in practical computations."

### Jamet2022GreenQSE
**[Jamet2022GreenQSE]**  
F. Jamet, A. Agarwal, and I. Rungger,  
``Quantum subspace expansion algorithm for Green's functions,''  
arXiv:2205.00094 (2022).  

- **contribution:** Jamet extends QSE to Green’s functions, supporting response-function observables from dynamic trajectories
- **relation:** Jamet extends QSE to Green’s functions, supporting response-function observables from dynamic trajectories
- **abstract:**
  - > "One of the major challenges for erroneous quantum computers is undoubtedly the control over the effect of noise."
  - > "Considering the rapid growth of available quantum resources that are not fully fault tolerant, it is crucial to develop practical hardware-friendly quantum error mitigation (QEM) techniques to suppress unwanted errors."
  - > "Here, we propose a novel generalized quantum subspace expansion method which can handle stochastic, coherent, and algorithmic errors in quantum computers."
  - > "We numerically demonstrate in both situations that we can suppress errors by orders of magnitude, and show that our protocol inherits the advantages of previous error-agnostic QEM techniques as well as overcoming their drawbacks."

### Dhawan2024ImagTimeGreen
**[Dhawan2024ImagTimeGreen]**  
D. Dhawan, D. Zgid, and M. Motta,  
``Quantum algorithm for imaginary-time Green's functions,''  
\emph{Journal of Chemical Theory and Computation} \textbf{20}, 4629--4638 (2024).  

- **contribution:** Dhawan develops imaginary-time Green-function algorithms, relevant for response/correlation validation beyond raw energies
- **relation:** Dhawan develops imaginary-time Green-function algorithms, relevant for response/correlation validation beyond raw energies
- **abstract:**
  - > "Green's function methods lead to ab initio, systematically improvable simulations of molecules and materials while providing access to multiple experimentally observable properties such as the density of states and the spectral function."
  - > "The calculation of the exact one-particle Green's function remains a significant challenge for classical computers and was attempted only on very small systems."
  - > "Here, we present a hybrid quantum-classical algorithm to calculate the imaginary-time one-particle Green's function."
  - > "We demonstrate the validity of this algorithm by simulating H<sub>2</sub> and H<sub>4</sub> on quantum simulators and on IBM's quantum devices."

### Gauthier2024OccupationQSE
**[Gauthier2024OccupationQSE]**  
B. Gauthier, P. Rosenberg, A. Foley, and M. Charlebois,  
``Occupation-number quantum-subspace-expansion approach to computing the single-particle Green function: An opportunity for noise filtering,''  
\emph{Physical Review A} \textbf{110}, 032624 (2024).  

- **contribution:** Gauthier brings occupation-number expansion and noise filtering for QSE observables, supporting reliability in noisy trajectory outputs
- **relation:** Gauthier brings occupation-number expansion and noise filtering for QSE observables, supporting reliability in noisy trajectory outputs
- **abstract:**
  - > "We introduce a hybrid quantum-classical algorithm to compute the Green function for strongly correlated electrons on noisy intermediate-scale quantum (NISQ) devices."
  - > "The technique consists in the construction of a nonorthogonal excitation basis composed of a set of single-particle excitations generated by occupation-number operators."
  - > "The excited sectors of the Hamiltonian in this basis can then be measured on the quantum device and a classical postprocessing procedure yields the Green function in the Lehmann representation."
  - > "A simulation of a four-site system carried out on classical hardware suggests that the approach can achieve similar accuracy for larger systems."

### Kokcu2024LinearResponse
**[Kokcu2024LinearResponse]**  
E. K\"okc\"u, H. A. Labib, J. K. Freericks, and A. F. Kemper,  
``A linear response framework for simulating bosonic and fermionic correlation functions illustrated on quantum computers,''  
\emph{Nature Communications} \textbf{15}, 3881 (2024).  

- **contribution:** Kokcu establishes linear-response QSE frameworks for bosonic and fermionic correlations, informing the response-type checks
- **relation:** Kokcu establishes linear-response QSE frameworks for bosonic and fermionic correlations, informing the response-type checks
- **abstract:**
  - > "Response functions are a fundamental aspect of physics; they represent the link between experimental observations and the underlying quantum many-body state."
  - > "However, this link is often under-appreciated, as the Lehmann formalism for obtaining response functions in linear response has no direct link to experiment."
  - > "Within the context of quantum computing, and via a linear response framework, we restore this link by making the experiment an inextricable part of the quantum simulation."
  - > "The linear response method provides a robust framework for using quantum computers to study systems in physics and chemistry."

### Umeano2025KitaevQSE
**[Umeano2025KitaevQSE]**  
C. Umeano, F. Jamet, L. P. Lindoy, I. Rungger, and O. Kyriienko,  
``Quantum subspace expansion approach for simulating dynamical response functions of Kitaev spin liquids,''  
\emph{Physical Review Materials} \textbf{9}, 034401 (2025).  

- **contribution:** Umeano demonstrates QSE-based dynamical-response workflows on complex spin systems, supporting generality of the spectral branch
- **relation:** Umeano demonstrates QSE-based dynamical-response workflows on complex spin systems, supporting generality of the spectral branch
- **abstract:**
  - > "We develop a quantum simulation-based approach for studying properties of strongly correlated magnetic materials at increasing scale."
  - > "We consider a paradigmatic example of a quantum spin liquid (QSL) state hosted by the honeycomb Kitaev model and use a trainable, symmetry-guided ansatz for preparing its ground state."
  - > "Applying the tools of quantum subspace expansion (QSE), Hamiltonian operator approximation, and overlap measurements, we simulate the QSL at zero temperature and finite magnetic field, thus moving outside of the symmetric subspace."
  - > "Our results show that quantum simulators offer an insight into quasiparticle properties of strongly correlated magnets and can become a valuable tool for studying material science."

### Backes2023HHDMFT
**[Backes2023HHDMFT]**  
S. Backes, Y. Murakami, S. Sakai, and R. Arita,  
``Dynamical mean-field theory for the Hubbard-Holstein model on a quantum device,''  
\emph{Physical Review B} \textbf{107}, 165155 (2023).  

- **contribution:** Backes applies DMFT to Hubbard-Holstein on quantum hardware, giving advanced dynamic-context precedent and benchmarking context
- **relation:** Backes applies DMFT to Hubbard-Holstein on quantum hardware, giving advanced dynamic-context precedent and benchmarking context
- **abstract:**
  - > "Utilizing quantum computers for real materials calculations is highly promising, but hinges on qubit fidelity and fault-tolerant quantum algorithms."
  - > "Here, the authors show that a representative model for real materials, the Hubbard-Holstein model, which describes interacting electrons on a lattice coupled to bosonic degrees of freedom, can be reliably investigated using today's Quantum processors."
  - > "Adapting state-of-the-art mathematical tools is shown to be critical to overcome limitations imposed by the noise level of current quantum computing devices."

### Denner2023HybridEP
**[Denner2023HybridEP]**  
M. M. Denner et al.,  
``A hybrid quantum-classical method for electron-phonon systems,''  
\emph{Communications Physics} \textbf{6}, 233 (2023).  

- **contribution:** Denner2023HybridEP gives a hybrid protocol for electron-phonon dynamics, useful as an external benchmark line
- **relation:** Denner2023HybridEP gives a hybrid protocol for electron-phonon dynamics, useful as an external benchmark line
- **abstract:**
  - > "Interactions between electrons and phonons play a crucial role in quantum materials."
  - > "Yet, there is no universal method that would simultaneously accurately account for strong electron-phonon interactions and electronic correlations."
  - > "By combining methods of the variational quantum eigensolver and the variational non-Gaussian solver, we develop a hybrid quantum-classical algorithm suitable for this type of correlated systems."
  - > "We benchmark our method by applying it to the paradigmatic Hubbard-Holstein model at half filling, and show that it correctly captures the competition between charge density wave and antiferromagnetic phases, quantitatively consistent with exact diagonalization."

### Dutta2024BosonicChemistry
**[Dutta2024BosonicChemistry]**  
R. Dutta et al.,  
``Simulating chemistry on bosonic quantum devices,''  
\emph{Journal of Chemical Theory and Computation} \textbf{20}, 6426--6441 (2024).  

- **contribution:** bosonic-device chemistry survey informs architecture-level assumptions and method choices for phonon-heavy dynamics
- **relation:** bosonic-device chemistry survey informs architecture-level assumptions and method choices for phonon-heavy dynamics
- **abstract:**
  - > "Bosonic quantum devices offer a novel approach to realize quantum computations, where the quantum two-level system (<i>qubit</i>) is replaced with the quantum (an)harmonic oscillator (<i>qumode</i>) as the fundamental building block of the quantum simulator."
  - > "The simulation of chemical structure and dynamics can then be achieved by representing or mapping the system Hamiltonians in terms of bosonic operators."
  - > "In this Perspective, we review recent progress and future potential of using bosonic quantum devices for addressing a wide range of challenging chemical problems, including the calculation of molecular vibronic spectra, the simulation of gas-phase and solution-phase adiabatic and nonadiabatic chemical dynamics, the efficient solution of molecular graph theory problems, and the calculations of electronic structure."

### Peng2025BosonHamiltonians
**[Peng2025BosonHamiltonians]**  
B. Peng, Y. Su, D. Claudino, K. Kowalski, G. H. Low, and M. Roetteler,  
``Quantum simulation of boson-related Hamiltonians: Techniques, effective Hamiltonian construction, and error analysis,''  
\emph{Quantum Science and Technology} \textbf{10}, 023002 (2025).  

- **contribution:** Peng provides bosonic Hamiltonian encoding and error-analysis guidance for algorithmic choices in dynamic McLachlan tests
- **relation:** Peng provides bosonic Hamiltonian encoding and error-analysis guidance for algorithmic choices in dynamic McLachlan tests
- **abstract:**
  - > "Elementary quantum mechanics proposes that a closed physical system consistently evolves in a reversible manner."
  - > "However, control and readout necessitate the coupling of the quantum system to the external environment, subjecting it to relaxation and decoherence."
  - > "Consequently, system-environment interactions are indispensable for simulating physically significant theories."
  - > "Specifically, we emphasize two aspects: (1) the discussion of recently developed quantum algorithms for these interacting models and the construction of effective Hamiltonians, and (2) a detailed analysis regarding a tightened error bound for truncating the bosonic modes for a class of fermion-boson interacting Hamiltonians."

### Fischer2025ShadowQSE
**[Fischer2025ShadowQSE]**  
L. E. Fischer, D. Bultrini, I. Tavernelli, and F. Tacchino,  
``Large-scale implementation of quantum subspace expansion with classical shadows,''  
arXiv:2510.25640 (2025).  

- **contribution:** Fischer scales classical-shadow QSE to large spaces, informing the noise-aware spectral-stack design
- **relation:** Fischer scales classical-shadow QSE to large spaces, informing the noise-aware spectral-stack design
- **abstract:**
  - > "We introduce a postprocessing technique for classical shadow measurement data that enhances the precision of ground state estimation through high-dimensional subspace expansion; the dimensionality is only limited by the amount of classical postprocessing resources rather than by quantum resources."
  - > "Crucial steps of our approach are the efficient identification of useful observables from shadow data, followed by our regularized subspace expansion that is designed to be numerically stable even when using noisy data."
  - > "We analytically investigate noise propagation within our method, and upper bound the statistical fluctuations due to the limited number of snapshots in classical shadows."
  - > "Furthermore, performance is guaranteed to be at least as good—and in many cases better—than direct energy estimation without using additional quantum resources, and the approach is thus a very natural alternative for estimating ground state energies directly from classical shadow data."
