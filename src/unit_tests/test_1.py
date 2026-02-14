from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit.circuit import Parameter
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from quantum.qubitization_module import PauliTerm
from quantum.pauli_polynomial_class import fermion_plus_operator, fermion_minus_operator, PauliPolynomial
import pytest
import numpy as np
import matplotlib.pyplot as plt

#
#  QISKIT test unit
#

def test_hadamard():
    # create quantum circuit
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    # simulate circuit
    backend = AerSimulator()
    job = backend.run(qc, shots=2000)
    result = job.result()
    counts = result.get_counts()
    # assert outcomes
    assert '0' in counts
    assert '1' in counts
    # assert probability ~ 0.5 for both
    p0 = counts['0'] / 2000
    p1 = counts['1'] / 2000
    assert p0 == pytest.approx(0.5, abs=0.05)
    assert p1 == pytest.approx(0.5, abs=0.05)

def test_bell_circuit():
    # Bell circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    assert qc.width() == 4
    assert qc.size() == 4
    # run circuit
    backend = AerSimulator()
    job = backend.run(qc, shots=2000)
    result = job.result()
    counts = result.get_counts()
    # assert results
    assert '00' in counts
    assert '11' in counts
    assert counts['00'] + counts['11'] > 900  # > 90% fidelity

def test_pauli_words():
    pt1 = PauliTerm(5, ps='eexyz', pc=1.)
    assert pt1.p_coeff == pytest.approx(1., abs=1.e-7)
    assert pt1.pw[0].symbol == 'e'
    assert pt1.pw[0].phase == pytest.approx(1., abs=1.e-8)
    assert pt1.pw[2].symbol == 'x'
    assert pt1.pw[2].phase == pytest.approx(1., abs=1.e-8)
    assert pt1.pw[4].symbol == 'z'
    assert pt1.pw[4].phase == pytest.approx(1., abs=1.e-8)

def test_pauli_term_product():
    pt1 = PauliTerm(5, ps='eexyz', pc=1.)
    pt2 = PauliTerm(5, ps='xyezy', pc=1j)
    r = pt1 * pt2
    assert r.p_coeff.real == pytest.approx(0., abs=1.e-7)
    assert r.p_coeff.imag == pytest.approx(1., abs=1.e-7)
    assert r.pw[0].symbol == 'x'
    assert r.pw[1].symbol == 'y'
    assert r.pw[2].symbol == 'x'
    assert r.pw[3].symbol == 'x'
    assert r.pw[4].symbol == 'x'
    # replace pt1
    pt3 = PauliTerm(5, ps='zzzzz', pc=1.)
    pt1 *= pt3
    assert pt1.p_coeff.real == pytest.approx(1., abs=1.e-7)
    assert pt1.p_coeff.imag == pytest.approx(0., abs=1.e-7)
    assert pt1.pw[0].symbol == 'z'
    assert pt1.pw[1].symbol == 'z'
    assert pt1.pw[2].symbol == 'y'
    assert pt1.pw[3].symbol == 'x'
    assert pt1.pw[4].symbol == 'e'

def test_pauli_pol_reduction():
    f2q_mode = "JW"
    c_jdagg = fermion_plus_operator(f2q_mode, 3, 0)
    pp = c_jdagg.return_polynomial()
    assert len(pp) == 2
    assert pp[0].p_coeff.real == pytest.approx(0., 1.e-7)
    assert pp[0].p_coeff.imag == pytest.approx(-0.5, 1.e-7)
    assert pp[0].pw[0].symbol == 'e'
    assert pp[0].pw[1].symbol == 'e'
    assert pp[0].pw[2].symbol == 'y'
    assert pp[1].p_coeff.real == pytest.approx(0.5, 1.e-7)
    assert pp[1].p_coeff.imag == pytest.approx(0., 1.e-7)
    assert pp[1].pw[0].symbol == 'e'
    assert pp[1].pw[1].symbol == 'e'
    assert pp[1].pw[2].symbol == 'x'
    c_jdagg2= fermion_plus_operator(f2q_mode, 3, 0)
    c_jdagg += c_jdagg2
    pp2 = c_jdagg.return_polynomial()
    assert len(pp2) == 2
    assert pp2[0].p_coeff.real == pytest.approx(1., 1.e-7)
    assert pp2[0].p_coeff.imag == pytest.approx(0., 1.e-7)
    assert pp2[0].pw[0].symbol == 'e'
    assert pp2[0].pw[1].symbol == 'e'
    assert pp2[0].pw[2].symbol == 'x'
    assert pp2[1].p_coeff.real == pytest.approx(0., 1.e-7)
    assert pp2[1].p_coeff.imag == pytest.approx(-1., 1.e-7)
    assert pp2[1].pw[0].symbol == 'e'
    assert pp2[1].pw[1].symbol == 'e'
    assert pp2[1].pw[2].symbol == 'y'
    c_j = fermion_minus_operator(f2q_mode, 3, 0)
    c_jdagg = fermion_plus_operator(f2q_mode, 3, 0)
    cc_j = c_j + c_jdagg
    pp3 = cc_j.return_polynomial()
    assert len(pp3) == 1
    assert pp3[0].p_coeff.real == pytest.approx(1., 1.e-7)
    assert pp3[0].p_coeff.imag == pytest.approx(0., 1.e-7)
    assert pp3[0].pw[0].symbol == 'e'
    assert pp3[0].pw[1].symbol == 'e'
    assert pp3[0].pw[2].symbol == 'x'

def test_pauli_pol_product():
    f2q_mode = "JW"
    c_j = fermion_minus_operator(f2q_mode, 3, 0)
    c_jdagg = fermion_plus_operator(f2q_mode, 3, 0)
    cc_j = c_j + c_jdagg
    cc_j2 = cc_j * cc_j
    pp = cc_j2.return_polynomial()
    assert len(pp) == 1
    assert pp[0].p_coeff.real == pytest.approx(1., 1.e-7)
    assert pp[0].p_coeff.imag == pytest.approx(0., 1.e-7)
    assert pp[0].pw[0].symbol == 'e'
    assert pp[0].pw[1].symbol == 'e'
    assert pp[0].pw[2].symbol == 'e'
    # test product complex number
    cc_j3 = 1j * cc_j2
    pp = cc_j3.return_polynomial()
    assert len(pp) == 1
    assert pp[0].p_coeff.real == pytest.approx(0., 1.e-7)
    assert pp[0].p_coeff.imag == pytest.approx(1., 1.e-7)
    assert pp[0].pw[0].symbol == 'e'
    assert pp[0].pw[1].symbol == 'e'
    assert pp[0].pw[2].symbol == 'e'
    cc_j4 = 2. * cc_j3
    pp = cc_j4.return_polynomial()
    assert len(pp) == 1
    assert pp[0].p_coeff.real == pytest.approx(0., 1.e-7)
    assert pp[0].p_coeff.imag == pytest.approx(2., 1.e-7)
    assert pp[0].pw[0].symbol == 'e'
    assert pp[0].pw[1].symbol == 'e'
    assert pp[0].pw[2].symbol == 'e'

def test_Ising_model():
    # Define the Ising Hamiltonian H = -J * Z_i Z_(i+1)
    # Let's use J=1 and apply it to a 2-qubit system for simplicity.
    J = 1
    nq = 2
    # H = - Z0 Z1
    hamiltonian = SparsePauliOp.from_list([
        ("ZZ", -J)
    ])
    # exact solution
    exact_ener = -J
    # -----------------------
    # Ansatz
    # -----------------------
    ansatz = TwoLocal(
        num_qubits=nq,
        rotation_blocks="ry",
        entanglement_blocks="cx",
        entanglement="full",
        reps=1
    )
    # -----------------------
    # Optimizer
    # -----------------------
    optimizer = SLSQP(maxiter=200)
    # -----------------------
    # VQE
    # -----------------------
    estimator = Estimator()
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer
    )
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    energy = result.eigenvalue.real
    assert exact_ener == pytest.approx(energy, 1.e-4)
    # exact simulation 3 qubits
    J1 = 1
    J2 = 0.5
    nq = 3
    # H = -J1 Z0 Z1 -J2 Z1 Z2
    hamiltonian = SparsePauliOp.from_list([
        ("ZZI", -J1),
        ("IZZ", -J2)
    ])
    # exact solution
    exact_ener2 = -J1 - J2
    # -----------------------
    # Ansatz
    # -----------------------
    ansatz = TwoLocal(
        num_qubits=nq,
        rotation_blocks="ry",
        entanglement_blocks="cx",
        entanglement="full",
        reps=1
    )
    # -----------------------
    # Optimizer
    # -----------------------
    optimizer = SLSQP(maxiter=200)
    # -----------------------
    # VQE
    # -----------------------
    estimator = Estimator()
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer
    )
    # -----------------------
    # Plot the ansatz circuit
    # -----------------------
    fig = ansatz.draw(output="mpl")
    fig.savefig("Ising_ansatz.png", dpi=300)
    plt.close(fig)
    result2 = vqe.compute_minimum_eigenvalue(hamiltonian)
    energy2 = result2.eigenvalue.real
    assert exact_ener2 == pytest.approx(energy2, 1.e-4)
    ground_circ = result2.optimal_circuit
    ground_params = list(result2.optimal_parameters.values())
    # Time evolution
    # Parameters
    n_steps = 100
    dt = 0.1          # time step
    B = 0.5           # field amplitude
    S_vals = []
    times = []
    circ = ground_circ.copy()
    # Observable: Z0
    Z0 = SparsePauliOp.from_list([("ZII", 1.0)])
    for step in range(n_steps):
        t = step * dt
        times.append(t)
        B_t = B * np.sin(t)  # example time-dependent field
        # perturbation as X rotation on qubit 0
        H_pert = SparsePauliOp.from_list([("XII", B_t)])
        H_total = hamiltonian + H_pert
        # Hamiltonian evolution via 1-step Trotter
        evo_gate = PauliEvolutionGate(
            H_total,
            time=dt,
            synthesis=SuzukiTrotter(order=2, reps=1)
        )
        circ.append(evo_gate, circ.qubits)
        # Measure Z0 expectation
        val = estimator.run(circuits=[circ], observables=[Z0], parameter_values=[ground_params]).result().values[0]
        S_vals.append(val)
    # -----------------------
    # Plot the full evolution circuit
    # -----------------------
    fig_evo = circ.draw(output="mpl", fold=-1)
    fig_evo.savefig("Ising_time_evolution_circuit.png", dpi=300)
    plt.close(fig_evo)
    # Save to file
    with open("Z0_vs_time.txt", "w") as f:
        f.write("# Step\t<Z0>\n")
        for step, val in enumerate(S_vals):
            f.write(f"{step}\t{val:.12f}\n")
    # -----------------------
    # Plot ⟨Z0⟩ vs time
    # -----------------------
    plt.figure(figsize=(6,4))
    plt.plot(times, S_vals, marker='o', linestyle='-', color='blue')
    plt.xlabel("Time")
    plt.ylabel("<Z0>")
    plt.title("Time Evolution of <Z0>")
    plt.grid(True)
    plt.savefig("Z0_vs_time.png", dpi=300)
    plt.close()

def test_reduced_Hubbard_dimer():
    f2q_mode = "JW"
    nq = 4
    t = 1.
    U = 2.
    H = PauliPolynomial(f2q_mode)
    # 4 qubits -> |1u,2d>, |1d,2u>, |1u,1d>, |2u,2d>
    # spin dw
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 0) * fermion_minus_operator(f2q_mode, nq, 2)
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 2) * fermion_minus_operator(f2q_mode, nq, 0)
    # spin up
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 0) * fermion_minus_operator(f2q_mode, nq, 3)
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 3) * fermion_minus_operator(f2q_mode, nq, 0)
    # spin up
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 1) * fermion_minus_operator(f2q_mode, nq, 2)
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 2) * fermion_minus_operator(f2q_mode, nq, 1)
    # spin dw
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 1) * fermion_minus_operator(f2q_mode, nq, 3)
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 3) * fermion_minus_operator(f2q_mode, nq, 1)
    Hpol = H.return_polynomial()
    pauli_list = []
    for it in range(len(Hpol)):
        pt = Hpol[it].pw2sparsePauliOp()
        pauli_list.append(pt)
    hamiltonian = SparsePauliOp.from_list(pauli_list)
    # exact solution
    exact_ener = -2*t
    # -----------------------
    # Ansatz
    # -----------------------
    # Hartree-Fock prep: |1u 1d> -> qubits 0 and 1
    hf_circ = QuantumCircuit(nq)
    hf_circ.x(0)  # 1u
    hf_circ.x(1)  # 1d
    # Excitation-preserving ansatz
    from qiskit.circuit.library import ExcitationPreserving
    ansatz = ExcitationPreserving(
        num_qubits=nq,
        mode="iswap",       # or 'fsim'
        entanglement="full",
        reps=1
    )
    # Prepend HF state
    ansatz = hf_circ.compose(ansatz)
    # -----------------------
    # Optimizer
    # -----------------------
    optimizer = SLSQP(maxiter=200)
    # -----------------------
    # VQE
    # -----------------------
    estimator = Estimator()
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer
    )
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    energy = result.eigenvalue.real
    assert exact_ener == pytest.approx(energy, 1.e-4)

def test_Hubbard_dimer():
    f2q_mode = "JW"
    nq = 4
    t = 1.   # hopping coeff.
    U = 2.   # Hubbard parameter
    H = PauliPolynomial(f2q_mode)
    # 4 qubits |1u>, |1d>, |2u>, |2d>
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 0) * fermion_minus_operator(f2q_mode, nq, 2)
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 2) * fermion_minus_operator(f2q_mode, nq, 0)
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 1) * fermion_minus_operator(f2q_mode, nq, 3)
    H += (-t) * fermion_plus_operator(f2q_mode, nq, 3) * fermion_minus_operator(f2q_mode, nq, 1)
    Hpol = H.return_polynomial()
    pauli_list = []
    for it in range(len(Hpol)):
        pt = Hpol[it].pw2sparsePauliOp()
        pauli_list.append(pt)
    # set hamiltonian
    hamiltonian = SparsePauliOp.from_list(pauli_list)
    # Number operator
    N = PauliPolynomial(f2q_mode)
    N += fermion_plus_operator(f2q_mode, nq, 0) * fermion_minus_operator(f2q_mode, nq, 0)
    N += fermion_plus_operator(f2q_mode, nq, 1) * fermion_minus_operator(f2q_mode, nq, 1)
    N += fermion_plus_operator(f2q_mode, nq, 2) * fermion_minus_operator(f2q_mode, nq, 2)
    N += fermion_plus_operator(f2q_mode, nq, 3) * fermion_minus_operator(f2q_mode, nq, 3)
    Npol = N.return_polynomial()
    pauli_list = []
    for it in range(len(Npol)):
        pt = Npol[it].pw2sparsePauliOp()
        pauli_list.append(pt)
    Nop = SparsePauliOp.from_list(pauli_list)
    N1 = PauliPolynomial(f2q_mode)
    N1 += fermion_plus_operator(f2q_mode, nq, 0) * fermion_minus_operator(f2q_mode, nq, 0)
    N1 += fermion_plus_operator(f2q_mode, nq, 1) * fermion_minus_operator(f2q_mode, nq, 1)
    Npol = N1.return_polynomial()
    pauli_list = []
    for it in range(len(Npol)):
        pt = Npol[it].pw2sparsePauliOp()
        pauli_list.append(pt)
    N1op = SparsePauliOp.from_list(pauli_list)
    # Spin operator
    Sz = PauliPolynomial(f2q_mode)
    Sz += 0.5 * fermion_plus_operator(f2q_mode, nq, 0) * fermion_minus_operator(f2q_mode, nq, 0)
    Sz -= 0.5 * fermion_plus_operator(f2q_mode, nq, 1) * fermion_minus_operator(f2q_mode, nq, 1)
    Sz += 0.5 * fermion_plus_operator(f2q_mode, nq, 2) * fermion_minus_operator(f2q_mode, nq, 2)
    Sz -= 0.5 * fermion_plus_operator(f2q_mode, nq, 3) * fermion_minus_operator(f2q_mode, nq, 3)
    Spol = Sz.return_polynomial()
    pauli_list = []
    for it in range(len(Spol)):
        pt = Spol[it].pw2sparsePauliOp()
        pauli_list.append(pt)
    Sop = SparsePauliOp.from_list(pauli_list)
    # exact solution
    exact_ener = -2*t
    # -----------------------
    # Ansatz
    # -----------------------
    hf_circ = QuantumCircuit(nq)
    hf_circ.x(0)  # 1↑ occupied
    hf_circ.x(1)  # 1↓ occupied
    # Apply UCC excitation
    from qiskit.circuit.library import ExcitationPreserving
    # Step 2: Excitation-preserving ansatz
    exc_circ = ExcitationPreserving(
        num_qubits=nq,
        mode="iswap",
        entanglement="full",
        reps=1
    )
    # Step 3: Combine HF + excitation
    ansatz = hf_circ.compose(exc_circ)
    # -----------------------
    # Optimizer
    # -----------------------
    optimizer = SLSQP(maxiter=200)
    # -----------------------
    # VQE
    # -----------------------
    estimator = Estimator()
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=[0.01] * ansatz.num_parameters
    )
    result = vqe.compute_minimum_eigenvalue(hamiltonian)
    energy = result.eigenvalue.real
    assert exact_ener == pytest.approx(energy, 1.e-4)
    # <N>
    exact_N = 2.0
    N_expect = estimator.run(
        circuits=[result.optimal_circuit],
        observables=[Nop],
        parameter_values=[list(result.optimal_parameters.values())],
    ).result().values[0]
    assert exact_N == pytest.approx(N_expect, 1.e-4)
    # <S>
    exact_S = 0.0
    S_expect = estimator.run(
        circuits=[result.optimal_circuit],
        observables=[Sop],
        parameter_values=[list(result.optimal_parameters.values())],
    ).result().values[0]
    assert abs(S_expect - exact_S) < 1.e-4
    # Example: save the full ansatz to PNG
    fig = ansatz.draw(output="mpl")
    fig.savefig("hubbard_dimer_ansatz.png", dpi=300)  # save as PNG
    plt.close(fig)  # close the figure to avoid showing it in tests
    # Parameters
    n_steps = 100
    dt = 0.1          # time step
    B = 10.0          # field amplitude
    with open("S_vs_time.txt", "w") as f:
        f.write("# Step\t<S>\n")  # header
        # Start from VQE ground state
        circ = result.optimal_circuit.copy()
        # Store expectation values
        S_list = []
        for step in range(n_steps):
            # 1) Apply external field on qubit 0 (Z rotation)
            t = step * dt
            B_t = B * np.sin(t)  # example: B(t) = B * sin(t)
            P = (-B_t) * fermion_plus_operator(f2q_mode, nq, 0) * fermion_minus_operator(f2q_mode, nq, 2)
            P += (-B_t) * fermion_plus_operator(f2q_mode, nq, 2) * fermion_minus_operator(f2q_mode, nq, 0)
            Ppol = P.return_polynomial()
            pauli_list = []
            for it in range(len(Ppol)):
                pt = Ppol[it].pw2sparsePauliOp()
                pauli_list.append(pt)
            # set hamiltonian
            H_pert = SparsePauliOp.from_list(pauli_list, num_qubits=nq)
            # perturbation at this time
            H_total = hamiltonian + H_pert
            # 2) Apply Hamiltonian evolution for dt using 1-step Trotter
            evo_gate = PauliEvolutionGate(H_total, time=dt, synthesis=SuzukiTrotter(order=2, reps=1))
            # append to your VQE ground state circuit
            circ = result.optimal_circuit.copy()
            circ.append(evo_gate, circ.qubits)
            # 3) Measure ⟨S⟩
            # Get VQE optimized parameters as a list
            param_values = list(result.optimal_parameters.values())
            S_val = estimator.run(circuits=[circ], observables=[N1op], parameter_values=[param_values]).result().values[0]
            S_list.append(S_val)
            # write to file
            f.write(f"{step}\t{S_val:.12f}\n")
    # Print results
    #for step, S_val in enumerate(S_list):
    #    print(f"Step {step}: ⟨S⟩ = {S_val:.6f}")