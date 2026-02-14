#
#   This module defines
#   the spin Hamiltonian 
#

from abc import ABC
import numpy as np
from numpy import linalg as LA
from src.common.phys_constants import gamma_e, eps, THz_to_ev
from src.utilities.log import log
from src.parallelization.mpi import mpi
from src.set_param_object import p
from src.quantum.pauli_polynomial_class import PauliPolynomial, fermion_plus_operator, fermion_minus_operator

#
#   function : set spin Hamiltonian
#

def set_spin_hamiltonian(struct0, B0, nuclear_config=None):
	# extract spin configuration
	struct0.extract_spin_state()
	# spin mult.
	if int(struct0.spin_multiplicity) == 3:
		if mpi.rank == mpi.root:
			log.info("\t SPIN TRIPLET CALCULATION")
		spin_hamilt = spin_triplet_hamiltonian()
	elif int(struct0.spin_multiplicity) == 2:
		if mpi.rank == mpi.root:
			log.info("\t SPIN DOUBLET CALCULATION")
		spin_hamilt = spin_doublet_hamiltonian()
	else:
		log.error("Wrong spin multiplicity : " + str(struct0.spin_multiplicity))
	spin_hamilt.set_zfs_levels(struct0, B0, nuclear_config)
	return spin_hamilt
#

class spin_hamiltonian(ABC):
	def __init__(self):
		# quantum spin states
		self.qs = None
	def Splus_mtxel(self, j1, m1, j2, m2):
		r = np.sqrt((j2 - m2) * (j2 + m2 + 1)) * delta(j1, j2) * delta(m1, m2+1)
		return r
	def Sminus_mtxel(self, j1, m1, j2, m2):
		r = np.sqrt((j2 + m2) * (j2 - m2 + 1)) * delta(j1, j2) * delta(m1, m2-1)
		return r
	def set_Splus(self):
		#
		#    S+  ->
		#    <j1,m1|S+|j2,m2> = sqrt((j2-m2)(j2+m2+1)) delta(j1,j2) delta(m1,m2+1)
		#
		for r in range(len(self.basis_vectors)):
			v1 = self.basis_vectors[r]
			for c in range(len(self.basis_vectors)):
				v2 = self.basis_vectors[c]
				[j1, m1] = v1
				[j2, m2] = v2
				self.Splus[r,c] = self.Splus_mtxel(j1, m1, j2, m2)
	def set_Sminus(self):
		#
		#    S-   ->
		#    <j1,m1|S-|j2,m2> = sqrt((j2+m2)(j2-m2+1)) delta(j1,j2) delta(m1,m2-1)
		#
		for r in range(len(self.basis_vectors)):
			v1 = self.basis_vectors[r]
			for c in range(len(self.basis_vectors)):
				v2 = self.basis_vectors[c]
				[j1, m1] = v1
				[j2, m2] = v2
				self.Sminus[r,c] = self.Sminus_mtxel(j1, m1, j2, m2)
	def set_Sx(self):
		self.Sx[:,:] = (self.Splus[:,:] + self.Sminus[:,:]) / 2.
	def set_Sy(self):
		self.Sy[:,:] = (self.Splus[:,:] - self.Sminus[:,:]) / (2.*1j)
	def set_Ssq(self):
		self.Ssq = np.matmul(self.Sx, self.Sx) + np.matmul(self.Sy, self.Sy) + np.matmul(self.Sz, self.Sz)
		assert np.abs(self.Ssq[0,0] - self.s*(self.s+1.)) < eps
		assert np.abs(self.Ssq[1,1] - self.s*(self.s+1.)) < eps
		assert np.abs(self.Ssq[2,2] - self.s*(self.s+1.)) < eps
	def return_spin_eigenstates(self):
		return self.qs
	def check_degeneracy(self):
		return False

#
#   spin triplet Hamiltonian class
#
#   for spin triplet ground states
#   
#   Hss = D [Sz^2 - S(S+1)/3] + E (Sx^2 - Sy^2) + \sum_i^N I_i Ahfi(i) S
#

class spin_triplet_hamiltonian(spin_hamiltonian):
	def __init__(self):
		super().__init__()
		self.s = 1.
		self.basis_vectors = [[self.s,1.], [self.s,0.], [self.s,-1.]]
		self.Splus = np.zeros((3,3), dtype=np.complex128)
		self.Sminus = np.zeros((3,3), dtype=np.complex128)
		self.Sx = np.zeros((3,3), dtype=np.complex128)
		self.Sy = np.zeros((3,3), dtype=np.complex128)
		self.Sz = np.zeros((3,3), dtype=np.complex128)
		self.qs = []
		# set spin matrices
		self.set_Sz()
		self.set_Splus()
		self.set_Sminus()
		self.set_Sx()
		self.set_Sy()
		self.set_Ssq()
	#   Sz
	def set_Sz(self):
		self.Sz[0,0] = 1.
		self.Sz[2,2] = -1.
	#   SDS
	def set_SDS(self, unprt_struct):
		Ddiag = unprt_struct.Ddiag*1.E-6
		#
		#  THz units
		#
		self.SDS = Ddiag[0]*np.matmul(self.Sx, self.Sx)
		self.SDS = self.SDS + Ddiag[1]*np.matmul(self.Sy, self.Sy)
		self.SDS = self.SDS + Ddiag[2]*np.matmul(self.Sz, self.Sz)
		self.SDS = self.SDS * THz_to_ev
		# eV units
		# eigenv.
		eig = LA.eig(self.SDS)[0]
		eig0 = LA.eig(self.H0)[0]
		np.testing.assert_almost_equal(eig, eig0, decimal=5)
	# hyperfine interaction
	def hyperfine_coupl(self, site, I, Ahfi):
		# hyperfine coupl. in eV from MHz
		Ahfi_ev = Ahfi * 1.E-6 * THz_to_ev
		I_Ahf = np.einsum("i,ij->j", I, Ahfi_ev[site,:,:])
		Hhf = I_Ahf[0] * self.Sx + I_Ahf[1] * self.Sy + I_Ahf[2] * self.Sz
		return Hhf
	# set ZFS energy levels
	def set_zfs_levels(self, unprt_struct, B, nuclear_config=None):
		Ddiag = unprt_struct.Ddiag*1.E-6  # THz
		Ddiag = Ddiag * THz_to_ev         # eV units
		# D = 3./2 Dz
		D = 3./2 * Ddiag[2]
		# E = (Dx - Dy)/2
		E = (Ddiag[0] - Ddiag[1]) / 2.
		# unperturbed H0 = D[Sz^2 - s(s+1)/3] + E(Sx^2 - Sy^2) - mu_B B Sz
		H0 = D * (np.matmul(self.Sz, self.Sz) - self.Ssq / 3.) 
		H0 +=E * (np.matmul(self.Sx, self.Sx) - np.matmul(self.Sy, self.Sy))
		# store unperturbed Hamiltonian
		self.H0 = np.copy(H0)
		# add B field + HFI if present
		H0 -=gamma_e * (B[0] * self.Sx + B[1] * self.Sy + B[2] * self.Sz)
		if nuclear_config is not None:
			for isp in range(nuclear_config.nsp):
				I = nuclear_config.nuclear_spins[isp]['I']
				site = nuclear_config.nuclear_spins[isp]['site']
				H0 += self.hyperfine_coupl(site, I, unprt_struct.Ahfi)
		# store eigenstates
		eig, eigv = LA.eig(H0)
		for s in range(3):
			self.qs.append({'eig':eig[s], 'eigv':eigv[:,s]})
		# check eigenvalues
		self.set_SDS(unprt_struct)
	# set Hamiltonian time t
	def set_hamilt_oft(self, t, B, unprt_struct=None, nuclear_config=None):
		# get unpert. Hamiltonian
		H = np.copy(self.H0)
		# add ext. magnetic field
		H -=gamma_e * (B[0][t] * self.Sx + B[1][t] * self.Sy + B[2][t] * self.Sz)
		# hyperfine interaction
		if nuclear_config is not None:
			for isp in range(nuclear_config.nsp):
				It = nuclear_config.nuclear_spins[isp]['I'][t]
				site = nuclear_config.nuclear_spins[isp]['site']
				H += self.hyperfine_coupl(site, It, unprt_struct.Ahfi)
		return H

#
#   spin doublet 
#   Hamiltonian class
#

class spin_doublet_hamiltonian(spin_hamiltonian):
	def __init__(self):
		super().__init__()
		self.s = 0.5
		self.basis_vectors = [[self.s, 0.5], [self.s, -0.5]]
		self.Splus = np.zeros((2,2), dtype=np.complex128)
		self.Sminus = np.zeros((2,2), dtype=np.complex128)
		self.Sx = np.zeros((2,2), dtype=np.complex128)
		self.Sy = np.zeros((2,2), dtype=np.complex128)
		self.Sz = np.zeros((2,2), dtype=np.complex128)
		self.qs = []
		# set spin matrices
		self.set_Sz()
		self.set_Splus()
		self.set_Sminus()
		self.set_Sx()
		self.set_Sy()
		self.set_Ssq()
	#   Sz
	def set_Sz(self):
		self.Sz[0,0] = 0.5
		self.Sz[1,1] =-0.5

#
#   full spin Hamiltonian object :
#      this object is created when we need the full 
#      quantum representation of the Hamiltonian
#      including environment DOF 

class quantum_spin_hamiltonian:
	def __init__(self, qbit_repr, NUCL_SPINS=False, PHONONS=False):
		# nuclear spins
		self.NUCL_SPINS = NUCL_SPINS
		# phonons
		self.PHONONS = PHONONS
		# isolated spin Hamiltonian object
		self.Hsp = None
		# nuclear spin config.
		self.nuclear_config = None
		# ph. object
		self.ph = None
		# system hamiltonian in qubit form
		self.__Hsys_q = None
		# n. qubits
		self.__nq = None
		# qubit repr. mode
		self.qbit_repr = qbit_repr
	def print_info(self):
		log.info("\n")
		log.info("\t " + p.sep)
		log.info("\t FULL SYSYTEM HAMILTONIAN DEFINITION")
		log.info("\t PHONONS: " + str(self.PHONONS))
		log.info("\t NUCLEAR SPINS: " + str(self.NUCL_SPINS))
		log.info("\t QUBIT REPR. MODE: " + self.qbit_repr)
		log.info("\t " + p.sep)
		log.info("\n")
	def compute_number_qubits(self):
		nq = len(self.Hsp.qs)
		if self.NUCL_SPINS:
			# spin states up / dw
			nq += 2*self.nuclear_config.nsp
		return nq
	def set_system_qubit_hamiltonian(self, struct_0, B, nuclear_config=None, ph=None):
		#  first build isolated spin hamiltonian
		self.Hsp = set_spin_hamiltonian(struct_0, B)
		# set nuclear config. if available
		if self.NUCL_SPINS:
			self.nuclear_config = nuclear_config
		# set ph. if present
		if self.PHONONS:
			self.ph = ph
		# n. qubits required
		self.__nq = self.compute_number_qubits()
		if mpi.rank == mpi.root:
			log.info("\t " + p.sep)
			log.info("\t " + "n. qubits in simulation: " + str(self.__nq))
			log.info("\t " + p.sep)
		self.__Hsys_q = PauliPolynomial(self.qbit_repr)
	def qubitize_spin_hamilt(self):
    	#
    	#  This function convert the spin Hamiltonian
    	#  from fermion basis -> qubit basis
    	#
		#  build Pauli polynomial
		Hq = PauliPolynomial(self.qbit_repr)
		#  fermionic qubit iq=0 -> len(Hsp.qs)
		nf = len(self.Hsp.qs)
		for iq in range(nf):
			cj = fermion_minus_operator(self.qbit_repr, self.__nq, iq)
			cjd= fermion_plus_operator(self.qbit_repr, self.__nq, iq)
			r = cjd * cj
			Hq += self.Hsp.qs[iq]['eig'] * r
		self.__Hsys_q += Hq
		if mpi.rank == mpi.root:
			log.info("\t size Hsys_q polynomial: " + str(self.__Hsys_q.count_number_terms()))
			self.__Hsys_q.visualize_polynomial()