#
#   This module defines
#   the nuclear spin configuration class
#   - number of non zero nuclear spins
#   - atomic sites with non zero spin
#   - nuclear spin value
#

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import yaml
import random
import logging
from src.common.phys_constants import gamma_n
from src.parallelization.mpi import mpi
from src.utilities.log import log
from src.set_param_object import p
from src.common.matrix_operations import norm_realv

#
class nuclear_spins_config():
	def __init__(self, nsp, B0):
		self.nsp = nsp
		self.B0 = np.array(B0)
		# applied mag. field (G) units
		self.nuclear_spins = []
		# I spins list
	def set_time(self, dt, T):
		# n. time steps
		nt = int(T / dt)
		self.time = np.linspace(0., T, nt)
		# finer array
		nt = int(T / (dt/2.))
		self.time_dense = np.linspace(0., T, nt)
		# micro sec units
	def set_orientation(self, ic):
		# theta angles
		np.random.seed(ic)
		th = np.random.normal(0., 1., self.nsp)
		m = max(np.abs(np.min(th)), np.max(th))
		th[:] = th[:] * np.pi / m
		if log.level <= logging.DEBUG:
			plt.hist(th, 30, density=True)
			plt.ylabel("theta distrib.")
			plt.show()
		# phi angles
		np.random.seed(ic+1)
		phi = np.random.normal(0., 1., self.nsp)
		m = max(np.abs(np.min(phi)), np.max(phi))
		phi[:] = phi[:] * 2.*np.pi / m
		if log.level <= logging.DEBUG:
			plt.hist(phi, 30, density=True)
			plt.ylabel("phi distrib.")
			plt.show()
		# directions array
		Iv = np.zeros((3, self.nsp))
		for isp in range(self.nsp):
			cth = np.cos(th[isp])
			sth = np.sin(th[isp])
			cphi= np.cos(phi[isp])
			sphi= np.sin(phi[isp])
			# spin vector
			Iv[0,isp] = sth * cphi
			Iv[1,isp] = sth * sphi
			Iv[2,isp] = cth
		return Iv
	# set nuclear spins Hamiltonian
	def set_nuclear_spins_hamilt(self):
		pass
	# set electron spin magnetization vector
	def set_electron_magnet_vector(self, Hss):
		Mt = Hss.Mt
		# ps units
		t = Hss.time
		# compute <Mt> -> spin magnetization
		T = t[-1]
		# ps units
		rx = integrate.simpson(Mt[0,:], t) / T
		ry = integrate.simpson(Mt[1,:], t) / T
		rz = integrate.simpson(Mt[2,:], t) / T
		M  = np.array([rx, ry, rz])
		return M
	# set nuclear configuration method
	def set_nuclear_spins(self, nat, ic):
		# set distribution of orientations
		if p.rnd_orientation:
			Iv = self.set_orientation(ic)
		else:
			# assume nuclear spin random initial orientation
			v = self.B0 / norm_realv(self.B0)
			Iv = np.zeros((3,self.nsp))
			isp = 0
			while isp < self.nsp:
				Iv[:,isp] = v[:]
				isp += 1
		# set spin list
		Ilist = []
		isp = 0
		while isp < self.nsp:
			# set spin vector
			I = np.zeros(3)
			# compute components
			# in cart. coordinates
			I[:] = 0.5 * Iv[:,isp]
			Ilist.append(I)
			isp += 1
		# set atom's site
		random.seed(ic)
		sites = random.sample(range(0, nat), self.nsp)
		if mpi.rank == mpi.root:
			log.info("\t nuclear spin sites : " + str(sites))
		# define dictionary
		keys = ['site', 'I']
		for isp in range(self.nsp):
			self.nuclear_spins.append(dict(zip(keys, [sites[isp], Ilist[isp]])))
	# set spin vector evolution
	# method to time evolve I
	def set_nuclear_spin_evol(self, Hss, unprt_struct):
		# dIa/dt = gamma_n B X Ia + (A_hf(a) M(t)) X Ia
		# B : applied magnetic field (Gauss)
		# spin_hamilt : spin Hamiltonian object
		# unprt_struct : unperturbed atomic structure
		M = self.set_electron_magnet_vector(Hss)
		if mpi.rank == mpi.root:
			log.info("\t <S>_t = " + str(M))
		# n. time steps integ.
		nt = len(self.time_dense)
		# time interv. (micro sec)
		dt = self.time[1]-self.time[0]
		# set [B] matrix
		Btilde = set_cross_prod_matrix(self.B0)
		# run over the spins active
		# in the configuration
		for isp in range(self.nsp):
			site = self.nuclear_spins[isp]['site']
			# set HFI matrix (MHz)
			A = np.zeros((3,3))
			A[:,:] = 2.*np.pi*unprt_struct.Ahfi[site-1,:,:]
			# set F(t) = gamma_n B + Ahf(a) M
			Ft = np.zeros((3,3,nt))
			# A(a) M
			AM = np.matmul(A, M)
			AM_tilde = set_cross_prod_matrix(AM)
			for i in range(nt):
				Ft[:,:,i] = gamma_n * Btilde[:,:]
				Ft[:,:,i] = Ft[:,:,i] + AM_tilde[:,:]
				# MHz units
			I0 = self.nuclear_spins[isp]['I']
			It = ODE_solver(I0, Ft, dt)
			self.nuclear_spins[isp]['It'] = It
	# compute nuclear spin derivatives at t=0
	def compute_nuclear_spin_derivatives(self, Hss, unprt_struct, n):
		# dIa/dt = gamma_n B X Ia + (A_hf(a) M(t)) X Ia
		# dt^(2)Ia = gamma_n B X dIa/dt + (A_hf(a) M(t)) X dIa/dt
		# ...
		# dt^(n)Ia = gamma_n B X dt^(n-1)Ia + (A_hf(a) M(t)) X d^(n-1)Ia
		# spin magnetization
		M = self.set_electron_magnet_vector(Hss)
		# set [B] antisymmetric matrix
		Btilde = set_cross_prod_matrix(self.B0)
		# iterate over spins
		for isp in range(self.nsp):
			# order 0 spin vector
			self.nuclear_spins[isp]['dIt'] = np.zeros((3,n+1))
			I0 = self.nuclear_spins[isp]['I']
			self.nuclear_spins[isp]['dIt'][:,0] = I0[:]
			# hyperfine interaction
			A = np.zeros((3,3))
			site = self.nuclear_spins[isp]['site']
			A[:,:] = 2.*np.pi*unprt_struct.Ahfi[site-1,:,:]
			AM = np.matmul(A, M)
			AM_tilde = set_cross_prod_matrix(AM)
			# generator : F
			F = np.zeros((3,3))
			F[:,:] = gamma_n * Btilde[:,:]
			F[:,:]+= AM_tilde[:,:]
			# iterate over nth order derivatives
			for i in range(1, n+1):
				dI0 = self.nuclear_spins[isp]['dIt'][:,i-1]
				# compute dI
				dI = np.matmul(F, dI0)
				self.nuclear_spins[isp]['dIt'][:,i] = dI[:]
				# musec^(-i) units
	# write I(t) on ext. file
	def write_It_on_file(self, out_dir, ic):
		# write file name
		name_file = out_dir + "/config-sp" + str(self.nsp) + "-" + str(ic+1) + ".yml"
		# set dictionary
		dict = {'time' : 0, 'nuclear spins' : []}
		dict['time'] = self.time
		dict['nuclear spins'] = self.nuclear_spins
		# save data
		with open(name_file, 'w') as out_file:
			yaml.dump(dict, out_file)