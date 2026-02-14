from mpi4py import MPI
import numpy as np
import random
from src.global_params import MPI_ROOT

#
# MPI class
#

class MPI_obj:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MPI_obj, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.root = MPI_ROOT
            self.initialized = True
    # collect array
    def collect_array(self, array):
        array_full = np.zeros(array.shape)
        array_list = self.comm.gather(array, root=self.root)
        if self.rank == self.root:
            for a in array_list:
                array_full = array_full + a
        array_full = self.comm.bcast(array_full, root=self.root)
        return array_full
    # collect list
    def collect_list(self, lst):
        list_full = []
        list_full = self.comm.gather(lst, root=self.root)
        lst = []
        if self.rank == self.root:
            for l1 in list_full:
                for l in l1:
                    lst.append(l)
        lst = self.comm.bcast(lst, root=self.root)
        return lst
    # collect atom displ array
    def collect_time_freq_array(self, f_ofwt):
        n = f_ofwt.shape[0]
        f_ofwt_full = np.zeros(n, dtype=type(f_ofwt))
        f_ofwt_list = self.comm.gather(f_ofwt[:], root=self.root)
        if self.rank == self.root:
            for f_ofwt in f_ofwt_list:
                f_ofwt_full[:] += f_ofwt[:]
        f_ofwt = self.comm.bcast(f_ofwt_full, root=self.root)
        return f_ofwt
    # split array between processes
    def split_ph_modes(self, nq, nl):
        data = []
        for iq in range(nq):
            for il in range(nl):
                data.append((iq,il))
        data = np.array(data)
        loc_proc_list = np.array_split(data, self.size)
        return loc_proc_list[self.rank]
    # split list of data
    def split_list(self, list_data):
        loc_proc_list = np.array_split(np.array(list_data), self.size)
        return list(loc_proc_list[self.rank])
    # random split list
    def random_split(self, list_data):
        if len(list_data) == 0:
            return []
        data = list(np.array(list_data))
        random.shuffle(data)
        # compute length for each process
        lengths = np.full(self.size, len(data) // self.size, dtype=int)
        lengths[:len(data) % self.size] += 1
        assert sum(lengths) == len(data)
        # partition the data
        chunks = []
        start_idx = 0
        for length in lengths:
            chunks.append(data[start_idx:start_idx + length])
            start_idx += length
        loc_proc_list = self.comm.scatter(chunks, root=self.root)
        return loc_proc_list
    # finalize procedure
    def finalize_procedure(self):
        MPI.Finalize()

#    
# mpi -> obj
#

mpi = MPI_obj()