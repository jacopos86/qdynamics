import time
import os
from src.parallelization.mpi import mpi
# set timer module
class timer_class:
    def __init__(self):
        self.start_time = 0.
    def start_execution(self):
        self.start_time = time.time()
    def end_execution(self):
        elapsed_time = time.time() - self.start_time
        cpu_elapsed_time = time.process_time() - self.start_time
        if mpi.rank == mpi.root:
            print("*\t WALL CLOCK TIME : " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            print("*\t CPU EXECUTION TIME : " + time.strftime("%H:%M:%S", time.gmtime(cpu_elapsed_time)))
            seconds = time.time()
            local_time = time.ctime(seconds)
            print("*\t EXECUTION ENDED: ", local_time)
timer = timer_class()