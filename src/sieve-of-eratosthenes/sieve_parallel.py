import os

import sys

import primesieve

try:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import numpy as np
finally:
    del os.environ["OMP_NUM_THREADS"]
    del os.environ["NUMEXPR_NUM_THREADS"]
    del os.environ["MKL_NUM_THREADS"]

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # processId of the current process
size = comm.Get_size()  # total number of processes in the communicator


def validate(max_length):
    if max_length / size <= np.sqrt(max_length):
        raise Exception("there are too many processes!")


def main():
    max_length = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    validate(max_length)

    start = int(rank * (max_length / size))
    end = int(rank * (max_length / size) + (max_length / size))

    primes = np.array(primesieve.primes(start, end), dtype=np.long)

    gather_and_print(primes)


def gather_and_print(primes):
    all_primes = None
    if rank == 0:
        all_primes = np.zeros([size, len(primes)], dtype=np.long)
    comm.Gather(primes, all_primes, root=0)
    if rank == 0:
        print(all_primes[np.nonzero(all_primes)])


if __name__ == "__main__":
    main()
