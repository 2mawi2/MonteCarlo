import os

import sys

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
size = comm.Get_size()
rank = comm.Get_rank()
is_master = rank == 0


def main():
    if is_master:
        worker_data = generate_worker_data()
    else:
        worker_data = None
    pi = calculate(worker_data)
    if is_master:
        error = abs(pi - np.pi)
        print("pi is approximately %.16f, error is %.16f" % (pi, error))


def generate_worker_data():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    n = iterations // size
    return np.random.random((size, n, 2))  # generate random x,y arrays for all workers


def calculate(worker_data):
    comm.Barrier()
    data = comm.scatter(worker_data, root=0)
    pi = compute_pi(data) / size
    pi = comm.reduce(pi, op=MPI.SUM, root=0)
    comm.Barrier()
    return pi


def compute_pi(samples):
    count = 0
    for x, y in samples:
        if x ** 2 + y ** 2 <= 1:
            count += 1
    a_c = float(count)
    a_s = len(samples)
    pi = 4 * a_c / a_s
    return pi


if __name__ == "__main__":
    main()
