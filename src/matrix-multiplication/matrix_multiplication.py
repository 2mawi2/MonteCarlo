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
    matrix_size = get_matrix_size()
    data = generate_worker_data(matrix_size) if is_master else None

    data = comm.scatter(data, root=0)  # distribute
    result = np.dot(data[0], data[1])  # calculate
    result = comm.gather(result, root=0)  # collect

    if is_master:
        print(np.reshape(result, (matrix_size, matrix_size)))


def get_matrix_size():
    matrix_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    if matrix_size * matrix_size != size:
        raise Exception("matrix_size^2 must be equal to cluster size")
    return matrix_size


def generate_worker_data(s):
    matrix1 = np.random.random((s, s))
    matrix2 = np.random.random((s, s))
    return [[row, column] for row in matrix1 for column in matrix2]


if __name__ == "__main__":
    main()
