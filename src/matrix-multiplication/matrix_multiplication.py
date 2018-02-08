import os
from mpi4py import MPI
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

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
is_master = rank == 0


def main():
    ms = get_matrix_size()
    m1, m2 = generate_matrices(ms)

    matrix1_column = comm.scatter(m1, root=0)  # scatter columns of matrix1
    m2 = comm.bcast(m2, root=0)  # bcast matrix2
    result = [np.dot(matrix1_column, row) for row in m2.transpose()]  # dot every column with transposed matrix2
    result = comm.gather(result, root=0)  # collect data to master

    if is_master:
        print(np.reshape(result, (ms, ms)))


def get_matrix_size():
    matrix_size = int(sys.argv[1]) if len(sys.argv) > 1 else size
    if matrix_size != size:
        raise Exception("matrix_size^2 must be greater or equal to cluster size")
    return matrix_size


def generate_matrices(ms):
    if is_master:
        return np.random.random((ms, ms)), np.random.random((ms, ms))
    else:
        return None, None


if __name__ == "__main__":
    main()
