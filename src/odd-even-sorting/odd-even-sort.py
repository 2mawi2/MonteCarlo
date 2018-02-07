import os

try:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import numpy as np
finally:
    del os.environ["OMP_NUM_THREADS"]
    del os.environ["NUMEXPR_NUM_THREADS"]
    del os.environ["MKL_NUM_THREADS"]

import utils


def oddEvenSort(x):
    is_sorted = False
    while not is_sorted:
        is_sorted = True
        for i in range(0, len(x) - 1, 2):
            if x[i] > x[i + 1]:
                x[i], x[i + 1] = x[i + 1], x[i]
                is_sorted = False
        for i in range(1, len(x) - 1, 2):
            if x[i] > x[i + 1]:
                x[i], x[i + 1] = x[i + 1], x[i]
                is_sorted = False
    return x


print(oddEvenSort(np.load(utils.get_numbers_file())))
