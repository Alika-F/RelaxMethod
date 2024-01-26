from __future__ import annotations
from mpi4py import MPI
import numpy as np
import time
from dataclasses import dataclass

@dataclass
class Input:
    matrix: np.ndarray
    b: np.ndarray
    correct: np.ndarray


@dataclass
class AlgParams:
    w: float
    stop_epsilon: float 
    max_iter: int


def read_data(filename) -> Input:
    arr = np.genfromtxt(filename)
    size = int(arr[0])

    matrix = arr[1: size*size + 1].reshape((size, size))
    bc = arr[size*size + 1:]
    l = int(len(bc) / 2)
    b = bc[: l]
    correct = bc[l:]

    return Input(matrix, b, correct)


###
def calc_relax(input: Input, params: AlgParams):
    global comm, rank, n_nodes

    if rank == 0:
        current_x = np.zeros(input.matrix.shape[0])
    else:
        current_x = np.zeros((1,))

    iter = 0

    while True:
        last_epsilon = update_x(current_x, input, params)
        iter += 1

        can_stop = comm.bcast(check_stop(last_epsilon, params.stop_epsilon, iter, params.max_iter), root=0)
        if can_stop:
            break
        
    return current_x


def update_x(x, input: Input, params: AlgParams) -> float:
    ll = comm.bcast(len(x), root=0)

    epsilons = np.zeros_like(x)
    for i in range(ll):
        if rank == 0:
            xi = x[i]
            x[i] = 0
        mul_res = multiply(input.matrix[min(len(input.matrix) - 1, i)], x)
        if rank == 0:
            new_xi = (input.b[i] - mul_res) * params.w / input.matrix[i, i] + (1 - params.w) * xi
            epsilons[i] = new_xi - xi
            x[i] = new_xi

    return np.linalg.norm(epsilons)


def check_stop(last_epsilon, stop_epsilon, iter, max_iter):
    return last_epsilon <= stop_epsilon or iter >= max_iter


def multiply(m_i: np.ndarray, x: np.ndarray):
    global rank, n_nodes, comm
    sum_len = comm.bcast(len(x), root=0)
    my_len = int(sum_len / n_nodes)

    my_matrix = np.empty(my_len, dtype='d')
    my_x = np.empty(my_len, dtype='d')

    comm.Scatter(m_i, my_matrix, root=0, )
    comm.Scatter(x, my_x, root=0, )

    res = (my_matrix * my_x).sum()
    sums = np.empty(n_nodes, dtype='d')
    comm.Gather(res, sums, root=0)

    return sums.sum()

###
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_nodes = comm.Get_size()


def run_test(test_name, nrepeats=1):
    if rank == 0:
        inp = read_data(test_name)
    else:
        inp = Input(np.zeros((1, 1)), np.zeros((1, )), np.zeros((1, )))

    params = AlgParams(w=1.1, stop_epsilon=0.00001, max_iter=10000)

    sum_time = 0
    for i in range(nrepeats):
        start = time.time()
        res = calc_relax(inp, params)
        end = time.time()
        sum_time += end - start

    if rank == 0:
        print(test_name, sum_time / nrepeats, flush=True)
        


def main():
    test_paths = [
        './tests/test0',
        './tests/test1',
        './tests/test2',
        './tests/test3',
        './tests/test4',
        
    ]
    for path in test_paths:
        run_test(path, 10)


if __name__ == '__main__':
    main()
