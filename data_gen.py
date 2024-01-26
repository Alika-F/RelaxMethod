import numpy as np


def is_pos_def(x) -> bool:
    return np.all(np.linalg.eigvals(x) > 0)


def generate_sym_matrix(matrix_size: int):
    new = np.random.sample((matrix_size, matrix_size))
    for i in range(len(new)):
        new[i, i:] = new[i:, i]
        new[i, i] = (np.random.sample((1, ))[0] + 1) * matrix_size
    return new


def generate_matrix_and_vector(matrix_size: int, scale_k: float = 1) -> tuple[np.ndarray, np.ndarray]:
    new = generate_sym_matrix(matrix_size)

    while not is_pos_def(new):
        new = generate_sym_matrix(matrix_size)

    return new * scale_k, np.random.sample((matrix_size, ))


def generate_system(rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix, b = generate_matrix_and_vector(rank, 1)
    solving = np.linalg.solve(matrix, b)
    return matrix, b, solving


def write_system(path, filename, a, b, correct):
    with open(path + '/' + filename, 'w') as f:
        f.write('%d\n' % a.shape[0])
        for s in a:
            f.write('\n'.join([str(t) for t in s]) + '\n')
        f.write('\n'.join([str(t) % t for t in b]) + '\n')
        f.write('\n'.join([str(t) % t for t in correct]) + '\n')


def create_test(rank: int, test_number: int):
    filename = 'test%d' % test_number
    matrix, b, solving = generate_system(rank)
    write_system('./tests', filename, matrix, b, solving)


def create_tests(ranks: list[int], names=None):
    for i in range(len(ranks)):
        name = i if names is None else names[i]
        create_test(ranks[i], name)
        print(name)


if __name__ == '__main__':
    create_tests([100,500, 1000,5000, 10000])
