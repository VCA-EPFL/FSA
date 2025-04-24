import numpy as np

def print_hex_vec(vec: np.ndarray):
    assert len(vec.shape) == 1
    print("  [" + "  ".join(f"{np.uint16(x):04x}" for x in vec) + "]")

def print_hex(name: str, matrix: np.ndarray):
    print(f"{name} (shape {matrix.shape}):")
    if len(matrix.shape) == 2:
        for row in matrix:
            print_hex_vec(row)
        print()
    elif len(matrix.shape) == 1:
        print_hex_vec(matrix)
        print()

def pad_matrix_tri_ur_ll(matrix: np.ndarray) -> np.ndarray:
    """
    x x  ->  x x 0
    x x      0 x x
    """
    ret = []
    for row in range(matrix.shape[0]):
        ret.append([0 for _ in range(row)] + list(matrix[row]) + [0 for _ in range(matrix.shape[0] - row - 1)])
    return np.array(ret, dtype=matrix.dtype)

def pad_matrix_tri_ul_lr(matrix: np.ndarray) -> np.ndarray:
    """
    x x   -> 0 x x
    x x      x x 0
    """
    ret = []
    for row in range(matrix.shape[0]):
        ret.append([0 for _ in range(matrix.shape[0] - row - 1)] + list(matrix[row]) + [0 for _ in range(row)])
    return np.array(ret, dtype=matrix.dtype)

def reverse_matrix_horizontally(matrix: np.ndarray) -> np.ndarray:
    """
    x y  ->  y x
    m n      n m
    """
    ret = []
    for row in matrix:
        ret.append(list(reversed(row)))
    return np.array(ret, dtype=matrix.dtype)

def reverse_matrix_vertically(matrix: np.ndarray) -> np.ndarray:
    """
    x y  ->  m n
    m n      x y
    """
    ret = []
    for row in reversed(matrix):
        ret.append(list(row))
    return np.array(ret, dtype=matrix.dtype)

class Int16Data:
    def __init__(self, dim: int, range: tuple, seed: int = 0):
        self.dim = dim
        np.random.seed(seed)
        self.Q = np.random.randint(
            *range, size=(dim, dim), dtype=np.int16
        )
        self.K = np.random.randint(
            *range, size=(dim, dim), dtype=np.int16
        )
        self.V = np.random.randint(
            *range, size=(dim, dim), dtype=np.int16
        )

        self.S = np.matmul(self.Q, self.K)
        self.K_padded = pad_matrix_tri_ul_lr(self.K)
        self.S_row_max = np.max(self.S, axis=1)
        self.delta_m = np.iinfo(np.int16).min - self.S_row_max
        self.S_minus_row_max = self.S - self.S.max(axis=1, keepdims=True)
        self.exp_s1 = 2 * self.S_minus_row_max
        self.P = 1 + self.exp_s1
        self.exp_sum = np.sum(self.P, axis=1)
        self.O = np.matmul(self.P, self.V)
        self.V_padded = pad_matrix_tri_ur_ll(reverse_matrix_vertically(self.V))
    
    def print_data(self):
        print_hex("Q", self.Q)
        print_hex("Q_transpose", self.Q.T)
        print_hex("K", self.K)
        print_hex("S", self.S)
        print_hex("K_padded", self.K_padded)
        print_hex("S_row_max", self.S_row_max)
        print_hex("-S_row_max", -self.S_row_max)
        print_hex("delta_m", self.delta_m)
        print_hex("S_minus_row_max", self.S_minus_row_max)
        print_hex("exp_stage_1", self.exp_s1)
        print_hex("P = exp_stage_2", self.P)
        print_hex("exp_sum", self.exp_sum)
        print_hex("V", self.V)
        print_hex("V_padded", self.V_padded)
        print_hex("O", self.O)
         