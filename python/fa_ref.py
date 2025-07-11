import numpy as np
from pyeasyfloat.float import FloatPoint
from pyeasyfloat.rounding import round_raw_float
from pyeasyfloat.backend import BaseFPBackend, PyEasyFloatBackend

type Matrix = list[list[FloatPoint]]

def mat_hex_str(mat: Matrix) -> str:
        s = ''
        for row in mat:
            for e in row:
                l = (1 + e.ew + e.mw + 1) // 4
                s += format(e.to_bits(), f"0{l}x")
                s += ' '
            s += '\n'
        return s

def mat_to_numpy_array(mat: Matrix) -> np.ndarray:
        return np.array([[x.to_numpy() for x in row] for row in mat])

def np_to_fp(x: np.float16 | np.float32 | np.float64 | float, ew: int, mw: int) -> FloatPoint:
    if isinstance(x, float):
        x = np.float64(x)
    fp = FloatPoint.from_numpy(x)
    np_ew, np_mw = fp.ew, fp.mw
    if (np_ew, np_mw) != (ew, mw):
        fp = round_raw_float(fp.to_raw(), ew, mw)
    return fp

def fp_to_np(x: FloatPoint) -> np.float64 | np.float32 | np.float16:
    return x.to_numpy()

def build_mat_from_numpy(arr: np.ndarray, ew: int, mw: int) -> Matrix:
        return [[np_to_fp(x, ew, mw) for x in row] for row in arr]

def neg_fp(x: FloatPoint) -> FloatPoint:
    nx = FloatPoint(x.ew, x.mw)
    nx.sign = not x.sign
    nx.exp = x.exp
    nx.mantissa = x.mantissa
    return nx

class FlashAttentionTile:
    backend: BaseFPBackend

    Q: Matrix  # [Br, d]
    K: Matrix  # [Bc, d]
    V: Matrix  # [Bc, d]

    S: Matrix
    S_low_precision: Matrix

    PrevRowMax: Matrix  # [Br, 1]
    RowMaxS: Matrix      # [Br, 1]
    AccRowMaxS: Matrix   # [Br, 1]
    NegRowMaxS: Matrix   # [Br, 1]
    DeltaRowMax: Matrix  # [Br, 1]
    ExpDeltaRowMaxS1: Matrix
    ExpDeltaRowMaxS2: Matrix

    SMinusRowMax: Matrix
    SExpStage1: Matrix
    P: Matrix
    RowSum: Matrix

    O: Matrix
    AccRowSum: Matrix
    AccRowSumReciprocal: Matrix
    AccO: Matrix
    NormO: Matrix

    def __init__(
        self,
        Q: np.ndarray, K: np.ndarray, V: np.ndarray,
        PrevRowMax: np.ndarray | Matrix,
        PrevRowSum: np.ndarray | Matrix,
        PrevO: np.ndarray | Matrix,
        mul_ew: int, mul_mw: int,
        acc_ew: int, acc_mw: int,
        backend: BaseFPBackend
    ):
        self.backend = backend
        self.mul_ew, self.mul_mw = mul_ew, mul_mw
        self.acc_ew, self.acc_mw = acc_ew, acc_mw

        self.Q = build_mat_from_numpy(Q, mul_ew, mul_mw)
        self.K = build_mat_from_numpy(K, mul_ew, mul_mw)
        self.V = build_mat_from_numpy(V, mul_ew, mul_mw)

        self.PrevRowMax = PrevRowMax if isinstance(PrevRowMax, list) else build_mat_from_numpy(PrevRowMax, acc_ew, acc_mw)
        self.AccRowSum = PrevRowSum if isinstance(PrevRowSum, list) else build_mat_from_numpy(PrevRowSum, acc_ew, acc_mw)
        self.AccO = PrevO if isinstance(PrevO, list) else build_mat_from_numpy(PrevO, acc_ew, acc_mw)

        br, d, bc = len(Q), len(Q[0]), len(K)
        self.S = [[FloatPoint.from_bits(0, acc_ew, acc_mw) for _ in range(bc)] for _ in range(br)]
        self.O = [[FloatPoint.from_bits(0, acc_ew, acc_mw) for _ in range(d)] for _ in range(br)]

        self.__mul_qk()

        self.RowMaxS = [[self.__max(self.S[row] + self.PrevRowMax[row])] for row in range(br)]
        self.NegRowMaxS = [[neg_fp(x) for x in row] for row in self.RowMaxS]
        self.DeltaRowMax = [[self.__sub(self.PrevRowMax[row][0], self.RowMaxS[row][0])] for row in range(br)]

        self.AccRowMaxS = [[
            self.RowMaxS[row][0] if self.DeltaRowMax[row][0].sign else self.PrevRowMax[row][0]
        ] for row in range(br)]

        log2e_over_sqrt_d = np_to_fp(np.log2(np.e) / np.sqrt(d), acc_ew, acc_mw)
        zero_fp = np_to_fp(np.float32(0), acc_ew, acc_mw)

        self.ExpDeltaRowMaxS1 = [[
            self.backend.fma(row[0], log2e_over_sqrt_d, zero_fp, acc_ew, acc_mw)
        ] for row in self.DeltaRowMax]

        self.ExpDeltaRowMaxS2 = [[
            self.backend.exp2(row[0], acc_ew, acc_mw, acc_ew, acc_mw, acc_ew, acc_mw)
        ] for row in self.ExpDeltaRowMaxS1]

        self.RowSum = [[np_to_fp(np.float32(0), acc_ew, acc_mw)] for _ in range(br)]

        self.S_low_precision = [
            [round_raw_float(x.to_raw(), mul_ew, mul_mw) for x in row]
            for row in self.S
        ]

        self.SMinusRowMax = [
            [self.__sub(self.S_low_precision[row][col], self.RowMaxS[row][0]) for col in range(bc)]
            for row in range(br)
        ]

        mul_log2e = np_to_fp(np.log2(np.e) / np.sqrt(d), mul_ew, mul_mw)
        zero_acc = np_to_fp(np.float32(0), acc_ew, acc_mw)

        self.SExpStage1 = [
            [self.backend.fma(self.SMinusRowMax[row][col], mul_log2e, zero_acc, mul_ew, mul_mw)
             for col in range(bc)]
            for row in range(br)
        ]

        self.P = [
            [self.backend.exp2(self.SExpStage1[row][col], mul_ew, mul_mw, mul_ew, mul_mw, acc_ew, acc_mw)
             for col in range(bc)]
            for row in range(br)
        ]

        one_fp = np_to_fp(np.float32(1), mul_ew, mul_mw)
        for row in range(br):
            for col in range(bc):
                self.RowSum[row][0] = self.backend.fma(
                    self.P[row][col], one_fp, self.RowSum[row][0], acc_ew, acc_mw
                )

        self.__mul_pv()
        self.__update_global()

    def __sub(self, a: FloatPoint, b: FloatPoint) -> FloatPoint:
        return self.backend.fma(a, np_to_fp(1.0, a.ew, a.mw), neg_fp(b), a.ew, a.mw)

    def __max(self, row: list[FloatPoint]) -> FloatPoint:
        m = row[0]
        for e in row[1:]:
            if self.__sub(m, e).sign:
                m = e
        return m

    def __mul_qk(self):
        br, d = len(self.Q), len(self.Q[0])
        bc = len(self.K)
        for row in range(br):
            for col in range(bc):
                for k in reversed(range(d)):
                    self.S[row][col] = self.backend.fma(
                        self.K[col][k], self.Q[row][k], self.S[row][col],
                        self.S[row][col].ew, self.S[row][col].mw
                    )

    def __mul_pv(self):
        br, bc = len(self.P), len(self.P[0])
        d = len(self.V[0])
        for row in range(br):
            for col in range(d):
                for i in reversed(range(bc)):
                    self.O[row][col] = self.backend.fma(
                        self.P[row][i], self.V[i][col], self.O[row][col],
                        self.O[row][col].ew, self.O[row][col].mw
                    )

    def __update_global(self):
        self.AccRowSumReciprocal = []
        self.NormO = []

        for row in range(len(self.RowSum)):
            old_sum = self.AccRowSum[row][0]
            new_sum = self.RowSum[row][0]
            scale = self.ExpDeltaRowMaxS2[row][0]

            self.AccRowSum[row][0] = self.backend.fma(old_sum, scale, new_sum, old_sum.ew, old_sum.mw)
            reciprocal = self.backend.reciprocal(self.AccRowSum[row][0])
            self.AccRowSumReciprocal.append([reciprocal])

            norm_row = []
            for col in range(len(self.O[0])):
                old_o = self.AccO[row][col]
                new_o = self.O[row][col]
                self.AccO[row][col] = self.backend.fma(old_o, scale, new_o, old_o.ew, old_o.mw)
                norm = self.backend.fma(self.AccO[row][col], reciprocal, FloatPoint.from_bits(0, self.acc_ew, self.acc_mw),
                                        old_o.ew, old_o.mw)
                norm_row.append(norm)
            self.NormO.append(norm_row)

    def __str__(self) -> str:
        def to_str(name: str, mat: Matrix) -> str:
            return f"{name} hex:\n{mat_hex_str(mat)}{name} float:\n{mat_to_numpy_array(mat)}\n"

        return "\n".join([
            to_str("Q", self.Q),
            to_str("K", self.K),
            to_str("V", self.V),
            to_str("S", self.S),
            to_str("PrevRowMax", self.PrevRowMax),
            to_str("RowMaxS", self.RowMaxS),
            to_str("-RowMaxS", self.NegRowMaxS),
            to_str("DeltaRowMax", self.DeltaRowMax),
            to_str("ExpDeltaRowMaxS1", self.ExpDeltaRowMaxS1),
            to_str("ExpDeltaRowMaxS2", self.ExpDeltaRowMaxS2),
            to_str("SMinusRowMax", self.SMinusRowMax),
            to_str("SExpS1", self.SExpStage1),
            to_str("P", self.P),
            to_str("RowSum", self.RowSum),
            to_str("O", self.O),
            to_str("AccRowSum", self.AccRowSum),
            to_str("AccRowSumReciprocal", self.AccRowSumReciprocal),
            to_str("AccO", self.AccO),
            to_str("NormO", self.NormO),
        ])