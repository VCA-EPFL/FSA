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
        return np.array([
            [np.uint32(round_raw_float(x.to_raw(), 8, 23).to_bits()).view(np.float32) for x in row]
            for row in mat
        ])

def np_to_fp(x: np.float16 | np.float32 | np.float64, ew: int, mw: int) -> FloatPoint:
    match x.dtype:
        case np.float16:
            np_ew, np_mw, ut = 5, 10, np.uint16
        case np.float32:
            np_ew, np_mw, ut = 8, 23, np.uint32
        case np.float64:
            np_ew, np_mw, ut = 11, 52, np.uint64
        case _:
            raise ValueError(f"unknown dtype: {x.dtype}")

    fp = FloatPoint.from_bits(x.view(ut), np_ew, np_mw)
    if (np_ew, np_mw) != (ew, mw):
        fp = round_raw_float(fp.to_raw(), ew, mw)
    return fp


def fp_to_np(x: FloatPoint) -> np.float64 | np.float32 | np.float16:
    if x.ew <= 5 and x.mw <= 10:
        target_ew, target_mw, itype, dtype = 5, 10, np.uint16, np.float16
    elif x.ew <= 8 and x.mw <= 23:
        target_ew, target_mw, itype, dtype = 8, 23, np.uint32, np.float32
    elif x.ew <= 11 and x.mw <= 52:
        target_ew, target_mw, itype, dtype = 11, 52, np.uint64, np.float64
    else:
        raise ValueError(f"Can not convert x({x.ew}, {x.mw}) to numpy float")
    return itype(round_raw_float(x.to_raw(), target_ew, target_mw).to_bits()).view(dtype)


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

    Q: Matrix # [Br, d]
    K: Matrix # [Bc, d]
    V: Matrix # [Bc, d]

    # S = Q @ K.T [Br, Bc]
    S: Matrix
    S_low_precision: Matrix

    PrevRowMax: Matrix # [Br, 1]
    RowMaxS: Matrix # [Br, 1]
    AccRowMaxS: Matrix # [Br, 1]
    NegRowMaxS: Matrix # [Br, 1]
    # RowMax(i-1) - RowMax(i)
    DeltaRowMax: Matrix # [Br, 1]
    ExpDeltaRowMaxS1: Matrix # [Br, 1]
    ExpDeltaRowMaxS2: Matrix #[Br, 1]

    SMinusRowMax: Matrix # [Br, Bc]
    SExpStage1: Matrix # [Br, Bc]
    P: Matrix # [Br, Bc]
    RowSum: Matrix # [Br, 1]

    O: Matrix # [Br, d]

    AccRowSum: Matrix # [Br, 1]
    AccRowSumReciprocal: Matrix # 1 / AccRowSum
    AccO: Matrix # [Br, d]
    NormO: Matrix



    def __init__(self,
                 Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                 PrevRowMax: np.ndarray | Matrix,
                 PrevRowSum: np.ndarray | Matrix,
                 PrevO: np.ndarray | Matrix,
                 mul_ew: int, mul_mw: int, acc_ew: int, acc_mw: int,
                 backend: BaseFPBackend
                ):
        self.backend = backend
        self.mul_ew = mul_ew
        self.mul_mw = mul_mw
        self.acc_ew = acc_ew
        self.acc_mw = acc_mw
        self.Q = build_mat_from_numpy(Q, mul_ew, mul_mw)
        self.K = build_mat_from_numpy(K, mul_ew, mul_mw)
        self.V = build_mat_from_numpy(V, mul_ew, mul_mw)

        if isinstance(PrevRowMax, list):
            self.PrevRowMax = PrevRowMax
        else:
            self.PrevRowMax = build_mat_from_numpy(PrevRowMax, acc_ew, acc_mw)
        if isinstance(PrevRowSum, list):
            self.AccRowSum = PrevRowSum
        else:
            self.AccRowSum = build_mat_from_numpy(PrevRowSum, acc_ew, acc_mw)
        if isinstance(PrevO, list):
            self.AccO = PrevO
        else:
            self.AccO = build_mat_from_numpy(PrevO, acc_ew, acc_mw)
        br, d, bc = len(Q), len(Q[0]), len(K)
        # [Br, Bc]
        self.S = [[FloatPoint.from_bits(0, acc_ew, acc_mw) for _ in range(bc)] for _ in range(br)]
        # [Br, d]
        self.O = [[FloatPoint.from_bits(0, acc_ew, acc_mw) for _ in range(d)] for _ in range(br)]
        self.__mul_qk()
        self.RowMaxS = [[self.__max(self.S[row] + self.PrevRowMax[row])] for row in range(br)]
        self.NegRowMaxS = [[neg_fp(x) for x in row] for row in self.RowMaxS]
        self.DeltaRowMax = [[self.__sub(self.PrevRowMax[row][0], self.RowMaxS[row][0])] for row in range(br)]
        self.AccRowMaxS = [[
            self.RowMaxS[row][0] if self.DeltaRowMax[row][0].sign else self.PrevRowMax[row][0]
        ] for row in range(br)]
        # e^((m0 - m1)/sqrt(dk)) = 2^((m0 - m1) * log2(e) / sqrt(dk))
        self.ExpDeltaRowMaxS1 = [[self.backend.fma(row[0],
                                                   np_to_fp(np.log2(np.e) / np.sqrt(d), acc_ew, acc_mw),
                                                   np_to_fp(np.float32(0), acc_ew, acc_mw),
                                                   acc_ew, acc_mw
                                                   )] for row in self.DeltaRowMax]
        self.ExpDeltaRowMaxS2 = [[self.backend.exp2(
            row[0], acc_ew, acc_mw, acc_ew, acc_mw, acc_ew, acc_mw
            )] for row in self.ExpDeltaRowMaxS1]

        self.RowSum = [[np_to_fp(np.float32(0), acc_ew, acc_mw)] for row in range(br)]

        self.S_low_precision = [[round_raw_float(x.to_raw(), mul_ew, mul_mw)
                                 for x in row] for row in self.S]

        self.SMinusRowMax = [
            [self.__sub(self.S_low_precision[row][col], self.RowMaxS[row][0]) for col in range(bc)]
            for row in range(br)
        ]

        # e^(x/sqrt(dk)) = 2^(x * log2(e) / sqrt(dk))
        self.SExpStage1 = [
            [self.backend.fma(
                self.SMinusRowMax[row][col],
                np_to_fp(np.log2(np.e) / np.sqrt(d), mul_ew, mul_mw),
                np_to_fp(np.float32(0), acc_ew, acc_mw),
                mul_ew, mul_mw
            ) for col in range(bc)]
            for row in range(br)
        ]

        self.P = [
            [ self.backend.exp2(
                self.SExpStage1[row][col],
                mul_ew, mul_mw, mul_ew, mul_mw, acc_ew, acc_mw
                ) for col in range(bc)]
            for row in range(br)
        ]
        for row in range(br):
            for col in range(bc):
                self.RowSum[row][0] = self.backend.fma(
                    self.P[row][col],
                    np_to_fp(np.float32(1), mul_ew, mul_mw),
                    self.RowSum[row][0],
                    acc_ew, acc_mw
                )
        self.__mul_pv()
        self.__update_global()


    def __sub(self, a: FloatPoint, b: FloatPoint) -> FloatPoint:
        """a - b"""
        one = np_to_fp(np.float64(1), a.ew, a.mw)
        return self.backend.fma(a, one, neg_fp(b), a.ew, a.mw)

    def __max(self, row: list[FloatPoint]) -> FloatPoint:
        m = row[0]
        for e in row[1:]:
            # m - e
            diff = self.__sub(m, e)
            if diff.sign:
                m = e
        return m

    def __mul_qk(self):
        # [Br, d] @ [Bc, d].T => [Br, Bc]
        br, dq = len(self.Q), len(self.Q[0])
        bc, dk = len(self.K), len(self.K[0])
        assert dq == dk
        for row in range(br):
            for col in range(bc):
                for d in reversed(range(dq)):
                    k = self.K[col][d]
                    q = self.Q[row][d]
                    s = self.S[row][col]
                    self.S[row][col] = self.backend.fma(k, q, s, s.ew, s.mw)

    def __mul_pv(self):
        br, bc_p = len(self.P), len(self.P[0])
        bc_v, d = len(self.V), len(self.V[0])
        assert bc_p == bc_v
        assert bc_p <= d

        for row in range(br):
            for col in range(d):
                for i in reversed(range(bc_p)):
                    v = self.V[i][col]
                    p = self.P[row][i]
                    o = self.O[row][col]
                    self.O[row][col] = self.backend.fma(p, v, o, o.ew, o.mw)

    def __update_global(self):
        self.AccRowSumReciprocal = []
        self.NormO = []
        for row in range(len(self.RowSum)):
            old_d = self.AccRowSum[row][0]
            new_d = self.RowSum[row][0]
            scale = self.ExpDeltaRowMaxS2[row][0]
            self.AccRowSum[row][0] = self.backend.fma(old_d, scale, new_d, old_d.ew, old_d.mw)
            accRowSumReciprocal = self.backend.reciprocal(self.AccRowSum[row][0])
            self.AccRowSumReciprocal.append([accRowSumReciprocal])
            normORow = []

            for col in range(len(self.O[0])):
                old_o = self.AccO[row][col]
                new_o = self.O[row][col]
                self.AccO[row][col] = self.backend.fma(old_o, scale, new_o, old_o.ew, old_o.mw)
                normO = self.backend.fma(
                    self.AccO[row][col], accRowSumReciprocal, FloatPoint.from_bits(0, self.acc_ew, self.acc_mw),
                    old_o.ew, old_o.mw
                )

                normORow.append(normO)
            self.NormO.append(normORow)


    def __str__(self):
        return f"""
Q hex:
{mat_hex_str(self.Q)}
Q float:
{str(mat_to_numpy_array(self.Q))}

K hex:
{mat_hex_str(self.K)}
K float:
{str(mat_to_numpy_array(self.K))}

S hex:
{mat_hex_str(self.S)}
S float:
{str(mat_to_numpy_array(self.S))}

PrevRowMax hex:
{mat_hex_str(self.PrevRowMax)}
PrevRowMax float:
{str(mat_to_numpy_array(self.PrevRowMax))}

RowMaxS hex:
{mat_hex_str(self.RowMaxS)}
RowMaxS float:
{str(mat_to_numpy_array(self.RowMaxS))}

-RowMaxS hex:
{mat_hex_str(self.NegRowMaxS)}
-RowMaxS float:
{str(mat_to_numpy_array(self.NegRowMaxS))}

DeltaRowMax hex:
{mat_hex_str(self.DeltaRowMax)}
DeltaRowMax float:
{str(mat_to_numpy_array(self.DeltaRowMax))}

ExpDeltaRowMaxS1 hex:
{mat_hex_str(self.ExpDeltaRowMaxS1)}
ExpDeltaRowMaxS1 float:
{str(mat_to_numpy_array(self.ExpDeltaRowMaxS1))}

ExpDeltaRowMaxS2 hex:
{mat_hex_str(self.ExpDeltaRowMaxS2)}
ExpDeltaRowMaxS2 float:
{str(mat_to_numpy_array(self.ExpDeltaRowMaxS2))}

SMinusRowMax hex:
{mat_hex_str(self.SMinusRowMax)}
SMinusRowMax float:
{str(mat_to_numpy_array(self.SMinusRowMax))}

SExpS1 hex:
{mat_hex_str(self.SExpStage1)}
SExpS1 float:
{str(mat_to_numpy_array(self.SExpStage1))}

P hex:
{mat_hex_str(self.P)}
P float:
{str(mat_to_numpy_array(self.P))}

RowSum hex:
{mat_hex_str(self.RowSum)}
RowSum float:
{str(mat_to_numpy_array(self.RowSum))}


O hex:
{mat_hex_str(self.O)}
O float:
{str(mat_to_numpy_array(self.O))}

AccRowSum hex:
{mat_hex_str(self.AccRowSum)}
AccRowSum float:
{str(mat_to_numpy_array(self.AccRowSum))}

AccRowSumReciprocal hex:
{mat_hex_str(self.AccRowSumReciprocal)}
AccRowSumReciprocal float:
{str(mat_to_numpy_array(self.AccRowSumReciprocal))}

AccO hex:
{mat_hex_str(self.AccO)}
AccO float:
{str(mat_to_numpy_array(self.AccO))}

NormO hex:
{mat_hex_str(self.NormO)}
NormO float:
{str(mat_to_numpy_array(self.NormO))}
"""