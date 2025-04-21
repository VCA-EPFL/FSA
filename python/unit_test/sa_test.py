from pyverilator import PyVerilator
import numpy as np
import argparse

class SignalWrapper:
    def __init__(self, sim: PyVerilator, index: int):
        self.sim = sim
        self.index = index

    def reset(self):
        raise NotImplementedError(f"Reset not implemented for {self.__class__.__name__}")

    def _to_signal_name(self, name: str) -> str:
        raise NotImplementedError(f"Signal name mapping not implemented for {self.__class__.__name__}")

    def __getattr__(self, name):
        if name in ['sim', 'index']:
            return super().__getattr__(name)
        else:
            return self.sim._read(self._to_signal_name(name))
    
    def __setattr__(self, name, value):
        if name in ['sim', 'index']:
            super().__setattr__(name, value)
        else:
            self.sim._write(self._to_signal_name(name), value)

class PEData(SignalWrapper):
    data: int
    def __init__(self, sim: PyVerilator, index: int):
        super().__init__(sim, index)
    
    def reset(self):
        self.data = 0
    
    def _to_signal_name(self, name: str) -> str:
        if name == 'data':
            return f"io_pe_data_{self.index}"
        else:
            raise ValueError(f"Invalid signal name: {name}")


class CmpCtrl(SignalWrapper):
    UPDATE_NEW_MAX = 0
    PROP_NEW_MAX = 1
    PROP_DIFF = 2
    PROP_ZERO = 3

    valid: int
    cmd: int

    def __init__(self, sim: PyVerilator, index: int):
        super().__init__(sim, index)
        assert index == 0, "CmpCtrl index must be 0"
    
    def reset(self):
        self.valid = 0
        self.cmd = 0
    
    def _to_signal_name(self, name: str) -> str:
        if name == 'valid':
            return f"io_cmp_ctrl_valid"
        else:
            return f"io_cmp_ctrl_bits_{name}"


class PECtrl(SignalWrapper):
    valid: int
    mac: int
    acc_ui: int
    load_reg_li: int
    load_reg_ui: int
    flow_lr: int
    flow_ud: int
    flow_du: int
    update_reg: int
    exp2: int

    def __init__(self, sim: PyVerilator, index: int):
        super().__init__(sim, index)
    
    def reset(self):
        self.valid = 0
        self.mac = 0
        self.acc_ui = 0
        self.load_reg_li = 0
        self.load_reg_ui = 0
        self.flow_lr = 0
        self.flow_ud = 0
        self.flow_du = 0
        self.update_reg = 0
        self.exp2 = 0
        
    
    def _to_signal_name(self, name: str) -> str:
        if name == 'valid':
            return f"io_pe_ctrl_{self.index}_valid"
        else:
            return f"io_pe_ctrl_{self.index}_bits_{name}"

class AccOut(SignalWrapper):
    valid: int
    data: int
    def __init__(self, sim: PyVerilator, index: int):
        super().__init__(sim, index)
    
    def _to_signal_name(self, name):
        if name == 'valid':
            return f"io_acc_out_{self.index}_valid"
        elif name == 'data':
            return f"io_acc_out_{self.index}_bits"
        else:
            raise ValueError(f"Invalid signal name: {name}")


class SystolicArray:
    def __init__(self, sim: PyVerilator, dim: int, elem_width: int, acc_width: int):
        self.sim = sim
        self.dim = dim
        self.elem_width = elem_width
        self.acc_width = acc_width
        self.cmp_ctrl = CmpCtrl(sim, 0)
        self.pe_ctrl: list[PECtrl] = [
            PECtrl(sim, i) for i in range(dim)
        ]
        self.pe_data: list[PEData] = [
            PEData(sim, i) for i in range(dim)
        ]
        self.acc_out: list[AccOut] = [
            AccOut(sim, i) for i in range(dim)
        ]

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

def test_attention_int16(sa: SystolicArray):
    sim = sa.sim
    sim.start_vcd_trace(PyVerilator.default_vcd_filename)
    sim.io.reset = 1
    sim.clock.tick()

    np.random.seed(0)
    Q = np.random.randint(-100, 100, size=(sa.dim, sa.dim), dtype=np.int16)
    K = np.random.randint(-100, 100, size=(sa.dim, sa.dim), dtype=np.int16)
    V = np.random.randint(-100, 100, size=(sa.dim, sa.dim), dtype=np.int16)
    S = np.matmul(Q, K)
    print_hex("Q", Q)
    print_hex("Q_transpose", Q.T)
    print_hex("K", K)
    print_hex("S", S)
    K_padded = pad_matrix_tri_ul_lr(K)
    print_hex("K_padded", K_padded)
    S_row_max = np.max(S, axis=1)
    print_hex("S_row_max", S_row_max)
    delta_m = np.iinfo(np.int16).min - S_row_max
    print_hex("delta_m", delta_m)
    S_minus_row_max = S - S.max(axis=1, keepdims=True)
    print_hex("S_minus_row_max", S_minus_row_max)
    exp_s1 = 2 * S_minus_row_max
    print_hex("exp_stage_1", exp_s1)
    P = 1 + exp_s1
    print_hex("P = exp_stage_2", P)
    exp_sum = np.sum(P, axis=1)
    print_hex("exp_sum", exp_sum)
    O = np.matmul(P, V)
    print_hex("V", V)
    V_padded = pad_matrix_tri_ur_ll(reverse_matrix_vertically(V))
    print_hex("V_padded", V_padded)
    print_hex("O", O)

    sim.io.reset = 0

    dut_O_cols = [[] for _ in range(sa.dim)]
    dut_delta_m = []
    dut_exp_sum = []

    for cycle in range(7 * sa.dim + 10):
        sa.cmp_ctrl.reset()
        [x.reset() for x in sa.pe_ctrl]
        [x.reset() for x in sa.pe_data]
        # Load Q
        if cycle < sa.dim:
            for i in range(sa.dim):
                sa.pe_data[i].data = Q[sa.dim - 1 - cycle, i]
                sa.pe_ctrl[i].valid = 1
                sa.pe_ctrl[i].load_reg_li = 1

        # Mul K * transpose(Q)
        for i in range(sa.dim):
            row = sa.dim - 1 - i
            if sa.dim + i <= cycle < 2 * sa.dim + i:
                sa.pe_data[row].data = K_padded[row, cycle - sa.dim]
                sa.pe_ctrl[row].valid = 1
                sa.pe_ctrl[row].mac = 1
                sa.pe_ctrl[row].acc_ui = 0
                sa.pe_ctrl[row].flow_lr = 1

        # Update new max, re-put S back
        if 2 * sa.dim <= cycle < 3 * sa.dim:
            sa.cmp_ctrl.valid = 1
            sa.cmp_ctrl.cmd = CmpCtrl.UPDATE_NEW_MAX
        
        # Propagate new max down
        if cycle == 3 * sa.dim:
            sa.cmp_ctrl.valid = 1
            sa.cmp_ctrl.cmd = CmpCtrl.PROP_NEW_MAX

        # Propagate old_max - new_max down
        if cycle == 3 * sa.dim + 1:
            sa.cmp_ctrl.valid = 1
            sa.cmp_ctrl.cmd = CmpCtrl.PROP_DIFF
        
        # Propagate zero down from CMP
        if cycle == 3 * sa.dim + 2:
            sa.cmp_ctrl.valid = 1
            sa.cmp_ctrl.cmd = CmpCtrl.PROP_ZERO

        # Propagate zero up from bottom for x = x*log2(e) + 0
        for i in range(sa.dim):
            row = sa.dim - 1 - i
            if 2 * sa.dim + 3 + i <= cycle < 3 * sa.dim + 3 + i:
                sa.pe_ctrl[row].valid = 1
                sa.pe_ctrl[row].flow_du = 1

        # Re-put S back
        for row in range(sa.dim):
            if 2 * sa.dim + 1 + row <= cycle < 3 * sa.dim + 1 + row:
                sa.pe_ctrl[row].valid = 1
                sa.pe_ctrl[row].flow_ud = 1
            if cycle == 3 * sa.dim:
                sa.pe_ctrl[row].valid = 1
                sa.pe_ctrl[row].load_reg_ui = 1
        
            # subtract row_max
            if cycle == 3 * sa.dim + 1 + row:
                sa.pe_ctrl[row].valid = 1
                sa.pe_ctrl[row].update_reg = 1
                sa.pe_ctrl[row].acc_ui = 1
                sa.pe_ctrl[row].flow_ud = 1 # pass down new_max
                sa.pe_ctrl[row].flow_lr = 1
                # s = s * 1 + (-row_max)
                sa.pe_data[row].data = 1
            
            # 1. pass down old_max - new_max
            # 2. compute exp stage 1: x = x * log2(e)
            if cycle == 3 * sa.dim + 2 + row:
                sa.pe_ctrl[row].valid = 1
                sa.pe_ctrl[row].flow_ud = 1 # pass down old_max - new_max
                sa.pe_ctrl[row].update_reg = 1 # compute exp stage 1
                sa.pe_ctrl[row].acc_ui = 0
                sa.pe_ctrl[row].flow_lr = 1
                sa.pe_data[row].data = 2 # (use log2(e) = 2 for testing)
            # exp stage 2: x = pow2(x)
            if cycle == 3 * sa.dim + 3 + row:
                sa.pe_ctrl[row].valid = 1
                sa.pe_ctrl[row].exp2 = 1
            # exp sum
            if cycle == 3 * sa.dim + 4 + row:
                sa.pe_ctrl[row].valid = 1
                sa.pe_ctrl[row].mac = 1
                sa.pe_ctrl[row].acc_ui = 1
                sa.pe_ctrl[row].flow_lr = 1
                sa.pe_data[row].data = 1 # x = x * 1 + 0

            # O = P @ V
            if 3 * sa.dim + 5 + row <= cycle < 4 * sa.dim + 5 + row:
                sa.pe_ctrl[row].valid = 1
                sa.pe_ctrl[row].mac = 1
                sa.pe_ctrl[row].acc_ui = 1
                sa.pe_ctrl[row].flow_lr = 1
                sa.pe_data[row].data = V_padded[row, cycle - (3 * sa.dim + 5)]

        
        for col in range(sa.dim):
            if 4 * sa.dim + 2 + col == cycle:
                dut_delta_m.append(sa.acc_out[col].data)
            if 4 * sa.dim + 4 + col == cycle:
                dut_exp_sum.append(sa.acc_out[col].data)
            if 4 * sa.dim + 5 + col <= cycle < 5 * sa.dim + 5 + col:
                dut_O_cols[col].append(sa.acc_out[col].data)

        sim.clock.tick()
    print("-----------------------")
    dut_delta_m = np.array(dut_delta_m, dtype=np.uint16)
    dut_exp_sum = np.array(dut_exp_sum, dtype=np.uint16)
    dut_O_cols = np.array(dut_O_cols, dtype=np.uint16)

    print_hex("delta_m (from dut)", dut_delta_m)
    print_hex("exp_sum (from dut)", dut_exp_sum)
    print_hex("O (from dut)", dut_O_cols)

    assert np.array_equal(dut_delta_m, delta_m.astype(np.uint16))
    assert np.array_equal(dut_exp_sum, exp_sum.astype(np.uint16))
    assert np.array_equal(dut_O_cols, O.astype(np.uint16))

    print("PASSED")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-file", type=str, default='SystolicArray.sv')
    parser.add_argument("--src-dir", type=str, default="../build")
    parser.add_argument("--build-dir", type=str, default="../build")
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--elem-width", type=int, default=16)
    parser.add_argument("--acc-width", type=int, default=16)
    args = parser.parse_args()
    sim = PyVerilator.build(args.top_file, [args.src_dir], build_dir=args.build_dir)
    sa = SystolicArray(sim, args.dim, args.elem_width, args.acc_width)
    test_attention_int16(sa)
    