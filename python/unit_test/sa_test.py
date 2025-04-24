import numpy as np
import argparse
from pyverilator import PyVerilator
from signal_wrapper import SignalWrapper
from data import *

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



def test_attention_int16(sa: SystolicArray):
    sim = sa.sim
    sim.start_vcd_trace(PyVerilator.default_vcd_filename)
    sim.io.reset = 1
    sim.clock.tick()

    data = Int16Data(sa.dim, (-100, 100), seed=0)

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
                sa.pe_data[i].data = data.Q[sa.dim - 1 - cycle, i]
                sa.pe_ctrl[i].valid = 1
                sa.pe_ctrl[i].load_reg_li = 1

        # Mul K * transpose(Q)
        for i in range(sa.dim):
            row = sa.dim - 1 - i
            if sa.dim + i <= cycle < 2 * sa.dim + i:
                sa.pe_data[row].data = data.K_padded[row, cycle - sa.dim]
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
                sa.pe_data[row].data = data.V_padded[row, cycle - (3 * sa.dim + 5)]

        
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

    data.print_data()
    print_hex("delta_m (from dut)", dut_delta_m)
    print_hex("exp_sum (from dut)", dut_exp_sum)
    print_hex("O (from dut)", dut_O_cols)

    assert np.array_equal(dut_delta_m, data.delta_m.astype(np.uint16))
    assert np.array_equal(dut_exp_sum, data.exp_sum.astype(np.uint16))
    assert np.array_equal(dut_O_cols, data.O.astype(np.uint16))

    print("PASSED")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-file", type=str, default='SystolicArray.sv')
    parser.add_argument("--src-dir", type=str, default="../build/systolic_array")
    parser.add_argument("--build-dir", type=str, default="../build/systolic_array")
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--elem-width", type=int, default=16)
    parser.add_argument("--acc-width", type=int, default=16)
    args = parser.parse_args()
    sim = PyVerilator.build(args.top_file, [args.src_dir], build_dir=args.build_dir)
    sa = SystolicArray(sim, args.dim, args.elem_width, args.acc_width)
    test_attention_int16(sa)
    