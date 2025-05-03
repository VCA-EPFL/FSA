from signal_wrapper import SignalWrapper
from pyverilator import PyVerilator
import argparse
import numpy as np
from data import *
from dataclasses import dataclass
from pyeasyfloat.backend import BaseFPBackend, PyEasyFloatBackend, HwBackend

class MatrixDesc:
    origin_addr: int
    dim: int
    rev_v: bool
    rev_h: bool
    delay_u: bool
    delay_d: bool

    def __init__(self,
                 origin_addr: int, dim: int,
                    rev_v: bool, rev_h: bool,
                    delay_u: bool, delay_d: bool):

        self.addr = origin_addr if not rev_h else origin_addr + dim - 1
        self.stride = 1 if not rev_h else -1
        self.delay = delay_u or delay_d

        self.revIn = False
        self.revOut = False
        if delay_u:
            if rev_v:
                self.revIn = True
                self.revOut = False
            else:
                self.revIn = True
                self.revOut = True
        elif delay_d:
            if rev_v:
                self.revIn = True
                self.revOut = False
            else:
                self.revIn = False
                self.revOut = False

    def to_rs(self) -> int:
        hi = (self.addr << 8) | self.stride & 0xff
        lo = (self.revIn << 31) | (self.revOut << 30) | (self.delay << 29)
        return (hi << 32) | lo

LOAD_STATIONARY = 0
ATTENTION_SCORE = 1
ATTENTION_VALUE = 2

class Instruction(SignalWrapper):
    valid: int
    ready: int
    funct7: int
    rs1: int
    rs2: int
    def __init__(self, sim: PyVerilator):
        super().__init__(sim, 0)
        self.reset()

    def reset(self):
        self.valid = 0
        self.funct7 = 0
        self.rs1 = 0
        self.rs2 = 0

    def _to_signal_name(self, name: str) -> str:
        if name in ['valid', 'ready']:
            return f"io_inst_{name}"
        else:
            return f"io_inst_bits_{name}"

    def set(self,
            func: int,
            rs1: int,
            rs2: int
        ) -> None:
        self.valid = 1
        self.funct7 = func
        self.rs1 = rs1
        self.rs2 = rs2

class SRAMAddr(SignalWrapper):
    en_sp: int
    en_acc: int
    addr: int
    read: int
    write: int
    def __init__(self, sim: PyVerilator):
        super().__init__(sim, 0)

    def reset(self):
        self.en_sp = 0
        self.en_acc = 0
        self.addr = 0
        self.read = 0
        self.write = 0

    def _to_signal_name(self, name: str) -> str:
        return f"io_debug_sram_io_{name}"


class SRAMData(SignalWrapper):
    rdata: int
    wdata: int
    def __init__(self, sim, index):
        super().__init__(sim, index)

    def reset(self):
        self.wdata = 0

    def _to_signal_name(self, name: str) -> str:
        return f"io_debug_sram_io_{name}_{self.index}"


def test_msaga(sim: PyVerilator, dim: int):
    inst = Instruction(sim)
    sram_addr = SRAMAddr(sim)
    sram_data = [SRAMData(sim, i) for i in range(dim)]

    sim.start_vcd_trace(PyVerilator.default_vcd_filename)
    sim.io.reset = 1
    sim.clock.tick()

    np.random.seed(0)

    Q = np.random.random((dim, dim)).astype(np.float32)
    K = np.random.random((dim, dim)).astype(np.float32)
    V = np.random.random((dim, dim)).astype(np.float32)
    PrevRowMax = np.full((dim, 1), np.float32(-np.inf))
    PrevRowSum = np.full((dim, 1), np.float32(0))
    PrevO = np.full((dim, dim), np.float32(0))

    tile = FlashAttentionTile(
        Q, K, V,
        PrevRowMax, PrevRowSum, PrevO,
        8, 23, 8, 23, PyEasyFloatBackend()
    )
    print(str(tile))

    sim.io.reset = 0

    for i in range(dim):
        sram_addr.en_sp = 1
        sram_addr.write = 1
        sram_addr.addr = i
        for j in range(dim):
            sram_data[j].wdata = tile.Q[i][j].to_bits()
        sim.clock.tick()

    for i in range(dim):
        sram_addr.en_sp = 1
        sram_addr.write = 1
        sram_addr.addr = dim + i
        for j in range(dim):
            sram_data[j].wdata = tile.K[i][j].to_bits()
        sim.clock.tick()

    V_t = build_mat_from_numpy(V.T, 8, 23)
    for i in range(dim):
        sram_addr.en_sp = 1
        sram_addr.write = 1
        sram_addr.addr = 2 * dim + i
        for j in range(dim):
            sram_data[j].wdata = V_t[i][j].to_bits()
        sim.clock.tick()

    sram_addr.en_sp = 0
    sram_addr.write = 0

    for i in range(dim + 1):
        sram_addr.en_acc = 1
        sram_addr.write = 1
        sram_addr.addr = i
        for j in range(dim):
            sram_data[j].wdata = 0
        sim.clock.tick()

    sram_addr.en_acc = 0
    sram_addr.write = 0

    for _ in range(10):
        sim.clock.tick()

    while not inst.ready:
        sim.clock.tick()

    Q_desc = MatrixDesc(
        origin_addr=0,
        dim=dim,
        rev_v=False, rev_h=True,
        delay_u=False, delay_d=False
    )
    while not inst.ready:
        sim.clock.tick()
    inst.set(LOAD_STATIONARY, Q_desc.to_rs(), 0)
    sim.clock.tick()
    inst.reset()

    K_desc = MatrixDesc(
        origin_addr=dim,
        dim=dim,
        rev_v=False, rev_h=False,
        delay_u=True, delay_d=False
    )
    D_desc = MatrixDesc(origin_addr=0, dim=dim, rev_v=False, rev_h=False, delay_u=False, delay_d=False)

    while not inst.ready:
        sim.clock.tick()
    inst.set(ATTENTION_SCORE, K_desc.to_rs(), D_desc.to_rs())
    sim.clock.tick()
    inst.reset()

    V_desc = MatrixDesc(
        origin_addr=2 * dim,
        dim=dim,
        rev_v=True, rev_h=False,
        delay_u=False, delay_d=True
    )
    O_desc = MatrixDesc(
        origin_addr=1,
        dim=1,
        rev_v=False, rev_h=False, delay_u=False, delay_d=False
    )
    while not inst.ready:
        sim.clock.tick()
    inst.set(ATTENTION_VALUE, V_desc.to_rs(), O_desc.to_rs())
    sim.clock.tick()
    inst.reset()


    while not inst.ready:
        sim.clock.tick()
    [sim.clock.tick() for _ in range(3 * dim)]

    dut_O = []
    for i in range(dim):
        sram_addr.en_acc = 1
        sram_addr.read = 1
        sram_addr.addr = i + 1
        sim.clock.tick()
        row = []
        for j in range(dim):
            row.append(sram_data[j].rdata)
        dut_O.append(row)

    sram_addr.en_acc = 0
    sram_addr.read = 0
    sim.clock.tick()

    dut_O = np.array(dut_O, dtype=np.uint32).view(np.float32).T
    print(str(dut_O))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-file", type=str, default='MSAGA.sv')
    parser.add_argument("--src-dir", type=str, default="../build/msaga")
    parser.add_argument("--build-dir", type=str, default="../build/msaga")
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--elem-width", type=int, default=16)
    parser.add_argument("--acc-width", type=int, default=16)
    args = parser.parse_args()
    sim = PyVerilator.build(args.top_file, [args.src_dir], build_dir=args.build_dir)
    test_msaga(sim, args.dim)