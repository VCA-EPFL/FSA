import argparse
import numpy as np
import torch
from signal_wrapper import SignalWrapper
from pyverilator import PyVerilator
from data import *
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
ATTENTION_LSE_SCALE = 3
ATTENTION_LSE_NORM = 4

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


def sim_attention_tile(
    sim: PyVerilator, inst: Instruction,
    dim: int,
    q_addr: int, k_addr: int, v_addr: int,
    d_addr: int, o_addr: int
    ):

    Q_desc = MatrixDesc(
        origin_addr=q_addr,
        dim=dim,
        rev_v=False, rev_h=True,
        delay_u=False, delay_d=False
    )
    K_desc = MatrixDesc(
        origin_addr=k_addr,
        dim=dim,
        rev_v=False, rev_h=False,
        delay_u=True, delay_d=False
    )
    V_desc = MatrixDesc(
        origin_addr=v_addr,
        dim=dim,
        rev_v=True, rev_h=False,
        delay_u=False, delay_d=True
    )
    O_desc = MatrixDesc(
        origin_addr=o_addr,
        dim=dim,
        rev_v=False, rev_h=False, delay_u=False, delay_d=False
    )
    D_desc = MatrixDesc(origin_addr=d_addr, dim=dim, rev_v=False, rev_h=False, delay_u=False, delay_d=False)

    while not inst.ready:
        sim.clock.tick()
    inst.set(LOAD_STATIONARY, Q_desc.to_rs(), 0)
    sim.clock.tick()
    inst.reset()

    while not inst.ready:
        sim.clock.tick()
    inst.set(ATTENTION_SCORE, K_desc.to_rs(), D_desc.to_rs())
    sim.clock.tick()
    inst.reset()

    while not inst.ready:
        sim.clock.tick()
    inst.set(ATTENTION_VALUE, V_desc.to_rs(), O_desc.to_rs())
    sim.clock.tick()
    inst.reset()


def sim_attention_lse_norm(
    sim: PyVerilator,
    inst: Instruction,
    sram_addr: SRAMAddr,
    sram_data: list[SRAMData],
    dim: int, ew: int, mw: int,
    d_addr: int, o_addr: int
):

    D_desc = MatrixDesc(origin_addr=d_addr, dim=dim, rev_v=False, rev_h=False, delay_u=False, delay_d=False)
    O_desc = MatrixDesc(
        origin_addr=o_addr,
        dim=dim,
        rev_v=False, rev_h=False, delay_u=False, delay_d=False
    )

    inst.funct7 = ATTENTION_LSE_SCALE
    while not inst.ready:
        sim.clock.tick()
    inst.set(ATTENTION_LSE_SCALE, 0, D_desc.to_rs())
    sim.clock.tick()
    inst.reset()

    inst.funct7 = ATTENTION_LSE_NORM
    while not inst.ready:
        sim.clock.tick()
    inst.set(ATTENTION_LSE_NORM, 0, O_desc.to_rs())
    sim.clock.tick()
    inst.reset()

    # LSE NORM takes dim cycles
    [sim.clock.tick() for _ in range(dim)]

    dut_O = []
    for i in range(dim):
        sram_addr.en_acc = 1
        sram_addr.read = 1
        sram_addr.addr = i + 1
        sim.clock.tick()
        row = []
        for j in range(dim):
            x = FloatPoint.from_bits(sram_data[j].rdata, ew, mw)
            x = fp_to_np(x)
            row.append(x)
        dut_O.append(row)

    sram_addr.en_acc = 0
    sram_addr.read = 0
    sim.clock.tick()

    dut_O = np.array(dut_O).T
    return dut_O


def write_sram(
        dim: int, ew: int, mw: int,
        sram_addr: SRAMAddr, sram_data: list[SRAMData],
        tile: np.ndarray,
        base: int, accRAM: bool = False
):
    for i in range(dim):
        if accRAM:
            sram_addr.en_acc = 1
        else:
            sram_addr.en_sp = 1
        sram_addr.write = 1
        sram_addr.addr = base + i
        for j in range(dim):
            sram_data[j].wdata = np_to_fp(tile[i, j], ew, mw).to_bits()
        sim.clock.tick()
    if accRAM:
        sram_addr.en_acc = 0
    else:
        sram_addr.en_sp = 0
    sram_addr.write = 0


def standard_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    d = Q.shape[-1]
    S = (Q @ K.T) / np.sqrt(d)
    row_max = np.max(S, axis=-1, keepdims=True)
    S = S - row_max
    S_exp = np.exp(S)
    exp_sum = np.sum(S_exp, axis=-1, keepdims=True)
    return (S_exp @ V) / exp_sum

def compare_impls(ref: np.ndarray, impls: list[np.ndarray], impl_nams: list[str]):
    def error_metrics(a, b):
        return {
            'MAE': np.mean(np.abs(a - b)),
            'MSE': np.mean((a - b) ** 2),
            'MaxErr': np.max(np.abs(a - b)),
            'RelErr': np.mean(np.abs((a - b) / (b + 1e-8)))
        }

    for impl, name in zip(impls, impl_nams):
        err = error_metrics(impl, ref)
        print(f'Error of {name} vs standard impl:', err)


def test_flash_attention(
        sim: PyVerilator,
        dim: int, blocks: int,
        np_dtype: np.float32 | np.float16,
        mulEW: int, mulMW: int,
        addEW: int, addMW: int
    ):
    """One outer loop of flash attention V2"""
    inst = Instruction(sim)
    sram_addr = SRAMAddr(sim)
    sram_data = [SRAMData(sim, i) for i in range(dim)]
    sim.start_vcd_trace(PyVerilator.default_vcd_filename)
    sim.io.reset = 1
    sim.clock.tick()
    sim.io.reset = 0

    # 1. prepare data in sram
    np.random.seed(0)
    Q_i = np.random.random((dim, dim)).astype(np_dtype)
    K = np.random.random((blocks * dim, dim)).astype(np_dtype)
    V = np.random.random((blocks * dim, dim)).astype(np_dtype)
    K_BLOCKS = np.split(K, blocks, axis=-2)
    V_BLOCKS = np.split(V, blocks, axis=-2)
    PrevO = np.full((dim, dim), np_dtype(0))
    PrevRowMax = np.full((dim, 1), np_dtype(-np.inf))
    PrevRowSum = np.full((dim, 1), np_dtype(0))

    def k_addr(j: int):
        return dim + 2 * dim * j

    def v_addr(j: int):
        return 2 * dim + 2 * dim * j

    def d_addr():
        return 0

    q_addr = lambda : 0
    d_addr = lambda : 0
    o_addr = lambda : 1

    write_sram(dim, mulEW, mulMW, sram_addr, sram_data, Q_i, q_addr())
    for j, (K_j, V_j) in enumerate(zip(K_BLOCKS, V_BLOCKS)):
        write_sram(dim, mulEW, mulMW, sram_addr, sram_data, K_j, k_addr(j))
        write_sram(dim, mulEW, mulMW, sram_addr, sram_data, V_j.T, v_addr(j))

    # we use accRAM row 0 to store exp sum, so O starts from row 1
    write_sram(dim, addEW, addMW, sram_addr, sram_data, PrevO.T, o_addr(), accRAM=True)
    sim.clock.tick()

    # 2. inner loop
    backend = PyEasyFloatBackend()
    for j, (K_j, V_j) in enumerate(zip(K_BLOCKS, V_BLOCKS)):
        # run hardware verilator simulation
        sim_attention_tile(sim, inst, dim, q_addr(), k_addr(j), v_addr(j), d_addr(), o_addr())
        # run software emulation of a single tile
        ref_tile = FlashAttentionTile(Q_i, K_j, V_j, PrevRowMax, PrevRowSum, PrevO, mulEW, mulMW, addEW, addMW, backend)
        PrevRowMax = ref_tile.AccRowMaxS
        PrevRowSum = ref_tile.AccRowSum
        PrevO = ref_tile.AccO

    dut_res = sim_attention_lse_norm(sim, inst, sram_addr, sram_data, dim, addEW, addMW, d_addr(), o_addr())

    torch_dtype = torch.float32 if np_dtype == np.float32 else torch.float16
    torch_res = np.array(torch.nn.functional.scaled_dot_product_attention(
        torch.tensor(Q_i, dtype=torch_dtype),
        torch.tensor(K, dtype=torch_dtype),
        torch.tensor(V, dtype=torch_dtype)
    ))

    standard_res = standard_attention(Q_i, K, V)

    print(f"Standard:")
    print(standard_res)
    print(f"PyEasyFloat:")
    print(mat_to_numpy_array(ref_tile.NormO))
    print(f"Torch:")
    print(torch_res)
    print(f"MSAGA:")
    print(dut_res)

    compare_impls(standard_res,
                  [mat_to_numpy_array(ref_tile.NormO), torch_res, dut_res],
                  ['PyEasyFloat', 'Torch', 'MSAGA']
                  )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-file", type=str, default='MSAGA.sv')
    parser.add_argument("--src-dir", type=str, default="../build/msaga")
    parser.add_argument("--build-dir", type=str, default="../build/msaga")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--ref-dtype", choices=['fp16', 'fp32'], default='fp16')
    parser.add_argument("--mul-ew", type=int, default=5)
    parser.add_argument("--mul-mw", type=int, default=10)
    parser.add_argument("--add-ew", type=int, default=5)
    parser.add_argument("--add-mw", type=int, default=10)
    args = parser.parse_args()
    ref_dtype = np.float16 if args.ref_dtype == 'fp16' else np.float32
    sim = PyVerilator.build(args.top_file, [args.src_dir], build_dir=args.build_dir, make_args=['-j8'])
    test_flash_attention(
        sim, args.dim, args.blocks,
        ref_dtype,
        args.mul_ew, args.mul_mw,
        args.add_ew, args.add_mw
    )