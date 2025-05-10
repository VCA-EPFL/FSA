import argparse
import numpy as np
import torch
from pyverilator import PyVerilator
from .data import *
from .signal_wrapper import *
from pyeasyfloat.backend import BaseFPBackend, PyEasyFloatBackend, HwBackend
import msaga as M

def sim_attention_tile(
    sim: PyVerilator, inst: Instruction,
    dim: int,
    Q: M.STile, K: M.STile, V_t: M.STile,
    D: M.ATile, O: M.ATile, accum: bool
    ):

    B = M.StaticInstBuilder(dim, dim)
    Q = Q.reverse(0)
    B.mx_load_stationary(Q, None, None)
    B.mx_attn_score(K, D, accum, None, None)
    B.mx_attn_value(V_t, O, accum, None, None)
    program = B.compile()

    for i in program.instructions:
        inst.bits = i.bits
        while not inst.ready:
            sim.clock.tick()
        inst.valid = 1
        sim.clock.tick()
        inst.reset()


def sim_attention_lse_norm(
    sim: PyVerilator,
    inst: Instruction,
    sram_addr: SRAMAddr,
    sram_data: list[SRAMData],
    dim: int, ew: int, mw: int,
    D: M.ATile, O: M.ATile
):

    B = M.StaticInstBuilder(dim, dim)
    B.mx_reciprocal(D, 0, 0)
    B.mx_attn_lse_norm(O, 0, 0)
    program = B.compile()

    for i in program.instructions:
        inst.bits = i.bits
        while not inst.ready:
            sim.clock.tick()
        inst.valid = 1
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

    accType = M.get_dtype(addEW, addMW)
    mulType = M.get_dtype(mulEW, mulMW)
    # row addr is used in SRAMs
    # spad
    q_row_addr = 0
    k_row_addr = lambda j: dim + 2 * dim * j
    v_row_addr = lambda j: 2 * dim + 2 * dim * j
    # acc
    d_row_addr = 0
    o_row_addr = 1
    # byte-addr, used for `Tile`s
    q_addr = q_row_addr * dim * mulType.itemsize
    k_addr = lambda j : k_row_addr(j) * dim * mulType.itemsize
    v_addr = lambda j : v_row_addr(j) * dim * mulType.itemsize
    d_addr = d_row_addr * dim * accType.itemsize
    o_addr = o_row_addr * dim * accType.itemsize

    # the following tiles do not hold any data, just hold address to generate the instructions
    Q_tile = M.STile((dim, dim), dtype=mulType, data_ptr=q_addr)
    K_tiles = [M.STile((dim, dim), dtype=mulType, data_ptr=k_addr(j)) for j in range(len(K_BLOCKS))]
    V_tiles = [M.STile((dim, dim), dtype=mulType, data_ptr=v_addr(j)) for j in range(len(V_BLOCKS))]
    O_tile = M.ATile((dim, dim), dtype=accType, data_ptr=o_addr)
    D_tile = M.ATile((1, dim), dtype=accType, data_ptr=d_addr)
    # initialize scratchpad
    write_sram(dim, mulEW, mulMW, sram_addr, sram_data, Q_i, q_row_addr)
    for j, (K_j, V_j) in enumerate(zip(K_BLOCKS, V_BLOCKS)):
        write_sram(dim, mulEW, mulMW, sram_addr, sram_data, K_j, k_row_addr(j))
        write_sram(dim, mulEW, mulMW, sram_addr, sram_data, V_j.T, v_row_addr(j))
    sim.clock.tick()

    # 2. inner loop
    backend = PyEasyFloatBackend()
    for j, (K_j, V_j) in enumerate(zip(K_BLOCKS, V_BLOCKS)):
        # run hardware verilator simulation
        sim_attention_tile(sim, inst, dim, Q_tile, K_tiles[j], V_tiles[j], D_tile, O_tile, accum=(j!=0))
        # run software emulation of a single tile
        ref_tile = FlashAttentionTile(Q_i, K_j, V_j, PrevRowMax, PrevRowSum, PrevO, mulEW, mulMW, addEW, addMW, backend)
        PrevRowMax = ref_tile.AccRowMaxS
        PrevRowSum = ref_tile.AccRowSum
        PrevO = ref_tile.AccO
    dut_res = sim_attention_lse_norm(sim, inst, sram_addr, sram_data, dim, addEW, addMW, D_tile, O_tile)

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
    parser.add_argument("--add-ew", type=int, default=8)
    parser.add_argument("--add-mw", type=int, default=23)
    args = parser.parse_args()
    ref_dtype = np.float16 if args.ref_dtype == 'fp16' else np.float32
    sim = PyVerilator.build(args.top_file, [args.src_dir], build_dir=args.build_dir, make_args=['-j8'])
    test_flash_attention(
        sim, args.dim, args.blocks,
        ref_dtype,
        args.mul_ew, args.mul_mw,
        args.add_ew, args.add_mw
    )