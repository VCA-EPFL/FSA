import os
import torch
import numpy as np
import fsa as F
import argparse
from fa_ref import *
from fsa.tensor import MTile, ATile, STile

@F.kernel
def scaled_dot_product_attention(Q: MTile, K: MTile, V_t: MTile, br: int, bc: int) -> MTile:
    assert (len(Q.shape), len(K.shape), len(V_t.shape)) == (2, 2, 2)
    seq_q, d = Q.shape
    seq_k, dk = K.shape
    dv, seq_v = V_t.shape
    assert d == dk and d == dv and seq_k == seq_v
    assert bc == d, "FSA requires bc == d"

    O_t: MTile = F.alloc_mem((d, seq_q), F.fp32)
    Q_BLOCKS = Q.split(br, dim=-2) # [br, d]
    K_BLOCKS = K.split(bc, dim=-2) # [bc, d]
    V_t_BLOCKS = V_t.split(bc, dim=-1) # [d, bc]
    O_t_BLOCKS = O_t.split(br, dim=-1) # [d, br]

    # [Br, d]
    Q_tiles = [F.alloc_spad((br, d)) for _ in range(2)]
    # log exp sum [Br, 1]
    L_tile = F.alloc_accumulator((1, br))
    # [d, Br]
    O_t_tile = F.alloc_accumulator((d, br))

    # double-buffer KV
    K_tiles = [F.alloc_spad((bc, d)) for _ in range(2)]
    V_t_tiles = [F.alloc_spad((d, bc)) for _ in range(2)]

    sem_q_lst = [F.Semaphore(id=0, n=2), F.Semaphore(id=1, n=2)]
    sem_k_lst = [F.Semaphore(id=2, n=2), F.Semaphore(id=3, n=2)]
    sem_v_lst = [F.Semaphore(id=4, n=2), F.Semaphore(id=5, n=2)]
    sem_o = F.Semaphore(id=6, n=2)

    for i, Q_i in enumerate(Q_BLOCKS):
        Q_tile = Q_tiles[i % 2]
        sem_q = sem_q_lst[i % 2]
        Q_tile_rev = Q_tile.reverse(dim=0)
        F.load_tile(Q_i, Q_tile, sem_q)
        for j, (K_j, V_t_j) in enumerate(zip(K_BLOCKS, V_t_BLOCKS)):
            is_first_iter = j == 0
            is_last_iter = j == len(K_BLOCKS) - 1
            buffer = j % 2
            K_tile, V_t_tile = K_tiles[buffer], V_t_tiles[buffer]
            sem_k, sem_v = sem_k_lst[buffer], sem_v_lst[buffer]

            F.mx_load_stationary(Q_tile_rev, sem_q, aq=is_first_iter, rl=is_last_iter)

            F.load_tile(K_j, K_tile, sem_k)
            F.mx_attn_score(K_tile, L_tile, not is_first_iter, sem_k)

            F.load_tile(V_t_j, V_t_tile, sem_v)
            F.mx_attn_value(V_t_tile, O_t_tile, not is_first_iter, sem_v)
        # end inner loop
        F.mx_reciprocal(L_tile, None)
        F.mx_attn_lse_norm(O_t_tile, sem_o, aq=False, rl=True)
        F.store_tile(O_t_tile, O_t_BLOCKS[i], sem_o)
    F.fence(mx=True, dma=True, stop=True)
    return O_t

def ref_pyeasyfloat(Q_np: np.ndarray, K_np: np.ndarray, V_np: np.ndarray, br: int, bc: int, verbose: bool) -> np.ndarray:
    row_blocks = Q_np.shape[0] // br
    col_blocks = K_np.shape[0] // bc
    d = Q_np.shape[-1]
    Q_BLOCKS = np.split(Q_np, row_blocks, axis=-2)
    K_BLOCKS = np.split(K_np, col_blocks, axis=-2)
    V_BLOCKS = np.split(V_np, col_blocks, axis=-2)
    backend = PyEasyFloatBackend()
    res = []
    for i, Q_i in enumerate(Q_BLOCKS):
        PrevO = np.full((br, d), np.float32(0))
        PrevRowMax = np.full((br, 1), np.float32(-np.inf))
        PrevRowSum = np.full((br, 1), np.float32(0))
        for j, (K_j, V_j) in enumerate(zip(K_BLOCKS, V_BLOCKS)):
            tile = FlashAttentionTile(
                Q_i, K_j, V_j,
                PrevRowMax, PrevRowSum, PrevO,
                mul_ew=5, mul_mw=10,
                acc_ew=8, acc_mw=23,
                backend=backend
            )
            if verbose:
                print(str(tile))
            PrevRowMax = tile.AccRowMaxS
            PrevRowSum = tile.AccRowSum
            PrevO = tile.AccO
        res.append(mat_to_numpy_array(tile.NormO))
    return np.concatenate(res, axis=0)

def ref_torch(Q_np: np.ndarray, K_np: np.ndarray, V_np: np.ndarray) -> np.ndarray:
    Q_torch = torch.from_numpy(Q_np)
    K_torch = torch.from_numpy(K_np)
    V_torch = torch.from_numpy(V_np)
    O_torch = torch.nn.functional.scaled_dot_product_attention(Q_torch, K_torch, V_torch)
    return O_torch.numpy()


def main(
        seq_q: int, seq_kv: int, d: int, br: int, bc: int,
        engine: F.engine.BaseEngine,
        diff_easyfloat: bool = False,
        easyfloat_verbose: bool = False
    ):
    np.random.seed(0)
    Q_np = np.random.rand(seq_q, d).astype(np.float16)
    K_np = np.random.rand(seq_kv, d).astype(np.float16)
    V_np = np.random.rand(seq_kv, d).astype(np.float16)

    impls = {}
    if engine:
        Q = F.from_numpy(Q_np)
        K = F.from_numpy(K_np)
        V_t = F.from_numpy(V_np.T)
        O_t = engine.execute(scaled_dot_product_attention(Q, K, V_t, br, bc))
        O = F.to_numpy(O_t).T
        impls['FSA'] = O

    if diff_easyfloat:
        print("Comparing with PyEasyFloat...")
        if easyfloat_verbose:
            print("PyEasyFloat verbose mode enabled.")
        O_pyeasyfloat = ref_pyeasyfloat(Q_np, K_np, V_np, br, bc, easyfloat_verbose)
        impls['PyEasyFloat'] = O_pyeasyfloat

    print("Comparing with Torch...")
    O_torch = ref_torch(Q_np, K_np, V_np)

    F.compare_matrices(
        ('torch', O_torch),
        impls
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_q', type=int, default=4, help='Sequence length for query')
    parser.add_argument('--seq_kv', type=int, default=4, help='Sequence length for key/value')
    parser.add_argument('--config', type=str, default='FSA4X4Fp16Config', help='Chisel generation config')
    parser.add_argument('--engine', type=str, default='Verilator', choices=['Verilator', 'FPGA'])
    parser.add_argument('--build_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='/tmp', help='Output directory')
    parser.add_argument('--diff', action='store_true', help='Compare result with PyEasyFloat')
    parser.add_argument('--diff_verbose', action='store_true', help='Enable verbose mode for PyEasyFloat')
    parser.add_argument('--diff_only', action='store_true', help='Only run PyEasyFloat, skip real hardware execution')
    parser.add_argument('--simulator_bin', type=str, default=None, help='[VerilatorOnly] Path to the simulator binary')
    parser.add_argument('--vcdfile', type=str, default=None, help='[VerilatorOnly] Path to the VCD file')
    parser.add_argument('--numactl', type=str, default=None, help='[VerilatorOnly] Command to run the simulator with NUMA control')
    parser.add_argument('--max_cycles', type=int, default=0, help='[VerilatorOnly] Maximum number of cycles to run the simulation')
    args = parser.parse_args()

    if args.build_dir is None:
        build_dir = os.path.join('..', '..', '..', 'sims', 'verilator')
    else:
        build_dir = args.build_dir
    long_name = 'chipyard.harness.TestHarness.' + args.config
    config_file = os.path.join(
        build_dir, 'generated-src', long_name,
        long_name + '.FSAConfig.json'
    )

    if args.diff_only:
        engine = None
    elif args.engine == 'Verilator':

        if args.simulator_bin is not None:
            simulator_bin = args.simulator_bin
        else:
            simulator_bin = os.path.join(build_dir, 'simulator-chipyard.harness-' + args.config + '-debug')
            if not os.path.isfile(simulator_bin):
                simulator_bin = os.path.join(build_dir, 'simulator-chipyard.harness-' + args.config)
        if os.path.isfile(simulator_bin):
            print(f"Using simulator binary: {simulator_bin}")
        else:
            raise FileNotFoundError(f"Simulator binary not found: {simulator_bin}")

        engine = F.VerilatorSimulator(
            simulator_bin,
            vcdfile=args.vcdfile,
            output_dir=args.output_dir,
            max_cycles=args.max_cycles,
            numactl_cmd=args.numactl
        )
    elif args.engine == 'FPGA':
        if args.build_dir is None:
            build_dir = os.path.join('..', '..', '..', 'fpga')
        else:
            build_dir = args.build_dir
        long_name = 'chipyard.fpga.u55c.U55CFPGATestHarness.' + args.config
        config_file = os.path.join(
            build_dir, 'generated-src', long_name,
            long_name + '.FSAConfig.json'
        )
        engine = F.FPGA()
    else:
        assert f"{args.engine} is not supported yet."


    if not os.path.isfile(config_file):
        print(f"Warning: Config file not found: {config_file}. Using default FSA config.")
    else:
        print(f"Loading config from: {config_file}")
        F.init(config_file)
        cfg = F.get_config()

    main(
        args.seq_q, args.seq_kv,
        d=cfg.sa_rows, br=cfg.sa_cols, bc=cfg.sa_rows, engine=engine,
        diff_easyfloat=args.diff, easyfloat_verbose=args.diff_verbose
    )
