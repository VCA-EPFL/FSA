import os
import torch
import numpy as np
import msaga as M
import argparse
from fa_ref import *
from msaga.tensor import MTile, ATile, STile

@M.kernel
def scaled_dot_product_attention(Q: MTile, K: MTile, V_t: MTile, br: int, bc: int) -> MTile:
    assert (len(Q.shape), len(K.shape), len(V_t.shape)) == (2, 2, 2)
    seq_q, d = Q.shape
    seq_k, dk = K.shape
    dv, seq_v = V_t.shape
    assert d == dk and d == dv and seq_k == seq_v
    assert bc == d, "MSAGA requires bc == d"

    O_t: MTile = M.alloc_mem((d, seq_q), M.fp32)
    Q_BLOCKS = Q.split(br, dim=-2) # [br, d]
    K_BLOCKS = K.split(bc, dim=-2) # [bc, d]
    V_t_BLOCKS = V_t.split(bc, dim=-1) # [d, bc]
    O_t_BLOCKS = O_t.split(br, dim=-1) # [d, br]

    # [Br, d]
    Q_tiles = [M.alloc_spad((br, d)) for _ in range(2)]
    # log exp sum [Br, 1]
    L_tile = M.alloc_accumulator((1, br))
    # [d, Br]
    O_t_tile = M.alloc_accumulator((d, br))

    # double-buffer KV
    K_tiles = [M.alloc_spad((bc, d)) for _ in range(2)]
    V_t_tiles = [M.alloc_spad((d, bc)) for _ in range(2)]

    sem_q_lst = [M.Semaphore(id=0, n=2), M.Semaphore(id=1, n=2)]
    sem_k_lst = [M.Semaphore(id=2, n=2), M.Semaphore(id=3, n=2)]
    sem_v_lst = [M.Semaphore(id=4, n=2), M.Semaphore(id=5, n=2)]
    sem_o = M.Semaphore(id=6, n=2)

    for i, Q_i in enumerate(Q_BLOCKS):
        Q_tile = Q_tiles[i % 2]
        sem_q = sem_q_lst[i % 2]
        Q_tile_rev = Q_tile.reverse(dim=0)
        M.load_tile(Q_i, Q_tile, sem_q)
        for j, (K_j, V_t_j) in enumerate(zip(K_BLOCKS, V_t_BLOCKS)):
            is_first_iter = j == 0
            is_last_iter = j == len(K_BLOCKS) - 1
            buffer = j % 2
            K_tile, V_t_tile = K_tiles[buffer], V_t_tiles[buffer]
            sem_k, sem_v = sem_k_lst[buffer], sem_v_lst[buffer]

            M.mx_load_stationary(Q_tile_rev, sem_q, aq=is_first_iter, rl=is_last_iter)

            M.load_tile(K_j, K_tile, sem_k)
            M.mx_attn_score(K_tile, L_tile, not is_first_iter, sem_k)

            M.load_tile(V_t_j, V_t_tile, sem_v)
            M.mx_attn_value(V_t_tile, O_t_tile, not is_first_iter, sem_v)
        # end inner loop
        M.mx_reciprocal(L_tile, None)
        M.mx_attn_lse_norm(O_t_tile, sem_o, aq=False, rl=True)
        M.store_tile(O_t_tile, O_t_BLOCKS[i], sem_o)
    M.fence(mx=True, dma=True, stop=True)
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
        engine: M.engine.BaseEngine,
        diff_easyfloat: bool = False,
        easyfloat_verbose: bool = False
    ):
    np.random.seed(0)
    Q_np = np.random.rand(seq_q, d).astype(np.float16)
    K_np = np.random.rand(seq_kv, d).astype(np.float16)
    V_np = np.random.rand(seq_kv, d).astype(np.float16)

    impls = {}
    if engine:
        Q = M.from_numpy(Q_np)
        K = M.from_numpy(K_np)
        V_t = M.from_numpy(V_np.T)
        O_t = engine.execute(scaled_dot_product_attention(Q, K, V_t, br, bc))
        O = M.to_numpy(O_t).T
        impls['MSAGA'] = O

    if diff_easyfloat:
        print("Comparing with PyEasyFloat...")
        if easyfloat_verbose:
            print("PyEasyFloat verbose mode enabled.")
        O_pyeasyfloat = ref_pyeasyfloat(Q_np, K_np, V_np, br, bc, easyfloat_verbose)
        impls['PyEasyFloat'] = O_pyeasyfloat

    print("Comparing with Torch...")
    O_torch = ref_torch(Q_np, K_np, V_np)

    M.compare_matrices(
        ('torch', O_torch),
        impls
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_q', type=int, default=4, help='Sequence length for query')
    parser.add_argument('--seq_kv', type=int, default=4, help='Sequence length for key/value')
    parser.add_argument('--config', type=str, default='SmallMSAGAConfig', help='Chisel generation config')
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
        long_name + '.MSAGAConfig.json'
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

        engine = M.VerilatorSimulator(
            simulator_bin,
            vcdfile=args.vcdfile,
            output_dir=args.output_dir,
            max_cycles=args.max_cycles,
            numactl_cmd=args.numactl
        )
    else:
        assert f"{args.engine} is not supported yet."


    if not os.path.isfile(config_file):
        print(f"Warning: Config file not found: {config_file}. Using default MSAGA config.")
    else:
        print(f"Loading config from: {config_file}")
        M.init(config_file)
        cfg = M.get_config()

    main(
        args.seq_q, args.seq_kv,
        d=cfg.sa_rows, br=cfg.sa_cols, bc=cfg.sa_rows, engine=engine,
        diff_easyfloat=args.diff, easyfloat_verbose=args.diff_verbose
    )
