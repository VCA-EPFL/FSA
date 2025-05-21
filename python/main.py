import os
import torch
import numpy as np
import msaga as M
from unit_test.data import *
from msaga.tensor import MTile, ATile, STile

simulator = M.VerilatorSimulator(
    os.path.join('..', '..', '..', 'sims', 'verilator', 'simulator-chipyard.harness-MSAGAConfig-debug')
)

@M.kernel(engine=simulator)
def scaled_dot_product_attention(Q: MTile, K: MTile, V_t: MTile, br: int, bc: int) -> MTile:
    assert (len(Q.shape), len(K.shape), len(V_t.shape)) == (2, 2, 2)
    seq_q, d = Q.shape
    seq_k, dk = K.shape
    dv, seq_v = V_t.shape
    assert d == dk and d == dv and seq_k == seq_v

    O_t: MTile = M.alloc_mem((d, seq_q), M.fp32)
    Q_BLOCKS = Q.split(br, dim=-2) # [br, d]
    K_BLOCKS = K.split(bc, dim=-2) # [bc, d]
    V_t_BLOCKS = V_t.split(bc, dim=-1) # [d, bc]
    O_t_BLOCKS = O_t.split(br, dim=-1) # [d, br]

    # [Br, d]
    Q_tile = M.alloc_spad((br, d))
    Q_tile_rev = Q_tile.reverse(dim=0)
    # log exp sum [Br, 1]
    L_tile = M.alloc_accumulator((1, br))
    # [d, Br]
    O_t_tile = M.alloc_accumulator((d, br))

    # double-buffer KV
    K_tiles = [M.alloc_spad((bc, d)) for _ in range(2)]
    V_t_tiles = [M.alloc_spad((d, bc)) for _ in range(2)]

    sem_q = M.Semaphore(id=1, n=2)
    sem_k_lst = [M.Semaphore(id=2, n=2), M.Semaphore(id=3, n=2)]
    sem_v_lst = [M.Semaphore(id=4, n=2), M.Semaphore(id=5, n=2)]
    sem_o = M.Semaphore(id=6, n=2)

    for i, Q_i in enumerate(Q_BLOCKS):
        M.load_tile(Q_i, Q_tile, sem_q, sem_q)
        for j, (K_j, V_t_j) in enumerate(zip(K_BLOCKS, V_t_BLOCKS)):
            is_first_iter = j == 0
            is_last_iter = j == len(K_BLOCKS) - 1
            buffer = j % 2
            K_tile, V_t_tile = K_tiles[buffer], V_t_tiles[buffer]
            sem_k, sem_v = sem_k_lst[buffer], sem_v_lst[buffer]
            if is_last_iter:
                M.mx_load_stationary(Q_tile_rev, sem_q, sem_q)
            else:
                M.mx_load_stationary(Q_tile_rev, sem_q, None)
            M.load_tile(K_j, K_tile, sem_k, sem_k)
            M.mx_attn_score(K_tile, L_tile, not is_first_iter, sem_k, sem_k)

            M.load_tile(V_t_j, V_t_tile, sem_v, sem_v)
            M.mx_attn_value(V_t_tile, O_t_tile, not is_first_iter, sem_v, sem_v)
        # end inner loop
        M.mx_reciprocal(L_tile, None, None)
        M.mx_attn_lse_norm(O_t_tile, None, sem_o)
        M.store_tile(O_t_tile, O_t_BLOCKS[i], sem_o, sem_o)
    M.fence(mx=True, dma=True, stop=True)
    return O_t

def ref_pyeasyfloat(Q_np: np.ndarray, K_np: np.ndarray, V_np: np.ndarray, br: int, bc: int) -> np.ndarray:
    row_blocks = Q_np.shape[0] // br
    col_blocks = K_np.shape[0] // bc
    d = Q_np.shape[-1]
    Q_BLOCKS = np.split(Q_np, row_blocks, axis=-2)
    K_BLOCKS = np.split(K_np, col_blocks, axis=-2)
    V_BLOCKS = np.split(V_np, col_blocks, axis=-2)
    backend = PyEasyFloatBackend()
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
            PrevRowMax = tile.AccRowMaxS
            PrevRowSum = tile.AccRowSum
            PrevO = tile.AccO
    return mat_to_numpy_array(tile.NormO)

def ref_torch(Q_np: np.ndarray, K_np: np.ndarray, V_np: np.ndarray) -> np.ndarray:
    Q_torch = torch.from_numpy(Q_np)
    K_torch = torch.from_numpy(K_np)
    V_torch = torch.from_numpy(V_np)
    print(Q_torch.dtype)
    O_torch = torch.nn.functional.scaled_dot_product_attention(Q_torch, K_torch, V_torch)
    return O_torch.numpy()


def main(seq_q: int, seq_kv: int, d: int):
    np.random.seed(0)
    Q_np = np.random.rand(seq_q, d).astype(np.float16)
    K_np = np.random.rand(seq_kv, d).astype(np.float16)
    V_np = np.random.rand(seq_kv, d).astype(np.float16)
    O_pyeasyfloat = ref_pyeasyfloat(Q_np, K_np, V_np, 4, 4)
    O_torch = ref_torch(Q_np, K_np, V_np)
    Q = M.from_numpy(Q_np)
    K = M.from_numpy(K_np)
    V_t = M.from_numpy(V_np.T)
    O_t = scaled_dot_product_attention(Q, K, V_t, 4, 4)
    O = M.to_numpy(O_t).T
    M.compare_matrices(
        ref=('torch', O_torch),
        impls={
            'MSAGA': O,
            'PyEasyFloat': O_pyeasyfloat
        }
    )

if __name__ == "__main__":
    main(seq_q=4, seq_kv=12, d=4)