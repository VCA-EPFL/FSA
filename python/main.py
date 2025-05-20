import os
import numpy as np
import msaga as M
from unit_test.data import *
from msaga.tensor import MTile, ATile, STile

simulator = M.VerilatorSimulator(os.path.join('..', '..', '..', 'sims', 'verilator', 'simulator-chipyard.harness-MSAGAConfig-debug'))

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
        M.mx_load_stationary(Q_tile_rev, sem_q, sem_q)
        for j, (K_j, V_t_j) in enumerate(zip(K_BLOCKS, V_t_BLOCKS)):
            is_first_iter = j == 0
            buffer = j % 2
            K_tile, V_t_tile = K_tiles[buffer], V_t_tiles[buffer]
            sem_k, sem_v = sem_k_lst[buffer], sem_v_lst[buffer]

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

def ref(Q_np: np.ndarray, K_np: np.ndarray, V_np: np.ndarray):
    seq_q, d = Q_np.shape
    backend = PyEasyFloatBackend()
    PrevRowMax = np.full((seq_q, 1), np.float16(-np.inf))
    PrevRowSum = np.full((seq_q, 1), np.float16(0))
    PrevO = np.full((seq_q, d), np.float32(0))
    tile = FlashAttentionTile(
        Q_np, K_np, V_np,
        PrevRowMax,
        PrevRowSum,
        PrevO,
        mul_ew=5,
        mul_mw=10,
        acc_ew=8,
        acc_mw=23,
        backend=backend
    )
    print(str(tile))

def main():
    np.random.seed(0)
    Q_np = np.random.rand(4, 4).astype(np.float16)
    K_np = np.random.rand(4, 4).astype(np.float16)
    V_np = np.random.rand(4, 4).astype(np.float16)
    ref(Q_np, K_np, V_np)
    Q = M.from_numpy(Q_np)
    K = M.from_numpy(K_np)
    V_t = M.from_numpy(V_np.T)
    O_t = scaled_dot_product_attention(Q, K, V_t, 4, 4)
    res = M.to_numpy(O_t)
    print(res)

if __name__ == "__main__":
    main()