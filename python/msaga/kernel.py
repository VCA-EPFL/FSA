from typing import Optional
from .instructions import *
from .tensor import MTile, STile, ATile
from .engine import BaseEngine
from .config import g_config
from .mem import g_mem_manger

class KernelContext:
    def __init__(self, engine: BaseEngine):
        self.rows = g_config.sa_rows
        self.cols = g_config.sa_cols
        self.instructions: list[Instruction] = []
        self.engine = engine

    def tile_row_addr(self, tile: ATile | STile) -> int:
        # on-chip SRAMs are not byte-addressed, they are row-addressed
        if isinstance(tile, STile):
            assert tile.shape[-1] == self.rows
        else:
            assert tile.shape[-1] == self.cols
        cols, itemsize = tile.shape[-1], tile.dtype.itemsize
        return tile.data_ptr // (cols * itemsize)

    def tile_stride(self, tile: ATile | STile) -> int:
        if isinstance(tile, STile):
            assert tile.shape[-1] == self.rows
        else:
            assert tile.shape[-1] == self.cols
        return tile.stride[-2] // tile.shape[-1]

    def push(self, inst: Instruction) -> None:
        self.instructions.append(inst)

__g_kernel_ctx: Optional[KernelContext] = None

def kernel(engine: BaseEngine):
    def decorator(func):
        def wrapper(*args, **kwargs):
            global __g_kernel_ctx
            assert __g_kernel_ctx is None
            __g_kernel_ctx = KernelContext(engine)
            ret = func(*args, **kwargs)
            assert (ret is None) or (isinstance(ret, MTile)) or (isinstance(ret, list[MTile])), \
                "the return type of MSAGA kernel function can only be one of MTile, list[MTile] or None"
            inst_bytes = sum(inst.n_bytes for inst in __g_kernel_ctx.instructions)
            assert inst_bytes <= g_config.inst_queue_size * 4, \
                f"Instruction queue size exceeded: {inst_bytes} > {g_config.inst_queue_size * 4}"
            __g_kernel_ctx.engine.execute(
                __g_kernel_ctx.instructions,
                g_mem_manger.mem_tensor_list,
                ret
            )
            __g_kernel_ctx = None
            return ret
        return wrapper
    return decorator

def check_kernel_ctx(func):
    def wrapper(*args, **kwargs):
        assert __g_kernel_ctx is not None, f"{func.__name__} can only be called within a MSAGA kernel!"
        func(*args, **kwargs)
    return wrapper

@check_kernel_ctx
def fence(mx: bool, dma: bool, stop: bool) -> None:
    __g_kernel_ctx.push(FenceInstruction(mx, dma, stop))

@check_kernel_ctx
def dma(func: int, mem: MTile, tile: ATile | STile, sem: Optional[Semaphore], aq: bool = True, rl: bool = True) -> None:
    assert mem.shape == tile.shape and len(mem.shape) == 2 and mem.dtype == tile.dtype
    rows, cols = mem.shape
    mem = DMAInstrucionMem(mem.data_ptr, mem.stride[-2] * mem.dtype.itemsize, cols * mem.dtype.itemsize)
    sram = DMAInstrucionSRAM(
        __g_kernel_ctx.tile_row_addr(tile),
        __g_kernel_ctx.tile_stride(tile),
        isAccum=isinstance(tile, ATile)
    )
    if sem is None:
        aq, rl = False, False
    header = DMAInstructionHeader(
        sem.id if sem else 0,
        acquireValid=aq, acquireSemValue=sem.value if sem and aq else 0,
        releaseValid=rl, releaseSemValue=sem.inc().value if rl else 0,
        func=func,
        repeat=rows
    )
    __g_kernel_ctx.push(DMAInstruction(header, sram, mem))

@check_kernel_ctx
def load_tile(mem: MTile, tile: STile, sem: Optional[Semaphore], aq: bool = True, rl: bool = True) -> None:
    dma(DMAFunc.LD_SRAM.value, mem, tile, sem, aq, rl)


@check_kernel_ctx
def store_tile(tile: ATile, mem: MTile, sem: Optional[Semaphore], aq: bool = True, rl: bool = True) -> None:
    dma(DMAFunc.ST_SRAM.value, mem, tile, sem, aq, rl)

def build_matrix_instruction_header(
        func: int, waitPrevAcc: bool,
        sem: Optional[Semaphore], aq: bool, rl: bool) -> MatrixInstructionHeader:
    if sem is None:
        aq, rl = False, False
    return MatrixInstructionHeader(
        sem.id if sem else 0,
        acquireValid=aq, acquireSemValue=sem.value if sem and aq else 0,
        releaseValid=rl, releaseSemValue=sem.inc().value if rl else 0,
        func=func,
        waitPrevAcc=waitPrevAcc
    )

@check_kernel_ctx
def mx_load_stationary(tile: STile, sem: Optional[Semaphore], aq: bool = True, rl: bool = True) -> None:
    assert len(tile.shape) == 2 and tile.shape[-1] == __g_kernel_ctx.rows
    header = build_matrix_instruction_header(MxFunc.LOAD_STATIONARY.value, False, sem, aq, rl)
    spad = MatrixInstructionSpad(
        __g_kernel_ctx.tile_row_addr(tile),
        __g_kernel_ctx.tile_stride(tile),
        False, False, False
    )
    acc = MatrixInstrucionAcc(0, 0, False)
    __g_kernel_ctx.push(MatrixInstruction(header, spad, acc))

@check_kernel_ctx
def mx_attn_score(k: STile, l: ATile, accumulate: bool, sem: Optional[Semaphore], aq: bool = True, rl: bool = True) -> None:
    assert len(k.shape) == 2 and l.shape == (1, __g_kernel_ctx.cols)
    header = build_matrix_instruction_header(
        MxFunc.ATTN_SCORE.value, False,
        sem, aq, rl
    )
    spad = MatrixInstructionSpad(
        __g_kernel_ctx.tile_row_addr(k),
        __g_kernel_ctx.tile_stride(k),
        True, True, True
    )
    acc = MatrixInstrucionAcc(
        __g_kernel_ctx.tile_row_addr(l),
        __g_kernel_ctx.tile_stride(l),
        not accumulate
    )
    __g_kernel_ctx.push(MatrixInstruction(header, spad, acc))

@check_kernel_ctx
def mx_attn_value(v_t: STile, o_t: ATile, accumulate: bool, sem: Optional[Semaphore], aq: bool = True, rl: bool = True) -> None:
    assert len(v_t.shape) == 2 and len(o_t.shape) == 2
    header = build_matrix_instruction_header(
        MxFunc.ATTN_VALUE.value, False,
        sem, aq, rl
    )
    spad = MatrixInstructionSpad(
        __g_kernel_ctx.tile_row_addr(v_t),
        __g_kernel_ctx.tile_stride(v_t),
        True, False, True
    )
    acc = MatrixInstrucionAcc(
        __g_kernel_ctx.tile_row_addr(o_t),
        __g_kernel_ctx.tile_stride(o_t),
        not accumulate
    )
    __g_kernel_ctx.push(MatrixInstruction(header, spad, acc))

@check_kernel_ctx
def mx_reciprocal(tile: ATile, sem: Optional[Semaphore], aq: bool = True, rl: bool = True) -> None:
    assert tile.shape == (1, __g_kernel_ctx.cols)
    header = build_matrix_instruction_header(
        MxFunc.ACC_RECIPROCOL.value, True,
        sem, aq, rl
    )
    spad = MatrixInstructionSpad(0, 0, False, False, False)
    acc = MatrixInstrucionAcc(
        __g_kernel_ctx.tile_row_addr(tile),
        __g_kernel_ctx.tile_stride(tile),
        False
    )
    __g_kernel_ctx.push(MatrixInstruction(header, spad, acc))

@check_kernel_ctx
def mx_attn_lse_norm(tile: ATile, sem: Optional[Semaphore], aq: bool = True, rl: bool = True) -> None:
    assert len(tile.shape) == 2 and tile.shape[-1] == __g_kernel_ctx.cols
    header = build_matrix_instruction_header(
        MxFunc.ATTN_LSE_NORM.value, True,
        sem, aq, rl
    )
    spad = MatrixInstructionSpad(0, 0, False, False, False)
    acc = MatrixInstrucionAcc(
        __g_kernel_ctx.tile_row_addr(tile),
        __g_kernel_ctx.tile_stride(tile),
        False
    )
    __g_kernel_ctx.push(MatrixInstruction(header, spad, acc))
