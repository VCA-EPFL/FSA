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
def dma(func: int, mem: MTile, tile: ATile | STile, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
    assert mem.shape == tile.shape and len(mem.shape) == 2 and mem.dtype == tile.dtype
    rows, cols = mem.shape
    mem = DMAInstrucionMem(mem.data_ptr, mem.stride[-2] * mem.dtype.itemsize, cols * mem.dtype.itemsize)
    sram = DMAInstrucionSRAM(
        __g_kernel_ctx.tile_row_addr(tile),
        __g_kernel_ctx.tile_stride(tile),
        isAccum=isinstance(tile, ATile)
    )
    header = DMAInstructionHeader(
        func,
        consume.to_inst_field() if consume else 0,
        produce.inc().to_inst_field() if produce else 0,
        repeat=rows
    )
    __g_kernel_ctx.push(DMAInstruction(header, sram, mem))

@check_kernel_ctx
def load_tile(mem: MTile, tile: STile, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
    dma(DMAFunc.LD_SRAM.value, mem, tile, consume, produce)


@check_kernel_ctx
def store_tile(tile: ATile, mem: MTile, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
    dma(DMAFunc.ST_SRAM.value, mem, tile, consume, produce)

@check_kernel_ctx
def mx_load_stationary(tile: STile, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
    assert len(tile.shape) == 2 and tile.shape[-1] == __g_kernel_ctx.rows
    header = MatrixInstructionHeader(
        MxFunc.LOAD_STATIONARY.value, False,
        consume.to_inst_field() if consume else 0,
        produce.inc().to_inst_field() if produce else 0
    )
    spad = MatrixInstructionSpad(
        __g_kernel_ctx.tile_row_addr(tile),
        __g_kernel_ctx.tile_stride(tile),
        False, False, False
    )
    acc = MatrixInstrucionAcc(0, 0, False)
    __g_kernel_ctx.push(MatrixInstruction(header, spad, acc))

@check_kernel_ctx
def mx_attn_score(k: STile, l: ATile, accumulate: bool, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
    assert len(k.shape) == 2 and l.shape == (1, __g_kernel_ctx.cols)
    header = MatrixInstructionHeader(
        MxFunc.ATTN_SCORE.value, False,
        consume.to_inst_field() if consume else 0,
        produce.inc().to_inst_field() if produce else 0
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
def mx_attn_value(v_t: STile, o_t: ATile, accumulate: bool, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
    assert len(v_t.shape) == 2 and len(o_t.shape) == 2
    header = MatrixInstructionHeader(
        MxFunc.ATTN_VALUE.value, False,
        consume.to_inst_field() if consume else 0,
        produce.inc().to_inst_field() if produce else 0
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
def mx_reciprocal(tile: ATile, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
    assert tile.shape == (1, __g_kernel_ctx.cols)
    header = MatrixInstructionHeader(
        MxFunc.ACC_RECIPROCOL.value, True,
        consume.to_inst_field() if consume else 0,
        produce.inc().to_inst_field() if produce else 0
    )
    spad = MatrixInstructionSpad(0, 0, False, False, False)
    acc = MatrixInstrucionAcc(
        __g_kernel_ctx.tile_row_addr(tile),
        __g_kernel_ctx.tile_stride(tile),
        False
    )
    __g_kernel_ctx.push(MatrixInstruction(header, spad, acc))

@check_kernel_ctx
def mx_attn_lse_norm(tile: ATile, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
    assert len(tile.shape) == 2 and tile.shape[-1] == __g_kernel_ctx.cols
    header = MatrixInstructionHeader(
        MxFunc.ATTN_LSE_NORM.value, True,
        consume.to_inst_field() if consume else 0,
        produce.inc().to_inst_field() if produce else 0
    )
    spad = MatrixInstructionSpad(0, 0, False, False, False)
    acc = MatrixInstrucionAcc(
        __g_kernel_ctx.tile_row_addr(tile),
        __g_kernel_ctx.tile_stride(tile),
        False
    )
    __g_kernel_ctx.push(MatrixInstruction(header, spad, acc))
