
from .static_builder import StaticInstBuilder, Semaphore
from .mem import CompoundMemoryManger
from .tensor import STile, ATile, MTile
from .dtype import *

# TODO: do not hard code!
__mem_manger = CompoundMemoryManger(
    mem_base=0x80000000, mem_size=0x10000, mem_align=8,
    spad_base=0, spad_size=0x1000, spad_align= 4 * fp16.itemsize, spad_dtype=fp16,
    acc_base=0, acc_size=0x1000, acc_align= 4 * fp32.itemsize, acc_dtype=fp32
)

alloc_spad = __mem_manger.alloc_spad
alloc_accumulator = __mem_manger.alloc_accumulator
alloc_mem = __mem_manger.alloc_mem