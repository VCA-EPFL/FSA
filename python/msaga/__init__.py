
from .static_builder import StaticInstBuilder, Semaphore
from .mem import CompoundMemoryManger
from .tensor import STile, ATile, MTile
from .dtype import *

__mem_manger = CompoundMemoryManger()

def alloc_spad() -> STile:
    return __mem_manger.alloc_spad()

def alloc_accumulator() -> ATile:
    return __mem_manger.alloc_accumulator()

def alloc_mem() -> MTile:
    return __mem_manger.alloc_mem()
