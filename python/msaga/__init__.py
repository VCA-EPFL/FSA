import numpy as np
from .dtype import *
from .kernel import *
from .tensor import STile, ATile, MTile
from .engine import VerilatorSimulator
from .instructions import Semaphore
from .config import g_config
from .mem import g_mem_manger
from .utils import *



def alloc_spad(shape: int | tuple[int, ...]) -> STile:
    return g_mem_manger.alloc_spad(shape)

def alloc_accumulator(shape: int | tuple[int, ...]) -> ATile:
    return g_mem_manger.alloc_accumulator(shape)

def alloc_mem(shape: int | tuple[int, ...], dtype: dtype) -> MTile:
    return g_mem_manger.alloc_mem(shape, dtype)

def from_numpy(array: np.ndarray) -> MTile:
    """Create a MTile from a numpy ndarray"""
    finfo = np.finfo(array.dtype)
    dtype = get_dtype(finfo.nexp, finfo.nmant)
    tile = g_mem_manger.alloc_mem(array.shape, dtype=dtype)
    tile.data = array.tobytes(order='C')
    return tile

def to_numpy(tile: MTile) -> np.ndarray:
    assert tile.data is not None
    assert tile.is_contiguous()
    arr = np.frombuffer(tile.data, dtype=to_numpy_dtype(tile.dtype))
    return arr.reshape(tile.shape)