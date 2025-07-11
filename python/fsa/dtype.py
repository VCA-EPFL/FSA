import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class dtype:
    itemsize: int

fp32 = dtype(4)
fp16 = dtype(2)
bf16 = dtype(2)
fp8 = dtype(1)

def get_dtype(ew: int, mw: int) -> dtype:
    match (ew, mw):
        case (8, 23):
            return fp32
        case (8, 7):
            return bf16
        case (5, 10):
            return fp16
        case (4, 3):
            return fp8
        case _:
            raise ValueError(f"Unknown dtype: e{ew}m{mw}")

def from_numpy_dtype(n_type: np.dtype):
    info = np.finfo(n_type)
    return get_dtype(info.nexp, info.nmant)

def to_numpy_dtype(t: dtype):
    type_dict = {
        fp32: np.float32,
        fp16: np.float16,
    }
    return type_dict[t]