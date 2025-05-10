from dataclasses import dataclass

@dataclass
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