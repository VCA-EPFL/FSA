from .dtype import *

@dataclass
class MSAGAConfig:
    sa_rows: int = 4
    sa_cols: int = 4
    e_type: dtype = fp16
    a_type: dtype = fp32
    mem_base: int = 0x80000000
    mem_size: int = 0x10000000
    mem_align: int = 32
    spad_base: int = 0
    spad_size: int = 0x1000
    acc_base: int = 0
    acc_size: int = 0x1000

g_config = MSAGAConfig()