from .dtype import *
import json

@dataclass(frozen=True)
class MSAGAConfig:
    sa_rows: int = 4
    sa_cols: int = 4
    inst_queue_size: int = 256
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

def load_config(config_file: str) -> MSAGAConfig:
    global g_config
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    cfg["e_type"] = eval(cfg["e_type"])
    cfg["a_type"] = eval(cfg["a_type"])
    g_config = MSAGAConfig(**cfg)
    return g_config
