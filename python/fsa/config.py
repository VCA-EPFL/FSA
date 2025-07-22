from .dtype import *
from .mem import CompoundMemoryManger
import json

@dataclass(frozen=True)
class FSAConfig:
    sa_rows: int = 16
    sa_cols: int = 16
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

@dataclass(frozen=True)
class FSAGlobalVariables:
    config: FSAConfig
    mem_manager: CompoundMemoryManger

__global_vars: FSAGlobalVariables = None

def init(config_file: str):
    global __global_vars
    assert __global_vars is None, "FSA is already initialized."

    with open(config_file, 'r') as f:
        cfg = json.load(f)
    cfg["e_type"] = eval(cfg["e_type"])
    cfg["a_type"] = eval(cfg["a_type"])
    config = FSAConfig(**cfg)
    mem_manger = CompoundMemoryManger(
        mem_base=config.mem_base,
        mem_size=config.mem_size,
        mem_align=config.mem_align,

        spad_base=config.spad_base,
        spad_size=config.spad_size,
        spad_align=config.sa_cols * config.e_type.itemsize,
        spad_dtype=config.e_type,

        acc_base=config.acc_base,
        acc_size=config.acc_size,
        acc_align=config.sa_cols * config.a_type.itemsize,
        acc_dtype=config.a_type
    )
    __global_vars = FSAGlobalVariables(config, mem_manger)

def require_initialized():
    global __global_vars
    if __global_vars is None:
        raise RuntimeError("FSA is not initialized. Call init() with a config file before using FSA.")
    return __global_vars

def get_config() -> FSAConfig:
    require_initialized()
    return __global_vars.config

def get_mem_manager() -> CompoundMemoryManger:
    require_initialized()
    return __global_vars.mem_manager
