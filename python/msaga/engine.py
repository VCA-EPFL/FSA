import struct
import os
import subprocess
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from .instructions import Instruction
from .tensor import MTile
from .utils import ElfWriter
from .config import g_config
from .dtype import to_numpy_dtype

class BaseEngine(ABC):
    @abstractmethod
    def execute(self, instructions: list[Instruction], input_tensors: list[MTile], output_tensors: MTile | list[MTile] | None) -> None:
        pass

class VerilatorSimulator(BaseEngine):
    def __init__(self, simulator_path: str,
                output_dir: str = '/tmp', max_cycles: int = 10000000, verbose=True,
                dram_sim: bool=False,
                vcdfile: Optional[str]=None,
                dram_sim_ini_dir: Optional[str]=None
                ):
        super().__init__()
        assert os.path.isfile(simulator_path)
        assert os.path.isdir(output_dir)
        self.simulator_path = simulator_path
        self.output_dir = output_dir
        self.max_cycles = max_cycles
        self.verbose = verbose
        self.dram_sim = dram_sim
        self.vcdfile = vcdfile
        if dram_sim_ini_dir:
            self.dram_sim_ini_dir = dram_sim_ini_dir
            assert os.path.isdir(dram_sim_ini_dir)
        else:
            # try to infer dram sim ini path
            try_path = os.path.join(
                os.path.dirname((simulator_path)),
                '..', '..', 'generators',
                'testchipip', 'src', 'main',
                'resources', 'dramsim2_ini'
            )
            assert os.path.isdir(try_path), \
                f"Can't find dramsim ini dir, please specify it explicitly.\nTried the following path: {try_path}"
            self.dram_sim_ini_dir = try_path

    @staticmethod
    def dump_mem_elf(filename: str, tensors: list[MTile]):
        segments = [
            (x.data_ptr, x.size, x.data)
            for x in tensors if x.data is not None
        ]
        writer = ElfWriter(segments, g_config.mem_align)
        writer.write_elf(filename)

    def execute(self, instructions: list[Instruction], input_tensors: list[MTile], output_tensors: MTile | list[MTile] | None) -> None:
        # prepare inputs for simulator
        ui32_lst = [elem for inst in instructions for elem in inst.to_ui32_list()]
        bytes  = struct.pack(f'{len(ui32_lst)}I', *ui32_lst)
        inst_file = os.path.join(self.output_dir, 'inst.bin')
        with open(inst_file, 'wb') as f:
            f.write(bytes)
        mem_file = os.path.join(self.output_dir, 'mem.elf')
        self.dump_mem_elf(mem_file, input_tensors)
        sim_cmd = [self.simulator_path, inst_file]
        if self.dram_sim:
            sim_cmd.append('+dramsim')
            sim_cmd.append(f'+dramsim_ini_dir={self.dram_sim_ini_dir}')
        sim_cmd.append(f'+loadmem={mem_file}')
        sim_cmd.append(f'+max-cycles={self.max_cycles}')
        if self.verbose:
            sim_cmd.append('+verbose')
        if self.vcdfile:
            sim_cmd.append(f'+vcdfile={self.vcdfile}')
        output_list: list[MTile]
        output_filenames: list[str] = []
        if isinstance(output_tensors, MTile):
            output_list = [output_tensors]
        elif isinstance(output_tensors, list):
            output_list = output_tensors
        else:
            output_list = []
        for out in output_list:
            out_filename = os.path.join(self.output_dir, hex(out.data_ptr) + ".bin")
            output_filenames.append(out_filename)
            sim_cmd.append(f'+dump-mem={out_filename}:{hex(out.data_ptr)}:{hex(out.size)}')
        print(f'Start simulation with cmd: {sim_cmd}')
        subprocess.run(sim_cmd, check=True)
        for (out, out_filename) in zip(output_list, output_filenames):
            arr = np.fromfile(out_filename, dtype=to_numpy_dtype(out.dtype))
            out.data = arr.tobytes(order='C')
        return output_tensors

