import struct
import os
import subprocess
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from .kernel import Kernel
from .tensor import MTile
from .utils import ElfWriter
from .config import get_config
from .dtype import to_numpy_dtype
from .xdma import dev_read, dev_write
from .xdma_mmio import MMIO

class BaseEngine(ABC):
    @abstractmethod
    def execute(self, kernel: Kernel) -> None | MTile | list[MTile]:
        pass

class VerilatorSimulator(BaseEngine):
    def __init__(self, simulator_path: str,
                output_dir: str = '/tmp', max_cycles: int = 10000000, verbose=True,
                dram_sim: bool=False,
                vcdfile: Optional[str]=None,
                dram_sim_ini_dir: Optional[str]=None,
                numactl_cmd: str = None,
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
        self.numactl_cmd = numactl_cmd
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
        writer = ElfWriter(segments, get_config().mem_align)
        writer.write_elf(filename)

    def execute(self, kernel: Kernel) -> None | MTile | list[MTile]:
        # prepare inputs for simulator
        ui32_lst = [elem for inst in kernel.instructions for elem in inst.to_ui32_list()]
        bytes  = struct.pack(f'{len(ui32_lst)}I', *ui32_lst)
        inst_file = os.path.join(self.output_dir, 'inst.bin')
        with open(inst_file, 'wb') as f:
            f.write(bytes)
        mem_file = os.path.join(self.output_dir, 'mem.elf')
        self.dump_mem_elf(mem_file, kernel.input)
        sim_cmd = [self.simulator_path, inst_file]
        if self.numactl_cmd:
            sim_cmd = self.numactl_cmd.split() + [self.simulator_path, inst_file]
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
        if isinstance(kernel.output, MTile):
            output_list = [kernel.output]
        elif isinstance(kernel.output, list):
            output_list = kernel.output
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
        return kernel.output

class FPGA(BaseEngine):
    def __init__(self, control_dev: str = "/dev/xdma0_user", mem_read_dev: str = "/dev/xdma0_c2h_0", mem_write_dev: str = "/dev/xdma0_h2c_0"):
        super().__init__()
        self.control_dev = control_dev
        self.mem_read_dev = mem_read_dev
        self.mem_write_dev = mem_write_dev
        self.INST_QUEUE_OFFSET = 0x0 # offset for instruction queue register
        self.SET_ACTIVE_OFFSET = 0x4  # offset for setting active state
        self.STATE_OFFSET = 0x8  # offset for device state register
        self.PERF_EXEC_TIME = 0xc
        self.PERF_MX_BUBBLE = 0x10
        self.PERF_MX_ACTIVE = 0x14
        self.PERF_DMA_ACTIVE = 0x18
        self.PERF_RAW_INST = 0x1c
        self.PERF_MX_INST = 0x20
        self.PERF_DMA_INST = 0x24
        self.PERF_FENCE_INST = 0x28
        self.PERF_ENQ_INST = 0x2c
        self.PERF_DEQ_INST = 0x30
        self.MMIO = MMIO(control_dev)
    def execute(self, kernel: Kernel) -> None | MTile | list[MTile]:
        output_tensors = kernel.output
        output_tensors = [output_tensors]
        for tensor in output_tensors:
            if tensor.size > 0:
                print(f"Reading back output tensor from addr {hex(tensor.data_ptr)}, size {tensor.size}")
                data = dev_read(self.mem_read_dev, tensor.data_ptr, tensor.size)
                data.tofile("/tmp/debug-fpga{hex(tensor.data_ptr)}.bin")



        ui32_lst = [elem for inst in kernel.instructions for elem in inst.to_ui32_list()]

        input_tensors = kernel.input

        # write to memory
        if isinstance(input_tensors, MTile):
            input_tensors = [input_tensors]
        elif not isinstance(input_tensors, list):
            raise TypeError("input_tensors must be a MTile or a list of MTiles")
        for tensor in input_tensors:
            if tensor.data is not None:
                # tensor to numpy array
                numpy_array = np.frombuffer(tensor.data, dtype=to_numpy_dtype(tensor.dtype))
                dev_write(self.mem_write_dev, tensor.data_ptr, numpy_array)
                numpy_array.tofile(f"/tmp/{hex(tensor.data_ptr)}")
        
        # make sure device in idle state
        state = self.MMIO.dev_mmio_read(self.STATE_OFFSET)

        if state != 0:
            raise RuntimeError(f"Device is not idle, current state: {state}")
        
        # activate device
        self.MMIO.dev_mmio_write(self.SET_ACTIVE_OFFSET, 0xffffffff)

        # make sure device is active
        state = self.MMIO.dev_mmio_read(addr = self.STATE_OFFSET)
        if state != 1:
            raise RuntimeError(f"Device is not active, current state: {state}")

        # write instructions to control device
        inst_bytes = struct.pack(f'{len(ui32_lst)}I', *ui32_lst)
        self.MMIO.dev_queue_mmio_write(self.INST_QUEUE_OFFSET, ui32_lst)

        # wait for device to finish
        cycles = 0
        while True:
            state = self.MMIO.dev_mmio_read(self.STATE_OFFSET)
            if state == 2:
                print("Device finished execution")
                print("Performance counters:")
                print(f"Execution time: {self.MMIO.dev_mmio_read(self.PERF_EXEC_TIME)} cycles")
                print(f"Max bubble cycles: {self.MMIO.dev_mmio_read(self.PERF_MX_BUBBLE)} cycles")
                print(f"Max active cycles: {self.MMIO.dev_mmio_read(self.PERF_MX_ACTIVE)} cycles")
                print(f"DMA active cycles: {self.MMIO.dev_mmio_read(self.PERF_DMA_ACTIVE)} cycles")
                print(f"Raw instructions: {self.MMIO.dev_mmio_read(self.PERF_RAW_INST)}")
                print(f"Max instructions: {self.MMIO.dev_mmio_read(self.PERF_MX_INST)}")
                print(f"DMA instructions: {self.MMIO.dev_mmio_read(self.PERF_DMA_INST)}")
                print(f"Fence instructions: {self.MMIO.dev_mmio_read(self.PERF_FENCE_INST)}")
                print(f"Enqueue instructions: {self.MMIO.dev_mmio_read(self.PERF_ENQ_INST)}")
                print(f"Dequeue instructions: {self.MMIO.dev_mmio_read(self.PERF_DEQ_INST)}")
                break
        # read back output tensors
        output_tensors = kernel.output
        if isinstance(output_tensors, MTile):
            output_tensors = [output_tensors]
        elif not isinstance(output_tensors, list):
            output_tensors = []

        for tensor in output_tensors:
            if tensor.size > 0:
                print(f"Reading back output tensor from addr {hex(tensor.data_ptr)}, size {tensor.size}")
                data = dev_read(self.mem_read_dev, tensor.data_ptr, tensor.size)
                data.tofile("/tmp/debug-fpga.bin")
                tensor.data = data

        return kernel.output
