from pyverilator import PyVerilator

class SignalWrapper:
    def __init__(self, sim: PyVerilator, index: int):
        self.sim = sim
        self.index = index

    def reset(self):
        raise NotImplementedError(f"Reset not implemented for {self.__class__.__name__}")

    def _to_signal_name(self, name: str) -> str:
        raise NotImplementedError(f"Signal name mapping not implemented for {self.__class__.__name__}")

    def __getattr__(self, name):
        if name in ['sim', 'index']:
            return super().__getattr__(name)
        else:
            return self.sim._read(self._to_signal_name(name))
    
    def __setattr__(self, name, value):
        if name in ['sim', 'index']:
            super().__setattr__(name, value)
        else:
            self.sim._write(self._to_signal_name(name), value)

class Instruction(SignalWrapper):
    valid: int
    ready: int
    bits: int
    def __init__(self, sim: PyVerilator):
        super().__init__(sim, 0)
        self.reset()

    def reset(self):
        self.valid = 0
        self.bits = 0

    def _to_signal_name(self, name: str) -> str:
        return f"io_debug_mx_inst_{name}"

    def set(self, bits: int) -> None:
        self.valid = 1
        self.bits = bits

class SRAMAddr(SignalWrapper):
    en_sp: int
    en_acc: int
    addr: int
    read: int
    write: int
    def __init__(self, sim: PyVerilator):
        super().__init__(sim, 0)

    def reset(self):
        self.en_sp = 0
        self.en_acc = 0
        self.addr = 0
        self.read = 0
        self.write = 0

    def _to_signal_name(self, name: str) -> str:
        return f"io_debug_sram_io_{name}"


class SRAMData(SignalWrapper):
    rdata: int
    wdata: int
    def __init__(self, sim, index):
        super().__init__(sim, index)

    def reset(self):
        self.wdata = 0

    def _to_signal_name(self, name: str) -> str:
        return f"io_debug_sram_io_{name}_{self.index}"