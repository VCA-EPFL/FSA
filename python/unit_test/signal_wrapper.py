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