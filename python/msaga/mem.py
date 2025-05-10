from .tensor import STile, ATile, MTile


class MemoryManger:
    pass

class CompoundMemoryManger:

    def alloc_spad(self) -> STile:
        pass

    def alloc_accumulator(self) -> ATile:
        pass

    def alloc_mem(self) -> MTile:
        pass

