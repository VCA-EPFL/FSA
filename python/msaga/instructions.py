from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Sequence
from enum import Enum

class InstructionType(Enum):
    FENCE = 0
    MX = 1
    DMA = 2

class MxFunc(Enum):
    LOAD_STATIONARY = 0
    ATTN_SCORE = 1
    ATTN_VALUE = 2
    ACC_RECIPROCOL = 3
    ATTN_LSE_NORM = 4

class DMAFunc(Enum):
    LD_SRAM = 0
    ST_SRAM = 1

class InstructionLike(ABC):

    def combine_fields(fs: Sequence[tuple[int|bool, int, int]]):
        bits = 0
        for x, msb, lsb in fs:
            if isinstance(x, bool):
                x = int(x)
            n_bits = msb - lsb + 1
            x &= ((1 << n_bits) - 1)
            bits |= x << lsb
        return bits

    @property
    @abstractmethod
    def bits(self) -> int:
        pass

class Instruction(InstructionLike):

    @property
    @abstractmethod
    def i_type(self) -> InstructionType:
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    def to_ui32_list(self) -> list[int]:
        n_pieces = self.width // 32
        res = []
        bits = self.bits
        for _ in range(n_pieces):
            ui32 = bits & 0xFFFFFFFF
            res.append(ui32)
            bits >>= 32
        return res

@dataclass
class FenceInstruction(Instruction):

    mx: bool
    dma: bool
    stop: bool

    @property
    def i_type(self) -> InstructionType:
        return InstructionType.FENCE

    @property
    def width(self) -> int:
        return 32

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields((
            (self.i_type.value, 31, 29),
            (self.mx, 28, 28),
            (self.dma, 27, 27),
            (self.stop, 26, 26),
        ))

@dataclass
class MatrixInstructionHeader(InstructionLike):
    func: int
    waitPrevAcc: bool
    consumer_semaphore: int
    producer_semaphore: int

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields((
            (self.func, 28, 24),
            (self.waitPrevAcc, 23, 23),
            (self.consumer_semaphore, 22, 15),
            (self.producer_semaphore, 14, 7)
        ))

@dataclass
class MatrixInstructionSpad(InstructionLike):
    addr: int
    stride: int
    revInput: bool
    revOutput: bool
    delayOutput: bool

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields((
            (self.addr, 31, 12),
            (self.stride, 11, 7),
            (self.revInput, 6, 6),
            (self.revOutput, 5, 5),
            (self.delayOutput, 4, 4),
        ))

@dataclass
class MatrixInstrucionAcc(InstructionLike):
    addr: int
    stride: int
    zero: bool

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields((
            (self.addr, 31, 12),
            (self.stride, 11, 7),
            (self.zero, 6, 6),
        ))

@dataclass
class MatrixInstruction(Instruction):
    header: MatrixInstructionHeader
    spad: MatrixInstructionSpad
    acc: MatrixInstrucionAcc

    @property
    def i_type(self) -> InstructionType:
        return InstructionType.MX

    @property
    def width(self) -> int:
        return 3 * 32

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields((
            (self.header.bits, 28, 0),
            (self.i_type.value, 31, 29),
            (self.spad.bits, 63, 32),
            (self.acc.bits, 95, 64),
        ))

@dataclass
class DMAInstructionHeader(InstructionLike):
    func: int
    consumer_semaphore: int
    producer_semaphore: int
    repeat: int

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields((
            (self.func, 28, 25),
            (self.consumer_semaphore, 24, 17),
            (self.producer_semaphore, 16, 9),
            (self.repeat, 8, 0)
        ))


@dataclass
class DMAInstrucionSRAM(InstructionLike):
    addr: int
    stride: int
    isAccum: bool

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields((
            (self.addr, 31, 12),
            (self.stride, 11, 7),
            (self.isAccum, 6, 6),
        ))


@dataclass
class DMAInstrucionMem(InstructionLike):
    addr: int
    stride: int
    size: int

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields((
            (self.addr, 63, 25),
            (self.stride, 24, 15),
            (self.size, 14, 5)
        ))

@dataclass
class DMAInstruction(Instruction):
    header: DMAInstructionHeader
    sram: DMAInstrucionSRAM
    mem: DMAInstrucionMem

    @property
    def i_type(self) -> InstructionType:
        return InstructionType.DMA

    @property
    def width(self) -> int:
        return 4 * 32

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields((
            (self.header.bits, 28, 0),
            (self.i_type.value, 31, 29),
            (self.sram.bits, 63, 32),
            (self.mem.bits, 127, 64),
        ))

class Semaphore:
    def __init__(self, id: int, n: int):
        assert 0 < id < 32 and 0 < n < 8
        self.id = id
        self.n = n
        self.value = 0

    def inc(self) -> 'Semaphore':
        if self.value == self.n - 1:
            self.value = 0
        else:
            self.value += 1
        return self

    def to_inst_field(self) -> int:
        #   id    v
        # |xxxxx|xxx|
        return (self.id << 3) | self.value