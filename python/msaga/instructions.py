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

@dataclass
class InstructionField:
    value: int | bool
    msb: int
    lsb: int
    signed: bool = False

    @property
    def width(self) -> int:
        return self.msb - self.lsb + 1

    def shifted_value(self) -> int:
        return (self.value & ((1 << self.width) - 1)) << self.lsb

    def __post_init__(self):
        if isinstance(self.value, bool):
            assert not self.signed, "Boolean fields cannot be signed"
            self.value = int(self.value)
        if self.signed:
            assert - (1 << (self.width - 1)) <= self.value < (1 << (self.width - 1)), \
                f"Value {self.value} cannot be represented in {self.width} bits as signed"
        else:
            assert 0 <= self.value < (1 << self.width), \
                f"Value {self.value} cannot be represented in {self.width} bits as unsigned"


class InstructionLike(ABC):

    def combine_fields(fs: Sequence[InstructionField]) -> int:
        bits = 0
        for f in fs:
            bits |= f.shifted_value()
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

    @property
    def n_bytes(self) -> int:
        return self.width // 8

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
            InstructionField(self.i_type.value, 31, 29),
            InstructionField(self.mx, 28, 28),
            InstructionField(self.dma, 27, 27),
            InstructionField(self.stop, 26, 26),
        ))

@dataclass
class MatrixInstructionHeader(InstructionLike):
    semId: int
    acquireValid: bool
    acquireSemValue: int
    releaseValid: bool
    releaseSemValue: int
    func: int
    waitPrevAcc: bool

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields((
            InstructionField(self.semId, 28, 24),
            InstructionField(self.acquireValid, 23, 23),
            InstructionField(self.acquireSemValue, 22, 20),
            InstructionField(self.releaseValid, 19, 19),
            InstructionField(self.releaseSemValue, 18, 16),
            InstructionField(self.func, 15, 11),
            InstructionField(self.waitPrevAcc, 10, 10)
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
            InstructionField(self.addr, 31, 12),
            InstructionField(self.stride, 11, 7, signed=True),
            InstructionField(self.revInput, 6, 6),
            InstructionField(self.revOutput, 5, 5),
            InstructionField(self.delayOutput, 4, 4),
        ))

@dataclass
class MatrixInstrucionAcc(InstructionLike):
    addr: int
    stride: int
    zero: bool

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields((
            InstructionField(self.addr, 31, 12),
            InstructionField(self.stride, 11, 7, signed=True),
            InstructionField(self.zero, 6, 6),
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
            InstructionField(self.header.bits, 28, 0),
            InstructionField(self.i_type.value, 31, 29),
            InstructionField(self.spad.bits, 63, 32),
            InstructionField(self.acc.bits, 95, 64),
        ))

@dataclass
class DMAInstructionHeader(InstructionLike):
    semId: int
    acquireValid: bool
    acquireSemValue: int
    releaseValid: bool
    releaseSemValue: int
    func: int
    repeat: int

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields((
            InstructionField(self.semId, 28, 24),
            InstructionField(self.acquireValid, 23, 23),
            InstructionField(self.acquireSemValue, 22, 20),
            InstructionField(self.releaseValid, 19, 19),
            InstructionField(self.releaseSemValue, 18, 16),
            InstructionField(self.func, 15, 12),
            InstructionField(self.repeat, 11, 3)
        ))


@dataclass
class DMAInstrucionSRAM(InstructionLike):
    addr: int
    stride: int
    isAccum: bool

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields((
            InstructionField(self.addr, 31, 12),
            InstructionField(self.stride, 11, 7, signed=True),
            InstructionField(self.isAccum, 6, 6),
        ))


@dataclass
class DMAInstrucionMem(InstructionLike):
    addr: int
    stride: int
    size: int

    @property
    def bits(self) -> int:
        return InstructionLike.combine_fields((
            InstructionField(self.addr, 63, 25),
            InstructionField(self.stride, 24, 10, signed=True),
            InstructionField(self.size, 9, 0)
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
            InstructionField(self.header.bits, 28, 0),
            InstructionField(self.i_type.value, 31, 29),
            InstructionField(self.sram.bits, 63, 32),
            InstructionField(self.mem.bits, 127, 64),
        ))

class Semaphore:
    def __init__(self, id: int, n: int):
        assert 0 <= id < 32 and 0 < n < 8
        self.id = id
        self.n = n
        self.value = 0

    def inc(self) -> 'Semaphore':
        if self.value == self.n - 1:
            self.value = 0
        else:
            self.value += 1
        return self