import struct
from typing import Optional
from .instructions import *
from .tensor import MTile, STile, ATile

class StaticProgram:

    def __init__(self, instructions: list[Instruction]):
        ui32_lst = [elem for inst in instructions for elem in inst.to_ui32_list()]
        self.bytes  = struct.pack(f'{len(ui32_lst)}I', *ui32_lst)
        self.instructions = instructions

    def save_to_file(self, filename: str):
        with open(filename, 'wb') as f:
            f.write(self.bytes)

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

class BaseInstructionBuilder:

    def __init__(self):
        self.instructions: list[Instruction] = []

    def _add(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def compile(self) -> StaticProgram:
        return StaticProgram(self.instructions)


class StaticInstBuilder(BaseInstructionBuilder):

    def __init__(self, systolic_array_rows: int, systolic_array_cols: int):
        super().__init__()
        self.rows = systolic_array_rows
        self.cols = systolic_array_cols

    def __tile_row_addr(self, tile: ATile | STile) -> int:
        # on-chip SRAMs are not byte-addressed, they are row-addressed
        if isinstance(tile, STile):
            assert tile.shape[-1] == self.rows
        else:
            assert tile.shape[-1] == self.cols
        cols, itemsize = tile.shape[-1], tile.dtype.itemsize
        return tile.data_ptr // (cols * itemsize)

    def __tile_stride(self, tile: ATile | STile) -> int:
        if isinstance(tile, STile):
            assert tile.shape[-1] == self.rows
        else:
            assert tile.shape[-1] == self.cols
        return tile.stride[-2] // tile.shape[-1]


    def dma(self, func: int, mem: MTile, tile: ATile | STile, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
        assert mem.shape == tile.shape and len(mem.shape) == 2 and mem.dtype == tile.dtype
        rows, cols = mem.shape
        mem = DMAInstrucionMem(mem.data_ptr, mem.stride[-2] * mem.dtype.itemsize, cols * mem.dtype.itemsize)
        sram = DMAInstrucionSRAM(
            self.__tile_row_addr(tile),
            self.__tile_stride(tile),
            isAccum=isinstance(tile, ATile)
        )
        header = DMAInstructionHeader(
            func,
            consume.to_inst_field() if consume else 0,
            produce.inc().to_inst_field() if produce else 0,
            repeat=rows
        )
        self._add(DMAInstruction(header, sram, mem))

    def load_tile(self, mem: MTile, tile: STile, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
        self.dma(DMAFunc.LD_SRAM.value, mem, tile, consume, produce)


    def store_tile(self, tile: ATile, mem: MTile, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
        self.dma(DMAFunc.ST_SRAM, mem, tile, consume, produce)

    def mx_load_stationary(self, tile: STile, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
        assert len(tile.shape) == 2 and tile.shape[-1] == self.rows
        header = MatrixInstructionHeader(
            MxFunc.LOAD_STATIONARY.value, False,
            consume.to_inst_field() if consume else 0,
            produce.inc().to_inst_field() if produce else 0
        )
        spad = MatrixInstructionSpad(
            self.__tile_row_addr(tile),
            self.__tile_stride(tile),
            False, False, False
        )
        acc = MatrixInstrucionAcc(0, 0, False)
        self._add(MatrixInstruction(header, spad, acc))

    def mx_attn_score(self, k: STile, l: ATile, accumulate: bool, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
        assert len(k.shape) == 2 and l.shape == (1, self.cols)
        header = MatrixInstructionHeader(
            MxFunc.ATTN_SCORE.value, False,
            consume.to_inst_field() if consume else 0,
            produce.inc().to_inst_field() if produce else 0
        )
        spad = MatrixInstructionSpad(
            self.__tile_row_addr(k),
            self.__tile_stride(k),
            True, True, True
        )
        acc = MatrixInstrucionAcc(
            self.__tile_row_addr(l),
            self.__tile_stride(l),
            not accumulate
        )
        self._add(MatrixInstruction(header, spad, acc))

    def mx_attn_value(self, v_t: STile, o_t: ATile, accumulate: bool, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
        assert len(v_t.shape) == 2 and len(o_t.shape) == 2
        header = MatrixInstructionHeader(
            MxFunc.ATTN_VALUE.value, False,
            consume.to_inst_field() if consume else 0,
            produce.inc().to_inst_field() if produce else 0
        )
        spad = MatrixInstructionSpad(
            self.__tile_row_addr(v_t),
            self.__tile_stride(v_t),
            True, False, True
        )
        acc = MatrixInstrucionAcc(
            self.__tile_row_addr(o_t),
            self.__tile_stride(o_t),
            not accumulate
        )
        self._add(MatrixInstruction(header, spad, acc))

    def mx_reciprocal(self, tile: ATile, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
        assert tile.shape == (1, self.cols)
        header = MatrixInstructionHeader(
            MxFunc.ACC_RECIPROCOL.value, True,
            consume.to_inst_field() if consume else 0,
            produce.inc().to_inst_field() if produce else 0
        )
        spad = MatrixInstructionSpad(0, 0, False, False, False)
        acc = MatrixInstrucionAcc(
            self.__tile_row_addr(tile),
            self.__tile_stride(tile),
            False
        )
        self._add(MatrixInstruction(header, spad, acc))

    def mx_attn_lse_norm(self, tile: ATile, consume: Optional[Semaphore], produce: Optional[Semaphore]) -> None:
        assert len(tile.shape) == 2 and tile.shape[-1] == self.cols
        header = MatrixInstructionHeader(
            MxFunc.ATTN_LSE_NORM.value, True,
            consume.to_inst_field() if consume else 0,
            produce.inc().to_inst_field() if produce else 0
        )
        spad = MatrixInstructionSpad(0, 0, False, False, False)
        acc = MatrixInstrucionAcc(
            self.__tile_row_addr(tile),
            self.__tile_stride(tile),
            False
        )
        self._add(MatrixInstruction(header, spad, acc))
