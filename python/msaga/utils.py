from elftools.elf.enums import *
from elftools.elf.constants import P_FLAGS
from elftools.elf.structs import ELFStructs

from .tensor import MTile
from .dtype import get_dtype
from .mem import g_mem_manger

import numpy as np


def from_numpy(array: np.ndarray) -> MTile:
    """Create a MTile from a numpy ndarray"""
    finfo = np.finfo(array.dtype)
    dtype = get_dtype(finfo.nexp, finfo.nmant)
    tile = g_mem_manger.alloc_mem(array.shape, dtype=dtype)
    tile.data = array.tobytes(order='C')
    return tile

class DictToClass:
    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                v = DictToClass(v)
            self.__setattr__(k, v)

class ElfWriter:
    def __init__(self, segments: list[tuple[int, int, bytes]], alignment: int):
        """
        segments: list[(BaseAddr, Size, Bytes)]
        """
        self.structs = ELFStructs(elfclass=64)
        self.structs.create_basic_structs()
        self.structs.create_advanced_structs(
            e_type=ENUM_E_TYPE['ET_NONE'],
            e_machine=ENUM_E_MACHINE['EM_RISCV'],
            e_ident_osabi=0
        )
        self.data_offset = 64 + 56 * len(segments)
        self.data_alignment = alignment
        self.segments: list[dict] = [
            self.__add_segment(addr, size, data)
            for (addr, size, data) in segments
        ]


    def __align(self, offset: int) -> int:
        if offset % self.data_alignment != 0:
            return offset + (self.data_alignment - (offset % self.data_alignment))
        return offset

    def __add_segment(self, addr: int, size: int, data: bytes) -> dict:
        self.data_offset = self.__align(self.data_offset)
        print(hex(addr), size, hex(self.data_offset))
        segment = {
            'p_type': ENUM_P_TYPE_RISCV['PT_LOAD'],
            'p_offset': self.data_offset,
            'p_vaddr': addr,
            'p_paddr': addr,
            'p_filesz': size,
            'p_memsz': size,
            'p_flags': P_FLAGS.PF_R | P_FLAGS.PF_W,
            'p_align': self.data_alignment,
            'data': data
        }
        self.data_offset += size
        return segment

    def write_elf(self, filename: str):
        with open(filename, 'wb') as f:
            # header
            f.write(self.structs.Elf_Ehdr.build(DictToClass({
                'e_ident': {
                    'EI_MAG': b'\x7fELF',
                    'EI_CLASS': 'ELFCLASS64',
                    'EI_DATA': 'ELFDATA2LSB',
                    'EI_VERSION': 1,
                    'EI_OSABI': 0,
                    'EI_ABIVERSION': 0,
                    'EI_PAD': bytes(7)
                },
                'e_type': ENUM_E_TYPE['ET_NONE'],
                'e_machine': ENUM_E_MACHINE['EM_RISCV'],
                'e_version': ENUM_E_VERSION['EV_CURRENT'],
                'e_entry': 0,
                'e_phoff': 64 if self.segments else 0,
                'e_shoff': 0,  # No section headers, we create segments directly
                'e_flags': 0,
                'e_ehsize': 64,
                'e_phentsize': 56,
                'e_phnum': len(self.segments),
                'e_shentsize': 0,
                'e_shnum': 0,
                'e_shstrndx': 0
            })))
            # segments
            for seg in self.segments:
                f.write(self.structs.Elf_Phdr.build(DictToClass(seg)))
            # segment data
            for seg in self.segments:
                cur_offset = f.tell()
                padding = seg['p_offset'] - cur_offset
                if padding > 0:
                    f.write(b'\x00' * padding)
                f.write(seg['data'])

def dump_mem_elf(filename: str) -> None:
    # TODO: merge consecutive segments
    segments = [
        (x.data_ptr, x.size, x.data)
        for x in g_mem_manger.mem_tensor_list if x.data is not None
    ]
    writer = ElfWriter(segments, g_mem_manger.mem.alignment)
    writer.write_elf(filename)