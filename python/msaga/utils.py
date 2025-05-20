from elftools.elf.enums import *
from elftools.elf.constants import P_FLAGS
from elftools.elf.structs import ELFStructs

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
        """
        ELF layout:
        0x00: ELF header
        0x40: program header table
        0x40 + 56 * len(segments): section data (shstrtab content)
        0x40 + 56 * len(segments) + len(shstrtab): segment data
        segments[-1].p_offset + segments[-1].p_filesz: section header table

        """
        self.shstrtab_data = b'\x00dummy_section\x00.shstrtab\x00'
        self.dummy_section_name_offset = 1         # Points to "dummy_section"
        self.shstrtab_name_offset = 15             # Points to ".shstrtab"
        self.data_offset = 64 + 56 * len(segments) + len(self.shstrtab_data)
        self.data_alignment = alignment
        self.segments: list[dict] = [
            self.__add_segment(addr, size, data)
            for (addr, size, data) in segments
        ]
        if len(self.segments) > 0:
            self.section_header_offset = self.segments[-1]['p_offset'] + self.segments[-1]['p_filesz']
        else:
            self.section_header_offset = self.data_offset


    def __align(self, offset: int) -> int:
        if offset % self.data_alignment != 0:
            return offset + (self.data_alignment - (offset % self.data_alignment))
        return offset

    def __add_segment(self, addr: int, size: int, data: bytes) -> dict:
        self.data_offset = self.__align(self.data_offset)
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
            # ELF header
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
                'e_type': ENUM_E_TYPE['ET_EXEC'],
                'e_machine': ENUM_E_MACHINE['EM_RISCV'],
                'e_version': ENUM_E_VERSION['EV_CURRENT'],
                'e_entry': 0,
                'e_phoff': 64 if self.segments else 0,
                'e_shoff': self.section_header_offset,
                'e_flags': 0,
                'e_ehsize': 64,
                'e_phentsize': 56,
                'e_phnum': len(self.segments),
                'e_shentsize': 64,
                'e_shnum': 2,
                'e_shstrndx': 1
            })))
            # program header table
            for seg in self.segments:
                f.write(self.structs.Elf_Phdr.build(DictToClass(seg)))
            # section data (.shstrtab content)
            f.write(self.shstrtab_data)
            # segment data
            for seg in self.segments:
                cur_offset = f.tell()
                padding = seg['p_offset'] - cur_offset
                if padding > 0:
                    f.write(b'\x00' * padding)
                f.write(seg['data'])
            assert f.tell() == self.section_header_offset
            # section header table
            # dummy_section
            f.write(self.structs.Elf_Shdr.build(DictToClass({
                'sh_name': self.dummy_section_name_offset,
                'sh_type': ENUM_SH_TYPE_RISCV['SHT_NULL'],
                'sh_flags': 0,
                'sh_addr': 0,
                'sh_offset': 0,
                'sh_size': 0,
                'sh_link': 0,
                'sh_info': 0,
                'sh_addralign': 0,
                'sh_entsize': 0
            })))
            # .shstrtab
            f.write(self.structs.Elf_Shdr.build(DictToClass({
                'sh_name': self.shstrtab_name_offset,
                'sh_type': ENUM_SH_TYPE_RISCV['SHT_STRTAB'],
                'sh_flags': 0,
                'sh_addr': 0,
                'sh_offset': 64 + 56 * len(self.segments),
                'sh_size': len(self.shstrtab_data),
                'sh_link': 0,
                'sh_info': 0,
                'sh_addralign': 1,
                'sh_entsize': 0
            })))
