from .tensor import STile, ATile, MTile, T
from .dtype import *
from typing import Type

class MemoryAllocator:
    def __init__(self, addr_base: int, size: int, alignment: int):
        self.addr_base = addr_base
        self.size = size
        self.alignment = alignment
        self.free_blocks = [(addr_base, size)]  # List of tuples (address, size)
        self.allocated_blocks = {}

    def _align_up(self, addr: int) -> int:
        """ Aligns the address to the nearest multiple of alignment. """
        if addr % self.alignment == 0:
            return addr
        return ((addr // self.alignment) + 1) * self.alignment

    def allocate(self, size: int) -> int:
        """ Allocates a block of memory with the specified size. """
        for index, (start, block_size) in enumerate(self.free_blocks):
            aligned_start = self._align_up(start)
            padding = aligned_start - start
            total_size = size + padding

            if total_size <= block_size:
                # Allocate the block
                self.free_blocks.pop(index)
                allocated_addr = aligned_start
                self.allocated_blocks[allocated_addr] = size

                # Split remaining free memory
                remaining_size = block_size - total_size
                if remaining_size > 0:
                    self.free_blocks.insert(index, (aligned_start + size, remaining_size))

                # print(f"Allocated {size} bytes at address {hex(allocated_addr)}")
                return allocated_addr

        raise RuntimeError(f"Allocation failed: not enough memory. Requested {size} bytes, but only {self.size - sum(self.allocated_blocks.values())} bytes available.")

    def deallocate(self, addr: int):
        """ Deallocates a previously allocated block of memory. """
        if addr in self.allocated_blocks:
            size = self.allocated_blocks.pop(addr)
            self.free_blocks.append((addr, size))
            self.free_blocks.sort()
            self._merge_free_blocks()
            # print(f"Deallocated {size} bytes from address {hex(addr)}")
        else:
            raise RuntimeError(f"Invalid deallocation attempt at address {hex(addr)}")

    def _merge_free_blocks(self):
        """ Merges contiguous free blocks to prevent fragmentation. """
        merged_blocks = []
        for block in sorted(self.free_blocks):
            if merged_blocks and merged_blocks[-1][0] + merged_blocks[-1][1] == block[0]:
                last_addr, last_size = merged_blocks.pop()
                merged_blocks.append((last_addr, last_size + block[1]))
            else:
                merged_blocks.append(block)
        self.free_blocks = merged_blocks

    def dump_memory(self):
        """ Prints the current memory layout. """
        print("\nAllocated Blocks:")
        for addr, size in self.allocated_blocks.items():
            print(f" - Address: {hex(addr)}, Size: {size} bytes")

        print("\nFree Blocks:")
        for addr, size in self.free_blocks:
            print(f" - Address: {hex(addr)}, Size: {size} bytes")
        print("\n")

class CompoundMemoryManger:

    def __init__(
        self,
        mem_base: int, mem_size: int, mem_align: int,
        spad_base: int, spad_size: int, spad_align: int, spad_dtype: dtype,
        acc_base: int, acc_size: int, acc_align: int, acc_dtype: dtype
    ):
        self.mem = MemoryAllocator(mem_base, mem_size, mem_align)
        self.spad = MemoryAllocator(spad_base, spad_size, spad_align)
        self.acc = MemoryAllocator(acc_base, acc_size, acc_align)
        self.spad_dtype = spad_dtype
        self.acc_dtype = acc_dtype
        self.mem_tensor_list: list[MTile] = []

    def alloc_spad(self, shape: int | tuple[int, ...]) -> STile:
        return self.__allocate(self.spad, shape, self.spad_dtype, STile)

    def alloc_accumulator(self, shape: int | tuple[int, ...]) -> ATile:
        return self.__allocate(self.acc, shape, self.acc_dtype, ATile)

    def alloc_mem(self, shape: int | tuple[int, ...], dtype: dtype) -> MTile:
        tile = self.__allocate(self.mem, shape, dtype, MTile)
        self.mem_tensor_list.append(tile)
        return tile

    @staticmethod
    def __allocate(allocator: MemoryAllocator, shape: int | tuple[int, ...], dtype: dtype, ret: Type[T]) -> T:
        data_ptr = allocator.allocate(CompoundMemoryManger.__shape_to_size(shape) * dtype.itemsize)
        if isinstance(shape, int):
            shape = tuple(shape)
        return ret(shape, dtype, data_ptr)

    @staticmethod
    def __shape_to_size(shape: int | tuple[int, ...]) -> int:
        if isinstance(shape, int):
            if shape <= 0:
                raise ValueError(f"Shape dimension must be positive, got {shape}")
            return shape
        elif isinstance(shape, tuple):
            if not all(isinstance(dim, int) and dim > 0 for dim in shape):
                raise ValueError(f"All shape dimensions must be positive integers, got {shape}")
            size = 1
            for dim in shape:
                size *= dim
            return size
        else:
            raise TypeError(f"Shape must be an int or tuple of ints, got {type(shape)}")
