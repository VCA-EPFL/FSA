# This file should NOT be rewritten with python's mmap wrapper
# Python's mmap wrapper would do a double write to the device even with a single write call
# The second write can break FSA's internal states

import os
import sys
import ctypes
import ctypes.util

# --- C Library and Constants Setup ---

# Constants from fcntl.h and mman.h, needed for C function calls
O_RDWR = os.O_RDWR
O_SYNC = getattr(os, 'O_SYNC', 0) # O_SYNC might not be available on all OSes
PROT_READ = 0x1
PROT_WRITE = 0x2
MAP_SHARED = 0x01

# Find and load the standard C library
try:
    libc = ctypes.CDLL(ctypes.util.find_library('c'))
except (FileNotFoundError, OSError):
    try:
        # Fallback for some systems where find_library might fail
        libc = ctypes.CDLL('libc.so.6')
    except OSError as e:
        print(f"Error: Could not find or load the C standard library (libc). {e}", file=sys.stderr)
        sys.exit(1)

# Define C function prototypes with ctypes for type safety and correctness
# int open(const char *pathname, int flags);
libc.open.argtypes = [ctypes.c_char_p, ctypes.c_int]
libc.open.restype = ctypes.c_int

# int close(int fd);
libc.close.argtypes = [ctypes.c_int]
libc.close.restype = ctypes.c_int

# void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p

# int munmap(void *addr, size_t length);
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munmap.restype = ctypes.c_int

# --- Endianness Helper Functions ---

if sys.byteorder == 'little':
    ltohs = lambda x: x
    ltohl = lambda x: x
    htols = lambda x: x
    htoll = lambda x: x
else:
    def swap16(x):
        return ((x << 8) & 0xFF00) | ((x >> 8) & 0x00FF)
    def swap32(x):
        return ((x << 24) & 0xFF000000) | \
               ((x <<  8) & 0x00FF0000) | \
               ((x >>  8) & 0x0000FF00) | \
               ((x >> 24) & 0x000000FF)
    ltohs = swap16
    ltohl = swap32
    htols = swap16
    htoll = swap32

# --- MMIO Library Class ---

class MMIO:
    """
    A class to perform memory-mapped I/O on a device using raw C library calls.
    This provides low-level access similar to the C implementation.
    """
    def __init__(self, device: str = '/dev/xdma0_user'):
        """
        Initializes the MMIO object and opens the target device.
        Args:
            device: The path to the character device (e.g., '/dev/mem').
        Raises:
            IOError: If the device cannot be opened.
        """
        self.device = device
        self.fd = libc.open(device.encode('utf-8'), O_RDWR | O_SYNC)
        if self.fd == -1:
            err = ctypes.get_errno()
            raise IOError(f"Failed to open device {device}: {os.strerror(err)}")

        try:
            self.page_size = os.sysconf("SC_PAGESIZE")
        except (ValueError, AttributeError):
            # Fallback if sysconf is not available or SC_PAGESIZE is not defined
            self.page_size = 4096
        
        self.access_width_map = {'b': 1, 'h': 2, 'w': 4}
        self.ctype_map = {'b': ctypes.c_uint8, 'h': ctypes.c_uint16, 'w': ctypes.c_uint32}

    def _access(self, addr: int, access_type: str = 'w', write_val: int = None):
        """Internal method to handle both read and write operations."""
        access_type = access_type.lower()
        if access_type not in self.access_width_map:
            raise ValueError(f"Invalid access type: '{access_type}'. Use 'b', 'h', or 'w'.")

        width = self.access_width_map[access_type]
        ctype = self.ctype_map[access_type]
        
        # Align address to page size
        offset = addr & (self.page_size - 1)
        base_addr = addr & (~(self.page_size - 1))
        map_size = offset + width

        # Map memory region
        map_ptr = libc.mmap(None, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, self.fd, base_addr)
        if map_ptr == -1:
            err = ctypes.get_errno()
            raise IOError(f"mmap failed for address 0x{base_addr:x}: {os.strerror(err)}")

        try:
            # Calculate the final pointer to the target data
            data_ptr = ctypes.cast(map_ptr + offset, ctypes.POINTER(ctype))
            
            if write_val is not None:
                # Write operation
                val_to_write = write_val
                if access_type == 'h': val_to_write = htols(write_val)
                if access_type == 'w': val_to_write = htoll(write_val)
                data_ptr.contents.value = val_to_write
                return None
            else:
                # Read operation
                raw_val = data_ptr.contents.value
                read_val = raw_val
                if access_type == 'h': read_val = ltohs(raw_val)
                if access_type == 'w': read_val = ltohl(raw_val)
                return read_val
        finally:
            # Ensure memory is always unmapped
            if libc.munmap(map_ptr, map_size) == -1:
                err = ctypes.get_errno()
                # We raise a warning here instead of an error because the primary I/O might have succeeded
                print(f"Warning: munmap failed for address 0x{map_ptr:x}: {os.strerror(err)}", file=sys.stderr)

    def dev_mmio_read(self, addr: int, access_type: str = 'w') -> int:
        """
        Reads a value from a memory-mapped address.
        Args:
            addr: The memory address to read from.
            access_type: The width of the read: 'b' (byte), 'h' (half-word), 'w' (word).
        Returns:
            The integer value read from the address.
        """
        return self._access(addr, access_type)

    def dev_mmio_write(self, addr: int, value: int, access_type: str = 'w'):
        """
        Writes a value to a memory-mapped address.
        Args:
            addr: The memory address to write to.
            value: The integer value to write.
            access_type: The width of the write: 'b' (byte), 'h' (half-word), 'w' (word).
        """
        self._access(addr, access_type, write_val=value)
    
    def dev_queue_mmio_write(self, write_addr: int, data_array: list[int]):
        """
        Writes a list of 32-bit integers to the same address as a queue-style write.
        Each write is 4 bytes, like queueing commands or data to a FIFO.
        Args:
            write_addr: The base address to write to.
            data_array: A list of 32-bit integers to write.
        """
        for data in data_array:
            self.dev_mmio_write(write_addr, data, 'w')

    def close(self):
        """Closes the device file descriptor."""
        if self.fd != -1:
            libc.close(self.fd)
            self.fd = -1
