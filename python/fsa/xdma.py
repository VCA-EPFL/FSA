import os
import time
import numpy as np

class DeviceError(IOError):
    """Custom exception for device I/O errors."""
    pass

def dev_read(device_name: str, address: int, size: int) -> np.ndarray:
    """
    Reads data from a device into a NumPy array.
    This function corresponds to reading from a device like /dev/xdma0_c2h_0.

    Args:
        device_name (str): Path to the device file (e.g., "/dev/xdma0_c2h_0").
        address (int): The starting address (offset) on the device to read from.
                       If address is 0, reading starts from the current position
                       (or beginning if device was just opened).
        size (int): The number of bytes to read.

    Returns:
        np.ndarray: A NumPy array of dtype uint8 containing the data read.

    Raises:
        DeviceError: If any error occurs during device opening, seeking, or reading.
        TypeError: If arguments are of incorrect type.
        ValueError: If size is non-positive.
    """
    # Validate arguments
    if not isinstance(device_name, str):
        raise TypeError("device_name must be a string.")
    if not isinstance(address, int):
        raise TypeError("address must be an integer.")
    if not isinstance(size, int):
        raise TypeError("size must be an integer.")
    if address < 0:
        raise ValueError("address must be a non-negative integer.")
    if size <= 0:
        raise ValueError("size must be a positive integer.")

    dev_fd = -1  # Initialize file descriptor
    try:
        # Open the device. O_RDWR is used to match the C example's general access mode.
        # For a pure read, O_RDONLY could also be used.
        dev_fd = os.open(device_name, os.O_RDWR)

        # Seek to the specified address if address is non-zero.
        # The C code's `if (addr)` implies seeking only if addr is non-zero.
        # os.lseek returns the new cursor position.
        if address != 0:
            current_pos = os.lseek(dev_fd, address, os.SEEK_SET)
            if current_pos != address:
                raise DeviceError(
                    f"Failed to seek to address 0x{address:X} in {device_name}. "
                    f"Current position: 0x{current_pos:X}"
                )

        # Read data from the device
        buffer = os.read(dev_fd, size)
        if len(buffer) != size:
            # This case might mean EOF was reached before reading 'size' bytes.
            raise DeviceError(
                f"Failed to read requested {size} bytes from {device_name}. "
                f"Actually read {len(buffer)} bytes."
            )

        # Convert the raw bytes to a NumPy array of unsigned 8-bit integers
        numpy_array = np.frombuffer(buffer, dtype=np.uint8)
        return numpy_array

    except FileNotFoundError:
        raise DeviceError(f"Device not found: {device_name}")
    except PermissionError:
        raise DeviceError(f"Permission denied for device: {device_name}")
    except OSError as e:
        # Catch other OS-level errors
        raise DeviceError(f"OS error during read operation on {device_name}: {e.errno} - {os.strerror(e.errno)}")
    except DeviceError: # Re-raise custom DeviceError
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise DeviceError(f"An unexpected error occurred during read on {device_name}: {e}")
    finally:
        # Ensure the device file descriptor is closed if it was opened
        if dev_fd != -1:
            try:
                os.close(dev_fd)
            except OSError as e:
                # Log or handle close error if necessary, but don't let it mask an original error.
                # In a more complex app, this might go to a logger.
                print(f"Warning: Failed to close device {device_name} (fd: {dev_fd}): {e}")


def dev_write(device_name: str, address: int, data: np.ndarray) -> int:
    """
    Writes data from a NumPy array to a device.
    This function corresponds to writing to a device like /dev/xdma0_h2c_0.

    Args:
        device_name (str): Path to the device file (e.g., "/dev/xdma0_h2c_0").
        address (int): The starting address (offset) on the device to write to.
                       If address is 0, writing starts from the current position.
        data (np.ndarray): The NumPy array containing data to write.
                           The array's data will be converted to bytes.

    Returns:
        int: 0 on success. (Number of bytes written could also be returned,
             but C version returns 0 for success).

    Raises:
        DeviceError: If any error occurs during device opening, seeking, or writing.
        TypeError: If arguments are of incorrect type.
        ValueError: If data array is empty or address is negative.
    """
    # Validate arguments
    if not isinstance(device_name, str):
        raise TypeError("device_name must be a string.")
    if not isinstance(address, int):
        raise TypeError("address must be an integer.")
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a NumPy array.")
    if address < 0:
        raise ValueError("address must be a non-negative integer.")
    
    # It's valid to write an empty array if its byte representation is empty,
    # but os.write might behave differently. The C code implies size > 0.
    # For consistency, let's ensure some bytes are written if array isn't truly "empty" conceptually.
    # data.tobytes() will handle various dtypes.
    
    # Ensure data is C-contiguous for tobytes() to work reliably
    if not data.flags['C_CONTIGUOUS']:
        data_contiguous = np.ascontiguousarray(data)
    else:
        data_contiguous = data
        
    data_bytes = data_contiguous.tobytes()
    size_to_write = len(data_bytes)

    if data.size > 0 and size_to_write == 0:
        # This can happen if the dtype is unusual (e.g., object array of empty strings)
        raise ValueError("Input NumPy array converted to an empty byte string, though it has elements.")
    
    # If size_to_write is 0 (e.g. from an empty array), os.write might write 0 bytes and succeed.
    # The C code implies size > 0 for a transaction.
    # If an empty write is intended, this will proceed. If not, the user should pass non-empty data.

    dev_fd = -1 # Initialize file descriptor
    try:
        # Open the device
        dev_fd = os.open(device_name, os.O_RDWR)

        # Seek to the specified address if address is non-zero
        if address != 0:
            current_pos = os.lseek(dev_fd, address, os.SEEK_SET)
            if current_pos != address:
                raise DeviceError(
                    f"Failed to seek to address 0x{address:X} in {device_name}. "
                    f"Current position: 0x{current_pos:X}"
                )

        # Write data to the device
        bytes_written = os.write(dev_fd, data_bytes)
        if bytes_written != size_to_write:
            raise DeviceError(
                f"Failed to write {size_to_write} bytes to {device_name}. "
                f"Actually wrote {bytes_written} bytes."
            )

        return 0  # Success, following C convention

    except FileNotFoundError:
        raise DeviceError(f"Device not found: {device_name}")
    except PermissionError:
        raise DeviceError(f"Permission denied for device: {device_name}")
    except OSError as e:
        raise DeviceError(f"OS error during write operation on {device_name}: {e.errno} - {os.strerror(e.errno)}")
    except DeviceError: # Re-raise custom DeviceError
        raise
    except Exception as e:
        raise DeviceError(f"An unexpected error occurred during write on {device_name}: {e}")
    finally:
        if dev_fd != -1:
            try:
                os.close(dev_fd)
            except OSError as e:
                print(f"Warning: Failed to close device {device_name} (fd: {dev_fd}): {e}")