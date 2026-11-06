from multiprocessing import shared_memory
import numpy as np


def share_array(arr):
    """
    Move numpy array to shared memory.
    Note that, this may lead to the array being present twice in memory. To prevent this
    overwrite the old reference with the first output value.

    arr, shm, shm_dict = share_array(arr)

    Parameters
    ----------
    arr : np.ndarray
        The array that should be moved to shared memory

    Returns
    -------
    array: np.ndarray
        The new numpy array to access the array

    shm: shared_memory.SharedMemory
        The SharedMemory object of the new array

    shm_dict: dict
        The dict containing values needed for easy reconstruction of the array
    """
    buffer_size = arr.nbytes
    shm = shared_memory.SharedMemory(create=True, size=buffer_size)

    array = np.ndarray(
        arr.shape,
        dtype=arr.dtype,
        buffer=shm.buf,
    )
    array[:] = arr
    shm_dict = {"name": shm.name, "shape": array.shape, "dtype": array.dtype}
    return array, shm, shm_dict


def empty_shared_array(shape, dtype):
    """
    Allocate a zero-filled numpy array directly in shared memory.

    Parameters
    ----------
    shape : tuple
        Shape of the array.
    dtype : np.dtype
        Data type of the array.

    Returns
    -------
    array: np.ndarray
        The new array, backed by shm.

    shm: shared_memory.SharedMemory
        The block holding the array; the caller is responsible for unlinking it.
    """
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return array, shm


def recreate_shared_array(shm_dict):
    """
    Recreate a shared memory array from a dict.

    Parameters
    ----------
    shm_dict: dict
        The dict as created by share_array.

    Returns
    -------
    array: np.ndarray
        The reconstructed numpy array

    shm: shared_memory.SharedMemory
        The SharedMemory object of the reconstructed array
    """
    shm = shared_memory.SharedMemory(name=shm_dict["name"])
    array = np.ndarray(shm_dict["shape"], dtype=shm_dict["dtype"], buffer=shm.buf)
    return array, shm


def unlink_shared(shm):
    shm.close()
    shm.unlink()


# Shared memory attached by pool workers through attach_shared_memory, kept until the
# worker exits
_ATTACHED = {}


def attach_shared_memory(name):
    """
    Attach a shared memory block by name; returns it and whether the caller should close it.

    Pool workers run thousands of small tasks on the same blocks, and closing the mapping
    after each one costs more than many of the tasks; in a worker the block is attached
    once and kept until the worker exits. In the main process it is attached afresh, so
    nothing there outlives the block's unlinking.
    """
    import multiprocessing

    if multiprocessing.parent_process() is None:
        return shared_memory.SharedMemory(name=name), True
    shm = _ATTACHED.get(name)
    if shm is None:
        shm = _ATTACHED[name] = shared_memory.SharedMemory(name=name)
    return shm, False


def release_shared_memory(shm, close):
    """Close a block from attach_shared_memory if it said to."""
    if close:
        shm.close()


class RowPrefixReleaser:
    """
    Return to the system the memory of a 2D shared-memory array, column range by column
    range, as a streaming consumer finishes with it.

    release_until(col) frees the whole pages holding columns [previous col, col) of every
    row, with madvise(MADV_REMOVE), which punches holes in the shared memory (tmpfs) file
    so the memory is really freed; the freed values then read as zeros, so call it only
    for columns nothing will read again. Pages shared with a column not yet released (or
    with the neighbouring row) are kept. If the array is not backed by shared memory the
    call fails, and the releaser simply stops releasing; the data is never touched.
    """

    def __init__(self, array):
        import ctypes
        import mmap

        self.array = array
        self.enabled = (
            array.ndim == 2 and array.flags.c_contiguous and hasattr(mmap, "MADV_REMOVE")
        )
        self.done = 0
        if self.enabled:
            self._advice = mmap.MADV_REMOVE
            self._page = mmap.PAGESIZE
            libc = ctypes.CDLL(None, use_errno=True)
            self._madvise = libc.madvise
            self._madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            self._madvise.restype = ctypes.c_int

    def release_until(self, col):
        """Release columns [previous col, col) of every row."""
        col = min(col, self.array.shape[1])
        if not self.enabled or col <= self.done:
            return
        page = self._page
        item = self.array.itemsize
        base = self.array.ctypes.data
        row_bytes = self.array.strides[0]
        for r in range(self.array.shape[0]):
            row = base + r * row_bytes
            # The page holding the previous boundary is now fully consumed too, but not
            # one that also holds the end of the row before
            start = max((row + self.done * item) // page * page, -(-row // page) * page)
            stop = (row + col * item) // page * page
            if stop > start and self._madvise(start, stop - start, self._advice) != 0:
                self.enabled = False
                return
        self.done = col


def release_rows(array, start, stop):
    """
    Return to the system the memory of rows start to stop of a C-contiguous array in
    shared memory, once nothing will read them again.

    Frees the whole pages inside those rows with madvise(MADV_REMOVE) (see
    RowPrefixReleaser); pages shared with rows outside the range are kept, so ranges
    released by different processes in any order never touch each other's rows. Freed
    values read as zeros. Returns False, changing nothing, if the memory is not shared.
    """
    import ctypes
    import mmap

    if not (array.flags.c_contiguous and hasattr(mmap, "MADV_REMOVE")) or stop <= start:
        return False
    page = mmap.PAGESIZE
    row_bytes = array.strides[0]
    first = array.ctypes.data + start * row_bytes
    last = array.ctypes.data + stop * row_bytes
    begin = -(-first // page) * page
    end = last // page * page
    if end <= begin:
        return True
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.madvise.restype = ctypes.c_int
    return libc.madvise(begin, end - begin, mmap.MADV_REMOVE) == 0
