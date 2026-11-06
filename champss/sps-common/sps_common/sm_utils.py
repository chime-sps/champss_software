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
