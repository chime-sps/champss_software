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
