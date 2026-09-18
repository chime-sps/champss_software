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


def share_nested_arrays(obj):
    """
    Recursively move all NumPy arrays in a nested dict/list/tuple
    structure into shared memory. Created by ChatGPT.

    Returns
    -------
    shared_obj
        Same structure, but NumPy arrays replaced by shm dictionaries.
    shm_objects
        List of SharedMemory objects that must be kept alive by the
        creating process.
    """

    shm_objects = []

    def recurse(x):
        if isinstance(x, np.ndarray):
            _, shm, shm_dict = share_array(x)
            shm_objects.append(shm)
            return shm_dict

        elif isinstance(x, dict):
            return {key: recurse(value) for key, value in x.items()}

        elif isinstance(x, list):
            return [recurse(value) for value in x]

        elif isinstance(x, tuple):
            return tuple(recurse(value) for value in x)

        else:
            return x

    shared_obj = recurse(obj)

    return shared_obj, shm_objects


def recreate_nested_arrays(obj):
    """
    Recursively recreate NumPy arrays from shared-memory dictionaries. Created by ChatGPT.

    Returns
    -------
    array_obj
        Same structure with NumPy arrays reconstructed from shared memory.
    shm_objects
        SharedMemory handles that must remain alive while arrays are used.
    """

    shm_objects = []

    def recurse(x):
        if isinstance(x, dict) and "name" in x and "shape" in x and "dtype" in x:
            array, shm = recreate_shared_array(x)
            shm_objects.append(shm)
            return array

        elif isinstance(x, dict):
            return {key: recurse(value) for key, value in x.items()}

        elif isinstance(x, list):
            return [recurse(value) for value in x]

        elif isinstance(x, tuple):
            return tuple(recurse(value) for value in x)

        else:
            return x

    reconstructed = recurse(obj)

    return reconstructed, shm_objects
