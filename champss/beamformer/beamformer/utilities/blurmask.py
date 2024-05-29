import numpy as np
from numpy.lib import stride_tricks


def blur_3x3(array, mean=False):
    r"""
    Takes the mean of/sums each element with its neighbours (within a 3x3 box centered
    on the element), \ effectively blurring the input array.

    If summing, the edge and corner sums are scaled up to match the rest.
    Uses strides to form the boxes - may(untested!) be speedier and lower on memory
    :param array: 2D integer numpy array
    :type array: ndarray
    :param mean: Take mean of elements in window, rather than sum
    :type mean: bool
    :return: 2D numpy array of the mean/sum of an element with its neighbours
    """
    # initialize output array to correct shape
    out_array = np.zeros(array.shape, dtype=float)

    strides = (array.strides[0], array.strides[1], array.strides[0], array.strides[1])

    # compute scaling factors
    window = (
        3  # more than a 3x3 box . . . maybe can't get away with without an enumerate
    )
    length = (window - 1) / 2
    total_number = window**2
    edge_number = (length + 1) * window
    corner_number = length**2 + 2 * length + 1

    if mean:
        inner_scale = 1 / total_number
        edge_scale = 1 / edge_number
        corner_scale = 1 / corner_number
    else:
        inner_scale = 1
        edge_scale = total_number / edge_number
        corner_scale = total_number / corner_number

    # make 3x3 windows for every inner element - aka no edges, no corners
    inner_shape = (
        array.shape[0] - window + 1,
        array.shape[1] - window + 1,
        window,
        window,
    )
    # as_strided creates a view not a copy which is awesome!
    inner_minesweep = stride_tricks.as_strided(
        array, shape=inner_shape, strides=strides
    )
    # sum windows and put into out_array
    out_array[1:-1, 1:-1] = inner_scale * inner_minesweep.sum(axis=(-1, -2))

    # make and fill edges
    # make a 2x3 window view for every element except last row and corners
    horizontal_edge_shape = (
        array.shape[0] - 1,
        array.shape[1] - window + 1,
        window - 1,
        window,
    )
    horizontal_edges = stride_tricks.as_strided(
        array, shape=horizontal_edge_shape, strides=strides
    )
    # sum appropriate 2x3 window, scale, and apply
    out_array[0, 1:-1] = edge_scale * horizontal_edges[0, :, :, :].sum(axis=(-1, -2))
    out_array[-1, 1:-1] = edge_scale * horizontal_edges[-1, :, :, :].sum(axis=(-1, -2))

    # make a 3x2 window view for every element except last column and corners
    vertical_edge_shape = (
        array.shape[0] - window + 1,
        array.shape[1] - 1,
        window,
        window - 1,
    )
    vertical_edges = stride_tricks.as_strided(
        array, shape=vertical_edge_shape, strides=strides
    )
    # sum appropriate 3x2 window, scale, and apply
    out_array[1:-1, 0] = edge_scale * vertical_edges[:, 0, :, :].sum(axis=(-1, -2))
    out_array[1:-1, -1] = edge_scale * vertical_edges[:, -1, :, :].sum(axis=(-1, -2))

    # fill corners
    # make a 2x2 window view for every element except last row and column
    corner_shape = (array.shape[0] - 1, array.shape[1] - 1, window - 1, window - 1)
    corners = stride_tricks.as_strided(array, shape=corner_shape, strides=strides)
    # sum appropriate 2x2 window, scale, and apply
    out_array[0, 0] = corner_scale * corners[0, 0, :, :].sum(axis=(-1, -2))
    out_array[0, -1] = corner_scale * corners[0, -1, :, :].sum(axis=(-1, -2))
    out_array[-1, 0] = corner_scale * corners[-1, 0, :, :].sum(axis=(-1, -2))
    out_array[-1, -1] = corner_scale * corners[-1, -1, :, :].sum(axis=(-1, -2))

    return out_array


def blur(array, window=3, mean=True, scale_edges=True, dtype=float):
    """
    Takes the mean of (or sums if set mean=False) each element with its neighbours.

    Effectively blurs the input array. Steps through array with ndenumerate and so may
    be slow for large arrays.
    :param array: 2D integer numpy array
    :type array: ndarray
    :param window: Neighbours are defined by a window x window box centered on the
        element, default is 3
    :type window: int
    :param dtype: Data type for output array. Default is float
    :type dtype: Type
    :param scale_edges: Edges and corners are summed over fewer elements. Set True
        (=default) to scale them up.
    :type scale_edges: bool
    :param mean: Compute the mean of the window, default is True. If mean=False computes
        a sum
    :type mean: bool
    :raises TypeError: window must be an odd integer
    :return: 2D numpy array of the sums/means
    """
    # check window
    if type(window) != int or not (window % 2):
        raise TypeError("window must be an odd integer")

    dist = int((window - 1) / 2)

    # initialize output array to correct shape
    out_array = np.zeros(array.shape, dtype=dtype)

    maxi, maxj = array.shape
    for (i, j), item in np.ndenumerate(array):
        box = array[
            max(i - dist, 0) : min(i + dist + 1, maxi),
            max(j - dist, 0) : min(j + dist + 1, maxj),
        ]
        if mean:
            out_array[i, j] = box.sum() / box.size
        elif scale_edges:
            out_array[i, j] = box.sum() * window**2 / box.size
        else:
            out_array[i, j] = box.sum()

    return out_array
