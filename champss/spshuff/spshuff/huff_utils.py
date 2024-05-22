import numpy as np
from spshuff import decode, encode

summary_dtype = np.dtype(np.float32)
data_dtype = np.dtype(np.float32)  # must match DTYPE_F in spshuff_defs.hpp!


# these must agree with the c++ definitions; should move to shared defs
edges5 = np.array([-1.24435754, -0.38228386, 0.38228386, 1.24435754, np.inf])
dequant5 = np.array([-1.8739534, -0.8309986, 0.0, 0.8309986, 1.8739534])


def feq(a, b, eps=1e-6):
    return np.all(np.abs(a - b) <= np.maximum(np.abs(a), np.abs(b)) * eps)


# here, the frac denotes the tolerable fraction of values that aren't
# epsilon-equal. Useful for unstable calculations.
def feq_frac(a, b, eps=1e-6, frac=0.0001):
    return np.sum(
        np.abs(a - b) > np.maximum(np.abs(a), np.abs(b)) * eps
    ) <= frac * np.prod(a.shape)


def get_level(val):
    for ilevel in range(5):
        if val < edges5[ilevel]:
            return ilevel
    return None


def quantize_dequantize(ar_in):
    assert len(ar_in.shape) == 1
    nsamples = len(ar_in)
    return decode(encode(ar_in), nsamples)


def py_quantize_dequantize(ar_in):
    n = len(ar_in)

    levels = np.empty((n,), dtype=int)
    for i in range(n):
        levels[i] = get_level(ar_in[i])
        assert levels[i] is not None

    return dequant5[levels]


def check_eof(f, nbyte_check=1):
    """Return True if the file has at least nbyte_check of data left, False
    otherwise.
    """
    nseek = len(f.read(nbyte_check))
    f.seek(-nseek, 1)
    return nseek == nbyte_check
