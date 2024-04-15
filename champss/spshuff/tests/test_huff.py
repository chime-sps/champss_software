import spshuff
from spshuff.huff_utils import py_quantize_dequantize, feq
import numpy as np
import time


def test_encode_decode(nsamp=1 * 1024 * 1024):
	ar_in = np.random.normal(size=nsamp).astype(np.float32)

	t0 = time.time()
	py_decoded = py_quantize_dequantize(ar_in)
	dt_py = time.time() - t0
	print("Python reference rate (Msamples/s): {}".format((float(nsamp)*1e-6)/dt_py))

	t1 = time.time()
	encoded = spshuff.encode(ar_in)
	dt_enc = time.time() - t1

	t2 = time.time()
	decoded = spshuff.decode(encoded, nsamp)
	dt_dec = time.time() - t2

	# time.sleep(120.)

	print("Encoding rate (Msamples/s): {}".format((float(nsamp)*1e-6)/dt_enc))
	print("Decoding rate (Msamples/s): {}".format((float(nsamp)*1e-6)/dt_dec))

	assert feq(decoded, py_decoded)


if __name__ == "__main__":
	test_encode_decode()
