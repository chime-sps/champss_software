#include "huffman.hpp"

namespace spshuff {
#if 0
}  // emacs pacifier
#endif

// quantize + huffman encode with automatic reallocation
py::array_t<uint32_t> encode(const py::array_t<DTYPE_F> &ar_in)
{
	assert(ar_in.ndim() == 1);

	const ssize_t nquant = ar_in.size();

	DTYPE_F* dptr = (DTYPE_F*) ar_in.data(0);

	std::vector<uint8_t> out0(nquant);
	quantize_naive5(dptr, get_ptr<uint8_t>(out0), nquant);

	// Compute an upper bound on bit size of encoded huffman data
	const ssize_t max_dst_size = encode_bound(nquant);

	// Allocate a uint32_t "safe" sized array. While this is a bit-level coding, I
	// find that chunking the huffman stream into a larger data type e.g. 32 or 64 bits
	// is helpful for efficiency.

	// number of uint32_t in the compressed data (fairly arbitrary choice of dtype)
	const ssize_t len_compressed = (max_dst_size/sizeof(uint32_t)) + 1;
	std::vector<uint32_t> compressed_data(len_compressed);

	// we need to set these values here, but we don't worry about their post-call state in this function
	ssize_t i0 = 0;
	ssize_t bit0 = 0;
	uint32_t tmp = 0;
	const ssize_t compressed_data_len = huff_encode_kernel(get_ptr<uint8_t>(out0), 
											get_ptr<uint32_t>(compressed_data), nquant, i0, bit0, tmp);
	const ssize_t compressed_data_size = compressed_data_len * sizeof(uint32_t);

	auto ret = py::array_t<uint32_t>(
				{compressed_data_len,},
				{sizeof(uint32_t),});

	std::memcpy(ret.mutable_data(0), get_ptr<uint32_t>(compressed_data), compressed_data_size);

	return ret;
}


py::array_t<DTYPE_F> decode(const py::array_t<uint32_t> &dat_encoded, const ssize_t nsamp)
{
	uint32_t* datptr = (uint32_t*) dat_encoded.data(0);
	const ssize_t nframe = dat_encoded.size();
	const ssize_t dat_size = nframe * sizeof(uint32_t);
	// std::vector<uint8_t> decode_buf(nsamp);
	std::vector<uint8_t> decode_buf(decode_bound(dat_size));

	const ssize_t recovered_samples = huff_decode_kernel(datptr, get_ptr<uint8_t>(decode_buf),
														nframe, nsamp);
	assert(recovered_samples == nsamp);

	std::vector<DTYPE_F> retvec(nsamp);

	dequantize(get_ptr<uint8_t>(decode_buf), get_ptr<DTYPE_F>(retvec), nsamp);

	auto ret = py::array_t<DTYPE_F>(
				{recovered_samples,},
				{sizeof(DTYPE_F),});

	std::memcpy(ret.mutable_data(0), get_ptr<DTYPE_F>(retvec), nsamp * sizeof(DTYPE_F));

	return ret;
}


PYBIND11_MODULE(huffman, m) {
    m.def("encode", &encode, 
    							"A function to quantize + huffman compress mean-zero gaussian, unit-variance data");
    m.def("decode", &decode, 
    							"A function to decompress + dequantize previously encoded data");
}
} // namespace spshuff