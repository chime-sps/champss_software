#include "spshuff_defs.hpp"

// const ssize_t huff_decode_bound(const ssize_t size_in);
// const ssize_t huff_encode_bound(const ssize_t samples_in);
// const ssize_t huff_encode_kernel(const char* in, uint32_t* out, const ssize_t n_in);
// const ssize_t huff_decode_kernel(const uint32_t* in, uint8_t* out, const uint8_t* verify, const ssize_t n_in);

const ssize_t decode_bound(const ssize_t nsamp);

// quantizes, the huffman compresses an array of floats
// returns an array of 32-bit integers containing the binary encoded huffman data
py::array_t<uint32_t> encode(const py::array_t<DTYPE_F> &data);

// converts an array of huffman encoded quantized samples
// returns an array of floats representing the dequanitzed, decompressed data
py::array_t<DTYPE_F> decode(const py::array_t<uint32_t> &dat_encoded, const ssize_t nsamp);