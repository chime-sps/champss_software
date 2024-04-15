#include "huffman.hpp"
#include <vector>
#include <memory>
#include <iostream>
#include <cassert>


// the edges of the N=5 bin encoding scheme (determined in quantization.py)
// -Inf and +Inf are implicit edges for the first and last bins
const DTYPE_F edges5[4] = {-1.24435754, -0.38228386, 0.38228386, 1.24435754};

// We can constrain this by symmetry, but we write the full form here
// for efficiency in the dequantization stage
const DTYPE_F dequant5[5] = {-1.8739534, -0.8309986, 0., 0.8309986, 1.8739534};

const uint8_t codes[5] = {7, 2, 0, 1, 3}; // value of each huffman code
const uint32_t codes32[5] = {7, 2, 0, 1, 3}; // value of each code (32 bit dtype for internal use)
const ssize_t lens[5] = {3, 2, 2, 2, 3}; // bit length of each code
const uint32_t check[5] = {7, 3, 3, 3, 7}; // used to match during decode
const ssize_t maxlen = 3; // maximum length of code, used for size bound

// hard-coded, used to estimate encoding efficiency
// we can compute the entropy of an encoding scheme (as a function of N) in quanztiaztion.py
const DTYPE_F entropy5 = 2.202916387949746;
const DTYPE_F bitsize = 2.321928094887362;


template <typename T>
T* get_ptr(std::vector<T> &vect)
{
	return &(vect[0]);
}


const int comparef(const DTYPE_F a, const DTYPE_F b)
{
	return (a < b) ? 0: 1;
}


template <typename T>
T* malloc_buf(const ssize_t n)
{
	return (T*) malloc(n * sizeof(T));
}


void quantize_naive5(const DTYPE_F* in, uint8_t* out,
					 const ssize_t n)
{
	const DTYPE_F e0 = edges5[0];
	const DTYPE_F e1 = edges5[1];
	const DTYPE_F e2 = edges5[2];
	const DTYPE_F e3 = edges5[3];
	int val;

	for(ssize_t i = 0; i < n; i++){
		const DTYPE_F v = in[i];
		val = 0;
		val += comparef(v, e0);
		val += comparef(v, e1);
		val += comparef(v, e2);
		val += comparef(v, e3);
		out[i] = (uint8_t) val;
	}
}


void dequantize(const uint8_t* levels, DTYPE_F* out, const ssize_t nsamp)
{
	for(ssize_t i = 0; i < nsamp; i++){
		out[i] = dequant5[levels[i]];
	}
}


// assumes a uint32_t "chunk" structure
// provides an upper bound on the number of distinct samples
// coincidentally is the same as an upper bound on the size of the
// resulting unpacked array.
const ssize_t decode_bound(const ssize_t size_in)
{
	return size_in * 4;
}


const ssize_t encode_bound(const ssize_t samples_in)
{
	ssize_t r = 0;
	ssize_t base = (samples_in * 3) / 8;

	if(((3 * samples_in) % 8) > 0){
		r = 1;
	}

	return base + r;
}


const ssize_t huff_encode_kernel(const uint8_t* in, uint32_t* out, const ssize_t n_in)
{
	ssize_t iout = 0; // tracks out index
	ssize_t bit_pos = 0; // tracks bit position within
	uint32_t hval;
	ssize_t this_size, ival;
	uint32_t shuttle;
	uint32_t tmp = 0;
	ssize_t r;
	for(ssize_t i = 0; i < n_in; i++){
		shuttle = 0;
		ival = (ssize_t) in[i];
		hval = (uint32_t) codes[ival];
		this_size = (ssize_t) lens[ival];
		shuttle = hval << bit_pos;
		tmp += shuttle;

		bit_pos += this_size;

		if(bit_pos >= 32){
			r = bit_pos - 32;
			out[iout] = tmp;

			tmp = 0;
			if(r > 0){
				tmp += hval >> (this_size - r);
			}
			iout++;
			bit_pos = r;
		}
	}

	if(bit_pos > 0){
		out[iout] = tmp;
	}

	return (iout + 1) * 4;
}


const ssize_t huff_decode_kernel(const uint32_t* in, uint8_t* out,
								 const ssize_t n_in, const ssize_t nsamp)
{
	assert(n_in > 0);

	ssize_t ichunk = 0;
	ssize_t iout = 0;
	ssize_t bpos = 0;
	ssize_t this_len;
	ssize_t imatch;

	// this is lazy - to not deal with bit overrun I use a 64 bit int
	uint64_t v = ((uint64_t) in[0]) + (((uint64_t) in[1]) << 32);
	uint32_t tmp2, tmp3;

	// fix ichunk == n_in - 1 handling
	// while(ichunk < n_in - 1){
	while(iout < nsamp){
		assert(ichunk < n_in);
		// extract 2 and 3 bit codes starting at bpos
		tmp2 = (uint64_t) ((v >> bpos) & 3);
		tmp3 = (uint64_t) ((v >> bpos) & 7);

		if(tmp2 == codes32[2]){
			imatch = 2;
		}
		else if(tmp2 == codes32[1]){
			imatch = 1;
		}
		else if(tmp2 == codes32[3]){
			imatch = 3;
		}
		else if(tmp3 == codes32[0]){
			imatch = 0;
		}
		else{
			imatch = 4;
		}

		this_len = lens[imatch];
		out[iout] = (uint8_t) imatch;
		// std::cout << (uint32_t) out[iout] << " | " << (uint32_t) verify[iout] << " | " << bpos << " | " << tmp2 << " | " << tmp3 << std::endl;
		bpos += this_len;
		iout++;

		if(bpos >= 32){
			ichunk++;
			bpos = bpos - 32;
			v = ((uint64_t) in[ichunk]) + (((uint64_t) in[ichunk + 1]) << 32);
		}
	}

	return iout;
}

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
	const ssize_t len_compressed = (max_dst_size/sizeof(uint32_t)) +1;
	std::vector<uint32_t> compressed_data(len_compressed);

	const ssize_t compressed_data_size = huff_encode_kernel(get_ptr<uint8_t>(out0), 
											get_ptr<uint32_t>(compressed_data), nquant);
	const ssize_t compressed_data_len = compressed_data_size / sizeof(uint32_t);

	auto ret = py::array_t<uint32_t>(
				{compressed_data_len,},
				{sizeof(uint32_t),});

	std::memcpy(ret.mutable_data(0), get_ptr<uint32_t>(compressed_data), compressed_data_size);

	return ret;
}


py::array_t<DTYPE_F> decode(const py::array_t<uint32_t> &dat_encoded, const ssize_t nsamp)
{
	uint32_t* datptr = (uint32_t*) dat_encoded.data(0);
	const ssize_t dat_size = dat_encoded.size() * sizeof(uint32_t);
	const ssize_t nframe = dat_size / sizeof(uint32_t);
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


// const ? huff_decode(source buffer, const ssize_t ndecode)
// {
// 	uint8_t* decode_buf = (uint8_t*) malloc(huff_decompress_bound(compressed_data_size));

// 	// NOTE: here I structure the program to actually determine the number of recovered samples from the
// 	// compressed stream but I really don't think this is a good idea. I think the better thing to do would be
// 	// to prepend the number of samples in a fixed-length datatype at the beginning of the huffman stream (e.g. 
// 	// if you're writing it to disk) and just recover the data using this info. Right now there's a degeneracy
// 	// between samples that correspond to zero and trailing zeroes in the last chunk. This confuses the decoder
// 	// and no amount of logic on the raw (encoded) sample chunk stream can break this degeneracy.

// 	auto start1 = std::chrono::high_resolution_clock::now();
// 	const ssize_t recovered_samples = huff_decode(compressed_data, decode_buf, (uint8_t*) out0, n_in);

// }