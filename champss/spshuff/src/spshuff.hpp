#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sys/types.h>
#include <sys/time.h>
#include <stdint.h>
#include <immintrin.h> 
#include <memory>
#include <random>
#include <cstring>
#include <cassert>
#include <vector>

namespace spshuff {
#if 0
}  // emacs pacifier
#endif

typedef float DTYPE_F;

// exposre the n=5 parameters for utility

// the edges of the N=5 bin encoding scheme (determined in quantization.py)
// -Inf and +Inf are implicit edges for the first and last bins
const DTYPE_F edges5[4] = {-1.24435754, -0.38228386, 0.38228386, 1.24435754};

// We can constrain this by symmetry, but we write the full form here
// for efficiency in the dequantization stage
const DTYPE_F dequant5[5] = {-1.8739534, -0.8309986, 0., 0.8309986, 1.8739534};
// hard-coded, used to estimate encoding efficiency
// we can compute the entropy of an encoding scheme (as a function of N) in quanztiaztion.py
const DTYPE_F entropy5 = 2.202916387949746;
const DTYPE_F bitsize = 2.321928094887362;

const uint8_t codes[5] = {7, 2, 0, 1, 3}; // value of each huffman code
const uint32_t codes32[5] = {7, 2, 0, 1, 3}; // value of each code (32 bit dtype for internal use)
const ssize_t lens[5] = {3, 2, 2, 2, 3}; // bit length of each code
const ssize_t len_delta[5] = {3, -1, 0, 0, 1}; // delta length from zero for simd code
const uint32_t check[5] = {7, 3, 3, 3, 7}; // used to match during decode
const ssize_t maxlen = 3; // maximum length of code, used for size bound


template <typename T>
inline T* get_ptr(std::vector<T> &vect)
{
	return &(vect[0]);
};


inline const int comparef(const DTYPE_F a, const DTYPE_F b)
{
	return (a < b) ? 0: 1;
}


template <typename T>
inline T* malloc_buf(const ssize_t n)
{
	return (T*) malloc(n * sizeof(T));
}


// decent reference method
inline void quantize_naive5_reference(const DTYPE_F* in, uint8_t* out,
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


// requires 32-byte aligned input!
inline void quantize_naive5(const DTYPE_F* in, uint8_t* out,
					 const ssize_t n)
{
	assert((n % 8) == 0);
	__m256 me0 = _mm256_broadcast_ss(edges5);
	__m256 me1 = _mm256_broadcast_ss(edges5 + 1);
	__m256 me2 = _mm256_broadcast_ss(edges5 + 2);
	__m256 me3 = _mm256_broadcast_ss(edges5 + 3);

	// 32-byte aligned alloc for AXV2 instructions
	alignas(32) int32_t tmp_out[8];

	for(ssize_t i = 0; i < n; i+= 8){
		const __m256 m_in = _mm256_loadu_ps(in + i); // Consider aligned input
		__m256i mval = _mm256_set1_epi32(0);
		__m256 tmp;

		tmp = _mm256_cmp_ps(me0, m_in, 0x11);
		mval = _mm256_sub_epi32(mval, _mm256_castps_si256(tmp));

		tmp = _mm256_cmp_ps(me1, m_in, 0x11);
		mval = _mm256_sub_epi32(mval, _mm256_castps_si256(tmp));

		tmp = _mm256_cmp_ps(me2, m_in, 0x11);
		mval = _mm256_sub_epi32(mval, _mm256_castps_si256(tmp));

		tmp = _mm256_cmp_ps(me3, m_in, 0x11);
		mval = _mm256_sub_epi32(mval, _mm256_castps_si256(tmp));

		// _mm256_store_si256((__m256i*) (out + i), mval);
		_mm256_store_si256((__m256i*) tmp_out, mval);
		for(ssize_t j = 0; j < 8; j++){
			out[i + j] = (uint8_t) tmp_out[j];
		}
	}
}


// requires 32-byte aligned input!
inline void quantize_naive5_simd2(const DTYPE_F* in, uint8_t* out,
					 const ssize_t n)
{
	assert((n % 16) == 0);
	__m256 me0 = _mm256_broadcast_ss(edges5);
	__m256 me1 = _mm256_broadcast_ss(edges5 + 1);
	__m256 me2 = _mm256_broadcast_ss(edges5 + 2);
	__m256 me3 = _mm256_broadcast_ss(edges5 + 3);

	// 32-byte aligned alloc for AXV2 instructions
	alignas(32) int32_t tmp_out[8];

	// shift by four
	const __m256i cshift = _mm256_set1_epi32(16);

	for(ssize_t i = 0; i < n; i+= 16){
		const __m256 m_in0 = _mm256_load_ps(in + i);
		const __m256 m_in1 = _mm256_load_ps(in + i + 8);
		__m256i mval = _mm256_set1_epi32(0);

		mval = _mm256_sub_epi32(mval, _mm256_castps_si256(_mm256_cmp_ps(me0, m_in0, 0x11)));
		mval = _mm256_sub_epi32(mval, _mm256_mullo_epi32(_mm256_castps_si256(_mm256_cmp_ps(me0, m_in1, 0x11)), cshift));

		mval = _mm256_sub_epi32(mval, _mm256_castps_si256(_mm256_cmp_ps(me1, m_in0, 0x11)));
		mval = _mm256_sub_epi32(mval, _mm256_mullo_epi32(_mm256_castps_si256(_mm256_cmp_ps(me1, m_in1, 0x11)), cshift));

		mval = _mm256_sub_epi32(mval, _mm256_castps_si256(_mm256_cmp_ps(me2, m_in0, 0x11)));
		mval = _mm256_sub_epi32(mval, _mm256_mullo_epi32(_mm256_castps_si256(_mm256_cmp_ps(me2, m_in1, 0x11)), cshift));

		mval = _mm256_sub_epi32(mval, _mm256_castps_si256(_mm256_cmp_ps(me3, m_in0, 0x11)));
		mval = _mm256_sub_epi32(mval, _mm256_mullo_epi32(_mm256_castps_si256(_mm256_cmp_ps(me3, m_in1, 0x11)), cshift));

		_mm256_store_si256((__m256i*) tmp_out, mval);
		for(ssize_t j = 0; j < 8; j++){
			uint32_t val = (uint32_t) tmp_out[j];
			uint32_t v0 = val >> 4;
			uint32_t v1 = val & 15;
			out[i + j] = (uint8_t) v1;
			out[i + j + 8] = (uint8_t) v0;
		}
	}
}


inline void comp_shift_add(const __m256 dat0, const __m256 dat1, const __m256 dat2, const __m256 dat3,
						   const __m256 level, __m256i& accumulate)
{
	accumulate = _mm256_sub_epi32(accumulate, _mm256_castps_si256(_mm256_cmp_ps(level, dat0, 0x11)));
	accumulate = _mm256_sub_epi32(accumulate, _mm256_sllv_epi32(_mm256_castps_si256(_mm256_cmp_ps(level, dat1, 0x11)), _mm256_set1_epi32(4)));
	accumulate = _mm256_sub_epi32(accumulate, _mm256_sllv_epi32(_mm256_castps_si256(_mm256_cmp_ps(level, dat2, 0x11)), _mm256_set1_epi32(8)));
	accumulate = _mm256_sub_epi32(accumulate, _mm256_sllv_epi32(_mm256_castps_si256(_mm256_cmp_ps(level, dat3, 0x11)), _mm256_set1_epi32(12)));
}


// requires 32-byte aligned input!
inline void quantize_naive5_simd4(const DTYPE_F* in, uint8_t* out,
					 const ssize_t n)
{
	assert((n % 32) == 0);
	const __m256 me0 = _mm256_broadcast_ss(edges5);
	const __m256 me1 = _mm256_broadcast_ss(edges5 + 1);
	const __m256 me2 = _mm256_broadcast_ss(edges5 + 2);
	const __m256 me3 = _mm256_broadcast_ss(edges5 + 3);

	// shift by eight bits each
	// const __m256i cshift1 = _mm256_set1_epi32(256); // shift by 4
	// const __m256i cshift2 = _mm256_set1_epi32(65536); // shift by 8
	// const __m256i cshift3 = _mm256_set1_epi32(16777216); // shift by 12

	// const __m256i cdeltas = _mm256_set1_epi

	const __m256i cshuffle = _mm256_setr_epi8(0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15,
											  0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15);
	const __m256i cpermute = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

	for(ssize_t i = 0; i < n/32; i++){
		const __m256 m_in0 = _mm256_load_ps(in + i * 32);
		const __m256 m_in1 = _mm256_load_ps(in + i * 32 + 8);
		const __m256 m_in2 = _mm256_load_ps(in + i * 32 + 16);
		const __m256 m_in3 = _mm256_load_ps(in + i * 32 + 24);
		__m256i mval = _mm256_set1_epi32(0);
		// __m256i mlen = _mm256_set1_epi32(0);

		comp_shift_add(m_in0, m_in1, m_in2, m_in3, me0, mval);
		comp_shift_add(m_in0, m_in1, m_in2, m_in3, me1, mval);
		comp_shift_add(m_in0, m_in1, m_in2, m_in3, me2, mval);
		comp_shift_add(m_in0, m_in1, m_in2, m_in3, me3, mval);

		// shuffle then permute to rearrange and store to output
		_mm256_store_si256((__m256i*) (out + i * 32), _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(mval, cshuffle), cpermute));
	}
}


inline void dequantize(const uint8_t* levels, DTYPE_F* out, const ssize_t nsamp)
{
	for(ssize_t i = 0; i < nsamp; i++){
		out[i] = dequant5[levels[i]];
	}
}


// assumes a uint32_t "chunk" structure
// provides an upper bound on the number of distinct samples
// coincidentally is the same as an upper bound on the size of the
// resulting unpacked array.
inline const ssize_t decode_bound(const ssize_t size_in)
{
	return size_in * 4;
}


inline const ssize_t encode_bound(const ssize_t samples_in)
{
	ssize_t r = 0;
	ssize_t base = (samples_in * 3) / 8;

	if(((3 * samples_in) % 8) > 0){
		r = 1;
	}

	return base + r;
}


inline const ssize_t encode_ceil_bound(const ssize_t samples_in)
{
	return ((encode_bound(samples_in)/sizeof(uint32_t)) + 1) * sizeof(uint32_t);
}


// now we return the overall bit position, as opposed to the
inline const ssize_t huff_encode_kernel(const uint8_t* in, uint32_t* out, const ssize_t n_in, ssize_t& i0, ssize_t& bit0, uint32_t& tmp)
{
	ssize_t iout = i0; // tracks out index
	ssize_t bit_pos = bit0; // tracks bit position within
	uint32_t this_tmp = tmp; // tracks temporary working buffer from last iteration, if applicable
	uint32_t hval;
	ssize_t this_size, ival;
	uint32_t shuttle;
	ssize_t r;
	for(ssize_t i = 0; i < n_in; i++){
		shuttle = 0;
		ival = (ssize_t) in[i];
		hval = (uint32_t) codes[ival];
		this_size = (ssize_t) lens[ival];
		shuttle = hval << bit_pos;
		this_tmp += shuttle;

		bit_pos += this_size;

		if(bit_pos >= 32){
			r = bit_pos - 32;
			out[iout] = this_tmp;

			this_tmp = 0;
			if(r > 0){
				this_tmp += hval >> (this_size - r);
			}
			iout++;
			bit_pos = r;
		}
	}

	ssize_t rlen = iout;
	if(bit_pos > 0){
		// TODO this might be redundant
		out[iout] = this_tmp;
		rlen++;
	}

	i0 = iout;
	bit0 = bit_pos;
	tmp = this_tmp;

	// retain the "legacy" return of safe length
	// more information is contained in the index, bit pair
	return rlen;
}


inline const ssize_t huff_decode_kernel(const uint32_t* in, uint8_t* out,
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

		// probably should use a map or a sparse array
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


// The naive5_encoder kernel performs the following operations:
//
//   - normalizes data, by subtracting mean and dividing rms
//      (where the mean and rms are assumed precomputed)
//   - quantizes (with 5 levels)
//   - huffman compresses
//   - coalesces the 2-bit and 3-bit huffman compressed tokens into
//      a contiguous bit stream, and writes it to an output buffer.
//
//
// Executive summary: the following function applies the kernel to
// a long sample array.
//
//   vector<uint8_t> encode(const float *samples, int64_t nsamples, float mean, float rms)
//   {
//       assert(nsamples % 64 == 0);            // currently assumed by kernel (could be changed)
//       assert(is_aligned_pointer(samples));   // currently assumed by kernel (could be changed)
//
//       int64_t nout = naive5_encoder::min_nbytes(nsamples);
//       vector<uint8_t> out(nout);
//
//       naive5_encoder encoder(&out[0], mean, rms);
//
//       for (int i = 0; i < nsamples; i += 64) {
//           encoder.encode64_aligned(samples + i);
//
//           // Note: can change mean/rms between calls to encode64_aligned(),
//           // by calling encoder.set_mean_and_rms().
//       }
//
//       int64_t nbits_written = encoder.flush();   // don't forget this!
//       return out;
//   }


struct n5_encoder {

    // --------------------  External interface  --------------------

    // The 'mean' and 'rms' arguments are used to normalize floating-point data,
    // prior to quantization (x_normalized = (x-mean)/rms). Huffman-compressed
    // output data will be streamed to the 'out' buffer.

    n5_encoder(void *out, float mean=0.0, float rms=1.0)
    {
	this->outbuf = reinterpret_cast<int64_t *> (out);
	this->set_mean_and_rms(mean, rms);
    }

    // To use the kernel, loop over calls to encode64_aligned(). As the name suggests,
    // each call processes 64 floating-point inputs, and 'p' must be an aligned pointer
    // (otherwise you will get a segfault!!)

    inline void encode64_aligned(const float *p);

    // After the last call to encode64_aligned(), you should call flush(), to ensure
    // that any internally buffered data is written to the output buffer. Returns the
    // total number of bits (not bytes!) written to the output buffer.
    
    inline int64_t flush();

    // Changes the mean and rms, between calls to encode64_aligned().
    
    inline void set_mean_and_rms(float mean, float rms);

    // Returns the minimum number of bytes needed to allocate the 'out' buffer.
    
    static int64_t min_nbytes(int64_t nsamples)
    {
	// Round up nbits to a multiple of 64, since kernel operates in 64-bit chunks.
	int64_t min_nbits = nsamples * 3;
	int64_t round_nbits = ((min_nbits + 63) / 64)  * 64;
	return round_nbits / 8;
    }

    // --------------------  Internal implementation  --------------------
    
    __m256i state;
    __m256i npad;
    __m256  params;

    __m256i nbits_c = _mm256_setzero_si256();
    int64_t shuttle = 0;
    int64_t *outbuf;

    static constexpr float unnormalized_level0 = 0.38228386f;
    static constexpr float unnormalized_level1 = 1.24435754f;
    
    template<int N> inline void _compute_syndrome(__m256 x, __m256 mean, __m256 level0, __m256 level1, __m256 sign_bit, __m256i c1);
    template<int B> inline void _set_state_bit(__m256 predicate, __m256i c1);
    template<int S> inline void _shuttle64(__m128i a, __m128i b, int64_t nowrite_flag);

    inline void _permute();
    inline void _huffman_compress();
    inline void _coalesce_4_8();
    inline void _coalesce_8_32();
    inline void _coalesce_32_64();
    inline void _shuttle();
};


// -------------------------------------------------------------------------------------------------


inline void n5_encoder::set_mean_and_rms(float mean, float rms)
{
    // Each 128-bit "half" of 'params' contains the following parameter vector:
    //
    //   params[0] = mean
    //   params[1] = level0 = 0.38228386 * rms
    //   params[2] = level1 = 1.24435754 * rms
    //   params[3] = sign_bit = -0.0
    //
    // Note that we multiply the quantization levels by the rms. This is equivalent
    // to dividing the data by the rms, but saves a few clock cycles.
    //
    // Packing the params into a single __m256 (rather than using four __m256's)
    // saves a few registers, which turns out to make the kernel a little faster.
    
    __m256 param0 = _mm256_set1_ps(mean);
    __m256 param1 = _mm256_set1_ps(rms * n5_encoder::unnormalized_level0);
    __m256 param2 = _mm256_set1_ps(rms * n5_encoder::unnormalized_level1);
    __m256 param3 = _mm256_set1_ps(-0.0);

    // Pack params into a single __m256
    __m256 param01 = _mm256_blend_ps(param0, param1, 0xaa);  // (10101010)_2
    __m256 param23 = _mm256_blend_ps(param2, param3, 0xaa);  // (10101010)_2    
    this->params = _mm256_blend_ps(param01, param23, 0xcc);  // (11001100)_2
}


inline void n5_encoder::encode64_aligned(const float *p)
{
    // Unpack params vector.
    const __m256 mean = _mm256_permute_ps(this->params, 0x00);      // (0000)_4
    const __m256 level0 = _mm256_permute_ps(this->params, 0x55);    // (1111)_4
    const __m256 level1 = _mm256_permute_ps(this->params, 0xaa);    // (2222)_4
    const __m256 sign_bit = _mm256_permute_ps(this->params, 0xff);  // (3333)_4
    const __m256i c1 = _mm256_set1_epi32(1);

    // Clear state bits, before computing syndromes.
    // (Note: this->npad will be initialized in _huffman_compress().)
    this->state = _mm256_setzero_si256();

    // We factor the encode kernel into a bunch of helper inlines, so that
    // the helpers can be separately unit-testsed. The meaning of each helper
    // is explained in comments below!
    
    _compute_syndrome<0> (_mm256_load_ps(p), mean, level0, level1, sign_bit, c1);
    _compute_syndrome<1> (_mm256_load_ps(p+8), mean, level0, level1, sign_bit, c1);
    _compute_syndrome<2> (_mm256_load_ps(p+16), mean, level0, level1, sign_bit, c1);
    _compute_syndrome<3> (_mm256_load_ps(p+24), mean, level0, level1, sign_bit, c1);
    _compute_syndrome<4> (_mm256_load_ps(p+32), mean, level0, level1, sign_bit, c1);
    _compute_syndrome<5> (_mm256_load_ps(p+40), mean, level0, level1, sign_bit, c1);
    _compute_syndrome<6> (_mm256_load_ps(p+48), mean, level0, level1, sign_bit, c1);
    _compute_syndrome<7> (_mm256_load_ps(p+56), mean, level0, level1, sign_bit, c1);

    _permute();
    _huffman_compress();
    
    _coalesce_4_8();
    _coalesce_8_32();
    _coalesce_32_64();
    
    _shuttle();
}


template<int B>
inline void n5_encoder::_set_state_bit(__m256 predicate, __m256i c1)
{
    // This helper function is called by _compute_syndrome().
    // The 'predicate' arg is an int32[8], in which each int32 is 0 ("false") or -1 ("true").
    // The 'c1' arg is an int32[8] containing { 1, ..., 1 }.
    //
    // This function interprets this->state as an int32[8], and is equivalent to:
    //
    //   for (int i = 0; i < 8; i++)
    //      if (predicate[i])
    //          state[i] |= (1 << B)

    __m256i mask = _mm256_slli_epi32(c1, B);
    __m256i x = _mm256_castps_si256(predicate);
    __m256i y = _mm256_and_si256(x, mask);
    
    this->state = _mm256_or_si256(this->state, y);
}


template<int N>
inline void n5_encoder::_compute_syndrome(__m256 x, __m256 mean, __m256 level0, __m256 level1, __m256 sign_bit, __m256i c1)
{
    // For each floating-point sample x, the "syndrome" of x is defined to be 3 bits:
    //   bit0 = (x >= 0)
    //   bit1 = (abs(x) >= level0)
    //   bit2 = (abs(x) >= level1)
    //
    // This function interprets this->state as int4[64], and stores the syndrome of x[i]
    // (where 0 <= i < 8) in state[8*i+B].
    //
    // The 'sign_bit' arg is a float32[8] containing { -0.0, ..., -0.0 }.
    // The 'c1' arg is an int32[8] containing { 1, ..., 1 }.
    
    x = _mm256_sub_ps(x, mean);
    
    // Small trick here: implement abs(x) by clearing the sign bit
    __m256 absx =  _mm256_andnot_ps(sign_bit, x);

    _set_state_bit<4*N> (_mm256_cmp_ps(x, sign_bit, _CMP_GE_OQ), c1);
    _set_state_bit<4*N+1> (_mm256_cmp_ps(absx, level0, _CMP_GE_OQ), c1);
    _set_state_bit<4*N+2> (_mm256_cmp_ps(absx, level1, _CMP_GE_OQ), c1);
}


inline void n5_encoder::_permute()
{
    // After 8 calls to _compute_syndrome(), this->state contains syndromes for 64
    // samples. If 'state' is interpreted as int4[64], then state[8*i+j] contains
    // the syndrome for sample (8*j+i).
    //
    // The _permute() function transposes 'state', so that state[8*i+j] contains the
    // syndrome for sample (8*i+j).
    //
    // Note: I suspect this implementation is a little suboptimal, but optimizing
    // further would only speed up the kernel by a few percent, so I didn't bother.
    
    // In the first part of this function, we permute int4's
    // such that the remaining permutation is "pure int8".
    //
    // Start: [ A0|A1, B0|B1 ]
    // Want:  [ A0|B0, A1|B1 ]

    __m256i a = _mm256_srli_epi16(state, 4);      // [ A1|*, *|* ]
    __m256i b = _mm256_slli_epi16(state, 4);      // [  *|*, *|B0 ]
    __m256i c = _mm256_blend_epi32(a, b, 0xaa);   // [ A1|*, *|B0 ]; 0xaa = (1010101010)_2
    __m256i d = _mm256_shuffle_epi32(c, 0xb1);    // [ *|B0, A1|* ]; 0xb1 = (2301)_4

    // (Low 4 bits in first 4 bytes) + (High 4 bits in next 4 bytes)
    __m256i mask = _mm256_set1_epi64x(0xf0f0f0f00f0f0f0fL);
    __m256i e = _mm256_and_si256(mask, state);      // [ A0|0, 0|B1 ]
    __m256i f = _mm256_andnot_si256(mask, d);       // [ 0|B0, A1|0 ]
    __m256i g = _mm256_or_si256(e, f);              // [ A0|B0, A1|B1 ]

    // In the second part of this function, we permute int8's.
    //
    // Start: [ a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 c3 d0 d1 d2 d3
    //          e0 e1 e2 e3 f0 f1 f2 f3 g0 g1 g2 g3 h0 h1 h2 h3 ]
    //
    // Want: [ a0 c0 e0 g0 b0 d0 f0 h0 a1 c1 e1 g1 b1 d1 f1 h1
    //         a2 c2 e2 g2 b2 d2 f2 h2 a3 c3 e3 g3 b3 d3 f3 h3 ]
    
    // h = [ a0 c0 - - b0 d0 - - a1 c1 - - b1 d1 - -
    //       - - e2 g2 - - f2 h2 - - e3 g3 - - f3 h3 ]
    
    __m256i ctl0 = _mm256_set_epi8(
	15, 7, 255, 255, 11, 3, 255, 255,
	14, 6, 255, 255, 10, 2, 255, 255,
        255, 255, 13, 5, 255, 255, 9, 1,
        255, 255, 12, 4, 255, 255, 8, 0
    );

    __m256i h = _mm256_shuffle_epi8(g, ctl0);

    // k = [ - - e0 g0 - - f0 h0 - - e1 g1 - - f1 h1
    //       a2 c2 - - b2 d2 - - a3 c3 - - b3 d3 - - ]

    __m256i ctl1 = _mm256_set_epi8(
	255, 255, 15, 7, 255, 255, 11, 3,
	255, 255, 14, 6, 255, 255, 10, 2,
        13, 5, 255, 255, 9, 1, 255, 255,
        12, 4, 255, 255, 8, 0, 255, 255
    );

    __m256i j = _mm256_permute2x128_si256(g, g, 0x01);
    __m256i k = _mm256_shuffle_epi8(j, ctl1);

    this->state = _mm256_or_si256(h, k);
}


inline void n5_encoder::_huffman_compress()
{
    // This function converts 3-bit syndromes to (2 or 3)-bit Huffman
    // compressed tokens.
    //
    // Input:
    //
    //   this->state = int4[64] containing 3-bit syndromes
    //
    // Output:
    //
    //   this->state = int4[64] containing (2 or 3)-bit tokens
    //
    //   this->npad = int4[64] containing the number of "padding"
    //     bits in each token (i.e. npad=1 for a 3-bit token, and
    //     npad=2 for a 2-bit token).
    
    __m256i state_l1 = _mm256_slli_epi16(state, 1);  // morally epi4
    __m256i state_l2 = _mm256_slli_epi16(state, 2);  // morally epi4
    __m256i state_r1 = _mm256_srli_epi16(state, 1);  // morally epi4
    __m256i state_r2 = _mm256_srli_epi16(state, 2);  // morally epi4

    // From spshuff:
    //   codes[5] = {7, 2, 0, 1, 3}; // value of each huffman code
    //   lens[5] = {3, 2, 2, 2, 3}; // bit length of each code
    //
    // Equivalent bitplane version:
    //   out0 = in2 | (in0 & in1)
    //   out1 = in2 | (~in0 & in1)
    //   out2 = ~in0 & in2
    
    __m256i out0 = _mm256_or_si256(state_r2, _mm256_and_si256(state, state_r1));
    __m256i out1 = _mm256_or_si256(state_r1, _mm256_andnot_si256(state_l1, state));
    __m256i out2 = _mm256_andnot_si256(state_l2, state);

    __m256i bit0 = _mm256_set1_epi8(0x11);
    __m256i bit1 = _mm256_set1_epi8(0x22); // _mm256_slli_epi16(bit0, 1);
    __m256i bit2 = _mm256_set1_epi8(0x44); // _mm256_slli_epi16(bit0, 2);

    this->state = _mm256_and_si256(out0, bit0);
    this->state = _mm256_or_si256(state, _mm256_and_si256(out1, bit1));
    this->state = _mm256_or_si256(state, _mm256_and_si256(out2, bit2));
    
    // npad = in2 ? 1 : 2
    this->npad = _mm256_sub_epi8(bit1, _mm256_and_si256(state_r2, bit0));  // morally epi4
}


inline void n5_encoder::_coalesce_4_8()
{
    // Concatenates 4-bit tokens in pairs, to get 8-bit tokens
    //
    // Input:
    //   this->state = int4[64] containing (2 or 3)-bit tokens
    //   this->npad = int4[64] containing (4 - token_nbits)
    //
    // Output:
    //   this->state = int8[32] containing tokens (4 <= token_nbits <= 6)
    //   this->npad = int8[32] containing (8 - token_nbits)

    __m256i low4_bits = _mm256_set1_epi8(0x0f);

    __m256i state0 = _mm256_and_si256(low4_bits, this->state);
    __m256i state1 = _mm256_andnot_si256(low4_bits, this->state);
    __m256i npad0 = _mm256_and_si256(low4_bits, this->npad);
    __m256i npad1 = _mm256_andnot_si256(low4_bits, this->npad);

    // Want _mm256_slrv_epi8(state1, npad0) here, but _mm256_slrv_epi8() doesn't exist!
    // The following workaround is equivalent (assuming npad0 is 1 or 2 everywhere)
    __m256i mask = _mm256_cmpeq_epi8(npad0, _mm256_set1_epi8(0x2));
    __m256i state1_r1 = _mm256_srli_epi16(state1, 1);
    __m256i state1_r2 = _mm256_srli_epi16(state1, 2);
    __m256i state1_r12 = _mm256_blendv_epi8(state1_r1, state1_r2, mask);
    
    this->state = _mm256_or_si256(state0, state1_r12);
    this->npad = _mm256_add_epi8(npad0, _mm256_srli_epi16(npad1, 4));  // morally _mm256_slri_epi8()
}


inline void n5_encoder::_coalesce_8_32()
{
    // Concatenates 8-bit tokens in quadruples, to get 32-bit tokens
    //
    // Input:
    //   this->state = int8[32] containing tokens (4 <= token_nbits <= 6)
    //   this->npad = int8[32] containing (8 - token_nbits)
    //
    // Output:
    //   this->state = int32[8] containing tokens (16 <= token_nbits <= 24)
    //   this->npad = int32[8] containing (32 - token_nbits)
    
    __m256i low8_bits = _mm256_set1_epi32(0x000000ff);
    
    __m256i state0 = _mm256_and_si256(state, low8_bits);
    __m256i state1 = _mm256_and_si256(state, _mm256_slli_epi32(low8_bits, 8));
    __m256i state2 = _mm256_and_si256(state, _mm256_slli_epi32(low8_bits, 16));
    __m256i state3 = _mm256_and_si256(state, _mm256_slli_epi32(low8_bits, 24));

    __m256i a = _mm256_add_epi8(npad, _mm256_srli_epi32(npad, 8));
    __m256i b = _mm256_add_epi8(a, _mm256_srli_epi32(npad, 16));
    __m256i c = _mm256_add_epi8(a, _mm256_srli_epi32(a, 16));
    
    __m256i npad0 = _mm256_and_si256(npad, low8_bits);
    __m256i npad01 = _mm256_and_si256(a, low8_bits);
    __m256i npad012 = _mm256_and_si256(b, low8_bits);
    __m256i npad0123 = _mm256_and_si256(c, low8_bits);

    state1 = _mm256_srlv_epi32(state1, npad0);
    state2 = _mm256_srlv_epi32(state2, npad01);
    state3 = _mm256_srlv_epi32(state3, npad012);

    __m256i state01 = _mm256_or_si256(state0, state1);
    __m256i state23 = _mm256_or_si256(state2, state3);
    
    this->state = _mm256_or_si256(state01, state23);
    this->npad = npad0123;
}


inline void n5_encoder::_coalesce_32_64()
{
    // Concatenates 32-bit tokens in pairs, to get 64-bit tokens
    //
    // Input:
    //   this->state = int32[8] containing tokens (16 <= token_nbits <= 24)
    //   this->npad = int32[8] containing (32 - token_nbits)
    //
    // Output:
    //   this->state = int64[4] containing tokens (32 <= token_nbits <= 48)
    //   this->npad = int64[4] containing (64 - token_nbits)
    
    __m256i low32_bits = _mm256_set1_epi64x(0x00000000ffffffffL);

    __m256i state0 = _mm256_and_si256(low32_bits, this->state);
    __m256i state1 = _mm256_andnot_si256(low32_bits, this->state);
    __m256i npad0 = _mm256_and_si256(low32_bits, this->npad);
    
    this->state = _mm256_or_si256(state0, _mm256_srlv_epi64(state1, npad0));
    this->npad = _mm256_add_epi64(npad0, _mm256_srli_epi64(npad, 32));
}


inline __m256i _cumsum64_restricted(__m256i x)
{
    // Standalone helper function for n5_encoder::_shuttle().
    //
    // The argument 'x' is interpreted as an int64[4], and each x[i]
    // is assumed to be < 2^32. (The "_restricted" name is intended to
    // emphasize that an extra assumption exists).
    //
    // Returns { 0, x[0], x[0]+x[1], x[0]+x[1]+x[2] },
    // an int64[4] packed into an __m256i.
    
    __m256i a = _mm256_shuffle_epi32(x, 0x4e);           // [x1 x0 * *]; 0x4e = (1032)_4
    __m256i b = _mm256_shuffle_epi32(x, 0x45);           // [0 x0 0 x2]; 0x45 = (1011)_4
    __m256i c = _mm256_add_epi64(x, a);                  // [(x0+x1) (x0+x1) * *]
    __m256i d = _mm256_permute2x128_si256(c, c, 0x01);   // [* * (x0+x1) (x0+x1)]
    __m256i e = _mm256_add_epi64(b, d);                  // [* * (x0+x1) (x0+x1+x2) ]
    __m256i f = _mm256_blend_epi32(b, e, 0xf0);          // [ 0 x0 (x0+x1) (x0+x1+x2) ]
    
    return f;
}


inline int64_t _gather_predicate64(__m256i predicate)
{
    // Standalone helper function for n5_encoder::_shuttle().
    //
    // The predicate is an int64[4], such that each element
    // predicate[i] is either 0 ("false") or -1 ("true").
    //
    // Converts predicate to an int16[4], and returns it as a single int64_t.
    
    __m256i a = _mm256_shuffle_epi32(predicate, 0x08);    // (0020)_2
    __m256i b = _mm256_shufflelo_epi16(a, 0x88);  // (2020)_2
    __m128i c = _mm256_extractf128_si256(b, 0);
    __m128i d = _mm256_extractf128_si256(b, 1);
    __m128i e = _mm_blend_epi32(c, d, 0x2);    // (00000010)_2
    int64_t f = _mm_extract_epi64(e, 0);
    
    return f;
}


inline __m256i _broadcast_last64(__m256i x)
{
    // Standalone helper function for n5_encoder::_shuttle().
    //
    // The 'x' argument is interpreted as int64[4].
    // Returns { x[3], x[3], x[3], x[3] }, an __m256i interpreted as int64[4].
    
    __m256i a = _mm256_permute2x128_si256(x, x, 0x11);
    __m256i b = _mm256_shuffle_epi32(a, 0xee);   // 0xee = (3232)_4
    
    return b;
}


template<int S>
inline void n5_encoder::_shuttle64(__m128i a128, __m128i b128, int64_t nowrite_flag)
{
    // Helper method called by n5_encoder::_shuttle().
    // See comments in _shuttle() for an explanation of what it does!

    shuttle |= _mm_extract_epi64(a128, S);

    if (nowrite_flag & 1)
	return;

    // 64-bit "transaction"
    *outbuf++ = shuttle;
    shuttle = _mm_extract_epi64(b128, S);
}


inline void n5_encoder::_shuttle()
{
    // When _shuttle() is called:
    //   this->state = int64[4] containing tokens (32 <= token_nbits <= 48)
    //   this->npad = int64[4] containing (64 - token_nbits)
    //
    // The _shuttle() function is responsible for coalescing the tokens,
    // and streaming them to the output array. This is done by buffering
    // bits, and writing them in 64-bit transactions. Buffering state is
    // maintained by the following members of 'n5_encoder':
    //
    //   this->nbits_c = total number of bits so far (including buffer)
    //   this->shuttle = int64_t containing bits which are buffered but unwritten
    //   this->outbuf = pointer to current buffer

    // int64 nbits_start[4] = total bitcount, up to start of each token
    // int64 nbits_end[4] = total bitcount, up to end of each token
    __m256i c64 = _mm256_set1_epi64x(64);
    __m256i nbits = _mm256_sub_epi64(c64, npad);
    __m256i nbits_start = _mm256_add_epi64(nbits_c, _cumsum64_restricted(nbits));
    __m256i nbits_end = _mm256_add_epi64(nbits_start, nbits);

    // Split bits in each token into pair (a,b)
    //
    //   a = contribution of token to first 64-bit transaction
    //     = token << (nbits_start % 64)
    //
    //   b = contribution of token to next 64-bit transaction (can be zero)
    //     = token >> (64 - (nbits_start % 64))
    
    __m256i c63 = _mm256_set1_epi64x(63);
    __m256i s = _mm256_and_si256(nbits_start, c63);
    __m256i a = _mm256_sllv_epi64(state, s);
    __m256i b = _mm256_srlv_epi64(state, _mm256_sub_epi64(c64,s));

    // For each token, compute a boolean 'nowrite_flag', which indicates
    // whether the token triggers a 64-bit transaction. We represent the
    // nowrite_flags as an int16_t[4], packed as a single int64_t.

    __m256i t1 = _mm256_andnot_si256(c63, nbits_start);
    __m256i t2 = _mm256_andnot_si256(c63, nbits_end);
    __m256i p = _mm256_cmpeq_epi64(t1, t2);
    int64_t nowrite_flags = _gather_predicate64(p);

    // For each token, call _shuttle64() with the appropriate (a, b, nowrite_flag),
    // to update internal buffering state (except nbits_c), and (possibly) write a
    // 64-bit transaction to the output array.
    
    __m128i a0 = _mm256_extractf128_si256(a, 0);
    __m128i a1 = _mm256_extractf128_si256(a, 1);
    __m128i b0 = _mm256_extractf128_si256(b, 0);
    __m128i b1 = _mm256_extractf128_si256(b, 1);
    
    _shuttle64<0> (a0, b0, nowrite_flags);
    _shuttle64<1> (a0, b0, nowrite_flags >> 16);
    _shuttle64<0> (a1, b1, nowrite_flags >> 32);
    _shuttle64<1> (a1, b1, nowrite_flags >> 48);

    // Only remaining step is updating nbits_c.
    this->nbits_c = _broadcast_last64(nbits_end);
}



inline int64_t n5_encoder::flush()
{
    __m128i a = _mm256_extractf128_si256(nbits_c, 0);
    int64_t b = _mm_extract_epi64(a, 0);

    if ((b & 63) != 0) {
	// Flush partially buffered transaction to output array.
	*outbuf = shuttle;
    }

    return b;
}

} // namespace spshuff
// This header is constructed to avoid Python.h-dependent declarations
