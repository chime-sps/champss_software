#include <memory>
#include <random>
#include <cstring>
#include <cassert>
#include <sstream>
#include <iostream>
#include <sys/time.h>
#include <spshuff.hpp>
#include "n5_encoder_testing.hpp"


using namespace std;
using namespace spshuff;


// -------------------------------------------------------------------------------------------------
//
// Unit test standalone inlines: _cumsum64_restricted(), _gather_predicate64(), _broadcast_last64()


static void test_cumsum64_restricted(std::mt19937 &rng, int niter)
{
    for (int iter = 0; iter < niter; iter++) {
    vector<int64_t> v(4, 0);
    for (int i = 0; i < 4; i++)
        v[i] = randint(rng, 0, 1L << 31);   // upper limit (1L << 32) segfaults here (?!!)

    vector<int64_t> w(4, 0);
    for (int i = 1; i < 4; i++)
        w[i] = w[i-1] + v[i-1];
    
    __m256i x = _mm256_loadu_si256((__m256i *) &v[0]);
    __m256i y = _cumsum64_restricted(x);
    
    vector<int64_t> u(4, 0);
    _mm256_storeu_si256((__m256i *) &u[0], y);
    
    for (int i = 0; i < 4; i++)
        assert(w[i] == u[i]);
    }

    cout << "test_cumsum64_restricted: passed, " << niter << " iterations" << endl;
}


static void test_gather_predicate64(std::mt19937 &rng, int niter)
{
    for (int iter = 0; iter < niter; iter++) {
    vector<int64_t> v(4, 0);
    for (int i = 0; i < 4; i++)
        v[i] = randint(rng,0,2) ? -1L: 0;
    
    __m256i x = _mm256_loadu_si256((__m256i *) &v[0]);
    int64_t y = _gather_predicate64(x);
    
    for (int i = 0; i < 4; i++) {
        bool t1 = (v[i] != 0);
        bool t2 = ((y & (1L << (16*i))) != 0);
        assert(t1 == t2);
    }
    }
    
    cout << "test_gather_predicate64: passed, " << niter << " iterations" << endl;
}


static void test_broadcast_last64(std::mt19937 &rng, int niter)
{
    for (int iter = 0; iter < niter; iter++) {
    vector<int64_t> v(4, 0);
    for (int i = 0; i < 4; i++) {
        v[i] = randint(rng, 0, 1L << 31);
        v[i] += randint(rng, 0, 1L << 31) << 31;
    }
    
    __m256i x = _mm256_loadu_si256((__m256i *) &v[0]);
    __m256i y = _broadcast_last64(x);
    
    vector<int64_t> w(4, 0);
    _mm256_storeu_si256((__m256i *) &w[0], y);

    for (int i = 0; i < 4; i++)
        assert(w[i] == v[3]);
    }   
    
    cout << "test_broadcast_last64: passed, " << niter << " iterations" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// Helper class used in some unit tests


struct m256i_wrapper {
    vector<uint8_t> v;

    m256i_wrapper() : v(32,0) { }
    m256i_wrapper(__m256i x) : v(32,0) { _mm256_storeu_si256((__m256i *) &v[0], x); }

    inline __m256i unwrap() const
    {
    return _mm256_loadu_si256((__m256i *) &v[0]);
    }

    inline bool get_bit(int b) const
    {
    assert(b >= 0 && b < 256);
    int i = b / 8;
    int s = 1 << (b % 8);
    return v[i] & s;
    }
    
    inline void set_bit(int b, bool val=true)
    {
    assert(b >= 0 && b < 256);
    int i = b / 8;
    int s = 1 << (b % 8);
    v[i] &= ~s;
    if (val) v[i] |= s;
    }

    inline uint8_t get_bits(int b, int n) const
    {
    assert(n > 0 && n <= 8);
    uint8_t ret = 0;
    for (int i = 0; i < n; i++)
        if (get_bit(b+i))
        ret |= (1 << i);
    return ret;
    }

    inline void set_bits(int b, int n, uint8_t x)
    {
    assert(n > 0 && n <= 8);

    for (int i = 0; i < n; i++)
        if (x & (1 << i))
        set_bit(b+i);
    }

    inline std::string bitstring(int b, int n) const
    {
    std::stringstream ss;
    for (int i = 0; i < n; i++)
        ss << int(get_bit(b+i));
    return ss.str();
    }

    inline std::string bitlist() const
    {
    std::stringstream ss;
    ss << "[";

    bool flag = false;
    for (int b = 0; b < 256; b++) {
        if (!get_bit(b))
        continue;
        if (flag)
        ss << ",";
        ss << b;
        flag = true;
    }

    ss << "]";
    return ss.str();
    }
};


static bool m256i_equal(__m256i x, __m256i y)
{
    m256i_wrapper wx(x);
    m256i_wrapper wy(y);

    for (int i = 0; i < 32; i++)
    if (wx.v[i] != wy.v[i])
        return false;

    return true;
}


// -------------------------------------------------------------------------------------------------
//
// Helper functions for interacting with 'struct n5_encoder'.


static void check_state_invariants(const n5_encoder &e, int nbits)
{
    assert(nbits >= 4 && nbits <= 256);
    assert((nbits & (nbits-1)) == 0);   // fails if nbits is not a power of 2
    
    m256i_wrapper state(e.state);
    m256i_wrapper npad(e.npad);
    
    for (int i = 0; i < 256/nbits; i++) {
        uint8_t p = npad.get_bits(i*nbits, min(nbits,8));
    assert (p >= nbits/4 && p <= nbits/2);
        
    for (int b = min(nbits,8); b < nbits; b++)
        assert(!npad.get_bit(i*nbits+b));
    
    for (int b = nbits-p; b < nbits; b++)
        assert(!state.get_bit(i*nbits+b));
    }
}


static std::string state_bitstring(const n5_encoder &e, int nbits, int i)
{
    m256i_wrapper state(e.state);
    m256i_wrapper npad(e.npad);

    uint8_t p = npad.get_bits(i*nbits, min(nbits,8));
    return state.bitstring(i*nbits, nbits-p);
}


static void print_state(const n5_encoder &e, int nbits, const std::string &label="state")
{
    std::cout << label << "(nbits=" << nbits << ")\n";
    check_state_invariants(e, nbits);
    
    for (int i = 0; i < 256/nbits; i++)
    std::cout << "    " << label << "[" << i << "]: " << state_bitstring(e,nbits,i) << std::endl;
}


static bool states_equal_noisy(const n5_encoder &e1, const n5_encoder &e2, int nbits, const std::string &label="states")
{
    check_state_invariants(e1, nbits);
    check_state_invariants(e2, nbits);
    
    if (m256i_equal(e1.state,e2.state) && m256i_equal(e1.npad,e2.npad))
    return true;

    std::cout << label << " mismatch\n";
    
    for (int i = 0; i < 256/nbits; i++) {
    std::string s1 = state_bitstring(e1, nbits, i);
    std::string s2 = state_bitstring(e2, nbits, i);
    
    if (s1 != s2)
        std::cout << "    " << i << " " << s1 << " " << s2 << std::endl;
    }

    return false;
}


static void randomize_state(n5_encoder &e, std::mt19937 &rng, int nbits)
{
    m256i_wrapper new_state;
    m256i_wrapper new_npad;

    for (int i = 0; i < 256/nbits; i++) {
    uint8_t p = randint(rng, nbits/4, nbits/2+1);
    new_npad.set_bits(i*nbits, min(nbits,8), p);

    for (int b = 0; b < nbits-p; b++)
        new_state.set_bit(i*nbits+b, randint(rng,0,2));
    }

    e.state = new_state.unwrap();
    e.npad = new_npad.unwrap();
}


// -------------------------------------------------------------------------------------------------
//
// Unit test n5_encoder::_compute_syndrome().


template<int N>
static void _test_compute_syndrome(std::mt19937 &rng)
{
    m256i_wrapper initial_state;
    
    // First, randomize all state bits
    for (int b = 0; b < 256; b++)
    initial_state.set_bit(b, randint(rng,0,2));

    // Now clear state bits which will be computed by _compute_syndrome().
    for (int i = 0; i < 8; i++)
    for (int j = 0; j < 3; j++)
        initial_state.set_bit(32*i+4*N+j, 0);

    float mean = uniform_rand(rng, -1.0, 1.0);
    float rms = uniform_rand(rng, 1.0, 2.0);
    vector<float> v = uniform_randvec(rng, 8, mean-2*rms, mean+2*rms);
    
    n5_encoder e(nullptr, mean, rms);
    e.state = initial_state.unwrap();

    // Arguments to _compute_syndrome().
    // (From n5_encoder::set_mean_and_rms() and n5_encoder::encode64_aligned().)
    __m256 x = _mm256_loadu_ps(&v[0]);
    __m256 m = _mm256_set1_ps(mean);
    __m256 l0 = _mm256_set1_ps(rms * n5_encoder::unnormalized_level0);
    __m256 l1 = _mm256_set1_ps(rms * n5_encoder::unnormalized_level1);
    __m256 sb = _mm256_set1_ps(-0.0);
    __m256i c1 = _mm256_set1_epi32(1);

    e._compute_syndrome<N> (x, m, l0, l1, sb, c1);

    m256i_wrapper final_state(e.state);
    const float eps = 1.0e-5;

    for (int i = 0; i < 8; i++) {
    // Non-syndrome bits should remain unchanged.
    for (int j = 0; j < 32; j++) {
        if ((j < 4*N) || (j >= 4*N+3))
        assert(initial_state.get_bit(32*i+j) == final_state.get_bit(32*i+j));
    }

    // Syndrome bit 0
    if (final_state.get_bit(32*i + 4*N))
        assert(v[i] >= mean - eps * rms);
    else
        assert(v[i] <= mean + eps * rms);

    // Syndrome bit 1
    if (final_state.get_bit(32*i + 4*N + 1))
        assert(fabs(v[i]-mean) >= (n5_encoder::unnormalized_level0 - eps) * rms);
    else
        assert(fabs(v[i]-mean) <= (n5_encoder::unnormalized_level0 + eps) * rms);
    
    // Syndrome bit 2
    if (final_state.get_bit(32*i + 4*N + 2))
        assert(fabs(v[i]-mean) >= (n5_encoder::unnormalized_level1 - eps) * rms);
    else
        assert(fabs(v[i]-mean) <= (n5_encoder::unnormalized_level1 + eps) * rms);
    }
}
    

static void test_compute_syndrome(std::mt19937 &rng, int niter)
{
    for (int iter = 0; iter < niter; iter++) {
    _test_compute_syndrome<0> (rng);
    _test_compute_syndrome<1> (rng);
    _test_compute_syndrome<2> (rng);
    _test_compute_syndrome<3> (rng);
    _test_compute_syndrome<4> (rng);
    _test_compute_syndrome<5> (rng);
    _test_compute_syndrome<6> (rng);
    _test_compute_syndrome<7> (rng);
    }
    
    cout << "test_compute_syndrome: passed, " << niter << " iterations" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// Unit test n5_encoder::_permute().


static void randomize_syndrome(n5_encoder &e, std::mt19937 &rng)
{
    m256i_wrapper new_state;
    m256i_wrapper new_npad;

    for (int i = 0; i < 64; i++) {
    new_npad.set_bits(4*i, 4, 1);
    new_state.set_bit(4*i, randint(rng,0,2));
    
    if (randint(rng,0,2)) {
        new_state.set_bit(4*i+1, true);
        new_state.set_bit(4*i+2, randint(rng,0,2));
    }
    }

    e.state = new_state.unwrap();
    e.npad = new_npad.unwrap();
}


static void reference_permute(n5_encoder &e)
{
    m256i_wrapper curr_state(e.state);
    m256i_wrapper new_state;

    for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
        uint8_t x = curr_state.get_bits(32*j + 4*i, 4);
        new_state.set_bits(32*i + 4*j, 4, x);
    }
    }

    e.state = new_state.unwrap();
}


static void _test_permute(const m256i_wrapper &w)
{
    n5_encoder e_in(nullptr);
    e_in.state = w.unwrap();
    
    n5_encoder e_slow = e_in;
    reference_permute(e_slow);

    n5_encoder e_fast = e_in;
    e_fast._permute();
    
    if (!m256i_equal(e_slow.state, e_fast.state)) {
    cout << "in: " << m256i_wrapper(e_in.state).bitlist() << endl;
    cout << "slow: " << m256i_wrapper(e_slow.state).bitlist() << endl;
    cout << "fast: " << m256i_wrapper(e_fast.state).bitlist() << endl;
    throw runtime_error("test_permute failed");
    }
}


static void test_permute(std::mt19937 &rng, int niter)
{
    for (int b = 0; b < 256; b++) {
    m256i_wrapper w;
    w.set_bit(b);
    _test_permute(w);
    }

    for (int i = 0; i < niter; i++) {
    m256i_wrapper w;
    for (int j = 0; j < 32; j++)
        w.set_bits(8*j, 8, randint(rng,0,256));
    _test_permute(w);
    }

    cout << "test_permute: passed, " << niter << " iterations" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// Unit test n5_encoder::_huffman_compress().


static void reference_huffman_compress(n5_encoder &e)
{
    m256i_wrapper curr_state(e.state);
    m256i_wrapper new_state;
    m256i_wrapper new_npad;
                
    for (int i = 0; i < 64; i++) {
    bool pos = curr_state.get_bit(4*i);

    if (curr_state.get_bit(4*i+2)) {
        new_state.set_bits(4*i, 3, pos ? 3 : 7);
        new_npad.set_bits(4*i, 4, 1);
    }
    else if (curr_state.get_bit(4*i+1)) {
        new_state.set_bits(4*i, 2, pos ? 1 : 2);
        new_npad.set_bits(4*i, 4, 2);
    }
    else {
        new_state.set_bits(4*i, 2, 0);
        new_npad.set_bits(4*i, 4, 2);
    }
    }

    e.state = new_state.unwrap();
    e.npad = new_npad.unwrap();
}


static void test_huffman_compress(std::mt19937 &rng, int niter)
{
    for (int i = 0; i < niter; i++) {
    n5_encoder e_in(nullptr);
    randomize_syndrome(e_in, rng);

    n5_encoder e_slow = e_in;
    reference_huffman_compress(e_slow);

    n5_encoder e_fast = e_in;
    e_fast._huffman_compress();

    if (!states_equal_noisy(e_slow, e_fast, 4, "slow/fast huffman compress")) {
        print_state(e_in, 4, "in");
        throw runtime_error("test_huffman_compress");
    }
    }

    cout << "test_huffman_compress: passed, " << niter << " iterations" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// Unit test n5_encoder::_coalesce_*()


static void reference_coalesce(n5_encoder &e, int nbits_in, int nbits_out)
{
    m256i_wrapper curr_state(e.state);
    m256i_wrapper curr_npad(e.npad);
    m256i_wrapper new_state;
    m256i_wrapper new_npad;

    int r = nbits_out / nbits_in;

    // Outer loop over output samples.
    for (int iout = 0; iout < 256/nbits_out; iout++) {
    int bout = 0;   // Current bit position within output sample

    // Inner loop over input samples to coalesce.
    for (int iin = iout*r; iin < (iout+1)*r; iin++) {
        uint8_t p = curr_npad.get_bits(iin*nbits_in, min(nbits_in,8));

        for (int b = 0; b < nbits_in-p; b++) {
        bool x = curr_state.get_bit(iin*nbits_in + b);
        new_state.set_bit(iout*nbits_out + bout, x);
        bout++;
        }
    }

    new_npad.set_bits(iout*nbits_out, min(nbits_out,8), nbits_out-bout);
    }

    e.state = new_state.unwrap();
    e.npad = new_npad.unwrap();
}


static void test_coalesce_4_8(std::mt19937 &rng, int niter)
{
    for (int i = 0; i < niter; i++) {
    n5_encoder e_in(nullptr);
    randomize_state(e_in, rng, 4);
    
    n5_encoder e_slow = e_in;
    reference_coalesce(e_slow, 4, 8);

    n5_encoder e_fast = e_in;
    e_fast._coalesce_4_8();

    if (!states_equal_noisy(e_slow, e_fast, 8, "slow/fast coalesce_4_8")) {
        print_state(e_in, 4, "in");
        throw runtime_error("test_coalesce_4_8() failed");
    }
    }

    cout << "test_coalesce_4_8: passed, " << niter << " iterations" << endl;
}


static void test_coalesce_8_32(std::mt19937 &rng, int niter)
{
    for (int i = 0; i < niter; i++) {
    n5_encoder e_in(nullptr);
    randomize_state(e_in, rng, 8);
    
    n5_encoder e_slow = e_in;
    reference_coalesce(e_slow, 8, 32);

    n5_encoder e_fast = e_in;
    e_fast._coalesce_8_32();

    if (!states_equal_noisy(e_slow, e_fast, 32, "slow/fast coalesce_8_32")) {
        print_state(e_in, 8, "in");
        throw runtime_error("test_coalesce_8_32() failed");
    }
    }

    cout << "test_coalesce_8_32: passed, " << niter << " iterations" << endl;
}

static void test_coalesce_32_64(std::mt19937 &rng, int niter)
{
    for (int i = 0; i < niter; i++) {
    n5_encoder e_in(nullptr);;
    randomize_state(e_in, rng, 32);
    
    n5_encoder e_slow = e_in;
    reference_coalesce(e_slow, 32, 64);

    n5_encoder e_fast = e_in;
    e_fast._coalesce_32_64();

    if (!states_equal_noisy(e_slow, e_fast, 64, "slow/fast coalesce_32_64")) {
        print_state(e_in, 32, "in");
        throw runtime_error("test_coalesce_32_64() failed");
    }
    }

    cout << "test_coalesce_32_64: passed, " << niter << " iterations" << endl;
}


// -------------------------------------------------------------------------------------------------
//
// Unit test n5_encoder::_shuttle().


static void test_shuttle(std::mt19937 &rng, int niter)
{
    const int max_shuttle = 100;
    const int max_samples = 64 * max_shuttle;
    const int bufsize = n5_encoder::min_nbytes(max_samples);

    vector<int8_t> buf_slow(bufsize);
    vector<int8_t> buf_fast(bufsize);
    
    for (int iter = 0; iter < niter; iter++) {
    int nshuttle = randint(rng, 1, max_shuttle+1);

    // slow writer
    memset(&buf_slow[0], 0, bufsize);
    int nbits_slow = 0;

    // fast writer
    memset(&buf_fast[0], 0, bufsize);
    n5_encoder e(&buf_fast[0]);
    
    for (int ishuttle = 0; ishuttle < nshuttle; ishuttle++) {
        randomize_state(e, rng, 64);
        e._shuttle();
        
        m256i_wrapper state(e.state);
        m256i_wrapper npad(e.npad);
        
        for (int i = 0; i < 4; i++) {
        uint8_t nb = 64 - npad.get_bits(64*i, 8);
        
        for (int b = 0; b < nb; b++) {
            if (state.get_bit(64*i+b))
            buf_slow[nbits_slow/8] |= (1 << (nbits_slow % 8));
            nbits_slow++;
        }
        }
    }

    int nbits_fast = e.flush();
    assert(nbits_fast == nbits_slow);

    for (int i = 0; i < bufsize; i++)
        assert(buf_slow[i] == buf_fast[i]);
    }
    
    cout << "test_shuttle: passed, " << niter << " iterations" << endl;
}
    

// -------------------------------------------------------------------------------------------------
//
// End-to-end unit test. This one has comments!


// Convert floating-point output of spshuff::dequantize() to integer level 0-4.
inline int dequantization_level(float x)
{
    // Roundoff tolerance (not sure if this is actually necessary)
    const float eps = 1.0e-5;
    
    for (int level = 0; level < 5; level++)
    if (fabs(x - spshuff::dequant5[level]) < eps)
        return level;

    throw runtime_error("dequantization_level() failed");
}


static void test_end_to_end(std::mt19937 &rng, int niter)
{
    const float eps = 1.0e-5;
    
    for (int iter = 0; iter < niter; iter++) {
    int nsamples = 64 * randint(rng, 1, 100);
    
    uptr<float> data_normalized = make_uptr<float> (nsamples);
    uptr<float> data_unnormalized = make_uptr<float> (64);
    
    int compressed_nbytes = n5_encoder::min_nbytes(nsamples);
    vector<uint8_t> data_compressed(compressed_nbytes);

    float mean = uniform_rand(rng, -1.0, 1.0);
    float rms = uniform_rand(rng, 1.0, 2.0);
    n5_encoder e(&data_compressed[0], mean, rms);
    
    // We simulate input samples in batches of 64 (corresponding
    // to one call to n5_encoder::encode64_kernel()).
    
    for (int i = 0; i < nsamples; i += 64) {
        // Make sure to test cases where n5_encoder::set_mean_and_rms()
        // is either called, or not called, between calls to encode64_kernel().

        if (randint(rng,0,2)) {
        mean = uniform_rand(rng, -1.0, 1.0);
        rms = uniform_rand(rng, 1.0, 2.0);
        e.set_mean_and_rms(mean, rms);
        }
        
        for (int j = 0; j < 64; j++) {
        float t = uniform_rand(rng, -2.0, 2.0);
        data_normalized[i+j] = t;
        data_unnormalized[j] = t*rms + mean;
        }

        e.encode64_aligned(&data_unnormalized[0]);
    }

    // Don't forget to call n5_encoder::flush()!
    ssize_t nbits_compressed = e.flush();

    // At this point, 'data_compressed' contains the output of the n5_encoder, and
    // 'data_normalized' contains the input samples (after normalizing by mean and rms).
    //
    // As our end-to-end test, we call spshuff decode functions on 'data_compressed',
    // then verify consistency with 'data_normalized'.
    
    vector<uint8_t> data_decoded(nsamples);
    vector<float> data_dequantized(nsamples);
    
    spshuff::huff_decode_kernel(
        (uint32_t *) (&data_compressed[0]),  // in
        &data_decoded[0],                    // out
        (nbits_compressed + 31) / 32,        // n_in (32-bit)
        nsamples                             // nsamp
    );
    
    spshuff::dequantize(
        &data_decoded[0],      // levels
        &data_dequantized[0],  // out
        nsamples               // nsamp
    );

    for (int i = 0; i < nsamples; i++) {
        // Convert spshuff::dequantize output to integer "level" 0-4.
        int level = dequantization_level(data_dequantized[i]);

        // Verify consistency with data_normalized[i].
        if (level > 0)
        assert(data_normalized[i] >= spshuff::edges5[level-1] - eps);
        if (level < 4)
        assert(data_normalized[i] <= spshuff::edges5[level] + eps);
    }
    }

    cout << "test_end_to_end: passed, " << niter << " iterations" << endl;
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_cumsum64_restricted(rng, 10000);
    test_gather_predicate64(rng, 10000);
    test_broadcast_last64(rng, 10000);
    test_compute_syndrome(rng, 1000);
    test_permute(rng, 10000);
    test_huffman_compress(rng, 10000);
    test_coalesce_4_8(rng, 10000);
    test_coalesce_8_32(rng, 10000);
    test_coalesce_32_64(rng, 10000);
    test_shuttle(rng, 1000);
    test_end_to_end(rng, 1000);
    
    return 0;
}