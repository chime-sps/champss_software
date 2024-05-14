#include <memory>
#include <random>
#include <cstring>
#include <cassert>
#include <sstream>
#include <iostream>
#include <sys/time.h>


// -------------------------------------------------------------------------------------------------
//
// Timing utils


inline struct timeval get_time()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t;
}

inline double time_diff(const struct timeval &t0, const struct timeval &t1)
{
    return (t1.tv_sec - t0.tv_sec) + 1.0e-6 * (t1.tv_usec - t0.tv_usec);
}


// -------------------------------------------------------------------------------------------------
//
// RNG utils


inline ssize_t randint(std::mt19937 &rng, ssize_t lo, ssize_t hi)
{
    assert(lo < hi);
    return std::uniform_int_distribution<>(lo,hi-1)(rng);   // note hi-1 here!
}


inline float uniform_rand(std::mt19937 &rng, float lo=0.0, float hi=1.0)
{
    return std::uniform_real_distribution<float>(lo,hi)(rng);
}


inline std::vector<float> uniform_randvec(std::mt19937 &rng, ssize_t n, float lo=0.0, float hi=1.0)
{
    std::vector<float> ret(n);
    std::uniform_real_distribution<float> d(lo, hi);
    
    for (ssize_t i = 0; i < n; i++)
	ret[i] = d(rng);

    return ret;
}


// -------------------------------------------------------------------------------------------------
//
// Aligned allocators


// We align to 128 bytes by default (size of an L3 cache line "pair")
template<typename T>
inline T *aligned_alloc(ssize_t nelts, ssize_t nalign=128, bool zero=true)
{
    if (nelts < 0)
	throw std::runtime_error("expected nelts >= 0");
    if (nalign <= 0)
	throw std::runtime_error("expected nalign > 0");
    if (nelts == 0)
	return NULL;

    void *p = NULL;
    if (posix_memalign(&p, nalign, nelts * sizeof(T)) != 0)
	throw std::runtime_error("couldn't allocate memory");

    if (zero)
	memset(p, 0, nelts * sizeof(T));

    return reinterpret_cast<T *> (p);
}

struct uptr_deleter {
    inline void operator()(const void *p) { free(const_cast<void *> (p)); }
};

template<typename T>
using uptr = std::unique_ptr<T[], uptr_deleter>;

// Usage: uptr<float> p = make_uptr<float> (nelts);
template<typename T>
inline uptr<T> make_uptr(size_t nelts, size_t nalign=128, bool zero=true)
{
    T *p = aligned_alloc<T> (nelts, nalign, zero);
    return uptr<T> (p);
}