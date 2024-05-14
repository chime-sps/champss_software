#include "spshuff.hpp"
#include <random>
#include <chrono>
#include <memory>
#include <iostream>

using namespace spshuff;

int main()
{
    const ssize_t chunk_size = 1024;
    const ssize_t nchunk = 1024;
    const ssize_t nquant = chunk_size * nchunk;

    // four-byte aligned
    // std::vector<float> in(nquant);
    std::shared_ptr<float> in((float*) aligned_alloc(32, nquant * 4), free);

    // generate a random normal stream
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0,1};
    for(ssize_t i = 0; i < nquant; i++){
        (&(*in))[i] = d(gen);
    }

    float* in_ptr = &(*(in));

    std::vector<uint8_t> quantized(nquant);


    auto t0 = std::chrono::high_resolution_clock::now();
    // perform the quantization and compression in the normal "lump" sense
    quantize_naive5(in_ptr, &(quantized[0]), nquant);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Compute an upper bound on bit size of encoded huffman data
    const ssize_t max_dst_size = encode_bound(nquant);

    // number of uint32_t in the compressed data (fairly arbitrary choice of dtype)
    const ssize_t len_compressed = (max_dst_size/sizeof(uint32_t)) + 1;
    std::vector<uint32_t> compressed_lump(len_compressed);
    std::vector<uint32_t> compressed_chunked(len_compressed);

    // we need to set these values here, but we don't worry about their post-call state in this function
    ssize_t i0 = 0;
    ssize_t bit0 = 0;
    uint32_t tmp = 0;
    const ssize_t compressed_data_len = huff_encode_kernel(&(quantized[0]), 
                                            &(compressed_lump[0]), nquant, i0, bit0, tmp);
    const ssize_t compressed_data_size = compressed_data_len * sizeof(uint32_t);


    auto t2 = std::chrono::high_resolution_clock::now();
    i0 = 0;
    bit0 = 0;
    tmp = 0;
    for(ssize_t i = 0; i < nchunk; i++){
        huff_encode_kernel(&(quantized[i * chunk_size]), &(compressed_chunked[0]), chunk_size, i0, bit0, tmp);
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    // check agreement
    for(ssize_t i = 0; i < compressed_data_len; i++){
        if(compressed_chunked[i] != compressed_lump[i]){ 
            std::cout << "FAIL" << std::endl;
            std::cout << compressed_chunked[i] << " " << compressed_lump[i] << std::endl;
            std::cout << "\t" << i << " " << i * (32/2.32) << std::endl;
            return 1;
        }
    }
    
    std::chrono::duration<double> tquant = t1 - t0;
    std::chrono::duration<double> tchunk = t3 - t2;

    std::cout << "quantize rate " << (1e-6 * nquant) / tquant.count() << " Msamp/s " << std::endl;
    std::cout << "compress rate " << (1e-6 * nquant) / tchunk.count() << " Msamp/s " << std::endl;
    std::cout << "PASS" << std::endl;
    return 0;
}