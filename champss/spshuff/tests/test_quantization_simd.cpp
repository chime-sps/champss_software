#include "spshuff.hpp"
#include <random>
#include <chrono>
#include <memory>
#include <iostream>
#include <cmath>

using namespace spshuff;

bool feq(const float a, const float b, const float eps=1e-6){
	const float asum = abs(a) + abs(b);
	if(asum > eps){
		return 2 * abs(a - b)/asum > eps? false : true;
	}
	else{
		return abs(a - b) > eps? false : true;
	}
}

int main(){
	const ssize_t nsamp = 16 * 1024 * 1024;
	const ssize_t ntrial = 128;

	// four-byte aligned
	std::shared_ptr<float> in((float*) aligned_alloc(32, nsamp * 4), free);

	// generate a random normal stream
	std::random_device rd{};
	std::mt19937 gen{rd()};
	std::normal_distribution<> d{0,1};
	for(ssize_t i = 0; i < nsamp; i++){
		(&(*in))[i] = d(gen);
	}

	float* in_ptr = &(*(in));

	std::vector<uint8_t> quantized_ref(nsamp);
	std::vector<uint8_t> quantized_simd(nsamp);
	// std::vector<uint8_t> quantized_simd2(nsamp);


	std::shared_ptr<uint8_t> quantized_simd4_alloc((uint8_t*) aligned_alloc(32, nsamp), free);
	uint8_t* quantized_simd4 = &(*quantized_simd4_alloc);

	auto t0 = std::chrono::high_resolution_clock::now();
	// for(ssize_t i = 0; i < ntrial; i++){
	// 	quantize_naive5_reference(in_ptr, &(quantized_ref[0]), nsamp);
	// }
	auto t1 = std::chrono::high_resolution_clock::now();
	// for(ssize_t i = 0; i < ntrial; i++){
	// 	quantize_naive5(in_ptr, &(quantized_simd[0]), nsamp);
	// }
	auto t2 = std::chrono::high_resolution_clock::now();
	for(ssize_t i = 0; i < ntrial; i++){
		// quantize_naive5_simd4(in_ptr, &(quantized_simd2[0]), nsamp);
		quantize_naive5_simd4(in_ptr, &(quantized_simd4[0]), nsamp);
	}
	auto t3 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> tquant_ref = t1 - t0;
	std::chrono::duration<double> tquant_simd = t2 - t1;
	std::chrono::duration<double> tquant_simd4 = t3 - t2;

	std::cout << "reference rate: " << nsamp * 1e-6 * ntrial / tquant_ref.count() << " Msamp/s" << std::endl;
	std::cout << "simd rate: " << nsamp * 1e-6 * ntrial / tquant_simd.count() << " Msamp/s" << std::endl;
	std::cout << "simd4 rate: " << nsamp * 1e-6 * ntrial / tquant_simd4.count() << " Msamp/s" << std::endl;
	std::cout << "reference 16k,1k time " << tquant_ref.count() / ntrial << " (s)" << std::endl;
	std::cout << "simd 16k,1k time " << tquant_simd.count() / ntrial << " (s)" << std::endl;
	std::cout << "simd4 16k,1k time " << tquant_simd4.count() / ntrial << " (s)" << std::endl;

	for(ssize_t i = 0; i < nsamp; i++){
		if( (quantized_ref[i] != quantized_simd[i]) || (quantized_ref[i] != quantized_simd4[i])){
			// do complicated logic to hide disagreements due to rounding
			// at a bin edge
			const float in_fail = in_ptr[i];
			bool fail = true;
			for(ssize_t j = 0; j < 4; j++){
				if(feq(in_fail, edges5[j])){
					fail = false;
				}
			}
			if(fail){
				std::cout << "disagreement: i, ref, simd, simd4" << std::endl;
				std::cout << i << " " << in_fail << " " << ((uint32_t) quantized_ref[i]) << " " << ((uint32_t) quantized_simd[i]) << " " << ((uint32_t) quantized_simd4[i]) << std::endl;
				std::cout << "FAIL" << std::endl;
				return 1;
			}
		}
	}

	std::cout << "PASS" << std::endl;
	return 0;
}