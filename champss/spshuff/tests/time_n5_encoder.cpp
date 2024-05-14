#include "spshuff.hpp"
#include <string>
#include <iostream>
#include "n5_encoder_testing.hpp"


using namespace std;
using namespace spshuff;


static void do_timing(ssize_t nouter, ssize_t ninner, const string &label, ssize_t nruns=4)
{
    cout << "Timing with nouter=" << nouter << ", ninner=" << ninner << " (" << label << ")" << endl;
	
    ssize_t nout = n5_encoder::min_nbytes(nouter * ninner);

    uptr<float> in = make_uptr<float> (ninner);
    vector<uint8_t> out(nout);

    for (int i = 0; i < nruns; i++) {
	// Specify nonzero mean, to prevent compiler from optimizing out the mean subtraction.
	n5_encoder enc(&out[0], 1.0);
	auto t0 = get_time();
	
	for (int j = 0; j < nouter; j++)
	    for (ssize_t k = 0; k < ninner; k += 64)
		enc.encode64_aligned(&in[k]);
	
	auto t1 = get_time();
	auto dt = time_diff(t0,t1);
	
	cout << "    Timing run " << i << "/" << nruns << ": "
	     << (nouter*ninner/1.0e6/dt) << " Msamp/sec" << endl;
    }
}    


int main(int argc, char **argv)
{
    do_timing(2048 * 1024, 512, "L1 cache");
    do_timing(32 * 1024, 32 * 1024, "L2 cache");
    do_timing(512, 2 * 1024 * 1024, "L3 cache");
    do_timing(2, 512 * 1024 * 1024, "DRAM");
    
    return 0;
}