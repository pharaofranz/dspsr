// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// A sample program demonstrating using Google C++ testing framework.
//
// Author: wan@google.com (Zhanyong Wan)


// This sample shows how to write a simple unit test for a function,
// using Google C++ testing framework.
//
// Writing a unit test using Google C++ testing framework is easy as 1-2-3:


// Step 1. Include necessary header files such that the stuff your
// test logic needs is declared.
//
// Don't forget gtest.h, which declares the testing framework.

#include <limits.h>

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/FilterbankCPU.h"
#include "dsp/Filterbank.h"
#include "dsp/FilterbankConfig.h"
#include "dsp/Memory.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include <cuda_runtime.h>
#endif

#include "gtest/gtest.h"

using namespace std;

namespace {

	ostream cerrStream(NULL);

	ostream& isVerbose(ostream &stream)
	{
		return (dsp::Filterbank::verbose) ? cerr : stream;
	}

	TEST(FilterbankSetGetChan, Positive) {

		dsp::Operation::verbose = false;
		dsp::Filterbank::Config filterbankConfig;
		dsp::Filterbank* filterbank = filterbankConfig.create();

		// given
		unsigned nchan = 128;

		// when
		filterbank->set_nchan(nchan);

		// then
		// ASSERT_* : Fatal Assertion
		ASSERT_EQ(filterbank->get_nchan(), nchan);
		cerrStream << isVerbose << filterbank->get_nchan() << std::endl;
	}

	TEST(FilterbankSetGetChan, Negative) {

		dsp::Operation::verbose = false;
		dsp::Filterbank::Config filterbankConfig;
		dsp::Filterbank* filterbank = filterbankConfig.create();

		// given
		unsigned nchan = -128;

		// when
		filterbank->set_nchan(nchan);

		// then
		ASSERT_EQ(filterbank->get_nchan(), nchan);
		cerrStream << isVerbose << nchan << " vs "<< filterbank->get_nchan() << std::endl;
	}

	TEST(FilterbankSetGetFrequencyResolution, Positive) {

		dsp::Operation::verbose = false;
		dsp::Filterbank::Config filterbankConfig;
		dsp::Filterbank* filterbank = filterbankConfig.create();

		// given
		unsigned frequencyResolution = 1024;

		// when
		filterbank->set_freq_res(frequencyResolution);

		// then
		// ASSERT_* : Fatal Assertion
		ASSERT_EQ(filterbank->get_freq_res(), frequencyResolution);
		cerrStream << isVerbose << frequencyResolution << " vs "<< filterbank->get_freq_res() << std::endl;
	}

	TEST(FilterbankSetGetFrequencyResolution, Negative) {

		dsp::Operation::verbose = false;
		dsp::Filterbank::Config filterbankConfig;
		dsp::Filterbank filterbank; // = filterbankConfig.create();

		// given
		unsigned frequencyResolution = -1024;

		// when
		filterbank.set_freq_res(frequencyResolution);

		// then
		ASSERT_EQ(filterbank.get_freq_res(), frequencyResolution);
		cerrStream << isVerbose << frequencyResolution << " vs "<< filterbank.get_freq_res() << std::endl;
	}

	TEST(FilterbankTransformation, WholeProcessing) {

		dsp::Operation::verbose = true;


		dsp::Filterbank::Config config;
		unsigned nloop;
		unsigned niter;
		unsigned gpu_id;
		bool real_to_complex;
		bool do_fwd_fft;
		bool cuda;

		// given

		gpu_id = 0;
		niter = 10;
		nloop = 0;
		real_to_complex = false;
		do_fwd_fft = true;
		cuda = false;

		config.set_freq_res( 1024 );

		unsigned nfloat = config.get_nchan() * config.get_freq_res();
		if (!real_to_complex)
			nfloat *= 2;

		unsigned size = sizeof(float) * nfloat;

		if (!nloop)
		{
			nloop = (1024) / size;
			if (nloop > 2000)
				nloop = 2000;
		}

		dsp::Memory* memory = 0;

#if HAVE_CUFFT
		cerr << "using GPU " << gpu_id << endl;
		cudaSetDevice(gpu_id); 

		cudaStream_t stream = 0;
		if (cuda)
		{
			cudaError_t err = cudaSetDevice (0);
			if (err != cudaSuccess)
				throw Error (InvalidState, "filterbank_speed",
						"cudaSetDevice failed: %s", cudaGetErrorString(err));

			err = cudaStreamCreate( &stream );
			if (err != cudaSuccess)
				throw Error (InvalidState, "filterbank_speed",
						"cudaStreamCreate failed: %s", cudaGetErrorString(err));

			memory = new CUDA::DeviceMemory(stream);

			cerr << "run on GPU" << endl;
			config.set_device( memory );
			config.set_stream( stream );
		}
		else
			memory = new dsp::Memory;
#else
		memory = new dsp::Memory;
#endif

		dsp::Filterbank* filterbank = config.create();
		filterbank->isSimulation = true;

		dsp::TimeSeries input;
		filterbank->set_input( &input );

		input.set_rate( 1e6 );
		input.set_state( Signal::Analytic );
		input.set_ndim( 2 );
		input.set_input_sample( 0 );
		input.set_memory ( memory );

		input.resize( size );
		input.zero();

		dsp::TimeSeries output;
		output.set_memory ( memory );

		filterbank->set_output( &output );

		filterbank->prepare();

		RealTimer timer;
		timer.start ();

		uint64_t nInputSize = input.internal_get_size();
		uint64_t nOutputSize = output.internal_get_size();

		cerr << "nInputSize vs nOutputSize: " << nInputSize << " vs " << nOutputSize 
			<< " size : nfloat = " << size << " : " << nfloat << endl; 

		nInputSize = input.get_ndat();
		nOutputSize = output.get_ndat();

		cerr << "nInputSize vs nOutputSize: " << nInputSize << " vs " << nOutputSize 
			<< " size : nfloat = " << size << " : " << nfloat << endl; 

		float* pInput = const_cast<float*>(input.get_datptr());
		
		for(auto i=0; i<nInputSize/4; i++)
		{
			*(pInput+i) = i;
			cerr<< *(pInput+i) << " "; 
		}

		cerr<< endl;

		filterbank->transformation();
  		//for (unsigned i=0; i<nloop; i++)
      	//	filterbank->operate();
		float* pOutput = const_cast<float*>(output.get_datptr());
		
		for(auto i=0; i<nOutputSize/4; i++)
		{
			cerr<< *(pOutput+i) << " "; 
		}

		cerr<< endl;


#if HAVE_CUFFT
		//check_error_stream ("CUDA::FilterbankEngine::finish", stream);
#endif

		timer.stop ();

		double total_time = timer.get_elapsed();

		double time_us = total_time * 1e6 / (nloop*niter);

		unsigned nfft = config.get_freq_res();
		unsigned nchan = config.get_nchan();

		double log2_nfft = log2(nfft);
		double log2_nchan = log2(nchan);

		double bwd = 2;
		if (nchan == 1)
			bwd = 1;

		double mflops = 5.0 * nfft * nchan * (bwd*log2_nfft + log2_nchan) / time_us;

		cerr << "nchan=" << nchan << " nfft=" << nfft << " time=" << time_us << "us"
			" log2(nfft)=" << log2_nfft << " log2(nchan)=" << log2_nchan << 
			" mflops=" << mflops << endl;

		cout << nchan << " " << nfft << " " << time_us << " "
			<< log2_nchan << " " << log2_nfft << " " << mflops << endl;


		// when

		// then
	}

}  // namespace

