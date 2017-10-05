//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/FilterbankCUDA.h

#ifndef __FilterbankCUDA_h
#define __FilterbankCUDA_h

#include "dsp/FilterbankEngine.h"
#include "dsp/LaunchConfig.h"

#include <cufft.h>

class elapsed
{
public:
	elapsed();
	void wrt(cudaEvent_t before);
	double total;
	cudaEvent_t after;
};

//! Discrete convolution filterbank step implemented using CUDA streams
class FilterbankEngineCUDA : public dsp::Filterbank::Engine
{
	unsigned nstream;
public:
	//! Default Constructor
	FilterbankEngineCUDA(cudaStream_t stream)
		: _planForward(0), _planBackward(0), _realToComplex(false), _dFft(0),
		_convolutionKernel(0), _nFilterPosition(0), _stream(stream) {}
	~FilterbankEngineCUDA() {}
	void setup(dsp::Filterbank*);
	void set_scratch(float*);
	void perform(	const dsp::TimeSeries* in, dsp::TimeSeries* out,
			uint64_t npart, uint64_t in_step, uint64_t out_step);
	void finish();
protected:
	bool verbose;
private:
	void _calculateDispatchDimensionsForCopy(dim3& threads, dim3& blocks);
	//! forward fft plan
	cufftHandle _planForward;
	//! backward fft plan
	cufftHandle _planBackward;
	//! Complex-valued data
	bool _realToComplex;
	//! inplace FFT in CUDA memory
	float2* _dFft;
	//! convolution kernel in CUDA memory
	float2* _convolutionKernel;
	//! device scratch sapce
	float* _scratch;
	//
	unsigned _nChannelSubbands;
	unsigned _frequencyResolution;
	unsigned _nFilterPosition;
	unsigned _nKeep;
	//
	CUDA::LaunchConfig1D _multiply;
	//
	cudaStream_t _stream;
};

#endif
