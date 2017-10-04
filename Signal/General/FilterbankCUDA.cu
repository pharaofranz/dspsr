//-*-C++-*-

/***************************************************************************
 *
 *   Copyright(C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

//#define _DEBUG 1

#include "dsp/FilterbankCUDA.h"
#include "CUFFTError.h"
#include "debug.h"

#include <cuda_runtime.h>

#include <iostream>
#include <assert.h>

void check_error_stream(const char*, cudaStream_t);

#ifdef _DEBUG
#define CHECK_ERROR(x,y) check_error_stream(x,y)
#else
#define CHECK_ERROR(x,y)
#endif

#define EXEC_OR_THROW(cmd) \
result = cmd; if(result != CUFFT_SUCCESS) { throw CUFFTError(result, __func__, "cmd"); }

__global__ void k_multiply(float2* dFft, float2* kernel)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	float x = dFft[i].x * kernel[i].x - dFft[i].y * kernel[i].y;
	dFft[i].y = dFft[i].x * kernel[i].y + dFft[i].y * kernel[i].x;
	dFft[i].x = x;
}

__global__ void k_ncopy(float2* outputData, unsigned outputStride,
			const float2* inputData, unsigned inputStride,
			unsigned toCopy)
{
	outputData += blockIdx.y * outputStride;
	inputData += blockIdx.y * inputStride;
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < toCopy) {
		outputData[index] = inputData[index];
	}
}

using namespace std;

void FilterbankEngineCUDA::setup(dsp::Filterbank* filterbank)
{
	// the CUDA engine does not maintain/compute the passband
	filterbank->set_passband(NULL);
	//
	_frequencyResolution = filterbank->get_freq_res();
	_nChannelSubbands = filterbank->get_nchan_subband();
	_realToComplex = (filterbank->get_input()->get_state() == Signal::Nyquist);
	DEBUG("FilterbankEngineCUDA::setup _nChannelSubbands=" << _nChannelSubbands
		<< " _frequencyResolution=" << _frequencyResolution);
	DEBUG("FilterbankEngineCUDA::setup scratch=" << _scratch);
	cufftResult result;
	// setup forward plan
	unsigned planSize = _realToComplex ? _frequencyResolution*_nChannelSubbands*2 : _frequencyResolution*_nChannelSubbands;
	cufftType planType = _realToComplex ? CUFFT_R2C : CUFFT_C2C;
	EXEC_OR_THROW(cufftPlan1d(&_planForward, planSize, planType, 1))
	// setup stream
	DEBUG("FilterbankEngineCUDA::setup setting _stream=" << _stream);
	EXEC_OR_THROW(cufftSetStream(_planForward, _stream))
	DEBUG("FilterbankEngineCUDA::setup fwd FFT plan set");
	if(_frequencyResolution > 1) {
		EXEC_OR_THROW(cufftPlan1d(&_planBackward, _frequencyResolution, CUFFT_C2C, _nChannelSubbands))
		EXEC_OR_THROW(cufftSetStream(_planBackward, _stream))
		DEBUG("FilterbankEngineCUDA::setup bwd FFT plan set");
	}
	_nKeep = _frequencyResolution;
	_multiply.init();
	_multiply.set_nelement(_nChannelSubbands * _frequencyResolution);
	if(filterbank->has_response()) {
		const dsp::Response* response = filterbank->get_response();
		unsigned nChannels = response->get_nchan();
		unsigned nData = response->get_ndat();
		unsigned nDimensions = response->get_ndim();
		assert( nChannels == filterbank->get_nchan() );
		assert( nData == _frequencyResolution );
		assert( nDimensions == 2 ); // complex
		unsigned memSize = nChannels * nData * nDimensions * sizeof(cufftReal);	
		// allocate space for the convolution kernel
		cudaMalloc((void**)&_convolutionKernel, memSize);
		_nFilterPosition = response->get_impulse_pos();
		unsigned nFilterTotal = _nFilterPosition + response->get_impulse_neg();
		// points kept from each small fft
		_nKeep = _frequencyResolution - nFilterTotal;
		// copy the kernel accross
		const float* kernel = filterbank->get_response()->get_datptr(0,0);
		if(_stream) {
			cudaMemcpyAsync(_convolutionKernel, kernel, memSize, cudaMemcpyHostToDevice, _stream);
		} else {
			cudaMemcpy(_convolutionKernel, kernel, memSize, cudaMemcpyHostToDevice);
		}
	}
}

void FilterbankEngineCUDA::set_scratch(float* scratch)
{
	_scratch = scratch;
}

void FilterbankEngineCUDA::finish()
{
	check_error_stream("FilterbankEngineCUDA::finish", _stream);
}

void FilterbankEngineCUDA::_calculateDispatchDimensions(dim3& threads, dim3& blocks)
{
	threads.x = _multiply.get_nthread();
	blocks.x = _nKeep / threads.x;
	if(_nKeep % threads.x) { // ensure there's enough blocks to process all data
		blocks.x++;
	}
	blocks.y = _nChannelSubbands;
}

void FilterbankEngineCUDA::perform(	const dsp::TimeSeries * in, dsp::TimeSeries * out,
					uint64_t nParts, const uint64_t inStep, const uint64_t outStep)
{
	verbose = dsp::Operation::record_time || dsp::Operation::verbose;
	//
	const unsigned nPolarizations = in->get_npol();
	const unsigned nInputChannels = in->get_nchan();
	const unsigned nOutputChannels = out->get_nchan();
	DEBUG("FilterbankEngineCUDA::perform _stream=" << _stream);
	// GPU scratch space
	DEBUG("FilterbankEngineCUDA::perform scratch=" << _scratch);
	float2* cscratch = (float2*)_scratch;
	//
	cufftResult result;
	DEBUG("FilterbankEngineCUDA::perform nInputChannels=" << nInputChannels);
	DEBUG("FilterbankEngineCUDA::perform nPolarizations=" << nPolarizations);
	DEBUG("FilterbankEngineCUDA::perform nParts=" << nParts);
	DEBUG("FilterbankEngineCUDA::perform _nKeep=" << _nKeep);
	DEBUG("FilterbankEngineCUDA::perform inStep=" << inStep);
	DEBUG("FilterbankEngineCUDA::perform outStep=" << outStep);
	for(unsigned iInputChannel = 0; iInputChannel < nInputChannels; iInputChannel++) {
		for(unsigned iPolarization = 0; iPolarization < nPolarizations; iPolarization++) {
			for(unsigned iPart = 0; iPart < nParts; iPart++) {
				DEBUG("FilterbankEngineCUDA::perform iPart " << iPart << " of " << nParts);
				uint64_t inOffset = iPart * inStep;
				uint64_t outOffset = iPart * outStep;
				DEBUG("FilterbankEngineCUDA::perform offsets in=" << inOffset << " out=" << outOffset);
				float* inputPtr = const_cast<float *>(in->get_datptr(iInputChannel, iPolarization)) + inOffset;
				DEBUG("FilterbankEngineCUDA::perform FORWARD FFT inptr=" << inputPtr << " outptr=" << cscratch);
				if(_realToComplex) {
					EXEC_OR_THROW(cufftExecR2C(_planForward, inputPtr, cscratch))
				} else {
					float2* cin = (float2*)inputPtr;
					EXEC_OR_THROW(cufftExecC2C(_planForward, cin, cscratch, CUFFT_FORWARD))
				}
				if(_convolutionKernel) {
					// complex numbers offset(_convolutionKernel is float2*)
					unsigned offset = iInputChannel * _nChannelSubbands * _frequencyResolution;
					DEBUG("FilterbankEngineCUDA::perform _multiply dedipersion kernel _stream=" << _stream);
					k_multiply<<<_multiply.get_nblock(), _multiply.get_nthread(), 0, _stream>>>(cscratch, _convolutionKernel + offset);
					CHECK_ERROR("FilterbankEngineCUDA::perform _multiply", _stream);
				}
				if(_planBackward) {
					DEBUG("FilterbankEngineCUDA::perform BACKWARD FFT");
					EXEC_OR_THROW(cufftExecC2C(_planBackward, cscratch, cscratch, CUFFT_INVERSE))
				}
				if(out) {
					float* outputPtr = out->get_datptr(iInputChannel * _nChannelSubbands, iPolarization) + outOffset;
					uint64_t outputSpan = 	out->get_datptr(iInputChannel * _nChannelSubbands + 1, iPolarization)
								- out->get_datptr(iInputChannel * _nChannelSubbands, iPolarization);
					//
					const float2* input = cscratch + _nFilterPosition;
					unsigned inputStride = _frequencyResolution;
					unsigned toCopy = _nKeep;
					dim3 threads, blocks;
					_calculateDispatchDimensions(threads, blocks);
					// divide by two for complex data
					float2* outputBase = (float2*)outputPtr;
					unsigned outputStride = outputSpan / 2;
					DEBUG("FilterbankEngineCUDA::perform output base=" << outputBase << " stride=" << outputStride);
					DEBUG("FilterbankEngineCUDA::perform input base=" << input << " stride=" << inputStride);
					DEBUG("FilterbankEngineCUDA::perform to_copy=" << toCopy);
					k_ncopy<<<blocks, threads, 0, _stream>>>(outputBase, outputStride,
										 input, inputStride, toCopy);
					CHECK_ERROR("FilterbankEngineCUDA::perform ncopy", _stream);
				} // if not benchmarking
			} // for each part
		} // for each polarization
	} // for each channel
	if(verbose) {
		check_error_stream("FilterbankEngineCUDA::perform", _stream);
	}
}
