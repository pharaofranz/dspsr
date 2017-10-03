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

__global__ void k_multiply(float2* d_fft, float2* kernel)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	float x = d_fft[i].x * kernel[i].x - d_fft[i].y * kernel[i].y;
	d_fft[i].y = d_fft[i].x * kernel[i].y + d_fft[i].y * kernel[i].x;
	d_fft[i].x = x;
}

__global__ void k_ncopy(float2* output_data, unsigned output_stride,
			const float2* input_data, unsigned input_stride,
			unsigned to_copy)
{
	output_data += blockIdx.y * output_stride;
	input_data += blockIdx.y * input_stride;
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < to_copy) {
		output_data[index] = input_data[index];
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
	_realToComplex =(filterbank->get_input()->get_state() == Signal::Nyquist);
	DEBUG("FilterbankEngineCUDA::setup _nChannelSubbands=" << _nChannelSubbands
		<< " _frequencyResolution=" << _frequencyResolution);
	DEBUG("FilterbankEngineCUDA::setup scratch=" << scratch);
	cufftResult result;
	if(_realToComplex) {
		DEBUG("FilterbankEngineCUDA::setup plan size=" << _frequencyResolution*_nChannelSubbands*2);
		result = cufftPlan1d(&_planForward, _frequencyResolution*_nChannelSubbands*2, CUFFT_R2C, 1);
		if(result != CUFFT_SUCCESS) {
			throw CUFFTError(	result, "FilterbankEngineCUDA::setup",
						"cufftPlan1d(_planForward, CUFFT_R2C)");
		}
	} else {
		DEBUG("FilterbankEngineCUDA::setup plan size=" << _frequencyResolution*_nChannelSubbands);
		result = cufftPlan1d(&_planForward, _frequencyResolution*_nChannelSubbands, CUFFT_C2C, 1);
		if(result != CUFFT_SUCCESS) {
			throw CUFFTError(	result, "FilterbankEngineCUDA::setup",
						"cufftPlan1d(_planForward, CUFFT_C2C)");
		}
	}
	DEBUG("FilterbankEngineCUDA::setup setting _stream=" << _stream);
	result = cufftSetStream(_planForward, _stream);
	if(result != CUFFT_SUCCESS) {
		throw CUFFTError(	result, "FilterbankEngineCUDA::setup",
					"cufftSetStream(_planForward)");
	}
	DEBUG("FilterbankEngineCUDA::setup fwd FFT plan set");
	if(_frequencyResolution > 1) {
		result = cufftPlan1d(&_planBackward, _frequencyResolution, CUFFT_C2C, _nChannelSubbands);
		if(result != CUFFT_SUCCESS) {
			throw CUFFTError(	result, "FilterbankEngineCUDA::setup",
						 "cufftPlan1d(_planBackward)");
		}
		result = cufftSetStream(_planBackward, _stream);
		if(result != CUFFT_SUCCESS) {
			throw CUFFTError(	result, "FilterbankEngineCUDA::setup",
						"cufftSetStream(_planBackward)");
		}
		DEBUG("FilterbankEngineCUDA::setup bwd FFT plan set");
	}
	_nKeep = _frequencyResolution;
	_multiply.init();
	_multiply.set_nelement(_nChannelSubbands * _frequencyResolution);
	if(filterbank->has_response()) {
		const dsp::Response* response = filterbank->get_response();
		unsigned nchan = response->get_nchan();
		unsigned ndat = response->get_ndat();
		unsigned ndim = response->get_ndim();
		assert( nchan == filterbank->get_nchan() );
		assert( ndat == _frequencyResolution );
		assert( ndim == 2 ); // complex
		unsigned mem_size = nchan * ndat * ndim * sizeof(cufftReal);	
		// allocate space for the convolution kernel
		cudaMalloc((void**)&_convolutionKernel, mem_size);
		_nFilterPosition = response->get_impulse_pos();
		unsigned nfilt_tot = _nFilterPosition + response->get_impulse_neg();
		// points kept from each small fft
		_nKeep = _frequencyResolution - nfilt_tot;
		// copy the kernel accross
		const float* kernel = filterbank->get_response()->get_datptr(0,0);
		if(_stream) {
			cudaMemcpyAsync(_convolutionKernel, kernel, mem_size, cudaMemcpyHostToDevice, _stream);
		} else {
			cudaMemcpy(_convolutionKernel, kernel, mem_size, cudaMemcpyHostToDevice);
		}
	}
}

void FilterbankEngineCUDA::set_scratch(float * _scratch)
{
	scratch = _scratch;
}

void FilterbankEngineCUDA::finish()
{
	check_error_stream("FilterbankEngineCUDA::finish", _stream);
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
	DEBUG("FilterbankEngineCUDA::perform scratch=" << scratch);
	float2* cscratch = (float2*)scratch;
	//
	cufftResult result;
	//DEBUG("FilterbankEngineCUDA::perform input_nchan=" << input_nchan);
	//DEBUG("FilterbankEngineCUDA::perform npol=" << npol);
	//DEBUG("FilterbankEngineCUDA::perform npart=" << npart);
	//DEBUG("FilterbankEngineCUDA::perform _nKeep=" << _nKeep);
	//DEBUG("FilterbankEngineCUDA::perform in_step=" << in_step);
	//DEBUG("FilterbankEngineCUDA::perform out_step=" << out_step);
	for(unsigned iInputChannel = 0; iInputChannel < nInputChannels; iInputChannel++) {
		for(unsigned iPolarization = 0; iPolarization < nPolarizations; iPolarization++) {
			for(unsigned iPart = 0; iPart < nParts; iPart++) {
				//DEBUG("FilterbankEngineCUDA::perform ipart " << ipart << " of " << npart);
				uint64_t inOffset = iPart * inStep;
				uint64_t outOffset = iPart * outStep;
				//DEBUG("FilterbankEngineCUDA::perform offsets in=" << in_offset << " out=" << out_offset);
				float* inputPtr = const_cast<float *>(in->get_datptr(iInputChannel, iPolarization)) + inOffset;
				//DEBUG("FilterbankEngineCUDA::perform FORWARD FFT inptr=" << input_ptr << " outptr=" << cscratch);
				if(_realToComplex) {
					result = cufftExecR2C(_planForward, inputPtr, cscratch);
					if(result != CUFFT_SUCCESS) {
						throw CUFFTError(result, "FilterbankEngineCUDA::perform", "cufftExecR2C");
					}
					CHECK_ERROR("FilterbankEngineCUDA::perform cufftExecR2C FORWARD", _stream);
				} else {
					float2* cin = (float2*)inputPtr;
					result = cufftExecC2C(_planForward, cin, cscratch, CUFFT_FORWARD);
					if(result != CUFFT_SUCCESS) {
						throw CUFFTError(result, "FilterbankEngineCUDA::perform", "cufftExecC2C");
					}
					CHECK_ERROR("FilterbankEngineCUDA::perform cufftExecC2C FORWARD", _stream);
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
					result = cufftExecC2C(_planBackward, cscratch, cscratch, CUFFT_INVERSE);
					if(result != CUFFT_SUCCESS) {
						throw CUFFTError(result, "FilterbankEngineCUDA::perform", "cufftExecC2C(inverse)");
					}
					CHECK_ERROR("FilterbankEngineCUDA::perform cufftExecC2C BACKWARD", _stream);
				}
				if(out) {
					float* outputPtr = out->get_datptr(iInputChannel * _nChannelSubbands, iPolarization) + outOffset;
					uint64_t outputSpan = 	out->get_datptr(iInputChannel * _nChannelSubbands + 1, iPolarization)
								- out->get_datptr(iInputChannel * _nChannelSubbands, iPolarization);
					//
					const float2* input = cscratch + _nFilterPosition;
					unsigned inputStride = _frequencyResolution;
					unsigned toCopy = _nKeep;
					{
						dim3 threads;
						threads.x = _multiply.get_nthread();
						//
						dim3 blocks;
						blocks.x = _nKeep / threads.x;
						if(_nKeep % threads.x) {
							blocks.x++;
						}
						blocks.y = _nChannelSubbands;
						// divide by two for complex data
						float2* outputBase = (float2*)outputPtr;
						unsigned outputStride = outputSpan / 2;
						DEBUG("FilterbankEngineCUDA::perform output base=" << outputBase << " stride=" << outputStride);
						DEBUG("FilterbankEngineCUDA::perform input base=" << input << " stride=" << inputStride);
						DEBUG("FilterbankEngineCUDA::perform to_copy=" << toCopy);
						k_ncopy<<<blocks, threads, 0, _stream>>>(outputBase, outputStride,
											 input, inputStride, toCopy);
						CHECK_ERROR("FilterbankEngineCUDA::perform ncopy", _stream);
					}
				} // if not benchmarking
			} // for each part
		} // for each polarization
	} // for each channel
	if(verbose) {
		check_error_stream("FilterbankEngineCUDA::perform", _stream);
	}
}
