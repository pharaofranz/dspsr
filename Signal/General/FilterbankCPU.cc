//-*-C++-*-

#include "dsp/FilterbankCPU.hpp"

#include <fstream>
#include <cassert>
#include <cstring>

#if 0
#define TESTING_LOG(s) std::cerr << s << std::endl
#define TESTING_LOG_LINE std::cerr << __LINE__ << ":" << __FUNCTION__ << std::endl
#else 
#define TESTING_LOG(s)
#define TESTING_LOG_LINE
#endif

FilterbankEngineCPU::FilterbankEngineCPU()
{
	//
}

FilterbankEngineCPU::~FilterbankEngineCPU()
{
	//
}

/**
 * Initialization function to setup FilterbankEngineCPU for use.
 * 
 * @param filterbank pointer to filterbank class that is setting up the engine for usage.
 */
void FilterbankEngineCPU::setup(dsp::Filterbank* filterbank)
{
	TESTING_LOG("FilterbankEngineCPU::setup()");
	// passband is unused by engine
	filterbank->set_passband(NULL);
	//
	_frequencyResolution = filterbank->get_freq_res();
	_nChannelSubbands = filterbank->get_nchan_subband();
	_isRealToComplex = (filterbank->get_input()->get_state() == Signal::Nyquist);
	//! Number of points in forwards FFT
	_nFftPoints = _nChannelSubbands*_frequencyResolution;
	//! Number of samples in forwards FFT
	const unsigned nSamplesForward = (_frequencyResolution*_nChannelSubbands) * (_isRealToComplex ? 2 : 1);
	//! Number of samples in Backwards FFT
	const unsigned nSamplesBackward = _frequencyResolution;
	//
	const FTransform::type transformType = (_isRealToComplex ? FTransform::frc : FTransform::fcc);
	_forward = FTransform::Agent::current->get_plan(nSamplesForward, transformType);
	_backward = FTransform::Agent::current->get_plan(nSamplesBackward, FTransform::bcc);
	_nPointsToKeep = _frequencyResolution;
	_nFilterPosition = 0;
	//
	if(filterbank->has_response()) {
		_response = filterbank->get_response();
		unsigned nChannels = _response->get_nchan();
		unsigned nData = _response->get_ndat();
		unsigned nDimensions = _response->get_ndim();
		//
		assert(nChannels == filterbank->get_nchan());
		assert(nData == _frequencyResolution);
		assert(nDimensions == 2);
		//! Complex samples dropped from beginning of cyclical convolution result
		unsigned nFilterPositive = _response->get_impulse_pos();
		_nFilterPosition = nFilterPositive;
		//! Complex samples dropped from end of cyclical convolution result
		unsigned nFilterNegative = _response->get_impulse_neg();
		//! Total number of samples in convolution filter kernel
		unsigned nFilterTotal = nFilterPositive + nFilterNegative;
		//! Number of FFT points to keep from each backwards FFT
		_nPointsToKeep = _frequencyResolution - nFilterTotal;
	}
	const bool isNormalized = FTransform::get_norm() == FTransform::normalized;
	const double scaleFactorNormalized = double(_nFftPoints)/double(_frequencyResolution);
	const double scaleFactorUnnormalized = double(_nFftPoints)*double(_frequencyResolution);
	_scaleFactor = isNormalized ? scaleFactorNormalized : scaleFactorUnnormalized;
	_nFftSamples = _isRealToComplex ? 2*_nFftPoints : _nFftPoints;
	_nSampleStep = _nFftSamples - _nSampleOverlap;
	//
	TESTING_LOG("_frequencyResolution=" << _frequencyResolution << " _nChannelSubbands=" << _nChannelSubbands << " _nPointsToKeep=" << _nPointsToKeep << " _scaleFactor=" << _scaleFactor
		<< " _nFftSamples=" << _nFftSamples << " _nSampleStep=" << _nSampleStep << std::endl);
}

/**
 * Performs the Convolving De-Dispersion filtering.
 * 
 * @param in[in] input time series data
 * @param out[out] output time series data
 * @param nParts number of parts the forward FFT is broken down into
 * @param inStep how far to step forward through the input data for each FFT part
 * @param outStep how far to step forward through the output data for each FFT part
 */
void FilterbankEngineCPU::perform(const dsp::TimeSeries* in, dsp::TimeSeries* out,uint64_t nParts, uint64_t inStep, uint64_t outStep)
{
	TESTING_LOG("FilterbankEngineCPU::perform()");
	const uint64_t nPolarizations = in->get_npol();
	const uint64_t nInputChannels = in->get_nchan();
	const uint64_t nOutputChannels = out->get_nchan();
	TESTING_LOG("nPolarizations=" << nPolarizations << " nInputChannels=" << nInputChannels << " nParts=" << nParts);
	for(uint64_t iInputChannel = 0; iInputChannel < nInputChannels; iInputChannel++) {
		for(uint64_t iPart = 0; iPart < nParts; iPart++) {
			const uint64_t inOffset = iPart*inStep;
			const uint64_t outOffset = iPart*outStep;
			for(uint64_t iPolarization = 0; iPolarization < nPolarizations; iPolarization++) {
				float* timeDomainInputPtr = const_cast<float*>(in->get_datptr(iInputChannel, iPolarization)) + inOffset;
				float* frequencyDomainPtr = _complexSpectrum[iPolarization];
				// perform forward FFT ot convert time domain data to the frequency domain
				if(_isRealToComplex) {
					_forward->frc1d(_nFftSamples, frequencyDomainPtr, timeDomainInputPtr);
				} else {
					_forward->fcc1d(_nFftSamples, frequencyDomainPtr, timeDomainInputPtr);
				}
				// apply filter response if available
				if(_response) {
					_response->operate(	_complexSpectrum[iPolarization], 
								iPolarization,
								iInputChannel*_nChannelSubbands,
								_nChannelSubbands);
				}
				// output data if output is available
				if(out) {
					for(uint64_t iSubband = 0; iSubband < _nChannelSubbands; iSubband++) {
						// perform a backwards FFT for each sub-band to convert frequency domain
						// data back into the time domain for output
						float* subbandPtr = frequencyDomainPtr + (iSubband*(_frequencyResolution*2));
						_backward->bcc1d(_frequencyResolution, _complexTime, subbandPtr);
						// Copy output data to output
						void* destinationPtr = out->get_datptr((iInputChannel*_nChannelSubbands)+iSubband, iPolarization)+outOffset;
						void* sourcePtr = _complexTime + _nFilterPosition*2;
						memcpy(destinationPtr, sourcePtr, _nPointsToKeep*sizeof(float)*2);
					}
				}
			}
		}
	}
}

void FilterbankEngineCPU::finish()
{
	//
}

/**
 * Setup scratch space for performing FFT calculations
 * 
 * @param scratch pointer to memory to use for scratch
 */
void FilterbankEngineCPU::set_scratch(float* scratch)
{
	TESTING_LOG("FilterbankEngineCPU::set_scratch(" << scratch << ")");
	unsigned bigFftSize = _nChannelSubbands * _frequencyResolution * 2;
	if(_isRealToComplex) {
		bigFftSize += 256;
	}
	_scratch = scratch;
	_complexSpectrum[0] = scratch;
	_complexSpectrum[1] = _complexSpectrum[0];
	_complexTime = _complexSpectrum[1] + bigFftSize;
	_windowedTimeDomain = _complexTime + 2 * _frequencyResolution;
}
