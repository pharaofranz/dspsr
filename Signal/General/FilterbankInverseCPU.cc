//-*-C++-*-

#include "dsp/FilterbankInverseCPU.h"

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

using namespace std;
/**
 * Initialization function to setup FilterbankInverseEngineCPU for use.
 * 
 * @param filterbank pointer to filterbank class that is setting up the engine for usage.
 */
void FilterbankInverseEngineCPU::setup(dsp::Filterbank* filterbank)
{
	TESTING_LOG("FilterbankInverseEngineCPU::setup()");
	// passband is unused by engine
	filterbank->set_passband(NULL);
	//
	_frequencyResolution = filterbank->get_freq_res();

	// _nChannelSubbands should be calculated after considering input channels and output channels
	_nChannelSubbands = 1; //filterbank->get_nchan_subband();
	_nInputChannels = filterbank->get_nInputChannel();
	_nPolarization = filterbank->get_nPolarization(); 

	_isRealToComplex = (filterbank->get_input()->get_state() == Signal::Nyquist);

	//! Number of points in forwards FFT
	// for Stretch goal
	_nFftPoints = _frequencyResolution;

	//! Number of samples in forwards FFT
	const unsigned nSamplesForward = _nFftPoints * (_isRealToComplex ? 2 : 1);

	//! Number of samples in Backwards FFT
	const unsigned nSamplesBackward = _frequencyResolution;
	//
	const FTransform::type transformType = (_isRealToComplex ? FTransform::frc : FTransform::fcc);
	_forward = FTransform::Agent::current->get_plan(nSamplesForward, transformType);
	_backward = FTransform::Agent::current->get_plan(nSamplesBackward, FTransform::bcc);
	_nPointsToKeep = _frequencyResolution;
	_nFilterPosition = 0;
	//
	if(filterbank->has_response()==true) {
		_response = filterbank->get_response();
		unsigned nChannels = _response->get_nchan();
		unsigned nData = _response->get_ndat();
		unsigned nDimensions = _response->get_ndim();
		//
		assert(nChannels == filterbank->get_nchan());
		assert(nData == _frequencyResolution);
		assert(nDimensions == 2);
		//! Complex samples discarded from beginning of cyclical convolution result
		unsigned nFilterPositive = _response->get_impulse_pos();
		_nFilterPosition = nFilterPositive;
		//! Complex samples discarded from end of cyclical convolution result
		unsigned nFilterNegative = _response->get_impulse_neg();
		//! Total number of samples in convolution filter kernel
		unsigned nFilterTotal = nFilterPositive + nFilterNegative;
	}
	const bool isNormalized = FTransform::get_norm() == FTransform::normalized;
	const double scaleFactorNormalized = double(_nFftPoints)/double(_frequencyResolution);
	const double scaleFactorUnnormalized = double(_nFftPoints)*double(_frequencyResolution);
	_scaleFactor = isNormalized ? scaleFactorNormalized : scaleFactorUnnormalized;
	_nFftSamples = nSamplesForward;
	_nSampleStep = _nFftSamples - _nSampleOverlap;
	//
}

void FilterbankInverseEngineCPU::perform(const dsp::TimeSeries* in, dsp::TimeSeries* out,uint64_t nParts, 
		uint64_t inStep, uint64_t outStep)
{
	TESTING_LOG("FilterbankInverseEngineCPU::perform()");
	const uint64_t nPolarizations = in->get_npol();
	const uint64_t nInputChannels = in->get_nchan();
	const uint64_t nOutputChannels = nInputChannels / nInputChannels;

	static const int floatsPerComplex = 2;
	static const int sizeOfComplexData = sizeof(float) * floatsPerComplex;

	// critically sampled number of channels retained from each fwd FFT
	int stitchOffset = in->get_oversampling_factor().normalize(_nFftSamples);
	assert ( (_nFftSamples - stitchOffset) % 2 == 0 );
	// discard half of the oversampled channels from each side -> / 2
	int numOffsetForDiscard = (_nFftSamples - stitchOffset) / 2;

	int numDataForCopy = stitchOffset * sizeOfComplexData;  

	// temp data for check loop count
	const int numTotalBytesToCopy = numDataForCopy * nInputChannels;
	const int nPointsToBwdFFT = stitchOffset * nInputChannels;

	_nPointsToKeep = nPointsToBwdFFT
		- (_response->get_impulse_pos() + _response->get_impulse_neg());

	// prepare memory space to stitching data
	dsp::Scratch* stitchedSpace;
	float* spaceForStitchingFft;

	stitchedSpace = new dsp::Scratch;
	spaceForStitchingFft = stitchedSpace->space<float>
		(nInputChannels * stitchOffset );

	for(auto iOutputChannels = 0; iOutputChannels < nOutputChannels; iOutputChannels++) {

		for(uint64_t iPart = 0; iPart < nParts; iPart++) {

			const uint64_t inOffset = iPart*inStep;

			for(uint64_t iPolarization = 0; iPolarization < nPolarizations; iPolarization++) {

				for(uint64_t iInputChannel = 0; iInputChannel < nInputChannels; iInputChannel++) {

					float* frequencyDomainPtr = _complexSpectrum[0];
					float* timeDomainInputPtr = const_cast<float*>
						(in->get_datptr(iInputChannel, iPolarization)) + inOffset;
					// perform forward FFT ot convert time domain data to the frequency domain
					if(_isRealToComplex==true) {
						_forward->frc1d(_nFftSamples, frequencyDomainPtr, timeDomainInputPtr);
					} else {
						_forward->fcc1d(_nFftSamples, frequencyDomainPtr, timeDomainInputPtr);
					}
					// discarding bandedge and stitching
					frequencyDomainPtr += numOffsetForDiscard;
					memcpy(spaceForStitchingFft + stitchOffset * iInputChannel, frequencyDomainPtr, numDataForCopy);	

				} // end of for nInputChannels 

				// multiply convolution kernel for same polarization of whole input channels  
				if(_response!=nullptr) {
					_response->operate(	spaceForStitchingFft, 
										iPolarization,
										0,  // was iInputChannel*_nChannelSubbands in Primary goal
										_nChannelSubbands);
				}

				// output data if output is available
				if(out!=nullptr) {

					// number of input data for Backward FFT should be 
					// adjusted because it is not power of 2 due to discarding bandedges
					_backward->bcc1d(nPointsToBwdFFT, _complexTime, spaceForStitchingFft);

					// Copy output data to output
					// start point of output data, so used 0,0
					void* destinationPtr = out->get_datptr(0,0) 
						+ (iPart * nPolarizations + iPolarization) * _nPointsToKeep;
					void* sourcePtr = _complexTime + _nFilterPosition*floatsPerComplex;

					memcpy(destinationPtr, sourcePtr, _nPointsToKeep * sizeOfComplexData);
				} // end of if(out!=nullptr)
			} // end of nPolarizations
		} // end of iPart 
	} // end of iOutputChannels
}

void FilterbankInverseEngineCPU::finish()
{
	//
}

/**
 * Setup scratch space for performing FFT calculations
 * 
 * @param scratch pointer to memory to use for scratch
 */
void FilterbankInverseEngineCPU::set_scratch(float* scratchPtr)
{
	TESTING_LOG("FilterbankInverseEngineCPU::set_scratch(" << scratch << ")");
	// initialize scratch space for FFTs
	unsigned bigFftSize = _nInputChannels * _frequencyResolution * 2 * sizeof(float) * _nPolarization; 
	
	if(_isRealToComplex) {                                                                                     
		bigFftSize += 256;
	}  

	scratch = scratchPtr;
	_complexSpectrum[0] = scratch;
	_complexSpectrum[1] = _complexSpectrum[0];
	_complexTime = _complexSpectrum[1] + bigFftSize;
	_windowedTimeDomain = _complexTime + 2 * _frequencyResolution;     
}
