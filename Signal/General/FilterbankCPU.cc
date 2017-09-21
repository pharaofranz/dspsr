//-*-C++-*-

#include "dsp/FilterbankCPU.hpp"

#include <fstream>
#include <cassert>
#include <cstring>

#define USE_NEW_CPU_CODE

FilterbankEngineCPU::FilterbankEngineCPU()
{
    //
}

FilterbankEngineCPU::~FilterbankEngineCPU()
{
    //
}

void FilterbankEngineCPU::setup(dsp::Filterbank* filterbank)
{
#ifdef USE_NEW_CPU_CODE
	std::cerr << "FilterbankEngineCPU::setup()" << std::endl;
    // passband is unused by engine
    filterbank->set_passband(NULL);
    
    _frequencyResolution = filterbank->get_freq_res();
    _nChannelSubbands = filterbank->get_nchan_subband();
    _isRealToComplex = (filterbank->get_input()->get_state() == Signal::Nyquist);
    
    const unsigned nSamplesForward = (_frequencyResolution*_nChannelSubbands) * (_isRealToComplex ? 2 : 1);
    const unsigned nSamplesBackward = (_frequencyResolution*_nChannelSubbands);
    const FTransform::type transformType = (_isRealToComplex ? FTransform::frc : FTransform::fcc);
    _forward = FTransform::Agent::current->get_plan(nSamplesForward, transformType);
    _backward = FTransform::Agent::current->get_plan(nSamplesBackward, FTransform::bcc);
    _nPointsToKeep = _frequencyResolution;
    
    if(filterbank->has_response()) {
        const dsp::Response* response = filterbank->get_response();
        
        unsigned nChannels = response->get_nchan();
        unsigned nData = response->get_ndat();
        unsigned nDimensions = response->get_ndim();
        
        assert(nChannels == filterbank->get_nchan());
        assert(nData == _frequencyResolution);
        assert(nDimensions == 2);
        
        //! Complex samples dropped from beginning of cyclical convolution result
        unsigned nFilterPositive = response->get_impulse_pos();
        //! Complex samples dropped from end of cyclical convolution result
        unsigned nFilterNegative = response->get_impulse_neg();
        
        unsigned nFilterTotal = nFilterPositive + nFilterNegative;
        
        _nPointsToKeep = _frequencyResolution - nFilterTotal;
    }
    
    const bool isNormalized = FTransform::get_norm() == FTransform::normalized;
    
    const double scaleFactorNormalized = double(_nFftPoints)/double(_frequencyResolution);
    
    const double scaleFactorUnnormalized = double(_nFftPoints)*double(_frequencyResolution);
    
    _scaleFactor = isNormalized ? scaleFactorNormalized : scaleFactorUnnormalized;
    
    _nFftSamples = _isRealToComplex ? 2*_nFftPoints : _nFftPoints;
    
    _nSampleStep = _nFftSamples - _nSampleOverlap;
#endif
}

void FilterbankEngineCPU::perform(const dsp::TimeSeries* in, dsp::TimeSeries* out,uint64_t nParts, uint64_t inStep, uint64_t outStep)
{
#ifdef USE_NEW_CPU_CODE
	//std::cerr << "FilterbankEngineCPU::perform()" << std::endl;
    const uint64_t nPolarizations = in->get_npol();
    const uint64_t nInputChannels = in->get_nchan();
    const uint64_t nOutputChannels = out->get_nchan();
    
    for(uint64_t i = 0; i < nInputChannels; i++) {
        for(uint64_t j = 0; j < nPolarizations; j++) {
            for(uint64_t k = 0; k < nParts; k++) {
		//std::cerr << "i:" << i << " j:" << j << " k:" << k << std::endl;
                const uint64_t inOffset = k*inStep;
                const uint64_t outOffset = k*outStep;
		//std::cerr << "inOffset:" << inOffset << " outOffset:" << outOffset << std::endl;
                float* timeDomainInputPtr = (float*)in->get_datptr(i, j) + inOffset;
                float* frequencyDomainPtr = _scratch;
                float* timeDomainOutputPtr = _scratch + _nFftSamples;
		//std::cerr << "_forward:" << _forward << std::endl;
                if(_isRealToComplex) {
			//std::cerr << "_forward->frc1d:" << _forward->frc1d << std::endl;
			//std::cerr << "frc1d(" << _nFftSamples << ", " << frequencyDomainPtr << ", " << timeDomainInputPtr << ")" << std::endl;
                    _forward->frc1d(_nFftSamples, frequencyDomainPtr, timeDomainInputPtr);
                } else {
			//std::cerr << "fcc1d(" << _nFftSamples << ", " << frequencyDomainPtr << ", " << timeDomainInputPtr << ")" << std::endl;
                    _forward->fcc1d(_nFftSamples, frequencyDomainPtr, timeDomainInputPtr);
                }
		//std::cerr << "bcc1d(" << _frequencyResolution << ", " << timeDomainOutputPtr << ", " << frequencyDomainPtr << ")" << std::endl;
                _backward->bcc1d(_frequencyResolution, timeDomainOutputPtr, frequencyDomainPtr);
                //
                if(out) {
                    float* outputPtr = out->get_datptr(i*_nChannelSubbands, j)+outOffset;
			memcpy(outputPtr, timeDomainOutputPtr, _frequencyResolution*sizeof(float));
                }
            }
        }
    }
	//std::cerr << "FilterbankEngineCPU::Perform() complete" << std::endl;
#endif
}

void FilterbankEngineCPU::finish()
{
    //
}


void FilterbankEngineCPU::set_scratch (float* scratch)
{
	//std::cerr << "FilterbankEngineCPU::set_scratch(" << scratch << ")" << std::endl;
    _scratch = scratch;
}
