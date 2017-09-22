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
    
	_nFftPoints = _nChannelSubbands*_frequencyResolution;

    const unsigned nSamplesForward = (_frequencyResolution*_nChannelSubbands) * (_isRealToComplex ? 2 : 1);
    const unsigned nSamplesBackward = _frequencyResolution;
    const FTransform::type transformType = (_isRealToComplex ? FTransform::frc : FTransform::fcc);
    _forward = FTransform::Agent::current->get_plan(nSamplesForward, transformType);
    _backward = FTransform::Agent::current->get_plan(nSamplesBackward, FTransform::bcc);
    _nPointsToKeep = _frequencyResolution;
	_nFilterPosition = 0;
    
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
	_nFilterPosition = nFilterPositive;
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

	std::cerr << "_frequencyResolution=" << _frequencyResolution << " _nChannelSubbands=" << _nChannelSubbands << " _nPointsToKeep=" << _nPointsToKeep << " _scaleFactor=" << _scaleFactor
		<< " _nFftSamples=" << _nFftSamples << " _nSampleStep=" << _nSampleStep << std::endl << std::endl;

#endif
}

void FilterbankEngineCPU::perform(const dsp::TimeSeries* in, dsp::TimeSeries* out,uint64_t nParts, uint64_t inStep, uint64_t outStep)
{
#ifdef USE_NEW_CPU_CODE
	//std::cerr << "FilterbankEngineCPU::perform()" << std::endl;
    	const uint64_t nPolarizations = in->get_npol();
    	const uint64_t nInputChannels = in->get_nchan();
    	const uint64_t nOutputChannels = out->get_nchan();

	//std::cerr << "nPolarizations=" << nPolarizations << " nInputChannels=" << nInputChannels << " nParts=" << nParts << std::endl;
    
    	for(uint64_t iInputChannel = 0; iInputChannel < nInputChannels; iInputChannel++) {
		for(uint64_t iPart = 0; iPart < nParts; iPart++) {
                	const uint64_t inOffset = iPart*inStep;
                	const uint64_t outOffset = iPart*outStep;
        		for(uint64_t iPolarization = 0; iPolarization < nPolarizations; iPolarization++) {
				//std::cerr << "iInputChannel:" << iInputChannel << " iPart:" << iPart << " iPolarization:" << iPolarization << std::endl;
				//std::cerr << "inOffset:" << inOffset << " outOffset:" << outOffset << std::endl;
                		float* timeDomainInputPtr = const_cast<float*>(in->get_datptr(iInputChannel, iPolarization)) + inOffset;
                		float* frequencyDomainPtr = _complexSpectrum[iPolarization];
				//std::cerr << "iInputChannel=" << iInputChannel << " iPolarization=" << iPolarization << " inOffset=" << inOffset << std::endl;
                		//float* timeDomainOutputPtr = _scratch + _nFftSamples;
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
                		
				//_backward->bcc1d(_frequencyResolution, _complexTime, frequencyDomainPtr);
                		//
                		if(out) {
					for(uint64_t iSubband = 0; iSubband < _nChannelSubbands; iSubband++) {
						//std::cerr << "_";
						//std::cerr << "_frequencyResolution=" << _frequencyResolution << " _complexTime=" << _complexTime << std::endl;
						_backward->bcc1d(_frequencyResolution, _complexTime, frequencyDomainPtr);
						//_forward->frc1d(_frequencyResolution, frequencyDomainPtr, timeDomainInputPtr);
						//
						frequencyDomainPtr += _frequencyResolution;
                    				void* destinationPtr = out->get_datptr((iInputChannel*_nChannelSubbands)+iSubband, iPolarization)+outOffset;
						void* sourcePtr = _complexTime + _nFilterPosition*2;
						memcpy(destinationPtr, sourcePtr, _nPointsToKeep*sizeof(float)*2);
						//std::cerr << "_nFilterPosition=" << _nFilterPosition << " _nPointsToKeep=" << _nPointsToKeep  << std::endl;
					}
					//std::cerr << std::endl;
                    			//void* outputPtr = out->get_datptr(iChan*_nChannelSubbands, iPol)+outOffset;
					//const uint64_t outputSpan = out->get_datptr(iChan*_nChannelSubbands+1, iPol) - out->get_datptr(iChan*_nChannelSubbands, iPol);
					//memcpy(destinationPtr, _complexTime, _frequencyResolution*sizeof(float));
                		}
				//std::cerr << "looped" << std::endl;
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
