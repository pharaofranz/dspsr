//-*-C++-*-

#pragma once

#ifndef __FilterbankCPU_hpp
#define __FilterbankCPU_hpp

#include "dsp/FilterbankEngine.h"
//
#include "dsp/Transformation.h"
//#include "dsp/LaunchConfig.h"

class FilterbankEngineCPU : public dsp::Filterbank::Engine
{
    public:
    FilterbankEngineCPU();
    ~FilterbankEngineCPU();
    
    void setup(dsp::Filterbank* filterbank);
    
    void set_scratch (float* scratch);
    
    void perform(const dsp::TimeSeries* in, dsp::TimeSeries* out,
                 uint64_t npart, uint64_t in_step, uint64_t out_step);
    
    void finish();
    
    protected:
    unsigned _frequencyResolution;
    unsigned _nChannelSubbands;
    bool _isRealToComplex;
    unsigned _nPointsToKeep;
    double _scaleFactor;
    unsigned _nFftPoints;
    unsigned _nFftSamples;
    unsigned _nSampleOverlap;
    unsigned _nSampleStep;
	unsigned _nFilterPosition;
    //
    FTransform::Plan* _forward;
    FTransform::Plan* _backward;
    //
    	float* _scratch;
    	float* _complexSpectrum[2];
	float* _complexTime;
	float* _windowedTimeDomain;
    
};

#endif
