//-*-C++-*-

#pragma once

#ifndef __FilterbankCPU_hpp
#define __FilterbankCPU_hpp

#include "dsp/FilterbankEngine.h"
//#include "dsp/LaunchConfig.h"

class FilterbankEngineCPU : public dsp::Filterbank::Engine
{
    public:
    FilterbankEngineCPU();
    ~FilterbankEngineCPU();
    
    void setup (dsp::Filterbank*);
    
    void perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
                  uint64_t npart, uint64_t in_step, uint64_t out_step);
    
    void finish ();
    
};

#endif