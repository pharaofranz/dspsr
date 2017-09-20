//-*-C++-*-
/***************************************************************************
 *
 *   Copyright(C) 2002-2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/Filterbank.h

#ifndef __Filterbank_h
#define __Filterbank_h

#include "dsp/Convolution.h"
#include <iostream>

namespace dsp {
    
    //! Breaks a single-band TimeSeries into multiple frequency channels
    /* This class implements the coherent filterbank technique described
       in Willem van Straten's thesis.  */
	//std::ostream _debugMessage(NULL);
    
	class Filterbank: public Convolution {
        
        public:
        
        //! Configuration options
        class Config;
		class DebugPrint;
        
        //! Null constructor
        Filterbank(const char* name = "Filterbank", Behaviour type = outofplace);
		
        //! Prepare all relevant attributes
        void prepare();
        
        //! Reserve the maximum amount of output space required
        void reserve();
        
        //! Get the minimum number of samples required for operation
        uint64_t get_minimum_samples() { return nsamp_fft; }
        
        //! Get the minimum number of samples lost
        uint64_t get_minimum_samples_lost() { return nsamp_overlap; }
        
        //! Set the number of channels into which the input will be divided
        void set_nchan(unsigned _nchan) { nchan = _nchan; }
        
        //! Get the number of channels into which the input will be divided
        unsigned get_nchan() const { return nchan; }
        
        unsigned get_nchan_subband() const {return nchan_subband; }
        
        //! Set the frequency resolution factor
        void set_freq_res(unsigned _freq_res) { freq_res = _freq_res; }
        void set_frequency_resolution(unsigned fres) { freq_res = fres; }
        
        //! Get the frequency resolution factor
        unsigned get_freq_res() const { return freq_res; } 
        unsigned get_frequency_resolution() const { return freq_res; }
        
        //! Engine used to perform discrete convolution step
        class Engine;
        void set_engine(Engine*);
	    
        protected:
        
        //! Perform the convolution transformation on the input TimeSeries
        virtual void transformation();
        
        //! Number of channels into which the input will be divided
        //! This is the final number of channels in the output
        unsigned nchan;
        
        //! Frequency resolution factor
        unsigned freq_res;
        
        // This is the number of channels each input channel will be divided into
        unsigned nchan_subband;
        
        //! Frequency channel overlap ratio
        //double overlap_ratio;
        
        //! Interface to alternate processing engine(e.g. GPU)
        //Reference::To<Engine> engine;
        
        private:
        
        void _makePreparations();
        void _prepareOutput(uint64_t ndat = 0, bool set_ndat = false);
        void _resizeOutput(bool reserve_extra = false);
        void _computeScaleFactor();
        void _computeSampleCounts();
        void _setupFftPlans();
		void _runForwardFft();
        //! Perform the filterbank step 
		void _initFilterbank();
        void _filterbank();
		void _filterbankCPU();
        
		//! Interface to alternate processing engine(e.g. GPU)
        Reference::To<Engine> _engine;

		float* _complexSpectrum[2];
		unsigned long in_step;
		unsigned long out_step;

		//! points kept from each small fft
		unsigned _nChannelsSmallFft;
		unsigned _nInputChannels;
    };
    
}

#endif

