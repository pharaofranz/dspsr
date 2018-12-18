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
#include <fstream>

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
        uint64_t get_minimum_samples() noexcept { return nsamp_fft; }
        
        //! Get the minimum number of samples lost
        uint64_t get_minimum_samples_lost() noexcept { return nsamp_overlap; }
        
        //! Set the number of channels into which the input will be divided
        void set_nchan(unsigned _nchan) noexcept { nchan = _nchan; }
        
        //! Get the number of channels into which the input will be divided
        unsigned get_nchan() const noexcept { return nchan; }
        
        unsigned get_nchan_subband() const noexcept {return nchan_subband; }
   
   		void set_isInverseFilterbank(const bool isInverseFilterbank) noexcept { 
            _isInverseFilterbank = isInverseFilterbank;
	    }

	
        //! Get the number of input channels
        unsigned get_nInputChannel() const noexcept { return input->get_nchan(); }
        
        //! Get the number of channels into which the input will be divided
        unsigned get_nPolarization() const noexcept { return input->get_npol(); }
		
	  //! Set the frequency resolution factor
        void set_freq_res(unsigned _freq_res) noexcept { freq_res = _freq_res; }
        void set_frequency_resolution(unsigned fres) noexcept { freq_res = fres; }
        
        //! Get the frequency resolution factor
        unsigned get_freq_res() const noexcept { return freq_res; } 
        unsigned get_frequency_resolution() const noexcept { return freq_res; }
        
        //! Set the Oversampling factor num & denom
        void set_oversampling_factor (const Rational& _osf) { oversampling_factor = _osf; }

        //! Get the Oversampling factor num & denom
        const Rational& get_oversampling_factor() {return input->get_oversampling_factor();}

        
        //! Engine used to perform discrete convolution step
        class Engine;
        void set_engine(Engine*);
        
		virtual void transformation();
	    
		// unit test example
		bool isSimulation;
       
	   	protected:
        
        //! Perform the convolution transformation on the input TimeSeries
        //virtual void transformation();
        
        //! Number of channels into which the input will be divided
        //! This is the final number of channels in the output
        unsigned nchan;
        
        //! Frequency resolution factor
        unsigned freq_res;
        
        // This is the number of channels each input channel will be divided into
        unsigned nchan_subband;
        
        //! Frequency channel overlap ratio
        //double overlap_ratio;
        
        //! Oversampling factor 
        Rational oversampling_factor;
 
        //! Interface to alternate processing engine(e.g. GPU)
        //Reference::To<Engine> engineunsigned tres_ratio;
        
        private:
        
        void _preparationForDataProcessing();
		unsigned _getTresRatio();
        void _configOutputStructure(uint64_t ndat, bool set_ndat);
		void _configWeightedOutput(unsigned tres_ratio);
		void _prepareOutput(uint64_t ndat = 0, bool set_ndat = false);
        void _reprepareOutputToMatchInput(bool reserve_extra = false);
        void _computeScaleFactor();
        void _computeSampleCounts();
        void _setupFftPlans();
		void _runForwardFft();
        
		//! Perform the filterbank step 
		void _setMinimumSamples();
		void _setupEngine();
		void _setOutputForFilterbank();
        void _calculateInStepOutStepForFilterbank();
		void _initScratchSpaceForFilterbank();
		void _runFilterbank();
        void _filterbank();
        
		//! Interface to alternate processing engine(e.g. GPU)
        Reference::To<Engine> _engine;

		float* _complexSpectrum[2];
		unsigned long in_step;
		unsigned long out_step;

		//! points kept from each small fft
		unsigned _nChannelsSmallFft;
		unsigned _nInputChannels;
		unsigned _bigFftSize;
		bool _isInverseFilterbank;
    };

}

#endif

