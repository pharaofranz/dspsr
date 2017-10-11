/***************************************************************************
 *
 *   Copyright(C) 2002-2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Filterbank.h"
#include "dsp/FilterbankEngine.h"

#include "dsp/WeightedTimeSeries.h"
#include "dsp/Response.h"
#include "dsp/Apodization.h"
#include "dsp/InputBuffering.h"
#include "dsp/Scratch.h"
#include "dsp/OptimalFFT.h"

#include "FTransform.h"

#include <fstream>

using namespace std;

// #define _DEBUG 1

#if 0 
#define TESTING_LOG(s) cerr << s << endl
#define TESTING_LOG_LINE cerr << __LINE__ << ":" << __FUNCTION__ << endl
#else 
#define TESTING_LOG(s)
#define TESTING_LOG_LINE
#endif

ostream cerrStream(NULL);

ostream& isVerbose(ostream &stream)
{
	return (dsp::Filterbank::verbose) ? cerr : stream;
}

dsp::Filterbank::Filterbank(const char* name, Behaviour behaviour)
: Convolution(name, behaviour), nchan(0), freq_res(1)
{
	set_buffering_policy(new InputBuffering(this));
}

void dsp::Filterbank::set_engine(Engine* engine)
{
	cerr << "dsp::Filterbank::set_engine(" << engine << ")" << endl;
	_engine = engine;
}

/**
 * Initializer function to prepare Filterbank for use.
 */
inline void dsp::Filterbank::prepare()
{
	cerrStream << isVerbose << "dsp::Filterbank::prepare" << endl;
	//
	_makePreparations();
	//
	prepared = true;
}


/**
 * These are preparations that could be performed once at the start of
 * the data processing
 */
inline void dsp::Filterbank::_makePreparations()
{
	TESTING_LOG("_makePreparations - start");
	_computeSampleCounts();
	_computeScaleFactor();
	matrix_convolution = false; // We don't do/use matrix convolution
	if(has_buffering_policy()==true) {
		get_buffering_policy()->set_minimum_samples(nsamp_fft);
	}
	_prepareOutput();
	if(_engine!=nullptr) {
		_engine->setup(this);
	} else {
		_setupFftPlans();
	}
	set_passband(NULL); // Passband is unused
	TESTING_LOG("_makePreparations - end");
}

/**
 * Prepares output data structures and values for output of processed timeseries data.
 */
void dsp::Filterbank::_prepareOutput(uint64_t ndat, bool set_ndat)
{
	TESTING_LOG("_prepareOutput - start");
	if(set_ndat==true) {
		cerrStream << isVerbose << "dsp::Filterbank::_prepareOutput set ndat=" 
			<< ndat << endl;
		output->set_npol( input->get_npol() );
		output->set_nchan( nchan );
		output->set_ndim( 2 );
		output->set_state( Signal::Analytic);
		output->resize( ndat );
	}
	TESTING_LOG_LINE;
	//
	output->copy_configuration( get_input() );
	output->set_nchan( nchan );
	output->set_ndim( 2 );
	output->set_state( Signal::Analytic );
	//
	WeightedTimeSeries* weighted_output = dynamic_cast<WeightedTimeSeries*>(output.get());
	//
	unsigned tres_ratio = nsamp_fft / freq_res;
	TESTING_LOG_LINE;
	//
	if(weighted_output!=nullptr) {
		weighted_output->set_reserve_kludge_factor(tres_ratio);		
		//
		weighted_output->set_reserve_kludge_factor(1);
		weighted_output->convolve_weights(nsamp_fft, nsamp_step);
		weighted_output->scrunch_weights(tres_ratio);
	}
	TESTING_LOG_LINE;
	/* the problem: copy_configuration copies the weights array, which
	   results in a call to resize_weights, which sets some offsets
	   according to the reserve(for later prepend).  However, the
	   offset is computed based on values that are about to be changed.
	   This kludge allows the offsets to reflect the correct values
	   that will be set later */
	if(set_ndat==true) {
		cerrStream << isVerbose << "dsp::Filterbank::_prepareOutput reset ndat=" 
			<< ndat << endl;
		output->resize(ndat);
	} else {
		ndat = input->get_ndat() / tres_ratio;
		cerrStream << isVerbose << "dsp::Filterbank::_prepareOutput scrunch ndat=" 
			<< ndat << endl;
		output->resize(ndat);
	}
	cerrStream << isVerbose << "dsp::Filterbank::_prepareOutput output ndat="
		<< output->get_ndat() << endl;
	TESTING_LOG_LINE;
	output->rescale(scalefac);
	cerrStream << isVerbose << "dsp::Filterbank::_prepareOutput scale="
		<< output->get_scale() <<endl;
	TESTING_LOG_LINE;
	/*
	 * output data will have new sampling rate
	 * NOTE: that nsamp_fft already contains the extra factor of two required
	 * when the input TimeSeries is Signal::Nyquist(real) sampled
	 */
	double ratechange = static_cast<double>(freq_res) / static_cast<double>(nsamp_fft);
	output->set_rate(input->get_rate() * ratechange);
	TESTING_LOG_LINE;
	if(freq_res == 1) {
		output->set_dual_sideband(true);
	}
	TESTING_LOG_LINE;
	/*
	 * if freq_res is even, then each sub-band will be centred on a frequency
	 * that lies on a spectral bin *edge* - not the centre of the spectral bin
	 */
	output->set_dc_centred(freq_res%2);
	TESTING_LOG_LINE;
	// dual sideband data produces a band swapped result
	if(input->get_dual_sideband()==true) { 
		_nInputChannels = input->get_nchan();
		if(_nInputChannels > 1) {
			output->set_nsub_swap(_nInputChannels);
		} else {
			output->set_swap(true);
		}
	}
	TESTING_LOG_LINE;
	// increment the start time by the number of samples discarded from the fft
	output->change_start_time(nfilt_pos);
	TESTING_LOG_LINE;
	cerrStream << isVerbose << "dsp::Filterbank::_prepareOutput start time += "
		<< nfilt_pos << " samps -> " << output->get_start_time() << endl;
	TESTING_LOG_LINE;
	// enable the Response to record its effect on the output Timeseries
	if(response!=nullptr) {
		response->mark(output);
	}
	TESTING_LOG("_prepareOutput - end");
}

inline void dsp::Filterbank::reserve()
{
	cerrStream << isVerbose << "dsp::Filterbank::reserve" << endl;
	//
	_resizeOutput(true);
}

/**
 * Re-prepares output after adjusting output size to match input size.
 */
inline void dsp::Filterbank::_resizeOutput(bool reserve_extra)
{
	const uint64_t ndat = input->get_ndat();
	// number of big FFTs(not including, but still considering, extra FFTs
	// required to achieve desired time resolution) that can fit into data
	npart = 0;
	//
	if(nsamp_step == 0) {
		throw Error(InvalidState, "dsp::Filterbank::_resizeOutput",
				"nsamp_step == 0 ... not properly prepared");
	}
	if(ndat > nsamp_overlap) {
		npart =(ndat-nsamp_overlap)/nsamp_step;
	}
	// on some iterations, ndat could be large enough to fit an extra part
	if(reserve_extra && has_buffering_policy()) {
		npart += 2;
	}
	// points kept from each small fft
	_nChannelsSmallFft = freq_res - nfilt_tot;
	//
	uint64_t output_ndat = npart * _nChannelsSmallFft;
	//
	cerrStream << isVerbose << "dsp::Filterbank::reserve input ndat=" << ndat 
		<< " overlap=" << nsamp_overlap << " step=" << nsamp_step
		<< " reserve=" << reserve_extra << " _nChannelsSmallFft=" << _nChannelsSmallFft
		<< " npart=" << npart << " output ndat=" << output_ndat << endl;
	//
#if DEBUGGING_OVERLAP
	// this exception is useful when debugging, but not at the end-of-file
	if( !has_buffering_policy() && ndat > 0
			&&(nsamp_step*npart + nsamp_overlap != ndat) )
		throw Error(InvalidState, "dsp::Filterbank::reserve",
				"npart=%u * step=%u + overlap=%u != ndat=%u",
				npart, nsamp_step, nsamp_overlap, ndat);
#endif
	// prepare the output TimeSeries
	_prepareOutput(output_ndat, true);
}

/**
 * Calculates the scale factor used to correct for whether or not FFT library uses
 * normalized values.
 */
inline void dsp::Filterbank::_computeScaleFactor()
{
	scalefac = (FTransform::get_norm() == FTransform::unnormalized) ?
		(double(n_fft) * double(freq_res)) :
		(double(n_fft) / double(freq_res));
}

/**
 * Calculates the various sample counts and buffer sizes, resolutions and offsets
 * required for operation of the filterbank engine.
 */
inline void dsp::Filterbank::_computeSampleCounts()
{
	TESTING_LOG("computeSampleCounts - start");
	//
	_nInputChannels = input->get_nchan();
	//! Number of channels outputted per input channel
	nchan_subband = nchan / _nInputChannels;
	//
	nfilt_pos = 0;
	nfilt_neg = 0;
	nfilt_tot = 0;
	freq_res = 1;
	//
	if(response) {
		response->match(input, nchan);
		//! Complex samples dropped from beginning of cyclical convolution result
		nfilt_pos = response->get_impulse_pos();
		//! Complex samples dropped from end of cyclical convolution result
		nfilt_neg = response->get_impulse_neg();
		//! number of complex samples invalid in result of small ffts
		nfilt_tot = nfilt_pos + nfilt_neg;
		//! Frequency resolution factor
		freq_res = response->get_ndat();
	}
	//! number of complex values in the result of the first fft
	n_fft = nchan_subband * freq_res;
	//
	const bool isNyquist =(input->get_state() == Signal::Nyquist);
	const int fftPointsPerSample = isNyquist ? 2 : 1;
	nsamp_fft = fftPointsPerSample * n_fft;
	nsamp_overlap = fftPointsPerSample * nfilt_tot * nchan_subband;
	//! number of timesamples between start of each big fft
	nsamp_step = nsamp_fft - nsamp_overlap;
	TESTING_LOG("computeSampleCounts - end");
}

/**
 * Create FFT plans that will be used when performing FFT/iFFTs that are optimized
 * to appropriates sizes for use in filterbank engine.
 */
inline void dsp::Filterbank::_setupFftPlans()
{
	TESTING_LOG("setupFftPlans - start");
	using namespace FTransform;
	OptimalFFT* optimal = 0;
	if(response && response->has_optimal_fft()) {
		optimal = response->get_optimal_fft();
		if(optimal) {
			FTransform::set_library(optimal->get_library(nsamp_fft));
		}
	}
	const bool isRealToComplex =(input->get_state() == Signal::Nyquist);
	forward = Agent::current->get_plan(nsamp_fft, isRealToComplex ? FTransform::frc : FTransform::fcc);
	if(optimal) {
		FTransform::set_library(optimal->get_library(freq_res));
	}
	if(freq_res > 1) {
		backward = Agent::current->get_plan(freq_res, FTransform::bcc);
	}
	TESTING_LOG("setupFftPlans - end");
}

void dsp::Filterbank::transformation()
{
	cerrStream << isVerbose << "dsp::Filterbank::transformation input ndat=" 
		<< input->get_ndat() << " nchan=" << _nInputChannels << endl;
	//
	if(prepared == false) {
		prepare();
	}
	_resizeOutput();
	//
	if(has_buffering_policy()==true) {
		get_buffering_policy()->set_next_start(nsamp_step * npart);
	}
	uint64_t output_ndat = output->get_ndat();
	// points kept from each small fft
	//_nChannelsSmallFft = freq_res - nfilt_tot;
	cerrStream << isVerbose << "dsp::Filterbank::transformation npart=" << npart 
		<< " _nChannelsSmallFft=" << _nChannelsSmallFft << " output_ndat=" << output_ndat << endl;

	// set the input sample
	int64_t input_sample = input->get_input_sample();
	if(output_ndat == 0) {
		output->set_input_sample(0);
	} else if(input_sample >= 0) {
		output->set_input_sample((input_sample / nsamp_step) * _nChannelsSmallFft);
	}
	cerrStream << isVerbose << "dsp::Filterbank::transformation after prepare output"
		" ndat=" << output->get_ndat() << 
		" input_sample=" << output->get_input_sample() << endl;
	//
	if(npart == 0) {
		cerrStream << isVerbose << "dsp::Filterbank::transformation empty result" << endl;
		return;
	}
	//
	_filterbank();
}

inline void dsp::Filterbank::_filterbank()
{
	// number of floats to step between input to filterbank
	in_step = nsamp_step * input->get_ndim();
	// freq_res - nfilt_tot : points kept from each small fft 
	//_nChannelsSmallFft = freq_res - nfilt_tot;
	// number of floats to step between output from filterbank
	out_step = (freq_res - nfilt_tot) * 2;
	// initialize scratch space for FFTs
	_bigFftSize = nchan_subband * freq_res * 2;
	if(input->get_state() == Signal::Nyquist) {
		_bigFftSize += 256;
	}
	// also need space to hold backward FFTs
	unsigned scratch_needed = _bigFftSize + 2 * freq_res;
	// divide up the scratch space
	_complexSpectrum[0] = scratch->space<float>(scratch_needed);
	_complexSpectrum[1] = _complexSpectrum[0];	
	// /////////////////////////////////////////////////////////////////////
	// PERFORM FILTERBANK VIA ENGINE(could be on GPU with CUDA or on CPU)
	// /////////////////////////////////////////////////////////////////////
	if(_engine!=nullptr) {
		//cerr << endl << "nsamp_fft=" << nsamp_fft << endl;
		cerrStream << isVerbose << "have engine"<<endl;
		//
		_engine->set_scratch(_complexSpectrum[0]);
		_engine->perform(input, output, npart, in_step, out_step);
		if(Operation::record_time == true) {
			_engine->finish();
		}
	}
	//
	cerrStream << isVerbose << "dsp::Filterbank::transformation return with output ndat="
		<< output->get_ndat() << endl;
}

