/***************************************************************************
 *
 *   Copyright (C) 2002-2011 by Willem van Straten
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

#define TESTING_LOG(s) cerr << s << endl
#define TESTING_LOG_LINE cerr << __LINE__ << ":" << __FUNCTION__ << endl

dsp::Filterbank::Filterbank (const char* name, Behaviour behaviour)
: Convolution (name, behaviour)
{
    nchan = 0;
    freq_res = 1;
    // overlap_ratio = 0;
    
    set_buffering_policy (new InputBuffering (this));
}

void dsp::Filterbank::set_engine (Engine* _engine)
{
    engine = _engine;
}

void dsp::Filterbank::prepare ()
{
    if (verbose)
        cerr << "dsp::Filterbank::prepare" << endl;
    
    make_preparations ();
    prepared = true;
}


/*
  These are preparations that could be performed once at the start of
  the data processing
*/
void dsp::Filterbank::make_preparations ()
{
    TESTING_LOG("make_preparations - start");
    computeSampleCounts();
    computeScaleFactor();
    if(has_buffering_policy()) {
        get_buffering_policy()->set_minimum_samples (nsamp_fft);
    }
    prepare_output();
    if(engine) {
        engine->setup(this);
    } else {
        setupFftPlans();
    }
    TESTING_LOG("make_preparations - end");
}

void dsp::Filterbank::prepare_output (uint64_t ndat, bool set_ndat)
{
    TESTING_LOG("prepare_output - start");
    if (set_ndat)
    {
        if (verbose)
            cerr << "dsp::Filterbank::prepare_output set ndat=" << ndat << endl;
        
        output->set_npol( input->get_npol() );
        output->set_nchan( nchan );
        output->set_ndim( 2 );
        output->set_state( Signal::Analytic);
        output->resize( ndat );
    }
    TESTING_LOG_LINE;
    
    WeightedTimeSeries* weighted_output;
    weighted_output = dynamic_cast<WeightedTimeSeries*> (output.get());
    
    TESTING_LOG_LINE;
    /* the problem: copy_configuration copies the weights array, which
       results in a call to resize_weights, which sets some offsets
       according to the reserve (for later prepend).  However, the
       offset is computed based on values that are about to be changed.
       This kludge allows the offsets to reflect the correct values
       that will be set later */
    
    unsigned tres_ratio = nsamp_fft / freq_res;
    TESTING_LOG_LINE;
    if (weighted_output)
        weighted_output->set_reserve_kludge_factor (tres_ratio);
    TESTING_LOG_LINE;
    output->copy_configuration ( get_input() );
    
    output->set_nchan( nchan );
    output->set_ndim( 2 );
    output->set_state( Signal::Analytic );
    TESTING_LOG_LINE;
    custom_prepare ();
    TESTING_LOG_LINE;
    if (weighted_output)
    {
        weighted_output->set_reserve_kludge_factor (1);
        weighted_output->convolve_weights (nsamp_fft, nsamp_step);
        weighted_output->scrunch_weights (tres_ratio);
    }
    TESTING_LOG_LINE;
    if (set_ndat)
    {
        if (verbose)
            cerr << "dsp::Filterbank::prepare_output reset ndat=" << ndat << endl;
        output->resize (ndat);
    }
    else
    {
        ndat = input->get_ndat() / tres_ratio;
        
        if (verbose)
            cerr << "dsp::Filterbank::prepare_output scrunch ndat=" << ndat << endl;
        output->resize (ndat);
    }
    TESTING_LOG_LINE;
    if (verbose)
        cerr << "dsp::Filterbank::prepare_output output ndat="
        << output->get_ndat() << endl;
    TESTING_LOG_LINE;
    output->rescale (scalefac);
    TESTING_LOG_LINE;
    if (verbose) cerr << "dsp::Filterbank::prepare_output scale="
        << output->get_scale() <<endl;
    TESTING_LOG_LINE;
    /*
     * output data will have new sampling rate
     * NOTE: that nsamp_fft already contains the extra factor of two required
     * when the input TimeSeries is Signal::Nyquist (real) sampled
     */
    double ratechange = double(freq_res) / double (nsamp_fft);
    output->set_rate (input->get_rate() * ratechange);
    TESTING_LOG_LINE;
    if (freq_res == 1)
        output->set_dual_sideband (true);
    TESTING_LOG_LINE;
    /*
     * if freq_res is even, then each sub-band will be centred on a frequency
     * that lies on a spectral bin *edge* - not the centre of the spectral bin
     */
    output->set_dc_centred (freq_res%2);
    TESTING_LOG_LINE;
    
#if 0
    // the centre frequency of each sub-band will be offset
    double channel_bandwidth = input->get_bandwidth() / nchan;
    double shift = double(freq_res-1)/double(freq_res);
    output->set_centre_frequency_offset ( 0.5*channel_bandwidth*shift );
#endif
    TESTING_LOG_LINE;
    // dual sideband data produces a band swapped result
    if (input->get_dual_sideband())
    {
        if (input->get_nchan() > 1)
            output->set_nsub_swap (input->get_nchan());
        else
            output->set_swap (true);
    }
    TESTING_LOG_LINE;
    // increment the start time by the number of samples dropped from the fft
    
    //cerr << "FILTERBANK OFFSET START TIME=" << nfilt_pos << endl;
    
    output->change_start_time (nfilt_pos);
    TESTING_LOG_LINE;
    if (verbose)
        cerr << "dsp::Filterbank::prepare_output start time += "
        << nfilt_pos << " samps -> " << output->get_start_time() << endl;
    TESTING_LOG_LINE;
    // enable the Response to record its effect on the output Timeseries
    if (response)
        response->mark (output);
    TESTING_LOG("prepare_output - end");
}

void dsp::Filterbank::reserve ()
{
    if (verbose)
        cerr << "dsp::Filterbank::reserve" << endl;
    
    resize_output (true);
}

void dsp::Filterbank::resize_output (bool reserve_extra)
{
    const uint64_t ndat = input->get_ndat();
    
    // number of big FFTs (not including, but still considering, extra FFTs
    // required to achieve desired time resolution) that can fit into data
    npart = 0;
    
    if (nsamp_step == 0)
        throw Error (InvalidState, "dsp::Filterbank::resize_output",
                     "nsamp_step == 0 ... not properly prepared");
    
    if (ndat > nsamp_overlap)
        npart = (ndat-nsamp_overlap)/nsamp_step;
    
    // on some iterations, ndat could be large enough to fit an extra part
    if (reserve_extra && has_buffering_policy())
        npart += 2;
    
    // points kept from each small fft
    unsigned nkeep = freq_res - nfilt_tot;
    
    uint64_t output_ndat = npart * nkeep;
    
    if (verbose)
        cerr << "dsp::Filterbank::reserve input ndat=" << ndat 
        << " overlap=" << nsamp_overlap << " step=" << nsamp_step
        << " reserve=" << reserve_extra << " nkeep=" << nkeep
        << " npart=" << npart << " output ndat=" << output_ndat << endl;
    
#if DEBUGGING_OVERLAP
    // this exception is useful when debugging, but not at the end-of-file
    if ( !has_buffering_policy() && ndat > 0
        && (nsamp_step*npart + nsamp_overlap != ndat) )
        throw Error (InvalidState, "dsp::Filterbank::reserve",
                     "npart=%u * step=%u + overlap=%u != ndat=%u",
                     npart, nsamp_step, nsamp_overlap, ndat);
#endif
    
    // prepare the output TimeSeries
    prepare_output (output_ndat, true);
}

void dsp::Filterbank::computeScaleFactor()
{
    TESTING_LOG("computeScaleFactor - start");
    if (FTransform::get_norm() == FTransform::unnormalized) {
        scalefac = double(n_fft) * double(freq_res);
    } else if (FTransform::get_norm() == FTransform::normalized) {
        scalefac = double(n_fft) / double(freq_res);
    }
    TESTING_LOG("computeScaleFactor - end");
}

void dsp::Filterbank::computeSampleCounts()
{
    TESTING_LOG("computeSampleCounts - start");
    //! Number of channels outputted per input channel
    nchan_subband = nchan / input->get_nchan();
    if(response) {
        //! Complex samples dropped from beginning of cyclical convolution result
        nfilt_pos = response->get_impulse_pos();
        //! Complex samples dropped from end of cyclical convolution result
        nfilt_neg = response->get_impulse_neg();
        //! number of complex samples invalid in result of small ffts
        nfilt_tot = nfilt_pos + nfilt_neg;
        //! Frequency resolution factor
        freq_res = response->get_ndat();
        //! number of complex values in the result of the first fft
        unsigned n_fft = nchan_subband * freq_res;
    }
    if(input->get_state() == Signal::Nyquist) {
        nsamp_fft = 2 * n_fft;
        nsamp_overlap = 2 * nfilt_tot * nchan_subband;
    } else if(input->get_state() == Signal::Analytic) {
        nsamp_fft = n_fft;
        nsamp_overlap = nfilt_tot * nchan_subband;
    }
    //! number of timesamples between start of each big fft
    nsamp_step = nsamp_fft - nsamp_overlap;
    TESTING_LOG("computeSampleCounts - end");
}

void dsp::Filterbank::setupFftPlans()
{
    TESTING_LOG("setupFftPlans - start");
    using namespace FTransform;
    OptimalFFT* optimal = 0;
    if(response && response->has_optimal_fft()) {
        optimal = response->get_optimal_fft();
    }
    if(optimal) {
        FTransform::set_library(optimal->get_library(nsamp_fft));
    }
    if(input->get_state() == Signal::Nyquist) {
        forward = Agent::current->get_plan(nsamp_fft, FTransform::frc);
    } else {
        forward = Agent::current->get_plan(nsamp_fft, FTransform::fcc);
    }
    if(optimal) {
        FTransform::set_library(optimal->get_library(freq_res));
    }
    if(freq_res > 1) {
        backward = Agent::current->get_plan(freq_res, FTransform::bcc);
    }
    TESTING_LOG("setupFftPlans - end");
}

void dsp::Filterbank::transformation ()
{
    if (verbose)
        cerr << "dsp::Filterbank::transformation input ndat=" << input->get_ndat()
        << " nchan=" << input->get_nchan() << endl;
    
    if (!prepared)
        prepare ();
    
    resize_output ();
    
    if (has_buffering_policy())
        get_buffering_policy()->set_next_start (nsamp_step * npart);
    
    uint64_t output_ndat = output->get_ndat();
    
    // points kept from each small fft
    unsigned nkeep = freq_res - nfilt_tot;
    
    if (verbose)
        cerr << "dsp::Filterbank::transformation npart=" << npart 
        << " nkeep=" << nkeep << " output_ndat=" << output_ndat << endl;
    
    // set the input sample
    int64_t input_sample = input->get_input_sample();
    if (output_ndat == 0)
        output->set_input_sample (0);
    else if (input_sample >= 0)
        output->set_input_sample ((input_sample / nsamp_step) * nkeep);
    
    if (verbose)
        cerr << "dsp::Filterbank::transformation after prepare output"
        " ndat=" << output->get_ndat() << 
        " input_sample=" << output->get_input_sample() << endl;
    
    if (!npart)
    {
        if (verbose)
            cerr << "dsp::Filterbank::transformation empty result" << endl;
        return;
    }
    
    filterbank ();
}

void dsp::Filterbank::filterbank ()
{
    // initialize scratch space for FFTs
    unsigned bigfftsize = nchan_subband * freq_res * 2;
    if (input->get_state() == Signal::Nyquist)
        bigfftsize += 256;
    
    // also need space to hold backward FFTs
    unsigned scratch_needed = bigfftsize + 2 * freq_res;
    
    if (apodization)
        scratch_needed += bigfftsize;
    
    if (matrix_convolution)
        scratch_needed += bigfftsize;
    
    // divide up the scratch space
    float* c_spectrum[2];
    c_spectrum[0] = scratch->space<float> (scratch_needed);
    c_spectrum[1] = c_spectrum[0];
    if (matrix_convolution)
        c_spectrum[1] += bigfftsize;
    
    float* c_time = c_spectrum[1] + bigfftsize;
    float* windowed_time_domain = c_time + 2 * freq_res;
    
    unsigned cross_pol = 1;
    if (matrix_convolution)
        cross_pol = 2;
    
    if (verbose)
        cerr << "dsp::Filterbank::transformation enter main loop" <<
        " cpol=" << cross_pol << " npol=" << input->get_npol() <<
        " npart=" << npart  << endl;
    if (engine) {
        if (verbose)
            cerr << "have engine"<<endl;
    }
    
    // number of floats to step between input to filterbank
    const unsigned long in_step = nsamp_step * input->get_ndim();
    
    // points kept from each small fft
    const unsigned nkeep = freq_res - nfilt_tot;
    
    // number of floats to step between output from filterbank
    const unsigned long out_step = nkeep * 2;
    
    // counters
    unsigned ipt, ipol, jpol, ichan;
    uint64_t ipart;
    
    const unsigned npol = input->get_npol();
    
    // offsets into input and output
    uint64_t in_offset, out_offset;
    
    // some temporary pointers
    float* time_dom_ptr = NULL;  
    float* freq_dom_ptr = NULL;
    
    // do a 64-bit copy
    uint64_t* data_into = NULL;
    uint64_t* data_from = NULL;
    
    // /////////////////////////////////////////////////////////////////////
    //
    // PERFORM FILTERBANK VIA ENGINE (e.g. on GPU)
    //
    // /////////////////////////////////////////////////////////////////////
    if (engine)
    {
        engine->set_scratch(c_spectrum[0]);
        engine->perform (input, output, npart, in_step, out_step);
        if (Operation::record_time)
            engine->finish ();
    }
    //  cerr << "output ndat=" <<output->get_ndat() << " output ptr=" << output->get_datptr(0,0) << endl;
    
    // /////////////////////////////////////////////////////////////////////
    //
    // PERFORM FILTERBANK DIRECTLY (CPU)
    //
    // /////////////////////////////////////////////////////////////////////
    else
    {
        for (unsigned input_ichan=0; input_ichan<input->get_nchan(); input_ichan++)
        {
            
            for (ipart=0; ipart<npart; ipart++)
            {
#ifdef _DEBUG
                cerr << "ipart=" << ipart << endl;
#endif
                in_offset = ipart * in_step;
                out_offset = ipart * out_step;
                
                for (ipol=0; ipol < npol; ipol++)
                {
                    for (jpol=0; jpol<cross_pol; jpol++)
                    {
                        if (matrix_convolution)
                            ipol = jpol;
                        
                        time_dom_ptr = const_cast<float*>(input->get_datptr (input_ichan, ipol));
                        
                        time_dom_ptr += in_offset;
                        
                        if (apodization)
                        {
                            apodization -> operate (time_dom_ptr, windowed_time_domain);
                            time_dom_ptr = windowed_time_domain;
                        }
                        if (input->get_state() == Signal::Nyquist)
                            forward->frc1d (nsamp_fft, c_spectrum[ipol], time_dom_ptr);
                        else
                            forward->fcc1d (nsamp_fft, c_spectrum[ipol], time_dom_ptr);
                        
                        
                    }
                    
                    if (matrix_convolution)
                    {
                        if (passband)
                            passband->integrate (c_spectrum[0], c_spectrum[1], input_ichan);
                        
                        // cross filt can be set only if there is a response
                        response->operate (c_spectrum[0], c_spectrum[1]);
                    }
                    else
                    {
                        if (passband)
                            passband->integrate (c_spectrum[ipol], ipol, input_ichan);
                        
                        if (response)
                            response->operate (c_spectrum[ipol], ipol,
                                               input_ichan*nchan_subband, nchan_subband);
                    }
                    
                    for (jpol=0; jpol<cross_pol; jpol++)
                    {
                        if (matrix_convolution)
                            ipol = jpol;
                        
                        if (freq_res == 1)
                        {
                            data_from = (uint64_t*)( c_spectrum[ipol] );
                            for (ichan=0; ichan < nchan_subband; ichan++)
                            {
                                data_into = (uint64_t*)( output->get_datptr (input_ichan*nchan_subband+ichan, ipol) + out_offset );
                                
                                *data_into = data_from[ichan];
                            }
                            continue;
                        }
                        
                        
                        // freq_res > 1 requires a backward fft into the time domain
                        // for each channel
                        
                        unsigned jchan = input_ichan * nchan_subband;
                        freq_dom_ptr = c_spectrum[ipol];
                        
                        for (ichan=0; ichan < nchan_subband; ichan++)
                        {
                            backward->bcc1d (freq_res, c_time, freq_dom_ptr);
                            
                            freq_dom_ptr += freq_res*2;
                            
                            data_into = (uint64_t*)( output->get_datptr (jchan+ichan, ipol) + out_offset);
                            data_from = (uint64_t*)( c_time + nfilt_pos*2 );  // complex nos.
                            
                            for (ipt=0; ipt < nkeep; ipt++)
                                data_into[ipt] = data_from[ipt];
                            
                        } // for each output channel
                        
                    } // for each cross poln
                    
                } // for each polarization
                
            } // for each big fft (ipart)
            
        } // for each input channel
        
    } // if no engine (on CPU)
    
    if (Operation::record_time && engine)
        engine->finish ();
    
    if (verbose)
        cerr << "dsp::Filterbank::transformation return with output ndat="
        << output->get_ndat() << endl;
}

