/***************************************************************************
 *
 *   Copyright (C) 2007-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/LoadToFil.h"

#include "dsp/IOManager.h"
#include "dsp/Unpacker.h"
#include "dsp/ExcisionUnpacker.h"

#include "dsp/FilterbankEngineCPU.h"
#include "dsp/TFPFilterbank.h"
#include "dsp/Filterbank.h"
#include "dsp/Detection.h"

#include "dsp/SampleDelay.h"
#include "dsp/DedispersionSampleDelay.h"

#include "dsp/FScrunch.h"
#include "dsp/TScrunch.h"
#include "dsp/PScrunch.h"
#include "dsp/PolnSelect.h"

#include "dsp/Rescale.h"

#include "dsp/SigProcDigitizer.h"
#include "dsp/SigProcOutputFile.h"

#if HAVE_CFITSIO
#if HAVE_fits
#include "dsp/FITSFile.h"
#include "dsp/MultiFile.h"
#include "dsp/FITSUnpacker.h"
#endif
#endif

using namespace std;

bool dsp::LoadToFil::verbose = false;

static void* const undefined_stream = (void *) -1;

dsp::LoadToFil::LoadToFil (Config* configuration) try
{
  kernel = NULL;

  if (configuration)
    set_configuration (configuration);
}
catch (Error& error)
{
  throw error += "LoadToFil ctor Config*=" + tostring ((void*)configuration);
}

//! Run through the data
void dsp::LoadToFil::set_configuration (Config* configuration)
{
  SingleThread::set_configuration (configuration);
  config = configuration;
}

dsp::LoadToFil::Config::Config()
{
  can_thread = true;

  // block size in MB
  block_size = 2.0;

  order = dsp::TimeSeries::OrderTFP;
 
  // by default, do not denormalize using DAT_SCL and DAT_OFFS
  apply_FITS_scale_and_offset = false;

  filterbank.set_nchan(0);
  filterbank.set_freq_res(0);
  filterbank.set_convolve_when(Filterbank::Config::Never);

  dispersion_measure = 0;
  dedisperse = false;
  coherent_dedisp = false;

  excision_enable = true;

  tscrunch_factor = 0;
  fscrunch_factor = 0;

  npol = 1;

  poln_select = -1;

  rescale_seconds = 10.0;
  rescale_constant = false;

  scale_fac = 1.0;

  nbits = 2;

  // by default, time series weights are not used
  weighted_time_series = false;
}

void dsp::LoadToFil::Config::set_quiet ()
{
  SingleThread::Config::set_quiet();
  LoadToFil::verbose = false;
}

void dsp::LoadToFil::Config::set_verbose ()
{
  SingleThread::Config::set_verbose();
  LoadToFil::verbose = true;
}

void dsp::LoadToFil::Config::set_very_verbose ()
{
  SingleThread::Config::set_very_verbose();
  LoadToFil::verbose = true;
}

void dsp::LoadToFil::construct () try
{
  SingleThread::construct ();

  /*
    The following lines "wire up" the signal path, using containers
    to communicate the data between operations.
  */

  // set up for optimal memory usage pattern

  Unpacker* unpacker = manager->get_unpacker();

#if HAVE_CFITSIO && HAVE_fits

  if (config->apply_FITS_scale_and_offset &&
      manager->get_info()->get_machine() == "FITS")
  {
    FITSUnpacker* fun = dynamic_cast<FITSUnpacker*> (manager->get_unpacker());
    fun->apply_scale_and_offset (true);
  }

#endif

  if (!config->dedisperse && unpacker->get_order_supported (config->order))
    unpacker->set_output_order (config->order);


  // get basic information about the observation

  Observation* obs = manager->get_info();
  const unsigned nchan = obs->get_nchan ();
  const unsigned npol = obs->get_npol ();
  const unsigned ndim = obs->get_ndim ();

  if (verbose)
  {
    cerr << "Source = " << obs->get_source() << endl;
    cerr << "Frequency = " << obs->get_centre_frequency() << endl;
    cerr << "Bandwidth = " << obs->get_bandwidth() << endl;
    cerr << "Sampling rate = " << obs->get_rate() << endl;
  }

  obs->set_dispersion_measure( config->dispersion_measure );

  // the unpacked input will occupy nbytes_per_sample
  double nbytes_per_sample = sizeof(float) * nchan * npol * ndim;

#if HAVE_CFITSIO
#if HAVE_fits
  // if PSRFITS input, set block to exact size of FITS row
  // this is needed to keep in sync with the callback
  if (manager->get_info()->get_machine() == "FITS")
  {
    FITSFile* tmp = dynamic_cast<FITSFile*> (manager->get_input());
    uint64_t block_size;

    if (!tmp)
    {
      MultiFile* mfile = dynamic_cast<MultiFile*> (manager->get_input());
      if (mfile)
      {
        block_size = mfile->get_block_size();
        tmp = dynamic_cast<FITSFile*> ( mfile->get_loader() );
      }
    }
    else
      block_size = tmp->get_block_size();
    if (tmp)
    {
      unsigned samples_per_row = tmp->get_samples_in_row();
      uint64_t current_bytes = manager->set_block_size (samples_per_row);
      manager->set_maximum_RAM (current_bytes);
      manager->set_block_size (samples_per_row);
      if (verbose)
        cerr << "digifil: block_size=" << (current_bytes/1024/1024) << " MB "
          "(" << samples_per_row << " samp)" << endl;
    }
    else
      cerr << "digifil: WARNING have FITS input but cannot set block size properly." << endl;
  }
  else
#endif
#endif
  {
    double MB = 1024.0 * 1024.0;
    uint64_t nsample = uint64_t( config->block_size*MB / nbytes_per_sample );

    if (verbose)
      cerr << "digifil: block_size=" << config->block_size << " MB "
        "(" << nsample << " samp)" << endl;

    manager->set_block_size( nsample );
  }
  
  bool do_pscrunch = (config->npol==1) && (obs->get_npol() > 1) 
    && (config->poln_select < 0);

  TimeSeries* timeseries = unpacked;

  if ( config->poln_select >= 0 )
  {
    PolnSelect *pselect = new PolnSelect;

    pselect->set_ipol (config->poln_select);
    pselect->set_input (timeseries);
    pselect->set_output (timeseries);

    operations.push_back( pselect );
  }

  if (!obs->get_detected())
  {
    bool do_detection = false;

    config->coherent_dedisp = 
      (config->filterbank.get_convolve_when() == Filterbank::Config::During)
      && (config->dispersion_measure != 0.0);

    if ( config->filterbank.get_nchan() )
    {
      if (verbose)
        cerr << "digifil: creating " << config->filterbank.get_nchan()
             << " channel filterbank" << endl;

      if ( config->coherent_dedisp )
      {
        cerr << "digifil: using coherent dedispersion" << endl;

        kernel = new Dedispersion;

        if (config->filterbank.get_freq_res())
          kernel->set_frequency_resolution (config->filterbank.get_freq_res());

        kernel->set_dispersion_measure( config->dispersion_measure );

        // TODO other FFT length/etc options as implemented in LoadToFold1?

      }

      if ( config->filterbank.get_freq_res() 
          || config->coherent_dedisp 
          || (config->npol>2) )
      {
        cerr << "digifil: using convolving filterbank" << endl;

        filterbank = new Filterbank;

        filterbank->set_nchan( config->filterbank.get_nchan() );
        filterbank->set_input( timeseries );
        filterbank->set_output( timeseries = new_TimeSeries() );
        filterbank->set_engine (new FilterbankEngineCPU);

        if (kernel)
          filterbank->set_response( kernel );

        if ( config->filterbank.get_freq_res() ) 
          filterbank->set_frequency_resolution ( 
              config->filterbank.get_freq_res() );

        operations.push_back( filterbank.get() );
        do_detection = true;
      }
      else
      {
        filterbank = new TFPFilterbank;

        filterbank->set_nchan( config->filterbank.get_nchan() );
        filterbank->set_input( timeseries );
        filterbank->set_output( timeseries = new_TimeSeries() );
        filterbank->set_engine (new FilterbankEngineCPU);

        operations.push_back( filterbank.get() );
      }
    }
    else
      do_detection = true;

    if ( config->dedisperse )
    {
      if (verbose)
        cerr << "digifil: removing dispersion delays" << endl;

      SampleDelay* delay = new SampleDelay;

      delay->set_input (timeseries);
      delay->set_output (timeseries = new_TimeSeries() );
      delay->set_function (new Dedispersion::SampleDelay);

      operations.push_back( delay );
    }

    if (do_detection)
    {
      if (verbose)
        cerr << "digifil: creating detection operation (npol=" <<
          config->npol << ")" << endl;
  
      Detection* detection = new Detection;

      detection->set_input( timeseries );
      detection->set_output( timeseries = new_TimeSeries() );

      if (config->npol==1) 
        detection->set_output_state(Signal::Intensity);
      else if (config->npol==2)
        detection->set_output_state(Signal::PPQQ);
      else if (config->npol==3)
        detection->set_output_state(Signal::NthPower);
      else if (config->npol==4)
      {
        detection->set_output_state(Signal::Coherence);
        //detection->set_output_ndim(4);
      }

      // detection will do pscrunch if necessary
      do_pscrunch = false;

      operations.push_back( detection );
    }
  }

  else { // Data already detected
    if ( config->dedisperse )
    {
      if (verbose)
        cerr << "digifil: removing dispersion delays (post-detection)" << endl;

      SampleDelay* delay = new SampleDelay;

      delay->set_input (timeseries);
      delay->set_output (timeseries = new_TimeSeries() );
      delay->set_function (new Dedispersion::SampleDelay);

      operations.push_back( delay );
    }
  }

  if ( config->fscrunch_factor )
  {
    FScrunch* fscrunch = new FScrunch;
    
    fscrunch->set_factor( config->fscrunch_factor );
    fscrunch->set_input( timeseries );
    fscrunch->set_output( timeseries = new_TimeSeries() );

    operations.push_back( fscrunch );
  }

  if ( config->tscrunch_factor )
  {
    TScrunch* tscrunch = new TScrunch;
    
    tscrunch->set_factor( config->tscrunch_factor );
    tscrunch->set_input( timeseries );
    tscrunch->set_output( timeseries = new_TimeSeries() );

    operations.push_back( tscrunch );
  }
  
  if ( config->rescale_seconds )
  {
    if (verbose)
      cerr << "digifil: creating rescale transformation" << endl;

    Rescale* rescale = new Rescale;

    rescale->set_input (timeseries);
    rescale->set_output (timeseries);
    rescale->set_constant (config->rescale_constant);
    rescale->set_interval_seconds (config->rescale_seconds);

    operations.push_back( rescale );
  }

  if (do_pscrunch)
  {
    if (verbose)
      cerr << "digifil: creating pscrunch transformation" << endl;

    PScrunch* pscrunch = new PScrunch;
    pscrunch->set_input (timeseries);
    pscrunch->set_output (timeseries = new_TimeSeries());
    if (config->npol == 1)
      pscrunch->set_output_state(Signal::Intensity);
    else if (config->npol == 2)
      pscrunch->set_output_state(Signal::PPQQ);
    else if (config->npol == 4)
      pscrunch->set_output_state(Signal::Coherence);
    else
      throw Error (InvalidState, "dsp::LoadToFil::construct",
                  "output polarisation invalid %d", config->npol);
    operations.push_back( pscrunch );
  }

  if (verbose)
    cerr << "digifil: creating output bitseries container" << endl;

  BitSeries* bitseries = new BitSeries;

  if (verbose)
    cerr << "digifil: creating sigproc digitizer" << endl;

  SigProcDigitizer* digitizer = new SigProcDigitizer;
  digitizer->set_nbit (config->nbits);
  digitizer->set_input (timeseries);
  digitizer->set_output (bitseries);
  digitizer->set_scale (config->scale_fac);
  // If Rescale is not in use, the scale/offset settings in the digitizer do
  // not make sense, so this disables them:
  if (config->rescale_seconds == 0.0) digitizer->use_digi_scales(false);

  operations.push_back( digitizer );

  if (verbose)
    cerr << "digifil: creating sigproc output file" << endl;

  const char* output_filename = 0;
  if (!config->output_filename.empty())
    output_filename = config->output_filename.c_str();

  outputFile = new SigProcOutputFile (output_filename);
  outputFile->set_input (bitseries);

  operations.push_back( outputFile.get() );
}
catch (Error& error)
{
  throw error += "dsp::LoadToFil::construct";
}

void dsp::LoadToFil::prepare () try
{
  SingleThread::prepare();

  // Check that block size is sufficient for the filterbanks,
  // increase it if not.
  if (filterbank && verbose)
    cerr << "digifil: filterbank minimum samples = " 
      << filterbank->get_minimum_samples() 
      << endl;

  if (filterbank) {
    if (filterbank->get_minimum_samples() > 
        manager->get_input()->get_block_size())
    {
      cerr << "digifil: increasing data block size from " 
        << manager->get_input()->get_block_size()
        << " to " << filterbank->get_minimum_samples() 
        << " samples" << endl;
      manager->set_block_size( filterbank->get_minimum_samples() );
    }
  }

  if (config->excision_enable==false)
  {
    dsp::ExcisionUnpacker* excision;
    excision = dynamic_cast<dsp::ExcisionUnpacker*> (manager->get_unpacker());
    if (excision)
      excision->set_cutoff_sigma(0.0);
  }

}
catch (Error& error)
{
  throw error += "dsp::LoadToFil::prepare";
}
