/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <stdlib.h>
#include <string.h>
#include "dsp/Apodization.h"
#include "dsp/CyclicFold.h"
#include "FTransform.h"

#include <assert.h>
#include <fstream>
using namespace std;

dsp::CyclicFold::CyclicFold()
{
  nlag = 0;
  mover = 1;
  npol = 0;
  set_name("CyclicFold");
}

dsp::CyclicFold::~CyclicFold()
{
}

void dsp::CyclicFold::prepare () 
{

  if (verbose) 
    cerr << "dsp::CyclicFold::prepare" << endl;

  // Init engine if it's not already
  if (!engine) 
    set_engine (new CyclicFoldEngine);
  else
    setup_engine();

  Fold::prepare ();
}

void dsp::CyclicFold::set_engine (Fold::Engine *_engine)
{
  if (verbose) 
    cerr << "dsp::CyclicFold::set_engine" << endl;

  Fold::set_engine(_engine);

  if (engine)
    setup_engine();
}

void dsp::CyclicFold::setup_engine ()
{
  if (verbose) 
    cerr << "dsp::CyclicFold::setup_engine" << endl;

  // Check for appropriate engine type
  CyclicFoldEngine *cfe = dynamic_cast<CyclicFoldEngine *>(engine.get());
  if (!cfe)
    throw Error (InvalidState, "dsp::CyclicFold::setup_engine",
        "Folding engine is not a CyclicFoldEngine");

  // Set params in fold engine
  cfe->set_nlag (nlag);
  cfe->set_mover (mover);
  cfe->set_npol (npol);
  cfe->set_profiles (output);
}

void dsp::CyclicFold::check_input() try
{

  if (input->get_detected ())
    throw Error (InvalidParam, "dsp::CyclicFold::check_input",
		 "input is already detected");

}
catch (Error &error) 
{
  throw error += "dsp::CyclicFold::check_input";
}

void dsp::CyclicFold::prepare_output() try
{
  if (verbose) 
    cerr << "dsp::CyclicFold::prepare_output start" << endl;

  // Need to do steps that happen in PhaseSeries::mixable
  // but without constraint that input chans, polns, etc need to
  // match exactly in input/output, since CyclicFold will 
  // alter the state of the data.

  const TimeSeries *in = get_input();
  PhaseSeries *out = get_output();

  MJD obsStart = in->get_start_time() + double (idat_start) / in->get_rate();
  MJD obsEnd;
  if (ndat_fold == 0) 
    obsEnd = in->get_end_time();
  else
    obsEnd = obsStart + double (ndat_fold) / in->get_rate();

  if (out->get_integration_length() == 0.0) 
  {
    if (verbose)
      cerr << "dsp::CyclicFold::prepare_output reset" << endl;

    out->Observation::operator = (*in);

    // Assumes complex lags in 'c2r' format
    unsigned nchan_out = (2*get_nlag() - 2)/get_mover();
    const unsigned nchan_in = in->get_nchan();
    out->set_nchan(nchan_out*nchan_in);

    if (verbose)
      cerr << "dsp::CyclicFold::prepare_output npol=" << npol << endl;

    out->set_npol(npol);
    out->set_ndim(1);
    if (npol==1) 
    {
      if (in->get_npol()==1)
        out->set_state(Signal::PP_State);
      else
        out->set_state(Signal::Intensity);
    }
    else if (npol==2) 
      out->set_state(Signal::PPQQ);
    else if (npol==4)
      out->set_state(Signal::Coherence);
    else
      throw Error (InvalidParam, "dsp::CyclicFold::prepare_output",
          "invalid npol=%d", npol);

    out->set_order (TimeSeries::OrderFPT);

    out->set_end_time(obsEnd);
    out->set_start_time(obsStart);

    uint64_t backup_ndat_total = out->get_ndat_total();

    out->resize (folding_nbin);
    out->zero();

    out->ndat_total = backup_ndat_total;

    if (verbose)
      cerr << "dsp::CyclicFold::prepare_output in->get_nchan()=" << in->get_nchan() << endl;

    if (in->get_nchan() == 1) {
      if (verbose)
	cerr << "dsp::CyclicFold::prepare_output setting swap because only one input channel" << endl;
      out->set_swap(true);
    }

    if (in->get_nchan() > 1 && in->get_swap() == false) {
      if (verbose)
	cerr << "dsp::CyclicFold::prepare_output setting nsub swap to " << out->get_nchan() << endl;
      out->set_nsub_swap (out->get_nchan());
    }

    if (in->get_dual_sideband()) {
      if (verbose)
	cerr << "dsp::CyclicFold::prepare_output input appears to be dual sideband" << endl;    
    }

    return;
  }

  // TODO some kind of test to replicate the desired parts of 
  // PhaseSeries::combinable?  Are these tests even needed in this
  // context?

  if (out->get_nbin() != folding_nbin)
    throw Error (InvalidState, "dsp::CyclicFold::prepare_output",
        "Output nbin not equal to current folding nbin");

  out->set_end_time (max(out->get_end_time(), obsEnd));
  out->set_start_time (min(out->get_start_time(), obsStart));

}
catch (Error &error)
{
  throw error += "dsp::CyclicFold::prepare_output";
}

dsp::CyclicFoldEngine::CyclicFoldEngine()
{
  nbin=npol=ndim=0;
  npol_out = 0;
  nlag = 0;
  mover = 1;
  binplan_size = 0;
  binplan[0] = NULL;
  binplan[1] = NULL;
  lagdata_size = 0;
  lagdata = NULL;
  idat_start = 0;
  lag2chan = NULL;
  use_set_bins = false;
}

dsp::CyclicFoldEngine::~CyclicFoldEngine()
{
	if (parent->verbose) {
		cerr << "dsp::CyclicFoldEngine::~CyclicFoldEngine freeing binplan" <<endl;
	}
  if (binplan[0]) delete [] binplan[0];
  if (binplan[1]) delete [] binplan[1];
  if (parent->verbose) {
	  cerr << "dsp::CyclicFoldEngine::~CyclicFoldEngine freeing lagdata" <<endl;
  }
  if (lagdata) delete [] lagdata;
  if (parent->verbose) {
	  cerr << "dsp::CyclicFoldEngine::~CyclicFoldEngine done" <<endl;
  }
}

void dsp::CyclicFoldEngine::set_nlag(unsigned _nlag) 
{
  if (nlag != _nlag)
  {
    if (parent->verbose)
      cerr << "dsp::CyclicFoldEngine::set_nlag nlag=" << _nlag << endl;
    // FIXME: it's inefficient to FFT the whole thing and downsample
    // better to FFT 1/mover at a time and add them, or just add them
    // and then FFT a small hunk
    // this will work, though, and be easier to implement
    unsigned nchan_spec = 2*_nlag - 2;
    lag2chan = FTransform::Agent::current->get_plan (nchan_spec,
        FTransform::bcr);
    nlag = _nlag;
  }
}

void dsp::CyclicFoldEngine::set_mover(unsigned _mover) 
{
  if (mover != _mover)
  {
    if (parent->verbose)
      cerr << "dsp::CyclicFoldEngine::set_mover mover=" << _mover << endl;
    // FIXME: should recompute nlag from nchan
    mover = _mover;

  }
}

void dsp::CyclicFoldEngine::set_ndat (uint64_t _ndat, uint64_t _idat_start)
{
  setup();

  if (parent->verbose)
    cerr << "dsp::CyclicFoldEngine::set_ndat ndat=" << _ndat << endl;

  if (_ndat > binplan_size) {

    if (parent->verbose)
      cerr << "dsp::CyclicFoldEngine::set_ndat alloc binplan" << endl;

    if (binplan[0]) delete [] binplan[0];
    if (binplan[1]) delete [] binplan[1];

    binplan[0] = new unsigned [_ndat];
    binplan[1] = new unsigned [_ndat];

    binplan_size = _ndat;
  }

  ndat_fold = _ndat;
  idat_start = _idat_start;

  if (parent->verbose)
    cerr << "dsp::CyclicFoldEngine::set_ndat "
      << "nlag=" << nlag << " "
      << "mover=" << mover << " "
      << "nbin=" << nbin << " "
      << "npol=" << npol_out << " "
      << "ndim=" << ndim << " "
      << "nchan=" << nchan << endl;

  uint64_t _lagdata_size = nlag * nbin * npol_out * ndim * nchan;

  if (parent->verbose)
    cerr << "dsp::CyclicFoldEngine::set_ndat lagdata_size=" << _lagdata_size << endl;

  if (_lagdata_size > lagdata_size) {
    if (parent->verbose)
      cerr << "dsp::CyclicFoldEngine::set_ndat alloc lagdata" << endl;
    if (lagdata) delete [] lagdata;
    lagdata = new float [_lagdata_size];
    lagdata_size = _lagdata_size;
    memset(lagdata, 0, sizeof(float)*lagdata_size);
  }

}

void dsp::CyclicFoldEngine::zero ()
{
	if (parent->verbose) {
		cerr << "dsp::CyclicFoldEngine::zero: zeroing profiles " << endl;
	}
  if (get_profiles()) 
      get_profiles()->zero();
  if (lagdata && lagdata_size>0) 
    memset(lagdata, 0, sizeof(float)*lagdata_size);
}

void dsp::CyclicFoldEngine::set_bin (uint64_t idat, double ibin, 
    double bins_per_sample) 
{
  // Lag folding for cyclic spectra needs phase evaluated at 
  // half-sample spacing.
  binplan[0][idat-idat_start] = unsigned(ibin);
  unsigned ibin1 = unsigned (ibin + 0.5*bins_per_sample);
  binplan[1][idat-idat_start] = ibin1 % nbin;
}
uint64_t dsp::CyclicFoldEngine::get_bin_hits (int ibin)
{
	return 0; // Fix this
}
uint64_t dsp::CyclicFoldEngine::set_bins (double phi, double phase_per_sample,uint64_t _ndat, uint64_t idat_start)
{
	return 0;
}

dsp::PhaseSeries* dsp::CyclicFoldEngine::get_profiles ()
{
  return out;
}

static inline void complex_conj_mult_acc(float *d_out, const float *in0,
    const float *in1) 
{
  d_out[0] += in0[0]*in1[0] + in0[1]*in1[1];
  d_out[1] += in0[1]*in1[0] - in0[0]*in1[1];
}

static inline void mult_acc(float *d_out, const float *in0,
    const float *in1)
{
  *d_out += (*in0) * (*in1);
}

float* dsp::CyclicFoldEngine::get_lagdata_ptr(unsigned ichan, 
    unsigned ipol, unsigned ibin)
{
  // Store data in internal lagdata array in order (fast->slow):
  //   lag, freq, poln, bin
  return lagdata + ndim*(ibin*npol_out*nchan*nlag
    + ipol*nchan*nlag
    + ichan*nlag);
}

void dsp::CyclicFoldEngine::fold ()
{
  const TimeSeries* in = parent->get_input();

  if (in->get_order() != TimeSeries::OrderFPT) 
    throw Error (InvalidState, "dsp::CyclicFoldEngine::fold",
        "Only FPT input order is currently supported.");

  if (in->get_state() != Signal::Analytic)
    throw Error (InvalidState, "dsp::CyclicFoldEngine::fold",
        "Only Analytic input data is currently supported");

  setup();

  if (parent->verbose) {
	  cerr << "dsp::CyclicFoldEngine::fold entering fold loop" << endl;
	  cerr << "idat_start=" << idat_start << " ndat_fold=" << ndat_fold << endl;
  }

  // Ignore blocks which don't contain enough data (avoids
  // triggering the ibin<nbin assertion below).
  if (ndat_fold <= nlag)
  {
    if (parent->verbose)
      cerr << "dsp::CyclicFoldEngine::fold ignoring a short data block "
        << "(ndat_fold=" << ndat_fold << " <= nlag=" << nlag << ")"
        << endl;
    return;
  }

  if (in->get_npol() == 2) 
  {

    for (unsigned ichan=0; ichan<nchan; ichan++) 
    {
      const float *pol0_in = in->get_datptr(ichan,0) + ndim*idat_start;
      const float *pol1_in = in->get_datptr(ichan,1) + ndim*idat_start;

#if 0
      ofstream fbin;
      fbin.open("cpuprefold.dat", ios::binary | ios::app);
      for (int nn=0; nn < ndat_fold*ndim; nn++){
    	  fbin.write((char *)(&(pol0_in[nn])),sizeof(float));
      }
      for (int nn=0; nn < ndat_fold*ndim; nn++){
    	  fbin.write((char *)(&(pol1_in[nn])),sizeof(float));
      }
      cerr << "done, dumping precyclic cpu, closing files" << endl;
      fbin.close();
#endif

      for (uint64_t idat=0; idat<ndat_fold-nlag; idat++) 
      {
        for (unsigned ilag=0; ilag<nlag; ilag++) 
        {
          const unsigned ibin = binplan[ilag%2][idat+ilag/2];
          assert(ibin<nbin);
          if (npol_out==1) 
          {
            complex_conj_mult_acc(get_lagdata_ptr(ichan,0,ibin) + ndim*ilag,
                pol0_in + ndim*idat, pol0_in + ndim*(idat+ilag));
            complex_conj_mult_acc(get_lagdata_ptr(ichan,0,ibin) + ndim*ilag,
                pol1_in + ndim*idat, pol1_in + ndim*(idat+ilag));
          }
          else
          {
            complex_conj_mult_acc(get_lagdata_ptr(ichan,0,ibin) + ndim*ilag,
                pol0_in + ndim*idat, pol0_in + ndim*(idat+ilag));
            complex_conj_mult_acc(get_lagdata_ptr(ichan,1,ibin) + ndim*ilag,
                pol1_in + ndim*idat, pol1_in + ndim*(idat+ilag));
          }
          if (npol_out==4) 
          {
            complex_conj_mult_acc(get_lagdata_ptr(ichan,2,ibin) + ndim*ilag,
                pol0_in + ndim*idat, pol1_in + ndim*(idat+ilag));
            complex_conj_mult_acc(get_lagdata_ptr(ichan,3,ibin) + ndim*ilag,
                pol1_in + ndim*idat, pol0_in + ndim*(idat+ilag));
          }
        } // lag
      } // dat
    } // chan

  }

  else if (in->get_npol() == 1) 
  {

    for (unsigned ichan=0; ichan<nchan; ichan++) 
    {
      const float *pol0_in = in->get_datptr(ichan,0) + ndim*idat_start;
      for (uint64_t idat=0; idat<ndat_fold-nlag; idat++) 
      {
        for (unsigned ilag=0; ilag<nlag; ilag++) 
        {
          const unsigned ibin = binplan[ilag%2][idat+ilag/2];
          assert(ibin<nbin);
          complex_conj_mult_acc(get_lagdata_ptr(ichan,0,ibin) + ndim*ilag,
              pol0_in + ndim*idat, pol0_in + ndim*(idat+ilag));
        } // lag
      } // dat
    } // chan

  }

  else
    throw Error (InvalidParam, "dsp::CyclicFoldEngine::fold", 
        "Invalid npol=%d", npol);

  synchronized = false;
}

void dsp::CyclicFoldEngine::synch (PhaseSeries* out)
{
  // FFT lag data to channel data and arrange it correctly in
  // the output PhaseSeries

  if (parent->verbose)
    cerr << "dsp::CyclicFoldEngine::synch" << endl;

  if (synchronized)
    return;

  if (parent->verbose)
    cerr << "dsp::CyclicFoldEngine::synch calling bcr FFT" << endl;

  // NOTE: this spectrum is oversampled by a factor mover
  unsigned nchan_spec = 2*nlag - 2;
  float *spec = new float[nchan_spec];

  unsigned nchan_spec_real = nchan_spec/mover;

#if 0 
  // In the 4-pol case, we need to sum/diff the lag functions to get
  // cross-terms equivalent to the usual Coherence products.
  // TODO still testing this..
  if (npol_out == 4)
  {
    for (unsigned ibin=0; ibin<nbin; ibin++) 
    {
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        float *lags2 = get_lagdata_ptr(ichan, 2, ibin);
        float *lags3 = get_lagdata_ptr(ichan, 3, ibin);
        for (unsigned ilag=0; ilag<nlag; ilag+=2) 
        {
          float pos_r = lags2[ilag];
          float pos_i = lags2[ilag+1];
          float neg_r = lags3[ilag];
          float neg_i = lags3[ilag+1];
          lags2[ilag]   = 0.5*(pos_r + neg_r);
          lags2[ilag+1] = 0.5*(pos_i - neg_i);
          lags3[ilag]   = 0.5*(pos_r - neg_r);
          lags3[ilag+1] = 0.5*(pos_i + neg_i);
        }
      }
    }
  }
#endif

#if 0
  ofstream fbin;
  fbin.open("cyclic.dat", ios::binary | ios::app);
  for (int nn=0; nn < lagdata_size; nn++){
	  fbin.write((char *)(&(lagdata[nn])),sizeof(float));

  }
  cerr << "done, dumping cyclic, closing files" << endl;
  fbin.close();
#endif

  if (parent->verbose)
    cerr << "dsp::CyclicFoldEngine::synch folding and saving spectra" << endl;
  for (unsigned ibin=0; ibin<nbin; ibin++) 
  {
    for (unsigned ipol=0; ipol<npol_out; ipol++) 
    {
      for (unsigned ichan=0; ichan<nchan; ichan++) 
      {
        float *lags = get_lagdata_ptr(ichan, ipol, ibin);

	if (mover>1) {
	  assert(ndim==2);
	  lags[0] *= 1;
	  lags[1] *= 1;
	  for (unsigned ilag=1; ilag<nlag; ilag++) {
	    float x = (M_PI/3)*mover*ilag/((float)(2*nlag-2));
	    // FIXME: this is horrible.
	    // This is a manual implementation of a Hanning window.
	    // Using the built-in window functions led to mysterious
	    // failures to write the second file.
	    // In any case dsp::Apodization::Parzen is not actually 
	    // a Parzen window.
	    float y = 0.5*(1+cos(2*M_PI*float(ilag)/float(2*nlag)));
	    float f = y*sin(x)/x;
	    lags[2*ilag] *= f;
	    lags[2*ilag+1] *= f;
	  }
	  
	}

        lag2chan->bcr1d(nchan_spec, spec, lags);
        for (unsigned schan=0; schan<nchan_spec_real; schan++) 
        {
          float* phasep = out->get_datptr(ichan*nchan_spec_real+schan,ipol);
	  // downsample by mover
          phasep[ibin] = spec[schan*mover];
        }
      }
    }
  }
  if (parent->verbose)
    cerr << "dsp::CyclicFoldEngine::synch finished computing spectra" << endl;

  delete [] spec;

  synchronized = true;
}
