//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/PhaseLockedFilterbank.h,v $
   $Revision: 1.5 $
   $Date: 2011/04/28 23:30:12 $
   $Author: demorest $ */

#ifndef __baseband_dsp_PhaseLockedFilterbank_h
#define __baseband_dsp_PhaseLockedFilterbank_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/PhaseSeries.h"
#include "dsp/TimeDivide.h"
#include "dsp/FZoom.h"

namespace Pulsar {
  class Predictor;
}

namespace dsp {

  //! Performs FFT at specific pulse phase windows
  /*! This class is particularly useful when maximum frequency resolution
    is required in dynamic spectra. */
  class PhaseLockedFilterbank 
    : public Transformation <TimeSeries, PhaseSeries> {

  public:

    //! Default constructor
    PhaseLockedFilterbank ();

    //! Set the number of channels into which the input will be divided
    void set_nchan (unsigned nchan);

    //! Get the number of channels into which the input will be divided
    unsigned get_nchan () const { return nchan; }

    //! Choose phase bins and period to obtain desired channel bandwidth
    void set_goal_chan_bw (double chanbw_mhz);

    //! Set the number of pulse phase windows in which to compute spectra
    void set_nbin (unsigned nbin);

    //! Get the number of pulse phase windows in which to compute spectra
    unsigned get_nbin () const { return nbin; }

    //! Set the number of polarization to compute
    void set_npol (unsigned npol);

    //! Get the number of polarizations
    unsigned get_npol () const { return npol; }

    //! Overlap windows to consume all time samples in a phase window
    void set_overlap (bool overlap);

    //! Has a folding predictor been set?
    bool has_folding_predictor() const { return bin_divider.get_predictor(); }

    //! Get the predictor
    const Pulsar::Predictor* get_folding_predictor() const
      { return bin_divider.get_predictor(); }

    //! The phase divider
    TimeDivide bin_divider;

    //! Get pointer to the output
    PhaseSeries* get_result() const;

    //! Normalize the spectra by the hits array
    void normalize_output ();

    //! Reset the output
    void reset ();

    //! Finalize anything
    virtual void finish ();

    void combine (const Operation*);

    void set_zoom(Reference::To<FZoom> zoom) { fzoom = zoom; }

    //! Return cached FZoom; use smart pointer in case it's 
    Reference::To<FZoom> get_zoom() { return fzoom; }

    class Engine;

    void set_engine (Engine*);

  protected:

    //! Perform the convolution transformation on the input TimeSeries
    virtual void transformation ();

    //! Number of channels into which the input will be divided
    unsigned nchan;

    //! Number of pulse phase windows in which to compute spectra
    unsigned nbin;

    //! Number of polarization products to compute
    unsigned npol;

    //! Goal channel bandwidth
    double goal_chan_bw;

    //! Flag set when built
    bool built;

    //! Prepare internal variables
    void prepare ();

    //! Set the idat_start and ndat_fold attributes
    virtual void set_limits (const Observation* input);

    //! Internal:  time to start processing data
    uint64_t idat_start;

    //! Internal:  number of samples to process
    uint64_t ndat_fold;

    //! Repeat and average FFTs within phase window if sufficient samples
    bool overlap;

    Reference::To<FZoom> fzoom;

    Reference::To<Engine> engine;

  };

  class PhaseLockedFilterbank::Engine : public OwnStream
  {
  public:

    virtual void prepare ( const TimeSeries* in, unsigned ) = 0;

    virtual void setup_phase_series (PhaseSeries*) = 0;

    virtual void sync_phase_series (PhaseSeries*) = 0;

    virtual void fpt_process (const TimeSeries* in,
        unsigned nblock, unsigned block_advance,
        unsigned phase_bin, unsigned idat_start,
        double* total_integrated) = 0;

    void do_nothing () {;}

  };
  
}

#endif
