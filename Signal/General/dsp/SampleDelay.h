//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/SampleDelay.h

#ifndef __baseband_dsp_SampleDelay_h
#define __baseband_dsp_SampleDelay_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"

#include <vector>

namespace dsp {

  class SampleDelayFunction;

  class SampleDelay : public Transformation<TimeSeries,TimeSeries> {

  public:

    //! Default constructor
    SampleDelay ();

    //! Set the delay function
    void set_function (SampleDelayFunction*);

    //! Computes the total delay and prepares the input buffer
    void prepare ();

    //! Prepares the output data buffer
    void prepare_output ();

    //! Get the minimum number of samples required for operation
    uint64_t get_minimum_samples () { return total_delay; }

    //! Applies the delays to the input
    void transformation ();

    //! Get the total delay (in samples)
    uint64_t get_total_delay () const;

    //! Get the zero delay (in samples)
    int64_t get_zero_delay () const;

    //! Engine used to perform application of delays
    class Engine;
    void set_engine (Engine*);

  protected:

    //! The total delay (in samples)
    uint64_t total_delay;

    //! The zero delay (in samples)
    int64_t zero_delay;

    //! Interface to alternate processing engine (e.g. GPU)
    Reference::To<Engine> engine;

    //! Flag set when delays have been initialized
    bool built;

    //! Initalizes the delays
    void build ();

    //! The sample delay function
    Reference::To<SampleDelayFunction> function;

  };

  class SampleDelay::Engine : public Reference::Able
  {
  public:

    virtual void set_delays (unsigned npol, unsigned nchan, int64_t zero_delay, SampleDelayFunction * function) = 0;

    virtual void retard(const TimeSeries* in, TimeSeries* out) = 0;

  };

}

#endif
