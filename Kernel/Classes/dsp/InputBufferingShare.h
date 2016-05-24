//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __InputBufferingShare_h
#define __InputBufferingShare_h

#include "dsp/InputBuffering.h"

class ThreadContext;

namespace dsp {

  //! Buffers the Transformation input
  class InputBuffering::Share : public BufferingPolicy {

  public:
    
    //! Default constructor
    Share ();

    //! Constructor
    Share (InputBuffering*, HasInput<TimeSeries>*);

    //! Destructor
    ~Share ();

    //! Sort-of clone
    Share* clone (HasInput<TimeSeries>*);

    //! Perform all buffering tasks required before transformation
    void pre_transformation ();
    
    //! Perform all buffering tasks required after transformation
    void post_transformation ();
    
    //! Set the first sample to be used from the input next time
    void set_next_start (uint64_t next_start_sample);

    //! Get the next contiguous sample following the current buffer
    virtual int64_t get_next_contiguous () const;
    
    //! Set the minimum number of samples that can be processed
    void set_minimum_samples (uint64_t samples);

    //! Detect non-contiguous input and reset buffer accordingly
    void set_round_robin_mode (bool enable=true) {round_robin_mode=enable;}

  protected:
    
    //! The target with input TimeSeries to be buffered
    HasInput<TimeSeries>* target;
    
    //! The shared input buffering policy
    Reference::To<InputBuffering> buffer;
    
    //! The reserve manager
    Reference::To<Reserve> reserve;

    //! Multi-threaded context information
    ThreadContext* context;

    //! Owner of the context attribute;
    bool context_owner;

    //! Run in round-robin mode
    bool round_robin_mode;

  };

}

#endif // !defined(__InputBuffering_h)
