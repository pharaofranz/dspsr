
/***************************************************************************
 *
 *   Copyright (C) 2018 by Willem van Straten and Axl Rogers
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
//-*-C++-*-

// dspsr/Signal/Pulsar/dsp/PulsarSimulator.h

#ifndef __PulsarSimulator_h
#define __PulsarSimulator_h

#include "dsp/Operation.h"

namespace dsp {

  //! An object that can fill a TimeSeries with simulated pulsar signal
  class PulsarSimulator : public Operation
  {

  public:
    
    //! Constructor
    PulsarSimulator ();
    
    //! Destructor
    virtual ~PulsarSimulator ();

    //! Do something
    void operation ();

  };

}

#endif // !defined(__PulsarSimulator_h)
