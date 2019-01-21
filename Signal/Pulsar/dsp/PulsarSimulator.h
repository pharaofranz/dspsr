
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

namespace Pulsar
{
  class PolnProfile;
  class Predictor;
}

namespace dsp {

  //! An object that can fill a TimeSeries with simulated pulsar signal
  class PulsarSimulator : public Operation
  {
    Reference::To<Pulsar::PolnProfile> profile;
    Reference::To<Pulsar::Predictor> predictor;

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
