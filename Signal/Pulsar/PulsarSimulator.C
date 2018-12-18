/***************************************************************************
 *
 *   Copyright (C) 2018 by Willem van Straten and Axl Rogers
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PulsarSimulator.h"

#include "Pulsar/Predictor.h"

using namespace std;
using namespace Pulsar;

//! Constructor
dsp::PulsarSimulator::PulsarSimulator () : Operation ("PulsarSimulator")
{
}

//! Destructor
dsp::PulsarSimulator::~PulsarSimulator ()
{
}

//! the main operation
void dsp::PulsarSimulator::operation ()
{
  cerr << "dsp::PulsarSimulator::operation not implemented" << endl;
}


