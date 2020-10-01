/***************************************************************************
 *
 *   Copyright (C) 2020 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PlasmaResponseProduct.h"

using namespace std;

dsp::PlasmaResponseProduct::PlasmaResponseProduct ()
{
}

//! Destructor
dsp::PlasmaResponseProduct::~PlasmaResponseProduct ()
{
}

//! Create a product to match the input
void dsp::PlasmaResponseProduct::prepare (const Observation* obs, unsigned nchan)
{
  if (verbose)
    cerr << "dsp::PlasmaResponseProduct::prepare nchan=" << nchan 
         << " nresp=" << response.size() << endl;

  for (unsigned iresp=0; iresp < response.size(); iresp++)
  {
    response[iresp]->prepare (obs, nchan);
  }

  PlasmaResponse::prepare (obs, nchan);
}

double dsp::PlasmaResponseProduct::delay_time (double freq) const
{
  double time = 0.0;
  
  for (unsigned iresp=0; iresp < response.size(); iresp++)
    time += response[iresp]->delay_time (freq);

  if (verbose)
    cerr << "dsp::PlasmaResponseProduct::delay_time freq=" << freq
         << " nresp=" << response.size() << " time=" << time << endl;

  return time;
}


//! Add a response to the product
void dsp::PlasmaResponseProduct::add_response (PlasmaResponse* _response)
{
  response.push_back (_response);
  _response->changed.connect (this, &PlasmaResponseProduct::set_component_changed);
}

    //! Called when a component has changed
void dsp::PlasmaResponseProduct::set_component_changed (const Response& response)
{
  set_not_built ();
}


using namespace std;

void dsp::PlasmaResponseProduct::build (unsigned _ndat, unsigned _nchan)
{
  if (verbose)
    cerr << "dsp::PlasmaResponseProduct::build nchan=" << _nchan 
         << " nfilt=" << _ndat << endl;

  for (unsigned iresp=0; iresp < response.size(); iresp++)
  {
    if (verbose)
      cerr << "dsp::PlasmaResponseProduct::build iresp=" << iresp << endl;
    response[iresp]->build (_ndat, _nchan);
  }

  if (verbose)
    cerr << "dsp::PlasmaResponseProduct::build copy response[0]->nchan=" 
         << response[0]->get_nchan() << " nfilt=" << response[0]->get_ndat() << endl;

  Response::operator = (*response[0]);

  if (verbose)
    cerr << "dsp::PlasmaResponseProduct::build nchan=" << nchan
         << " nfilt=" << ndat << endl;

  for (unsigned iresp=1; iresp < response.size(); iresp++)
  {
    if (verbose)
      cerr << "dsp::PlasmaResponseProduct::build multiply iresp=" << iresp << endl;
    Response::operator *= (*response[iresp]);
  }

  if (verbose)
    cerr << "dsp::PlasmaResponseProduct::build done" << endl;
}

