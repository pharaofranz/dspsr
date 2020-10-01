/***************************************************************************
 *
 *   Copyright (C) 2020 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "dsp/Debirefraction.h"
#include "dsp/Observation.h"
#include "dsp/OptimalFFT.h"

#include "ThreadContext.h"
#include "Error.h"

#include <complex>

using namespace std;

dsp::Debirefraction::Debirefraction ()
{
}

//! Set the dispersion measure in \f${\rm pc\,cm}^{-3}\f$
void dsp::Debirefraction::set_rotation_measure (double _rotation_measure)
{
  if (verbose)
    cerr << "dsp::Debirefraction::set_rotation_measure " << _rotation_measure << endl;

  if (get_rotation_measure() != _rotation_measure)
    set_not_built();

   birefringence.set_rotation_measure (_rotation_measure);
}

double dsp::Debirefraction::get_rotation_measure () const
{
  return birefringence.get_rotation_measure().get_value();
}

void dsp::Debirefraction::prepare (const Observation* input, unsigned channels)
{
  set_rotation_measure ( input->get_rotation_measure() );
  PlasmaResponse::prepare ( input, channels );
}

void dsp::Debirefraction::build_setup (double chan_freq)
{
  birefringence.set_reference_frequency (chan_freq);
}

Jones<float> dsp::Debirefraction::build_compute (double chan_freq, double freq)
{
  birefringence.set_frequency (freq);
  return birefringence.evaluate ();
}

void dsp::Debirefraction::build (unsigned ndat, unsigned nchan)
{
  vector< Jones<float> > response (ndat * nchan);
  PlasmaResponse::build (response, "rotation", get_rotation_measure(), this, ndat, nchan);

  // always zap DC channel
  response[0] = 0;
  
  resize (1, nchan, ndat, 8);
  set (response);
}

/*!
  return x squared
  */
template <typename T> inline T sqr (T x) { return x*x; }

double dsp::Debirefraction::delay_time (double freq) const
{
  Calibration::Faraday temp;

  temp.set_rotation_measure (get_rotation_measure());
  temp.set_reference_wavelength (0.0);
  temp.set_frequency (freq);

  double abs_delta_PA = temp.get_rotation () / (2.0*M_PI);
  double abs_dgd_mus = 2.0 * fabs(abs_delta_PA) / freq;
  return abs_dgd_mus * 1e-6;
}

