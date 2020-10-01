/***************************************************************************
 *
 *   Copyright (C) 2002-2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "dsp/Dedispersion.h"
#include "dsp/Observation.h"
#include "dsp/OptimalFFT.h"

#include "ThreadContext.h"
#include "Error.h"

#include <complex>

using namespace std;

/*! Although the value:

  \f$ DM\,({\rm pc\,cm^{-3}})=2.410331(2)\times10^{-4}D\,({\rm s\,MHz^{2}}) \f$

  has been derived from "fundamental and primary physical and
  astronomical constants" (section 3 of Backer, Hama, van Hook and
  Foster 1993. ApJ 404, 636-642), the rounded value is in standard
  use by pulsar astronomers (page 129 of Manchester and Taylor 1977).
*/
const double dsp::Dedispersion::dm_dispersion = 2.41e-4;

dsp::Dedispersion::Dedispersion ()
{
  dispersion_measure = 0.0;
  fractional_delay = false;
  build_delays = false;
  build_setup_delay = 0.0;
}

//! Set the dispersion measure in \f${\rm pc\,cm}^{-3}\f$
void dsp::Dedispersion::set_dispersion_measure (double _dispersion_measure)
{
  if (verbose)
    cerr << "dsp::Dedispersion::set_dispersion_measure "
            " dm=" << _dispersion_measure << endl;
 
  if (dispersion_measure != _dispersion_measure)
    set_not_built();

  dispersion_measure = _dispersion_measure;
}

//! Set the flag to add fractional inter-channel delay
void dsp::Dedispersion::set_fractional_delay (bool _fractional_delay)
{
  if (fractional_delay != _fractional_delay)
    set_not_built();

  fractional_delay = _fractional_delay;
}

void dsp::Dedispersion::prepare (const Observation* input, unsigned channels)
{
  if (verbose)
    cerr << "dsp::Dedispersion::prepare nchan=" << channels 
         << " dm=" << input->get_dispersion_measure() << endl;

  set_dispersion_measure ( input->get_dispersion_measure() );
  PlasmaResponse::prepare ( input, channels );
}

void dsp::Dedispersion::build (unsigned _ndat, unsigned _nchan)
{
  if (verbose)
    cerr << "dsp::Dedispersion::build nchan=" << _nchan << " nfilt=" << _ndat
         << " dm=" << get_dispersion_measure() << endl;

  // calculate the complex frequency response function
  vector<float> phases (_ndat * _nchan);

  build (phases, _ndat, _nchan);

  resize (1, _nchan, _ndat, 2);
  complex<float>* phasors = reinterpret_cast< complex<float>* > ( buffer );
  uint64_t npt = ndat * nchan;

  for (unsigned ipt=0; ipt<npt; ipt++)
    phasors[ipt] = polar (float(1.0), phases[ipt]);

#ifdef _DEBUG
  for (unsigned ipt=0; ipt<npt; ipt++)
    cerr << "Dedispersion::build ipt=" << ipt << " " << buffer[ipt*2] << " " << buffer[ipt*2+1] << endl;
#endif

  // always zap DC channel
  phasors[0] = 0;

  if (verbose)
    cerr << "dsp::Dedispersion::build done.  nchan=" << nchan 
         << " nfilt=" << ndat << endl;
}

void dsp::Dedispersion::build (std::vector<float>& phases, 
                               unsigned _npts, unsigned _nchan)
{
  if (verbose)
    cerr << "dsp::Dedispersion::build std::vector<float> nchan=" << nchan 
         << " nfilt=" << _npts << endl;

  PlasmaResponse::build (phases, "dispersion", dispersion_measure, 
                         this, _npts, _nchan);
}

/*!
  return x squared
  */
template <typename T> inline T sqr (T x) { return x*x; }

double dsp::Dedispersion::delay_time (double freq) const
{
  double dispersion = dispersion_measure/dm_dispersion;
  return dispersion * 1.0/sqr(freq);
}

void dsp::Dedispersion::build_setup (double chan_freq)
{
  if (!fractional_delay)
  {
    build_setup_delay = 0.0;
    return;
  }

  double doppler = get_Doppler_shift();
  double centrefreq = get_centre_frequency() / doppler;
  double bw = get_bandwidth() / doppler;

  double chanwidth = bw / double(nchan);
  double highest_freq = centrefreq + 0.5*fabs(bw-chanwidth);

  double samp_int = 1.0/chanwidth; // sampint in microseconds, for
                                   // quadrature nyquist data eg fb.

  // when divided by MHz, yields a dimensionless value
  double dispersion_per_MHz = 1e6 * dispersion_measure / dm_dispersion;

  // Compute the DM delay in microseconds; when multiplied by the
  // frequency in MHz, the powers of ten cancel each other
  build_setup_delay = dispersion_per_MHz * ( 1.0/sqr(chan_freq) -
				             1.0/sqr(highest_freq) );
  // Modulo one sample and invert it
  build_setup_delay = - fmod(build_setup_delay, samp_int);
}

double dsp::Dedispersion::build_compute (double chan_freq, double freq)
{
  double bw = get_bandwidth();
  double sign = bw / fabs (bw);

  double dispersion_per_MHz = 1e6 * dispersion_measure / dm_dispersion;

  double coeff = -sign * 2*M_PI * dispersion_per_MHz / sqr(chan_freq);

  // additional phase turn for fractional dispersion delay shift
  double delay_phase = -2.0*M_PI * freq * build_setup_delay;

  double result = coeff*sqr(freq)/(chan_freq+freq) + delay_phase;

  if (build_delays && freq != 0.0)
    result /= (2.0*M_PI * freq);

  return result;
}

//! Build delays in microseconds instead of phases
void dsp::Dedispersion::set_build_delays (bool delays)
{
  build_delays = delays;
}

