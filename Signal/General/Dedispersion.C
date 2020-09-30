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
}

//! Set the dispersion measure in \f${\rm pc\,cm}^{-3}\f$
void dsp::Dedispersion::set_dispersion_measure (double _dispersion_measure)
{
  if (verbose) {
    cerr << "dsp::Dedispersion::set_dispersion_measure " << _dispersion_measure << endl;
  }
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
  set_dispersion_measure ( input->get_dispersion_measure() );
  PlasmaResponse::prepare ( input, channels );
}

void dsp::Dedispersion::build (unsigned ndat, unsigned nchan)
{
  // calculate the complex frequency response function
  vector<float> phases (ndat * nchan);

  build (phases, ndat, nchan);

  resize (1, nchan, ndat, 2);
  complex<float>* phasors = reinterpret_cast< complex<float>* > ( buffer );
  uint64_t npt = ndat * nchan;

  for (unsigned ipt=0; ipt<npt; ipt++)
    phasors[ipt] = polar (float(1.0), phases[ipt]);

  // always zap DC channel
  phasors[0] = 0;
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

void dsp::Dedispersion::build (vector<float>& phases,
			       unsigned _ndat, unsigned _nchan)
{
  if (verbose)
    cerr << "dsp::Dedispersion::build"
      "\n  centre frequency = " << get_centre_frequency() <<
      "\n  bandwidth = " << get_bandwidth() <<
      "\n  dispersion measure = " << dispersion_measure <<
      "\n  Doppler shift = " << get_Doppler_shift() <<
      "\n  ndat = " << ndat <<
      "\n  nchan = " << _nchan <<
      "\n  centred on DC = " << dc_centred <<
      "\n  fractional delay compensation = " << fractional_delay << endl;

  double doppler = get_Doppler_shift();
  double centrefreq = get_centre_frequency() / doppler;
  double bw = get_bandwidth() / doppler;

  double sign = bw / fabs (bw);
  double chanwidth = bw / double(_nchan);
  double binwidth = chanwidth / double(_ndat);

  double lower_cfreq = centrefreq - 0.5*bw;
  if (!dc_centred)
    lower_cfreq += 0.5*chanwidth;

  double highest_freq = centrefreq + 0.5*fabs(bw-chanwidth);

  double samp_int = 1.0/chanwidth; // sampint in microseconds, for
                                   // quadrature nyquist data eg fb.
  double delay = 0.0;

  // when divided by MHz, yields a dimensionless value
  double dispersion_per_MHz = 1e6 * dispersion_measure / dm_dispersion;

  phases.resize (_ndat * _nchan);

  frequency_output.resize( _nchan );
  bandwidth_output.resize( _nchan );

  for (unsigned ichan = 0; ichan < _nchan; ichan++)
  {
    double chan_cfreq = lower_cfreq + double(ichan) * chanwidth;

    frequency_output[ichan] = chan_cfreq;
    bandwidth_output[ichan] = chanwidth;

    if (fractional_delay)
    {
      // Compute the DM delay in microseconds; when multiplied by the
      // frequency in MHz, the powers of ten cancel each other
      delay = dispersion_per_MHz * ( 1.0/sqr(chan_cfreq) -
				     1.0/sqr(highest_freq) );
      // Modulo one sample and invert it
      delay = - fmod(delay, samp_int);
    }

    double coeff = -sign * 2*M_PI * dispersion_per_MHz / sqr(chan_cfreq);

    unsigned spt = ichan * _ndat;
    for (unsigned ipt = 0; ipt < _ndat; ipt++)
    {
      // frequency offset from centre frequency of channel
      double freq = double(ipt)*binwidth - 0.5*chanwidth;

      // additional phase turn for fractional dispersion delay shift
      double delay_phase = -2.0*M_PI * freq * delay;

      phases[spt+ipt] = coeff*sqr(freq)/(chan_cfreq+freq) + delay_phase;

      if (build_delays && freq != 0.0)
	phases[spt+ipt] /= (2.0*M_PI * freq);

#ifdef _DEBUG
      cerr << ichan*_ndat + ipt << " " << chan_cfreq+freq << " "
	   << phases[spt+ipt] << endl;
#endif
    }
  }
}

//! Build delays in microseconds instead of phases
void dsp::Dedispersion::set_build_delays (bool delays)
{
  build_delays = delays;
}
