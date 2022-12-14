/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/DedispersionSampleDelay.h"
#include "dsp/Observation.h"

using namespace std;

dsp::Dedispersion::SampleDelay::SampleDelay ()
{
  centre_frequency = 0.0;
  bandwidth = 0.0;
  dispersion_measure = 0.0;
  sampling_rate = 0.0;
}

#define SQR(x) (x*x)

bool dsp::Dedispersion::SampleDelay::match (const Observation* obs)
{
  // cerr << "dsp::Dedispersion::SampleDelay::match" << endl;

  bool changed =
    dispersion_measure != obs->get_dispersion_measure() ||
    centre_frequency   != obs->get_centre_frequency() ||
    bandwidth          != obs->get_bandwidth() ||
    sampling_rate      != obs->get_rate() ||
    delays.size()      != obs->get_nchan();

  if (!changed)
    return false;

  centre_frequency = obs->get_centre_frequency();
  bandwidth = obs->get_bandwidth();
  dispersion_measure = obs->get_dispersion_measure();
  sampling_rate = obs->get_rate();

  bool error = (sampling_rate == 0 || bandwidth == 0 || centre_frequency == 0);

  if (verbose || error)
    std::cerr << "dsp::Dedispersion::SampleDelay::match"
	      << "\n  centre frequency = " << centre_frequency
	      << "\n  bandwidth = " << bandwidth
	      << "\n  dispersion measure = " << dispersion_measure
	      << "\n  sampling rate = " << sampling_rate << endl;

  if (error)
    throw Error (InvalidParam, "dsp::Dedispersion::SampleDelay::match",
		 "invalid input");

  delays.resize( obs->get_nchan() );
  freqs.resize( obs->get_nchan() );
  
  // when divided by MHz, yields a dimensionless value
  double dispersion = dispersion_measure / dm_dispersion;

  for (unsigned ichan = 0; ichan < obs->get_nchan(); ichan++) {

    freqs[ichan] = obs->get_centre_frequency (ichan);
    
    // Compute the DM delay in seconds
    double delay = dispersion * (1.0/SQR(centre_frequency) - 1.0/SQR(freqs[ichan]));

    delays[ichan] = int64_t( floor(delay*sampling_rate + 0.5) );

    // cerr << "freq=" << freq << " delay=" << delay*1e3 << " ms = " 
    //      << delays[ichan] << " samps" << endl;

  }

  return true;
}

//! Return the dispersion delay for the given frequency channel
int64_t dsp::Dedispersion::SampleDelay::get_delay (unsigned ichan, unsigned ipol)
{
  return delays[ichan];
}

//! Return the deispersion delay for the centre of the frequency channel range
int64_t dsp::Dedispersion::SampleDelay::get_delay_range (unsigned schan, unsigned echan, unsigned ipol)
{
  if (schan > freqs.size() || echan > freqs.size())
    throw Error (InvalidParam, "dsp::Dedispersion::SampleDelay::get_delay_range",
                 "channel range invalid");

  double dispersion = dispersion_measure / dm_dispersion;
  double freq = (freqs[schan] + freqs[echan]) / 2;
  double delay = dispersion * (1.0/SQR(centre_frequency) - 1.0/SQR(freq));

  return int64_t( floor(delay*sampling_rate + 0.5) );
}

void dsp::Dedispersion::SampleDelay::mark (Observation* observation)
{
  observation->set_dispersion_measure (dispersion_measure);
}

