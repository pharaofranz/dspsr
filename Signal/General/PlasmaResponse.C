/***************************************************************************
 *
 *   Copyright (C) 2020 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "dsp/PlasmaResponse.h"
#include "dsp/Observation.h"
#include "dsp/OptimalFFT.h"

#include "ThreadContext.h"
#include "Error.h"

#include <complex>

using namespace std;

float dsp::PlasmaResponse::smearing_buffer = 0.1;

dsp::PlasmaResponse::PlasmaResponse ()
{
  centre_frequency = -1.0;
  bandwidth = 0.0;

  Doppler_shift = 1.0;
  dc_centred = false;

  frequency_resolution_set = false;
  times_minimum_nfft = 0;

  smearing_samples_set = false;

  built = false;
  context = 0;

  oversampling_factor = Rational(1,1);
}

//! Set the dimensions of the data and update the built attribute
void dsp::PlasmaResponse::resize (unsigned _npol, unsigned _nchan,
				unsigned _ndat, unsigned _ndim)
{
  if (npol != _npol || nchan != _nchan || ndat != _ndat || ndim != _ndim)
    built = false;

  Shape::resize (_npol, _nchan, _ndat, _ndim);
}

//! Set the centre frequency of the band-limited signal in MHz
void dsp::PlasmaResponse::set_centre_frequency (double _centre_frequency)
{
  if (centre_frequency != _centre_frequency)
    built = false;

  centre_frequency = _centre_frequency;
}

//! Returns the centre frequency of the specified channel in MHz
double dsp::PlasmaResponse::get_centre_frequency (int ichan) const
{
  throw Error (InvalidState,
	       "PlasmaResponse::get_centre_frequency (ichan)"
	       "not implemented");
}

//! Set the bandwidth of the signal in MHz
void dsp::PlasmaResponse::set_bandwidth (double _bandwidth)
{
  if (bandwidth != _bandwidth)
    built = false;

  bandwidth = _bandwidth;
}

//! Set the Doppler shift due to the Earth's motion
void dsp::PlasmaResponse::set_Doppler_shift (double _Doppler_shift)
{
  if (Doppler_shift != _Doppler_shift)
    built = false;

  Doppler_shift = _Doppler_shift;
}

//! Set the flag for a bin-centred spectrum
void dsp::PlasmaResponse::set_dc_centred (bool _dc_centred)
{
  if (dc_centred != _dc_centred)
    built = false;

  dc_centred = _dc_centred;
}

//! Set the number of channels
void dsp::PlasmaResponse::set_nchan (unsigned _nchan)
{
  if (nchan != _nchan)
    built = false;

  nchan = _nchan;
}

void dsp::PlasmaResponse::set_frequency_resolution (unsigned nfft)
{
  if (verbose)
    cerr << "dsp::PlasmaResponse::set_frequency_resolution ("<<nfft<<")"<<endl;
  resize (npol, nchan, nfft, ndim);

  frequency_resolution_set = true;
}

void dsp::PlasmaResponse::set_smearing_samples (unsigned tot)
{
  unsigned pos = tot / 2;
  unsigned neg = pos;
  if (tot % 2)
    neg ++;
  set_smearing_samples (pos, neg);
}

void dsp::PlasmaResponse::set_smearing_samples (unsigned pos, unsigned neg)
{
  impulse_pos = pos;
  impulse_neg = neg;
  smearing_samples_set = true;
}

void dsp::PlasmaResponse::set_times_minimum_nfft (unsigned times)
{
  if (verbose)
    cerr << "dsp::PlasmaResponse::set_times_minimum_nfft ("<<times<<")"<<endl;

  times_minimum_nfft = times;
  frequency_resolution_set = true;
}

void dsp::PlasmaResponse::prepare (const Observation* input, unsigned channels)
{
  if (verbose)
    cerr << "dsp::PlasmaResponse::prepare input.nchan=" << input->get_nchan()
	 << " channels=" << channels << "\n\t"
      " centre frequency=" << input->get_centre_frequency() <<
      " bandwidth=" << input->get_bandwidth () <<
      " dispersion measure=" << input->get_dispersion_measure() << endl;

  set_centre_frequency ( input->get_centre_frequency() );
  set_bandwidth ( input->get_bandwidth() );
  set_dc_centred ( input->get_dc_centred() );

  frequency_input.resize( input->get_nchan() );
  bandwidth_input.resize( input->get_nchan() );

  for (unsigned ichan=0; ichan<input->get_nchan(); ichan++)
  {
    frequency_input[ichan] = input->get_centre_frequency( ichan );
    bandwidth_input[ichan] = input->get_bandwidth() / input->get_nchan();
  }

  if (!channels)
    channels = input->get_nchan();

  set_nchan (channels);

  if (!built)
    prepare ();
}

/*! The signal at sky frequencies lower than the centre frequency
  arrives later.  So the finite impulse response (FIR) of the
  dispersion relation, d(t), should have non-zero values for t>0 up to
  the smearing time in the lower half of the band.  However, this
  class represents the inverse, or dedispersion, frequency response
  function, the FIR of which is given by h(t)=d^*(-t).  Therefore,
  h(t) has non-zero values for t>0 up to the smearing time in the
  upper half of the band.

  Noting that the first impulse_pos complex time samples are discarded
  from each cyclical convolution result, it may also be helpful to
  note that each time sample depends upon the preceding impulse_pos
  points.
*/

unsigned smearing_samples_threshold = 16 * 1024 * 1024;

void dsp::PlasmaResponse::prepare ()
{
  if (!smearing_samples_set)
  {
    unsigned threshold = smearing_samples_threshold / nchan;
    supported_channels = vector<bool> (nchan, true);
    unsigned ichan = 0;

    while( (impulse_neg = smearing_samples (-1)) > threshold )
    {
      supported_channels[ichan] = false;
      ichan ++;
      if (ichan == nchan)
	throw Error (InvalidState,
		     "dsp::PlasmaResponse::prepare",
		     "smearing samples=%u exceeds threshold=%u",
		     impulse_neg, threshold);
    }

    if (verbose)
      cerr << "dsp::PlasmaResponse::prepare "
	   << ichan << " unsupported channels" << endl;

    impulse_pos = smearing_samples (1);
  }

  if (psrdisp_compatible)
  {
    cerr << "dsp::PlasmaResponse::prepare psrdisp compatibility\n"
      "   using symmetric impulse response function" << endl;
    impulse_pos = impulse_neg;
  }
}

/*! Builds a frequency response function (kernel) suitable for phase-coherent
  dispersion removal, based on the centre frequency, bandwidth, and number
  of channels in the input Observation.

  \param input Observation for which a dedispersion kernel will be built.

  \param channels If specified, over-rides the number of channels of the
  input Observation.  This parameter is useful if the Observation is to be
  simultaneously divided into filterbank channels during convolution.
 */
void dsp::PlasmaResponse::match (const Observation* input, unsigned channels)
{
  if (verbose){
    cerr << "dsp::PlasmaResponse::match before lock" << endl;
  }

  input_nchan = input->get_nchan();
  oversampling_factor = input->get_oversampling_factor();

  ThreadContext::Lock lock (context);

  if (verbose){
    cerr << "dsp::PlasmaResponse::match after lock" << endl;
  }

  prepare (input, channels);

  if (!built)
    build ();

  Response::match (input, channels);

  buffer[0] = buffer[1] = 0.0;

  if (verbose)
    cerr << "dsp::PlasmaResponse::match exit" << endl;

}

void dsp::PlasmaResponse::build ()
{
  if (built)
    return;

  // prepare internal variables
  if (frequency_resolution_set)
  {
    if (times_minimum_nfft)
      set_frequency_resolution( times_minimum_nfft * get_minimum_ndat() );
    check_ndat ();
  }
  else
  {
    if (optimal_fft)
      optimal_fft->set_simultaneous (nchan > 1);
    set_optimal_ndat ();
    if (verbose) {
      std::cerr << "dsp::PlasmaResponse::build:"
        << " calculating oversampled ndat and discard regions"
        << " ndat=" << ndat
        << " impulse_pos=" << impulse_pos
        << " impulse_neg=" << impulse_neg
        << std::endl;
    }
    if (oversampling_factor.doubleValue() > 1.0) {
      calc_oversampled_discard_region(
        &impulse_neg, &impulse_pos, input_nchan/nchan, oversampling_factor);
      calc_oversampled_fft_length(
        &ndat, input_nchan/nchan, oversampling_factor);
      if (ndat <= impulse_neg + impulse_pos) {
        ndat = 2*(impulse_neg + impulse_pos);
        calc_oversampled_fft_length(
          &ndat, input_nchan/nchan, oversampling_factor, 1);
      }
    }
    if (verbose) {
      std::cerr << "dsp::PlasmaResponse::build:"
        << " ndat=" << ndat
        << " impulse_pos=" << impulse_pos
        << " impulse_neg=" << impulse_neg
        << std::endl;
    }
  }

  build (ndat, nchan);

  whole_swapped = false;
  swap_divisions = 0;

  built = true;

  changed.send (*this);
}

/*!
  return x squared
  */
template <typename T> inline T sqr (T x) { return x*x; }

/*
  \param cfreq centre frequency, in MHz
  \param bw bandwidth, in MHz
  \retval dispersion smearing time across the specified band, in seconds
*/
double dsp::PlasmaResponse::smearing_time (double cfreq, double bw) const
{
  return delay_time (cfreq - fabs(0.5*bw), cfreq + fabs(0.5*bw));
}

double dsp::PlasmaResponse::delay_time (double freq1, double freq2) const
{
  return delay_time (freq1) - delay_time (freq2);
}

double dsp::PlasmaResponse::get_effective_smearing_time () const
{
  return smearing_time (0);
}

//! Return the effective number of smearing samples
unsigned dsp::PlasmaResponse::get_effective_smearing_samples () const
{
  if (verbose)
    cerr << "dsp::PlasmaResponse::get_effective_smearing_samples" << endl;

  return smearing_samples (0);
}

/*!
  Calculate the smearing time over the band (or the sub-band with
  the lowest centre frequency) in seconds.  This will determine the
  number of points "nsmear" that must be thrown away for each FFT.
*/
double dsp::PlasmaResponse::smearing_time (int half) const
{
  if ( ! (half==0 || half==-1 || half == 1) )
    throw Error (InvalidParam, "dsp::PlasmaResponse::smearing_time",
		 "invalid half=%d", half);

  double abs_bw = fabs (bandwidth);
  double ch_abs_bw = abs_bw / double(nchan);
  double lower_ch_cfreq = centre_frequency - (abs_bw - ch_abs_bw) / 2.0;

  unsigned ichan=0;
  while (ichan < supported_channels.size() && !supported_channels[ichan])
  {
    lower_ch_cfreq += ch_abs_bw;
    ichan++;
  }

  // calculate the smearing (in the specified half of the band)
  if (half)
  {
    ch_abs_bw /= 2.0;
    lower_ch_cfreq += double(half) * ch_abs_bw;
  }

  if (verbose)
    cerr << "dsp::PlasmaResponse::smearing_time freq=" << lower_ch_cfreq
	 << " bw=" << ch_abs_bw << endl;

  double tsmear = smearing_time (lower_ch_cfreq, ch_abs_bw);

  if (verbose)
  {
    string band = "band";
    if (nchan>1)
      band = "worst channel";

    string side;
    if (half == 1)
      side = "upper half of the ";
    else if (half == -1)
      side = "lower half of the ";

    cerr << "dsp::PlasmaResponse::smearing_time in the " << side << band << ": "
	 << float(tsmear*1e3) << " ms" << endl;
  }

  return tsmear;
}

unsigned dsp::PlasmaResponse::smearing_samples (int half) const
{
  double tsmear = smearing_time (half);

  // the sampling rate of the resulting complex time samples
  double ch_abs_bw = fabs (bandwidth) / double (nchan);
  double sampling_rate = ch_abs_bw * 1e6;

  if (verbose)
    cerr << "dsp::PlasmaResponse::smearing_samples = "
	 << int64_t(tsmear * sampling_rate) << endl;

  // add another ten percent, just to be sure that the pollution due
  // to the cyclical convolution effect is minimized
  if (psrdisp_compatible)
  {
     cerr << "dsp::PlasmaResponse::prepare psrdisp compatibility\n"
       "   increasing smearing time by 5 instead of "
          << smearing_buffer*100.0 << " percent" << endl;
    tsmear *= 1.05;
  }
  else
    tsmear *= (1.0 + smearing_buffer);

  // smear across one channel in number of time samples.
  unsigned nsmear = unsigned (ceil(tsmear * sampling_rate));

  if (psrdisp_compatible)
  {
    cerr << "dsp::PlasmaResponse::prepare psrdisp compatibility\n"
      "   rounding smear samples down instead of up" << endl;
     nsmear = unsigned (tsmear * sampling_rate);
  }

  if (verbose)
  {
    // recalculate the smearing time simply for display of new value
    tsmear = double (nsmear) / sampling_rate;
    cerr << "dsp::PlasmaResponse::smearing_samples effective smear time: "
	 << tsmear*1e3 << " ms (" << nsmear << " pts)." << endl;
  }

  return nsmear;
}

