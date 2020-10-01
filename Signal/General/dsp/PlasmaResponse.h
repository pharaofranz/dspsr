//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2020 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/PlasmaResponse.h

#ifndef __PlasmaResponse_h
#define __PlasmaResponse_h

#include "dsp/Response.h"
#include "Rational.h"

class ThreadContext;

namespace dsp {

  class PlasmaResponse: public Response {

  public:

    //! Fractional smearing added to reduce cyclical convolution effects
    static float smearing_buffer;

    //! Null constructor
    PlasmaResponse ();

    //! Set up and calculate the impulse_pos and impulse_neg attributes
    virtual void prepare (const Observation* input, unsigned channels);

    //! Calculate the impulse_pos and impulse_neg attributes
    void prepare ();

    //! Match the dedispersion kernel to the input Observation
    virtual void match (const Observation* input, unsigned channels=0);

    //! Set the dimensions of the data
    virtual void resize (unsigned npol, unsigned nchan,
			 unsigned ndat, unsigned ndim);

    //! Set the flag for a bin-centred spectrum
    virtual void set_dc_centred (bool dc_centred);

    //! Set the number of channels into which the band will be divided
    virtual void set_nchan (unsigned nchan);

    //! Set the centre frequency of the band-limited signal in MHz
    void set_centre_frequency (double centre_frequency);

    //! Return the centre frequency of the band-limited signal in MHz
    double get_centre_frequency () const { return centre_frequency; }

    //! Returns the centre frequency of the specified channel in MHz
    double get_centre_frequency (int ichan) const;

    //! Set the bandwidth of signal in MHz
    void set_bandwidth (double bandwidth);

    //! Return the bandwidth of signal in MHz
    double get_bandwidth () const { return bandwidth; }

    //! Set the Doppler shift due to the Earth's motion
    void set_Doppler_shift (double Doppler_shift);

    //! Return the doppler shift due to the Earth's motion
    double get_Doppler_shift () const { return Doppler_shift; }

    //! Set the frequency resolution in each channel of the kernel
    void set_frequency_resolution (unsigned nfft);

    //! Get the frequency resolution in each channel of the kernel
    unsigned get_frequency_resolution () const { return ndat; }

    //! Set the number of smearing samples in each channel of the kernel
    void set_smearing_samples (unsigned pos, unsigned neg);
    void set_smearing_samples (unsigned total);

    //! Set the frequency resolution this many times the minimum required
    void set_times_minimum_nfft (unsigned times);

    //! Return the smearing across the entire band time in seconds
    double get_smearing_time () const {
      return smearing_time (centre_frequency, bandwidth);
    }

    //! Return the smearing time across the worst sub-band in seconds
    double get_effective_smearing_time () const;

    //! Return the effective number of smearing samples
    unsigned get_effective_smearing_samples () const;

    //! Return the dispersion delay between freq1 and freq2
    /*! If freq2 is higher than freq1, delay_time is positive */
    double delay_time (double freq1, double freq2) const;

    //! Return the dispersion delay for the given frequency
    virtual double delay_time (double freq) const = 0;

    //! Return the dispersion delay for the given frequency
    double delay_time () const { return delay_time (centre_frequency); }

    //! Return the smearing time, given the centre frequency and bandwidth
    double smearing_time (double centre_frequency, double bandwidth) const;

    //! Build the dedispersion frequency response kernel
    virtual void build ();

    //! Set the built flag to false (only build can set it to true)
    void set_not_built () { built = false; }

    //!
    ThreadContext* context;

    std::vector<double> frequency_input;
    std::vector<double> bandwidth_input;

    std::vector<double> frequency_output;
    std::vector<double> bandwidth_output;

  protected:

    friend class PlasmaResponseProduct;
    
    //! Construction of filter defined in derived classes
    virtual void build (unsigned ndat, unsigned nchan) = 0;

    //! Worker function to help in derived classes
    template<typename T, typename C>
    void build (std::vector<T>& response, const char* name, double measure, C* child,
                unsigned _ndat, unsigned _nchan);

  private:

    //! Centre frequency of the band-limited signal in MHz
    double centre_frequency;

    //! Bandwidth of signal in MHz
    double bandwidth;

    //! Doppler shift due to the Earths' motion
    double Doppler_shift;

    //! Flag set when set_frequency_resolution() method is called
    bool frequency_resolution_set;

    //! Flag set when set_smearing_samples() method is called
    bool smearing_samples_set;

    //! Choose filter length this many times the minimum length
    unsigned times_minimum_nfft;

    //! Flag that the response and bandpass attributes reflect the state
    bool built;

    //! Supported frequency channels
    /*! Set to false when the dispersive smearing is too large */
    std::vector<bool> supported_channels;

    //! Return the effective smearing time in seconds (worker function)
    double smearing_time (int half) const;

    //! Return the number of complex samples of smearing (worker function)
    unsigned smearing_samples (int half) const;

    Rational oversampling_factor;
  };

}

template<typename T, typename C>
void dsp::PlasmaResponse::build (std::vector<T>& response, 
                                 const char* name, double measure, C* child,
                                 unsigned _ndat, unsigned _nchan)
{
  if (verbose)
    cerr << "dsp::PlasmaResponse::build"
      "\n  centre frequency = " << get_centre_frequency() <<
      "\n  bandwidth = " << get_bandwidth() <<
      "\n  " << name << " measure = " << measure <<
      "\n  Doppler shift = " << get_Doppler_shift() <<
      "\n  ndat = " << _ndat <<
      "\n  nchan = " << _nchan <<
      "\n  centred on DC = " << dc_centred << std::endl;

  double doppler = get_Doppler_shift();
  double centrefreq = get_centre_frequency() / doppler;
  double bw = get_bandwidth() / doppler;

  double sign = bw / fabs (bw);
  double chanwidth = bw / double(_nchan);
  double binwidth = chanwidth / double(_ndat);

  double lower_cfreq = centrefreq - 0.5*bw;
  if (!dc_centred)
    lower_cfreq += 0.5*chanwidth;

  response.resize (_ndat * _nchan);

  frequency_output.resize( _nchan );
  bandwidth_output.resize( _nchan );

  for (unsigned ichan = 0; ichan < _nchan; ichan++)
  {
#ifdef _DEBUG
    cerr << "dsp::PlasmaResponse::build ichan=" << ichan << std::endl;
#endif

    double chan_cfreq = lower_cfreq + double(ichan) * chanwidth;

    frequency_output[ichan] = chan_cfreq;
    bandwidth_output[ichan] = chanwidth;

    child->build_setup (chan_cfreq);

    unsigned spt = ichan * _ndat;
    for (unsigned ipt = 0; ipt < _ndat; ipt++)
    {
      // frequency offset from centre frequency of channel
      double freq = double(ipt)*binwidth - 0.5*chanwidth;

      response[spt+ipt] = child->build_compute (chan_cfreq, freq);
#ifdef _DEBUG
      cerr << ichan << " " << ipt << " " << response[spt+ipt] << std::endl;
#endif
    }
  }
}

#endif
