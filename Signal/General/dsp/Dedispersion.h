//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 - 2020 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/Dedispersion.h

#ifndef __Dedispersion_h
#define __Dedispersion_h

#include "dsp/PlasmaResponse.h"
#include "dsp/SampleDelayFunction.h"

namespace dsp {

  //! Phase-coherent dispersion removal frequency response function
  /* This class implements the phase-coherent dedispersion kernel, as
     determined by the frequency response of the interstellar
     medium. */

  class Dedispersion : public PlasmaResponse 
  {

  public:

    //! Conversion factor between dispersion measure, DM, and dispersion, D
    static const double dm_dispersion;

    //! Null constructor
    Dedispersion ();

    //! Return the dispersion delay for the given frequency
    double delay_time (double freq) const;

    //! Set up and calculate the impulse_pos and impulse_neg attributes
    void prepare (const Observation* input, unsigned channels);

    //! Set the dispersion measure in \f${\rm pc\,cm}^{-3}\f$
    void set_dispersion_measure (double dm);

    //! Get the dispersion measure in \f${\rm pc\,cm}^{-3}\f$
    double get_dispersion_measure () const { return dispersion_measure; }

    //! Set the flag to add fractional inter-channel delay
    void set_fractional_delay (bool fractional_delay);

    //! Get the flag to add fractional inter-channel delay
    bool get_fractional_delay () const { return fractional_delay; }

    //! Build delays in microseconds instead of phases
    void set_build_delays (bool delay = true);

    //! Compute the phases for a dedispersion kernel
    void build (std::vector<float>& phases, unsigned npts, unsigned nchan);

    class SampleDelay;

  protected:

    //! Dispersion measure (in \f${\rm pc cm}^{-3}\f$)
    double dispersion_measure;

    //! Flag to add fractional inter-channel delay
    bool fractional_delay;

    //! Build method returns delay in microseconds instead of phase
    bool build_delays;

    void build (unsigned ndat, unsigned nchan);

    friend class PlasmaResponse;

    //! Set up for the specified channel
    void build_setup (double chan_freq);
    double build_setup_delay;

    //! Return the phase of the kernel response 
    double build_compute (double chan_freq, double freq);

    //! Supported frequency channels
    /*! Set to false when the dispersive smearing is too large */
    std::vector<bool> supported_channels;
  };

}

#endif
