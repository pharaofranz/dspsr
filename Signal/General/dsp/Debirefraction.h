//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2020 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/Debirefraction.h

#ifndef __Debirefraction_h
#define __Debirefraction_h

#include "dsp/PlasmaResponse.h"
#include "Pulsar/Faraday.h"

namespace dsp {

  //! Phase-coherent birefringence removal frequency response function
  /* This class implements the phase-coherent birefraction removal kernel, as
     determined by the frequency response of the interstellar medium. */

  class Debirefraction : public PlasmaResponse 
  {

  public:

    //! Null constructor
    Debirefraction ();

    //! Return the dispersion delay for the given frequency
    double delay_time (double freq) const;

    //! Set up and calculate the impulse_pos and impulse_neg attributes
    void prepare (const Observation* input, unsigned channels);

    //! Set the dispersion measure in \f${\rm pc\,cm}^{-3}\f$
    void set_rotation_measure (double dm);

    //! Get the dispersion measure in \f${\rm pc\,cm}^{-3}\f$
    double get_rotation_measure () const;

  protected:

    Calibration::Faraday birefringence;

    void build (unsigned ndat, unsigned nchan);

    friend class PlasmaResponse;

    //! Set up for the specified channel
    void build_setup (double chan_freq);

    //! Called in build to compute the value of the response
    Jones<float> build_compute (double chan_freq, double freq);

    //! Supported frequency channels
    /*! Set to false when the dispersive smearing is too large */
    std::vector<bool> supported_channels;
  };

}

#endif
