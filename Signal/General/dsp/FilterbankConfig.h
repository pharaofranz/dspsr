//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/FilterbankConfig.h

#ifndef __FilterbankConfig_h
#define __FilterbankConfig_h

#include "dsp/Filterbank.h"

namespace dsp
{
  class Filterbank::Config
  {
  public:

    //! When dedispersion takes place with respect to filterbank
    enum When
    {
      Before,
      During,
      After,
      Never
    };

    Config ()
    : _memory(0), _stream(0), _nChannel(1), 
      _frequencyResolution(0), _when(After) { }

    void set_nchan (unsigned n) noexcept { _nChannel = n; }
    unsigned get_nchan () const noexcept { return _nChannel; }

    void set_freq_res (unsigned n) noexcept { _frequencyResolution = n; }
    unsigned get_freq_res () const noexcept { return _frequencyResolution; }

    void set_convolve_when (When w) noexcept { _when = w; }
    When get_convolve_when () const noexcept { return _when; }

    //! Set the device on which the unpacker will operate
    void set_device (Memory* mem) noexcept { _memory = mem; }

    //! Set the stream information for the device
    void set_stream (void* ptr) noexcept { _stream = ptr; }

    //! Return a new Filterbank instance and configure it
    Filterbank* create ();

  private:

    Memory* _memory;
    void* _stream;
    unsigned _nChannel;
    unsigned _frequencyResolution;
    When _when;

  };

  //! Insertion operator
  std::ostream& operator << (std::ostream&, const Filterbank::Config&);

  //! Extraction operator
  std::istream& operator >> (std::istream&, Filterbank::Config&);
}

#endif
