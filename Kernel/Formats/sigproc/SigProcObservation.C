/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "Pulsar/Site.h"
#include "Horizon.h"
#include "Meridian.h"

#include "dsp/SigProcObservation.h"
#include "FilePtr.h"

#include <iostream>
#include <string.h>

extern "C" {
#include "filterbank.h"
}

// This is probably not really the best practice, but I want
// to avoid having this require obsys.dat to function.
//#include "tempo_impl.h" // PSRCHIVE does not install this file
namespace Tempo
{
  extern std::string itoa_code (const std::string& telescope_name);
}

using namespace std;

dsp::SigProcObservation::SigProcObservation (const char* filename)
{
  if (filename)
    load (filename);
}

void dsp::SigProcObservation::load (const char* filename)
{
  FilePtr fptr = fopen (filename, "r");
  if (!fptr)
    throw Error (FailedSys, "dsp::SigProcObservation::load",
                 "fopen (%s)", filename);

  load (fptr);
}

dsp::SigProcObservation::SigProcObservation (FILE* header)
{
  if (header)
    load (header);
}

void dsp::SigProcObservation::load (FILE* header)
{
  sigproc_verbose = verbose;

  header_bytes = read_header (header);

  if (header_bytes < 1)
    throw Error (FailedCall, "dsp::SigProcObservation::load",
                 "read_header failed");

  load_global ();
}

#define SIGPROC_PARKES 4

static std::string get_sigproc_telescope_name (int _id)
{
  // Info from sigproc's aliases.c
  switch (_id) {
    case 0:
      return "Fake";
    case 1:
      return "Arecibo";
    case 2:
      return "Ooty";
    case 3:
      return "Nancay";
    case SIGPROC_PARKES:
      return "Parkes";
    case 5:
      return "Jodrell";
    case 6:
      return "GBT";
    case 7:
      return "GMRT";
    case 8:
      return "Effelsberg";
    case 10:
      return "SRT";
    case 11:
      return "LOFAR";
    case 12:
      return "VLA";
    case 64:
      return "MeerKAT";
    default:
      return "unknown";
      break;
  }

  return "unknown";
}

static int get_sigproc_telescope_id (string name)
{
  // Use psrchive's routine to convert various aliases into an ITOA
  // 2-char code, then convert those to sigproc codes using the info
  // found in sigproc's aliases.c
  try 
  {
    string itoa = Tempo::itoa_code(name);

    // If it's 2-char it might be an ITOA code already
    if ( itoa.empty() && name.length()==2 ) 
    {
      itoa = name;
    }

    // TODO if it's 1-char it might be a Tempo code..

    // Convert ITOA to sigproc code
    if      (itoa == "AO") return 1;
    else if (itoa == "NC") return 3;
    else if (itoa == "PK") return SIGPROC_PARKES;
    else if (itoa == "JB") return 5;
    else if (itoa == "GB") return 6;
    else if (itoa == "GM") return 7;
    else if (itoa == "EF") return 8;
    else if (itoa == "SR") return 10;
    else if (itoa == "LF") return 11;
    else if (itoa == "VL") return 12;
    else if (itoa == "MK") return 64;
    else return 0;
  }
  catch (Error &error)
  {
    cerr << "SigProcObservation: Error looking up telescope code" << endl;
  }

  return 0;
}

static std::string get_sigproc_machine_name (int _id, int _telescope)
{
  // Info from sigproc's aliases.c
  switch (_id) {
    case 0:
      return "FAKE";
    case 1:
      return "PSPM";
    case 2:
      return "WAPP";
    case 3:
      return "AOFTM";
    case 4:
      return "BPP";
    case 5:
      return "OOTY";
    case 6:
      return "SCAMP";
    case 7:
      return "GMRTFB";
    case 8:
      return "PULSAR2000";
    case 10:
      if (_telescope == SIGPROC_PARKES)
        return "BPSR";
      else
        return "ARTEMIS";
    case 11:
      return "COBALT";
    default:
      return "?????";
      break;
  }

  return "?????";
}


void dsp::SigProcObservation::load_global ()
{
  // set_receiver (buffer);

  set_source( source_name );

  set_type( Signal::Pulsar );

  // set_calfreq(calfreq);

  set_centre_frequency (fch1 + 0.5 * (foff * (nchans-1)));
  set_bandwidth (foff * nchans);

  set_nchan (nchans);
  set_npol (nifs);
  set_nbit (nbits);
  set_ndim (1);

  // set_ndat (nsamples);

  if (nifs == 1)
    set_state( Signal::Intensity );

  if (nifs == 2)
    set_state( Signal::PPQQ );

  if (nifs == 4)
    set_state( Signal::Coherence );

  set_rate( 1.0/tsamp );
  set_start_time( tstart );

  sky_coord coord;
  coord.ra().setHourMS (src_raj);
  coord.dec().setDegMS (src_dej);
  set_coordinates (coord);

  set_format ("SigProc");
  set_machine ( get_sigproc_machine_name(machine_id, telescope_id) );
  set_telescope ( get_sigproc_telescope_name(telescope_id) );
}

void dsp::SigProcObservation::unload (FILE* header)
{
  sigproc_verbose = verbose;

  unload_global ();
  filterbank_header (header);
}

void dsp::SigProcObservation::unload_global ()
{
  // set_receiver (buffer);

  // set the machine to a non-zero value so that header will be written
  machine_id = 0;
  telescope_id = 0;

  telescope_id = get_sigproc_telescope_id(get_telescope());

 /*
  * We need to get all the codes in here, but I am not sure
  * what the DSPSR 'names' for all the hardware is.
  *
  * M.Keith 2008-07-14
  */
  if(get_machine().compare("BPSR")==0)machine_id=10;
  else if(get_machine().compare("SCAMP")==0)machine_id=6;
  else if(get_machine().compare("COBALT")==0)machine_id=11;
  else if(get_machine().compare("ROACH")==0)machine_id=83;

  // This is the 'rawfilename' parameter in the header.
  // inpfile is possibly uninitialized here so avoid setting
  // either a blank string or a unterminated one
  inpfile[79] = '\0';
  if (inpfile[0]=='\0') strcpy(inpfile, "unknown");

  strcpy( source_name, get_source().c_str() );

  fch1 = get_centre_frequency (0);
  foff = get_bandwidth() / get_nchan();

  nchans = get_nchan ();
  nifs = get_npol ();
  obits = get_nbit ();
  
  tsamp = 1.0 / get_rate ();

  tstart = get_start_time().in_days();

  src_raj = coordinates.ra().getHourMS ();
  src_dej = coordinates.dec().getDegMS ();

  // cerr << "raj=" << src_raj << " dej=" << src_dej << endl;

  const Pulsar::Site* location = 0;
  Horizon horizon;
  Directional* directional = &horizon;
  try
  {
    // configure the observatory location
    location = Pulsar::Site::location (get_telescope());
    double lat, lon, rad;
    location->get_sph (lat, lon, rad);
    Angle latitude;
    Angle longitude;
    latitude.setRadians( lat );
    longitude.setRadians( lon );
    directional->set_observatory_latitude( latitude.getRadians() );
    directional->set_observatory_longitude( longitude.getRadians() );

    // configure the source coordinates
    directional->set_source_coordinates( coordinates );

    MJD mjd(tstart);
    directional->set_epoch( mjd );

    double rad2deg = 180.0/M_PI;
    az_start = horizon.get_azimuth() * rad2deg;
    za_start = horizon.get_zenith() * rad2deg;
  }
  catch (Error& error)
  {
    az_start = za_start = 0.0;
  }

  for (unsigned ipol=0; ipol < get_npol(); ipol++)
    ::ifstream[ipol] = 'Y';
}

