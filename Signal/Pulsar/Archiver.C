/***************************************************************************
 *
 *   Copyright (C) 2002-2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/on_host.h"

#include "dsp/Archiver.h"
#include "dsp/PhaseSeries.h"
#include "dsp/Response.h"
#include "dsp/Operation.h"
#include "dsp/TwoBitCorrection.h"
#include "dsp/OutputArchive.h"

#include "Pulsar/Interpreter.h"
#include "Pulsar/Integration.h"
#include "Pulsar/IntegrationExpert.h"
#include "Pulsar/Dedisperse.h"

#include "Pulsar/Profile.h"
#include "Pulsar/FourthMoments.h"

#include "Pulsar/Check.h"

#include "Pulsar/dspReduction.h"
#include "Pulsar/TwoBitStats.h"
#include "Pulsar/DigitiserCounts.h"
#include "Pulsar/Passband.h"
#include "Pulsar/Telescope.h"
#include "Pulsar/Receiver.h"
#include "Pulsar/Backend.h"

#include "Pulsar/FITSHdrExtension.h"

#include "Pulsar/Predictor.h"
#include "Error.h"

#include <iomanip>
#include <assert.h>
#include <time.h>

using namespace std;

unsigned dsp::Archiver::verbose = 1;

dsp::Archiver::Archiver ()
{
  archive_software = "Software Unknown";
  archive_dedispersed = false;
  profiles = 0;
  minimum_integration_length = 0;
  store_dynamic_extensions = true;
  fourth_moments = 0;
  subints_per_file = 0;
  use_single_archive = false;
  force_archive_class = false;

  /* PLEASE DON'T FORGET TO ALSO UPDATE THE COPY CONSTRUCTOR */
}

dsp::Archiver::Archiver (const Archiver& copy)
  : PhaseSeriesUnloader (copy)
{
  if (copy.single_archive)
    single_archive = copy.single_archive->clone();

  script = copy.script;

  extensions.resize( copy.extensions.size() );
  for (unsigned iext=0; iext < extensions.size(); iext++)
    extensions[iext] = copy.extensions[iext]->clone();

  archive_class_name = copy.archive_class_name;
  force_archive_class = copy.force_archive_class;
  store_dynamic_extensions = copy.store_dynamic_extensions;
  archive_software = copy.archive_software;
  archive_dedispersed = copy.archive_dedispersed;
  minimum_integration_length = copy.minimum_integration_length;
  fourth_moments = copy.fourth_moments;
  subints_per_file = copy.subints_per_file;
  use_single_archive = copy.use_single_archive;

  profiles = 0;
}

dsp::Archiver::~Archiver ()
{
}

//! Clone operator
dsp::Archiver* dsp::Archiver::clone () const
{
  return new Archiver (*this);
}

void dsp::Archiver::set_minimum_integration_length (double seconds)
{
  minimum_integration_length = seconds;
}

void dsp::Archiver::set_archive_class (const string& class_name)
{
  archive_class_name = class_name;
}

void dsp::Archiver::set_force_archive_class (bool force)
{
  force_archive_class = force;
}

void dsp::Archiver::set_archive (Pulsar::Archive* archive)
{
  single_archive = archive;
}

//! Get the Pulsar::Archive instance to which all data were added
Pulsar::Archive* dsp::Archiver::get_archive ()
{
  return single_archive;
}

bool dsp::Archiver::has_archive () const
{
  return single_archive;
}

//! Set the post-processing script
void dsp::Archiver::set_script (const std::vector<std::string>& jobs)
{ 
  script = jobs;
}

//! Add a Pulsar::Archive::Extension to those added to the output archive
void dsp::Archiver::add_extension (Pulsar::Archive::Extension* extension)
{
  extensions.push_back (extension);
}

//! Return a new Archive instance
Pulsar::Archive* dsp::Archiver::new_Archive() const
{
  if (!force_archive_class) try
  {
    const OutputArchive* out = profiles->get_extensions()->get<OutputArchive>();
    if (out)
    {
      if (verbose > 2)
        cerr << "dsp::Archiver::new_Archive using OutputArchive policy" << endl;

      Pulsar::Archive* output = out->new_Archive();

      if (verbose > 2)
        cerr << "dsp::Archiver::new_Archive output=" << output 
             << " nsubint=" << output->get_nsubint() << endl;

      return output;
    }
  }
  catch (Error& error)
  {
    if (verbose > 2)
      cerr << "dsp::Archiver::new_Archive using OutputArchive policy failed"
           << error << endl;
  }

  if (verbose > 2)
    cerr << "dsp::Archiver::new_Archive new " << archive_class_name << endl;

  return Pulsar::Archive::new_Archive (archive_class_name);
}

void dsp::Archiver::unload (const PhaseSeries* _profiles) try
{
  if (!single_archive && archive_class_name.size() == 0)
    throw Error (InvalidState, "dsp::Archiver::unload", 
        	 "neither Archive nor class name specified");

  if (!_profiles)
    throw Error (InvalidState, "dsp::Archiver::unload",
        	 "Profile data not provided");

  if (verbose > 2)
    cerr << "dsp::Archiver::unload profiles=" << _profiles << endl;
  
  if (_profiles->get_nbin() == 0 || _profiles->get_ndat_folded() == 0)
  {
    if (verbose > 2)
      cerr << "dsp::Archiver::unload ignoring empty sub-integration" << endl;
    return;
  }

  on_host( _profiles, profiles );

  uint64_t ndat_folded = profiles->get_ndat_folded();
  uint64_t ndat_total = profiles->get_ndat_total();
  double percent = double(ndat_folded)/double(ndat_total) * 100.0;

  if (verbose > 2)
    cerr << "dsp::Archiver::unload folded " << ndat_folded << " out of "
         << ndat_total << " total samples: " << percent << "%" << endl;

  uint64_t ndat_expected = profiles->get_ndat_expected();
  if (ndat_expected && ndat_expected < 0.9 * ndat_total)
  {
    /*
      ndat_expected is the number of samples expected to be totalled in
      the sub-integration.  This number can possibly differ from ndat_total
      due to different rounding in different thread (an untested assertion).
    */

    if (verbose > 2)
      cerr << "dsp::Archiver::unload ignoring incomplete sub-integration \n\t"
        "expected=" << ndat_expected << " total=" << ndat_total << endl;

    return;
  }

  if (profiles->get_integration_length() < minimum_integration_length)
  {
    cerr << "dsp::Archiver::unload ignoring " 
         << profiles->get_integration_length() << " seconds of data" << endl;

    return;
  }

  if (use_single_archive)
  {
    // Generate new archive if needed
    if (!single_archive)
    {
      if (verbose > 2)
	cerr << "dsp::Archiver::unload creating new single Archive" << endl;
      single_archive = new_Archive();
    }

    // refer to the single archive to which all sub-integration will be written
    archive = single_archive;

    // add the profile data
    add (archive, profiles);

    // See if we've reached max subints per file, if not, we're done for now
    if (subints_per_file == 0 || (archive->get_nsubint() < subints_per_file))
      return;

    // Max subint limit reached so we need to unload this file and
    // start a new one next time.
    finish();
    single_archive = 0;

    return;
  }

  if (!archive)
    archive = new_Archive();

  if (verbose > 2)
    cerr << "dsp::Archiver::unload set Pulsar::Archive" << endl;

  set (archive, profiles);

  if (verbose > 2)
    cerr << "dsp::Archiver::unload postprocess" << endl;

  postprocess (archive);

  if (verbose > 2)
    cerr << "dsp::Archiver::unload archive '"
         << archive->get_filename() << "'" << endl;
    
  RealTimer timer;
  if (dsp::Operation::record_time)
    timer.start();

  if (verbose > 2)
    cerr << "dsp::Archiver::unload archive=" << archive.get() << endl;

  archive -> unload();
  
  if (dsp::Operation::record_time)
  {
    timer.stop();
    cerr << "dsp::Archiver::unload in " << timer << endl;
  }

}
catch (Error& error)
{
  throw error += "dsp::Archiver::unload";
}

void dsp::Archiver::finish () try
{
  if (!single_archive)
  {
    if (verbose > 2)
      cerr << "dsp::Archiver::finish no archive" << endl;
    return;
  }

  cerr << "dsp::Archiver::finish archive '"
       << single_archive->get_filename() << "' with "
       << single_archive->get_nsubint () << " integrations" << endl;

  postprocess ( single_archive );

  if (verbose > 2)
    cerr << "dsp::Archiver::finish archive=" << single_archive.get() << endl;

  if (single_archive->get_nsubint ())
    single_archive->unload ();
}
catch (Error& error)
{
  throw error += "dsp::Archiver::finish";
}

void dsp::Archiver::postprocess (Pulsar::Archive* data) try
{
  if (script.size() == 0)
    return;

  if (verbose > 2)
    cerr << "dsp::Archiver::postprocess data=" << data << endl;

  if (verbose > 2)
    cerr << "dsp::Archiver::postprocess script starting" << endl;

  if (!interpreter)
    interpreter = standard_shell();

  interpreter->set( data );
  interpreter->script( script );

  if (verbose > 2)
    cerr << "dsp::Archiver::postprocess script finished" << endl;

  Pulsar::Archive* processed = interpreter->get ();

  if ( processed != data )
    throw Error (InvalidState, "dsp::Archiver::postprocess",
                 "cannot handle interpreter input != output");

  if (verbose > 2)
    cerr << "dsp::Archiver::postprocess processed=" << processed << endl;
}
catch (Error& error)
{
  throw error += "dsp::Archiver::postprocess";
}

void dsp::Archiver::add (Pulsar::Archive* archive, const PhaseSeries* phase)
try 
{
  if (verbose > 2)
    cerr << "dsp::Archiver::add Pulsar::Archive" << endl;

  if (!archive)
    throw Error (InvalidParam, "dsp::Archiver::add Pulsar::Archive",
        	 "no Archive");

  if (!phase) 
    throw Error (InvalidParam, "dsp::Archiver::add Pulsar::Archive",
        	 "no PhaseSeries");

  unsigned nsub = archive->get_nsubint();

  if (!nsub)
  {
    set (archive, phase);
    return;
  }

  unsigned npol = phase->get_npol() * phase->get_ndim();

  // simple sanity check
  if (archive->get_npol() != npol)
    throw Error (InvalidParam, "dsp::Archiver::add Pulsar::Archive",
        	 "Pulsar::Archive::npol=%d != PhaseSeries::npol=%d",
        	 archive->get_npol(), npol);
  if (archive->get_nchan() != phase->get_nchan())
    throw Error (InvalidParam, "dsp::Archiver::add Pulsar::Archive",
        	 "Pulsar::Archive::nchan=%d != PhaseSeries::nchan=%d",
        	 archive->get_nchan(), phase->get_nchan());
  if (archive->get_nbin() != phase->get_nbin())
    throw Error (InvalidParam, "dsp::Archiver::add Pulsar::Archive",
        	 "Pulsar::Archive::nbin=%d != PhaseSeries::nbin=%d",
        	 archive->get_nbin(), phase->get_nbin());
  
  archive-> resize (nsub + 1);
  set (archive-> get_Integration(nsub), phase, nsub);
}
catch (Error& error)
{
  throw error += "dsp::Archiver::add Pulsar::Archive";
}

unsigned dsp::Archiver::get_npol (const PhaseSeries* phase) const
{
  unsigned npol = phase->get_npol();
  unsigned ndim = phase->get_ndim();

  unsigned effective_npol = npol * ndim;

  fourth_moments = 0;

  if ( phase->get_state() == Signal::FourthMoment )
  {
    if (verbose > 2)
      cerr << "dsp::Archiver::get_npol fourth moments" << endl;

    if (effective_npol != 14)
      throw Error (InvalidParam, "dsp::Archiver::get_npol",
        	   "state==FourthMoment and PhaseSeries::npol=%u != 14", 
        	   effective_npol);

    fourth_moments = effective_npol - 4;
    effective_npol = 4;
  }

  return effective_npol;
}

void dsp::Archiver::set (Pulsar::Archive* archive, const PhaseSeries* phase)
try
{
  if (verbose > 2)
    cerr << "dsp::Archiver::set Pulsar::Archive" << endl;

  if (!archive)
    throw Error (InvalidParam, "dsp::Archiver::set Pulsar::Archive",
        	 "no Archive");

  if (!phase)
    throw Error (InvalidParam, "dsp::Archiver::set Pulsar::Archive",
        	 "no PhaseSeries");

  const unsigned npol  = get_npol (phase);
  const unsigned nchan = phase->get_nchan();
  const unsigned nbin  = phase->get_nbin();
  const unsigned nsub  = 1;

  if (verbose > 2)
    cerr << "dsp::Archiver::set Pulsar::Archive nsub=" << nsub 
         << " npol=" << npol << " nchan=" << nchan 
         << " nbin=" << nbin << " fourth=" << fourth_moments << endl;

  archive-> resize (nsub, npol, nchan, nbin);

  Pulsar::FITSHdrExtension* ext;
  ext = archive->get<Pulsar::FITSHdrExtension>();
  
  if (ext)
  {
    if (verbose > 2)
      cerr << "dsp::Archiver::set Pulsar::Archive FITSHdrExtension" << endl;

    // Make sure the start time is aligned with pulse phase zero
    // as this is what the PSRFITS format expects.

    MJD initial = phase->get_start_time();

    if (phase->has_folding_predictor())
    {
      Pulsar::Phase inphs = phase->get_folding_predictor()->phase(initial);
      double dtime = inphs.fracturns() * phase->get_folding_period();
      initial -= dtime;
    }

    ext->set_start_time (initial);

    // In keeping with tradition, I'll set this to a value that should
    // work in most places for the next 50 years or so ;)

    ext->set_coordmode("J2000");

    // Set the ASCII date stamp from the system clock (in UTC)

    time_t thetime;
    time(&thetime);
    string time_str = asctime(gmtime(&thetime));

    // Cut off the line feed character
    time_str = time_str.substr(0,time_str.length() - 1);

    ext->set_date_str(time_str);
  }

  archive-> set_telescope ( phase->get_telescope() );
  archive-> set_type ( phase->get_type() );

  switch (phase->get_state())
  {
  case Signal::NthPower:
  case Signal::PP_State:
  case Signal::QQ_State:
    archive->set_state (Signal::Intensity);
    break;
  
  case Signal::FourthMoment:
    archive->set_state (Signal::Stokes);
    break;

  default:
    archive-> set_state ( phase->get_state() );
  }

  archive-> set_scale ( Signal::FluxDensity );

  if (verbose > 2)
    cerr << "dsp::Archiver::set Archive source=" << phase->get_source()
         << "\n  coord=" << phase->get_coordinates()
         << "\n  bw=" << phase->get_bandwidth()
         << "\n  freq=" << phase->get_centre_frequency () << endl;

  archive-> set_source ( phase->get_source() );
  archive-> set_coordinates ( phase->get_coordinates() );
  archive-> set_bandwidth ( phase->get_bandwidth() );
  archive-> set_centre_frequency ( phase->get_centre_frequency() );
  archive-> set_dispersion_measure ( phase->get_dispersion_measure() );

  archive-> set_dedispersed( archive_dedispersed );
  archive-> set_faraday_corrected (false);

  /* *********************************************************************
     *********************************************************************

     Set any available extensions

     *********************************************************************
     ********************************************************************* */

  /*
    A valid Backend extension may have already been created; e.g. by
    the OutputArchive extension class.  Therefore, if the Backend extension
    already exists, do not modifiy it.
  */

  Pulsar::Backend* backend = archive -> get<Pulsar::Backend>();
  if (!backend)
  {
    backend = new Pulsar::Backend;

    if (verbose > 2)
      cerr << "dsp::Archiver::set Pulsar::Backend extension" << endl;

    set (backend);

    archive->add_extension( backend );
  }


  // Note, this is now called before the set(Integration,...) call below
  // so that the DigitiserCounts extension gets set up correctly the
  // first time.

  Pulsar::dspReduction* dspR = archive -> getadd<Pulsar::dspReduction>();
  if (dspR)
  {
    if (verbose > 2)
      cerr << "dsp::Archiver::set Pulsar::dspReduction extension" << endl;
    set (dspR);
  }

  for (unsigned isub=0; isub < nsub; isub++)
    set (archive->get_Integration(isub), phase, isub, nsub);

  if (store_dynamic_extensions)
  {
    Pulsar::TwoBitStats* tbc = archive -> getadd<Pulsar::TwoBitStats>();
    if (tbc)
    {
      if (verbose > 2)
        cerr << "dsp::Archiver::set Pulsar::TwoBitStats extension" << endl;
      set (tbc);
    }

    Pulsar::Passband* pband = archive -> getadd<Pulsar::Passband>();
    if (pband)
    {
      if (verbose > 2)
        cerr << "dsp::Archiver::set Pulsar::Passband extension" << endl;
      set (pband);
    }
  }

  Pulsar::Telescope* telescope = archive -> getadd<Pulsar::Telescope>();

  try
  {
    telescope->set_coordinates ( phase -> get_telescope() );
  }
  catch (Error& error)
  {
    if (verbose > 2)
      cerr << "dsp::Archiver WARNING could not set telescope coordinates\n\t"
           << error.get_message() << endl;
  }

  // default Receiver extension
  Pulsar::Receiver* receiver = archive -> getadd<Pulsar::Receiver>();
  receiver->set_name ( phase -> get_receiver() );
  receiver->set_basis ( phase -> get_basis() );

  for (unsigned iext=0; iext < extensions.size(); iext++)
    archive -> add_extension ( extensions[iext] );

  // set_model must be called after the Integration::MJD has been set
  if( phase->has_folding_predictor() )
  {
    if (verbose > 2)
      cerr << "dsp::Archiver::set has predictor" << endl;
    archive-> set_model ( phase->get_folding_predictor(), false );
  }
  else if (verbose > 2)
    cerr << "dsp::Archiver::set PhaseSeries has no predictor" << endl;

  if (phase->has_pulsar_ephemeris())
    archive-> set_ephemeris( phase->get_pulsar_ephemeris(), false );

  archive-> set_filename (get_filename (phase));

  if (verbose > 2) cerr << "dsp::Archiver set archive filename to '"
        	    << archive->get_filename() << "'" << endl;


}
catch (Error& error)
{
  throw error += "dsp::Archiver::set Pulsar::Archive";
}


void dsp::Archiver::set (Pulsar::Integration* integration,
        		 const PhaseSeries* phase,
        		 unsigned isub, unsigned nsub) 
try
{
  if (verbose > 2)
    cerr << "dsp::Archiver::set Pulsar::Integration" << endl;

  const unsigned npol  = phase->get_npol();
  const unsigned nchan = phase->get_nchan();
  const unsigned ndim  = phase->get_ndim();
  const unsigned nbin  = phase->get_nbin();

  const unsigned effective_npol = get_npol (phase);

  if ( integration->get_npol() != effective_npol)
    throw Error (InvalidParam, "dsp::Archiver::set (Pulsar::Integration)",
                 "Integration.npol=%d != PhaseSeries.npol=%d", 
                 integration->get_npol(), effective_npol);

  if ( integration->get_nchan() != nchan)
    throw Error (InvalidParam, "dsp::Archiver::set (Pulsar::Integration)",
                 "Integration.nchan=%d != PhaseSeries.nchan=%d", 
                 integration->get_nchan(), nchan);

  integration-> set_epoch ( phase->get_mid_time() );
  integration-> set_duration ( phase->get_integration_length() );

  integration-> set_folding_period ( phase->get_folding_period () );

  if (verbose > 2)
    cerr << "dsp::Archiver::set"
         << " epoch=" << integration->get_epoch().printdays(13)
         << " duration=" << integration->get_duration()
         << " period=" << integration->get_folding_period() << endl;

  if (archive_dedispersed)
  {
    Pulsar::Dedisperse* corrected = new Pulsar::Dedisperse;
    corrected->set_dispersion_measure( phase->get_dispersion_measure() );
    corrected->set_reference_frequency( phase->get_centre_frequency() );
    integration->add_extension( corrected );
  }

  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    unsigned chan = phase->get_unswapped_ichan (ichan);

    Reference::To<Pulsar::MoreProfiles> more;

    if (fourth_moments)
    {
      more = new Pulsar::FourthMoments;
      more->resize( fourth_moments, nbin );
    }

    if (more)
      integration->get_Profile(0,chan)->add_extension(more);

    corrupted_profiles = 0;

    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      for (unsigned idim=0; idim<ndim; idim++)
      {
        unsigned poln = ipol*ndim+idim;

        if (nsub > 1)
          idim = isub;

        Pulsar::Profile* profile = 0;

        double scale = phase->get_scale ();

        if (more && poln >= effective_npol)
        {
          profile = more->get_Profile (poln - effective_npol);
          scale *= scale; // FourthMoments start with Stokes squared
        }
        else
          profile = integration->get_Profile (poln, chan);

        if (verbose > 2)
          cerr << "dsp::Archiver::set Pulsar::Integration ipol=" << poln
               << " ichan=" << chan << " nbin=" << profile->get_nbin() << endl;

        set (profile, phase, scale, ichan, ipol, idim);

      }
    }

    if (corrupted_profiles)
    {
      cerr << "dsp::Archiver::set Pulsar::Integration ichan=" << ichan
             << " contains corrupted data" << endl;

      integration->set_weight (ichan, 0);
    }
    else if (fourth_moments)
    {
      cerr << "dsp::Archiver::set fourth_moments=true" << endl;
      const unsigned * hits = 0;
      if (phase->get_zeroed_data())
        hits = phase->get_hits(ichan);
      else
        hits = phase->get_hits();
      raw_to_central (chan, more, integration, hits);
    }
  }

  if (hist_unpacker)
  {
    // Add DigitiserCounts histograms for this subint.
    Pulsar::Archive *arch = const_cast<Pulsar::Archive *> 
      ( integration -> expert() -> get_parent() );
    Pulsar::DigitiserCounts *dcnt = arch -> getadd<Pulsar::DigitiserCounts>();
    if (dcnt)
    {
      if (verbose > 2)
	cerr << "dsp::Archiver::set Pulsar::DigitiserCounts extension" << endl;
      set (dcnt, isub);
    }
  }
}

catch (Error& error)
{
  throw error += "dsp::Archiver::set Pulsar::Integration";
}

/*! subtract the mean squared from each moment */
void dsp::Archiver::raw_to_central (unsigned ichan,
        			    Pulsar::MoreProfiles* moments,
        			    const Pulsar::Integration* means,
        			    const unsigned* hits)
{
  const unsigned npol = means->get_npol();
  const unsigned nbin = means->get_nbin();

  unsigned index = 0;
  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    const float* mean_i = means->get_Profile(ipol, ichan)->get_amps();

    for (unsigned jpol=ipol; jpol<npol; jpol++)
    {
      const float* mean_j = means->get_Profile(jpol, ichan)->get_amps();

      float* moment = moments->get_Profile(index)->get_amps();

      for (unsigned ibin=0; ibin < nbin; ibin++)
      {
        double pi = mean_i[ibin];
        double pj = mean_j[ibin];

        // divide by hits again to form variance of mean
        moment[ibin] = (moment[ibin] - pi*pj) / hits[ibin];
      }

      index ++;
    }
  }

  assert (index == moments->get_size());
}

void dsp::Archiver::set (Pulsar::Profile* profile,
        		 const PhaseSeries* phase, double scale,
        		 unsigned ichan, unsigned ipol, unsigned idim)
try
{
  if (verbose > 2)
    cerr << "dsp::Archiver::set Pulsar::Profile"
      " ichan=" << ichan << " ipol=" << ipol << " idim=" << idim << endl;

  const unsigned nbin = phase->get_nbin();
  const unsigned npol = phase->get_npol();
  const unsigned ndim = phase->get_ndim();
  const unsigned nchan = phase->get_nchan();

  assert (ipol < npol);
  assert (idim < ndim);

  profile-> resize (nbin);
  profile-> set_centre_frequency (phase->get_centre_frequency (ichan));
  profile-> set_weight (1.0);

  float* into = profile->get_amps ();
  const float* from = 0;
  unsigned nstride = 0;

  const unsigned * hits = 0;
  if (phase->get_zeroed_data())
    hits = phase->get_hits(ichan);
  else
    hits = phase->get_hits();

  if (phase->get_order() == TimeSeries::OrderFPT)
  {
    from = phase->get_datptr (ichan, ipol) + idim;
    nstride = ndim;
  }
  else if (phase->get_order() == TimeSeries::OrderTFP)
  {
    from = phase->get_dattfp() + ichan * npol*ndim + ipol * ndim + idim;
    nstride = nchan * npol * ndim;
  }
  
  unsigned zeroes = 0;

  if (verbose > 2)
    cerr << "dsp::Archiver::set Pulsar::Profile scale=" << scale << endl;

  if (scale == 0 || !isfinite(scale))
    throw Error (InvalidParam, string(), "invalid scale=%lf", scale);

  unsigned not_finite = 0;
  unsigned hits_sum = 0;

  for (unsigned ibin = 0; ibin<nbin; ibin++)
  {
    hits_sum += hits[ibin];

    if (hits[ibin] == 0)
    {
      zeroes ++;
      into[ibin] = 0.0;
    }
    else if (!isfinite(*from))
    {
      not_finite ++;
      if (verbose > 2)
        cerr << "non-finite: hit=" << hits[ibin] << endl;
    }
    else
      into[ibin] = *from / (scale * double( hits[ibin] ));

    from += nstride;
  }

  if (not_finite)
  {
    if (verbose > 2)
      cerr << "dsp::Archiver::set Pulsar::Profile WARNING " << not_finite
           << " out of " << nbin << " bins with non-finite amplitudes" << endl;

    for (unsigned ibin = 0; ibin<nbin; ibin++)
      into[ibin] = 0;

    corrupted_profiles ++;

    profile->set_weight(0);
  }

  if (zeroes)
  {
    if (verbose > 2)
      cerr << "dsp::Archiver::set Pulsar::Profile WARNING " << zeroes 
                 << " out of " << nbin << " bins with zero hits" << endl;

    // find the mean of the hit bins
    double sum = 0.0;
    unsigned count = 0;
    for (unsigned ibin = 0; ibin<nbin; ibin++)
    if (hits[ibin] != 0)
    {
      sum += into[ibin];
      count ++;
    }

    // avoid division by zero
    if (count == 0)
      count = 1;

    // set the unhit bins to the mean
    double mean = sum / count;
    for (unsigned ibin = 0; ibin<nbin; ibin++)
      if (hits[ibin] == 0)
        into[ibin] = mean;
  }
  if (phase->get_zeroed_data())
  {
    float chan_weight = (float) hits_sum / (float) phase->get_ndat_total();
    profile->set_weight (chan_weight);
  }

}
catch (Error& error)
{
  throw error += "dsp::Archiver::set Pulsar::Profile";
}

