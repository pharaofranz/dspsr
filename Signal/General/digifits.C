/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*
  digifits converts any file format recognized by dspsr into PSRFITS
  search mode (".sf") format.

  Liberally cribbed from digifil.
 */

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/LoadToFITS.h"
#include "dsp/LoadToFITSN.h"
#include "dsp/FilterbankConfig.h"

#include "CommandLine.h"
#include "FTransform.h"

#include <stdlib.h>

using namespace std;

// The LoadToFITS configuration parameters
Reference::To<dsp::LoadToFITS::Config> config;

// names of data files to be processed
vector<string> filenames;

void parse_options (int argc, char** argv);

int main (int argc, char** argv) try
{
  config = new dsp::LoadToFITS::Config;

  parse_options (argc, argv);

  Reference::To<dsp::Pipeline> engine;
  if (config->get_total_nthread() > 1)
    engine = new dsp::LoadToFITSN (config);
  else
    engine = new dsp::LoadToFITS (config);

  engine->set_input( config->open (argc, argv) );
  engine->construct ();   
  engine->prepare ();   
  engine->run();
  engine->finish();
}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

void parse_options (int argc, char** argv) try
{
  CommandLine::Menu menu;
  CommandLine::Argument* arg;
  
  menu.set_help_header ("digifits - convert dspsr input to PSRFITS search mode output");
  menu.set_version ("digifits " + tostring(dsp::version) +
		    " <" + FTransform::get_library() + ">");

  config->add_options (menu);

  // Need to rename default threading option due to conflict
  // with original digifil -t (time average) setting.
  arg = menu.find("t");
  arg->set_short_name('\0'); 
  arg->set_long_name("threads");
  arg->set_type("nthread");

  menu.add ("\n" "Source options:");

  arg = menu.add (config->dispersion_measure, 'D', "dm");
  arg->set_help (" set the dispersion measure");

  menu.add ("\n" "Processing options:");

  arg = menu.add (config->block_size, 'B', "MB");
  arg->set_help ("block size in megabytes");

  string ram_limit;
  arg = menu.add (ram_limit, 'U', "MB");
  arg->set_help ("upper limit on RAM usage");
  arg->set_long_help ("specify the floating point number of megabytes; e.g. -U 256 \n");

  arg = menu.add (&config->filterbank, 
      &dsp::Filterbank::Config::set_freq_res, 
      'x', "nfft");
  arg->set_help ("set backward FFT length in voltage filterbank");

  arg = menu.add (config->coherent_dedisp, "do_dedisp", "bool");
  arg->set_help ("enable coherent dedispersion (default: false)");

  arg = menu.add (config->rescale_constant, 'c');
  arg->set_help ("keep offset and scale constant");

  arg = menu.add (config->rescale_seconds, 'I', "secs");
  arg->set_help ("rescale interval in seconds");

  arg = menu.add (config->apply_FITS_scale_and_offset, "scloffs");
  arg->set_help ("denormalize using DAT_SCL and DAT_OFFS [PSRFITS]");

  menu.add ("\n" "Output options:");

  arg = menu.add (config->tsamp, 't', "tsamp");
  arg->set_help ("integration time (s) per output sample (default=64mus)");

  arg = menu.add (config->npol, 'p', "npol");
  arg->set_help ("output 1 (Intensity), 2 (AABB), or 4 (Coherence) products");

  arg = menu.add (config->nbits, 'b', "bits");
  arg->set_help ("number of bits per sample output to file [1,2,4,8]");

  arg = menu.add (config->filterbank, 'F', "nchan[:D]");
  arg->set_help ("create a filterbank (voltages only)");
  arg->set_long_help
    ("Specify number of filterbank channels; e.g. -F 256\n"
     "Select coherently dedispersing filterbank with -F 256:D\n"
     "Set leakage reduction factor with -F 256:<N>\n");

  arg = menu.add (config->nsblk, "nsblk", "N");
  arg->set_help ("output block size in samples (default=2048)");

  arg = menu.add (config->integration_length, 'L', "seconds");
  arg->set_help ("set maximum file length");

  arg = menu.add (config->dedisperse, 'K');
  arg->set_help ("remove inter-channel dispersion delays");

  arg = menu.add (config->start_time_delay, 's', "seconds");
  arg->set_help ("delay start time");

  arg = menu.add (config->fscrunch_factor, 'f', "nchan");
  arg->set_help ("decimate in frequency by this factor");

  arg = menu.add (config->output_filename, 'o', "file");
  arg->set_help ("output filename");

  arg = menu.add (config->upper_sideband_output, 'u');
  arg->set_help ("produced upper side-band output");

  //bool revert = false;
  //arg = menu.add (revert, 'p');
  //arg->set_help ("revert to FPT order");

  menu.parse (argc, argv);

  if (!ram_limit.empty())
  {
    double MB = fromstring<double> (ram_limit);
    config->set_maximum_RAM (uint64_t( MB * 1024.0 * 1024.0 ));
  }

  //if (revert)
  //  config->order = dsp::TimeSeries::OrderFPT;
}
catch (Error& error)
{
  cerr << error << endl;
  exit (-1);
}
catch (std::exception& error)
{
  cerr << error.what() << endl;
  exit (-1);
}

