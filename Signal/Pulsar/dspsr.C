/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/dsp.h"
#include "dsp/File.h"
#include "dsp/MultiFile.h"

#include "dsp/LoadToFoldConfig.h"
#include "dsp/LoadToFold1.h"
#include "dsp/LoadToFoldN.h"

#include "Pulsar/Archive.h"
#include "Pulsar/Parameters.h"
#include "Pulsar/Predictor.h"

#include "FTransform.h"

#include "load_factory.h"
#include "dirutil.h"
#include "strutil.h"

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

using namespace std;

void parse_options (int argc, char** argv);

// sets up the pipeline using the following attributes

void prepare (dsp::Pipeline* engine, dsp::Input* input);

// number of seconds to adjust clocks by
double offset_clock = 0.0;

// set the MJD
string mjd_string;

// set the telescope code
string telescope;

// bandwidth
double bandwidth = 0.0;
// centre_frequency
double centre_frequency = 0.0;
// Pulsar name
string pulsar_name;

// Specified command line options that apply to baseband data only
string baseband_options;

// The LoadToFold configuration parameters
Reference::To<dsp::LoadToFold::Config> config;

// Number of threads used to process the data
unsigned nthread = 0;

int main (int argc, char** argv) try
{
  config = new dsp::LoadToFold::Config;

  parse_options (argc, argv);

  Reference::To<dsp::Pipeline> engine;

  if (config->get_total_nthread() > 1)
  {
    if(dsp::Observation::verbose)
      cerr << "using dsp::LoadToFoldN" << endl;

    engine = new dsp::LoadToFoldN (config);
  }
  else
  {
    if(dsp::Observation::verbose)
      cerr << "using dsp::LoadToFold" << endl;

    engine = new dsp::LoadToFold (config);
  }

  bool time_prep = dsp::Operation::record_time || config->get_cuda_ndevice();

  RealTimer preptime;
  if (time_prep)
    preptime.start();

  prepare (engine, config->open (argc, argv));

  if (time_prep)
  {
    preptime.stop();
    cerr << "dspsr: prepared in " << preptime << endl;
  }

  engine->run();
  engine->finish();

  return 0;
}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

void input_prepare (dsp::Input* input)
{
  dsp::Observation* info = input->get_info();

  if (bandwidth != 0)
  {
    cerr << "dspsr: over-riding bandwidth"
      " old=" << info->get_bandwidth() << " MHz"
      " new=" << bandwidth << " MHz" << endl;
    info->set_bandwidth (bandwidth);

    if (info->get_state() == Signal::Nyquist)
    {
      info->set_rate( fabs(bandwidth) * 2e6 );
      cerr << "dspsr: corrected Nyquist (real-valued) sampling rate="
           << info->get_rate() << " Hz" << endl;
    }
    else if (info->get_state () == Signal::Analytic)
    {
      info->set_rate( fabs(bandwidth) * 1e6 );
      cerr << "dspsr: corrected Analytic (complex-valued) sampling rate="
           << info->get_rate() << " Hz" << endl;
    }
  }

  if (centre_frequency != 0)
  {
    cerr << "dspsr: over-riding centre_frequency"
      " old=" << info->get_centre_frequency() <<
      " new=" << centre_frequency << endl;
    info->set_centre_frequency (centre_frequency);
  }

  if (!telescope.empty())
  {
    cerr << "dspsr: over-riding telescope code"
      " old=" << info->get_telescope() <<
      " new=" << telescope << endl;
    info->set_telescope (telescope);
  }

  if (!pulsar_name.empty())
  {
    cerr << "dspsr: over-riding source name"
      " old=" << info->get_source() <<
      " new=" << pulsar_name << endl;
    info->set_source( pulsar_name );
  }

  if (!mjd_string.empty())
  {
    MJD mjd (mjd_string);
    cerr << "dspsr: over-riding start time"
      " old=" << info->get_start_time() <<
      " new=" << mjd << endl;
    info->set_start_time( mjd );
  }

  if (offset_clock)
  {
     MJD old = info->get_start_time();
     cerr << "dspsr: offset clock by " << offset_clock << " seconds" << endl;
     info->set_start_time( old + offset_clock );
  }

  info->set_dispersion_measure (config->dispersion_measure);
}

void prepare (dsp::Pipeline* engine, dsp::Input* input)
{
  config->input_prepare.set( input_prepare );

  engine->set_input( input );

  dsp::Observation* info = input->get_info();

  if (info->get_detected() && !baseband_options.empty())
    throw Error (InvalidState, "prepare",
		 "input type " + input->get_name() +
		 " yields detected data and the command line option(s):"
		 "\n\n" + baseband_options + "\n\n"
		 " are specific to baseband (undetected) data.");

  engine->construct ();
  engine->prepare ();
}

#include "CommandLine.h"

void parse_options (int argc, char** argv) try
{

  CommandLine::Menu menu;
  CommandLine::Argument* arg;

  menu.set_help_header ("dspsr - digital signal processing of pulsar signals");
  menu.set_version ("dspsr " + tostring(dsp::version) +
		    " <" + FTransform::get_library() + ">");

  /* ***********************************************************************

  General Processing Options

  *********************************************************************** */

  config->add_options (menu);

  arg = menu.add (config->optimal_order, "order");
  arg->set_help ("order data optimally when possible [default:true]");

  string ram_min;
  arg = menu.add (ram_min, "minram", "MB");
  arg->set_help ("minimum RAM usage in MB");

  string ram_limit;
  arg = menu.add (ram_limit, 'U', "MB|minX");
  arg->set_help ("upper limit on RAM usage");
  arg->set_long_help
    ("specify either the floating point number of megabytes; e.g. -U 256 \n"
     "or a multiple of the minimum possible block size; e.g. -U minX2 \n");

  arg = menu.add (config->apply_FITS_scale_and_offset, "scloffs");
  arg->set_help ("denormalize using DAT_SCL and DAT_OFFS [PSRFITS]");

  /* ***********************************************************************

  Source Options

  *********************************************************************** */

  menu.add ("\n" "Source options:");

  arg = menu.add (bandwidth, 'B', "bandwidth");
  arg->set_help ("set the bandwidth in MHz");

  arg = menu.add (centre_frequency, 'f', "frequency");
  arg->set_help ("set the centre frequency in MHz");

  arg = menu.add (telescope, 'k', "telescope");
  arg->set_help ("set the telescope name");

  arg = menu.add (pulsar_name, 'N', "name");
  arg->set_help ("set the source name");

  /* ***********************************************************************

  Clock/Time Options

  *********************************************************************** */

  menu.add ("\n" "Clock/Time options:");

  arg = menu.add (offset_clock, 'C', "offset");
  arg->set_help ("adjust clock by offset seconds");

  arg = menu.add (mjd_string, 'm', "MJD");
  arg->set_help ("set the start MJD of the observation");

  /* ***********************************************************************

  RFI Removal (SK) Options

  *********************************************************************** */
  menu.add ("\n" "RFI removal options:");

  vector<string> unpack;
  arg = menu.add (unpack, '2', "code");
  arg->set_help ("unpacker options (\"2-bit\" excision)");
  arg->set_long_help
    (" -2c<cutoff>    threshold for impulsive interference excision \n"
     " -2n<sample>    number of samples used to estimate undigitized power \n"
     " -2t<threshold> two-bit sampling threshold at record time \n");


  arg = menu.add (config->sk_zap, "skz");
  arg->set_help ("apply spectral kurtosis filterbank RFI zapping");

  arg = menu.add (config->nosk_too, "noskz_too");
  arg->set_help ("also produce un-zapped version of output");

  arg = menu.add (config.get(), &dsp::LoadToFold::Config::set_sk_m, "skzm", "samples");
  arg->set_help ("samples to integrate for spectral kurtosis statistics");

  arg = menu.add (config.get(), &dsp::LoadToFold::Config::set_sk_noverlap, "skzover", "integer");
  arg->set_help ("oversampling factor (skzover must evenly divide skzm)");

  arg = menu.add (config.get(), &dsp::LoadToFold::Config::set_sk_std_devs, "skzs", "stddevs");
  arg->set_help ("number of std deviations to use for spectral kurtosis excisions");

  arg = menu.add (config->sk_chan_start, "skz_start", "chan");
  arg->set_help ("first channel where signal is expected");

  arg = menu.add (config->sk_chan_end, "skz_end", "chan");
  arg->set_help ("last channel where signal is expected");

#if HAVE_YAMLCPP
  arg = menu.add (config->sk_config, "skz_config", "sk.yaml");
  arg->set_help ("load SK configuration from YAML file");
#endif

  arg = menu.add (config->sk_no_fscr, "skz_no_fscr");
  arg->set_help ("do not use SKDetector Fscrunch feature");

  arg = menu.add (config->sk_no_tscr, "skz_no_tscr");
  arg->set_help ("do not use SKDetector Tscrunch feature");

  arg = menu.add (config->sk_no_ft, "skz_no_ft");
  arg->set_help ("do not use SKDetector despeckeler");

#ifdef HAVE_CUFFT
  arg = menu.add (config->sk_nthreads, "skzn", "threads");
  arg->set_help ("number of CPU threads for spectral kurtosis filterbank");
#endif

  arg = menu.add (config->sk_fold, "sk_fold");
  arg->set_help ("fold the SKFilterbank output");

  /* ***********************************************************************

  Dispersion removal Options

  *********************************************************************** */

  menu.add ("\n" "Dispersion removal options:");

  arg = menu.add (config->filterbank, 'F', "<N>[:D]");
  arg->set_help ("create an N-channel filterbank");
  arg->set_long_help
    ("<N> is the number of channels output by the filterank; e.g. -F 256 \n"
     "\n"
     "Reduce the spectral leakage function bandwidth with -F 256:<M> \n"
     "where <M> is the reduction factor."
     "\n"
     "If DM != 0, coherent dedispersion will be performed \n"
     " - after the filterbank with -F 256 or -F 256:<M>\n"
     " - during the filterbank with -F 256:D \n"
     " - before the filterbank with -F 256:B \n" );

  arg = menu.add (config->inverse_filterbank, "IF", "<N>[:D]");
  arg->set_help( "create inverse filterbank with N output channels");
  arg->set_long_help
    ("<N> is the number of channels output by the inverse filterank; e.g. -IF 256 \n"
     "\n"
     "If DM != 0, coherent dedispersion will be performed \n"
     " - after the inverse filterbank with -F 256:<M>:<O>\n"
     "   where M is the size of the forward FFT and\n"
     "   O is the size of the overlap region\n"
     " - during the inverse filterbank with -F 256:D \n");

  arg = menu.add (config->do_deripple, "dr");
  arg->set_help( "Apply deripple correction to inverse filterbank");

  arg = menu.add (config->plfb_nbin, 'G', "nbin");
  arg->set_help ("create phase-locked filterbank");

  arg = menu.add (config->cyclic_nchan, "cyclic", "N");
  arg->set_help ("form cyclic spectra with N channels (per input channel)");

  arg = menu.add (config->cyclic_mover, "cyclicoversample", "M");
  arg->set_help ("use M times as many lags to improve cyclic channel isolation (4 is recommended)");

  string taper_help = "Available tapering functions:\n"
     "\t hanning, welch, bartlett, tukey, top_hat (default: none)\n";

  arg = menu.add (config->temporal_apodization_type, "t-taper", "name");
  arg->set_help ("name of temporal apodization/tapering/window function");
  arg->set_long_help (taper_help);

  arg = menu.add (config->spectral_apodization_type, "f-taper", "name");
  arg->set_help ("name of spectral apodization/tapering/window function");
  arg->set_long_help (taper_help);

  double dm = -1.0;
  arg = menu.add (dm, 'D', "dm");
  arg->set_help ("over-ride dispersion measure");

  arg = menu.add (config->interchan_dedispersion, 'K');
  arg->set_help ("remove inter-channel dispersion delays");

  string fft_length;
  arg = menu.add (fft_length, 'x', "nfft|minX");
  arg->set_help ("over-ride optimal transform length");
  arg->set_long_help
    ("either specify the desired transform length; e.g, -x 32768 \n"
     "or request the minimum possible length be used via -x min\n"
     "or a multiple of the minimum length; e.g. -x minX2");

  arg = menu.add (config->zap_rfi, 'R');
  arg->set_help ("apply time-variable narrow-band RFI filter");

  arg = menu.add (config->calibrator_database_filename, "pac", "dbase");
  arg->set_help ("pac database for phase-coherent matrix convolution");
  arg->set_long_help
    ("specify the name of a database created by pac from which to select\n"
     "the polarization calibrator to be used for matrix convolution");

  arg = menu.add (config->use_fft_bench, "fft-bench");
  arg->set_help ("use benchmark data to choose optimal FFT length");

  /* ***********************************************************************

  Detection Options

  *********************************************************************** */

  menu.add ("\n" "Detection options:");

  arg = menu.add (config->npol, 'd', "npol");
  arg->set_help ("0=P,Q; 1=PP+QQ; 2=PP,QQ; 3=(PP+QQ)^2; 4=PP,QQ,PQ,QP");

  arg = menu.add (config->ndim, 'n', "ndim");
  arg->set_help ("[experimental] ndim of output when npol=4");

  arg = menu.add (config->fourth_moment, '4');
  arg->set_help ("compute fourth-order moments");

  /* ***********************************************************************

  Folding Options

  *********************************************************************** */

  menu.add ("\n" "Folding options:");

  int nbin = 0;
  arg = menu.add (nbin, 'b', "nbin");
  arg->set_help ("number of phase bins in folded profile");

  arg = menu.add (config->folding_period, 'c', "period");
  arg->set_help ("folding period (in seconds)");

  arg = menu.add (config->reference_epoch, "cepoch", "MJD");
  arg->set_help ("reference epoch for phase=0 (when -c is used)");

  arg = menu.add (config->reference_phase, 'p', "phase");
  arg->set_help ("reference phase of rising edge of bin zero");

  vector<string> ephemeris;
  arg = menu.add (ephemeris, 'E', "file");
  arg->set_help ("pulsar ephemeris used to generate predictor");

  vector<string> predictor;
  arg = menu.add (predictor, 'P', "file");
  arg->set_help ("phase predictor used for folding");

  string predictors_file;
  arg = menu.add (predictors_file, 'w', "file");
  arg->set_help ("phase predictors used for folding.");

  arg = menu.add (config->additional_pulsars, 'X', "name");
  arg->set_help ("additional pulsar to be folded");

#if HAVE_CUFFT
  arg = menu.add (config->asynchronous_fold, "asynch-fold");
  arg->set_help ("fold on CPU while processing on GPU");
#endif

  /* ***********************************************************************

  Division Options

  *********************************************************************** */

  menu.add ("\n" "Time division options:");

  arg = menu.add (config->single_archive, 'A');
  arg->set_help ("output single archive with multiple integrations");

  arg = menu.add (config->subints_per_archive, "nsub", "N");
  arg->set_help ("output archives with N integrations each");

  arg = menu.add (config.get(), &dsp::LoadToFold::Config::single_pulse, 's');
  arg->set_help ("create single pulse sub-integrations");

  arg = menu.add (config->integration_turns, "turns", "N");
  arg->set_help ("create integrations of specified number of spin periods");

  arg = menu.add (config->integration_length, 'L', "seconds");
  arg->set_help ("create integrations of specified duration");

  arg = menu.add (config->integration_reference_epoch, "Lepoch", "MJD");
  arg->set_help ("start time of first sub-integration (when -L is used)");

  arg = menu.add (config->minimum_integration_length, "Lmin", "seconds");
  arg->set_help ("minimum integration length output");

  arg = menu.add (config->fractional_pulses, 'y');
  arg->set_help ("output partially completed integrations");

  /* ***********************************************************************

  Output Archive Options

  *********************************************************************** */

  menu.add ("\n" "Output archive options:");

  arg = menu.add (config.get(),
		  &dsp::LoadToFold::Config::set_archive_class, 'a', "archive");
  arg->set_help ("output archive class name");

  arg = menu.add (config->archive_extension, 'e', "ext");
  arg->set_help ("output filename extension");

  arg = menu.add (config->archive_filename, 'O', "name");
  arg->set_help ("output filename");

  arg = menu.add (config->filename_convention, "fname", "convention");
  arg->set_help ("multiple output filename convention");

  arg = menu.add (config->pdmp_output, 'Y');
  arg->set_help ("output pdmp extras");

  arg = menu.add (config->no_dynamic_extensions, "no_dyn");
  arg->set_help ("disable dynamic extensions");

  vector<string> jobs;
  arg = menu.add (jobs, 'j', "job");
  arg->set_help ("psrsh command run before output");

  string script;
  arg = menu.add (script, 'J', "a.psh");
  arg->set_help ("psrsh script run before output");

  /* ***********************************************************************

  Output Archive Options

  *********************************************************************** */


  menu.parse (argc, argv);

  if (config->integration_length && config->minimum_integration_length < 0)
  {
    /*
      rationale: If data are divided into blocks, and blocks are
      sent down different data reduction paths, then it is possible
      for blocks on different paths to overlap by a small amount.

      The minimum integration length is a simple attempt to avoid
      producing a small overlap archive with the same name as the
      full integration length archive.

      If minimum_integration_length is not specified, a default of 10%
      of the integration length is applied.
    */

    config->minimum_integration_length = 0.1 * config->integration_length;
  }

  // interpret the unpacker options

  for (unsigned i=0; i<unpack.size(); i++)
  {
    const char* carg = unpack[i].c_str();

    int scanned = sscanf (carg, "n%u", &config->excision_nsample);
    if (scanned == 1)
    {
      cerr << "dspsr: Using " << config->excision_nsample
	   << " samples to estimate undigitized power" << endl;
      continue;
    }

    scanned = sscanf (carg, "c%f", &config->excision_cutoff);
    if (scanned == 1)
    {
      cerr << "dspsr: Setting impulsive interference excision threshold to "
	   << config->excision_cutoff << endl;
      continue;
    }

    scanned = sscanf (carg, "t%f", &config->excision_threshold);
    if (scanned == 1)
    {
      cerr << "dspsr: Setting two-bit sampling threshold to "
	   << config->excision_threshold << endl;
      continue;
    }
  }

  // interpret the nbin argument
  if (nbin < 0)
  {
    config->force_sensible_nbin = true;
    config->nbin = -nbin;
  }
  if (nbin > 0)
    config->nbin = nbin;

  // over-ride the dispersion measure
  if (dm != -1.0)
  {
    config->dispersion_measure = dm;

    if (dm == 0.0)
    {
      cerr << "dspsr: Disabling coherent dedispersion" << endl;
      config->coherent_dedispersion = false;
    }
  }

  for (unsigned i=0; i<ephemeris.size(); i++)
  {
    cerr << "dspsr: Loading ephemeris from " << ephemeris[i] << endl;
    config->ephemerides.push_back
      ( factory<Pulsar::Parameters> (ephemeris[i]) );
  }

  for (unsigned i=0; i<predictor.size(); i++)
  {
    cerr << "dspsr: Loading phase model from " << predictor[i] << endl;
    config->predictors.push_back
      ( factory<Pulsar::Predictor> (predictor[i]) );
  }

  if(!predictors_file.empty()) {

    cerr << "dspsr: Loading phase models from " << predictors_file << endl;

    vector<char> buffer (10240);
    char* buf = &buffer[0];

    FILE* fptr = fopen (predictors_file.c_str(), "r");
    if (!fptr)
      throw Error (FailedSys, "parse_options",
		   "fopen (%s)", predictors_file.c_str());

    string key_string;
    // choose first non commented and non empty line and attempt to parse header.
    while( fgets (buf, buffer.size(), fptr) ==buf ){

      string temp  = buf;
      temp = stringtok ( temp, "#\n", false);  // get rid of comments and empty lines

      if(temp.empty())
        continue;

      key_string = temp;
      break;

    }
    if(key_string.empty())
      throw Error(InvalidState,"parse_options","Bad input file to -w flag.");

    string delim = " \t\n";

    vector<string> keys;
    string key_next;
    string key_rest;

    cerr << " read header string: " << key_string << endl;

    do {

      string_split_on_any( key_string, key_next, key_rest, delim );

      if(key_next.empty() && !key_rest.empty())
	throw Error (InvalidState,"dspsr", "Key in candiate file was empty.");

      if(key_next.empty() && key_rest.empty())
	key_next = key_string;

      cerr<< "Considering Key = '" << key_next << "'"<<endl;
      keys.push_back(key_next);

      key_string = key_rest;


    }while(!key_rest.empty());

    int nkeys = keys.size();
    int nline = 1;


    cerr << "loaded " << nkeys << " keys." << endl;


    while ( fgets (buf, buffer.size(), fptr) == buf ) {

      nline++;

      string value_string = buf;
      value_string = stringtok (value_string, "#\n", false);  // get rid of comments and empty lines

      if(value_string.empty())
	continue;

      vector<string> values(nkeys);
      stringstream lines;
      string value_next;
      string value_rest;

      for(int i=0; i< nkeys; i++ ) {

        string_split_on_any( value_string, value_next, value_rest, delim );




	if(value_next.empty() && !value_rest.empty()){
	  stringstream err;
	  cerr <<  "Value in candiate file was empty on line " << nline << endl;
	  throw Error (InvalidState,"dspsr", err.str().c_str());
	}

	if(value_next.empty() && value_rest.empty())
	  value_next = value_string;

	if(dsp::Observation::verbose)
	  cerr<< "Considering Key = '" << keys.at(i) << "' value='" << value_next << "'" <<endl;

	lines << keys.at(i) << ": " << value_next << endl;
	value_string = value_rest;
      }

      string ascii_predictor = lines.str();
      char* line_buffer = (char*) (ascii_predictor.c_str());

      FILE* virtual_ptr = fmemopen( line_buffer, strlen(line_buffer) ,"r" );
      if (virtual_ptr)
        config->predictors.push_back ( factory<Pulsar::Predictor> ( virtual_ptr ));
    }
  }


  for (unsigned i=0; i<jobs.size(); i++)
    separate (jobs[i], config->jobs, ",");

  if (!script.empty())
    loadlines (script, config->jobs);

  if (!pulsar_name.empty())
  {
    if (file_exists(pulsar_name.c_str()))
    {
      cerr << "dspsr: Loading source names from " << pulsar_name << endl;
      vector <string> names;
      stringfload (&names, pulsar_name);

      if (names.size())
	pulsar_name = names[0];
      for (unsigned i=1; i < names.size(); i++)
	config->additional_pulsars.push_back ( names[i] );
    }
    else
      cerr << "dspsr: Source name set to " << pulsar_name << endl;
  }

  if (!ram_min.empty())
  {
    double MB = fromstring<double> (ram_min);
    cerr << "dspsr: Using at least " << MB << " MB" << endl;
    config->set_minimum_RAM (uint64_t( MB * 1024.0 * 1024.0 ));
  }

  if (!ram_limit.empty())
  {
    if (ram_limit == "min")
      config->set_times_minimum_ndat( 1 );

    else
    {
      unsigned times = 0;
      if ( sscanf(ram_limit.c_str(), "minX%u", &times) == 1 )
	config->set_times_minimum_ndat( times );

      else
      {
	double MB = fromstring<double> (ram_limit);
	config->set_maximum_RAM (uint64_t( MB * 1024.0 * 1024.0 ));
      }
    }
  }

  if (!fft_length.empty())
  {
    char* carg = strdup( fft_length.c_str() );
    char* colon = strchr (carg, ':');
    if (colon)
    {
      *colon = '\0';
      colon++;
      if (sscanf (colon, "%d", &config->nsmear) < 1)
      {
	fprintf (stderr,
		 "Error parsing '%s' as filterbank frequency resolution\n",
		 colon);
	exit (-1);
      }
    }

    unsigned times = 0;

    if (string(carg) == "min")
      config->times_minimum_nfft = 1;
    else if ( sscanf(carg, "minX%u", &times) == 1 )
      config->times_minimum_nfft = times;
    else
    {
      unsigned nfft = strtol (carg, 0, 10);
      if (colon && config->nsmear >= nfft)
      {
	cerr << "dspsr -x: nfft=" << nfft
	     << " must be greater than nsmear=" << config->nsmear << endl;
	exit (-1);
      }
      config->filterbank.set_freq_res( nfft );
    }
    delete [] carg;
  }
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
