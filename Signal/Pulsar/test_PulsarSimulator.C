/***************************************************************************
 *
 *   Copyright (C) 2018 by Willem van Straten and Axl Rogers
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <iostream>
#include <unistd.h>

#include "dsp/PulsarSimulator.h"
#include "dsp/TimeSeries.h"

#include "strutil.h"
#include "dirutil.h"

static char* args = "b:n:op:t:vV";

using namespace std;

void usage ()
{
  cout << "test_Fold - test phase coherent dedispersion kernel\n"
    "Usage: test_Fold [" << args << "] file1 [file2 ...] \n"
       << endl;
}

int main (int argc, char** argv) 
{
try {

  Error::verbose = true;
  Error::complete_abort = true;

  char* metafile = 0;
  bool verbose = false;

  // number of time samples loaded from file at a time
  int block_size = 512*1024;
  int blocks = 0;
  int ndim = 1;
  int nbin = 1024;
  Signal::State output_state = Signal::Coherence;
  bool inplace_detection = true;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'V':
      dsp::Operation::verbose = true;
      dsp::Observation::verbose = true;
    case 'v':
      verbose = true;
      break;

    case 'b':
      nbin = atoi (optarg);
      break;

    case 't':
      blocks = atoi (optarg);
      break;

    case 'n':
      ndim = atoi (optarg);
      break;

    case 'o': inplace_detection = false; break;

    case 'p':
      {
	unsigned npol = atoi(optarg);
	if( npol==1 ) output_state = Signal::Intensity;
	if( npol==4 ) output_state = Signal::Coherence;
	break;
      }

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  vector <string> filenames;

  if (metafile)
    stringfload (&filenames, metafile);
  else 
    for (int ai=optind; ai<argc; ai++)
      dirglob (&filenames, argv[ai]);

#if 0
  if (filenames.size() == 0) {
    usage ();
    return -1;
  }
#endif

  if (verbose)
    cerr << "Creating TimeSeries instance" << endl;
  dsp::TimeSeries voltages;


  dsp::PulsarSimulator simulator;
  simulator.operate();

  fprintf(stderr,"biyee!\n");
  return 0;
}

catch (Error& error) {
  cerr << "Error thrown: " << error << endl;
  return -1;
}

catch (string& error) {
  cerr << "exception thrown: " << error << endl;
  return -1;
}

catch (...) {
  cerr << "unknown exception thrown." << endl;
  return -1;
}
 fprintf(stderr,"At end of main()\n"); 
}
