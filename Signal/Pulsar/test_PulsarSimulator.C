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

#include "Pulsar/Archive.h"
#include "Pulsar/Parameters.h"

#include "strutil.h"
#include "dirutil.h"

static char* args = "a:e:p:vV";

// e = ephemeris = pulsar parameters from catalog
// p = predictor = output of tempo / tempo2
// a = archive = pulse phase bins that define Stokes parameters

using namespace std;
using namespace Pulsar;

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
  bool verbose = false;

  char* archive_filename = 0;   // open and use as in getStokes.C
  char* parameter_filename = 0; // open and use as in test_Predictor.C
  char* predictor_filename = 0; // open and use to avoid calling the Generator (which takes heaps of time)

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'a':
      archive_filename = optarg;
      break;

    case 'e':
      parameter_filename = optarg;
      break;

    case 'p':
      predictor_filename = optarg;
      break;

    case 'V':
      dsp::Operation::verbose = true;
      dsp::Observation::verbose = true;
    case 'v':
      verbose = true;
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }


  if (!archive_filename)
  {
     cerr << "Please specify the name of the file containing the average pulse profile with -a" << endl;
     return -1;
  }

  if (!parameter_filename && !predictor_filename)
  {
     cerr << "Please specify the name of the file containing either the pulsar parameters (-e) or predictor (-p)" << endl;
     return -1;
  }

  cerr << "Loading average pulse profile from " << archive_filename << endl;
  Archive* archive = 0; // do as in getStokes.C to get a PolnProfile

  if (parameter_filename)
  {
    cerr << "Loading average pulse profile from " << archive_filename << endl;
    Parameters* parameters = 0; // do as in test_Predictor.C to get a Predictor
  }
  else if (predictor_filename)
  {
    // load a Predictor directly from file (same syntax as loading Parameters from file)
  }

  if (verbose)
    cerr << "Creating TimeSeries instance" << endl;
  dsp::TimeSeries voltages;


  dsp::PulsarSimulator simulator;

   // simulator.set_Predictor (predictor);
   // simulator.set_PolnProfile (profile);

  simulator.operate();

  cerr << "biyee!" << endl;
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
