/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/CPSRFile.h"
#include "pspmDbase.h"
#include "pspm++.h"

#include "tempo++.h"
#include "dirutil.h"
#include "Error.h"

// CPSR header and unpacking routines
#define cpsr 1
#include "pspm_search_header.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

using namespace std;

//! Construct and open file
dsp::CPSRFile::CPSRFile (const char* filename)
  : File ("CPSR")
{ 
 tapenum = filenum = -1;
  if (filename)
    open (filename);
}

//! Virtual destructor
dsp::CPSRFile::~CPSRFile(){ }

bool dsp::CPSRFile::is_valid (const char* filename) const
{
  int fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    return false;

  PSPM_SEARCH_HEADER* header = pspm_read (fd);
  ::close (fd);

  if (!header || !PSPMverify(header, verbose))
    return false;

  return true;
}


// #define _DEBUG 1

//
// NEW! 1 Aug 01
// It has been determined that PSPM headers are coming off DLT tape
// corrupted.   Luckily, a copy of most headers can be found on disk
// on orion.  An ascii database is created from these files, and
// served by the pspmDbase::server and pspmDbase::entry clases

static pspmDbase::server cpsr_hdr;

/*!
  Opens a CPSR (PSPM) data file and prepares the attributes of the various
  base classes, including:  <UL>
  <LI> Input::info
  <LI> File::header_bytes
  <LI> File::fd
  </UL>
*/

void dsp::CPSRFile::open_file (const char* filename)
{
  if (verbose)
    cerr << "dsp::CPSRFile::open " << filename << endl;

  if ( sizeof(PSPM_SEARCH_HEADER) != PSPM_HEADER_SIZE )
   {
    cerr << "dsp::CPSRFile:: PSPM header size is invalid.\n"
            "dsp::CPSRFile:: PSPM header size " << PSPM_HEADER_SIZE << ".\n"
            "dsp::CPSRFile:: for this architecture: " 
         << sizeof(PSPM_SEARCH_HEADER) << endl;
    
    throw Error (InvalidState, "dsp::CPSRFile::open", 
		 "Architecture Error: Invalid PSPM header size");
  }

  header_bytes = sizeof(PSPM_SEARCH_HEADER);

  if (verbose)
    cerr << "dsp::CPSRFile::open_file header_bytes=" << header_bytes << endl;

  fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::CPSRFile::open", 
		 "failed open(%s)", filename);

  PSPM_SEARCH_HEADER* header = pspm_read (fd);
  if (!header) {
    ::close (fd);
    throw Error (FailedCall, "dsp::CPSRFile::open",
		 "failed pspm_read(%s) %s\n", filename, strerror(errno));
  }

  pspmDbase::entry hdr;
  try { hdr = pspmDbase::Entry (header); }
  catch (...) {
    cerr << "dsp::CPSRFile::open WARNING no match found in pspmDbase for "
	 << PSPMidentifier (header) << endl;

    hdr.create (header);
  }
    
  tapenum = hdr.tape;
  filenum = hdr.file;
  
#if _DEBUG

  fprintf (stderr, "File size:       %ld\n", header->file_size);
  fprintf (stderr, "Large File Size: "I64"\n", header->ll_file_size);
  fprintf (stderr, "Large Offset:    "I64"\n", header->ll_file_offset);
  fprintf (stderr, "MJD in header    %40.38Lf\n", header->mjd_start);
  
  fprintf (stderr, "tick offset:     %30.28lf\n\n", header->tick_offset);
  fprintf (stderr, "tape_num:        %d\n", header->tape_num);
  fprintf (stderr, "tape_file_number:%d\n", header->tape_file_number);
  fprintf (stderr, "scan_num:        %d\n", header->scan_num);
  fprintf (stderr, "scan_file_number %d\n", header->scan_file_number);
  fprintf (stderr, "file_size        %d\n\n", header->file_size);

  fprintf (stderr, "LMST in header   %lf\n", header->pasmon_lmst);
  
  fprintf (stderr, "Pulsar:          %s\n",  header->psr_name);
  fprintf (stderr, "Date:            %s\n",  header->date);
  fprintf (stderr, "Start Time:      %s\n",  header->start_time);
  
  fprintf (stderr, "pasmon_daynumber:%d\n",  header->pasmon_daynumber);
  fprintf (stderr, "pasmon_ast:      %d\n",  header->pasmon_ast);
  
  fprintf (stderr, "Centre Freq:     %lf\n", header->rf_freq);
  fprintf (stderr, "Sampling Period: %lf\n", header->samp_rate);
  fprintf (stderr, "Bandwidth:       %lf\n", header->bw);
  fprintf (stderr, "SIDEBAND:        %d:",  header->SIDEBAND);
  switch (header->SIDEBAND) 
    {
    case UNKNOWN_SIDEBAND:
	fprintf (stderr, "Unknown (assume USB)\n");
	break;
      case SSB_LOWER:
	fprintf (stderr, "SSB Lower\n");
	break;
      case SSB_UPPER:
	fprintf (stderr, "SSB Upper\n");
	break;
      case DSB_SKYFREQ:
	fprintf (stderr, "DSB Sky frequency order\n");
	break;
      case DSB_REVERSED:
	fprintf (stderr, "DSB Sky reversed frequency order\n");
	break;
      default:
	fprintf (stderr, "Internal error\n");
	break;
      }
    
  fprintf (stderr, "Telescope:       %d\n",  header->observatory);
  fprintf (stderr, "Channels:        %ld\n", header->num_chans);
  fprintf (stderr, "Bit Mode:        %ld\n", header->bit_mode);
    
  fprintf (stderr, "dsp::CPSRFile::open - header size %d\n", header_bytes);
    
  fflush (stderr);
#endif
    
  char modestr[30];
  sprintf (modestr, "%d", hdr.nbit);
  get_info()->set_mode (modestr);
  
  get_info()->set_start_time (hdr.start);

  /* redwards --- set "position" from the header.. should replace cal_ra/dec*/
  double ra_deg, dec_deg;
  ra_deg = 15.0 * floor(header->user_ra/10000.0)
    + 15.0/60.0 * floor(fmod(header->user_ra,10000.0)/100.0)
    + 15.0/3600.0 * fmod(header->user_ra,100.0);
  if (header->user_dec == 0.0)
    dec_deg = 0.0;
  else
    dec_deg = header->user_dec/fabs(header->user_dec)* // sign
	( floor(fabs(header->user_dec)/10000.0)         // magnitude
	  +1.0/60.0*floor(fmod(fabs(header->user_dec),10000.0)/100.0)
	  +1.0/3600.0*fmod(fabs(header->user_dec),100.0));
    
  get_info()->set_coordinates (sky_coord(ra_deg, dec_deg));
    
  get_info()->set_source (hdr.name);
    
  /* IMPORTANT: tsamp is the sampling period in microseconds */
  get_info()->set_rate (1e6/hdr.tsamp);
  get_info()->set_bandwidth (hdr.bandwidth);
    
  // CBR linefeed data has incorrect bandwidth of 8 MHz
  if( fabs(get_info()->get_bandwidth()) < 9.9 ){
    get_info()->set_bandwidth( 10.0 );
    get_info()->set_rate( 10.0e6 );
    fprintf(stderr,"Detected CBR file with incorrect bandwidth of file '%s' of '%f'- set rate to %f and bw to %f\n",
	    filename, hdr.bandwidth, get_info()->get_rate(), get_info()->get_bandwidth());
  }

  // IMPORTANT: both telescope and centre_freq should be set before calling
  // default_basis
  get_info()->set_telescope( "parkes" ); // string(1, hdr.ttelid) );
  get_info()->set_centre_frequency (hdr.frequency);
    
  // CPSR samples the analytic representation of the voltages
  get_info()->set_state (Signal::Analytic);
  // number of channels = number of polarizations * ((I and Q)=2);
  get_info()->set_npol (hdr.ndigchan / 2);
  get_info()->set_nbit (hdr.nbit);
   
  get_info()->set_ndat (hdr.ndat);

  uint64_t fsize = filesize(filename);

  if( fsize-header_bytes != get_info()->get_nbytes() ){
    get_info()->set_ndat( get_info()->get_nsamples(fsize-header_bytes) );
    fprintf(stderr,"WARNING: dsp::CPSRFile::open_file(): Your ndat in header (" UI64 ") doesn't match file size (" UI64 " bytes) (The number of samples has been reduced to " UI64 " because the header size is " UI64 ")\n",
	    uint64_t(hdr.ndat),uint64_t(fsize),get_info()->get_ndat(),
	    uint64_t(header_bytes));
  }

  //throw Error(InvalidState,"dsp::CPSRFile::open_file()",
  //      "ndat in header ("UI64") doesn't match file size ("UI64")\n",
  //      uint64_t(hdr.ndat),uint64_t(fsize));

  fprintf(stderr,"Have set CPSR ndat to be " UI64 "\n",uint64_t(hdr.ndat));
  
  if (hdr.ndat < 1) {
    ::close (fd);
    throw Error (InvalidState, "dsp::CPSRFile::open",
		   "Total data: %d\n", hdr.ndat);
  }

  get_info()->set_machine ("CPSR");

  if (verbose)
    cerr << "dsp::CPSRFile::open exit" << endl;
}
  











