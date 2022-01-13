/***************************************************************************
 *
 *   Copyright (C) 2011-2021 by Paul Demorest and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/VDIFFile.h"
#include "dsp/ASCIIObservation.h"
#include "vdifio.h"
#include "Error.h"

#include "coord.h"
#include "strutil.h"	
#include "ascii_header.h"

#include <iomanip>

#include <time.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

using namespace std;

static const int uninitialised_frame_number = -2;

dsp::VDIFFile::VDIFFile (const char* filename,const char* headername)
  : BlockFile ("VDIF")
{
  stream = 0;
  datafile[0] = '\0';
  current_frame_number = uninitialised_frame_number;
}

dsp::VDIFFile::~VDIFFile ( )
{
}

bool dsp::VDIFFile::is_valid (const char* filename) const
{
  if (verbose)
    cerr << "dsp::VDIFFile::is_valid filename=" << filename << endl;

  // Open the header file, check for INSTRUMENT=VDIF
  // TODO use a different keyword?
  FILE *fptr = fopen(filename, "r");
  if (!fptr) 
  {
    if (verbose)
      cerr << "dsp::VDIFFile::is_valid Error opening filename=" << filename << endl;
    return false;
  }

  char header[4096];
  fread(header, sizeof(char), 4096, fptr);
  fclose(fptr);

  char inst[64];
  if ( ascii_header_get(header, "INSTRUMENT", "%s", inst) < 0 )
  {
    if (verbose)
      cerr << "dsp::VDIFFile::is_valid no INSTRUMENT line" << endl;
    return false;
  }
  if ( std::string(inst) != "VDIF" )
  {
    if (verbose)
      cerr << "dsp::VDIFFile::is_valid INSTRUMENT != 'VDIF'" << endl;
    return false;
  }

  // TODO check for DATAFILE line?

  // Old code below.  Could use to also test datafile.
#if 0 

  // Read one header
  char rawhdr[VDIF_HEADER_BYTES];
  size_t rv = fread(rawhdr, sizeof(char), VDIF_HEADER_BYTES, fptr);
  fclose(fptr);
  if (rv != VDIF_HEADER_BYTES) {
      if (verbose) 
          cerr << "VDIFFile: Error reading header." << endl;
    return false;
  }

  // See if some basic values make sense
  int nbytes = getVDIFFrameBytes(rawhdr);
  if (nbytes<0 || nbytes>MAX_VDIF_FRAME_BYTES) {
      if (verbose) 
          cerr << "VDIFFFile: Frame bytes = " << nbytes << endl;
      return false;
  }

  int mjd = getVDIFFrameMJD(rawhdr);
  if (mjd<30000 || mjd>70000) {
      if (verbose) 
          cerr << "VDIFFFile: MJD = " << mjd << endl;
      return false;
  }

  int nchan = getVDIFNumChannels(rawhdr);
  if (nchan<0 || nchan>nbytes*8) {
      if (verbose) 
          cerr << "VDIFFFile: nchan = " << nchan << endl;
      return false;
  }
#endif
	
  // Everything looks ok
  return true;
}

void dsp::VDIFFile::open_file (const char* filename)
{	

  // This is the header file
  FILE *fptr = fopen (filename, "r");
  if (!fptr)
    throw Error (FailedSys, "dsp::VDIFFile::open_file",
        "fopen(%s) failed", filename);

  // Read the header
  char header[4096];
  fread(header, sizeof(char), 4096, fptr);
  fclose(fptr);

  // Get the data file
  if (ascii_header_get (header, "DATAFILE", "%s", datafile) < 0)
  {
    strncpy (datafile, filename, VDIF_MAX_FILENAME_LENGTH);
    char* ext = strstr (datafile, ".hdr");
    if (!ext)
      throw Error (InvalidParam, "dsp::VDIFFile::open_file",
                   "no DATAFILE and no .hdr to strip");
    *ext = '\0';
  }

  // Parse the standard ASCII info.  Timestamps are in VDIF packets
  // so not required.  Also we'll assume VDIF's "nchan" really gives
  // the number of polns for now, and NCHAN is 1.  NBIT is in VDIF packets.
  // We'll compute TSAMP from the bandwidth.  NDIM (real vs complex sampling)
  // is in VDIF packets via the iscomplex param.
  ASCIIObservation* info_tmp = new ASCIIObservation;
  info = info_tmp;

  info_tmp->set_required("UTC_START", false);
  info_tmp->set_required("OBS_OFFSET", false);
  info_tmp->set_required("NBIT", false);
  info_tmp->set_required("NDIM", false);
  info_tmp->set_required("TSAMP", false);
  info_tmp->load(header);

  fd = ::open(datafile, O_RDONLY);
  if (fd<0) 
    throw Error (FailedSys, "dsp::VDIFFile::open_file",
              "open(%s) failed", datafile);
	
  // Read until we get a valid frame
  bool got_valid_frame = false;
  char rawhdr_bytes[VDIF_HEADER_BYTES];
  vdif_header *rawhdr = (vdif_header *)rawhdr_bytes;
  int nbyte, legacymode;
  while (!got_valid_frame) 
  {
    size_t rv = read(fd, rawhdr_bytes, VDIF_HEADER_BYTES);
    if (rv != VDIF_HEADER_BYTES) 
        throw Error (FailedSys, "VDIFFile::open_file",
                "Error reading first header");

    // Get frame size
    nbyte = getVDIFFrameBytes(rawhdr);
    if (verbose) cerr << "VDIFFile::open_file FrameBytes = " << nbyte << endl;
    // Get the legacy mode
    legacymode = getVDIFFrameLegacyMode(rawhdr);
    if (verbose) cerr << "VDIFFile::open_file LegacyMode = " << legacymode << endl;

    header_bytes = 0;
    block_bytes = nbyte;
    block_header_bytes = (legacymode) ? VDIF_LEGACY_HEADER_BYTES : VDIF_HEADER_BYTES;

    // If this first frame is invalid, go to the next one
    if (getVDIFFrameInvalid(rawhdr)==0)
      got_valid_frame = true;
    else
    {
      rv = lseek(fd, nbyte-VDIF_HEADER_BYTES, SEEK_CUR);
      if (rv<0) 
        throw Error (FailedSys, "VDIFFile::lseek",
            "Error seeking to next VDIF frame");
    }
  }

  // Rewind file
  lseek(fd, 0, SEEK_SET);

  // Get basic params

  int nbit = getVDIFBitsPerSample(rawhdr);
  if (verbose) cerr << "VDIFFile::open_file NBIT = " << nbit << endl;
  get_info()->set_nbit (nbit);

  bool iscomplex = rawhdr->iscomplex;
  if (iscomplex) 
  {
    get_info()->set_ndim(2);
    get_info()->set_state(Signal::Analytic);
  }
  else
  {
    get_info()->set_ndim(1);
    get_info()->set_state(Signal::Nyquist);
  }
  if (verbose) cerr << "VDIFFile::open_file iscomplex = " << iscomplex << endl;

  // Each poln shows up as a different channel but this 
  // could also be different freq channels...
  int vdif_nchan = getVDIFNumChannels(rawhdr);
  if (verbose) cerr << "VDIFFile::open_file NCHAN = " << vdif_nchan << endl;

  int expected_nchan = get_info()->get_npol() * get_info()->get_nchan();
  if (vdif_nchan != expected_nchan)
    throw Error (InvalidState, "dsp::VDIFFile::open_file",
                 "vdif_nchan=%d != expected=%d (npol=%d * nchan=%d)",
                 vdif_nchan, expected_nchan, get_info()->get_npol(), get_info()->get_nchan());

  get_info()->set_rate( fabs((double) get_info()->get_bandwidth()) * 1e6 
      / (double) get_info()->get_nchan() 
      * (get_info()->get_state() == Signal::Nyquist ? 2.0 : 1.0));
  if (verbose) cerr << "VDIFFile::open_file rate = " << get_info()->get_rate() << endl;

  // Figure frames per sec from bw, pkt size, etc
  int frame_data_size = nbyte - block_header_bytes;
  int bits_per_byte = 8;
  int bits_per_frame = bits_per_byte * frame_data_size;
  int bits_per_sample
    = get_info()->get_nbit()
    * get_info()->get_nchan()
    * get_info()->get_npol();

  int samples_per_second = get_info()->get_rate();
    
  frames_per_second = bits_per_sample * samples_per_second / bits_per_frame; 
 
  if (verbose)
    cerr << "VDIFFile::open_file frame_data_size=" 
	 << frame_data_size << " frames_per_sec=" 
	 << frames_per_second << endl;

  // Set load resolution equal to one frame? XXX
  // This broke file unloading somehow ... wtf..
  //resolution = info.get_nsamples(frame_data_size);

  int mjd = getVDIFFrameMJD(rawhdr);
  int sec = getVDIFFrameSecond(rawhdr);
  int fn = getVDIFFrameNumber(rawhdr);
  if (verbose) cerr << "VDIFFile::open_file MJD = " << mjd << endl;
  if (verbose) cerr << "VDIFFile::open_file sec = " << sec << endl;
  if (verbose) cerr << "VDIFFile::open_file fn  = " << fn << endl;
  get_info()->set_start_time( MJD(mjd,sec,(double)fn/frames_per_second) );

  // Figures out how much data is in file based on header sizes, etc.
  set_total_samples();

  current_frame_number = uninitialised_frame_number;
}

void dsp::VDIFFile::reopen ()
{
  if (fd > 0)
    throw Error (InvalidState, "dsp::VDIFFile::reopen", "file already open");

  fd = ::open(datafile, O_RDONLY);
  if (fd<0)
    throw Error (FailedSys, "dsp::VDIFFile::open_file",
              "open(%s) failed", datafile);

}

//! On the first seek, set the index of the first packet
int64_t dsp::VDIFFile::seek_bytes (uint64_t bytes)
{
  return BlockFile::seek_bytes (bytes);
}

//! Check that next packet follows the packet that was just read
/*! Return the number of missing packets */
uint64_t dsp::VDIFFile::skip_extra ()
{
  /*
    WvS: Instead of fread from fptr, use read from int fd
    as in BlockFile::load_bytes

  // Read one header
  char rawhdr[VDIF_HEADER_BYTES];
  size_t rv = fread(rawhdr, sizeof(char), VDIF_HEADER_BYTES, fptr);

  */

  char rawhdr[VDIF_HEADER_BYTES];
  ssize_t bytes_read = read (fd, rawhdr, VDIF_HEADER_BYTES);

  vdif_header* hdr = static_cast<vdif_header*> ( (void*) rawhdr );

  if (bytes_read < 0)
    perror ("dsp::VDIFFile::skip_extra read error");

  bool valid = ! getVDIFFrameInvalid (hdr);

  if (!valid)
    cerr << "dsp::VDIFFile::skip_extra getVDIFFrameInvalid returns false"
	 << endl;

  int next_frame_number = getVDIFFrameNumber (hdr);

  uint64_t skipped_packets = 0;
  
  if (current_frame_number != uninitialised_frame_number)
  {
    if (current_frame_number == next_frame_number)
    {
      if (verbose)
	cerr << "dsp::VDIFFile::skip_extra end of data" << endl;
      return 0;
    }
    
    if (current_frame_number + 1 == frames_per_second)
    {
      // expect next_frame_number == 0
      current_frame_number = -1;
    }
    
    skipped_packets = next_frame_number - current_frame_number - 1;

    if (skipped_packets)
      cerr << "dsp::VDIFFile::skip_extra"
	" next_frame_number=" << next_frame_number <<
	" current_frame_number=" << current_frame_number <<
	" skipped_packets=" << skipped_packets << endl;
  }

  current_frame_number = next_frame_number;
      
  return skipped_packets;
}

bool dsp::VDIFFile::contiguous (const File* that) const
{
  const VDIFFile* like = dynamic_cast<const VDIFFile*> (that);

  if (like == NULL)
    return false;

  /* this is where you could compare the last packet of this with
     the first packet of like; i.e. if

     this = pointer to VDIFFile A
     like = pointer to VDIFFile B

     ensure that A and B are contiguous.

     However, because the VDIFF unpackers deal with dropped packets,
     and assuming that files do not contain overlapping content,
     contiguity requires only (strict weak) ordering in time.
  */

  return this->get_info()->get_start_time() < that->get_info()->get_start_time();
}
