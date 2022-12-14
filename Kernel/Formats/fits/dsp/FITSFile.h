//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Jonathan Khoo
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FITSFile_h
#define __FITSFile_h

#include "fitsio.h"

#include "Pulsar/Archive.h"

#include "dsp/File.h"
#include "dsp/BitSeries.h"

namespace dsp
{

  class FITSUnpacker;

  //! Loads BitSeries data from a PSRFITS data file
  class FITSFile : public File
  {
    public:
      //! Construct and open file
      FITSFile(const char* filename = 0);

      //! Returns true if filename appears to name a valid FITS file
      bool is_valid(const char* filename) const;

      //! Close the file
      void close ();

      //! Reopen the file
      void reopen ();

      void add_extensions (Extensions*);
      
      unsigned get_samples_in_row() const { return samples_in_row; }

      unsigned get_bytes_per_row() { return bytes_per_row; }

      class Extension;

    protected:
      friend class FITSUnpacker;

      //! Open the file
      virtual void open_file(const char* filename);

      //! Load nbyte bytes of sampled data from the device into buffer.
      virtual int64_t load_bytes(unsigned char* buffer, uint64_t bytes);

      //! Set the current byte offset in the file
      int64_t seek_bytes (uint64_t bytes);

      void set_samples_in_row (unsigned n) { samples_in_row = n; }

      void set_bytes_per_row(const unsigned bytes) { bytes_per_row = bytes; }

      void set_data_colnum(const int colnum) { data_colnum = colnum; }

      int get_data_colnum() const { return data_colnum; }

      Reference::To<Pulsar::Archive> archive;

      //! Column number of the DATA column in the SUBINT table.
      int data_colnum;

      //! Column number of the DAT_SCL column in the SUBINT table.
      int scl_colnum;

      //! Column number of the DAT_OFFS column in the SUBINT table.
      int offs_colnum;

      //! Store the instance of fitsfile, so it is only opened once.
      fitsfile* fp;

      //! Number of samples per row
      unsigned samples_in_row;

      //! Number of bytes per row in the SUBINT table.
      unsigned bytes_per_row;

      //! Current byte offset in the file
      uint64_t current_byte;

      //! The last row of data read
      unsigned current_row;

      //! Store reference spectrum
      std::vector<float> dat_scl;

      //! Store reference offset
      std::vector<float> dat_offs;

      //! Number of rows in the SUBINT table
      unsigned number_of_rows;
      void set_number_of_rows (unsigned N) { number_of_rows = N; }
      unsigned get_number_of_rows () const { return number_of_rows; }

      //! Using signed ingegers?
      int signint;

      //! Offset to conver unsigned integers to signed integers
      float zero_off;
  };

  class FITSFile::Extension : public BitSeries::Extension
  {
    public:

      class Row
      {
      public:

        //! Scale factors
        std::vector<float> dat_scl;

        //! Offsets
        std::vector<float> dat_offs;

        //! Number of samples to unpack using these values
        uint64_t nsamp;
      };

      //! Default contructor
      Extension () { zero_off = 0.0; }

      //! Offset to convert unsigned integers to signed integers
      float zero_off;

      //! Scales and offsets relevant to each row
      std::vector<Row> rows;
  };

}

#endif

