//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/cpsr2/dsp/CPSR2File.h,v $
   $Revision: 1.5 $
   $Date: 2002/11/06 06:30:41 $
   $Author: hknight $ */


#ifndef __CPSR2File_h
#define __CPSR2File_h

#include "dsp/File.h"

namespace dsp {

  //! Loads Timeseries data from a CPSR2 data file
  class CPSR2File : public File 
  {
  public:
   
    //! Returns true if filename appears to name a valid CPSR2 file
    bool is_valid (const char* filename) const;
    
    static int get_header (char* cpsr2_header, const char* filename);

    //! Construct and open file
    CPSR2File (const char* filename=0) { if (filename) open (filename); }

  protected:

    //! Open the file
    virtual void open_it (const char* filename);

    // set the number of bytes in header attribute- called by open_it() and by dsp::ManyFile::switch_to_file()
    virtual void set_header_bytes();

  };

}

#endif // !defined(__CPSR2File_h)
  
