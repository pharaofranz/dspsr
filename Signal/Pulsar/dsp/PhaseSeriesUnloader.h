
/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/PhaseSeriesUnloader.h,v $
   $Revision: 1.11 $
   $Date: 2006/08/04 00:08:10 $
   $Author: straten $ */

#ifndef __PhaseSeriesUnloader_h
#define __PhaseSeriesUnloader_h

#include <string>

#include "Reference.h"

namespace dsp {

  class PhaseSeries;

  //! Base class for things that can unload PhaseSeries data somewhere

  class PhaseSeriesUnloader : public Reference::Able {

  public:
    
    //! Constructor
    PhaseSeriesUnloader ();
    
    //! Destructor
    virtual ~PhaseSeriesUnloader ();
    
    //! Set the PhaseSeries from which Profile data will be constructed
    void set_profiles (const PhaseSeries* profiles);

    //! Defined by derived classes
    virtual void unload () = 0;

    //! Creates a good filename for the PhaseSeries data archive
    virtual std::string get_filename (const PhaseSeries* data) const;

    //! Set the filename (pattern) to be used by get_filename
    virtual void set_filename (const char* filename);
    void set_filename (const std::string& filename)
    { set_filename (filename.c_str()); }

    //! Set the extension to be used by get_filename
    virtual void set_extension (const char* extension);
    void set_extension (const std::string& extension)
    { set_extension (extension.c_str()); }

    //! allow the archive filename to be over-ridden by a pulse number
    void set_force_filename (bool _force_filename)
    { force_filename = _force_filename; }

    bool get_force_filename () const { return force_filename; }

    //! place output files in a sub-directory named by source
    void set_source_filename (bool _source_filename)
    { source_filename = _source_filename; }

    bool get_source_filename(){ return source_filename; }

  protected:

    //! Helper function that makes sure a given filename is unique
    std::string make_unique (const std::string& filename,
			     const std::string& fname_extension,
			     const PhaseSeries* data) const;

    //! PhaseSeries from which Profile data will be constructed
    Reference::To<const PhaseSeries> profiles;

    //! The filename pattern
    std::string filename_pattern;

    //! The filename extension
    std::string filename_extension;

    //! The filename path
    std::string filename_path;

    //! Put each output file in a sub-directory named by source
    bool source_filename;

    //! Force make_unique() to return 'filename' [false]
    bool force_filename;

  };

}

#endif // !defined(__PhaseSeriesUnloader_h)
