//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Signal/Pulsar/dsp/SubFold.h,v $
   $Revision: 1.5 $
   $Date: 2005/07/29 17:39:11 $
   $Author: wvanstra $ */

#ifndef __SubFold_h
#define __SubFold_h

#include "dsp/Fold.h"

#include "Phase.h"
#include "Callback.h"

namespace dsp {

  //! Fold data into sub-integrations
  /*! The SubFold class is useful for producing multiple sub-integrations
    from a single observation.  Given a polyco and the number of pulses
    to integrate, this class can be used to produce single pulse profiles.

    If no PhaseSeriesUnloader is set (see SubFold::set_unloader),
    the SubFold class will store folded sub-integrations in the subints 
    vector.  If set, the SubFold class will call unloader->unload (output)
  */

  class PhaseSeriesUnloader;

  class SubFold : public Fold {

  public:
    
    //! Constructor
    SubFold ();
    
    //! Destructor
    ~SubFold ();
    
    /** @name PhaseSeries events
     *  The attached callback methods should be of the form:
     *
     *  void method (const PhaseSeries& data);
     */
    //@{

    //! Attach methods to receive completed PhaseSeries instances
    Callback<PhaseSeries> complete;

    //! Attach methods to receive partially completed PhaseSeries instances
    Callback<PhaseSeries> partial;

    //@}

    //! Set the start time from which to begin counting sub-integrations
    void set_start_time (MJD start_time);

    //! Get the start time from which to begin counting sub-integrations
    MJD get_start_time () const { return start_time; }

    //! Set the number of seconds to fold into each sub-integration
    void set_subint_seconds (double subint_seconds);

    //! Get the number of seconds to fold into each sub-integration
    double get_subint_seconds () const { return subint_seconds; }

    //! Set the number of turns to fold into each sub-integration
    void set_subint_turns (unsigned subint_turns);

    //! Get the number of turns to fold into each sub-integration
    unsigned get_subint_turns () const { return subint_turns; }

    //! Prepare to fold with the current attributes
    void prepare ();

    /** @name deprecated methods 
     *  Use of these methods is deprecated in favour of attaching
     *  callback methods to the completed event. */
    //@{

    //! Set the file unloader
    void set_unloader (PhaseSeriesUnloader* unloader);

    //! Get the file unloader
    PhaseSeriesUnloader* get_unloader () const;

    //! Decide wether or not to keep the folded profile
    virtual bool keep (PhaseSeries* data) { return true; }

    //@}

  protected:

    //! Folds the TimeSeries data into one or more sub-integrations
    virtual void transformation ();

    //! Disable Fold class from setting the idat_start and ndat_fold attributes
    virtual void set_limits (const Observation* input);

    //! If no unloader is set, sub-integrations are stored here
    vector< Reference::To<PhaseSeries> > subints;

    //! File unloading flag
    Reference::To<PhaseSeriesUnloader> unloader;
    
    //! The start time from which to begin counting sub-integrations
    MJD start_time;

    //! Interval over which to fold each sub-integration (in seconds)
    double subint_seconds;

    //! Number of turns to fold into each sub-integration
    unsigned subint_turns;

    //! Calculates the boundaries within which to fold the input TimeSeries
    bool bound (bool& more_data, bool& subint_full);

    //! Calculates the boundaries of the sub-integration containing time
    void set_boundaries (const MJD& time);

  private:

    //! The start of the current sub-integration
    MJD lower;

    //! The end of the current sub-integration
    MJD upper;

    //! The phase at which folding starts
    Phase start_phase;

  };

}

#endif // !defined(__SubFold_h)
