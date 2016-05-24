/***************************************************************************
 *
 *   Copyright (C) 2016 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/Subint.h"

namespace dsp {

/* MTK
Specialize the transform method for PhaseLockedFilterbank, which buffers.

The issue here is that Subint blocks while waiting for an integration to
finish, and other threads need those threads to put their samples in the
buffer.  So one gets a deadlock.  The solution, here, perhaps not a good 
one, is to buffer *before* any call to unload.  Then in the calls to
PhaseLockedFilterbank::transformation, it will check to make sure that
processing hasn't "gone ahead" before it buffers.

The remaining case, where a block straddles a sub-int boundary, is handled
in the same way.  The remaining samples are buffered so execution can
proceed, and an additional call to transformation is made to process the
remaining data.
*/

template <>
void Subint<PhaseLockedFilterbank>::transformation () try
{
  if (PhaseLockedFilterbank::verbose)
    std::cerr << "Subint<PhaseLockedFilterbank>::transformation" << std::endl;

  if (divider.get_turns() == 0 && divider.get_seconds() == 0.0)
    throw Error (InvalidState, "Subint<PhaseLockedFilterbank>::tranformation",
		 "sub-integration length not specified");

  if (!built)
    prepare ();

  // flag that the input TimeSeries contains data for another sub-integration
  bool more_data = true;
  bool first_division = true;

  while (more_data)
  {
      //std::cerr << "SUBINT <PLFBQ>: entering loop at sample " << PhaseLockedFilterbank::get_input()->get_input_sample() << " with integ= " << PhaseLockedFilterbank::get_result()->get_integration_length() << std::endl;

    divider.set_bounds( PhaseLockedFilterbank::get_input() );

    if (!divider.get_fractional_pulses())
      PhaseLockedFilterbank::get_output()->set_ndat_expected( divider.get_division_ndat() );

    more_data = divider.get_in_next ();

    if (first_division && divider.get_new_division())
    {
      /* A new division has been started and there is still data in
         the current integration.  This is a sign that the current
         input comes from uncontiguous data, which can arise when
         processing in parallel. */

      //std::cerr << "SUBINT <PLFBQ>: unload partial 1 at sample " << PhaseLockedFilterbank::get_input()->get_input_sample() << " with integ= " << PhaseLockedFilterbank::get_result()->get_integration_length() << std::endl;

      if (PhaseLockedFilterbank::get_result()->get_integration_length() > 0)
      {
        PhaseLockedFilterbank::get_buffering_policy()->set_next_start (
            PhaseLockedFilterbank::get_input()->get_ndat() -
            PhaseLockedFilterbank::get_ndat_fft() + 1);
      }
      unload_partial ();
    }

    if (!divider.get_is_valid())
      continue;

    PhaseLockedFilterbank::transformation ();

    /*
    if (firstgo)
    {
      std::cerr << "getting out of dodge" << std::endl;
      return;
    }
    */


    if (!divider.get_end_reached())
      continue;

    if (first_division)
    {
      /* When the end of the first division is reached, it is not 100%
         certain that a complete sub-integration is available */

      //std::cerr << "SUBINT <PLFBQ>: unload partial 2 at sample " << PhaseLockedFilterbank::get_input()->get_input_sample() << " with integ= " << PhaseLockedFilterbank::get_result()->get_integration_length() << std::endl;

      if (PhaseLockedFilterbank::get_result()->get_integration_length() > 0)
      {
        PhaseLockedFilterbank::get_buffering_policy()->set_next_start (
            PhaseLockedFilterbank::get_input()->get_ndat() -
            PhaseLockedFilterbank::get_ndat_fft() + 1);
      }
      unload_partial ();

      divider.set_bounds( PhaseLockedFilterbank::get_input() );
      PhaseLockedFilterbank::transformation ();
      more_data = divider.get_in_next ();

      first_division = false;
      continue;
    }

    if (PhaseLockedFilterbank::verbose)
      std::cerr << "Subint<PhaseLockedFilterbank>::transformation sub-integration completed" << std::endl;

    PhaseSeries* result = PhaseLockedFilterbank::get_result ();

    complete.send (result);

    if (unloader && keep(result))
    {
      if (PhaseLockedFilterbank::verbose)
        std::cerr << "dsp::Subint<PhaseLockedFilterbank>::transformation this=" << this << " unloader=" << unloader.get() << std::endl;

      unloader->unload (result);
      zero_output ();
    }
  }
}
catch (Error& error)
{
  throw error += "dsp::Subint<PhaseLockedFilterbank>::transformation";
}

}

