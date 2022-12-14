/***************************************************************************
 *
 *   Copyright (C) 2007-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/InputBufferingShare.h"
#include "dsp/Reserve.h"

#include "ThreadContext.h"

using namespace std;

dsp::InputBuffering::Share::Share ()
{
  name = "InputBuffering::Share";
  context = 0;
  context_owner = false;

  reserve = new Reserve;
}

dsp::InputBuffering::Share::Share (InputBuffering* _buffer,
				   HasInput<TimeSeries>* _target)
{
  name = "InputBuffering::Share";

  buffer = _buffer;
  target = _target;

  context = new ThreadContext;
  context_owner = true;

  reserve = new Reserve;
}

dsp::InputBuffering::Share*
dsp::InputBuffering::Share::clone (HasInput<TimeSeries>* _target)
{
  Share* result = new Share;
  result -> buffer = buffer;
  result -> target = _target;
  result -> context = context;

  return result;
}

dsp::InputBuffering::Share::~Share ()
{
  if (context && context_owner)
    delete context;
}

//! Set the minimum number of samples that can be processed
void dsp::InputBuffering::Share::set_maximum_samples (uint64_t samples) try
{
  reserve->reserve ( target->get_input(), samples );
}
catch (Error& error)
{
  throw error += "dsp::InputBuffering::Share::set_maximum_samples";
}

/*! Copy remaining data from the target Transformation's input to buffer */
void dsp::InputBuffering::Share::set_next_start (uint64_t next) try
{
  // do nothing if the thread has no data
  if (target->get_input()->get_ndat() == 0)
    return;

  if (Operation::verbose)
    cerr << "dsp::InputBuffering::Share::set_next_start lock context="
         << context << endl;

  ThreadContext::Lock lock (context);

  if (Operation::verbose)
  {
    cerr << "dsp::InputBuffering::Share::set_next_start next=" << next << endl;
    buffer->set_cerr (cerr);
  }

  buffer->reserve = reserve;
  buffer->set_target (target);
  buffer->set_next_start (next);

  if (context)
  {
    if (Operation::verbose)
      cerr << "dsp::InputBuffering::Share::set_next_start broadcast" << endl;
    context->broadcast();
  }
}
catch (Error& error)
{
  throw error += "dsp::InputBuffering::Share::set_next_start";
}

/*! Prepend buffered data to target Transformation's input TimeSeries */
void dsp::InputBuffering::Share::pre_transformation () try
{
  if (Operation::verbose)
    cerr << "dsp::InputBuffering::Share::pre_transformation lock context="
         << context << endl;

  ThreadContext::Lock lock (context);

  int64_t want = target->get_input()->get_input_sample();

  // don't wait for data preceding the first loaded block
  if (want <= 0)
    return;

  if (Operation::verbose)
    cerr << "dsp::InputBuffering::Share::pre_transformation want=" << want << endl;

  while ( buffer->get_next_contiguous() != want )
  {
    if (buffer->get_next_contiguous() > want)
      throw Error (InvalidState,
                   "dsp::InputBuffering::Share::pre_transformation",
                   "have=%" PRIu64 " > want=%" PRIu64,
                   buffer->get_next_contiguous(), want);

    if (Operation::verbose)
      cerr << "dsp::InputBuffering::Share::pre_transformation want=" << want
	   << "; have=" << buffer->get_next_contiguous() << endl;

    context->wait();
  }

  if (Operation::verbose)
  {
    cerr << "dsp::InputBuffering::Share::pre_transformation working" << endl;
    buffer->set_cerr (cerr);
  }

  buffer->set_target (target);
  buffer->pre_transformation ();

  if (Operation::verbose)
    cerr << "dsp::InputBuffering::Share::pre_transformation exiting" << endl;
}
catch (Error& error)
{
  throw error += "dsp::InputBuffering::Share::pre_transformation";
}

/*! No action required after transformation */
void dsp::InputBuffering::Share::post_transformation ()
{
}
