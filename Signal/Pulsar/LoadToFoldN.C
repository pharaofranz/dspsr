/***************************************************************************
 *
 *   Copyright (C) 2007-2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LoadToFoldN.h"
#include "dsp/LoadToFoldConfig.h"

#include "dsp/Input.h"
#include "dsp/InputBufferingShare.h"

#include "dsp/Dedispersion.h"
#include "dsp/Fold.h"
#include "dsp/Subint.h"
#include "dsp/UnloaderShare.h"
#include "FTransformAgent.h"
#include "ThreadContext.h"

#include "dsp/CyclicFold.h"
#include "dsp/PhaseLockedFilterbank.h"

#include <fstream>
#include <stdlib.h>
#include <errno.h>

using namespace std;

//! Constructor
dsp::LoadToFoldN::LoadToFoldN (LoadToFold::Config* config)
{
  configuration = config;
  set_nthread (configuration->get_total_nthread());
}
    
//! Set the number of thread to be used
void dsp::LoadToFoldN::set_nthread (unsigned nthread)
{
  MultiThread::set_nthread (nthread);

  FTransform::nthread = nthread;

  if (configuration)
    set_configuration (configuration);
}

dsp::LoadToFold* dsp::LoadToFoldN::at (unsigned i)
{
  return dynamic_cast<LoadToFold*>( threads.at(i).get() );
}

//! Set the configuration to be used in prepare and run
void dsp::LoadToFoldN::set_configuration (LoadToFold::Config* config)
{
  configuration = config;

  MultiThread::set_configuration (config);

  for (unsigned i=0; i<threads.size(); i++)
    at(i)->set_configuration( config );
}

void dsp::LoadToFoldN::share ()
{
  MultiThread::share ();

  if (at(0)->kernel && !at(0)->kernel->context)
    at(0)->kernel->context = new ThreadContext;

  at(0)->prepare_fold ();

  if (at(0)->output_subints()) 
  {
    bool subints_ok = prepare_subint_archival <Fold> ();
    if (!subints_ok)
      subints_ok = prepare_subint_archival <CyclicFold> ();
    if (!subints_ok)
      throw Error (InvalidState, "dsp::LoadToFoldN::share",
          "folder is not a recognized Subint<> type.");
  }
}

template <class T>
bool dsp::LoadToFoldN::prepare_subint_archival ()
{
  unsigned nfold = at(0)->fold.size();

  if (Operation::verbose)
    cerr << "dsp::LoadToFoldN::prepare_subint_archival nfold=" << nfold
	 << endl;

  if( at(0)->unloader.size() != nfold )
    throw Error( InvalidState, "dsp::LoadToFoldN::prepare_subint_archival",
		 "unloader vector size=%u != fold vector size=%u",
		 at(0)->unloader.size(), nfold );

  unloader.resize( nfold );

  for (unsigned i=1; i<threads.size(); i++)
    at(i)->unloader.resize( nfold );

  /*
    Note that, at this point, only thread[0] has been prepared.
    Therefore, only thread[0] will have an initialized fold array
  */

  for (unsigned ifold = 0; ifold < nfold; ifold ++)
  {
    Subint<T>* subfold = 
      dynamic_cast< Subint<T>* >( at(0)->fold[ifold].get() );

    if (!subfold)
      return false;

    unloader[ifold] = new UnloaderShare( threads.size() );
    unloader[ifold]->copy( subfold->get_divider() );
    unloader[ifold]->set_context( new ThreadContext );

    Reference::To<PhaseSeriesUnloader> primary_unloader = at(0)->unloader[ifold];

    if (configuration->concurrent_archives())
    {
      if (Operation::verbose)
        cerr << "dsp::LoadToFoldN::prepare_subint_archival ifold=" << ifold
             << " set_wait_all (false)" << endl;
      unloader[ifold]->set_wait_all (false);
    }
    else
    {
      if (Operation::verbose)
        cerr << "dsp::LoadToFoldN::prepare_subint_archival ifold=" << ifold
             << " set_unloader ptr=" << (void*) primary_unloader.get() << endl;
      unloader[ifold]->set_unloader( primary_unloader );
    }

    for (unsigned i=0; i<threads.size(); i++) 
    {
      UnloaderShare::Submit* submit = unloader[ifold]->new_Submit (i);

      if (configuration->concurrent_archives())
      {
        PhaseSeriesUnloader* unique_unloader = primary_unloader->clone();
        if (Operation::verbose)
          cerr << "dsp::LoadToFoldN::prepare_subint_archival ifold=" << ifold
               << " clone=" << (void*) unique_unloader << endl;
        submit->set_unloader( unique_unloader );
      }
      else
      {
        submit->set_unloader( primary_unloader );
      }

      if (Operation::verbose)
	cerr << "dsp::LoadToFoldN::prepare_subint_archival submit ptr="
	     << submit << endl;

      at(i)->unloader[ifold] = submit;

      subfold = dynamic_cast< Subint<T>* >( at(i)->fold[ifold].get() );

      if (!subfold)
	return false;

      subfold->set_unloader( submit );
    }

    if (configuration->get_nfold() > 1)
    {
      if (Operation::verbose)
        cerr << "dsp::LoadToFoldN::prepare_subint_archival"
                " nfold=" << configuration->get_nfold() <<
                " set_wait_all (false)" << endl;

      unloader[ifold]->set_wait_all (false);
    }
  }

  if (Operation::verbose)
    cerr << "dsp::LoadToFoldN::prepare_subint_archival done" << endl;

  return true;
}

void dsp::LoadToFoldN::finish ()
{
  if (Operation::verbose)
    cerr << "dsp::LoadToFoldN::finish this=" << (void*) this << endl;

  MultiThread::finish ();

  for (unsigned i=0; i<unloader.size(); i++)
  {
    if (Operation::verbose)
      cerr << "dsp::LoadToFoldN::finish unloader[" << i << "]" << endl;

    unloader[i]->finish();
  }
}

//! The creator of new LoadToFold1 threadss
dsp::LoadToFold* dsp::LoadToFoldN::new_thread ()
{
  return new LoadToFold;
}

