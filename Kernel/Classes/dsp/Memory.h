//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Memory_h_
#define __dsp_Memory_h_

#include "Reference.h"
#include <inttypes.h>

namespace dsp {

  //! Manages memory allocation and destruction
  class Memory : public Reference::Able
  {
  protected:
    static Memory* manager;

  public:
    static void* allocate (size_t nbytes);
    static void free (void*);
    static void set_manager (Memory*);
    static Memory* get_manager ();

    virtual void* do_allocate (size_t nbytes);
    virtual void  do_free (void*);
    virtual void  do_zero (void* ptr, size_t nbytes);
    virtual void  do_copy (void* to, const void* from, size_t bytes);
    virtual void  cross_copy (void* to, Memory* memto, 
        const void* from, const Memory* memfrom, size_t bytes);
    virtual bool  on_host () const { return true; }
    virtual int get_device () const { return -1; }
  };

}

#endif
