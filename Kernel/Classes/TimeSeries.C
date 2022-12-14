
 /***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "config.h"
#include "dsp/TimeSeries.h"
#include "dsp/Memory.h"

#ifdef HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/TimeSeriesCUDA.h"
#endif

#include "fsleep.h"
#include "Error.h"

#include <algorithm>
#include <memory>

#include <stdio.h>
#include <assert.h>
#include <string.h>

using namespace std;

bool dsp::TimeSeries::auto_delete = true;

dsp::TimeSeries::TimeSeries () : DataSeries()
{
  init ();
}

dsp::TimeSeries::TimeSeries (const TimeSeries& ts) : DataSeries()
{
  init ();
  operator=(ts);
}

void dsp::TimeSeries::init ()
{
  order = OrderFPT;

  DataSeries::set_nbit( 8 * sizeof(float) );
  data = 0;

  reserve_ndat = 0;
  reserve_nfloat = 0;
  input_sample = -1;
  zeroed_data = false;
  engine = 0;
}

dsp::TimeSeries* dsp::TimeSeries::clone () const
{
  return new TimeSeries (*this);
}

dsp::TimeSeries* dsp::TimeSeries::null_clone () const
{
  if (verbose)
    cerr << "dsp::TimeSeries::null_clone" << endl;

  TimeSeries* result = new TimeSeries;
  result->null_work (this);
  return result;
}

void dsp::TimeSeries::null_work (const TimeSeries* from)
{
  order = from->order;

#ifdef HAVE_CUDA
  memory = from->memory;
  if (from->engine)
  {
    set_engine (new CUDA::TimeSeriesEngine(memory));
  }
#else
  memory = from->memory;
#endif
}

dsp::TimeSeries::~TimeSeries()
{
}

    //! Get the order
dsp::TimeSeries::Order dsp::TimeSeries::get_order () const
{
  return order;
}

//! Set the order
void dsp::TimeSeries::set_order (Order o)
{
  order = o;
}

void dsp::TimeSeries::set_nbit (unsigned _nbit)
{
  if (verbose)
    cerr << "dsp::TimeSeries::set_nbit (" << _nbit << ") ignored" << endl;
}

dsp::TimeSeries&
dsp::TimeSeries::use_data(float* _buffer, uint64_t _ndat)
{
  if( get_nchan() != 1 || get_npol() != 1 )
    throw Error(InvalidState,"dsp::TimeSeries::use_data()",
		"This function is only for nchan=1 npol=1 TimeSeries --- you had %d %d",
		get_nchan(), get_npol());

  if( !_buffer )
    throw Error(InvalidState,"dsp::TimeSeries::use_data()",
		"Input data was null!");

  resize( 0 );
  buffer = (unsigned char*)_buffer;
  data = _buffer;
  size = sizeof(float) * _ndat;
  subsize = sizeof(float) * _ndat;
  set_ndat( _ndat );

  return *this;
}

//! Allocate the space required to store nsamples time samples.
/*!
  \pre The dimensions of each time sample (nchan, npol, ndim) should
  have been set prior to calling this method.
  \post If: <UL>
  <LI> nsamples == 0, the data buffer is completely de-allocated </LI>
  <LI> nsamples < previous resize(nsamples), the data buffer and its data
  is not modified.  Only the interpretation of the size of each data block
  is changed.
  <LI> nsamples > previous resize(nsamples), the data buffer may be deleted
  and the current data fill be lost

  If the reshape flag is set, the reserve_ndat is ignored.
  </UL>
*/
void dsp::TimeSeries::resize (uint64_t nsamples)
{
  if (verbose)
  {
    cerr << "dsp::TimeSeries::resize (" << nsamples << ") data=" << data
	 << " buffer=" << (void*)buffer << " ndat=" << get_ndat() << endl;

    if (data && buffer)
      cerr << "dsp::TimeSeries::resize (" << nsamples << ") offset="
	   << int64_t((data-(float*)buffer)) << endl;
  }

  uint64_t reserve_step = get_ndim();
  if (order == OrderTFP)
    reserve_step *= get_nchan() * get_npol();

  uint64_t fake_ndat = reserve_nfloat / reserve_step;
  if (reserve_nfloat % reserve_step)
    fake_ndat ++;

  if (verbose)
    cerr << "dsp::TimeSeries::resize reserve_ndat="
	 << reserve_ndat << " fake_ndat=" << fake_ndat << endl;

  // change in the required number of samples
  int64_t ndat_change = int64_t(nsamples+fake_ndat) - int64_t(get_ndat_allocated());

  if (nsamples || auto_delete)
  {
    // only resize the data series when additional space is required
    if (ndat_change > 0 || auto_delete || reshape_required())
      DataSeries::resize (nsamples+fake_ndat);
  }

  // offset the data pointer and reset the number of samples
  data = (float*)buffer + reserve_nfloat;

  set_ndat( nsamples );

  return;
}

void dsp::TimeSeries::decrease_ndat (uint64_t new_ndat)
{
  if (new_ndat > get_ndat())
    throw Error (InvalidParam, "dsp::TimeSeries::decrease_ndat",
                 "new ndat=" UI64 " > old ndat=" UI64, new_ndat, get_ndat());

  if (verbose)
    cerr << "dsp::TimeSeries::decrease_ndat from " << get_ndat()
         << " to " << new_ndat << endl;

  Observation::set_ndat( new_ndat );
}

void dsp::TimeSeries::copy (const Observation* obs)
{
  const TimeSeries* ts = dynamic_cast<const TimeSeries*> (obs);
  if (ts)
    copy (ts);
  else
    DataSeries::copy (obs);
}

void dsp::TimeSeries::copy (const DataSeries* ds)
{
  const TimeSeries* ts = dynamic_cast<const TimeSeries*> (ds);
  if (ts)
    copy (ts);
  else
    DataSeries::copy (ds);
}

dsp::TimeSeries& dsp::TimeSeries::operator = (const TimeSeries& copy)
{
  DataSeries::operator=(copy);
  order = copy.order;
  data = (float*)buffer;
  return *this;
}

//! Offset the base pointer by offset time samples
void dsp::TimeSeries::seek (int64_t offset)
{
  if (verbose)
    cerr << "dsp::TimeSeries::seek " << offset << endl;

  if (offset == 0)
    return;

  if (offset > int64_t(get_ndat()))
    throw Error (InvalidRange, "dsp::TimeSeries::seek",
		 "offset=" I64 " > ndat=" UI64 ";"
                 " attempt to seek past end of data",
                 offset, get_ndat());

  float* fbuffer = (float*)buffer;

  int64_t reserve_step = get_ndim();
  if (order == OrderTFP)
    reserve_step *= get_nchan() * get_npol();

  int64_t current_offset = int64_t(data - fbuffer);
  int64_t float_offset = offset * reserve_step;

  if (-float_offset > current_offset)
    throw Error (InvalidRange, "dsp::TimeSeries::seek",
		 "offset=" I64 " > current_offset=" I64 ";"
                 " attempt to seek before start of data",
                 offset, current_offset/get_ndim());

  if (verbose)
    cerr << "dsp::TimeSeries::seek current_offset=" << current_offset
         << " float_offset=" << float_offset << endl;

  data += float_offset;
  assert (data >= fbuffer);

  set_ndat( get_ndat() - offset );

  input_sample += offset;
  assert (input_sample >= 0);

  change_start_time (offset);
}

//! Returns a uchar pointer to the first piece of data
unsigned char* dsp::TimeSeries::get_data()
{
  if (!data)
    throw Error (InvalidState,"dsp::TimeSeries::get_data",
		"Data buffer not initialized.  ndat=" UI64, get_ndat());

#ifdef _DEBUG
    cerr << "dsp::TimeSeries::get_data data-buffer="
         << int(data-(float*)buffer) << endl;
#endif

  return (unsigned char*) data;
}

//! Returns a uchar pointer to the first piece of data
const unsigned char* dsp::TimeSeries::get_data() const
{
  if (!data)
    throw Error (InvalidState,"dsp::TimeSeries::get_data() const",
		"Data buffer not initialized.  ndat=" UI64, get_ndat());

  return ((const unsigned char*)data);
}

//! Return pointer to the specified data block
float* dsp::TimeSeries::get_datptr (unsigned ichan, unsigned ipol)
{
  if (order != OrderFPT)
    throw Error (InvalidState, "dsp::TimeSeries::get_datptr",
		 "Not in Frequency, Polarization, Time Order");

  return reinterpret_cast<float*>( get_udatptr(ichan,ipol) );
}

//! Return pointer to the specified data block
const float*
dsp::TimeSeries::get_datptr (unsigned ichan, unsigned ipol) const
{
  if (order != OrderFPT)
    throw Error (InvalidState, "dsp::TimeSeries::get_datptr",
		 "Not in Frequency, Polarization, Time Order");

  return reinterpret_cast<const float*>( get_udatptr(ichan,ipol) );
}

//! Return pointer to the specified data block
float* dsp::TimeSeries::get_dattfp ()
{
  if (order != OrderTFP)
    throw Error (InvalidState, "dsp::TimeSeries::get_dattfp",
		 "Not in Time, Frequency, Polarization Order");

  return reinterpret_cast<float*>( get_udatptr(0,0) );
}

//! Return pointer to the specified data block
const float* dsp::TimeSeries::get_dattfp () const
{
  if (order != OrderTFP)
    throw Error (InvalidState, "dsp::TimeSeries::get_dattfp",
		 "Not in Time, Frequency, Polarization Order");

  return reinterpret_cast<const float*>( get_udatptr(0,0) );
}

uint64_t dsp::TimeSeries::get_nfloat_span () const
{
  return internal_get_subsize() / sizeof(float);
}

double dsp::TimeSeries::mean (unsigned ichan, unsigned ipol)
{
  if (get_ndim() != 1)
    throw Error (InvalidState, "dsp::TimeSeries::mean", "ndim != 1");

  float* data = get_datptr (ichan, ipol);
  double mean = 0.0;

  uint64_t _ndat = get_ndat();

  for (unsigned idat=0; idat<_ndat; idat++)
    mean += data[idat];

  return mean/double(get_ndat());
}

//! Match the internal memory layout of another TimeSeries
void dsp::TimeSeries::internal_match (const DataSeries* ds)
{
  const TimeSeries* ts = dynamic_cast<const TimeSeries*> (ds);
  if (ts)
    internal_match (ts);
  else
    DataSeries::internal_match (ds);
}

//! Match the internal memory layout of another DataSeries
void dsp::TimeSeries::internal_match (const TimeSeries* other)
{
  DataSeries::internal_match (other);

  order = other->order;
  reserve_ndat = other->reserve_ndat;
  reserve_nfloat = other->reserve_nfloat;
  input_sample = other->input_sample;

  uint64_t offset = other->data - (float*)other->buffer;

  data = (float*)buffer + offset;
}

void dsp::TimeSeries::copy_configuration (const Observation* copy)
{
  if( copy==this )
    return;

  DataSeries::copy_configuration ( copy );

  if (verbose)
    cerr << "dsp::TimeSeries::copy_configuration ndat=" << get_ndat()
	 << endl;
}

dsp::TimeSeries& dsp::TimeSeries::operator += (const TimeSeries& add)
{
  if( get_ndat()==0 )
    return operator=( add );

  if (!combinable (add))
    throw Error (InvalidState, "TimeSeries::operator+=",
		 "TimeSeries are not combinable");

  if (get_ndat() != add.get_ndat())
    throw Error (InvalidState, "TimeSeries::operator+=",
		 "ndat=" UI64 " != " UI64, get_ndat(), add.get_ndat());

  uint64_t npt = get_ndat() * get_ndim();

  if (order == OrderTFP)
  {
    npt *= get_nchan() * get_npol();
    float* data1 = get_dattfp ();
    const float* data2 = add.get_dattfp ();

    for (uint64_t ipt=0; ipt<npt; ipt++)
      data1[ipt] += data2[ipt];

    return *this;
  }

  for (unsigned ichan=0; ichan<get_nchan(); ichan++)
  {
    for (unsigned ipol=0; ipol<get_npol(); ipol++)
    {
      float* data1 = get_datptr (ichan, ipol);
      const float* data2 = add.get_datptr (ichan, ipol);

      for (uint64_t ipt=0; ipt<npt; ipt++)
        data1[ipt] += data2[ipt];
    }
  }

  return *this;
}

dsp::TimeSeries& dsp::TimeSeries::operator *= (float mult){
  if( verbose )
    fprintf(stderr,"In dsp::TimeSeries::operator*=()\n");

  if( fabs(mult-1.0) < 1.0e-9 )
    return *this;

  for (unsigned ichan=0; ichan<get_nchan(); ichan++){
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {
      float* dat = get_datptr(ichan,ipol);
      unsigned npts = unsigned(get_ndat()*get_ndim());

      for( unsigned i=0; i<npts; ++i)
	dat[i] *= mult;
    }
  }

  rescale( mult );

  return *this;
}

void dsp::TimeSeries::prepend_checks (const dsp::TimeSeries* pre,
				      uint64_t pre_ndat)
{
  if (pre->input_sample + pre_ndat != uint64_t(input_sample))
    throw Error (InvalidState, "dsp::TimeSeries::prepend_checks",
                 "data to be prepended end sample=" I64 "; "
                 "not contiguous with start sample=" I64,
                 pre->input_sample + pre_ndat, input_sample);
}

void dsp::TimeSeries::prepend (const dsp::TimeSeries* pre, uint64_t pre_ndat)
try
{
  if (!pre)
    return;

  if (!pre_ndat)
    pre_ndat = pre->get_ndat();

  prepend_checks (pre, pre_ndat);

  if (verbose)
    cerr << "dsp::TimeSeries::prepend " << pre_ndat << " samples" << endl;

  seek (-int64_t(pre_ndat));
  copy_data (pre, 0, pre_ndat);
}
catch (Error& error)
{
  throw error += "dsp::TimeSeries::prepend";
}

void dsp::TimeSeries::copy_data (const dsp::TimeSeries* copy,
                                 uint64_t idat_start, uint64_t copy_ndat) try
{
  if (verbose)
    cerr << "dsp::TimeSeries::copy_data to ndat=" << get_ndat()
	 << " from ndat=" << copy->get_ndat()
	 << "\n  idat_start=" << idat_start
	 << " copy_ndat=" << copy_ndat << endl;

  if (copy_ndat > get_ndat())
    throw Error (InvalidParam, "dsp::TimeSeries::copy_data",
		 "copy ndat=" UI64 " > this ndat=" UI64, copy_ndat, get_ndat());

  if (copy->get_ndim() != get_ndim())
    throw Error (InvalidParam, "dsp::TimeSeries::copy_data",
		 "copy ndim=%u > this ndim=%u", copy->get_ndim(), get_ndim());

  uint64_t offset = idat_start * get_ndim();
  uint64_t byte_count = copy_ndat * get_ndim() * sizeof(float);

  if (copy_ndat)
  {
    switch (order)
    {
    case OrderFPT:
      if (engine)
      {
        engine->copy_data_fpt (copy, idat_start, copy_ndat);
      }
      else
      {
        for (unsigned ichan=0; ichan<get_nchan(); ichan++)
        {
          for (unsigned ipol=0; ipol<get_npol(); ipol++)
          {
            float* to = get_datptr (ichan, ipol);
            const float* from = copy->get_datptr(ichan,ipol) + offset;
            memory->do_copy (to, from, size_t(byte_count));
          }
        }
      }
      break;

    case OrderTFP:
      {
      uint64_t times = get_nchan() * get_npol();
      offset *= times;
      byte_count *= times;

      float* to = get_dattfp ();
      const float* from = copy->get_dattfp() + offset;
      memory->do_copy (to, from, size_t(byte_count));
      }
      break;
    }
  }

  input_sample = copy->input_sample + idat_start;
}
catch (Error& error)
{
  throw error += "dsp::TimeSeries::copy_data";
}

#if 0
/*!
  \retval number of timesamples actually appended.
  If zero, then none were and we assume 'this' is full.
  If nonzero, but not equal to little->get_ndat(), then 'this' is full too.
  If equal to little->get_ndat(), it may/may not be full.
*/
uint64_t dsp::TimeSeries::append (const dsp::TimeSeries* little)
{
  if( verbose )
    fprintf(stderr,"In dsp::TimeSeries::append()\n");

  uint64_t ncontain = get_ndat();
  uint64_t ncopy = little->get_ndat();

  append_checks(ncontain,ncopy,little);
  if( ncontain==maximum_ndat() )
    return 0;

  if ( get_ndat() == 0 )
  {
    Observation::operator=( *little );
    set_ndat (0);
  }
  else if( !contiguous(*little) )
    throw Error (InvalidState, "dsp::TimeSeries::append()",
		 "next TimeSeries is not contiguous " + get_reason());

  for( unsigned ichan=0; ichan<get_nchan(); ichan++)
  {
    for( unsigned ipol=0; ipol<get_npol(); ipol++)
    {
      const float* from = little->get_datptr (ichan,ipol);
      float* to = get_datptr(ichan,ipol) + ncontain*get_ndim();

      memcpy (to, from, size_t(ncopy*get_ndim()*sizeof(float)));
    }
  }

  set_ndat (ncontain + ncopy);

  if( verbose )
    fprintf(stderr,"Returning from dsp::TimeSeries::append() with " UI64 "\n",
	    ncopy);

  return ncopy;
}

void dsp::TimeSeries::append_checks(uint64_t& ncontain,uint64_t& ncopy,
				    const TimeSeries* little){
  if( verbose ){
    fprintf(stderr,"dsp::TimeSeries::append_checks() ncopy=" UI64 "\n",ncopy);
    fprintf(stderr,"dsp::TimeSeries::append_checks() ncontain=" UI64 "\n",ncontain);
    fprintf(stderr,"dsp::TimeSeries::append_checks() maximum_ndat()=" UI64 "\n",maximum_ndat());
    fprintf(stderr,"dsp::TimeSeries::append_checks() nchan=%d npol=%d\n",get_nchan(),get_npol());
    fprintf(stderr,"dsp::TimeSeries::append_checks() subsize=" UI64 "\n",subsize);
  }

  if ( maximum_ndat() <= ncontain + ncopy ){
    ncopy = maximum_ndat() - ncontain;
    if( verbose )
      fprintf(stderr,"dsp::TimeSeries::append()- this append will fill up the timeseries from ndat="UI64" with ncopy="UI64" to ndat="UI64".\n",
	      get_ndat(), ncopy, get_ndat()+ncopy/little->get_ndim());
  }
}

#endif

dsp::TimeSeries& dsp::TimeSeries::swap_data (DataSeries& ds)
{
  TimeSeries* ts = dynamic_cast<TimeSeries*> (&ds);
  if (ts)
    swap_data (*ts);
  else
    DataSeries::swap_data(ds);

  return *this;
}
  
dsp::TimeSeries& dsp::TimeSeries::swap_data (TimeSeries& ts)
{
  DataSeries::swap_data( ts );
  float* tmp = data; data = ts.data; ts.data = tmp;

  return *this;
}

void dsp::TimeSeries::check (float min, float max)
{
  if (get_detected()) {
    min *= fabs(min);
    max *= max;
  }

  min *= scale;
  max *= scale;

  uint64_t _ndat = get_ndat();

  for (unsigned ichan=0; ichan<get_nchan(); ichan++)
    for (unsigned ipol=0; ipol<get_npol(); ipol++) {

      float* dat = get_datptr (ichan, ipol);

      for (uint64_t idat=0; idat<_ndat; idat++)
	for (unsigned idim=0; idim<get_ndim(); idim++) {
	  if (isnan (*dat) || *dat < min || *dat > max)
	    cerr << "dsp::TimeSeries::check data ["
	      "f=" << ichan << ", "
	      "p=" << ipol << ", "
	      "t=" << idat << ", "
	      "d=" << idim << "] = " << *dat << endl;
	  dat ++;
	}
    }
}

bool from_range(unsigned char* fr,const dsp::TimeSeries* tseries)
{
  unsigned char* fr_min = (unsigned char*)tseries->get_datptr(0,0);
  unsigned char* fr_max = (unsigned char*)(tseries->get_datptr(tseries->get_nchan()-1,tseries->get_npol()-1)+tseries->get_ndat());
  if( fr>=fr_min && fr<fr_max)
    return true;
  fprintf(stderr,"fr_min=%p fr_max=%p fr=%p\n",
	  fr_min,fr_max,fr);
  return false;
}


/*! This method is used by the InputBuffering policy */
void dsp::TimeSeries::change_reserve (int64_t change) const
{
  if (verbose)
    cerr << "dsp::TimeSeries::change_reserve (" << change << ")" << endl;

  TimeSeries* thiz = const_cast<TimeSeries*>(this);

  uint64_t reserve_step = get_ndim();
  if (order == OrderTFP)
    reserve_step *= get_nchan() * get_npol();

  if (change < 0)
  {
    uint64_t decrease = -change;
    if (decrease > reserve_ndat)
      throw Error (InvalidState, "dsp::TimeSeries::change_reserve",
		   "decrease=" I64 "; reserve_ndat=" UI64,
		   decrease, reserve_ndat);

    thiz->reserve_ndat -= decrease;
    thiz->reserve_nfloat -= decrease * reserve_step;
  }
  else
  {
    thiz->reserve_ndat += change;
    thiz->reserve_nfloat += change * reserve_step;
  }

  if (match)
  {
    match->reserve_ndat = reserve_ndat;
    match->reserve_nfloat = reserve_nfloat;
  }
}

void dsp::TimeSeries::set_match (TimeSeries* other)
{
  match = other;
}

void dsp::TimeSeries::finite_check () const
{
  unsigned nfloat = get_ndat() * get_ndim();
  unsigned nchan = get_nchan();
  unsigned npol = get_npol();
  unsigned non_finite = 0;

  if (verbose)
    cerr << "dsp::TimeSeries::finite_check nchan=" << nchan << " npol=" << npol
	 << " ndim=" << get_ndim() << " ndat=" << get_ndat() << endl;

  for (unsigned ichan=0; ichan < nchan; ichan++)
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      const float* from = get_datptr (ichan, ipol);
      for (unsigned ibin=0; ibin < nfloat; ibin++)
	if (!isfinite(from[ibin]))
	{
#ifdef _DEBUG
	  cerr << "not finite ichan=" << ichan << " ipol=" << ipol
	       << " ptr=" << from << " ibin=" << ibin << endl;
#endif
	  non_finite ++;
	}
    }

  if (non_finite)
    throw Error (InvalidParam, "dsp::TimeSeries::finite_check",
		 "%u/%u non-finite values",
		 non_finite, nfloat * nchan * npol);
}

void dsp::TimeSeries::set_engine( Engine* _engine )
{
  engine = _engine;
  engine->prepare (this);
}


unsigned dsp::TimeSeries::get_stride () const
{
  if (order == OrderFPT) {

    if (get_npol() > 1) {
      if (verbose) {
        std::cerr << "dsp::TimeSeries::get_stride: npol > 1" << std::endl;
      }
      return get_datptr(0, 1) - get_datptr(0, 0);
    } else {
      if (get_nchan() > 1) {
        if (verbose) {
          std::cerr << "dsp::TimeSeries::get_stride: nchan > 1" << std::endl;
          std::cerr << "dsp::TimeSeries::get_stride: ndat="
            << get_ndat() << " get_datptr(1, 0) - get_datptr(0, 0)="
            << get_datptr(1, 0) - get_datptr(0, 0)
            << std::endl;
        }
        return get_datptr(1, 0) - get_datptr(0, 0);
      } else {
        if (verbose) {
          std::cerr << "dsp::TimeSeries::get_stride: nchan = 1 and npol = 1" << std::endl;
          std::cerr << "dsp::TimeSeries::get_stride: ndat="
            << get_ndat() << " get_nfloat_span()=" << get_nfloat_span()
            << std::endl;
        }
        return (unsigned) get_nfloat_span();
      }
    }
  } else {
    throw "dsp::TimeSeries::get_stride not supported for OrderTFP";
  }
}
