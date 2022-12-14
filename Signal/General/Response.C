/***************************************************************************
 *
 *   Copyright (C) 2002-2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Response.h"
#include "dsp/Observation.h"
#include "dsp/OptimalFFT.h"

#include "Error.h"
#include "Jones.h"
#include "cross_detect.h"

#include <vector>
#include <assert.h>
#include <math.h>

using namespace std;

//#define _DEBUG

/*! If specified, this attribute restricts the value for ndat chosen by
  the set_optimal_ndat method, enabling the amount of RAM used by the calling
  process to be limited. */
unsigned dsp::Response::ndat_max = 0;

dsp::Response::Response ()
{
  impulse_pos = impulse_neg = 0;

  whole_swapped = dc_centred = false;
  swap_divisions = 0;

  npol = 2;
  ndim = 1;
  nchan = 1;
  input_nchan = 1;
  step = 1;
}

//! Destructor
dsp::Response::~Response ()
{
}

//! Copy constructor
dsp::Response::Response (const Response& response)
{
  operator = (response);
}


//! Assignment operator
const dsp::Response& dsp::Response::operator = (const Response& response)
{
  if (this == &response)
    return *this;

  Shape::operator = ( response );

  input_nchan = response.input_nchan;
  impulse_pos = response.impulse_pos;
  impulse_neg = response.impulse_neg;
  whole_swapped = response.whole_swapped;
  swap_divisions = response.swap_divisions;
  dc_centred = response.dc_centred;
  step = response.step;

  changed.send (*this);
  return *this;
}

//! Multiplication operator
const dsp::Response& dsp::Response::operator *= (const Response& response)
{
  if (ndat*nchan != response.ndat * response.nchan)
    throw Error (InvalidParam, "dsp::Response::operator *=",
		 "ndat=%u*nchan=%u != other ndat=%u*nchan=%u", ndat, nchan,
                 response.ndat, response.nchan);

  if (npol < response.npol)
    throw Error (InvalidParam, "dsp::Response::operator *=",
		 "npol=%d < *.npol=%d", npol, response.npol);

  /*
    perform A = A * B where
    A = this->buffer
    B = response->buffer
    (so B operates on A's buffer)
  */

  if (response.ndim > ndim)
    throw Error (InvalidParam, "dsp::Response::operator *=",
      "response.ndim=%u < ndim=%u, incorrect ndim order for operation",
      response.ndim, ndim);

  unsigned original_step = response.step;
  response.step = ndim / response.ndim;

   for (unsigned istep=0; istep < step; istep++)
    for (unsigned ipol=0; ipol < npol; ipol++)
     for (unsigned ichan=0; ichan < nchan; ichan++)
     	response.operate (buffer + offset*ipol + ichan*ndat*ndim + istep, ipol, ichan);

  response.step = original_step;

  changed.send (*this);
  return *this;
}

//! Calculate the impulse_pos and impulse_neg attributes
void dsp::Response::prepare (const Observation* input, unsigned channels)
{
  impulse_neg = impulse_pos = 0;
}

/*! The ordering of frequency channels in the response function depends
  upon:
  <UL>
  <LI> the state of the input Observation (real or complex); and </LI>
  <LI> Operations to be performed upon the Observation
       (e.g. simultaneous filterbank) </LI>
  </UL>
  As well, sub-classes of Response may need to dynamically check, refine, or
  define their frequency response function based on the state of the
  input Observation or the number of channels into which it will be divided.

  \param input Observation to which the frequency response is to be matched

  \param channels If specified, the number of filterbank channels into
  which the input Observation will be divided.  Response::match does not
  use this parameter, but sub-classes may find it useful.

 */
void dsp::Response::match (const Observation* input, unsigned channels)
{
  if (verbose)
    cerr << "dsp::Response::match input.nchan=" << input->get_nchan()
	 << " channels=" << channels << endl;
  input_nchan = input->get_nchan();
  if ( input->get_nchan() == 1 ) {

    // if the input Observation is single-channel, complex sampled
    // data, then the first forward FFT performed on this data will
    // result in a swapped spectrum
    if ( input->get_dual_sideband() && !whole_swapped ) {
      if (verbose)
	cerr << "dsp::Response::match swap whole" << endl;
      doswap ();
    }
  }
  else  {

    // if the filterbank channels are centred on DC
    if ( input->get_dc_centred() && !dc_centred ) {
      if (verbose)
	cerr << "dsp::Response::match rotate half channel" << endl;

      if ( swap_divisions )
        doswap ( swap_divisions );

      rotate (-int(ndat/2));
      dc_centred = true;
    }

    // if the input Observation is multi-channel, complex sampled data,
    // then each FFT performed will result in little swapped spectra
    if ( input->get_dual_sideband() && swap_divisions != input->get_nchan() )
    {
      if (verbose)
	cerr << "dsp::Response::match swap channels"
	  " (nchan=" << input->get_nchan() << ")" << endl;
      doswap ( input->get_nchan() );
    }

    // the ordering of the filterbank channels may be swapped
    if ( input->get_swap() && !whole_swapped ) {
      if (verbose)
	cerr << "dsp::Response::match swap whole (nchan=" << nchan << ")"
	     << endl;
      doswap ();
    }
  }
}

//! Returns true if the dimension and ordering match
bool dsp::Response::matches (const Shape* shape)
{
  const Response* response = dynamic_cast<const Response*> (shape);

  if (!response)
    return false;

  return
    whole_swapped == response->whole_swapped &&
    swap_divisions == response->swap_divisions &&
    dc_centred == response->dc_centred &&

    nchan == response->get_nchan() &&
    ndat == response->get_ndat();
}

//! Match the frequency response to another Response
void dsp::Response::match (const Response* response)
{
  if (matches (response))
    return;

  if (verbose)
    cerr << "dsp::Response::match Response" << endl;

  resize (npol, response->get_nchan(),
	  response->get_ndat(), ndim);

  input_nchan = response->input_nchan;
  whole_swapped = response->whole_swapped;
  swap_divisions = response->swap_divisions;
  dc_centred = response->dc_centred;

  zero();
}

void dsp::Response::mark (Observation* output)
{

}

//! Set the flag for a bin-centred spectrum
void dsp::Response::set_dc_centred (bool _dc_centred)
{
  dc_centred = _dc_centred;
}

void dsp::Response::naturalize ()
{
  if (verbose)
    cerr << "dsp::Response::naturalize" << endl;

  if ( whole_swapped )
  {
    if (verbose)
      cerr << "dsp::Response::naturalize whole bandpass swap" << endl;
    doswap ();
  }

  if ( swap_divisions )
  {
    if (verbose)
      cerr << "dsp::Response::naturalize sub-bandpass swap" << endl;
    doswap ( swap_divisions );
  }

  if ( dc_centred )
  {
    if (verbose)
      cerr << "dsp::Response::naturalize rotation" << endl;
    rotate (ndat/2);
    dc_centred = false;
  }
}

/*!  Using the impulse_pos and impulse_neg attributes, this method
  determines the minimum acceptable ndat for use in convolution.  This
  is given by the smallest power of two greater than or equal to the
  twice the sum of impulse_pos and impulse_neg. */
unsigned dsp::Response::get_minimum_ndat () const
{
  double impulse_tot = impulse_pos + impulse_neg;

  if (impulse_tot == 0)
    return 0;

  unsigned min = unsigned( pow (2.0, ceil( log(impulse_tot)/log(2.0) )) );
  while (min <= impulse_tot)
    min *= 2;

  if (verbose)
    cerr << "dsp::Response::get_minimum_ndat impulse_tot=" << impulse_tot
	 << " min power of two=" << min << endl;

  return min;
}

extern "C" uint64_t
optimal_fft_length (uint64_t nbadperfft, uint64_t nfft_max, char verbose);

/*!  Using the get_minimum_ndat method and the max_ndat static attribute,
  this method determines the optimal ndat for use in convolution. */
void dsp::Response::set_optimal_ndat ()
{
  unsigned ndat_min = get_minimum_ndat ();

  if (verbose)
    cerr << "Response::set_optimal_ndat minimum ndat=" << ndat_min << endl;

  if (ndat_max && ndat_max < ndat_min)
    throw Error (InvalidState, "Response::set_optimal_ndat",
		  "specified maximum ndat (%d) < required minimum ndat (%d)",
		  ndat_max, ndat_min);

  int64_t optimal_ndat = 0;

  if (optimal_fft)
  {
    optimal_fft->set_nchan (nchan);
    optimal_ndat = optimal_fft->get_nfft (impulse_pos+impulse_neg);
  }
  else
  {
    optimal_ndat = optimal_fft_length (impulse_pos+impulse_neg,
				       ndat_max, verbose);
    if (optimal_ndat < 0)
      throw Error (InvalidState, "Response::set_optimal_ndat",
		   "optimal_fft_length failed");
  }

  resize (npol, nchan, unsigned(optimal_ndat), ndim);
}

void dsp::Response::set_optimal_fft (OptimalFFT* policy)
{
  optimal_fft = policy;
}

dsp::OptimalFFT* dsp::Response::get_optimal_fft () const
{
  return optimal_fft;
}

bool dsp::Response::has_optimal_fft () const
{
  return optimal_fft;
}

void dsp::Response::check_ndat () const
{
  if (ndat_max && ndat > ndat_max)
    throw Error (InvalidState, "Response::check_ndat",
		 "specified maximum ndat (%d) < specified ndat (%d)",
		 ndat_max, ndat);

  unsigned ndat_min = get_minimum_ndat ();

  if (verbose)
    cerr << "Response::check_ndat minimum ndat=" << ndat_min << endl;

  if (ndat < ndat_min)
    throw Error (InvalidState, "dsp::Response::check_ndat",
		 "specified ndat (%d) < required minimum ndat (%d)",
		 ndat, ndat_min);
}

//! Get the passband
vector<float> dsp::Response::get_passband (unsigned ipol, int ichan) const
{
  assert (ndim == 1);

  // output all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan < 0) {
    npts *= nchan;
    ichan = 0;
  }

  register float* f_p = buffer + offset * ipol + ichan * ndat * ndim;

  vector<float> retval (npts);
  for (unsigned ipt=0; ipt < npts; ipt++)
    retval[ipt]=f_p[ipt];

  return retval;
}

// /////////////////////////////////////////////////////////////////////////

/*! Multiplies an array of complex points by the complex response

  \param ipol the polarization of the data (Response may optionally
  contain a different frequency response function for each polarization)

  \param data an array of nchan*ndat complex numbers */

void dsp::Response::operate (float* data, unsigned ipol, int ichan) const
{
  if( ichan < 0 )
    operate(data,ipol,ichan,nchan);
  else
    operate(data,ipol,ichan,1);
}

//! Multiply spectrum by complex frequency response
void
dsp::Response::operate (float* spectrum, unsigned poln, int ichan_start, unsigned nchan_op) const
{
  assert (ndim == 2);

  unsigned ipol = poln;

  // one filter may apply to two polns
  if (ipol >= npol)
    ipol = 0;

  // do all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan_start < 0) {
    npts *= nchan;
    ichan_start = 0;
  }
  else
    npts *= nchan_op;

  register float* d_from = spectrum;
  register float* f_p = buffer + offset * ipol + ichan_start * ndat * ndim;

  /*
    this operates on spectrum; i.e.  A = A * B where
    A = spectrum
    B = this->buffer
  */

#ifdef _DEBUG
  cerr << "dsp::Response::operate nchan=" << nchan << " ipol=" << ipol
       << " buf=" << buffer << " f_p=" << f_p
       << " off=" << offset(ipol) << endl;
#endif

  // the idea is that by explicitly calling the values from the
  // arrays into local stack space, the routine should run faster
  register float d_r;
  register float d_i;
  register float f_r;
  register float f_i;

  // cerr << "dsp::Response::operate step=" << step << endl;

  for (unsigned ipt=0; ipt<npts; ipt++)
  {
    d_r = d_from[0];
    d_i = d_from[1];
    f_r = f_p[0];
    f_i = f_p[1];

    d_from[0] = f_r * d_r - f_i * d_i;
    d_from[1] = f_i * d_r + f_r * d_i;

    d_from += 2 * step;
    f_p += 2;
  }

  // cerr << "dsp::Response::operate done" << endl;
}

//! Multiply spectrum by complex frequency response
void
dsp::Response::operate (float* input_spectrum, float * output_spectrum,
                        unsigned poln, int ichan_start, unsigned nchan_op) const
{
  assert (ndim == 2);

  unsigned ipol = poln;

  // one filter may apply to two polns
  if (ipol >= npol)
    ipol = 0;

  // do all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan_start < 0) {
    npts *= nchan;
    ichan_start = 0;
  }
  else
    npts *= nchan_op;

  register float* d_from = input_spectrum;
  register float* d_into = output_spectrum;
  register float* f_p = buffer + offset * ipol + ichan_start * ndat * ndim;

  /*
    this operates on spectrum; i.e.  C = A * B where
    A = input_spectrum
    B = this->buffer
    C = output_spectrum
  */

#ifdef _DEBUG
  cerr << "dsp::Response::operate nchan=" << nchan << " ipol=" << ipol
       << " buf=" << buffer << " f_p=" << f_p
       << " off=" << offset(ipol) << endl;
#endif

  // the idea is that by explicitly calling the values from the
  // arrays into local stack space, the routine should run faster
  register float d_r;
  register float d_i;
  register float f_r;
  register float f_i;

  for (unsigned ipt=0; ipt<npts; ipt++)
  {
    d_r = d_from[0];
    d_i = d_from[1];
    f_r = f_p[0];
    f_i = f_p[1];

    d_into[0] = f_r * d_r - f_i * d_i;
    d_into[1] = f_i * d_r + f_r * d_i;

    d_from += 2 * step;
    d_into += 2 * step;
    f_p += 2;
  }
}

// /////////////////////////////////////////////////////////////////////////

/*! Adds the square of each complex point to the current power spectrum

  \param data an array of nchan*ndat complex numbers

  \param ipol the polarization of the data (Response may optionally
  integrate a different power spectrum for each polarization)

*/
void dsp::Response::integrate (float* data, unsigned ipol, int ichan)
{
  assert (ndim == 1);
  assert (npol != 4);

  // may be used to integrate total intensity from two polns
  if (ipol >= npol)
    ipol = 0;

  // do all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan < 0) {
    npts *= nchan;
    ichan = 0;
  }

  register float* d_p = data;
  register float* f_p = buffer + offset * ipol + ichan * ndat * ndim;

#ifdef _DEBUG
  cerr << "dsp::Response::integrate ipol=" << ipol
       << " buf=" << buffer << " f_p=" << f_p
       << "off=" << offset(ipol) << endl;
#endif

  register float d;
  register float t;

  for (unsigned ipt=0; ipt<npts; ipt++) {
    d = *d_p; d_p ++; // Re
    t = d*d;
    d = *d_p; d_p ++; // Im

    *f_p += t + d*d;
    f_p ++;
  }
}

void dsp::Response::set (const vector<complex<float> >& filt)
{
  // one poln, one channel, complex
  resize (1, 1, filt.size(), 2);
  float* f = buffer;

  for (unsigned idat=0; idat<filt.size(); idat++) {
    // Re
    *f = filt[idat].real();
    f++;
    // Im
    *f = filt[idat].imag();
    f++;
  }
}

// /////////////////////////////////////////////////////////////////////////
//
// Response::operate - multiplies two complex arrays by complex matrix Response
// ndat = number of complex points
//
void dsp::Response::operate (float* data1, float* data2, int ichan) const
{
  assert (ndim == 8);

  // do all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan < 0) {
    npts *= nchan;
    ichan = 0;
  }

  float* d1_rp = data1;
  float* d1_ip = data1 + 1;
  float* d2_rp = data2;
  float* d2_ip = data2 + 1;

  float* f_p = buffer + ichan * ndat * ndim;

  register float d_r;
  register float d_i;
  register float f_r;
  register float f_i;

  register float r1_r;
  register float r1_i;
  register float r2_r;
  register float r2_i;

  for (unsigned ipt=0; ipt<npts; ipt++) {

    // ///////////////////////
    // multiply: r1 = f11 * d1
    d_r = *d1_rp;
    d_i = *d1_ip;
    f_r = *f_p; f_p ++;
    f_i = *f_p; f_p ++;

    r1_r = f_r * d_r - f_i * d_i;
    r1_i = f_i * d_r + f_r * d_i;

    // ///////////////////////
    // multiply: r2 = f21 * d1
    f_r = *f_p; f_p ++;
    f_i = *f_p; f_p ++;

    r2_r = f_r * d_r - f_i * d_i;
    r2_i = f_i * d_r + f_r * d_i;

    // ////////////////////////////
    // multiply: d2 = r2 + f22 * d2
    d_r = *d2_rp;
    d_i = *d2_ip;
    f_r = *f_p; f_p ++;
    f_i = *f_p; f_p ++;

    *d2_rp = r2_r + f_r * d_r - f_i * d_i;
    d2_rp += 2;
    *d2_ip = r2_i + f_i * d_r + f_r * d_i;
    d2_ip += 2;

    // ////////////////////////////
    // multiply: d1 = r1 + f12 * d2
    f_r = *f_p; f_p ++;
    f_i = *f_p; f_p ++;

    *d1_rp = r1_r + f_r * d_r - f_i * d_i;
    d1_rp += 2;
    *d1_ip = r1_i + f_i * d_r + f_r * d_i;
    d1_ip += 2;
  }
}

void dsp::Response::integrate (float* data1, float* data2, int ichan)
{
  if (ndim != 1)
    throw Error (InvalidState, "dsp::Response::integrate",
                 "ndim=%u != 1", ndim);

  if (npol != 4)
    throw Error (InvalidState, "dsp::Response::integrate",
                 "npol=%u != 4", npol);

  // do all channels at once if ichan < 0
  unsigned npts = ndat;
  if (ichan < 0) {
    npts *= nchan;
    ichan = 0;
  }

  float* data = buffer + ichan * ndat * ndim;

  if (verbose)
    cerr << "dsp::Response::integrate::cross_detect_int" << endl;

  cross_detect_int (npts, data1, data2,
		    data, data + offset,
		    data + 2*offset, data + 3*offset, 1);
}

void dsp::Response::set (const vector<Jones<float> >& response)
{
  if (verbose)
    cerr << "dsp::Response::set" <<endl;

  // one poln, one channel, Jones
  resize (1, 1, response.size(), 8);

  float* f = buffer;

  for (unsigned idat=0; idat<response.size(); idat++) {

    // for efficiency, the elements of a Jones matrix Response
    // are ordered as: f11, f21, f22, f12

    for (int j=0; j<2; j++)
      for (int i=0; i<2; i++) {
	complex<double> element = response[idat]( (i+j)%2, j );
	// Re
	*f = element.real();
	f++;
	// Im
	*f = element.imag();
	f++;
      }
  }
}

// ////////////////////////////////////////////////////////////////
//
// dsp::Response::doswap swaps the passband(s)
//
// If 'each_chan' is true, then the nchan units (channels) into which
// the Response is logically divided will be swapped individually
//
void dsp::Response::doswap (unsigned divisions)
{
  if (nchan == 0)
    throw Error (InvalidState, "dsp::Response::swap",
		 "invalid nchan=%d", nchan);

  unsigned half_npts = (ndat * ndim * nchan) / (2 * divisions);

  if (half_npts < 2)
    throw Error (InvalidState, "dsp::Response::swap",
		 "invalid npts=%d (ndat=%u ndim=%u nchan=%u)",
                 half_npts, ndat, ndim, nchan);

#ifdef _DEBUG
  cerr << "dsp::Response::swap"
    " nchan=" << nchan <<
    " ndat=" << ndat <<
    " ndim=" << ndim <<
    " npts=" << half_npts
       << endl;
#endif

  float* ptr1 = 0;
  float* ptr2 = 0;
  float  temp = 0;

  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    ptr1 = buffer + offset * ipol;
    ptr2 = ptr1 + half_npts;

    for (unsigned idiv=0; idiv<divisions; idiv++)
    {

      for (unsigned ipt=0; ipt<half_npts; ipt++)
      {
	temp = *ptr1;
	*ptr1 = *ptr2; ptr1++;
	*ptr2 = temp; ptr2++;
      }

      ptr1+=half_npts;
      ptr2+=half_npts;
    }
  }

  if (divisions == 1)
    whole_swapped = !whole_swapped;
  else if (divisions == swap_divisions)
    swap_divisions = 0;
  else
    swap_divisions = divisions;
}

void dsp::Response::flagswap (unsigned divisions)
{
  if (divisions == 1)
    whole_swapped = true;
  else
    swap_divisions = divisions;
}

void dsp::Response::calc_lcf (
	unsigned a, unsigned b, const Rational& osf, vector<unsigned>& result)
{
  // if (verbose) {
  //   std::cerr << "dsp::Response::calc_lcf:"
  //     << " a=" << a
  //     << " b=" << b
  //     << " osf=" << osf
  //     << std::endl;
  // }
  //   std::cerr << "dsp::Response::calc_lcf: quot=" << a / (osf.get_denominator()*b) << std::endl;
  //   std::cerr << "dsp::Response::calc_lcf: rem=" << a % (osf.get_denominator()*b) << std::endl;
  // }
  result[0] = a / (osf.get_denominator()*b);
  result[1] = a % (osf.get_denominator()*b);
}


void dsp::Response::calc_oversampled_fft_length (
  unsigned* _fft_length,
  unsigned _nchan,
  const Rational& osf,
  int direction
)
{
  if (verbose) {
    std::cerr << "dsp::Response::calc_oversampled_fft_length:"
      << " _fft_length=" << *_fft_length
      << " _nchan=" << _nchan
      << " osf=" << osf
      << std::endl;
  }
  vector<unsigned> max_fft_length_lcf(2);
	calc_lcf(*_fft_length, _nchan, osf, max_fft_length_lcf);
  while (max_fft_length_lcf[1] != 0 || fmod(log2(max_fft_length_lcf[0]), 1) != 0){
    if (direction == -1) {
      if (max_fft_length_lcf[1] != 0) {
        (*_fft_length) -= max_fft_length_lcf[1];
      } else {
        (*_fft_length) -= 2;
      }
    } else if (direction == 1) {
      if (max_fft_length_lcf[1] != 0) {
        (*_fft_length) = osf.get_denominator()*_nchan*(max_fft_length_lcf[0] + 1);
      } else {
        (*_fft_length) += 2;
      }
    }
    calc_lcf(*_fft_length, _nchan, osf, max_fft_length_lcf);
  }
  // if (verbose) {
  //   std::cerr << "dsp::Response::calc_oversampled_fft_length: result"
  //     << " _fft_length=" << *_fft_length
  //     << std::endl;
  // }
}

void dsp::Response::calc_oversampled_discard_region(
  unsigned* _discard_neg,
  unsigned* _discard_pos,
  unsigned _nchan,
  const Rational& osf
)
{
  if (verbose) {
    std::cerr << "dsp::Response::calc_oversampled_discard_region"
      << " _discard_neg=" << *_discard_neg
      << " _discard_pos=" << *_discard_pos
      << " _nchan=" << _nchan
      << " osf=" << osf
      << std::endl;
  }
	vector<unsigned> n = {*_discard_pos, *_discard_neg};
  vector<vector<unsigned>> lcfs(2);
  unsigned min_n;
	vector<unsigned> lcf(2);
  for (int i=0; i<n.size(); i++) {
    min_n = n[i];
    calc_lcf(min_n, _nchan, osf, lcf);
    if (lcf[1] != 0) {
      min_n += osf.get_denominator()*_nchan - lcf[1];
			lcf[0] += 1;
	    lcf[1] = 0;
    }
    lcfs[i] = lcf;
    n[i] = min_n;
  }

  *_discard_pos = n[0];
  *_discard_neg = n[1];
  // if (verbose) {
  //   std::cerr << "dsp::Response::calc_oversampled_discard_region: result"
  //     << " _discard_neg=" << *_discard_neg
  //     << " _discard_pos=" << *_discard_pos
  //     << std::endl;
  // }
}
