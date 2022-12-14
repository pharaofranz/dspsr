/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TScrunch.h"
#include "dsp/InputBuffering.h"

#include "Error.h"

using namespace std;

dsp::TScrunch::TScrunch (Behaviour place)
  : Transformation <TimeSeries, TimeSeries> ("TScrunch", place)
{
  factor = 0;
  time_resolution = 0;
  use_tres = false;
  prepared = false;

  set_buffering_policy (new InputBuffering (this));
}

void dsp::TScrunch::set_factor( unsigned samples )
{
  factor = samples;
  use_tres = false;
}

void dsp::TScrunch::set_time_resolution( double microseconds )
{
  time_resolution = microseconds;
  use_tres = true;
}

unsigned dsp::TScrunch::get_factor() const
{
  if (use_tres)
  {
    if( time_resolution <= 0.0 )
      throw Error(InvalidState,"dsp::Tscrunch::get_factor",
		  "invalid time resolution:%lf", time_resolution);
    double in_tsamp = 1.0e6/input->get_rate();  // in microseconds
    factor = unsigned(time_resolution/in_tsamp + 0.00001);

    if ( factor<1 )
      factor = 1;

    use_tres = false;
    time_resolution = 0.0;
  }

  return factor;
}

double dsp::TScrunch::get_time_resolution() const
{
  if (!time_resolution)
    time_resolution = 1.0e6/(input->get_rate()*double(factor));

  return time_resolution;
}

void dsp::TScrunch::set_engine( Engine* _engine )
{
  engine = _engine;
}

// Initial preparation of relevant attributes
void dsp::TScrunch::prepare ()
{
  sfactor = get_factor();
  if (verbose)
    cerr << "dsp::TScrunch::prepare factor=" << sfactor << endl;

  if (has_buffering_policy())
  {
    if (verbose)
      cerr << "dsp::TScrunch::prepare set_maximum_samples(" << sfactor << ")" << endl;
    get_buffering_policy()->set_maximum_samples (sfactor);
  }

  if (verbose)
    cerr << "dsp::TScrunch::prepare prepare_output()" << endl;
  prepare_output ();
  prepared = true;
}

// reserve the maximum required output space
void dsp::TScrunch::reserve ()
{
  if (verbose)
    cerr << "dsp::TScrunch::reserve prepare_output()" << endl;
  prepare_output();

  if (verbose)
    cerr << "dsp::TScrunch::reserve input ndat=" << get_input()->get_ndat()
         << " scrunch=" << sfactor << " output ndat=" << output_ndat << endl;

  // only resize if output of place
  if (input.get() != output.get())
    output->resize (output_ndat);
}

// preapre the output TimeSeries
void dsp::TScrunch::prepare_output ()
{
  output_ndat = get_input()->get_ndat() / sfactor;

  if (input.get() != output.get())
  {
    if (verbose)
      cerr << "dsp::TScrunch::prepare_output copying configuration" << endl;
    get_output()->copy_configuration (get_input());
  }

  // this is necessary if we buffer further down the line otherwise, samples are misaligned
  get_output()->set_input_sample (get_input()->get_input_sample() / sfactor);
  output->rescale (sfactor);
  output->set_rate (input->get_rate() / sfactor);
  output->set_order (input->get_order());
}

void dsp::TScrunch::transformation ()
{
  if (verbose)
    cerr << "dsp::TScrunch::transformation" << endl;

  if (!prepared)
    prepare ();

  // ensure the output TimeSeries is large enough
  reserve ();

  if (sfactor == 1)
    throw Error(InvalidState,"dsp::TScrunch::transformation",
                "cannot support Tscrunch of 1");

  if (!sfactor)
    throw Error (InvalidState, "dsp::TScrunch::get_factor",
		   "scrunch factor not set");

  if( !input->get_detected() )
    throw Error(InvalidState,"dsp::TScrunch::transformation()",
		"invalid input state: " + tostring(input->get_state()));

  if (verbose)
    cerr << "dsp::TScrunch::transformation input ndat=" << input->get_ndat()
         << " output ndat=" << output_ndat << " sfactor=" << sfactor << endl;

  if (has_buffering_policy())
    get_buffering_policy()->set_next_start (output_ndat * sfactor);

  if (sfactor==1)
  {
    if (verbose)
      cerr << "dsp::TScrunch::transformation sfactor=1, using copy operator" << endl;
    if( input.get() != output.get() )
      output->operator=( *input );
    output->set_input_sample(input->get_input_sample());
    return;
  }

  switch (input->get_order())
  {
    case TimeSeries::OrderFPT:
      if (engine)
        engine->fpt_tscrunch (get_input(), get_output(), sfactor);
      else
        fpt_tscrunch ();
      break;

    case TimeSeries::OrderTFP:
      tfp_tscrunch ();
      break;
  }

  if( input.get() == output.get() )
    output->set_ndat( output_ndat );
}

void dsp::TScrunch::fpt_tscrunch ()
{
  const unsigned input_nchan = input->get_nchan();
  const unsigned input_npol = input->get_npol();
  const unsigned ndim = input->get_ndim();

  for (unsigned idim=0; idim < ndim; ++idim)
  {
    for (unsigned ichan=0; ichan<input_nchan; ichan++)
    {
      for (unsigned ipol=0; ipol<input_npol; ipol++)
      {
        const float* in = input->get_datptr(ichan, ipol) + idim;
        float* out = output->get_datptr(ichan, ipol) + idim;

        unsigned input_idat=0;

        for (unsigned output_idat=0;
            output_idat<output_ndat*ndim; output_idat+=ndim)
        {
          unsigned stop = (input_idat + sfactor*ndim);

          out[output_idat] = in[input_idat];
          input_idat += ndim;

          for( ; input_idat<stop; input_idat += ndim)
            out[output_idat] += in[input_idat];

        }
      } // for each ipol
    } // for each ichan
  if (idim > 0)
    cerr <<"this is a problem right now"<<std::endl;
  } // for each idim
}

void dsp::TScrunch::tfp_tscrunch ()
{
  const unsigned input_nchan = input->get_nchan();
  const unsigned input_npol = input->get_npol();
  const unsigned input_ndim = input->get_ndim();

  const unsigned nfloat = input_nchan * input_npol * input_ndim;

  const float* indat = input->get_dattfp ();
  float* outdat = output->get_dattfp ();

  for( unsigned output_idat=0; output_idat<output_ndat; ++output_idat)
  {
    for (unsigned ifloat=0; ifloat < nfloat; ifloat++)
      outdat[ifloat] = indat[ifloat];

    for (unsigned ifactor=1; ifactor < sfactor; ifactor++)
    {
      indat += nfloat;
      for (unsigned ifloat=0; ifloat < nfloat; ifloat++)
        outdat[ifloat] += indat[ifloat];
    }

    indat += nfloat;
    outdat += nfloat;
  }
}
