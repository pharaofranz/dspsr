/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TransferCUDA.h"

#include "Error.h"

#include <iostream>
using namespace std;

//! Default constructor- always inplace
dsp::TransferCUDA::TransferCUDA(cudaStream_t _stream)
  : Transformation<TimeSeries,TimeSeries> ("CUDA::Transfer", outofplace)
{
  stream = _stream;
  kind = cudaMemcpyHostToDevice;
}

//! Do stuff
void dsp::TransferCUDA::transformation ()
{
  if (verbose) {
    cerr << "dsp::TransferCUDA::transformation" << endl;
  }
  prepare ();
  if (kind == cudaMemcpyHostToDevice)
  {
    if (verbose)
      cerr << "dsp::TransferCUDA::transformation synchornizing pre H2D transfer" << endl;
    if (stream)
      cudaStreamSynchronize(stream);
    else
      cudaThreadSynchronize();
  }

  if (verbose)
  {
    cerr << "dsp::TransferCUDA::transformation input ndat="
         << input->get_ndat() << " ndim=" << input->get_ndim();
    if (input->get_order() == TimeSeries::OrderFPT)
    {
      if (input->get_npol() > 1)
        cerr << " span=" << input->get_datptr (0,1) - input->get_datptr(0,0);
      cerr << " offset=" << input->get_datptr(0,0) - (float*)input->internal_get_buffer() << endl;
    }
    else
      cerr << endl;
  }

  cudaError error;
  if (stream)
    error = cudaMemcpyAsync (output->internal_get_buffer(),
                             input->internal_get_buffer(),
                             input->internal_get_size(),
                             kind,
                             stream);
  else
    error = cudaMemcpy (output->internal_get_buffer(),
                             input->internal_get_buffer(),
                             input->internal_get_size(), kind);
  if (error != cudaSuccess) {
    if (verbose) {
      std::cerr << "dsp::TransferCUDA::transformation failed with error " << cudaGetErrorString(error) << std::endl;
    }
    throw Error (InvalidState, "dsp::TransferCUDA::transformation",
                 cudaGetErrorString (error));
  }

  if (verbose)
  {
    cerr << "dsp::TransferCUDA::transformation output ndat="
       << output->get_ndat() << " ndim=" << output->get_ndim();
    if (output->get_order() == TimeSeries::OrderFPT)
    {
      if (output->get_npol() > 1)
        cerr << " span=" << output->get_datptr (0, 1) - output->get_datptr(0,0);
      cerr << " offset=" << output->get_datptr(0,0) - (float*)output->internal_get_buffer() << endl;
    }
    else
      cerr << endl;
  }

  if (kind == cudaMemcpyDeviceToHost)
  {
    if (verbose)
      cerr << "dsp::TransferCUDA::transformation synchornizing post D2H transfer" << endl;
    if (stream)
      cudaStreamSynchronize(stream);
    else
      cudaThreadSynchronize();
  }
}

void dsp::TransferCUDA::prepare ()
{
  if (verbose) {
    std::cerr << "dsp::TransferCUDA::prepare" << std::endl;
  }
  output->set_match( const_cast<TimeSeries*>(input.get()) );
  output->internal_match( input );
  output->copy_configuration( input );
}
