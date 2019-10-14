#include "dsp/ScalarAdd.h"

dsp::ScalarAdd::ScalarAdd () : Transformation<TimeSeries,TimeSeries> ("ScalarAdd", outofplace)
{
  scalar = 0.0; // initialize scalar to something
}

dsp::ScalarAdd::~ScalarAdd () {}

void dsp::ScalarAdd::prepare ()
{
  output->copy_configuration (input);
  output->set_state(input->get_state());
  output->set_npol(input->get_npol());
  output->set_nchan(input->get_nchan());
  output->set_ndim(input->get_ndim());
  output->set_ndat(input->get_ndat());
  output->resize(input->get_ndat());
}


void dsp::ScalarAdd::transformation ()
{
  if (! prepared) {
    prepare ();
  }
  if (input->get_order() == dsp::TimeSeries::OrderFPT) {

    float* out_ptr;

    unsigned ndim = input->get_ndim();

    for (unsigned ichan=0; ichan<input->get_nchan(); ichan++) {
      for (unsigned ipol=0; ipol<input->get_npol(); ipol++) {
        const float* in_ptr = input->get_datptr(ichan, ipol);
        out_ptr = output->get_datptr(ichan, ipol);
        for (unsigned idat=0; idat<input->get_ndat(); idat++) {
          for (unsigned idim=0; idim<ndim; idim++) {
            out_ptr[idat*ndim + idim] = scalar + in_ptr[idat*ndim + idim];
          }
        }
      }
    }

  } else {
    throw Error (
      InvalidState,
      "dsp::ScalarAdd::transformation",
      "OrderTFP is not supported"
    );
  }
}
