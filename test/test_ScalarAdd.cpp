#include "catch.hpp"

#include "dsp/ScalarAdd.h"

#include "util/util.hpp"
#include "util/TestConfig.hpp"


TEST_CASE("Can get and set ScalarAdd::scalar", "[ScalarAdd]")
{
  dsp::ScalarAdd adder;

  float expected_value = 5.0;

  CHECK(adder.get_scalar() == 0); // from constructor

  adder.set_scalar(expected_value);

  REQUIRE(adder.get_scalar() == expected_value);

}

TEST_CASE ("Can operate on data", "[ScalarAdd]")
{

  dsp::ScalarAdd adder;

  Reference::To<dsp::TimeSeries> in = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out = new dsp::TimeSeries;

  in->set_state(Signal::Nyquist);
  in->set_nchan (16);
  in->set_npol (2);
  in->set_ndat (100);
  in->set_ndim (1);
  in->resize (100);

  float* in_ptr;

  for (unsigned ichan = 0; ichan < in->get_nchan(); ichan++) {
    for (unsigned ipol = 0; ipol < in->get_npol(); ipol++) {
      in_ptr = in->get_datptr(ichan, ipol);
      for (unsigned idat = 0; idat < in->get_ndat(); idat++) {
        in_ptr[idat] = 1.0;
      }
    }
  }

  float scalar = 4.0;

  adder.set_input(in);
  adder.set_output(out);
  adder.set_scalar(scalar);

  adder.prepare();

  adder.operate();

  bool allclose = true;
  unsigned nclose = 0;
  unsigned total_size = out->get_ndat() * out->get_nchan() * out->get_ndim() * out->get_npol();
  float* out_ptr;

  for (unsigned ichan = 0; ichan < out->get_nchan(); ichan++) {
    for (unsigned ipol = 0; ipol < out->get_npol(); ipol++) {
      out_ptr = out->get_datptr(ichan, ipol);
      for (unsigned idat = 0; idat < out->get_ndat(); idat++) {
        if (out_ptr[idat] != (1.0 + scalar)) {
          allclose = false;
        } else {
          nclose++;
        }
      }
    }
  }

  std::cerr << nclose << "/" << total_size
    << "(" << 100 * (float) nclose / total_size << "%)" << std::endl;

  REQUIRE(allclose == true);

}


TEST_CASE ("Can operate on data from file", "[ScalarAdd]")
{
  test::util::TestConfig test_config;

  std::string file_name = test_config.get_field<std::string>("ScalarAdd.file_name");

  const std::string file_path = test::util::get_test_data_dir() + "/" + file_name;

  dsp::ScalarAdd adder;

  Reference::To<dsp::TimeSeries> in = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out = new dsp::TimeSeries;

  dsp::IOManager manager;

  manager.set_output(in);

  manager.open(file_path);
  manager.set_block_size(1024);

  while (! manager.get_input()->eod()) {
    manager.operate();
  }


  float scalar = 4.0;

  adder.set_input(in);
  adder.set_output(out);
  adder.set_scalar(scalar);

  adder.prepare();

  adder.operate();

  bool allclose = true;
  unsigned nclose = 0;
  unsigned total_size = out->get_ndat() * out->get_nchan() * out->get_ndim() * out->get_npol();
  float* out_ptr;

  for (unsigned ichan = 0; ichan < out->get_nchan(); ichan++) {
    for (unsigned ipol = 0; ipol < out->get_npol(); ipol++) {
      out_ptr = out->get_datptr(ichan, ipol);
      for (unsigned idat = 0; idat < out->get_ndat(); idat++) {
        if (out_ptr[idat] != (1.0 + scalar)) {
          allclose = false;
        } else {
          nclose++;
        }
      }
    }
  }

  std::cerr << nclose << "/" << total_size
    << "(" << 100 * (float) nclose / total_size << "%)" << std::endl;

  REQUIRE(allclose == true);

}
