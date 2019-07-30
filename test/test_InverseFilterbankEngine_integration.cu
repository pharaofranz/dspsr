#include <vector>

#include "catch.hpp"

#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/InverseFilterbankEngineCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "util.hpp"
#include "InverseFilterbank_test_config.h"


void check_error (const char*);


TEST_CASE (
  "InverseFilterbankEngineCPU and InverseFilterbankEngineCUDA produce same output",
  "[InverseFilterbankEngineCPU]"
)
{
  int idx = 2;
  test_config::TestShape test_shape = test_config::test_shapes[idx];
  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory* device_memory = new CUDA::DeviceMemory(cuda_stream);
  CUDA::InverseFilterbankEngineCUDA engine_cuda(cuda_stream);
  dsp::InverseFilterbankEngineCPU engine_cpu;

  Reference::To<dsp::TimeSeries> in = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> in_gpu = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out_gpu = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out_cuda = new dsp::TimeSeries;

  Rational os_factor (4, 3);
  unsigned npart = test_shape.npart;

  util::IntegrationTestConfiguration<dsp::InverseFilterbank> config(
    os_factor, npart, test_shape.npol,
    test_shape.nchan, test_shape.output_nchan,
    test_shape.ndat, test_shape.overlap
  );
  config.filterbank->set_pfb_dc_chan(true);
  config.filterbank->set_pfb_all_chan(true);

  config.setup (in, out);

  engine_cpu.setup(config.filterbank);
  std::vector<float *> scratch_cpu = config.allocate_scratch<dsp::Memory> ();
  engine_cpu.set_scratch(scratch_cpu[0]);
  engine_cpu.perform(
    in, out, npart
  );
  engine_cpu.finish();

  auto transfer = util::transferTimeSeries(cuda_stream, device_memory);

  transfer(in, in_gpu, cudaMemcpyHostToDevice);
  transfer(out, out_gpu, cudaMemcpyHostToDevice);

  config.filterbank->set_device(device_memory);
  engine_cuda.setup(config.filterbank);
  std::vector<float *> scratch_cuda = config.allocate_scratch<CUDA::DeviceMemory>(device_memory);
  engine_cuda.set_scratch(scratch_cuda[0]);
  engine_cuda.perform(
    in_gpu, out_gpu, npart
  );
  engine_cuda.finish();

  // now lets compare the two time series
  transfer(out_gpu, out_cuda, cudaMemcpyDeviceToHost);
  std::cerr << "out_gpu->get_ndat()=" << out_gpu->get_ndat()  << std::endl;
  std::cerr << "out_cuda->get_ndat()=" << out_cuda->get_ndat() << std::endl;

  REQUIRE(util::allclose(out_cuda, out, test_config::thresh));
}
