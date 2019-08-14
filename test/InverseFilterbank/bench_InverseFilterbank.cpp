#include <complex>
#include <vector>
#include <iostream>

#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/InverseFilterbankEngineCUDA.h"
#include "dsp/InverseFilterbank.h"
#include "dsp/TransferCUDA.h"
#include "dsp/MemoryCUDA.h"
#include "Rational.h"
#include "CommandLine.h"

#include "util.hpp"
#include "InverseFilterbankTestConfig.hpp"

class BenchmarkOptions {
public:
  unsigned niter = 10;

  // npart, input_nchan, output_nchan, input_npol, output_npol, input_ndat, output_ndat, overlap_pos, overlap_neg
  util::TestShape test_shape{10, 256, 1, 2, 2, 128, 0, 16, 16};
};

void check_error (const char*);

void parse_options (BenchmarkOptions& options, int argc, char** argv)
{
  CommandLine::Menu menu;
  CommandLine::Argument* arg;

  menu.set_help_header ("benchmark InverseFilterbank CUDA and CPU engines");

  arg = menu.add (options.niter, "n", "niter");
  arg->set_help ("Specify the number of iterations to run [default:10]");

  arg = menu.add (options.test_shape.input_npol, "npol");
  arg->set_help ("Specify the number of input polarisations [default:2]");

  arg = menu.add (options.test_shape.npart, "npart");
  arg->set_help ("Specify the number of parts [default:10]");

  arg = menu.add (options.test_shape.input_nchan, "input_nchan");
  arg->set_help ("Specify the number of input channels [default:256]");

  arg = menu.add (options.test_shape.output_nchan, "output_nchan");
  arg->set_help ("Specify the number of output channels [default:1]");

  arg = menu.add (options.test_shape.input_ndat, "input_ndat");
  arg->set_help ("Specify the size of the forward FFT [default:128]");

  arg = menu.add (options.test_shape.overlap_pos, "overlap");
  arg->set_help ("Specify the size of the input overlap region [default:16]");

  menu.parse (argc, argv);

  if (options.test_shape.overlap_pos > options.test_shape.input_ndat/2)
  {
    throw "Overlap region cannot be larger than input FFT";
  }
  if (options.test_shape.output_nchan >= options.test_shape.input_nchan)
  {
    throw "Number of output channels cannot be greater than input channels";
  }

}

int main (int argc, char** argv)
{
  BenchmarkOptions options;

  parse_options (options, argc, argv);

  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory* device_memory = new CUDA::DeviceMemory(cuda_stream);

  CUDA::InverseFilterbankEngineCUDA engine_cuda(cuda_stream);
  dsp::InverseFilterbankEngineCPU engine_cpu;

  Reference::To<dsp::TimeSeries> in = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> in_gpu = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> out_gpu = new dsp::TimeSeries;

  Rational os_factor (4, 3);
  unsigned npart = options.test_shape.npart;
  unsigned npol = options.test_shape.input_npol;

  util::InverseFilterbank::InverseFilterbankProxy proxy(
    os_factor, npart, npol,
    options.test_shape.input_nchan, options.test_shape.output_nchan,
    options.test_shape.input_ndat, options.test_shape.overlap_pos
  );

  // config.filterbank->set_device(device_memory);
  proxy.filterbank->set_pfb_dc_chan(true);
  proxy.filterbank->set_pfb_all_chan(true);

  proxy.setup(in, out, false, false);

  engine_cpu.setup(proxy.filterbank);
  float* scratch_cpu = proxy.allocate_scratch(engine_cpu.get_total_scratch_needed());
  engine_cpu.set_scratch(scratch_cpu);


  auto t0 = util::now();

  for (unsigned iiter=0; iiter<options.niter; iiter++) {
    engine_cpu.perform(
      in, out, npart
    );
    engine_cpu.finish();
  }

  util::delta<std::milli> delta_cpu = util::now() - t0;

  std::cerr << "CPU engine " << delta_cpu.count() / options.niter << " ms per loop"
    << std::endl;

  auto transfer = util::transferTimeSeries(cuda_stream, device_memory);

  transfer(in, in_gpu, cudaMemcpyHostToDevice);
  transfer(out, out_gpu, cudaMemcpyHostToDevice);

  engine_cuda.setup(proxy.filterbank);
  proxy.set_memory<CUDA::DeviceMemory>(device_memory);
  float* scratch_cuda = proxy.allocate_scratch(engine_cuda.get_total_scratch_needed());
  engine_cuda.set_scratch(scratch_cuda);

  t0 = util::now();

  for (unsigned iiter=0; iiter<options.niter; iiter++) {
    engine_cuda.perform(
      in_gpu, out_gpu, npart
    );
    engine_cuda.finish();
  }

  util::delta<std::milli> delta_cuda = util::now() - t0;

  std::cerr << "CUDA engine " << delta_cuda.count() / options.niter << " ms per loop"
    << std::endl;

  if (delta_cpu > delta_cuda) {
    std::cerr << "CUDA engine " << delta_cpu.count() / delta_cuda.count()
      << " times faster" << std::endl;
  } else {
    std::cerr << "CPU engine " << delta_cuda.count() / delta_cpu.count()
      << " times faster" << std::endl;
  }
}
