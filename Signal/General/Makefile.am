
lib_LTLIBRARIES = libdspdsp.la

nobase_include_HEADERS = dsp/ACFilterbank.h dsp/Detection.h	       \
	dsp/ResponseProduct.h dsp/Accumulator.h			       \
	dsp/Apodization.h dsp/SampleDelay.h dsp/AutoCorrelation.h      \
	dsp/SampleDelayFunction.h dsp/Example.h dsp/Shape.h	       \
	dsp/Bandpass.h dsp/Simultaneous.h dsp/Stats.h dsp/Filterbank.h \
	dsp/BitStatsPlotter.h dsp/Switcher.h dsp/Buffer.h	       \
	dsp/IncoherentFilterbank.h dsp/TScrunch.h dsp/LevelHistory.h   \
	dsp/TimeOrder.h dsp/Chomper.h dsp/LevelMonitor.h	       \
	dsp/ExcisionHistoryPlotter.h dsp/ExcisionStatsPlotter.h	       \
	dsp/Convolution.h dsp/Dedispersion.h dsp/DedispersionHistory.h \
	dsp/RFIFilter.h dsp/DedispersionSampleDelay.h dsp/Response.h   \
	dsp/Rescale.h dsp/BandpassMonitor.h dsp/PScrunch.h	       \
	dsp/FourthMoment.h dsp/PolnCalibration.h dsp/Dump.h	       \
	dsp/OptimalFFT.h dsp/OptimalFilterbank.h dsp/on_host.h	       \
	dsp/FScrunch.h dsp/FilterbankBench.h dsp/FilterbankConfig.h    \
	dsp/FilterbankEngine.h dsp/FilterbankEngineCPU.h  \
	dsp/filterbank_engine.h dsp/FZoom.h  \
	dsp/GeometricDelay.h \
	dsp/DelayStartTime.h \
	dsp/TFPFilterbank.h dsp/RFIZapper.h dsp/SKFilterbank.h	       \
	dsp/Resize.h dsp/SKDetector.h dsp/SKMasker.h		       \
	dsp/Pipeline.h dsp/SingleThread.h dsp/MultiThread.h            \
	dsp/PolnSelect.h dsp/PolnReshape.h dsp/SpectralKurtosis.h \
  dsp/SKComputer.h dsp/ScalarFilter.h \
	dsp/InverseFilterbank.h dsp/InverseFilterbankEngine.h \
	dsp/InverseFilterbankConfig.h \
	dsp/InverseFilterbankEngineCPU.h \
	dsp/InverseFilterbankResponse.h

libdspdsp_la_SOURCES = optimize_fft.c cross_detect.c cross_detect.h  \
	cross_detect.ic stokes_detect.c stokes_detect.h		     \
	stokes_detect.ic ACFilterbank.C TScrunch.C      \
	TimeOrder.C Apodization.C AutoCorrelation.C Filterbank.C     \
	FilterbankEngineCPU.C \
	IncoherentFilterbank.C Bandpass.C LevelMonitor.C RFIFilter.C \
	Chomper.C Response.C ResponseProduct.C Convolution.C	     \
	Dedispersion.C SampleDelay.C DelayStartTime.C DedispersionHistory.C	\
	Shape.C DedispersionSampleDelay.C Detection.C Rescale.C	     \
	PScrunch.C BandpassMonitor.C FourthMoment.C Stats.C	     \
	PolnCalibration.C Dump.C OptimalFFT.C FScrunch.C	     \
	FilterbankBench.C OptimalFilterbank.C FilterbankConfig.C \
	FZoom.C GeometricDelay.C mfilter.c \
	TFPFilterbank.C RFIZapper.C SKFilterbank.C \
	Resize.C SKDetector.C SKMasker.C \
	SingleThread.C MultiThread.C dsp_verbosity.C \
	PolnSelect.C PolnReshape.C SpectralKurtosis.C ScalarFilter.C \
	InverseFilterbank.C \
	InverseFilterbankConfig.C \
	InverseFilterbankEngineCPU.C \
	InverseFilterbankResponse.C


bin_PROGRAMS = dmsmear digitxt digimon digihist filterbank_speed
# bin_PROGRAMS += test_InverseFilterbankResponse test_FIRFilter

if HAVE_CUFFT

nobase_include_HEADERS += CUFFTError.h dsp/LaunchConfig.h \
	dsp/DelayStartTimeCUDA.h \
  dsp/FilterbankEngineCUDA.h dsp/filterbank_cuda.h \
  dsp/TransferCUDA.h dsp/TransferBitSeriesCUDA.h \
  dsp/SKMaskerCUDA.h dsp/DetectionCUDA.h dsp/FZoomCUDA.h \
  dsp/FScrunchCUDA.h dsp/TScrunchCUDA.h \
  dsp/PScrunchCUDA.h dsp/PolnSelectCUDA.h \
  dsp/ConvolutionCUDA.h dsp/ConvolutionCUDASpectral.h \
  dsp/ConvolutionCUDACallbacks.h dsp/SpectralKurtosisCUDA.h \
  dsp/SKComputerCUDA.h dsp/SKDetectorCUDA.h dsp/SKFilterbankCUDA.h \
  dsp/SampleDelayCUDA.h dsp/InverseFilterbankEngineCUDA.h

libdspdsp_la_SOURCES += CUFFTError.C LaunchConfig.C FilterbankEngineCUDA.cu \
	DelayStartTimeCUDA.cu \
  TransferCUDA.C TransferBitSeriesCUDA.C DetectionCUDA.cu             \
  SKMaskerCUDA.cu FZoomCUDA.cu FScrunchCUDA.cu TScrunchCUDA.cu        \
  PScrunchCUDA.cu PolnSelectCUDA.cu SpectralKurtosisCUDA.cu \
  SKComputerCUDA.cu SKDetectorCUDA.cu SKFilterbankCUDA.cu \
  ConvolutionCUDA.cu ConvolutionCUDASpectral.cu ConvolutionCUDACallbacks.cu \
  SampleDelayCUDA.cu InverseFilterbankEngineCUDA.cu

bin_PROGRAMS += fftbatch_speed
fftbatch_speed_SOURCES = fftbatch_speed.C

if HAVE_CUFFT_CALLBACKS

bin_PROGRAMS += cufft_callback_bench

cufft_callback_bench_SOURCES = cufft_callback_bench.cu

ConvolutionCUDACallbacks.lo: ConvolutionCUDACallbacks.cu
	$(top_srcdir)/config/cudalt.py $(top_builddir)/libtool $@ $(CUDA_NVCC) -dc -c $<
	$(CUDA_NVCC) -o ConvolutionCUDACallbacks_DC.o -dlink ConvolutionCUDACallbacks.o -lcufft_static

cufft_callback_bench.lo: cufft_callback_bench.cu
	$(top_srcdir)/config/cudalt.py $(top_builddir)/libtool $@ $(CUDA_NVCC) -dc -c $<
	$(CUDA_NVCC) -o cufft_callback_bench_DC.o -dlink cufft_callback_bench.o -lcufft_static

endif
endif


dmsmear_SOURCES = dmsmear.C
digitxt_SOURCES = digitxt.C
digimon_SOURCES = digimon.C
digihist_SOURCES = digihist.C
filterbank_speed_SOURCES = filterbank_speed.C
# test_InverseFilterbankResponse_SOURCES = test_InverseFilterbankResponse.C
# test_FIRFilter_SOURCES = test_FIRFilter.C

check_PROGRAMS = test_PolnCalibration test_OptimalFFT test_Apodization

test_PolnCalibration_SOURCES = test_PolnCalibration.C
test_OptimalFFT_SOURCES = test_OptimalFFT.C
test_Apodization_SOURCES = test_Apodization.C

libdspdsp_la_LIBADD = @YAMLCPP_LIBS@

if HAVE_PGPLOT

  ###########################################################################
  #
  # start PGPLOT-specific code
  #

  libdspdsp_la_SOURCES += ExcisionStatsPlotter.C BitStatsPlotter.C
  libdspdsp_la_LIBADD += @PGPLOT_LIBS@
  bin_PROGRAMS += digistat passband

  digistat_SOURCES = digistat.C
  digistat_LDADD = @PSRPLOT_LIBS@ $(LDADD)

  passband_SOURCES = passband.C
  passband_LDADD = @PSRPLOT_LIBS@ $(LDADD)

  #
  # end PGPLOT-specific code
  #
  ###########################################################################

endif

if HAVE_fits

  ##########################################################################
  #
  # start PSRFITS-specific code
  #

  nobase_include_HEADERS += dsp/LoadToFITS.h dsp/LoadToFITSN.h
  libdspdsp_la_LIBADD += @CFITSIO_LIBS@
  libdspdsp_la_SOURCES += LoadToFITS.C LoadToFITSN.C

  bin_PROGRAMS += digifits
  digifits_SOURCES = digifits.C
  digifits_LDADD = @CFITSIO_LIBS@ $(LDADD)

  #
  # end PSRFITS-specific code
  #
  ##########################################################################

endif

if HAVE_sigproc

  ###########################################################################
  #
  # start sigproc-specific code
  #

  nobase_include_HEADERS += dsp/LoadToFil.h dsp/LoadToFilN.h
  libdspdsp_la_SOURCES += LoadToFil.C LoadToFilN.C

  bin_PROGRAMS += digifil
  digifil_SOURCES = digifil.C


if HAVE_dada
  bin_PROGRAMS += the_decimator
  the_decimator_SOURCES = the_decimator.C
  the_decimator_LDADD = $(LDADD) @OPENSSL_LIBS@ @PSRXML_LIBS@ @PSRDADA_LIBS@
  the_decimator_CPPFLAGS = $(AM_CPPFLAGS) $(CPPFLAGS) @PSRXML_CFLAGS@ @PSRDADA_CFLAGS@
endif

  #
  # end sigproc-specific code
  #
  ###########################################################################

endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

LDADD = libdspdsp.la \
	$(top_builddir)/Kernel/libdspbase.la \
	$(top_builddir)/Signal/Statistics/libdspstats.la \
	@PGPLOT_LIBS@ @CUDA_LIBS@ @YAMLCPP_LIBS@

AM_CPPFLAGS += @PGPLOT_CFLAGS@ @CUFFT_CFLAGS@ @CFITSIO_CFLAGS@ @YAMLCPP_CFLAGS@
AM_CXXFLAGS = @OPENMP_CFLAGS@

if HAVE_CUFFT
if HAVE_CUFFT_CALLBACKS
LDADD += ConvolutionCUDACallbacks_DC.o @CUFFT_CALLBACKS_LIBS@
else
LDADD += @CUFFT_LIBS@
endif
endif

libs: install-libLTLIBRARIES

