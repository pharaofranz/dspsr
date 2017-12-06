#nobase_include_HEADERS = dsp/ACFilterbank.h dsp/Detection.h        \
#						 dsp/ResponseProduct.h dsp/Accumulator.h                \
#						 dsp/Apodization.h dsp/SampleDelay.h dsp/AutoCorrelation.h      \
#						 dsp/SampleDelayFunction.h dsp/Example.h dsp/Shape.h        \
#						 dsp/Bandpass.h dsp/Simultaneous.h dsp/Stats.h dsp/Filterbank.h \
#						 dsp/BitStatsPlotter.h dsp/Switcher.h dsp/Buffer.h          \
#						 dsp/IncoherentFilterbank.h dsp/TScrunch.h dsp/LevelHistory.h   \
#						 dsp/TimeOrder.h dsp/Chomper.h dsp/LevelMonitor.h           \
#						 dsp/ExcisionHistoryPlotter.h dsp/ExcisionStatsPlotter.h        \
#						 dsp/Convolution.h dsp/Dedispersion.h dsp/DedispersionHistory.h \
#						 dsp/RFIFilter.h dsp/DedispersionSampleDelay.h dsp/Response.h   \
#						 dsp/Rescale.h dsp/BandpassMonitor.h dsp/PScrunch.h         \
#						 dsp/FourthMoment.h dsp/PolnCalibration.h dsp/Dump.h        \
#						 dsp/OptimalFFT.h dsp/OptimalFilterbank.h dsp/on_host.h         \
#						 dsp/FScrunch.h dsp/FilterbankBench.h dsp/FilterbankConfig.h    \
#						 dsp/FilterbankEngine.h dsp/filterbank_engine.h dsp/FZoom.h  \
#						 dsp/GeometricDelay.h                           \
#						 dsp/TFPFilterbank.h dsp/RFIZapper.h dsp/SKFilterbank.h         \
#						 dsp/Resize.h dsp/SKDetector.h dsp/SKMasker.h               \
#						 dsp/Pipeline.h dsp/SingleThread.h dsp/MultiThread.h            \
#						 dsp/PolnSelect.h dsp/PolnReshape.h dsp/SpectralKurtosis.h \
#						 dsp/SKComputer.h dsp/FilterbankCPU.hpp

#nobase_include_HEADERS += CUFFTError.h dsp/LaunchConfig.h \
#						  dsp/FilterbankCUDA.h dsp/filterbank_cuda.h \
#						  dsp/TransferCUDA.h dsp/TransferBitSeriesCUDA.h \
#						  dsp/SKMaskerCUDA.h dsp/DetectionCUDA.h dsp/FZoomCUDA.h \
#						  dsp/FScrunchCUDA.h dsp/TScrunchCUDA.h \
#						  dsp/PScrunchCUDA.h dsp/PolnSelectCUDA.h \                                               
#						  dsp/ConvolutionCUDA.h dsp/ConvolutionCUDASpectral.h \
#						  dsp/ConvolutionCUDACallbacks.h dsp/SpectralKurtosisCUDA.h \
#						  dsp/SKComputerCUDA.h dsp/SKDetectorCUDA.h dsp/SKFilterbankCUDA.h

check_PROGRAMS = sample1_gtest #unittest

#unittest_SOURCES = unit_test.cc gtest_main.cc
#unittest_LDFLAGS = -lgtest -pthread
sample1_gtest_SOURCES = sample1_unittest.cc sample1.cc gtest_main.cc
sample1_gtest_LDFLAGS = -lgtest -pthread

include $(top_srcdir)/config/Makefile.include 
