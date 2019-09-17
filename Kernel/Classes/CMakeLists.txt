set(classes_sources
  ascii_header.c Rational.C ASCIIObservation.C
  InputBufferingShare.C Reserve.C
  BitSeries.C SubByteTwoBitCorrection.C
  DADAFile.C DummyFile.C TestInput.C BitTable.C BitUnpacker.C
  BlockFile.C
  TimeSeries.C ChannelOrder.C DataSeries.C
  TwoBitCorrection.C Digitizer.C MultiFile.C TwoBitTable.C
  EightBitUnpacker.C Unpacker.C File.C Observation.C
  Unpacker_create.C Operation.C WeightedTimeSeries.C
  FourBitUnpacker.C dsp.C HistUnpacker.C
  dspExtension.C IOManager.C Scratch.C Input.C Seekable.C
  InputBuffering.C ExcisionUnpacker.C
  TwoBitLookup.C TwoBitFour.C TwoBit1or2.C NLowLookup.C
  UnpackerIterator.C ObservationChange.C PrestoObservation.C
  CloneArchive.C SignalPath.C Multiplex.C Memory.C
  OperationThread.C FloatUnpacker.C OutputFile.C
  ObservationInterface.C
  GenericEightBitUnpacker.C
  GenericFourBitUnpacker.C
  GenericTwoBitCorrection.C
  CommandLineHeader.C OutputFileShare.C
  FIRFilter.C
)

set(include_dirs ${PSRCHIVE_INCLUDE_DIRS} ${Epsic_INCLUDE_DIRS})

if(HAVE_MPI)
  message(STATUS "Kernel/Classes: adding MPI sources")
  set(classes_sources
    ${classes_sources}
    MPIRoot.C MPITrans.C MPIServer.C mpi_Observation.C
  )
  set(include_dirs
    ${include_dirs}
    ${MPI_CXX_INCLUDE_DIRS}
  )
endif()

if(HAVE_CUDA)
  message(STATUS "Kernel/Classes: adding CUDA sources")
  set(classes_sources
    ${classes_sources}
    MemoryCUDA.C check_error.C GenericEightBitUnpackerCUDA.cu TimeSeriesCUDA.cu
  )
endif()

set(test_BlockIterator_sources
  test_BlockIterator.C
)

set(test_environ_sources
  test_environ.c
)


# the following will allow us to get the config.h file
include_directories(${CMAKE_BINARY_DIR})

add_library(Classes SHARED ${classes_sources})
target_include_directories(Classes PRIVATE . ${include_dirs})
target_link_libraries(Classes PRIVATE ${CUDA_LIBRARIES})

add_executable(test_BlockIterator ${test_BlockIterator_sources})
target_include_directories(test_BlockIterator PRIVATE . ${include_dirs})

add_executable(test_environ ${test_environ_sources})
target_include_directories(test_environ PRIVATE .)
