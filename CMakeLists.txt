cmake_minimum_required(VERSION 3.1)

project(dspsr VERSION "2.0.0" LANGUAGES CXX C)
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(HAVE_CUDA 1)
else()
  set(HAVE_CUDA 0)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})

find_package(MPI)
find_package(PSRCHIVE)
find_package(Epsic)

# if (MPI_FOUND)
#   set(HAVE_MPI 1)
# else()
#   set(HAVE_MPI 0)
# endif()

configure_file(config.h.in config.h)

add_subdirectory(Kernel)
# add_subdirectory(Signal)
# add_subdirectory(Management)
