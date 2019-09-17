find_path(Epsic_INCLUDE_DIR
  Jacobi.h
  HINTS ${EPSIC_INSTALL_DIR}/include/epsic
)

find_library(Epsic_LIBRARY
  NAMES epsic
  HINTS ${EPSIC_INSTALL_DIR}/lib
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Epsic
  FOUND_VAR Epsic_FOUND
  REQUIRED_VARS
    Epsic_LIBRARY
    Epsic_INCLUDE_DIR
  # VERSION_VAR Epsic_VERSION
)

if(Epsic_FOUND)
  set(Epsic_LIBRARIES ${Epsic_LIBRARY})
  set(Epsic_INCLUDE_DIRS ${Epsic_INCLUDE_DIR})
  # set(Epsic_DEFINITIONS ${PC_Epsic_CFLAGS_OTHER})
endif()

# message(STATUS "Epsic_INCLUDE_DIR=${Epsic_INCLUDE_DIR}")
# message(STATUS "Epsic_LIBRARY=${PSRCHIVE_LIBRARY}")
