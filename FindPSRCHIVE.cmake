execute_process(COMMAND psrchive --prefix OUTPUT_VARIABLE return_val)
string(STRIP ${return_val} psrchive_prefix)

execute_process(COMMAND psrchive --version OUTPUT_VARIABLE return_val)
string(STRIP ${return_val} psrchive_version)


find_path(PSRCHIVE_INCLUDE_DIR
  Error.h
  HINTS ${psrchive_prefix}/include
)

find_library(PSRCHIVE_LIBRARY
  NAMES psrbase
  HINTS ${psrchive_prefix}/lib
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(PSRCHIVE
  FOUND_VAR PSRCHIVE_FOUND
  REQUIRED_VARS
    PSRCHIVE_LIBRARY
    PSRCHIVE_INCLUDE_DIR
  VERSION_VAR PSRCHIVE_VERSION
)

if(PSRCHIVE_FOUND)
  set(PSRCHIVE_LIBRARIES ${PSRCHIVE_LIBRARY})
  set(PSRCHIVE_INCLUDE_DIRS ${PSRCHIVE_INCLUDE_DIR})
  # set(PSRCHIVE_DEFINITIONS ${PC_PSRCHIVE_CFLAGS_OTHER})
endif()

# message(STATUS "PSRCHIVE_INCLUDE_DIR=${PSRCHIVE_INCLUDE_DIR}")
# message(STATUS "PSRCHIVE_LIBRARY=${PSRCHIVE_LIBRARY}")
