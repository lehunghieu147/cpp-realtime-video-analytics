# FindONNXRuntime.cmake
# Locates ONNX Runtime library and headers
#
# Usage:
#   find_package(ONNXRuntime REQUIRED)
#   target_link_libraries(myapp PRIVATE ONNXRuntime::ONNXRuntime)
#
# Set ONNXRUNTIME_ROOT env variable or cmake variable to help find it.

# Search paths
set(_SEARCH_PATHS
    ${ONNXRUNTIME_ROOT}
    $ENV{ONNXRUNTIME_ROOT}
    /usr/local
    /usr
)

# Find header
find_path(ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATH_SUFFIXES include include/onnxruntime
    PATHS ${_SEARCH_PATHS}
)

# Find library
find_library(ONNXRuntime_LIBRARY
    NAMES onnxruntime
    PATH_SUFFIXES lib lib64
    PATHS ${_SEARCH_PATHS}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR
)

# Create imported target
if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::ONNXRuntime)
    add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}"
    )
endif()
