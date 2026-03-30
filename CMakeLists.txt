cmake_minimum_required(VERSION 3.20)
project(real_time_video_analytics LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings — catch bugs early
add_compile_options(-Wall -Wextra -Wpedantic)

# Optional sanitizers for debugging
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
if(ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "Cannot enable ASAN and TSAN simultaneously")
endif()
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()
if(ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

# --- Dependencies ---

# nlohmann/json (header-only JSON library)
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(json)

# cpp-httplib (header-only HTTP server library)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.7
)
FetchContent_MakeAvailable(httplib)

# OpenCV — installed via system package manager (apt install libopencv-dev)
find_package(OpenCV REQUIRED)

# ONNX Runtime — found via custom cmake module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(ONNXRuntime REQUIRED)

# --- Subdirectories ---
add_subdirectory(src)
add_subdirectory(tests)
