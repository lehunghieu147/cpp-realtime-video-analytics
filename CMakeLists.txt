cmake_minimum_required(VERSION 3.20)
project(real_time_video_analytics LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings — catch bugs early
add_compile_options(-Wall -Wextra -Wpedantic)

# --- Dependencies ---

# nlohmann/json (header-only JSON library)
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(json)

# OpenCV — installed via system package manager (apt install libopencv-dev)
find_package(OpenCV REQUIRED)

# ONNX Runtime — found via custom cmake module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(ONNXRuntime REQUIRED)

# --- Subdirectories ---
add_subdirectory(src)
add_subdirectory(tests)
