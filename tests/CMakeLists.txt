# Test executable for inference engine
add_executable(test_inference
    test-inference.cpp
    ../src/inference/inference-engine.cpp
    ../src/inference/pre-process.cpp
    ../src/inference/post-process.cpp
)

# Link dependencies
target_link_libraries(test_inference
    PRIVATE
        ${OpenCV_LIBS}
        ONNXRuntime::ONNXRuntime
        pthread
)

# Include directories
# src/ as root so includes like "inference/xxx.hpp" work
target_include_directories(test_inference
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${OpenCV_INCLUDE_DIRS}
)

# Test executable for pipeline (BoundedQueue, FrameData, AnalyticsResult)
add_executable(test_pipeline
    test-pipeline.cpp
)

# Link dependencies
target_link_libraries(test_pipeline
    PRIVATE
        ${OpenCV_LIBS}
        pthread
)

# Include directories
target_include_directories(test_pipeline
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${OpenCV_INCLUDE_DIRS}
)
