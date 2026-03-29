# Source files
add_executable(video_analytics
    main.cpp
    capture/opencv-capture.cpp
)

# Link dependencies
target_link_libraries(video_analytics
    PRIVATE
        ${OpenCV_LIBS}
        ONNXRuntime::ONNXRuntime
        nlohmann_json::nlohmann_json
        pthread
)

# Include directories — src/ as root so includes like "capture/xxx.hpp" work
target_include_directories(video_analytics
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)
