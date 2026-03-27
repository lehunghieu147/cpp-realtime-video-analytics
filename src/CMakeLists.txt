# Main executable
add_executable(video_analytics main.cpp)

# Link dependencies
target_link_libraries(video_analytics
    PRIVATE
        ${OpenCV_LIBS}
        ONNXRuntime::ONNXRuntime
        nlohmann_json::nlohmann_json
        pthread
)

# Include directories
target_include_directories(video_analytics
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)
