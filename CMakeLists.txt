
# DO NOT set minimum version to 3.27 and above
# https://forums.developer.nvidia.com/t/cmake-error-opencv-cuda-cmakelists-txt/260691
cmake_minimum_required(VERSION 3.26)

project(YoloOpencvPoseVideoInference)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(main
    src/main.cpp
    src/pose.hpp
    src/posenet.hpp
    src/posenet.cpp
    src/imgprocess.hpp
    src/imgprocess.cpp
)

target_link_libraries(main PRIVATE
    opencv_core opencv_dnn opencv_imgproc opencv_videoio
    # opencv_world ocv.3rdparty.v4l ocv.3rdparty.gtk3 ocv.3rdparty.ffmpeg
    Boost::program_options
)
