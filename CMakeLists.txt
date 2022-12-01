cmake_minimum_required(VERSION 3.2)
project(FastestDet)

set(CMAKE_CXX_STANDARD 14)

add_executable(FastestDet main.cpp anchor_creator.cpp
        anchor_creator.h
        config.cpp
        config.h
        http.h
        utils.cpp
        utils.h)
find_package(OpenCV REQUIRED)
target_link_libraries(FastestDet ${OpenCV_LIBS})
set(ncnn_DIR "/home/boyd/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)
target_link_libraries(FastestDet ncnn)