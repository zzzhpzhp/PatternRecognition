cmake_minimum_required(VERSION 3.5)

project(fit_curve)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(fit_curve main.cpp)
target_link_libraries(fit_curve ${OpenCV_LIBS})