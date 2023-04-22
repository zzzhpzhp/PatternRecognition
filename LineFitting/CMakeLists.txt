cmake_minimum_required(VERSION 3.5)

project(line_fitting)

set(CMAKE_CXX_STANDARD 11)

# Find OpenCV library
find_package(OpenCV REQUIRED)

# Include directories and libraries
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

# Add executable
add_executable(${PROJECT_NAME} main.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
