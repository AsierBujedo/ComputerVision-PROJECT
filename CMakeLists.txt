cmake_minimum_required(VERSION 3.10)

# Set the project name
project(ComputerVisionProject VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Glob all source files in the src directory
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Add the executable
add_executable(ComputerVisionProject ${SOURCES})

# Find OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(ComputerVisionProject ${OpenCV_LIBS})