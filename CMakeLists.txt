cmake_minimum_required(VERSION 3.10)
project(BeamSim3D)

set(CMAKE_CXX_STANDARD 14)

# Find Raylib
find_package(raylib REQUIRED)

# Include directories
include_directories(${raylib_INCLUDE_DIRS})

# Add executable
add_executable(beam_sim main.cpp Physics.cpp)

# Link libraries
target_link_libraries(beam_sim ${raylib_LIBRARIES})
