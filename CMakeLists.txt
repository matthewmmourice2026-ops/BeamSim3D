cmake_minimum_required(VERSION 3.15)
project(BeamSim3D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories("C:/raylib/raylib/src")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Link directories
link_directories("C:/raylib/raylib/src")

# Add executable
add_executable(beam_sim main.cpp Physics.cpp)

# Link libraries
target_link_libraries(beam_sim raylib opengl32 gdi32 winmm)

# Copy assets to the build directory
file(COPY assets/vehicle.json DESTINATION ${CMAKE_BINARY_DIR})
file(COPY assets/terrain.png DESTINATION ${CMAKE_BINARY_DIR})
file(COPY assets/engine.wav DESTINATION ${CMAKE_BINARY_DIR})
