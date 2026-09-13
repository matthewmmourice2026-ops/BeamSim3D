cmake_minimum_required(VERSION 3.15)
project(BeamSim3D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
add_executable(beam_sim main.cpp Physics.cpp)
target_include_directories(beam_sim PRIVATE ${CMAKE_SOURCE_DIR})

# Headless batch data-gen tool, no raylib window needed
add_executable(throw_sim ThrowSim.cpp)

# OS-Specific Linking
if (APPLE)
    # Apple Silicon Homebrew paths
    include_directories(/opt/homebrew/include)

    # macOS requires specific frameworks for windowing and graphics
    find_library(RAYLIB_LIBRARY raylib PATHS /opt/homebrew/lib REQUIRED)
    find_library(COCOA_LIBRARY Cocoa)
    find_library(IOKIT_LIBRARY IOKit)
    find_library(COREVIDEO_LIBRARY CoreVideo)
    find_library(OPENGL_LIBRARY OpenGL)

    target_link_libraries(beam_sim PRIVATE ${RAYLIB_LIBRARY} ${COCOA_LIBRARY} ${IOKIT_LIBRARY} ${COREVIDEO_LIBRARY} ${OPENGL_LIBRARY})
    
elseif (WIN32)
    # Your existing Windows paths
    include_directories("C:/raylib/raylib/src")
    link_directories("C:/raylib/raylib/src")
    target_link_libraries(beam_sim PRIVATE raylib opengl32 gdi32 winmm)
endif()