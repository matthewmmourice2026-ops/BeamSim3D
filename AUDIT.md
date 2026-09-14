# BeamSim3D Project Audit

## Project Overview
BeamSim3D is a 3D physics simulation engine designed to model soft-body dynamics, vehicle mechanics (engines, transmissions, differentials), and terrain interaction using the raylib library. The project focuses on structural integrity through beam systems and realistic vehicle behavior for simulations involving deformable bodies or complex mechanical structures.

## Core File Summary

### `CMakeLists.txt`
**Role**: Build System Configuration.
- Handles cross-platform compilation settings.
- Includes specific logic for macOS (Apple Silicon) to link necessary frameworks: Cocoa, IOKit, CoreVideo, and OpenGL via Homebrew paths (`/opt/homebrew`).
- Manages dependencies like `raylib`.

### `main.cpp`
**Role**: Application Entry Point & Rendering Loop.
- Initializes the windowing system, audio device, and rendering context using raylib.
- Handles high-level asset loading (terrain textures, engine sounds).
- Contains the main simulation loop where physics updates are called every frame based on `GetFrameTime()`.
- Manages camera logic and renders 3D primitives representing nodes, beams, and wheels.

### `Physics.h`
**Role**: Data Structures & Class Definitions.
- **Node3D**: Represents individual points in space with mass, velocity, position, and angular properties.
- **Beam3D**: Defines structural connections between nodes (stiffness, damping, break thresholds).
- **Vehicle Components**: Structs for `Wheel`, `Engine` (RPM/Torque curves), `Transmission` (Gear ratios), and `Differential`.
- **SoftBody Class**: The primary container managing the collection of all physical components.

### `Physics.cpp`
**Role**: Physics Engine Implementation.
- Implements logic for loading vehicle configurations from JSON files.
- Handles center of mass calculations.
- Contains methods for gravity, ground collisions, traction, and powertrain updates (engine/transmission).

### `vehicle.json`
**Role**: Data Configuration.
- Defines the specific layout of vehicles without changing code.
- Includes node coordinates, beam connections, wheel placement, and powertrain specifications (idle RPM, gear ratios, etc.).

## Recent Refactors & Compatibility
- **Raylib 5.0 Update**: The camera system was refactored to remove deprecated `SetCameraMode` calls. It now uses explicit projection settings (`CAMERA_PERSPECTIVE`) and the updated `UpdateCamera(&camera, CAMERA_CUSTOM)` call for modern raylib compatibility.
- **Audio Lifecycle Management**: Added proper initialization (`InitAudioDevice`) and cleanup (`CloseAudioDevice`) to ensure stable audio handling across different platforms.
- **Render Pipeline Correction**: Fixed a rendering bug where 2D terrain textures were being incorrectly projected into the 3D camera matrix; they are now drawn before entering `BeginMode3D`.
- **Physics Loop Integration**: Integrated `GetFrameTime()` to provide consistent, frame-rate independent physics updates via `vehicle.update(deltaTime)`.
- **CMake Target Linking**: Refactored macOS build logic to use target-specific include and link directories (`target_include_directories`, `target_link_directories`) for better modularity and added explicit OpenGL library discovery.

## Build & Execution
- **Requirements**: Raylib library must be installed on the host system.
- **macOS Users**: Ensure Homebrew is used as it maps to `/opt/homebrew` in the CMake configuration.
- **Build Process**: Standard CMake workflow (Configure -> Build).
