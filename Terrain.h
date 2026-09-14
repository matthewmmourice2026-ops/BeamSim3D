#ifndef TERRAIN_H
#define TERRAIN_H

#include <cmath>

// Shared by ThrowSim.cpp (physics) and ThrowGame.cpp (rendering), so the
// visual terrain and the actual simulated terrain can never drift apart.
inline float terrainHeight(float x) {
    return 2.0f * std::sin(x * 0.05f) + 0.5f * std::sin(x * 0.13f);
}

// d/dx of terrainHeight, used to build correct terrain surface normals
// for lighting instead of guessing them.
inline float terrainSlope(float x) {
    return 2.0f * 0.05f * std::cos(x * 0.05f) + 0.5f * 0.13f * std::cos(x * 0.13f);
}

#endif // TERRAIN_H
