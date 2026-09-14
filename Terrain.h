#ifndef TERRAIN_H
#define TERRAIN_H

#include <cmath>

// Shared by ThrowSim.cpp (physics) and ThrowGame.cpp (rendering), so the
// visual terrain and the actual simulated terrain can never drift apart.
inline float terrainHeight(float x) {
    return 2.0f * std::sin(x * 0.05f) + 0.5f * std::sin(x * 0.13f);
}

#endif // TERRAIN_H
