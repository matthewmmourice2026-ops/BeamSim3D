#ifndef TRACK_H
#define TRACK_H

#include <raylib.h>
#include <raymath.h>
#include <vector>
#include <cmath>
#include <cstdio>

// Closed-loop race track: a Catmull-Rom spline through a handful of control
// points, evaluated ONCE at startup into an evenly-spaced arc-length table
// (center/leftEdge/rightEdge). Every other query (nearest point, raycasts,
// mesh) reads from that table - nothing here is recomputed per frame except
// small windowed searches. Flat (constant Y) by design - this is a ground
// track for a simple kinematic car model, not terrain with elevation.
struct Track {
    static constexpr float halfWidth = 6.0f;
    static constexpr float ds = 1.0f; // arc-length spacing of the sample table
    static constexpr float trackY = 0.0f;

    std::vector<Vector3> center, leftEdge, rightEdge;
    float totalLength = 0.0f;
    Model mesh{};

    void build(const std::vector<Vector2>& controlPoints);
    int wrapIndex(int i) const {
        int n = (int)center.size();
        int r = i % n;
        return r < 0 ? r + n : r;
    }
    float nearestArcLength(Vector3 pos, float hintArcLength, float* outLateralDist) const;
    float raycastEdgeDistance(Vector3 origin, Vector2 dirXZ, float hintArcLength, float maxRange) const;
    Vector3 pointAt(float s) const { return center[wrapIndex((int)(s / ds))]; }
    Vector3 tangentAt(float s) const;

private:
    static Vector2 catmullRom(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float t);
};

inline Vector2 Track::catmullRom(Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, float t) {
    float t2 = t * t, t3 = t2 * t;
    Vector2 r;
    r.x = 0.5f * ((2.0f * p1.x) + (-p0.x + p2.x) * t + (2.0f * p0.x - 5.0f * p1.x + 4.0f * p2.x - p3.x) * t2 +
                  (-p0.x + 3.0f * p1.x - 3.0f * p2.x + p3.x) * t3);
    r.y = 0.5f * ((2.0f * p1.y) + (-p0.y + p2.y) * t + (2.0f * p0.y - 5.0f * p1.y + 4.0f * p2.y - p3.y) * t2 +
                  (-p0.y + 3.0f * p1.y - 3.0f * p2.y + p3.y) * t3);
    return r;
}

inline void Track::build(const std::vector<Vector2>& cp) {
    int n = (int)cp.size();

    // 1. Densely sample the closed spline (uniform Catmull-Rom, wrapped
    // neighbor indices) and walk it accumulating raw chord length.
    const int samplesPerSegment = 40;
    std::vector<Vector2> raw;
    std::vector<float> rawArcLen;
    float acc = 0.0f;
    for (int seg = 0; seg < n; seg++) {
        Vector2 p0 = cp[(seg - 1 + n) % n], p1 = cp[seg], p2 = cp[(seg + 1) % n], p3 = cp[(seg + 2) % n];
        for (int s = 0; s < samplesPerSegment; s++) {
            float t = (float)s / (float)samplesPerSegment;
            Vector2 pt = catmullRom(p0, p1, p2, p3, t);
            if (!raw.empty()) acc += Vector2Distance(raw.back(), pt);
            raw.push_back(pt);
            rawArcLen.push_back(acc);
        }
    }
    // Close the loop distance (last raw sample back to the first).
    float closingDist = Vector2Distance(raw.back(), raw.front());
    totalLength = acc + closingDist;

    // 2. Resample at fixed ds spacing by walking the raw table.
    center.clear();
    int rawCount = (int)raw.size();
    int rawIdx = 0;
    for (float s = 0.0f; s < totalLength; s += ds) {
        while (rawIdx < rawCount - 1 && rawArcLen[rawIdx + 1] < s) rawIdx++;
        Vector2 a = raw[rawIdx];
        Vector2 b = (rawIdx + 1 < rawCount) ? raw[rawIdx + 1] : raw[0];
        float segStart = rawArcLen[rawIdx];
        float segLen = ((rawIdx + 1 < rawCount) ? rawArcLen[rawIdx + 1] : totalLength) - segStart;
        float t = segLen > 0.0001f ? (s - segStart) / segLen : 0.0f;
        Vector2 pt = Vector2Lerp(a, b, Clamp(t, 0.0f, 1.0f));
        center.push_back((Vector3){ pt.x, trackY, pt.y });
    }
    totalLength = (float)center.size() * ds; // snap to the actual table size used everywhere else

    // 3. Tangent (central difference, wrapped) -> perpendicular offset for edges.
    int count = (int)center.size();
    leftEdge.resize(count);
    rightEdge.resize(count);
    float minWidth = 1e9f;
    for (int i = 0; i < count; i++) {
        Vector3 prev = center[wrapIndex(i - 1)];
        Vector3 next = center[wrapIndex(i + 1)];
        Vector2 tangent = Vector2Normalize((Vector2){ next.x - prev.x, next.z - prev.z });
        Vector2 perp = { -tangent.y, tangent.x };
        leftEdge[i]  = (Vector3){ center[i].x - perp.x * halfWidth, trackY, center[i].z - perp.y * halfWidth };
        rightEdge[i] = (Vector3){ center[i].x + perp.x * halfWidth, trackY, center[i].z + perp.y * halfWidth };
        minWidth = fminf(minWidth, Vector3Distance(leftEdge[i], rightEdge[i]));
    }
    // Sanity check: a corner tighter than the control points intended (or a
    // bad control-point layout) can pinch the edges toward/across each
    // other. Warn loudly rather than silently shipping a broken track -
    // see the "spline pinching" note in the implementation plan.
    if (minWidth < halfWidth * 2.0f * 0.9f) {
        printf("WARNING: Track.h - track width pinches to %.2f (expected ~%.2f) on a sharp corner; "
               "widen that corner's control points or reduce halfWidth.\n", minWidth, halfWidth * 2.0f);
    }

    // 4. Ribbon mesh, same mechanics as ThrowGame.cpp's buildTerrainModel()
    // (manual Mesh fill, uint16 index buffer, UploadMesh/LoadModelFromMesh)
    // but the two edge vertices per sample come from leftEdge/rightEdge
    // instead of a fixed +/-halfWidth offset, and normals are flat up
    // (constant-Y track) instead of slope-derived.
    int vertexCount = count * 2;
    int triangleCount = count * 2; // closed loop: one quad pair per sample, wrapping
    Mesh m = { 0 };
    m.vertexCount = vertexCount;
    m.triangleCount = triangleCount;
    m.vertices = (float*)MemAlloc(vertexCount * 3 * sizeof(float));
    m.normals = (float*)MemAlloc(vertexCount * 3 * sizeof(float));
    m.texcoords = (float*)MemAlloc(vertexCount * 2 * sizeof(float));
    m.colors = (unsigned char*)MemAlloc(vertexCount * 4 * sizeof(unsigned char));
    m.indices = (unsigned short*)MemAlloc(triangleCount * 3 * sizeof(unsigned short));

    for (int i = 0; i < count; i++) {
        for (int side = 0; side < 2; side++) {
            int v = i * 2 + side;
            Vector3 p = side == 0 ? leftEdge[i] : rightEdge[i];
            m.vertices[v * 3 + 0] = p.x;
            m.vertices[v * 3 + 1] = p.y;
            m.vertices[v * 3 + 2] = p.z;
            m.normals[v * 3 + 0] = 0.0f;
            m.normals[v * 3 + 1] = 1.0f;
            m.normals[v * 3 + 2] = 0.0f;
            // Repeat every ~4 units along the racing line (not one tile
            // stretched over the whole loop) so a tiled road texture reads
            // as asphalt grain instead of a single blurred smear.
            m.texcoords[v * 2 + 0] = ((float)i * ds) / 4.0f;
            m.texcoords[v * 2 + 1] = (float)side;
            // Faint alternating stripe every ~10 samples along the racing
            // line, like a curb/rumble strip - cheap visual read of motion
            // and lap progress without a texture asset.
            bool stripe = (i / 10) % 2 == 0;
            m.colors[v * 4 + 0] = stripe ? 70 : 60;
            m.colors[v * 4 + 1] = stripe ? 70 : 60;
            m.colors[v * 4 + 2] = stripe ? 78 : 66;
            m.colors[v * 4 + 3] = 255;
        }
    }
    int idx = 0;
    for (int i = 0; i < count; i++) {
        int iNext = wrapIndex(i + 1);
        unsigned short bottomA = i * 2, topA = i * 2 + 1;
        unsigned short bottomB = iNext * 2, topB = iNext * 2 + 1;
        m.indices[idx++] = bottomA; m.indices[idx++] = topA; m.indices[idx++] = bottomB;
        m.indices[idx++] = topA;    m.indices[idx++] = topB; m.indices[idx++] = bottomB;
    }

    UploadMesh(&m, false);
    mesh = LoadModelFromMesh(m);
}

inline float Track::nearestArcLength(Vector3 pos, float hintArcLength, float* outLateralDist) const {
    int n = (int)center.size();
    int hintIdx = wrapIndex((int)(hintArcLength / ds));
    int window = 25; // see "windowed search" safeguard note - generous vs. per-frame movement
    float bestDist2 = 1e18f;
    int bestIdx = hintIdx;
    for (int di = -window; di <= window; di++) {
        int idx = wrapIndex(hintIdx + di);
        float dx = pos.x - center[idx].x, dz = pos.z - center[idx].z;
        float d2 = dx * dx + dz * dz;
        if (d2 < bestDist2) { bestDist2 = d2; bestIdx = idx; }
    }
    if (outLateralDist) *outLateralDist = sqrtf(bestDist2);
    (void)n;
    return bestIdx * ds;
}

// 2D ray (origin + dir*t, t>=0) vs. segment (a to b, u in [0,1]) intersection.
static inline bool raySegmentIntersect(float ox, float oz, float dx, float dz,
                                        float ax, float az, float bx, float bz,
                                        float& outT, float& outU) {
    float ex = bx - ax, ez = bz - az;
    float denom = dx * ez - dz * ex;
    if (fabsf(denom) < 1e-8f) return false; // parallel
    float t = ((ax - ox) * ez - (az - oz) * ex) / denom;
    float u = ((ax - ox) * dz - (az - oz) * dx) / denom;
    outT = t;
    outU = u;
    return true;
}

inline float Track::raycastEdgeDistance(Vector3 origin, Vector2 dirXZ, float hintArcLength, float maxRange) const {
    float best = maxRange;
    int hintIdx = wrapIndex((int)(hintArcLength / ds));
    int window = (int)(maxRange / ds) + 5;
    const std::vector<Vector3>* edges[2] = { &leftEdge, &rightEdge };
    for (int e = 0; e < 2; e++) {
        const std::vector<Vector3>& edge = *edges[e];
        for (int di = -window; di <= window; di++) {
            int i0 = wrapIndex(hintIdx + di);
            int i1 = wrapIndex(hintIdx + di + 1);
            float t, u;
            if (raySegmentIntersect(origin.x, origin.z, dirXZ.x, dirXZ.y,
                                     edge[i0].x, edge[i0].z, edge[i1].x, edge[i1].z, t, u) &&
                t >= 0.0f && t < best && u >= 0.0f && u <= 1.0f) {
                best = t;
            }
        }
    }
    return best;
}

inline Vector3 Track::tangentAt(float s) const {
    int i = wrapIndex((int)(s / ds));
    Vector3 prev = center[wrapIndex(i - 1)];
    Vector3 next = center[wrapIndex(i + 1)];
    Vector2 t = Vector2Normalize((Vector2){ next.x - prev.x, next.z - prev.z });
    return (Vector3){ t.x, 0.0f, t.y };
}

#endif // TRACK_H
