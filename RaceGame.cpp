#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include "Track.h"
#include "CarPhysics.h"
#include "CarBrain.h"

// Racing mode: 6 cars, each a tiny neural-net "brain" (CarBrain.h), improve
// generation over generation via a genetic algorithm - no gradients, no
// PyTorch, no subprocess. Simulate a generation, score by track progress,
// keep the best brain, mutate the rest, repeat. Pure C++: a 6-car MLP
// forward pass is a few hundred multiply-adds/frame, trivially fast
// in-process, so unlike ThrowGame.cpp's predict_server.py bridge (built for
// a pretrained PyTorch checkpoint) there is nothing here that benefits from
// a separate process. Run from build/ (./race_game), same as throw_game.
//
// Uses the same custom Blinn-Phong lighting shader as ThrowGame.cpp
// (Shaders/lighting.vs/.fs) on the track and cars, plus the postprocess
// vignette/contrast pass at final composite - same visual language as the
// throw game instead of flat-shaded primitives.

static const char* BRAIN_PATH = "../race_brain.txt";
static const char* SETTINGS_PATH = "../race_settings.txt";
static const char* LOG_DIR = "../race_logs";

static constexpr int POP_SIZE = 6;
static constexpr float MUTATION_RATE = 0.15f;
static constexpr float MUTATION_STRENGTH = 0.3f;
static constexpr float MAX_SENSOR_RANGE = 25.0f;
static const float SENSOR_ANGLES_DEG[7] = { -60, -40, -20, 0, 20, 40, 60 };
static constexpr float WALL_RESTITUTION = 0.35f;  // energy kept after bouncing off a wall
// Was 1.0 - negligible next to the hundreds of fitness points a lap is
// worth, so evolution had no real reason to stop bouncing off walls (a
// car that crudely bounced its way forward could still out-score a
// careful driver). Raised so hitting something is a genuinely costly
// mistake, not a rounding error - roughly half a lap's worth of progress
// per hit, comparable to HEALTH_DAMAGE_PER_HIT's real consequence (dying).
static constexpr float WALL_HIT_PENALTY = 15.0f;
static constexpr float HEALTH_MAX = 100.0f;
static constexpr float HEALTH_DAMAGE_PER_HIT = 20.0f; // 5 solid hits before a car dies
// A generation can't be eliminated by crashing anymore (cars bounce and
// keep driving - see the per-frame wall-collision response below), so lap
// mode needs its own hard backstop: if no car ever reaches the lap target
// (typical for early, badly-evolved generations), this still ends the
// generation instead of running forever.
static constexpr float SAFETY_MAX_TIME = 120.0f;

// --- Upgrade/currency system ---
// Coins earned by distance/laps; the brain itself decides which part to
// buy next (Brain::OUT indices 2-4 are engine/tire/armor preference
// scores - see CarBrain.h) whenever it can afford at least one tier.
static constexpr int COINS_PER_LAP = 50;
static constexpr float DISTANCE_PER_COIN_TICK = 50.0f;
static constexpr int COINS_PER_DISTANCE_TICK = 5;
static constexpr int UPGRADE_TIER_COUNT = 3;

enum class UpgradeCategory { ENGINE = 0, TIRES = 1, ARMOR = 2 };

struct UpgradeTierInfo { int cost; const char* name; };

static const UpgradeTierInfo ENGINE_TIERS[UPGRADE_TIER_COUNT] = { { 80, "Engine Tune" }, { 160, "Turbo Kit" }, { 280, "Race Engine" } };
static const UpgradeTierInfo TIRE_TIERS[UPGRADE_TIER_COUNT]   = { { 60, "Sport Tires" }, { 140, "Race Tires" }, { 240, "Slick Tires" } };
static const UpgradeTierInfo ARMOR_TIERS[UPGRADE_TIER_COUNT]  = { { 70, "Reinforced Frame" }, { 150, "Roll Cage" }, { 260, "Armor Plating" } };
static constexpr float ENGINE_TIER_BONUS = 0.15f; // +15% top speed/accel per tier, stacking
static constexpr float TIRE_TIER_BONUS = 0.15f;   // +15% turn rate per tier, stacking
static constexpr float ARMOR_TIER_BONUS = 25.0f;  // +25 max health per tier, stacking

// A 4th thing the brain can spend coins on: instead of a stat boost, this
// one-shot purchase carries this generation's tiers (engine/tire/armor)
// into the SAME car slot's next generation instead of resetting to stock.
// Caveat, explained in the garage menu too: it carries by slot, not by
// brain identity - the brain occupying that slot next generation is a
// mutated descendant of this generation's best (see selectAndMutate), not
// necessarily the exact brain that earned the parts. Still matches the
// ask ("this car's picks don't reset") - just worth knowing why.
static constexpr int PERSIST_PACK_COST = 250;
static const char* PERSIST_PACK_NAME = "Keep Setup Pack";

static const Color CAR_COLORS[POP_SIZE] = { RED, GOLD, LIME, SKYBLUE, VIOLET, ORANGE };

// --- Reused ThrowGame.cpp-style helpers (reimplemented locally - see the
// implementation plan for why ThrowGame.cpp itself isn't touched) ---

static void DrawTextOutlined(const char* text, int x, int y, int fontSize, Color color) {
    DrawText(text, x + 2, y + 2, fontSize, (Color){ 0, 0, 0, 170 });
    DrawText(text, x, y, fontSize, color);
}

static void DrawPanel(int x, int y, int w, int h, Color accent) {
    DrawRectangle(x, y, w, h, (Color){ 10, 12, 18, 165 });
    DrawRectangleLines(x, y, w, h, (Color){ 255, 255, 255, 30 });
    DrawRectangle(x, y, 3, h, accent);
}
static void DrawPanel(int x, int y, int w, int h) {
    DrawPanel(x, y, w, h, (Color){ 120, 170, 255, 220 });
}

// --- Camera/mouse settings persistence, same key-value-per-line convention
// as ThrowGame.cpp's GameSettings/loadSettings/saveSettings ---

enum class RaceEndMode { TIME_LIMIT = 0, LAP_COUNT = 1 };

struct RaceSettings {
    float mouseSensitivity = 0.15f;
    bool invertY = false;
    RaceEndMode endMode = RaceEndMode::TIME_LIMIT;
    float timeLimitSeconds = 45.0f;
    int lapTarget = 3;
};

static RaceSettings loadSettings(const std::string& path) {
    RaceSettings s;
    std::ifstream in(path);
    if (!in.is_open()) return s;
    std::string key;
    while (in >> key) {
        if (key == "mouseSensitivity") in >> s.mouseSensitivity;
        else if (key == "invertY") in >> s.invertY;
        else if (key == "endMode") { int m; in >> m; s.endMode = m == 1 ? RaceEndMode::LAP_COUNT : RaceEndMode::TIME_LIMIT; }
        else if (key == "timeLimitSeconds") in >> s.timeLimitSeconds;
        else if (key == "lapTarget") in >> s.lapTarget;
    }
    return s;
}
static void saveSettings(const std::string& path, const RaceSettings& s) {
    std::ofstream out(path);
    out << "mouseSensitivity " << s.mouseSensitivity << "\n";
    out << "invertY " << (s.invertY ? 1 : 0) << "\n";
    out << "endMode " << (s.endMode == RaceEndMode::LAP_COUNT ? 1 : 0) << "\n";
    out << "timeLimitSeconds " << s.timeLimitSeconds << "\n";
    out << "lapTarget " << s.lapTarget << "\n";
}

// --- Wall-collision spark particles, same pattern as ThrowGame.cpp's
// impact bursts: simple pos/vel/life, unlit, shrink+fade over lifetime ---
struct Particle { Vector3 pos, vel; float life, maxLife; };

static void spawnWallSparks(std::vector<Particle>& particles, Vector3 origin) {
    for (int i = 0; i < 10; i++) {
        Particle p;
        p.pos = origin;
        p.vel = (Vector3){ (float)GetRandomValue(-400, 400) / 100.0f, (float)GetRandomValue(100, 400) / 100.0f, (float)GetRandomValue(-400, 400) / 100.0f };
        p.maxLife = (float)GetRandomValue(20, 40) / 100.0f;
        p.life = p.maxLife;
        particles.push_back(p);
    }
}

// --- Obstacles / traffic: static pillars and slow lateral-patrolling
// "traffic" hazards on the track surface. Detected by the SAME 7 sensor
// rays the cars already use for track edges (see raycastObstacleDistance
// below, taken as the min alongside the edge distance) - no brain
// architecture change needed for cars to "see" them, the existing spatial
// awareness just gets more to look at. ---
struct Obstacle {
    Vector3 pos;
    float radius;
    bool moving;
    Vector3 patrolA, patrolB; // moving obstacles ping-pong between these
    float patrolT;            // 0..1 along the patrol path
    int patrolDir;            // +1 or -1
};

static constexpr float TRAFFIC_PATROL_SPEED = 0.22f; // patrolT units/sec

static std::vector<Obstacle> buildObstacles(const Track& track) {
    std::vector<Obstacle> obstacles;

    auto placeStatic = [&](float sFrac, float lateralFrac) {
        float s = sFrac * track.totalLength;
        Vector3 c = track.pointAt(s);
        Vector3 t = track.tangentAt(s);
        Vector2 perp = { -t.z, t.x };
        float lateral = lateralFrac * (Track::halfWidth - 1.5f); // stay inboard of the edges
        Obstacle o{};
        o.pos = { c.x + perp.x * lateral, Track::trackY, c.z + perp.y * lateral };
        o.radius = 2.0f;
        o.moving = false;
        obstacles.push_back(o);
    };
    placeStatic(0.15f, -0.5f);
    placeStatic(0.35f, 0.6f);
    placeStatic(0.55f, -0.6f);
    placeStatic(0.78f, 0.4f);

    auto placeTraffic = [&](float sFrac) {
        float s = sFrac * track.totalLength;
        Vector3 c = track.pointAt(s);
        Vector3 t = track.tangentAt(s);
        Vector2 perp = { -t.z, t.x };
        float lateral = Track::halfWidth - 1.5f;
        Obstacle o{};
        o.patrolA = { c.x - perp.x * lateral, Track::trackY, c.z - perp.y * lateral };
        o.patrolB = { c.x + perp.x * lateral, Track::trackY, c.z + perp.y * lateral };
        o.pos = o.patrolA;
        o.radius = 1.6f;
        o.moving = true;
        o.patrolT = 0.0f;
        o.patrolDir = 1;
        obstacles.push_back(o);
    };
    placeTraffic(0.25f);
    placeTraffic(0.65f);

    return obstacles;
}

static void updateObstacles(std::vector<Obstacle>& obstacles, float dt) {
    for (auto& o : obstacles) {
        if (!o.moving) continue;
        o.patrolT += TRAFFIC_PATROL_SPEED * dt * o.patrolDir;
        if (o.patrolT >= 1.0f) { o.patrolT = 1.0f; o.patrolDir = -1; }
        if (o.patrolT <= 0.0f) { o.patrolT = 0.0f; o.patrolDir = 1; }
        o.pos = Vector3Lerp(o.patrolA, o.patrolB, o.patrolT);
    }
}

// --- Visible track-edge walls: a physical red/white barrier rendered along
// Track::leftEdge/rightEdge, so a wall hit is something the player can SEE,
// not just a number changing. Purely visual - collision already happens
// against these same edge arrays via raycastEdgeDistance/nearestArcLength
// (see the per-agent wall-collision block in main()), which is what applies
// WALL_HIT_PENALTY and HEALTH_DAMAGE_PER_HIT; this mesh doesn't touch
// physics. Backface culling is disabled for the draw call (not per-triangle
// winding tricks) so the ribbon reads correctly from both the track side
// and the outside, regardless of which edge it's on. ---
static constexpr float WALL_HEIGHT = 1.4f;
static constexpr int WALL_STRIPE_SAMPLES = 6;

static Model buildWallModel(const Track& track) {
    int count = (int)track.center.size();
    int vertexCount = count * 4; // (bottom,top) x (left,right)
    int triangleCount = count * 4; // 2 quads (left,right) x 2 tris, per sample->next
    Mesh m = { 0 };
    m.vertexCount = vertexCount;
    m.triangleCount = triangleCount;
    m.vertices = (float*)MemAlloc(vertexCount * 3 * sizeof(float));
    m.normals = (float*)MemAlloc(vertexCount * 3 * sizeof(float));
    m.texcoords = (float*)MemAlloc(vertexCount * 2 * sizeof(float));
    m.colors = (unsigned char*)MemAlloc(vertexCount * 4 * sizeof(unsigned char));
    m.indices = (unsigned short*)MemAlloc(triangleCount * 3 * sizeof(unsigned short));

    int idx = 0;
    for (int side = 0; side < 2; side++) { // 0 = left edge, 1 = right edge
        for (int i = 0; i < count; i++) {
            Vector3 prev = track.center[track.wrapIndex(i - 1)];
            Vector3 next = track.center[track.wrapIndex(i + 1)];
            Vector2 tangent = Vector2Normalize((Vector2){ next.x - prev.x, next.z - prev.z });
            Vector2 perp = { -tangent.y, tangent.x };
            Vector3 outward = side == 0 ? (Vector3){ -perp.x, 0.0f, -perp.y } : (Vector3){ perp.x, 0.0f, perp.y };
            Vector3 edgePt = side == 0 ? track.leftEdge[i] : track.rightEdge[i];
            bool stripe = (i / WALL_STRIPE_SAMPLES) % 2 == 0;
            Color c = stripe ? (Color){ 205, 30, 30, 255 } : (Color){ 225, 225, 225, 255 };

            int base = (side * count + i) * 2;
            for (int j = 0; j < 2; j++) { // 0 = bottom, 1 = top
                int v = base + j;
                Vector3 p = { edgePt.x, Track::trackY + (j == 0 ? 0.0f : WALL_HEIGHT), edgePt.z };
                m.vertices[v * 3 + 0] = p.x; m.vertices[v * 3 + 1] = p.y; m.vertices[v * 3 + 2] = p.z;
                m.normals[v * 3 + 0] = outward.x; m.normals[v * 3 + 1] = outward.y; m.normals[v * 3 + 2] = outward.z;
                m.texcoords[v * 2 + 0] = (float)i / 4.0f; m.texcoords[v * 2 + 1] = (float)j;
                m.colors[v * 4 + 0] = c.r; m.colors[v * 4 + 1] = c.g; m.colors[v * 4 + 2] = c.b; m.colors[v * 4 + 3] = c.a;
            }
        }
        for (int i = 0; i < count; i++) {
            int iNext = track.wrapIndex(i + 1);
            unsigned short bottomA = (unsigned short)((side * count + i) * 2);
            unsigned short topA = bottomA + 1;
            unsigned short bottomB = (unsigned short)((side * count + iNext) * 2);
            unsigned short topB = bottomB + 1;
            m.indices[idx++] = bottomA; m.indices[idx++] = topA; m.indices[idx++] = bottomB;
            m.indices[idx++] = topA;    m.indices[idx++] = topB; m.indices[idx++] = bottomB;
        }
    }

    UploadMesh(&m, false);
    return LoadModelFromMesh(m);
}

// --- Procedural ground/road textures. No external asset files (same
// no-asset-fetching convention as main.cpp's beam_sim fallback) - grass and
// asphalt are both GenImagePerlinNoise() tinted/contrasted, then tiled via
// REPEAT wrap + texcoords that repeat every few units (see Track.h's build()
// and buildGroundPlane() below) instead of stretching one tile over the
// whole surface. ---
static Texture2D generateGrassTexture() {
    Image img = GenImagePerlinNoise(256, 256, 0, 0, 4.5f);
    ImageColorContrast(&img, 25.0f);
    ImageColorTint(&img, (Color){ 80, 150, 65, 255 });
    ImageColorBrightness(&img, -10);
    Texture2D tex = LoadTextureFromImage(img);
    UnloadImage(img);
    GenTextureMipmaps(&tex);
    SetTextureFilter(tex, TEXTURE_FILTER_TRILINEAR);
    SetTextureWrap(tex, TEXTURE_WRAP_REPEAT);
    return tex;
}

static Texture2D generateRoadTexture() {
    Image img = GenImagePerlinNoise(256, 256, 50, 50, 7.0f);
    ImageColorContrast(&img, 15.0f);
    ImageColorTint(&img, (Color){ 90, 90, 98, 255 });
    ImageColorBrightness(&img, -35);
    Texture2D tex = LoadTextureFromImage(img);
    UnloadImage(img);
    GenTextureMipmaps(&tex);
    SetTextureFilter(tex, TEXTURE_FILTER_TRILINEAR);
    SetTextureWrap(tex, TEXTURE_WRAP_REPEAT);
    return tex;
}

// Flat quad sized around the track's bounding box (+ margin) so grass
// extends well past every edge/obstacle. Texcoords repeat every
// GRASS_TILE_SIZE units (not one tile across the whole quad) so the noise
// texture reads as ground texture instead of a single blurred gradient.
static constexpr float GRASS_TILE_SIZE = 20.0f;
static constexpr float GROUND_MARGIN = 60.0f;

static Model buildGroundPlane(const Track& track) {
    float minX = 1e9f, maxX = -1e9f, minZ = 1e9f, maxZ = -1e9f;
    for (const auto& p : track.center) {
        minX = fminf(minX, p.x); maxX = fmaxf(maxX, p.x);
        minZ = fminf(minZ, p.z); maxZ = fmaxf(maxZ, p.z);
    }
    minX -= GROUND_MARGIN; maxX += GROUND_MARGIN;
    minZ -= GROUND_MARGIN; maxZ += GROUND_MARGIN;
    float sizeX = maxX - minX, sizeZ = maxZ - minZ;
    float groundY = Track::trackY - 0.05f; // just under the track so there's no z-fighting

    Mesh m = { 0 };
    m.vertexCount = 4;
    m.triangleCount = 2;
    m.vertices = (float*)MemAlloc(4 * 3 * sizeof(float));
    m.normals = (float*)MemAlloc(4 * 3 * sizeof(float));
    m.texcoords = (float*)MemAlloc(4 * 2 * sizeof(float));
    m.colors = (unsigned char*)MemAlloc(4 * 4 * sizeof(unsigned char));
    m.indices = (unsigned short*)MemAlloc(6 * sizeof(unsigned short));

    Vector3 corners[4] = {
        { minX, groundY, minZ }, { maxX, groundY, minZ },
        { maxX, groundY, maxZ }, { minX, groundY, maxZ },
    };
    float us[4] = { 0.0f, sizeX / GRASS_TILE_SIZE, sizeX / GRASS_TILE_SIZE, 0.0f };
    float vs[4] = { 0.0f, 0.0f, sizeZ / GRASS_TILE_SIZE, sizeZ / GRASS_TILE_SIZE };
    for (int i = 0; i < 4; i++) {
        m.vertices[i * 3 + 0] = corners[i].x;
        m.vertices[i * 3 + 1] = corners[i].y;
        m.vertices[i * 3 + 2] = corners[i].z;
        m.normals[i * 3 + 0] = 0.0f; m.normals[i * 3 + 1] = 1.0f; m.normals[i * 3 + 2] = 0.0f;
        m.texcoords[i * 2 + 0] = us[i];
        m.texcoords[i * 2 + 1] = vs[i];
        m.colors[i * 4 + 0] = 255; m.colors[i * 4 + 1] = 255; m.colors[i * 4 + 2] = 255; m.colors[i * 4 + 3] = 255;
    }
    unsigned short idx[6] = { 0, 2, 1, 0, 3, 2 };
    memcpy(m.indices, idx, sizeof(idx));

    UploadMesh(&m, false);
    return LoadModelFromMesh(m);
}

// 2D (XZ) ray-vs-circle nearest hit distance, same closed-form approach as
// Track.h's raySegmentIntersect but for obstacles instead of track edges.
static float raycastObstacleDistance(Vector3 origin, Vector2 dir, const std::vector<Obstacle>& obstacles, float maxRange) {
    float best = maxRange;
    for (const auto& o : obstacles) {
        float ox = o.pos.x - origin.x, oz = o.pos.z - origin.z;
        float proj = ox * dir.x + oz * dir.y; // distance along the ray to the closest approach
        if (proj < 0.0f || proj > best) continue;
        float closestX = origin.x + dir.x * proj, closestZ = origin.z + dir.y * proj;
        float dx = o.pos.x - closestX, dz = o.pos.z - closestZ;
        float distSq = dx * dx + dz * dz;
        if (distSq > o.radius * o.radius) continue; // ray misses the circle entirely
        float halfChord = sqrtf(o.radius * o.radius - distSq);
        float t = proj - halfChord; // near intersection point
        if (t >= 0.0f && t < best) best = t;
    }
    return best;
}

// Shared bounce response for both wall hits and obstacle hits: reflect
// velocity off the given normal (like bumping a guardrail), damp it, push
// the car to `insetDist` from `pushFrom` along that normal (not exactly
// onto the boundary - see the wall-collision comment where this used to
// be inlined: clamping exactly to a boundary re-triggered a bounce almost
// every frame), and enforce a minimum post-bounce speed so a near-zero
// heavily-damped bounce doesn't just sit there and immediately re-collide.
static void bounceCarState(CarState& state, Vector3 pushFrom, Vector2 normal, float insetDist) {
    const float MIN_BOUNCE_SPEED = 5.0f;
    state.pos.x = pushFrom.x + normal.x * insetDist;
    state.pos.z = pushFrom.z + normal.y * insetDist; // Vector2's .y here holds the world Z component
    Vector2 vel = { cosf(state.heading) * state.speed, sinf(state.heading) * state.speed };
    float vDotN = vel.x * normal.x + vel.y * normal.y;
    Vector2 reflected = { (vel.x - 2.0f * vDotN * normal.x) * WALL_RESTITUTION,
                           (vel.y - 2.0f * vDotN * normal.y) * WALL_RESTITUTION };
    state.speed = fmaxf(Vector2Length(reflected), MIN_BOUNCE_SPEED);
    // Reflected could be near-zero-length (near head-on hit) - heading from
    // the surface normal itself (pointing back away from what was hit)
    // rather than an undefined atan2(0,0) in that case.
    Vector2 headingDir = Vector2Length(reflected) > 0.01f ? reflected : (Vector2){ -normal.x, -normal.y };
    state.heading = atan2f(headingDir.y, headingDir.x);
}

// --- Genetic-algorithm population ---

struct CarAgent {
    Brain brain;
    CarState state;
    float rawArcLength = 0.0f;
    float fitness = 0.0f;         // selection metric: distance traveled minus wall-hit penalties
    float distanceTraveled = 0.0f; // never penalized - pure progress, used for lap counting
    int lapsCompleted = 0;
    int wallHits = 0;
    float healthMax = HEALTH_MAX; // raised by armor upgrades
    float health = HEALTH_MAX;    // wall hits drain this; degrades performance as it drops
    bool alive = true;            // false once health hits 0 - stops updating for the rest of the generation

    int coins = 0;
    float distanceSinceCoinTick = 0.0f;
    int engineTier = 0, tireTier = 0, armorTier = 0; // 0 = stock, up to UPGRADE_TIER_COUNT
    std::string lastUpgrade = "none";                // for the garage menu - what the brain last picked
    bool keepSetupBought = false; // if true when this generation ends, tiers carry into next gen instead of resetting

    float prevSteer = 0.0f, prevThrottle = 0.0f; // fed back in as brain inputs next frame - see CarBrain.h
};

struct Population {
    std::array<CarAgent, POP_SIZE> agents;
    int generation = 0;
    float bestEverFitness = -1e9f;
    Brain bestEverBrain{};
};

// weightCount acts as a cheap format-version guard: if Brain's architecture
// ever changes, an old file is rejected instead of misreading garbage.
static bool loadBrainFile(const std::string& path, int& generation, float& bestFitness, Brain& bestBrain) {
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::string key;
    int weightCount = 0;
    bool gotWeights = false;
    while (in >> key) {
        if (key == "generation") in >> generation;
        else if (key == "bestFitness") in >> bestFitness;
        else if (key == "weightCount") in >> weightCount;
        else if (key == "weights") {
            if (weightCount != BRAIN_WEIGHT_COUNT) return false;
            float* w = brainWeights(bestBrain);
            for (int i = 0; i < BRAIN_WEIGHT_COUNT; i++) in >> w[i];
            gotWeights = true;
        }
    }
    return gotWeights;
}

static void saveBrainFile(const std::string& path, const Population& pop) {
    std::ofstream out(path);
    out << "generation " << pop.generation << "\n";
    out << "bestFitness " << pop.bestEverFitness << "\n";
    out << "weightCount " << BRAIN_WEIGHT_COUNT << "\n";
    out << "weights";
    const float* w = brainWeights(pop.bestEverBrain);
    for (int i = 0; i < BRAIN_WEIGHT_COUNT; i++) out << " " << w[i];
    out << "\n";
}

// One row per generation, every car's stats - easy to load into a
// spreadsheet/plot later, unlike parsing many small per-generation files.
static void appendSummaryLog(const std::string& dir, const Population& pop) {
    std::filesystem::create_directories(dir);
    std::string path = dir + "/summary.csv";
    bool needsHeader = !std::filesystem::exists(path);
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) return;
    if (needsHeader) {
        out << "generation,bestEverFitness";
        for (int i = 0; i < POP_SIZE; i++) out << ",car" << i << "_fitness,car" << i << "_distance,car" << i << "_laps,car" << i << "_wallHits,car" << i << "_finalHealth,car" << i << "_died";
        out << "\n";
    }
    out << pop.generation << "," << pop.bestEverFitness;
    for (const auto& a : pop.agents) out << "," << a.fitness << "," << a.distanceTraveled << "," << a.lapsCompleted << "," << a.wallHits << "," << a.health << "," << (a.alive ? 0 : 1);
    out << "\n";
}

// Full weight snapshot of that generation's winning brain - lets any past
// generation's best driver be reloaded and replayed later, not just the
// single latest bestEverBrain that race_brain.txt overwrites in place.
static void logGenerationBrain(const std::string& dir, int generation, const Brain& winner, float fitness) {
    std::filesystem::create_directories(dir);
    char filename[128];
    snprintf(filename, sizeof(filename), "%s/gen_%06d_brain.txt", dir.c_str(), generation);
    std::ofstream out(filename);
    if (!out.is_open()) return;
    out << "generation " << generation << "\n";
    out << "fitness " << fitness << "\n";
    out << "weightCount " << BRAIN_WEIGHT_COUNT << "\n";
    out << "weights";
    const float* w = brainWeights(winner);
    for (int i = 0; i < BRAIN_WEIGHT_COUNT; i++) out << " " << w[i];
    out << "\n";
}

static void placeOnStartGrid(Population& pop, const Track& track) {
    Vector3 startPos = track.pointAt(0.0f);
    Vector3 tangent = track.tangentAt(0.0f);
    float heading = atan2f(tangent.z, tangent.x);
    Vector2 perp = { -tangent.z, tangent.x };
    for (int i = 0; i < POP_SIZE; i++) {
        int row = i / 3, col = i % 3;
        float lateral = (col - 1) * 3.0f;
        float back = row * 4.0f;
        CarAgent& a = pop.agents[i];
        a.state.pos = (Vector3){
            startPos.x + perp.x * lateral - tangent.x * back,
            Track::trackY,
            startPos.z + perp.y * lateral - tangent.z * back
        };
        a.state.heading = heading;
        a.state.speed = 0.0f;
        a.rawArcLength = 0.0f;
        a.fitness = 0.0f;
        a.distanceTraveled = 0.0f;
        a.lapsCompleted = 0;
        a.wallHits = 0;
        a.healthMax = HEALTH_MAX;
        a.health = HEALTH_MAX;
        a.alive = true;
        // Upgrades reset with everything else each generation, same as
        // position/fitness/health - a "career" of permanent upgrades across
        // generations would fight the genetic algorithm's own reset (the
        // brain in car slot N next generation is a mutated DIFFERENT brain,
        // not the one that earned last generation's parts), so each
        // generation is one clean test of "how well does this brain drive
        // AND spend its earnings," starting from stock every time.
        a.coins = 0;
        a.distanceSinceCoinTick = 0.0f;
        a.engineTier = 0;
        a.tireTier = 0;
        a.armorTier = 0;
        a.lastUpgrade = "none";
        a.keepSetupBought = false; // must be re-bought each generation to keep carrying forward
        a.prevSteer = 0.0f;
        a.prevThrottle = 0.0f;
    }
}

// Every action every car's brain takes, every frame: raw steering/throttle
// output, resulting speed/health, and any upgrade purchased that frame.
// Opened fresh per generation (race_logs/gen_NNNNNN_actions.csv) so a long
// run doesn't grow one unbounded file - the summary.csv above is the
// cheap aggregate view, this is the full trace for a specific generation
// you want to inspect closely.
static std::ofstream openActionLog(const std::string& dir, int generation) {
    std::filesystem::create_directories(dir);
    char filename[128];
    snprintf(filename, sizeof(filename), "%s/gen_%06d_actions.csv", dir.c_str(), generation);
    std::ofstream out(filename);
    if (out.is_open()) out << "frame,car,steering,throttle,speed,health,coins,upgrade\n";
    return out;
}

static void logAction(std::ofstream& log, int frame, int carIndex, float steering, float throttle,
                       float speed, float health, int coins, const std::string& upgrade) {
    if (!log.is_open()) return;
    log << frame << "," << carIndex << "," << steering << "," << throttle << "," << speed << ","
        << health << "," << coins << "," << upgrade << "\n";
}

// Picks the highest-scoring AFFORDABLE upgrade category from the brain's
// own output (out[2]=engine, out[3]=tires, out[4]=armor - see CarBrain.h).
// Returns the tier name bought, or empty if nothing was affordable/picked.
static std::string tryPurchaseUpgrade(CarAgent& agent, const float out[Brain::OUT]) {
    int tiers[3] = { agent.engineTier, agent.tireTier, agent.armorTier };
    const UpgradeTierInfo* tables[3] = { ENGINE_TIERS, TIRE_TIERS, ARMOR_TIERS };

    int bestCat = -1; // 0-2 = engine/tire/armor tier, 3 = keep-setup pack
    float bestScore = -1e9f;
    for (int cat = 0; cat < 3; cat++) {
        if (tiers[cat] >= UPGRADE_TIER_COUNT) continue; // maxed out
        if (tables[cat][tiers[cat]].cost > agent.coins) continue; // can't afford this tier yet
        float score = out[2 + cat];
        if (score > bestScore) { bestScore = score; bestCat = cat; }
    }
    if (!agent.keepSetupBought && PERSIST_PACK_COST <= agent.coins && out[5] > bestScore) {
        bestScore = out[5];
        bestCat = 3;
    }
    if (bestCat < 0) return "";

    if (bestCat == 3) {
        agent.coins -= PERSIST_PACK_COST;
        agent.keepSetupBought = true;
        agent.lastUpgrade = PERSIST_PACK_NAME;
        return PERSIST_PACK_NAME;
    }

    const UpgradeTierInfo& tier = tables[bestCat][tiers[bestCat]];
    agent.coins -= tier.cost;
    if (bestCat == 0) agent.engineTier++;
    else if (bestCat == 1) agent.tireTier++;
    else { agent.armorTier++; agent.healthMax += ARMOR_TIER_BONUS; agent.health += ARMOR_TIER_BONUS; }
    agent.lastUpgrade = tier.name;
    return tier.name;
}

static void selectAndMutate(Population& pop) {
    std::array<int, POP_SIZE> order = { 0, 1, 2, 3, 4, 5 };
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return pop.agents[a].fitness > pop.agents[b].fitness;
    });

    Brain best = pop.agents[order[0]].brain;
    Brain second = pop.agents[order[1]].brain;

    if (pop.agents[order[0]].fitness > pop.bestEverFitness) {
        pop.bestEverFitness = pop.agents[order[0]].fitness;
        pop.bestEverBrain = best;
    }

    std::array<Brain, POP_SIZE> next;
    next[0] = best; // elitism: unchanged
    next[1] = mutate(crossover(best, second), MUTATION_RATE, MUTATION_STRENGTH);
    for (int i = 2; i < POP_SIZE; i++) next[i] = mutate(best, MUTATION_RATE, MUTATION_STRENGTH);

    for (int i = 0; i < POP_SIZE; i++) pop.agents[i].brain = next[i];

    appendSummaryLog(LOG_DIR, pop);
    logGenerationBrain(LOG_DIR, pop.generation, best, pop.agents[order[0]].fitness);

    pop.generation++;
    printf("Generation %d done | best=%.1f second=%.1f best-ever=%.1f\n",
           pop.generation, pop.agents[order[0]].fitness, pop.agents[order[1]].fitness, pop.bestEverFitness);
}

// ~12 hand-placed control points forming a gentle rounded oval with one
// mild S-bend for interest. Kept generous relative to halfWidth=6 (no
// corner tighter than ~3x halfWidth) so the spline can't pinch its own
// edges - see the "spline pinching" safeguard in the implementation plan.
static std::vector<Vector2> trackControlPoints() {
    return {
        {   0.0f, -70.0f }, {  55.0f, -68.0f }, { 100.0f, -35.0f },
        { 100.0f,   0.0f }, {  70.0f,  15.0f }, {  40.0f,   0.0f }, // gentle inward S-bend
        {  60.0f,  45.0f }, { 100.0f,  70.0f }, {  30.0f,  75.0f },
        { -60.0f,  65.0f }, {-100.0f,  25.0f }, {-100.0f, -30.0f },
        { -60.0f, -65.0f }
    };
}

int main() {
    SetRandomSeed((unsigned int)GetTime());

    const int screenWidth = 1280;
    const int screenHeight = 800;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(screenWidth, screenHeight, "BeamSim3D - Race Game");
    SetWindowMinSize(480, 300);
    SetExitKey(KEY_NULL); // ESC drives pause, not quit - window close button still works

    Shader postShader = LoadShader(0, "../Shaders/postprocess.fs");
    RenderTexture2D sceneTarget = LoadRenderTexture(screenWidth, screenHeight);

    // Same lighting shader ThrowGame.cpp uses on its terrain - one warm
    // directional light + ambient + distance fog. All of fogColor/
    // fogStart/fogEnd MUST be set (even to "basically off" values) - an
    // unset fog uniform defaults to 0 in GLSL, which makes fogStart==
    // fogEnd==0 and the whole scene renders solid black. Learned that the
    // hard way earlier tonight on ThrowGame's terrain.
    Shader litShader = LoadShader("../Shaders/lighting.vs", "../Shaders/lighting.fs");
    int locLightDir = GetShaderLocation(litShader, "lightDir");
    int locLightColor = GetShaderLocation(litShader, "lightColor");
    int locAmbientColor = GetShaderLocation(litShader, "ambientColor");
    int locViewPos = GetShaderLocation(litShader, "viewPos");
    int locShininess = GetShaderLocation(litShader, "shininess");
    int locFogColor = GetShaderLocation(litShader, "fogColor");
    int locFogStart = GetShaderLocation(litShader, "fogStart");
    int locFogEnd = GetShaderLocation(litShader, "fogEnd");

    Vector3 lightDir = Vector3Normalize((Vector3){ -0.4f, -1.0f, -0.35f });
    Vector3 lightColor = { 1.0f, 0.97f, 0.9f };
    Vector3 ambientColor = { 0.30f, 0.33f, 0.40f };
    float shininess = 20.0f;
    Vector3 fogColor = { 90.0f / 255.0f, 100.0f / 255.0f, 130.0f / 255.0f }; // matches the sky gradient's horizon color
    float fogStart = 90.0f, fogEnd = 260.0f; // track is small (halfWidth=6) - fog kicks in much closer than ThrowGame's
    SetShaderValue(litShader, locLightDir, &lightDir, SHADER_UNIFORM_VEC3);
    SetShaderValue(litShader, locLightColor, &lightColor, SHADER_UNIFORM_VEC3);
    SetShaderValue(litShader, locAmbientColor, &ambientColor, SHADER_UNIFORM_VEC3);
    SetShaderValue(litShader, locShininess, &shininess, SHADER_UNIFORM_FLOAT);
    SetShaderValue(litShader, locFogColor, &fogColor, SHADER_UNIFORM_VEC3);
    SetShaderValue(litShader, locFogStart, &fogStart, SHADER_UNIFORM_FLOAT);
    SetShaderValue(litShader, locFogEnd, &fogEnd, SHADER_UNIFORM_FLOAT);

    Track track;
    track.build(trackControlPoints());
    track.mesh.materials[0].shader = litShader;
    Texture2D roadTexture = generateRoadTexture();
    track.mesh.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = roadTexture;
    std::vector<Obstacle> obstacles = buildObstacles(track);

    Model groundModel = buildGroundPlane(track);
    Texture2D grassTexture = generateGrassTexture();
    groundModel.materials[0].shader = litShader;
    groundModel.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = grassTexture;

    Model wallModel = buildWallModel(track);
    wallModel.materials[0].shader = litShader;

    Population pop;
    int loadedGen = 0;
    float loadedBest = -1e9f;
    Brain loadedBrain{};
    if (loadBrainFile(BRAIN_PATH, loadedGen, loadedBest, loadedBrain)) {
        pop.generation = loadedGen;
        pop.bestEverFitness = loadedBest;
        pop.bestEverBrain = loadedBrain;
        pop.agents[0].brain = loadedBrain;
        for (int i = 1; i < POP_SIZE; i++) pop.agents[i].brain = mutate(loadedBrain, MUTATION_RATE, MUTATION_STRENGTH);
        printf("Loaded race_brain.txt: resuming from generation %d (best-ever %.1f)\n", loadedGen, loadedBest);
    } else {
        for (auto& agent : pop.agents) agent.brain = randomBrain();
        printf("No race_brain.txt found - starting fresh from generation 0\n");
    }
    placeOnStartGrid(pop, track);
    std::ofstream actionLog = openActionLog(LOG_DIR, pop.generation);
    int frameInGen = 0;
    std::vector<Particle> particles;

    RaceSettings settings = loadSettings(SETTINGS_PATH);
    bool paused = false;
    bool showSettings = false;
    bool showGarage = false;
    int followCar = -1; // -1 = auto-follow current leader

    Vector3 followTarget = pop.agents[0].state.pos;
    float orbitYaw = 0.0f;
    float orbitRadius = 30.0f, orbitHeight = 18.0f;
    const float orbitRadiusMin = 8.0f, orbitRadiusMax = 90.0f;
    const float orbitHeightMin = 3.0f, orbitHeightMax = 50.0f;

    double genStartTime = GetTime();
    double pauseStartTime = 0.0;
    double totalPausedDuration = 0.0;

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_ESCAPE)) {
            paused = !paused;
            if (paused) pauseStartTime = GetTime();
            else totalPausedDuration += GetTime() - pauseStartTime;
        }
        if (IsKeyPressed(KEY_TAB)) showSettings = !showSettings;
        if (IsKeyPressed(KEY_G)) showGarage = !showGarage;
        if (IsKeyPressed(KEY_F11)) ToggleFullscreen();
        for (int i = 0; i < POP_SIZE; i++) {
            if (IsKeyPressed(KEY_ONE + i)) followCar = i;
        }
        if (IsKeyPressed(KEY_ZERO)) followCar = -1;

        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            Vector2 mouseDelta = GetMouseDelta();
            orbitYaw += mouseDelta.x * settings.mouseSensitivity * 0.02f;
            float heightDelta = mouseDelta.y * settings.mouseSensitivity * 0.3f * (settings.invertY ? -1.0f : 1.0f);
            orbitHeight = Clamp(orbitHeight + heightDelta, orbitHeightMin, orbitHeightMax);
        }
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) orbitRadius = Clamp(orbitRadius - wheel * 3.0f, orbitRadiusMin, orbitRadiusMax);

        if (showSettings) {
            bool changed = false;
            if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) { settings.mouseSensitivity = Clamp(settings.mouseSensitivity + 0.05f, 0.02f, 2.0f); changed = true; }
            if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) { settings.mouseSensitivity = Clamp(settings.mouseSensitivity - 0.05f, 0.02f, 2.0f); changed = true; }
            if (IsKeyPressed(KEY_I)) { settings.invertY = !settings.invertY; changed = true; }
            if (IsKeyPressed(KEY_M)) { settings.endMode = settings.endMode == RaceEndMode::TIME_LIMIT ? RaceEndMode::LAP_COUNT : RaceEndMode::TIME_LIMIT; changed = true; }
            if (IsKeyPressed(KEY_RIGHT_BRACKET)) {
                if (settings.endMode == RaceEndMode::TIME_LIMIT) settings.timeLimitSeconds = Clamp(settings.timeLimitSeconds + 5.0f, 10.0f, SAFETY_MAX_TIME);
                else settings.lapTarget = std::min(settings.lapTarget + 1, 50);
                changed = true;
            }
            if (IsKeyPressed(KEY_LEFT_BRACKET)) {
                if (settings.endMode == RaceEndMode::TIME_LIMIT) settings.timeLimitSeconds = Clamp(settings.timeLimitSeconds - 5.0f, 10.0f, SAFETY_MAX_TIME);
                else settings.lapTarget = std::max(settings.lapTarget - 1, 1);
                changed = true;
            }
            if (changed) saveSettings(SETTINGS_PATH, settings);
        }

        // Clamp dt: bounds worst-case per-frame car movement regardless of
        // frame hitches, which is what keeps Track's windowed searches safe
        // without needing a larger window - see the implementation plan's
        // "windowed search must not let a car escape" safeguard.
        float dt = paused ? 0.0f : fminf(GetFrameTime(), 1.0f / 30.0f);

        if (!paused) {
            frameInGen++;
            updateObstacles(obstacles, dt);
            bool lapTargetReached = false;
            bool allDead = true;
            for (int carIdx = 0; carIdx < POP_SIZE; carIdx++) {
                CarAgent& agent = pop.agents[carIdx];
                if (!agent.alive) continue;
                allDead = false;

                float inputs[Brain::IN];
                for (int r = 0; r < 7; r++) {
                    float a = agent.state.heading + SENSOR_ANGLES_DEG[r] * DEG2RAD;
                    Vector2 dir = { cosf(a), sinf(a) };
                    float edgeDist = track.raycastEdgeDistance(agent.state.pos, dir, agent.rawArcLength, MAX_SENSOR_RANGE);
                    float obstacleDist = raycastObstacleDistance(agent.state.pos, dir, obstacles, MAX_SENSOR_RANGE);
                    inputs[r] = fminf(edgeDist, obstacleDist) / MAX_SENSOR_RANGE;
                }
                inputs[7] = agent.state.speed / 40.0f;
                inputs[8] = agent.prevSteer;
                inputs[9] = agent.prevThrottle;

                float nearestTraffic = MAX_SENSOR_RANGE;
                for (const auto& o : obstacles) {
                    if (!o.moving) continue;
                    float d = Vector3Distance(agent.state.pos, o.pos) - o.radius;
                    nearestTraffic = fminf(nearestTraffic, fmaxf(d, 0.0f));
                }
                inputs[10] = nearestTraffic / MAX_SENSOR_RANGE;

                float out[Brain::OUT];
                forward(agent.brain, inputs, out);
                agent.prevSteer = out[0];
                agent.prevThrottle = out[1];
                // Damaged cars drive worse; engine/tire upgrades drive
                // better - both scale the same physics knobs CarPhysics.h
                // exposes, just in opposite directions and kept separate
                // (engine affects speed/accel, tires affect turn rate) so
                // one upgrade category doesn't silently buff the other.
                float damageFactor = 0.5f + 0.5f * (agent.health / agent.healthMax);
                float enginePerf = damageFactor * (1.0f + ENGINE_TIER_BONUS * agent.engineTier);
                float turnPerf = damageFactor * (1.0f + TIRE_TIER_BONUS * agent.tireTier);
                updateCar(agent.state, out[0], out[1], dt, enginePerf, turnPerf);

                float lateralDist;
                float sRaw = track.nearestArcLength(agent.state.pos, agent.rawArcLength, &lateralDist);
                float delta = sRaw - agent.rawArcLength;
                if (delta > track.totalLength * 0.5f) delta -= track.totalLength;
                if (delta < -track.totalLength * 0.5f) delta += track.totalLength;
                agent.rawArcLength = sRaw;
                agent.fitness += delta;
                agent.distanceTraveled += delta;

                // Wall collision: bounce (not an instant elimination on the
                // very first touch), but drain health - once health runs
                // out the car actually dies for the rest of the generation.
                // Reflect velocity off the track-boundary normal (like
                // bumping a guardrail), lose some speed to the impact, and
                // push position back IN from the boundary by a margin - not
                // exactly onto it. Clamping exactly to the boundary was
                // measured to bounce a car every single frame (thousands of
                // hits/generation): a heavily-damped near-zero-speed bounce
                // sitting right at the edge gets pushed straight back out by
                // the very next frame's (often near-random, for an
                // unevolved brain) steering output, re-triggering
                // immediately. The margin plus a minimum post-bounce speed
                // guarantees real clearance before another hit is possible.
                if (lateralDist > Track::halfWidth) {
                    const float WALL_MARGIN = 1.0f;
                    Vector3 centerPt = track.pointAt(sRaw);
                    Vector2 normal = Vector2Normalize((Vector2){ agent.state.pos.x - centerPt.x, agent.state.pos.z - centerPt.z });
                    bounceCarState(agent.state, centerPt, normal, Track::halfWidth - WALL_MARGIN);
                    agent.wallHits++;
                    agent.fitness -= WALL_HIT_PENALTY;
                    agent.health = fmaxf(agent.health - HEALTH_DAMAGE_PER_HIT, 0.0f);
                    if (agent.health <= 0.0f) agent.alive = false;
                    spawnWallSparks(particles, (Vector3){ agent.state.pos.x, 0.8f, agent.state.pos.z });
                }

                // Obstacle collision - same bounce/damage consequence as a
                // wall hit, just against a circle instead of the track edge.
                for (const auto& obs : obstacles) {
                    float odx = agent.state.pos.x - obs.pos.x, odz = agent.state.pos.z - obs.pos.z;
                    float centerDist = sqrtf(odx * odx + odz * odz);
                    const float CAR_RADIUS = 0.9f;
                    float minDist = obs.radius + CAR_RADIUS;
                    if (centerDist < minDist && centerDist > 0.0001f) {
                        Vector2 normal = { odx / centerDist, odz / centerDist };
                        bounceCarState(agent.state, obs.pos, normal, minDist + 0.5f);
                        agent.wallHits++;
                        agent.fitness -= WALL_HIT_PENALTY;
                        agent.health = fmaxf(agent.health - HEALTH_DAMAGE_PER_HIT, 0.0f);
                        if (agent.health <= 0.0f) agent.alive = false;
                        spawnWallSparks(particles, (Vector3){ agent.state.pos.x, 0.8f, agent.state.pos.z });
                        break; // one obstacle hit per frame is enough to resolve
                    }
                }

                int newLaps = (int)(agent.distanceTraveled / track.totalLength);
                bool earnedCoins = false;
                if (newLaps > agent.lapsCompleted) {
                    agent.lapsCompleted = newLaps;
                    agent.coins += COINS_PER_LAP;
                    earnedCoins = true;
                }
                if (settings.endMode == RaceEndMode::LAP_COUNT && agent.lapsCompleted >= settings.lapTarget) lapTargetReached = true;

                agent.distanceSinceCoinTick += delta;
                while (agent.distanceSinceCoinTick >= DISTANCE_PER_COIN_TICK) {
                    agent.distanceSinceCoinTick -= DISTANCE_PER_COIN_TICK;
                    agent.coins += COINS_PER_DISTANCE_TICK;
                    earnedCoins = true;
                }

                // Only worth evaluating the upgrade decision right after
                // coins changed - no point re-running the same affordability
                // check every single frame in between.
                std::string upgradeThisFrame;
                if (earnedCoins) upgradeThisFrame = tryPurchaseUpgrade(agent, out);

                logAction(actionLog, frameInGen, carIdx, out[0], out[1], agent.state.speed, agent.health, agent.coins, upgradeThisFrame);
            }

            for (auto& p : particles) {
                p.vel.y -= 9.8f * dt;
                p.pos = Vector3Add(p.pos, Vector3Scale(p.vel, dt));
                p.life -= dt;
            }
            particles.erase(std::remove_if(particles.begin(), particles.end(), [](const Particle& p) { return p.life <= 0.0f; }), particles.end());

            double elapsed = GetTime() - genStartTime - totalPausedDuration;
            bool timeModeEnd = settings.endMode == RaceEndMode::TIME_LIMIT && elapsed >= settings.timeLimitSeconds;
            if (allDead || timeModeEnd || lapTargetReached || elapsed >= SAFETY_MAX_TIME) {
                // Capture which slots bought the Keep Setup Pack (and their
                // tiers) BEFORE placeOnStartGrid resets everything to
                // stock, so it can be reapplied after - see PERSIST_PACK_
                // COST's comment for the by-slot-not-by-brain caveat.
                std::array<bool, POP_SIZE> carryOver{};
                std::array<int, POP_SIZE> carryEngine{}, carryTire{}, carryArmor{};
                for (int i = 0; i < POP_SIZE; i++) {
                    carryOver[i] = pop.agents[i].keepSetupBought;
                    carryEngine[i] = pop.agents[i].engineTier;
                    carryTire[i] = pop.agents[i].tireTier;
                    carryArmor[i] = pop.agents[i].armorTier;
                }

                selectAndMutate(pop);
                saveBrainFile(BRAIN_PATH, pop);
                placeOnStartGrid(pop, track);

                for (int i = 0; i < POP_SIZE; i++) {
                    if (!carryOver[i]) continue;
                    CarAgent& a = pop.agents[i];
                    a.engineTier = carryEngine[i];
                    a.tireTier = carryTire[i];
                    a.armorTier = carryArmor[i];
                    a.healthMax = HEALTH_MAX + ARMOR_TIER_BONUS * a.armorTier;
                    a.health = a.healthMax;
                    a.lastUpgrade = "carried over";
                }

                genStartTime = GetTime();
                totalPausedDuration = 0.0;
                actionLog.close();
                actionLog = openActionLog(LOG_DIR, pop.generation);
                frameInGen = 0;
            }
        }

        // Follow the pinned car (even if it died - stay on it, don't jump
        // away), or the current fitness leader among the living, falling
        // back to whoever has the highest fitness if every car has died.
        int target = followCar;
        if (target < 0) {
            target = 0;
            for (int i = 1; i < POP_SIZE; i++) {
                bool iBetter = pop.agents[target].alive
                    ? (pop.agents[i].alive && pop.agents[i].fitness > pop.agents[target].fitness)
                    : (pop.agents[i].alive || pop.agents[i].fitness > pop.agents[target].fitness);
                if (iBetter) target = i;
            }
        }
        Vector3 desiredTarget = pop.agents[target].state.pos;
        followTarget = Vector3Lerp(followTarget, desiredTarget, 0.08f);

        Camera camera = { 0 };
        camera.target = followTarget;
        camera.position = (Vector3){
            followTarget.x + orbitRadius * cosf(orbitYaw),
            followTarget.y + orbitHeight,
            followTarget.z + orbitRadius * sinf(orbitYaw)
        };
        camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
        camera.fovy = 45.0f;
        camera.projection = CAMERA_PERSPECTIVE;
        SetShaderValue(litShader, locViewPos, &camera.position, SHADER_UNIFORM_VEC3);

        BeginTextureMode(sceneTarget);
        ClearBackground((Color){ 25, 28, 38, 255 });
        DrawRectangleGradientV(0, 0, screenWidth, screenHeight, (Color){ 40, 45, 65, 255 }, (Color){ 90, 100, 130, 255 });

        BeginMode3D(camera);
        DrawModel(groundModel, (Vector3){ 0, 0, 0 }, 1.0f, WHITE);
        DrawModel(track.mesh, (Vector3){ 0, 0, 0 }, 1.0f, WHITE);
        rlDisableBackfaceCulling();
        DrawModel(wallModel, (Vector3){ 0, 0, 0 }, 1.0f, WHITE);
        rlEnableBackfaceCulling();
        BeginShaderMode(litShader); // DrawCube uses whatever shader is active - track's Model carries its own
        for (const auto& obs : obstacles) {
            Vector3 base = { obs.pos.x, Track::trackY, obs.pos.z };
            Color col = obs.moving ? (Color){ 255, 140, 20, 255 } : (Color){ 120, 30, 30, 255 };
            DrawCylinder(base, obs.radius, obs.radius, 3.2f, 12, col);
            DrawCylinderWires(base, obs.radius, obs.radius, 3.2f, 12, (Color){ 0, 0, 0, 130 });
        }
        for (int i = 0; i < POP_SIZE; i++) {
            const CarAgent& a = pop.agents[i];
            // Dead cars sit greyed-out and dim where they died instead of
            // vanishing - a visible reminder of the generation's casualties.
            Color bodyCol = a.alive ? CAR_COLORS[i] : Fade(GRAY, 0.5f);
            Color cabinCol = a.alive ? Fade(BLACK, 0.55f) : Fade(GRAY, 0.35f);
            Vector3 pos = { a.state.pos.x, 0.55f, a.state.pos.z };

            // Simple multi-box car silhouette (body + cabin + 4 wheels)
            // instead of a single box, rotated to face the car's actual
            // heading - the old single DrawCube never rotated at all, so
            // cars visually never turned even though they were steering.
            rlPushMatrix();
            rlTranslatef(pos.x, pos.y, pos.z);
            rlRotatef(-a.state.heading * RAD2DEG, 0.0f, 1.0f, 0.0f);
            DrawCube((Vector3){ 0, 0, 0 }, 2.2f, 0.7f, 1.15f, bodyCol);
            DrawCubeWires((Vector3){ 0, 0, 0 }, 2.2f, 0.7f, 1.15f, (Color){ 0, 0, 0, 120 });
            DrawCube((Vector3){ 0.15f, 0.5f, 0 }, 1.0f, 0.55f, 0.95f, cabinCol);
            const float wx = 0.95f, wz = 0.62f, wy = -0.38f;
            Color wheelCol = (Color){ 25, 25, 28, 255 };
            DrawCube((Vector3){  wx, wy,  wz }, 0.45f, 0.45f, 0.3f, wheelCol);
            DrawCube((Vector3){  wx, wy, -wz }, 0.45f, 0.45f, 0.3f, wheelCol);
            DrawCube((Vector3){ -wx, wy,  wz }, 0.45f, 0.45f, 0.3f, wheelCol);
            DrawCube((Vector3){ -wx, wy, -wz }, 0.45f, 0.45f, 0.3f, wheelCol);
            rlPopMatrix();
        }
        EndShaderMode();
        for (const auto& p : particles) {
            DrawSphere(p.pos, 0.08f, Fade(ORANGE, p.life / p.maxLife));
        }
        EndMode3D();

        // Floating health bar above each car - projected from world space so
        // it stays anchored over the car regardless of camera angle. Green
        // -> yellow -> red as health drops, empty outline once dead.
        for (int i = 0; i < POP_SIZE; i++) {
            const CarAgent& a = pop.agents[i];
            Vector2 screenPos = GetWorldToScreenEx((Vector3){ a.state.pos.x, 1.6f, a.state.pos.z }, camera, screenWidth, screenHeight);
            int barW = 34, barH = 5;
            int bx = (int)screenPos.x - barW / 2, by = (int)screenPos.y;
            DrawRectangle(bx - 1, by - 1, barW + 2, barH + 2, (Color){ 0, 0, 0, 150 });
            float frac = Clamp(a.health / HEALTH_MAX, 0.0f, 1.0f);
            Color healthCol = frac > 0.5f ? LIME : (frac > 0.2f ? GOLD : RED);
            DrawRectangle(bx, by, (int)(barW * frac), barH, a.alive ? healthCol : (Color){ 90, 90, 90, 180 });
        }

        // --- HUD --- (sizes/positions tuned for the 1280x800 internal
        // canvas - was 1000x600 with 13-15px text, which read as blocky/
        // "retro-by-accident" rather than a deliberate style. Bigger canvas
        // + bigger type, same raylib default font.)
        int panelW = 340, panelH = 118;
        DrawPanel(10, 10, panelW, panelH);
        DrawTextOutlined(TextFormat("Generation %d", pop.generation), 22, 18, 24, RAYWHITE);
        double elapsedShown = paused ? (pauseStartTime - genStartTime - totalPausedDuration) : (GetTime() - genStartTime - totalPausedDuration);
        if (settings.endMode == RaceEndMode::TIME_LIMIT) {
            DrawTextOutlined(TextFormat("Time %.1f / %.0fs", elapsedShown, settings.timeLimitSeconds), 22, 50, 18, (Color){ 200, 210, 230, 255 });
        } else {
            int leadLaps = 0;
            for (const auto& a : pop.agents) leadLaps = std::max(leadLaps, a.lapsCompleted);
            DrawTextOutlined(TextFormat("Lap %d / %d  (%.0fs)", leadLaps, settings.lapTarget, elapsedShown), 22, 50, 18, (Color){ 200, 210, 230, 255 });
        }
        DrawTextOutlined(TextFormat("Best-ever: %.1f", pop.bestEverFitness), 22, 72, 18, GOLD);
        DrawTextOutlined(TextFormat("Hit red/white wall: -%.0f fitness, -%.0f HP", WALL_HIT_PENALTY, HEALTH_DAMAGE_PER_HIT),
                          22, 96, 14, (Color){ 235, 120, 120, 255 });

        int boardW = 350, boardH = 30 + POP_SIZE * 28;
        DrawPanel(10, panelH + 20, boardW, boardH, GOLD);
        DrawTextOutlined("LEADERBOARD", 22, panelH + 28, 17, RAYWHITE);
        std::array<int, POP_SIZE> rank = { 0, 1, 2, 3, 4, 5 };
        std::sort(rank.begin(), rank.end(), [&](int a, int b) { return pop.agents[a].fitness > pop.agents[b].fitness; });
        for (int row = 0; row < POP_SIZE; row++) {
            int i = rank[row];
            const CarAgent& a = pop.agents[i];
            int y = panelH + 54 + row * 28;
            DrawRectangle(22, y + 3, 12, 12, a.alive ? CAR_COLORS[i] : (Color){ 90, 90, 90, 255 });
            const char* healthStr = a.alive ? TextFormat("hp %.0f", a.health) : "DEAD";
            DrawTextOutlined(TextFormat("%d. Car %d  %.1f  lap %d  %s", row + 1, i + 1, a.fitness, a.lapsCompleted, healthStr), 40, y, 16,
                              (i == target) ? RAYWHITE : (Color){ 190, 190, 190, 255 });
        }

        if (showSettings) {
            int sx = screenWidth - 380, sy = 10, sw = 360, sh = 178;
            DrawPanel(sx, sy, sw, sh);
            DrawTextOutlined("SETTINGS (TAB to close)", sx + 12, sy + 10, 20, RAYWHITE);
            DrawTextOutlined(TextFormat("Mouse sensitivity: %.2f  (-/+ to adjust)", settings.mouseSensitivity), sx + 12, sy + 42, 17, (Color){ 220, 220, 220, 255 });
            DrawTextOutlined(TextFormat("Invert Y: %s  (I to toggle)", settings.invertY ? "ON" : "OFF"), sx + 12, sy + 68, 17, (Color){ 220, 220, 220, 255 });
            const char* modeStr = settings.endMode == RaceEndMode::TIME_LIMIT ? "Time limit" : "Lap count";
            DrawTextOutlined(TextFormat("Generation end: %s  (M to toggle)", modeStr), sx + 12, sy + 94, 17, (Color){ 220, 220, 220, 255 });
            if (settings.endMode == RaceEndMode::TIME_LIMIT) {
                DrawTextOutlined(TextFormat("Time per generation: %.0fs  ([ / ] to adjust)", settings.timeLimitSeconds), sx + 12, sy + 120, 17, (Color){ 220, 220, 220, 255 });
            } else {
                DrawTextOutlined(TextFormat("Laps per generation: %d  ([ / ] to adjust)", settings.lapTarget), sx + 12, sy + 120, 17, (Color){ 220, 220, 220, 255 });
            }
            DrawTextOutlined("Right-drag: orbit camera | Scroll: zoom", sx + 12, sy + 150, 16, (Color){ 180, 200, 220, 255 });
        }

        // --- Garage: NFS-style shop screen showing what the brain is
        // buying and what it costs, not just that fitness numbers moved. ---
        if (showGarage) {
            int gx = (screenWidth - 980) / 2, gy = 110, gw = 980, gh = 540;
            DrawPanel(gx, gy, gw, gh, GOLD);
            DrawTextOutlined("GARAGE - AI upgrade decisions (G to close)", gx + 18, gy + 14, 24, RAYWHITE);
            DrawTextOutlined("Each car's brain spends its own coins - highest-scoring affordable part wins, every time it can afford one.",
                              gx + 18, gy + 44, 16, (Color){ 190, 200, 215, 255 });

            int colCar = gx + 18, colCoins = gx + 115, colEngine = gx + 230, colTires = gx + 430, colArmor = gx + 630, colPick = gx + 830;
            int headerY = gy + 76;
            DrawTextOutlined("Car", colCar, headerY, 16, GOLD);
            DrawTextOutlined("Coins", colCoins, headerY, 16, GOLD);
            DrawTextOutlined("Engine (next cost)", colEngine, headerY, 16, GOLD);
            DrawTextOutlined("Tires (next cost)", colTires, headerY, 16, GOLD);
            DrawTextOutlined("Armor (next cost)", colArmor, headerY, 16, GOLD);
            DrawTextOutlined("Last AI pick", colPick, headerY, 16, GOLD);

            auto tierCell = [](int tier, const UpgradeTierInfo* table) -> const char* {
                if (tier >= UPGRADE_TIER_COUNT) return TextFormat("T%d (MAX)", tier);
                return TextFormat("T%d (%d)", tier, table[tier].cost);
            };

            for (int row = 0; row < POP_SIZE; row++) {
                const CarAgent& a = pop.agents[row];
                int y = headerY + 30 + row * 32;
                Color rowCol = a.alive ? RAYWHITE : (Color){ 130, 130, 130, 255 };
                DrawRectangle(colCar, y + 3, 12, 12, a.alive ? CAR_COLORS[row] : (Color){ 90, 90, 90, 255 });
                DrawTextOutlined(TextFormat("Car %d", row + 1), colCar + 20, y, 16, rowCol);
                DrawTextOutlined(TextFormat("%d", a.coins), colCoins, y, 16, rowCol);
                DrawTextOutlined(tierCell(a.engineTier, ENGINE_TIERS), colEngine, y, 16, rowCol);
                DrawTextOutlined(tierCell(a.tireTier, TIRE_TIERS), colTires, y, 16, rowCol);
                DrawTextOutlined(tierCell(a.armorTier, ARMOR_TIERS), colArmor, y, 16, rowCol);
                DrawTextOutlined(a.lastUpgrade.c_str(), colPick, y, 16, rowCol);
            }

            int noteY = headerY + 30 + POP_SIZE * 32 + 18;
            DrawTextOutlined(TextFormat("Engine: +%.0f%% speed/accel per tier   Tires: +%.0f%% turn rate per tier   Armor: +%.0f max health per tier",
                                         ENGINE_TIER_BONUS * 100.0f, TIRE_TIER_BONUS * 100.0f, ARMOR_TIER_BONUS),
                              gx + 18, noteY, 16, (Color){ 190, 200, 215, 255 });
            DrawTextOutlined(TextFormat("Coins: +%d per lap, +%d per %.0f units traveled. Upgrades reset each generation UNLESS the brain buys the Keep Setup Pack (%d coins, one-shot).",
                                         COINS_PER_LAP, COINS_PER_DISTANCE_TICK, DISTANCE_PER_COIN_TICK, PERSIST_PACK_COST),
                              gx + 18, noteY + 22, 16, (Color){ 190, 200, 215, 255 });
            DrawTextOutlined("Keep Setup carries THIS slot's tiers into next generation - next gen's brain in that slot is still a mutated descendant, not the same brain.",
                              gx + 18, noteY + 44, 16, (Color){ 190, 200, 215, 255 });
        }

        if (paused) {
            DrawRectangle(0, 0, screenWidth, screenHeight, (Color){ 0, 0, 0, 130 });
            const char* label = "PAUSED";
            int fontSize = 56;
            int textW = MeasureText(label, fontSize);
            DrawTextOutlined(label, (screenWidth - textW) / 2, screenHeight / 2 - 50, fontSize, RAYWHITE);
        }

        DrawPanel(0, screenHeight - 36, screenWidth, 36);
        DrawTextOutlined("ESC pause | TAB settings | G garage | 1-6 follow car | 0 auto-follow | right-drag/scroll camera",
                          18, screenHeight - 28, 17, (Color){ 220, 220, 220, 255 });

        EndTextureMode();

        int winW = GetScreenWidth();
        int winH = GetScreenHeight();
        float fitScale = fminf((float)winW / screenWidth, (float)winH / screenHeight);
        float destW = screenWidth * fitScale;
        float destH = screenHeight * fitScale;
        float destX = (winW - destW) * 0.5f;
        float destY = (winH - destH) * 0.5f;

        BeginDrawing();
        ClearBackground(BLACK);
        BeginShaderMode(postShader);
        DrawTexturePro(sceneTarget.texture,
                        (Rectangle){ 0, 0, (float)screenWidth, -(float)screenHeight },
                        (Rectangle){ destX, destY, destW, destH },
                        (Vector2){ 0, 0 }, 0.0f, WHITE);
        EndShaderMode();
        EndDrawing();
    }

    UnloadModel(track.mesh);   // also frees roadTexture (UnloadMaterial frees non-default map textures)
    UnloadModel(groundModel);  // also frees grassTexture, same reason
    UnloadModel(wallModel);
    UnloadShader(postShader);
    UnloadShader(litShader);
    UnloadRenderTexture(sceneTarget);
    CloseWindow();
    return 0;
}
