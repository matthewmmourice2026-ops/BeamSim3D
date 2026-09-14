#include <raylib.h>
#include <raymath.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <limits>
#include "Terrain.h"

// Throwing game combining the physics engine and the trained model.
// Arrow keys adjust velocity, [ and ] adjust mass, SPACE throws.
// Run this from build/ (./throw_game), same as throw_sim and beam_sim.

static const float SIM_DT = 0.01f;        // must match ThrowSim.cpp's dt
static const float PLAYBACK_SPEED = 2.5f; // watch the flight faster than real time

static const float VX_MIN = -15.0f, VX_MAX = 15.0f;
static const float VY_MIN = 5.0f, VY_MAX = 25.0f;
static const float MASS_MIN = 0.5f, MASS_MAX = 5.0f;

static bool runCommand(const std::string& cmd, std::vector<Vector2>& outPoints) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;

    char buffer[256] = {0};
    while (fgets(buffer, sizeof(buffer), pipe)) {
        float x, y;
        if (sscanf(buffer, "%f,%f", &x, &y) == 2) {
            outPoints.push_back({ x, y });
        }
    }
    int status = pclose(pipe);
    return status == 0 && !outPoints.empty();
}

static bool runFinal(const std::string& cmd, float& outX, float& outY) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;

    char buffer[256] = {0};
    bool gotLine = fgets(buffer, sizeof(buffer), pipe) != nullptr;
    int status = pclose(pipe);

    if (!gotLine || status != 0) return false;
    return sscanf(buffer, "%f,%f", &outX, &outY) == 2;
}

struct ThrowState {
    float vx0, vy0, mass;
    std::vector<Vector2> realTrajectory;
    bool haveReal;
    float predX, predY;
    bool havePred;
    double startTime;
};

static ThrowState runThrow(float vx0, float vy0, float mass) {
    ThrowState s;
    s.vx0 = vx0;
    s.vy0 = vy0;
    s.mass = mass;

    s.haveReal = runCommand(
        "./throw_sim --trajectory " + std::to_string(vx0) + " " + std::to_string(vy0) + " " + std::to_string(mass),
        s.realTrajectory);

    s.havePred = runFinal(
        "python3 ../predict.py " + std::to_string(vx0) + " " + std::to_string(vy0) + " " + std::to_string(mass),
        s.predX, s.predY);

    s.startTime = GetTime();
    return s;
}

// Terrain visual mesh, a ribbon following terrainHeight(x) that's flat
// across z (matches the physics, which never varies with z either).
static std::vector<Vector3> buildTerrainMesh() {
    std::vector<Vector3> points;
    const float halfWidth = 15.0f;
    const float xMin = -110.0f, xMax = 110.0f, step = 2.0f;

    for (float x = xMin; x <= xMax; x += step) {
        float h = terrainHeight(x);
        points.push_back({ x, h, -halfWidth });
        points.push_back({ x, h, halfWidth });
    }
    return points;
}

int main() {
    SetRandomSeed((unsigned int)GetTime());

    const int screenWidth = 1000;
    const int screenHeight = 600;
    InitWindow(screenWidth, screenHeight, "BeamSim3D - Throw Game");

    std::vector<Vector3> terrainMesh = buildTerrainMesh();

    float vx = 10.0f, vy = 15.0f, mass = 2.0f;
    ThrowState state = runThrow(vx, vy, mass);
    bool scored = false;

    int throwCount = 0;
    float totalError = 0.0f;
    float bestError = std::numeric_limits<float>::infinity();
    float worstError = 0.0f;
    int streak = 0; // consecutive throws under 1 unit of error

    std::vector<Vector3> realTrail, predTrail;

    Vector3 followTarget = { 0.0f, 1.0f, 0.0f };
    float orbitYaw = 0.0f;
    const float orbitRadius = 60.0f, orbitHeight = 35.0f;

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        float realDuration = state.haveReal ? (state.realTrajectory.size() - 1) * SIM_DT : 0.0f;
        float elapsed = (float)(GetTime() - state.startTime) * PLAYBACK_SPEED;
        float animT = realDuration > 0.0f ? Clamp(elapsed / realDuration, 0.0f, 1.0f) : 1.0f;
        bool animDone = animT >= 1.0f;
        bool flying = !animDone;

        // Adjust throw parameters only between throws
        if (animDone) {
            float dt = GetFrameTime();
            if (IsKeyDown(KEY_RIGHT)) vx = Clamp(vx + 10.0f * dt, VX_MIN, VX_MAX);
            if (IsKeyDown(KEY_LEFT)) vx = Clamp(vx - 10.0f * dt, VX_MIN, VX_MAX);
            if (IsKeyDown(KEY_UP)) vy = Clamp(vy + 10.0f * dt, VY_MIN, VY_MAX);
            if (IsKeyDown(KEY_DOWN)) vy = Clamp(vy - 10.0f * dt, VY_MIN, VY_MAX);
            if (IsKeyDown(KEY_RIGHT_BRACKET)) mass = Clamp(mass + 2.0f * dt, MASS_MIN, MASS_MAX);
            if (IsKeyDown(KEY_LEFT_BRACKET)) mass = Clamp(mass - 2.0f * dt, MASS_MIN, MASS_MAX);

            if (IsKeyPressed(KEY_SPACE)) {
                state = runThrow(vx, vy, mass);
                scored = false;
                realTrail.clear();
                predTrail.clear();
            }
        }

        Vector2 realPos = { 0.0f, 1.0f };
        if (state.haveReal) {
            int idx = (int)(animT * (state.realTrajectory.size() - 1));
            realPos = state.realTrajectory[idx];
        }

        Vector2 predPos = { 0.0f, 1.0f };
        if (state.havePred) {
            // Not a physically simulated path (the model only predicts the
            // endpoint), just a parabolic arc for visual flight, timed to
            // land alongside the real ball.
            float arcHeight = 8.0f;
            predPos.x = Lerp(0.0f, state.predX, animT);
            predPos.y = Lerp(1.0f, state.predY, animT) + arcHeight * sinf(3.14159265f * animT);
        }

        if (flying) {
            if (state.haveReal) realTrail.push_back({ realPos.x, realPos.y + 1.0f, 0.0f });
            if (state.havePred) predTrail.push_back({ predPos.x, predPos.y + 1.0f, 0.0f });
        }

        if (animDone && !scored && state.haveReal && state.havePred) {
            float dx = state.realTrajectory.back().x - state.predX;
            float dy = state.realTrajectory.back().y - state.predY;
            float error = sqrtf(dx * dx + dy * dy);
            throwCount++;
            totalError += error;
            if (error < bestError) bestError = error;
            if (error > worstError) worstError = error;
            streak = (error < 1.0f) ? streak + 1 : 0;
            scored = true;
        }

        // Camera follows the real ball while it's flying, otherwise settles
        // over the landing spot. Manual orbit instead of raylib's built-in
        // CAMERA_ORBITAL so it can smoothly track a moving target.
        Vector3 desiredTarget = flying
            ? (Vector3){ realPos.x, realPos.y, 0.0f }
            : (Vector3){ state.haveReal ? state.realTrajectory.back().x : 0.0f,
                         state.haveReal ? state.realTrajectory.back().y : 1.0f, 0.0f };
        followTarget = Vector3Lerp(followTarget, desiredTarget, 0.08f);
        orbitYaw += 0.15f * GetFrameTime();

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

        BeginDrawing();
        ClearBackground(SKYBLUE);

        BeginMode3D(camera);

        DrawTriangleStrip3D(terrainMesh.data(), (int)terrainMesh.size(), DARKGREEN);

        for (size_t i = 1; i < realTrail.size(); i++) {
            DrawLine3D(realTrail[i - 1], realTrail[i], MAROON);
        }
        for (size_t i = 1; i < predTrail.size(); i++) {
            DrawLine3D(predTrail[i - 1], predTrail[i], ORANGE);
        }

        if (state.haveReal) {
            DrawSphere((Vector3){ realPos.x, realPos.y + 1.0f, 0.0f }, 1.0f, RED);
        }
        if (state.havePred) {
            DrawSphere((Vector3){ predPos.x, predPos.y + 1.0f, 0.0f }, 1.0f, GOLD);
        }
        if (animDone && state.haveReal && state.havePred) {
            DrawLine3D((Vector3){ state.realTrajectory.back().x, state.realTrajectory.back().y + 1.0f, 0.0f },
                       (Vector3){ state.predX, state.predY + 1.0f, 0.0f }, WHITE);
        }

        EndMode3D();

        if (animDone) {
            DrawText(TextFormat("Throw: vx=%.1f vy=%.1f mass=%.1f  (arrows to adjust, [ ] for mass)", vx, vy, mass), 10, 10, 20, BLACK);
        } else {
            DrawText(TextFormat("In flight: vx=%.1f vy=%.1f mass=%.1f", state.vx0, state.vy0, state.mass), 10, 10, 20, BLACK);
        }

        if (state.haveReal) {
            DrawText(TextFormat("Real landing (red):  x=%.2f y=%.2f", state.realTrajectory.back().x, state.realTrajectory.back().y), 10, 35, 20, MAROON);
        } else {
            DrawText("Real landing: throw_sim call failed (run this from build/)", 10, 35, 20, MAROON);
        }
        if (state.havePred) {
            DrawText(TextFormat("AI predicted (gold): x=%.2f y=%.2f", state.predX, state.predY), 10, 60, 20, ORANGE);
        } else {
            DrawText("AI prediction: predict.py call failed (need venv with torch active)", 10, 60, 20, ORANGE);
        }

        if (animDone && state.haveReal && state.havePred) {
            float dx = state.realTrajectory.back().x - state.predX;
            float dy = state.realTrajectory.back().y - state.predY;
            float error = sqrtf(dx * dx + dy * dy);
            DrawText(TextFormat("AI error: %.3f units", error), 10, 90, 20, DARKPURPLE);
        } else if (!animDone) {
            DrawText("In flight...", 10, 90, 20, GRAY);
        }

        if (throwCount > 0) {
            DrawText(TextFormat("Throws: %d   Avg error: %.3f   Best: %.3f   Worst: %.3f   Streak (<1.0): %d",
                                 throwCount, totalError / throwCount, bestError, worstError, streak),
                      10, 120, 18, DARKGRAY);
        }

        DrawText("SPACE to throw", 10, screenHeight - 30, 20, GRAY);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
