#include <raylib.h>
#include <raymath.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Throwing game combining the physics engine and the trained model.
// Press SPACE for a new random throw. Run this from build/ (./throw_game),
// same as throw_sim and beam_sim.

static const float SIM_DT = 0.01f;       // must match ThrowSim.cpp's dt
static const float PLAYBACK_SPEED = 2.5f; // watch the flight faster than real time

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

static float randRange(float lo, float hi) {
    return lo + (hi - lo) * (float)GetRandomValue(0, 10000) / 10000.0f;
}

struct ThrowState {
    float vx0, vy0, mass;
    std::vector<Vector2> realTrajectory;
    bool haveReal;
    float predX, predY;
    bool havePred;
    double startTime;
};

static ThrowState runThrow() {
    ThrowState s;
    s.vx0 = randRange(-15.0f, 15.0f);
    s.vy0 = randRange(5.0f, 25.0f);
    s.mass = randRange(0.5f, 5.0f);

    s.haveReal = runCommand(
        "./throw_sim --trajectory " + std::to_string(s.vx0) + " " + std::to_string(s.vy0) + " " + std::to_string(s.mass),
        s.realTrajectory);

    s.havePred = runFinal(
        "python3 ../predict.py " + std::to_string(s.vx0) + " " + std::to_string(s.vy0) + " " + std::to_string(s.mass),
        s.predX, s.predY);

    s.startTime = GetTime();
    return s;
}

int main() {
    SetRandomSeed((unsigned int)GetTime());

    const int screenWidth = 1000;
    const int screenHeight = 600;
    InitWindow(screenWidth, screenHeight, "BeamSim3D - Throw Game");

    ThrowState state = runThrow();

    Camera camera = { 0 };
    camera.position = (Vector3){ 0.0f, 40.0f, 130.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            state = runThrow();
        }

        float realDuration = state.haveReal ? (state.realTrajectory.size() - 1) * SIM_DT : 0.0f;
        float elapsed = (float)(GetTime() - state.startTime) * PLAYBACK_SPEED;
        float animT = realDuration > 0.0f ? Clamp(elapsed / realDuration, 0.0f, 1.0f) : 1.0f;
        bool animDone = animT >= 1.0f;

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

        BeginDrawing();
        ClearBackground(SKYBLUE);

        UpdateCamera(&camera, CAMERA_ORBITAL);
        BeginMode3D(camera);

        DrawPlane((Vector3){ 0.0f, 0.0f, 0.0f }, (Vector2){ 200.0f, 20.0f }, DARKGREEN);
        DrawGrid(20, 10.0f);

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

        DrawText(TextFormat("Throw: vx=%.1f vy=%.1f mass=%.1f", state.vx0, state.vy0, state.mass), 10, 10, 20, BLACK);
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

        DrawText("Press SPACE for a new throw", 10, screenHeight - 30, 20, GRAY);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
