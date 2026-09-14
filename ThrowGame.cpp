#include <raylib.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

// Throwing game combining the physics engine and the trained model.
// Press SPACE for a new random throw. Run this from build/ (./throw_game),
// same as throw_sim and beam_sim.

static bool runCommand(const std::string& cmd, float& outX, float& outY) {
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
    float realX, realY;
    float predX, predY;
    bool haveReal, havePred;
};

static ThrowState runThrow() {
    ThrowState s;
    s.vx0 = randRange(-15.0f, 15.0f);
    s.vy0 = randRange(5.0f, 25.0f);
    s.mass = randRange(0.5f, 5.0f);

    s.haveReal = runCommand(
        "./throw_sim " + std::to_string(s.vx0) + " " + std::to_string(s.vy0) + " " + std::to_string(s.mass),
        s.realX, s.realY);

    s.havePred = runCommand(
        "python3 ../predict.py " + std::to_string(s.vx0) + " " + std::to_string(s.vy0) + " " + std::to_string(s.mass),
        s.predX, s.predY);

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

        BeginDrawing();
        ClearBackground(SKYBLUE);

        UpdateCamera(&camera, CAMERA_ORBITAL);
        BeginMode3D(camera);

        DrawPlane((Vector3){ 0.0f, 0.0f, 0.0f }, (Vector2){ 200.0f, 20.0f }, DARKGREEN);
        DrawGrid(20, 10.0f);

        if (state.haveReal) {
            DrawSphere((Vector3){ state.realX, state.realY + 1.0f, 0.0f }, 1.0f, RED);
        }
        if (state.havePred) {
            DrawSphere((Vector3){ state.predX, state.predY + 1.0f, 0.0f }, 1.0f, GOLD);
        }
        if (state.haveReal && state.havePred) {
            DrawLine3D((Vector3){ state.realX, state.realY + 1.0f, 0.0f },
                       (Vector3){ state.predX, state.predY + 1.0f, 0.0f }, WHITE);
        }

        EndMode3D();

        DrawText(TextFormat("Throw: vx=%.1f vy=%.1f mass=%.1f", state.vx0, state.vy0, state.mass), 10, 10, 20, BLACK);
        if (state.haveReal) {
            DrawText(TextFormat("Real landing (red):  x=%.2f y=%.2f", state.realX, state.realY), 10, 35, 20, MAROON);
        } else {
            DrawText("Real landing: throw_sim call failed (run this from build/)", 10, 35, 20, MAROON);
        }
        if (state.havePred) {
            DrawText(TextFormat("AI predicted (gold): x=%.2f y=%.2f", state.predX, state.predY), 10, 60, 20, ORANGE);
        } else {
            DrawText("AI prediction: predict.py call failed (need venv with torch active)", 10, 60, 20, ORANGE);
        }

        if (state.haveReal && state.havePred) {
            float dx = state.realX - state.predX;
            float dy = state.realY - state.predY;
            float error = sqrtf(dx * dx + dy * dy);
            DrawText(TextFormat("AI error: %.3f units", error), 10, 90, 20, DARKPURPLE);
        }

        DrawText("Press SPACE for a new throw", 10, screenHeight - 30, 20, GRAY);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
