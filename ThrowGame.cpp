#include <raylib.h>
#include <cstdio>
#include <cstdlib>
#include <string>

// Small first step towards a game combining the physics engine and the
// trained model: run one throw, ask throw_sim for the real landing point
// and predict.py for the AI's guess, show both in the window. Run this
// from build/ (./throw_game), same as throw_sim and beam_sim.

static bool runCommand(const std::string& cmd, float& outX, float& outY) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;

    char buffer[256] = {0};
    bool gotLine = fgets(buffer, sizeof(buffer), pipe) != nullptr;
    int status = pclose(pipe);

    if (!gotLine || status != 0) return false;

    return sscanf(buffer, "%f,%f", &outX, &outY) == 2;
}

int main() {
    const float vx0 = 15.0f, vy0 = 25.0f, mass = 2.5f;

    float realX = 0.0f, realY = 0.0f;
    bool haveReal = runCommand(
        "./throw_sim " + std::to_string(vx0) + " " + std::to_string(vy0) + " " + std::to_string(mass),
        realX, realY);

    float predX = 0.0f, predY = 0.0f;
    bool havePred = runCommand(
        "python3 ../predict.py " + std::to_string(vx0) + " " + std::to_string(vy0) + " " + std::to_string(mass),
        predX, predY);

    const int screenWidth = 1000;
    const int screenHeight = 600;
    InitWindow(screenWidth, screenHeight, "BeamSim3D - Throw Game (step 1)");

    Camera camera = { 0 };
    camera.position = (Vector3){ 0.0f, 40.0f, 130.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(SKYBLUE);

        UpdateCamera(&camera, CAMERA_ORBITAL);
        BeginMode3D(camera);

        DrawPlane((Vector3){ 0.0f, 0.0f, 0.0f }, (Vector2){ 200.0f, 20.0f }, DARKGREEN);
        DrawGrid(20, 10.0f);

        if (haveReal) {
            DrawSphere((Vector3){ realX, realY + 1.0f, 0.0f }, 1.0f, RED);
        }
        if (havePred) {
            DrawSphere((Vector3){ predX, predY + 1.0f, 0.0f }, 1.0f, GOLD);
        }

        EndMode3D();

        DrawText(TextFormat("Throw: vx=%.1f vy=%.1f mass=%.1f", vx0, vy0, mass), 10, 10, 20, BLACK);
        if (haveReal) {
            DrawText(TextFormat("Real landing (red):      x=%.2f y=%.2f", realX, realY), 10, 35, 20, MAROON);
        } else {
            DrawText("Real landing: throw_sim call failed (run this from build/)", 10, 35, 20, MAROON);
        }
        if (havePred) {
            DrawText(TextFormat("AI predicted (gold):     x=%.2f y=%.2f", predX, predY), 10, 60, 20, ORANGE);
        } else {
            DrawText("AI prediction: predict.py call failed (need venv with torch active)", 10, 60, 20, ORANGE);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
