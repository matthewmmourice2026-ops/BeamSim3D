#include "raylib.h"
#include "Physics.h"
#include <vector>
#include <cstdlib>
#include <ctime>
#include <sstream>

struct Barrier {
    Vector3 position;
    float size;
};

int main() {
    const int screenWidth = 800;
    const int screenHeight = 600;

    InitWindow(screenWidth, screenHeight, "Soft Body Physics Prototype");

    Camera camera = { 0 };
    camera.position = (Vector3){ 10.0f, 10.0f, 10.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    SoftBody softBody;
    softBody.loadConfig("vehicle.json");

    std::vector<Barrier> barriers;
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < 10; i++) {
        Barrier barrier;
        barrier.position = (Vector3){ static_cast<float>(rand() % 20 - 10), static_cast<float>(rand() % 5), static_cast<float>(rand() % 20 - 10) };
        barrier.size = static_cast<float>(rand() % 2 + 1);
        barriers.push_back(barrier);
    }

    SetTargetFPS(60);

    bool isMouseDragging = false;
    float lastMouseX = 0.0f;
    float lastMouseY = 0.0f;
    float cameraDistance = 10.0f;

    while (!WindowShouldClose()) {
        float accelerationForce = 0.0f;
        float steeringTorque = 0.0f;

        if (IsKeyDown(KEY_W)) accelerationForce = 10.0f;
        if (IsKeyDown(KEY_S)) accelerationForce = -10.0f;
        if (IsKeyDown(KEY_A)) steeringTorque = -0.1f;
        if (IsKeyDown(KEY_D)) steeringTorque = 0.1f;

        if (IsKeyDown(KEY_UP)) {
            for (auto& beam : softBody.beams) {
                beam.stiffness += 10.0f;
            }
        }
        if (IsKeyDown(KEY_DOWN)) {
            for (auto& beam : softBody.beams) {
                beam.stiffness -= 10.0f;
            }
        }

        if (IsKeyDown(KEY_R)) {
            softBody.reset();
        }

        for (auto& wheel : softBody.wheels) {
            if (wheel.isDriven) {
                wheel.torque = accelerationForce;
            }
        }

        for (auto& wheel : softBody.wheels) {
            if (!wheel.isDriven) {
                wheel.node->theta += steeringTorque;
            }
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        BeginMode3D(camera);

        for (const auto& beam : softBody.beams) {
            if (!beam.isBroken) {
                float distance = std::sqrt(
                    std::pow(beam.node2->position[0] - beam.node1->position[0], 2) +
                    std::pow(beam.node2->position[1] - beam.node1->position[1], 2) +
                    std::pow(beam.node2->position[2] - beam.node1->position[2], 2)
                );

                float stress = std::abs(distance - beam.restLength);
                Color beamColor = Color{ 0, 255, 0, 255 }; // Green
                if (stress > beam.breakThreshold * 0.5f) {
                    beamColor = Color{ 255, 0, 0, 255 }; // Red
                } else if (stress > beam.breakThreshold * 0.25f) {
                    beamColor = Color{ 255, 255, 0, 255 }; // Yellow
                }

                DrawLine3D(
                    (Vector3){ beam.node1->position[0], beam.node1->position[1], beam.node1->position[2] },
                    (Vector3){ beam.node2->position[0], beam.node2->position[1], beam.node2->position[2] },
                    beamColor
                );
            }
        }

        for (const auto& barrier : barriers) {
            DrawCube(barrier.position, barrier.size, barrier.size, barrier.size, RED);
        }

        EndMode3D();

        // Telemetry HUD
        DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, 20, BLACK);
        DrawText(TextFormat("Intact Beams: %i", std::count_if(softBody.beams.begin(), softBody.beams.end(), [](const Beam3D& beam) { return !beam.isBroken; })), 10, 40, 20, BLACK);
        DrawText(TextFormat("Broken Beams: %i", std::count_if(softBody.beams.begin(), softBody.beams.end(), [](const Beam3D& beam) { return beam.isBroken; })), 10, 70, 20, BLACK);
        DrawText(TextFormat("Speed: %.2f m/s", std::sqrt(std::pow(softBody.chassisCenter.x, 2) + std::pow(softBody.chassisCenter.y, 2) + std::pow(softBody.chassisCenter.z, 2))), 10, 100, 20, BLACK);

        EndDrawing();

        softBody.update(GetFrameTime());

        // Camera controls
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            isMouseDragging = true;
            lastMouseX = GetMouseX();
            lastMouseY = GetMouseY();
        }

        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
            isMouseDragging = false;
        }

        if (isMouseDragging) {
            float deltaX = GetMouseX() - lastMouseX;
            float deltaY = GetMouseY() - lastMouseY;

            camera.yaw -= deltaX * 0.5f;
            camera.pitch -= deltaY * 0.5f;

            lastMouseX = GetMouseX();
            lastMouseY = GetMouseY();
        }

        if (IsMouseWheelMoved()) {
            cameraDistance -= GetMouseWheelMove() * 0.5f;
            cameraDistance = std::max(cameraDistance, 1.0f);
        }

        camera.position = (Vector3){ softBody.chassisCenter.x + cameraDistance * std::cos(camera.yaw) * std::cos(camera.pitch),
                                     softBody.chassisCenter.y + cameraDistance * std::sin(camera.pitch),
                                     softBody.chassisCenter.z + cameraDistance * std::sin(camera.yaw) * std::cos(camera.pitch) };
        camera.target = softBody.chassisCenter;
    }

    CloseWindow();

    return 0;
}
