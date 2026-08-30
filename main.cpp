#include "raylib.h"
#include "Physics.h"
#include <vector>
#include <cstdlib>
#include <ctime>

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
                DrawLine3D(
                    (Vector3){ beam.node1->position[0], beam.node1->position[1], beam.node1->position[2] },
                    (Vector3){ beam.node2->position[0], beam.node2->position[1], beam.node2->position[2] },
                    BLACK
                );
            }
        }

        for (const auto& barrier : barriers) {
            DrawCube(barrier.position, barrier.size, barrier.size, barrier.size, RED);
        }

        EndMode3D();

        EndDrawing();

        softBody.update(GetFrameTime());

        // Camera tracking
        Vector3 chassisCenter = { 0.0f, 0.0f, 0.0f };
        for (int i = 0; i < 8; i++) {
            chassisCenter.x += softBody.nodes[i].position[0];
            chassisCenter.y += softBody.nodes[i].position[1];
            chassisCenter.z += softBody.nodes[i].position[2];
        }
        chassisCenter.x /= 8.0f;
        chassisCenter.y /= 8.0f;
        chassisCenter.z /= 8.0f;

        camera.position = (Vector3){ chassisCenter.x + 10.0f, chassisCenter.y + 10.0f, chassisCenter.z + 10.0f };
        camera.target = chassisCenter;
    }

    CloseWindow();

    return 0;
}
