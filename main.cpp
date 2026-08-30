#include "raylib.h"
#include "Physics.h"

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
    const int numNodes = 8;
    const float nodeSpacing = 1.0f;

    for (int i = 0; i < numNodes; i++) {
        Node3D node = { 0 };
        node.position[0] = (i % 2) * nodeSpacing;
        node.position[1] = (i / 4) * nodeSpacing + 5.0f; // Start higher to allow crumpling
        node.position[2] = (i / 2 % 2) * nodeSpacing;
        node.mass = 1.0f;
        softBody.nodes.push_back(node);
    }

    for (int i = 0; i < numNodes; i++) {
        for (int j = i + 1; j < numNodes; j++) {
            float distance = std::sqrt(
                std::pow(softBody.nodes[j].position[0] - softBody.nodes[i].position[0], 2) +
                std::pow(softBody.nodes[j].position[1] - softBody.nodes[i].position[1], 2) +
                std::pow(softBody.nodes[j].position[2] - softBody.nodes[i].position[2], 2)
            );

            if (distance < 1.1f) {
                Beam3D beam = { &softBody.nodes[i], &softBody.nodes[j], distance, 100.0f, 0.1f, 10.0f, 1000.0f, false };
                softBody.beams.push_back(beam);
            }
        }
    }

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
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

        EndMode3D();

        EndDrawing();

        softBody.update(GetFrameTime());
    }

    CloseWindow();

    return 0;
}
