#include "Physics.h"
#include <raylib.h>

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

    SoftBody vehicle;
    vehicle.loadConfig("vehicle.json");

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();

            ClearBackground(RAYWHITE);

            BeginMode3D(camera);

                // Draw the vehicle's nodes, beams, and wheels
                for (const auto& node : vehicle.nodes) {
                    DrawSphere((Vector3){ node.position[0], node.position[1], node.position[2] }, 0.1f, RED);
                }

                for (const auto& beam : vehicle.beams) {
                    DrawLine3D((Vector3){ beam.node1->position[0], beam.node1->position[1], beam.node1->position[2] },
                               (Vector3){ beam.node2->position[0], beam.node2->position[1], beam.node2->position[2] }, BLUE);
                }

                for (const auto& wheel : vehicle.wheels) {
                    DrawSphere((Vector3){ wheel.node->position[0], wheel.node->position[1], wheel.node->position[2] }, wheel.radius, GREEN);
                }

            EndMode3D();

            // Telemetry HUD
            Vector3 velocity = Vector3Subtract(vehicle.chassisCenter, (Vector3){ vehicle.chassisCenter.x, vehicle.chassisCenter.y - 0.1f, vehicle.chassisCenter.z });
            float speed = Vector3Length(velocity);
            std::string speedText = "Speed: " + std::to_string(static_cast<int>(speed)) + " m/s";
            std::string rpmText = "RPM: " + std::to_string(static_cast<int>(vehicle.engine.rpm));
            std::string gearText = "Gear: " + std::to_string(vehicle.transmission.currentGear);
            int brokenBeams = 0;
            for (const auto& beam : vehicle.beams) {
                if (beam.isBroken) {
                    brokenBeams++;
                }
            }
            std::string beamsText = "Beams: " + std::to_string(brokenBeams) + "/" + std::to_string(vehicle.beams.size());

            DrawText(speedText.c_str(), 10, 10, 20, BLACK);
            DrawText(rpmText.c_str(), 10, 40, 20, BLACK);
            DrawText(gearText.c_str(), 10, 70, 20, BLACK);
            DrawText(beamsText.c_str(), 10, 100, 20, BLACK);

        EndDrawing();

        vehicle.update(GetFrameTime());

        // Input Handling
        if (IsKeyPressed(KEY_R)) {
            vehicle.resetVehicle();
        }
    }

    CloseWindow();

    return 0;
}
