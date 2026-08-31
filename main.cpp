#include "Physics.h"
#include <raylib.h>

int main() {
    const int screenWidth = 800;
    const int screenHeight = 450;

    InitWindow(screenWidth, screenHeight, "BeamSim3D");

    SoftBody vehicle;
    vehicle.loadConfig("src/assets/vehicle.json");

    Image terrainImage = LoadImage("src/assets/terrain.png");
    if (terrainImage.data == NULL) {
        terrainImage = GenImagePerlinNoise(64, 64, 0, 0, 1.0f);
    }

    Texture2D terrainTexture = LoadTextureFromImage(terrainImage);
    UnloadImage(terrainImage);

    Sound engineSound = LoadSound("src/assets/engine.wav");
    bool audioLoaded = engineSound.frameCount > 0;

    Camera camera = { 0 };
    camera.position = (Vector3){ 0.0f, 10.0f, 10.0f };
    camera.target = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.type = CAMERA_PERSPECTIVE;

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(SKYBLUE);

        BeginMode3D(camera);

        // Draw the terrain
        DrawTextureEx(terrainTexture, (Vector2){ 0, 0 }, 0.0f, 1.0f, WHITE);

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

        EndDrawing();
    }

    UnloadTexture(terrainTexture);
    if (audioLoaded) {
        UnloadSound(engineSound);
    }

    CloseWindow();

    return 0;
}
