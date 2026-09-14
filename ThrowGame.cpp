#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <limits>
#include "Terrain.h"
#include "ThrowRanges.h"

// Throwing game combining the physics engine and the trained model.
// Arrow keys adjust velocity, [ and ] adjust mass, SPACE throws.
// Run this from build/ (./throw_game), same as throw_sim and beam_sim.

static const float SIM_DT = 0.01f;        // must match ThrowSim.cpp's dt
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

// Terrain as a proper lit Model instead of an unlit immediate-mode strip.
// Vertex normals come from the analytic derivative of terrainHeight, not
// guessed, so the lighting shader gets real slope information.
static Model buildTerrainModel() {
    const float halfWidth = 15.0f;
    const float xMin = -110.0f, xMax = 110.0f, step = 2.0f;

    std::vector<float> xs;
    for (float x = xMin; x <= xMax; x += step) xs.push_back(x);
    int sampleCount = (int)xs.size();
    int vertexCount = sampleCount * 2;
    int triangleCount = (sampleCount - 1) * 2;

    Mesh mesh = { 0 };
    mesh.vertexCount = vertexCount;
    mesh.triangleCount = triangleCount;
    mesh.vertices = (float*)MemAlloc(vertexCount * 3 * sizeof(float));
    mesh.normals = (float*)MemAlloc(vertexCount * 3 * sizeof(float));
    mesh.texcoords = (float*)MemAlloc(vertexCount * 2 * sizeof(float));
    mesh.colors = (unsigned char*)MemAlloc(vertexCount * 4 * sizeof(unsigned char));
    mesh.indices = (unsigned short*)MemAlloc(triangleCount * 3 * sizeof(unsigned short));

    for (int i = 0; i < sampleCount; i++) {
        float x = xs[i];
        float h = terrainHeight(x);
        float slope = terrainSlope(x);
        Vector3 n = Vector3Normalize((Vector3){ -slope, 1.0f, 0.0f });

        for (int side = 0; side < 2; side++) {
            int v = i * 2 + side;
            float z = side == 0 ? -halfWidth : halfWidth;

            mesh.vertices[v * 3 + 0] = x;
            mesh.vertices[v * 3 + 1] = h;
            mesh.vertices[v * 3 + 2] = z;

            mesh.normals[v * 3 + 0] = n.x;
            mesh.normals[v * 3 + 1] = n.y;
            mesh.normals[v * 3 + 2] = n.z;

            mesh.texcoords[v * 2 + 0] = (float)i / (sampleCount - 1);
            mesh.texcoords[v * 2 + 1] = (float)side;

            mesh.colors[v * 4 + 0] = 255;
            mesh.colors[v * 4 + 1] = 255;
            mesh.colors[v * 4 + 2] = 255;
            mesh.colors[v * 4 + 3] = 255;
        }
    }

    int idx = 0;
    for (int i = 0; i < sampleCount - 1; i++) {
        unsigned short bottomA = i * 2, topA = i * 2 + 1;
        unsigned short bottomB = (i + 1) * 2, topB = (i + 1) * 2 + 1;

        mesh.indices[idx++] = bottomA; mesh.indices[idx++] = topA; mesh.indices[idx++] = bottomB;
        mesh.indices[idx++] = topA;    mesh.indices[idx++] = topB; mesh.indices[idx++] = bottomB;
    }

    UploadMesh(&mesh, false);
    return LoadModelFromMesh(mesh);
}

int main() {
    SetRandomSeed((unsigned int)GetTime());

    const int screenWidth = 1000;
    const int screenHeight = 600;
    InitWindow(screenWidth, screenHeight, "BeamSim3D - Throw Game");

    // --- Lighting setup ---
    Shader litShader = LoadShader("../Shaders/lighting.vs", "../Shaders/lighting.fs");
    int locLightDir = GetShaderLocation(litShader, "lightDir");
    int locLightColor = GetShaderLocation(litShader, "lightColor");
    int locAmbientColor = GetShaderLocation(litShader, "ambientColor");
    int locViewPos = GetShaderLocation(litShader, "viewPos");
    int locShininess = GetShaderLocation(litShader, "shininess");

    Vector3 lightDir = Vector3Normalize((Vector3){ -0.4f, -1.0f, -0.35f });
    Vector3 lightColor = { 1.0f, 0.97f, 0.9f };
    Vector3 ambientColor = { 0.28f, 0.32f, 0.36f };
    float shininess = 24.0f;
    SetShaderValue(litShader, locLightDir, &lightDir, SHADER_UNIFORM_VEC3);
    SetShaderValue(litShader, locLightColor, &lightColor, SHADER_UNIFORM_VEC3);
    SetShaderValue(litShader, locAmbientColor, &ambientColor, SHADER_UNIFORM_VEC3);
    SetShaderValue(litShader, locShininess, &shininess, SHADER_UNIFORM_FLOAT);

    Model terrainModel = buildTerrainModel();
    terrainModel.materials[0].shader = litShader;
    terrainModel.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = (Color){ 60, 120, 55, 255 };

    Model sphereModel = LoadModelFromMesh(GenMeshSphere(1.0f, 24, 24));
    sphereModel.materials[0].shader = litShader;

    Shader postShader = LoadShader(0, "../Shaders/postprocess.fs");
    RenderTexture2D sceneTarget = LoadRenderTexture(screenWidth, screenHeight);

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

        SetShaderValue(litShader, locViewPos, &camera.position, SHADER_UNIFORM_VEC3);

        // --- Draw the scene into an offscreen texture ---
        BeginTextureMode(sceneTarget);
        ClearBackground(SKYBLUE);

        DrawRectangleGradientV(0, 0, screenWidth, screenHeight,
                                (Color){ 90, 150, 230, 255 }, (Color){ 200, 225, 245, 255 });

        BeginMode3D(camera);

        DrawModel(terrainModel, (Vector3){ 0, 0, 0 }, 1.0f, WHITE);

        for (size_t i = 1; i < realTrail.size(); i++) {
            DrawLine3D(realTrail[i - 1], realTrail[i], MAROON);
        }
        for (size_t i = 1; i < predTrail.size(); i++) {
            DrawLine3D(predTrail[i - 1], predTrail[i], ORANGE);
        }

        // Blob shadows under both balls, grounded at the terrain height
        // beneath their current x position.
        if (state.haveReal) {
            float groundY = terrainHeight(realPos.x);
            DrawCylinder((Vector3){ realPos.x, groundY + 0.03f, 0.0f }, 1.1f, 1.1f, 0.03f, 16, (Color){ 0, 0, 0, 90 });
        }
        if (state.havePred) {
            float groundY = terrainHeight(predPos.x);
            DrawCylinder((Vector3){ predPos.x, groundY + 0.03f, 0.0f }, 1.1f, 1.1f, 0.03f, 16, (Color){ 0, 0, 0, 70 });
        }

        if (state.haveReal) {
            sphereModel.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = RED;
            DrawModel(sphereModel, (Vector3){ realPos.x, realPos.y + 1.0f, 0.0f }, 1.0f, WHITE);
        }
        if (state.havePred) {
            sphereModel.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = GOLD;
            DrawModel(sphereModel, (Vector3){ predPos.x, predPos.y + 1.0f, 0.0f }, 1.0f, WHITE);
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

        EndTextureMode();

        // --- Composite with the post-process pass (vignette + contrast) ---
        BeginDrawing();
        ClearBackground(BLACK);
        BeginShaderMode(postShader);
        DrawTextureRec(sceneTarget.texture,
                        (Rectangle){ 0, 0, (float)screenWidth, -(float)screenHeight },
                        (Vector2){ 0, 0 }, WHITE);
        EndShaderMode();
        EndDrawing();
    }

    UnloadShader(litShader);
    UnloadShader(postShader);
    UnloadModel(terrainModel);
    UnloadModel(sphereModel);
    UnloadRenderTexture(sceneTarget);
    CloseWindow();
    return 0;
}
