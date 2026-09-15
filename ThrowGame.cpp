#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <limits>
#include "Terrain.h"
#include "ThrowRanges.h"

// Throwing game combining the physics engine and the trained model.
// Arrow keys adjust velocity, [ and ] adjust mass, A/D move your landing
// guess, SPACE throws, R replays the last throw.
// Run this from build/ (./throw_game), same as throw_sim and beam_sim.

static const float SIM_DT = 0.01f;        // must match ThrowSim.cpp's dt
static const float PLAYBACK_SPEED = 2.5f; // watch the flight faster than real time
static const char* STATS_PATH = "../game_stats.txt"; // repo root, survives build/ wipes

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

// predict.py prints exactly 8 comma-separated floats in a fixed order:
// x, y, maxHeight, timeToLand, bounceCount, apexTime, finalVx, uncertainty
// (predicted std dev of landing-position error). Only x, y, and
// uncertainty matter here, the rest are skipped with %*f rather than
// guessed at with a partial-match trick.
static bool runFinal(const std::string& cmd, float& outX, float& outY, float& outUncertainty) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;

    char buffer[256] = {0};
    bool gotLine = fgets(buffer, sizeof(buffer), pipe) != nullptr;
    int status = pclose(pipe);

    if (!gotLine || status != 0) return false;
    return sscanf(buffer, "%f,%f,%*f,%*f,%*f,%*f,%*f,%f", &outX, &outY, &outUncertainty) == 3;
}

static float randRangeF(float lo, float hi) {
    return lo + (hi - lo) * (float)GetRandomValue(0, 10000) / 10000.0f;
}

// Cheap legibility boost without a custom font file: a dark offset copy
// behind the real text, like a soft drop shadow. Keeps HUD text readable
// over a busy 3D scene (sky, terrain, particles) instead of just flat text.
static void DrawTextOutlined(const char* text, int x, int y, int fontSize, Color color) {
    DrawText(text, x + 2, y + 2, fontSize, (Color){ 0, 0, 0, 170 });
    DrawText(text, x, y, fontSize, color);
}

static void DrawPanel(int x, int y, int w, int h) {
    DrawRectangle(x, y, w, h, (Color){ 15, 15, 20, 110 });
}

struct ThrowState {
    float vx0, vy0, mass, height0, windAccel;
    std::vector<Vector2> realTrajectory;
    bool haveReal;
    float predX, predY, predUncertainty;
    bool havePred;
    double startTime;
};

// height0 and windAccel are environmental, not player-controlled - randomized
// fresh each throw, same as the target zone, rather than adding more keys.
static ThrowState runThrow(float vx0, float vy0, float mass, float height0, float windAccel) {
    ThrowState s;
    s.vx0 = vx0;
    s.vy0 = vy0;
    s.mass = mass;
    s.height0 = height0;
    s.windAccel = windAccel;

    std::string argsStr = std::to_string(vx0) + " " + std::to_string(vy0) + " " + std::to_string(mass) + " " +
                           std::to_string(height0) + " " + std::to_string(windAccel);

    s.haveReal = runCommand("./throw_sim --trajectory " + argsStr, s.realTrajectory);
    s.havePred = runFinal("python3 ../predict.py " + argsStr, s.predX, s.predY, s.predUncertainty);

    s.startTime = GetTime();
    return s;
}

// --- Persistent stats, saved to a plain text file outside build/ so a
// build wipe doesn't erase progress ---
struct GameStats {
    int totalThrows = 0;
    float aiErrorSum = 0.0f;
    float bestError = std::numeric_limits<float>::infinity();
    float worstError = 0.0f;
    int streak = 0;
    int bestStreak = 0;
    int targetScore = 0;
    int playerWins = 0;
    int aiWins = 0;
};

static GameStats loadStats(const std::string& path) {
    GameStats s;
    std::ifstream in(path);
    if (!in.is_open()) return s;
    std::string key;
    while (in >> key) {
        if (key == "totalThrows") in >> s.totalThrows;
        else if (key == "aiErrorSum") in >> s.aiErrorSum;
        else if (key == "bestError") in >> s.bestError;
        else if (key == "worstError") in >> s.worstError;
        else if (key == "streak") in >> s.streak;
        else if (key == "bestStreak") in >> s.bestStreak;
        else if (key == "targetScore") in >> s.targetScore;
        else if (key == "playerWins") in >> s.playerWins;
        else if (key == "aiWins") in >> s.aiWins;
    }
    return s;
}

static void saveStats(const std::string& path, const GameStats& s) {
    std::ofstream out(path);
    out << "totalThrows " << s.totalThrows << "\n";
    out << "aiErrorSum " << s.aiErrorSum << "\n";
    out << "bestError " << s.bestError << "\n";
    out << "worstError " << s.worstError << "\n";
    out << "streak " << s.streak << "\n";
    out << "bestStreak " << s.bestStreak << "\n";
    out << "targetScore " << s.targetScore << "\n";
    out << "playerWins " << s.playerWins << "\n";
    out << "aiWins " << s.aiWins << "\n";
}

// --- Simple burst particles for landing impact, unlit, cheap ---
struct Particle {
    Vector3 pos, vel;
    float life, maxLife;
};

static void spawnBurst(std::vector<Particle>& particles, Vector3 origin) {
    for (int i = 0; i < 24; i++) {
        Particle p;
        p.pos = origin;
        p.vel = (Vector3){ randRangeF(-6.0f, 6.0f), randRangeF(2.0f, 8.0f), randRangeF(-6.0f, 6.0f) };
        p.maxLife = randRangeF(0.4f, 0.7f);
        p.life = p.maxLife;
        particles.push_back(p);
    }
}

// Terrain as a proper lit Model instead of an unlit immediate-mode strip.
// Vertex normals come from the analytic derivative of terrainHeight, not
// guessed, so the lighting shader gets real slope information.
static Model buildTerrainModel() {
    const float halfWidth = 45.0f;
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

    // The scene always renders internally at this fixed resolution, then
    // gets scaled (letterboxed) to fit whatever the real window size is.
    // That way HUD layout code never has to think about window size - only
    // the final composite step does.
    const int screenWidth = 1000;
    const int screenHeight = 600;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(screenWidth, screenHeight, "BeamSim3D - Throw Game");
    SetWindowMinSize(480, 300);

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
    float guessX = 0.0f;
    float targetX = randRangeF(-50.0f, 250.0f);
    ThrowState state = runThrow(vx, vy, mass, randRangeF(HEIGHT_MIN, HEIGHT_MAX), randRangeF(WIND_MIN, WIND_MAX));
    bool scored = false;
    bool wasFlying = true;

    GameStats stats = loadStats(STATS_PATH);
    float lastPlayerError = 0.0f;
    bool lastPlayerWon = false;
    int lastRoundScore = 0;

    std::vector<Vector3> realTrail, predTrail;
    std::vector<Particle> particles;
    float shakeTimer = 0.0f;
    const float shakeDuration = 0.25f;

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
        float dt = GetFrameTime();

        if (IsKeyPressed(KEY_F11)) ToggleFullscreen();

        // Adjust throw parameters and guess only between throws
        if (animDone) {
            if (IsKeyDown(KEY_RIGHT)) vx = Clamp(vx + 10.0f * dt, VX_MIN, VX_MAX);
            if (IsKeyDown(KEY_LEFT)) vx = Clamp(vx - 10.0f * dt, VX_MIN, VX_MAX);
            if (IsKeyDown(KEY_UP)) vy = Clamp(vy + 10.0f * dt, VY_MIN, VY_MAX);
            if (IsKeyDown(KEY_DOWN)) vy = Clamp(vy - 10.0f * dt, VY_MIN, VY_MAX);
            if (IsKeyDown(KEY_RIGHT_BRACKET)) mass = Clamp(mass + 2.0f * dt, MASS_MIN, MASS_MAX);
            if (IsKeyDown(KEY_LEFT_BRACKET)) mass = Clamp(mass - 2.0f * dt, MASS_MIN, MASS_MAX);
            if (IsKeyDown(KEY_D)) guessX += 30.0f * dt;
            if (IsKeyDown(KEY_A)) guessX -= 30.0f * dt;

            if (IsKeyPressed(KEY_SPACE)) {
                state = runThrow(vx, vy, mass, randRangeF(HEIGHT_MIN, HEIGHT_MAX), randRangeF(WIND_MIN, WIND_MAX));
                targetX = randRangeF(-50.0f, 250.0f);
                scored = false;
                realTrail.clear();
                predTrail.clear();
            } else if (IsKeyPressed(KEY_R)) {
                // Replay the same throw's cached trajectory, no new
                // subprocess calls and no re-scoring (already counted).
                state.startTime = GetTime();
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
            predPos.y = Lerp(state.height0, state.predY, animT) + arcHeight * sinf(3.14159265f * animT);
        }

        if (flying) {
            if (state.haveReal) realTrail.push_back({ realPos.x, realPos.y + 1.0f, 0.0f });
            if (state.havePred) predTrail.push_back({ predPos.x, predPos.y + 1.0f, 0.0f });
        }

        // Landing edge: just transitioned from flying to landed this frame
        if (wasFlying && !flying && state.haveReal) {
            spawnBurst(particles, (Vector3){ realPos.x, realPos.y + 1.0f, 0.0f });
            shakeTimer = shakeDuration;
        }
        wasFlying = flying;

        if (animDone && !scored && state.haveReal && state.havePred) {
            float realX = state.realTrajectory.back().x;
            float realY = state.realTrajectory.back().y;
            float dx = realX - state.predX;
            float dy = realY - state.predY;
            float aiError = sqrtf(dx * dx + dy * dy);

            stats.totalThrows++;
            stats.aiErrorSum += aiError;
            if (aiError < stats.bestError) stats.bestError = aiError;
            if (aiError > stats.worstError) stats.worstError = aiError;
            stats.streak = (aiError < 1.0f) ? stats.streak + 1 : 0;
            if (stats.streak > stats.bestStreak) stats.bestStreak = stats.streak;

            lastPlayerError = fabsf(realX - guessX);
            lastPlayerWon = lastPlayerError < aiError;
            if (lastPlayerWon) stats.playerWins++; else stats.aiWins++;

            lastRoundScore = (int)fmaxf(0.0f, 100.0f - fabsf(realX - targetX));
            stats.targetScore += lastRoundScore;

            scored = true;
            saveStats(STATS_PATH, stats);
        }

        // Update particles
        for (auto& p : particles) {
            p.vel.y -= 9.8f * dt;
            p.pos = Vector3Add(p.pos, Vector3Scale(p.vel, dt));
            p.life -= dt;
        }
        particles.erase(
            std::remove_if(particles.begin(), particles.end(), [](const Particle& p) { return p.life <= 0.0f; }),
            particles.end());

        // Camera follows the real ball while it's flying, otherwise settles
        // over the landing spot. Manual orbit instead of raylib's built-in
        // CAMERA_ORBITAL so it can smoothly track a moving target.
        Vector3 desiredTarget = flying
            ? (Vector3){ realPos.x, realPos.y, 0.0f }
            : (Vector3){ state.haveReal ? state.realTrajectory.back().x : 0.0f,
                         state.haveReal ? state.realTrajectory.back().y : 1.0f, 0.0f };
        followTarget = Vector3Lerp(followTarget, desiredTarget, 0.08f);
        orbitYaw += 0.15f * dt;

        Camera camera = { 0 };
        camera.target = followTarget;
        camera.position = (Vector3){
            followTarget.x + orbitRadius * cosf(orbitYaw),
            followTarget.y + orbitHeight,
            followTarget.z + orbitRadius * sinf(orbitYaw)
        };

        if (shakeTimer > 0.0f) {
            float t = shakeTimer / shakeDuration;
            camera.position.x += randRangeF(-0.6f, 0.6f) * t;
            camera.position.y += randRangeF(-0.6f, 0.6f) * t;
            camera.position.z += randRangeF(-0.6f, 0.6f) * t;
            shakeTimer -= dt;
        }

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

        // Player's guess marker (sky blue disc on the ground)
        {
            float gY = terrainHeight(guessX);
            DrawCylinder((Vector3){ guessX, gY + 0.05f, 0.0f }, 0.8f, 0.8f, 0.08f, 16, Fade(SKYBLUE, 0.85f));
        }
        // Target marker (magenta flag)
        {
            float tY = terrainHeight(targetX);
            DrawCylinder((Vector3){ targetX, tY, 0.0f }, 0.05f, 0.05f, 4.0f, 8, MAGENTA);
            DrawSphere((Vector3){ targetX, tY + 4.0f, 0.0f }, 0.5f, MAGENTA);
        }

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

        for (const auto& p : particles) {
            DrawSphere(p.pos, 0.12f, Fade(GOLD, p.life / p.maxLife));
        }

        EndMode3D();

        // --- Top HUD panel: throw params, real/AI landing, headline result ---
        int panelW = 470;
        int panelH = (animDone && state.haveReal && state.havePred) ? 178 : 148;
        DrawPanel(8, 8, panelW, panelH);

        int lineY = 16;
        if (animDone) {
            DrawTextOutlined(TextFormat("vx=%.1f  vy=%.1f  mass=%.1f", vx, vy, mass), 18, lineY, 20, RAYWHITE);
        } else {
            DrawTextOutlined(TextFormat("In flight: vx=%.1f  vy=%.1f  mass=%.1f", state.vx0, state.vy0, state.mass), 18, lineY, 20, RAYWHITE);
        }
        lineY += 22;

        // height0/windAccel are environmental (randomized per throw, not
        // player-adjustable), shown for the throw that's currently in
        // flight or just landed - not the next one, which hasn't rolled yet.
        DrawTextOutlined(TextFormat("Launch height %.1f   Wind %+.1f", state.height0, state.windAccel), 18, lineY, 15, (Color){ 180, 210, 230, 255 });
        lineY += 20;

        if (state.haveReal) {
            DrawTextOutlined(TextFormat("Real landing:  x=%.2f  y=%.2f", state.realTrajectory.back().x, state.realTrajectory.back().y), 18, lineY, 17, (Color){ 255, 120, 120, 255 });
        } else {
            DrawTextOutlined("Real landing: throw_sim call failed (run this from build/)", 18, lineY, 17, (Color){ 255, 120, 120, 255 });
        }
        lineY += 22;

        if (state.havePred) {
            DrawTextOutlined(TextFormat("AI predicted:  x=%.2f  y=%.2f  (+/-%.2f, 68%% confident)", state.predX, state.predY, state.predUncertainty), 18, lineY, 17, GOLD);
        } else {
            DrawTextOutlined("AI prediction: predict.py call failed (need venv with torch active)", 18, lineY, 17, GOLD);
        }
        lineY += 30;

        if (animDone && state.haveReal && state.havePred) {
            float dx = state.realTrajectory.back().x - state.predX;
            float dy = state.realTrajectory.back().y - state.predY;
            float aiError = sqrtf(dx * dx + dy * dy);
            Color resultColor = lastPlayerWon ? (Color){ 110, 230, 140, 255 } : (Color){ 200, 140, 255, 255 };
            DrawTextOutlined(lastPlayerWon ? "YOU WIN THIS ROUND" : "AI WINS THIS ROUND", 18, lineY, 22, resultColor);
            lineY += 26;
            DrawTextOutlined(TextFormat("AI error %.3f   Your error %.3f   +%d target pts",
                                         aiError, lastPlayerError, lastRoundScore),
                              18, lineY, 16, (Color){ 220, 220, 220, 255 });
        } else if (!animDone) {
            DrawTextOutlined("In flight...", 18, lineY, 18, (Color){ 210, 210, 210, 255 });
        }

        // --- Secondary stats panel, smaller/quieter than the headline ---
        if (stats.totalThrows > 0) {
            DrawPanel(8, panelH + 14, panelW, 52);
            DrawTextOutlined(TextFormat("Throws %d   AI avg %.3f   Best %.3f   Streak %d (best %d)",
                                         stats.totalThrows, stats.aiErrorSum / stats.totalThrows, stats.bestError, stats.streak, stats.bestStreak),
                              18, panelH + 20, 15, (Color){ 190, 190, 190, 255 });
            DrawTextOutlined(TextFormat("You vs AI: %d - %d   Target score: %d",
                                         stats.playerWins, stats.aiWins, stats.targetScore),
                              18, panelH + 40, 15, (Color){ 190, 190, 190, 255 });
        }

        // --- Bottom control hint bar, full width ---
        DrawPanel(0, screenHeight - 30, screenWidth, 30);
        DrawTextOutlined("SPACE throw | R replay | F11 fullscreen | A/D your guess | blue disc = guess | magenta flag = target",
                          14, screenHeight - 24, 15, (Color){ 220, 220, 220, 255 });

        EndTextureMode();

        // --- Composite with the post-process pass (vignette + contrast) ---
        // Scaled + letterboxed to fit the actual window/fullscreen size,
        // keeping the fixed internal resolution's aspect ratio intact.
        int winW = GetScreenWidth();
        int winH = GetScreenHeight();
        float fitScale = fminf((float)winW / screenWidth, (float)winH / screenHeight);
        float destW = screenWidth * fitScale;
        float destH = screenHeight * fitScale;
        float destX = (winW - destW) * 0.5f;
        float destY = (winH - destH) * 0.5f;

        BeginDrawing();
        ClearBackground(BLACK);
        BeginShaderMode(postShader);
        DrawTexturePro(sceneTarget.texture,
                        (Rectangle){ 0, 0, (float)screenWidth, -(float)screenHeight },
                        (Rectangle){ destX, destY, destW, destH },
                        (Vector2){ 0, 0 }, 0.0f, WHITE);
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
