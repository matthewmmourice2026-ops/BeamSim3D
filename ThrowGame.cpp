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
#include <unistd.h>
#include <sys/wait.h>
#include <csignal>
#include <thread>
#include <atomic>
#include "Terrain.h"
#include "ThrowRanges.h"

// Throwing game combining the physics engine and the trained model.
// Arrow keys adjust velocity, [ and ] adjust mass, A/D move your landing
// guess, SPACE throws, R replays the last throw.
// Run this from build/ (./throw_game), same as throw_sim and beam_sim.

static const float SIM_DT = 0.01f;        // must match ThrowSim.cpp's dt
static const float PLAYBACK_SPEED = 2.5f; // watch the flight faster than real time
static const char* STATS_PATH = "../game_stats.txt"; // repo root, survives build/ wipes
static const char* SETTINGS_PATH = "../game_settings.txt";
// Same column schema as build/throw_results.csv (ThrowSim.cpp), so
// main.py/kfold_eval.py can concatenate this straight in as more training
// rows - real played throws, not just ThrowSim.cpp's uniform-random ones.
static const char* GAME_THROWS_PATH = "../game_throws.csv";

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

// throw_sim's single-throw mode (argc==6, no --trajectory) prints the
// exact same 7 output columns build/throw_results.csv has: final_x,
// final_y, maxHeight, timeToLand, bounceCount, apexTime, finalVx. Used
// here to get the authoritative final row for logging, separate from the
// --trajectory call above which only gives x,y per step (no bounceCount/
// apexTime/finalVx - those aren't recoverable from position samples alone).
static bool runFinalRow(const std::string& cmd, float out[7]) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;

    char buffer[256] = {0};
    bool gotLine = fgets(buffer, sizeof(buffer), pipe) != nullptr;
    int status = pclose(pipe);
    if (!gotLine || status != 0) return false;

    return sscanf(buffer, "%f,%f,%f,%f,%f,%f,%f",
                  &out[0], &out[1], &out[2], &out[3], &out[4], &out[5], &out[6]) == 7;
}

// Appends one played throw to game_throws.csv, writing the header first if
// the file doesn't exist yet. Called once per real throw (not replays -
// R replays the cached trajectory, no new physics run to log).
static void logGameThrow(const std::string& path, float vx0, float vy0, float mass, float height0,
                          float windAccel, const float finalRow[7]) {
    bool needsHeader = false;
    {
        std::ifstream check(path);
        needsHeader = !check.is_open() || check.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) return;
    if (needsHeader) {
        out << "vx0,vy0,mass,height0,windAccel,final_x,final_y,maxHeight,timeToLand,bounceCount,apexTime,finalVx\n";
    }
    out << vx0 << "," << vy0 << "," << mass << "," << height0 << "," << windAccel << ","
        << finalRow[0] << "," << finalRow[1] << "," << finalRow[2] << "," << finalRow[3] << ","
        << finalRow[4] << "," << finalRow[5] << "," << finalRow[6] << "\n";
}

// predict_server.py prints exactly 8 comma-separated floats in a fixed
// order: x, y, maxHeight, timeToLand, bounceCount, apexTime, finalVx,
// uncertainty (predicted std dev of landing-position error). Only x, y,
// and uncertainty matter here, the rest are skipped with %*f rather
// than guessed at with a partial-match trick.

// Spawning `python3 predict.py ...` fresh per throw paid Python startup
// + torch import + checkpoint load every single call - hundreds of ms
// to a couple seconds, dwarfing the actual forward pass. This keeps one
// warm predict_server.py subprocess alive for the game's lifetime and
// talks to it over a pair of pipes, so a throw only pays the forward
// pass. popen() can't do this (it's one-directional), so it's a manual
// fork + pipe + exec.
struct PredictServer {
    pid_t pid = -1;
    FILE* toChild = nullptr;
    FILE* fromChild = nullptr;
};
static PredictServer g_predictServer;
// Set true only after g_predictServer's FILE* pointers are fully written -
// every read of those pointers elsewhere must check this first. That
// ordering (write pointers, then store true; check true, then read
// pointers) is what makes this safe without a mutex: std::atomic's
// default seq_cst ordering turns it into a publish/subscribe handoff
// between the loader thread and the main thread.
static std::atomic<bool> g_predictServerReady{false};
static std::atomic<bool> g_predictServerFailed{false};
static std::thread g_predictServerThread;

// Runs on a background thread (see main()) - torch import + checkpoint
// load takes ~0.5-1s, and blocking the main thread on that used to leave
// the game window dark/unresponsive at launch. fork()+exec() from a
// non-main thread is safe (the exec() right after fork() in the child
// means the child never runs any other thread's code), unlike fork()
// alone in a multithreaded process.
static void startPredictServer() {
    int inPipe[2];  // parent writes -> child stdin
    int outPipe[2]; // child stdout -> parent reads
    if (pipe(inPipe) != 0 || pipe(outPipe) != 0) { g_predictServerFailed = true; return; }

    pid_t pid = fork();
    if (pid < 0) { g_predictServerFailed = true; return; }

    if (pid == 0) {
        // Child: stdin <- inPipe[0], stdout -> outPipe[1]
        dup2(inPipe[0], STDIN_FILENO);
        dup2(outPipe[1], STDOUT_FILENO);
        close(inPipe[0]);
        close(inPipe[1]);
        close(outPipe[0]);
        close(outPipe[1]);
        execlp("python3", "python3", "../predict_server.py", (char*)nullptr);
        _exit(127); // only reached if execlp failed
    }

    // Parent
    close(inPipe[0]);
    close(outPipe[1]);
    g_predictServer.pid = pid;
    g_predictServer.toChild = fdopen(inPipe[1], "w");
    g_predictServer.fromChild = fdopen(outPipe[0], "r");
    if (!g_predictServer.toChild || !g_predictServer.fromChild) { g_predictServerFailed = true; return; }

    // Block (this thread only) for the "ready" line printed after the
    // model/checkpoint finish loading, so the first real throw doesn't
    // race the warm-up.
    char buffer[64] = {0};
    if (fgets(buffer, sizeof(buffer), g_predictServer.fromChild) != nullptr) {
        g_predictServerReady = true;
    } else {
        g_predictServerFailed = true;
    }
}

static void stopPredictServer() {
    if (g_predictServerThread.joinable()) g_predictServerThread.join();
    if (g_predictServer.toChild) fclose(g_predictServer.toChild); // EOF tells the child to exit
    if (g_predictServer.fromChild) fclose(g_predictServer.fromChild);
    if (g_predictServer.pid > 0) waitpid(g_predictServer.pid, nullptr, 0);
}

// All 8 model outputs, not just x/y/uncertainty: x, y, maxHeight,
// timeToLand, bounceCount, apexTime, finalVx, uncertainty.
static bool predictThrow(float vx0, float vy0, float mass, float height0, float windAccel,
                          float& outX, float& outY, float& outMaxHeight, float& outTimeToLand,
                          float& outBounceCount, float& outApexTime, float& outFinalVx, float& outUncertainty) {
    if (!g_predictServerReady.load()) return false;

    fprintf(g_predictServer.toChild, "%f %f %f %f %f\n", vx0, vy0, mass, height0, windAccel);
    if (fflush(g_predictServer.toChild) != 0) return false;

    char buffer[256] = {0};
    if (!fgets(buffer, sizeof(buffer), g_predictServer.fromChild)) return false;
    return sscanf(buffer, "%f,%f,%f,%f,%f,%f,%f,%f", &outX, &outY, &outMaxHeight, &outTimeToLand,
                  &outBounceCount, &outApexTime, &outFinalVx, &outUncertainty) == 8;
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

// Card-style HUD panel: darker/more opaque than before for legibility over
// a busy 3D scene, a faint border to separate it from the background, and
// a colored accent strip on the left edge so panels read as distinct UI
// elements instead of a flat translucent smear.
static void DrawPanel(int x, int y, int w, int h, Color accent) {
    DrawRectangle(x, y, w, h, (Color){ 10, 12, 18, 165 });
    DrawRectangleLines(x, y, w, h, (Color){ 255, 255, 255, 30 });
    DrawRectangle(x, y, 3, h, accent);
}
static void DrawPanel(int x, int y, int w, int h) {
    DrawPanel(x, y, w, h, (Color){ 120, 170, 255, 220 });
}

struct ThrowState {
    float vx0, vy0, mass, height0, windAccel;
    std::vector<Vector2> realTrajectory;
    bool haveReal;
    // All 7 regression outputs the model predicts (not just x/y), plus its
    // predicted uncertainty on the landing position.
    float predX, predY, predMaxHeight, predTimeToLand, predBounceCount, predApexTime, predFinalVx, predUncertainty;
    bool havePred;
    // Ground truth for the same 7 outputs, from throw_sim's single-throw
    // mode, so the full prediction can be checked against reality, not
    // just x/y.
    float realMaxHeight, realTimeToLand, realBounceCount, realApexTime, realFinalVx;
    bool haveFinalRow;
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
    s.havePred = predictThrow(vx0, vy0, mass, height0, windAccel, s.predX, s.predY, s.predMaxHeight,
                               s.predTimeToLand, s.predBounceCount, s.predApexTime, s.predFinalVx, s.predUncertainty);

    float finalRow[7];
    s.haveFinalRow = runFinalRow("./throw_sim " + argsStr, finalRow);
    if (s.haveFinalRow) {
        logGameThrow(GAME_THROWS_PATH, vx0, vy0, mass, height0, windAccel, finalRow);
        // finalRow layout matches the CSV columns: final_x, final_y,
        // maxHeight, timeToLand, bounceCount, apexTime, finalVx.
        s.realMaxHeight = finalRow[2];
        s.realTimeToLand = finalRow[3];
        s.realBounceCount = finalRow[4];
        s.realApexTime = finalRow[5];
        s.realFinalVx = finalRow[6];
    }

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

struct GameSettings {
    float mouseSensitivity = 0.15f; // scales orbit yaw/height per pixel of mouse drag
    bool invertY = false;
};

static GameSettings loadSettings(const std::string& path) {
    GameSettings s;
    std::ifstream in(path);
    if (!in.is_open()) return s;
    std::string key;
    while (in >> key) {
        if (key == "mouseSensitivity") in >> s.mouseSensitivity;
        else if (key == "invertY") in >> s.invertY;
    }
    return s;
}

static void saveSettings(const std::string& path, const GameSettings& s) {
    std::ofstream out(path);
    out << "mouseSensitivity " << s.mouseSensitivity << "\n";
    out << "invertY " << (s.invertY ? 1 : 0) << "\n";
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
    // Terrain used to end at +/-110, but max-power throws (VX_MAX=65,
    // VY_MAX=100, MASS_MAX=20, WIND_MAX=3) land past x=2600 under the real
    // physics - the ball was flying off the end of the visible ground.
    // Widened with margin; step raised to keep vertex count sane
    // (mesh.indices is uint16, capped at 65535).
    const float xMin = -150.0f, xMax = 3200.0f, step = 4.0f;

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

            // Height/slope-tinted vertex colors (multiply the flat material
            // diffuse below) instead of flat white, so the terrain reads as
            // rolling grass instead of a single-tone slab: darker, cooler
            // green in valleys, warmer highlight near ridge tops, extra
            // shading on steep faces like a self-shadowed crevice.
            float hillT = Clamp((h + 2.5f) / 5.0f, 0.0f, 1.0f);
            float slopeShade = 1.0f - Clamp(fabsf(slope) * 1.5f, 0.0f, 0.35f);
            mesh.colors[v * 4 + 0] = (unsigned char)(Lerp(150.0f, 255.0f, hillT) * slopeShade);
            mesh.colors[v * 4 + 1] = (unsigned char)(Lerp(200.0f, 255.0f, hillT) * slopeShade);
            mesh.colors[v * 4 + 2] = (unsigned char)(Lerp(140.0f, 210.0f, hillT) * slopeShade);
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

    // Writing to predict_server.py's pipe after it has died (crash, bad
    // checkpoint, etc.) would raise SIGPIPE, whose default action kills
    // this whole process instantly - every future throw would take the
    // game down with it instead of just failing that one prediction.
    // Ignore it so the write fails with EPIPE and predictThrow() degrades
    // gracefully like it already does on any other failure.
    signal(SIGPIPE, SIG_IGN);

    // The scene always renders internally at this fixed resolution, then
    // gets scaled (letterboxed) to fit whatever the real window size is.
    // That way HUD layout code never has to think about window size - only
    // the final composite step does.
    const int screenWidth = 1000;
    const int screenHeight = 600;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(screenWidth, screenHeight, "BeamSim3D - Throw Game");
    SetWindowMinSize(480, 300);

    // startPredictServer() takes ~0.5-1s (torch import + checkpoint load
    // in predict_server.py). Used to block here before the main loop even
    // started, so the window stayed dark/unresponsive for that whole
    // stretch. Now runs on a background thread - the window and game are
    // interactive immediately; predictThrow() just returns false (shown
    // as "AI loading..." in the HUD, see the havePred fallback below)
    // until g_predictServerReady flips true.
    g_predictServerThread = std::thread(startPredictServer);

    // --- Lighting setup ---
    Shader litShader = LoadShader("../Shaders/lighting.vs", "../Shaders/lighting.fs");
    int locLightDir = GetShaderLocation(litShader, "lightDir");
    int locLightColor = GetShaderLocation(litShader, "lightColor");
    int locAmbientColor = GetShaderLocation(litShader, "ambientColor");
    int locViewPos = GetShaderLocation(litShader, "viewPos");
    int locShininess = GetShaderLocation(litShader, "shininess");
    int locFogColor = GetShaderLocation(litShader, "fogColor");
    int locFogStart = GetShaderLocation(litShader, "fogStart");
    int locFogEnd = GetShaderLocation(litShader, "fogEnd");

    Vector3 lightDir = Vector3Normalize((Vector3){ -0.4f, -1.0f, -0.35f });
    Vector3 lightColor = { 1.0f, 0.97f, 0.9f };
    Vector3 ambientColor = { 0.28f, 0.32f, 0.36f };
    float shininess = 24.0f;
    // Matches the sky gradient's top color so the terrain's far edge
    // (the mesh now runs out to x=3200) dissolves into the sky instead of
    // hard-cutting.
    Vector3 fogColor = { 200.0f / 255.0f, 225.0f / 255.0f, 245.0f / 255.0f };
    float fogStart = 140.0f, fogEnd = 480.0f;
    SetShaderValue(litShader, locLightDir, &lightDir, SHADER_UNIFORM_VEC3);
    SetShaderValue(litShader, locLightColor, &lightColor, SHADER_UNIFORM_VEC3);
    SetShaderValue(litShader, locAmbientColor, &ambientColor, SHADER_UNIFORM_VEC3);
    SetShaderValue(litShader, locShininess, &shininess, SHADER_UNIFORM_FLOAT);
    SetShaderValue(litShader, locFogColor, &fogColor, SHADER_UNIFORM_VEC3);
    SetShaderValue(litShader, locFogStart, &fogStart, SHADER_UNIFORM_FLOAT);
    SetShaderValue(litShader, locFogEnd, &fogEnd, SHADER_UNIFORM_FLOAT);

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
    float orbitRadius = 60.0f, orbitHeight = 35.0f;
    const float orbitRadiusMin = 20.0f, orbitRadiusMax = 160.0f;
    const float orbitHeightMin = 5.0f, orbitHeightMax = 100.0f;

    GameSettings settings = loadSettings(SETTINGS_PATH);
    bool showSettings = false;
    bool paused = false;

    // Custom pause key instead of quitting: ESC is raylib's default exit
    // key, so free it up here and let WindowShouldClose() still catch the
    // OS close button.
    SetExitKey(KEY_NULL);

    SetTargetFPS(60);

    double pauseStartTime = 0.0;
    double totalPausedDuration = 0.0;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_ESCAPE)) {
            paused = !paused;
            if (paused) pauseStartTime = GetTime();
            else totalPausedDuration += GetTime() - pauseStartTime;
        }
        if (IsKeyPressed(KEY_TAB)) showSettings = !showSettings;

        double pausedNow = totalPausedDuration + (paused ? GetTime() - pauseStartTime : 0.0);
        float realDuration = state.haveReal ? (state.realTrajectory.size() - 1) * SIM_DT : 0.0f;
        float elapsed = (float)(GetTime() - state.startTime - pausedNow) * PLAYBACK_SPEED;
        float animT = realDuration > 0.0f ? Clamp(elapsed / realDuration, 0.0f, 1.0f) : 1.0f;
        bool animDone = animT >= 1.0f;
        bool flying = !animDone;
        float dt = paused ? 0.0f : GetFrameTime();

        if (IsKeyPressed(KEY_F11)) ToggleFullscreen();

        // Mouse camera: right-drag orbits (yaw + height), scroll zooms
        // (radius). Works even while paused, since it's just a view
        // control, not gameplay.
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            Vector2 mouseDelta = GetMouseDelta();
            orbitYaw += mouseDelta.x * settings.mouseSensitivity * 0.02f;
            float heightDelta = mouseDelta.y * settings.mouseSensitivity * 0.3f * (settings.invertY ? -1.0f : 1.0f);
            orbitHeight = Clamp(orbitHeight + heightDelta, orbitHeightMin, orbitHeightMax);
        }
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            orbitRadius = Clamp(orbitRadius - wheel * 6.0f, orbitRadiusMin, orbitRadiusMax);
        }

        // Settings panel: TAB to open, +/- adjust sensitivity, I toggles
        // invert Y, saved to disk on change so it survives a restart.
        if (showSettings) {
            bool changed = false;
            if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
                settings.mouseSensitivity = Clamp(settings.mouseSensitivity + 0.05f, 0.02f, 2.0f);
                changed = true;
            }
            if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
                settings.mouseSensitivity = Clamp(settings.mouseSensitivity - 0.05f, 0.02f, 2.0f);
                changed = true;
            }
            if (IsKeyPressed(KEY_I)) {
                settings.invertY = !settings.invertY;
                changed = true;
            }
            if (changed) saveSettings(SETTINGS_PATH, settings);
        }

        // Adjust throw parameters and guess only between throws, and never while paused
        if (animDone && !paused) {
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
            // endpoint), just a parabolic arc for visual flight. This used
            // to reuse animT (scaled to realDuration, the REAL trajectory's
            // full length including every bounce/settle step - can be many
            // real seconds for a bouncy throw), so on a throw with a lot of
            // bounces the gold ball crawled through its single arc for the
            // whole stretched-out duration, looking stuck/slow. Give it its
            // own, much shorter duration based on the model's own predicted
            // timeToLand instead - it reaches its landing spot promptly and
            // then just sits there while the real ball keeps settling.
            float predDuration = Clamp(state.predTimeToLand, 0.3f, realDuration > 0.0f ? realDuration : 999.0f);
            float predAnimT = Clamp(elapsed / predDuration, 0.0f, 1.0f);
            float arcHeight = 8.0f;
            predPos.x = Lerp(0.0f, state.predX, predAnimT);
            predPos.y = Lerp(state.height0, state.predY, predAnimT) + arcHeight * sinf(3.14159265f * predAnimT);
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

        // Comet-style trails: older segments fade out instead of a flat
        // solid line, so the flight path reads as motion, not a static rod.
        for (size_t i = 1; i < realTrail.size(); i++) {
            float trailAlpha = 0.25f + 0.65f * ((float)i / (float)(realTrail.size() - 1));
            DrawLine3D(realTrail[i - 1], realTrail[i], Fade(MAROON, trailAlpha));
        }
        for (size_t i = 1; i < predTrail.size(); i++) {
            float trailAlpha = 0.25f + 0.65f * ((float)i / (float)(predTrail.size() - 1));
            DrawLine3D(predTrail[i - 1], predTrail[i], Fade(ORANGE, trailAlpha));
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
            float lifeT = p.life / p.maxLife;
            // Warm impact-dust look: shrinks and cools from bright gold to a
            // duller orange as it dies, instead of a flat gold dot fading
            // out at constant size.
            Color sparkColor = ColorLerp(ORANGE, GOLD, lifeT);
            DrawSphere(p.pos, 0.06f + 0.10f * lifeT, Fade(sparkColor, lifeT));
        }

        EndMode3D();

        // World-space labels for the guess/target markers, projected with
        // the internal render-texture resolution (not the real window size
        // GetWorldToScreen would use) since this all draws into sceneTarget.
        {
            Vector2 guessScreen = GetWorldToScreenEx((Vector3){ guessX, terrainHeight(guessX) + 1.4f, 0.0f }, camera, screenWidth, screenHeight);
            const char* guessLabel = "YOUR GUESS";
            int guessLabelW = MeasureText(guessLabel, 14);
            DrawTextOutlined(guessLabel, (int)guessScreen.x - guessLabelW / 2, (int)guessScreen.y, 14, SKYBLUE);

            Vector2 targetScreen = GetWorldToScreenEx((Vector3){ targetX, terrainHeight(targetX) + 5.2f, 0.0f }, camera, screenWidth, screenHeight);
            const char* targetLabel = "TARGET";
            int targetLabelW = MeasureText(targetLabel, 14);
            DrawTextOutlined(targetLabel, (int)targetScreen.x - targetLabelW / 2, (int)targetScreen.y, 14, MAGENTA);
        }

        // --- Brand mark, top-right corner ---
        {
            const char* brand = "BEAMSIM3D";
            int brandW = MeasureText(brand, 16);
            DrawTextOutlined(brand, screenWidth - brandW - 14, 12, 16, (Color){ 210, 225, 245, 235 });
        }

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
            DrawTextOutlined(TextFormat("AI predicted:  x=%.2f  y=%.2f  (~%.2f avg error)", state.predX, state.predY, state.predUncertainty), 18, lineY, 17, GOLD);
        } else if (g_predictServerFailed.load()) {
            DrawTextOutlined("AI prediction: predict_server.py failed to start (need venv with torch active)", 18, lineY, 17, GOLD);
        } else {
            DrawTextOutlined("AI prediction: loading model...", 18, lineY, 17, GOLD);
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
            DrawPanel(8, panelH + 14, panelW, 52, GOLD);
            DrawTextOutlined(TextFormat("Throws %d   AI avg %.3f   Best %.3f   Streak %d (best %d)",
                                         stats.totalThrows, stats.aiErrorSum / stats.totalThrows, stats.bestError, stats.streak, stats.bestStreak),
                              18, panelH + 20, 15, (Color){ 190, 190, 190, 255 });
            DrawTextOutlined(TextFormat("You vs AI: %d - %d   Target score: %d",
                                         stats.playerWins, stats.aiWins, stats.targetScore),
                              18, panelH + 40, 15, (Color){ 190, 190, 190, 255 });
        }

        // --- Full AI output panel: every model output, not just x/y ---
        {
            int px = screenWidth - 300, py = 112, pw = 292, ph = 190;
            DrawPanel(px, py, pw, ph, GOLD);
            DrawTextOutlined("AI FULL PREDICTION", px + 10, py + 8, 16, RAYWHITE);

            if (state.havePred) {
                DrawTextOutlined("metric          real       AI", px + 10, py + 30, 13, (Color){ 170, 190, 210, 255 });
                int ry = py + 48;
                bool hr = state.haveReal;
                bool hf = state.haveFinalRow;
                float realX = hr ? state.realTrajectory.back().x : 0.0f;
                float realY = hr ? state.realTrajectory.back().y : 0.0f;

                auto row = [&](const char* label, float realVal, bool haveReal_, float aiVal) {
                    if (haveReal_) {
                        DrawTextOutlined(TextFormat("%-14s %8.2f %8.2f", label, realVal, aiVal), px + 10, ry, 13, (Color){ 225, 225, 225, 255 });
                    } else {
                        DrawTextOutlined(TextFormat("%-14s %8s %8.2f", label, "--", aiVal), px + 10, ry, 13, (Color){ 225, 225, 225, 255 });
                    }
                    ry += 17;
                };

                row("x", realX, hr, state.predX);
                row("y", realY, hr, state.predY);
                row("maxHeight", state.realMaxHeight, hf, state.predMaxHeight);
                row("timeToLand", state.realTimeToLand, hf, state.predTimeToLand);
                row("bounceCount", state.realBounceCount, hf, state.predBounceCount);
                row("apexTime", state.realApexTime, hf, state.predApexTime);
                row("finalVx", state.realFinalVx, hf, state.predFinalVx);

                DrawTextOutlined(TextFormat("predicted avg landing error: ~%.2f", state.predUncertainty),
                                  px + 10, ry + 2, 13, (Color){ 200, 220, 240, 255 });
            } else if (g_predictServerFailed.load()) {
                DrawTextOutlined("predict_server.py failed to start", px + 10, py + 36, 14, (Color){ 220, 220, 220, 255 });
            } else {
                DrawTextOutlined("Loading model...", px + 10, py + 36, 14, (Color){ 220, 220, 220, 255 });
            }
        }

        // --- Settings panel (TAB) ---
        if (showSettings) {
            int sx = screenWidth - 300, sy = 8, sw = 292, sh = 96;
            DrawPanel(sx, sy, sw, sh);
            DrawTextOutlined("SETTINGS (TAB to close)", sx + 10, sy + 8, 17, RAYWHITE);
            DrawTextOutlined(TextFormat("Mouse sensitivity: %.2f  (-/+ to adjust)", settings.mouseSensitivity),
                              sx + 10, sy + 34, 15, (Color){ 220, 220, 220, 255 });
            DrawTextOutlined(TextFormat("Invert Y: %s  (I to toggle)", settings.invertY ? "ON" : "OFF"),
                              sx + 10, sy + 56, 15, (Color){ 220, 220, 220, 255 });
            DrawTextOutlined("Right-drag: orbit camera | Scroll: zoom", sx + 10, sy + 78, 14, (Color){ 180, 200, 220, 255 });
        }

        // --- Pause overlay ---
        if (paused) {
            DrawRectangle(0, 0, screenWidth, screenHeight, (Color){ 0, 0, 0, 130 });
            const char* label = "PAUSED";
            int fontSize = 44;
            int textW = MeasureText(label, fontSize);
            DrawTextOutlined(label, (screenWidth - textW) / 2, screenHeight / 2 - 40, fontSize, RAYWHITE);
            const char* sub = "ESC to resume";
            int subW = MeasureText(sub, 18);
            DrawTextOutlined(sub, (screenWidth - subW) / 2, screenHeight / 2 + 12, 18, (Color){ 210, 210, 210, 255 });
        }

        // --- Bottom control hint bar, full width ---
        DrawPanel(0, screenHeight - 30, screenWidth, 30);
        DrawTextOutlined("SPACE throw | R replay | F11 fullscreen | ESC pause | TAB settings | right-drag/scroll camera | A/D your guess",
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
    stopPredictServer();
    return 0;
}
