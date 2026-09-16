#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include <string>
#include "Terrain.h"
#include "ThrowRanges.h"

struct ThrowResult {
    float vx0, vy0, mass, height0, windAccel;
    float finalX, finalY;
    float maxHeight;
    float timeToLand;
    float bounceCount;
    float apexTime;
    float finalVx;
};

struct TrajectoryPoint {
    float x, y;
};

static ThrowResult simulateThrow(float vx0, float vy0, float mass, float height0, float windAccel,
                                  std::vector<TrajectoryPoint>* trajectory = nullptr) {
    const float dt = 0.01f;
    const float gravity = 9.81f;
    const float dragCoeff = 0.2f;
    const float restitution = 0.4f;
    const float groundFriction = 0.7f;
    const float restEps = 0.05f;
    const int maxSteps = 5000;
    float maxHeight = 0.0f;  // Starting height at x=0
    float apexTime = 0.0f;
    int bounceCount = 0;
    float timeToLand = maxSteps * dt; // default if it never fully settles
    float x = 0.0f, y = height0;
    float vx = vx0, vy = vy0;

    if (trajectory) {
        trajectory->push_back({ x, y });
    }

    for (int step = 0; step < maxSteps; ++step) {
        // windAccel is a constant horizontal push for the whole flight,
        // not a drag term - simplest physically-reasonable model of a
        // steady crosswind without adding wind-relative-velocity drag.
        float ax = -dragCoeff * vx / mass + windAccel;
        float ay = -gravity - dragCoeff * vy / mass;

        vx += ax * dt;
        vy += ay * dt;
        x += vx * dt;
        y += vy * dt;

        float ground = terrainHeight(x);
        if (y <= ground) {
            // Only count a real impact, not the constant near-zero
            // re-triggers of this branch every frame once it's resting
            // on the ground (gravity keeps nudging y a hair below ground
            // each frame, which would otherwise inflate the count into
            // the hundreds).
            if (std::fabs(vy) > 0.5f) {
                bounceCount++;
            }
            y = ground;
            vy = -vy * restitution;
            vx *= groundFriction;
            if (std::fabs(vy) < restEps) {
                vy = 0.0f;
            }
        }

        if (y > maxHeight) {
            maxHeight = y;
            apexTime = (step + 1) * dt;
        }

        if (trajectory) {
            trajectory->push_back({ x, y });
        }

        if (y == ground && std::fabs(vy) < restEps && std::fabs(vx) < restEps) {
            timeToLand = (step + 1) * dt;
            break;
        }
    }

    return { vx0, vy0, mass, height0, windAccel, x, y, maxHeight, timeToLand, (float)bounceCount, apexTime, vx };
}

int main(int argc, char** argv) {
    // Single-throw mode: ./throw_sim <vx0> <vy0> <mass> <height0> <windAccel>
    // prints the outputs and exits. Lets other tools (e.g. compare_predictions.py)
    // ask the real physics engine for ground truth instead of re-implementing
    // it elsewhere.
    if (argc == 6) {
        float vx0 = std::stof(argv[1]);
        float vy0 = std::stof(argv[2]);
        float mass = std::stof(argv[3]);
        float height0 = std::stof(argv[4]);
        float windAccel = std::stof(argv[5]);
        ThrowResult r = simulateThrow(vx0, vy0, mass, height0, windAccel);
        std::cout << r.finalX << "," << r.finalY << "," << r.maxHeight << ","
                   << r.timeToLand << "," << r.bounceCount << "," << r.apexTime << "," << r.finalVx
                   << std::endl;
        return 0;
    }

    // Trajectory mode: ./throw_sim --trajectory <vx0> <vy0> <mass> <height0>
    // <windAccel> prints one "x,y" line per simulation step, for animating
    // the flight path instead of just showing where it lands.
    if (argc == 7 && std::string(argv[1]) == "--trajectory") {
        float vx0 = std::stof(argv[2]);
        float vy0 = std::stof(argv[3]);
        float mass = std::stof(argv[4]);
        float height0 = std::stof(argv[5]);
        float windAccel = std::stof(argv[6]);
        std::vector<TrajectoryPoint> trajectory;
        simulateThrow(vx0, vy0, mass, height0, windAccel, &trajectory);
        for (const auto& p : trajectory) {
            std::cout << p.x << "," << p.y << "\n";
        }
        return 0;
    }

    const int throwCount = 700000;
    // Uniform random sampling barely ever lands near any one specific
    // corner of the 5D input space (vx,vy,mass,height,wind all near an
    // extreme simultaneously) - that's exactly the region the model does
    // worst in (e.g. vx=65,vy=100,mass=20 all maxed at once: ~60 units of
    // x error vs a ~4 average). This adds a second, smaller batch of
    // corner-biased throws - independently per axis, so many end up with
    // several parameters extreme at once - to give the network real
    // exposure to that region instead of extrapolating into it blind.
    const int cornerCount = 200000;
    const float cornerPower = 0.35f; // <1 concentrates samples toward the axis extremes

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> vxDist(VX_MIN, VX_MAX);
    std::uniform_real_distribution<float> vyDist(VY_MIN, VY_MAX);
    std::uniform_real_distribution<float> massDist(MASS_MIN, MASS_MAX);
    std::uniform_real_distribution<float> heightDist(HEIGHT_MIN, HEIGHT_MAX);
    std::uniform_real_distribution<float> windDist(WIND_MIN, WIND_MAX);
    std::uniform_real_distribution<float> unit01(0.0f, 1.0f);

    // Maps a uniform [0,1] draw to a value in [lo,hi] biased toward the
    // two ends: u=0.5 (the range's midpoint) still maps to the midpoint,
    // but u near 0 or 1 now lands much closer to lo/hi than a uniform
    // draw would, at a rate controlled by power (smaller = more biased).
    // extremity in [0,1] additionally scales how far toward that end this
    // particular draw is allowed to reach - see jointCornerSample below,
    // which uses a single shared extremity per row across all 5 axes so
    // rows are jointly extreme, not just marginally so per axis.
    auto biasedSample = [&](float lo, float hi, float power, float extremity) {
        float u = unit01(gen);
        float signedUnit = 2.0f * u - 1.0f; // [-1,1]
        float biased = std::copysign(std::pow(std::fabs(signedUnit), power), signedUnit) * extremity;
        float mid = (lo + hi) * 0.5f;
        float halfRange = (hi - lo) * 0.5f;
        return mid + biased * halfRange;
    };

    // First attempt at corner coverage biased each of the 5 axes
    // independently (fixed extremity=1.0 for all) - most rows ended up
    // with only 1-2 axes actually extreme, since 5 independent draws
    // rarely land extreme together, diluting exposure to genuinely
    // all-maxed throws like the vx=65,vy=100,mass=20 case that motivated
    // this in the first place (measured: didn't improve that throw's
    // error, or the model's overall x accuracy - see IMPROVEMENTS.md).
    // Fixed below: draw ONE shared extremity per row (how extreme this
    // row is, overall) and apply it to every axis via biasedSample's
    // extremity parameter, so a high-extremity row pushes most/all axes
    // toward their bounds together, not independently.

    std::ofstream out("throw_results.csv");
    out << "vx0,vy0,mass,height0,windAccel,final_x,final_y,maxHeight,timeToLand,bounceCount,apexTime,finalVx\n";

    for (int i = 0; i < throwCount; ++i) {
        float vx0 = vxDist(gen);
        float vy0 = vyDist(gen);
        float mass = massDist(gen);
        float height0 = heightDist(gen);
        float windAccel = windDist(gen);
        ThrowResult r = simulateThrow(vx0, vy0, mass, height0, windAccel);

        out << r.vx0 << "," << r.vy0 << "," << r.mass << "," << r.height0 << "," << r.windAccel << ","
            << r.finalX << "," << r.finalY << "," << r.maxHeight << ","
            << r.timeToLand << "," << r.bounceCount << "," << r.apexTime << "," << r.finalVx << "\n";
    }

    // Half genuinely near-corner (every axis pinned close to a randomly
    // chosen min/max per row, small jitter) - directly guarantees coverage
    // of throws like "everything maxed at once", not just a smooth bias
    // toward it. Half joint-extremity biased (a shared, continuously-
    // varying extremity per row applied to every axis) - broader coverage
    // of "fairly extreme but not literally pinned" throws in between the
    // true corners and the uniform-sampled interior.
    const int pinnedCount = cornerCount / 2;
    std::uniform_int_distribution<int> coinFlip(0, 1);
    for (int i = 0; i < pinnedCount; ++i) {
        float jitter = 0.04f; // fraction of each axis's range, small deliberate spread off the exact bound
        auto pinned = [&](float lo, float hi) {
            float bound = coinFlip(gen) == 0 ? lo : hi;
            float span = (hi - lo) * jitter;
            std::uniform_real_distribution<float> jitterDist(-span, span);
            float v = bound + jitterDist(gen);
            return v < lo ? lo : (v > hi ? hi : v);
        };
        float vx0 = pinned(VX_MIN, VX_MAX);
        float vy0 = pinned(VY_MIN, VY_MAX);
        float mass = pinned(MASS_MIN, MASS_MAX);
        float height0 = pinned(HEIGHT_MIN, HEIGHT_MAX);
        float windAccel = pinned(WIND_MIN, WIND_MAX);
        ThrowResult r = simulateThrow(vx0, vy0, mass, height0, windAccel);

        out << r.vx0 << "," << r.vy0 << "," << r.mass << "," << r.height0 << "," << r.windAccel << ","
            << r.finalX << "," << r.finalY << "," << r.maxHeight << ","
            << r.timeToLand << "," << r.bounceCount << "," << r.apexTime << "," << r.finalVx << "\n";
    }
    for (int i = pinnedCount; i < cornerCount; ++i) {
        float extremity = 0.4f + 0.6f * unit01(gen); // [0.4, 1.0] - always at least moderately extreme
        float vx0 = biasedSample(VX_MIN, VX_MAX, cornerPower, extremity);
        float vy0 = biasedSample(VY_MIN, VY_MAX, cornerPower, extremity);
        float mass = biasedSample(MASS_MIN, MASS_MAX, cornerPower, extremity);
        float height0 = biasedSample(HEIGHT_MIN, HEIGHT_MAX, cornerPower, extremity);
        float windAccel = biasedSample(WIND_MIN, WIND_MAX, cornerPower, extremity);
        ThrowResult r = simulateThrow(vx0, vy0, mass, height0, windAccel);

        out << r.vx0 << "," << r.vy0 << "," << r.mass << "," << r.height0 << "," << r.windAccel << ","
            << r.finalX << "," << r.finalY << "," << r.maxHeight << ","
            << r.timeToLand << "," << r.bounceCount << "," << r.apexTime << "," << r.finalVx << "\n";
    }

    out.close();
    std::cout << throwCount << " uniform + " << cornerCount << " corner-biased throws simulated. "
               << "Results in throw_results.csv" << std::endl;

    return 0;
}
