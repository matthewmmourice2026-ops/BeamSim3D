#include <iostream>
#include <fstream>
#include <random>
#include <cmath>

struct ThrowResult {
    float vx0, vy0, mass;
    float finalX, finalY;
};

// Rolling hills instead of flat ground, so the resting height actually
// depends on where the object lands, not always 0.
static float terrainHeight(float x) {
    return 2.0f * std::sin(x * 0.05f) + 0.5f * std::sin(x * 0.13f);
}

static ThrowResult simulateThrow(float vx0, float vy0, float mass) {
    const float dt = 0.01f;
    const float gravity = 9.81f;
    const float dragCoeff = 0.2f;
    const float restitution = 0.4f;
    const float groundFriction = 0.7f;
    const float restEps = 0.05f;
    const int maxSteps = 5000;

    float x = 0.0f, y = 1.0f;
    float vx = vx0, vy = vy0;

    for (int step = 0; step < maxSteps; ++step) {
        float ax = -dragCoeff * vx / mass;
        float ay = -gravity - dragCoeff * vy / mass;

        vx += ax * dt;
        vy += ay * dt;
        x += vx * dt;
        y += vy * dt;

        float ground = terrainHeight(x);
        if (y <= ground) {
            y = ground;
            vy = -vy * restitution;
            vx *= groundFriction;
            if (std::fabs(vy) < restEps) {
                vy = 0.0f;
            }
        }

        if (y == ground && std::fabs(vy) < restEps && std::fabs(vx) < restEps) {
            break;
        }
    }

    return { vx0, vy0, mass, x, y };
}

int main() {
    const int throwCount = 1000;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> vxDist(-15.0f, 15.0f);
    std::uniform_real_distribution<float> vyDist(5.0f, 25.0f);
    std::uniform_real_distribution<float> massDist(0.5f, 5.0f);

    std::ofstream out("throw_results.csv");
    out << "vx0,vy0,mass,final_x,final_y\n";

    for (int i = 0; i < throwCount; ++i) {
        float vx0 = vxDist(gen);
        float vy0 = vyDist(gen);
        float mass = massDist(gen);

        ThrowResult r = simulateThrow(vx0, vy0, mass);

        out << r.vx0 << "," << r.vy0 << "," << r.mass << ","
            << r.finalX << "," << r.finalY << "\n";
    }

    out.close();
    std::cout << throwCount << " throws simulated. Results in throw_results.csv" << std::endl;

    return 0;
}
