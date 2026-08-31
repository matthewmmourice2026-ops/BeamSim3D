#ifndef PHYSICS_H
#define PHYSICS_H

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <raylib.h>
#include <raymath.h>

struct Node3D {
    float position[3];
    float velocity[3];
    float force[3];
    float mass;
    float omega; // Angular velocity
    float theta; // Angular position
};

struct Beam3D {
    Node3D* node1;
    Node3D* node2;
    float restLength;
    float stiffness;
    float damping;
    float deformThreshold;
    float breakThreshold;
    bool isBroken;
};

struct Wheel {
    Node3D* node = nullptr;
    float radius = 0.35f;
    float torque = 0.0f;
    float angularVelocity = 0.0f;
};

struct Engine {
    float rpm = 1000.0f;
    float idle_rpm = 1000.0f;
    float idleRpm = 1000.0f;
    float max_rpm = 7000.0f;
    float maxRpm = 7000.0f;
    float peak_torque = 300.0f;
    float peakTorque = 300.0f;
    float torqueCurve[100];
    float powerCurve[100];
};

struct Transmission {
    std::vector<float> gearRatios;
    int currentGear = 1;
    float clutchEngagement = 0.0f;
};

struct Differential {
    float torqueDistribution; // 0 for open differential, 1 for locked differential
};

struct Triangle {
    Node3D* node1;
    Node3D* node2;
    Node3D* node3;
};

class SoftBody {
public:
    std::vector<Node3D> nodes;
    std::vector<Beam3D> beams;
    std::vector<Wheel> wheels;
    std::vector<Triangle> triangles;
    Engine engine;
    Transmission transmission;
    Differential differential;
    Vector3 chassisCenter;
    float steeringAngle;

    void update(float deltaTime);
    void applyGravity();
    void applyGroundCollision();
    void applyWheelTorque(float deltaTime);
    void applySteering(float deltaTime);
    void applyTraction(float deltaTime);
    void updateEngine(float deltaTime);
    void updateTransmission(float deltaTime);
    void updateDifferential(float deltaTime);
    void loadConfig(const std::string& filename);
    void reset();
    void calculateCenterOfMass();
    bool loadHeightmap(const std::string& filename);
    void applyAerodynamicForces(float deltaTime);
};

#endif // PHYSICS_H
