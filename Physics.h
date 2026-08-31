#ifndef PHYSICS_H
#define PHYSICS_H

#include <vector>
#include <cmath>
#include <fstream> // Include for std::ifstream
#include <nlohmann/json.hpp>

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
    Node3D* node;
    float radius;
    float torque; // Torque applied to the wheel
    float angularVelocity; // Angular velocity of the wheel
    bool isDriven; // Whether the wheel is driven
};

struct Engine {
    float rpm;
    float maxRpm;
    float torqueCurve[100];
    float powerCurve[100];
};

struct Transmission {
    float gearRatios[6];
    int currentGear;
    float clutchEngagement;
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
    void loadHeightmap(const std::string& filename);
    void applyAerodynamicForces(float deltaTime);
};

#endif // PHYSICS_H
