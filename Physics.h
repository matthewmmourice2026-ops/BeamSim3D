#ifndef PHYSICS_H
#define PHYSICS_H

#include <vector>
#include <cmath>

struct Node3D {
    float position[3];
    float velocity[3];
    float force[3];
    float mass;
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

class SoftBody {
public:
    std::vector<Node3D> nodes;
    std::vector<Beam3D> beams;

    void update(float deltaTime);
    void applyGravity();
    void applyGroundCollision();
};

#endif // PHYSICS_H
