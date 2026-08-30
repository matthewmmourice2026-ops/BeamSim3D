#include "Physics.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void SoftBody::update(float deltaTime) {
    applyGravity();
    applyGroundCollision();
    updateEngine(deltaTime);
    updateTransmission(deltaTime);
    updateDifferential(deltaTime);
    applyWheelTorque(deltaTime);
    applySteering(deltaTime);
    applyTraction(deltaTime);

    for (auto& beam : beams) {
        if (beam.isBroken) continue;

        float distance = std::sqrt(
            std::pow(beam.node2->position[0] - beam.node1->position[0], 2) +
            std::pow(beam.node2->position[1] - beam.node1->position[1], 2) +
            std::pow(beam.node2->position[2] - beam.node1->position[2], 2)
        );

        float forceMagnitude = beam.stiffness * (distance - beam.restLength);
        float direction[3] = {
            (beam.node2->position[0] - beam.node1->position[0]) / distance,
            (beam.node2->position[1] - beam.node1->position[1]) / distance,
            (beam.node2->position[2] - beam.node1->position[2]) / distance
        };

        if (std::abs(forceMagnitude) > beam.deformThreshold) {
            beam.restLength = distance;
        }

        if (std::abs(forceMagnitude) > beam.breakThreshold) {
            beam.isBroken = true;
            continue;
        }

        beam.node1->force[0] += forceMagnitude * direction[0];
        beam.node1->force[1] += forceMagnitude * direction[1];
        beam.node1->force[2] += forceMagnitude * direction[2];

        beam.node2->force[0] -= forceMagnitude * direction[0];
        beam.node2->force[1] -= forceMagnitude * direction[1];
        beam.node2->force[2] -= forceMagnitude * direction[2];
    }

    for (auto& node : nodes) {
        node.velocity[0] += node.force[0] / node.mass * deltaTime;
        node.velocity[1] += node.force[1] / node.mass * deltaTime;
        node.velocity[2] += node.force[2] / node.mass * deltaTime;

        node.position[0] += node.velocity[0] * deltaTime;
        node.position[1] += node.velocity[1] * deltaTime;
        node.position[2] += node.velocity[2] * deltaTime;

        node.force[0] = 0.0f;
        node.force[1] = 0.0f;
        node.force[2] = 0.0f;
    }
}

void SoftBody::applyGravity() {
    for (auto& node : nodes) {
        node.force[1] -= 9.81f * node.mass;
    }
}

void SoftBody::applyGroundCollision() {
    for (auto& wheel : wheels) {
        float distanceToGround = wheel.node->position[1] - wheel.radius;
        if (distanceToGround < 0.0f) {
            float penetrationDepth = -distanceToGround;
            float suspensionForce = wheel.springStiffness * penetrationDepth;
            wheel.node->force[1] += suspensionForce;
        }
    }
}

void SoftBody::applyWheelTorque(float deltaTime) {
    for (auto& wheel : wheels) {
        wheel.angularVelocity += wheel.torque / wheel.radius * deltaTime;
    }
}

void SoftBody::applySteering(float deltaTime) {
    if (steeringAngle != 0.0f) {
        for (auto& wheel : wheels) {
            if (wheel.isDriven) {
                float steeringFactor = std::atan2(wheel.node->position[2] - chassisCenter[2], wheel.node->position[0] - chassisCenter[0]);
                wheel.node->position[0] += steeringAngle * std::cos(steeringFactor);
                wheel.node->position[2] += steeringAngle * std::sin(steeringFactor);
            }
        }
    }
}

void SoftBody::applyTraction(float deltaTime) {
    for (auto& wheel : wheels) {
        if (wheel.isDriven) {
            float tractionForce = wheel.torque * wheel.friction;
            wheel.node->force[0] += tractionForce * std::cos(wheel.node->theta);
            wheel.node->force[2] += tractionForce * std::sin(wheel.node->theta);
        }
    }
}

void SoftBody::updateEngine(float deltaTime) {
    engine.rpm += engine.torqueCurve[int(engine.rpm)] * deltaTime;
    if (engine.rpm > engine.maxRpm) engine.rpm = engine.maxRpm;
}

void SoftBody::updateTransmission(float deltaTime) {
    if (transmission.currentGear == 0) {
        transmission.clutchEngagement = 0.0f;
    } else {
        transmission.clutchEngagement += (1.0f - transmission.clutchEngagement) * deltaTime;
    }

    float effectiveTorque = engine.torqueCurve[int(engine.rpm)] * transmission.gearRatios[transmission.currentGear] * transmission.clutchEngagement;
    for (auto& wheel : wheels) {
        if (wheel.isDriven) {
            wheel.torque = effectiveTorque;
        }
    }
}

void SoftBody::updateDifferential(float deltaTime) {
    if (differential.torqueDistribution == 1.0f) {
        // Locked differential
        float totalTorque = 0.0f;
        for (auto& wheel : wheels) {
            if (wheel.isDriven) {
                totalTorque += wheel.torque;
            }
        }
        float avgTorque = totalTorque / wheels.size();
        for (auto& wheel : wheels) {
            if (wheel.isDriven) {
                wheel.torque = avgTorque;
            }
        }
    }
    // Open differential logic can be implemented here if needed
}

void SoftBody::loadConfig(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << filename << std::endl;
        return;
    }

    json config = json::parse(file);

    nodes.clear();
    beams.clear();
    wheels.clear();

    for (const auto& nodeJson : config["nodes"]) {
        Node3D node;
        node.id = nodeJson["id"];
        node.position[0] = nodeJson["x"];
        node.position[1] = nodeJson["y"];
        node.position[2] = nodeJson["z"];
        node.mass = nodeJson["mass"];
        node.omega = 0.0f;
        node.theta = 0.0f;
        nodes.push_back(node);
    }

    for (const auto& beamJson : config["beams"]) {
        Beam3D beam;
        beam.node1 = &nodes[beamJson["node1"]];
        beam.node2 = &nodes[beamJson["node2"]];
        beam.restLength = beamJson["restLength"];
        beam.stiffness = beamJson["stiffness"];
        beam.damping = beamJson["damping"];
        beam.deformThreshold = 0.0f; // Assuming default value
        beam.breakThreshold = beamJson["breakThreshold"];
        beam.isBroken = false;
        beams.push_back(beam);
    }

    for (const auto& wheelJson : config["wheels"]) {
        Wheel wheel;
        wheel.node = &nodes[wheelJson["node_id"]];
        wheel.radius = wheelJson["radius"];
        wheel.springStiffness = wheelJson["spring_stiffness"];
        wheel.friction = wheelJson["friction"];
        wheel.torque = 0.0f;
        wheel.angularVelocity = 0.0f;
        wheel.isDriven = wheelJson["is_driven"];
        wheels.push_back(wheel);
    }

    file.close();
}

void SoftBody::reset() {
    loadConfig("vehicle.json");
}
