#include "Physics.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <raylib.h>

using json = nlohmann::json;

unsigned char heightmapData[64][64];

float GetTerrainHeight(float x, float z) {
    int ix = static_cast<int>(x + 32);
    int iz = static_cast<int>(z + 32);
    if (ix < 0 || ix >= 64 || iz < 0 || iz >= 64) return 0.0f;
    return static_cast<float>(heightmapData[ix][iz]) / 255.0f * 10.0f; // Scale height to a reasonable range
}

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

    calculateCenterOfMass();
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

            // Basic friction
            float frictionForce = 0.5f * node.velocity[0];
            wheel.node->force[0] -= frictionForce;
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
    float totalAngularVelocity = 0.0f;
    int drivenWheelCount = 0;
    for (const auto& wheel : wheels) {
        if (wheel.isDriven) {
            totalAngularVelocity += wheel.angularVelocity;
            drivenWheelCount++;
        }
    }

    if (drivenWheelCount > 0) {
        float averageAngularVelocity = totalAngularVelocity / drivenWheelCount;
        engine.rpm = averageAngularVelocity * transmission.gearRatios[transmission.currentGear] * transmission.finalDrive;
    }

    // Basic torque curve
    float torque = 0.0f;
    if (engine.rpm < engine.idle_rpm) {
        torque = 0.0f;
    } else if (engine.rpm > engine.max_rpm) {
        torque = 0.0f;
    } else {
        float rpmRange = engine.max_rpm - engine.idle_rpm;
        float rpmFraction = (engine.rpm - engine.idle_rpm) / rpmRange;
        torque = engine.peak_torque * (1.0f - std::pow(rpmFraction - 0.5f, 2));
    }

    engine.torqueCurve[int(engine.rpm)] = torque;
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
        wheel.friction = wheelJson["friction"];
        wheel.torque = 0.0f;
        wheel.angularVelocity = 0.0f;
        wheel.isDriven = wheelJson["is_driven"];
        wheels.push_back(wheel);
    }

    engine.idle_rpm = config["powertrain"]["engine"]["idle_rpm"];
    engine.max_rpm = config["powertrain"]["engine"]["max_rpm"];
    engine.peak_torque = config["powertrain"]["engine"]["peak_torque"];

    for (int i = 0; i < 6; i++) {
        transmission.gearRatios[i] = config["powertrain"]["transmission"]["gear_ratios"][i];
    }
    transmission.finalDrive = config["powertrain"]["transmission"]["final_drive"];

    file.close();
}

void SoftBody::reset() {
    loadConfig("vehicle.json");
}

void SoftBody::calculateCenterOfMass() {
    chassisCenter = { 0.0f, 0.0f, 0.0f };
    float totalMass = 0.0f;

    for (const auto& node : nodes) {
        chassisCenter.x += node.position[0] * node.mass;
        chassisCenter.y += node.position[1] * node.mass;
        chassisCenter.z += node.position[2] * node.mass;
        totalMass += node.mass;
    }

    chassisCenter.x /= totalMass;
    chassisCenter.y /= totalMass;
    chassisCenter.z /= totalMass;
}

void SoftBody::loadHeightmap(const std::string& filename) {
    Image image = LoadImage(filename.c_str());
    if (image.format != PIXELFORMAT_GRAYSCALE) {
        std::cerr << "Heightmap image must be grayscale." << std::endl;
        return;
    }

    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            heightmapData[x][y] = GetPixelColor(image, x, y).r;
        }
    }

    UnloadImage(image);
}
