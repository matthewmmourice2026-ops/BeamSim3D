#include "Physics.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

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
    for (auto& node : nodes) {
        if (node.position[1] < 0.0f) {
            node.position[1] = 0.0f;
            node.velocity[1] *= -0.8f; // Damping effect

            // Ground friction
            float frictionForce = 0.5f * node.velocity[0];
            node.force[0] -= frictionForce;
        }
    }
}

void SoftBody::applyWheelTorque(float deltaTime) {
    for (auto& wheel : wheels) {
        wheel.angularVelocity += wheel.torque / wheel.radius * deltaTime;
    }
}

void SoftBody::applySteering(float deltaTime) {
    // Implement steering logic here
}

void SoftBody::applyTraction(float deltaTime) {
    // Implement traction logic here
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
        wheel.torque = effectiveTorque;
    }
}

void SoftBody::updateDifferential(float deltaTime) {
    if (differential.torqueDistribution == 1.0f) {
        // Locked differential
        float totalTorque = 0.0f;
        for (auto& wheel : wheels) {
            totalTorque += wheel.torque;
        }
        float avgTorque = totalTorque / wheels.size();
        for (auto& wheel : wheels) {
            wheel.torque = avgTorque;
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

    std::string line;
    std::vector<Node3D> newNodes;
    std::vector<Beam3D> newBeams;

    while (std::getline(file, line)) {
        if (line.find("nodes") != std::string::npos) {
            std::getline(file, line);
            while (line.find("}") == std::string::npos) {
                std::getline(file, line);
                if (line.find("{") != std::string::npos) {
                    Node3D node;
                    node.mass = 0.0f;
                    node.omega = 0.0f;
                    node.theta = 0.0f;
                    while (line.find("}") == std::string::npos) {
                        std::getline(file, line);
                        if (line.find("id") != std::string::npos) {
                            std::stringstream ss(line);
                            std::string temp;
                            ss >> temp >> node.id;
                        } else if (line.find("x") != std::string::npos) {
                            std::stringstream ss(line);
                            std::string temp;
                            ss >> temp >> node.position[0];
                        } else if (line.find("y") != std::string::npos) {
                            std::stringstream ss(line);
                            std::string temp;
                            ss >> temp >> node.position[1];
                        } else if (line.find("z") != std::string::npos) {
                            std::stringstream ss(line);
                            std::string temp;
                            ss >> temp >> node.position[2];
                        } else if (line.find("mass") != std::string::npos) {
                            std::stringstream ss(line);
                            std::string temp;
                            ss >> temp >> node.mass;
                        }
                    }
                    newNodes.push_back(node);
                }
            }
        } else if (line.find("beams") != std::string::npos) {
            std::getline(file, line);
            while (line.find("}") == std::string::npos) {
                std::getline(file, line);
                if (line.find("{") != std::string::npos) {
                    Beam3D beam;
                    beam.restLength = 0.0f;
                    beam.stiffness = 0.0f;
                    beam.damping = 0.0f;
                    beam.deformThreshold = 0.0f;
                    beam.breakThreshold = 0.0f;
                    beam.isBroken = false;
                    while (line.find("}") == std::string::npos) {
                        std::getline(file, line);
                        if (line.find("node1") != std::string::npos) {
                            std::stringstream ss(line);
                            std::string temp;
                            ss >> temp >> beam.node1;
                        } else if (line.find("node2") != std::string::npos) {
                            std::stringstream ss(line);
                            std::string temp;
                            ss >> temp >> beam.node2;
                        } else if (line.find("restLength") != std::string::npos) {
                            std::stringstream ss(line);
                            std::string temp;
                            ss >> temp >> beam.restLength;
                        } else if (line.find("stiffness") != std::string::npos) {
                            std::stringstream ss(line);
                            std::string temp;
                            ss >> temp >> beam.stiffness;
                        } else if (line.find("damping") != std::string::npos) {
                            std::stringstream ss(line);
                            std::string temp;
                            ss >> temp >> beam.damping;
                        } else if (line.find("breakThreshold") != std::string::npos) {
                            std::stringstream ss(line);
                            std::string temp;
                            ss >> temp >> beam.breakThreshold;
                        }
                    }
                    newBeams.push_back(beam);
                }
            }
        }
    }

    nodes = newNodes;
    beams = newBeams;

    file.close();
}

void SoftBody::reset() {
    loadConfig("vehicle.json");
}
