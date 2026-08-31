#include "Physics.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <cmath>
#include <limits>

using json = nlohmann::json;

void SoftBody::loadConfig(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << filename << std::endl;
        // Generate a basic vehicle
        nodes.clear();
        beams.clear();
        wheels.clear();
        triangles.clear();

        // Define 8 nodes for a box (width 2, length 4, height 1)
        nodes.push_back({ {-1.0f, 2.0f, -2.0f}, {-1.0f, 2.0f, -2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });
        nodes.push_back({ { 1.0f, 2.0f, -2.0f}, { 1.0f, 2.0f, -2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });
        nodes.push_back({ { 1.0f, 2.0f,  2.0f}, { 1.0f, 2.0f,  2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });
        nodes.push_back({ {-1.0f, 2.0f,  2.0f}, {-1.0f, 2.0f,  2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });
        nodes.push_back({ {-1.0f, 1.0f, -2.0f}, {-1.0f, 1.0f, -2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });
        nodes.push_back({ { 1.0f, 1.0f, -2.0f}, { 1.0f, 1.0f, -2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });
        nodes.push_back({ { 1.0f, 1.0f,  2.0f}, { 1.0f, 1.0f,  2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });
        nodes.push_back({ {-1.0f, 1.0f,  2.0f}, {-1.0f, 1.0f,  2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });

        // Define beams for the box
        beams.push_back({ &nodes[0], &nodes[1], Vector3Distance(nodes[0].position, nodes[1].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[0], &nodes[2], Vector3Distance(nodes[0].position, nodes[2].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[1], &nodes[3], Vector3Distance(nodes[1].position, nodes[3].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[2], &nodes[3], Vector3Distance(nodes[2].position, nodes[3].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[0], &nodes[4], Vector3Distance(nodes[0].position, nodes[4].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[1], &nodes[5], Vector3Distance(nodes[1].position, nodes[5].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[2], &nodes[6], Vector3Distance(nodes[2].position, nodes[6].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[3], &nodes[7], Vector3Distance(nodes[3].position, nodes[7].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[4], &nodes[5], Vector3Distance(nodes[4].position, nodes[5].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[5], &nodes[7], Vector3Distance(nodes[5].position, nodes[7].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[6], &nodes[4], Vector3Distance(nodes[6].position, nodes[4].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[7], &nodes[6], Vector3Distance(nodes[7].position, nodes[6].position), 100.0f, 0.1f, 1000.0f, false });

        // Define wheels
        wheels.push_back({ &nodes[4], 0.5f, 1000.0f, 0.8f, true });
        wheels.push_back({ &nodes[5], 0.5f, 1000.0f, 0.8f, true });
        wheels.push_back({ &nodes[6], 0.5f, 1000.0f, 0.8f, false });
        wheels.push_back({ &nodes[7], 0.5f, 1000.0f, 0.8f, false });

        // Define engine and transmission
        engine.rpm = 1000;
        engine.max_rpm = 7000;
        engine.peak_torque = 300;

        transmission.finalDrive = 3.5f;
        transmission.gearRatios = {-3.0f, 0.0f, 3.5f, 2.1f, 1.4f, 1.0f};
        transmission.currentGear = 2;

        std::cout << "Fallback vehicle generated" << std::endl;
        return;
    }

    try {
        json config = json::parse(file, nullptr, false);
        if (config.is_discarded()) {
            throw std::runtime_error("Failed to parse config file");
        }

        nodes.clear();
        beams.clear();
        wheels.clear();
        triangles.clear();

        for (const auto& nodeJson : config["nodes"]) {
            Node3D node;
            node.position[0] = nodeJson["x"];
            node.position[1] = nodeJson["y"];
            node.position[2] = nodeJson["z"];
            node.initialPosition[0] = nodeJson["x"];
            node.initialPosition[1] = nodeJson["y"];
            node.initialPosition[2] = nodeJson["z"];
            node.velocity[0] = 0.0f;
            node.velocity[1] = 0.0f;
            node.velocity[2] = 0.0f;
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

        if (config.count("triangles")) {
            for (const auto& triangleJson : config["triangles"]) {
                Triangle triangle;
                triangle.node1 = &nodes[triangleJson["node1"]];
                triangle.node2 = &nodes[triangleJson["node2"]];
                triangle.node3 = &nodes[triangleJson["node3"]];
                triangles.push_back(triangle);
            }
        }

        file.close();
    } catch (...) {
        std::cerr << "Error parsing config file: " << filename << std::endl;
        // Generate a basic vehicle
        nodes.clear();
        beams.clear();
        wheels.clear();
        triangles.clear();

        // Define 8 nodes for a box (width 2, length 4, height 1)
        nodes.push_back({ {-1.0f, 2.0f, -2.0f}, {-1.0f, 2.0f, -2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });
        nodes.push_back({ { 1.0f, 2.0f, -2.0f}, { 1.0f, 2.0f, -2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });
        nodes.push_back({ { 1.0f, 2.0f,  2.0f}, { 1.0f, 2.0f,  2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });
        nodes.push_back({ {-1.0f, 2.0f,  2.0f}, {-1.0f, 2.0f,  2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });
        nodes.push_back({ {-1.0f, 1.0f, -2.0f}, {-1.0f, 1.0f, -2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });
        nodes.push_back({ { 1.0f, 1.0f, -2.0f}, { 1.0f, 1.0f, -2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });
        nodes.push_back({ { 1.0f, 1.0f,  2.0f}, { 1.0f, 1.0f,  2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });
        nodes.push_back({ {-1.0f, 1.0f,  2.0f}, {-1.0f, 1.0f,  2.0f}, {0.0f, 0.0f, 0.0f}, 100.0f, 0.0f, 0.0f });

        // Define beams for the box
        beams.push_back({ &nodes[0], &nodes[1], Vector3Distance(nodes[0].position, nodes[1].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[0], &nodes[2], Vector3Distance(nodes[0].position, nodes[2].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[1], &nodes[3], Vector3Distance(nodes[1].position, nodes[3].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[2], &nodes[3], Vector3Distance(nodes[2].position, nodes[3].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[0], &nodes[4], Vector3Distance(nodes[0].position, nodes[4].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[1], &nodes[5], Vector3Distance(nodes[1].position, nodes[5].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[2], &nodes[6], Vector3Distance(nodes[2].position, nodes[6].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[3], &nodes[7], Vector3Distance(nodes[3].position, nodes[7].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[4], &nodes[5], Vector3Distance(nodes[4].position, nodes[5].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[5], &nodes[7], Vector3Distance(nodes[5].position, nodes[7].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[6], &nodes[4], Vector3Distance(nodes[6].position, nodes[4].position), 100.0f, 0.1f, 1000.0f, false });
        beams.push_back({ &nodes[7], &nodes[6], Vector3Distance(nodes[7].position, nodes[6].position), 100.0f, 0.1f, 1000.0f, false });

        // Define wheels
        wheels.push_back({ &nodes[4], 0.5f, 1000.0f, 0.8f, true });
        wheels.push_back({ &nodes[5], 0.5f, 1000.0f, 0.8f, true });
        wheels.push_back({ &nodes[6], 0.5f, 1000.0f, 0.8f, false });
        wheels.push_back({ &nodes[7], 0.5f, 1000.0f, 0.8f, false });

        // Define engine and transmission
        engine.rpm = 1000;
        engine.max_rpm = 7000;
        engine.peak_torque = 300;

        transmission.finalDrive = 3.5f;
        transmission.gearRatios = {-3.0f, 0.0f, 3.5f, 2.1f, 1.4f, 1.0f};
        transmission.currentGear = 2;

        std::cout << "Fallback vehicle generated" << std::endl;
    }
}

void SoftBody::calculateCenterOfMass() {
    if (nodes.empty()) {
        chassisCenter = {0, 0, 0};
        return;
    }

    Vector3 sum = {0, 0, 0};
    float totalMass = 0.0f;

    for (const auto& node : nodes) {
        sum.x += node.position[0] * node.mass;
        sum.y += node.position[1] * node.mass;
        sum.z += node.position[2] * node.mass;
        totalMass += node.mass;
    }

    chassisCenter.x = sum.x / totalMass;
    chassisCenter.y = sum.y / totalMass;
    chassisCenter.z = sum.z / totalMass;

    if (std::isnan(chassisCenter.x) || std::isnan(chassisCenter.y) || std::isnan(chassisCenter.z)) {
        chassisCenter = {0.0f, 1.0f, 0.0f};
    }
}
