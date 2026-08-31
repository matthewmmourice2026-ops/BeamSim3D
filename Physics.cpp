void SoftBody::loadConfig(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << filename << std::endl;
        procedurallyGenerateVehicle();
        return;
    }

    json config = json::parse(file, nullptr, false);
    if (config.is_discarded()) {
        std::cerr << "Failed to parse config file: " << filename << std::endl;
        procedurallyGenerateVehicle();
        return;
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
}
