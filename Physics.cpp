#include "Physics.h"

void SoftBody::update(float deltaTime) {
    applyGravity();
    applyGroundCollision();

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
        }
    }
}
