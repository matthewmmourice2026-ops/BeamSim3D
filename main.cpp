// Draw the vehicle's nodes, beams, and wheels
for (const auto& node : vehicle.nodes) {
    DrawSphere((Vector3){ node.position[0], node.position[1], node.position[2] }, 0.1f, RED);
}

for (const auto& beam : vehicle.beams) {
    DrawLine3D((Vector3){ beam.node1->position[0], beam.node1->position[1], beam.node1->position[2] },
               (Vector3){ beam.node2->position[0], beam.node2->position[1], beam.node2->position[2] }, BLUE);
}

for (const auto& wheel : vehicle.wheels) {
    DrawSphere((Vector3){ wheel.node->position[0], wheel.node->position[1], wheel.node->position[2] }, wheel.radius, GREEN);
}
