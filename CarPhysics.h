#ifndef CAR_PHYSICS_H
#define CAR_PHYSICS_H

#include <raylib.h>
#include <cmath>

// Simple kinematic car model: direct integration of speed and heading, no
// forces/mass/slip. Deliberately NOT a rigid-body model - cars corner
// instantly with no drift, which looks a little robotic but keeps the
// genetic-algorithm testbed simple enough to debug by inspection. See the
// "Kinematic, not dynamic" note in the implementation plan.
struct CarState {
    Vector3 pos;
    float heading; // radians, 0 = +X
    float speed;   // signed: negative = reverse
};

// enginePerf scales top speed/accel (damage pulls it toward 0.5, engine
// upgrades push it above 1.0); turnPerf scales turn rate the same way for
// tire upgrades/damage - kept separate so an engine upgrade doesn't
// silently also improve handling and vice versa. Both default to 1.0
// (undamaged, stock parts).
inline void updateCar(CarState& c, float steering, float throttle, float dt, float enginePerf = 1.0f, float turnPerf = 1.0f) {
    steering = Clamp(steering, -1.0f, 1.0f);
    throttle = Clamp(throttle, -1.0f, 1.0f);
    enginePerf = Clamp(enginePerf, 0.05f, 3.0f);
    turnPerf = Clamp(turnPerf, 0.05f, 3.0f);

    const float MAX_SPEED = 40.0f * enginePerf;
    const float ACCEL = 25.0f * enginePerf;
    const float BRAKE = 40.0f;
    const float DRAG = 0.35f;
    const float MAX_TURN_RATE = 2.4f * turnPerf; // rad/s at full turn authority

    float accel = (throttle >= 0.0f) ? throttle * ACCEL : throttle * BRAKE;
    c.speed += accel * dt;
    c.speed -= c.speed * DRAG * dt;
    c.speed = Clamp(c.speed, -MAX_SPEED * 0.4f, MAX_SPEED);

    // No steering authority standing still - avoids cars spinning in place.
    float turnAuthority = Clamp(fabsf(c.speed) / MAX_SPEED, 0.15f, 1.0f);
    float turnDir = c.speed >= 0.0f ? 1.0f : -1.0f;
    c.heading += steering * MAX_TURN_RATE * turnAuthority * turnDir * dt;

    c.pos.x += cosf(c.heading) * c.speed * dt;
    c.pos.z += sinf(c.heading) * c.speed * dt;
}

#endif // CAR_PHYSICS_H
