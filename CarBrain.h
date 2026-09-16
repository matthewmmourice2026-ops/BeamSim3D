#ifndef CAR_BRAIN_H
#define CAR_BRAIN_H

#include <raylib.h>
#include <cmath>
#include <cstring>

// A car's "brain": a tiny MLP, trained by a genetic algorithm (mutation +
// selection on the raw weights), not gradient descent - see the
// implementation plan's "pure C++, no Python" decision. Flat, padding-free
// POD (guarded by the static_assert below) so mutate/crossover/save/load
// are all trivial float[] operations instead of needing per-layer
// serialization code.
struct Brain {
    // 7 sensor rays + normalized speed + [prevSteer, prevThrottle] (its own
    // last action fed back in, so it can reason about what it was just
    // doing instead of reacting to each frame in isolation - short-term
    // memory, not a full RNN) + nearest-moving-traffic distance (traffic
    // can be timed/waited-out differently than a wall that never moves, but
    // the shared ray sensors can't tell the brain which is which).
    static constexpr int IN = 11;
    static constexpr int HID = 16; // widened from 10 for the extra inputs' worth of reasoning capacity
    // steering, throttle, then 4 purchase-preference scores (engine,
    // tires, armor, "keep setup" pack) - the SAME evolved brain decides
    // both how to drive and, when it's earned enough currency, what to
    // spend on next (highest score among affordable options wins). This
    // ties every purchase decision into the genetic algorithm exactly
    // like driving skill: a brain that spends well out-competes one that
    // doesn't, same as one that drives well - not a hardcoded rule.
    static constexpr int OUT = 6;

    float w1[HID * IN];
    float b1[HID];
    float w2[OUT * HID];
    float b2[OUT];
};

static_assert(sizeof(Brain) == (Brain::HID * Brain::IN + Brain::HID + Brain::OUT * Brain::HID + Brain::OUT) * sizeof(float),
              "Brain must be a flat, padding-free float array");

constexpr int BRAIN_WEIGHT_COUNT = sizeof(Brain) / sizeof(float);

inline float* brainWeights(Brain& b) { return reinterpret_cast<float*>(&b); }
inline const float* brainWeights(const Brain& b) { return reinterpret_cast<const float*>(&b); }

inline void forward(const Brain& b, const float in[Brain::IN], float out[Brain::OUT]) {
    float hidden[Brain::HID];
    for (int h = 0; h < Brain::HID; h++) {
        float sum = b.b1[h];
        for (int i = 0; i < Brain::IN; i++) sum += b.w1[h * Brain::IN + i] * in[i];
        hidden[h] = tanhf(sum);
    }
    for (int o = 0; o < Brain::OUT; o++) {
        float sum = b.b2[o];
        for (int h = 0; h < Brain::HID; h++) sum += b.w2[o * Brain::HID + h] * hidden[h];
        out[o] = tanhf(sum); // [0]=steering [1]=throttle [2..4]=engine/tire/armor scores [5]=keep-setup-pack score, all in [-1,1]
    }
}

// Box-Muller, same GetRandomValue-based style as ThrowGame.cpp's randRangeF.
inline float brainRandGaussian() {
    float u1 = Clamp((float)GetRandomValue(1, 10000) / 10000.0f, 1e-6f, 1.0f);
    float u2 = (float)GetRandomValue(0, 10000) / 10000.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

inline Brain randomBrain() {
    Brain b;
    float* w = brainWeights(b);
    for (int i = 0; i < BRAIN_WEIGHT_COUNT; i++) w[i] = (float)GetRandomValue(-1000, 1000) / 1000.0f;
    return b;
}

inline Brain mutate(const Brain& parent, float rate, float strength) {
    Brain child = parent;
    float* w = brainWeights(child);
    for (int i = 0; i < BRAIN_WEIGHT_COUNT; i++) {
        if ((float)GetRandomValue(0, 10000) / 10000.0f < rate) w[i] += brainRandGaussian() * strength;
    }
    return child;
}

inline Brain crossover(const Brain& a, const Brain& b) {
    Brain child;
    const float* wa = brainWeights(a);
    const float* wb = brainWeights(b);
    float* wc = brainWeights(child);
    for (int i = 0; i < BRAIN_WEIGHT_COUNT; i++) wc[i] = (GetRandomValue(0, 1) == 0) ? wa[i] : wb[i];
    return child;
}

#endif // CAR_BRAIN_H
