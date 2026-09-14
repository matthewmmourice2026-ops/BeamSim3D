#ifndef THROW_RANGES_H
#define THROW_RANGES_H

// Playable/trainable range for throw parameters, shared between
// ThrowSim.cpp (dataset generation) and ThrowGame.cpp (player controls)
// so they can't drift apart. A trained model is only meaningful for
// inputs inside the range it actually saw during training - if these
// two ever disagree, the game would let you throw at values the model
// has never seen, producing silent garbage predictions with no error.
static const float VX_MIN = -15.0f, VX_MAX = 65.0f;
static const float VY_MIN = 5.0f, VY_MAX = 100.0f;
static const float MASS_MIN = 0.5f, MASS_MAX = 20.0f;

#endif // THROW_RANGES_H
