import torch

# Must match ThrowSim.cpp's dragCoeff constant.
DRAG_COEFF = 0.2


def add_engineered_features(raw_inputs: torch.Tensor) -> torch.Tensor:
    """Takes the raw [vx0, vy0, mass] columns and appends a physics-informed
    feature: the asymptotic horizontal range under pure exponential drag
    decay (ax = -k*vx/m integrated to t=infinity, ignoring terrain/bounce).
    This is the dominant factor driving final_x, so it gives the network a
    head start instead of learning that relationship from raw inputs alone.
    Works for both a single row (shape (1, 3)) and a batch (shape (N, 3)).
    """
    vx0 = raw_inputs[:, 0]
    mass = raw_inputs[:, 2]
    x_inf = (vx0 * mass / DRAG_COEFF).unsqueeze(1)
    return torch.cat([raw_inputs, x_inf], dim=1)
