import torch

# Must match ThrowSim.cpp's dragCoeff constant.
DRAG_COEFF = 0.2


def add_engineered_features(raw_inputs: torch.Tensor) -> torch.Tensor:
    """Takes the raw [vx0, vy0, mass, height0, windAccel] columns and appends
    a physics-informed feature: the asymptotic horizontal range under pure
    exponential drag decay (ax = -k*vx/m integrated to t=infinity, ignoring
    terrain/bounce/wind/height). This is the dominant factor driving
    final_x, so it gives the network a head start instead of learning that
    relationship from raw inputs alone. vx0 is column 0, mass is column 2 -
    unaffected by height0/windAccel being appended after mass, not between
    vx0 and mass. Works for both a single row (shape (1, 5)) and a batch
    (shape (N, 5)).
    """
    vx0 = raw_inputs[:, 0]
    mass = raw_inputs[:, 2]
    x_inf = (vx0 * mass / DRAG_COEFF).unsqueeze(1)
    return torch.cat([raw_inputs, x_inf], dim=1)
