import torch

# Must match ThrowSim.cpp's dragCoeff constant.
DRAG_COEFF = 0.2
# Must match ThrowSim.cpp's gravity constant.
GRAVITY = 9.81


def physics_baseline_x(raw_inputs: torch.Tensor) -> torch.Tensor:
    """NOT currently used by add_engineered_features() - see IMPROVEMENTS.md.
    Tried as a 7th input feature (fc1 widened 6->7); the resulting checkpoint
    regressed on every output (x MAE 1.871->4.979, y 0.178->0.475, maxHeight
    0.404->3.235), not just x. Confounded with patience=35 (vs. the 100 that
    produced the deployed checkpoint) possibly stopping training well short
    of convergence, so this isn't a clean verdict on the feature itself -
    kept here, unused, for a properly-controlled retry (same patience as the
    deployed checkpoint, isolate this one variable) rather than deleted.

    Analytical wind-aware horizontal displacement, evaluated at an
    estimated flight time t_est.

    Solving dvx/dt = -c*vx/m + w (c=dragCoeff, m=mass, w=windAccel) gives
    vx(t) = (vx0 - m*w/c)*exp(-c*t/m) + m*w/c, and integrating that gives
    x(t) = (vx0 - terminal_vx)*tau*(1 - exp(-t/tau)) + terminal_vx*t, with
    terminal_vx = m*w/c and tau = m/c.

    t_est is a leak-free (no ground-truth columns used) vertical-ballistics
    estimate: the time for a vertical-drag-free projectile launched at vy0
    from height0 to first reach the ground, t_est = (vy0 + sqrt(vy0^2 +
    2*g*height0)) / g. This is a known, measured-limited estimate (see the
    IMPROVEMENTS.md / experiment notes this feature was added alongside):
    it covers only the first ballistic descent, not ThrowSim.cpp's
    post-impact bounce/friction/settling phase, so on throws that bounce a
    lot it can understate the real flight time by 10+ seconds and this
    baseline alone is a poor standalone predictor of final_x (empirically
    ~40x worse MAE than the trained model). It's included as one more
    input feature for the network to weigh, not as a replacement for
    learning - see x_inf below for the same caveat pattern.
    """
    vx0 = raw_inputs[:, 0]
    vy0 = raw_inputs[:, 1]
    mass = raw_inputs[:, 2]
    height0 = raw_inputs[:, 3]
    windAccel = raw_inputs[:, 4]

    t_est = (vy0 + torch.sqrt(vy0 ** 2 + 2 * GRAVITY * height0)) / GRAVITY
    terminal_vx = mass * windAccel / DRAG_COEFF
    tau = mass / DRAG_COEFF
    return (vx0 - terminal_vx) * tau * (1 - torch.exp(-t_est / tau)) + terminal_vx * t_est


def add_engineered_features(raw_inputs: torch.Tensor) -> torch.Tensor:
    """Takes the raw [vx0, vy0, mass, height0, windAccel] columns and appends
    a physics-informed feature: the asymptotic horizontal range under pure
    exponential drag decay (ax = -k*vx/m integrated to t=infinity, ignoring
    terrain/bounce/wind/height/finite-time). This is the dominant factor
    driving final_x, so it gives the network a head start instead of
    learning that relationship from raw inputs alone. vx0 is column 0, mass
    is column 2 - unaffected by height0/windAccel being appended after
    mass, not between vx0 and mass. Works for both a single row (shape
    (1, 5)) and a batch (shape (N, 5)).

    (physics_baseline_x() above is a second, more sophisticated engineered
    feature that was tried alongside this one and reverted - see its own
    docstring and IMPROVEMENTS.md.)
    """
    vx0 = raw_inputs[:, 0]
    mass = raw_inputs[:, 2]
    x_inf = (vx0 * mass / DRAG_COEFF).unsqueeze(1)
    return torch.cat([raw_inputs, x_inf], dim=1)
