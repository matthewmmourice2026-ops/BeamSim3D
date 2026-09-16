import math
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import add_engineered_features
from models.ImprovedNeuralNetwork import ImprovedNeuralNetwork

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("usage: predict.py <vx0> <vy0> <mass> <height0> <windAccel>", file=sys.stderr)
        sys.exit(1)

    vx0, vy0, mass, height0, windAccel = (
        float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]),
    )

    repo_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(repo_root, "models", "ImprovedNeuralNetwork.pth")
    
    try:
        checkpoint = torch.load(checkpoint_path)
    except FileNotFoundError:
        print(f"Checkpoint file not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    model = ImprovedNeuralNetwork()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = torch.tensor([[vx0, vy0, mass, height0, windAccel]], dtype=torch.float32)
    x = add_engineered_features(x)
    x = (x - checkpoint["input_mean"]) / checkpoint["input_std"]
    with torch.no_grad():
        predictions = model(x)

    landing_x, landing_y, max_height, time_to_land, bounce_count, apex_time, final_vx, log_var = (
        predictions[0, 0].item(), predictions[0, 1].item(), predictions[0, 2].item(),
        predictions[0, 3].item(), predictions[0, 4].item(), predictions[0, 5].item(), predictions[0, 6].item(),
        predictions[0, 7].item(),
    )
    # log_var = log of the per-axis landing-position error variance
    # (sigma^2), so sigma = exp(0.5*log_var) is the per-axis std. With dx,
    # dy each ~ N(0, sigma^2), the radial landing-position error
    # sqrt(dx^2+dy^2) follows a Rayleigh(sigma) distribution, whose MEAN
    # is sigma*sqrt(pi/2) - not sigma itself. Reporting that mean (rather
    # than raw sigma) is what's directly comparable to an actual mean
    # landing error, same units as x/y.
    sigma = math.exp(0.5 * log_var)
    uncertainty_std = sigma * math.sqrt(math.pi / 2.0)

    print(f"{landing_x},{landing_y},{max_height},{time_to_land},{bounce_count},{apex_time},{final_vx},{uncertainty_std}")
