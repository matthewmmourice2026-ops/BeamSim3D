import math
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import add_engineered_features
from models.ImprovedNeuralNetwork import ImprovedNeuralNetwork

# Persistent version of predict.py: the game was shelling out to
# `python3 predict.py ...` once per throw, which pays Python startup +
# torch import + checkpoint load on every single call - the dominant
# cost, not the actual forward pass. This loads the model once and then
# answers one request per stdin line, so a throw only pays a forward pass.
repo_root = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(repo_root, "models", "ImprovedNeuralNetwork.pth")
checkpoint = torch.load(checkpoint_path)

model = ImprovedNeuralNetwork()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

input_mean = checkpoint["input_mean"]
input_std = checkpoint["input_std"]

print("ready", flush=True)

for line in sys.stdin:
    parts = line.split()
    if len(parts) != 5:
        print("error", flush=True)
        continue
    vx0, vy0, mass, height0, windAccel = (float(p) for p in parts)

    x = torch.tensor([[vx0, vy0, mass, height0, windAccel]], dtype=torch.float32)
    x = add_engineered_features(x)
    x = (x - input_mean) / input_std
    with torch.no_grad():
        predictions = model(x)

    landing_x, landing_y, max_height, time_to_land, bounce_count, apex_time, final_vx, log_var = (
        predictions[0, 0].item(), predictions[0, 1].item(), predictions[0, 2].item(),
        predictions[0, 3].item(), predictions[0, 4].item(), predictions[0, 5].item(), predictions[0, 6].item(),
        predictions[0, 7].item(),
    )
    # See predict.py: log_var is log per-axis variance, radial error is
    # Rayleigh(sigma)-distributed, and its MEAN (sigma*sqrt(pi/2)) - not
    # raw sigma - is what's comparable to an actual mean landing error.
    sigma = math.exp(0.5 * log_var)
    uncertainty_std = sigma * math.sqrt(math.pi / 2.0)

    print(f"{landing_x},{landing_y},{max_height},{time_to_land},{bounce_count},{apex_time},{final_vx},{uncertainty_std}", flush=True)
