import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ImprovedNeuralNetwork import ImprovedNeuralNetwork

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: predict.py <vx0> <vy0> <mass>", file=sys.stderr)
        sys.exit(1)

    vx0, vy0, mass = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])

    repo_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint = torch.load(os.path.join(repo_root, "models", "ImprovedNeuralNetwork.pth"))

    model = ImprovedNeuralNetwork()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = torch.tensor([[vx0, vy0, mass]], dtype=torch.float32)
    x = (x - checkpoint["input_mean"]) / checkpoint["input_std"]
    with torch.no_grad():
        y = model(x)

    print(f"{y[0, 0].item()},{y[0, 1].item()}")
