import random
import subprocess

import torch

from models.ImprovedNeuralNetwork import ImprovedNeuralNetwork

THROW_SIM = "build/throw_sim"


def ground_truth(vx0, vy0, mass):
    result = subprocess.run(
        [THROW_SIM, str(vx0), str(vy0), str(mass)],
        capture_output=True, text=True, check=True,
    )
    final_x, final_y, max_height = result.stdout.strip().split(",")
    return float(final_x), float(final_y), float(max_height)


def load_model():
    checkpoint = torch.load("models/ImprovedNeuralNetwork.pth")
    model = ImprovedNeuralNetwork()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["input_mean"], checkpoint["input_std"]


def predict(model, input_mean, input_std, vx0, vy0, mass):
    x = torch.tensor([[vx0, vy0, mass]], dtype=torch.float32)
    x = (x - input_mean) / input_std
    with torch.no_grad():
        y = model(x)
    return y[0, 0].item(), y[0, 1].item(), y[0, 2].item()


if __name__ == "__main__":
    # Fixed seed so this test set stays the same across runs, makes
    # before/after comparisons between model versions fair. Same ranges
    # ThrowSim.cpp uses to generate throw_results.csv, but these specific
    # throws are never in that file.
    random.seed(123)
    num_test_throws = 30
    # Must match ThrowRanges.h (no shared include across C++/Python, so
    # keep these in sync by hand if that file changes).
    test_throws = [
        (random.uniform(-15.0, 65.0), random.uniform(5.0, 100.0), random.uniform(0.5, 20.0))
        for _ in range(num_test_throws)
    ]

    model, input_mean, input_std = load_model()

    print(f"{'vx0':>7} {'vy0':>7} {'mass':>6} | {'real_x':>9} {'pred_x':>9} {'err_x':>7} | "
          f"{'real_y':>8} {'pred_y':>8} {'err_y':>7} | {'real_h':>8} {'pred_h':>8} {'err_h':>7}")

    x_errors, y_errors, h_errors = [], [], []
    for vx0, vy0, mass in test_throws:
        real_x, real_y, real_h = ground_truth(vx0, vy0, mass)
        pred_x, pred_y, pred_h = predict(model, input_mean, input_std, vx0, vy0, mass)
        err_x, err_y, err_h = abs(real_x - pred_x), abs(real_y - pred_y), abs(real_h - pred_h)
        x_errors.append(err_x)
        y_errors.append(err_y)
        h_errors.append(err_h)

        print(f"{vx0:7.1f} {vy0:7.1f} {mass:6.1f} | {real_x:9.3f} {pred_x:9.3f} {err_x:7.3f} | "
              f"{real_y:8.3f} {pred_y:8.3f} {err_y:7.3f} | {real_h:8.3f} {pred_h:8.3f} {err_h:7.3f}")

    print(f"\nMean absolute error: x = {sum(x_errors) / len(x_errors):.3f}, "
          f"y = {sum(y_errors) / len(y_errors):.3f}, max_height = {sum(h_errors) / len(h_errors):.3f}")
