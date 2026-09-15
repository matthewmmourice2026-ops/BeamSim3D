import math
import random
import subprocess

import torch

from features import add_engineered_features
from models.ImprovedNeuralNetwork import ImprovedNeuralNetwork

THROW_SIM = "build/throw_sim"


def ground_truth(vx0, vy0, mass, height0, wind_accel):
    result = subprocess.run(
        [THROW_SIM, str(vx0), str(vy0), str(mass), str(height0), str(wind_accel)],
        capture_output=True, text=True, check=True,
    )
    values = result.stdout.strip().split(",")
    return tuple(float(v) for v in values)


def load_model():
    checkpoint = torch.load("models/ImprovedNeuralNetwork.pth")
    model = ImprovedNeuralNetwork()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["input_mean"], checkpoint["input_std"]


def predict(model, input_mean, input_std, vx0, vy0, mass, height0, wind_accel):
    x = torch.tensor([[vx0, vy0, mass, height0, wind_accel]], dtype=torch.float32)
    x = add_engineered_features(x)
    x = (x - input_mean) / input_std
    with torch.no_grad():
        y = model(x)
    return tuple(y[0, i].item() for i in range(y.shape[1]))


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
        (random.uniform(-15.0, 65.0), random.uniform(5.0, 100.0), random.uniform(0.5, 20.0),
         random.uniform(0.5, 20.0), random.uniform(-3.0, 3.0))
        for _ in range(num_test_throws)
    ]

    model, input_mean, input_std = load_model()

    # Output order: final_x, final_y, maxHeight, timeToLand, bounceCount,
    # apexTime, finalVx. Full per-row table for the first 3 (readable at
    # 7 columns), aggregate-only MAE for the other 4 (21 columns of
    # per-row detail would be unreadable in a terminal).
    print(f"{'vx0':>7} {'vy0':>7} {'mass':>6} {'h0':>6} {'wind':>6} | {'real_x':>9} {'pred_x':>9} {'err_x':>7} | "
          f"{'real_y':>8} {'pred_y':>8} {'err_y':>7} | {'real_h':>8} {'pred_h':>8} {'err_h':>7}")

    x_errors, y_errors, h_errors = [], [], []
    t2l_errors, bounce_errors, apex_errors, vx_errors = [], [], [], []
    landing_errors, predicted_stds = [], []

    for vx0, vy0, mass, height0, wind_accel in test_throws:
        real_x, real_y, real_h, real_t2l, real_bounce, real_apex, real_vx = ground_truth(vx0, vy0, mass, height0, wind_accel)
        pred_x, pred_y, pred_h, pred_t2l, pred_bounce, pred_apex, pred_vx, pred_log_var = predict(model, input_mean, input_std, vx0, vy0, mass, height0, wind_accel)

        err_x, err_y, err_h = abs(real_x - pred_x), abs(real_y - pred_y), abs(real_h - pred_h)
        x_errors.append(err_x)
        y_errors.append(err_y)
        h_errors.append(err_h)

        t2l_errors.append(abs(real_t2l - pred_t2l))
        bounce_errors.append(abs(real_bounce - pred_bounce))
        apex_errors.append(abs(real_apex - pred_apex))
        vx_errors.append(abs(real_vx - pred_vx))

        # Actual landing-position error vs. the model's own predicted
        # uncertainty for that same throw - there's no ground-truth
        # "uncertainty" column to MAE against (uncertainty isn't a
        # physical quantity), so this pair is what gets checked for
        # calibration below instead.
        landing_errors.append(math.sqrt(err_x ** 2 + err_y ** 2))
        predicted_stds.append(math.exp(0.5 * pred_log_var))

        print(f"{vx0:7.1f} {vy0:7.1f} {mass:6.1f} {height0:6.1f} {wind_accel:6.1f} | {real_x:9.3f} {pred_x:9.3f} {err_x:7.3f} | "
              f"{real_y:8.3f} {pred_y:8.3f} {err_y:7.3f} | {real_h:8.3f} {pred_h:8.3f} {err_h:7.3f}")

    def mae(errors):
        return sum(errors) / len(errors)

    def pearson_corr(xs, ys):
        n = len(xs)
        mean_x, mean_y = sum(xs) / n, sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        if var_x == 0 or var_y == 0:
            return 0.0
        return cov / math.sqrt(var_x * var_y)

    print(f"\nMean absolute error: x = {mae(x_errors):.3f}, "
          f"y = {mae(y_errors):.3f}, max_height = {mae(h_errors):.3f}")
    print(f"Mean absolute error: timeToLand = {mae(t2l_errors):.3f}, "
          f"bounceCount = {mae(bounce_errors):.3f}, apexTime = {mae(apex_errors):.3f}, "
          f"finalVx = {mae(vx_errors):.3f}")

    # Calibration check for the uncertainty head: does predicted
    # uncertainty actually track actual landing error? Correlation near
    # +1 = well calibrated (confident when it should be, unsure when it
    # should be). Near 0 = the uncertainty head learned nothing useful.
    corr = pearson_corr(landing_errors, predicted_stds)
    print(f"\nUncertainty calibration: mean actual landing error = {mae(landing_errors):.3f}, "
          f"mean predicted std = {mae(predicted_stds):.3f}, correlation = {corr:.3f}")
