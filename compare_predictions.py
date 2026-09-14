import subprocess

import torch

from models.ImprovedNeuralNetwork import ImprovedNeuralNetwork

THROW_SIM = "build/throw_sim"


def ground_truth(vx0, vy0, mass):
    result = subprocess.run(
        [THROW_SIM, str(vx0), str(vy0), str(mass)],
        capture_output=True, text=True, check=True,
    )
    final_x, final_y = result.stdout.strip().split(",")
    return float(final_x), float(final_y)


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
    return y[0, 0].item(), y[0, 1].item()


if __name__ == "__main__":
    # A handful of throws the model has never seen (not from throw_results.csv)
    test_throws = [
        (15.0, 25.0, 2.5),
        (-10.0, 15.0, 1.0),
        (8.0, 20.0, 4.5),
        (-14.0, 8.0, 0.7),
        (3.0, 22.0, 3.3),
    ]

    model, input_mean, input_std = load_model()

    print(f"{'vx0':>7} {'vy0':>7} {'mass':>6} | {'real_x':>9} {'pred_x':>9} {'err_x':>7} | {'real_y':>8} {'pred_y':>8} {'err_y':>7}")

    x_errors, y_errors = [], []
    for vx0, vy0, mass in test_throws:
        real_x, real_y = ground_truth(vx0, vy0, mass)
        pred_x, pred_y = predict(model, input_mean, input_std, vx0, vy0, mass)
        err_x, err_y = abs(real_x - pred_x), abs(real_y - pred_y)
        x_errors.append(err_x)
        y_errors.append(err_y)

        print(f"{vx0:7.1f} {vy0:7.1f} {mass:6.1f} | {real_x:9.3f} {pred_x:9.3f} {err_x:7.3f} | "
              f"{real_y:8.3f} {pred_y:8.3f} {err_y:7.3f}")

    print(f"\nMean absolute error: x = {sum(x_errors) / len(x_errors):.3f}, y = {sum(y_errors) / len(y_errors):.3f}")
