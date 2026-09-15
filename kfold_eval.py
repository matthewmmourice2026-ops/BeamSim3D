import copy

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from features import add_engineered_features
from models.ImprovedNeuralNetwork import ImprovedNeuralNetwork

# 5-fold cross validation. This is for reporting a more honest accuracy
# number, not for producing the weights main.py saves for actual use.
# Mirrors main.py's training setup (MPS, engineered feature, mini-batches,
# Huber loss, weight decay, LR scheduler) so the numbers are comparable.

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

df = pd.read_csv("build/throw_results.csv")
inputs = torch.tensor(df[["vx0", "vy0", "mass", "height0", "windAccel"]].values, dtype=torch.float32).to(device)
targets = torch.tensor(df[["final_x", "final_y", "maxHeight", "timeToLand", "bounceCount", "apexTime", "finalVx"]].values, dtype=torch.float32).to(device)
inputs = add_engineered_features(inputs)

k = 5
epochs = 50000
patience = 150
batch_size = 2048


def uncertainty_loss(outputs, targets):
    """Same as main.py's - trains outputs[:, 7] (log-variance) to predict
    the model's own landing-position error, detached from the point
    estimate so it doesn't interfere with the main Huber loss."""
    xy_error_sq = (outputs[:, 0].detach() - targets[:, 0]) ** 2 + (outputs[:, 1].detach() - targets[:, 1]) ** 2
    log_var = outputs[:, 7]
    return (0.5 * torch.exp(-log_var) * xy_error_sq + 0.5 * log_var).mean()

torch.manual_seed(42)
n = inputs.shape[0]
perm = torch.randperm(n)
fold_size = n // k
fold_losses = []

for fold in range(k):
    val_start = fold * fold_size
    val_end = n if fold == k - 1 else val_start + fold_size
    val_idx = perm[val_start:val_end]
    train_idx = torch.cat([perm[:val_start], perm[val_end:]])

    train_inputs, train_targets = inputs[train_idx], targets[train_idx]
    val_inputs, val_targets = inputs[val_idx], targets[val_idx]

    input_mean = train_inputs.mean(dim=0)
    input_std = train_inputs.std(dim=0)
    train_inputs = (train_inputs - input_mean) / input_std
    val_inputs = (val_inputs - input_mean) / input_std

    train_loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=batch_size, shuffle=True)

    model = ImprovedNeuralNetwork().to(device)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.75, patience=55)

    # Matches main.py: early stopping/scheduler watch val_loss + val_unc_loss
    # combined, not val_loss alone, so the uncertainty head gets a real
    # chance to finish calibrating instead of stopping the moment the
    # point-estimate loss plateaus.
    best_combined_loss = float("inf")
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            main_loss = criterion(outputs[:, :7], batch_targets)
            loss = main_loss + uncertainty_loss(outputs, batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs[:, :7], val_targets)
            val_unc_loss = uncertainty_loss(val_outputs, val_targets)

        combined_val_loss = val_loss.item() + val_unc_loss.item()
        scheduler.step(combined_val_loss)

        if combined_val_loss < best_combined_loss:
            best_combined_loss = combined_val_loss
            best_val_loss = val_loss.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    print(f"Fold {fold + 1}/{k}: best val loss = {best_val_loss:.4f} (stopped at epoch {epoch + 1})")
    fold_losses.append(best_val_loss)

fold_losses = torch.tensor(fold_losses)
print(f"\nMean val loss across {k} folds: {fold_losses.mean().item():.4f}")
print(f"Std dev: {fold_losses.std().item():.4f}")
