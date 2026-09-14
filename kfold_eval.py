import copy

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models.ImprovedNeuralNetwork import ImprovedNeuralNetwork

# 5-fold cross validation. This is for reporting a more honest accuracy
# number, not for producing the weights main.py saves for actual use.

df = pd.read_csv("build/throw_results.csv")
inputs = torch.tensor(df[["vx0", "vy0", "mass"]].values, dtype=torch.float32)
targets = torch.tensor(df[["final_x", "final_y"]].values, dtype=torch.float32)

k = 5
epochs = 1000
patience = 50

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

    model = ImprovedNeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_inputs)
        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_inputs), val_targets)

        if val_loss.item() < best_val_loss:
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
