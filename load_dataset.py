import pandas as pd
import torch

df = pd.read_csv("build/throw_results.csv")

inputs = torch.tensor(df[["vx0", "vy0", "mass"]].values, dtype=torch.float32)
targets = torch.tensor(df[["final_x", "final_y"]].values, dtype=torch.float32)

print(f"inputs:  {tuple(inputs.shape)}, dtype={inputs.dtype}")
print(f"targets: {tuple(targets.shape)}, dtype={targets.dtype}")
print(inputs[:5])
print(targets[:5])
