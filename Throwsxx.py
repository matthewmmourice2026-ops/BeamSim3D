import sys
import os
import torch
import random

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ImprovedNeuralNetwork import ImprovedNeuralNetwork

# 1. Load the trained model + input normalization stats
try:
    checkpoint = torch.load('models/ImprovedNeuralNetwork.pth')
except FileNotFoundError:
    print("Checkpoint file not found. Please ensure the model is trained and saved correctly.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the checkpoint: {e}")
    sys.exit(1)

model = ImprovedNeuralNetwork()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # Set the model to evaluation mode

# The mean and standard deviation used during training
input_mean = checkpoint["input_mean"]
input_std = checkpoint["input_std"]

# 2. Create a brand-new, unseen physics throw
# Generate random values for velocity and mass
velocity_x = random.uniform(10.0, 35.0)  # Example range
velocity_y = random.uniform(1.0, 10.0)   # Example range
mass = random.uniform(0.5, 20.0)        # Example range

new_throw = [velocity_x, velocity_y, mass]

# Normalize the input data
x_test = torch.tensor([new_throw], dtype=torch.float32)
x_test = (x_test - input_mean) / input_std

# 3. Make the Prediction
with torch.no_grad():
    y_hat_prediction = model(x_test)

# 4. Print the results
print("\n--- INFERENCE TEST ---")
print(f"Starting Parameters (Vel X, Vel Y, Mass): {new_throw}")
print(f"AI Predicted Landing Coordinates (X, Y): {y_hat_prediction.numpy()[0]}")