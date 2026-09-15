import math
import sys
import os
import torch
import random

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features import add_engineered_features
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
# Generate random values for velocity, mass, launch height, and wind
velocity_x = random.uniform(10.0, 200.0)
velocity_y = random.uniform(1.0, 100.0)
mass = random.uniform(0.1, 90.0)
height0 = random.uniform(0.5, 20.0)
wind_accel = random.uniform(-3.0, 3.0)

new_throw = [velocity_x, velocity_y, mass, height0, wind_accel]

# Normalize the input data
x_test = torch.tensor([new_throw], dtype=torch.float32)
x_test = add_engineered_features(x_test)
x_test = (x_test - input_mean) / input_std

# 3. Make the Prediction
with torch.no_grad():
    predictions = model(x_test)

# Model returns [landing_x, landing_y, max_height, time_to_land,
# bounce_count, apex_time, final_vx], matching main.py's target column
# order (final_x, final_y, maxHeight, timeToLand, bounceCount, apexTime,
# finalVx)
landing_x = predictions[0, 0].item()
landing_y = predictions[0, 1].item()
max_height = predictions[0, 2].item()
time_to_land = predictions[0, 3].item()
bounce_count = predictions[0, 4].item()
apex_time = predictions[0, 5].item()
final_vx = predictions[0, 6].item()
log_var = predictions[0, 7].item()
uncertainty_std = math.exp(0.5 * log_var)

# 4. Print the results
print("\n--- INFERENCE TEST ---")
print(f"Starting Parameters (Vel X, Vel Y, Mass): {new_throw}")
print(f"AI Predicted Landing Coordinates (X, Y): ({landing_x}, {landing_y})")
print(f"AI Predicted Maximum Height: {max_height}")
print(f"AI Predicted Time To Land: {time_to_land}")
print(f"AI Predicted Bounce Count: {bounce_count}")
print(f"AI Predicted Apex Time: {apex_time}")
print(f"AI Predicted Final Horizontal Velocity: {final_vx}")
print(f"AI Predicted Uncertainty: +/-{uncertainty_std:.3f} units (68% confidence)")
