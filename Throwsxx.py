# Throwsxx.py
import sys
import os
import torch

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ImprovedNeuralNetwork import ImprovedNeuralNetwork

# 1. Load the trained model (assuming it's saved as 'ImprovedNeuralNetwork.pth' in the current directory)
model_path = 'models/ImprovedNeuralNetwork.py'  # Adjust the path based on your directory structure
model = ImprovedNeuralNetwork()
model.load_state_dict(torch.load(model_path[:-3] + '.pth'))  # Load the weights from 'ImprovedNeuralNetwork.pth'
model.eval()  # Set the model to evaluation mode

# 2. Create a brand-new, unseen physics throw
# Example Inputs: [Velocity X, Velocity Y, Mass]
# Change these numbers to whatever you want!
new_throw = [15.0, 25.0, 2.5] 

# 3. Convert the raw numbers into a PyTorch Tensor
x_test = torch.tensor([new_throw], dtype=torch.float32)

# 4. Make the Prediction
with torch.no_grad():
    y_hat_prediction = model(x_test)

# 5. Print the results
print("\n--- INFERENCE TEST ---")
print(f"Starting Parameters (Vel X, Vel Y, Mass): {new_throw}")
print(f"AI Predicted Landing Coordinates (X, Y): {y_hat_prediction.numpy()[0]}")