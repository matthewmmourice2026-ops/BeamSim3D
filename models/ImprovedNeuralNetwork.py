import torch
import torch.nn as nn
import torch.optim as optim

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self):
        super(ImprovedNeuralNetwork, self).__init__()
        
        # Input layer to hidden layer (5 raw inputs: vx0, vy0, mass,
        # height0, windAccel + 1 engineered feature, see features.py)
        self.fc1 = nn.Linear(6, 250)
        self.bn1 = nn.BatchNorm1d(250)  # Batch Normalization
        self.dropout1 = nn.Dropout(0.0000000001)  # Dropout to reduce overfitting
        
        # Hidden layer 1 to hidden layer 2
        self.fc2 = nn.Linear(250, 250)
        self.bn2 = nn.BatchNorm1d(250)
        self.dropout2 = nn.Dropout(0.0000000001)

        # Hidden layer 2 to hidden layer 3
        self.fc3 = nn.Linear(250, 250)
        self.bn3 = nn.BatchNorm1d(250)  # Batch Normalization
        self.dropout3 = nn.Dropout(0.0000000001)

        # Hidden layer 3 to hidden layer 4
        self.fc4 = nn.Linear(250, 300)
        self.bn4 = nn.BatchNorm1d(300)
        self.dropout4 = nn.Dropout(0.0000000001)

        # Hidden layer 4 to hidden layer 5
        self.fc5 = nn.Linear(300, 450)
        self.bn5 = nn.BatchNorm1d(450)
        self.dropout5 = nn.Dropout(0.0000000001)

        # Hidden layer 5 to hidden layer 6
        self.fc6 = nn.Linear(450, 600)
        self.bn6 = nn.BatchNorm1d(600)
        self.dropout6 = nn.Dropout(0.0000000001)

        # Hidden layer 6 to hidden layer 7
        self.fc7 = nn.Linear(600, 1200)
        self.bn7 = nn.BatchNorm1d(1200)
        self.dropout7 = nn.Dropout(0.0000000001)

        # Hidden layer 7 to hidden layer 8
        self.fc8 = nn.Linear(1200, 2000)
        self.bn8 = nn.BatchNorm1d(2000)
        self.dropout8 = nn.Dropout(0.0000000001)

        # Hidden layer 8 to hidden layer 9
        self.fc9 = nn.Linear(2000, 2000)
        self.bn9 = nn.BatchNorm1d(2000)
        self.dropout9 = nn.Dropout(0.0000000001)

        # Hidden layer 9 to output layer
        # 8 outputs: final_x, final_y, maxHeight, timeToLand, bounceCount,
        # apexTime, finalVx, and a log-variance uncertainty head (index 7)
        # predicting the model's own landing-position error, trained via a
        # separate NLL loss in main.py/kfold_eval.py - not a regression
        # target with a ground-truth column like the other 7.
        self.fc10 = nn.Linear(2000, 8)

    def forward(self, x):
        # Apply ReLU activation and batch normalization to the first layer
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        # Apply ReLU activation and batch normalization to the second layer
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        # Apply ReLU activation and batch normalization to the third layer
        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        # Apply ReLU activation and batch normalization to the fourth layer
        x = self.fc4(x)
        x = nn.functional.relu(x)
        x = self.bn4(x)
        x = self.dropout4(x)

        # Apply ReLU activation and batch normalization to the fifth layer
        x = self.fc5(x)
        x = nn.functional.relu(x)
        x = self.bn5(x)
        x = self.dropout5(x)

        # Apply ReLU activation and batch normalization to the sixth layer
        x = self.fc6(x)
        x = nn.functional.relu(x)
        x = self.bn6(x)
        x = self.dropout6(x)

        # Apply ReLU activation and batch normalization to the seventh layer
        x = self.fc7(x)
        x = nn.functional.relu(x)
        x = self.bn7(x)
        x = self.dropout7(x)
        
        # Apply ReLU activation and batch normalization to the eighth layer
        x = self.fc8(x)
        x = nn.functional.relu(x)
        x = self.bn8(x)
        x = self.dropout8(x)

        # Apply ReLU activation and batch normalization to the ninth layer
        x = self.fc9(x)
        x = nn.functional.relu(x)
        x = self.bn9(x)
        x = self.dropout9(x)

        # Output layer
        x = self.fc10(x)
        return x

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from features import add_engineered_features

    net = ImprovedNeuralNetwork()
    print(net)

    # Dummy input to see the output
    net.eval()
    input_tensor = add_engineered_features(torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0]]))
    with torch.no_grad():
        output = net(input_tensor)
    print(output)