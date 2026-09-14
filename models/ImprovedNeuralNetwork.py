import torch
import torch.nn as nn
import torch.optim as optim

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self):
        super(ImprovedNeuralNetwork, self).__init__()
        
        # Input layer to hidden layer
        self.fc1 = nn.Linear(3, 92)
        self.bn1 = nn.BatchNorm1d(92)  # Batch Normalization
        self.dropout1 = nn.Dropout(0.0001)  # Dropout to reduce overfitting
        
        # Hidden layer 1 to hidden layer 2
        self.fc2 = nn.Linear(92, 140)
        self.bn2 = nn.BatchNorm1d(140)
        self.dropout2 = nn.Dropout(0.00001)

        # Hidden layer 2 to hidden layer 3
        self.fc3 = nn.Linear(140, 200)
        self.bn3 = nn.BatchNorm1d(200)  # Batch Normalization
        self.dropout3 = nn.Dropout(0.00001)

        # Hidden layer 3 to hidden layer 4
        self.fc4 = nn.Linear(200, 290)
        self.bn4 = nn.BatchNorm1d(290)
        self.dropout4 = nn.Dropout(0.00001)

        # Hidden layer 4 to hidden layer 5
        self.fc5 = nn.Linear(290, 450)
        self.bn5 = nn.BatchNorm1d(450)
        self.dropout5 = nn.Dropout(0.00001)

        # Hidden layer 5 to hidden layer 6
        self.fc6 = nn.Linear(450, 600)
        self.bn6 = nn.BatchNorm1d(600)
        self.dropout6 = nn.Dropout(0.00001)

        # Hidden layer 6 to output layer
        self.fc7 = nn.Linear(600, 3)

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
        
        # Output layer
        x = self.fc7(x)
        return x

if __name__ == "__main__":
    net = ImprovedNeuralNetwork()
    print(net)

    # Dummy input to see the output
    net.eval()
    input_tensor = torch.tensor([[1.0, 2.0, 3.0]])
    with torch.no_grad():
        output = net(input_tensor)
    print(output)