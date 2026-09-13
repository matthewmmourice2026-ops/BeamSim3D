import torch
import torch.nn as nn

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self):
        super(ImprovedNeuralNetwork, self).__init__()
        
        # Input layer to hidden layer
        self.fc1 = nn.Linear(3, 16)
        self.bn1 = nn.BatchNorm1d(16)  # Batch Normalization
        self.dropout1 = nn.Dropout(0.2)  # Dropout to reduce overfitting
        
        # Hidden layer to hidden layer
        self.fc2 = nn.Linear(16, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        
        # Hidden layer to output layer
        self.fc3 = nn.Linear(32, 2)

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
        
        # Output layer
        x = self.fc3(x)
        
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