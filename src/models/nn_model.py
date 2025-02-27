# src/models/nn_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from src.models.abstract_model import AbstractModel

class NeuralNetwork(nn.Module):
    """
    A simple fully connected neural network with one hidden layer.
    You can extend this architecture (e.g., more layers) as needed.
    """
    def __init__(self, input_dim, hidden_dim=16):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Forward pass of the neural network.
        
        :param x: Input tensor of shape (batch_size, input_dim).
        :return: Output tensor of shape (batch_size, 1).
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class NNModel(AbstractModel):
    """
    A neural network-based model that predicts the next token price.
    By default, it sums fees and sums rewards to produce a single input feature
    for each. If you want more detailed control, you can replace the sums with
    a vector of fee/reward parameters.
    """

    def __init__(self):
        """
        Construct the neural network model, define the loss function (MSE),
        and the optimizer (Adam).
        
        For simplicity, we fix input_dim to 6:
            [ current_price, sum_of_fees, sum_of_rewards, volume, liquidity, circulation ]
        If you'd like to pass each fee/reward individually, you can dynamically
        set input_dim at runtime or rewrite the logic below.
        """
        super().__init__()
        self.input_dim = 6  # (price, sum_of_fees, sum_of_rewards, volume, liquidity, circulation)
        self.model = NeuralNetwork(self.input_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def predict(self, P, params, volume, liquidity, circulation):
        """
        Predict the next price using the neural network.
        
        :param P: Current token price (float).
        :param params: Dictionary with "fees" and "rewards" sub-dicts.
        :param volume: Market volume (float).
        :param liquidity: Market liquidity (float).
        :param circulation: Token circulation (float).
        :return: Predicted next price (float).
        """
        # Summation of all fees and rewards. 
        sum_of_fees = sum(params["fees"].values())
        sum_of_rewards = sum(params["rewards"].values())

        # Construct input vector
        input_vec = [
            P,
            sum_of_fees,
            sum_of_rewards,
            volume,
            liquidity,
            circulation
        ]
        input_tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0)  # shape: (1, input_dim)

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.item()

    def update(self, data):
        """
        Train the neural network with provided data.
        
        :param data: A tuple (inputs, targets) where:
            - inputs is a tensor of shape (batch_size, input_dim)
            - targets is a tensor of shape (batch_size, 1)
        :return: The loss value (float) after the update step.
        """
        inputs, targets = data
        self.model.train()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
