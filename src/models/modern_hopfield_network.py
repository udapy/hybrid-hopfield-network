import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernHopfieldLayer(nn.Module):
    def __init__(self, input_size, hidden_size, beta=1.0):
        super(ModernHopfieldLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.beta = beta

        self.W_Q = nn.Linear(input_size, hidden_size, bias=False)
        self.W_K = nn.Linear(input_size, hidden_size, bias=False)
        self.W_V = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size**0.5)
        attention = F.softmax(energy * self.beta, dim=-1)

        output = torch.matmul(attention, V)
        return output


class ModernHopfieldNetwork(nn.Module):
    def __init__(self, config):
        super(ModernHopfieldNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.hopfield = ModernHopfieldLayer(
            config["model"]["input_size"],
            config["model"]["hidden_size"],
            config["model"]["beta"],
        )
        self.fc = nn.Linear(
            config["model"]["hidden_size"], config["model"]["num_classes"]
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.hopfield(x)
        x = self.fc(x)
        return x
