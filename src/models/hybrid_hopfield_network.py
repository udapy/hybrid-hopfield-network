import torch.nn as nn

from .hybrid_hopfield_layer import HybridHopfieldLayer


class HybridHopfieldNetwork(nn.Module):
    def __init__(self, config):
        super(HybridHopfieldNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.hopfield = HybridHopfieldLayer(
            config["model"]["input_size"],
            config["model"]["hidden_size"],
            config["model"]["beta"],
            config["model"]["sparsity_threshold"],
        )
        self.fc = nn.Linear(
            config["model"]["hidden_size"], config["model"]["num_classes"]
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.hopfield(x)
        x = self.fc(x)
        return x
