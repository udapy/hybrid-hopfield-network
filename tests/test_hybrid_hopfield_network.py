import pytest
import torch
import yaml

from src.models.hybrid_hopfield_network import HybridHopfieldNetwork


@pytest.fixture
def config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def model(config):
    return HybridHopfieldNetwork(config)


def test_forward(model):
    x = torch.randn(32, 1, 28, 28)
    output = model(x)
    assert output.shape == (32, 10)  # Assuming 10 classes for MNIST
