import pytest
import torch

from src.models.hybrid_hopfield_layer import HybridHopfieldLayer


@pytest.fixture
def hybrid_layer():
    input_size = 784
    hidden_size = 256
    beta = 1.0
    sparsity_threshold = 0.5
    return HybridHopfieldLayer(input_size, hidden_size, beta, sparsity_threshold)


def test_forward(hybrid_layer):
    x = torch.randn(32, 784)
    output = hybrid_layer(x)
    assert output.shape == (32, 256)


def test_sparsemax(hybrid_layer):
    x = torch.randn(32, 10)
    output = hybrid_layer.sparsemax(x)
    assert output.shape == x.shape
    assert torch.all(output >= 0) and torch.all(output <= 1)
    assert torch.allclose(output.sum(dim=1), torch.ones(32))
