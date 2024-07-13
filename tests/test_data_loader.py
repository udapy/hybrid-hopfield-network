import pytest
import torch
import yaml

from src.data.data_loader import MNISTDataLoader


@pytest.fixture
def config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def data_loader(config):
    return MNISTDataLoader(config)


def test_get_data_loaders(data_loader):
    train_loader, test_loader = data_loader.get_data_loaders()
    assert len(train_loader.dataset) == 60000
    assert len(test_loader.dataset) == 10000


def test_add_noise():
    dummy_image = torch.zeros(1, 28, 28)
    noisy_image = MNISTDataLoader.add_noise(dummy_image, 0.5)
    assert not torch.all(noisy_image == 0)
    assert torch.all(noisy_image >= 0) and torch.all(noisy_image <= 1)
