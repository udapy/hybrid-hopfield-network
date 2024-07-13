import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTDataLoader:
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def get_data_loaders(self):
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=self.transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=self.transform
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.config["data"]["batch_size"], shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["data"]["test_batch_size"],
            shuffle=False,
        )

        return train_loader, test_loader

    @staticmethod
    def add_noise(images, noise_factor):
        noisy_images = images + noise_factor * torch.randn(*images.shape)
        return torch.clamp(noisy_images, 0.0, 1.0)
