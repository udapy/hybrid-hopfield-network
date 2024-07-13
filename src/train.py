import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .data.data_loader import MNISTDataLoader
from .utils.metrics import calculate_accuracy


def train(model, train_loader, test_loader, config, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0
        train_acc = 0
        for batch_idx, (data, target) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        ):
            data, target = data.to(device), target.to(device)
            noisy_data = MNISTDataLoader.add_noise(data, config["data"]["noise_factor"])

            optimizer.zero_grad()
            output = model(noisy_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calculate_accuracy(output, target)

            if (batch_idx + 1) % config["training"]["log_interval"] == 0:
                print(
                    f"Train Epoch: {epoch+1} [{batch_idx+1}/{len(train_loader)}]\t"
                    f"Loss: {train_loss/(batch_idx+1):.4f}\t"
                    f"Accuracy: {train_acc/(batch_idx+1):.4f}"
                )

        # Evaluation
        model.eval()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                noisy_data = MNISTDataLoader.add_noise(
                    data, config["data"]["noise_factor"]
                )
                output = model(noisy_data)
                test_loss += criterion(output, target).item()
                test_acc += calculate_accuracy(output, target)

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    return model
