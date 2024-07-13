import torch
import yaml

from src.data.data_loader import MNISTDataLoader
from src.models.hybrid_hopfield_network import HybridHopfieldNetwork
from src.train import train
from src.utils.visualization import visualize_results


def main():
    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Load data
    data_loader = MNISTDataLoader(config)
    train_loader, test_loader = data_loader.get_data_loaders()

    # Initialize model
    model = HybridHopfieldNetwork(config).to(device)

    # Train model
    trained_model = train(model, train_loader, test_loader, config, device)

    # Visualize results
    visualize_results(trained_model, test_loader, device)


if __name__ == "__main__":
    main()
