import matplotlib.pyplot as plt
import torch


def visualize_results(model, test_loader, device, num_samples=5):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            noisy_data = test_loader.dataset.add_noise(data, 0.5)
            output = model(noisy_data)
            _, predicted = torch.max(output.data, 1)

            fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
            for i in range(num_samples):
                axs[i, 0].imshow(data[i].cpu().squeeze(), cmap="gray")
                axs[i, 0].set_title(f"Original (Label: {target[i].item()})")
                axs[i, 1].imshow(noisy_data[i].cpu().squeeze(), cmap="gray")
                axs[i, 1].set_title("Noisy Input")
                axs[i, 2].imshow(output[i].cpu().reshape(28, 28).detach(), cmap="gray")
                axs[i, 2].set_title(f"Reconstructed (Pred: {predicted[i].item()})")

            plt.tight_layout()
            plt.show()
            break
