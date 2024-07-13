import torch


def calculate_accuracy(output, target):
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == target).sum().item()
    total = target.size(0)
    return correct / total
