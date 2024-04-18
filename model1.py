import torch


def Model1():
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(8),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 16, 5, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 7, kernel_size=5, padding=2),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 8, 7, kernel_size=5, padding=2),
        torch.nn.BatchNorm2d(8),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(8, 4, 5, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(4, 1, 3, kernel_size=3, padding=1),
        torch.nn.Sigmoid()
    )