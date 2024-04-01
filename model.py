import torch

def create_model():
    return torch.nn.Sequential(
        torch.nn.Conv2d(4, 8, 3, padding=1),
        torch.nn.BatchNorm2d(8),
        torch.nn.ReLU(),
        torch.nn.Upsample(scale_factor=2),
        torch.nn.Conv2d(8, 16, 5, padding=2),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.Upsample(scale_factor=2),
        torch.nn.Conv2d(16, 16, 7, padding=3),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 16, 7, padding=3),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 8, 7, padding=3),
        torch.nn.BatchNorm2d(8),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(8, 4, 5, padding=2),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(4, 1, 3, padding=1),
        torch.nn.Sigmoid()
    )