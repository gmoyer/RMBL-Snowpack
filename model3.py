import torch

from model2 import Decoder, Regular, Sigmoid
from preprocess import FEATURE_IMAGE_SIZE

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=2):
        super(Encoder, self).__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=scale, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class Linear(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Combine(torch.nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU()
    def forward(self, x, m):
        m1, m2 = m.chunk(2, dim=1)  # split m into two tensors along the channel dimension
        m1 = m1.view(m1.shape[0], m1.shape[1], 1, 1).expand_as(x)
        m2 = m2.view(m2.shape[0], m2.shape[1], 1, 1).expand_as(x)
        x = x * m1 + m2
        x = self.leaky_relu(x)
        return x


class Model3(torch.nn.Module):
    def __init__(self):
        super(Model3, self).__init__()

        self.encoder = torch.nn.ModuleList([
            Encoder(3, 4, scale=4),  # 1/4 original size
            Encoder(4, 8, scale=4),  # 1/16
            Encoder(8, 16, scale=4),  # 1/64
            torch.nn.Flatten(),
            Linear(16 * (FEATURE_IMAGE_SIZE // 64) * (FEATURE_IMAGE_SIZE // 64), 64),
            Linear(64, 32)
        ])

        self.layers = torch.nn.ModuleList([
            Regular(3, 8),
            Regular(8, 16),
            Regular(16, 32),
            Regular(32, 16),
            Combine(),
            Regular(16, 8),
            Encoder(8, 4), #1/2 original size
            Encoder(4, 1), #1/4 original size
            Regular(1, 1, kernel_size=1),
            Sigmoid()
        ])

    def forward(self, x):
        m = torch.clone(x)
        for layer in self.encoder:
            m = layer(m)
        for layer in self.layers:
            if isinstance(layer, Combine):
                x = layer(x, m)
            else:
                x = layer(x)
        return x