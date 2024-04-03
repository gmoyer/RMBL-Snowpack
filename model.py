import torch


def resize(x, size):
        return torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super(Encoder, self).__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.skip_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        skip = self.skip_conv(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x, skip
class Decoder(torch.nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=3, stride=2):
        super(Decoder, self).__init__()
        in_channels += skip_channels
        padding = kernel_size // 2
        self.upsample = torch.nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
    def forward(self, x, skip):
        x = self.upsample(x)

        skip = resize(skip, x.shape[-2:])
        x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class Regular(torch.nn.Module):
    def __init__(self, in_channels, out_channels, skip_num=None, kernel_size=3):
        super(Regular, self).__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

        self.skip_num = skip_num
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Sigmoid(torch.nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(x)
    

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # in-channels must match for corresponding skips
        self.layers = torch.nn.ModuleList([
            Encoder(3, 8),
            Encoder(8, 16),
            Encoder(16, 32),
            Encoder(32, 64),
            Decoder(64, 64, 32),
            Decoder(32, 32, 16),
            Decoder(16, 16, 8, stride=1),
            Decoder(8, 8, 4, stride=1),
            Regular(4, 1),
            Sigmoid()
        ])

    
    def forward(self, x):
        skips = []
        for layer in self.layers:
            if isinstance(layer, Encoder):
                x, skip = layer(x)
                skips.append(skip)
            elif isinstance(layer, Decoder):
                skip = skips.pop()
                x = layer(x, skip)
            else:
                x = layer(x)
        return x


def create_model():
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.BatchNorm2d(8),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 16, 5, padding=2),
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