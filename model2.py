import torch


def resize(x, size):
        return torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=2):
        super(Encoder, self).__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.skip_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.pool = torch.nn.MaxPool2d(scale)
        # self.bn = torch.nn.BatchNorm2d(out_channels)
        # self.relu = torch.nn.ReLU()
    def forward(self, x):
        skip = self.skip_conv(x)

        x = self.conv(x)
        x = self.pool(x)
        # x = self.bn(x)
        # x = self.relu(x)
        return x, skip
class Decoder(torch.nn.Module):
    def __init__(self, skip_channels, in_channels, out_channels, kernel_size=3, scale=2):
        super(Decoder, self).__init__()
        in_channels += skip_channels
        padding = kernel_size // 2
        self.upsample = torch.nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
        # self.bn = torch.nn.BatchNorm2d(out_channels)
        # self.relu = torch.nn.ReLU()
    def forward(self, x, skip):
        x = self.upsample(x)

        skip = resize(skip, x.shape[-2:])
        x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        # x = self.bn(x)
        # x = self.relu(x)
        return x
class Regular(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Regular, self).__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

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
    

class Model2(torch.nn.Module):
    def __init__(self):
        super(Model2, self).__init__()

        # in-channels must match for corresponding skips
        self.layers = torch.nn.ModuleList([
                Regular(3, 8),
                Regular(8, 16),
            Encoder(16, 16, scale=4), # 1/4 original size
                Regular(16, 32),
                Regular(32, 64),
            Encoder(64, 64, scale=4), # 1/16
                Regular(64, 128),
                Regular(128, 64),
            Decoder(64, 64, 64, scale=2), # 1/8
                Regular(64, 32),
                Regular(32, 16),
            Decoder(16, 16, 16, scale=2), # 1/4
                Regular(16, 8),
                Regular(8, 4),
                Regular(4, 1),
                Regular(1, 1, kernel_size=1),
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