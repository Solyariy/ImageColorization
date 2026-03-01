import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, skip_connection=None):
        x = self.upsample(x)
        if skip_connection is not None:
            diffY = skip_connection.size()[2] - x.size()[2]
            diffX = skip_connection.size()[3] - x.size()[3]
            x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)


class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        original_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            new_conv1.weight.copy_(original_conv1.weight.sum(dim=1, keepdim=True))

        self.encoder0 = nn.Sequential(
            new_conv1,
            resnet.bn1,
            resnet.relu
        )
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        return [e0, e1, e2, e3, e4]


class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = UpBlock(512, 256, skip_channels=256)
        self.up2 = UpBlock(256, 128, skip_channels=128)
        self.up3 = UpBlock(128, 64, skip_channels=64)
        self.up4 = UpBlock(64, 64, skip_channels=64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, features):
        e0, e1, e2, e3, e4 = features

        d1 = self.up1(e4, skip_connection=e3)
        d2 = self.up2(d1, skip_connection=e2)
        d3 = self.up3(d2, skip_connection=e1)
        d4 = self.up4(d3, skip_connection=e0)

        return self.final(d4)


class ColorizationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.decoder = UNetDecoder()

    def forward(self, L_input):
        features = self.encoder(L_input)
        ab_output = self.decoder(features)
        return ab_output
