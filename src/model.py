import torch
import torch.nn as nn
from torchvision import models


class UpBlock(nn.Module):
    """
    A helper block for the Decoder.
    Responsibility: Upsample -> Concatenate -> Convolve.
    """

    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
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
    """
    The 'Eye' of the model.
    Responsibility: Extract features at 5 different scales (e0...e4).
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)

        # Modify input to accept 1 channel (L) instead of 3 (RGB)
        self.encoder0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu
        )
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

    def forward(self, x):
        e0 = self.encoder0(x)  # 64x128x128
        e1 = self.encoder1(e0)  # 64x64x64
        e2 = self.encoder2(e1)  # 128x32x32
        e3 = self.encoder3(e2)  # 256x16x16
        e4 = self.encoder4(e3)  # 512x8x8

        # Return all features for skip connections
        return [e0, e1, e2, e3, e4]


class UNetDecoder(nn.Module):
    """
    The 'Hand' of the model.
    Responsibility: Take features and paint the colors.
    """

    def __init__(self):
        super().__init__()
        # 512 + 256 -> 256
        self.up1 = UpBlock(512, 256, skip_channels=256)
        # 256 + 128 -> 128
        self.up2 = UpBlock(256, 128, skip_channels=128)
        # 128 + 64 -> 64
        self.up3 = UpBlock(128, 64, skip_channels=64)
        # 64 + 64 -> 64
        self.up4 = UpBlock(64, 64, skip_channels=64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, features):
        # Unpack the list of features from the Encoder
        e0, e1, e2, e3, e4 = features

        d1 = self.up1(e4, skip_connection=e3)
        d2 = self.up2(d1, skip_connection=e2)
        d3 = self.up3(d2, skip_connection=e1)
        d4 = self.up4(d3, skip_connection=e0)

        return self.final(d4)


class ColorizationModel(nn.Module):
    """
    The Main Controller.
    Responsibility: Connect Encoder to Decoder.
    """

    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.decoder = UNetDecoder()

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output


# --- VERIFICATION ---
if __name__ == "__main__":
    model = ColorizationModel()
    x = torch.randn(4, 1, 256, 256)
    out = model(x)
    print(f"Output Shape: {out.shape}")
    assert out.shape == (4, 2, 256, 256)
    print("✅ Optimized Modular Architecture Verified.")