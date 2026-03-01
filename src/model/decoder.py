import torch
from torch import nn


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


class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = UpBlock(in_channels=512,out_channels=256, skip_channels=256)
        self.up2 = UpBlock(in_channels=256, out_channels=128, skip_channels=128)
        self.up3 = UpBlock(in_channels=128, out_channels=64, skip_channels=64)
        self.up4 = UpBlock(in_channels=64, out_channels=64, skip_channels=64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, features):
        e0, e1, e2, e3, e4 = features

        d1 = self.up1(e4, skip_connection=e3)
        d2 = self.up2(d1, skip_connection=e2)
        d3 = self.up3(d2, skip_connection=e1)
        d4 = self.up4(d3, skip_connection=e0)

        return self.final(d4)
