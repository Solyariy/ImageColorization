import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def disc_block(in_f, out_f, stride=2, normalize=True):
            layers = [
                nn.Conv2d(
                    in_f, out_f, kernel_size=4, stride=stride, padding=1, bias=False
                )
            ]

            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *disc_block(in_channels, 64, normalize=False),
            *disc_block(64, 128),
            *disc_block(128, 256),
            *disc_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, L, ab):
        x = torch.cat([L, ab], dim=1)
        return self.model(x)
