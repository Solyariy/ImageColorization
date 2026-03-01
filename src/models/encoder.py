import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


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
