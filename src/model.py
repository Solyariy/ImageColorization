import torch
import torch.nn as nn
from torchvision import models




class ColorizationModel(nn.Module):
    """
    The Main Controller.
    Responsibility: Connect Encoder to Decoder.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output
