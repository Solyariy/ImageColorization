import torch.nn as nn

from src.models.encoder import UNetEncoder
from src.models.decoder import UNetDecoder


class ColorizationModel(nn.Module):
    def __init__(self, encoder = UNetEncoder, decoder = UNetDecoder):
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, L_input):
        features = self.encoder(L_input)
        ab_output = self.decoder(features)
        return ab_output
