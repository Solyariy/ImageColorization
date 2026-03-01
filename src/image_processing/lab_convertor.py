import kornia
import torch
from torch import nn


class LabConvertor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_batch):
        lab_batch = kornia.color.rgb_to_lab(img_batch)

        L = lab_batch[:, [0], :, :] / 50 - 1
        ab = lab_batch[:, 1:, :, :] / 128

        return {"L": L, "ab": ab}

    @staticmethod
    def unscale_to_rgb(L_input, ab_input):
        lab = torch.cat([L_input, ab_input], dim=1)

        L_unscaled = (lab[:, [0], :, :] + 1.0) * 50.0
        ab_unscaled = lab[:, 1:, :, :] * 128.0
        lab_unscaled = torch.cat([L_unscaled, ab_unscaled], dim=1)

        rgb_output = kornia.color.lab_to_rgb(lab_unscaled)
        rgb_output = torch.clamp(rgb_output, 0.0, 1.0)
        return rgb_output
