import kornia
from torch import nn


class LabConvertor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_batch):
        lab_batch = kornia.color.rgb_to_lab(img_batch)

        L = lab_batch[:, [0], :, :] / 50 - 1
        ab = lab_batch[:, 1:, :, :] / 128

        return {"L": L, "ab": ab}
