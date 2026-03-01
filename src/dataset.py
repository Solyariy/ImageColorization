import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import kornia
from src.config import main_config
import random

from src.enums import RunTypeEnum


class ColorizationDataset(Dataset):
    def __init__(self, root_dir, split: RunTypeEnum = RunTypeEnum.TRAIN):
        self.files = glob.glob(os.path.join(root_dir, "*.jpg"))
        random.shuffle(self.files)

        split_idx = int(len(self.files) * 0.8)
        if split == "train":
            self.files = self.files[:split_idx]
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(
                        (main_config.IMG_SIZE, main_config.IMG_SIZE),
                        transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.files = self.files[split_idx:]
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(
                        (main_config.IMG_SIZE, main_config.IMG_SIZE),
                        transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transforms(img)

        return img


class DataProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_batch):
        lab_batch = kornia.color.rgb_to_lab(img_batch)

        L = lab_batch[:, [0], :, :] / 50 - 1
        ab = lab_batch[:, 1:, :, :] / 128

        return {"L": L, "ab": ab}


if __name__ == "__main__":
    ds = ColorizationDataset(f"{main_config.DATA_PATH}/landscape_images", split="train")
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = DataProcessor().to(device)

    rgb_batch = next(iter(dl))
    print(f"1. CPU Output Shape: {rgb_batch.shape} (RGB)")

    rgb_batch = rgb_batch.to(device)

    data = processor(rgb_batch)
    L = data["L"]
    ab = data["ab"]

    print(f"2. GPU L Shape: {L.shape}")
    print(f"3. GPU ab Shape: {ab.shape}")
    print(f"4. Max L Value: {L.max().item():.2f}")
