import os
import glob
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.setup.config import main_config
import random

from src.setup.enums import RunTypeEnum
from src.image_processing.lab_convertor import LabConvertor


class ColorizationDataset(Dataset):
    def __init__(self, root_dir, split: RunTypeEnum = RunTypeEnum.TRAIN):
        self.files = glob.glob(os.path.join(root_dir, "*.jpg"))
        random.shuffle(self.files)

        split_idx = int(len(self.files) * 0.8)
        if split == RunTypeEnum.TRAIN:
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




def make_dataloaders(
        root_dir: Path,
        split: RunTypeEnum,
        batch_size=16,
        n_workers=4,
        pin_memory=True
):
    dataset = ColorizationDataset(root_dir=root_dir, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=pin_memory
    )
    return dataloader


if __name__ == "__main__":
    dl = make_dataloaders(root_dir=main_config.LANDSCAPE_IMAGES, split=RunTypeEnum.TRAIN)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = LabConvertor().to(device)

    rgb_batch = next(iter(dl))
    print(f"1. CPU Output Shape: {rgb_batch.shape} (RGB)")

    rgb_batch = rgb_batch.to(device)

    data = processor(rgb_batch)
    L = data["L"]
    ab = data["ab"]

    print(f"2. GPU L Shape: {L.shape}")
    print(f"3. GPU ab Shape: {ab.shape}")
    print(f"4. Max L Value: {L.max().item():.2f}")
