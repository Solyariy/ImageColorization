import os
import glob
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.setup.config import main_config
import numpy as np

from src.setup.enums import RunTypeEnum


class ColorizationDataset(Dataset):
    def __init__(self, root_dir: str = None, split: RunTypeEnum = RunTypeEnum.TRAIN, single_image: str = None):
        if single_image:
            self.files = [single_image]
        else:
            self.files = glob.glob(os.path.join(root_dir, "*.jpg"))
            np.random.shuffle(self.files)

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
        pin_memory=False,
        image_path: str = None
):
    dataset = ColorizationDataset(root_dir=root_dir, split=split, single_image=image_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=pin_memory
    )
    return dataloader
