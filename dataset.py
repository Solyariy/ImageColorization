import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import kornia

IMG_SIZE = 256


class ColorizationDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.files = glob.glob(os.path.join(root_dir, "*.jpg"))

        split_idx = int(len(self.files) * 0.8)
        if split == "train":
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]

        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    (IMG_SIZE, IMG_SIZE), transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transforms(img)

        return img


class GPUDataProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_batch):
        lab_batch = kornia.color.rgb_to_lab(img_batch)

        L = lab_batch[:, [0], :, :] / 50 - 1
        ab = lab_batch[:, 1:, :, :] / 128

        return {"L": L, "ab": ab}


if __name__ == "__main__":
    ds = ColorizationDataset("data/landscape_images/color", split="train")
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = GPUDataProcessor().to(device)

    rgb_batch = next(iter(dl))
    print(f"1. CPU Output Shape: {rgb_batch.shape} (RGB)")

    rgb_batch = rgb_batch.to(device)

    data = processor(rgb_batch)
    L = data["L"]
    ab = data["ab"]

    print(f"2. GPU L Shape: {L.shape}")
    print(f"3. GPU ab Shape: {ab.shape}")
    print(f"4. Max L Value: {L.max().item():.2f} (Should be ~1.0)")
