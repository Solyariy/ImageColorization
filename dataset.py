import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb

IMG_SIZE = 256


class ColorizationDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.files = glob.glob(os.path.join(root_dir, "*.jpg"))

        split_idx = int(len(self.files) * 0.8)
        if split == 'train':
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]

        self.transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")

        img = self.transforms(img)

        img_lab = rgb2lab(np.array(img)).astype("float32")

        img_lab[:, :, 0] = (img_lab[:, :, 0] / 50) - 1
        img_lab[:, :, 1:] = img_lab[:, :, 1:] / 128

        img_lab = transforms.ToTensor()(img_lab)

        L = img_lab[[0], ...]
        ab = img_lab[[1, 2], ...]

        return {'L': L, 'ab': ab}
