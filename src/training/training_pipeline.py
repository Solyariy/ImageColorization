import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.image_processing.dataset import ColorizationDataset
from src.image_processing.lab_convertor import LabConvertor
from src.models import ColorizationModel
from src.setup.config import MainConfig, TrainingConfig
from src.setup.enums import RunTypeEnum


def setup_data():
    dataset = ColorizationDataset(
        root_dir=f"{MainConfig.DATA_PATH}/landscape_images",
        split=RunTypeEnum.TRAIN
    )
    loader = DataLoader(
        dataset,
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return loader


def train_one_epoch(model, loader, processor, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0
    loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{TrainingConfig.EPOCHS}")

    for rgb_batch in loop:
        rgb_batch = rgb_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        processed_data = processor(rgb_batch)
        L, real_ab = processed_data["L"], processed_data["ab"]

        fake_ab = model(L)
        loss = criterion(fake_ab, real_ab)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader)


def save_checkpoint(model, epoch):
    path = MainConfig.TRAINED_MODELS / f"baseline_unet_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to {path}\n")


def train_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_loader = setup_data()
    model = ColorizationModel().to(device)
    processor = LabConvertor().to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=TrainingConfig.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    for epoch in range(TrainingConfig.EPOCHS):
        avg_loss = train_one_epoch(
            model, train_loader, processor, optimizer, criterion, device, epoch
        )

        print(f"\nEpoch [{epoch + 1}/{TrainingConfig.EPOCHS}] Completed. Avg Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == TrainingConfig.EPOCHS:
            save_checkpoint(model, epoch)


if __name__ == "__main__":
    train_baseline()
