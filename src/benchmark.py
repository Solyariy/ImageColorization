import time
import torch
import torch.nn as nn

from src.image_processing.dataset import make_dataloaders
from src.image_processing.lab_convertor import LabConvertor
from src.model import ColorizationModel
from src.setup.config import main_config
from src.setup.enums import RunTypeEnum


def run_benchmark(model):
    device = torch.device("mps")
    model = model.to(device)
    model.train()

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    batch_sizes = [16, 32, 64]
    worker_counts = [0, 2, 4, 6, 8]

    test_batches = 10

    processor = LabConvertor().to(device)

    print(f"{'Batch Size':<12} | {'Workers':<9} | {'Speed (it/s)':<15} | {'Images/sec':<15}")
    print("-" * 55)

    for bs in batch_sizes:
        for nw in worker_counts:
            loader = make_dataloaders(
                root_dir=main_config.LANDSCAPE_IMAGES,
                split=RunTypeEnum.TRAIN,
                n_workers=nw,
                batch_size=bs
            )
            loader_iter = iter(loader)

            try:
                for _ in range(2):
                    batch = next(loader_iter)
                    rgb_batch = batch.to(device)

                    # Process into Normalized L and ab channels
                    processed_data = processor(rgb_batch)
                    L = processed_data["L"]
                    ab = processed_data["ab"]
                    out = model(L)
                    loss = criterion(out, ab)
                    loss.backward()
                    optimizer.zero_grad()
            except StopIteration:
                continue

            start_time = time.time()
            valid_batches = 0

            for _ in range(test_batches):
                try:
                    rgb_batch = next(loader_iter)
                    rgb_batch = rgb_batch.to(device)

                    # Process into Normalized L and ab channels
                    processed_data = processor(rgb_batch)
                    L = processed_data["L"]
                    ab = processed_data["ab"]

                    out = model(L)
                    loss = criterion(out, ab)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    valid_batches += 1
                except StopIteration:
                    break

            end_time = time.time()

            if valid_batches > 0:
                duration = end_time - start_time
                it_per_sec = valid_batches / duration
                img_per_sec = (valid_batches * bs) / duration
                print(f"{bs:<12} | {nw:<9} | {it_per_sec:<15.2f} | {img_per_sec:<15.2f}")


if __name__ == '__main__':
    run_benchmark(model=ColorizationModel())
