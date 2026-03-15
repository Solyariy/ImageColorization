import time

import torch
import torch.nn as nn

from src.image_processing.dataset import make_dataloaders
from src.image_processing.lab_convertor import LabConvertor
from src.model import ColorizationModel
from src.setup.config import BenchmarkConfig, TrainingConfig, MainConfig, main_config
from src.setup.enums import RunTypeEnum


def setup_benchmark_components(model, device):
    model = model.to(device)
    model.train()

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE)
    processor = LabConvertor().to(device)

    return model, criterion, optimizer, processor


def run_warmup(loader_iter, model, processor, criterion, optimizer, device):
    try:
        for _ in range(2):
            batch = next(loader_iter).to(device, non_blocking=True)
            processed_data = processor(batch)
            out = model(processed_data["L"])
            loss = criterion(out, processed_data["ab"])
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
        return True
    except StopIteration:
        return False


def measure_throughput(loader_iter, model, processor, criterion, optimizer, device, batch_size):
    device_type = device.type if isinstance(device, torch.device) else device
    if device_type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.perf_counter()

    valid_batches = 0

    for _ in range(BenchmarkConfig.TEST_BATCHES):
        try:
            rgb_batch = next(loader_iter).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            processed_data = processor(rgb_batch)
            out = model(processed_data["L"])
            loss = criterion(out, processed_data["ab"])

            loss.backward()
            optimizer.step()
            valid_batches += 1
        except StopIteration:
            break

    if device_type == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        duration_sec = start_event.elapsed_time(end_event) / 1000
    else:
        duration_sec = time.perf_counter() - start_time

    if valid_batches > 0:
        it_per_sec = valid_batches / duration_sec
        img_per_sec = (valid_batches * batch_size) / duration_sec
        return it_per_sec, img_per_sec

    return 0, 0


def run_benchmark(model):
    device = main_config.DEVICE
    model, criterion, optimizer, processor = setup_benchmark_components(model, device)

    print(f"{'Batch Size':<12} | {'Workers':<9} | {'Speed (it/s)':<15} | {'Images/sec':<15}")
    print("-" * 55)

    for bs in BenchmarkConfig.BATCH_SIZES:
        for nw in BenchmarkConfig.WORKER_COUNTS:
            loader = make_dataloaders(
                root_dir=MainConfig.LANDSCAPE_IMAGES,
                split=RunTypeEnum.TRAIN,
                n_workers=nw,
                batch_size=bs,
                pin_memory=True
            )
            loader_iter = iter(loader)

            if not run_warmup(loader_iter, model, processor, criterion, optimizer, device):
                continue

            it_per_sec, img_per_sec = measure_throughput(
                loader_iter, model, processor, criterion, optimizer, device, bs
            )

            if it_per_sec > 0:
                print(f"{bs:<12} | {nw:<9} | {it_per_sec:<15.2f} | {img_per_sec:<15.2f}")


if __name__ == "__main__":
    run_benchmark(model=ColorizationModel())
