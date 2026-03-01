import uuid
from datetime import datetime

import torch


def get_datetime():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_uuid_str():
    return str(uuid.uuid4())
