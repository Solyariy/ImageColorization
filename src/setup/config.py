from pathlib import Path

from src.setup.utils import get_device


class MainConfig:
    DATA_PATH: Path = Path(__file__).parents[2] / "data"
    LANDSCAPE_IMAGES: Path = DATA_PATH / "landscape_images"

    TRAINED_MODELS: Path = DATA_PATH.parent / "trained_models"
    PREDICTED_IMAGES: Path = DATA_PATH.parent / "predicted_data"

    DEVICE: str = get_device()

    IMG_SIZE = 256


class BenchmarkConfig:
    BATCH_SIZES: list[int] = [16, 32, 64]
    WORKER_COUNTS: list[int] = [0, 2, 4, 6, 8]
    TEST_BATCHES: int = 10


class TrainingConfig:
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 8
    LEARNING_RATE: float = 2e-4
    EPOCHS: int = 30
    GAN_EPOCHS: int = 100
    BETAS: tuple[float, float] = (0.5, 0.999)
    LAMBDA_L1: int = 100


main_config = MainConfig
