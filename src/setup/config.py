from pathlib import Path

from src.setup.utils import get_device


class MainConfig:
    DATA_PATH: Path = Path(__file__).parents[2] / "data"
    LANDSCAPE_IMAGES: Path = DATA_PATH / "landscape_images"

    TRAINED_MODELS: Path = DATA_PATH.parent / "trained_models"
    PREDICTED_IMAGES: Path = DATA_PATH.parent / "predicted_data"

    DEVICE: str = get_device()

    IMG_SIZE = 256


main_config = MainConfig
