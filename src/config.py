from pathlib import Path


class MainConfig:
    DATA_PATH: Path = Path(__file__).parents[1] / "data"
    LANDSCAPE_IMAGES: Path = DATA_PATH / "landscape_images"

    IMG_SIZE = 256


main_config = MainConfig
